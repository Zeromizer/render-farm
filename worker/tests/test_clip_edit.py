"""clip_edit: param validation, grid / window / canvas maths, and a prep -> compose round
trip of every mode on a tiny synthetic clip (fake take = the generation input itself),
which is what catches shape bugs in compose and the proof sheet before a GPU run.
The pixel side needs OpenCV + ffmpeg; those tests skip without them."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _params(**over):
    p = {"org_id": "o", "job_id": "j", "source": {"bucket": "assets", "path": "a"}, "mode": "region",
         "prompt": "empty street", "regions": [{"box": [10, 10, 100, 40]}]}
    p.update(over)
    return p


try:
    from runners import clip_edit as _runner  # noqa: E402  (needs the worker venv: db -> supabase)
except ImportError:
    _runner = None

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "edit"))
    import clip_edit as CE  # noqa: E402
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    CE.L.find_ffmpeg_tool("ffmpeg")
except Exception:  # noqa: BLE001 - no OpenCV / ffmpeg on this box
    CE = None


@unittest.skipUnless(_runner, "runner imports need the worker venv")
class ValidateTest(unittest.TestCase):
    def test_defaults(self):
        p = _runner.validate(_params())
        self.assertEqual((p["takes"], p["seed"]), (2, 6332))

    def test_bad_mode(self):
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(mode="morph"))

    def test_region_needs_boxes(self):
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(regions=[]))
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(regions=[{"box": [10, 10, 12, 40]}]))

    def test_keyframed_region(self):
        k = [{"at_s": 0, "box_norm": [0.9, 0.3, 1, 0.7]}, {"at_s": 1, "box_norm": [0.7, 0.3, 1, 0.7]}]
        _runner.validate(_params(regions=[{"keys": k}]))
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(regions=[{"keys": k[:1]}]))

    def test_bridge_needs_second_clip(self):
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(mode="bridge", seconds=1.5))
        _runner.validate(_params(mode="bridge", seconds=1.5, source_b={"bucket": "assets", "path": "b"}))

    def test_seconds_range(self):
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(mode="extend", seconds=30))

    def test_needs_prompt(self):
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(prompt=" "))


@unittest.skipIf(CE is None, "OpenCV / ffmpeg not available")
class MathsTest(unittest.TestCase):
    def test_snap_up(self):
        self.assertEqual([CE.snap_up(n) for n in (1, 5, 6, 22, 23, 193)], [5, 5, 22, 22, 39, 209])

    def test_window_covers_range(self):
        lo, n = CE.window_around(300, 100, 150, 24)
        self.assertLessEqual(lo, 100)
        self.assertGreaterEqual(lo + n - 1, 150)
        self.assertEqual((n - 5) % 17, 0)

    def test_window_too_long(self):
        with self.assertRaises(RuntimeError):
            CE.window_around(500, 0, 400, 0)

    def test_canvas_budget(self):
        w, h = CE.canvas_for(9 / 16, 300, CE.MAX_AREA, 720 * 1280)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertLessEqual(w * h * 300, CE.PIXEL_FRAMES * 1.1)


@unittest.skipIf(CE is None, "OpenCV / ffmpeg not available")
class RoundTripTest(unittest.TestCase):
    """Every mode through prep and compose with the generation input as the 'take'."""

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        bg = cv2.GaussianBlur(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8), (0, 0), 2)

        def clip(path, n, shift):
            frames = []
            for i in range(n):
                f = bg.copy()
                x = 40 + shift * i
                cv2.rectangle(f, (x, 70), (x + 50, 110), (40, 200, 90), -1)
                cv2.putText(f, "AB", (x + 8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                frames.append(f)
            CE.L.write_clip(path, frames, 24)
        cls.a = os.path.join(cls.d, "a.mp4")
        cls.b = os.path.join(cls.d, "b.mp4")
        clip(cls.a, 41, 2)   # off the 17k+5 grid on purpose
        clip(cls.b, 30, -1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def _run(self, name, spec, expect_frames):
        w = os.path.join(self.d, name)
        os.makedirs(w, exist_ok=True)
        spec = dict(spec, work_dir=w)
        CE.prep(spec)
        out = os.path.join(w, "out.mp4")
        CE.compose({"work_dir": w, "takes": [os.path.join(w, "gen_src.mp4")] * 2, "seeds": [1, 2], "out": out,
                    "proof": os.path.join(w, "proof.png"), "report": os.path.join(w, "report.json")})
        frames, _ = CE.L.read_clip(out)
        self.assertEqual(len(frames), expect_frames)
        self.assertIsNotNone(cv2.imread(os.path.join(w, "proof.png")))
        rep = json.load(open(os.path.join(w, "report.json")))
        self.assertEqual(len(rep["takes"]), 2)
        return rep

    def test_region_tracked(self):
        self._run("region", {"mode": "region", "clip": self.a,
                             "regions": [{"box": [40, 70, 90, 110], "frame": 0, "track": True}]}, 41)

    def test_region_static_full(self):
        self._run("full", {"mode": "region", "clip": self.a, "crop": "full",
                           "regions": [{"box": [0, 0, 320, 60], "track": False, "start_frame": 10, "end_frame": 30}]}, 41)

    def test_extend(self):
        rep = self._run_len("extend", {"mode": "extend", "clip": self.a, "seconds": 1.0})
        self.assertGreater(rep["generated_frames"], 0)

    def test_prepend_and_bridge(self):
        rep = self._run_len("prepend", {"mode": "prepend", "clip": self.a, "seconds": 1.0})
        self.assertGreater(rep["generated_frames"], 0)
        rep = self._run_len("bridge", {"mode": "bridge", "clip": self.a, "clip_b": self.b, "seconds": 0.5})
        self.assertGreater(rep["generated_frames"], 0)

    def test_region_norm_seconds(self):
        self._run("norm", {"mode": "region", "clip": self.a,
                           "regions": [{"box_norm": [0.1, 0.38, 0.3, 0.62], "at_s": 0.0, "end_s": 1.0}]}, 41)

    def test_region_keyframed(self):
        self._run("keys", {"mode": "region", "clip": self.a, "regions": [{"keys": [
            {"frame": 0, "box_norm": [0.1, 0.38, 0.3, 0.62]}, {"frame": 40, "box_norm": [0.35, 0.38, 0.6, 0.62]}]}]}, 41)

    def test_audio(self):
        self._run("audio", {"mode": "audio", "clip": self.a, "start_frame": 5, "end_frame": 30}, 41)

    def _run_len(self, name, spec):
        """Like _run, but the expected length is the timeline's (clip + generated)."""
        w = os.path.join(self.d, name)
        os.makedirs(w, exist_ok=True)
        CE.prep(dict(spec, work_dir=w))
        plan = json.load(open(os.path.join(w, "plan.json")))
        out = os.path.join(w, "out.mp4")
        CE.compose({"work_dir": w, "takes": [os.path.join(w, "gen_src.mp4")], "seeds": [1], "out": out,
                    "proof": os.path.join(w, "proof.png"), "report": os.path.join(w, "report.json")})
        self.assertEqual(len(CE.L.read_clip(out)[0]), plan["frames"])
        return json.load(open(os.path.join(w, "report.json")))


if __name__ == "__main__":
    unittest.main()
