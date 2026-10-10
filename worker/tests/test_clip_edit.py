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

    def test_clean_and_anchors(self):
        _runner.validate(_params(clean={"prompt": "open paving"}))
        _runner.validate(_params(anchors=[{"image": {"bucket": "assets", "path": "c"}, "at_s": 2}]))
        for bad in ({"clean": {"prompt": " "}},
                    {"clean": {"prompt": "open paving", "every": 2}},
                    {"clean": {"prompt": "open paving"}, "anchors": [{"image": {"bucket": "a", "path": "c"}}]},
                    {"anchors": [{"at_s": 2}]},
                    {"mode": "extend", "seconds": 2, "clean": {"prompt": "open paving"}}):
            with self.assertRaises(RuntimeError):
                _runner.validate(_params(**bad))

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

    def test_region_anchor(self):
        # the cleaned frame is the plain background, shifted 3 px the way an image model's
        # redraw can be; it must line up, land in the box on its frames, and those frames
        # must go to H3 unmasked but for a band at the box edge (a moving object on a still
        # camera needs a cleaned frame per stretch: each carries only its own cleaned box)
        rng = np.random.default_rng(3)
        bg = cv2.GaussianBlur(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8), (0, 0), 2)
        img = os.path.join(self.d, "clean.png")
        cv2.imwrite(img, np.roll(bg, 3, axis=1))
        keys = [{"frame": 0, "box": [40, 70, 90, 110]}, {"frame": 40, "box": [120, 70, 170, 110]}]
        rep = self._run("anchor", {"mode": "region", "clip": self.a, "regions": [{"keys": keys}],
                                   "anchors": [{"frame": f, "image": img} for f in (0, 17, 34)],
                                   "anchor_every": 10}, 41)
        self.assertEqual(rep["warnings"], [])
        w = os.path.join(self.d, "anchor")
        plan = json.load(open(os.path.join(w, "plan.json")))
        self.assertEqual(plan["anchor_frames"], [0, 17, 34])   # key frames only
        masks, _ = CE.L.read_clip(os.path.join(w, "gen_mask.mp4"))
        lo = plan["window"][0]
        for a in plan["anchor_frames"]:   # only a band inside the box edge is left for H3
            on = float((masks[a - lo] > 127).mean())
            self.assertTrue(0 < on < 0.6 * float((masks[a - lo + 1] > 127).mean()), (a, on))
        self.assertGreater(int(masks[25 - lo].max()), 200)
        tl, _ = CE.L.read_clip(os.path.join(w, "timeline.mp4"))
        patch = tl[17][75:105, 76:122].astype(np.int16)   # where the green box was on frame 17
        self.assertLess(float(np.abs(patch - bg[75:105, 76:122].astype(np.int16)).mean()), 12.0)

    def test_clean_inputs_picks_the_fullest_frame(self):
        # the box grows from frame 0 to 20 and holds: the main cleaned frame is the key frame
        # of the fullest stretch, the first boxed key frame joins it (the entry), each written
        # at the image model's size with its mask
        keys = [{"frame": 0, "box": [280, 70, 320, 110]}, {"frame": 20, "box": [200, 70, 320, 110]},
                {"frame": 40, "box": [200, 70, 320, 110]}]
        w = os.path.join(self.d, "clean")
        os.makedirs(w, exist_ok=True)
        CE.prep({"mode": "region", "clip": self.a, "regions": [{"keys": keys, "track": False}], "work_dir": w})
        pre = os.path.join(w, "c_")
        CE.clean_inputs({"work_dir": w, "clip": self.a, "size": [320, 176], "out_prefix": pre})
        cj = json.load(open(os.path.join(w, "clean.json")))
        self.assertEqual(cj["main"], 34)   # the fullest stretch is 20-40; 34 is its key frame
        self.assertEqual([f["frame"] for f in cj["frames"]], [0, 17, 34])   # 17 ends the entry
        m = cv2.imread(cj["frames"][2]["mask"], cv2.IMREAD_GRAYSCALE)
        self.assertEqual(m.shape, (176, 320))
        self.assertEqual(int(m[90, 300]), 255)
        self.assertEqual(int(m[90, 100]), 0)
        self.assertIsNotNone(cv2.imread(cj["frames"][0]["image"]))
        CE.clean_inputs({"work_dir": w, "clip": self.a, "size": [320, 176], "frame": 5, "out_prefix": pre})
        self.assertEqual(json.load(open(os.path.join(w, "clean.json")))["main"], 5)

    def test_anchor_groups_on_entries(self):
        # the box grows over the first clip (an object entering): the middle 4-frame latent of
        # that clip is anchored whole, besides the key frames
        rng = np.random.default_rng(3)
        bg = cv2.GaussianBlur(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8), (0, 0), 2)
        img = os.path.join(self.d, "clean2.png")
        cv2.imwrite(img, bg)
        keys = [{"frame": 0, "box": [280, 70, 320, 110]}, {"frame": 20, "box": [200, 70, 320, 110]},
                {"frame": 40, "box": [200, 70, 320, 110]}]
        w = os.path.join(self.d, "groups")
        os.makedirs(w, exist_ok=True)
        CE.prep({"mode": "region", "clip": self.a, "regions": [{"keys": keys, "track": False}], "work_dir": w,
                 "anchors": [{"frame": 34, "image": img}], "anchor_every": 17})
        plan = json.load(open(os.path.join(w, "plan.json")))
        self.assertEqual(plan["anchor_frames"], [0, 9, 10, 11, 12, 17, 34])

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
