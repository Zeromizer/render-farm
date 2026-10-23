"""face_swap: param validation and the prompt, piece planning, and a prep -> build -> score ->
finish round trip on a tiny synthetic clip (fake SAM masks, fake renders = the motion
references themselves), which is what catches shape bugs in the stitch and the proof sheet
before a GPU run. The pixel side needs OpenCV + ffmpeg; those tests skip without them."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _params(**over):
    p = {"org_id": "o", "job_id": "j", "target": {"bucket": "assets", "path": "t"},
         "face": {"bucket": "assets", "path": "f"}, "consent": {"confirmed": True},
         "subject": "man", "person": "a wide round face and short thick black hair with a fringe"}
    p.update(over)
    return p


try:
    from runners import face_swap as _runner  # noqa: E402  (needs the worker venv: db -> supabase)
except (ImportError, KeyError):  # no worker venv / no .env on this box
    _runner = None

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faceswap"))
    import face_swap as FS  # noqa: E402
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    FS.ffmpeg()
except Exception:  # noqa: BLE001 - no OpenCV / ffmpeg on this box
    FS = None


@unittest.skipUnless(_runner, "runner imports need the worker venv")
class ValidateTest(unittest.TestCase):
    def test_defaults(self):
        p = _runner.validate(_params())
        self.assertEqual((p["tries"], p["seed"], p["props"]), (2, 43, []))

    def test_consent_required(self):
        for c in (None, {}, {"confirmed": "yes"}, {"confirmed": False}):
            with self.assertRaises(RuntimeError):
                _runner.validate(_params(consent=c))

    def test_inputs_required(self):
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(face=None))
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(person="round"))

    def test_props_and_range(self):
        p = _runner.validate(_params(props=["cap", {"object": "headphones", "description": "grey over-ear headphones"}],
                                     start_s=2, end_s=40))
        self.assertEqual(p["props"][0], {"object": "cap", "description": "cap"})
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(props=["a", "b", "c", "d"]))
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(start_s=5, end_s=4))
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(start_s=0, end_s=61))

    def test_prompt(self):
        p = _runner.validate(_params(scene="the same light blue t-shirt, in the same dim room."))
        s = _runner.build_prompt(p, [{"object": "headphones", "description": "grey over-ear headphones"}])
        self.assertIn("The man from <Picture 1>", s)
        self.assertIn("his face, a wide round face", s)
        self.assertIn("He wears the grey over-ear headphones from <Picture 2>, exactly as worn in <Video 1>.", s)
        self.assertTrue(s.endswith("He wears the same light blue t-shirt, in the same dim room."))
        q = _runner.build_prompt(_runner.validate(_params(subject="person")), [])
        self.assertIn("their face", q)
        self.assertNotIn("<Picture 2>", q)


@unittest.skipUnless(FS, "needs OpenCV + ffmpeg")
class PlanTest(unittest.TestCase):
    def test_pieces_cover_the_clip(self):
        self.assertEqual(FS.plan_pieces(100, 243, 195), [0])
        self.assertEqual(FS.plan_pieces(243, 243, 195), [0])
        self.assertEqual(FS.plan_pieces(813, 243, 195), [0, 195, 390, 570])
        for n in (244, 500, 1440):
            s = FS.plan_pieces(n, 243, 195)
            self.assertEqual(s[-1] + 243, n)
            self.assertTrue(all(b - a <= 195 for a, b in zip(s, s[1:])))

    def test_canvas(self):
        self.assertEqual(FS.canvas_for(720, 1280)[1], (448, 768))
        self.assertEqual(FS.canvas_for(1920, 1080)[1], (768, 448))
        self.assertEqual(FS.canvas_for(1080, 1080)[1], (576, 576))
        for w, h in FS.CANVAS.values():
            self.assertEqual((w % 32, h % 32), (0, 0))


@unittest.skipUnless(FS, "needs OpenCV + ffmpeg")
class RoundTripTest(unittest.TestCase):
    """A 9 s portrait clip of a moving 'head' on a ceiling, pieces of 124 / stride 100."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="fs_test_")
        self.n = 9 * FS.FPS
        W, H = 360, 640
        self.target = os.path.join(self.dir, "target.mp4")
        frames, face, hair = [], [], []
        for i in range(self.n):
            f = np.full((H, W, 3), (90, 100, 110), np.uint8)
            cx, cy = 180 + int(40 * np.sin(i / 7)), 300
            cv2.circle(f, (cx, cy - 70), 70, (20, 20, 20), -1)       # hair
            cv2.ellipse(f, (cx, cy), (60, 80), 0, 0, 360, (120, 150, 200), -1)  # face
            frames.append(f)
            fm = np.zeros((H, W), np.uint8)
            cv2.ellipse(fm, (cx, cy), (60, 80), 0, 0, 360, 255, -1)
            hm = np.zeros((H, W), np.uint8)
            cv2.circle(hm, (cx, cy - 70), 70, 255, -1)
            hm[fm > 0] = 0
            face.append(cv2.cvtColor(fm, cv2.COLOR_GRAY2BGR))
            hair.append(cv2.cvtColor(hm, cv2.COLOR_GRAY2BGR))
        FS.L.write_clip(self.target, frames, FS.FPS)
        self.face_src, self.hair_src = face, hair

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _mask_video(self, name, masks, size):
        path = os.path.join(self.dir, name)
        FS.L.write_clip(path, [cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) for m in masks], FS.FPS,
                        lossless=True)
        return path

    def test_round_trip(self):
        FS.prep({"work_dir": self.dir, "target": self.target, "cloud": False})
        plan = json.load(open(os.path.join(self.dir, "plan.json")))
        self.assertEqual(plan["canvas"], [448, 768])
        self.assertEqual(plan["frames"], self.n)
        self.assertEqual(plan["starts"], [0, 92])
        size = tuple(plan["canvas"])
        # the masks SAM would give on src.mp4 (the target scaled 360x640 -> 448x768, same aspect)
        masks = {"face": [self._mask_video("face.mp4", self.face_src, size)],
                 "hair": [self._mask_video("hair.mp4", self.hair_src, size)], "hand": [], "props": {}}
        FS.build({"work_dir": self.dir, "masks": masks, "hair": "from_photo"})
        built = json.load(open(os.path.join(self.dir, "build.json")))
        self.assertEqual([p["start"] for p in built["pieces"]], [0, 92])
        for p in built["pieces"]:
            self.assertEqual(FS.probe_frames(p["path"])[0], plan["piece"])
        # the hair is painted out of the motion reference: the dark disc above the face is gone
        ref = next(FS.frames(os.path.join(self.dir, "ref.mp4")))
        self.assertGreater(ref[120:160, 200:250].mean(), 60)
        # fake renders = the motion references themselves: they follow the motion
        out = os.path.join(self.dir, "score.json")
        FS.score({"work_dir": self.dir, "renders": built["pieces"], "out": out})
        scores = json.load(open(out))
        self.assertTrue(all(s["mean"] > 0.9 for s in scores))
        picks = [{**p, "seed": 43, "tries": 1, "mean": 1.0, "min": 1.0, "face_mask": masks["face"][0] if not i else None}
                 for i, p in enumerate(built["pieces"])]
        res = {k: os.path.join(self.dir, f"swap{k}") for k in (".mp4", "-proof.png", ".json")}
        FS.finish({"work_dir": self.dir, "picks": picks, "chin_clean": True, "destubble": True, "out": res[".mp4"],
                   "proof": res["-proof.png"], "report": res[".json"]})
        self.assertEqual(FS.probe_frames(res[".mp4"])[:3], (self.n, 448, 768))
        rep = json.load(open(res[".json"]))
        self.assertEqual(len(rep["joins"]), 1)
        self.assertLess(rep["joins"][0]["mismatch"], 3)
        self.assertGreater(rep["sync_mean"], 0.7)  # hair painted out + chin passes: lower than the raw pieces
        self.assertTrue(os.path.exists(res["-proof.png"]))


if __name__ == "__main__":
    unittest.main()
