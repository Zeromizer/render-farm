"""clip_edit move mode: param validation, route / track / camera maths, and a prep -> build ->
compose round trip on a tiny synthetic locked-off clip (a red block driving across a grey
road with a soft shadow; SAM's masks and the render's are faked from the known boxes),
which catches shape bugs and checks the old car AND its shadow are gone before a GPU run.
The pixel side needs OpenCV + ffmpeg; those tests skip without them."""
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _params(**over):
    p = {"org_id": "o", "job_id": "j", "source": {"bucket": "assets", "path": "a"}, "mode": "move",
         "prompt": "top-down shot, the red car turns left", "object": "red car", "point_norm": [0.25, 0.5],
         "route": [[0.5, 0.5], [0.6, 0.2], [0.6, -0.2]]}
    p.update(over)
    return p


try:
    from runners import clip_edit as _runner  # noqa: E402  (needs the worker venv: db -> supabase)
except ImportError:
    _runner = None

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "edit"))
    import clip_move as CM  # noqa: E402
    import cv2  # noqa: E402
    import numpy as np  # noqa: E402
    CM.L.find_ffmpeg_tool("ffmpeg")
except Exception:  # noqa: BLE001 - no OpenCV / ffmpeg on this box
    CM = None


@unittest.skipUnless(_runner, "runner imports need the worker venv")
class ValidateMoveTest(unittest.TestCase):
    def test_defaults(self):
        p = _runner.validate(_params())
        self.assertEqual((p["takes"], p["seed"]), (1, 6332))

    def test_needs_object_point_route(self):
        for bad in ({"object": ""}, {"point_norm": [1.2, 0.5]}, {"point_norm": None}, {"route": []},
                    {"route": [[0.5, 2.0]]}, {"route": [[0.1, 0.1]] * 13}):
            with self.assertRaises(RuntimeError, msg=str(bad)):
                _runner.validate(_params(**bad))

    def test_route_may_leave_the_shot(self):
        _runner.validate(_params(route=[[0.5, 0.5], [1.4, 0.5]]))

    def test_timing_and_ttm(self):
        _runner.validate(_params(start_s=1.0, hold_s=0.5, arrive_s=4.0, ttm=[1, 2], ease="linear", turn=False))
        for bad in ({"start_s": 3, "arrive_s": 2}, {"ttm": [0, 2]}, {"ttm": [3, 2]}, {"ease": "bounce"},
                    {"hold_s": -1}):
            with self.assertRaises(RuntimeError, msg=str(bad)):
                _runner.validate(_params(**bad))

    def test_other_modes_keep_two_takes(self):
        p = _runner.validate({"org_id": "o", "job_id": "j", "source": {"bucket": "a", "path": "b"},
                              "mode": "region", "prompt": "x", "regions": [{"box": [1, 1, 50, 50]}]})
        self.assertEqual(p["takes"], 2)


@unittest.skipUnless(CM, "needs OpenCV + numpy + ffmpeg")
class GeometryTest(unittest.TestCase):
    def test_snap_and_canvas(self):
        self.assertEqual(CM.snap_up(121), 124)
        self.assertEqual(CM.snap_up(124), 124)
        w, h = CM.canvas_for(1920, 1080, 124)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertLessEqual(w * h, 1344 * 768 * 1.05)
        w2, h2 = CM.canvas_for(1920, 1080, 362)  # long clips get a smaller canvas
        self.assertLess(w2 * h2, w * h)
        self.assertEqual(CM.canvas_for(640, 360, 124), CM.L.gen_dims(640 / 360, 640 * 360))

    def test_poses_hold_and_arrive(self):
        ps = CM.poses([(0, 0), (100, 0), (100, 100)], 60, start=10, hold=5, arrive=40)
        self.assertEqual(ps[0][:2], ps[15][:2])                       # still until start + hold
        self.assertAlmostEqual(ps[40][0], 100, delta=0.5)
        self.assertAlmostEqual(ps[40][1], 100, delta=0.5)
        self.assertEqual(ps[40][:2], ps[59][:2])                      # still after arriving
        self.assertAlmostEqual(ps[16][2], 0.0, delta=0.3)              # heading east first
        self.assertAlmostEqual(ps[39][2], math.pi / 2, delta=0.3)      # then south

    def test_pick_track_by_point(self):
        a = [np.zeros((40, 80), bool) for _ in range(5)]
        b = [np.zeros((40, 80), bool) for _ in range(5)]
        for m in a:
            m[5:15, 5:20] = True
        for m in b:
            m[20:30, 50:70] = True
        self.assertEqual(CM.pick_track([a, b], 2, (0.75, 0.6), 80, 40), 1)
        self.assertEqual(CM.pick_track([a, b], 2, (0.15, 0.25), 80, 40), 0)
        with self.assertRaises(RuntimeError):
            CM.pick_track([a, b], 2, (0.5, 0.95), 80, 40)

    def test_fill_gaps_slides_the_mask(self):
        ms = [np.zeros((20, 100), bool) for _ in range(7)]
        for f in (0, 6):
            ms[f][5:10, 10 + 10 * f:20 + 10 * f] = True
        out = CM.fill_gaps(ms)
        c = CM.centroid(out[3])
        self.assertIsNotNone(c)
        self.assertAlmostEqual(c[0], 14.5 + 30, delta=2)

    def test_camera_drift(self):
        rng = np.random.default_rng(1)
        base = (rng.random((180, 320, 3)) * 255).astype(np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), 2)
        still = [base.copy() for _ in range(30)]
        pan = [np.roll(base, 2 * i, axis=1) for i in range(30)]
        self.assertLess(CM.camera_drift(still), 0.005)
        self.assertGreater(CM.camera_drift(pan), 0.1)


def _road(w, h):
    img = np.full((h, w, 3), 96, np.uint8)
    cv2.line(img, (0, h // 2), (w, h // 2), (230, 230, 230), 2)
    cv2.line(img, (w // 2, 0), (w // 2, h), (230, 230, 230), 2)
    return img


def _car_box(f, w, h, n, path):
    """Car centre at frame f along path ('right' = old path east; 'up' = new path)."""
    if path == "right":
        cx, cy = int(w * 0.2 + w * 0.6 * f / (n - 1)), int(h * 0.4)
    else:
        t = f / (n - 1)
        cx, cy = int(w * 0.2 + w * 0.3 * min(1, 2 * t)), int(h * 0.4 - h * 0.6 * max(0, 2 * t - 1))
    return cx - 14, cy - 8, cx + 14, cy + 8


def _draw_car(img, box):
    x0, y0, x1, y1 = box
    sh = np.zeros(img.shape[:2], np.float32)
    cv2.rectangle(sh, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 6)[..., None]
    out = (img.astype(np.float32) * (1 - 0.35 * sh)).astype(np.uint8)  # soft shadow halo
    cv2.rectangle(out, (x0, y0), (x1, y1), (30, 30, 220), -1)
    return out


def _mask_video(path, boxes, w, h):
    ms = []
    for b in boxes:
        m = np.zeros((h, w), bool)
        if b is not None:
            m[max(0, b[1]):max(0, b[3]), max(0, b[0]):max(0, b[2])] = True
        ms.append(m)
    CM.write_masks(path, ms)


@unittest.skipUnless(CM, "needs OpenCV + numpy + ffmpeg")
class RoundTripTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="clip_move_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_prep_build_compose(self):
        w, h, n = 320, 192, 40
        road = _road(w, h)
        frames = [_draw_car(road, _car_box(f, w, h, n, "right")) for f in range(n)]
        clip = os.path.join(self.dir, "clip.mp4")
        CM.L.write_clip(clip, frames, CM.FPS)
        CM.prep({"clip": clip, "work_dir": self.dir})
        with open(os.path.join(self.dir, "plan.json")) as fh:
            plan = json.load(fh)
        W, H = plan["canvas"]
        self.assertEqual(plan["length"], 39 if n == 39 else CM.snap_up(n))
        self.assertLess(plan["camera_drift"], CM.MAX_CAMERA_DRIFT)
        sx, sy = W / w, H / h
        sc = lambda b: (int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy))  # noqa: E731
        # a second, parked "car" so pick_track has to choose
        src_masks = [os.path.join(self.dir, "src_obj0.mp4"), os.path.join(self.dir, "src_obj1.mp4")]
        _mask_video(src_masks[0], [(int(W * 0.8), int(H * 0.8), int(W * 0.9), int(H * 0.9))] * plan["length"], W, H)
        _mask_video(src_masks[1], [sc(_car_box(min(f, n - 1), w, h, n, "right")) for f in range(plan["length"])], W, H)
        CM.build({"work_dir": self.dir, "sam_masks": src_masks, "point_norm": [0.2, 0.4],
                  "route": [[0.5, 0.4], [0.5, 0.1], [0.5, -0.3]], "hold_s": 0, "arrive_s": (n - 1) / CM.FPS})
        with open(os.path.join(self.dir, "poses.json")) as fh:
            pz = json.load(fh)
        self.assertEqual(pz["track"], 1)
        for f in ("move_ref.mp4", "move_refmask.mp4", "oldmask.mp4", "first.png", "route.png", "plate.png"):
            self.assertTrue(os.path.exists(os.path.join(self.dir, f)), f)
        plate = cv2.imread(os.path.join(self.dir, "plate.png"))
        self.assertLess(float(np.abs(plate.astype(int) - cv2.resize(road, (W, H)).astype(int)).mean()), 4.0,
                        "the plate should be the empty road")
        # fake take: the reference itself; the render's car mask from the drawn poses
        take = os.path.join(self.dir, "move_ref.mp4")
        ref_frames, _ = CM.L.read_clip(take)
        drawn = pz["drawn"]
        car_boxes = []
        for f in range(plan["length"]):
            d = drawn[f] if f < len(drawn) else None
            car_boxes.append(None if d is None else (int(d[0]) - 16, int(d[1]) - 10, int(d[0]) + 16, int(d[1]) + 10))
        tm = os.path.join(self.dir, "take_obj0.mp4")
        _mask_video(tm, car_boxes, W, H)
        out = os.path.join(self.dir, "edit.mp4")
        CM.compose({"work_dir": self.dir, "clip": clip, "takes": [take], "take_masks": [[tm]], "seeds": [6332],
                    "out": out, "proof": os.path.join(self.dir, "proof.png"),
                    "report": os.path.join(self.dir, "report.json")})
        with open(os.path.join(self.dir, "report.json")) as fh:
            rep = json.load(fh)
        self.assertEqual(rep["best_take"], 1)
        res, _ = CM.L.read_clip(out)
        self.assertEqual(len(res), n)
        self.assertEqual(res[0].shape[:2], (h, w))
        # where the old car (and its shadow) was late in the clip: road again
        f = n - 4
        x0, y0, x1, y1 = _car_box(f, w, h, n, "right")
        patch = res[f][y0 - 10:y1 + 10, x0 - 10:x1 + 10].astype(int)
        want = road[y0 - 10:y1 + 10, x0 - 10:x1 + 10].astype(int)
        self.assertLess(float(np.abs(patch - want).mean()), 8.0, "old car or its shadow is still there")
        # far from both cars: the source as decoded, untouched (one re-encode of drift)
        dec, _ = CM.L.read_clip(clip)
        self.assertLess(float(np.abs(res[f][h - 30:, :60].astype(int) - dec[f][h - 30:, :60].astype(int)).mean()), 3.0)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "proof.png")))

    def test_moving_camera_is_refused(self):
        w, h, n = 320, 192, 40
        rng = np.random.default_rng(3)
        tex = cv2.GaussianBlur((rng.random((h, w * 2, 3)) * 255).astype(np.uint8), (0, 0), 2)
        frames = [tex[:, 3 * f:3 * f + w].copy() for f in range(n)]
        clip = os.path.join(self.dir, "pan.mp4")
        CM.L.write_clip(clip, frames, CM.FPS)
        with self.assertRaises(RuntimeError) as cm:
            CM.prep({"clip": clip, "work_dir": self.dir})
        self.assertIn("camera moves", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
