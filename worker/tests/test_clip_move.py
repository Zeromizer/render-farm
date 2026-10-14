"""clip_edit move mode: param validation, route / track / camera maths, the LTX plate windows,
and prep -> build -> compose round trips on tiny synthetic clips (a red block driving across a
grey road with a soft shadow; SAM's masks, the LTX plate and the render are faked from the
known geometry): a locked-off one on the median plate, and a panning camera on a faked LTX
plate with a timed route. They catch shape bugs and check the old car AND its shadow are gone
and the new car lands where the route says, before a GPU run.
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

    def test_timed_route(self):
        _runner.validate(_params(route=[{"at_s": 2.0, "x": 0.5, "y": 0.4}, {"at_s": 4.0, "x": 0.7, "y": 1.2}]))
        _runner.validate(_params(route=[{"frame": 48, "x": 0.5, "y": 0.4}]))
        for bad in ([{"at_s": 2.0, "x": 0.5, "y": 0.4}, [0.6, 0.2]],       # mixed kinds
                    [{"at_s": 2.0, "x": 0.5}],                              # no y
                    [{"at_s": 25.0, "x": 0.5, "y": 0.4}],                   # past 20 s
                    [{"at_s": 2.0, "x": 0.5, "y": 1.8}]):                   # off the -0.5..1.5 range
            with self.assertRaises(RuntimeError, msg=str(bad)):
                _runner.validate(_params(route=bad))

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

    def test_poses_keep_a_driving_car_moving(self):
        ps = CM.poses([(0, 0), (200, 0)], 60, start=0, hold=0, arrive=50, v0=6.0)
        self.assertAlmostEqual(ps[1][0] - ps[0][0], 6.0, delta=0.8)    # leaves at its own speed
        self.assertAlmostEqual(ps[50][0], 200, delta=0.5)              # and still arrives on time
        self.assertTrue(all(b[0] >= a[0] - 1e-6 for a, b in zip(ps, ps[1:])))
        rest = CM.poses([(0, 0), (200, 0)], 60, start=0, hold=0, arrive=50)
        self.assertLess(rest[1][0] - rest[0][0], 0.5)                  # v0 0: from rest, as before
        pt = CM.poses_timed([(0, 0), (100, 0), (150, 0)], [20, 50], 60, go=0, v0=5.0)
        self.assertAlmostEqual(pt[1][0] - pt[0][0], 5.0, delta=0.8)

    def test_poses_timed_hits_each_point_on_time(self):
        route = [(0, 0), (100, 0), (100, 100), (0, 100)]
        ps = CM.poses_timed(route, [20, 50, 70], 90, go=5)
        self.assertEqual(ps[0][:2], ps[5][:2])                        # still until it leaves
        for f, (x, y) in zip((20, 50, 70), route[1:]):
            self.assertAlmostEqual(ps[f][0], x, delta=1.0)
            self.assertAlmostEqual(ps[f][1], y, delta=1.0)
        self.assertEqual(ps[70][:2], ps[89][:2])                      # stops at the last point
        # never backs up along the route
        dist = [0.0]
        for a, b in zip(ps[5:71], ps[6:71]):
            dist.append(dist[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.assertAlmostEqual(dist[-1], 300, delta=25)                # ~ the spline's length, no detours

    def test_ltx_windows_and_dims(self):
        self.assertEqual(CM.ltx_windows(40), [(0, 41)])
        self.assertEqual(CM.ltx_windows(121), [(0, 121)])
        for n in (124, 250, 362):
            ws = CM.ltx_windows(n)
            self.assertEqual(ws[0][0], 0)
            self.assertEqual(ws[-1][0] + ws[-1][1], n)               # the last one ends on the last frame
            for (a, c), (b, _) in zip(ws, ws[1:]):
                self.assertGreaterEqual(a + c - b, CM.LTX_OVERLAP)    # enough overlap to crossfade
            self.assertTrue(all(c == CM.LTX_FRAMES for _, c in ws))
        self.assertEqual(CM.ltx_dims(1344, 768), (1024, 576))
        self.assertEqual(CM.ltx_dims(768, 1344), (576, 1024))
        w, h = CM.ltx_dims(1920, 1080)
        self.assertEqual((w % 32, h % 32), (0, 0))

    def test_camera_track_follows_a_pan(self):
        rng = np.random.default_rng(1)
        tex = cv2.GaussianBlur((rng.random((180, 480, 3)) * 255).astype(np.uint8), (0, 0), 2)
        frames = [tex[:, 3 * i:3 * i + 320].copy() for i in range(30)]
        Hs, drift, lost = CM.camera_track(frames)
        self.assertEqual(lost, 0)
        x, y = CM.warp_pt(Hs[29], 200, 90)                            # frame 0 -> frame 29: 87 px left
        self.assertAlmostEqual(x, 200 - 87, delta=2.0)
        self.assertAlmostEqual(y, 90, delta=2.0)
        G = CM.relative(Hs, 10)                                       # frame 10 -> frame 29: 57 px left
        self.assertAlmostEqual(CM.warp_pt(G[29], 200, 90)[0], 200 - 57, delta=2.0)

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

    def test_main_body_drops_strays(self):
        m = np.zeros((100, 200), bool)
        m[10:40, 10:70] = True       # the car
        m[10:40, 74:90] = True       # a piece of it across a lane line: kept
        m[70:85, 150:165] = True     # a stray piece on something else: dropped
        (out,), cut = CM.main_body([m], 832)  # sizes tuned at 832 wide
        self.assertEqual(cut, 1)
        self.assertTrue(out[20, 80])
        self.assertFalse(out[75, 155])

    def test_remove_old_takes_the_hard_shadow_keeps_other_cars(self):
        road = np.full((200, 400, 3), (90, 95, 100), np.uint8)
        frame = road.copy()
        frame[80:110, 150:210] = (40, 200, 220)                       # the old car
        frame[110:118, 160:215] = (36, 38, 40)                        # its hard shadow, road tint
        frame[118:124, 160:215] = (32, 40, 56)                        # and warm, the way H3 draws it
        frame[60:90, 222:262] = (245, 245, 245)                       # a white car right beside it
        m = np.zeros((200, 400), bool)
        m[80:110, 150:210] = True
        out = CM.remove_old(frame, m, road, 832)
        self.assertLess(float(np.abs(out[112:122, 165:210] - road[112:122, 165:210]).mean()), 6.0, "shadow left")
        self.assertGreater(float(out[70:80, 235:250].mean()), 230, "the white car got painted over")

    def test_camera_drift(self):
        rng = np.random.default_rng(1)
        base = (rng.random((180, 320, 3)) * 255).astype(np.uint8)
        base = cv2.GaussianBlur(base, (0, 0), 2)
        still = [base.copy() for _ in range(30)]
        pan = [np.roll(base, 2 * i, axis=1) for i in range(30)]
        self.assertLess(CM.camera_drift(still), 0.005)
        self.assertGreater(CM.camera_drift(pan), 0.1)


def _stitch_check(test, d):
    """Two plate windows of flat colours crossfade over their overlap, n frames out."""
    a, b = os.path.join(d, "a.mp4"), os.path.join(d, "b.mp4")
    for pth, v in ((a, 40), (b, 200)):
        wr = CM.Writer(pth, 64, 32, CM.FPS, lossless=True)
        for _ in range(20):
            wr.write(np.full((32, 64, 3), v, np.uint8))
        wr.close()
    out = os.path.join(d, "plate.mp4")
    CM.stitch_plate([a, b], [(0, 20), (12, 20)], 30, out, (64, 32))
    res, _ = CM.L.read_clip(out)
    test.assertEqual(len(res), 30)
    vals = [int(r[16, 32, 1]) for r in res]
    test.assertLess(abs(vals[5] - 40), 3)
    test.assertLess(abs(vals[25] - 200), 3)
    test.assertTrue(all(x <= y + 2 for x, y in zip(vals, vals[1:])), vals)   # a smooth ramp, no jump
    test.assertTrue(60 < vals[16] < 180, vals)


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
        # the red car waits at its start for 24 frames, then drives off east
        park = 24
        old_box = lambda f: _car_box(max(0, f - park) * (n - 1) // (n - 1 - park), w, h, n, "right")  # noqa: E731
        frames = [_draw_car(road, old_box(f)) for f in range(n)]
        # other traffic: once it has gone, a white car creeps past just below where it waited
        # (most of the frames there that are free of the red car then show the white car)
        for f in range(park + 1, n):
            x = int(w * 0.12 + (f - park) * 2)
            cv2.rectangle(frames[f], (x, int(h * 0.47)), (x + 30, int(h * 0.55)), (245, 245, 245), -1)
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
        _mask_video(src_masks[1], [sc(old_box(min(f, n - 1))) for f in range(plan["length"])], W, H)
        # SAM's generic "car" pass finds the white car too
        white = os.path.join(self.dir, "all_obj0.mp4")
        _mask_video(white, [(int((w * 0.12 + (f - park) * 2) * sx), int(h * 0.47 * sy), int((w * 0.12 + (f - park) * 2 + 30) * sx),
                             int(h * 0.55 * sy)) if park < f < n else None for f in range(plan["length"])], W, H)
        CM.build({"work_dir": self.dir, "sam_masks": src_masks, "other_masks": [white], "point_norm": [0.2, 0.4],
                  "route": [[0.5, 0.4], [0.5, 0.1], [0.5, -0.3]], "hold_s": 0, "arrive_s": (n - 1) / CM.FPS})
        with open(os.path.join(self.dir, "poses.json")) as fh:
            pz = json.load(fh)
        self.assertEqual(pz["track"], 1)
        for f in ("move_ref.mp4", "move_refmask.mp4", "oldmask.mp4", "first.png", "route.png", "plate.png"):
            self.assertTrue(os.path.exists(os.path.join(self.dir, f)), f)
        plate = cv2.imread(os.path.join(self.dir, "plate.png"))
        self.assertLess(int(plate[int(H * 0.52), :int(W * 0.45)].max(axis=1).max()), 200,
                        "the passing white car leaked into the plate")
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
        x0, y0, x1, y1 = old_box(f)
        patch = res[f][y0 - 10:y1 + 10, x0 - 10:x1 + 10].astype(int)
        want = road[y0 - 10:y1 + 10, x0 - 10:x1 + 10].astype(int)
        self.assertLess(float(np.abs(patch - want).mean()), 8.0, "old car or its shadow is still there")
        # far from both cars: the source as decoded, untouched (one re-encode of drift)
        dec, _ = CM.L.read_clip(clip)
        self.assertLess(float(np.abs(res[f][h - 30:, :60].astype(int) - dec[f][h - 30:, :60].astype(int)).mean()), 3.0)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "proof.png")))

    def test_stitch_plate(self):
        _stitch_check(self, self.dir)

    def _pan_clip(self, w, h, n, pan, car_world):
        """A textured road the camera pans across `pan` px a frame, a red car at car_world(f)
        (world px); returns (frames, empty frames, car box in frame px per frame)."""
        rng = np.random.default_rng(3)
        world = cv2.GaussianBlur((rng.random((h, w + pan * n + 8, 3)) * 120 + 60).astype(np.uint8), (0, 0), 1.5)
        cv2.line(world, (0, h // 2), (world.shape[1], h // 2), (230, 230, 230), 2)
        frames, empty, boxes = [], [], []
        for f in range(n):
            bg = world[:, pan * f:pan * f + w].copy()
            cx, cy = car_world(f)
            box = (int(cx - pan * f) - 14, int(cy) - 8, int(cx - pan * f) + 14, int(cy) + 8)
            empty.append(bg)
            frames.append(_draw_car(bg, box))
            boxes.append(box)
        return frames, empty, boxes

    def test_moving_camera_needs_the_ltx_plate(self):
        w, h, n = 320, 192, 40
        frames, _, boxes = self._pan_clip(w, h, n, 3, lambda f: (80 + 4 * f, h * 0.4))
        clip = os.path.join(self.dir, "pan.mp4")
        CM.L.write_clip(clip, frames, CM.FPS)
        CM.prep({"clip": clip, "work_dir": self.dir})
        with open(os.path.join(self.dir, "plan.json")) as fh:
            plan = json.load(fh)
        self.assertTrue(plan["camera_moving"])
        self.assertTrue(os.path.exists(os.path.join(self.dir, plan["ltx"]["files"][0])))
        m = os.path.join(self.dir, "src_obj0.mp4")
        _mask_video(m, boxes + [boxes[-1]] * (plan["length"] - n), *plan["canvas"])
        with self.assertRaises(RuntimeError) as cm:
            CM.build({"work_dir": self.dir, "sam_masks": [m], "point_norm": [0.27, 0.4], "route": [[0.5, 0.1]]})
        self.assertIn("LTX", str(cm.exception))

    def test_moving_camera_round_trip(self):
        """A panning camera; the car should leave its lane and be at (0.5, 0.15) OF FRAME 30
        at frame 30 (a timed point): the drawn car must land there although the camera has
        moved 90 px since the start, the old car must be gone, the plate is the faked LTX one."""
        w, h, n, pan = 320, 192, 40, 3
        frames, empty, boxes = self._pan_clip(w, h, n, pan, lambda f: (80 + 4 * f, h * 0.4))
        clip = os.path.join(self.dir, "pan.mp4")
        CM.L.write_clip(clip, frames, CM.FPS)
        CM.prep({"clip": clip, "work_dir": self.dir})
        with open(os.path.join(self.dir, "plan.json")) as fh:
            plan = json.load(fh)
        W, H = plan["canvas"]
        length = plan["length"]
        self.assertTrue(plan["camera_moving"])
        sx, sy = W / w, H / h
        sc = lambda b: (int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy))  # noqa: E731
        m = os.path.join(self.dir, "src_obj0.mp4")
        _mask_video(m, [sc(b) for b in boxes] + [sc(boxes[-1])] * (length - n), W, H)
        # the faked LTX plate: the empty road at the LTX canvas, one window per plan
        lw, lh = plan["ltx"]["canvas"]
        wins = []
        for i, (s0, c) in enumerate(plan["ltx"]["windows"]):
            pth = os.path.join(self.dir, f"plate_{i}.mp4")
            wr = CM.Writer(pth, lw, lh, CM.FPS, lossless=True)
            for j in range(c):
                wr.write(cv2.resize(empty[min(s0 + j, n - 1)], (lw, lh), interpolation=cv2.INTER_AREA))
            wr.close()
            wins.append(pth)
        CM.build({"work_dir": self.dir, "sam_masks": [m], "plate_windows": wins, "point_norm": [0.27, 0.4],
                  "hold_s": 0, "route": [{"frame": 18, "x": 0.45, "y": 0.4}, {"frame": 30, "x": 0.5, "y": 0.15}]})
        self.assertTrue(os.path.exists(os.path.join(self.dir, "plate_ltx.mp4")))
        with open(os.path.join(self.dir, "poses.json")) as fh:
            pz = json.load(fh)
        self.assertTrue(pz["camera_moving"])
        d30 = pz["drawn"][30]
        self.assertAlmostEqual(d30[0], 0.5 * W, delta=6, msg=f"drawn car at frame 30: {d30}")
        self.assertAlmostEqual(d30[1], 0.15 * H, delta=6, msg=f"drawn car at frame 30: {d30}")
        # fake take = the reference; the render's car mask from the drawn poses
        take = os.path.join(self.dir, "move_ref.mp4")
        car_boxes = [None if d is None else (int(d[0]) - 16, int(d[1]) - 10, int(d[0]) + 16, int(d[1]) + 10)
                     for d in pz["drawn"]]
        tm = os.path.join(self.dir, "take_obj0.mp4")
        _mask_video(tm, car_boxes, W, H)
        out = os.path.join(self.dir, "edit.mp4")
        CM.compose({"work_dir": self.dir, "clip": clip, "takes": [take], "take_masks": [[tm]], "seeds": [6332],
                    "out": out, "proof": os.path.join(self.dir, "proof.png"),
                    "report": os.path.join(self.dir, "report.json")})
        with open(os.path.join(self.dir, "report.json")) as fh:
            rep = json.load(fh)
        self.assertEqual((rep["plate"], rep["camera_moving"]), ("ltx", True))
        res, _ = CM.L.read_clip(out)
        self.assertEqual(len(res), n)
        f = n - 4
        x0, y0, x1, y1 = boxes[f]
        patch = res[f][y0 - 10:y1 + 10, x0 - 10:x1 + 10].astype(int)
        want = empty[f][y0 - 10:y1 + 10, x0 - 10:x1 + 10].astype(int)
        self.assertLess(float(np.abs(patch - want).mean()), 8.0, "old car or its shadow is still there")
        # and the new car is in the composite where it was drawn
        cx, cy = int(d30[0] / sx), int(d30[1] / sy)
        self.assertGreater(int(res[30][cy, cx, 2]) - int(res[30][cy, cx, 1]), 100, "no red car at the timed point")


if __name__ == "__main__":
    unittest.main()
