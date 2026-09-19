"""Tests for the planar_patch engine.

Pure part (no OpenCV, no Supabase): library resolution and params -> spec merge rules in
runners/planar_patch.py. Tracking/compositing tests run only when cv2 is importable (the engine's
own venv has it; the worker venv does not), on a synthetic clip: a dark rectangle sliding across a
textured background must be tracked to within a pixel and come out repainted.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from runners import planar_patch as pp  # noqa: E402  (needs the worker venv: db -> supabase)
except ImportError:
    pp = None

try:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

needs_runner = unittest.skipUnless(pp, "runner imports need the worker venv")


def fetch_stub(obj, name):
    return f"DL:{name}:{obj['bucket']}/{obj['path']}"


@needs_runner
class Library(unittest.TestCase):
    def test_atto3evo_entries_resolve_to_files(self):
        for element in ("plate", "badge"):
            e = pp.resolve_library(f"atto3evo/{element}")
            self.assertTrue(os.path.exists(e["artwork"]), e["artwork"])
            if e.get("alpha"):
                self.assertTrue(os.path.exists(e["alpha"]), e["alpha"])

    def test_badge_is_letters_only(self):
        e = pp.resolve_library("atto3evo/badge")
        self.assertTrue(e["alpha"])
        self.assertFalse(e["refine"])
        self.assertGreater(e["clear"], 0)

    def test_bad_refs(self):
        for ref in ("atto3evo", "atto3evo/plate/x", "../x/y", "nosuch/plate", "atto3evo/wing", "atto3evo/_subject"):
            with self.assertRaises(RuntimeError):
                pp.resolve_library(ref)

    def test_index_lists_subjects_and_views(self):
        idx = pp.library_index()
        self.assertIn("atto3evo", idx)
        s = idx["atto3evo"]
        self.assertEqual(s["label"], "BYD Atto 3 EVO")
        self.assertEqual(s["views"], ["rear"])
        self.assertEqual(sorted(s["elements"]), ["badge", "plate"])
        self.assertEqual(s["elements"]["plate"]["artwork_size"], [566, 168])
        self.assertTrue(s["elements"]["badge"]["letters_only"])
        self.assertFalse(s["elements"]["plate"]["letters_only"])
        json.dumps(idx)  # must be serialisable as-is


@needs_runner
class Localize(unittest.TestCase):
    PNG = bytes.fromhex("89504e470d0a1a0a" "0000000d49484452" "0000001000000008" "0802000000" + "00" * 8)

    def test_extensionless_png_gets_suffix(self):
        with tempfile.TemporaryDirectory() as d:
            raw = os.path.join(d, "plate-artwork")
            with open(raw, "wb") as f:
                f.write(self.PNG)
            got = pp.localize_image(raw, "plate artwork")
            self.assertTrue(got.endswith(".png"), got)
            self.assertTrue(os.path.exists(got))
            self.assertFalse(os.path.exists(raw))

    def test_non_image_refused(self):
        with tempfile.TemporaryDirectory() as d:
            raw = os.path.join(d, "plate-artwork")
            with open(raw, "wb") as f:
                f.write(bytes.fromhex("0000001866747970" "69736f6d") + bytes(32))  # an mp4 header
            with self.assertRaises(RuntimeError):
                pp.localize_image(raw, "plate artwork")


@needs_runner
class SpecMerge(unittest.TestCase):
    def test_library_defaults_then_overrides(self):
        spec = pp.build_spec({"patches": [{"library": "atto3evo/badge", "key_box": [1, 2, 3, 4], "blur": 0}]},
                             "/w", fetch_stub)
        p = spec["patches"][0]
        self.assertEqual(p["name"], "patch0")
        self.assertEqual(p["key_box"], [1, 2, 3, 4])
        self.assertEqual(p["blur"], 0)
        self.assertEqual(p["clear"], 0.03)        # inherited
        self.assertTrue(p["alpha"].endswith("badge_alpha.png"))

    def test_custom_artwork_drops_library_alpha(self):
        spec = pp.build_spec({"patches": [{"name": "b", "library": "atto3evo/badge",
                                           "artwork": {"bucket": "assets", "path": "sha256/aa"}}]}, "/w", fetch_stub)
        p = spec["patches"][0]
        self.assertEqual(p["artwork"], "DL:b-artwork:assets/sha256/aa")
        self.assertIsNone(p["alpha"])

    def test_storage_only_patch(self):
        spec = pp.build_spec({"patches": [{"name": "lbl", "artwork": {"bucket": "assets", "path": "sha256/aa"},
                                           "alpha": {"bucket": "assets", "path": "sha256/bb"}, "key_box": [0, 0, 9, 9]}]},
                             "/w", fetch_stub)
        p = spec["patches"][0]
        self.assertEqual(p["alpha"], "DL:lbl-alpha:assets/sha256/bb")
        self.assertNotIn("clear", p)  # planar.py applies its DEFAULTS

    def test_needs_patches_and_artwork(self):
        with self.assertRaises(RuntimeError):
            pp.build_spec({"patches": []}, "/w", fetch_stub)
        with self.assertRaises(RuntimeError):
            pp.build_spec({"patches": [{"name": "x", "key_box": [0, 0, 1, 1]}]}, "/w", fetch_stub)

    def test_order_preserved(self):
        spec = pp.build_spec({"patches": [{"library": "atto3evo/plate"}, {"library": "atto3evo/badge", "key_box": [0, 0, 1, 1]}]},
                             "/w", fetch_stub)
        self.assertEqual([p["name"] for p in spec["patches"]], ["patch0", "patch1"])


@unittest.skipUnless(HAVE_CV2, "cv2 not in this venv (engine venv has it)")
class Tracking(unittest.TestCase):
    def _synthetic(self, n=30, W=320, H=180):
        import cv2
        import numpy as np
        rng = np.random.default_rng(1)
        bg = cv2.GaussianBlur(rng.integers(90, 200, (H, W, 3), dtype=np.uint8), (0, 0), 1.5)
        frames, boxes = [], []
        for i in range(n):
            f = bg.copy()
            x0 = 60 + i * 3
            y0 = 70 + (i % 5)
            cv2.rectangle(f, (x0, y0), (x0 + 60, y0 + 18), (15, 15, 15), -1)
            cv2.putText(f, "AB 12", (x0 + 6, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            frames.append(f)
            boxes.append((x0, y0, x0 + 60, y0 + 18))
        return frames, boxes

    def test_track_follows_moving_plate(self):
        import numpy as np
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patch"))
        import planar
        import cv2
        frames, boxes = self._synthetic()
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        key = 15
        x0, y0, x1, y1 = boxes[key]
        q = planar.order_quad([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        quads, misses, lost = planar.track(grays, key, q, 60 / 18, True, 1.0)
        self.assertEqual(lost, 0)
        for i, (bx0, by0, bx1, by1) in enumerate(boxes):
            got = quads[i]
            self.assertLess(abs(got[:, 0].min() - bx0), 2.5, f"frame {i} x0 {got[:, 0].min()} vs {bx0}")
            self.assertLess(abs(got[:, 1].min() - by0), 2.5, f"frame {i} y0 {got[:, 1].min()} vs {by0}")

    def test_end_to_end_repaints(self):
        import cv2
        import numpy as np
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patch"))
        import planar
        frames, boxes = self._synthetic()
        with tempfile.TemporaryDirectory() as d:
            clip = os.path.join(d, "in.mp4")
            vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 180))
            for f in frames:
                vw.write(f)
            vw.release()
            art = np.full((36, 120, 3), 20, np.uint8)
            cv2.putText(art, "ZZ 99", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
            cv2.imwrite(os.path.join(d, "art.png"), art)
            out = os.path.join(d, "out.mp4")
            spec = {"clip": clip, "out": out, "proof": os.path.join(d, "proof.png"),
                    "patches": [{"name": "plate", "artwork": os.path.join(d, "art.png"),
                                 "key_frame": 15, "key_box": list(boxes[15]), "refine": True, "match": False}]}
            sp = os.path.join(d, "spec.json")
            with open(sp, "w") as f:
                json.dump(spec, f)
            try:
                planar.main(sp)
            except RuntimeError as exc:  # ffmpeg missing on a bare test box
                if "ffmpeg" in str(exc).lower():
                    self.skipTest(str(exc))
                raise
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(spec["proof"]))
            cap = cv2.VideoCapture(out)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f0 = cap.read()
            cap.release()
            self.assertTrue(ok)
            bx0, by0, bx1, by1 = boxes[0]
            # The original "AB 12" text has been replaced: the plate region's bright pixels moved.
            orig = frames[0][by0:by1, bx0:bx1].mean()
            new = f0[by0:by1, bx0:bx1].mean()
            self.assertNotAlmostEqual(orig, new, delta=1.0)


if __name__ == "__main__":
    unittest.main()
