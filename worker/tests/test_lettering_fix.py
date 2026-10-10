"""lettering_fix: param validation, H3 grid / window maths and the inpaint graph shape.
The pixel side (worker/lettering/lettering.py) needs OpenCV, which the worker venv does
not have, so only its pure helpers are covered here when cv2 imports."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogen import graphs_inpaint  # noqa: E402


def _params(**over):
    p = {"org_id": "o", "job_id": "j", "source": {"bucket": "assets", "path": "a"},
         "reference": {"bucket": "assets", "path": "r"}, "prompt": "front of a green car",
         "elements": [{"name": "plate", "box": [10, 10, 100, 40]}]}
    p.update(over)
    return p


try:
    from runners import lettering_fix as _runner  # noqa: E402  (needs the worker venv: db -> supabase)
except ImportError:
    _runner = None


@unittest.skipUnless(_runner, "runner imports need the worker venv")
class ValidateTest(unittest.TestCase):
    def setUp(self):
        self.mod = _runner

    def test_defaults(self):
        v = self.mod.validate(_params())
        self.assertEqual(v["takes"], 2)
        self.assertEqual(v["seed"], 6332)

    def test_requires_reference(self):
        with self.assertRaises(RuntimeError):
            self.mod.validate(_params(reference=None))

    def test_bad_box(self):
        with self.assertRaises(RuntimeError):
            self.mod.validate(_params(elements=[{"name": "plate", "box": [10, 10, 12, 40]}]))

    def test_duplicate_names(self):
        els = [{"name": "plate", "box": [0, 0, 50, 20]}, {"name": "plate", "box": [0, 30, 50, 50]}]
        with self.assertRaises(RuntimeError):
            self.mod.validate(_params(elements=els))

    def test_too_many_takes(self):
        with self.assertRaises(RuntimeError):
            self.mod.validate(_params(takes=5))

    def test_unnamed_elements_get_names(self):
        v = self.mod.validate(_params(elements=[{"box": [0, 0, 50, 20]}, {"box": [0, 30, 50, 50]}]))
        self.assertEqual([e["name"] for e in v["elements"]], ["element0", "element1"])


class GraphTest(unittest.TestCase):
    def test_turbo_graph(self):
        g, meta = graphs_inpaint.build("s.mp4", "m.mp4", "a.mp4", "car", 960, 544, 158, 7, "x/y")
        self.assertEqual(meta["steps"], 4)
        self.assertIn("lora", g)
        self.assertEqual(g["sample"]["inputs"]["latent_image"], ["feather", 0])
        self.assertEqual(g["latmask"]["class_type"], "vloMaskToLatentMask")
        self.assertEqual(g["cond"]["inputs"]["length"], 158)

    def test_full_steps(self):
        g, meta = graphs_inpaint.build("s", "m", "a", "car", 960, 544, 124, 1, "p", turbo=False)
        self.assertEqual(meta["steps"], 20)
        self.assertNotIn("lora", g)

    def test_rejects_off_grid(self):
        with self.assertRaises(ValueError):
            graphs_inpaint.build("s", "m", "a", "car", 960, 544, 120, 1, "p")
        with self.assertRaises(ValueError):
            graphs_inpaint.build("s", "m", "a", "car", 950, 544, 124, 1, "p")


try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lettering"))
    import lettering  # noqa: E402
except ImportError:  # no OpenCV in this interpreter
    lettering = None


@unittest.skipIf(lettering is None, "OpenCV not installed")
class PixelHelpersTest(unittest.TestCase):
    def test_snap_len(self):
        self.assertEqual(lettering.snap_len(158), 158)
        self.assertEqual(lettering.snap_len(150), 141)
        self.assertEqual(lettering.snap_len(3), 5)

    def test_window_keeps_anchor(self):
        self.assertEqual(lettering.window_for(158, 157), (0, 158))
        s, n = lettering.window_for(400, 399)
        self.assertEqual(s + n - 1, 399)
        self.assertEqual((n - 5) % 17, 0)
        s, n = lettering.window_for(400, 10)
        self.assertEqual(s, 0)
        self.assertLessEqual(n, lettering.MAX_WINDOW)

    def test_gen_dims_aligned(self):
        w, h = lettering.gen_dims(16 / 9)
        self.assertEqual((w % 32, h % 32), (0, 0))


if __name__ == "__main__":
    unittest.main()
