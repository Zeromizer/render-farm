"""Pure tests for the turntable work plan: two-anchor backward compatibility,
four-anchor ordering/direction, partial-reference rejection, staged segments
and ready-segment reuse. No ComfyUI, ffmpeg or Supabase needed.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studio import turntable as tt  # noqa: E402

REF = lambda n: {"bucket": "assets", "path": f"sha256/{n * 8}"}  # noqa: E731
TWO = {"front": REF("a"), "rear": REF("b"), "car": "A red hatchback"}
FOUR = dict(TWO, left=REF("c"), right=REF("d"))


class TwoAnchorCompat(unittest.TestCase):
    def test_defaults_unchanged(self):
        pl = tt.plan(TWO)
        self.assertEqual(pl["variant"], "two")
        self.assertEqual(pl["requested"], ["front_to_rear", "rear_to_front"])
        self.assertEqual(pl["generate"], ["front_to_rear", "rear_to_front"])
        self.assertEqual(pl["seconds"], 10)
        self.assertTrue(pl["complete"] and pl["loops"])
        self.assertEqual(pl["anchors"], ["front", "rear"])
        self.assertEqual([s["seconds"] for s in pl["segments"]], [10, 10])

    def test_seconds_per_half_still_honoured(self):
        self.assertEqual(tt.plan(dict(TWO, seconds_per_half=6))["seconds"], 6)
        with self.assertRaises(ValueError):
            tt.plan(dict(TWO, seconds_per_half=12))

    def test_labels_match_documented_phase_text(self):
        labels = tt.labels_for(tt.plan(TWO))
        self.assertEqual(labels["front_to_rear"], "turntable half 1 (front to rear)")
        self.assertEqual(labels["rear_to_front"], "turntable half 2 (rear to front)")

    def test_four_anchor_names_rejected_in_two_anchor_mode_without_sides(self):
        # Naming a quarter implies four-anchor mode; without side photos the missing anchors are reported.
        with self.assertRaises(ValueError) as cm:
            tt.plan(dict(TWO, segments=["front_to_left"]))
        self.assertIn("left", str(cm.exception))


class FourAnchor(unittest.TestCase):
    def test_order_and_anchor_chain(self):
        pl = tt.plan(FOUR)
        self.assertEqual(pl["variant"], "four")
        self.assertEqual(pl["requested"], ["front_to_left", "left_to_rear", "rear_to_right", "right_to_front"])
        chain = [(s["start"], s["end"]) for s in pl["segments"]]
        self.assertEqual(chain, [("front", "left"), ("left", "rear"), ("rear", "right"), ("right", "front")])
        # each segment starts where the previous ended, and the loop closes
        for a, b in zip(chain, chain[1:] + chain[:1]):
            self.assertEqual(a[1], b[0])
        self.assertEqual(pl["seconds"], 5)
        self.assertEqual(pl["anchors"], ["front", "left", "rear", "right"])
        self.assertTrue(pl["complete"])

    def test_direction_is_consistent_clockwise(self):
        for name in tt.SEQUENCES["four"] + tt.SEQUENCES["two"]:
            self.assertIn("clockwise", tt.SWEEP[name], name)
            self.assertIn("as seen from above", tt.SWEEP[name], name)
            self.assertNotIn("anticlockwise", tt.SWEEP[name], name)
        self.assertIn("clockwise", tt.REPAIR_MOTION)
        # left flank faces the camera first (vehicle's left), then rear, then right
        self.assertIn("left side swings round", tt.SWEEP["front_to_left"])
        self.assertIn("right side swings round", tt.SWEEP["rear_to_right"])

    def test_partial_side_reference_rejected(self):
        for only in ("left", "right"):
            with self.assertRaises(ValueError) as cm:
                tt.plan(dict(TWO, **{only: REF("z")}))
            other = "right" if only == "left" else "left"
            self.assertIn(f"turntable.{other} is missing", str(cm.exception))

    def test_malformed_ref_rejected(self):
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, left={"bucket": "assets"}))
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, right="sha256/abc"))

    def test_labels(self):
        labels = tt.labels_for(tt.plan(FOUR))
        self.assertEqual(labels["front_to_left"], "quarter 1/4 (front to left)")
        self.assertEqual(labels["rear_to_right"], "quarter 3/4 (rear to right)")

    def test_seconds_limits(self):
        self.assertEqual(tt.plan(dict(FOUR, seconds_per_quarter=8))["seconds"], 8)
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, seconds_per_quarter=11))
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, seconds_per_quarter=2))


class StagedSegments(unittest.TestCase):
    def test_single_quarter_is_partial_and_does_not_loop(self):
        pl = tt.plan(dict(FOUR, segments=["front_to_left"]))
        self.assertEqual(pl["generate"], ["front_to_left"])
        self.assertEqual(pl["anchors"], ["front", "left"])
        self.assertFalse(pl["complete"])
        self.assertFalse(pl["loops"])
        self.assertEqual(tt.labels_for(pl)["front_to_left"], "quarter 1/4 (front to left)")

    def test_single_quarter_needs_only_its_two_photos(self):
        pl = tt.plan({"front": REF("a"), "left": REF("c"), "right": REF("d"), "car": "x", "segments": ["front_to_left"]})
        self.assertEqual(pl["anchors"], ["front", "left"])

    def test_consecutive_subset_ok_non_consecutive_rejected(self):
        pl = tt.plan(dict(FOUR, segments=["left_to_rear", "rear_to_right"]))
        self.assertEqual([s["position"] for s in pl["segments"]], [2, 3])
        with self.assertRaises(ValueError) as cm:
            tt.plan(dict(FOUR, segments=["front_to_left", "rear_to_right"]))
        self.assertIn("consecutive", str(cm.exception))
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, segments=["rear_to_right", "left_to_rear"]))   # wrong order
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, segments=["front_to_rear"]))                  # a two-anchor name

    def test_unknown_or_duplicate_segment_rejected(self):
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, segments=["front_to_back"]))
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, segments=["front_to_left", "front_to_left"]))

    def test_ready_segments_skip_generation(self):
        ready = {"front_to_left": {"bucket": "renders", "path": "outputs/j1-front_to_left.mp4"},
                 "left_to_rear": {"bucket": "assets", "path": "sha256/" + "e" * 64}}
        pl = tt.plan(dict(FOUR, ready_segments=ready))
        self.assertEqual(pl["reuse"], ["front_to_left", "left_to_rear"])
        self.assertEqual(pl["generate"], ["rear_to_right", "right_to_front"])
        self.assertEqual(pl["anchors"], ["front", "rear", "right"])
        self.assertTrue(pl["complete"])

    def test_assembly_only_needs_no_photos(self):
        ready = {n: {"bucket": "renders", "path": f"outputs/j1-{n}.mp4"} for n in tt.SEQUENCES["four"]}
        pl = tt.plan({"ready_segments": ready})
        self.assertEqual(pl["variant"], "four")
        self.assertEqual(pl["generate"], [])
        self.assertEqual(pl["anchors"], [])
        self.assertTrue(pl["loops"])

    def test_ready_segment_outside_request_or_unknown_rejected(self):
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, segments=["front_to_left"],
                         ready_segments={"left_to_rear": {"bucket": "renders", "path": "x.mp4"}}))
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, ready_segments={"nope": {"bucket": "renders", "path": "x.mp4"}}))
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, ready_segments={"front_to_left": "outputs/x.mp4"}))

    def test_car_required_only_when_generating(self):
        with self.assertRaises(ValueError):
            tt.plan({"front": REF("a"), "rear": REF("b")})
        ready = {n: {"bucket": "renders", "path": f"outputs/j1-{n}.mp4"} for n in tt.SEQUENCES["two"]}
        tt.plan({"ready_segments": ready})   # no car needed


class Progress(unittest.TestCase):
    def test_ranges_are_increasing_and_cover_the_span(self):
        for n in (1, 2, 3, 4):
            r = tt.progress_ranges(n)
            self.assertEqual(r[0][0], 5)
            self.assertEqual(r[-1][1], 85)
            for (a, b), (c, d) in zip(r, r[1:]):
                self.assertLess(a, b)
                self.assertEqual(b, c)
        self.assertEqual(tt.progress_ranges(0), [])


if __name__ == "__main__":
    unittest.main()
