"""Pure tests for the turntable work plan: two-anchor backward compatibility,
four-anchor ordering/direction, partial-reference rejection, staged segments
and ready-segment reuse. No ComfyUI, ffmpeg or Supabase needed.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studio import post  # noqa: E402
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


class BridgedJoins(unittest.TestCase):
    def test_bridged_by_default_only_for_a_complete_rotation(self):
        self.assertTrue(tt.plan(FOUR)["bridge"])
        self.assertTrue(tt.plan(TWO)["bridge"])
        self.assertFalse(tt.plan(dict(FOUR, segments=["front_to_left"]))["bridge"])
        self.assertFalse(tt.plan(dict(FOUR, joins="anchor"))["bridge"])
        with self.assertRaises(ValueError):
            tt.plan(dict(FOUR, joins="crossfade"))

    def test_all_ready_segments_still_bridge(self):
        ready = {n: {"bucket": "renders", "path": f"outputs/j1-{n}.mp4"} for n in tt.SEQUENCES["four"]}
        pl = tt.plan({"ready_segments": ready})
        self.assertEqual(pl["generate"], [])
        self.assertTrue(pl["bridge"])

    def test_gen_frames_follow_the_cut_motion(self):
        steady = [2.0] * 124
        self.assertEqual(tt.bridge_gen_frames(steady, steady), 42)          # 30 + 12 frames at the same speed
        eased = [2.0] * 94 + [1.0] * 30                                      # brakes into the photo
        self.assertEqual(tt.bridge_gen_frames(eased, steady), 27)            # (30 * 1 + 12 * 2) / 2
        self.assertEqual(tt.bridge_gen_frames([0.0] * 124, [0.0] * 124), tt.BRIDGE_GEN[0])
        self.assertEqual(tt.bridge_gen_frames([2.0] * 94 + [10.0] * 30, steady), tt.BRIDGE_GEN[1])   # a cut that covered far more

    def test_windows_are_on_the_grid_with_context_in_range(self):
        self.assertEqual(tt.bridge_window(25), (107, 41, 25))
        self.assertEqual(tt.bridge_window(43), (124, 40, 44))
        self.assertEqual(tt.bridge_window(12), (90, 39, 12))
        for gen in range(tt.BRIDGE_GEN[0], tt.BRIDGE_GEN[1] + 1):
            length, c, g = tt.bridge_window(gen)
            self.assertEqual((length - 5) % 17, 0)
            self.assertTrue(tt.BRIDGE_CONTEXT[0] <= c <= tt.BRIDGE_CONTEXT[1], gen)
            self.assertIn(g - gen, (0, 1))
            self.assertEqual(length, 2 * c + g)

    def test_second_window_changes_the_geometry(self):
        self.assertEqual(tt.bridge_window_alt(107, 41), (107, 39, 29))
        self.assertEqual(tt.bridge_window_alt(90, 31), (90, 33, 24))       # too little context: grow it
        self.assertEqual(tt.bridge_window_alt(141, 40), (141, 42, 57))     # would generate too much: grow it

    def test_layout_covers_every_kept_and_generated_frame_once(self):
        keeps, gens = [82, 82, 83, 82], [31, 43, 25, 29]
        pieces = tt.bridge_layout(keeps, gens)
        self.assertEqual(len(pieces), 8)
        self.assertEqual(sum(e - s for _, s, e in pieces), sum(keeps) + sum(gens))
        # segment 1 opens on the last bridge's landing side, right after that bridge's new frames
        self.assertEqual(pieces[0], (3, 82 + 29, 82 + 29 + 41))
        self.assertEqual(pieces[1], (0, 41, 82 + 31))
        # segment 3 (odd length) splits 41 / 42
        self.assertEqual(pieces[4], (1, 82 + 43, 82 + 43 + 41))
        self.assertEqual(pieces[5], (2, 41, 83 + 25))

    def test_two_anchor_layout_alternates(self):
        pieces = tt.bridge_layout([200, 200], [30, 34])
        self.assertEqual([b for b, _, _ in pieces], [1, 0, 0, 1])

    def test_flash_and_jump_detection(self):
        ramp = [(100 + i * 0.5, 128.0, 128.0) for i in range(40)]
        self.assertEqual(tt.flash_frames(ramp, 0, 40), [])
        flash = list(ramp)
        flash[20] = (flash[20][0], 131.0, 125.0)                             # one magenta-ish frame
        self.assertEqual(tt.flash_frames(flash, 0, 40), [20])
        self.assertEqual(tt.flash_frames(flash, 0, 20), [])                  # outside the checked window
        judder = [1.0, 1.0, 1.0, 2.0] * 10                                   # H3's one-in-four double step
        self.assertEqual(tt.jump_frames(judder, 0, 40), [])
        snap = list(judder)
        snap[21] = 4.5
        self.assertEqual(tt.jump_frames(snap, 0, 40), [21])

    def test_take_choice(self):
        t = lambda n, seam, f=(), j=(): {"take": n, "seam_jump": seam, "flashes": list(f), "jumps": list(j)}  # noqa: E731
        self.assertEqual(tt.pick_bridge_take([t(1, 2.4), t(2, 1.9)])["take"], 2)
        self.assertEqual(tt.pick_bridge_take([t(1, 1.5, f=[90]), t(2, 2.2)])["take"], 2)
        self.assertIsNone(tt.pick_bridge_take([t(1, 5.1), t(2, 1.8, j=[113])]))


class Dejudder(unittest.TestCase):
    """H3's one-in-four long step (post.judder_weights / dejudder_times), on synthetic motion."""

    @staticmethod
    def displacement_per_output_step(true_steps, times):
        C = [0.0]
        for x in true_steps:
            C.append(C[-1] + x)
        pos = post._interp(times, list(range(len(C))), C)
        return [b - a for a, b in zip(pos, pos[1:])]

    @staticmethod
    def profile(true_steps):
        """motion_profile-style: one luma value per step, displacement ** (1 / gamma), m[0] copies m[1]."""
        luma = [x ** (1 / post.JUDDER_GAMMA) for x in true_steps]
        return [luma[1]] + luma[1:]

    def test_long_step_is_evened_out(self):
        true = [1.0, 1.0, 1.0, 1.85] * 30
        m = self.profile(true)
        n_out = int(round((len(m) + 1) * 60 / 24))
        uniform = [k * len(m) / (n_out - 1) for k in range(n_out)]
        before = self.displacement_per_output_step(true, uniform)[10:-10]
        after = self.displacement_per_output_step(true, post.dejudder_times(m, n_out))[10:-10]
        self.assertGreater(max(before) / min(before), 1.7)
        self.assertLess(max(after) / min(after), 1.15)

    def test_clean_clip_and_its_pacing_are_left_alone(self):
        ease = [2.0 - 1.5 * i / 119 for i in range(120)]     # slows to a quarter speed, no judder
        m = self.profile(ease)
        self.assertTrue(all(abs(w - 1) < 1e-9 for w in post.judder_weights(m[1:])))
        n_out = int(round((len(m) + 1) * 60 / 24))
        uniform = [k * len(m) / (n_out - 1) for k in range(n_out)]
        times = post.dejudder_times(m, n_out)
        self.assertLess(max(abs(a - b) for a, b in zip(times, uniform)), 0.35)
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], len(m))

    def test_period4_mean_flattens_the_pattern(self):
        v = [1.0, 1.0, 1.0, 2.0] * 6
        mid = post.period4_mean(v)[4:-4]
        self.assertTrue(all(abs(x - 1.25) < 1e-9 for x in mid))


if __name__ == "__main__":
    unittest.main()
