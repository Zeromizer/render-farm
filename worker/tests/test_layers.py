"""The verdict of worker/extract/layers.py on drawn pictures: letters as bars, a
subject as a blob. Needs numpy and cv2 only (no detector, no video), so it is
skipped on a bare Python."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "extract"))

try:
    import cv2
    import numpy as np
    import layers
except Exception:  # noqa: BLE001 - the extractor's venv is not this Python
    layers = None

H, W = 400, 640
TOP, BASE = 100, 200          # the word's cap line and baseline
BARS = [(60 + 60 * k, 100 + 60 * k) for k in range(9)]   # nine letters, 40 wide, 20 apart


def word(cut_by=None):
    """Nine bar letters; with cut_by, the letters are hidden where it is."""
    s = np.zeros((H, W), bool)
    for x0, x1 in BARS:
        s[TOP:BASE, x0:x1] = True
    if cut_by is not None:
        s &= ~cut_by
    return s


def blob(cx, cy, rx, ry):
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    return m.astype(bool)


@unittest.skipIf(layers is None, "needs numpy and cv2")
class Judge(unittest.TestCase):
    def test_a_roof_cutting_the_letters_is_behind(self):
        roof = blob(320, 270, 170, 85)            # rises 15 px into the word, under four letters
        layer, nums = layers.judge(word(cut_by=roof), roof)
        self.assertEqual(layer, "behind", nums)
        self.assertLess(nums["sure"], layers.SURE_DEEP)   # decided by the cut letters, not by depth
        self.assertGreaterEqual(nums["span"], layers.SPAN_MIN)

    def test_a_head_under_the_arm_of_a_t_is_clear(self):
        # The open-world remake of 2026-11-27: a runner's head rose into the
        # space under a T's arm. Those columns end short of the baseline by the
        # letter's own shape, and nowhere near the head.
        s = word()
        x0, x1 = BARS[4]
        s[TOP:BASE, x0 - 10:x1 + 10] = False
        s[TOP:TOP + 20, x0 - 10:x1 + 10] = True   # the arm
        s[TOP:BASE, x0 + 12:x1 - 12] = True       # the stem
        head = blob(x0 + 1, 215, 10, 60)          # under the left arm, against the stem
        layer, nums = layers.judge(s, head)
        self.assertEqual(layer, "clear", nums)
        self.assertEqual(nums["span"], 0)

    def test_letters_that_carry_on_over_the_subject_are_in_front(self):
        body = blob(320, 230, 120, 110)
        layer, nums = layers.judge(word(), body)
        self.assertEqual(layer, "front", nums)

    def test_a_subject_away_from_the_word_is_clear(self):
        layer, _ = layers.judge(word(), blob(320, 330, 150, 50))
        self.assertEqual(layer, "clear")

    def test_a_subject_well_inside_the_word_with_no_letters_on_it_is_behind(self):
        person = blob(320, 200, 70, 130)          # stands through the word, hiding two letters
        layer, nums = layers.judge(word(cut_by=person), person)
        self.assertEqual(layer, "behind", nums)

    def test_a_shallow_graze_claims_nothing(self):
        roof = blob(320, 290, 170, 95)            # rises 5 px into the word: within the outline's own error
        layer, _ = layers.judge(word(cut_by=roof), roof)
        self.assertNotEqual(layer, "behind")


@unittest.skipIf(layers is None, "needs numpy and cv2")
class Settle(unittest.TestCase):
    def test_takes_a_majority_of_the_frames(self):
        self.assertEqual(layers.settle(["behind", "behind", "unknown"]), "behind")
        self.assertEqual(layers.settle(["behind", "unknown", "unknown"]), "unknown")
        self.assertEqual(layers.settle(["clear", "unknown", "unknown"]), "unknown")
        self.assertEqual(layers.settle(["clear"]), "clear")

    def test_behind_and_front_both_seen_is_a_contradiction(self):
        self.assertEqual(layers.settle(["behind", "behind", "front"]), "unknown")


if __name__ == "__main__":
    unittest.main()
