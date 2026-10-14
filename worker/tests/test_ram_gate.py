"""Pure tests for the H3 RAM gate (videogen/ram_gate.py): fake clock and readings,
no Windows memory API, ComfyUI or Supabase needed.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proc  # noqa: E402
from videogen import ram_gate  # noqa: E402


class FakeBox:
    """Readings are consumed one per read(); sleep() advances the fake clock."""

    def __init__(self, readings, cancel_after=None):
        self.readings, self.t, self.waits, self.logs = list(readings), 0.0, [], []
        self.cancel_after = cancel_after

    def read(self):
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]

    def sleep(self, s):
        self.t += s

    def clock(self):
        return self.t

    def cancel(self):
        return self.cancel_after is not None and self.t >= self.cancel_after

    def gate(self, need=12, max_wait=600):
        return ram_gate.wait_for_ram(need, max_wait, self.cancel, self.waits.append, self.logs.append,
                                     read=self.read, sleep=self.sleep, clock=self.clock, poll_s=10)


class RamGate(unittest.TestCase):
    def test_enough_ram_passes_immediately_without_logging(self):
        box = FakeBox([20.0])
        self.assertEqual(box.gate(), 0.0)
        self.assertEqual((box.waits, box.logs), ([], []))

    def test_disabled_when_need_is_zero(self):
        box = FakeBox([0.5])
        self.assertEqual(box.gate(need=0), 0.0)

    def test_waits_until_two_consecutive_good_readings(self):
        # short, short, good, short (blip resets), good, good -> done
        box = FakeBox([3.0, 4.0, 13.0, 5.0, 14.0, 15.0])
        waited = box.gate()
        self.assertEqual(waited, 50.0)
        self.assertEqual(box.waits, [3.0, 4.0, 5.0])  # phase updated only on short readings
        self.assertIn("after waiting", box.logs[-1])

    def test_gives_up_after_max_wait_and_starts_anyway(self):
        box = FakeBox([2.0])
        waited = box.gate(max_wait=60)
        self.assertEqual(waited, 60.0)
        self.assertIn("starting anyway", box.logs[-1])

    def test_cancel_while_waiting_raises(self):
        box = FakeBox([2.0], cancel_after=30)
        with self.assertRaises(proc.Canceled):
            box.gate()

    def test_real_reader_returns_a_plausible_number(self):
        gb = ram_gate.available_gb()
        self.assertGreater(gb, 0)
        self.assertLess(gb, 1024)


if __name__ == "__main__":
    unittest.main()
