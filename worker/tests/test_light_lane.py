"""Light lane admission (worker/light_lane.py room()). The real psutil, config
and db are used where they import (the worker venv); a module that cannot
import (no psutil, no .env) is stubbed, so this also runs on a bare Python.
Each test patches what room() reads, so the numbers do not depend on the box."""
import importlib
import sys
import types
import unittest
from unittest import mock


def _real_or_stub(name, **attrs):
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 - missing package or .env
        stub = types.ModuleType(name)
        stub.__dict__.update(attrs)
        sys.modules[name] = stub


_real_or_stub("psutil")
_real_or_stub("config", COMFYUI_URL="http://127.0.0.1:8188", LIGHT_LANE_MIN_AVAIL_RAM_GB=8,
              LIGHT_LANE_MIN_COMMIT_GB=8, LIGHT_LANE_MIN_FREE_VRAM_MIB=2048)
_real_or_stub("db")

import light_lane  # noqa: E402

GB = 1 << 30


class RoomTest(unittest.TestCase):
    def setUp(self):
        self.ram = mock.patch.object(light_lane, "psutil", types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(available=30 * GB)), create=True)
        self.commit = mock.patch.object(light_lane, "_commit_free_gb", return_value=40.0)
        self.vram = mock.patch.object(light_lane, "_free_vram_mib", return_value=12000)
        for p in (self.ram, self.commit, self.vram):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(lambda: light_lane.main_engine.__setitem__(0, None))

    def test_idle_loop_always_has_room_even_with_comfy_busy(self):
        light_lane.main_engine[0] = None
        with mock.patch.object(light_lane, "_comfy_busy", return_value=True) as busy:
            self.assertEqual(light_lane.room(), (True, ""))
        busy.assert_not_called()

    def test_python_spike_driving_comfy_pauses_the_lane(self):
        # 2026-10-18: a python spike ran H3 itself; its engine is not in PAUSE_DURING.
        light_lane.main_engine[0] = "python"
        with mock.patch.object(light_lane, "_comfy_busy", return_value=True):
            ok, why = light_lane.room()
        self.assertFalse(ok)
        self.assertIn("ComfyUI busy", why)

    def test_python_job_without_comfy_work_still_admits(self):
        light_lane.main_engine[0] = "python"
        with mock.patch.object(light_lane, "_comfy_busy", return_value=False):
            self.assertEqual(light_lane.room(), (True, ""))

    def test_pause_engines_still_pause_without_asking_comfy(self):
        light_lane.main_engine[0] = "video_gen"
        with mock.patch.object(light_lane, "_comfy_busy") as busy:
            ok, why = light_lane.room()
        self.assertFalse(ok)
        self.assertIn("video_gen", why)
        busy.assert_not_called()

    def test_switch_off_skips_the_comfy_check(self):
        light_lane.main_engine[0] = "python"
        with mock.patch.object(light_lane, "PAUSE_ON_COMFY", False), \
                mock.patch.object(light_lane, "_comfy_busy", return_value=True):
            self.assertEqual(light_lane.room(), (True, ""))


class ComfyBusyTest(unittest.TestCase):
    def _answer(self, payload):
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = payload
        return mock.patch.object(light_lane.urllib.request, "urlopen", return_value=resp)

    def test_running_or_pending_prompt_is_busy(self):
        with self._answer(b'{"queue_running": [[1, "p"]], "queue_pending": []}'):
            self.assertTrue(light_lane._comfy_busy())
        with self._answer(b'{"queue_running": [], "queue_pending": [[2, "q"]]}'):
            self.assertTrue(light_lane._comfy_busy())

    def test_empty_queue_is_not_busy(self):
        with self._answer(b'{"queue_running": [], "queue_pending": []}'):
            self.assertFalse(light_lane._comfy_busy())

    def test_no_answer_is_not_busy(self):
        with mock.patch.object(light_lane.urllib.request, "urlopen", side_effect=OSError("refused")):
            self.assertFalse(light_lane._comfy_busy())


if __name__ == "__main__":
    unittest.main()
