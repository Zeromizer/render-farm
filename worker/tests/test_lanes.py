"""Light lane slots and the music lane (worker/light_lane.py, worker/music_lane.py).
The real psutil, config and db are used where they import; a module that cannot
import is stubbed, so this also runs on a bare Python. Each test patches what the
code reads, so nothing depends on the box."""
import importlib
import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _real_or_stub(name, **attrs):
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 - missing package or .env
        stub = types.ModuleType(name)
        stub.__dict__.update(attrs)
        sys.modules[name] = stub


_real_or_stub("psutil")
_real_or_stub("config", COMFYUI_URL="http://127.0.0.1:8188", LIGHT_LANE_MIN_AVAIL_RAM_GB=8, LIGHT_LANE_MIN_COMMIT_GB=8,
              LIGHT_LANE_MIN_FREE_VRAM_MIB=2048, LIGHT_LANE_SLOTS=4, KINETOGRAPH_CLI="", KINETOGRAPH_ROUTE="off",
              KINETOGRAPH_PARALLEL=2, KINETOGRAPH_MIN_FREE_VRAM_MIB=4096, MUSIC_LANE_MIN_AVAIL_RAM_GB=12,
              MUSIC_LANE_MIN_FREE_VRAM_MIB=6500, MATTE_MIN_FREE_VRAM_MIB=10500, MATTE_VRAM_WAIT_SECONDS=120,
              MATTE_LOAD_SECONDS=45,
              COMFY_ENGINES={"video_gen", "audio_gen", "footage", "voiceover", "lettering_fix", "clip_edit", "face_swap"},
              GPU_EXCLUSIVE_ENGINES={"video_gen", "audio_gen", "footage", "voiceover", "lettering_fix", "clip_edit", "face_swap", "matte"})
_real_or_stub("db")

import config  # noqa: E402
import db  # noqa: E402
import light_lane as ll  # noqa: E402
import locks  # noqa: E402
import music_lane as ml  # noqa: E402

GB = 1 << 30


class Base(unittest.TestCase):
    def setUp(self):
        ll._running.clear()
        ll._idle.set()
        ll._stop.clear()
        ll.main_engine[0] = None
        ll.music_engine[0] = None
        ll._matte_hold_until[0] = 0.0
        ml._stop.clear()
        ml._idle.set()
        self.addCleanup(ll._running.clear)
        for name, value in (("KINETOGRAPH_CLI", "cli.js"), ("KINETOGRAPH_ROUTE", "on"), ("KINETOGRAPH_PARALLEL", 2)):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)

    def box(self, ram=40, commit=40, vram=8000, comfy=False):
        patches = [mock.patch.object(ll.psutil, "virtual_memory", create=True, return_value=types.SimpleNamespace(available=ram * GB)),
                   mock.patch.object(ll, "_commit_free_gb", return_value=commit),
                   mock.patch.object(ll, "_free_vram_mib", return_value=vram),
                   mock.patch.object(ll, "_comfy_busy", return_value=comfy)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class Limits(Base):
    def test_hyperframes_runs_parallel_only_with_kinetograph_on(self):
        self.assertEqual(ll.engine_limit("hyperframes"), 2)
        with mock.patch.object(config, "KINETOGRAPH_ROUTE", "test"):
            self.assertEqual(ll.engine_limit("hyperframes"), 1)
        with mock.patch.object(config, "KINETOGRAPH_CLI", ""):
            self.assertEqual(ll.engine_limit("hyperframes"), 1)

    def test_hyperframes_runs_one_beside_a_matte(self):
        ll.main_engine[0] = "matte"
        self.assertEqual(ll.engine_limit("hyperframes"), 1)
        ll.main_engine[0] = "python"
        self.assertEqual(ll.engine_limit("hyperframes"), 2)

    def test_every_other_engine_is_one(self):
        for engine in ll.LIGHT_ENGINES - {"hyperframes"}:
            self.assertEqual(ll.engine_limit(engine), 1, engine)

    def test_a_full_engine_leaves_the_claim_list(self):
        ll._running.update(reference_extract=1, hyperframes=1)
        left = ll._open_engines()
        self.assertNotIn("reference_extract", left)
        self.assertIn("hyperframes", left)
        ll._running["hyperframes"] = 2
        self.assertNotIn("hyperframes", ll._open_engines())


class MatteClearing(Base):
    def wait_for(self, readings, seconds=120):
        """clear_for_matte with light jobs running, a VRAM reading per pass and no real waiting."""
        ll._idle.clear()
        self.addCleanup(ll._idle.set)
        logged = []
        clock = [0.0]

        def wait(timeout):
            clock[0] += timeout
            return False
        with mock.patch.object(ll, "_free_vram_mib", side_effect=readings), \
                mock.patch.object(ll.time, "monotonic", lambda: clock[0]), \
                mock.patch.object(ll._idle, "wait", wait), \
                mock.patch.object(config, "MATTE_MIN_FREE_VRAM_MIB", 10500), \
                mock.patch.object(config, "MATTE_VRAM_WAIT_SECONDS", seconds), \
                mock.patch.object(config, "MATTE_LOAD_SECONDS", 45):
            ll.clear_for_matte(logged.append)
        return logged, clock[0]

    def test_room_on_the_card_starts_at_once(self):
        logged, waited = self.wait_for([12000])
        self.assertEqual((logged, waited), ([], 0.0))

    def test_it_waits_for_renders_to_leave_room(self):
        logged, waited = self.wait_for([6000, 8000, 11000])
        self.assertEqual(waited, 2 * ll.POLL_SECONDS * 5)
        self.assertEqual(len(logged), 2)

    def test_the_wait_is_bounded(self):
        logged, waited = self.wait_for([6000] * 20, seconds=30)
        self.assertEqual(waited, 30.0)
        self.assertIn("anyway", logged[-1])

    def test_no_reading_does_not_hold_it_up(self):
        self.assertEqual(self.wait_for([None])[1], 0.0)

    def test_an_idle_lane_has_nothing_to_wait_for(self):
        with mock.patch.object(ll, "_free_vram_mib", side_effect=AssertionError("no need to read")):
            ll.clear_for_matte(lambda m: None)

    def test_the_hold_runs_past_the_wait_for_the_load(self):
        _, waited = self.wait_for([11000])
        self.assertEqual(ll._matte_hold_until[0], waited + 45)

    def test_light_jobs_are_held_during_the_wait(self):
        seen = []

        def reading():
            seen.append(ll._matte_hold_until[0])
            return 11000
        ll._idle.clear()
        self.addCleanup(ll._idle.set)
        with mock.patch.object(ll, "_free_vram_mib", reading):
            ll.clear_for_matte(lambda m: None)
        self.assertEqual(seen, [float("inf")])


class Admission(Base):
    def test_first_job_on_an_idle_box_always_starts(self):
        self.box(ram=1, commit=1, vram=1)
        self.assertEqual(ll.room(False), (True, ""))

    def test_a_second_job_meets_the_bars_with_the_loop_idle(self):
        self.box(vram=100)
        ok, why = ll.room(True)
        self.assertFalse(ok)
        self.assertIn("free VRAM", why)

    def test_a_bed_is_not_a_reason_to_wait(self):
        self.box(comfy=True)
        ll.main_engine[0] = "python"
        ll.music_engine[0] = "audio_gen"
        self.assertEqual(ll.room(True), (True, ""))

    def test_a_python_spike_on_comfy_still_pauses(self):
        self.box(comfy=True)
        ll.main_engine[0] = "python"
        self.assertFalse(ll.room(False)[0])

    def test_pause_engines_still_pause(self):
        self.box()
        ll.main_engine[0] = "video_gen"
        self.assertFalse(ll.room(True)[0])

    def test_beside_a_matte_a_light_job_meets_the_higher_bar(self):
        self.box(vram=3545)  # 9280578e came in at this and took the card to 431 MiB free
        ll.main_engine[0] = "matte"
        ok, why = ll.room(True)
        self.assertFalse(ok)
        self.assertIn("< 4096", why)

    def test_the_usual_bar_applies_beside_anything_else(self):
        self.box(vram=3545)
        ll.main_engine[0] = "python"
        self.assertEqual(ll.room(True), (True, ""))

    def test_nothing_starts_while_a_matte_loads(self):
        self.box()
        ll.main_engine[0] = "matte"
        ll._matte_hold_until[0] = time.monotonic() + 30
        self.assertEqual(ll.room(False), (False, "matte is loading"))

    def test_a_hold_outlives_no_matte(self):
        self.box()
        ll._matte_hold_until[0] = time.monotonic() + 30
        ll.main_engine[0] = "python"
        self.assertEqual(ll.room(True), (True, ""))

    def test_a_bed_makes_the_next_light_job_meet_the_bars(self):
        self.box(ram=2)
        ll.music_engine[0] = "audio_gen"
        job = {"id": "j", "engine": "hyperframes"}
        with mock.patch.object(ll.db, "claim_conditional", create=True, return_value=job) as claim:
            self.assertEqual(ll._claim_next()[0], None)
        claim.assert_not_called()


class Claiming(Base):
    def test_claim_counts_and_leave_settles(self):
        self.box()
        with mock.patch.object(ll.db, "claim_conditional", create=True, return_value={"id": "j", "engine": "hyperframes"}):
            job, why = ll._claim_next()
        self.assertEqual((job["id"], why), ("j", ""))
        self.assertFalse(ll.is_idle())
        ll._leave("hyperframes")
        self.assertTrue(ll.is_idle())

    def test_nothing_to_claim_leaves_the_lane_idle(self):
        self.box()
        with mock.patch.object(ll.db, "claim_conditional", create=True, return_value=None):
            self.assertEqual(ll._claim_next(), (None, ""))
        self.assertTrue(ll.is_idle())

    def test_a_failed_claim_leaves_the_lane_idle(self):
        self.box()
        with mock.patch.object(ll.db, "claim_conditional", create=True, side_effect=OSError("down")):
            with self.assertRaises(OSError):
                ll._claim_next()
        self.assertTrue(ll.is_idle())

    def test_a_pause_job_taken_meanwhile_stops_the_claim(self):
        self.box()
        real = ll.room

        def looked(busy):
            answer = real(busy)
            ll.main_engine[0] = "video_gen"
            return answer
        with mock.patch.object(ll, "room", side_effect=looked), \
             mock.patch.object(ll.db, "claim_conditional", create=True) as claim:
            self.assertEqual(ll._claim_next(), (None, ""))
        claim.assert_not_called()

    def test_slots_never_pass_an_engine_limit(self):
        self.box()
        queue = [{"id": str(i), "engine": "hyperframes" if i % 2 else "reference_extract"} for i in range(12)]
        guard = threading.Lock()
        peak, now = {}, {}

        def claim(include=None, **_):
            with guard:
                for i, job in enumerate(queue):
                    if job["engine"] in include:
                        return queue.pop(i)
            return None

        def work(job, log):
            with guard:
                now[job["engine"]] = now.get(job["engine"], 0) + 1
                peak[job["engine"]] = max(peak.get(job["engine"], 0), now[job["engine"]])
            time.sleep(0.02)
            with guard:
                now[job["engine"]] -= 1

        with mock.patch.object(ll.db, "claim_conditional", create=True, side_effect=claim), \
             mock.patch.object(ll, "_pending_light", side_effect=lambda: bool(queue) or (ll._stop.set() or False)), \
             mock.patch.object(ll, "POLL_SECONDS", 0.001):
            threads = [threading.Thread(target=ll._loop, args=(work, "no-such-flag", lambda m: None)) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
        ll._idle.wait(5)
        self.assertEqual(queue, [])
        self.assertEqual(peak.get("reference_extract"), 1)
        self.assertLessEqual(peak.get("hyperframes"), 2)
        self.assertTrue(ll.is_idle())


class MusicRoom(Base):
    def test_a_quiet_box_has_room(self):
        self.box()
        self.assertEqual(ml.room(), (True, ""))

    def test_light_work_and_python_are_no_reason_to_wait(self):
        self.box()
        ll._running["hyperframes"] = 2
        ll.main_engine[0] = "python"
        self.assertEqual(ml.room(), (True, ""))

    def test_waits_for_any_comfy_engine_in_the_loop(self):
        self.box()
        for engine in ("video_gen", "voiceover", "clip_edit"):
            ll.main_engine[0] = engine
            self.assertFalse(ml.room()[0], engine)

    def test_waits_for_matte(self):
        # Its VRAM, not ComfyUI: a p90 matte beside a bed is over the card
        self.box()
        ll.main_engine[0] = "matte"
        self.assertIn("matte", ml.room()[1])

    def test_matte_shares_the_lock_with_every_comfy_engine(self):
        self.assertIn("matte", config.GPU_EXCLUSIVE_ENGINES)
        self.assertLessEqual(config.COMFY_ENGINES, config.GPU_EXCLUSIVE_ENGINES)
        self.assertNotIn("python", config.GPU_EXCLUSIVE_ENGINES)

    def test_a_bed_needs_room_for_its_real_peak(self):
        self.box(vram=6000)
        self.assertIn("VRAM", ml.room()[1])
        mock.patch.stopall()
        self.box(vram=7000)
        self.assertEqual(ml.room(), (True, ""))

    def test_waits_for_comfy_busy_from_outside(self):
        self.box(comfy=True)
        self.assertIn("ComfyUI busy", ml.room()[1])

    def test_waits_on_ram_commit_and_vram(self):
        for kw, word in (({"ram": 2}, "RAM"), ({"commit": 2}, "commit"), ({"vram": 1000}, "VRAM")):
            self.box(**kw)
            self.assertIn(word, ml.room()[1])
            mock.patch.stopall()


class MusicLoop(Base):
    def run_once(self, job, gated=frozenset(), work=None):
        self.box()
        seen = []

        def done(j, log):
            seen.append((j, ll.music_engine[0], ml.is_idle()))
            ml._stop.set()

        def none(**kw):
            ml._stop.set()
            return None
        claim = mock.Mock(side_effect=(lambda **kw: job) if job else none)
        with mock.patch.object(ml, "_pending", return_value=True), mock.patch.object(ml.db, "claim_conditional", claim, create=True), \
             mock.patch.object(ml.db, "gated_engines", create=True, return_value=set(gated)), mock.patch.object(ml, "POLL_SECONDS", 0.001):
            ml._loop(work or done, "no-such-flag", lambda m: None)
        return seen, claim

    def test_a_bed_runs_and_marks_the_lane_busy_meanwhile(self):
        seen, claim = self.run_once({"id": "m", "engine": "audio_gen"})
        self.assertEqual(seen, [({"id": "m", "engine": "audio_gen"}, "audio_gen", False)])
        claim.assert_called_once_with(include={"audio_gen"})
        self.assertIsNone(ll.music_engine[0])
        self.assertTrue(ml.is_idle())

    def test_a_cloud_gated_engine_is_not_claimed(self):
        ml._stop.clear()
        with mock.patch.object(ml._stop, "wait", side_effect=lambda s=None: ml._stop.set()):
            _, claim = self.run_once({"id": "m", "engine": "audio_gen"}, gated={"audio_gen"})
        claim.assert_not_called()
        self.assertIsNone(ll.music_engine[0])

    def test_a_crash_in_the_job_does_not_leave_it_marked(self):
        def boom(j, log):
            ml._stop.set()
            raise RuntimeError("worker bug")
        self.run_once({"id": "m", "engine": "audio_gen"}, work=boom)
        self.assertIsNone(ll.music_engine[0])
        self.assertTrue(ml.is_idle())


class Recycle(Base):
    def test_the_loop_decides_for_itself(self):
        ll.main_engine[0] = "python"
        self.assertTrue(ll.may_recycle("main"))

    def test_a_bed_cannot_restart_comfy_under_a_python_spike(self):
        ll.main_engine[0] = "python"
        self.assertFalse(ll.may_recycle("music"))

    def test_a_bed_may_recycle_with_the_loop_idle(self):
        self.assertTrue(ll.may_recycle("music"))


class Locks(unittest.TestCase):
    def test_limited_is_one_semaphore_per_key(self):
        self.assertIs(locks.limited("t-a", 2), locks.limited("t-a", 2))
        s = locks.limited("t-b", 2)
        self.assertTrue(s.acquire(blocking=False))
        self.assertTrue(s.acquire(blocking=False))
        self.assertFalse(s.acquire(blocking=False))
        s.release()
        s.release()

    def test_comfy_lock_held_elsewhere_refuses_a_recycle(self):
        lock = locks.named("t-comfyui")
        got = []
        with lock:
            t = threading.Thread(target=lambda: got.append(lock.acquire(blocking=False)))
            t.start()
            t.join()
        self.assertEqual(got, [False])


if __name__ == "__main__":
    unittest.main()
