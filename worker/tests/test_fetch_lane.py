"""Fetch lane (worker/fetch_lane.py): several threads, so a stalled download does
not hold the files behind it, and a stall is cut and retried. config and db are
stubbed where they cannot import, so this also runs on a bare Python."""
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


_real_or_stub("config", HEARTBEAT_SECONDS=15, ASSETS_BUCKET="assets")
_real_or_stub("db")

import httpx  # noqa: E402

import fetch_lane as fl  # noqa: E402


class Download(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(fl.time, "sleep")
        p.start()
        self.addCleanup(p.stop)

    def test_a_stalled_read_is_retried_and_logged(self):
        calls, lines = [], []

        def once(url, dest, beat):
            calls.append(url)
            if len(calls) < 3:
                raise httpx.ReadTimeout("stalled")
            return "abc", 10

        with mock.patch.object(fl, "_fetch_once", once):
            self.assertEqual(fl._download("u", "d", lambda: None, lines.append), ("abc", 10))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(lines), 2)
        self.assertIn("ReadTimeout", lines[0])

    def test_gives_up_after_every_try(self):
        with mock.patch.object(fl, "_fetch_once", side_effect=httpx.ReadTimeout("stalled")) as once:
            with self.assertRaisesRegex(RuntimeError, f"after {fl.ATTEMPTS} attempts"):
                fl._download("u", "d", lambda: None, lambda m: None)
        self.assertEqual(once.call_count, fl.ATTEMPTS)

    def test_a_refusal_is_never_retried(self):
        lines = []
        with mock.patch.object(fl, "_fetch_once", side_effect=fl.Refused("no")) as once:
            with self.assertRaises(fl.Refused):
                fl._download("u", "d", lambda: None, lines.append)
        self.assertEqual(once.call_count, 1)
        self.assertEqual(lines, [])

    def test_the_read_timeout_is_the_short_one(self):
        seen = {}

        class Stop(Exception):
            pass

        def stream(method, url, timeout, follow_redirects):
            seen.update(read=timeout.read, connect=timeout.connect)
            raise Stop

        with mock.patch.object(fl, "check_url"), mock.patch.object(fl.httpx, "stream", stream):
            with self.assertRaises(Stop):
                fl._fetch_once("https://x.volces.com/a", "d", lambda: None)
        self.assertEqual(seen, {"read": fl.READ_TIMEOUT_SECONDS, "connect": 20.0})


class Threads(unittest.TestCase):
    def test_a_stall_does_not_hold_the_files_behind_it(self):
        """Real threads and the real _loop: row 1 stalls until the other five are done."""
        rows = [{"id": str(i)} for i in range(1, 7)]
        taken, done = threading.Lock(), []
        rest_done = threading.Event()

        def claim():
            with taken:
                return rows.pop(0) if rows else None

        def run(job, log):
            if job["id"] == "1":
                self.assertTrue(rest_done.wait(5), "the stalled row held the others")
            done.append(job["id"])
            if len(done) == 5:
                rest_done.set()

        with mock.patch.object(fl, "_claim", claim), mock.patch.object(fl, "_run", run), \
                mock.patch.object(fl, "POLL_SECONDS", 0.01), mock.patch.object(fl, "THREADS", 3):
            first = fl.start(lambda m: None)
            deadline = time.time() + 6
            while len(done) < 6 and time.time() < deadline:
                time.sleep(0.01)
        self.assertEqual(sorted(done), ["1", "2", "3", "4", "5", "6"])
        self.assertEqual(done[-1], "1")
        self.assertTrue(first.name.startswith("fetch-lane"))

    def test_each_thread_is_named_for_its_own_client(self):
        started = []

        class Fake:
            def __init__(self, target, args, name, daemon):
                started.append((name, daemon))

            def start(self):
                pass

        with mock.patch.object(fl.threading, "Thread", Fake), mock.patch.object(fl, "THREADS", 4):
            fl.start(lambda m: None)
        self.assertEqual(started, [(f"fetch-lane-{i}", True) for i in range(4)])


if __name__ == "__main__":
    unittest.main()
