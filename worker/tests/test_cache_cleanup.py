"""Disk hygiene: asset-cache age + LRU cap, and HyperFrames scratch cleanup.

The HyperFrames CLI stages renders in tmpdir()/hf-render-* on Windows and leaks
them when a render fails or is killed; the runner points TEMP into the job's
work dir and removes it on every exit path.
"""
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assets  # noqa: E402
import config  # noqa: E402
import proc  # noqa: E402
from runners import hyperframes  # noqa: E402


class AssetCacheCleanup(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="asset-cache-test-")
        self.patches = [mock.patch.object(config, "ASSETS_DIR", self.dir)]
        for p in self.patches:
            p.start()
        self.now = time.time()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def put(self, name, size, age_days):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as f:
            f.write(b"x" * size)
        t = self.now - age_days * 86400
        os.utime(p, (t, t))
        return p

    def test_age_limit_defaults_to_seven_days(self):
        self.assertEqual(config.ASSET_CACHE_MAX_AGE_DAYS, 7)
        old = self.put("a" * 64, 10, 8)
        fresh = self.put("b" * 64, 10, 6)
        part = self.put("c" * 64 + ".part", 10, 9)
        assets.cleanup_old(lambda m: None, now=self.now)
        self.assertFalse(os.path.exists(old))
        self.assertFalse(os.path.exists(part))
        self.assertTrue(os.path.exists(fresh))

    def test_size_cap_evicts_least_recently_used_first(self):
        with mock.patch.object(config, "ASSET_CACHE_MAX_GB", 250 / 1e9):  # 250-byte cap
            oldest = self.put("1" * 64, 100, 3)
            middle = self.put("2" * 64, 100, 2)
            newest = self.put("3" * 64, 100, 1)
            assets.cleanup_old(lambda m: None, now=self.now)
        self.assertFalse(os.path.exists(oldest))
        self.assertTrue(os.path.exists(middle))
        self.assertTrue(os.path.exists(newest))

    def test_cache_hit_refreshes_last_use(self):
        p = self.put("d" * 64, 10, 6)
        assets._download("d" * 64, 10, lambda m: None)
        self.assertGreater(os.path.getmtime(p), self.now - 60)


class HyperframesScratch(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="hf-scratch-test-")
        self.repo = os.path.join(self.root, "repo")
        self.work = os.path.join(self.root, "work")
        os.makedirs(self.repo)
        os.makedirs(self.work)
        with open(os.path.join(self.repo, "index.html"), "w") as f:
            f.write("<html></html>")
        self.seen_temp = []
        self.patches = [
            mock.patch.object(hyperframes, "_npx", return_value="npx"),
            mock.patch.object(hyperframes, "_pinned_version", return_value="0.8.26"),
            mock.patch.object(hyperframes, "_ensure_browser", return_value=None),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def fake_cli(self, outcome):
        def run_streaming(cmd, cwd, on_line, cancel_check, timeout_seconds, env=None):
            temp = env["TEMP"]
            self.seen_temp.append(temp)
            scratch = os.path.join(temp, "hf-render-abc123")
            os.makedirs(scratch)
            with open(os.path.join(scratch, "frame-0001.png"), "wb") as f:
                f.write(b"\0" * 1024)
            if outcome == "fail":
                raise RuntimeError("render exited 1")
            if outcome == "cancel":
                raise proc.Canceled()
            out = cmd[cmd.index("--output") + 1]
            with open(out, "wb") as f:
                f.write(b"mp4")
        return run_streaming

    def render(self):
        job = {"params": {"format": "mp4"}}
        hb = mock.Mock(progress=0)
        return hyperframes.run(job, self.repo, self.work, hb, lambda m: None, lambda: False, 60)

    def assert_scratch_gone(self):
        self.assertEqual(len(self.seen_temp), 1)
        self.assertTrue(os.path.normcase(self.seen_temp[0]).startswith(os.path.normcase(self.work)))
        self.assertFalse(os.path.exists(self.seen_temp[0]))

    def test_success_keeps_output_and_removes_scratch(self):
        with mock.patch.object(proc, "run_streaming", self.fake_cli("ok")):
            out, ext, _ = self.render()
        self.assertTrue(os.path.exists(out))
        self.assertEqual(ext, "mp4")
        self.assert_scratch_gone()

    def test_failure_removes_scratch(self):
        with mock.patch.object(proc, "run_streaming", self.fake_cli("fail")):
            with self.assertRaises(RuntimeError):
                self.render()
        self.assert_scratch_gone()

    def test_cancel_removes_scratch(self):
        with mock.patch.object(proc, "run_streaming", self.fake_cli("cancel")):
            with self.assertRaises(proc.Canceled):
                self.render()
        self.assert_scratch_gone()

    def test_browser_ensure_keeps_system_temp(self):
        self.assertEqual(hyperframes._env().get("TEMP"), os.environ.get("TEMP"))


if __name__ == "__main__":
    unittest.main()
