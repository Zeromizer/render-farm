"""Routing and fallback of worker/runners/kinetograph.py. Nothing here renders: the
subprocess is a small Python child, so this runs on a bare Python and every way
out of render() is tested for what it leaves behind for HyperFrames."""
import importlib
import json
import os
import sys
import tempfile
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


_real_or_stub("config", KINETOGRAPH_CLI="", KINETOGRAPH_ROUTE="off", KINETOGRAPH_CONCURRENCY=4,
              KINETOGRAPH_TIMEOUT_SECONDS=600, KINETOGRAPH_MIN_FREE_VRAM_MIB=4096, KINETOGRAPH_MIN_COMMIT_GB=8)

import config  # noqa: E402
import proc  # noqa: E402
from runners import kinetograph as kt  # noqa: E402

BUNDLE = """
FX.define("TRAN-09", { swap(t) { FXKit.gl(t, root, frag); } });
FX.define("TYPE-01", { in(t) { t.to(el, {y: 0}); } });
FX.define("CAM-18", { in(t) { x.putImageData(id, 0, 0); } });
"""
ROOT = '<div id="root" data-composition-id="main" data-start="0" data-duration="6" data-fps="30"></div>'


class Enabled(unittest.TestCase):
    def check(self, cli, route, params, want):
        with mock.patch.object(config, "KINETOGRAPH_CLI", cli), mock.patch.object(config, "KINETOGRAPH_ROUTE", route):
            self.assertEqual(kt.enabled(params), want)

    def test_off_unless_configured(self):
        self.check("", "on", {}, False)
        self.check("cli.js", "off", {"renderer": "kinetograph"}, False)

    def test_test_mode_takes_only_jobs_that_ask(self):
        self.check("cli.js", "test", {}, False)
        self.check("cli.js", "test", {"renderer": "kinetograph"}, True)

    def test_on_takes_all_but_jobs_that_refuse(self):
        self.check("cli.js", "on", {}, True)
        self.check("cli.js", "on", {"renderer": "hyperframes"}, False)


class Shape(unittest.TestCase):
    def reasons(self, params=None, entry="index.html", version="0.8.155", quality="standard"):
        return kt.shape_reasons(params or {}, entry, version, quality)

    def test_the_replayed_shapes_pass(self):
        self.assertEqual(self.reasons(), [])
        self.assertEqual(self.reasons(entry="video-9x16.html"), [])
        self.assertEqual(self.reasons({"crf": 10}, quality="high"), [])
        self.assertEqual(self.reasons(quality="draft"), [])

    def test_shapes_with_no_equivalent_stay(self):
        for params in ({"fps": 24}, {"resolution": "portrait-4k"}, {"variables": {"a": 1}}, {"gpu": True},
                       {"project_dir": "fx/TYPE-01"}, {"format": "webm"}):
            self.assertTrue(self.reasons(params), params)

    def test_non_root_entry_and_older_projects_stay(self):
        self.assertTrue(self.reasons(entry="compositions/feed-4x5.html"))
        self.assertTrue(self.reasons(version="0.8.26"))
        self.assertTrue(self.reasons(version="0.8.154"))
        self.assertEqual(self.reasons(version="0.8.160"), [])

    def test_high_needs_a_crf(self):
        self.assertTrue(self.reasons(quality="high"))


class Content(unittest.TestCase):
    def test_a_plain_composition_passes(self):
        self.assertEqual(kt.content_reasons('<div class="a"></div><script>tl.to(a, {x: 1});</script>', ""), [])

    def test_hf_modules_come_from_the_bundle(self):
        self.assertEqual(kt.hf_modules(BUNDLE), {"TRAN-09", "CAM-18"})

    def test_a_module_is_found_by_mention_whatever_the_call(self):
        # 408d8b58: the delivery used TRAN-09, a call-shaped regex did not see it
        self.assertEqual(kt.content_reasons('const shot = {fx: "TRAN-09"}; run(shot);', BUNDLE), ["fx TRAN-09"])

    def test_a_safe_module_passes(self):
        self.assertEqual(kt.content_reasons('FX.enter(tl, "TYPE-01", el);', BUNDLE), [])

    def test_fx_calls_with_no_bundle_stay(self):
        self.assertEqual(kt.content_reasons('FX.enter(tl, "TYPE-01", el);', ""), ["fx calls without a bundle to read"])

    def test_each_blocker(self):
        for text, why in (('c.getContext("webgl2")', "webgl"), ("c.getContext('webgpu')", "webgl"),
                          ('<script src="shaders.iife.js">', "webgpu"), ("new THREE.Scene()", "three"),
                          ('<iframe src="a.html">', "iframe"), ('<audio data-fx-chain="x">', "data-fx-chain"),
                          ("g.putImageData(i, 0, 0)", "putImageData"),
                          (".a{mix-blend-mode:screen;transform-style:preserve-3d}", "blend mode with 3D")):
            self.assertIn(why, kt.content_reasons(text, ""), text)

    def test_blend_alone_or_3d_alone_passes(self):
        self.assertEqual(kt.content_reasons(".a{mix-blend-mode:screen}", ""), [])
        self.assertEqual(kt.content_reasons(".a{transform-style:preserve-3d}", ""), [])


class ReadProject(unittest.TestCase):
    def test_bundle_kept_apart_and_vendored_dirs_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "node_modules"))
            for name, body in (("index.html", "COMP"), ("fx-kit.js", "KIT"), ("gsap.min.js", "GSAP"),
                               ("node_modules/x.js", "DEP"), ("clip.mp4", "BIN")):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    f.write(body)
            self.assertEqual(kt.read_project(d), ("COMP", "KIT"))


class Duration(unittest.TestCase):
    def test_reads_the_root_tag(self):
        self.assertEqual(kt.expected_seconds(ROOT), 6.0)

    def test_none_without_a_statement(self):
        self.assertIsNone(kt.expected_seconds('<div data-duration="3"></div>'))


class Render(unittest.TestCase):
    """render() end to end, with a child Python standing in for `node cli.js`."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        d = self.dir.name
        self.tool = os.path.join(d, "tool")
        self.cli = os.path.join(self.tool, "node_modules", "kinetograph", "dist", "cli.js")
        os.makedirs(os.path.dirname(self.cli))
        open(self.cli, "w").close()
        self.project = os.path.join(d, "project")
        self.tmp = os.path.join(d, "tmp")
        os.makedirs(self.project)
        os.makedirs(self.tmp)
        self.out = os.path.join(d, "out.mp4")
        with open(os.path.join(self.project, "index.html"), "w", encoding="utf-8") as f:
            f.write(ROOT)
        self.job = {"params": {}}
        self.hb = types.SimpleNamespace(progress=0)
        self.logged = []
        for name, value in (("KINETOGRAPH_CLI", self.cli), ("KINETOGRAPH_ROUTE", "on")):
            p = mock.patch.object(config, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(kt, "room", return_value=(True, ""))
        p.start()
        self.addCleanup(p.stop)

    def child(self, body):
        """Make the render command run `body` in a child Python; argv[1] is the output."""
        script = os.path.join(self.dir.name, "child.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(body)
        real = proc.run_streaming

        def fake(cmd, **kw):
            self.cmd = cmd
            self.env = kw["env"]
            return real([sys.executable, script, self.out], **kw)
        p = mock.patch.object(kt.proc, "run_streaming", fake)
        p.start()
        self.addCleanup(p.stop)

    def render(self, cancel=lambda: False, timeout=600):
        return kt.render(self.job, self.project, "index.html", "0.8.155", "standard", self.out, self.tmp, None,
                         self.hb, self.logged.append, {"cancel_check": cancel, "timeout_seconds": timeout})

    def probe_says(self, **info):
        p = mock.patch.object(kt.subprocess, "run",
                              return_value=types.SimpleNamespace(stdout=json.dumps(info)))
        p.start()
        self.addCleanup(p.stop)

    GOOD = {"streams": [{"codec_type": "video"}], "format": {"duration": "6.0"}}

    def test_a_good_render_is_returned_and_labelled(self):
        self.child("import sys\nprint('rendering frame 90/180', flush=True)\nopen(sys.argv[1], 'wb').write(b'x')\n")
        self.probe_says(**self.GOOD)
        self.assertEqual(self.render(), self.out)
        self.assertEqual(self.hb.progress, 47)
        self.assertTrue(self.job["_done_phase"].startswith("done: kinetograph "))
        self.assertIn("--strict", self.cmd)
        self.assertEqual(self.cmd[self.cmd.index("--crf") + 1], "16")
        self.assertEqual(self.cmd[self.cmd.index("--concurrency") + 1], "4")

    def test_the_render_cannot_see_the_workers_secrets(self):
        self.child("import sys\nopen(sys.argv[1], 'wb').write(b'x')\n")
        self.probe_says(**self.GOOD)
        with mock.patch.dict(os.environ, {"SUPABASE_SERVICE_KEY": "secret", "PATH": os.environ["PATH"]}):
            self.render()
        self.assertNotIn("SUPABASE_SERVICE_KEY", self.env)
        self.assertEqual(self.env["TEMP"], self.tmp)

    def test_a_strict_refusal_falls_back_and_clears_the_file(self):
        self.child("import sys\nopen(sys.argv[1], 'wb').write(b'partial')\nprint('unsupported content')\nsys.exit(1)\n")
        self.hb.progress = 40
        self.assertIsNone(self.render())
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.hb.progress, 3)
        self.assertTrue(self.job["_done_phase"].startswith("done: hyperframes (kinetograph: exit code 1"))

    def test_a_silent_hang_is_killed_and_falls_back(self):
        self.child("import time\ntime.sleep(60)\n")
        with mock.patch.object(config, "KINETOGRAPH_TIMEOUT_SECONDS", 1):
            self.assertIsNone(self.render())
        self.assertIn("exceeded 1s", self.job["_done_phase"])

    def test_cancel_goes_through(self):
        self.child("import time\nfor i in range(60):\n    print('x', flush=True)\n    time.sleep(1)\n")
        with mock.patch.object(kt.proc.time, "monotonic", side_effect=iter(range(0, 100000, 10))):
            with self.assertRaises(proc.Canceled):
                self.render(cancel=lambda: True)
        self.assertNotIn("_done_phase", self.job)

    def test_exit_0_with_no_file_falls_back(self):
        self.child("pass\n")
        self.assertIsNone(self.render())
        self.assertIn("no output file", self.job["_done_phase"])

    def test_exit_0_with_the_wrong_length_falls_back(self):
        self.child("import sys\nopen(sys.argv[1], 'wb').write(b'x')\n")
        self.probe_says(streams=[{"codec_type": "video"}], format={"duration": "2.0"})
        self.assertIsNone(self.render())
        self.assertIn("duration 2.00 s", self.job["_done_phase"])
        self.assertFalse(os.path.exists(self.out))

    def test_lost_audio_falls_back(self):
        with open(os.path.join(self.project, "index.html"), "a", encoding="utf-8") as f:
            f.write('<audio src="a.m4a"></audio>')
        self.child("import sys\nopen(sys.argv[1], 'wb').write(b'x')\n")
        self.probe_says(**self.GOOD)
        self.assertIsNone(self.render())
        self.assertIn("output has none", self.job["_done_phase"])

    def test_no_room_never_starts_it(self):
        with mock.patch.object(kt, "room", return_value=(False, "free VRAM 1000 MiB < 4096")), \
             mock.patch.object(kt.proc, "run_streaming") as run:
            self.assertIsNone(self.render())
        run.assert_not_called()

    def test_a_missing_cli_stays_on_hyperframes(self):
        os.remove(self.cli)
        with mock.patch.object(kt.proc, "run_streaming") as run:
            self.assertIsNone(self.render())
        run.assert_not_called()

    def test_off_reads_nothing(self):
        with mock.patch.object(config, "KINETOGRAPH_ROUTE", "off"), mock.patch.object(kt, "read_project") as read:
            self.assertIsNone(self.render())
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
