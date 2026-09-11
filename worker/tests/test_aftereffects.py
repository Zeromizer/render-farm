"""aftereffects engine tests without After Effects: schema bounds, font
preflight, the whole pipeline on the fake host (real ffmpeg for ProRes 4444,
VP9 alpha, alpha statistics, contact sheet), the failure paths the handoff
asks for (missing input, JSX failure, cancellation, timeout, incomplete
render) and the runner's retry / publish guard.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest tests.test_aftereffects
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile

os.environ.setdefault("SUPABASE_URL", "https://example.invalid")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proc  # noqa: E402
from aftereffects import ae_host, fonts, media, pipeline, schema  # noqa: E402
from aftereffects.errors import AEError  # noqa: E402
from aftereffects.fake_host import FakeHost  # noqa: E402

PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090775"
    "3de0000000c49444154789c63f8cfc00000030101009d9c2f7f0000000049454e44ae426082")


def small_request(**over):
    req = {
        "schema_version": 1, "recipe": "text_overlay_v1", "composition": "Main",
        "org_id": "org-1", "job_id": "job-1", "assets": [],
        "settings": {
            "composition": {"width": 320, "height": 180, "fps": 24, "duration_s": 1.0},
            "shapes": [{"id": "pill", "size": [120, 40], "radius": 20, "color": "#FF6A00", "position": [160, 120],
                        "in_s": 0.1, "out_s": 1.0, "fade_in_s": 0.2, "fade_out_s": 0.2, "scale_from": 50}],
            "texts": [{"id": "headline", "text": "Hello\nWorld", "font": "ArialMT", "size": 30, "color": "#FFFFFF",
                       "position": [160, 60], "in_s": 0.0, "out_s": 1.0, "fade_in_s": 0.2, "slide_from": [0, 20]},
                      {"id": "cta", "text": "Go", "font": "Arial-BoldMT", "size": 20, "position": [160, 150],
                       "in_s": 0.3, "out_s": 0.9}],
            "images": [],
        },
    }
    req.update(over)
    return req


def quiet(_msg):
    pass


class Schema(unittest.TestCase):
    def test_default_normalizes(self):
        r = schema.validate_request(small_request())
        self.assertEqual(r["settings"]["composition"]["frames"], 24)
        self.assertEqual(r["settings"]["texts"][0]["justify"], "center")
        self.assertEqual(r["settings"]["texts"][1]["color"], "#FFFFFF")
        self.assertEqual(r["output_profile"], "prores4444_alpha")

    def test_rejects(self):
        bad = [
            dict(schema_version=2),
            dict(recipe="anything_v9"),
            dict(org_id=None),
            dict(settings=dict(small_request()["settings"], composition={"width": 321, "height": 180, "fps": 24, "duration_s": 1})),
            dict(settings=dict(small_request()["settings"], composition={"width": 320, "height": 180, "fps": 31, "duration_s": 1})),
        ]
        for over in bad:
            with self.assertRaises(AEError) as cm:
                schema.validate_request(small_request(**over))
            self.assertEqual(cm.exception.code, "INVALID_REQUEST", over)
        s = small_request()["settings"]
        s["texts"][1]["id"] = "headline"
        with self.assertRaisesRegex(AEError, "duplicate layer id"):
            schema.validate_request(small_request(settings=s))
        s = small_request()["settings"]
        s["images"] = [{"id": "logo", "asset": "logo"}]
        with self.assertRaisesRegex(AEError, "not in the request's assets"):
            schema.validate_request(small_request(settings=s))
        s = small_request()["settings"]
        s["texts"][0]["fade_in_s"] = 0.9
        s["texts"][0]["fade_out_s"] = 0.9
        with self.assertRaisesRegex(AEError, "fades"):
            schema.validate_request(small_request(settings=s))
        s = small_request()["settings"]
        s["texts"][0]["text"] = "x" * 501
        with self.assertRaisesRegex(AEError, "longer than"):
            schema.validate_request(small_request(settings=s))
        with self.assertRaisesRegex(AEError, "'..'"):
            schema.validate_request(small_request(assets=[{"name": "a", "bucket": "assets", "path": "x/../y"}]))

    def test_changes(self):
        r = schema.validate_request(small_request())
        ch = schema.validate_changes([{"id": "headline", "text": "New", "in_s": 0.2}], r)
        self.assertEqual(ch[0]["text"], "New")
        with self.assertRaisesRegex(AEError, "not a text layer"):
            schema.validate_changes([{"id": "pill", "text": "x"}], r)
        with self.assertRaisesRegex(AEError, "changes nothing"):
            schema.validate_changes([{"id": "cta"}], r)
        with self.assertRaisesRegex(AEError, "not a layer"):
            schema.validate_changes([{"id": "nope", "text": "x"}], r)


class Fonts(unittest.TestCase):
    def test_windows_fonts(self):
        self.assertEqual(fonts.missing(["ArialMT", "Arial-BoldMT"]), [])
        self.assertEqual(fonts.missing(["NoSuchFont-Aimotion"]), ["NoSuchFont-Aimotion"])


class PipelineFake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = media.tool("ffmpeg")

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="ae-")
        os.makedirs(os.path.join(self.td, "inputs"))

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def run_pipe(self, host, req=None, assets=None, timeout=120, cancel=lambda: False, **kw):
        req = schema.validate_request(req or small_request())
        return pipeline.run(req, self.td, host, assets or {}, quiet, cancel, timeout, **kw)

    def test_full_run_sequence(self):
        phases = []
        res = self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg), phase=phases.append)
        self.assertTrue(res["ok"])
        self.assertEqual(phases, list(pipeline.PHASES))
        c = res["checks"]
        self.assertEqual((c["width"], c["height"], c["frames"], c["fps"]), (320, 180, 24, 24.0))
        # prores_ks writes 4444 from yuva444p10le; the decoder reports the stream as 12-bit
        self.assertIn(c["pix_fmt"], ("yuva444p10le", "yuva444p12le"))
        self.assertEqual(c["alpha"]["raw_mid_frame"]["min"], 0)
        self.assertEqual(c["alpha"]["raw_mid_frame"]["max"], 255)
        self.assertEqual(res["review_info"]["raw_alpha_frame"]["min"], 0)
        self.assertEqual(c["codec"], "prores")
        self.assertGreater(c["alpha"]["frames_with_transparent_pixels"], 0)
        self.assertGreater(c["alpha"]["frames_with_opaque_pixels"], 0)
        self.assertGreater(c["alpha"]["frames_fully_transparent"], 0, "frame 0 is empty before the first fade")
        self.assertEqual(res["review_info"]["codec"], "vp9")
        self.assertEqual(res["review_info"]["pix_fmt"], "yuva420p")
        for k in ("master", "review", "contact_sheet", "bundle", "manifest", "project"):
            self.assertTrue(os.path.exists(res[k]), k)
        with zipfile.ZipFile(res["bundle"]) as z:
            names = set(z.namelist())
        self.assertIn("project.aep", names)
        self.assertIn("settings.json", names)
        self.assertIn("recipes/text_overlay_v1.jsx", names)
        self.assertIn("recipes/lib.jsx", names)
        layers = {L["name"]: L for L in res["inspect"]["editable_layers"]}
        self.assertEqual(layers["headline"]["kind"], "text")
        self.assertEqual(layers["headline"]["text"], "Hello\nWorld")
        self.assertEqual(res["provenance"]["host"]["kind"], "fake")
        self.assertEqual(res["provenance"]["recipe_revision"], pipeline.recipe_revision("text_overlay_v1"))
        self.assertIn("render_s", res["timings"])
        with open(res["manifest"], encoding="utf-8") as f:
            self.assertTrue(json.load(f)["ok"])

    def test_full_run_mov_path(self):
        res = self.run_pipe(FakeHost(log=quiet, om_kind="mov", ffmpeg=self.ffmpeg))
        self.assertEqual(res["checks"]["source"]["kind"], "mov")
        self.assertEqual(res["checks"]["source"]["codec"], "qtrle")
        self.assertEqual(res["checks"]["frames"], 24)

    def test_image_asset_and_missing_input(self):
        req = small_request(assets=[{"name": "logo", "bucket": "assets", "path": "sha256/abc", "kind": "image"}])
        req["settings"]["images"] = [{"id": "logo_l", "asset": "logo", "position": [40, 40], "in_s": 0, "out_s": 1}]
        with self.assertRaises(AEError) as cm:
            self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg), req=req, assets={})
        self.assertEqual(cm.exception.code, "ASSET_MISSING")
        # outside the workspace -> refused even if it exists
        outside = os.path.join(tempfile.gettempdir(), "ae-outside.png")
        with open(outside, "wb") as f:
            f.write(PNG_1x1)
        with self.assertRaises(AEError) as cm:
            self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg), req=req, assets={"logo": outside})
        self.assertEqual(cm.exception.code, "INVALID_REQUEST")
        # staged + wrong hash -> mismatch
        staged = os.path.join(self.td, "inputs", "logo.png")
        with open(staged, "wb") as f:
            f.write(PNG_1x1)
        req["assets"][0]["sha256"] = "0" * 64
        with self.assertRaises(AEError) as cm:
            self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg), req=req, assets={"logo": staged})
        self.assertEqual(cm.exception.code, "ASSET_HASH_MISMATCH")
        # right hash -> renders with the footage layer
        req["assets"][0]["sha256"] = pipeline.sha256_file(staged)
        res = self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg), req=req, assets={"logo": staged})
        self.assertTrue(res["ok"])
        self.assertIn("logo", res["provenance"]["inputs"])
        with zipfile.ZipFile(res["bundle"]) as z:
            self.assertIn("assets/logo.png", z.namelist())

    def test_font_missing_before_ae(self):
        req = small_request()
        req["settings"]["texts"][0]["font"] = "NoSuchFont-Aimotion"
        host = FakeHost(log=quiet, ffmpeg=self.ffmpeg)
        with self.assertRaises(AEError) as cm:
            self.run_pipe(host, req=req)
        self.assertEqual(cm.exception.code, "FONT_MISSING")
        self.assertEqual(host.launched, [], "AE must not be launched for a font we know is missing")

    def test_jsx_failure_reports_line(self):
        host = FakeHost(log=quiet, ffmpeg=self.ffmpeg, fail_author=("JSX_ERROR", "Unable to call 'addProperty'", 88))
        with self.assertRaises(AEError) as cm:
            self.run_pipe(host)
        self.assertEqual(cm.exception.code, "JSX_ERROR")
        self.assertIn(":88)", str(cm.exception))
        self.assertIn("addProperty", str(cm.exception))
        with open(os.path.join(self.td, "out", "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertFalse(m["ok"])
        self.assertEqual(m["error"]["jsx_line"], 88)
        # a coded failure keeps its code
        host = FakeHost(log=quiet, ffmpeg=self.ffmpeg, fail_author=("OM_TEMPLATE_MISSING", "no template", 200))
        with self.assertRaises(AEError) as cm:
            self.run_pipe(host)
        self.assertEqual(cm.exception.code, "OM_TEMPLATE_MISSING")

    def test_no_manifest(self):
        with self.assertRaises(AEError) as cm:
            self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg, no_manifest_stage="author"))
        self.assertEqual(cm.exception.code, "AE_NO_MANIFEST")

    def test_cancel_during_render(self):
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(proc.Canceled):
            self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg, hang_stage="render"), cancel=cancel)
        with open(os.path.join(self.td, "out", "manifest.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["error"]["code"], "CANCELED")
        self.assertFalse(os.path.exists(os.path.join(self.td, "out", "overlay-master.mov")))

    def test_timeout_during_author(self):
        with self.assertRaises(proc.TimedOut):
            self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg, hang_stage="author"), timeout=0.5)
        with open(os.path.join(self.td, "out", "manifest.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["error"]["code"], "TIMEOUT")

    def test_incomplete_render_is_not_success(self):
        with self.assertRaises(AEError) as cm:
            self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg, drop_frames=2))
        self.assertEqual(cm.exception.code, "RENDER_INCOMPLETE")

    def test_recipe_revision_pin(self):
        with self.assertRaises(AEError) as cm:
            self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg), req=small_request(recipe_revision="0" * 12))
        self.assertEqual(cm.exception.code, "RECIPE_REVISION_MISMATCH")
        res = self.run_pipe(FakeHost(log=quiet, ffmpeg=self.ffmpeg),
                            req=small_request(recipe_revision=pipeline.recipe_revision("text_overlay_v1")))
        self.assertTrue(res["ok"])

    def test_revise_one_text_and_timing(self):
        host = FakeHost(log=quiet, ffmpeg=self.ffmpeg)
        res = self.run_pipe(host)
        ws2 = os.path.join(self.td, "rev")
        os.makedirs(os.path.join(ws2, "project"))
        os.makedirs(os.path.join(ws2, "inputs"))
        src = os.path.join(ws2, "project", "source.aep")
        shutil.copy2(res["project"], src)
        req = schema.validate_request(small_request())
        rev = pipeline.revise(req, [{"id": "headline", "text": "Changed", "in_s": 0.25, "out_s": 0.95}], src, {},
                              ws2, host, quiet, lambda: False, 120)
        self.assertTrue(rev["ok"])
        layers = {L["name"]: L for L in rev["inspect"]["editable_layers"]}
        self.assertEqual(layers["headline"]["text"], "Changed")
        self.assertAlmostEqual(layers["headline"]["in_s"], 0.25)
        self.assertEqual(rev["checks"]["frames"], 24)
        self.assertEqual(rev["revised_request"]["settings"]["texts"][0]["text"], "Changed")
        with self.assertRaises(AEError):
            pipeline.revise(req, [{"id": "headline", "in_s": 0.99, "out_s": 0.5}], src, {}, ws2, host, quiet,
                            lambda: False, 120)


class SlotTest(unittest.TestCase):
    def test_second_holder_waits_then_times_out(self):
        with ae_host.Slot(wait_seconds=5):
            err = []

            def other():
                try:
                    with ae_host.Slot(wait_seconds=1):
                        err.append("acquired")
                except AEError as e:
                    err.append(e.code)

            t = threading.Thread(target=other)
            t.start()
            t.join(10)
            self.assertEqual(err, ["AE_SLOT_TIMEOUT"])
        with ae_host.Slot(wait_seconds=1):
            pass


class RunnerTest(unittest.TestCase):
    """The farm adapter around the pipeline, with db / storage stubbed."""

    def setUp(self):
        from runners import aftereffects as runner
        self.runner = runner
        self.td = tempfile.mkdtemp(prefix="ae-run-")
        self.uploads = []
        self.phases = []
        self.row = {"status": "processing", "attempts": 1, "cancel_requested": False}
        self._orig = (runner.fetch_row, runner.db.upload_file, runner.db.set_phase, runner.make_host,
                      runner.stage_assets)
        runner.fetch_row = lambda jid: dict(self.row)
        runner.db.upload_file = lambda remote, local, mime: self.uploads.append((remote, mime, os.path.getsize(local))) or remote
        runner.db.set_phase = lambda jid, phase, progress=None: self.phases.append(phase)
        runner.make_host = lambda log: FakeHost(log=quiet, ffmpeg=media.tool("ffmpeg"))
        runner.stage_assets = lambda request, inputs, log, download=None: {}

    def tearDown(self):
        r = self.runner
        r.fetch_row, r.db.upload_file, r.db.set_phase, r.make_host, r.stage_assets = self._orig
        shutil.rmtree(self.td, ignore_errors=True)

    def job(self, attempts=1):
        return {"id": "j-1", "engine": "aftereffects", "attempts": attempts,
                "params": {"aftereffects": small_request()}}

    def test_success_publishes_sidecars_then_master(self):
        class HB:
            progress = 0
        hb = HB()
        out, ext, mime = self.runner.run(self.job(), None, self.td, hb, quiet, lambda: False, 120)
        self.assertEqual((ext, mime), ("mov", "video/quicktime"))
        self.assertTrue(out.endswith("j-1-master.mov"))
        self.assertEqual([u[0] for u in self.uploads],
                         ["outputs/j-1-review.webm", "outputs/j-1-bundle.zip", "outputs/j-1-contact.png",
                          "outputs/j-1-manifest.json"])
        self.assertIn("uploading", self.phases)
        self.assertTrue(os.path.isdir(os.path.join(self.td, "attempt-1")))
        self.assertGreaterEqual(hb.progress, 90)

    def test_superseded_claim_publishes_nothing(self):
        class HB:
            progress = 0
        self.row["attempts"] = 2   # reclaimed and re-run elsewhere while we rendered
        with self.assertRaises(AEError) as cm:
            self.runner.run(self.job(attempts=1), None, self.td, HB(), quiet, lambda: False, 120)
        self.assertEqual(cm.exception.code, "CLAIM_SUPERSEDED")
        self.assertEqual(self.uploads, [])

    def test_cancel_at_publish(self):
        class HB:
            progress = 0
        self.row["cancel_requested"] = True
        with self.assertRaises(proc.Canceled):
            self.runner.run(self.job(), None, self.td, HB(), quiet, lambda: False, 120)
        self.assertEqual(self.uploads, [])

    def test_retry_uses_fresh_attempt_and_cleans_stale(self):
        stale = os.path.join(self.td, "attempt-1")
        os.makedirs(stale)
        with open(os.path.join(stale, "pids.json"), "w") as f:
            json.dump([[999999, "AfterFX.exe"]], f)   # no such process: nothing is killed
        with open(os.path.join(stale, "leftover.txt"), "w") as f:
            f.write("x")
        self.row["attempts"] = 2

        class HB:
            progress = 0
        out, _, _ = self.runner.run(self.job(attempts=2), None, self.td, HB(), quiet, lambda: False, 120)
        self.assertIn("attempt-2", out)
        self.assertFalse(os.path.exists(stale), "stale attempt workspace must be gone")

    def test_invalid_request_fails_fast(self):
        class HB:
            progress = 0
        job = self.job()
        job["params"]["aftereffects"]["schema_version"] = 7
        with self.assertRaises(AEError) as cm:
            self.runner.run(job, None, self.td, HB(), quiet, lambda: False, 120)
        self.assertEqual(cm.exception.code, "INVALID_REQUEST")
        self.assertEqual(self.uploads, [])


if __name__ == "__main__":
    unittest.main()
