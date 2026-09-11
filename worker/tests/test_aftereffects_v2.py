"""text_overlay_v2: schema bounds/normalization, fake-host pipeline plumbing,
reopen verification logic. Real-AE behaviour is covered by the smoke scenes.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest tests.test_aftereffects_v2
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("SUPABASE_URL", "https://example.invalid")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aftereffects import media, pipeline, schema, schema_v2  # noqa: E402
from aftereffects.errors import AEError  # noqa: E402
from aftereffects.fake_host import FakeHost  # noqa: E402

PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090775"
    "3de0000000c49444154789c63f8cfc00000030101009d9c2f7f0000000049454e44ae426082")


def v2_request(**over):
    req = {
        "schema_version": 1, "recipe": "text_overlay_v2", "composition": "Main",
        "org_id": "org-1", "job_id": "job-1", "assets": [],
        "settings": {
            "composition": {"width": 320, "height": 180, "fps": 30, "duration_s": 1.0,
                            "motion_blur": {"enabled": True, "shutter_angle": 180}},
            "layers": [
                {"id": "lamp", "kind": "shape", "shape": "ellipse", "size": [40, 36], "color": "#D82026",
                 "position": [280, 40], "in_f": 0, "out_f": 30,
                 "effects": [{"type": "glow", "color": "#D82026", "radius": 10, "intensity": 0.6},
                             {"type": "bevel_highlight", "thickness": 2, "angle": 90, "intensity": 0.47}],
                 "keyframes": {"brightness": [{"f": 0, "v": 1.0}, {"f": 2, "v": 1.12}, {"f": 8, "v": 1.0}]}},
                {"id": "word", "kind": "text", "text": "BLUE", "font": "Arial-BoldMT", "size": 60, "color": "#30579A",
                 "stretch": [1.38, 2.0], "position": [160, 20], "anchor": {"x": "center", "y": "top"},
                 "in_f": 3, "out_f": 30, "clip": {"bottom": 150}, "stroke": {"color": "#A7B2BC", "width": 0.35},
                 "keyframes": {"scale": [{"f": 3, "v": [103, 103], "ease": "ease_out"}, {"f": 9, "v": [100, 100], "ease": "hold"}]},
                 "motion_blur": True},
                {"id": "num", "kind": "text", "text": "3", "font": "ArialMT", "size": 80, "color": "#FFFFFF",
                 "position": [60, 120], "in_f": 0, "out_f": 12, "fade_in_f": 3},
            ],
        },
    }
    req.update(over)
    return req


def quiet(_m):
    pass


class SchemaV2(unittest.TestCase):
    def test_normalizes(self):
        r = schema.validate_request(v2_request())
        L = {x["id"]: x for x in r["settings"]["layers"]}
        self.assertEqual(L["word"]["in_s"], 0.1)
        self.assertEqual(L["word"]["anchor"], {"x": "center", "y": "top"})
        self.assertEqual(L["lamp"]["anchor"], {"x": "center", "y": "center"})
        self.assertEqual(L["word"]["clip"], {"left": 0, "top": 0, "right": 320, "bottom": 150})
        self.assertEqual(L["word"]["keyframes"]["scale"][0]["ease"], {"in": [0.0, 0.1], "out": [0.0, 33.3]})
        self.assertEqual(L["word"]["keyframes"]["scale"][1]["ease"], "hold")
        self.assertEqual(L["num"]["keyframes"]["opacity"], [
            {"f": 0, "v": 0.0, "t": 0.0, "ease": "linear"}, {"f": 3, "v": 100.0, "t": 0.1, "ease": "linear"}])
        self.assertEqual(L["lamp"]["effects"][0]["match_name"], "ADBE Glo2")
        self.assertEqual(L["lamp"]["effects"][0]["composite"], "on_top")
        self.assertEqual(r["settings"]["composition"]["motion_blur"]["shutter_phase"], -90)
        self.assertEqual([t["id"] for t in schema.text_layers(r)], ["word", "num"])
        self.assertEqual(schema.layer_ids(r), ["lamp", "word", "num"])

    def test_seconds_accepted_frames_win(self):
        req = v2_request()
        req["settings"]["layers"][2].update(in_s=0.5, out_s=0.9)
        del req["settings"]["layers"][2]["in_f"]
        del req["settings"]["layers"][2]["out_f"]
        r = schema.validate_request(req)
        self.assertEqual((r["settings"]["layers"][2]["in_f"], r["settings"]["layers"][2]["out_f"]), (15, 27))

    def test_rejects(self):
        def bad(mutate, code="INVALID_REQUEST", rx=None):
            req = v2_request()
            mutate(req["settings"])
            with self.assertRaises(AEError) as cm:
                schema.validate_request(req)
            self.assertEqual(cm.exception.code, code, str(cm.exception))
            if rx:
                self.assertRegex(str(cm.exception), rx)
        bad(lambda s: s["layers"].append({"id": "word", "kind": "text", "text": "x", "font": "ArialMT"}), rx="duplicate")
        bad(lambda s: s["layers"][1].update(matte={"layer": "nope"}), code="MATTE_TARGET_MISSING")
        bad(lambda s: s["layers"][1].update(matte={"layer": "word"}), code="MATTE_TARGET_MISSING")
        bad(lambda s: s["layers"][1]["keyframes"]["scale"].insert(0, {"f": 1, "v": [1, 1]}), code="KEYFRAME_INVALID")
        bad(lambda s: s["layers"][0]["effects"].append({"type": "inner_shadow"}), code="EFFECT_MISSING")
        bad(lambda s: s["layers"][0]["effects"][0].update(bogus=1), rx="unknown parameter")
        bad(lambda s: s["layers"][1].update(stretch=[0, 1]), rx="stretch")
        bad(lambda s: s["layers"][1].update(clip={"top": 100, "bottom": 50}), rx="empty clip")
        bad(lambda s: s["layers"][1].update(keyframes={"effects.5.radius": [{"f": 3, "v": 1}]}), rx="effect")
        bad(lambda s: s["layers"][1].update(keyframes={"opacity": [{"f": 5, "v": 1}, {"f": 5, "v": 2}]}), rx="increase")
        bad(lambda s: s["layers"][1].update(anchor={"y": "diagonal"}), rx="anchor")
        bad(lambda s: s["layers"][0].update(anchor={"y": "baseline"}), rx="baseline")
        bad(lambda s: s["layers"][1].update(keyframes={"scale": [{"f": 3, "v": [1, 1], "ease": "bounce"}]}), rx="ease")
        bad(lambda s: s.update(layers=[]), rx="layers")
        bad(lambda s: s["layers"][2].update(fade_in_f=7, fade_out_f=7), rx="fades")
        bad(lambda s: s.update(output_extras={"proof_frames_f": list(range(13))}), rx="proof")

    def test_no_revise_for_v2(self):
        r = schema.validate_request(v2_request())
        with self.assertRaisesRegex(AEError, "no revision script"):
            schema.validate_changes([{"id": "word", "text": "x"}], r)
        self.assertEqual(len(pipeline.recipe_revision("text_overlay_v2")), 12)
        self.assertNotEqual(pipeline.recipe_revision("text_overlay_v2"), pipeline.recipe_revision("text_overlay_v1"))

    def test_capabilities(self):
        from aftereffects import capabilities
        r = capabilities.report()
        self.assertIn("text_overlay_v2", r["recipes"])
        self.assertIn("glow", r["recipes"]["text_overlay_v2"]["effects"])
        self.assertEqual(r["recipes"]["text_overlay_v1"]["revision"], pipeline.recipe_revision("text_overlay_v1"))
        self.assertEqual(len(r["capabilities_sha256"]), 64)


class VerifyInspect(unittest.TestCase):
    def test_matches_and_detects(self):
        r = schema.validate_request(v2_request())
        model = {"settings": r["settings"], "assets": {}}
        summary = FakeHost._comp_summary_v2(model, "Main")
        self.assertEqual(schema_v2.verify_inspect(r["settings"], summary), [])
        broken = copy.deepcopy(summary)
        for L in broken["layers"]:
            if L["name"] == "word":
                L["inner"]["font"] = "ArialMT"
                L["masks"] = 0
            if L["name"] == "lamp":
                L["effects"] = L["effects"][:1]
                L["keys"] = {}
        probs = schema_v2.verify_inspect(r["settings"], broken)
        self.assertTrue(any("font" in p for p in probs))
        self.assertTrue(any("masked precomp" in p for p in probs))
        self.assertTrue(any("effects" in p for p in probs))
        self.assertTrue(any("brightness has None keys" in p for p in probs))
        broken["motion_blur"] = False
        self.assertIn("comp motion blur off", schema_v2.verify_inspect(r["settings"], broken))


class PipelineV2Fake(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="ae2-")
        os.makedirs(os.path.join(self.td, "inputs"))

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_full_run_with_matte_and_proofs(self):
        req = v2_request(assets=[{"name": "wear", "bucket": "assets", "path": "sha256/x", "kind": "image"}])
        req["settings"]["layers"].append({"id": "wear_l", "kind": "image", "asset": "wear", "position": [160, 60],
                                          "in_f": 3, "out_f": 30, "matte": {"layer": "word", "mode": "alpha"}})
        req["settings"]["output_extras"] = {"proof_frames_f": [0, 15]}
        staged = os.path.join(self.td, "inputs", "wear.png")
        with open(staged, "wb") as f:
            f.write(PNG_1x1)
        r = schema.validate_request(req)
        res = pipeline.run(r, self.td, FakeHost(log=quiet, ffmpeg=media.tool("ffmpeg")), {"wear": staged}, quiet,
                           lambda: False, 120)
        self.assertTrue(res["ok"])
        self.assertEqual(res["checks"]["frames"], 30)
        self.assertEqual([p["frame"] for p in res["proofs"]], [0, 15])
        for p in res["proofs"]:
            self.assertTrue(os.path.exists(p["path"]))
        names = {L["name"] for L in res["inspect"]["editable_layers"]}
        self.assertEqual(names, {"lamp", "word", "num", "wear_l"})
        self.assertEqual(res["provenance"]["recipe"], "text_overlay_v2")
        self.assertEqual(len(res["provenance"]["capabilities_sha256"]), 64)
        with open(res["manifest"], encoding="utf-8") as f:
            self.assertTrue(json.load(f)["ok"])

    def test_v1_still_runs(self):
        from tests.test_aftereffects import small_request
        r = schema.validate_request(small_request())
        res = pipeline.run(r, self.td, FakeHost(log=quiet, ffmpeg=media.tool("ffmpeg")), {}, quiet, lambda: False, 120)
        self.assertTrue(res["ok"])
        self.assertEqual(res["proofs"], [])


if __name__ == "__main__":
    unittest.main()
