"""The /object_info preflight (videogen/h3_preflight.py) against fake schemas:
a complete install passes, a missing node pack / weight / wrong combo string
produces one message with install hints, absent optional probes are dropped."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogen import graphs_h3, h3_preflight  # noqa: E402

graphs_h3.TILE_DEV_DEFAULT = True   # these suites exercise tile/decoded (dev-only in production)

U = graphs_h3.validate_upscale_params({"method": "h3_latent_upscale", "shorter_size": 1080, "variant": "tile",
                                       "latent": {"bucket": "b", "path": "p"}}, "upscale", (480, 832), 124)
ABS = os.path.abspath(os.path.join(os.sep, "tmp", "x.mmh3"))


def fake_object_info(graph, *, drop=(), weights=(graphs_h3.UPSCALER_WEIGHT,), geometry=(graphs_h3.GEOMETRY_MODE_DIMS,)):
    """A schema that accepts every literal the graph uses, minus `drop` classes."""
    info = {}
    for node in graph.values():
        cls = node["class_type"]
        if cls in drop:
            continue
        req = info.setdefault(cls, {"input": {"required": {}, "optional": {}}})["input"]["required"]
        for name, val in node["inputs"].items():
            if "." in name:
                continue
            if isinstance(val, list):
                req[name] = ["LINK", {}]
            elif cls == graphs_h3.UPSCALER_CLASS and name == "model_name":
                req[name] = [list(weights), {}]
            elif name == "geometry_mode":
                req[name] = [list(geometry), {}]
            elif isinstance(val, str):
                prev = req.get(name)
                opts = list(prev[0]) if prev and isinstance(prev[0], list) else []
                req[name] = [sorted(set(opts) | {val, "other"}), {}]
            else:
                req[name] = ["INT" if isinstance(val, int) else "FLOAT", {}]
    return info


class Preflight(unittest.TestCase):
    def setUp(self):
        self.g, _ = graphs_h3.build_latent_upscale(U, ABS, (480, 832), "video_gen/j-h3up")

    def test_complete_install_passes(self):
        info = fake_object_info(self.g)
        g2, res = h3_preflight.run(info, self.g)
        self.assertEqual(g2, self.g)
        self.assertIsNone(h3_preflight.error_message(res))

    def test_missing_pack_and_weight_named_together(self):
        info = fake_object_info(self.g, drop=("MMH3Load", "MMH3H3LatentUpscalePrepare"), weights=("something_else.safetensors",))
        res = h3_preflight.check(info, self.g)
        msg = h3_preflight.error_message(res)
        self.assertIn("MMH3Load", msg)
        self.assertIn("MMH3H3LatentUpscalePrepare", msg)
        self.assertIn(graphs_h3.MMH3_REPO, msg)
        self.assertIn(graphs_h3.UPSCALER_WEIGHTS_URL, msg)
        self.assertIn(graphs_h3.UPSCALER_FOLDER, msg)
        with self.assertRaises(RuntimeError):
            h3_preflight.run(info, self.g)

    def test_missing_upscaler_node(self):
        info = fake_object_info(self.g, drop=(graphs_h3.UPSCALER_CLASS,))
        msg = h3_preflight.error_message(h3_preflight.check(info, self.g))
        self.assertIn(graphs_h3.UPSCALER_REPO, msg)

    def test_wrong_combo_string_lists_options(self):
        info = fake_object_info(self.g, geometry=("scale", "target_size"))
        msg = h3_preflight.error_message(h3_preflight.check(info, self.g))
        self.assertIn("geometry_mode", msg)
        self.assertIn("target_size", msg)
        self.assertIn("graphs_h3.py", msg)

    def test_optional_probe_dropped_silently(self):
        info = fake_object_info(self.g, drop=(graphs_h3.PROBE_CLASS,))
        g2, res = h3_preflight.run(info, self.g)
        self.assertNotIn("probe_refine", g2)
        self.assertEqual(sorted(res["dropped_optional"]), ["probe_prep", "probe_refine", "probe_tile"])

    def test_missing_required_input_reported(self):
        info = fake_object_info(self.g)
        info["MMH3H3NativeTileRefine"]["input"]["required"]["new_knob"] = ["INT", {}]
        msg = h3_preflight.error_message(h3_preflight.check(info, self.g))
        self.assertIn("new_knob", msg)

    def test_v3_combo_shapes(self):
        info = fake_object_info(self.g)
        info[graphs_h3.UPSCALER_CLASS]["input"]["required"]["model_name"] = [{"type": "COMBO", "options": [graphs_h3.UPSCALER_WEIGHT]}, {}]
        info["MMH3Load"]["input"]["required"]["file"] = ["COMBO", {"options": ["(none)"]}]
        self.assertIsNone(h3_preflight.error_message(h3_preflight.check(info, self.g)))


if __name__ == "__main__":
    unittest.main()


class UploadCombosAndInstalledSchema(unittest.TestCase):
    def test_upload_file_combo_is_not_enum_checked(self):
        # LoadVideo.file lists the input folder (empty on a fresh install) and carries
        # video_upload: true; the runner uploads the file right before /prompt.
        graph = {"load": {"class_type": "LoadVideo", "inputs": {"file": "clip.mp4"}}}
        info = {"LoadVideo": {"input": {"required": {"file": ["COMBO", {"options": [], "video_upload": True}]}}}}
        res = h3_preflight.check(info, graph)
        self.assertEqual(res["bad_enum"], [])
        self.assertIsNone(h3_preflight.error_message(res))

    def test_plain_combo_still_enum_checked(self):
        graph = {"n": {"class_type": "X", "inputs": {"mode": "nope"}}}
        info = {"X": {"input": {"required": {"mode": ["COMBO", {"options": ["a", "b"]}]}}}}
        self.assertEqual(len(h3_preflight.check(info, graph)["bad_enum"]), 1)

    def test_all_four_graphs_pass_against_the_pc_fixture(self):
        """tests/fixtures/object_info_pc.json is the trimmed /object_info of the render PC
        (ComfyUI 0.34.0, mmh3_media bca81b8c, upscaler d7c01b90, 2026-09-10). The same four
        graphs smoke.py --preflight-only builds; MMH3Put's five advanced string inputs are
        required there, and LoadVideo.file is an upload combo."""
        import json
        fx = os.path.join(os.path.dirname(__file__), "fixtures", "object_info_pc.json")
        if not os.path.exists(fx):
            self.skipTest("fixture not present")
        with open(fx, encoding="utf-8") as f:
            info = json.load(f)
        gen, _ = graphs_h3.build_generation_with_packet("t2v", {"prompt": "x"}, {}, "video_gen/t", "mmh3/t")
        u = graphs_h3.validate_upscale_params({"method": "h3_latent_upscale", "shorter_size": 1080, "variant": "tile",
                                               "latent": {"bucket": "b", "path": "p"}}, "upscale")
        tile, _ = graphs_h3.build_latent_upscale(u, ABS, (480, 832), "video_gen/t")
        full, _ = graphs_h3.build_latent_upscale(dict(u, variant="full"), ABS, (480, 832), "video_gen/t")
        dec, _ = graphs_h3.build_decoded_upscale(
            graphs_h3.validate_upscale_params({"method": "h3_latent_upscale", "shorter_size": 1080, "variant": "decoded"}, "upscale"),
            "x.mp4", (480, 832), "video_gen/t")
        for name, graph in (("generation+packet", gen), ("tile", tile), ("full", full), ("decoded", dec)):
            res = h3_preflight.check(info, graph)
            self.assertIsNone(h3_preflight.error_message(res), f"{name}: {h3_preflight.error_message(res)}")
            self.assertEqual(res["unknown_inputs"], [], name)
