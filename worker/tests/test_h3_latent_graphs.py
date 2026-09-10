"""Pure tests for the H3 latent path graph builders (videogen/graphs_h3.py):
the default generation graph is untouched, the packet graph and the F07
upscale graphs have the expected node classes and every link resolves, the
32-aligned refine geometry and the crop maths are right. No ComfyUI needed.

    cd worker && python -m unittest discover -s tests -t .
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogen import graphs, graphs_h3  # noqa: E402

graphs_h3.TILE_DEV_DEFAULT = True   # these suites exercise tile/decoded (dev-only in production)

P_T2V = {"prompt": "A red car turns slowly on a grey floor, tyre noise", "duration_s": 5, "resolution": "480p",
         "ratio": "9:16", "seed": 7, "turbo": True}
U_TILE = {"method": "h3_latent_upscale", "variant": "tile", "shorter_size": 1080,
          "latent": {"bucket": "renders", "path": "outputs/j-latent.mmh3"}}
ABS = os.path.abspath(os.path.join(os.sep, "tmp", "x.mmh3"))


def classes(g):
    return {n["class_type"] for n in g.values()}


class DefaultGraphUntouched(unittest.TestCase):
    """graphs.build is the validated path; this locks its shape so the packet
    variant can never change it by accident."""

    def test_snapshot(self):
        g, meta = graphs.build("t2v", P_T2V, {}, "video_gen/j")
        self.assertEqual(set(g), {"unet", "lora", "clip", "vae", "avae", "cond", "guider", "sampler", "sched",
                                  "noise", "sample", "dec", "adec", "video", "save"})
        self.assertEqual(g["cond"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(g["sample"]["inputs"]["latent_image"], ["cond", 1])
        self.assertEqual(g["sched"]["inputs"], {"model": ["lora", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0})
        self.assertEqual((meta["width"], meta["height"], meta["length"]), (480, 832, 124))


class PacketGeneration(unittest.TestCase):
    def test_same_chain_plus_packet_nodes(self):
        base, bmeta = graphs.build("t2v", P_T2V, {}, "video_gen/j")
        g, meta = graphs_h3.build_generation_with_packet("t2v", P_T2V, {}, "video_gen/j", "mmh3/video_gen/j")
        graphs_h3.check_links(g)
        for key in ("unet", "lora", "clip", "vae", "avae", "guider", "sampler", "sched", "noise", "dec", "adec", "video", "save"):
            self.assertEqual(g[key], base[key], key)
        self.assertEqual(g["sample"]["inputs"]["latent_image"], ["cond", 1])
        self.assertEqual(g["cond"]["class_type"], "MMH3H3AutoCondition")
        self.assertEqual(g["cond"]["inputs"]["frames_override"], 124)
        self.assertEqual(g["cond"]["inputs"]["width_override"], 480)
        self.assertEqual(g["cond"]["inputs"]["height_override"], 832)
        self.assertTrue({"MMH3Create", "MMH3H3GenerationSettings", "MMH3H3AutoCondition", "MMH3H3ModelOptimizations",
                         "MMH3PackH3Result", "MMH3Save"} <= classes(g))
        self.assertEqual(g["pkt_settings"]["inputs"]["resolution"], "Custom")
        self.assertEqual(g["pkt_settings"]["inputs"]["resolution.width"], 480)
        self.assertEqual(g["pkt_settings"]["inputs"]["duration_seconds"], round(124 / 24, 3))
        pr = g["pkt_result"]["inputs"]
        self.assertEqual(pr["latent"], ["sample", 0])
        self.assertEqual(pr["operation"], "generate")
        self.assertEqual(pr["latent_origin"], "sampler_output")
        prof = json.loads(pr["sampling_profile_json"])
        self.assertEqual((prof["profile"], prof["steps"], prof["sampler"], prof["video_shift"]),
                         ("turbo (8 steps)", 8, "res_multistep", 12.0))
        loras = json.loads(pr["applied_loras_json"])
        self.assertEqual(loras[0]["name"], graphs.TURBO_LORAS["fl2va"][8])
        self.assertEqual(g["pkt_save"]["inputs"]["filename_prefix"], "mmh3/video_gen/j")
        self.assertEqual(meta["seed"], bmeta["seed"])
        self.assertEqual(meta["length"], bmeta["length"])

    def test_i2v_frames_linked_into_packet(self):
        g, _ = graphs_h3.build_generation_with_packet("i2v", P_T2V, {"first_frame": "a.png", "last_frame": "b.png"},
                                                       "video_gen/j", "mmh3/video_gen/j")
        graphs_h3.check_links(g)
        self.assertEqual(g["pkt_create"]["inputs"]["first_frame"], ["load_first_frame", 0])
        self.assertEqual(g["pkt_result"]["inputs"]["last_frame"], ["load_last_frame", 0])

    def test_r2v_and_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            graphs_h3.build_generation_with_packet("r2v", P_T2V, {}, "v", "m")
        with self.assertRaises(ValueError):
            graphs_h3.build_generation_with_packet("t2v", dict(P_T2V, prompt=" "), {}, "v", "m")

    def test_full_schedule_has_no_lora_provenance(self):
        g, _ = graphs_h3.build_generation_with_packet("t2v", dict(P_T2V, turbo=False), {}, "v", "m")
        self.assertNotIn("lora", g)
        self.assertEqual(json.loads(g["pkt_result"]["inputs"]["applied_loras_json"]), [])
        self.assertEqual(json.loads(g["pkt_result"]["inputs"]["sampling_profile_json"])["profile"], "standard (20 steps)")


class Geometry(unittest.TestCase):
    def test_vertical_480_to_1080(self):
        d = graphs_h3.target_dims(480, 832, {"shorter_size": 1080})
        self.assertEqual(d["req"], (1080, 1872))
        self.assertEqual(d["refine"], (1088, 1888))
        self.assertEqual(graphs_h3.crop_args(d["refine"], d["req"]), "crop=1080:1872:4:8")

    def test_landscape_768_to_1080(self):
        d = graphs_h3.target_dims(1344, 768, {"shorter_size": 1080})
        self.assertEqual(d["req"], (1890, 1080))
        self.assertEqual(d["refine"], (1920, 1088))
        self.assertAlmostEqual(d["scale"][1], 1088 / 768, places=3)

    def test_factor_already_aligned_needs_no_crop(self):
        d = graphs_h3.target_dims(832, 480, {"factor": 2})
        self.assertEqual(d["req"], (1664, 960))
        self.assertEqual(d["refine"], (1664, 960))
        self.assertIsNone(graphs_h3.crop_args(d["refine"], d["req"]))

    def test_limits(self):
        with self.assertRaises(ValueError):
            graphs_h3.target_dims(480, 848, {"shorter_size": 2160})   # 4.5x
        with self.assertRaises(ValueError):
            graphs_h3.target_dims(1344, 768, {"shorter_size": 480})   # downscale

    def test_frame_grid(self):
        self.assertTrue(graphs_h3.on_frame_grid(124))
        self.assertFalse(graphs_h3.on_frame_grid(120))
        self.assertEqual(graphs_h3.grid_frames(120), 107)
        # decoded import: AV-exact lengths only (PC 2026-09-10, mmh3_media audio floor vs round)
        self.assertEqual([graphs_h3.av_boundary_frames(n) for n in (38, 39, 73, 124, 141, 243, 260)],
                         [0, 39, 39, 90, 141, 243, 243])


class UpscaleGraphs(unittest.TestCase):
    def setUp(self):
        self.u = graphs_h3.validate_upscale_params(U_TILE, "upscale", (480, 832), 124)

    def test_tile_graph(self):
        g, meta = graphs_h3.build_latent_upscale(self.u, ABS, (480, 832), "video_gen/j-h3up")
        graphs_h3.check_links(g)
        self.assertEqual(classes(g), {"MMH3Load", "UNETLoader", "CLIPLoader", "VAELoader", "MMH3H3LatentUpscalePrepare",
                                      "MinimaxH3LatentUpscaler3D", "MMH3H3RefineLoRAs", "MMH3H3ModelOptimizations",
                                      "MMH3H3AutoCondition", "MMH3H3UpscaleRefineSampling", "MiniMaxH3SigmaShift",
                                      "BasicGuider", "KSamplerSelect", "BasicScheduler", "VAEDecode",
                                      "MMH3H3NativeTileRefine", "VAEDecodeAudio", "CreateVideo", "SaveVideo", "PreviewAny",
                                      "MMH3Inspect", "MiniMaxH3ImageToVideo"})
        # PC 2026-09-10: the tile node must not receive the packet (it would demand an F16 control
        # configuration), and per-tile sampling cannot take first/last-frame conditioning rows, so the
        # tile guider is text-only at the refine size, from the packet's own prompt.
        self.assertNotIn("packet", g["tile"]["inputs"])
        self.assertEqual(g["guider"]["inputs"]["conditioning"], ["cond_text", 0])
        self.assertEqual(g["cond_text"]["inputs"]["prompt"], ["inspect", 11])
        self.assertEqual(g["cond_text"]["inputs"]["length"], ["prep", 5])
        self.assertNotIn("first_frame", g["cond_text"]["inputs"])
        self.assertNotIn("lora", g)
        self.assertNotIn("noise", g)
        self.assertEqual(g["pkt_load"]["inputs"], {"file": "(none)", "verify": "on_access", "path_override": ABS})
        self.assertEqual(g["prep"]["inputs"]["target_width"], 1088)
        self.assertEqual(g["prep"]["inputs"]["target_height"], 1888)
        up = g["up"]["inputs"]
        self.assertEqual((up["mode"], up["mode.width"], up["mode.height"], up["model_name"]),
                         ("target dimensions", ["prep", 3], ["prep", 4], graphs_h3.UPSCALER_WEIGHT))
        self.assertTrue(up["force_unload"])
        t = g["tile"]["inputs"]
        for name in ("video", "guider", "sampler", "sigmas", "video_vae", "source_audio_latent", "frames", "seed",
                     "tile_width", "tile_height", "overlap", "context_padding", "traversal", "overlap_mode",
                     "blend_mode", "context_source"):
            self.assertIn(name, t, name)
        self.assertEqual((t["tile_width"], t["tile_height"], t["overlap"], t["context_padding"]), (640, 384, 64, 64))
        # PC 2026-09-10 default: reprocess + half_cosine (context_only + hard showed seams)
        self.assertEqual((t["traversal"], t["overlap_mode"], t["blend_mode"], t["context_source"]),
                         ("snake", "reprocess", "half_cosine", "composited"))
        self.assertEqual(g["dec_up"]["inputs"]["samples"], ["up", 0])
        self.assertEqual(g["dec"]["inputs"]["samples"], ["tile", 0])
        self.assertEqual(g["adec"]["inputs"]["samples"], ["tile", 0])
        self.assertEqual(g["sched"]["inputs"]["steps"], ["refine", 2])
        self.assertEqual(g["sched"]["inputs"]["denoise"], ["refine", 7])
        self.assertEqual(g["refine"]["inputs"]["denoise_override"], 0.0)
        self.assertEqual(g["loras"]["inputs"]["unknown_policy"], "error")
        self.assertEqual(g["opt"]["inputs"]["attention"], "Default")
        self.assertTrue(g["save"]["inputs"]["filename_prefix"].startswith("video_gen/"))
        self.assertNotIn("pkt_save", g)
        self.assertEqual(meta["crop"], "crop=1080:1872:4:8")
        self.assertEqual(meta["tiles"]["count"], graphs_h3.tile_count((1088, 1888), self.u))
        self.assertGreaterEqual(meta["tiles"]["count"], 6)

    def test_tile_blend_modes(self):
        # PC 2026-09-10: context_only/hard leaves visible seams; the node's own rules for the alternatives.
        u = graphs_h3.validate_upscale_params(dict(self.u, overlap_mode="context_only", blend_mode="hard"), "upscale")
        g, meta = graphs_h3.build_latent_upscale(u, ABS, (480, 832), "video_gen/j-h3up")
        self.assertEqual((g["tile"]["inputs"]["overlap_mode"], g["tile"]["inputs"]["blend_mode"]),
                         ("context_only", "hard"))
        self.assertEqual((meta["tiles"]["overlap_mode"], meta["tiles"]["blend_mode"]), ("context_only", "hard"))
        with self.assertRaisesRegex(ValueError, "needs overlap_mode 'reprocess'"):
            graphs_h3.validate_upscale_params(dict(self.u, overlap_mode="context_only", blend_mode="linear"), "upscale")
        with self.assertRaisesRegex(ValueError, "needs blend_mode 'linear' or 'half_cosine'"):
            graphs_h3.validate_upscale_params(dict(self.u, overlap_mode="reprocess", blend_mode="hard"), "upscale")
        with self.assertRaisesRegex(ValueError, "tile_overlap > 0"):
            graphs_h3.validate_upscale_params(dict(self.u, overlap_mode="reprocess", blend_mode="linear",
                                                   tile_overlap=0), "upscale")
        with self.assertRaisesRegex(ValueError, "upscale.blend_mode must be one of"):
            graphs_h3.validate_upscale_params(dict(self.u, blend_mode="feather"), "upscale")

    def test_full_graph(self):
        u = graphs_h3.validate_upscale_params(dict(U_TILE, variant="full", denoise=0.3, steps_override=8, seed=-1),
                                              "upscale", (480, 832), 22)
        g, meta = graphs_h3.build_latent_upscale(u, ABS, (480, 832), "video_gen/j-h3up", packet_prefix="mmh3/video_gen/j-upscaled")
        graphs_h3.check_links(g)
        self.assertNotIn("tile", g)
        self.assertEqual(g["sample"]["inputs"]["latent_image"], ["target", 0])
        self.assertEqual(g["target"]["inputs"], {"video_latent": ["up", 0], "audio_latent": ["prep", 2]})
        self.assertEqual(g["noise"]["inputs"]["noise_seed"], ["cond", 2])
        self.assertEqual(g["cond"]["inputs"]["seed_override"], -1)
        self.assertEqual(g["avcomb"]["inputs"], {"video_latent": ["avsep", 0], "audio_latent": ["avsep", 1]})
        self.assertEqual(g["dec"]["inputs"]["samples"], ["avcomb", 0])
        self.assertEqual(g["sched"]["inputs"]["steps"], 8)
        self.assertEqual(g["refine"]["inputs"]["denoise_override"], 0.3)
        self.assertEqual(g["pkt_result"]["inputs"]["operation"], "latent_upscale_refine")
        self.assertEqual(g["pkt_result"]["inputs"]["latent_origin"], "derived")
        self.assertEqual(g["pkt_save"]["inputs"]["filename_prefix"], "mmh3/video_gen/j-upscaled")
        self.assertIn("MMH3H3LatentUpscaleReport", classes(g))
        self.assertEqual(meta["operation"], "latent_upscale_refine")

    def test_decoded_graph(self):
        u = graphs_h3.validate_upscale_params({"method": "h3_latent_upscale", "variant": "decoded", "shorter_size": 1080},
                                              "upscale", (480, 832), 120)
        self.assertEqual((u["denoise"], u["steps_override"]), (0.375, 8))
        g, meta = graphs_h3.build_decoded_upscale(u, "video_gen/src.mp4", (480, 832), "video_gen/j-h3up")
        graphs_h3.check_links(g)
        self.assertEqual(g["load"]["inputs"]["file"], "video_gen/src.mp4")
        self.assertEqual(g["pkt_put"]["inputs"]["resource"], ["load", 0])
        self.assertTrue(g["pkt_put"]["inputs"]["primary"])
        self.assertEqual(g["prep"]["class_type"], "MMH3H3DecodedUpscalePrepare")
        self.assertEqual(g["prep"]["inputs"]["missing_audio_policy"], "error")
        # PC 2026-09-10: turbo_override needs a recorded (here: explicitly empty) LoRA list
        self.assertEqual(g["pkt_loras"]["inputs"], {"packet": ["pkt_put", 0], "action": "mark no LoRAs", "loras_list": "[]"})
        self.assertEqual(g["prep"]["inputs"]["packet"], ["pkt_loras", 0])
        self.assertEqual(g["loras"]["inputs"]["unknown_policy"], "continue_without_source_loras")
        self.assertEqual(g["loras"]["inputs"]["turbo_override"], graphs.TURBO_LORAS["fl2va"][8])
        self.assertEqual(g["pkt_create"]["inputs"]["prompt"], graphs_h3.DECODED_PROMPT)
        self.assertEqual(g["sched"]["inputs"]["steps"], 8)
        self.assertEqual(g["refine"]["inputs"]["denoise_override"], 0.375)
        self.assertIn("tile", g)
        self.assertEqual(meta["variant"], "decoded")

    def test_probes_can_be_disabled(self):
        g, _ = graphs_h3.build_latent_upscale(self.u, ABS, (480, 832), "v", probes=False)
        self.assertNotIn("PreviewAny", classes(g))

    def test_preflight_requirements_name_the_weight(self):
        g, _ = graphs_h3.build_latent_upscale(self.u, ABS, (480, 832), "v")
        req = graphs_h3.preflight_requirements(g)
        self.assertEqual(req["enums"][("MinimaxH3LatentUpscaler3D", "model_name")], graphs_h3.UPSCALER_WEIGHT)
        self.assertIn("MMH3H3NativeTileRefine", req["classes"])

    def test_relative_latent_path_rejected(self):
        with self.assertRaises(ValueError):
            graphs_h3.build_latent_upscale(self.u, "x.mmh3", (480, 832), "v")


if __name__ == "__main__":
    unittest.main()
