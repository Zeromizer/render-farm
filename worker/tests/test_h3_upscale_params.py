"""Validation messages for upscale.method h3_latent_upscale and the provisional
runtime estimates. Pure; no ComfyUI/ffmpeg/Supabase."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogen import estimate, graphs_h3  # noqa: E402

LAT = {"bucket": "renders", "path": "outputs/j-latent.mmh3"}
BASE = {"method": "h3_latent_upscale", "shorter_size": 1080, "latent": LAT}


def err(u, mode="upscale", dims=None, frames=None):
    try:
        graphs_h3.validate_upscale_params(u, mode, dims, frames)
    except ValueError as exc:
        return str(exc)
    return None


class Validation(unittest.TestCase):
    def test_defaults(self):
        u = graphs_h3.validate_upscale_params({"method": "h3_latent_upscale", "latent": LAT}, "upscale")
        self.assertEqual(u["variant"], "tile")
        self.assertEqual(u["shorter_size"], 1080)
        self.assertEqual((u["denoise"], u["steps_override"], u["seed"]), (0.0, 0, 0))
        self.assertTrue(u["force_unload"])
        self.assertEqual((u["attention"], u["fp16_accumulation"]), ("Default", "Default"))
        self.assertEqual(u["tile_width"], 640)
        self.assertEqual(u["fidelity"]["ssim_min"], 0.8)

    def test_latent_required_for_tile_and_full(self):
        for variant in ("tile", "full"):
            msg = err({"method": "h3_latent_upscale", "variant": variant, "shorter_size": 1080})
            self.assertIn("outputs/<id>-latent.mmh3", msg)
            self.assertIn("decoded", msg)
        self.assertIsNone(err({"method": "h3_latent_upscale", "variant": "decoded", "shorter_size": 1080}))

    def test_generation_mode_does_not_need_latent_ref(self):
        self.assertIsNone(err({"method": "h3_latent_upscale", "shorter_size": 1080}, "t2v"))
        self.assertIn("r2v", err({"method": "h3_latent_upscale", "shorter_size": 1080}, "r2v"))

    def test_denoise_and_steps(self):
        self.assertIn("denoise", err(dict(BASE, denoise=0.6)))
        self.assertIn("denoise", err(dict(BASE, denoise=0.01)))
        self.assertEqual(graphs_h3.validate_upscale_params(dict(BASE, denoise=0.3), "upscale")["denoise"], 0.3)
        self.assertIn("steps_override", err(dict(BASE, steps_override=40)))

    def test_variant_and_sizes(self):
        self.assertIn("variant", err(dict(BASE, variant="mega")))
        self.assertIn("factor or shorter_size", err(dict(BASE, factor=2)))
        self.assertIn("tile_width", err(dict(BASE, tile_width=600)))
        self.assertIn("fp16_accumulation", err(dict(BASE, fp16_accumulation="Fast")))
        self.assertIn("attention", err(dict(BASE, attention="H3 SLA")))

    def test_full_guard(self):
        msg = err(dict(BASE, variant="full"), dims=(480, 832), frames=124)
        self.assertIn("tile", msg)
        self.assertIn("allow_large_full", msg)
        self.assertIsNone(err(dict(BASE, variant="full"), dims=(480, 832), frames=22))
        self.assertIsNone(err(dict(BASE, variant="full", allow_large_full=True), dims=(480, 832), frames=124))

    def test_frame_grid_check(self):
        self.assertIn("17k+5", err(BASE, dims=(480, 832), frames=120))
        self.assertIsNone(err({"method": "h3_latent_upscale", "variant": "decoded"}, dims=(480, 832), frames=120))

    def test_dims_attached(self):
        u = graphs_h3.validate_upscale_params(BASE, "upscale", (480, 832), 124)
        self.assertEqual(u["_dims"]["refine"], (1088, 1888))


class Estimates(unittest.TestCase):
    def test_positive_and_monotonic(self):
        u = dict(BASE)
        a = estimate.latent_upscale_seconds(u, 480, 848, 22)
        b = estimate.latent_upscale_seconds(u, 480, 848, 124)
        c = estimate.latent_upscale_seconds(dict(u, tile_width=320, tile_height=192), 480, 848, 124)
        self.assertGreater(a, 60)
        self.assertGreater(b, a)
        self.assertGreater(c, b)   # more tiles, more time

    def test_full_variant_and_hint(self):
        p = estimate.latent_upscale_plan(dict(BASE, variant="full"), 480, 848, 22)
        self.assertEqual(p["tiles"], 1)
        h = estimate.latent_upscale_hint(BASE, 480, 848, 124)
        self.assertEqual(set(h), {"steps", "step_seconds", "load_seconds", "tail_seconds"})
        self.assertGreater(h["steps"], 3)

    def test_job_seconds_dispatch(self):
        secs = estimate.job_seconds({"video_gen": {"source": {"bucket": "b", "path": "p"}, "upscale": BASE}})
        self.assertGreater(secs, 120)
        gen = estimate.job_seconds({"video_gen": {"prompt": "x", "duration_s": 5, "resolution": "480p", "ratio": "9:16",
                                                  "save_latent": True, "upscale": {"method": "h3_latent_upscale", "shorter_size": 1080}}})
        self.assertGreater(gen, secs)
        self.assertEqual(estimate.upscale_seconds({"method": "lanczos"}, 5), estimate.LANCZOS_S)
        self.assertGreater(estimate.upscale_seconds({"method": "seedvr2"}, 5), 100)


if __name__ == "__main__":
    unittest.main()
