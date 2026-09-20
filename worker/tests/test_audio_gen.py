"""audio_gen: the exported graph is filled in by node class (so a re-export with new
node ids needs no code change), wires are left alone, and the finishing pass delivers
exactly the length asked for. Real ffmpeg on a synthetic tone; no ComfyUI, no network.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiogen import graphs  # noqa: E402

try:
    from runners import audio_gen  # noqa: E402  (needs the worker venv: db -> supabase)
except ImportError:
    audio_gen = None

needs_runner = unittest.skipUnless(audio_gen, "runner imports need the worker venv")

# The shape of a flattened "Save (API format)" export, with ids no code should depend on.
EXPORT = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "yue2_3b_int8_convrot.safetensors"}},
    "9": {"class_type": "LoraLoader", "inputs": {"lora_name": "ar_lora_inst_v3abc_comfyui.safetensors",
                                                 "strength_model": 0.0, "strength_clip": 1.0,
                                                 "model": ["4", 0], "clip": ["4", 1]}},
    "21": {"class_type": "YuE2GenerateABC", "inputs": {"clip": ["9", 1], "style": "old", "lyrics": "old",
                                                       "seed": 1, "mode": "melody", "temperature": 0.7}},
    "22": {"class_type": "YuE2GenerateMusic", "inputs": {"clip": ["9", 1], "style": "old", "lyrics": "old",
                                                         "seed": 1, "mode": "melody", "max_duration": 360.0,
                                                         "abc": ["21", 0]}},
    "23": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["22", 0]}},
    "24": {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": ["22", 1], "batch_size": 1}},
    "25": {"class_type": "KSampler", "inputs": {"model": ["9", 0], "positive": ["22", 0], "negative": ["23", 0],
                                                "latent_image": ["24", 0], "seed": 42, "steps": 32, "cfg": 1.0,
                                                "sampler_name": "dpm_2", "scheduler": "sgm_uniform", "denoise": 1.0}},
    "26": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["25", 0], "vae": ["4", 2]}},
    "30": {"class_type": "SaveAudio", "inputs": {"audio": ["26", 0], "filename_prefix": "audio/ComfyUI"}},
}


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        with open(os.path.join(self.dir, "yue2_inst.api.json"), "w", encoding="utf-8") as f:
            json.dump(EXPORT, f)
        self._here = graphs._HERE
        graphs._HERE = self.dir

    def tearDown(self):
        graphs._HERE = self._here
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_fills_both_generate_nodes_and_the_save_prefix(self):
        g, meta = graphs.build("yue2_inst", "warm lo-fi", "[intro 0:00-0:03]", 30, 77, "audio_gen/job1")
        for nid in ("21", "22"):
            self.assertEqual(g[nid]["inputs"]["style"], "warm lo-fi")
            self.assertEqual(g[nid]["inputs"]["lyrics"], "[intro 0:00-0:03]")
            self.assertEqual(g[nid]["inputs"]["seed"], 77)
            self.assertEqual(g[nid]["inputs"]["mode"], "full")
        self.assertEqual(g["22"]["inputs"]["max_duration"], 32.0)
        self.assertEqual(g["30"]["inputs"]["filename_prefix"], "audio_gen/job1")
        self.assertEqual(g["25"]["inputs"]["seed"], 77)   # one seed drives all three stages
        self.assertEqual(g["25"]["inputs"]["steps"], 32)
        self.assertEqual((meta["text"], meta["music"], meta["save"], meta["sampler"]), (2, 1, 1, 1))

    def test_refuses_the_stock_template_whose_text_and_seed_arrive_by_wire(self):
        raw = json.loads(json.dumps(EXPORT))
        raw["22"]["inputs"]["style"] = ["40", 0]      # PrimitiveStringMultiline "Text (Style)"
        raw["25"]["inputs"]["seed"] = ["41", 0]       # SeedNode
        raw["22"]["inputs"]["abc"] = ""               # the template switch, defaulting to off
        with open(os.path.join(self.dir, "yue2_inst.api.json"), "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with self.assertRaisesRegex(RuntimeError, "not flattened.*YuE2GenerateMusic.style.*abc.*KSampler.seed"):
            graphs.build("yue2_inst", "s", "t", 15, 1, "p")

    def test_the_progress_pattern_reads_token_units_and_still_reads_steps(self):
        if audio_gen is None:
            self.skipTest("runner imports need the worker venv")
        from videogen import comfy_client
        for line in ("412/750 [00:31<00:25, 13.2token/s]", " 12/32 [00:04<00:07,  2.61it/s]",
                     "3/8 [01:30<02:30, 30.1s/it]", "5/9 [00:10<00:08, 2.0s/token]"):
            self.assertIsNotNone(comfy_client._TQDM.search(line), line)

    def test_leaves_wires_and_the_template_alone(self):
        g, _ = graphs.build("yue2_inst", "s", "t", 15, 1, "p")
        self.assertEqual(g["22"]["inputs"]["abc"], ["21", 0])
        self.assertEqual(g["9"]["inputs"]["strength_clip"], 1.0)
        again, _ = graphs.load("yue2_inst")
        self.assertEqual(again["22"]["inputs"]["style"], "old")

    def test_cfg_is_only_set_when_asked(self):
        g, _ = graphs.build("yue2_inst", "s", "t", 15, 1, "p")
        self.assertNotIn("cfg_scale", g["22"]["inputs"])
        g, _ = graphs.build("yue2_inst", "s", "t", 15, 1, "p", cfg_scale=1.01)
        self.assertEqual(g["22"]["inputs"]["cfg_scale"], 1.01)

    def test_names_the_fix_when_the_export_is_missing_or_wrong(self):
        os.remove(os.path.join(self.dir, "yue2_inst.api.json"))
        with self.assertRaisesRegex(RuntimeError, "Save \\(API format\\)"):
            graphs.build("yue2_inst", "s", "t", 15, 1, "p")
        with self.assertRaisesRegex(RuntimeError, "must be one of"):
            graphs.build("suno", "s", "t", 15, 1, "p")
        with open(os.path.join(self.dir, "yue2_inst.api.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in EXPORT.items() if k != "30"}, f)
        with self.assertRaisesRegex(RuntimeError, "no SaveAudio"):
            graphs.build("yue2_inst", "s", "t", 15, 1, "p")


@needs_runner
class FinishTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _tone(self, seconds):
        path = os.path.join(self.dir, f"tone-{seconds}.flac")
        subprocess.run([audio_gen._tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi",
                        "-i", f"sine=frequency=440:duration={seconds}", "-ac", "1", path], check=True)
        return path

    def test_an_overlong_take_is_cut_to_the_request(self):
        out = os.path.join(self.dir, "out.wav")
        length, have = audio_gen.finish(self._tone(12.5), out, 10, lambda *_: None)
        self.assertAlmostEqual(length, 10.0, places=2)
        self.assertAlmostEqual(audio_gen._duration(out), 10.0, delta=0.02)
        self.assertGreater(have, 12)

    def test_a_slightly_short_take_keeps_its_own_length(self):
        out = os.path.join(self.dir, "out.wav")
        length, _ = audio_gen.finish(self._tone(9.0), out, 10, lambda *_: None)
        self.assertAlmostEqual(length, 9.0, delta=0.05)

    def test_a_take_that_gave_up_early_fails_with_the_lengths_in_the_message(self):
        with self.assertRaisesRegex(RuntimeError, "stopped at 3.0 s of the 10 s"):
            audio_gen.finish(self._tone(3.0), os.path.join(self.dir, "out.wav"), 10, lambda *_: None)


if __name__ == "__main__":
    unittest.main()
