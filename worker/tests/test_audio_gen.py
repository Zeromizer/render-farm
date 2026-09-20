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

from audiogen import graphs, progress  # noqa: E402

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
                                                       "seed": 1, "mode": "melody", "temperature": 0.7,
                                                       "max_abc_tokens": 8192}},
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

    def test_the_score_is_capped_so_a_runaway_seed_cannot_cost_five_minutes(self):
        # 3 of 7 measured seeds never emitted the score's end token and ran to 8192.
        self.assertEqual([graphs.abc_token_cap(s) for s in (15, 30, 45, 120, 180)],
                         [3000, 3000, 3300, 6300, 8192])
        g, _ = graphs.build("yue2_inst", "s", "t", 45, 1, "p")
        self.assertEqual(g["21"]["inputs"]["max_abc_tokens"], 3300)

    def test_the_committed_export_is_flattened_and_fills(self):
        graphs._HERE = self._here          # the real file, not the fixture
        if not os.path.exists(os.path.join(self._here, "yue2_inst.api.json")):
            self.skipTest("no committed export yet")
        g, meta = graphs.build("yue2_inst", "warm lo-fi", "[intro 0:00-0:03]", 30, 5, "audio_gen/x")
        self.assertEqual((meta["text"], meta["music"], meta["save"], meta["sampler"]), (2, 1, 1, 1))
        music = next(n for n in g.values() if n["class_type"] == "YuE2GenerateMusic")
        self.assertEqual((music["inputs"]["style"], music["inputs"]["max_duration"]), ("warm lo-fi", 32.0))

    def test_leaves_wires_and_the_template_alone(self):
        g, _ = graphs.build("yue2_inst", "s", "t", 15, 1, "p")
        self.assertEqual(g["22"]["inputs"]["abc"], ["21", 0])
        self.assertEqual(g["9"]["inputs"]["strength_clip"], 1.0)
        again, _ = graphs.load("yue2_inst")
        self.assertEqual(again["22"]["inputs"]["style"], "old")

    def test_cfg_is_set_only_when_asked_and_only_where_the_export_has_it(self):
        # ComfyUI v0.36.0 has no cfg_scale on the node; sending one fails validation.
        g, meta = graphs.build("yue2_inst", "s", "t", 15, 1, "p", cfg_scale=1.01)
        self.assertNotIn("cfg_scale", g["22"]["inputs"])
        self.assertEqual(meta["cfg"], 0)

        newer = json.loads(json.dumps(EXPORT))
        newer["22"]["inputs"]["cfg_scale"] = 1.0
        with open(os.path.join(self.dir, "yue2_inst.api.json"), "w", encoding="utf-8") as f:
            json.dump(newer, f)
        g, _ = graphs.build("yue2_inst", "s", "t", 15, 1, "p")
        self.assertEqual(g["22"]["inputs"]["cfg_scale"], 1.0)
        g, meta = graphs.build("yue2_inst", "s", "t", 15, 1, "p", cfg_scale=1.01)
        self.assertEqual((g["22"]["inputs"]["cfg_scale"], meta["cfg"]), (1.01, 1))

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


# Lines as ComfyUI logged them on the first queue job (258fccdf: 15 s bed, 151 s wall).
ABC = "YuE2 ABC sampling:  28%|##8       | 2280/8192 [01:43<04:28, 22.0token/s]"
MUSIC = "YuE2 music sampling:  38%|###7      | 161/425 [00:07<00:12, 21.0token/s]"
DIFF = " 78%|#######8  | 25/32 [00:06<00:01,  4.10it/s]"


class ProgressTests(unittest.TestCase):
    def test_reads_the_stage_from_the_description_and_tokens_per_second_as_a_rate(self):
        stage, i, n, rate = progress.parse(ABC)
        self.assertEqual((stage, i, n), ("abc", 2280, 8192))
        self.assertAlmostEqual(rate, 1 / 22.0)        # was read as 22 SECONDS per token
        self.assertEqual(progress.parse(MUSIC)[:3], ("music", 161, 425))
        self.assertEqual(progress.parse(DIFF)[:3], ("diffusion", 25, 32))
        self.assertAlmostEqual(progress.parse("3/8 [01:30<02:30, 30.1s/it]")[3], 30.1)
        self.assertIsNone(progress.parse("got prompt"))

    def test_the_stage_is_found_however_the_line_is_dressed(self):
        # tqdm redraws with a carriage return; ComfyUI may prefix a time or a level. The
        # first live job read its score stage as "rendering audio 311/3000" because of "\r".
        for dressed in ("\r" + ABC, "[2026-09-20 21:44:31] " + ABC, "\r\x1b[A" + ABC, "  " + ABC):
            self.assertEqual(progress.parse(dressed)[:3], ("abc", 2280, 8192), repr(dressed))
        self.assertEqual(progress.parse("\r" + MUSIC)[0], "music")
        self.assertEqual(progress.parse("\r" + DIFF)[0], "diffusion")

    def test_the_exact_bytes_comfyui_logged_on_the_first_live_job(self):
        # /internal/logs/raw, job cd8a0075: leading "\r", block glyphs, no space before the unit,
        # and the KSampler line carries no description at all.
        raw = {
            "\rYuE2 ABC sampling:  50%|\u2588\u2588\u2588\u2588\u2588     | 1514/3000 [01:16<01:14, 19.93token/s]":
                ("abc", 1514, 3000, 1 / 19.93),
            "\rYuE2 music sampling: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2589| 423/425 [00:21<00:00, 20.13token/s]":
                ("music", 423, 425, 1 / 20.13),
            "\r 97%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u258b| 31/32 [00:05<00:00,  5.00it/s]":
                ("diffusion", 31, 32, 1 / 5.0),
        }
        for line, (stage, i, n, rate) in raw.items():
            got = progress.parse(line)
            self.assertEqual(got[:3], (stage, i, n), repr(line))
            self.assertAlmostEqual(got[3], rate, places=6)
        label, left = progress.estimate(*progress.parse(next(iter(raw))), duration_s=15)
        self.assertEqual(label, "writing the score (1514 tokens)")
        self.assertTrue(50 < left < 80, left)          # it had 40 s to go; a normal score is assumed

    def test_time_left_is_minutes_not_days(self):
        # That job reported ~2828 min left here and ~91 min left with 20 s to go.
        label, left = progress.estimate(*progress.parse(ABC.replace("2280", "1000")), duration_s=15)
        self.assertEqual(label, "writing the score (1000 tokens)")
        self.assertTrue(60 < left < 110, left)         # ~55 s of score + ~19 s of music + diffusion
        label, left = progress.estimate(*progress.parse(MUSIC), duration_s=15)
        self.assertEqual(label, "writing the music 161/425")
        self.assertTrue(20 < left < 25, left)
        label, left = progress.estimate(*progress.parse(DIFF), duration_s=15)
        self.assertEqual(label, "rendering audio 25/32")
        self.assertTrue(left < 5, left)

    def test_a_runaway_score_is_costed_to_the_cap_not_to_a_normal_ending(self):
        _, normal = progress.estimate("abc", 2100, 3000, 1 / 21, duration_s=15)
        _, runaway = progress.estimate("abc", 2300, 3000, 1 / 21, duration_s=15)
        self.assertGreater(runaway, normal)            # it now expects 3000, not 2200
        self.assertLess(runaway, 70)

    def test_the_reader_makes_wait_print_the_honest_numbers(self):
        now = [100.0]
        entries = []
        reader = progress.Reader(15, lambda: entries, clock=lambda: now[0])
        self.assertIsNone(reader("2026-09-20T00:00:00"))        # nothing logged yet: "loading model"
        entries.append({"t": "2026-09-20T00:00:30", "m": MUSIC})
        now[0] = 200.0                                           # 100 s in
        step, total, rate = reader("2026-09-20T00:00:00")
        _, left = progress.estimate(*progress.parse(MUSIC), duration_s=15)
        self.assertAlmostEqual((total - step) * rate, left, places=6)   # wait(): eta = (n-i)*rate + tail
        self.assertAlmostEqual(step / total, 100 / (100 + left), delta=0.002)
        self.assertEqual(reader.label, "writing the music 161/425")
        # Lines from before this prompt belong to the previous job.
        self.assertIsNone(progress.Reader(15, lambda: entries)("2026-09-20T00:01:00"))


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

    def test_the_fade_scales_with_the_bed(self):
        self.assertEqual([audio_gen.fade_seconds(s) for s in (8, 15, 30, 45, 120)], [0.8, 0.8, 1.5, 2.0, 2.0])

    def test_a_slightly_short_take_keeps_its_own_length(self):
        out = os.path.join(self.dir, "out.wav")
        length, _ = audio_gen.finish(self._tone(9.0), out, 10, lambda *_: None)
        self.assertAlmostEqual(length, 9.0, delta=0.05)

    def test_a_take_that_gave_up_early_fails_with_the_lengths_in_the_message(self):
        with self.assertRaisesRegex(RuntimeError, "stopped at 3.0 s of the 10 s"):
            audio_gen.finish(self._tone(3.0), os.path.join(self.dir, "out.wav"), 10, lambda *_: None)


if __name__ == "__main__":
    unittest.main()
