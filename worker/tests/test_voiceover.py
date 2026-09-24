"""voiceover: spoken timings land on the WRITTEN tokens, whatever the voice said.
Pure alignment + param validation + the still-image matte loader. No models,
no network, no GPU.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voiceover"))

import align  # noqa: E402

try:
    from runners import voiceover  # noqa: E402  (needs the worker venv: db -> supabase)
except Exception:  # noqa: BLE001 - ImportError, or config KeyError without .env
    voiceover = None

needs_runner = unittest.skipUnless(voiceover, "runner imports need the worker venv + .env")

try:
    from PIL import Image  # noqa: E402
except ImportError:
    Image = None


class AlignTest(unittest.TestCase):
    def test_exact_words_copy_times(self):
        spoken = [(0.10, 0.40, "Create"), (0.45, 0.60, "an"), (0.62, 1.00, "event.")]
        out = align.align("Create an event.", spoken, line_start=2.0)
        self.assertEqual([w["w"] for w in out], ["Create", "an", "event"])
        self.assertEqual(out[0]["start"], 2.1)
        self.assertEqual(out[2]["end"], 3.0)
        self.assertTrue(out[2]["eos"])
        self.assertTrue(all(w["matched"] for w in out))

    def test_spoken_spelling_maps_back_to_written_brand(self):
        # The screen says e.MAS; the voice was handed "ee-Mars" and whisper heard "E-Mars".
        spoken = [(0.0, 0.3, "Meet"), (0.3, 0.5, "the"), (0.5, 1.0, "E-Mars"), (1.0, 1.2, "7.")]
        out = align.align("Meet the e.MAS 7.", spoken)
        brand = out[2]
        self.assertEqual(brand["w"], "e.MAS")
        self.assertAlmostEqual(brand["start"], 0.5, places=3)
        self.assertAlmostEqual(brand["end"], 1.0, places=3)
        self.assertFalse(brand["matched"])
        self.assertEqual(out[3]["w"], "7")

    def test_multiword_synth_token_is_split(self):
        # edge-tts returns "6.4 seconds" as ONE WordBoundary token.
        spoken = [(0.0, 0.4, "In"), (0.4, 1.4, "6.4 seconds")]
        out = align.align("In 6.4 seconds", spoken)
        self.assertEqual([w["w"] for w in out], ["In", "6.4", "seconds"])
        self.assertTrue(out[1]["matched"] and out[2]["matched"])
        self.assertLess(out[1]["end"], out[2]["end"])

    def test_unspoken_word_is_filled_between_neighbours(self):
        spoken = [(0.0, 0.3, "one"), (1.0, 1.3, "three")]
        out = align.align("one two three", spoken)
        self.assertGreaterEqual(out[1]["start"], out[0]["end"])
        self.assertLessEqual(out[1]["end"], out[2]["start"])
        self.assertFalse(out[1]["matched"])

    def test_times_are_monotonic_and_bounded_when_nothing_heard(self):
        out = align.align("a b c d", [], line_end=2.0)
        starts = [w["start"] for w in out]
        self.assertEqual(starts, sorted(starts))
        self.assertLessEqual(out[-1]["end"], 2.0)
        self.assertEqual(align.match_rate(out), 0.0)

    def test_extra_spoken_filler_is_ignored(self):
        spoken = [(0.0, 0.2, "um"), (0.2, 0.5, "hello"), (0.5, 0.9, "world")]
        out = align.align("hello world", spoken)
        self.assertEqual(out[0]["start"], 0.2)
        self.assertEqual(align.match_rate(out), 1.0)

    def test_currency_and_punctuation_display(self):
        toks = align.written_tokens('It costs "$126,000." Really?')
        self.assertEqual([t["w"] for t in toks], ["It", "costs", "$126,000", "Really"])
        self.assertEqual([t["eos"] for t in toks], [False, False, True, True])


@needs_runner
class ValidateTest(unittest.TestCase):
    def base(self, **kw):
        p = {"org_id": "o", "job_id": "j", "lines": [{"text": "Hello there."}]}
        p.update(kw)
        return p

    def test_defaults_to_omnivoice_and_fills_say(self):
        engine, lines = voiceover.validate(self.base())
        self.assertEqual(engine, "omnivoice")
        self.assertEqual(lines[0]["say"], "Hello there.")
        self.assertEqual(lines[0]["id"], "l1")

    def test_rejects_unknown_engine_empty_and_oversize(self):
        with self.assertRaises(RuntimeError):
            voiceover.validate(self.base(engine="elevenlabs"))
        with self.assertRaises(RuntimeError):
            voiceover.validate(self.base(lines=[{"text": "  "}]))
        with self.assertRaises(RuntimeError):
            voiceover.validate(self.base(lines=[{"text": "x" * 700}]))

    def test_edge_cannot_clone(self):
        with self.assertRaises(RuntimeError):
            voiceover.validate(self.base(engine="edge", ref={"bucket": "assets", "path": "p"}))


@unittest.skipUnless(Image, "Pillow needed")
class StillMatteLoaderTest(unittest.TestCase):
    def test_transparent_source_is_flattened_on_white(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "matte"))
        import matte as matte_cli  # noqa: E402

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "sticker")  # extensionless, like sha256/<hex>
            im = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
            im.paste((255, 0, 0, 255), (10, 10, 20, 20))
            im.save(src, format="PNG")
            frames, size = matte_cli.load_still(src, os.path.join(tmp, "f"))
            self.assertEqual(size, (40, 30))
            with Image.open(frames[0]) as out:
                self.assertEqual(out.mode, "RGB")
                self.assertEqual(out.getpixel((0, 0)), (255, 255, 255))
                self.assertEqual(out.getpixel((15, 15)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
