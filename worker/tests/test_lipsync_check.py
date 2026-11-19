"""Tests for the lipsync_check engine.

Pure parts: params validation (runners/lipsync_check.py, needs the worker venv for its imports),
the verdict rules and per-window summary (lipsync/lipsync_check.py, stdlib only), and the speech
measure on a synthetic wav (needs numpy + scipy, i.e. the engine's venv).

End to end (opt-in, downloads ~145 MB of weights + their demo clip): LIPSYNC_E2E=1 with the engine
venv's python reproduces syncnet_python's README demo and the offset sign:

    example.avi             -> offset_frames_25 -3 (their demo clip: voice 3 frames early)
    voice delayed 160 ms    -> about 0
    voice advanced 160 ms   -> about -7

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest tests.test_lipsync_check
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

WORKER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKER)

_spec = importlib.util.spec_from_file_location("lipsync_compute", os.path.join(WORKER, "lipsync", "lipsync_check.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

try:
    from runners import lipsync_check as runner  # noqa: E402  (needs the worker venv: db -> supabase)
except (ImportError, KeyError):  # KeyError: config.py reads SUPABASE_URL at import
    runner = None

try:
    import numpy as np
    from scipy.io import wavfile
    HAVE_NP = True
except ImportError:
    HAVE_NP = False


class Verdict(unittest.TestCase):
    def test_no_face(self):
        self.assertEqual(lc.verdict(False, 9, 0, 1)[0], "no_face")

    def test_in_sync_inside_the_asymmetric_tolerance(self):
        for off in (-lc.OFFSET_EARLY_OK, 0, lc.OFFSET_LATE_OK):
            self.assertEqual(lc.verdict(True, 8.0, off, 0.9)[0], "in_sync", off)

    def test_offset_names_frames_and_direction(self):
        v, why = lc.verdict(True, 8.0, lc.OFFSET_LATE_OK + 1, 0.9)
        self.assertEqual(v, f"off_by_{lc.OFFSET_LATE_OK + 1}_frames")
        self.assertIn(f"{(lc.OFFSET_LATE_OK + 1) * 40} ms late", why)
        v, why = lc.verdict(True, 8.0, -(lc.OFFSET_EARLY_OK + 1), 0.9)
        self.assertEqual(v, f"off_by_{lc.OFFSET_EARLY_OK + 1}_frames")
        self.assertIn("early", why)

    def test_low_confidence_is_no_signal_whatever_the_offset(self):
        for off in (9, 0):
            self.assertEqual(lc.verdict(True, lc.CONF_MIN - 0.01, off, 0.9)[0], "no_sync_signal")

    def test_calibration_populations_split(self):
        # 2026-11-19: unsynced Seedance-over-VO max 2.16, weakest real in-sync presenter 2.93.
        self.assertEqual(lc.verdict(True, 2.16, 0, 0.9)[0], "no_sync_signal")
        v, why = lc.verdict(True, 2.93, 2, 0.9)
        self.assertEqual(v, "in_sync")
        self.assertIn("weak", why)

    def test_silence_under_the_face(self):
        v, why = lc.verdict(True, 9.0, 0, lc.SPEECH_MIN / 2)
        self.assertEqual(v, "no_sync_signal")
        self.assertIn("speech", why)


class Summary(unittest.TestCase):
    def test_no_tracks(self):
        s = lc.summarise({"label": "w1", "start_s": 0, "end_s": 3}, [])
        self.assertFalse(s["face"])
        self.assertEqual(s["verdict"], "no_face")

    def test_best_confidence_track_decides(self):
        t = lambda conf, v: {"start_s": 0, "end_s": 2, "face_px": 200, "offset_frames_25": 0, "offset_frames": 0,
                             "offset_ms": 0, "confidence": conf, "min_dist": 7, "speech_fraction": 0.9,
                             "verdict": v, "reason": ""}
        s = lc.summarise({"label": "w1"}, [t(0.8, "no_sync_signal"), t(6.0, "in_sync")])
        self.assertTrue(s["face"])
        self.assertEqual(s["verdict"], "in_sync")
        self.assertEqual(len(s["tracks"]), 2)


@unittest.skipUnless(HAVE_NP, "needs numpy + scipy (the engine venv)")
class Speech(unittest.TestCase):
    def test_tone_then_silence(self):
        sr = 16000
        t = np.arange(sr) / sr
        tone = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        a = np.concatenate([tone, np.zeros(sr, dtype=np.int16)])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.wav")
            wavfile.write(p, sr, a)
            self.assertAlmostEqual(lc.speech_fraction(p), 0.5, delta=0.05)
            self.assertGreater(lc.speech_fraction(p, 0, 1), 0.95)


@unittest.skipUnless(runner, "runner imports need the worker venv")
class Params(unittest.TestCase):
    def test_needs_bucket_and_path(self):
        with self.assertRaises(RuntimeError):
            runner.parse_params({"check": {"bucket": "assets"}})

    def test_bucket_whitelist(self):
        with self.assertRaises(RuntimeError):
            runner.parse_params({"check": {"bucket": "secrets", "path": "x"}})

    def test_windows_cleaned(self):
        b, p, w, fps = runner.parse_params({"check": {"bucket": "renders", "path": "outputs/a.mp4", "fps": 30,
                                                       "windows": [{"start_s": "4", "end_s": 8, "label": "shot 2/a"}]}})
        self.assertEqual((b, p, fps), ("renders", "outputs/a.mp4", 30.0))
        self.assertEqual(w, [{"label": "shot_2_a", "start_s": 4.0, "end_s": 8.0}])

    def test_short_window_refused(self):
        with self.assertRaises(RuntimeError):
            runner.parse_params({"check": {"bucket": "assets", "path": "sha256/x",
                                           "windows": [{"start_s": 1, "end_s": 1.5}]}})


@unittest.skipUnless(os.environ.get("LIPSYNC_E2E") == "1", "set LIPSYNC_E2E=1 (downloads weights, runs SyncNet)")
class EndToEnd(unittest.TestCase):
    """Their README demo through our driver, then the sign check with the voice shifted 4 frames."""

    def _measure(self, video, d):
        out = os.path.join(d, os.path.basename(video) + ".json")
        models = os.environ.get("LIPSYNC_MODELS") or os.path.join(d, "models")
        subprocess.run([sys.executable, os.path.join(WORKER, "lipsync", "lipsync_check.py"), video, out,
                        "--models-dir", models, "--work-dir", os.path.join(d, "work")], check=True)
        with open(out, encoding="utf-8") as f:
            return json.load(f)["windows"][0]

    def test_demo_and_sign(self):
        import urllib.request

        ffmpeg = lc.find_ffmpeg_tool("ffmpeg")
        with tempfile.TemporaryDirectory() as d:
            ex = os.path.join(d, "example.avi")
            urllib.request.urlretrieve("http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi", ex)
            w = self._measure(ex, d)
            self.assertEqual(w["offset_frames_25"], -3)
            self.assertGreater(w["confidence"], 7)
            late = os.path.join(d, "late4.avi")
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", ex, "-af", "adelay=160:all=1", "-c:v", "copy", late],
                           check=True)
            self.assertIn(self._measure(late, d)["offset_frames_25"], (0, 1))
            early = os.path.join(d, "early4.avi")
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", ex, "-af", "atrim=start=0.16,asetpts=PTS-STARTPTS,apad",
                            "-shortest", "-c:v", "copy", early], check=True)
            self.assertIn(self._measure(early, d)["offset_frames_25"], (-7, -8))


if __name__ == "__main__":
    unittest.main()
