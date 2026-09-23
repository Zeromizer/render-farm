"""_cut audio: the selected audio is the audio of the selected frames.

Synthetic sources make the CONTENT checkable, not only the duration: video
frame k is flat grey at luma 16 + 4k (limited range), and the audio during frame k's slot is a
pure tone at 400 + 100k Hz. A cut [a, b) must therefore deliver grey levels
a..b-1 exactly once each and, slot by slot, tones a..b-1 - at every source
sample rate, because the trim offsets used to be computed at 48 kHz but applied
before the resample (a 32 kHz source was cut 1.5x too far in) and the end of
the cut was never bounded.

Tolerance: the output is AAC in MP4. The encoder primes 1024 samples, which
the MP4 edit list hides on decode, and a stream ends on a whole 1024-sample
frame, so decoded length may exceed the exact frame-derived length by up to
one AAC frame. Anything beyond that is drift, not encoder rounding.
"""
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners import footage  # noqa: E402
from videogen import segments  # noqa: E402

R24 = {"num": 24, "den": 1}
FF = segments._tool("ffmpeg")
SLOT = 2000                    # 48 kHz samples per 24 fps frame
AAC_FRAME = 1024
FRAMES = 51                    # the delivered length of a raw90/head39 take


def tone(k):
    return 400 + 100 * k


def make_source(path, rate=None, frames=FRAMES, silent_until=0):
    """Frame k: grey 4k. Audio (if rate): tone(k) during frame k's slot,
    silence before frame `silent_until`."""
    dur = frames / 24
    cmd = [FF, "-v", "error", "-y", "-f", "lavfi",
           "-i", f"nullsrc=s=64x48:r=24:d={dur},geq=lum='16+N*4':cb=128:cr=128"]
    if rate:
        expr = (f"if(lt(floor(t*24),{silent_until}),0,"
                f"0.5*sin(2*PI*(400+100*floor(t*24))*t))")
        cmd += ["-f", "lavfi", "-i", f"aevalsrc='{expr}':s={rate}:c=mono:d={dur}",
                "-c:a", "aac", "-ar", str(rate), "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-qp", "0", "-g", "1", "-pix_fmt", "yuv420p", "-frames:v", str(frames), path]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def grey_levels(path):
    r = subprocess.run([FF, "-v", "error", "-i", path, "-fps_mode", "passthrough",
                        "-vf", "scale=1:1:flags=area,format=gray", "-f", "rawvideo", "-"],
                       check=True, capture_output=True)
    # stored luma 16+4k is limited range; gray = (Y-16)*255/219, so k = gray*219/255/4
    return [round(v * 219 / 255 / 4) for v in r.stdout]


def pcm48(path):
    r = subprocess.run([FF, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "48000",
                        "-f", "s16le", "-"], check=True, capture_output=True)
    n = len(r.stdout) // 2
    return struct.unpack(f"<{n}h", r.stdout[:2 * n])


def goertzel(x, f, rate=48000):
    w = 2 * math.pi * f / rate
    c = 2 * math.cos(w)
    s1 = s2 = 0.0
    for v in x:
        s0 = v + c * s1 - s2
        s2, s1 = s1, s0
    return s1 * s1 + s2 * s2 - c * s1 * s2


def slot_tone(samples, j, candidates):
    """Index into `candidates` of the dominant tone in output slot j (its
    middle 60 %, away from slot edges and AAC transients), or None if silent."""
    win = samples[j * SLOT + 400: j * SLOT + 1600]
    if not win or max(abs(v) for v in win) < 300:
        return None
    return max(range(len(candidates)), key=lambda i: goertzel(win, candidates[i]))


class CutAudioContent(unittest.TestCase):
    """_cut at 32/44.1/48 kHz sources, nonzero trims, bounded ends, silence."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="footage_cut_audio_")
        cls.src = {r: make_source(os.path.join(cls.work, f"tone{r}.mp4"), r) for r in (32000, 44100, 48000)}
        cls.silent = make_source(os.path.join(cls.work, "silent.mp4"), None)
        cls.cands = [tone(k) for k in range(FRAMES)]

    def cut(self, src, a, b, name):
        info = footage.probe_media(src)
        want = footage.allocate_frames(a, b, info["fps"], R24)
        out = os.path.join(self.work, name)
        footage._cut(src, a, b, info, R24, 64, 48, want, out)
        return out, want

    def check_part(self, out, a, b, want, label):
        # video: exactly frames a..b-1, once each, in order (no drop, no duplicate)
        self.assertEqual(footage.probe_media(out)["frame_count"], want, label)
        self.assertEqual(grey_levels(out), list(range(a, b)), label)
        # audio: length is the video's, within one AAC frame
        pcm = pcm48(out)
        exact = want * SLOT
        self.assertGreaterEqual(len(pcm), exact - AAC_FRAME, f"{label}: audio short by {exact - len(pcm)}")
        self.assertLessEqual(len(pcm), exact + AAC_FRAME, f"{label}: audio long by {len(pcm) - exact}")
        # audio content: slot j carries the tone of source frame a+j
        for j in sorted({0, 1, want // 2, want - 2, want - 1}):
            self.assertEqual(slot_tone(pcm, j, self.cands), a + j, f"{label}: slot {j}")
        return pcm

    def test_nonzero_trim_selects_the_right_audio_at_every_source_rate(self):
        for rate, src in self.src.items():
            with self.subTest(rate=rate):
                out, want = self.cut(src, 12, 51, f"b12_51_{rate}.mp4")
                self.assertEqual(want, 39)
                self.check_part(out, 12, 51, want, f"{rate} Hz [12,51)")

    def test_head_cut_is_bounded_at_the_end(self):
        """[0,12) must not carry the rest of the source's audio."""
        for rate, src in self.src.items():
            with self.subTest(rate=rate):
                out, want = self.cut(src, 0, 12, f"h0_12_{rate}.mp4")
                self.check_part(out, 0, 12, want, f"{rate} Hz [0,12)")

    def test_interior_cut(self):
        out, want = self.cut(self.src[32000], 17, 34, "i17_34.mp4")
        self.check_part(out, 17, 34, want, "32 kHz [17,34)")

    def test_silent_source_gets_exact_length_silence(self):
        out, want = self.cut(self.silent, 5, 29, "silent5_29.mp4")
        self.assertEqual(footage.probe_media(out)["frame_count"], want)
        self.assertEqual(grey_levels(out), list(range(5, 29)))
        pcm = pcm48(out)
        self.assertLessEqual(abs(len(pcm) - want * SLOT), AAC_FRAME)
        self.assertLess(max(abs(v) for v in pcm), 50)

    def test_mixed_assembly_keeps_audio_on_its_frames(self):
        """silent[0,12) + 32 kHz[12,51) + 44.1 kHz[0,6): the join must not
        shift the tones relative to the frames they belong to."""
        parts, cursor, marks = [], 0, []
        for i, (src, a, b) in enumerate([(self.silent, 0, 12), (self.src[32000], 12, 51),
                                          (self.src[44100], 0, 6)]):
            out, want = self.cut(src, a, b, f"mix{i}.mp4")
            parts.append(out)
            marks.append((cursor, a, want))
            cursor += want
        final = os.path.join(self.work, "mix.mp4")
        footage._concat(parts, final, lambda *_: None)
        self.assertEqual(footage.probe_media(final)["frame_count"], cursor)
        self.assertEqual(grey_levels(final), list(range(0, 12)) + list(range(12, 51)) + list(range(0, 6)))
        pcm = pcm48(final)
        # _concat applies each part's edit list, so the join adds no priming
        self.assertLessEqual(abs(len(pcm) - cursor * SLOT), AAC_FRAME)
        self.assertIsNone(slot_tone(pcm, 5, self.cands))            # silent part stays silent
        for start, a, want in marks[1:]:
            for j in sorted({0, 1, want // 2, want - 2, want - 1}):
                self.assertEqual(slot_tone(pcm, start + j, self.cands), a + j, f"mix slot {start + j}")


if __name__ == "__main__":
    unittest.main()
