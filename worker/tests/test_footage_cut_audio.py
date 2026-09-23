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
one AAC frame. Anything beyond that is drift, not encoder rounding. Whether a
decoder drops the END padding differs by FFmpeg build (9.0.1 drops it, 8.1.1
keeps it), so _concat must not rely on it; the padded-part test pins that.
"""
import json
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


def probe_streams(path):
    """(video start, audio start, video duration, audio duration, video frame pts list)."""
    fp = segments._tool("ffprobe")
    r = subprocess.run([fp, "-v", "error", "-show_entries", "stream=codec_type,start_time,duration",
                        "-of", "json", path], check=True, capture_output=True, text=True)
    st = {s["codec_type"]: s for s in json.loads(r.stdout)["streams"]}
    r = subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                        "-of", "csv=p=0", path], check=True, capture_output=True, text=True)
    pts = [float(x.strip(",")) for x in r.stdout.split() if x.strip(",")]
    return (float(st["video"]["start_time"]), float(st["audio"]["start_time"]),
            float(st["video"]["duration"]), float(st["audio"]["duration"]), pts)


def padded_part(path, src, a, b, extra):
    """A part as _cut would make it, but whose audio runs `extra` samples PAST its
    video - what a decoder that keeps AAC end padding hands _concat. Built as
    exact video + separately trimmed PCM, muxed without any frame limit."""
    n = b - a
    v = path + ".v.mp4"
    w = path + ".a.wav"
    subprocess.run([FF, "-v", "error", "-y", "-i", src, "-an",
                    "-vf", f"select='between(n\,{a}\,{b - 1})',setpts=N/FRAME_RATE/TB", "-frames:v", str(n),
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", v], check=True, capture_output=True)
    subprocess.run([FF, "-v", "error", "-y", "-i", src, "-vn",
                    "-af", (f"aresample=48000,atrim=start_sample={a * SLOT}:end_sample={b * SLOT + extra},"
                           f"asetpts=N/SR/TB,apad=whole_len={n * SLOT + extra}"),  # also past the source end
                    "-ac", "2", "-c:a", "pcm_s16le", w], check=True, capture_output=True)
    subprocess.run([FF, "-v", "error", "-y", "-i", v, "-i", w, "-map", "0:v:0", "-map", "1:a:0",
                    "-c", "copy", path], check=True, capture_output=True)
    return path


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


    def assert_join_timing(self, final, total, label):
        v0, a0, vd, ad, pts = probe_streams(final)
        self.assertAlmostEqual(v0, 0.0, delta=0.001, msg=f"{label}: video start {v0}")
        self.assertAlmostEqual(a0, 0.0, delta=0.001, msg=f"{label}: audio start {a0}")
        self.assertEqual(len(pts), total, label)
        for i, x in enumerate(pts):                     # native 24 fps, zero-based, ordered
            self.assertAlmostEqual(x, i / 24, delta=0.001, msg=f"{label}: frame {i} pts {x}")
        self.assertAlmostEqual(vd, total / 24, delta=0.001, msg=label)
        self.assertLessEqual(abs(ad - vd), AAC_FRAME / 48000, f"{label}: audio {ad}s vs video {vd}s")

    def test_join_is_zero_based_and_tones_stay_on_their_frames_at_every_join(self):
        parts, cursor, marks = [], 0, []
        for i, (src, a, b) in enumerate([(self.silent, 0, 12), (self.src[32000], 12, 51),
                                          (self.src[44100], 0, 6)]):
            out, want = self.cut(src, a, b, f"jt{i}.mp4")
            parts.append(out)
            marks.append((cursor, a, want, src is self.silent))
            cursor += want
        final = os.path.join(self.work, "jt.mp4")
        footage._concat(parts, final, lambda *_: None)
        self.assert_join_timing(final, cursor, "3-part")
        pcm = pcm48(final)
        for start, a, want, silent in marks:            # both sides of every join
            for j in (0, want - 1):
                self.assertEqual(slot_tone(pcm, start + j, self.cands), None if silent else a + j,
                                 f"slot {start + j}")

    def test_long_chain_does_not_accumulate_per_part_padding(self):
        """Ten parts across all rates and silence: the total stays within ONE
        final AAC frame, not one per part."""
        plan = [(self.src[32000], 0, 7), (self.silent, 7, 13), (self.src[44100], 13, 21),
                (self.src[48000], 21, 26), (self.src[32000], 26, 33), (self.src[44100], 33, 38),
                (self.silent, 38, 41), (self.src[48000], 41, 45), (self.src[32000], 45, 49),
                (self.src[44100], 49, 51)]
        parts, cursor, marks = [], 0, []
        for i, (src, a, b) in enumerate(plan):
            out, want = self.cut(src, a, b, f"chain{i}.mp4")
            parts.append(out)
            marks.append((cursor, a, want, src is self.silent))
            cursor += want
        final = os.path.join(self.work, "chain.mp4")
        footage._concat(parts, final, lambda *_: None)
        self.assertEqual(cursor, 51)
        self.assertEqual(grey_levels(final), list(range(51)))
        self.assert_join_timing(final, cursor, "10-part")
        pcm = pcm48(final)
        self.assertLessEqual(abs(len(pcm) - cursor * SLOT), AAC_FRAME, "10-part length")
        for start, a, want, silent in marks:
            for j in sorted({0, want - 1}):
                self.assertEqual(slot_tone(pcm, start + j, self.cands), None if silent else a + j,
                                 f"chain slot {start + j}")

    def test_concat_bounds_parts_whose_decoded_audio_is_longer_than_their_video(self):
        """Simulates a decoder that keeps AAC end padding: every part's audio runs
        a whole AAC frame past its video. _concat must still join exact lengths."""
        spans = [(0, 12), (12, 30), (30, 39), (39, 51)]
        parts = [padded_part(os.path.join(self.work, f"pad{i}.mov"), self.src[48000], a, b, AAC_FRAME)
                 for i, (a, b) in enumerate(spans)]
        for p, (a, b) in zip(parts, spans):
            self.assertEqual(len(pcm48(p)), (b - a) * SLOT + AAC_FRAME)   # the fixture really is long
        final = os.path.join(self.work, "pad.mp4")
        footage._concat(parts, final, lambda *_: None)
        self.assert_join_timing(final, 51, "padded parts")
        pcm = pcm48(final)
        self.assertLessEqual(abs(len(pcm) - 51 * SLOT), AAC_FRAME, "padded parts: no per-part growth")
        for a, b in spans:
            for k in (a, b - 1):
                self.assertEqual(slot_tone(pcm, k, self.cands), k, f"padded slot {k}")


if __name__ == "__main__":
    unittest.main()
