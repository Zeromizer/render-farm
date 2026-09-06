"""Extensionless / content-addressed inputs: the type comes from the bytes,
the local file gets a matching suffix, junk is rejected, and the runner's
_fetch keeps the {bucket, path} contract. Real ffprobe is used for the
decode check on tiny synthetic files; a fake download stands in for storage.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videogen import media_type, segments  # noqa: E402

# 1x1 PNG (valid) and the smallest headers of the other containers we accept.
PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080200000090775"
    "3de0000000c49444154789c63f8cfc00000030101009d9c2f7f0000000049454e44ae426082")
HEADS = {
    ".jpg": b"\xff\xd8\xff\xe0" + b"\0" * 60,
    ".webp": b"RIFF\x10\0\0\0WEBPVP8 " + b"\0" * 48,
    ".mp4": b"\0\0\0\x18ftypisom\0\0\0\0isomiso2mp41" + b"\0" * 40,
    ".mov": b"\0\0\0\x14ftypqt  \0\0\0\0qt  " + b"\0" * 44,
    ".m4a": b"\0\0\0\x18ftypM4A \0\0\0\0M4A mp42isom" + b"\0" * 40,
    ".webm": b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08\x42\x82\x84webm" + b"\0" * 30,
    ".mkv": b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08\x42\x82\x88matroska" + b"\0" * 30,
    ".wav": b"RIFF\x24\0\0\0WAVEfmt " + b"\0" * 52,
    ".flac": b"fLaC" + b"\0" * 60,
    ".ogg": b"OggS" + b"\0" * 60,
    ".mp3": b"ID3\x04\0\0\0\0\0\0" + b"\0" * 54,
}


def _have_ffprobe():
    try:
        return subprocess.run([segments._tool("ffprobe"), "-version"], capture_output=True).returncode == 0
    except Exception:  # noqa: BLE001
        return False


class Sniff(unittest.TestCase):
    def test_signatures(self):
        self.assertEqual(media_type.sniff_bytes(PNG_1x1), ("image", ".png"))
        for ext, head in HEADS.items():
            kind, got = media_type.sniff_bytes(head)
            self.assertEqual(got, ext, ext)
            self.assertEqual(kind, media_type.kind_of_ext(ext), ext)
        self.assertEqual(media_type.sniff_bytes(b"\xff\xfb\x90\x00" + b"\0" * 20), ("audio", ".mp3"))
        self.assertEqual(media_type.sniff_bytes(b"hello world, not media at all"), (None, None))
        self.assertEqual(media_type.sniff_bytes(b""), (None, None))


class EnsureExtension(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="mt-")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _write(self, name, data):
        p = os.path.join(self.td, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_extensionless_png_gets_png_suffix(self):
        p = self._write("front", PNG_1x1)          # what assets/sha256/<hex> downloads as
        out = media_type.ensure_extension(p, ("image",), name="front", probe=False)
        self.assertTrue(out.endswith("front.png"))
        self.assertTrue(os.path.exists(out) and not os.path.exists(p))

    def test_wrong_suffix_is_corrected(self):
        p = self._write("last_frame.jpg", PNG_1x1)
        out = media_type.ensure_extension(p, ("image",), probe=False)
        self.assertTrue(out.endswith("last_frame.png"))

    def test_kind_mismatch_rejected(self):
        p = self._write("front", HEADS[".mp4"])
        with self.assertRaises(RuntimeError) as cm:
            media_type.ensure_extension(p, ("image",), name="front", probe=False)
        self.assertIn("video", str(cm.exception))
        self.assertIn("image", str(cm.exception))

    def test_junk_rejected_with_empty_suffix_named(self):
        p = self._write("front", b"not a media file at all, just text\n" * 4)
        with self.assertRaises(RuntimeError) as cm:
            media_type.ensure_extension(p, ("image",), name="front", probe=False)
        self.assertIn("unsupported or corrupt", str(cm.exception))
        self.assertIn("''", str(cm.exception))

    def test_every_supported_container_maps_to_its_kind(self):
        for ext, head in HEADS.items():
            kind = media_type.kind_of_ext(ext)
            p = self._write("ref_" + ext.strip("."), head)
            out = media_type.ensure_extension(p, (kind,), probe=False)
            self.assertTrue(out.endswith(ext), ext)

    @unittest.skipUnless(_have_ffprobe(), "ffprobe not available")
    def test_probe_accepts_real_png_and_rejects_truncated_mp4(self):
        p = self._write("front", PNG_1x1)
        out = media_type.ensure_extension(p, ("image",), name="front")
        self.assertTrue(out.endswith(".png"))
        p = self._write("source", HEADS[".mp4"])   # a header with no streams
        with self.assertRaises(RuntimeError) as cm:
            media_type.ensure_extension(p, ("video",), name="source")
        self.assertIn("does not decode", str(cm.exception))


@unittest.skipUnless(_have_ffprobe(), "ffprobe not available")
class RunnerFetch(unittest.TestCase):
    """runners.video_gen._fetch with a fake storage download: scoped {bucket, path}
    contract, extensionless platform object, rejection of junk."""

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="vgf-")
        try:
            from runners import video_gen
        except Exception as exc:  # noqa: BLE001 - needs the worker's .env for the db client
            self.skipTest(f"runners.video_gen not importable here: {exc}")
        self.vg = video_gen
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _fake_download(self, data):
        def download(bucket, path, work_dir, name, log):
            self.calls.append((bucket, path, name))
            local = os.path.join(work_dir, name + os.path.splitext(path)[1])   # what gate_common.download does
            with open(local, "wb") as f:
                f.write(data)
            return local
        return download

    def test_extensionless_asset_becomes_png(self):
        ref = {"bucket": "assets", "path": "sha256/" + "ab" * 32}
        out = self.vg._fetch(ref, self.td, "front", print, ("image",), download=self._fake_download(PNG_1x1))
        self.assertTrue(out.endswith("front.png"))
        self.assertEqual(self.calls, [("assets", ref["path"], "front")])

    def test_ref_shape_enforced(self):
        for bad in ({"path": "x"}, {"bucket": "assets"}, "assets/x", None):
            with self.assertRaises(RuntimeError):
                self.vg._fetch(bad, self.td, "front", print, ("image",), download=self._fake_download(PNG_1x1))

    def test_junk_object_rejected(self):
        ref = {"bucket": "assets", "path": "sha256/" + "cd" * 32}
        with self.assertRaises(RuntimeError) as cm:
            self.vg._fetch(ref, self.td, "ready_front_to_left", print, ("video",), download=self._fake_download(b"junk" * 40))
        self.assertIn("ready_front_to_left", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
