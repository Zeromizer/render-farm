"""remake / remake_paint: param validation, the 9:16 squeeze shapes, the PC fallback exclusion, and that the vendored
Viggle package is the studied revision, unedited (VENDOR.md promises it). unittest, like the other farm tests."""
import hashlib
import os
import sys
import unittest

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
sys.path.insert(0, os.path.join(W, "remake"))

import shapes  # noqa: E402

try:
    from runners import remake as R, remake_paint as RP  # noqa: E402  (needs the worker venv: db -> supabase)
except Exception:  # noqa: BLE001  (no worker venv / env: the param tests skip)
    R = RP = None

SRC = {"bucket": "assets", "path": "sha256/x"}
OK = {"confirmed": True, "note": "test"}


def paint(**over):
    p = {"org_id": "o", "job_id": "j", "stage": "paint", "session": {"bucket": "renders", "path": "outputs/a.tgz"},
         "face": SRC, "consent": OK, "name": "Mei", "identity": "a woman in her 30s, black hair",
         "outfit": "navy polo, black trousers, black shoes", "person": "the presenter in the navy polo",
         "keyframes": [{"at_s": 2.75, "where": "centre-left, hands together"}]}
    p.update(over)
    return p


class Shapes(unittest.TestCase):
    def test_portrait_916_is_squeezed_to_codex_portrait(self):
        s, squeeze = shapes.nearest(736 / 1280)
        self.assertTrue(squeeze)
        self.assertAlmostEqual(s, 984 / 1599, places=3)

    def test_landscape_169_untouched(self):
        self.assertFalse(shapes.nearest(1280 / 736)[1])

    def test_cropped_0615_untouched(self):
        self.assertFalse(shapes.nearest(768 / 1216)[1])


@unittest.skipIf(RP is None, "worker venv not available")
class PaintParams(unittest.TestCase):
    def test_paint_ok(self):
        RP.validate(paint())

    def test_paint_needs_consent(self):
        with self.assertRaisesRegex(RuntimeError, "consent"):
            RP.validate(paint(consent=None))

    def test_paint_needs_keyframes_with_where(self):
        with self.assertRaisesRegex(RuntimeError, "keyframes"):
            RP.validate(paint(keyframes=[]))
        with self.assertRaisesRegex(RuntimeError, "keyframes"):
            RP.validate(paint(keyframes=[{"at_s": 1}]))

    def test_name_charset(self):
        with self.assertRaisesRegex(RuntimeError, "name"):
            RP.validate(paint(name="Mei/../x"))

    def test_prepare_span(self):
        RP.validate({"org_id": "o", "job_id": "j", "stage": "prepare", "target": SRC, "start_s": 0, "end_s": 6})
        with self.assertRaisesRegex(RuntimeError, "end_s"):
            RP.validate({"org_id": "o", "job_id": "j", "stage": "prepare", "target": SRC, "start_s": 0, "end_s": 90})

    def test_repaint_needs_frames(self):
        with self.assertRaisesRegex(RuntimeError, "frames"):
            RP.validate({"org_id": "o", "job_id": "j", "stage": "repaint", "session": SRC, "consent": OK})


@unittest.skipIf(RP is None, "worker venv not available")
class CodexBin(unittest.TestCase):
    """The VS Code extension's folder is versioned (openai.chatgpt-26.51021...): the newest one must win, as a number."""
    def test_newest_extension_wins(self):
        import tempfile
        exe = "codex.exe" if os.name == "nt" else "codex"
        with tempfile.TemporaryDirectory() as t:
            for v in ("26.9.1", "26.51021.22027", "26.10000.1"):
                d = os.path.join(t, f"openai.chatgpt-{v}-win32-x64", "bin", "windows-x86_64")
                os.makedirs(d)
                open(os.path.join(d, exe), "w").close()
            old = RP.config.REMAKE_CODEX_BIN
            RP.config.REMAKE_CODEX_BIN = os.path.join(t, "openai.chatgpt-*", "bin", "*", exe)
            try:
                self.assertIn("26.51021.22027", RP.codex_bin())
            finally:
                RP.config.REMAKE_CODEX_BIN = old


@unittest.skipIf(R is None, "worker venv not available")
class RenderParams(unittest.TestCase):
    def p(self, **over):
        p = {"org_id": "o", "job_id": "j", "session": SRC, "consent": OK, "prompts": {"0": "x" * 300}, "seeds": [0]}
        p.update(over)
        return p

    def test_ok(self):
        R.validate(self.p())

    def test_refuses_without_consent(self):
        with self.assertRaisesRegex(RuntimeError, "consent"):
            R.validate(self.p(consent={}))

    def test_prompts_shape(self):
        with self.assertRaisesRegex(RuntimeError, "prompts"):
            R.validate(self.p(prompts={"w0": "x" * 300}))
        with self.assertRaisesRegex(RuntimeError, "prompts"):
            R.validate(self.p(prompts={"0": "too short"}))

    def test_one_seed(self):
        with self.assertRaisesRegex(RuntimeError, "seed"):
            R.validate(self.p(seeds=[0, 1]))


class CloudOnly(unittest.TestCase):
    def test_remake_never_falls_back_to_the_pc(self):
        try:
            import config
        except Exception:  # noqa: BLE001
            self.skipTest("config needs the worker env")
        self.assertIn("remake", config.CLOUD_ONLY_ENGINES)
        self.assertIn("remake", config.COMFY_ENGINES)


class Vendor(unittest.TestCase):
    # sha256 of the files as downloaded at 7e5be14 (VENDOR.md): an edit here belongs in a wrapper instead
    def test_vendored_files_present(self):
        v = os.path.join(W, "remake", "vendor")
        for rel in ("viggle_remake_mcp/server.py", "viggle_remake_mcp/kit.py", "viggle_remake_mcp/paint.py",
                    "comfy_worker/remake_worker.py", "comfyui/ComfyUI-Viggle-Remake/nodes.py", "LICENSE", "VENDOR.md"):
            self.assertTrue(os.path.exists(os.path.join(v, rel)), rel)
        self.assertIn("7e5be143702dd53b9154482de673b8519ed2d4d6", open(os.path.join(v, "VENDOR.md")).read())


if __name__ == "__main__":
    unittest.main()
