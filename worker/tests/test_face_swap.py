"""face_swap: the runner's param validation, the fsh3 config the engine writes (and the prompt fsh3 builds from it),
and the HTTP ComfyUI client the stages use (against a fake server). The stages themselves were validated on the
spike (spikes/faceswap_h3); the GPU path is the E2E. OpenCV-dependent tests skip without it."""
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest

WORKER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = os.path.join(WORKER, "faceswap")
sys.path.insert(0, WORKER)


def _params(**over):
    p = {"org_id": "o", "job_id": "j", "target": {"bucket": "assets", "path": "t"},
         "face": {"bucket": "assets", "path": "f"}, "consent": {"confirmed": True, "note": "Ken agreed"},
         "subject": "man", "person": "a wide round face and short thick black hair with a fringe"}
    p.update(over)
    return p


try:
    from runners import face_swap as _runner  # noqa: E402  (needs the worker venv: db -> supabase)
except (ImportError, KeyError):  # no worker venv / no .env on this box
    _runner = None

try:
    sys.path.insert(0, FS)
    import cv2  # noqa: E402,F401
    import engine  # noqa: E402
    import fsh3  # noqa: E402
except Exception:  # noqa: BLE001 - no OpenCV on this box
    engine = fsh3 = None


@unittest.skipUnless(_runner, "runner imports need the worker venv")
class ValidateTest(unittest.TestCase):
    def test_defaults(self):
        p = _runner.validate(_params())
        self.assertEqual((p["tries"], p["seed"], p["props"]), (2, 43, []))

    def test_consent_required(self):
        for c in (None, {}, {"confirmed": "yes"}, {"confirmed": False}):
            with self.assertRaises(RuntimeError):
                _runner.validate(_params(consent=c))
        text = _runner.consent_text({"note": "Ken agreed", "kind": "real_person", "by_user": "u"}, "fj")
        self.assertTrue(text.startswith("Ken agreed | kind=real_person by_user=u"))
        self.assertIn("farm_job=fj", text)

    def test_inputs(self):
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(face=None))
        with self.assertRaises(RuntimeError):
            _runner.validate(_params(subject="boy"))
        _runner.validate(_params(person=None))   # empty: fsh3.describe fills it

    def test_props_and_range(self):
        p = _runner.validate(_params(props=["cap", {"object": "headphones", "description": "grey over-ear headphones"}],
                                     start_s=2, end_s=40))
        self.assertEqual(p["props"], [{"object": "cap"}, {"object": "headphones",
                                                          "description": "grey over-ear headphones"}])
        for bad in ({"props": ["a", "b", "c", "d"]}, {"start_s": 5, "end_s": 4}, {"start_s": 0, "end_s": 61},
                    {"tries": 5}, {"hair": "bald"}):
            with self.assertRaises(RuntimeError):
                _runner.validate(_params(**bad))


@unittest.skipUnless(engine, "needs OpenCV")
class ConfigTest(unittest.TestCase):
    def test_prompt_from_settings(self):
        c = {**fsh3.DEFAULT_CONFIG, **engine.config_for({
            "subject": "man", "person": "a wide round face with full cheeks, short thick black hair with a fringe.",
            "scene": "the same light blue t-shirt with a yellow print, in the same dim room",
            "performance": "the same hand gestures with his index finger toward the lens, the same close selfie framing",
            "props": [{"object": "headphones", "description": "light grey over-ear ones on his head"}],
            "tries": 3, "seed": 43, "destubble": True})}
        self.assertEqual(c["seeds"], [43, 44, 45])
        self.assertTrue(c["destubble"] and c["chin_clean"])
        s = fsh3.prompt(c)
        self.assertTrue(s.startswith("The man from <Picture 1> performs the exact performance of the person in "
                                     "<Video 1>. Keep his identity from <Picture 1>: his face, a wide round face"))
        self.assertIn("He wears the headphones from <Picture 2>: light grey over-ear ones on his head, worn exactly "
                      "as in <Video 1>.", s)
        self.assertIn("speech timing, the same hand gestures with his index finger toward the lens, the same close "
                      "selfie framing and camera motion. He wears the same light blue t-shirt", s)
        self.assertEqual(fsh3.ref_images(c), ["face.jpg", "headphones.png"])

    def test_empty_texts_are_left_for_describe(self):
        c = engine.config_for({"subject": "person", "props": [{"object": "cap"}]})
        self.assertEqual((c["subject"], c["possessive"], c["face_desc"], c["scene_desc"]),
                         ("the person", "their", "", ""))
        self.assertNotIn("sentence", c["props"][0])
        q = engine.config_for({"subject": "person", "scene": "the same grey hoodie"})
        self.assertEqual(q["scene_desc"], "the same hand gestures, the same framing and camera motion. They wear the "
                                          "same grey hoodie.")

    def test_vlm_prop_colour_is_dropped(self):
        self.assertEqual(engine.uncoloured("The man wears the black over-ear headphones from {pic}, worn exactly as in "
                                           "<Video 1>."),
                         "The man wears the over-ear headphones from {pic}, worn exactly as in <Video 1>.")
        self.assertEqual(engine.uncoloured("the light grey and white baseball cap"), "the baseball cap")

    def test_short_clip_is_one_padded_piece(self):
        # E2E 2026-10-24: a 10 s clip (240 frames < 243) planned a piece at -3
        self.assertEqual(fsh3.piece_starts(240), [0])
        self.assertEqual(fsh3.piece_len(240), 243)
        self.assertEqual(fsh3.piece_len(100), 107)
        self.assertEqual(fsh3.piece_len(813), 243)
        self.assertEqual(fsh3.piece_starts(813), [0, 195, 390, 570])
        for n in (244, 500, 1440):
            st = fsh3.piece_starts(n)
            self.assertEqual(st[-1] + 243, n)
            self.assertTrue(all(0 <= a < b for a, b in zip(st, st[1:])))

    def test_short_clip_pieces_on_disk(self):
        import numpy as np
        d = tempfile.mkdtemp(prefix="fs_pieces_")
        try:
            job = fsh3.Job("j", root=d)
            job.save_cfg({**engine.config_for({"subject": "man", "person": "a round face and short black hair",
                                                "scene": "the same blue t-shirt"}), "ratio": "9:16"})
            engine.setup_env({"comfy_url": "http://127.0.0.1:1", "work_dir": d})
            w = fsh3.Writer(job.p("ref.mp4"), 448, 768)
            for i in range(100):
                w.write(np.full((768, 448, 3), i, np.uint8))
            w.close()
            fsh3.pieces(job)
            self.assertEqual(fsh3.count(job.p("pieces", "ref_000.mp4")), 107)
            g = json.load(open(job.p("graphs", "p000_s43.json")))
            self.assertEqual(g["cond"]["inputs"]["length"], 107)
            self.assertEqual([s for _, s in fsh3.pending(job)], [0])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class _Fake(http.server.BaseHTTPRequestHandler):
    """Just enough of ComfyUI's API: /upload/image, /prompt, /history, /view, /free."""
    store = {}

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/upload/image":
            sub = body.split(b'name="subfolder"\r\n\r\n')[1].split(b"\r\n")[0].decode()
            name = body.split(b'filename="')[1].split(b'"')[0].decode()
            data = body.split(b'filename="')[1].split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
            _Fake.store[f"in/{sub}/{name}"] = data
            return self._json({"name": name, "subfolder": sub, "type": "input"})
        if self.path == "/prompt":
            _Fake.store["graph"] = json.loads(body)["prompt"]
            return self._json({"prompt_id": "p1"})
        return self._json({})

    def do_GET(self):
        if self.path.startswith("/history/"):
            return self._json({"p1": {"status": {"messages": []}, "outputs": {
                "save": {"videos": [{"filename": "x_obj0.mp4", "subfolder": "fs_job", "type": "output"}]}}}})
        if self.path.startswith("/view?"):
            b = b"VIDEO"
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)


class ComfyHttpTest(unittest.TestCase):
    def setUp(self):
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), _Fake)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.dir = tempfile.mkdtemp(prefix="fs_comfy_")
        os.environ.update(COMFYUI_URL=f"http://127.0.0.1:{self.srv.server_port}", FSH3_SUBFOLDER="fs_job",
                          FSH3_FETCH_DIR=self.dir)
        sys.modules.pop("comfy", None)
        sys.path.insert(0, FS)
        import comfy
        self.comfy = comfy

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        sys.modules.pop("comfy", None)
        shutil.rmtree(self.dir, ignore_errors=True)
        for k in ("COMFYUI_URL", "FSH3_SUBFOLDER", "FSH3_FETCH_DIR"):
            os.environ.pop(k, None)

    def test_upload_run_fetch(self):
        src = os.path.join(self.dir, "face.jpg")
        with open(src, "wb") as f:
            f.write(b"JPEGDATA")
        self.assertEqual(self.comfy.put_input(src), "fs_job/face.jpg")
        self.assertEqual(_Fake.store["in/fs_job/face.jpg"], b"JPEGDATA")
        files = self.comfy.run({"save": {"class_type": "SaveVideo"}}, poll=0.01)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].replace("\\", "/").endswith("fs_job/x_obj0.mp4"))
        with open(files[0], "rb") as f:
            self.assertEqual(f.read(), b"VIDEO")
        self.comfy.free()


if __name__ == "__main__":
    unittest.main()
