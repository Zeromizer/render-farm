"""YuE2 "cover" from a reference track on the render PC's ComfyUI: SheetSage2 transcribes the
reference into an ABC score, then YuE2 (+ the instrumental AR LoRA, as in audio_gen) plays that
score in a new arrangement. Keeps the reference's melody and form, not its sound.

params.args = [<job json string>, <out dir>]. Job JSON:
  {"reference": "spikes/yue2_cover/in/ref.wav", "start_s": 0, "duration_s": 30,
   "tags": "[chorus 0:00-0:12]\\n...",            # timed section tags, as audio_gen sends
   "takes": [{"name": "melody_s1", "abc": "melody", "style": "...", "seed": 1},
             {"name": "full_s1",   "abc": "full",   "style": "...", "seed": 1},
             {"name": "control_s1","abc": "generate","style": "...", "seed": 1}]}
abc: "melody" / "full" = SheetSage2 score of the reference in that mode (YuE2 mode matches);
"generate" = the live audio_gen path (YuE2 writes its own score from the text), the control.
A take may carry its own "reference", "duration_s" (output length; may exceed the reference, to
see what YuE2 plays past the end of the score) and "tags". Writes <out>/<name>.wav (48 kHz,
trimmed to duration_s, 0.8 s fade), <out>/abc_<reference>_<mode>.txt, <out>/reference_<name>.wav
(the clip SheetSage2 heard) and <out>/results.json.

Needs sheetsage2_bf16.safetensors in ComfyUI's models/audio_encoders; downloaded once if missing.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

import httpx
import numpy as np
import soundfile as sf

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI")
T = httpx.Timeout(30.0, read=600.0)
CKPT = "yue2_3b_int8_convrot.safetensors"
LORA = "ar_lora_inst_v3abc_comfyui.safetensors"
ENCODER = "sheetsage2_bf16.safetensors"
ENCODER_URL = "https://huggingface.co/Comfy-Org/YuE2/resolve/main/audio_encoders/" + ENCODER
SR = 48000


def log(msg):
    print(msg, flush=True)


def ensure_server(wait_s=240):
    def up():
        try:
            return httpx.get(COMFY + "/system_stats", timeout=5).status_code == 200
        except httpx.HTTPError:
            return False
    if up():
        return
    bat = os.path.join(COMFY_DIR, "run-headless.bat")
    log(f"comfyui not running; launching {bat}")
    subprocess.Popen(["cmd", "/c", bat], cwd=COMFY_DIR, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if up():
            return
        time.sleep(2)
    raise RuntimeError("ComfyUI did not come up")


def encoder_names():
    # /models/<folder>, not /object_info: V3 nodes publish combos as ["COMBO", {"options": [...]}].
    return httpx.get(COMFY + "/models/audio_encoders", timeout=T).json()


def models_dir():
    """The models/ folder that holds the YuE2 checkpoint (layouts differ between installs)."""
    for root, dirs, files in os.walk(COMFY_DIR):
        if root[len(COMFY_DIR):].count(os.sep) > 4:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "venv", "custom_nodes", "input", "output", "temp")]
        if os.path.basename(root) == "checkpoints" and CKPT in files:
            return os.path.dirname(root)
    raise RuntimeError(f"cannot find models/checkpoints/{CKPT} under {COMFY_DIR}")


def ensure_encoder():
    if ENCODER in encoder_names():
        return
    dest_dir = os.path.join(models_dir(), "audio_encoders")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, ENCODER)
    log(f"downloading {ENCODER} -> {dest}")
    part = dest + ".part"
    with httpx.stream("GET", ENCODER_URL, follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0)) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    os.replace(part, dest)
    log(f"{ENCODER}: {os.path.getsize(dest) / 1e6:.0f} MB")
    for _ in range(10):
        if ENCODER in encoder_names():
            return
        time.sleep(2)
    raise RuntimeError(f"ComfyUI does not list {ENCODER} after download (folder {dest_dir})")


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "yue2_cover", "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def graph(ref_name, take, tags, max_duration, prefix):
    mode = take["abc"]
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": LORA,
                                                     "strength_model": 0.0, "strength_clip": 1.0}},
        "5": {"class_type": "YuE2GenerateMusic",
              "inputs": {"clip": ["2", 1], "style": take["style"], "lyrics": tags, "seed": int(take["seed"]),
                         "mode": "full" if mode == "generate" else mode, "max_duration": max_duration,
                         "temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {"class_type": "EmptyYuE2LatentAudio", "inputs": {"seconds": ["5", 1], "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["2", 0], "seed": int(take["seed"]), "steps": 32, "cfg": 1.0,
                         "sampler_name": "dpm_2", "scheduler": "sgm_uniform", "positive": ["5", 0],
                         "negative": ["6", 0], "latent_image": ["7", 0], "denoise": 1.0}},
        "9": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveAudio", "inputs": {"audio": ["9", 0], "filename_prefix": prefix}},
    }
    if mode == "generate":
        g["4"] = {"class_type": "YuE2GenerateABC",
                  "inputs": {"clip": ["2", 1], "style": take["style"], "lyrics": tags, "seed": int(take["seed"]),
                             "mode": "full", "max_abc_tokens": 3000, "temperature": 0.7, "top_p": 0.9,
                             "top_k": 30, "repetition_penalty": 1.005, "penalty_window": 100}}
    else:
        g["11"] = {"class_type": "LoadAudio", "inputs": {"audio": ref_name}}
        g["12"] = {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": ENCODER}}
        g["4"] = {"class_type": "SheetSage2AudioToABC",
                  "inputs": {"audio_encoder": ["12", 0], "audio": ["11", 0], "mode": mode}}
        g["13"] = {"class_type": "PreviewAny", "inputs": {"source": ["4", 0]}}
    g["5"]["inputs"]["abc"] = ["4", 0]
    return g


def run(g, timeout_s=2400):
    r = httpx.post(COMFY + "/prompt", json={"prompt": g, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:2000]}")
    pid = r.json()["prompt_id"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            e = httpx.get(COMFY + f"/history/{pid}", timeout=60).json().get(pid)
        except httpx.TimeoutException:
            continue
        if e:
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = st.get("messages") or []
                err = next((m[1] for m in msgs if m[0] == "execution_error"), {})
                raise RuntimeError(f"execution error at {err.get('node_type')}: {err.get('exception_message')}")
            return e.get("outputs") or {}, time.monotonic() - t0
        time.sleep(2)
    raise RuntimeError("timed out")


def fetch_audio(outputs, dest):
    for node_out in outputs.values():
        for f in node_out.get("audio", []) or []:
            params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
            with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as out:
                    for chunk in resp.iter_bytes(1 << 20):
                        out.write(chunk)
            return
    raise RuntimeError("no audio in outputs")


def preview_text(outputs):
    texts = []
    for node_out in outputs.values():
        for v in node_out.get("text", []) or []:
            texts.append(v if isinstance(v, str) else json.dumps(v))
    return "\n".join(texts)


def trim_fade(src, dest, duration_s):
    y, sr = sf.read(src, always_2d=True)
    if sr != SR:
        raise RuntimeError(f"expected {SR} Hz from VAEDecodeAudio, got {sr}")
    n = min(len(y), int(round(duration_s * sr)))
    y = y[:n].copy()
    fade = int(min(2.0, max(0.8, 0.05 * duration_s)) * sr)
    y[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    sf.write(dest, y, sr, subtype="PCM_16")
    return n / sr


def clear_out(out):
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        q = os.path.join(out, name)
        shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)


def main():
    job = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    clear_out(out)
    tags = job["tags"]
    takes = job["takes"]
    ensure_server()
    if any(t["abc"] != "generate" for t in takes):
        ensure_encoder()

    refs = {}   # reference path -> (ComfyUI input name, clip seconds, local stem)

    def reference(path):
        if path not in refs:
            start = float(job.get("start_s", 0))
            y, sr = sf.read(path, always_2d=True)
            y = y[int(start * sr): int((start + float(job.get("ref_max_s", 600))) * sr)]
            stem = "reference_" + os.path.splitext(os.path.basename(path))[0]
            local = os.path.join(out, stem + ".wav")
            sf.write(local, y, sr, subtype="PCM_16")
            refs[path] = (upload(local), len(y) / sr, stem)
            log(f"reference {path}: {len(y) / sr:.2f}s from {start:.1f}s @ {sr} Hz")
        return refs[path]

    results = []
    try:
        for i, take in enumerate(takes):
            ref_name, ref_s, stem = reference(take.get("reference", job["reference"]))
            dur = float(take.get("duration_s") or job.get("duration_s") or ref_s)
            max_duration = round(round((dur + 2.0) / 0.04) * 0.04, 2)
            raw = os.path.join(out, f"{take['name']}.raw.flac")
            outputs, secs = run(graph(ref_name, take, take.get("tags", tags), max_duration, f"yue2_cover/{take['name']}"))
            fetch_audio(outputs, raw)
            length = trim_fade(raw, os.path.join(out, f"{take['name']}.wav"), dur)
            os.remove(raw)
            abc = preview_text(outputs)
            if abc and take["abc"] != "generate":
                path = os.path.join(out, f"abc_{stem}_{take['abc']}.txt")
                if not os.path.exists(path):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(abc)
            results.append({**take, "ref_s": round(ref_s, 2), "wall_s": round(secs, 1),
                            "length_s": round(length, 3), "abc_chars": len(abc)})
            log(f"{take['name']}: {secs:.0f}s wall, {length:.2f}s audio")
            log(f"PROGRESS {int((i + 1) * 100 / len(takes))}")
    finally:
        with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"job": job, "results": results}, f, indent=1)
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)


if __name__ == "__main__":
    main()
