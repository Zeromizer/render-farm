"""Spike: vlo's LTX 2.5 clean-plate workflow (Clean-Plate IC-LoRA) on the PC's ComfyUI.

The question is whether a learned video clean plate can replace clip_move's temporal-median
plate: shadows removed by the model, and moving cameras possible. The graph follows vlo's
vlo_ltx2_5_clean_plate (vlo da20a26) and Lightricks' LTX-2.5_V2V_ICLoRA_Single_Stage_Distilled,
on CORE nodes only so nothing new is installed in the production ComfyUI:
  - the IC-LoRA guide is core LTXVAddGuide (the LTXVideo pack's IC-LoRA guide is the same
    append_keyframe call when the reference downscale factor is 1, as the card says it is);
  - the IC-LoRA loads with LoraLoaderModelOnly;
  - no NAG (KJNodes), so the negative prompt does nothing at CFG 1;
  - the PC has the DEV transformer, so the distilled LoRA makes it the 8-step model.
No attention mask: the IC-LoRA is trained mask-free and clears every vehicle and person.

params.args = [<job json string>, <out dir>]. Job JSON:
  {"clips": [{"name", "clip": <signed url>, "prompt", "negative"?, "width"?, "height"?,
              "frames"?, "seed"?}],
   "plate_lora"?: 1.0 (0 = leave the clean-plate LoRA out: a plumbing run), "distilled_lora"?: 1.0}
Writes <out>/{<name>_src.mp4, <name>_plate.mp4, <name>_compare.mp4, <name>_strip.jpg, report.json}.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

import httpx

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
T = httpx.Timeout(30.0, read=600.0)

UNET = "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
DISTILLED = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
LORA = "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"
TEXT = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VAE = "ltx-2.5-video-vae-conv-bf16.safetensors"
AVAE = "ltx-2.5-audio-vae-bf16.safetensors"
NEG = "People, crowds, vehicles"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # LTX distilled 8-step


def log(*a):
    print(*a, flush=True)


def ensure_comfy(wait_s=600):
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        try:
            if httpx.get(COMFY + "/system_stats", timeout=10).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise RuntimeError("ComfyUI did not answer")


def upload(path):
    with open(path, "rb") as f:
        r = httpx.post(COMFY + "/upload/image", files={"image": (os.path.basename(path), f)},
                       data={"subfolder": "ltx_plate", "type": "input", "overwrite": "true"}, timeout=T)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def run_outputs(graph, timeout_s=3600):
    r = httpx.post(COMFY + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=T)
    if r.status_code != 200:
        raise RuntimeError(f"/prompt rejected {r.status_code}: {r.text[:3000]}")
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
            return e.get("outputs") or {}
        time.sleep(3)
    raise RuntimeError("timed out")


def fetch_first(outputs, dest):
    for node_out in outputs.values():
        for key in ("videos", "images", "gifs"):
            for f in node_out.get(key, []) or []:
                name = f.get("filename", "")
                if f.get("type", "output") != "output" or not name.lower().endswith(".mp4"):
                    continue
                params = {"filename": name, "subfolder": f.get("subfolder", ""), "type": "output"}
                with httpx.stream("GET", COMFY + "/view", params=params, timeout=T) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as out:
                        for chunk in resp.iter_bytes(1 << 20):
                            out.write(chunk)
                return dest
    raise RuntimeError("no mp4 in the outputs")


def free():
    try:
        httpx.post(COMFY + "/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    except httpx.HTTPError:
        pass


def vram():
    try:
        d = httpx.get(COMFY + "/system_stats", timeout=10).json()["devices"][0]
        return round(d["vram_free"] / 2**30, 1), round(d["vram_total"] / 2**30, 1)
    except Exception:  # noqa: BLE001
        return None


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,nb_read_frames", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    streams = json.loads(out)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    return int(v["nb_read_frames"]), any(s["codec_type"] == "audio" for s in streams)


def prep(src, dest, w, h, frames):
    """Stretch to w x h (vlo stretches too), 24 fps, 8k+1 frames, with an audio track (the
    stock graph encodes audio; a silent one is added when the clip has none)."""
    n, has_audio = probe(src)
    frames = min(frames or n, n)
    frames = (frames - 1) // 8 * 8 + 1
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src]
    if not has_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0",
            "-vf", f"scale={w}:{h}:flags=lanczos,fps=24,setsar=1", "-frames:v", str(frames),
            "-t", f"{frames / 24:.4f}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", dest]
    subprocess.run(cmd, check=True)
    return frames


def build(video, prompt, negative, w, h, frames, seed, prefix, plate_lora=1.0, distilled_lora=1.0):
    g = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "30": {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["3", 0], "lora_name": DISTILLED, "strength_model": distilled_lora}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT, "type": "ltxv", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": prompt}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": negative}},
        "11": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w, "height": h, "length": frames, "batch_size": 1}},
        "12": {"class_type": "LTXVAddGuide",
               "inputs": {"positive": ["8", 0], "negative": ["9", 0], "vae": ["6", 0], "latent": ["11", 0],
                          "image": ["2", 0], "frame_idx": 0, "strength": 1.0}},
        "13": {"class_type": "LTXVConditioning", "inputs": {"positive": ["12", 0], "negative": ["12", 1], "frame_rate": 24.0}},
        "14": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["2", 1], "audio_vae": ["7", 0]}},
        "15": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["12", 2], "audio_latent": ["14", 0]}},
        "17": {"class_type": "ManualSigmas", "inputs": {"sigmas": SIGMAS}},
        "19": {"class_type": "CFGGuider", "inputs": {"model": ["30", 0], "positive": ["13", 0], "negative": ["13", 1], "cfg": 1.0}},
        "20": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "21": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "22": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["21", 0], "guider": ["19", 0], "sampler": ["20", 0], "sigmas": ["17", 0], "latent_image": ["15", 0]}},
        "23": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["22", 0]}},
        "24": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["13", 0], "negative": ["13", 1], "latent": ["23", 0]}},
        "25": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["24", 2], "vae": ["6", 0], "tile_size": 512,
                                                          "overlap": 64, "temporal_size": 4096, "temporal_overlap": 8}},
        "26": {"class_type": "CreateVideo", "inputs": {"images": ["25", 0], "fps": 24.0}},
        "27": {"class_type": "SaveVideo", "inputs": {"video": ["26", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
    }
    if plate_lora:
        g["4"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["30", 0], "lora_name": LORA, "strength_model": plate_lora}}
        g["19"]["inputs"]["model"] = ["4", 0]
    return g


def check_graph(graph):
    """Fail early, with every mismatch listed, if a node or input name is not what the PC has."""
    info = httpx.get(COMFY + "/object_info", timeout=T).json()
    problems = []
    for nid, n in graph.items():
        ct = n["class_type"]
        if ct not in info:
            problems.append(f"{nid}: missing node {ct}")
            continue
        spec = info[ct]["input"]
        known = set(spec.get("required", {})) | set(spec.get("optional", {}))
        for k in n["inputs"]:
            if k not in known:
                problems.append(f"{nid} {ct}: unknown input {k} (has {sorted(known)})")
        for k in spec.get("required", {}):
            if k not in n["inputs"]:
                problems.append(f"{nid} {ct}: required input {k} not set")
        for k, v in n["inputs"].items():
            s = spec.get("required", {}).get(k) or spec.get("optional", {}).get(k)
            if not s or isinstance(v, list):
                continue
            opts = s[0] if isinstance(s[0], list) else (s[1] or {}).get("options") if s[0] == "COMBO" and len(s) > 1 else None
            if opts is not None and v not in opts:
                problems.append(f"{nid} {ct}: {k}={v!r} not on the PC")
    if problems:
        raise RuntimeError("graph does not match the PC:\n" + "\n".join(problems))


def compare(src, plate, out_mp4, out_jpg):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", plate, "-filter_complex",
                    "[1:v][0:v]scale2ref[p][s];[s][p]hstack", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-an", out_mp4], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out_mp4, "-vf",
                    "select='not(mod(n\\,24))',scale=960:-2,tile=1x5", "-frames:v", "1", out_jpg], check=True)


def main():
    p = json.loads(sys.argv[1])
    out = os.path.abspath(sys.argv[2])
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):  # the python engine's out dir is a shared cache: clear it
        q = os.path.join(out, name)
        shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)
    work = os.path.join(out, "work")
    os.makedirs(work)
    ensure_comfy()
    plate_lora, distilled_lora = float(p.get("plate_lora", 1.0)), float(p.get("distilled_lora", 1.0))
    report = {"plate_lora": plate_lora, "distilled_lora": distilled_lora, "clips": []}
    for c in p["clips"]:
        name = c["name"]
        raw = os.path.join(work, f"{name}_raw.mp4")
        with httpx.stream("GET", c["clip"], timeout=T, follow_redirects=True) as r:
            r.raise_for_status()
            with open(raw, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        w, h = int(c.get("width", 1024)), int(c.get("height", 576))
        src = os.path.join(out, f"{name}_src.mp4")
        frames = prep(raw, src, w, h, c.get("frames"))
        graph = build(upload(src), c["prompt"], c.get("negative", NEG), w, h, frames,
                      int(c.get("seed", 6332)), f"ltx_plate/{name}", plate_lora, distilled_lora)
        check_graph(graph)
        before = vram()
        t0 = time.monotonic()
        try:
            outs = run_outputs(graph)
        except RuntimeError as e:
            log(f"{name}: FAILED {e}")
            report["clips"].append({"name": name, "error": str(e)[:2000], "size": [w, h], "frames": frames})
            free()
            continue
        secs = round(time.monotonic() - t0)
        plate = fetch_first(outs, os.path.join(out, f"{name}_plate.mp4"))
        free()
        compare(src, plate, os.path.join(out, f"{name}_compare.mp4"), os.path.join(out, f"{name}_strip.jpg"))
        log(f"{name}: {w}x{h} {frames}f in {secs}s (vram before {before})")
        report["clips"].append({"name": name, "seconds": secs, "size": [w, h], "frames": frames, "vram_before": before})
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
