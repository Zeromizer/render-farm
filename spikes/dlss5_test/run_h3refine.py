"""Spike (2026-10-24): the Reddit "H3 3D latent upscale + 2-step refine at 0.4" on an existing mp4.

Uses the archived decoded-upscale graph (feat/h3-latent-upscale, copied to h3up.py) with two
changes: a turbo_lora override (the DMAD 4-step LoRA) and full_tail (whole-frame refine instead of
tiles: the 5090 has the VRAM). Driven from the laptop through an ssh tunnel to the pod's ComfyUI.

usage: python run_h3refine.py SRC.mp4 OUT_DIR LABEL --denoise 0.4 --steps 2 [--lora file] [--tile]
"""
import argparse, json, os, subprocess, sys, time, uuid
import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "worker"))
sys.path.insert(0, HERE)
import h3up  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out_dir"); ap.add_argument("label")
ap.add_argument("--denoise", type=float, default=0.4)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--lora", default="")
ap.add_argument("--tile", action="store_true")
ap.add_argument("--prompt", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--comfy", default="http://127.0.0.1:18188")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
    "-show_entries", "stream=width,height,nb_read_packets", "-of", "json", a.src]))["streams"][0]
w, h, n = int(probe["width"]), int(probe["height"]), int(probe["nb_read_packets"])
frames = h3up.av_boundary_frames(n)
trim = os.path.join(a.out_dir, f"src_{frames}f.mp4")
if not os.path.exists(trim):
    # video cut to the AV-exact length; audio left long (the prep node slices it)
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", a.src, "-map", "0:v", "-map", "0:a?",
                           "-filter:v", f"trim=end_frame={frames},setpts=PTS-STARTPTS", "-c:v", "libx264",
                           "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", trim])
u = h3up.validate_upscale_params({"method": "h3_latent_upscale", "variant": "decoded", "denoise": a.denoise,
                                  "steps_override": a.steps, "shorter_size": 1080, "seed": a.seed,
                                  "missing_audio_policy": "silence" if False else "error"},
                                 "upscale", (w, h), frames, dev=True)
u["turbo_lora"] = a.lora or None
u["full_tail"] = not a.tile
C = a.comfy
name = f"h3up_{uuid.uuid4().hex[:8]}.mp4"
g, meta = h3up.build_decoded_upscale(u, name, (w, h), f"h3up/{a.label}", prompt=a.prompt or None, probes=False)
h3up.check_links(g)
json.dump(g, open(os.path.join(a.out_dir, f"{a.label}-graph.json"), "w"), indent=1)
print(json.dumps(meta))
if a.dry:
    sys.exit(0)
with open(trim, "rb") as f:
    httpx.post(C + "/upload/image", files={"image": (name, f, "video/mp4")}, data={"overwrite": "true"},
               timeout=300).raise_for_status()
t0 = time.time()
r = httpx.post(C + "/prompt", json={"prompt": g, "client_id": str(uuid.uuid4())}, timeout=120)
if r.status_code != 200:
    print(r.text[:4000]); sys.exit(1)
pid = r.json()["prompt_id"]
while True:
    time.sleep(10)
    e = httpx.get(C + f"/history/{pid}", timeout=60).json().get(pid)
    if not e:
        continue
    st = e.get("status", {})
    if st.get("status_str") == "error" or (st.get("completed") is False and st.get("messages")
                                            and any(m[0] == "execution_error" for m in st["messages"])):
        print(json.dumps(st)[:6000]); sys.exit(1)
    if st.get("completed"):
        break
el = time.time() - t0
for node in e["outputs"].values():
    for key in ("images", "videos", "gifs"):
        for it in node.get(key, []):
            if it.get("type") != "output":
                continue
            dst = os.path.join(a.out_dir, f"{a.label}-raw.mp4")
            with httpx.stream("GET", C + "/view", params={"filename": it["filename"], "subfolder": it.get("subfolder", ""),
                                                          "type": it.get("type", "output")}, timeout=600) as resp:
                with open(dst, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            crop = h3up.crop_args(meta["refine"], meta["requested"])
            final = os.path.join(a.out_dir, f"{a.label}.mp4")
            vf = ["-vf", crop] if crop else []
            subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", dst, *vf, "-c:v", "libx264", "-crf", "12",
                                   "-pix_fmt", "yuv420p", "-c:a", "copy", final])
            print(f"done {a.label} in {el:.0f}s -> {final}")
json.dump({"meta": meta, "seconds": round(el), "label": a.label}, open(os.path.join(a.out_dir, f"{a.label}.json"), "w"))
