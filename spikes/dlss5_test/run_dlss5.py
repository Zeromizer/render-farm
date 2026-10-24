"""Spike (2026-10-24): run videos through the PC ComfyUI's DLSS5 Enhance Video File node
(Blueforcer/ComfyUI-DLSS5-Enhancer) and collect the results.

usage: python run_dlss5.py OUT_DIR SPEC.json
SPEC: [{"src": "spikes/dlss5_test/in/x.mp4", "label": "x_15x_i080", "upscale": "1.5x (Quality)",
        "intensity": 0.8, "style": "Default", "max_frames": 0}, ...]
"""
import json, os, shutil, sys, time, uuid
import httpx

COMFY = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
spec = json.load(open(sys.argv[2]))
report = []
for i, s in enumerate(spec):
    src = os.path.abspath(s["src"])
    g = {
        "set": {"class_type": "DLSS5Settings", "inputs": {
            "upscaling_mode": s.get("upscale", "1x (DLAA / native)"), "nr_preset": "Default",
            "nr_style": s.get("style", "Default"), "nr_intensity": float(s.get("intensity", 0.8)),
            "local_tone_strength": float(s.get("tone", 1.0)), "local_structure_strength": float(s.get("structure", 1.5)),
            "skin_structure_strength": float(s.get("skin", 2.0)), "automatic_mask": True,
            "dlss_model_preset": "M", "motion": "auto", "scene_change_threshold": 0.24,
            "warmup_frames": int(s.get("warmup", 4)), "runtime_dir": ""}},
        "vid": {"class_type": "DLSS5EnhanceVideoFile", "inputs": {
            "video_path": src, "settings": ["set", 0], "codec": s.get("codec", "H.264"), "container": "MP4", "quality": s.get("quality", "Max"),
            "filename_prefix": s["label"], "output_directory": out, "max_frames": int(s.get("max_frames", 0)),
            "copy_audio": True, "verify_neural_rendering": True}},
    }
    t0 = time.time()
    r = httpx.post(COMFY + "/prompt", json={"prompt": g, "client_id": str(uuid.uuid4())}, timeout=120)
    if r.status_code != 200:
        report.append({"label": s["label"], "error": r.text[:3000]}); print(r.text[:3000]); continue
    pid = r.json()["prompt_id"]
    while True:
        time.sleep(3)
        e = httpx.get(COMFY + f"/history/{pid}", timeout=60).json().get(pid)
        if e and e.get("status", {}).get("completed") is not None and (e["status"].get("completed") or e["status"].get("status_str") == "error"):
            break
        print(f"PROGRESS {int(100 * i / len(spec))}", flush=True)
    st = e["status"]
    rep = {"label": s["label"], "seconds": round(time.time() - t0, 1), "status": st.get("status_str")}
    if st.get("status_str") == "error":
        rep["messages"] = st.get("messages")
    else:
        rep["outputs"] = e.get("outputs")
    report.append(rep); print(json.dumps(rep)[:1500], flush=True)
json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
try:   # keep the runtime's own log for the verify step
    for root in [os.path.expanduser("~")]:
        pass
except Exception:
    pass
