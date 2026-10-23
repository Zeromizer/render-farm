"""Pod-side: run one ComfyUI API graph and append a stats line (wall time, peak VRAM). Stdlib only.
usage: run_graphs.py <comfy_url> <stats.jsonl> graph.json"""
import json, subprocess, sys, threading, time, urllib.request, uuid


def http(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def vram():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.split()[0])
    except Exception:
        return 0


base, stats, path = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
peak, stop = [0], threading.Event()


def sample():
    while not stop.is_set():
        peak[0] = max(peak[0], vram())
        stop.wait(2)


threading.Thread(target=sample, daemon=True).start()
t0, rec = time.time(), {"graph": path.rsplit("/", 1)[-1]}
try:
    pid = http(f"{base}/prompt", {"prompt": json.load(open(path)), "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        time.sleep(3)
        h = http(f"{base}/history/{pid}")
        if pid in h:
            e = h[pid]
            rec["status"] = e.get("status", {}).get("status_str")
            rec["error"] = [m[1].get("exception_message") for m in e.get("status", {}).get("messages", []) if m[0] == "execution_error"]
            break
except Exception as ex:                      # e.g. ComfyUI died (OOM): record it, let the pod script carry on
    rec["status"], rec["error"] = "failed", str(ex)[:300]
stop.set()
rec.update(wall_s=round(time.time() - t0, 1), peak_vram_mib=peak[0])
print(json.dumps(rec), flush=True)
open(stats, "a").write(json.dumps(rec) + "\n")
