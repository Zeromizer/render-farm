"""Where does the PC's ComfyUI read models from? Prints launcher, extra paths and every
checkpoints/ dir holding the YuE2 ckpt. args = [<out dir>]."""
import glob, json, os, sys
import httpx
D = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI")
out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
lines = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)
for bat in glob.glob(os.path.join(D, "*.bat")):
    p("==", bat); p(open(bat, encoding="utf-8", errors="replace").read())
for y in glob.glob(os.path.join(D, "**", "extra_model_paths.yaml"), recursive=True)[:5]:
    p("==", y); p(open(y, encoding="utf-8", errors="replace").read())
for c in glob.glob(os.path.join(D, "**", "checkpoints", "yue2_3b_int8_convrot.safetensors"), recursive=True):
    p("ckpt at", c)
for c in glob.glob(os.path.join(D, "**", "audio_encoders"), recursive=True):
    p("audio_encoders dir", c, os.listdir(c))
for f in ("audio_encoders", "checkpoints", "loras"):
    try:
        p("GET /models/" + f, httpx.get("http://127.0.0.1:8188/models/" + f, timeout=20).text[:600])
    except Exception as e:
        p("GET /models/" + f, "failed", e)
try:
    st = httpx.get("http://127.0.0.1:8188/system_stats", timeout=20).json()
    p("system", json.dumps(st.get("system"), indent=0)[:1500])
except Exception as e:
    p("system_stats failed", e)
log = os.path.join(D, "comfyui-headless.log")
if os.path.exists(log):
    with open(log, encoding="utf-8", errors="replace") as f:
        head = f.read(6000)
    p("== log head"); p(head)
open(os.path.join(out, "diag.txt"), "w", encoding="utf-8").write("\n".join(lines))
