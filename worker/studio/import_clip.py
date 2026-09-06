"""Add an existing mp4 to the studio library (run while the studio is stopped; it loads items at start).

  ..\.venv\Scripts\python studio\import_clip.py C:\path\clip.mp4 --label "name" --prompt "..." --kind generate
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from studio import server  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--label", default=""); ap.add_argument("--prompt", default="")
ap.add_argument("--kind", default="generate", choices=("generate", "upscale", "interpolate", "turntable"))
ap.add_argument("--remote", help="bucket path of the clip if it came from the farm (outputs/<job>.mp4)")
ap.add_argument("--piece", action="append", default=[], help="extra local mp4s to attach as pieces")
a = ap.parse_args()
server._load_items()
it = server._new_item(a.kind, label=a.label, prompt=a.prompt)
pieces = []
for i, p in enumerate(a.piece):
    rel = f"videos/{it['id']}_piece{i + 1}.mp4"
    import shutil; shutil.copyfile(p, os.path.join(server.ROOT, rel)); pieces.append(rel)
server._set(it, pieces=pieces)
server._finalize(it, a.video, remote=a.remote)
print("imported", it["id"], it["info"])
