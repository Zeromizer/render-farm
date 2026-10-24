"""Read-only: list a render-platform job workspace on the PC (sizes, media, what index.html references)."""
import json, os, re, sys
ws = sys.argv[1]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
rep = {"ws": ws, "files": []}
total = 0
for root, dirs, files in os.walk(ws):
    dirs[:] = [d for d in dirs if d not in ("node_modules", ".git")]
    for f in files:
        p = os.path.join(root, f); s = os.path.getsize(p); total += s
        rel = os.path.relpath(p, ws)
        if rel.lower().endswith((".mp4", ".mov", ".webm", ".png", ".jpg", ".jpeg", ".webp", ".html", ".json", ".mp3", ".wav", ".m4a", ".css", ".js")):
            rep["files"].append([rel, s])
rep["total_bytes_excl_node_modules"] = total
rep["has_node_modules"] = os.path.isdir(os.path.join(ws, "node_modules"))
idx = os.path.join(ws, "index.html")
if os.path.exists(idx):
    html = open(idx, encoding="utf-8", errors="replace").read()
    rep["index_len"] = len(html)
    rep["media_refs"] = sorted(set(re.findall(r'''(?:src|href)=["']([^"']+\.(?:mp4|mov|webm|png|jpg|jpeg|webp|mp3|wav|m4a))["']''', html, re.I)))
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(html)
json.dump(rep, open(os.path.join(out, "probe.json"), "w"), indent=1)
print(json.dumps({k: rep[k] for k in rep if k != "files"})[:3000])
