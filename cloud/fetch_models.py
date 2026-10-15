"""Download the public model files in cloud/models-manifest.json onto the volume.

Run on a pod (data-centre bandwidth; far faster than uploading from home):
    python cloud/fetch_models.py [--root /workspace/ComfyUI] [--verify]
Entries with source "local-only" are skipped and listed: the PC uploads those
over S3. HF_TOKEN in the environment is used for gated repos. Each file is
written to <path>.part and renamed once its sha256 matches the manifest.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def present(dest, e, verify):
    if not os.path.exists(dest):
        return False
    if verify and e.get("sha256"):
        return sha256(dest) == e["sha256"]
    return not e.get("bytes") or os.path.getsize(dest) == e["bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/ComfyUI")
    ap.add_argument("--manifest", default=os.path.join(HERE, "models-manifest.json"))
    ap.add_argument("--verify", action="store_true", help="re-hash files already present")
    a = ap.parse_args()
    with open(a.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest["files"] if isinstance(manifest, dict) else manifest
    token = os.environ.get("HF_TOKEN")
    waiting, failed = [], []
    for e in entries:
        dest = os.path.join(a.root, e["path"])
        if present(dest, e, a.verify):
            continue
        if e.get("source") == "local-only" or not e.get("url"):
            waiting.append(e["path"])
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        req = urllib.request.Request(e["url"])
        if token and "huggingface.co" in e["url"]:
            req.add_header("Authorization", f"Bearer {token}")
        print(f"get {e['path']} ({e.get('bytes', 0) / 1e9:.1f} GB)", flush=True)
        try:
            with urllib.request.urlopen(req) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 24):
                    f.write(chunk)
            if e.get("sha256") and sha256(part) != e["sha256"]:
                raise RuntimeError("sha256 mismatch")
            os.replace(part, dest)
        except Exception as ex:  # noqa: BLE001 - report every failure, exit 1 at the end
            failed.append(f"{e['path']}: {ex}")
            if os.path.exists(part):
                os.remove(part)
    if waiting:
        print("\nwaiting for the PC's S3 upload (local-only):\n  " + "\n  ".join(waiting))
    if failed:
        print("\nFAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
