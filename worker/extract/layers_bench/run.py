"""Run the layers stage over real files on the farm and bring the verdicts back.

A farm python job (engine "python", requirements worker/extract/requirements.txt
so it reuses the extractor's cached venv):

  script  worker/extract/layers_bench/run.py
  args    bench_out name=path [name=path ...]
  output  bench_out
  assets  the videos, placed at those paths (content-addressed, so a
          reference already in the assets bucket needs no upload)

For each video it runs extract.py --stages layers (which pulls in the text
pass), then keeps the layers section, the blocks it judged and the sheets, plus
<name>_dump/*.npz: the letters and subject mask of every frame judged, which
replay.py re-judges anywhere numpy and cv2 are (no detector, no video).
Child output is streamed line by line: a quiet python job cannot be cancelled.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(os.path.dirname(HERE), "extract.py")


def main():
    out_root = os.path.abspath(sys.argv[1])
    os.makedirs(out_root, exist_ok=True)
    jobs = [a.split("=", 1) for a in sys.argv[2:]]
    summary = {}
    for n, (name, path) in enumerate(jobs):
        work = os.path.join(out_root, f"_{name}")
        print(f"== {name}: {path}", flush=True)
        p = subprocess.Popen([sys.executable, "-u", EXTRACT, os.path.abspath(path), work, "--stages", "layers"],
                             cwd=os.path.dirname(EXTRACT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env={**os.environ, "LAYERS_DUMP": os.path.join(out_root, f"{name}_dump")},
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(f"  {line.rstrip()}", flush=True)
        code = p.wait()
        spec_path = os.path.join(work, "motion_spec.json")
        if code != 0 or not os.path.exists(spec_path):
            summary[name] = {"error": f"extract exited {code}"}
        else:
            with open(spec_path, encoding="utf-8") as f:
                spec = json.load(f)
            blocks = (spec.get("text") or {}).get("blocks") or []
            layers = spec.get("layers") or {}
            for row in layers.get("blocks") or []:
                b = blocks[row["block"]] if row.get("block", -1) < len(blocks) else {}
                row["text"], row["t0"], row["t1"] = b.get("text"), b.get("t0"), b.get("t1")
                row["size"] = b.get("size_ratio")
            summary[name] = {"layers": layers, "timings": spec.get("timings")}
            for fn in sorted(os.listdir(work)):
                if fn.startswith("kf_layers_"):
                    shutil.copy(os.path.join(work, fn), os.path.join(out_root, f"{name}_{fn}"))
        shutil.rmtree(work, ignore_errors=True)
        print(f"PROGRESS {int(100 * (n + 1) / len(jobs))}", flush=True)
    with open(os.path.join(out_root, "layers_bench.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
