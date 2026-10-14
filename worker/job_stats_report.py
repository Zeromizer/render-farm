"""Per-engine summary of worker/job_stats.jsonl (written by job_stats.py).

    .venv\Scripts\python.exe worker\job_stats_report.py [--days N] [--jobs]
"""
import argparse
import collections
import json
import os
from datetime import datetime, timedelta, timezone

from job_stats import STATS_FILE

COLS = [("vram_job_peak_mib", "vram+ MiB"), ("vram_peak_mib", "vram peak"), ("gpu_util_avg", "gpu avg%"),
        ("ram_job_peak_gb", "ram+ GB"), ("ram_peak_gb", "ram peak"), ("worker_tree_peak_gb", "tree GB"),
        ("comfyui_peak_gb", "comfy GB"), ("cpu_avg", "cpu avg%"), ("seconds", "secs")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--jobs", action="store_true", help="also list each job")
    ap.add_argument("--solo", action="store_true",
                    help="only jobs that had the box to themselves (no light-lane overlap)")
    a = ap.parse_args()
    if not os.path.exists(STATS_FILE):
        print("no stats yet")
        return
    cut = datetime.now(timezone.utc) - timedelta(days=a.days)
    recs = []
    with open(STATS_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00")) >= cut:
                if not (a.solo and r.get("max_concurrent", 1) > 1):
                    recs.append(r)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["engine"]].append(r)
    over = sum(1 for r in recs if r.get("max_concurrent", 1) > 1)
    print(f"{len(recs)} jobs ({over} overlapped another), last {a.days:g} days. Each cell: median / max")
    print(f"{'engine':18s} {'n':>4s} " + " ".join(f"{h:>15s}" for _, h in COLS))
    for eng, rs in sorted(by.items()):
        cells = []
        for k, _ in COLS:
            v = sorted(x[k] for x in rs if x.get(k) is not None)
            cells.append(f"{v[len(v) // 2]:>7g}/{v[-1]:<7g}" if v else f"{'-':>15s}")
        print(f"{eng:18s} {len(rs):>4d} " + " ".join(cells))
    if a.jobs:
        for r in recs:
            print(r["finished_at"], r["engine"], r["job_id"][:8], r["outcome"], r.get("lane", "main"),
                  "overlapped" if r.get("max_concurrent", 1) > 1 else "solo",
                  " ".join(f"{h}={r.get(k)}" for k, h in COLS))


if __name__ == "__main__":
    main()
