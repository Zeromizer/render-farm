"""Re-judge a bench run's dumps with the rule as it stands in layers.py.

  python replay.py <bench_out> [--brief] [name ...]

<bench_out>/<name>_dump/*.npz hold the letters and the subject mask of every
frame the bench judged (run.py sets LAYERS_DUMP). judge() needs nothing else,
so a threshold or a rule can be tried here in a second, on any machine with
numpy and cv2, instead of a detector run per try.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layers  # noqa: E402


def load(path):
    z = np.load(path, allow_pickle=False)
    h, w = (int(v) for v in z["shape"])
    bits = lambda key: np.unpackbits(z[key])[: h * w].reshape(h, w).astype(bool)
    text, label, was, t, t0 = (str(v) for v in z["meta"])
    return bits("strokes"), bits("mask"), {"text": text, "subject": label, "was": was, "t": t, "t0": t0}


def main():
    args = [a for a in sys.argv[1:] if a != "--brief"]
    brief = "--brief" in sys.argv
    root, only = args[0], set(args[1:])
    for d in sorted(os.listdir(root)):
        if not d.endswith("_dump") or (only and d[:-5] not in only):
            continue
        print(f"== {d[:-5]}")
        blocks = {}
        for fn in sorted(os.listdir(os.path.join(root, d))):
            strokes, mask, meta = load(os.path.join(root, d, fn))
            layer, nums = layers.judge(strokes, mask)
            why = nums.pop("why", "")
            flag = " " if layer == meta["was"] else "*"
            key = fn.split("_")[0] + " " + meta["text"][:24]
            blocks.setdefault(key, []).append(layer)
            if not brief:
                print(f" {flag} {meta['t']:>6} {layer:8} (was {meta['was']:8}) {meta['subject']:8} "
                      + " ".join(f"{k}={v}" for k, v in nums.items())
                      + f'  "{meta["text"][:24]}" {why}')
        # the verdict each block would get from these frames alone (frames with
        # no subject or no readable letters never reach judge(), so are not here)
        for key, seen in blocks.items():
            print(f"   block {key:32} {layers.settle(seen):8} {seen}")


if __name__ == "__main__":
    main()
