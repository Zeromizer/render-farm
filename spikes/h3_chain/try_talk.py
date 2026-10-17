"""Spike: a ~45 s walking selfie vlog chained from H3 shots, no visible joins.

Shot 1 is FL2VA text-to-video. Every later shot is Ref2VA (the source workflow's model and
sampler: er_sde, beta, sigma shift 12/3, ref2v turbo-4) that opens on the previous shot's real
last `overlap` frames, picture and voice latents both held (AimH3Hold head), with frames of shot 1
as the character references. The film is shot 1 plus each later shot from frame `overlap` on;
each join sits inside the held stretch both clips share and is blended there (4 frames of
picture, 30 ms of sound), with the sound cut at the same frames as the picture.

Each shot gets its own spoken line, written to end before the shot's last ~1.6 s so the held
stretch carries a pause rather than a word.

params.args = [<job json string>, <out dir>]. Job JSON (all optional):
  {"scene", "lines": [...], "width": 480, "height": 832, "first_frames": 260, "frames": 243,
   "overlap": 39, "seed": 2024, "ref_frames": [24, 150]}
Writes <out>/{shot1..N.mp4, talk.mp4, sheet.jpg, joins.jpg, report.json}.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np

import try_chain as tc

SCENE = ("Vertical selfie vlog filmed on a phone held at arm's length. A stylish young woman in her late "
         "twenties with shoulder-length dark wavy hair, cream knit top and small gold hoop earrings walks "
         "along a sunny city sidewalk lined with parked cars and trees, talking straight into the phone in a "
         "playful, confident vlog style. The phone bobs gently with her steps and the street slides past "
         "behind her. Natural afternoon light, realistic handheld phone footage. Clear close-mic voice, "
         "soft street ambience.")
LINES = [
    "Okay, real talk. Ladies like hot cars. Don't let anyone tell you otherwise.",
    "And I don't mean loud and flashy. I mean a car that looks good just sitting there.",
    "Clean lines, a gorgeous colour, and an interior that feels special the second you get in.",
    "And electric? Silent, smooth, and so quick off the lights. Honestly, obsessed.",
    "So if you want to impress me, park something beautiful outside. That's it. Bye!",
]


def log(*a):
    print(*a, flush=True)


def prompt_for(scene, line, k):
    if k == 0:
        return f'{scene} She says: "{line}" Then she smiles and keeps walking for a moment.'
    return (f"The woman is the same woman as in <Picture 1> and <Picture 2>, same face, hair and outfit. {scene} "
            f'She keeps walking and talking, and says: "{line}" Then she smiles and keeps walking for a moment.')


def prompt_fresh(scene, line):
    """A fresh odd shot: the bridge lands on it ~3 s in, so it opens on a quiet walk (~4 s)."""
    return (f"The woman is the same woman as in <Picture 1> and <Picture 2>, same face, hair and outfit. {scene} "
            f'She walks quietly for about four seconds, glancing at the street, then says: "{line}" '
            "Then she smiles and keeps walking for a moment.")


def extract_png(video, idx, dest):
    tc.ff("-i", video, "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", dest)
    return dest


def make_chain(segs, dest, xf=4, axf=0.03, sr=48000):
    """segs = [(path, start, end)]: each clip's frames [start, end) in order. The frames just before a
    clip's start must be the same moments as the previous clip's last frames (a held stretch), and the
    joins are blended inside them. Returns (cumulative frame index of each join, total frames)."""
    spf = sr // tc.FPS
    first = cv2.VideoCapture(segs[0][0])
    h, w = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    first.release()
    wav = dest + ".f32"
    x = int(axf * sr)
    t = np.linspace(0, np.pi / 2, x, dtype=np.float32)[:, None]
    fo, fi = np.cos(t), np.sin(t)
    audio = None
    for pth, start, end in segs:
        da = tc.read_audio(pth, sr)
        if audio is None:
            audio = da[start * spf:end * spf]
            continue
        audio[-x:] = audio[-x:] * fo + da[start * spf - x:start * spf] * fi
        audio = np.concatenate([audio, da[start * spf:end * spf]])
    audio.astype(np.float32).tofile(wav)
    proc = subprocess.Popen([tc._ffmpeg("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", str(tc.FPS), "-i", "-", "-f", "f32le", "-ar", str(sr), "-ac", "2",
                             "-i", wav, "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac",
                             "-b:a", "192k", "-shortest", dest], stdin=subprocess.PIPE, creationflags=tc.NO_WINDOW)
    joins, written, held = [], 0, []  # held = the last xf frames, not written until the next clip arrives
    for ci, (pth, start, end) in enumerate(segs):
        cap = cv2.VideoCapture(pth)
        i = 0
        pre = []
        while True:
            ok, f = cap.read()
            if not ok or i >= end:
                break
            if i < start - (xf if ci else 0):
                i += 1
                continue
            if ci > 0 and i < start:
                pre.append(f)
                i += 1
                if len(pre) == xf:  # blend the previous clip's held tail with this clip's copy of it
                    for j in range(xf):
                        wgt = (j + 1) / (xf + 1)
                        g = (1 - wgt) * held[j].astype(np.float32) + wgt * pre[j].astype(np.float32)
                        proc.stdin.write(np.clip(g, 0, 255).astype(np.uint8).tobytes())
                        written += 1
                    held = []
                    joins.append(written)
                continue
            held.append(f)
            if len(held) > xf:
                proc.stdin.write(held.pop(0).tobytes())
                written += 1
            i += 1
        cap.release()
    for f in held:
        proc.stdin.write(f.tobytes())
        written += 1
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed writing the chain")
    os.remove(wav)
    return joins, written


def main():
    p = {"scene": SCENE, "lines": LINES, "width": 480, "height": 832, "first_frames": 260, "frames": 243,
         "overlap": 39, "seed": 2024, "ref_frames": [24, 150]}
    p.update(json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].strip() else {})
    out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "out")
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        q = os.path.join(out, name)
        shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else os.remove(q)
    work = os.path.join(out, "work")
    os.makedirs(work)
    run_id = uuid.uuid4().hex[:8]
    K = p["overlap"]
    report = {"params": {k: v for k, v in p.items() if k != "scene"}, "scene": p["scene"], "shots": [], "joins": []}

    tc.install_node()
    log("PROGRESS 3")
    lines = p["lines"]
    n_shots = len(lines)
    scheme = p.get("scheme", "chain")
    names = [f"talk_{run_id}_{k + 1}" for k in range(n_shots)]
    paths, refs = [None] * n_shots, []
    done = [0]

    def gen(k, g):
        t0 = time.monotonic()
        dest = tc.fetch_mp4(tc.run_outputs(g), os.path.join(out, f"shot{k + 1}.mp4"))
        secs = round(time.monotonic() - t0)
        n = len(tc.read_frames(dest, (16, 16)))
        report["shots"].append({"shot": k + 1, "line": lines[k], "frames": n, "seconds": secs})
        log(f"shot {k + 1}: {n} frames in {secs}s")
        paths[k] = dest
        done[0] += 1
        log(f"PROGRESS {5 + int(85 * done[0] / n_shots)}")

    def shot_params(k, prompt, frames):
        return {"prompt": prompt, "width": p["width"], "height": p["height"], "frames": frames, "overlap": K, "steps": 8}

    r2v = {"family": "ref2va", "sampler": "er_sde", "scheduler": "beta", "shift": (12, 3)}

    # shot 1: fresh FL2VA, or the newest earlier run's shot 1 (its latent is talk_<id>_1)
    if p.get("reuse_s1"):
        hits = sorted(glob.glob(os.path.join(tc.COMFY_DIR, "output", "h3_chain", "talk_*_1_0*.mp4")), key=os.path.getmtime)
        if not hits:
            raise RuntimeError("reuse_s1: no earlier talk shot 1")
        names[0] = "_".join(os.path.basename(hits[-1]).split("_")[:3])
        paths[0] = os.path.join(out, "shot1.mp4")
        shutil.copyfile(hits[-1], paths[0])
        report["shots"].append({"shot": 1, "line": lines[0], "reused": os.path.basename(hits[-1])})
        done[0] += 1
    else:
        gen(0, tc.build(shot_params(0, prompt_for(p["scene"], lines[0], 0), p["first_frames"]), int(p["seed"]),
                        f"h3_chain/{names[0]}", save_as=names[0]))
    refs = [tc.upload(extract_png(paths[0], i, os.path.join(work, f"ref{i}.png"))) for i in p["ref_frames"]]

    if scheme == "chain":
        # every shot continues the one before (chain depth grows: drift accumulates)
        for k in range(1, n_shots):
            gen(k, tc.build(shot_params(k, prompt_for(p["scene"], lines[k], k), p["frames"]), int(p["seed"]) + k,
                            f"h3_chain/{names[k]}", save_as=names[k], holds=[(names[k - 1], "head", K)],
                            ref_images=refs, **r2v))
        segs = [(paths[0], 0, None)] + [(paths[k], K, None) for k in range(1, n_shots)]
    else:
        # the source workflow's order: odd shots fresh from the same refs, even shots bridge them
        # (head = the previous shot's real end, tail = the next shot's real frames from `land_from`,
        # past its ease-in). No shot is more than one step from a fresh one.
        tail, land = int(p.get("tail", 22)), int(p.get("land_from", 51))
        odd_frames = int(p.get("odd_frames", 294))
        for k in range(2, n_shots, 2):
            gen(k, tc.build(shot_params(k, prompt_fresh(p["scene"], lines[k]), odd_frames), int(p["seed"]) + k,
                            f"h3_chain/{names[k]}", save_as=names[k], ref_images=refs, **r2v))
        for k in range(1, n_shots, 2):
            gen(k, tc.build(shot_params(k, prompt_for(p["scene"], lines[k], k), p["frames"]), int(p["seed"]) + k,
                            f"h3_chain/{names[k]}", save_as=names[k],
                            holds=[(names[k - 1], "head", K), (names[k + 1], "tail", tail, True, land)],
                            ref_images=refs, **r2v))
        segs = [(paths[0], 0, None)] + [(paths[k], K if k % 2 else land + tail, None) for k in range(1, n_shots)]
    tc.free()
    segs = [(pth, s, len(tc.read_frames(pth, (16, 16)))) for pth, s, _ in segs]
    report["segments"] = [{"shot": i + 1, "from": s, "to": e} for i, (_, s, e) in enumerate(segs)]

    final = os.path.join(out, "talk.mp4")
    joins, total = make_chain(segs, final)
    report["total_frames"], report["seconds"] = total, round(total / tc.FPS, 2)
    g = tc.read_frames(final, (240, 416), gray=True)
    strips = []
    fr = tc.read_frames(final, (240, 416))
    for j in joins:
        st = tc.seam_stats(g, j)
        st["audio"] = tc.audio_seam(final, j / tc.FPS)
        st["speed"] = tc.bg_speed(g, j, half=6)
        report["joins"].append(st)
        strips.append(np.hstack([fr[i] for i in range(j - 2, j + 2)]))
    if strips:
        cv2.imwrite(os.path.join(out, "joins.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 88])
    report["luma_per_second"] = [round(float(np.mean([x.mean() for x in g[i:i + 24]])), 1) for i in range(0, len(g), 24)]
    # luma per shot (does a long chain darken?)
    report["luma_per_shot"] = [round(float(np.mean([x.mean() for x in tc.read_frames(pth, (60, 104), gray=True)])), 1)
                               for pth in paths]
    tc.ff("-i", final, "-vf", "select='not(mod(n\\,90))',scale=160:-2,tile=6x2", "-frames:v", "1",
          os.path.join(out, "sheet.jpg"))
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    shutil.rmtree(work, ignore_errors=True)
    log(json.dumps(report["joins"], default=float)[:1500])
    log("PROGRESS 100")


if __name__ == "__main__":
    main()
