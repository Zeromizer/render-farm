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


def extract_png(video, idx, dest):
    tc.ff("-i", video, "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1", dest)
    return dest


def make_chain(paths, K, dest, xf=4, axf=0.03, sr=48000):
    """paths[0] whole, then paths[i][K:] for i >= 1; joins blended inside the shared held stretch.
    Returns the cumulative frame index of each join."""
    spf = sr // tc.FPS
    first = cv2.VideoCapture(paths[0])
    h, w = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    first.release()
    wav = dest + ".f32"
    x = int(axf * sr)
    t = np.linspace(0, np.pi / 2, x, dtype=np.float32)[:, None]
    fo, fi = np.cos(t), np.sin(t)
    audio = tc.read_audio(paths[0], sr)
    n0 = len(tc.read_frames(paths[0], (16, 16)))
    audio = audio[:n0 * spf]
    for pth in paths[1:]:
        da = tc.read_audio(pth, sr)
        n = len(tc.read_frames(pth, (16, 16)))
        audio[-x:] = audio[-x:] * fo + da[K * spf - x:K * spf] * fi
        audio = np.concatenate([audio, da[K * spf:n * spf]])
    audio.astype(np.float32).tofile(wav)
    proc = subprocess.Popen([tc._ffmpeg("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", str(tc.FPS), "-i", "-", "-f", "f32le", "-ar", str(sr), "-ac", "2",
                             "-i", wav, "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac",
                             "-b:a", "192k", "-shortest", dest], stdin=subprocess.PIPE, creationflags=tc.NO_WINDOW)
    joins, written, held = [], 0, []  # held = the last xf frames, not written until the next clip arrives
    for ci, pth in enumerate(paths):
        cap = cv2.VideoCapture(pth)
        i = 0
        pre = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if ci > 0 and i < K - xf:
                i += 1
                continue
            if ci > 0 and i < K:
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
    paths, refs = [], []
    n_shots = len(p["lines"])
    for k, line in enumerate(p["lines"]):
        sp = {"prompt": prompt_for(p["scene"], line, k), "width": p["width"], "height": p["height"],
              "frames": p["first_frames"] if k == 0 else p["frames"], "overlap": K, "steps": 8}
        name = f"talk_{run_id}_{k + 1}"
        if k == 0:
            g = tc.build(sp, int(p["seed"]), f"h3_chain/{name}", save_as=name)
        else:
            g = tc.build(sp, int(p["seed"]) + k, f"h3_chain/{name}", save_as=name, family="ref2va",
                         holds=[(f"talk_{run_id}_{k}", "head", K)], ref_images=refs,
                         sampler="er_sde", scheduler="beta", shift=(12, 3))
        t0 = time.monotonic()
        dest = tc.fetch_mp4(tc.run_outputs(g), os.path.join(out, f"shot{k + 1}.mp4"))
        secs = round(time.monotonic() - t0)
        n = len(tc.read_frames(dest, (16, 16)))
        report["shots"].append({"shot": k + 1, "line": line, "frames": n, "seconds": secs})
        log(f"shot {k + 1}: {n} frames in {secs}s")
        paths.append(dest)
        if k == 0:
            refs = [tc.upload(extract_png(dest, i, os.path.join(work, f"ref{i}.png"))) for i in p["ref_frames"]]
        log(f"PROGRESS {5 + int(85 * (k + 1) / n_shots)}")
    tc.free()

    final = os.path.join(out, "talk.mp4")
    joins, total = make_chain(paths, K, final)
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
