"""Lip-sync measure: does the mouth on screen follow the soundtrack, and by how much is it off?

Pure compute (no Supabase, no worker imports): runs in its own cached venv
(worker/venvs.py keyed on lipsync/requirements.txt); runners/lipsync_check.py owns the I/O.

It is SyncNet (Chung & Zisserman, "Out of time: automated lip sync in the wild", ACCV-W 2016)
exactly as joonson/syncnet_python 907c0b5 runs it: S3FD face detection on 25 fps frames,
PySceneDetect shots, IoU face tracks, a 224x224 face crop per track with its own audio, then
SyncNetInstance.evaluate (vshift 15) -> AV offset, min dist and confidence. The vendored files in
lipsync/vendor are unmodified; track_shot / crop_video / bb_intersection_over_union below are
copied from their run_pipeline.py (a script with argparse at import time, so it cannot be
imported). Reproduced their demo before any change: example.avi -> offset 3, min dist 5.358,
confidence 10.081 (README: 3 / 5.353 / 10.021).

Our deviations, all parameters, none in the model path:
  * min_track 25 frames (1 s) instead of 100: ad shots run 1.5-4 s.
  * each window is trimmed out of the video first; without windows the whole video is one window
    and every face track in it is reported (the talking spans find themselves).
  * speech_fraction (share of 40 ms audio frames above the window's speech floor) so a silent or
    music-only track reads as no speech rather than as a bad sync.

Offset sign (measured 2026-11-19 on their example.avi, offset 3): delaying its audio by 160 ms
(4 frames) gave 0, advancing it by 160 ms gave 8. So SyncNet's "AV offset" is the number of 25 fps
frames the AUDIO must be DELAYED to sync, i.e. positive = the voice comes EARLY. We report
offset_frames_25 = -(their offset), so positive = the voice comes LATE, negative = early.

Usage: lipsync_check.py <video> <out.json> --models-dir D --work-dir W [--windows windows.json]
       [--fps-out 30] [--facedet-scale 0.25] [--min-track 25] [--min-face-size 100]
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")

# syncnet_python's download_model.sh, pinned by content.
WEIGHTS = {
    "syncnet_v2.model": ("http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
                         "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"),
    "sfd_face.pth": ("https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth",
                     "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491"),
}

# Calibrated 2026-11-19 on the laptop CPU (full numbers in runners/lipsync_check.py):
#   unsynced  MaczTech 431eebf5 v3, 10 face tracks: confidence 0.55-2.16, min dist 10.96-14.95
#   in sync   real presenter clips (Downloads "video (42)", "0715 v3"), 5 tracks: 2.93-8.18, 7.08-11.80
#   their demo example.avi: 8.34 (shifted copies 7.39 / 7.70)
# The weakest positive (2.93, a 157 px full-body face) still moved by exactly +4 frames when its
# voice was delayed 160 ms, so it is a real signal: one threshold between the populations.
CONF_MIN = 2.5          # below: the mouth does not follow this voice (no_sync_signal)
CONF_STRONG = 4.0       # below this a verdict says "weak" in its reason
# Offset tolerance in 25 fps frames, from ITU-R BT.1359 acceptability (voice early <= 90 ms,
# late <= 185 ms), kept inside it: early <= 2 frames (80 ms), late <= 3 frames (120 ms).
OFFSET_EARLY_OK = 2
OFFSET_LATE_OK = 3
SPEECH_MIN = 0.25       # a track with less speech than this is judged no speech
REF_BIAS = 0            # syncnet_v2's offset on in-sync material (see header)


def log(msg):
    print(msg, flush=True)


def progress(n):
    print(f"PROGRESS {int(n)}", flush=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(models_dir):
    os.makedirs(models_dir, exist_ok=True)
    out = {}
    for name, (url, sha) in WEIGHTS.items():
        path = os.path.join(models_dir, name)
        if not (os.path.exists(path) and _sha256(path) == sha):
            log(f"downloading {name}")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            got = _sha256(tmp)
            if got != sha:
                os.remove(tmp)
                raise RuntimeError(f"{name}: sha256 {got} != pinned {sha}")
            os.replace(tmp, path)
        out[name] = path
    return out


def find_ffmpeg_tool(name):
    """FFMPEG_DIR, PATH, then the winget layout (same search as extract/common.py)."""
    import glob

    env_dir = os.environ.get("FFMPEG_DIR")
    if env_dir:
        exe = os.path.join(env_dir, name + (".exe" if os.name == "nt" else ""))
        if os.path.exists(exe):
            return exe
    found = shutil.which(name)
    if found:
        return found
    pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages",
                           "Gyan.FFmpeg*", "ffmpeg-*", "bin", f"{name}.exe")
    hits = sorted(glob.glob(pattern), reverse=True)
    if hits:
        return hits[0]
    raise RuntimeError(f"{name} not found: set FFMPEG_DIR or install ffmpeg")


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


# ---------- copied from syncnet_python run_pipeline.py (907c0b5) ----------

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(opt, scenefaces):
    import numpy as np
    from scipy.interpolate import interp1d

    iouThres = 0.5
    tracks = []
    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if track == []:
                    track.append(face)
                    framefaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > opt.min_track:
            framenum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            frame_i = np.arange(framenum[0], framenum[-1] + 1)
            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_face_size:
                tracks.append({'frame': frame_i, 'bbox': bboxes_i})
    return tracks


def crop_video(opt, track, cropfile, flist, audio_wav, ffmpeg):
    import cv2
    import numpy as np
    from scipy import signal

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut = cv2.VideoWriter(cropfile + 't.avi', fourcc, opt.frame_rate, (224, 224))
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs = opt.crop_scale
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image = cv2.imread(flist[frame])
        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi
        face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    audiotmp = cropfile + '_audio.wav'
    audiostart = (track['frame'][0]) / opt.frame_rate
    audioend = (track['frame'][-1] + 1) / opt.frame_rate
    vOut.release()
    _run([ffmpeg, "-y", "-loglevel", "error", "-i", audio_wav, "-ss", "%.3f" % audiostart, "-to", "%.3f" % audioend,
          audiotmp])
    _run([ffmpeg, "-y", "-loglevel", "error", "-i", cropfile + 't.avi', "-i", audiotmp, "-c:v", "copy", "-c:a", "copy",
          cropfile + '.avi'])
    os.remove(cropfile + 't.avi')
    return {'track': track, 'proc_track': dets, 'audio': audiotmp}

# ---------- end of the copy ----------


def detect_faces(DET, flist, opt, prog):
    import cv2

    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
        dets.append([{'frame': fidx, 'bbox': (b[:-1]).tolist(), 'conf': b[-1]} for b in bboxes])
        if fidx % 25 == 0:
            prog(fidx / max(1, len(flist)))
    return dets


def scene_detect(video_avi):
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(video_avi)
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    if not scenes:
        scenes = [(video.base_timecode, video.base_timecode + video.duration)]
    return [(a.get_frames(), b.get_frames()) for a, b in scenes]


def speech_fraction(wav_path, start_s=None, end_s=None):
    """Share of 40 ms frames whose RMS is within 30 dB of the loud end of the clip and above -50 dBFS."""
    import numpy as np
    from scipy.io import wavfile

    sr, a = wavfile.read(wav_path)
    a = a.astype(np.float64) / 32768.0
    if start_s is not None:
        a = a[int(start_s * sr):int(end_s * sr)]
    hop = int(sr * 0.04)
    n = len(a) // hop
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(a[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    floor = max(-50.0, float(np.percentile(db, 95)) - 30.0)
    return float(np.mean(db > floor))


def verdict(face, conf, offset, speech):
    """offset: 25 fps frames, + = voice late."""
    if not face:
        return "no_face", "no face big enough to read lips on (>= min_face_size px) for >= min_track frames"
    if speech < SPEECH_MIN:
        return "no_sync_signal", f"little speech under this face ({speech:.0%} of the span)"
    if conf < CONF_MIN:
        return "no_sync_signal", f"the mouth does not follow the voice (confidence {conf:.2f} < {CONF_MIN})"
    weak = " (weak face signal)" if conf < CONF_STRONG else ""
    if -OFFSET_EARLY_OK <= offset <= OFFSET_LATE_OK:
        return "in_sync", f"confidence {conf:.2f}, voice {offset * 40:+d} ms{weak}"
    late = "late" if offset > 0 else "early"
    return f"off_by_{abs(offset)}_frames", f"the voice is {abs(offset) * 40} ms {late} (confidence {conf:.2f}){weak}"


def measure_window(win, video, work, weights, opt, fps_out, S, DET, ffmpeg, prog):
    import glob

    import numpy as np

    ref = win["label"]
    d = os.path.join(work, ref)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "frames"))
    avi = os.path.join(d, "video.avi")
    trim = []
    if win.get("start_s") is not None:
        trim = ["-ss", "%.3f" % win["start_s"], "-to", "%.3f" % win["end_s"]]
    # run_pipeline: convert to 25 fps (-qscale:v 2 -async 1 -r 25), frames, 16 kHz mono audio
    _run([ffmpeg, "-y", "-loglevel", "error", *trim, "-i", video, "-qscale:v", "2", "-async", "1", "-r", "25", avi])
    _run([ffmpeg, "-y", "-loglevel", "error", "-i", avi, "-qscale:v", "2", "-threads", "1", "-f", "image2",
          os.path.join(d, "frames", "%06d.jpg")])
    wav = os.path.join(d, "audio.wav")
    _run([ffmpeg, "-y", "-loglevel", "error", "-i", avi, "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", wav])
    flist = sorted(glob.glob(os.path.join(d, "frames", "*.jpg")))
    base = win.get("start_s") or 0.0

    faces = detect_faces(DET, flist, opt, lambda f: prog(f * 0.6))
    tracks = []
    for a, b in scene_detect(avi):
        if b - a >= opt.min_track:
            tracks.extend(track_shot(opt, faces[a:b]))
    prog(0.65)

    results = []
    for i, tr in enumerate(tracks):
        cropfile = os.path.join(d, "%05d" % i)
        crop = crop_video(opt, tr, cropfile, flist, wav, ffmpeg)
        eopt = argparse.Namespace(batch_size=20, vshift=15, tmp_dir=os.path.join(d, "pytmp"), reference="t%05d" % i)
        offset, conf, _dists = S.evaluate(eopt, videofile=cropfile + ".avi")
        off = -int(offset) - REF_BIAS  # positive = voice late (see header)
        t0 = base + tr["frame"][0] / 25.0
        t1 = base + (tr["frame"][-1] + 1) / 25.0
        sp = speech_fraction(wav, tr["frame"][0] / 25.0, (tr["frame"][-1] + 1) / 25.0)
        size = float(np.mean(np.maximum(tr["bbox"][:, 2] - tr["bbox"][:, 0], tr["bbox"][:, 3] - tr["bbox"][:, 1])))
        v, why = verdict(True, float(conf), off, sp)
        results.append({"start_s": round(t0, 3), "end_s": round(t1, 3), "face_px": round(size),
                        "offset_frames_25": off, "offset_frames": round(off * fps_out / 25.0, 2),
                        "offset_ms": off * 40, "confidence": round(float(conf), 3),
                        "min_dist": round(float(_min_dist(_dists)), 3), "speech_fraction": round(sp, 3),
                        "verdict": v, "reason": why})
        prog(0.65 + 0.35 * (i + 1) / max(1, len(tracks)))
    shutil.rmtree(os.path.join(d, "frames"), ignore_errors=True)
    return results


def _min_dist(dists):
    import numpy as np

    mdist = np.mean(np.stack([np.asarray(x) for x in dists], 1), 1)
    return float(np.min(mdist))


def summarise(win, tracks):
    """One verdict per window: the speaking face (highest confidence) decides."""
    out = {"label": win["label"], "start_s": win.get("start_s"), "end_s": win.get("end_s")}
    if not tracks:
        v, why = verdict(False, 0, 0, 0)
        return {**out, "face": False, "verdict": v, "reason": why, "tracks": []}
    best = max(tracks, key=lambda t: t["confidence"])
    keys = ("offset_frames_25", "offset_frames", "offset_ms", "confidence", "min_dist", "speech_fraction",
            "verdict", "reason")
    return {**out, "face": True, **{k: best[k] for k in keys}, "tracks": tracks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--windows")
    ap.add_argument("--fps-out", type=float, default=0)
    ap.add_argument("--facedet-scale", type=float, default=0.25)
    ap.add_argument("--min-track", type=int, default=25)
    ap.add_argument("--min-face-size", type=int, default=100)
    a = ap.parse_args()

    t_start = time.time()
    ffmpeg = find_ffmpeg_tool("ffmpeg")
    # SyncNetInstance.evaluate calls bare "ffmpeg": put ours first on PATH rather than edit their code.
    os.environ["PATH"] = os.path.dirname(ffmpeg) + os.pathsep + os.environ.get("PATH", "")
    fps_out = a.fps_out
    if not fps_out:
        ffprobe = find_ffmpeg_tool("ffprobe")
        r = _run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                  "-of", "csv=p=0", a.video]).strip()
        num, _, den = r.partition("/")
        fps_out = float(num) / float(den or 1)

    weights = ensure_weights(a.models_dir)
    progress(3)
    sys.path.insert(0, VENDOR)
    import torch  # noqa: F401
    from detectors import s3fd as s3fd_mod
    from SyncNetInstance import SyncNetInstance

    s3fd_mod.PATH_WEIGHT = weights["sfd_face.pth"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DET = s3fd_mod.S3FD(device=device)
    S = SyncNetInstance()
    S.loadParameters(weights["syncnet_v2.model"])
    log(f"models loaded on {device}")

    windows = [{"label": "all"}]
    if a.windows:
        with open(a.windows, encoding="utf-8") as f:
            windows = json.load(f) or windows
        for i, w in enumerate(windows):
            w.setdefault("label", f"w{i + 1}")
            if w.get("start_s") is None or w.get("end_s") is None or w["end_s"] <= w["start_s"]:
                raise ValueError(f"window {w['label']} needs start_s < end_s")

    opt = argparse.Namespace(facedet_scale=a.facedet_scale, crop_scale=0.40, min_track=a.min_track,
                             frame_rate=25, num_failed_det=25, min_face_size=a.min_face_size)
    out = []
    for i, w in enumerate(windows):
        log(f"window {w['label']}: {w.get('start_s', 0)}-{w.get('end_s', 'end')}")

        def prog(f, i=i):
            progress(5 + 95 * (i + f) / len(windows))

        tracks = measure_window(w, a.video, a.work_dir, weights, opt, fps_out, S, DET, ffmpeg, prog)
        s = summarise(w, tracks)
        log(f"  -> {s['verdict']}: {s['reason']}")
        out.append(s)

    counts = {}
    for s in out:
        key = "off" if s["verdict"].startswith("off_by_") else s["verdict"]
        counts[key] = counts.get(key, 0) + 1
    result = {"ok": True, "fps": fps_out, "device": device,
              "model": "SyncNet syncnet_v2 + S3FD (joonson/syncnet_python 907c0b5)",
              "thresholds": {"conf_min": CONF_MIN, "conf_strong": CONF_STRONG,
                             "early_ok_frames_25": OFFSET_EARLY_OK, "late_ok_frames_25": OFFSET_LATE_OK,
                             "speech_min": SPEECH_MIN},
              "windows": out, "counts": counts, "seconds": round(time.time() - t_start, 1)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    progress(100)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    main()
