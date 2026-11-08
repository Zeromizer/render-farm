"""facecheck.py src.png kf.png name=photo.png ...: for each cast person, the painted keyframe's face most like their photo
-> {name: [cos to the photo, cos to the nearest source face, face width, change under it] or null}; the last is the
mean |kf - src| (0-255) over the torso box under the face, 1.5 face widths wide and 1.4 tall from .1 below the chin. Run by paint.py in $REMAKE_FACE_PY
(any python with onnxruntime + opencv); SCRFD det_10g + ArcFace w600k_r50 from $REMAKE_FACE_MODELS (insightface
buffalo_l). Standalone on purpose: the MCP's own venv needs neither library."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

M = Path(os.environ.get("REMAKE_FACE_MODELS", Path.home() / ".insightface/models/buffalo_l"))
o = ort.SessionOptions()
o.intra_op_num_threads, o.inter_op_num_threads = 4, 1  # all cores by default: it starved the render workers
DET = ort.InferenceSession(str(M / "det_10g.onnx"), o, providers=["CPUExecutionProvider"])
REC = ort.InferenceSession(str(M / "w600k_r50.onnx"), o, providers=["CPUExecutionProvider"])
ARC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
               np.float32)


def faces(rgb, thr=0.5, size=640):
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    pad = np.zeros((size, size, 3), np.uint8)
    pad[:round(h * s), :round(w * s)] = cv2.resize(rgb, (round(w * s), round(h * s)))
    blob = cv2.dnn.blobFromImage(pad, 1 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=False)
    outs = [x[0] if x.ndim == 3 else x for x in DET.run(None, {DET.get_inputs()[0].name: blob})]
    boxes, kpss, scores = [], [], []
    for i, st in enumerate([8, 16, 32]):
        n = size // st
        ac = np.repeat((np.stack(np.mgrid[:n, :n][::-1], -1).reshape(-1, 2) * st).astype(np.float32), 2, axis=0)
        sc, bb, kp = outs[i][:, 0], outs[i + 3] * st, outs[i + 6] * st
        k = sc >= thr
        boxes.append(np.concatenate([ac[k] - bb[k, :2], ac[k] + bb[k, 2:]], 1))
        kpss.append((np.tile(ac[k], 5) + kp[k]).reshape(-1, 5, 2))
        scores.append(sc[k])
    boxes, kpss, scores = np.concatenate(boxes) / s, np.concatenate(kpss) / s, np.concatenate(scores)
    keep = cv2.dnn.NMSBoxes(np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1).tolist(), scores.tolist(), thr, 0.4)
    out = []
    for j in np.array(keep).reshape(-1):
        m, _ = cv2.estimateAffinePartial2D(kpss[j].astype(np.float32), ARC)
        crop = cv2.dnn.blobFromImage(cv2.warpAffine(rgb, m, (112, 112)), 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False)
        e = REC.run(None, {REC.get_inputs()[0].name: crop})[0][0]
        out.append((boxes[j], e / np.linalg.norm(e)))
    return out


def load(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


k = load(sys.argv[2])
sr = cv2.resize(load(sys.argv[1]), k.shape[1::-1])
src, kf = faces(sr), faces(k)  # same canvas: matched by box centre
mid = lambda b: np.array([b[0] + b[2], b[1] + b[3]]) / 2


def torso(b):
    w, h = b[2] - b[0], b[3] - b[1]
    x0, x1 = (int(np.clip(v, 0, k.shape[1])) for v in (b[0] - .25 * w, b[2] + .25 * w))
    y0, y1 = (int(np.clip(v, 0, k.shape[0])) for v in (b[3] + .1 * h, b[3] + 1.5 * h))
    return round(float(np.abs(k[y0:y1, x0:x1].astype(float) - sr[y0:y1, x0:x1]).mean()), 1) if x1 - x0 > 3 and y1 - y0 > 3 else -1


res = {}
for a in sys.argv[3:]:
    name, photo = a.split("=", 1)
    ph = faces(load(photo))
    if not kf or not ph:  # no face in the keyframe, or none in the photo to compare with

        res[name] = None
        continue
    p = max(ph, key=lambda x: x[0][2] - x[0][0])[1]
    b, e = max(kf, key=lambda x: float(x[1] @ p))
    s = max(src, key=lambda x: -np.linalg.norm(mid(x[0]) - mid(b)))[1] @ e if src else 0.0
    res[name] = [round(float(e @ p), 3), round(float(s), 3), int(b[2] - b[0]), torso(b)]
print(json.dumps(res))
