# faceswap_h3: face swap by H3 motion transfer

Puts the person from a photo into a target video's performance (head moves, expression, lips, hands), keeping
the person's own face shape and hair and optionally the target's headphones. Built 2026-10-23 for an internal,
consented test. `fsh3.py` reproduces the accepted result (`FINAL_v4_H3_motion_headphones_less-stubble.mp4`) at
38.6 dB PSNR from the same renders.

## Method

MiniMax H3 ref2va reference-to-video (farm `videogen.graphs.build("r2v", ...)`, turbo 4-step LoRA, 448x768):

| input | what |
|---|---|
| `<Picture 1>` | the person's ORIGINAL photo (gives face + hair) |
| `<Picture 2>` | the target's headphones, cut out of a clean frame of the target video |
| `<Video 1>` | the target clip with its hair ERASED (gives motion only) |

The prompt (`prompt()` in `fsh3.py`) says: Picture 1's man performs Video 1 exactly, keeps his identity
(`face_desc`), wears Picture 2's headphones, and matches the gestures, framing, outfit and room (`scene_desc`).
Both descriptions live in `out/<name>/config.json`; edit them for each person and target.

## Steps

```
.venv\Scripts\python.exe fsh3.py prep --name JOB --face photo.jpg --target video.mp4
.venv\Scripts\python.exe fsh3.py masks --name JOB             # SAM3 on local ComfyUI: hair/headphones/face/hand
.venv\Scripts\python.exe fsh3.py cutout --name JOB --frame 300  # a frame with both cups visible, no hand in front
.venv\Scripts\python.exe fsh3.py mole --name JOB --frame 301 --x 150 --y 515   # optional, see below
.venv\Scripts\python.exe fsh3.py erase-hair --name JOB
.venv\Scripts\python.exe fsh3.py pieces --name JOB            # seeds from config (43,44)
.venv\Scripts\python.exe fsh3.py cloud --name JOB             # one-shot RTX 5090 pod, downloads renders/
.venv\Scripts\python.exe fsh3.py score --name JOB             # best seed per piece -> picks.json
.venv\Scripts\python.exe fsh3.py finish --name JOB            # -> out/JOB/final.mp4 + final_side-by-side.mp4
```

Setup: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`. The tool needs:
- the local ComfyUI (127.0.0.1:8188) with the SAM3 weights, for `masks` and `finish`;
- render-farm `.env` (RunPod keys), for `cloud`;
- the farm's `worker/videogen` graph builders (imported, not copied).

Run the local steps only while the farm is idle, because they share the 4080 with `video_gen`/`clip_edit` jobs.

## Why each step exists

- **Original photo, not an edited one.** Generating the photo with headphones (Qwen-Image-Edit) restyled the hair
  into a bowl fringe. Instead the headphones go in as their own reference (`<Picture 2>`), which keeps the real hair.
- **Erase the target's hair from `<Video 1>`.** Otherwise H3 copies the target's long side hair below the cups.
  The erase mask is SAM3 hair dilated 51 px, minus face, hands and headphones (slightly dilated), filled with Telea.
  A clean 5 s test is not enough: 10 s pieces regrew the hair until it was erased.
- **Optional `mole`.** It removes a dark mole from the target's face before H3 can copy it. The tool LK-tracks it from
  a seed point and snaps to the dark spot. This only partly helps; the chin clean in `finish` does most of the work.
- **10 s pieces with 2 s overlaps** (`PIECE=243`, `STRIDE=195`). 5 s pieces joined by short crossfades show the
  face jumping in size and colour at every join, and 15-frame overlaps were still visible. A single long pass
  runs out of memory on the 5090.
- **Two seeds, picked by sync (`score`).** An H3 piece can silently improvise its motion for several seconds.
  For the test video, s43 drifted at 16-26 s and s44 fixed it. The score is a windowed (3 s) correlation of the
  motion energy against the target: about 0.8 is normal and below 0.3 means that stretch drifted. If every seed
  for a piece scores a mean below 0.45, re-roll it with `pieces --only <start> --seeds 45,46,47` and then `cloud`.
- **Chin clean on the OUTPUT (`_chin_clean`).** H3 copies the target's mole. Small dark specks in the bottom 22%
  of a SAM3 face mask (mouth excluded) are filled with Telea. Tracking the mole in the source drifted over 34 s
  and hit the shirt, so the cleanup runs on masks from the output itself.
- **Stitch (`_stitch`).** It colour-matches pieces with a LAB mean/std transform chained over the overlaps, then
  crossfades for 24 frames. The crossfade is capped at the overlap length, otherwise it blends in the wrong frames.
- **Stubble reduction (`_destubble`).** H3 adds a chin shadow by itself; cleaning it out of the reference did
  nothing. The fix ADDS a smooth L/a/b offset that pulls the chin towards a cheek tone averaged over ±7 frames,
  plus a 5 px luminance closing for dots. The zone is the face mask averaged over ±3 frames and ramped in from 66%
  to 80% of the face height, with the mouth excluded. An earlier version replaced chin pixels with a bilateral-blurred
  copy and used a per-frame reference colour. Shawn rejected it as blurry and flickery. Never do either; check
  flicker by looking at consecutive frames, because an average metric missed it.

## Cloud

`cloud.py` and `pod/pod.sh` create a one-shot RunPod pod: RTX 5090, EU-RO-1, on the farm network volume.
- **Isolation:** the pod is named `fsh3-*`, which the farm's cloud lane ignores.
- **What it runs:** it unpacks the farm's ComfyUI bundle and copies in only the H3 ref2va, text-encoder, VAE and
  turbo-LoRA weights from the volume, so nothing is installed or downloaded. Then it runs `graphs/*.json` and
  writes to `fsh3/jobs/<job>/out/`.
- **Shutdown:** the pod removes itself, with a 2 h cap. `wait_job` deletes it if it is still listed afterwards,
  because self-removal failed once.

Costs and limits:
- **Time:** about 80 s per 10 s piece, plus a few minutes of boot. A 34 s video with two seeds is about 8 renders,
  roughly 15 min of pod time, around $0.25 at ~$1.19/h.
- **5090 availability:** RTX 5090s are often unavailable in EU-RO-1; `create_pod` retries every 30 s for up to 30 min.
- **Disk:** a 250 GB container disk gets "no instances"; use 200 GB.

## What did not work (don't retry)

- **BFS head-swap LoRAs** (LTX-2.5, H3, LTX-2.3): the target's hair leaked through and the target's jaw stayed.
- **H3 masked inpaint to remove hair:** it only nudges masked pixels towards their surroundings, so the hair regrows.
- **LTX clean-plate IC-LoRA:** it removes whole people, not hair.
- **Feeding H3 a motion reference smaller than 448x768:** the side hair came back.
