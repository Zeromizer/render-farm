# faceswap_h3: face swap by H3 motion transfer

Puts the person from a photo into a target video's performance (head moves, expression, lips, hands), keeping
the person's own face shape and hair, and optionally props the target wears (headphones, a cap, glasses). Built
2026-10-23 for an internal, consented test. Re-running `finish` on the original renders reproduces the accepted
result (`FINAL_v4_H3_motion_headphones_less-stubble.mp4`) at 38.6 dB PSNR, and the streamed finish matches the
in-memory one at 49.5 dB.

## Method

MiniMax H3 ref2va reference-to-video (farm `videogen.graphs.build("r2v", ...)`, turbo 4-step LoRA, ~480p):

| input | what |
|---|---|
| `<Picture 1>` | the person's ORIGINAL photo (gives face + hair) |
| `<Picture 2>`, `<Picture 3>`, ... | each prop to keep, cut out of a clean frame of the target video |
| `<Video 1>` | the target clip with its hair ERASED (gives motion only) |

The prompt (`prompt()` in `fsh3.py`) says: the subject from Picture 1 performs Video 1 exactly, keeps their
identity (`face_desc`), wears each prop picture (`props[].sentence`), and matches the gestures, framing, outfit and
room (`scene_desc`). All of it lives in `out/<name>/config.json`.

## Steps

```
.venv\Scripts\python.exe fsh3.py prep --name JOB --face photo.jpg --target video.mp4 --consent "who consented, to what"
#   then set in out\JOB\config.json: "subject"/"possessive" (e.g. "the woman"/"her"; default "the person"/"their"),
#   "props" (e.g. [{"noun": "headphones"}]), "destubble": true only for a clean-shaven subject
.venv\Scripts\python.exe fsh3.py masks --name JOB        # SAM3 on local ComfyUI: face, hair, hand, each prop
.venv\Scripts\python.exe fsh3.py cutout --name JOB       # props from a frontal frame (--noun N --frame F to override)
.venv\Scripts\python.exe fsh3.py describe --name JOB     # fills empty face/scene/prop text (--dry to preview); REVIEW it
.venv\Scripts\python.exe fsh3.py mole --name JOB --frame 301 --x 150 --y 515   # optional, see below
.venv\Scripts\python.exe fsh3.py erase-hair --name JOB
.venv\Scripts\python.exe fsh3.py pieces --name JOB       # one graph per piece, first seed
.venv\Scripts\python.exe fsh3.py render --name JOB       # 5090 pod -> score -> re-roll drifted pieces -> ...
.venv\Scripts\python.exe fsh3.py finish --name JOB       # -> final.mp4, final_side-by-side.mp4, final.json
```

`cloud` (one pod run of the pending graphs) and `score` (sync table + picks) are also separate commands. Every
step is a plain function taking a `Job` (`fsh3.Job(name, root=None)`), so an engine can call them directly.

Setup: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`. The tool needs:
- the local ComfyUI (`COMFYUI_URL`, default 127.0.0.1:8188; `COMFYUI_DIR`, default `C:\ComfyUI`) with SAM 3.1
  (`masks`, `finish`) and Qwen3-VL 4B `qwen3vl_4b_fp8_scaled` (`describe`). Both are on the cloud volume too;
- render-farm `.env` (RunPod keys) for `cloud`/`render`, or `FSH3_ENV=<path>` from a worktree. `FSH3_GPU` and
  `FSH3_IMAGE` override the pod's GPU and image;
- the farm's `worker/videogen` graph builders (imported, not copied), and ffmpeg on PATH.

Run the local steps only while the farm is idle, because they share the 4080 with `video_gen`/`clip_edit` jobs.

## Settings (config.json)

| key | default | meaning |
|---|---|---|
| `subject`, `possessive` | `the person`, `their` | how the prompt names the person; set by whoever submits the job, never guessed |
| `face_desc`, `scene_desc` | empty | filled by `describe` when empty; keep them positive (naming a flaw tends to add it) |
| `props` | `[]` | `{"noun"}`; `sentence` (with `{pic}`) and `cutout_frame` get filled in |
| `ratio` | `auto` | working canvas: 9:16 448x768 (validated), 16:9 768x448, 1:1 576x576, 3:4 480x640, 4:3 640x480 |
| `mole` | `null` | `{frame, x, y}` set by the `mole` step |
| `chin_clean` | `true` | fill small dark specks on the output chin |
| `destubble` | `false` | chin shadow reduction, clean-shaven subjects only |
| `seeds` | `[43, 44, 45, 46]` | first seed for every piece, the rest only for re-rolls |
| `sync_min_mean`, `sync_min_window` | 0.45, 0.3 | a piece is re-rolled below either |

## Why each step exists

- **Original photo, not an edited one.** Generating the photo with headphones (Qwen-Image-Edit) restyled the hair
  into a bowl fringe. Instead each prop goes in as its own reference picture, which keeps the real hair.
- **Main person only (`masks`).** The face track with the largest area over the clip is the main person. Hair,
  hand and prop tracks count only if they sit near that face (1 face height; hands 2.5) in most frames, so a
  second person in frame does not get erased or merged in. Frames with no face found are counted and printed.
- **Prop cut-out frame.** It is the frame with no hand over the prop, the prop not cut by the frame edge or
  shrunk by occlusion, the face near its median size and the prop centred on it. Picking the largest prop chose a
  side-on close-up with one cup cut off; this rule chose a frontal view like the hand-picked one.
- **Erase the target's hair from `<Video 1>`.** Otherwise H3 copies the target's long side hair below the cups.
  The erase mask is SAM3 hair dilated 51 px, minus face, hands and props (slightly dilated), filled with Telea.
  A clean 5 s test is not enough: 10 s pieces regrew the hair until it was erased.
- **Descriptions (`describe`).** The local Qwen3-VL 4B (Krea 2's text encoder) answers short questions about the
  photo, a frontal target frame and each prop cut-out. It runs in thinking mode: without it, this checkpoint
  reasons out loud inside its answer or returns nothing. Its video input made up the outfit (a jacket and tie for
  a t-shirt), so it only gets single frames. The model is small: review the text before rendering.
- **Optional `mole`.** It removes a dark mole from the target's face before H3 can copy it. The tool LK-tracks it
  from a seed point and snaps to the dark spot. This only partly helps; the chin clean in `finish` does most of
  the work.
- **10 s pieces with 2 s overlaps** (`PIECE=243`, `STRIDE=195`). 5 s pieces joined by short crossfades show the
  face jumping in size and colour at every join, and 15-frame overlaps were still visible. A single long pass
  runs out of memory on the 5090.
- **One seed, re-roll what drifts (`render`).** An H3 piece can silently improvise its motion for several seconds.
  In the test video, 1 of 4 pieces drifted with seed 43 (16-26 s) and seed 44 fixed it. The sync score is a
  windowed (3 s) correlation of motion energy against the target: about 0.8 is normal, below 0.3 means that
  stretch drifted. Only drifted pieces get the next seed, so the test video takes 5 renders instead of 8. The
  thresholds come from that one video, so treat them as provisional.
- **Chin clean on the OUTPUT.** H3 copies the target's mole. Small dark specks in the bottom 22% of a SAM3 face
  mask (mouth excluded) are filled with Telea. Tracking the mole in the source drifted over 34 s and hit the
  shirt, so the cleanup runs on masks from the output itself.
- **Stitch.** It colour-matches pieces with a LAB mean/std transform chained over the overlaps, then crossfades
  for 24 frames, capped at the overlap length (otherwise it blends in the wrong frames). It streams: the cleaned
  pieces are stored losslessly and read frame by frame, so memory does not grow with the video length.
- **Stubble reduction (`destubble`, off by default).** H3 adds a chin shadow by itself; cleaning it out of the
  reference did nothing. The fix ADDS a smooth L/a/b offset that pulls the chin towards a cheek tone averaged
  over ±7 frames, plus a 5 px luminance closing for dots. The zone is the face mask averaged over ±3 frames and
  ramped in from 66% to 80% of the face height, with the mouth excluded. An earlier version replaced chin pixels
  with a bilateral-blurred copy and used a per-frame reference colour; Shawn rejected it as blurry and flickery.
  Never do either; check flicker by looking at consecutive frames, because an average metric missed it. It also
  lightens a real beard, hence off unless the subject is clean-shaven.
- **Provenance.** `prep` refuses to run without a consent statement and writes `provenance.json` (input
  sha256s, consent, time). `finish` copies it into `final.json` with the output hashes, prompt, picks and sync.

## Cloud

`cloud.py` and `pod/pod.sh` create a one-shot RunPod pod: RTX 5090, EU-RO-1, on the farm network volume.
- **Isolation:** the pod is named `fsh3-*`, which the farm's cloud lane ignores.
- **What it runs:** it unpacks the farm's ComfyUI bundle and copies in only the H3 ref2va, text-encoder, VAE and
  turbo-LoRA weights from the volume, so nothing is installed or downloaded. Then it runs the pending graphs and
  writes to `fsh3/jobs/<job>/out/`.
- **Shutdown:** the pod removes itself, with a 2 h cap. `wait_job` deletes it if it is still listed afterwards,
  because self-removal failed once.

Costs and limits:
- **Time:** about 80 s per 10 s piece, plus a few minutes of boot per pod. A 34 s video is 4 renders plus one
  pod per re-roll round; around $0.15-0.30 at ~$1.19/h.
- **4080:** 10 s pieces at 448x768 were never run on the PC (5 s used ~31 of 31 GB host RAM), so render on the cloud.
- **Not yet rendered:** the landscape/square canvases (sizes follow H3's divisible-by-32 rule but are untested).
- **5090 availability:** RTX 5090s are often unavailable in EU-RO-1; `create_pod` retries every 30 s for up to 30 min.
- **Disk:** a 250 GB container disk gets "no instances"; use 200 GB.

## What did not work (don't retry)

- **BFS head-swap LoRAs** (LTX-2.5, H3, LTX-2.3): the target's hair leaked through and the target's jaw stayed.
- **H3 masked inpaint to remove hair:** it only nudges masked pixels towards their surroundings, so the hair regrows.
- **LTX clean-plate IC-LoRA:** it removes whole people, not hair.
- **Feeding H3 a motion reference smaller than 448x768:** the side hair came back.
- **Qwen3-VL 4B video input for the scene text:** it invented the outfit.
