# H3 latent upscale: PC handoff (GPU side)

Written 2026-09-10 on the laptop, branch `feat/h3-latent-upscale` of `Zeromizer/render-farm`.
Everything that does not need the GPU is done and unit-tested (68 tests, `cd worker && python -m
unittest discover -s tests -t .`). What follows needs the RTX 4080 SUPER box: install the node pack
and the weights, confirm five option strings, measure, calibrate, run the acceptance list.

Background: brief `C:\Users\shawn\Downloads\brief-h3-latent-upscale.md` (laptop). What changed and why:
`worker/runners/video_gen.py` docstring (`save_latent`, `upscale.method h3_latent_upscale`),
`worker/videogen/graphs_h3.py` (the graphs, with the reference-node mapping in its docstring),
`docs/video_gen-platform-brief.md` section "Latent upscale (H3 native)".

## 0. Findings you should know before touching anything

- The worker was decode-only: no H3 clip generated before this branch has a latent. Clips
  from before can only go through `variant: "decoded"` (mp4 -> VAE encode -> latent upscale ->
  refine; lower fidelity, experimental). New generations save a packet when `save_latent` is on.
- `refine_strength` from the brief does not exist upstream; the knob is `denoise` (0 = source-aware,
  0.375 for our turbo-8 packets) plus `steps_override`.
- F05 Upscale + Stitch is out of scope: turntable quarters are independent i2v clips, not an mmh3
  continuation chain. Quarters get per-segment packets (`outputs/<id>-<segment>-latent.mmh3`) and
  are upscaled as standalone jobs.
- The worker keeps its own sampler chain (`res_multistep`/`simple`/8 steps/shift 12,3) and writes
  the packet's `sampling_profile_json` / `applied_loras_json` itself. It does NOT use
  `MMH3H3SamplingPreset` (euler, shift 6/3, LoRA paths we do not have).
- The `.mmh3` schema is v2 only. The upscaler node repo has no licence file; its weights are Apache-2.0.

## 1. Install (once)

```
cd C:\ComfyUI\custom_nodes
git clone https://github.com/einhorn13/mmh3_media ComfyUI_mmh3_media
C:\ComfyUI\.venv\Scripts\pip install -r ComfyUI_mmh3_media\requirements.txt
git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
mkdir C:\ComfyUI\models\latent_upscale_models
:: download minimax_h3_latent_upscaler_3d_bf16.safetensors (~691 MB) from
:: https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler into that folder
```
Restart ComfyUI (`C:\ComfyUI\run-headless.bat`, or let the worker relaunch it). Note in
`comfyui-headless.log` whether `ComfyUI_mmh3_media` imported cleanly: it carries backports of
ComfyUI PRs #15860/#15975 and may want a newer ComfyUI than the current checkout. Record the ComfyUI
version (`/system_stats`) and both node-repo commits (the worker writes them into `-upscale.json`).

Laptop-side reference: the vendored example JSONs and upstream commits are in
`worker/videogen/recipes/reference/README.md` (mmh3_media `bca81b8c`, upscaler `d7c01b90`).

## 2. Worker checkout

```
cd c:\Coding\render-farm     (the PC checkout; confirm the path)
git fetch && git checkout feat/h3-latent-upscale
```
Let any running job finish, then restart the worker (kill the real `python.exe` child, never the
`pythonw` supervisor stub; it relaunches). The default generation graph is unchanged, so existing
jobs behave exactly as before.

## 3. Preflight and the five unknown option strings

```
cd worker\videogen
..\..\.venv\Scripts\python smoke.py --preflight-only --object-info-dump ..\tests\fixtures\object_info_pc.json
```
This fetches `/object_info`, builds the four graphs (generation+packet, tile, full, decoded) and
checks every node class and every literal combo value. Fix whatever it names in
`worker/videogen/graphs_h3.py` (one constant each), re-run until all four say OK, and commit the
trimmed fixture. Things I could not confirm from the sources and expect to adjust:

| constant in graphs_h3.py | what to confirm |
|---|---|
| `GEOMETRY_MODE_DIMS = "target_dimensions"` | option list of `MMH3H3LatentUpscalePrepare.geometry_mode` (`UPSCALE_GEOMETRY_MODES`) |
| `UPSCALER_MODE_DIMS = "target dimensions"` + dotted `mode.width` / `mode.height` | whether `MinimaxH3LatentUpscaler3D` is a DynamicCombo (dotted keys) or flat `width`/`height` inputs; `/prompt` node_errors will say |
| `PUT_ROLE = "auxiliary"` with `primary: true` | that `MMH3H3DecodedUpscalePrepare` then finds the primary video (`get_primary("video")`) |
| `LOAD_FILE_NONE = "(none)"` | that headless `/prompt` accepts it with `path_override`; if not, upload the packet to `input/video_gen/` and use `input::video_gen/<name>.mmh3` |
| `PROBE_CLASS = "PreviewAny"` | present in this ComfyUI (otherwise the probes are dropped and refine settings come from the saved packet only) |

Also compare the tile node's inputs against `MMH3H3NativeTileRefine` in
`nodes_upscale.py`; the preflight reports unknown/missing inputs.

## 4. Parity A/B of the packet-capable generation graph

Same prompt and seed twice, `--save-latent` vs without:
```
..\..\.venv\Scripts\python smoke.py --mode i2v --first-frame <still.png> --ratio 9:16 --duration 5 --seed 42 --out C:\tmp\a.mp4
..\..\.venv\Scripts\python smoke.py --mode i2v --first-frame <still.png> --ratio 9:16 --duration 5 --seed 42 --save-latent --out C:\tmp\b.mp4
```
Expect frame-identical output (`studio/post.py: frame_diff` of frame 0 and the last frame <= 1) and a
`C:\tmp\b.mmh3` whose `packet.json` shows our LoRA name, `profile "turbo (8 steps)"`, `frames 124`,
`480x832`. Record the wall-time delta of the packet save. Only after this passes, flip
`H3_SAVE_LATENT_DEFAULT` to `true` in the platform (`lib/media/video-provider.ts`).

## 5. Upscale runs (record every number)

Per run write down: phase timings from the smoke log (`load`, `sampling`, `decode`, `crop`,
`fidelity`), `VRAM during refine` (min free / peak used estimate / nvidia-smi), the ComfyUI process
peak RAM from Task Manager (the 243-frame SeedVR2 decode once hit 82 GB virtual and died), the
fidelity verdict, and visual notes on badge / grille / wheels / plate / dealer logo in the
`-compare.mp4` (source | upscaled).

```
:: tile (default), the acceptance case: typical clip at the default duration
smoke.py --mode upscale --source C:\tmp\b.mp4 --latent C:\tmp\b.mmh3 --method h3_latent_upscale --variant tile --shorter-size 1080
:: tile on a 768p 16:9 5 s clip and a 480p 9:16 10 s clip (RAM is the question at 10 s)
:: full on 3 s and 5 s of 480p 9:16 (the 5 s one is guarded: add --allow-large-full to measure it)
smoke.py --mode upscale --source ... --latent ... --method h3_latent_upscale --variant full --shorter-size 1080 --allow-large-full
:: decoded on 2-3 EXISTING H3 clips from the library (native 24 fps, not RIFE'd masters):
::   select a.id, a.storage_path, a.width, a.height, a.duration_seconds from rp_assets a
::   where a.source_url like 'minimax:h3:%' order by created_at desc;
smoke.py --mode upscale --source <existing.mp4> --method h3_latent_upscale --variant decoded --shorter-size 1080
:: and the same sources through seedvr2 and lanczos for the halo / ringing comparison
smoke.py --mode upscale --source <existing.mp4> --method seedvr2 --shorter-size 1080
smoke.py --mode upscale --source <existing.mp4> --method lanczos --shorter-size 1080
```
Acceptance from the brief: tile end-to-end at the default duration without OOM (document peak VRAM
and time); full for a short clip (document the length limit); no ringing/halo vs SeedVR2 on the
same source; fidelity JSON present.

## 6. Failure drills (each must fail with the planned message, nothing falls back)

1. tile without `--latent` (or a job without `upscale.latent`): message names `outputs/<id>-latent.mmh3` and `decoded`.
2. Weight renamed: preflight names the weight, the HF URL and `latent_upscale_models`.
3. `ComfyUI_mmh3_media` dir renamed: preflight lists the missing `MMH3*` classes with the repo URL.
4. A 60 fps RIFE'd master as `decoded` source: "needs a 24 fps source".
5. 768p 10 s `full` without `--allow-large_full`: the pixel-frame guard message.
6. Queue path: `insert_test_job.py --engine video_gen --params "{\"source\": {...}, \"upscale\": {\"method\": \"h3_latent_upscale\"}}"` -> row `failed` with the same text.

## 7. Regression

One lanczos, one seedvr2, one single-quarter turntable job through `insert_test_job.py`; all three
paths are untouched by this branch except the added kwargs, and must behave as before.

## 8. Calibrate and close

- `worker/videogen/estimate.py`: `H3UP_*` constants from the measured runs; `graphs_h3.FULL_MAX_PXF`
  from where `full` actually fits; fidelity thresholds (`graphs_h3.FIDELITY_DEFAULTS`) anchored on a
  known-good pair (lanczos vs slightly blurred lanczos) and a known-bad pair (raw SeedVR2).
- `README.md` video_gen section and `docs/video_gen-platform-brief.md`: replace "provisional" numbers.
- Commit the fixture and constants on `feat/h3-latent-upscale`, merge to `main`, and tell the laptop
  session (intercom) so the platform side can be deployed and `H3_SAVE_LATENT_DEFAULT` flipped.
