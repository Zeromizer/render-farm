# Brief: prepare the render PC for the RunPod cloud GPU lane

For the Claude session on the home render PC (`C:\Coding\render-farm`). Written 2026-10-15 from
the laptop. Branch `feat/cloud-gpu-lane`.

## What is happening

Shawn is adding a rented **RTX PRO 6000 Blackwell (96 GB)** on RunPod as a second, on-demand GPU
lane. The PC keeps working exactly as today; nothing on the PC is switched off.

- **Moves to the cloud (later, once proven):** the heavy ComfyUI engines `video_gen`,
  `lettering_fix`, `clip_edit`, `audio_gen`, plus `matte` and Blender.
- **Stays on the PC for good:**
  - the relay and `rp-agent-shawn` (the Claude/Codex logins stay home)
  - the Krea fallback ComfyUI at 127.0.0.1:8188
  - `url_fetch` (it needs the home IP)
  - `voiceover`
  - workspace renders that clone `C:\rp-workspace\...`
- **How the cloud gets jobs:**
  - **Routing:** the existing `farm_engine_capabilities` gating, with a new capability `gpu-cloud`.
  - **Starting a pod:** a small controller thread in *this* worker creates a RunPod pod when
    cloud jobs are queued.
  - **Stopping a pod:** the pod terminates itself after 15 idle minutes.
  - **Fallback:** if no pod comes up, the controller un-gates and the PC takes the jobs as today.
- **RunPod side (already created):**
  - Network volume `render-farm`, id `dk8mrhu2wh`, 300 GB, data centre **EU-RO-1**. It is
    mounted at `/workspace` on every pod; it is the persistent disk.
  - GPU type: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (Workstation Edition as second choice).

Nothing below changes the running worker. Do not restart anything for steps 1 to 3.

## 1. Report the live ComfyUI and Blender setup

Reply over intercom with:

- **ComfyUI:**
  - `git -C C:\ComfyUI describe --tags` and `git -C C:\ComfyUI rev-parse HEAD`
  - Memory notes disagree on whether production is v0.36.0 or v0.37.0. Which is actually live?
  - The exact launch line in `C:\ComfyUI\run-headless.bat`.
  - `pip freeze` of the ComfyUI venv. Commit it as `cloud/pc-comfyui-freeze.txt`.
  - Each folder in `C:\ComfyUI\custom_nodes` with its git commit. Is anything beyond
    ComfyUI-vlo actually used by the farm graphs?
- **Blender:** version and path of `BLENDER_EXE` (config default says 4.5.9).
- **Disk:** free space on C:.

## 2. Generate `cloud/models-manifest.json`

List **every model file the farm's graphs load**. Sources to scan:
- `worker/videogen/graphs*.py`
- `worker/audiogen/yue2_inst.api.json`
- the Krea 2 graph used by the relay (`C:\rp-relay\src\krea.js` and its graph)
- the SeedVR2, SAM 3.1 and LTX 2.5 plate graphs
- any LoRA named via env (`RP_KREA_LORA`)

One entry per file:

```json
{
  "path": "models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
  "bytes": 0,
  "sha256": "...",
  "source": "hf",
  "url": "https://huggingface.co/<repo>/resolve/main/<file>"
}
```

- **`path`:** relative to the ComfyUI root, using the same subfolder the PC uses. The cloud
  mirrors the layout under `/workspace/ComfyUI`.
- **`source`:** `"hf"` (or another public direct URL) when the file can be downloaded straight
  from the source. The pod has data-centre bandwidth, so this is much faster than uploading
  from home.
- **`"local-only"`:** use this when there is no public URL. That includes the gated LTX
  clean-plate IC-LoRA, the custom LoRAs (`atto3evo_krea2_v3_1000`,
  `ar_lora_inst_v3abc_comfyui`) and anything you converted or quantised yourself.
  Leave `url` empty.
- **`sha256`** lets the pod verify each download. Hashing the big files takes a while; that is
  fine.

Commit the manifest on `feat/cloud-gpu-lane` and push. Also report the total size of all files
and the total of the `local-only` files.

## 3. Upload the `local-only` files to the network volume

EU-RO-1 volumes are S3-compatible, so this needs **no running pod**.

1. Wait for Shawn to create a RunPod **S3 API key** (Console > Credentials > S3 API keys) and
   put it in the PC's `C:\Coding\render-farm\.env` as `RUNPOD_S3_ACCESS_KEY` /
   `RUNPOD_S3_SECRET_KEY`.
2. Do not ask for the key over intercom.
3. Upload with any S3 client (`aws s3 cp` or boto3):
   - Endpoint: `https://s3api-eu-ro-1.runpod.io`
   - Region: `EU-RO-1`
   - Bucket: `dk8mrhu2wh`
   - Key: `ComfyUI/<manifest path>`
4. Run it overnight; home upload bandwidth is the limit.
5. When done, report each file's key and size.

If the S3 endpoint misbehaves, tell the laptop. The fallback is `runpodctl send` from the PC
plus `runpodctl receive` on a short-lived pod.

## 4. Later, after the laptop lands the code (not yet)

You will get a separate message when this is ready. For reference, the steps will be:

1. `git pull` `feat/cloud-gpu-lane` (or main once merged) into `C:\Coding\render-farm`.
2. Add `CLOUD_LANE=1`, `RUNPOD_API_KEY`, `RUNPOD_VOLUME_ID=dk8mrhu2wh`,
   `RUNPOD_DATACENTER=EU-RO-1` and `CLOUD_MAX_HOURS_PER_DAY` to `.env`. Shawn pastes the keys
   himself.
3. Restart with the `worker\restart-requested` flag file. Never kill `pythonw.exe`.

## Leave unchanged

The relay, `rp-agent-shawn`, the Krea fallback, `url_fetch`, `voiceover`, the TTS studio,
workspace renders, and the PC's ComfyUI version.
