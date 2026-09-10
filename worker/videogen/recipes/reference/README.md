# Reference workflows (vendored, UI format)

Copied verbatim from https://github.com/einhorn13/mmh3_media (MIT), `example_workflows/`,
commit `bca81b8cc96efa2c6ac93f3bcad29c812ba89586` (2026-09-09). The worker does not load
these files: they are the source of truth the API-format builders in
`worker/videogen/graphs_h3.py` were transcribed from, kept here so a node-name mismatch
on the PC can be resolved against the exact graph the reference author ships.

| file | worker builder |
|---|---|
| `mmh3_f01_fl2va.json` | `graphs_h3.build_generation_with_packet` (packet capture at generation time) |
| `mmh3_f07_latent_upscale.json` | `graphs_h3.build_latent_upscale(variant="full")` |
| `mmh3_f07_native_tile_upscale.json` | `graphs_h3.build_latent_upscale(variant="tile")` |

External node: `MinimaxH3LatentUpscaler3D` from
https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (commit
`d7c01b9011f2e8439493f6c02c29995a27df276f`, 2026-08-28; no licence file in that repo),
weights `minimax_h3_latent_upscaler_3d_bf16.safetensors` from
https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler (Apache-2.0) in
`ComfyUI/models/latent_upscale_models/`.

To refresh: re-download the three files and update the commit hash above; then diff the
node types / widget values against `graphs_h3.py` (the tests in
`worker/tests/test_h3_latent_graphs.py` list the expected class sets).
