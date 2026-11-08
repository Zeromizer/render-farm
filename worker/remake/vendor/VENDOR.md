# Vendored: Viggle Remake (code only)

Source: https://huggingface.co/Viggle/Viggle-Remake at revision `7e5be143702dd53b9154482de673b8519ed2d4d6`
(the revision reproduced on a 5090 in the 2026-11-08 study; see memory `viggle-remake-evaluated`).

Copied **unmodified**:

| here | upstream |
|---|---|
| `viggle_remake_mcp/` | `mcp/viggle_remake_mcp/` (the MCP package: kit, paint, prompts, jobs, server, data/) |
| `comfy_worker/remake_worker.py`, `setup.sh` | `mcp/comfy_worker/` (the 32 GB-card ComfyUI render worker) |
| `comfyui/ComfyUI-Viggle-Remake/` | `comfyui/ComfyUI-Viggle-Remake/` (the node pack) |
| `pyproject.toml` | `mcp/pyproject.toml` (their pinned dependencies) |

Licence: the code is Apache-2.0 (`LICENSE`). The weights are NOT here; they are a Model Derivative of MiniMax H3
under the MiniMax H3 Community License (`NOTICE`). Products built on them must show "MiniMax H3" in the UI and
should credit "Powered by Viggle Remake (https://huggingface.co/Viggle/Viggle-Remake)".

Our own code that wraps these lives one level up in `worker/remake/` and never edits files in this folder; a fix is
a wrapper or a monkeypatch there, so an upgrade is a re-copy plus a diff of the wrappers.
