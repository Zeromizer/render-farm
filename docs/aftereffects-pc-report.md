# PC report: direct Astra -> After Effects integration (2026-09-11)

Return report for `docs/after-effects-pc-handoff.md`. Everything below ran on
the render PC (Windows 11, RTX 4080 SUPER, 31 GB RAM, user `shawn_fku5qux`).
**Nothing was activated**: the shared Supabase queue, the live worker on
`main`, both relays and production are exactly as they were.

## 1. Files changed, commits, commands

Branch `feat/aftereffects-engine` of `Zeromizer/render-farm`, developed in the
git worktree `C:\Coding\render-farm-ae` so the live checkout
`C:\Coding\render-farm` (main, running worker) was never touched.

| commit | content |
| --- | --- |
| `01321f6` | the engine: `worker/aftereffects/` (schema, fonts, ae_host, fake_host, media, pipeline, preflight, smoke, recipes/), `worker/runners/aftereffects.py`, `worker/tests/test_aftereffects.py`, capability-gated claiming (`docs/aftereffects-claiming.sql`, `schema.sql`), worker wiring (`render_worker.py`, `db.py`, `config.py`, `insert_test_job.py`), MCP enum (`mcp/index.js`), README, `.env.example`, `docs/aftereffects-platform-contract.md` |
| `0a534a8` | clock-driven cancel/timeout (`worker/aftereffects/clocked.py`), smoke scenarios |
| (this file) | `docs/aftereffects-pc-report.md` |

Commands (from `C:\Coding\render-farm-ae\worker`, using the live venv):

```
..\..\render-farm\.venv\Scripts\python.exe -m aftereffects.preflight
..\..\render-farm\.venv\Scripts\python.exe -m aftereffects.smoke --workspace C:\Coding\ae-smoke --revise
..\..\render-farm\.venv\Scripts\python.exe -m aftereffects.smoke --workspace C:\Coding\ae-smoke --asset logo=C:\Coding\ae-smoke\fixtures\logo.png
..\..\render-farm\.venv\Scripts\python.exe -m aftereffects.smoke --workspace C:\Coding\ae-smoke --cancel-after-s 12
..\..\render-farm\.venv\Scripts\python.exe -m aftereffects.smoke --workspace C:\Coding\ae-smoke --timeout-minutes 0.2
..\..\render-farm\.venv\Scripts\python.exe -m aftereffects.smoke --workspace C:\Coding\ae-smoke --fake --revise   (no AE needed)
set SUPABASE_URL=x& set SUPABASE_SERVICE_KEY=x& ..\..\render-farm\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

Queue path once activated: `insert_test_job.py --engine aftereffects --params @request.json`.

## 2. Runtime preflight

| item | result |
| --- | --- |
| After Effects | **26.5.0.89 (After Effects 2026)**, `C:\Program Files\Adobe\Adobe After Effects 2026\Support Files\AfterFX.exe` and `aerender.exe`. Installed through Creative Cloud at 08:56 today (absent at the start of this session); the operator's first launch + sign-in instance (pid 39512, "Untitled Project") stayed open throughout and was never touched. |
| AE opens for this user | yes: `AfterFX.exe -m -noui -r probe.jsx` in a fresh instance ran and quit in 6 s; `app.fonts.getFontsByPostScriptName("ArialMT")` resolves; output-module templates present: `Lossless with Alpha`, `High Quality with Alpha`, `TIFF Sequence with Alpha`, `Alpha Only`, H.264 presets; render templates: `Best Settings`, `Draft Settings`, … |
| Script file access | works. The recipe sets `Pref_SCRIPTING_FILE_NETWORK_SECURITY` in the legacy pref section (the `v2` section is refused by AE 26.5). **`app.preferences.saveToDisk()` + `reload()` switches the permission off again and every write becomes a 0-byte file**; found the hard way, removed, documented in `lib.jsx`. |
| FFmpeg / ffprobe | 9.0.1 (winget Gyan build), `prores_ks`, `libvpx-vp9`, `qtrle`, `png` encoders present |
| RAM / disk | 31.1 GB total, ~7.6 GB free at the time (the operator's AE + CC + TTS workers running); 132 GB free on C: |
| Fonts | `ArialMT`, `Arial-BoldMT`, Segoe UI family, 188 PostScript names read from the Windows font tables |
| GPU | not used by this engine (AE 2026 CPU multi-frame rendering; no CUDA effects requested) |

`python -m aftereffects.preflight` prints READY.

## 3. Smoke test (real After Effects)

Recipe `text_overlay_v1`: 5 s, 1080x1920, 30 fps, transparent; text layers
`headline` ("Motorised Blinds / From $299", Arial-BoldMT 110, slide-in +
fade, built-in Drop Shadow) and `cta` ("Book a free measure", ArialMT 56,
fade), shape `pill` (rounded rect, scale-up + fades). Artifacts copied to
`C:\Coding\ae-smoke\deliverables\`:

| artifact | v1 (`deliverables\v1`) | v2 revision (`deliverables\v2`) |
| --- | --- | --- |
| AEP bundle (aep + settings.json + recipes/) | `smoke-bundle.zip` 18.6 KB | `smoke-v2-bundle.zip` 9.8 KB |
| alpha master ProRes 4444 | `smoke-master.mov` 55.4 MB, prores 4444, yuva444p12le (decoded), 1080x1920, 30 fps, 150 frames, 5.000 s | `smoke-v2-master.mov` 49.4 MB, same checks |
| review VP9 alpha | `smoke-review.webm` 469 KB, vp9 yuva420p (decoded via libvpx) | `smoke-v2-review.webm` 444 KB |
| contact sheet | `smoke-contact.png` (6 frames, light + dark rows) | `smoke-v2-contact.png` |
| manifest | `manifest.json` (checks, timings, provenance, editable layers) | `manifest.json` |
| log | `C:\Coding\ae-smoke\last-smoke.log` | same file |

Verification recorded in the manifests: frame count 150 = 5 s x 30; size and
fps match the request; alpha: every frame has transparent pixels, 125 frames
have opaque pixels, 8 frames fully transparent (before the first fade-in),
none fully opaque; raw alpha of frame 75 spans 0..255 in the master and in
the review copy. The AVI AE writes is **premultiplied**; the master is
converted to **straight** alpha (text-edge pixels measured 253/255 at alpha
183 after conversion vs 182 before). The contact sheet was inspected: white
text with shadow and the orange pill read correctly over both backgrounds.
The saved `.aep` was reopened by a second AE instance and its layers listed:
`cta` text / `headline` text / `pill` shape with position and opacity
keyframes, in/out points 0.8–5.0, 0.2–5.0, 0.6–5.0 s.

Revision: `revise_v1.jsx` opened the delivered `.aep`, changed `headline` to
"Motorised Blinds / Now $249" and its timing to 0.5–4.6 s, saved
`project-revised.aep`, re-rendered; the reopened project shows the new text
and timing, 150 frames, alpha verified again.

Measured timings (real AE, both runs within 1 s of each other):

| step | v1 | v2 |
| --- | --- | --- |
| AfterFX authoring (launch + script + save + quit) | 5.0 s | 5.0 s (revise) |
| aerender (launch ~4 s + 150 frames) | 9.5 s | 10.5 s |
| ProRes 4444 master (incl. unpremultiply) | 2.3 s | 2.3 s |
| VP9 review | 3.2 s | 2.9 s |
| verify (probe, frame count, alpha stats, raw frame) | 2.5 s | 2.5 s |
| contact sheet + review decode check | 0.5 s | 0.5 s |
| reopen / inspect (second AfterFX instance) | 5.0 s | 5.0 s |
| **total** | **28.5 s** | **29.2 s** |

A third run with an imported PNG asset (`--asset logo=…`) rendered a footage
layer in 28.6 s. Intermediate AVIs (1.24 GB each) are deleted once the master
exists.

## 4. Test results

Real After Effects (smoke tool, `C:\Coding\ae-smoke\scenario-*.log`):

| case | result |
| --- | --- |
| JSX failure | reported as `JSX_ERROR: author script failed (text_overlay_v1.jsx:115): After Effects error: Unable to call 'addComp' because of parameter 1. Value is undefined.` (a real bug found and fixed this way; line numbers come from AE) |
| foreign instance / isolation | a script run whose `AE_JOB_TOKEN` does not match refuses with `AE_NOT_ISOLATED`, writes its manifest, does **not** quit the instance |
| cancellation during aerender (`--cancel-after-s 12`) | `CANCELED` after 14.8 s; both recorded pids (AfterFX 22848, aerender 32776) killed; operator's AE untouched |
| cancellation during authoring (`--cancel-after-s 2`) | `CANCELED` at 5.1 s (caught at the phase boundary after the 5 s author step) |
| timeout during aerender (`--timeout-minutes 0.2`) | `TIMEOUT` after 14.7 s, processes killed |
| image asset | imported and rendered (footage layer, no missing-footage flag) |
| missing font | `FONT_MISSING` from the Windows-font-table preflight before AE launches (fake host run; AE-side check present but not exercised with a real missing font) |

Unit tests (`tests/test_aftereffects.py`, fake host + real ffmpeg, 21 tests,
whole suite 50 tests OK): schema bounds and rejects; font check; full
pipeline (sequence and QuickTime paths) with bundle, manifest, alpha
statistics, review decode; missing input / outside-workspace input / hash
mismatch / good asset; JSX failure with line; AE exits without manifest;
cancel mid-render; timeout mid-author; incomplete render (fewer frames)
rejected; recipe revision pin; revision of one text + timing; slot mutex
timeout; runner: sidecars uploaded before the master, superseded claim
publishes nothing, cancel at publish, retry uses `attempt-2` and removes the
stale `attempt-1`, invalid request fails before any upload.

**Untested / not exercised** (labelled as the handoff asks):

- A real Supabase round trip (insert -> claim -> upload -> `done`): the row
  would be claimed by the current unpatched worker and failed with
  `unknown engine`, so it was not inserted. Runner-level publish/retry logic
  is covered only with stubbed `db`.
- Retry publication with two real attempts against the live table.
- `AE_SLOT_TIMEOUT` with two real AE jobs (mutex tested in-process only).
- Missing font, missing effect and missing output template *inside* AE
  (the code paths exist; only the Python-side font preflight ran for real).
- Cancellation during the ffmpeg steps against real AE output (clock-driven
  runner is unit-tested with the fake host).
- The demo Blinds ad integrated overlay test, final-composition alpha check,
  timing/audio, token usage: blocked on activation (below) and the laptop's
  platform changes.
- Fonts other than Arial/Segoe; effects other than the built-in Drop Shadow.

## 5. Contract and activation

Contract: `docs/aftereffects-platform-contract.md` (request schema v1 with
bounds, result objects + MIME types, manifest, phases, error codes,
capability advertisement, migration, host facts, timings).

Activation prerequisites, in order, none done:

1. Apply `docs/aftereffects-claiming.sql` in the Aimotion Supabase SQL
   editor (adds `farm_engine_capabilities`, replaces `claim_farm_job()` with
   `claim_farm_job(p_capabilities text[] default '{}')`). Safe for the
   running worker: it calls with no argument and simply never claims gated
   engines; the preview lane's `claim_farm_preview_job` is untouched.
2. Merge `feat/aftereffects-engine` into `main`, `git pull` in
   `C:\Coding\render-farm`, restart the worker when idle (kill the real
   `python.exe` child of the supervisor, not the pythonw stub). Log line to
   expect: `render worker starting (... capabilities=['aftereffects'])`.
3. Laptop: platform queue serialization + overlay tools per the contract;
   then the demo Blinds overlay through the demo environment.

Do not insert `aftereffects` rows before steps 1–2.

## Notes for whoever picks this up

- `AfterFX.exe -m -noui -r job.jsx` gives a fresh instance; the launch env
  carries `AE_JOB_TOKEN`, the recipe checks `$.getenv` and only ever
  `app.quit()`s its own instance. AfterFX has no useful exit code; the
  manifest file is the completion signal.
- Setting `layer.inPoint` on a text/shape layer shifts it (out point moves
  too); set inPoint then outPoint and verify (the recipe does).
- aerender's `-outputSettings "Color: …"` is read-only in 26.5, hence the
  ffmpeg-side unpremultiply (`blend=all_mode=divide`; ffmpeg's own
  `unpremultiply` filter left the pixels unchanged on this input).
- The operator's AE instance (started 08:57) is still open; it was left alone.
