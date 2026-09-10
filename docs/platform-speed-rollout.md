# Platform speed rollout

## Changes

- HyperFrames still jobs accept `params.snapshot_times: [0,1,2,...]` (2–24 distinct ascending timestamps). One CLI invocation and browser session captures every frame with `--no-end`. Image resolution is unchanged.
- The worker uploads `outputs/<farm-id>-slide-0.png`, `-slide-1.png`, etc., then `outputs/<farm-id>.json`. The row becomes done only after every upload succeeds. The JSON contains timestamps and bucket paths, never local paths:
  `{"version": 1, "job_id": "<farm-id>", "bucket": "renders", "count": 3, "snapshots": [{"index": 0, "at": 0, "bucket": "renders", "path": "outputs/<farm-id>-slide-0.png"}, ...]}`.
  Each PNG is bound to its timestamp by the CLI's own `frame-NN-at-<t>s.png` name and verified, so a missing or reordered frame fails the job instead of shifting slides.
- Aimotion creates one rp_renders row per card, all sharing the farm job ID. Its slide index resolves the image and provenance timestamp. No database schema change is needed for batching.
- Optional preview_worker.py claims only HyperFrames stills, uses its own checkout/assets/work cache and log, a separate singleton mutex, software browser capture and conservative resource admission (at least 8 GiB free RAM and CPU below 70%). It never runs video generation or unrestricted GPU jobs. The main worker remains available as fallback.

## PC-side rollout (required before enabling platform batching)

1. Pull this branch. Let the current main-worker job finish before restarting the worker through its existing supervisor. Do not terminate active renders.
2. Run `python -m unittest discover -s worker/tests -p test_snapshot_batch.py`.
3. Run one existing approved composition through the updated runner with `output_kind: still`, `snapshot_times: [0,1,2]`, and a pinned git commit. Verify three correctly ordered PNGs, the uploaded JSON, cancellation, and a simulated upload failure never marking the batch done. Compare each image with the old single-timestamp output at identical resolution/browser settings.
4. Only after this succeeds, set Vercel production `FARM_STORYBOARD_BATCH_ENABLED=true` and redeploy Aimotion. It is deliberately OFF until the PC upgrade is verified. Old single-frame requests remain supported.
5. Apply `docs/preview-worker.sql` in the same Supabase project. This adds a service-role-only claim RPC using `FOR UPDATE SKIP LOCKED`, so the main and preview workers cannot claim the same job.
6. Before starting the optional preview process, measure a representative MiniMax job alone, then the same class of job with a still batch alongside it. Compare completion time, free RAM, CPU, GPU VRAM, output images and desktop responsiveness. Do not keep the preview lane enabled if it causes resource pressure or materially slows MiniMax. Admission checks are conservative, not a substitute for this host benchmark.
7. Launch the preview lane windowlessly from the farm root: `Start-Process -FilePath .\.venv\Scripts\pythonw.exe -ArgumentList 'worker/preview_worker.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden`. The optional worker uses `worker/preview-worker.log`. It initially waits for two CPU samples. Do not start a second unrestricted render_worker.py. Keep the existing main worker supervisor unchanged.

## Recovery

- Turn `FARM_STORYBOARD_BATCH_ENABLED` off and redeploy to revert submissions to individual stills. Allow already queued batches to finish on an updated worker.
- Stop only the optional preview worker when it is idle to revert to the main worker. A batch interrupted mid-job remains nonterminal and is reclaimed through existing farm rules.
- Keep original render versions and storyboard approvals. Do not regenerate assets to recover a queue or upload failure.

## Platform changes already included

Project progress Details includes task queue/runtime measurements, fetched only while the panel is open. Intervals overlapping within a category are merged; categories can overlap and are not a summed project total. Agent instructions use local snapshots during iteration, group formal storyboard submissions, and overlap independent implementation with footage generation after approval.

## PC validation record (render PC, 2026-09-07, branch at 52c9a16)

Composition: `compositions/r6-storyboard.html` of the approved job-07ebf4fd workspace, pinned commit `11d02da4`, quality `draft`, 1080x1920.

| measurement | result |
|---|---|
| `python -m unittest discover -s worker/tests -t worker` | 48 tests pass (3 branch tests + 16 batch regressions + earlier video_gen tests) |
| three single stills at 0, 1, 2 s (new worker) | 3.5 s each after the first clone (5.9 s incl. clone), 15.3 s wall for the three including claim gaps |
| one batch job `snapshot_times: [0, 1, 2]` | 4.5 s claim to done, one browser session, `outputs/<id>-slide-0..2.png` + `outputs/<id>.json` |
| image equality | every batch slide is byte-identical to the single still at the same time from both the old worker (jobs 7810c8f6, 25da7d5d) and the new one (same size, luma diff 0.000) |
| cancel mid-batch (job e87127fb) | `cancel_requested` set during rendering; row ended `canceled`, no objects under `outputs/<id>*` |
| upload failure / short batch | covered by `tests/test_snapshot_batch_upload.py` with storage faked: the row is never marked done |
| ordinary still, video render (mp4, 15 fps draft: 8 s), MiniMax t2v 480p 3 s (73 s) | all fine on the restarted main worker; TTS workers paused and relaunched around the MiniMax job |

Preview-lane host benchmark (same 480p 3 s MiniMax job, seed 5; storyboard batch through the CLI with `--no-browser-gpu`, `PRODUCER_BROWSER_GPU_MODE=software`, `OMP_NUM_THREADS=2`, as `preview_worker.py` sets it):

| case | MiniMax wall | batch wall | CPU avg/max | free RAM min | VRAM max | GPU util |
|---|---|---|---|---|---|---|
| MiniMax alone | 73 s | - | 37 % / 82 % | 0.43 GiB | 15.5 GB | 79 % |
| MiniMax + software batch started at sampling 2/8 | 70 s | 5.5 s | 29 % / 72 % | 0.99 GiB | 15.5 GB | 89 % |
| software batch alone, idle box | - | 6.6 s | - | - | - | - |
| hardware (main-lane) batch alone | - | 4.5-6.2 s | - | - | - | - |

MiniMax was not slowed by the concurrent batch and the batch finished in normal time. The constraint on this
31 GB box is RAM: `--fast-disk` weight streaming leaves 0.4-1 GiB free during a MiniMax job, so the preview
lane's admission check (8 GiB free, CPU < 70 %) refuses to start while MiniMax runs; it can only add throughput
between GPU jobs. `preview_resources.can_start()` was observed returning False throughout a MiniMax job and True
on the idle box. Desktop responsiveness was not measured from the PC session (no input-latency probe); CPU stayed
under 20 % during the batch window, which is the only proxy available.

Preview worker dry run: `preview_worker.py` started windowlessly, wrote `worker/preview-worker.log`, created its own
cache at `%LOCALAPPDATA%\render-farm-preview\{repos,assets,work}`, and a second copy exited immediately
("another render-farm preview-worker instance is already running"). It was stopped again: the claim RPC is not
applied yet (see below), and a claim of a non-still job is now handed back to the queue rather than failed.

### SQL status

`docs/preview-worker.sql` is **not applied** to project otznmoiakqhtoeannldu as of this record: the render PC has no
SQL credential (no psql, no Supabase CLI, no personal access token; the worker only holds the service key, which
cannot run DDL). PostgREST lists `/rpc/claim_farm_job` and `/rpc/reclaim_stale_farm_jobs` only. Apply it from the
platform side (SQL editor or migration), then verify with the OpenAPI listing (`GET /rest/v1/` with the service
key) that `/rpc/claim_farm_preview_job` appears and that an `anon` call is rejected. The SQL itself is
service-role-only (revoke from public/anon/authenticated, grant to service_role) and claims with `FOR UPDATE SKIP
LOCKED`, so it cannot hand the same row to both lanes provided `claim_farm_job` also takes the row lock or
re-checks `status = 'pending'` in its UPDATE; the PC could not read that function's body to confirm.

### Preview lane operations (only after the SQL is applied and a second host benchmark is clean)

- Start (farm root, windowless): `Start-Process -FilePath .\.venv\Scripts\pythonw.exe -ArgumentList 'worker/preview_worker.py' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden`
- Status: `Get-CimInstance Win32_Process | ? { $_.CommandLine -like '*preview_worker.py*' }` and `worker/preview-worker.log`.
- Stop (when idle): `Get-CimInstance Win32_Process | ? { $_.CommandLine -like '*preview_worker.py*' } | % { Stop-Process -Id $_.ProcessId -Force }`.
  A batch interrupted mid-job stays `processing` until `reclaim_stale_farm_jobs` returns it to pending after the stale window; the main worker then renders it.
- Recovery: it has no supervisor by design; if it dies, start it again. It never runs video_gen or non-still jobs, never touches the main worker's cache, and holds the `render-farm-preview-worker` mutex, so at most one copy runs.
- Tuning: `PREVIEW_MIN_FREE_GB` (floor 8) and `PREVIEW_MAX_CPU_PERCENT` (cap 70) in `.env`; `RENDER_PREVIEW_CACHE_DIR` moves its cache.

Decision 2026-09-07: **preview lane left disabled** (SQL not applied; the RAM floor means it cannot run alongside MiniMax on this host anyway). Main-lane batching is verified and ready for platform activation.
