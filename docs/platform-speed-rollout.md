# Platform speed rollout

## Changes

- HyperFrames still jobs accept `params.snapshot_times: [0,1,2,...]` (2–24 distinct ascending timestamps). One CLI invocation and browser session captures every frame with `--no-end`. Image resolution is unchanged.
- The worker uploads `outputs/<farm-id>-slide-0.png`, `-slide-1.png`, etc., then `outputs/<farm-id>.json`. The row becomes done only after every upload succeeds. The JSON contains timestamps and bucket paths, never local paths.
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
