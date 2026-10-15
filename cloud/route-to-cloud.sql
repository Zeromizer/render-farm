-- Cloud GPU lane routing (docs/cloud-gpu-brief.md). Paste into the Supabase SQL editor.

-- 1. Once, any time (harmless to the running PC): which worker claimed a job.
--    The PC's cloud lane reads it to tell a working pod from a broken one.
alter table farm_render_jobs add column if not exists claimed_by text;

-- 2. Only after the cloud pod has passed its smoke jobs: send the heavy GPU
--    engines to the cloud. The PC stops claiming them within a minute (it
--    re-reads the gating) and takes one back only after CLOUD_FALLBACK_MINUTES.
insert into farm_engine_capabilities (engine, capability) values
  ('video_gen', 'gpu-cloud'), ('lettering_fix', 'gpu-cloud'), ('clip_edit', 'gpu-cloud')
-- matte stays on the PC too: TensorRT there does a job in ~16 s, while
-- onnxruntime-gpu on the PRO 6000 ran 3.7 s/frame (376 s) at go-live.
-- audio_gen (YuE2) stays on the PC: it decodes step by step, so the PRO 6000
-- is only ~15% faster (30 s bed: ~118 s vs ~139 s), it fits the 4080, and with
-- H3 gone the PC's ComfyUI is free for it (Shawn, 2026-10-15).
on conflict (engine) do update set capability = excluded.capability;

-- Back to the PC (all of them; add "and engine = '...'" for one):
-- delete from farm_engine_capabilities where capability = 'gpu-cloud';
