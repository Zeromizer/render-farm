-- Cloud GPU lane routing (docs/cloud-gpu-brief.md). Paste into the Supabase SQL editor.

-- 1. Once, any time (harmless to the running PC): which worker claimed a job.
--    The PC's cloud lane reads it to tell a working pod from a broken one.
alter table farm_render_jobs add column if not exists claimed_by text;

-- 2. Only after the cloud pod has passed its smoke jobs: send the heavy GPU
--    engines to the cloud. The PC stops claiming them within a minute (it
--    re-reads the gating) and takes one back only after CLOUD_FALLBACK_MINUTES.
insert into farm_engine_capabilities (engine, capability) values
  ('video_gen', 'gpu-cloud'), ('lettering_fix', 'gpu-cloud'), ('clip_edit', 'gpu-cloud'),
  ('audio_gen', 'gpu-cloud'), ('matte', 'gpu-cloud')
on conflict (engine) do update set capability = excluded.capability;

-- Back to the PC (all of them; add "and engine = '...'" for one):
-- delete from farm_engine_capabilities where capability = 'gpu-cloud';
