-- Render farm queue schema (Aimotion Supabase project: otznmoiakqhtoeannldu)
-- Apply by pasting this whole file into the Supabase dashboard SQL editor.
--
-- Table is named farm_render_jobs (an unrelated legacy 'render_jobs' table
-- from an old videogen pipeline already exists in this project, with an
-- incompatible status check constraint).

create table if not exists farm_render_jobs (
  id            uuid primary key default gen_random_uuid(),
  status        text not null default 'pending'
      check (status in ('pending','processing','done','failed','canceled')),
  engine        text not null,                     -- remotion | blender
  repo_url      text not null,
  git_ref       text not null default 'main',      -- branch / tag / commit sha
  params        jsonb not null default '{}',
  output_ext    text,
  output_path   text,                              -- outputs/<id>.<ext> in 'renders' bucket
  signed_url    text,
  signed_url_expires_at timestamptz,
  progress      int not null default 0,            -- 0..100
  phase         text,                              -- cloning | installing | rendering | uploading
  error         text,
  attempts      int not null default 0,
  max_attempts  int not null default 2,
  timeout_minutes int not null default 120,
  cancel_requested boolean not null default false,
  created_at    timestamptz default now(),
  claimed_at    timestamptz,
  heartbeat_at  timestamptz,
  completed_at  timestamptz
);

create index if not exists farm_render_jobs_status_idx on farm_render_jobs (status, created_at);

-- Atomic claim: one worker owns the job; SKIP LOCKED makes concurrent workers safe.
create or replace function claim_farm_job()
returns setof farm_render_jobs language sql as $$
  update farm_render_jobs
  set status = 'processing', claimed_at = now(), heartbeat_at = now(),
      attempts = attempts + 1, progress = 0, phase = 'cloning', error = null
  where id = (
    select id from farm_render_jobs
    where status = 'pending' and cancel_requested = false
    order by created_at
    for update skip locked
    limit 1)
  returning *;
$$;

-- Requeue processing jobs whose worker died (stale heartbeat); fail after max_attempts.
create or replace function reclaim_stale_farm_jobs(p_stale_minutes int default 5)
returns int language sql as $$
  with stale as (
    select id, attempts, max_attempts from farm_render_jobs
    where status = 'processing'
      and heartbeat_at < now() - make_interval(mins => p_stale_minutes)
    for update skip locked
  ), upd as (
    update farm_render_jobs j
    set status = case when s.attempts >= s.max_attempts then 'failed' else 'pending' end,
        error  = case when s.attempts >= s.max_attempts
                      then 'worker died / heartbeat stale after ' || s.attempts || ' attempts'
                      else j.error end
    from stale s where j.id = s.id
    returning 1
  ) select count(*)::int from upd;
$$;

-- Cleanup of the earlier mistaken migration on the legacy table (safe to run;
-- drops only what that migration added, keeps videogen's own columns/data):
drop function if exists claim_render_job();
drop function if exists reclaim_stale_jobs(int);
alter table render_jobs
  drop column if exists engine,
  drop column if exists repo_url,
  drop column if exists git_ref,
  drop column if exists params,
  drop column if exists output_ext,
  drop column if exists output_path,
  drop column if exists signed_url,
  drop column if exists signed_url_expires_at,
  drop column if exists progress,
  drop column if exists phase,
  drop column if exists error,
  drop column if exists attempts,
  drop column if exists max_attempts,
  drop column if exists timeout_minutes,
  drop column if exists cancel_requested,
  drop column if exists heartbeat_at,
  drop column if exists completed_at;
