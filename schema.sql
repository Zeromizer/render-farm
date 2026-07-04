-- Render farm queue schema (Aimotion Supabase project: otznmoiakqhtoeannldu)
-- Apply by pasting this whole file into the Supabase dashboard SQL editor.

create table if not exists render_jobs (
  id            uuid primary key default gen_random_uuid(),
  status        text not null default 'pending',
      -- pending | processing | done | failed | canceled
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

create index if not exists render_jobs_status_idx on render_jobs (status, created_at);

-- Atomic claim: one worker owns the job; SKIP LOCKED makes concurrent workers safe.
create or replace function claim_render_job()
returns setof render_jobs language sql as $$
  update render_jobs
  set status = 'processing', claimed_at = now(), heartbeat_at = now(),
      attempts = attempts + 1, progress = 0, phase = 'cloning', error = null
  where id = (
    select id from render_jobs
    where status = 'pending' and cancel_requested = false
    order by created_at
    for update skip locked
    limit 1)
  returning *;
$$;

-- Requeue processing jobs whose worker died (stale heartbeat); fail after max_attempts.
create or replace function reclaim_stale_jobs(p_stale_minutes int default 5)
returns int language sql as $$
  with stale as (
    select id, attempts, max_attempts from render_jobs
    where status = 'processing'
      and heartbeat_at < now() - make_interval(mins => p_stale_minutes)
    for update skip locked
  ), upd as (
    update render_jobs j
    set status = case when s.attempts >= s.max_attempts then 'failed' else 'pending' end,
        error  = case when s.attempts >= s.max_attempts
                      then 'worker died / heartbeat stale after ' || s.attempts || ' attempts'
                      else j.error end
    from stale s where j.id = s.id
    returning 1
  ) select count(*)::int from upd;
$$;
