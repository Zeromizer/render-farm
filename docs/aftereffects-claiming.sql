-- Capability-aware claiming for the render farm (Aimotion Supabase project).
-- Additive; apply in the SQL editor BEFORE deploying worker code that
-- advertises capabilities. NOT applied yet - see docs/aftereffects-platform-contract.md.
--
-- Engines listed in farm_engine_capabilities are only claimed by a worker
-- that passes the matching capability. A worker that calls claim_farm_job()
-- with no argument (every worker deployed before this change) gets the empty
-- default and therefore never claims a gated engine; ordinary engines are
-- unaffected. The preview lane's claim_farm_preview_job (hyperframes stills
-- only) is untouched and still cannot claim anything else.

create table if not exists farm_engine_capabilities (
  engine     text primary key,
  capability text not null
);
insert into farm_engine_capabilities (engine, capability)
  values ('aftereffects', 'aftereffects')
  on conflict (engine) do nothing;

-- The signature changes, so the old zero-argument function has to go first.
-- Between the drop and the create a polling worker sees one failed claim,
-- which it retries quietly (render_worker.py "claim error").
drop function if exists claim_farm_job();

create or replace function claim_farm_job(p_capabilities text[] default '{}')
returns setof farm_render_jobs language sql as $$
  update farm_render_jobs
  set status = 'processing', claimed_at = now(), heartbeat_at = now(),
      attempts = attempts + 1, progress = 0, phase = 'cloning', error = null
  where id = (
    select j.id from farm_render_jobs j
    where j.status = 'pending' and j.cancel_requested = false
      and not exists (
        select 1 from farm_engine_capabilities c
        where c.engine = j.engine
          and not (c.capability = any (coalesce(p_capabilities, '{}')))
      )
    order by j.priority, j.created_at
    for update skip locked
    limit 1)
  returning *;
$$;

revoke all on function claim_farm_job(text[]) from public, anon, authenticated;
grant execute on function claim_farm_job(text[]) to service_role;

-- Verification after applying (expect the aftereffects row to stay pending):
--   insert into farm_render_jobs (engine, repo_url, params, priority)
--     values ('aftereffects', '-', '{"aftereffects": {"schema_version": 1}}', 900);
--   select id, engine, status from claim_farm_job();               -- claims something else or nothing
--   select id, engine, status from claim_farm_job('{aftereffects}'); -- claims it
-- then cancel / delete that probe row.
