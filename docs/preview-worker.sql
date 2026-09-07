-- Additive, service-role only. Competes safely with claim_farm_job using row locks.
create or replace function public.claim_farm_preview_job()
returns setof public.farm_render_jobs
language sql
set search_path = public
as $$
  update public.farm_render_jobs
  set status='processing', claimed_at=now(), heartbeat_at=now(),
      attempts=attempts+1, progress=0, phase='cloning', error=null
  where id=(
    select id from public.farm_render_jobs
    where status='pending' and cancel_requested=false
      and engine='hyperframes' and params->>'output_kind'='still'
    order by priority,created_at
    for update skip locked limit 1
  ) returning *;
$$;
revoke all on function public.claim_farm_preview_job() from public, anon, authenticated;
grant execute on function public.claim_farm_preview_job() to service_role;
