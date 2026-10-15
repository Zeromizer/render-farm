"""Cloud lane: an on-demand RunPod GPU (RTX PRO 6000, 96 GB) beside the PC.

Why: the PC's 16 GB card caps H3 at 768p/10 s, streams weights from disk and
runs one job at a time. Shawn rents a 96 GB card by the second instead of
buying one (2026-10-15, docs/cloud-gpu-brief.md).

Routing: engines gated in farm_engine_capabilities with capability
"gpu-cloud" are never taken by the PC's claim_farm_job() (it advertises
none), and the PC's light-lane loop excludes them too. The pod's worker
(WORKER_ENGINES set) claims only those engines. If no pod takes a job within
CLOUD_FALLBACK_MINUTES, the PC's loop takes it after all (render_worker.py).

Two halves in this module:
  PC   start(log): a thread that creates a pod when gated jobs wait and none
       is up, keeps a per-day hour budget, and terminates a pod that has sat
       idle past its own exit (a safety net; the pod normally ends itself).
  pod  terminate_self(log): called by the pod's worker after IDLE_EXIT_MINUTES
       with nothing to do. Terminate, not stop: everything worth keeping is on
       the network volume, and a stopped pod can come back with no GPU.
"""
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx

import config
import db

API = "https://rest.runpod.io/v1"
POD_PREFIX = "rf-cloud"
TICK_SECONDS = 30
CREATE_BACKOFF_SECONDS = 300
BROKEN_BACKOFF_SECONDS = 2 * 3600
# A pod's first boot installs packages and loads ComfyUI before it claims anything.
BOOT_GRACE_MINUTES = 20
STATE_PATH = os.path.join(config.CACHE_DIR, "cloud_lane.json")


def _headers(key=None):
    return {"Authorization": f"Bearer {key or config.RUNPOD_API_KEY}"}


def list_pods():
    r = httpx.get(f"{API}/pods", headers=_headers(), timeout=30)
    r.raise_for_status()
    return [p for p in r.json() if (p.get("name") or "").startswith(POD_PREFIX)]


def delete_pod(pod_id, key=None):
    httpx.delete(f"{API}/pods/{pod_id}", headers=_headers(key), timeout=30).raise_for_status()


def create_pod(engines):
    body = {
        "name": f"{POD_PREFIX}-{datetime.now(timezone.utc):%m%d-%H%M%S}",
        "imageName": config.RUNPOD_IMAGE,
        "cloudType": "SECURE",
        "gpuTypeIds": config.RUNPOD_GPU_TYPES,
        "gpuCount": 1,
        "networkVolumeId": config.RUNPOD_VOLUME_ID,
        "volumeMountPath": "/workspace",
        "containerDiskInGb": 200,  # models staged locally (cloud/stage-models.sh) + bundle
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "dockerStartCmd": ["bash", "-c", "bash /workspace/render-farm/cloud/boot.sh"],
        "env": {
            # RunPod secrets: the values never sit in the pod's settings.
            "SUPABASE_URL": "{{ RUNPOD_SECRET_supabase_url }}",
            "SUPABASE_SERVICE_KEY": "{{ RUNPOD_SECRET_supabase_service_key }}",
            "WORKER_ENGINES": ",".join(sorted(engines)),
            "IDLE_EXIT_MINUTES": f"{config.IDLE_EXIT_MINUTES:g}",
            "RF_REF": config.CLOUD_GIT_REF,
        },
    }
    if config.RUNPOD_DATACENTER:
        body["dataCenterIds"] = [config.RUNPOD_DATACENTER]
    r = httpx.post(f"{API}/pods", headers=_headers(), json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"RunPod {r.status_code}: {r.text[:300]}")
    return r.json()


def _load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def _pending(engines):
    return len(db.sb.table("farm_render_jobs").select("id")
               .in_("engine", sorted(engines)).eq("status", "pending")
               .execute().data or [])


def _pod_work(pod_id):
    """(busy, last): whether this pod is running a job now, and the epoch
    seconds of the latest heartbeat/finish of a job it claimed (None: never)."""
    rows = (db.sb.table("farm_render_jobs").select("heartbeat_at,completed_at,status")
            .eq("claimed_by", f"cloud:{pod_id}").order("claimed_at", desc=True).limit(5)
            .execute().data or [])
    if any(r["status"] == "processing" for r in rows):
        return True, time.time()
    stamps = [r.get(k) for r in rows for k in ("completed_at", "heartbeat_at") if r.get(k)]
    return False, (max(datetime.fromisoformat(t).timestamp() for t in stamps) if stamps else None)


def _tick(state, log):
    today = datetime.now(timezone.utc).astimezone().date().isoformat()  # PC local day
    if state.get("day") != today:
        state.update(day=today, seconds=0.0)
    engines = db.gated_engines(config.CLOUD_CAPABILITY)
    if not engines:
        return
    pending = _pending(engines)
    now = time.time()
    pods = [p for p in list_pods() if p.get("desiredStatus") != "TERMINATED"]
    state["seconds"] += TICK_SECONDS * len(pods)
    over_budget = state["seconds"] >= config.CLOUD_MAX_HOURS_PER_DAY * 3600
    started = state.setdefault("started", {})

    spare = 0  # pods that will take a waiting job soon: booting, or up and idle
    alive = 0
    for p in pods:
        born = started.setdefault(p["id"], now)
        busy, worked = _pod_work(p["id"])
        why = None
        if over_budget:
            why = "daily budget used"
        elif worked is None and now - born > BOOT_GRACE_MINUTES * 60:
            # Booted but never claimed: broken image/boot. Don't keep paying for
            # it or re-create it every few minutes; the PC fallback takes the jobs.
            why = f"claimed nothing in {BOOT_GRACE_MINUTES} min"
            state["retry_at"] = now + BROKEN_BACKOFF_SECONDS
        elif worked is not None and not busy and now - worked > (config.IDLE_EXIT_MINUTES + 10) * 60:
            why = f"idle {(now - worked) / 60:.0f} min and did not end itself"
        if why:
            log(f"terminating pod {p['id']} ({why})")
            delete_pod(p["id"])
            started.pop(p["id"], None)
            continue
        alive += 1
        if not busy:
            spare += 1

    # Parallel lane: one more pod whenever more jobs wait than pods can take
    # them, up to CLOUD_MAX_PODS. One create per tick, so a burst of jobs does
    # not order a fleet before the first pod has even booted.
    if (pending > spare and alive < config.CLOUD_MAX_PODS and not over_budget
            and now >= state.get("retry_at", 0)):
        try:
            pod = create_pod(engines)
            started[pod["id"]] = now
            log(f"{pending} cloud job(s) waiting, {alive} pod(s) up ({spare} free): "
                f"created pod {pod['id']} (${pod.get('costPerHr', '?')}/h)")
        except Exception as e:  # noqa: BLE001 - out of stock etc.; the PC fallback covers it
            state["retry_at"] = now + CREATE_BACKOFF_SECONDS
            log(f"pod create failed, retrying in {CREATE_BACKOFF_SECONDS // 60} min: {str(e)[:200]}")
    elif pending and over_budget and state.get("budget_logged") != today:
        state["budget_logged"] = today
        log(f"cloud budget of {config.CLOUD_MAX_HOURS_PER_DAY:g} h used today; "
            f"jobs fall back to the PC after {config.CLOUD_FALLBACK_MINUTES:g} min")


def _loop(log):
    state = _load_state()
    err_logged = False
    while True:
        try:
            _tick(state, log)
            _save_state(state)
            err_logged = False
        except Exception as e:  # noqa: BLE001 - never take the worker down
            if not err_logged:
                log(f"tick error (retrying quietly): {str(e)[:200]}")
                err_logged = True
        time.sleep(TICK_SECONDS)


def start(log):
    if not (config.RUNPOD_API_KEY and config.RUNPOD_VOLUME_ID):
        raise RuntimeError("CLOUD_LANE=1 needs RUNPOD_API_KEY and RUNPOD_VOLUME_ID")
    threading.Thread(target=_loop, args=(lambda m: log(f"[cloud] {m}"),),
                     name="cloud-lane", daemon=True).start()
    log(f"[cloud] cloud lane started (gpu {config.RUNPOD_GPU_TYPES[0]}, "
        f"volume {config.RUNPOD_VOLUME_ID}, {config.CLOUD_MAX_HOURS_PER_DAY:g} h/day)")


def fallback_cutoff():
    """ISO time before which a cloud-gated job may be taken by the PC instead."""
    return (datetime.now(timezone.utc) - timedelta(minutes=config.CLOUD_FALLBACK_MINUTES)).isoformat()


def terminate_self(log):
    """On the pod: end this pod. RunPod puts the pod's id and a pod-scoped API
    key in its environment; runpodctl uses them, the REST call is the fallback."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        log("idle exit: RUNPOD_POD_ID not set (not on RunPod?); just exiting")
        return
    log(f"idle exit: terminating pod {pod_id}")
    try:
        r = subprocess.run(["runpodctl", "remove", "pod", pod_id], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return
        log(f"runpodctl failed ({(r.stderr or r.stdout).strip()[:160]}); trying the REST API")
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"runpodctl unavailable ({e}); trying the REST API")
    delete_pod(pod_id, key=os.environ.get("RUNPOD_API_KEY"))
