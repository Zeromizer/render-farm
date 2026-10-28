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
# Dud pods (never claimed) in a row, with no pod seen working in between,
# before the image counts as broken. One dud beside a working twin is a bad
# host, not a bad image (2gkxb8097ojt1m on 2026-10-28 paused the lane 2 h).
BROKEN_AFTER_DUDS = 2
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


def create_pod(engines, gpu_types=None):
    body = {
        "name": f"{POD_PREFIX}-{datetime.now(timezone.utc):%m%d-%H%M%S}",
        "imageName": config.RUNPOD_IMAGE,
        "cloudType": "SECURE",
        "gpuTypeIds": gpu_types or config.RUNPOD_GPU_TYPES,
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
    """(count, age_seconds of the oldest) of gated jobs waiting for a pod."""
    rows = (db.sb.table("farm_render_jobs").select("created_at")
            .in_("engine", sorted(engines)).eq("status", "pending")
            .order("created_at").execute().data or [])
    if not rows:
        return 0, 0.0
    return len(rows), time.time() - datetime.fromisoformat(rows[0]["created_at"]).timestamp()


def _gpu_choice(oldest_age):
    """The GPU list for a new pod, and whether it includes the fallback.

    RunPod picks any listed type at once, with no order of preference over
    time, so the fallback types are only listed once a job has waited
    CLOUD_GPU_FALLBACK_MINUTES. The preferred types stay first in the list:
    a 5090 that turns up later is still taken over a PRO 6000."""
    fb = config.RUNPOD_FALLBACK_GPU_TYPES
    if fb and oldest_age >= config.CLOUD_GPU_FALLBACK_MINUTES * 60:
        return config.RUNPOD_GPU_TYPES + [g for g in fb if g not in config.RUNPOD_GPU_TYPES], True
    return config.RUNPOD_GPU_TYPES, False


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
    pending, oldest_age = _pending(engines)
    gpu_types, with_fallback = _gpu_choice(oldest_age)
    now = time.time()
    pods = [p for p in list_pods() if p.get("desiredStatus") != "TERMINATED"]
    state["seconds"] += TICK_SECONDS * len(pods)
    over_budget = state["seconds"] >= config.CLOUD_MAX_HOURS_PER_DAY * 3600
    started = state.setdefault("started", {})
    # Pods rented from the preferred list only; one of these that has not
    # claimed a job when the fallback opens is replaced rather than waited on.
    preferred_only = state.setdefault("preferred_only", {})

    spare = 0  # pods that will take a waiting job soon: booting, or up and idle
    alive = 0
    any_worked = False
    for p in pods:
        born = started.setdefault(p["id"], now)
        busy, worked = _pod_work(p["id"])
        any_worked = any_worked or worked is not None
        why = None
        if over_budget:
            why = "daily budget used"
        elif (worked is None and with_fallback and preferred_only.get(p["id"])
              and now - born > config.CLOUD_GPU_FALLBACK_MINUTES * 60):
            # A 5090 can be rented and then never start (runtime null for 20
            # min, pjbdk2m0ydwmg8 on 2026-10-15). Normal boots claim in ~2 min,
            # so past the fallback mark try again with the fallback listed --
            # no 2 h backoff, the fallback types are still untried.
            why = f"claimed nothing in {config.CLOUD_GPU_FALLBACK_MINUTES:g} min; retrying with the fallback GPU"
        elif worked is None and now - born > BOOT_GRACE_MINUTES * 60:
            # Booted but never claimed: a bad host, or a broken image/boot. Stop
            # paying for it; whether to pause creates is decided after the loop.
            why = f"claimed nothing in {BOOT_GRACE_MINUTES} min"
            state["duds"] = state.get("duds", 0) + 1
        elif worked is not None and not busy and now - worked > (config.IDLE_EXIT_MINUTES + 10) * 60:
            why = f"idle {(now - worked) / 60:.0f} min and did not end itself"
        if why:
            log(f"terminating pod {p['id']} ({why})")
            delete_pod(p["id"])
            started.pop(p["id"], None)
            preferred_only.pop(p["id"], None)
            continue
        alive += 1
        if not busy:
            spare += 1

    if any_worked:
        state["duds"] = 0  # a pod claimed on this image: it boots fine
    elif state.get("duds", 0) >= BROKEN_AFTER_DUDS:
        # Broken image/boot: don't keep re-creating it; the PC fallback takes the jobs.
        state["duds"] = 0
        state["retry_at"] = now + BROKEN_BACKOFF_SECONDS
        until = datetime.fromtimestamp(state["retry_at"], timezone.utc)
        log(f"creates paused until {until:%H:%M}Z (broken-boot backoff: "
            f"{BROKEN_AFTER_DUDS} pods in a row claimed nothing)")

    # Parallel lane: one more pod whenever more jobs wait than pods can take
    # them, up to CLOUD_MAX_PODS. One create per tick, so a burst of jobs does
    # not order a fleet before the first pod has even booted.
    if (pending > spare and alive < config.CLOUD_MAX_PODS and not over_budget
            and now >= state.get("retry_at", 0)):
        try:
            pod = create_pod(engines, gpu_types)
            started[pod["id"]] = now
            if not with_fallback and config.RUNPOD_FALLBACK_GPU_TYPES:
                preferred_only[pod["id"]] = True
            gpu = (pod.get("machine") or {}).get("gpuTypeId") or pod.get("gpuTypeId") or "/".join(gpu_types)
            log(f"{pending} cloud job(s) waiting (oldest {oldest_age / 60:.0f} min), {alive} pod(s) up "
                f"({spare} free): created pod {pod['id']} on {gpu} (${pod.get('costPerHr', '?')}/h)"
                f"{' [fallback open]' if with_fallback else ''}")
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
