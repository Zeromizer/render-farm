"""aftereffects runner: a trusted After Effects recipe authored and rendered on
the native Windows host, delivered as a ProRes 4444 alpha master plus sidecars.

params (jsonb):
  aftereffects:          see worker/aftereffects/schema.py (schema_version 1)
    schema_version, recipe, recipe_revision?, composition, settings, assets[],
    output_profile, org_id, job_id, order_id?, version?

The Astra/Codex container never hosts AE: it submits this row (through the
platform's queue serialization) and waits, exactly like matte / video_gen.

Outputs in the 'renders' bucket, all uploaded before the row turns done:
  outputs/<jid>.mov             ProRes 4444 RGB+A master  (the row's output_path)
  outputs/<jid>-review.webm     VP9 alpha review copy
  outputs/<jid>-bundle.zip      project.aep + assets/ + settings.json + recipes/
  outputs/<jid>-contact.png     sampled frames over light and dark backgrounds
  outputs/<jid>-manifest.json   checks, timings, provenance, editable layers
  outputs/<jid>-proof-<fffff>.png  one PNG per requested proof frame, frame
                                number zero-padded to 5 digits (-proof-00042.png)

Isolation and retries: each attempt works in <work_dir>/attempt-<n>; a retry
first kills only the AfterFX/aerender pids the stale attempt recorded, then
removes that attempt's directory. Before anything is uploaded the row is
re-read: it must still be processing under this attempt number, else the
result is dropped (CLAIM_SUPERSEDED) so a reclaimed job never gets two
publishers. AE work on this machine is serialized through ae_host.Slot.

Errors reach the row as "<CODE>: message" (aftereffects/errors.py).
"""
import os
import shutil
import time

import db
import proc
from runners import gate_common
from aftereffects import ae_host, pipeline, schema, staging
from aftereffects.errors import AEError

SIDE_CARS = (("review", "-review.webm", "video/webm"),
             ("bundle", "-bundle.zip", "application/zip"),
             ("contact_sheet", "-contact.png", "image/png"),
             ("manifest", "-manifest.json", "application/json"))
SLOT_WAIT_SECONDS = int(os.environ.get("AE_SLOT_WAIT_SECONDS", "900"))

def host_available():
    """True when this worker may advertise the 'aftereffects' capability."""
    if os.environ.get("AE_FAKE_HOST") == "1":
        return True
    return ae_host.find_after_effects() is not None


def make_host(log):
    if os.environ.get("AE_FAKE_HOST") == "1":
        from aftereffects.fake_host import FakeHost
        log("aftereffects: AE_FAKE_HOST=1 - using the fake host (test fixture, outputs are NOT After Effects renders)")
        from aftereffects import media
        return FakeHost(log=log, om_kind=os.environ.get("AE_OM_KIND", "sequence"), ffmpeg=media.tool("ffmpeg"))
    return ae_host.RealHost(log=log)


CAPABILITIES_REMOTE = "capabilities/aftereffects.json"


def publish_capabilities(log):
    """Upsert docs/aftereffects-capabilities.json content to the renders bucket
    at capabilities/aftereffects.json so the platform can preflight fonts and
    recipe features without a DB migration. Best effort, called at worker start."""
    import json
    import tempfile
    from aftereffects import capabilities
    try:
        report = capabilities.report()
        tmp = os.path.join(tempfile.gettempdir(), "aftereffects-capabilities.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        remote = db.upload_file(CAPABILITIES_REMOTE, tmp, "application/json")
        log(f"aftereffects capabilities -> {remote} (sha {report['capabilities_sha256'][:12]}, "
            f"{len(report['fonts'])} fonts, recipes {sorted(report['recipes'])})")
        return remote
    except Exception as e:  # noqa: BLE001 - never block the worker over this
        log(f"aftereffects capabilities publish failed (ignored): {str(e)[:200]}")
        return None


def fetch_row(jid):
    rows = db.sb.table("farm_render_jobs").select("status,attempts,cancel_requested").eq("id", jid).execute().data
    return rows[0] if rows else None


def assert_current_claim(jid, attempt):
    """The row must still be this attempt's, processing, and not canceled."""
    row = fetch_row(jid)
    if row is None:
        raise AEError("CLAIM_SUPERSEDED", "job row vanished before publish")
    if row.get("cancel_requested"):
        raise proc.Canceled()
    if row.get("status") != "processing" or int(row.get("attempts") or 0) != attempt:
        raise AEError("CLAIM_SUPERSEDED",
                      f"row is {row.get('status')} at attempt {row.get('attempts')}; this worker ran attempt {attempt}")


def cleanup_stale_attempts(work_dir, current_attempt, log):
    """Kill the processes a previous attempt recorded and drop its directory."""
    if not os.path.isdir(work_dir):
        return
    for name in os.listdir(work_dir):
        if not name.startswith("attempt-"):
            continue
        try:
            n = int(name.split("-", 1)[1])
        except ValueError:
            continue
        if n == current_attempt:
            continue
        stale = os.path.join(work_dir, name)
        pids_file = os.path.join(stale, "pids.json")
        if os.path.exists(pids_file):
            try:
                import json
                with open(pids_file, encoding="utf-8") as f:
                    ae_host.kill_recorded(json.load(f), log)
            except (OSError, ValueError) as e:
                log(f"stale attempt {name}: could not read pids ({e})")
        shutil.rmtree(stale, ignore_errors=True)
        if os.path.exists(stale):
            try:
                os.rename(stale, os.path.join(work_dir, f"stale-{name}-{int(time.time())}"))
            except OSError:
                pass
        log(f"removed stale workspace {name}")


def stage_assets(request, inputs_dir, log, download=None):
    """Content-sniffed staging (aftereffects/staging.py): SVG by document
    prolog, everything else by magic + ffprobe. Storage names carry no suffix."""
    return staging.stage_assets(request, inputs_dir, log, download or gate_common.download)


def run(job, repo, work_dir, heartbeat, log, cancel_check, timeout_seconds):
    jid = job["id"]
    started = time.monotonic()
    params = (job.get("params") or {}).get("aftereffects")
    if not params:
        raise AEError("INVALID_REQUEST", "aftereffects job needs params.aftereffects")
    request = schema.validate_request(params)
    attempt = int(job.get("attempts") or 1)
    host = make_host(log)
    log(f"aftereffects: host={host.info().get('kind')} version={host.info().get('version')} "
        f"recipe={request['recipe']}@{pipeline.recipe_revision(request['recipe'])} attempt={attempt}")

    cleanup_stale_attempts(work_dir, attempt, log)
    ws = os.path.join(work_dir, f"attempt-{attempt}")
    inputs = os.path.join(ws, "inputs")
    os.makedirs(inputs, exist_ok=True)

    def progress(frac):
        heartbeat.progress = 3 + int(frac * 93)

    def phase(name):
        db.set_phase(jid, name, heartbeat.progress)

    with ae_host.Slot(wait_seconds=SLOT_WAIT_SECONDS, cancel_check=cancel_check):
        db.set_phase(jid, "downloading", 2)
        assets_local = stage_assets(request, inputs, log)
        left = timeout_seconds - (time.monotonic() - started)
        if left <= 0:
            raise AEError("TIMEOUT", "no time left after asset download")
        try:
            result = pipeline.run(request, ws, host, assets_local, log, cancel_check, left,
                                  progress=progress, phase=phase, name=jid)
        except proc.TimedOut as e:
            raise AEError("TIMEOUT", str(e))

    for k, v in (result.get("timings") or {}).items():
        log(f"  timing {k}={v}")
    c = result["checks"]
    log(f"aftereffects verified: {c['width']}x{c['height']} {c['fps']}fps {c['frames']} frames "
        f"{c['pix_fmt']} alpha-mean {c['alpha']['mean_alpha_avg']:.1f}")

    assert_current_claim(jid, attempt)
    db.set_phase(jid, "uploading", 97)
    for key, suffix, mime in SIDE_CARS:
        local = result[key]
        remote = db.upload_file(f"outputs/{jid}{suffix}", local, mime)
        log(f"  sidecar {remote} ({os.path.getsize(local)} bytes)")
    for pf in result.get("proofs") or []:
        remote = db.upload_file(f"outputs/{jid}-proof-{pf['frame']:05d}.png", pf["path"], "image/png")
        log(f"  proof frame {remote}")
    assert_current_claim(jid, attempt)   # the primary goes up next, in run_job
    profile = schema.OUTPUT_PROFILES[request["output_profile"]]
    return result["master"], profile["master_ext"], profile["master_mime"]
