"""RunPod side: the network volume over S3, and a one-shot RTX 5090 pod that renders a job's graphs.

The pod is named "fsh3-*" so the farm's cloud lane (which only manages "rf-cloud*") ignores it.
It removes itself when done (2 h hard cap in the pod script). Keys come from render-farm's .env.
"""
import os
import time
from datetime import datetime, timezone

import boto3
import httpx
from botocore.config import Config
from dotenv import dotenv_values

# render-farm's .env; FSH3_ENV points elsewhere (a git worktree has no .env of its own)
ENV = dotenv_values(os.environ.get("FSH3_ENV") or os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
API = "https://rest.runpod.io/v1"
GPU = os.environ.get("FSH3_GPU", "NVIDIA GeForce RTX 5090")      # Shawn's choice for this work (not the PRO 6000)
IMAGE = os.environ.get("FSH3_IMAGE", "runpod/pytorch:1.4.0-cu1281-torch280-ubuntu2404")
PREFIX = "fsh3"


def s3():
    return boto3.client("s3", endpoint_url="https://s3api-eu-ro-1.runpod.io", region_name="eu-ro-1",
                        aws_access_key_id=ENV["RUNPOD_S3_ACCESS_KEY"], aws_secret_access_key=ENV["RUNPOD_S3_SECRET_KEY"],
                        config=Config(retries={"max_attempts": 5}, read_timeout=300))


def put(local, key):
    s3().upload_file(local, ENV["RUNPOD_VOLUME_ID"], key)


def get(key, local):
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    s3().download_file(ENV["RUNPOD_VOLUME_ID"], key, local)


def ls(prefix):
    c, out, kw = s3(), [], {"Bucket": ENV["RUNPOD_VOLUME_ID"], "Prefix": prefix}
    while True:
        r = c.list_objects_v2(**kw)
        out += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        kw["ContinuationToken"] = r["NextContinuationToken"]


def rm_prefix(prefix):
    c = s3()
    for k in ls(prefix):
        c.delete_object(Bucket=ENV["RUNPOD_VOLUME_ID"], Key=k)


def _h():
    return {"Authorization": f"Bearer {ENV['RUNPOD_API_KEY']}"}


def pods():
    return [p for p in httpx.get(f"{API}/pods", headers=_h(), timeout=30).json() if (p.get("name") or "").startswith(PREFIX)]


def delete_pod(pod_id):
    httpx.delete(f"{API}/pods/{pod_id}", headers=_h(), timeout=30)


def create_pod(job, tries=60, wait=30, log=print):
    """Create the 5090 pod for a job, retrying while EU-RO-1 has none free. Returns the pod id."""
    body = {"name": f"{PREFIX}-{job}-{datetime.now(timezone.utc):%m%d-%H%M%S}", "imageName": IMAGE, "cloudType": "SECURE",
            "gpuTypeIds": [GPU], "gpuCount": 1, "networkVolumeId": ENV["RUNPOD_VOLUME_ID"], "volumeMountPath": "/workspace",
            "containerDiskInGb": 200,   # 250 GB got "no instances" on 5090 hosts
            "dockerStartCmd": ["bash", "-c", f"bash /workspace/fsh3/pod.sh {job}"]}
    if ENV.get("RUNPOD_DATACENTER"):
        body["dataCenterIds"] = [ENV["RUNPOD_DATACENTER"]]
    for i in range(tries):
        r = httpx.post(f"{API}/pods", headers=_h(), json=body, timeout=60)
        if r.status_code < 300:
            pod = r.json()
            log(f"pod {pod['id']} on {(pod.get('machine') or {}).get('gpuTypeId')} at ${pod.get('costPerHr')}/h")
            return pod["id"]
        log(f"no 5090 yet ({r.text[:80]}); retry {i + 1}/{tries}")
        time.sleep(wait)
    raise RuntimeError("no RTX 5090 available")


def wait_job(job, pod_id, timeout=3 * 3600, log=print):
    """Block until the pod's log says it finished, then make sure the pod is gone."""
    t0, seen = time.time(), 0
    key = f"fsh3/jobs/{job}/out/pod.log"
    while time.time() - t0 < timeout:
        time.sleep(20)
        try:
            get(key, os.path.join(os.environ.get("TEMP", "."), f"fsh3_{job}_pod.log"))
            lines = open(os.path.join(os.environ.get("TEMP", "."), f"fsh3_{job}_pod.log"), encoding="utf-8", errors="replace").read().splitlines()
        except Exception:
            continue
        for l in lines[seen:]:
            if l.startswith("[fsh3"):
                log(l)
        seen = len(lines)
        if any("finishing" in l for l in lines):
            break
    time.sleep(20)
    for p in pods():
        if p["id"] == pod_id:
            log(f"pod {pod_id} still listed; deleting it")
            delete_pod(pod_id)
