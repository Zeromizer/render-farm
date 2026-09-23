"""Archive stale render-platform job workspaces under C:\\rp-workspace.

For every job-<jobId> dir untouched for STALE_DAYS (files AND database), keep only the
job's final render(s) downloaded from the 'renders' bucket, delete everything else,
write a label file and rename the dir to ARCHIVED_job-<jobId>. Matching stale order-*
and manager-* dirs are deleted outright. Also sweeps %TEMP%\\hf-render-*, hyperframes-*
and remotion-* dirs untouched for a day that no process holds open.

Orphan dirs (job ids not in rp_jobs), order/manager dirs that map to no job and
unrecognised names are reported, never deleted.

Dry run by default (read-only). --execute deletes. Nightly (tools/archive-nightly.ps1):

    .venv\\Scripts\\python.exe tools\\archive-stale-workspaces.py --execute --idle-gate --max-jobs 30

Every run appends its receipt to %LOCALAPPDATA%\\render-farm\\logs\\archive-YYYYMMDD.json
(one JSON object per line) and prints a one-line summary last.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timedelta, timezone

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "worker"))
from db import sb  # noqa: E402  (uses the worker .env service key)

WORKSPACE = r"C:\rp-workspace"
SITE = "https://render-platform-sandy.vercel.app/jobs/"
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
STALE_DAYS = 7
TEMP_MIN_AGE_DAYS = 1
TEMP_PREFIXES = ("hf-render-", "hyperframes-", "remotion-")
LABEL = "ARCHIVED - safe to delete.txt"
LOG_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "render-farm", "logs")
ORDER_TERMINAL = {"done", "cancelled", "canceled", "failed"}
TURN_ACTIVE = {"claimed", "running", "queued"}
TURN_BUSY = {"claimed", "running"}  # idle gate: a long-queued zombie turn must not block every night
FARM_ACTIVE = {"pending", "processing", "queued", "claimed"}
BUSY_MARKER = "busy, skipped"

NOW = datetime.now(timezone.utc)
CUTOFF = NOW - timedelta(days=STALE_DAYS)


def gb(b):
    return round(b / 1e9, 2)


def ts(dt):
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "-"


def parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def long(p):
    p = os.path.abspath(p)
    return p if p.startswith("\\\\?\\") else "\\\\?\\" + p


def allrows(table, cols, **eq):
    out, i = [], 0
    while True:
        q = sb.table(table).select(cols)
        for k, v in eq.items():
            q = q.eq(k, v)
        r = q.range(i, i + 999).execute().data
        out += r
        i += 1000
        if len(r) < 1000:
            return out


def _is_link(st):
    return bool(getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT) or stat.S_ISLNK(st.st_mode)


def dir_stats(path):
    """(bytes, files, newest mtime as datetime) without following links."""
    size = files = 0
    newest = 0.0
    stack = [long(path)]
    while stack:
        p = stack.pop()
        try:
            it = os.scandir(p)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                newest = max(newest, st.st_mtime)
                if _is_link(st):
                    continue
                if stat.S_ISDIR(st.st_mode):
                    stack.append(e.path)
                else:
                    size += st.st_size
                    files += 1
    return size, files, (datetime.fromtimestamp(newest, timezone.utc) if newest else None)


def _unlink_any(p):
    """Remove a file or a link of any kind. Links made inside the Linux container
    (WSL reparse points) cannot be opened from Windows - rmtree's scandir into
    them raised WinError 1920 - but the link itself unlinks or rmdirs fine."""
    try:
        os.unlink(p)
    except OSError:
        try:
            os.rmdir(p)
        except OSError:
            os.chmod(p, stat.S_IWRITE)
            os.unlink(p)


def delete_tree(path):
    """Delete a tree bottom-up without ever following a link; long paths and
    read-only files included. Returns a list of failures (empty = fully gone)."""
    fails = []
    root = long(path)
    try:
        st = os.lstat(root)
    except FileNotFoundError:
        return fails
    if _is_link(st) or not stat.S_ISDIR(st.st_mode):
        try:
            _unlink_any(root)
        except OSError as e:
            fails.append(f"{root}: {e}")
        return fails

    def rec(d):
        try:
            entries = list(os.scandir(d))
        except OSError as e:
            fails.append(f"{d}: {e}")
            return
        for e in entries:
            try:
                st = e.stat(follow_symlinks=False)
                if _is_link(st) or not stat.S_ISDIR(st.st_mode):
                    _unlink_any(e.path)
                else:
                    rec(e.path)
            except OSError as ex:
                fails.append(f"{e.path}: {ex}")
        try:
            os.rmdir(d)
        except OSError as ex:
            try:
                os.chmod(d, stat.S_IWRITE)
                os.rmdir(d)
            except OSError:
                fails.append(f"{d}: {ex}")

    rec(root)
    return fails


def storage_index():
    """name -> metadata for every object under renders/outputs."""
    idx, off = {}, 0
    while True:
        r = sb.storage.from_("renders").list("outputs", {"limit": 1000, "offset": off})
        for o in r:
            idx["outputs/" + o["name"]] = o.get("metadata") or {}
        off += 1000
        if len(r) < 1000:
            return idx


def busy_reasons():
    """Empty list = idle: no farm job processing or claimable, ComfyUI queue empty,
    no agent turn claimed/running."""
    reasons = []
    farm = allrows("farm_render_jobs", "id,status", status="processing") + allrows("farm_render_jobs", "id,status", status="pending")
    for f in farm:
        reasons.append(f"farm job {f['id'][:8]} {f['status']}")
    for s in TURN_BUSY:
        for t in allrows("rp_agent_turns", "id,status", status=s):
            reasons.append(f"agent turn {t['id'][:8]} {s}")
    try:
        q = httpx.get(f"{COMFYUI_URL}/queue", timeout=5).json()
        n_run, n_pend = len(q.get("queue_running") or []), len(q.get("queue_pending") or [])
        if n_run or n_pend:
            reasons.append(f"ComfyUI running {n_run} pending {n_pend}")
    except httpx.ConnectError:
        pass  # ComfyUI not running: nothing it could be doing
    except Exception as e:
        reasons.append(f"ComfyUI queue unreadable: {str(e)[:80]}")
    return reasons


def pick_finals(renders):
    """Latest done job-level render set: prefer quality=final, then highest version.
    Returns every slide/format row of that version."""
    rows = [r for r in renders if r["status"] == "done" and r["order_id"] is None and r["output_path"]]
    if not rows:
        return []
    finals = [r for r in rows if r["quality"] == "final"]
    pool = finals or rows
    v = max(r["version"] or 0 for r in pool)
    return sorted((r for r in pool if (r["version"] or 0) == v), key=lambda r: (r["format"] or "", r["slide"] or 0))


def final_name(r, multi):
    ext = (r.get("output_ext") or os.path.splitext(r["output_path"])[1].lstrip(".") or "bin").lstrip(".")
    name = f"FINAL_v{r['version'] or 0}"
    if multi:
        name += f"_{r['format'] or 'default'}_s{r['slide'] or 0}"
    return f"{name}.{ext}"


def build_plan(max_jobs):
    jobs = {j["id"]: j for j in allrows("rp_jobs", "id,title,status,updated_at,client_id,brand_id,output_kind")}
    clients = {c["id"]: c["name"] for c in allrows("rp_clients", "id,name")}
    brands = {b["id"]: b["name"] for b in allrows("rp_brands", "id,name")}
    turns = allrows("rp_agent_turns", "id,job_id,status,created_at")
    orders = allrows("rp_orders", "id,job_id,status")
    renders = allrows("rp_renders", "id,job_id,order_id,status,quality,version,format,slide,output_path,output_ext")
    farm = [f for f in allrows("farm_render_jobs", "id,status,repo_url") if f["status"] in FARM_ACTIVE]

    turn_job = {t["id"]: t["job_id"] for t in turns}
    order_job = {o["id"]: o["job_id"] for o in orders}
    db_reasons = {}  # job_id -> reasons it is NOT db-stale
    for jid, j in jobs.items():
        if parse(j["updated_at"]) and parse(j["updated_at"]) > CUTOFF:
            db_reasons.setdefault(jid, []).append(f"rp_jobs.updated_at {ts(parse(j['updated_at']))}")
    for t in turns:
        if t["status"] in TURN_ACTIVE:
            db_reasons.setdefault(t["job_id"], []).append(f"turn {t['id'][:8]} {t['status']}")
        elif parse(t["created_at"]) > CUTOFF:
            db_reasons.setdefault(t["job_id"], []).append(f"turn {t['id'][:8]} created {ts(parse(t['created_at']))}")
    for o in orders:
        if o["status"] not in ORDER_TERMINAL:
            db_reasons.setdefault(o["job_id"], []).append(f"order {o['id'][:8]} {o['status']}")
    active_repos = {os.path.normcase(os.path.normpath(f["repo_url"])) for f in farm if f["repo_url"] and f["repo_url"] != "-"}
    renders_by_job = {}
    for r in renders:
        renders_by_job.setdefault(r["job_id"], []).append(r)
    objects = storage_index()

    plan = {"archive": [], "deferred": [], "skip": [], "children": [], "mismatch": [], "temp": [], "unknown": []}
    by_job_children = {}
    for e in sorted(os.scandir(WORKSPACE), key=lambda e: e.name):
        if not e.is_dir(follow_symlinks=False):
            continue
        name = e.name
        if name.startswith("ARCHIVED_") or name.startswith(".archive-staging-"):
            continue
        kind, _, ident = name.partition("-")
        if kind in ("job", "manager"):
            jid = ident
        elif kind == "order":
            jid = order_job.get(ident) or turn_job.get(ident)
        else:
            plan["unknown"].append({"dir": name, "reason": "unrecognised dir name (report only)"})
            continue
        size, files, newest = dir_stats(e.path)
        rec = {"dir": name, "path": e.path, "kind": kind, "job_id": jid, "gb": gb(size), "bytes": size,
               "files": files, "newest": ts(newest), "file_stale": bool(newest is None or newest < CUTOFF)}
        if kind != "job":
            if not jid or jid not in jobs:
                plan["skip"].append({**rec, "reason": "maps to no job (report only)"})
            else:
                by_job_children.setdefault(jid, []).append(rec)
            continue

        j = jobs.get(jid)
        if not j:
            plan["skip"].append({**rec, "reason": "job id not in rp_jobs (orphan, report only)"})
            continue
        rec["title"] = j["title"]
        reasons = list(db_reasons.get(jid, []))
        if os.path.normcase(os.path.normpath(e.path)) in active_repos:
            reasons.append("active farm_render_jobs row points here")
        db_stale = not reasons
        if db_stale != rec["file_stale"]:
            plan["mismatch"].append({"dir": name, "title": j["title"], "files_newest": rec["newest"],
                                     "file_stale": rec["file_stale"], "db_stale": db_stale, "db_activity": reasons[:4]})
        if not (db_stale and rec["file_stale"]):
            continue
        finals = pick_finals(renders_by_job.get(jid, []))
        if not finals:
            plan["skip"].append({**rec, "reason": "no done job-level render (reference/brand-kit/draft job?)"})
            continue
        missing = [f["output_path"] for f in finals if f["output_path"] not in objects]
        if missing:
            plan["skip"].append({**rec, "reason": f"final object missing in bucket: {missing[:2]}"})
            continue
        rec["finals"] = [{"id": f["id"], "version": f["version"], "quality": f["quality"], "format": f["format"],
                          "slide": f["slide"], "path": f["output_path"], "size": objects[f["output_path"]].get("size"),
                          "etag": objects[f["output_path"]].get("eTag"), "name": final_name(f, len(finals) > 1)}
                         for f in finals]
        rec["final_bytes"] = sum(f["size"] or 0 for f in rec["finals"])
        rec["client"] = clients.get(j["client_id"]) or "-"
        rec["brand"] = brands.get(j["brand_id"]) or "-"
        rec["last_activity"] = max(x for x in [rec["newest"], ts(parse(j["updated_at"]))] if x != "-")
        plan["archive"].append(rec)

    plan["archive"].sort(key=lambda a: a["last_activity"])  # oldest activity first
    plan["deferred"] = [{"dir": a["dir"], "title": a["title"], "last_activity": a["last_activity"], "gb": a["gb"]}
                        for a in plan["archive"][max_jobs:]]
    plan["archive"] = plan["archive"][:max_jobs]

    selected = {a["job_id"] for a in plan["archive"]}
    for jid, kids in by_job_children.items():
        for k in kids:
            if jid in selected and k["file_stale"]:
                plan["children"].append(k)

    temp = os.environ.get("TEMP") or os.path.join(os.environ["LOCALAPPDATA"], "Temp")
    for e in os.scandir(temp):
        if e.is_dir(follow_symlinks=False) and e.name.startswith(TEMP_PREFIXES):
            size, files, newest = dir_stats(e.path)
            if newest is None or newest < NOW - timedelta(days=TEMP_MIN_AGE_DAYS):
                plan["temp"].append({"dir": e.path, "gb": gb(size), "bytes": size, "newest": ts(newest)})
    return plan


def execute(plan, idle_gate):
    free_before = shutil.disk_usage("C:\\").free
    done, failed, stopped = [], [], None

    def gate(what):
        nonlocal stopped
        if idle_gate:
            busy = busy_reasons()
            if busy:
                stopped = f"went busy before {what}: {'; '.join(busy[:3])}"
        return stopped is None

    archived_ids = set()
    for a in plan["archive"]:
        if not gate(a["dir"]):
            break
        staging = os.path.join(WORKSPACE, f".archive-staging-{a['job_id']}")
        try:
            os.makedirs(staging, exist_ok=True)
            for f in a["finals"]:
                data = sb.storage.from_("renders").download(f["path"])
                if f["size"] is not None and len(data) != f["size"]:
                    raise RuntimeError(f"{f['path']}: size {len(data)} != {f['size']}")
                etag = (f["etag"] or "").strip('"')
                if etag and "-" not in etag and hashlib.md5(data).hexdigest() != etag:
                    raise RuntimeError(f"{f['path']}: md5 mismatch")
                f["sha256"] = hashlib.sha256(data).hexdigest()
                with open(os.path.join(staging, f["name"]), "wb") as fh:
                    fh.write(data)
        except Exception as ex:  # any download problem: skip the whole job, touch nothing
            delete_tree(staging)
            failed.append({"dir": a["dir"], "stage": "download", "error": str(ex)})
            continue

        fails = delete_tree(a["path"])
        target = os.path.join(WORKSPACE, "ARCHIVED_" + a["dir"])
        if not os.path.exists(a["path"]):
            os.rename(staging, target)
        else:  # partial removal: keep finals inside what is left
            for n in os.listdir(staging):
                shutil.move(os.path.join(staging, n), os.path.join(a["path"], n))
            os.rmdir(staging)
            try:
                os.rename(a["path"], target)
            except OSError as ex:
                fails.append(f"rename: {ex}")
                target = a["path"]
        finals_txt = "\n".join(f"  {f['name']}  render {f['id']}  v{f['version']} {f['quality']} {f['format']} slide {f['slide']}  sha256 {f['sha256']}"
                               for f in a["finals"])
        label = (f"Job: {a['title']}\nJob id: {a['job_id']}\nClient: {a['client']}\nBrand: {a['brand']}\n"
                 f"Archived at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
                 f"Site: {SITE}{a['job_id']}\nFinal renders kept:\n{finals_txt}\n\n"
                 f"Composition source and working files were removed on {NOW.date()}; this job can no longer be "
                 f"revised or re-rendered. Delete this folder whenever you like.\n")
        if fails:
            label += f"\nPARTIAL: {len(fails)} item(s) could not be removed.\n"
        with open(os.path.join(target, LABEL), "w", encoding="utf-8") as fh:
            fh.write(label)
        archived_ids.add(a["job_id"])
        (failed if fails else done).append({"dir": a["dir"], "kind": "job", "archived_as": target, "failures": fails[:20]})

    if stopped is None:
        for c in plan["children"]:
            if c["job_id"] not in archived_ids:
                continue
            if not gate(c["dir"]):
                break
            fails = delete_tree(c["path"])
            (failed if fails else done).append({"dir": c["dir"], "kind": c["kind"], "failures": fails[:20]})

    temp_freed = 0
    if stopped is None:
        for t in plan["temp"]:
            if not gate(t["dir"]):
                break
            probe = t["dir"] + ".deleting"
            try:
                os.rename(t["dir"], probe)  # fails if a process holds files open
            except OSError as ex:
                failed.append({"dir": t["dir"], "kind": "temp", "stage": "in use", "error": str(ex)})
                continue
            fails = delete_tree(probe)
            if not fails:
                temp_freed += t["bytes"]
            (failed if fails else done).append({"dir": t["dir"], "kind": "temp", "failures": fails[:20]})

    free_after = shutil.disk_usage("C:\\").free
    not_reached = [a["dir"] for a in plan["archive"] if a["job_id"] not in archived_ids
                   and not any(f["dir"] == a["dir"] for f in failed)]
    return {"freed_gb": gb(free_after - free_before), "free_before_gb": gb(free_before),
            "free_after_gb": gb(free_after), "archived": len(archived_ids),
            "children_deleted": sum(1 for d in done if d["kind"] in ("order", "manager")),
            "temp_freed_gb": gb(temp_freed), "ok": len(done), "failed": failed,
            "stopped": stopped, "not_reached": not_reached}


def write_log(entry):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"archive-{datetime.now().strftime('%Y%m%d')}.json")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="actually delete")
    mode.add_argument("--dry-run", action="store_true", help="report only (the default)")
    ap.add_argument("--idle-gate", action="store_true",
                    help="skip unless the farm, ComfyUI and the agent relay are idle; re-checked before each deletion")
    ap.add_argument("--max-jobs", type=int, default=30, help="archive at most N jobs per run, oldest activity first")
    ap.add_argument("--json", help="also write the full plan/receipt to this file")
    args = ap.parse_args()
    run = "execute" if args.execute else "dry-run"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if args.idle_gate:
        busy = busy_reasons()
        if busy:
            summary = f"{BUSY_MARKER}: {'; '.join(busy[:3])}"
            write_log({"at": started, "mode": run, "summary": summary})
            print(summary)
            return

    plan = build_plan(args.max_jobs)
    reclaim = sum(a["bytes"] - a["final_bytes"] for a in plan["archive"])
    children_b = sum(c["bytes"] for c in plan["children"])
    temp_b = sum(t["bytes"] for t in plan["temp"])
    counts = {"archive": len(plan["archive"]), "deferred": len(plan["deferred"]),
              "skipped": len(plan["skip"]) + len(plan["unknown"]), "children": len(plan["children"]),
              "temp": len(plan["temp"])}

    if args.execute:
        r = execute(plan, args.idle_gate)
        plan["receipt"] = r
        fails = len(r["failed"])
        summary = (f"archive {run}: archived {r['archived']} / deferred {counts['deferred'] + len(r['not_reached'])} / "
                   f"skipped {counts['skipped']} / order+manager {r['children_deleted']} / temp {gb(temp_b)} GB planned, "
                   f"{r['temp_freed_gb']} GB freed / freed {r['freed_gb']} GB / C: free {r['free_before_gb']} -> "
                   f"{r['free_after_gb']} GB / failures {fails}" + (f" / STOPPED {r['stopped']}" if r["stopped"] else ""))
    else:
        free = gb(shutil.disk_usage("C:\\").free)
        summary = (f"archive {run}: would archive {counts['archive']} ({gb(reclaim)} GB) / deferred {counts['deferred']} / "
                   f"skipped {counts['skipped']} / order+manager {counts['children']} ({gb(children_b)} GB) / "
                   f"temp {counts['temp']} ({gb(temp_b)} GB) / C: free {free} GB")
    plan["summary"] = summary

    slim = {k: v for k, v in plan.items() if k not in ("archive", "children", "temp")}
    slim["archive"] = [{k: a[k] for k in ("dir", "title", "last_activity", "gb")} for a in plan["archive"]]
    slim["children"] = [c["dir"] for c in plan["children"]]
    slim["temp"] = plan["temp"]
    log_path = write_log({"at": started, "mode": run, **slim})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=1, default=str)
    print(f"log: {log_path}")
    print(summary)


if __name__ == "__main__":
    main()
