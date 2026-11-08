"""The render queue: a directory of files shared with the workers, $REMAKE_HOME/queue.

    <name>.json  a job, written by submit (tmp + rename)      <name>.run   claimed by a worker (renamed from .json)
    <name>.done  the worker's JSON result                      <name>.err   the worker's traceback

A job is {mode, cond, audio, refs, prompt, num_frames, seed, out, pin, tier [, prefix, prefix_at]}: `pin` holds
each ref's frame index from the window start, `tier` is 480 (a draft, at most 832x480 pixels) or 736 (a final, at most
1280x736; the cond sets the exact size and aspect), and a hop's `prefix` is the
take it continues, whose frames [prefix_at, prefix_at + 17) become its first 17. Workers take jobs in sorted-name
order and leave a hop queued until its prefix exists, so submit queues hops at once.

Every worker heartbeats every 10 s to $REMAKE_HOME/workers/<host>_<gpu>.json =
{t, host, gpu, pid, job, modes, max_frames: {"480": N, "736": M}}; one younger than 120 s is live. A .run older than
120 s that no live heartbeat names is an orphan (its worker died); a live worker puts it back to .json.
"""
import json
import subprocess
import time
from pathlib import Path

from .common import ARM, FPS, QUEUE, WORKERS, jr, kind
from .kit import prefix_take, take_fp
from .prompts import fp, stamp

LIVE = 120


def workers(mode=None):
    """Live heartbeats (of workers that render `mode`)."""
    out = [json.loads(p.read_text()) for p in WORKERS.glob("*.json")]
    return [hb for hb in out if time.time() - hb["t"] < LIVE and (mode is None or mode in hb["modes"])]


def max_frames(mode, tier):
    """The longest render every live worker of `mode` takes at this tier (480 | 736), or None with none up."""
    fs = [hb["max_frames"].get(str(tier), 0) for hb in workers(mode)]
    return min((x for x in fs if x > 0), default=None)


def job(p):
    """A queued job's JSON, or None once a worker has claimed it."""
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


def cut720(d, i, w):
    """win<i>/cond720.mp4: the render span of case/cond720.mp4 (same frames as the draft cond, final canvas)."""
    s, e = w["render"]
    out = d / "case" / f"win{i}/cond720.mp4"
    if not out.exists():  # cut_window drops it when the window is cut again
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", d / "case" / "cond720.mp4", "-vf",
                        f"select=between(n\\,{s}\\,{e - 1}),setpts=N/{FPS}/TB", "-r", str(FPS), "-c:v", "libx264",
                        "-crf", "12", "-pix_fmt", "yuv420p", out], check=True)
    return out


def submit(d, sid, seeds, windows=None, canvas="720"):
    C = d / "case"
    ws = jr(d, "refs.json")["windows"]
    canvas, mode = str(canvas), kind(d)
    tier = 480 if canvas == "480" else 736
    todo = [i for i in (windows or range(len(ws))) if not ws[i].get("passthrough")]
    live = max_frames(mode, tier)
    assert live or canvas == "480", (f"no live {mode} worker renders finals (tier 736): start "
                                     "viggle-remake-worker on a card that takes it, or submit drafts (canvas 480)")
    bad = []
    for i in todo:
        pj, sj = C / f"win{i}/prompt.json", C / f"win{i}/stamp.json"
        n = ws[i]["render"][1] - ws[i]["render"][0]
        if not ws[i]["refs"] and "prefix_of" not in ws[i]:
            bad.append(f"w{i}: no painted keyframe (paint, then windows again)")
        elif any(not (C / "paint" / ARM / f"f{f}.png").exists() for f in ws[i]["refs"]):
            bad.append(f"w{i}: a keyframe is not painted")
        elif not (C / f"win{i}/prompt.txt").exists():
            bad.append(f"w{i}: no prompt (write_prompts / set_prompt)")
        elif (not pj.exists() or not sj.exists() or json.loads(pj.read_text()).get("fp") != fp(d, ws[i])
              or json.loads(sj.read_text()) != stamp(d, ws[i])):
            bad.append(f"w{i}: stale prompt - the window, its keyframes, their paints or case.json changed since it "
                       "was written (write_prompts / set_prompt for this window)")
        elif live and n > live:
            bad.append(f"w{i}: {n} frames, but the live workers take at most {live} at tier {tier} "
                       f"(windows sizes the windows to them)")
    assert not bad, "not submitted:\n" + "\n".join(bad)
    QUEUE.mkdir(parents=True, exist_ok=True)
    (C / "takes").mkdir(exist_ok=True)
    n, skip, gone = 0, 0, []
    for s in seeds:  # seed-major, longest windows first: every window gets a take before any gets its second
        for i in sorted(todo, key=lambda i: ws[i]["render"][0] - ws[i]["render"][1]):
            w, (r0, r1) = ws[i], ws[i]["render"]
            tfp = take_fp(d, i, w, s, canvas)
            out = C / "takes" / f"w{i}_{canvas}_s{s}_{tfp}.mp4"
            # name order = queue order: drafts (4), then hops (5), then finals (6); seed-major across sessions,
            # longest first
            hop = "prefix_of" in w
            name = f"{'5' if hop else '4' if canvas == '480' else '6'}0_s{s:02d}_{999 - (r1 - r0):03d}_{sid}_{canvas}_w{i:02d}_{tfp}"
            for p in QUEUE.glob(f"*_s{s:02d}_*_{sid}_{canvas}_w{i:02d}_*.json"):  # a queued job of older inputs
                if p.stem != name:
                    p.unlink(missing_ok=True)
                    gone.append(p.stem)
            if out.exists() or any((QUEUE / f"{name}{x}").exists() for x in (".json", ".run")):
                skip += 1
                continue
            for x in (".err", ".done"):  # a retry starts clean
                (QUEUE / f"{name}{x}").unlink(missing_ok=True)
            job = {"mode": mode, "cond": str(C / f"win{i}/cond.mp4") if canvas == "480" else str(cut720(d, i, w)),
                   "audio": str(C / f"win{i}/audio.m4a"), "refs": [str(C / "paint" / ARM / f"f{f}.png") for f in w["refs"]],
                   "prompt": (C / f"win{i}/prompt.txt").read_text(), "num_frames": r1 - r0, "seed": s, "out": str(out),
                   "pin": [f - r0 for f in w["refs"]], "tier": tier}
            if hop:
                job.update(prefix=str(prefix_take(d, w["prefix_of"], s, canvas)), prefix_at=w["prefix_at"])
            tmp = QUEUE / f".{name}.tmp"
            tmp.write_text(json.dumps(job, indent=1))
            tmp.rename(QUEUE / f"{name}.json")
            n += 1
    st = state(sid)
    return {"queued": n, "skipped_existing": skip, "withdrawn_stale": gone, "windows": todo, "canvas": canvas,
            "seeds": seeds, "hops_stuck": st["hops_stuck"]}


def state(sid):
    """This session's jobs by state. A queued hop waits while its prefix take is on disk or a job still pending makes
    it; it is stuck when nothing will (that job failed, or it was never submitted)."""
    st = {k: [] for k in (".json", ".run", ".done", ".err")}
    for p in QUEUE.glob(f"*_{sid}_*"):
        if p.suffix in st:
            st[p.suffix].append(p)
    beats = workers()
    names = {hb["job"] for hb in beats}
    orphans = [p for p in st[".run"] if p.stem not in names and p.exists() and time.time() - p.stat().st_mtime > LIVE]
    queued = {p.stem: j for p in st[".json"] if (j := job(p))}
    running = {p.stem: j for p in st[".run"] if p not in orphans and (j := job(p))}
    hops = {k: j for k, j in queued.items() if j.get("prefix") and not Path(j["prefix"]).exists()}
    stuck = set(hops)
    while True:  # a hop is not stuck once a pending job that is not stuck makes its prefix
        outs = {j["out"] for k, j in {**queued, **running}.items() if k not in stuck}
        left = {k for k in stuck if hops[k]["prefix"] not in outs}
        if left == stuck:
            break
        stuck = left
    return {"queued": len(queued) - len(stuck), "running": len(running), "done": len(st[".done"]), "failed": len(st[".err"]),
            "orphaned": [p.stem for p in orphans], "errors": [p.read_text()[-600:] for p in st[".err"][:2]],
            "hops_waiting": sorted(set(hops) - stuck), "hops_stuck": sorted(stuck), "workers": beats}


def wait(sid, max_s=540):
    """Block until none of this session's jobs is pending (queued and not stuck, or running), no live worker is up,
    or max_s (<= 1800) passes."""
    t0 = time.time()
    while True:
        st = state(sid)
        if st["queued"] + st["running"] == 0 or not st["workers"] or time.time() - t0 > min(max_s, 1800):
            break
        time.sleep(10)
    out = {k: st[k] for k in ("queued", "running", "done", "failed", "errors", "hops_waiting", "hops_stuck")}
    out["orphaned"] = [f"{n}: its worker died; a live worker requeues it" for n in st["orphaned"]]
    out["hops_stuck"] = [f"{n}: nothing pending makes the take it continues (that window failed or was not submitted): "
                         "submit the window before it" for n in st["hops_stuck"]]
    out["all_queue"] = len(list(QUEUE.glob("*.json"))) + len(list(QUEUE.glob("*.run")))
    out["workers_alive"] = [f"{hb['host']}:{hb['gpu']} {hb['job'] or 'idle'} modes {hb['modes']} max_frames {hb['max_frames']}"
                            for hb in st["workers"]]
    if not st["workers"]:
        out["note"] = "no live worker: queued jobs wait until one starts (viggle-remake-worker)"
    return out
