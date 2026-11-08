"""remake engine driver: a person swap with Viggle Remake, through the vendored Viggle MCP's own tool functions
(vendor/viggle_remake_mcp/server.py), called in the order their AGENTS.md prescribes. Two stages, two machines:

  driver.py paint  <spec.json>   PC (no GPU): ingest, case files, character sheet, keyframes painted on the Codex
                                 subscription (REMAKE_PAINT=pool + shapes.install), windows, prompt briefs
  driver.py render <spec.json>   cloud 5090: the agent's prompts stored through their lint (set_prompt), their
                                 ComfyUI 0.42 worker (comfy_worker/remake_worker.py) renders, their strip + finish

The session folder ($REMAKE_HOME/sessions/<sid>) is the hand-over: paint packs it into a tar the runner uploads, render
unpacks it into its own REMAKE_HOME. Nothing in vendor/ is edited; deviations are here and listed in VENDOR.md.

Stdout: PHASE <text> / PROGRESS <0-100> lines for the runner; anything else is log.

paint spec:
  work_dir, home (REMAKE_HOME), sid, target (video), photo (identity picture: a photo or a cast sheet), tar_out,
  report_out, start_s, dur_s, crop, name (character name, letters/digits), identity (who: face, hair, build, age),
  outfit (the whole outfit to the shoes; mode face = the SOURCE person's outfit), person (how to recognise the person
  to replace in the source), others [{desc}] (people who stay), keyframes [{frame | at_s, where}] (default: one per
  shot with the person, mid-shot), shots_with_person (shot indices; default all), kind person|anthro|creature,
  note (to every paint), tar_in + repaint {frames, note} (repaint keyframes of an earlier paint job's session)
render spec:
  work_dir, home, sid, tar_in, prompts {window: text}, seeds [0], comfy_worker_dir, report_out, final_out, sbs_out,
  strip_out, worker_timeout_s
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(HERE))


def emit(kind, text):
    print(f"{kind} {text}", flush=True)


def load(spec_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    os.environ["REMAKE_HOME"] = spec["home"]               # their common.py reads it at import
    os.environ.setdefault("REMAKE_POOL_DIR", str(Path(spec["work_dir"]) / "pool"))
    return spec


def texts(r):
    return [x for x in (r if isinstance(r, list) else [r]) if isinstance(x, str)]


# ---------------------------------------------------------------- paint (PC)

def pack(d, sid, out):
    with tarfile.open(out, "w:gz") as t:
        t.add(d, arcname=sid, filter=lambda ti: None if ti.name.endswith(".tmp") else ti)


def paint_edit(spec, S, kit, d, sid, warnings):
    """Edit session (their AGENTS.md "Edit sessions"): the object is a pseudo-person X1 replaced by `name`, people and
    other things kept ("none"); one epaint per keyframe with the instruction (what changes, the new object's OWN shape,
    what fills the space) and the pictures of what the edit brings in as refs."""
    nf, bounds = kit.nframes(d), kit.shot_bounds(d)
    kfs = []
    for k in spec.get("keyframes") or []:
        f = int(k["frame"]) if k.get("frame") is not None else round(float(k["at_s"]) * 24)
        kfs.append((min(max(f, 0), nf - 1), k["where"], (k.get("instruction") or "").strip()))
    assert kfs, ("keyframes are required: one per shot that shows the object, where it is clearly visible and large, "
                 ">= 3 frames from a cut, each with `where` (where the object is)")
    shot_of = lambda f: next(j for j, (a, b) in enumerate(bounds) if a <= f < b)  # noqa: E731
    with_obj = set(spec.get("shots_with_object") or []) | {shot_of(f) for f, _, _ in kfs}
    missing = sorted(with_obj - {shot_of(f) for f, _, _ in kfs})
    if missing:
        warnings.append(f"shots {missing} show the object but have no keyframe: they keep the original object")
    name = spec["name"]
    people = [{"id": "X1", "desc": spec["object"], "replace_with": name}]
    people += [{"id": f"K{i + 1}", "desc": o["desc"], "replace_with": "none"} for i, o in enumerate(spec.get("others") or [])]
    shots = [{"start": a, "end": b, "people": (["X1"] if i in with_obj else []), "desc": ""}
             for i, (a, b) in enumerate(bounds)]
    where = {str(f): {"X1": w, **({"_note": spec["note"]} if spec.get("note") else {})} for f, w, _ in kfs}
    print(S.save_case(sid, plan_grid={"people": people, "shots": shots},
                      case={"chars": {name: spec["replacement"]}, "outfits": {name: "none"}, "edit": spec["edit"]},
                      where=where), flush=True)
    emit("PROGRESS", 15)
    refs = [S.into_session(d, r) for r in spec.get("refs") or []]   # copied into photos/, session-relative
    st = {"instruction": spec["instruction"], "instructions": {}, "refs": refs}
    emit("PHASE", f"painting {len(kfs)} keyframe(s)")
    for f, _, own in kfs:
        text = own or spec["instruction"]
        st["instructions"][str(f)] = text
        for x in texts(S.epaint(sid, f, text, refs=refs)):
            print(x, flush=True)
            if '"error"' in x:
                warnings.append(f"paint failed: {x[:300]}")
    json.dump(st, open(d / "case" / "farm_edit.json", "w", encoding="utf-8"), indent=1)


def paint(spec):
    os.environ["REMAKE_PAINT"] = "pool"                    # Codex subscription, one helper codex per picture
    if spec.get("codex_auth"):
        os.environ["REMAKE_CODEX_AUTH"] = spec["codex_auth"]
    if spec.get("codex_bin"):
        os.environ["PATH"] = str(Path(spec["codex_bin"]).parent) + os.pathsep + os.environ.get("PATH", "")
    from viggle_remake_mcp import kit, paint as pt, server as S  # noqa: E402
    import shapes
    shapes.install(pt, log=lambda m: print(m, flush=True))
    sid, home = spec["sid"], Path(spec["home"])
    d = home / "sessions" / sid
    warnings = []

    stage = spec.get("stage") or "paint"
    if stage == "prepare":                                  # ingest + shot strips: the agent picks keyframes from them
        emit("PHASE", "ingesting the video")
        kind = spec.get("kind") or "swap"                   # swap: a person; edit: an object (a car) replaced
        assert kind in ("swap", "edit"), "kind is swap or edit"
        print(S.new_session(sid, kind), flush=True)
        r = S.ingest(sid, spec["target"], float(spec.get("start_s") or 0.0), spec.get("dur_s"), spec.get("crop"))
        print(texts(r)[0], flush=True)
        report = {"sid": sid, "stage": "prepare", "kind": kind, "frames": kit.nframes(d), "fps": 24,
                  "cuts": json.load(open(d / "case" / "cuts.json")), "shots": kit.shot_bounds(d),
                  "meta": json.load(open(d / "case" / "meta.json")), "ingest": texts(r)[0],
                  "strips": sorted(x.relative_to(d).as_posix() for x in (d / "case" / "look").glob("strips_*.jpg"))}
        json.dump(report, open(spec["report_out"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        pack(d, sid, spec["tar_out"])
        emit("PROGRESS", 100)
        return

    emit("PHASE", "unpacking the session")
    with tarfile.open(spec["tar_in"]) as t:
        t.extractall(home / "sessions")
    kind = (d / "kind").read_text().strip()
    if stage == "repaint":
        rp = spec.get("repaint") or {}
        frames = [int(f) for f in rp.get("frames") or []]
        assert frames, "repaint needs frames"
        emit("PHASE", f"repainting {frames}")
        for f in frames:
            if kind == "edit":                              # their epaint again, with the corrected instruction
                st = json.load(open(d / "case" / "farm_edit.json"))
                text = rp.get("instruction") or st["instructions"].get(str(f)) or st["instruction"]
                st["instructions"][str(f)] = text
                json.dump(st, open(d / "case" / "farm_edit.json", "w", encoding="utf-8"), indent=1)
                for x in texts(S.epaint(sid, f, text, refs=st["refs"], force=True)):
                    print(x, flush=True)
            else:
                for x in texts(S.paint(sid, [f], note=rp.get("note") or "", force=True, quality="medium")):
                    print(x, flush=True)
    elif kind == "edit":
        paint_edit(spec, S, kit, d, sid, warnings)
    else:
        nf = kit.nframes(d)
        bounds = kit.shot_bounds(d)
        kfs = []
        for k in spec.get("keyframes") or []:
            f = int(k["frame"]) if k.get("frame") is not None else round(float(k["at_s"]) * 24)
            kfs.append((min(max(f, 0), nf - 1), k["where"]))
        assert kfs, ("keyframes are required: one per shot that shows the person, where they are largest with the "
                     "whole face visible, >= 3 frames from a cut, each with `where` (their position and pose)")
        shot_of = lambda f: next(j for j, (a, b) in enumerate(bounds) if a <= f < b)  # noqa: E731
        with_person = set(spec.get("shots_with_person") or []) | {shot_of(f) for f, _ in kfs}
        missing = sorted(with_person - {shot_of(f) for f, _ in kfs})
        if missing:
            warnings.append(f"shots {missing} show the person but have no keyframe: they render as the source person")
        name = spec["name"]
        people = [{"id": "P1", "desc": spec["person"], "replace_with": name}]
        people += [{"id": f"K{i + 1}", "desc": o["desc"], "replace_with": "none"} for i, o in enumerate(spec.get("others") or [])]
        shots = [{"start": a, "end": b, "people": (["P1"] if i in with_person else []), "desc": ""}
                 for i, (a, b) in enumerate(bounds)]
        where = {str(f): {"P1": w, **({"_note": spec["note"]} if spec.get("note") else {})} for f, w in kfs}
        print(S.save_case(sid, plan_grid={"people": people, "shots": shots},
                          case={"chars": {name: spec["identity"]}, "outfits": {name: spec["outfit"]}},
                          where=where), flush=True)
        emit("PROGRESS", 10)
        emit("PHASE", "painting the character sheet")
        for x in texts(S.sheet(sid, name, "photo", photo=spec["photo"], kind=spec.get("kind") or "person")):
            print(x, flush=True)
        emit("PROGRESS", 30)
        # their rule: the two keyframes of one shot go in one call (one look); other frames one per call
        groups, by_shot = [], {}
        for f, _ in kfs:
            by_shot.setdefault(shot_of(f), []).append(f)
        for fs in by_shot.values():
            groups += [fs[:2]] + [[f] for f in fs[2:]]
        emit("PHASE", f"painting {len(kfs)} keyframe(s)")
        for x in texts(S.paint_many(sid, groups, quality="medium")):
            print(x, flush=True)
            if '"error"' in x:
                warnings.append(f"paint failed: {x[:300]}")
            if "kept_source_face" in x:
                warnings.append(f"a keyframe kept the source face: {x[:300]}")
    emit("PROGRESS", 75)

    emit("PHASE", "cutting windows and writing briefs")
    print(S.windows(sid), flush=True)
    # swap: their brief per window (write_prompts); edit: their cut_windows (guide + the edit FORMAT EXAMPLE), the
    # prompt is written by hand from it, as their AGENTS.md says for edit sessions
    r = S.write_prompts(sid) if kind == "swap" else S.cut_windows(sid)
    brief = texts(r)[0]
    rf = json.load(open(d / "case" / "refs.json"))
    painted = {int(p.stem[1:]) for p in (d / "case" / "paint" / "g2").glob("f*.png") if p.stem[1:].isdigit()}
    report = {
        "sid": sid, "frames": kit.nframes(d), "cuts": json.load(open(d / "case" / "cuts.json")),
        "meta": json.load(open(d / "case" / "meta.json")),
        "keyframes": sorted(painted),
        "comparisons": sorted(p.relative_to(d).as_posix() for p in (d / "case" / "paint").glob("cmp_*.jpg")),
        "kind": kind,
        "sheet": f"case/chars/{spec.get('name') or ''}_sheet.png" if kind == "swap" else None,
        "windows": [{"w": i, "render": w["render"], "own": w["own"], "refs": w.get("refs", []),
                     "passthrough": bool(w.get("passthrough")), "sheet": f"case/win{i}/sheet.jpg"}
                    for i, w in enumerate(rf["windows"])],
        "brief": brief,
        "budget": json.loads(S.budget(sid)),
        "warnings": warnings,
    }
    json.dump(report, open(spec["report_out"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    emit("PHASE", "packing the session")
    pack(d, sid, spec["tar_out"])
    emit("PROGRESS", 100)


# ---------------------------------------------------------------- render (cloud)

def start_worker(spec, log):
    """Their comfy worker (unchanged), as their run_worker.sh starts it, on this job's own REMAKE_HOME queue."""
    cw = Path(spec["comfy_worker_dir"])
    py = cw / "ComfyUI" / "venv" / "bin" / "python"
    env = dict(os.environ, REMAKE_HOME=spec["home"], PYTHONUNBUFFERED="1")
    logf = open(Path(spec["work_dir"]) / "remake_worker.log", "w")
    p = subprocess.Popen([str(py), "-u", "remake_worker.py", "--home", spec["home"]], cwd=cw, env=env,
                         stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    return p, logf


def wait_heartbeat(home, p, timeout=900):
    wd = Path(home) / "workers"
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            raise RuntimeError(f"the render worker exited ({p.returncode}) while loading; see remake_worker.log")
        if any(wd.glob("*.json")):
            return time.time() - t0
        time.sleep(3)
    raise RuntimeError("the render worker did not come up in 15 min")


def render(spec):
    from viggle_remake_mcp import jobs, kit, server as S  # noqa: E402
    sid, home = spec["sid"], Path(spec["home"])
    d = home / "sessions" / sid
    emit("PHASE", "unpacking the session")
    with tarfile.open(spec["tar_in"]) as t:
        t.extractall(home / "sessions")
    (home / "queue").mkdir(parents=True, exist_ok=True)

    emit("PHASE", "loading the Viggle model")
    p, logf = start_worker(spec, print)
    try:
        secs = wait_heartbeat(spec["home"], p)
        print(f"worker up in {secs:.0f}s", flush=True)
        emit("PROGRESS", 8)
        print(S.windows(sid), flush=True)                  # sized for the live worker, as their AGENTS.md says
        if (d / "kind").read_text().strip() == "edit":
            S.cut_windows(sid)                              # cuts each window for its span
        else:
            S.write_prompts(sid)                            # cuts each window for its span (briefs ignored here)
        for w, text in (spec.get("prompts") or {}).items():
            print(S.set_prompt(sid, int(w), text), flush=True)   # their lint: raises on a bad prompt
        seeds = [int(s) for s in spec.get("seeds") or [0]]
        print(S.submit(sid, seeds=seeds, canvas="720"), flush=True)
        emit("PHASE", "rendering")
        t0, deadline = time.time(), time.time() + float(spec.get("worker_timeout_s") or 7200)
        while True:
            st = jobs.state(sid)
            left, done = st["queued"] + st["running"], st["done"]
            if left == 0 or st["hops_stuck"]:
                break
            if p.poll() is not None:
                raise RuntimeError(f"the render worker died ({p.returncode}); see remake_worker.log")
            if time.time() > deadline:
                raise RuntimeError("render timed out")
            # their worker reports per window; the sampler's "k/6" in its log moves the bar inside one
            try:
                with open(Path(spec["work_dir"]) / "remake_worker.log", "rb") as f:
                    f.seek(max(0, f.seek(0, 2) - 4000))
                    steps = re.findall(rb"(\d)/6 \[", f.read())
                step = int(steps[-1]) / 6 if steps and st["running"] else 0.0
            except OSError:
                step = 0.0
            emit("PROGRESS", int(10 + 75 * (done + step) / max(1, done + left)))
            time.sleep(10)
        st = jobs.state(sid)
        if st["failed"] or st["hops_stuck"]:
            raise RuntimeError(f"render failed ({st['failed']} window(s), stuck hops {st['hops_stuck']}): "
                               + (st["errors"][0] if st["errors"] else ""))
        print(f"rendered in {time.time() - t0:.0f}s", flush=True)
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001
            p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
        logf.close()

    emit("PHASE", "review strip and finish")
    emit("PROGRESS", 88)
    st = texts(S.strip(sid))[0]
    print(st, flush=True)
    rf = json.load(open(d / "case" / "refs.json"))
    seed = (spec.get("seeds") or [0])[0]
    picks = {str(i): int(seed) for i, w in enumerate(rf["windows"]) if not w.get("passthrough")}
    print(S.finish(sid, picks), flush=True)
    final = d / "case" / "final" / "full_720.mp4"
    shutil.copyfile(final, spec["final_out"])
    strips = sorted((d / "case" / "review").glob("720_w*.jpg"))
    if strips:
        shutil.copyfile(strips[0], spec["strip_out"])
    # side by side: source span | result, the source stretched back the way finish does
    src = d / "case" / "cond720.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(final), "-filter_complex",
                    "[0:v]scale=-2:720,setsar=1[a];[1:v]scale=-2:720,setsar=1[b];[a][b]hstack=shortest=1[v]",
                    "-map", "[v]", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", spec["sbs_out"]], check=True)
    flags = [l.strip() for l in st.splitlines() if re.search(r"cut timing|cut f\d+|extra cut|dropped|edge bands", l)]
    json.dump({"sid": sid, "picks": picks, "strip": st, "flags": flags,
               "windows": [{"w": i, "render": w["render"], "refs": w.get("refs", [])} for i, w in enumerate(rf["windows"])],
               "worker_log_tail": (Path(spec["work_dir"]) / "remake_worker.log").read_text(errors="replace")[-3000:]},
              open(spec["report_out"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    emit("PROGRESS", 100)


if __name__ == "__main__":
    cmd, spec_path = sys.argv[1], sys.argv[2]
    spec = load(spec_path)
    {"paint": paint, "render": render}[cmd](spec)
