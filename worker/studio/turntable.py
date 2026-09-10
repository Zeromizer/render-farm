"""Car turntable flow: a seamless, constant-speed 360 of a product from
straight-on photos, for background removal. Two variants share one pipeline:

  two-anchor   front + rear photos      -> front_to_rear, rear_to_front (2 x 180)
  four-anchor  front + left + rear + right -> front_to_left, left_to_rear,
                                            rear_to_right, right_to_front (4 x 90)

"left"/"right" are the VEHICLE's sides (the side to a seated driver's left),
not the viewer's. The rotation is clockwise as seen from above throughout:
from the front view the vehicle's left flank swings round to face the camera
first, then the rear, then the right flank, then the front again. That is the
same direction the proven two-anchor flow used ("left side sweeps past").

HOW THE ANCHORS CONDITION THE MODEL (verified against the installed ComfyUI
0.34 nodes, 2026-09-06): MiniMaxH3ImageToVideo takes exactly first_frame and
last_frame; MiniMaxH3ReferenceToVideo takes ref images but no frame anchors.
There is no node that accepts four stills at once, so a side photo influences
the model ONLY as the end frame of one quarter and the start frame of the
next. Between anchors the geometry is the model's guess, guided by the prompt.

Proven 2026-09-06 (two-anchor) on the Proton e.MAS 7:

1. Pad the photos to the generation canvas with ONE common scale (no crop,
   letterboxed in each photo's own background colour) so every anchor frame IS
   the photo and the car keeps its relative size between views.
2. Each segment is image-to-video with first_frame = its start photo and
   last_frame = its end photo, so shared anchors are pixel exact and a complete
   sequence loops.
   PROMPT RULE: never name what you do not want in detail. A prompt saying
   "no overhead view, no top-down view, no grey studio floor" produced an
   aerial shot over a town. Short positive wording plus the drift check works.
3. Drift: a segment whose corners leave the flat backdrop (grey floor,
   overhead camera) is regenerated with another seed, up to 3 tries.
4. Repair: the model occasionally 'teleports' the car mid-clip to reach the
   last frame (one frame with 4-6x the normal motion). The clean part is kept,
   its last clean frame becomes a new start anchor, and the remainder to the
   segment's own end photo is regenerated, so a repair never leaves its
   quarter or changes direction.
5. Segments can be supplied ready-made (ready_segments) and are then not
   regenerated: approve quarters one at a time, then assemble.
6. Join on the shared frames, check every transition and the loop seam,
   time-remap to constant angular speed (the model eases at every anchor),
   RIFE to the output fps, lanczos to the output size (a resize, not detail).
"""
import os
import time

from videogen import estimate, graphs
from studio import post

SEQUENCES = {
    "two": ("front_to_rear", "rear_to_front"),
    "four": ("front_to_left", "left_to_rear", "rear_to_right", "right_to_front"),
}
ANCHORS = ("front", "left", "rear", "right")
VIEW = {"front": "straight-on front view", "rear": "straight-on rear view",
        "left": "straight-on view of its left side", "right": "straight-on view of its right side"}
# What sweeps past the camera in each segment; every text is the same clockwise turn.
SWEEP = {
    "front_to_rear": ("turning exactly 180 degrees clockwise as seen from above so that its left side sweeps past "
                      "the camera"),
    "rear_to_front": ("continuing the same clockwise turn as seen from above for exactly 180 degrees so that its "
                      "right side sweeps past the camera"),
    "front_to_left": ("turning exactly 90 degrees clockwise as seen from above so that its left side swings round "
                      "to face the camera"),
    "left_to_rear": ("continuing the same clockwise turn as seen from above for exactly 90 degrees so that its left "
                     "side swings away and the rear comes round to face the camera"),
    "rear_to_right": ("continuing the same clockwise turn as seen from above for exactly 90 degrees so that its "
                      "right side swings round to face the camera"),
    "right_to_front": ("continuing the same clockwise turn as seen from above for exactly 90 degrees so that its "
                       "right side swings away and the front comes round to face the camera"),
}
BASE_PROMPT = (
    "{car} on a seamless plain light studio background, exactly as in the given frames. One single rigid car, "
    "rotating in place about its own vertical axis at a perfectly constant, slow angular speed, no easing in or out, "
    "no pauses, no speed changes, {motion}. Its paint colour, rims, tyres, roofline, doors, windows, lamps, badges "
    "and spoiler stay exactly as in the frames throughout. The car stays the same size and stays centred; the whole "
    "car including all wheels stays fully inside the frame with empty space on both sides. Fixed camera at eye "
    "level, fixed framing, no camera movement, no zoom. No turntable, no floor, no ground shadow, no reflection, no "
    "horizon, just the plain background. {details} Soft even studio lighting that does not change, no people, no "
    "text overlays. Quiet studio ambience."
)
REPAIR_MOTION = ("continuing the same clockwise turn as seen from above from the exact view in the first frame until "
                 "it reaches the {target} shown in the last frame; the same side of the car stays visible throughout "
                 "and it never turns the other way")
DEFAULTS = {"resolution": "768p", "ratio": "16:9", "seconds_per_half": 10, "seconds_per_quarter": 5, "seed": 21,
            "fps": 60, "shorter_size": 1080, "density": 10, "details": ""}
SECONDS_LIMITS = {"two": (4, 10), "four": (3, 10)}
TRANSITION_MAX_DIFF = 6.0      # mean luma diff (256 px grey) of the frame a seam shares: same anchor ~0-3, front vs rear view ~10


def anchors_of(name):
    a, b = name.split("_to_")
    return a, b


def _is_ref(x):
    return isinstance(x, dict) and bool(x.get("bucket")) and bool(x.get("path"))


def plan(tt):
    """Validate params.video_gen.turntable and lay out the work. Pure (no I/O).

    Returns {variant, sequence, segments: [{name, start, end, seconds, position, ready}],
             requested, generate, reuse, anchors, complete, loops, seconds}.
    Raises ValueError with a message the platform can show verbatim."""
    if not isinstance(tt, dict):
        raise ValueError("turntable must be an object")
    has = {a: _is_ref(tt.get(a)) for a in ANCHORS}
    for a in ANCHORS:
        if tt.get(a) not in (None, "") and not has[a]:
            raise ValueError(f"turntable.{a} must be {{bucket, path}}")
    names = list(tt.get("segments") or []) + list((tt.get("ready_segments") or {}).keys()
                                                   if isinstance(tt.get("ready_segments"), dict) else [])
    four = has["left"] or has["right"] or any(n in SEQUENCES["four"] for n in names)
    if four and not (has["left"] and has["right"]) and (has["left"] or has["right"]):
        missing = "right" if has["left"] else "left"
        raise ValueError(f"four-anchor turntable needs both side photos: turntable.{missing} is missing "
                         "(left/right are the vehicle's own sides)")
    variant = "four" if four else "two"
    seq = SEQUENCES[variant]
    key = "seconds_per_quarter" if four else "seconds_per_half"
    seconds = tt.get(key)
    seconds = float(DEFAULTS[key] if seconds in (None, "") else seconds)
    lo, hi = SECONDS_LIMITS[variant]
    if not lo <= seconds <= hi:
        raise ValueError(f"turntable.{key} must be {lo}-{hi} (got {seconds:g}); 768p cannot exceed 10 s per segment")

    requested = tt.get("segments")
    if requested in (None, "", []):
        requested = list(seq)
    if not isinstance(requested, (list, tuple)) or not all(isinstance(s, str) for s in requested):
        raise ValueError("turntable.segments must be a list of segment names")
    bad = [s for s in requested if s not in seq]
    if bad:
        raise ValueError(f"unknown turntable segment(s) {bad} for the {variant}-anchor sequence; "
                         f"valid names in order: {list(seq)}")
    if len(set(requested)) != len(requested):
        raise ValueError("turntable.segments repeats a segment")
    idx = [seq.index(s) for s in requested]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError(f"turntable.segments must be consecutive in rotation order {list(seq)}, got {list(requested)}")

    ready = tt.get("ready_segments") or {}
    if not isinstance(ready, dict):
        raise ValueError("turntable.ready_segments must be an object {segment_name: {bucket, path}}")
    for name, ref in ready.items():
        if name not in seq:
            raise ValueError(f"ready_segments.{name}: unknown segment for the {variant}-anchor sequence {list(seq)}")
        if name not in requested:
            raise ValueError(f"ready_segments.{name} is not in the requested segments {list(requested)}")
        if not _is_ref(ref):
            raise ValueError(f"ready_segments.{name} must be {{bucket, path}} (a clip from a previous job's "
                             f"outputs/<id>-{name}.mp4 or an asset the platform owns)")

    segments = []
    for name in requested:
        a, b = anchors_of(name)
        segments.append({"name": name, "start": a, "end": b, "seconds": seconds, "position": seq.index(name) + 1,
                         "ready": ready.get(name)})
    generate = [s["name"] for s in segments if not s["ready"]]
    needed = sorted({x for s in segments if not s["ready"] for x in (s["start"], s["end"])}, key=ANCHORS.index)
    missing = [a for a in needed if not has[a]]
    if missing:
        raise ValueError(f"turntable needs the {', '.join(missing)} photo(s) to generate {generate}")
    if not (tt.get("car") or "").strip() and generate:
        raise ValueError("turntable.car (one line describing the car) is required to generate segments")
    complete = list(requested) == list(seq)
    return {"variant": variant, "sequence": list(seq), "segments": segments, "requested": list(requested),
            "generate": generate, "reuse": [s["name"] for s in segments if s["ready"]], "anchors": needed,
            "complete": complete, "loops": complete, "seconds": seconds}


def progress_ranges(n, lo=5, hi=85):
    """Equal, increasing (lo, hi) job-progress spans for n generations."""
    if n <= 0:
        return []
    w = (hi - lo) / n
    return [(int(round(lo + i * w)), int(round(lo + (i + 1) * w))) for i in range(n)]


def labels_for(pl):
    """Phase label per segment name, e.g. 'quarter 1/4 (front to left)' or 'turntable half 2 (rear to front)'."""
    out = {}
    for s in pl["segments"]:
        a, b = s["start"], s["end"]
        if pl["variant"] == "four":
            out[s["name"]] = f"quarter {s['position']}/4 ({a} to {b})"
        else:
            out[s["name"]] = f"turntable half {s['position']} ({a} to {b})"
    return out


def run(opts, api, log):
    """opts: the params.video_gen.turntable object plus
             anchor_local {front|left|rear|right: local photo path} for every anchor plan() lists,
             ready_local  {segment_name: local 24 fps mp4} for every ready_segments entry
       api:  work_dir; upload(local_path) -> anchor ref for submit(); submit(vg, label, prange, after_seconds,
             after_text) -> handle (runs an i2v generation, reporting progress inside prange); wait(handle, label)
             -> local mp4; phase(text, progress); check_cancel()
       returns {video, joined, segments: {name: {...}}, order, complete, loops, seams, loop_seam, defects,
                frames, fps, width, height, timing, plateau}"""
    pl = plan(opts)
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if v not in (None, "")}}
    work = api.work_dir
    os.makedirs(work, exist_ok=True)
    labels = labels_for(pl)
    timing = {}
    defects = []
    unit = "quarter" if pl["variant"] == "four" else "half"

    # Ready segments first: they fix the canvas when nothing is generated.
    ready_clips = {}
    for name in pl["reuse"]:
        path = (opts.get("ready_local") or {}).get(name)
        if not path or not os.path.exists(path):
            raise RuntimeError(f"ready segment {name}: local clip missing")
        inf = post.info(path)
        if abs(inf["fps"] - graphs.FPS) > 0.2:
            raise RuntimeError(f"ready segment {name} is {inf['fps']} fps; segments must be the native "
                               f"{graphs.FPS} fps clips (outputs/<id>-{name}.mp4), not the interpolated master")
        ready_clips[name] = (path, inf)
    if pl["generate"]:
        w, h = graphs.dims(o["resolution"], o["ratio"])
    else:
        first = next(iter(ready_clips.values()))[1]
        w, h = first["width"], first["height"]
    for name, (path, inf) in ready_clips.items():
        if (inf["width"], inf["height"]) != (w, h):
            raise RuntimeError(f"ready segment {name} is {inf['width']}x{inf['height']} but this job's canvas is "
                               f"{w}x{h}; all segments of one rotation must share one canvas")

    anchors = {}
    if pl["generate"]:
        api.phase("padding photos", 3)
        photos = {a: opts["anchor_local"][a] for a in pl["anchors"]}
        padded = post.pad_photos_common(photos, w, h, work)
        for a, info in padded.items():
            x0, y0, x1, y1 = info["box"]
            log(f"anchor {a}: subject {x0:.2f}-{x1:.2f} x, {y0:.2f}-{y1:.2f} y of the canvas (scale {info['scale']:.3f})")
            anchors[a] = dict(api.upload(info["local"]), local=info["local"])
        if pl["variant"] == "four" and {"front", "left"} <= set(padded):
            fw = padded["front"]["box"][2] - padded["front"]["box"][0]
            lw = padded["left"]["box"][2] - padded["left"]["box"][0]
            if lw < fw:
                defects.append(f"the left photo's subject ({lw:.2f} of the width) is narrower than the front "
                               f"photo's ({fw:.2f}); a side view of a car should be wider. Check the photo roles")

    def prompt(motion, details=None):
        return BASE_PROMPT.format(car=o.get("car", ""), motion=motion,
                                  details=o["details"] if details is None else details).replace("  ", " ").strip()

    ranges = dict(zip(pl["generate"], progress_ranges(len(pl["generate"]))))
    gen_left = {"n": len(pl["generate"])}

    def gen(first, last, text, seconds, label, seed_offset, seg_range, segment=None, piece=0):
        api.check_cancel()
        vg = {"mode": "i2v", "prompt": text, "duration_s": seconds, "resolution": o["resolution"], "ratio": o["ratio"],
              "seed": int(o["seed"]) + seed_offset, "first_frame": first, "last_frame": last}
        after_gens = max(0, gen_left["n"] - 1)
        after = after_gens * estimate.generation_seconds(vg, "i2v") + estimate.TURNTABLE_POST_S
        after_text = (f" - then {after_gens} {unit}{'s' if after_gens > 1 else ''} + join" if after_gens
                      else f" - then join + {int(o['fps'])} fps")
        t0 = time.monotonic()      # the worker's api runs the generation inside submit(); time both calls
        handle = api.submit(vg, label, prange=seg_range, after_seconds=after, after_text=after_text,
                            segment=segment, piece=piece)
        log(f"{label}: job {handle}")
        path = api.wait(handle, label)
        timing[label] = round(time.monotonic() - t0, 1)
        return path

    def gen_checked(seg, seg_range):
        """Generate a segment; reject and reseed when the scene drifts away from the flat backdrop
        (grey floor / overhead camera), which the cut detector cannot see because it is gradual."""
        name, a, b = seg["name"], seg["start"], seg["end"]
        base = labels[name]
        for attempt in range(3):
            label = base + (f" try {attempt + 1}" if attempt else "")
            path = gen(anchors[a], anchors[b], prompt(f"{SWEEP[name]}, from the {VIEW[a]} to the {VIEW[b]}"),
                       seg["seconds"], label, seed_offset=seg["position"] - 1 + 100 * attempt, seg_range=seg_range,
                       segment=name, piece=0)
            drifted, share, worst = post.background_drift(path)
            log(f"{label}: backdrop drift {share:.0%} of frames (worst {worst:.0f})" + (" -> regenerating" if drifted else ""))
            if not drifted:
                return path, attempt + 1
        raise RuntimeError(f"{base}: the scene kept drifting off the plain backdrop after 3 seeds")

    def repair(path, seg, seg_range, depth=0):
        """Return the clean clips that together cover the segment's intended quarter/half, in order."""
        m = post.motion_profile(path)
        cut, plateau = post.find_cut(m)
        if cut is None:
            log(f"{os.path.basename(path)}: clean ({len(m)} frames, plateau {plateau:.1f})")
            return [path]
        if depth >= 2:
            raise RuntimeError(f"{labels[seg['name']]} keeps cutting mid-turn (frame {cut}); try another seed")
        keep_end = max(cut - 4, 12)      # last clean frame; back off from the cut
        log(f"{os.path.basename(path)}: cut at frame {cut} (motion {m[cut]:.1f} vs plateau {plateau:.1f}), keeping 0..{keep_end}")
        anchor_png = post.extract_frame(path, keep_end, os.path.join(work, f"anchor_{seg['name']}_{depth}.png"))
        anchor = api.upload(anchor_png)
        kept = post.trim_copy(path, os.path.join(work, f"kept_{seg['name']}_{depth}.mp4"), 0, keep_end + 1)
        # Seconds left is roughly the unfinished share of the segment, never under 3 s.
        remaining = max(3.0, round(seg["seconds"] * (1 - keep_end / len(m)) + 1.0))
        piece = gen(anchor, anchors[seg["end"]], prompt(REPAIR_MOTION.format(target=VIEW[seg["end"]])), remaining,
                    f"{labels[seg['name']]} repair ({remaining:.0f}s)", seed_offset=10 + depth, seg_range=seg_range,
                    segment=seg["name"], piece=depth + 1)
        return [kept] + repair(piece, seg, seg_range, depth + 1)

    # ---- per segment: generate (or reuse), check, repair, concat to one native-rate clip
    seg_out = {}
    for seg in pl["segments"]:
        name = seg["name"]
        if seg["ready"]:
            path, inf = ready_clips[name]
            seg_out[name] = {"local": path, "frames": inf["frames"], "pieces": 1, "attempts": 0, "repairs": 0,
                             "reused": True, "seconds": inf["duration"]}
            log(f"{labels[name]}: reusing {os.path.basename(path)} ({inf['frames']} frames)")
            continue
        seg_range = ranges[name]
        raw, attempts = gen_checked(seg, seg_range)
        api.phase(f"{labels[name]}: checking for cuts", seg_range[1] - 1)
        clips = repair(raw, seg, seg_range)
        gen_left["n"] -= 1
        if len(clips) == 1:
            seg_clip = clips[0]
        else:
            pieces = []
            for i, c in enumerate(clips):
                n = post.info(c)["frames"]
                pieces.append((c, 0 if i == 0 else 1, n))     # drop the shared anchor frame between pieces
            seg_clip = post.concat_trimmed(pieces, os.path.join(work, f"seg_{name}_24.mp4"))
        inf = post.info(seg_clip)
        seg_out[name] = {"local": seg_clip, "frames": inf["frames"], "pieces": len(clips), "attempts": attempts,
                         "repairs": len(clips) - 1, "reused": False, "seconds": inf["duration"]}

    order = [s["name"] for s in pl["segments"]]

    # ---- transitions: the frame a seam shares must be the same image on both sides
    api.phase("checking segment transitions", 86)
    seams = []
    pairs = list(zip(order, order[1:]))
    if pl["loops"] and len(order) > 1:
        pairs.append((order[-1], order[0]))
    frames_png = {}

    def edge(name, which):
        key = (name, which)
        if key not in frames_png:
            dest = os.path.join(work, f"edge_{name}_{which}.png")
            src = seg_out[name]["local"]
            frames_png[key] = post.extract_frame(src, 0, dest) if which == "first" else post.last_frame_png(src, dest)
        return frames_png[key]

    for a, b in pairs:
        d = post.frame_diff(edge(a, "last"), edge(b, "first"))
        entry = {"between": [a, b], "frame_diff": round(d, 2), "loop_seam": (a, b) == (order[-1], order[0]) and pl["loops"]}
        seams.append(entry)
        log(f"seam {a} -> {b}: shared frame differs by {d:.1f}")
        if d > TRANSITION_MAX_DIFF:
            if seg_out[a]["reused"] or seg_out[b]["reused"]:
                raise RuntimeError(f"segments {a} and {b} do not meet: their shared frame differs by {d:.0f} "
                                   f"(limit {TRANSITION_MAX_DIFF:.0f}). They were made from different anchors, "
                                   "seeds or canvases; regenerate one of them")
            defects.append(f"seam {a} -> {b}: shared frame differs by {d:.0f}")

    # ---- join on the shared frames
    api.phase("joining quarters" if pl["variant"] == "four" else "joining pieces", 88)
    pieces = []
    for i, name in enumerate(order):
        n = seg_out[name]["frames"]
        a = 0 if i == 0 else 1                                   # drop the shared anchor frame
        b = n - 1 if (i == len(order) - 1 and pl["loops"]) else n   # drop the final frame only when it loops
        pieces.append((seg_out[name]["local"], a, b))
    joined = post.concat_trimmed(pieces, os.path.join(work, "joined_24fps.mp4"))
    # Motion across each seam of the joined clip, relative to the steady-state plateau.
    m = post.motion_profile(joined)
    plateau = post.plateau_of(m)
    at = 0
    for i, name in enumerate(order[:-1]):
        at += pieces[i][2] - pieces[i][1]
        jump = m[at] if at < len(m) else 0.0
        seams[i]["motion_jump"] = round(jump, 2)
        if plateau and jump > 2.5 * plateau:
            defects.append(f"seam {name} -> {order[i + 1]}: motion {jump:.1f} vs plateau {plateau:.1f} (visible pop)")
    if pl["loops"] and seams and seams[-1].get("loop_seam"):
        wrap = post.frame_diff(post.last_frame_png(joined, os.path.join(work, "edge_joined_last.png")),
                               post.extract_frame(joined, 0, os.path.join(work, "edge_joined_first.png")))
        seams[-1]["motion_jump"] = round(wrap, 2)
        if plateau and wrap > 2.5 * plateau:
            defects.append(f"loop seam: motion {wrap:.1f} vs plateau {plateau:.1f} (visible pop when looping)")

    # ---- constant-speed remap + RIFE, then encode (a resize to shorter_size, not added detail)
    api.phase("interpolating", 90)
    frames_dir = os.path.join(work, "frames")
    n_out, before, plateau2 = post.remap_constant_speed(joined, frames_dir, fps=int(o["fps"]), density=int(o["density"]),
                                                        tmp_root=work, log=log)
    api.phase("encoding", 93)
    tag = "turntable" if pl["complete"] else "segments_" + "-".join(order)
    out = os.path.join(work, f"{tag}_{o['shorter_size']}p{o['fps']}.mp4")
    audio_src = next((seg_out[n]["local"] for n in order if not seg_out[n]["reused"]), seg_out[order[0]]["local"])
    post.encode_frames(frames_dir, out, int(o["fps"]), o["shorter_size"], audio_src=audio_src, loop_audio=True)
    inf = post.info(out)
    return {"video": out, "joined": joined, "segments": seg_out, "order": order, "variant": pl["variant"],
            "complete": pl["complete"], "loops": pl["loops"], "seams": seams, "defects": defects,
            "frames": n_out, "fps": int(o["fps"]), "width": inf["width"], "height": inf["height"],
            "canvas": [w, h], "plateau": plateau2, "profile_before": before, "timing": timing}
