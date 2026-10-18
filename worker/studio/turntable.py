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

BRIDGED JOINS (joins="bridge", the default for a complete rotation, 2026-10-18).
Every anchored segment eases into its end photo and out of its start photo, so a
join on the shared frame stalls (the Noah four-anchor job: 0.2x speed for half a
second before each side). Step 6's remap hides that by evening out PIXEL motion,
which whips through the side views (they move fewer pixels per degree) and softens
the frames. Instead each segment loses its eased ends (BRIDGE_CUT) and H3 draws the
turn through the photo view in one move: a clip_edit bridge (Ref2VA inpaint, the
shared photo as a reference image) sized to the angle the cut frames covered. The
last bridge lands back on the first segment, so the rotation loops with no
shared-frame join at the model's own steady speed; it is RIFE'd to the output fps
without a remap. Every bridge take is checked for a seam jump, a motion jump and a
one-frame colour flash; a join with no clean take after a second window falls the
whole job back to the anchor joins of step 6 (a defect says so).

ORBIT (method="orbit", THE DEFAULT for a complete rotation made from photos, 2026-10-18). One H3
generation draws the whole turn: FL2VA with the 360 orbit LoRA (graphs.ORBIT_LORA), the front photo as
first AND last frame (so it loops), and every other view pinned at its angle's frame with core
MiniMaxH3AddGuide (90 = left, 180 = rear, 270 = right, plus optional extra `views` at any angle). No
joins, no remap: the speed is the model's own and even. Measured on the Toyota Noah (spike/h3-orbit):
  - 10 s (243 frames) keeps plates and badges legible all the way round; 5 s garbles them between pins.
  - every pin should come from ONE consistent source (same camera); mixing sources jerks at the seams.
    pad_photos_common keeps each photo's own framing (one scale, no per-view re-centring): placing every
    view on its own bounding box made the car bob ~60 px twice per turn.
  - turbo-8 stacked under the orbit LoRA matches 20 full steps (13 vs 28 min on the PC).
The clip is cut at the anchor frames into the usual segments (front_to_left ... or front_to_rear ...),
so the manifest, segment review, single-quarter regeneration and assembly keep working; a partial
request (`segments`) or any `ready_segments` uses the quarters method above. The master is RIFE'd from
the native 10 s to the rotation time the old method had (seconds_per_quarter x 4 / seconds_per_half x 2).
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
            "fps": 60, "shorter_size": 1080, "density": 10, "details": "", "joins": "bridge", "method": "orbit",
            "orbit_seconds": 10}
JOINS = ("bridge", "anchor")
METHODS = ("orbit", "quarters")
ANCHOR_ANGLE = {"front": 0, "left": 90, "rear": 180, "right": 270}
ORBIT_SECONDS = (5, 10)          # one generation; 768p cannot exceed 10 s, and under ~10 s lettering garbles
ORBIT_MAX_VIEWS = 23             # extra views (15 degrees apart at most), on top of the four anchors
ORBIT_PIN_MAX = 14.0             # frame_diff of a pinned frame vs its photo (clean pins measured 2-5)
ORBIT_TRIES = 2
# Settings per take, as measured on the Noah (E2E 2026-10-19). Turbo-8 at LoRA 1.0 with only the four photos
# pinned turned the backdrop dark grey between pins (both seeds); at LoRA 0.8 it stayed clean. With 8+ views
# LoRA 1.0 is clean. The second take runs 20 full steps (LoRA 1.0, clean with four photos, ~24 min on the PC).
ORBIT_FEW_VIEWS = 8
ORBIT_STRENGTH = {"few": 0.8, "many": 1.0}
ORBIT_RETRY = {"turbo": False, "steps": 20, "orbit_strength": 1.0}
SECONDS_LIMITS = {"two": (4, 10), "four": (3, 10)}
TRANSITION_MAX_DIFF = 6.0      # mean luma diff (256 px grey) of the frame a seam shares: same anchor ~0-3, front vs rear view ~10

# Bridged joins (module docstring). Measured on H3 FL2VA quarters (2026-10-18): the ease into the
# end photo spans ~30 frames, the one out of the start photo ~12.
BRIDGE_CUT = (12, 30)            # frames dropped at a segment's start / end
BRIDGE_MIN_KEEP = 24             # a segment must keep this much of itself between two bridges
BRIDGE_LENGTHS = (141, 124, 107, 90)   # H3 windows (17k+5), longest first: more context, steadier speed
BRIDGE_CONTEXT = (30, 44)        # original frames H3 sees on each side
BRIDGE_GEN = (12, 60)            # generated frames per bridge
BRIDGE_SEAM_MAX = 4.0            # clip_edit seam_jump of a usable take (clean 1.6-2.6; an overshoot that
                                 # snapped back 3 frames scored 4.8-5.1)
BRIDGE_JUMP_MAX = 2.5            # a frame's motion vs its local median
BRIDGE_FLASH_MAX = 1.5           # a frame's mean Y/U/V vs its neighbours' (H3 once flashed magenta for one frame)
BRIDGE_MOTION = ("continuing the same clockwise turn as seen from above at the same constant speed, passing "
                 "straight through the {view} without slowing, pausing or stopping")


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
    joins = tt.get("joins") or DEFAULTS["joins"]
    if joins not in JOINS:
        raise ValueError(f"turntable.joins must be one of {list(JOINS)} (got {joins!r})")
    method = tt.get("method") or DEFAULTS["method"]
    if method not in METHODS:
        raise ValueError(f"turntable.method must be one of {list(METHODS)} (got {method!r})")
    views = parse_views(tt.get("views"), {a for a in ANCHORS if has[a]})
    # The orbit is one generation of the whole turn: a partial request or reused segments need quarters.
    orbit = method == "orbit" and complete and not ready
    if views and not orbit:
        raise ValueError("turntable.views (extra angles) need the orbit method and a complete rotation "
                         "without ready_segments")
    orbit_seconds = tt.get("orbit_seconds")
    orbit_seconds = float(DEFAULTS["orbit_seconds"] if orbit_seconds in (None, "") else orbit_seconds)
    if not ORBIT_SECONDS[0] <= orbit_seconds <= ORBIT_SECONDS[1]:
        raise ValueError(f"turntable.orbit_seconds must be {ORBIT_SECONDS[0]}-{ORBIT_SECONDS[1]} (got {orbit_seconds:g})")
    return {"variant": variant, "sequence": list(seq), "segments": segments, "requested": list(requested),
            "generate": generate, "reuse": [s["name"] for s in segments if s["ready"]], "anchors": needed,
            "complete": complete, "loops": complete, "seconds": seconds,
            "method": "orbit" if orbit else "quarters", "views": views, "orbit_seconds": orbit_seconds,
            "bridge": joins == "bridge" and complete and not orbit}


def parse_views(views, anchors_given):
    """turntable.views: [{angle, image: {bucket, path}}] extra pinned angles for the orbit, in degrees of
    the same clockwise turn (front 0, left 90, rear 180, right 270). Returns [(angle, ref)] by angle."""
    if views in (None, "", []):
        return []
    if not isinstance(views, list) or not all(isinstance(v, dict) for v in views):
        raise ValueError("turntable.views must be a list of {angle, image: {bucket, path}}")
    if len(views) > ORBIT_MAX_VIEWS:
        raise ValueError(f"turntable.views: at most {ORBIT_MAX_VIEWS} extra angles (got {len(views)})")
    taken = {ANCHOR_ANGLE[a] for a in anchors_given}
    out = []
    for i, v in enumerate(views):
        try:
            angle = float(v.get("angle"))
        except (TypeError, ValueError):
            raise ValueError(f"turntable.views[{i}].angle must be a number of degrees")
        if not 0 < angle < 360:
            raise ValueError(f"turntable.views[{i}].angle must be between 0 and 360 exclusive (got {angle:g}); "
                             "0 is the front photo")
        if not _is_ref(v.get("image")):
            raise ValueError(f"turntable.views[{i}].image must be {{bucket, path}}")
        if any(abs(angle - t) < 5 for t in taken):
            raise ValueError(f"turntable.views[{i}]: angle {angle:g} is within 5 degrees of another view")
        taken.add(angle)
        out.append((angle, v["image"]))
    return sorted(out, key=lambda x: x[0])


def orbit_frame(angle, length):
    """Frame of a `length`-frame orbit (front at 0 and length-1) where the view at `angle` sits."""
    return int(round(angle / 360.0 * (length - 1)))


def orbit_cuts(variant, length):
    """[(segment name, first frame, last frame inclusive)] of an orbit clip: consecutive segments share
    their anchor frame, exactly like generated quarters/halves."""
    seq = SEQUENCES[variant]
    out = []
    for name in seq:
        a, b = anchors_of(name)
        lo = orbit_frame(ANCHOR_ANGLE[a], length)
        hi = length - 1 if b == "front" else orbit_frame(ANCHOR_ANGLE[b], length)
        out.append((name, lo, hi))
    return out


def bridge_gen_frames(m_a, m_b, cut=BRIDGE_CUT):
    """Frames a bridge generates: what the cut (eased) frames of the two segments covered,
    at the speed the segments run at just outside the cut. m_a / m_b: post.motion_profile
    of the segment before / after the join (value i = change from frame i-1 to i)."""
    cs, ce = cut
    n = len(m_a)
    covered = sum(m_a[n - ce:]) + sum(m_b[1:cs + 1])
    edge = list(m_a[max(0, n - ce - 8):n - ce]) + list(m_b[cs + 1:cs + 9])
    speed = sum(edge) / len(edge) if edge else 0.0
    lo, hi = BRIDGE_GEN
    if speed <= 0:
        return lo
    return max(lo, min(hi, int(round(covered / speed))))


def bridge_window(gen):
    """(length, context, gen) on H3's 17k+5 grid: the longest window whose context on each side
    stays within BRIDGE_CONTEXT. gen grows by a frame when that makes the two contexts equal."""
    for length in BRIDGE_LENGTHS:
        c = (length - gen) // 2
        if BRIDGE_CONTEXT[0] <= c <= BRIDGE_CONTEXT[1]:
            return length, c, length - 2 * c
    raise ValueError(f"no H3 window fits a {gen}-frame bridge")


def bridge_window_alt(length, c):
    """The second-round window: same length, 2 frames less context per side (4 more generated
    frames), or more context when that would generate too many. H3's glitches come with the
    geometry (2026-10-18: one window flashed on both seeds, the next one did not)."""
    c2 = c - 2 if c - 2 >= BRIDGE_CONTEXT[0] and length - 2 * (c - 2) <= BRIDGE_GEN[1] else c + 2
    return length, c2, length - 2 * c2


def bridge_layout(keeps, gens):
    """Frame pieces of the bridged loop, as [(bridge index, start, end)]. Bridge k joins segment k
    to segment k+1 (the last one to the first); its output is segment k's kept frames, gens[k] new
    frames, then segment k+1 from its cut start. Segment k comes half from bridge k-1 (its landing
    side) and half from bridge k (its leaving side), so both bridges' handoff blends are used."""
    n = len(keeps)
    out = []
    for k in range(n):
        h = keeps[k] // 2
        prev = (k - 1) % n
        b0 = keeps[prev] + gens[prev]
        out.append((prev, b0, b0 + h))
        out.append((k, h, keeps[k] + gens[k]))
    return out


def flash_frames(yuv, lo, hi, limit=BRIDGE_FLASH_MAX):
    """Frames in [lo, hi) whose mean Y, U or V is off the average of its two neighbours by more
    than limit: a one-frame colour flash. yuv: [(Y, U, V)] per frame (post.yuv_profile)."""
    bad = []
    for i in range(max(1, lo), min(len(yuv) - 1, hi)):
        if any(abs(yuv[i][c] - (yuv[i - 1][c] + yuv[i + 1][c]) / 2) > limit for c in range(3)):
            bad.append(i)
    return bad


def jump_frames(motion, lo, hi, limit=BRIDGE_JUMP_MAX, k=6):
    """Frames in [lo, hi) that move more than limit x the median of the 13 around them: a snap
    or skip (H3 FL2VA segments double one step in four, so the limit sits above 2)."""
    bad = []
    for i in range(max(1, lo), min(len(motion), hi)):
        win = sorted(motion[max(0, i - k):i + k + 1])
        med = win[len(win) // 2]
        if med > 0.05 and motion[i] > limit * med:
            bad.append(i)
    return bad


def pick_bridge_take(takes):
    """takes: [{take, seam_jump, flashes, jumps, ...}] -> the cleanest usable take, or None."""
    ok = [t for t in takes if not t["flashes"] and not t["jumps"] and t["seam_jump"] <= BRIDGE_SEAM_MAX]
    return min(ok, key=lambda t: t["seam_jump"]) if ok else None


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
             view_local   [(angle, local photo path)] for every plan()["views"] entry (orbit only),
             ready_local  {segment_name: local 24 fps mp4} for every ready_segments entry
       api:  work_dir; upload(local_path) -> anchor ref for submit(); submit(vg, label, prange, after_seconds,
             after_text) -> handle (runs an i2v generation, reporting progress inside prange); wait(handle, label)
             -> local mp4; orbit(spec) -> local mp4 (one pinned 360 generation, run_orbit);
             phase(text, progress); check_cancel(); bridge(spec) (bridged joins: one clip_edit
             bridge, see bridge_joins) -> {"report": compose report, "takes": [composed take mp4s]}
       returns {video, joined, segments: {name: {...}}, order, complete, loops, seams, loop_seam, defects,
                frames, fps, width, height, timing, plateau}"""
    pl = plan(opts)
    if pl["method"] == "orbit":
        return run_orbit(pl, opts, api, log)
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

    def gen(first, last, text, seconds, label, seed_offset, seg_range):
        api.check_cancel()
        vg = {"mode": "i2v", "prompt": text, "duration_s": seconds, "resolution": o["resolution"], "ratio": o["ratio"],
              "seed": int(o["seed"]) + seed_offset, "first_frame": first, "last_frame": last}
        after_gens = max(0, gen_left["n"] - 1)
        after = after_gens * estimate.generation_seconds(vg, "i2v") + estimate.TURNTABLE_POST_S
        after_text = (f" - then {after_gens} {unit}{'s' if after_gens > 1 else ''} + join" if after_gens
                      else f" - then join + {int(o['fps'])} fps")
        t0 = time.monotonic()      # the worker's api runs the generation inside submit(); time both calls
        handle = api.submit(vg, label, prange=seg_range, after_seconds=after, after_text=after_text)
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
                       seg["seconds"], label, seed_offset=seg["position"] - 1 + 100 * attempt, seg_range=seg_range)
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
                    f"{labels[seg['name']]} repair ({remaining:.0f}s)", seed_offset=10 + depth, seg_range=seg_range)
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

    bridged = None
    if pl["bridge"] and len(order) > 1:
        try:
            bridged = bridge_joins(order, seg_out, prompt, o, api, log)
        except (RuntimeError, OSError, ValueError) as exc:
            # a cancel is not one of these and still stops the job
            log(f"bridged joins failed, joining on the photos instead: {exc}")
            defects.append(f"bridged joins failed ({str(exc)[:160]}); joined on the photo frames and re-timed")
    if bridged:
        return finish_bridged(bridged, order, seg_out, seams, defects, pl, o, api, log, timing, (w, h))

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
            "canvas": [w, h], "plateau": plateau2, "profile_before": before, "timing": timing, "joins": "anchor"}


def bridge_joins(order, seg_out, prompt, o, api, log):
    """Bridge every join of a complete rotation (module docstring). Returns {pieces, outputs,
    bridges} for finish_bridged, or raises RuntimeError when a join has no usable take."""
    work = os.path.join(api.work_dir, "bridges")
    os.makedirs(work, exist_ok=True)
    cs, ce = BRIDGE_CUT
    n_seg = len(order)
    frames = [seg_out[name]["frames"] for name in order]
    keeps = [f - cs - ce for f in frames]
    if min(keeps) < BRIDGE_MIN_KEEP:
        raise RuntimeError(f"segments too short to bridge ({min(frames)} frames; each needs {BRIDGE_MIN_KEEP + cs + ce})")
    motion = {name: post.motion_profile(seg_out[name]["local"]) for name in order}
    outputs, gens, info = [], [], []
    span = (86 - 70) / n_seg
    for k, name in enumerate(order):
        nxt = order[(k + 1) % n_seg]
        anchor = anchors_of(name)[1]
        api.check_cancel()
        a_trim = post.trim_copy(seg_out[name]["local"], os.path.join(work, f"keep_{name}.mp4"), cs, frames[k] - ce)
        ref = post.extract_frame(seg_out[nxt]["local"], 0, os.path.join(work, f"ref_{anchor}.png"))
        gen = bridge_gen_frames(motion[name], motion[nxt])
        windows = [bridge_window(gen)]
        windows.append(bridge_window_alt(*windows[0][:2]))
        text = prompt(BRIDGE_MOTION.format(view=VIEW[anchor]))
        label = f"bridge {k + 1}/{n_seg} ({anchor})"
        chosen, tried = None, []
        for r, (length, c, g) in enumerate(windows):
            api.phase(f"{label}: {g} frames" + (" (second window)" if r else ""), int(70 + span * k))
            seeds = [int(o["seed"]) + 500 + 10 * k + 100 * r + 1009 * i for i in range(2)]
            got = api.bridge({"clip": a_trim, "clip_b": seg_out[nxt]["local"], "b_from_s": cs / float(graphs.FPS),
                              "seconds": g / float(graphs.FPS), "context_s": c / float(graphs.FPS), "prompt": text, "ref": ref,
                              "seeds": seeds, "label": label, "work_dir": os.path.join(work, f"{k + 1}_{r + 1}")})
            rep = got["report"]
            g_real = int(rep.get("generated_frames") or g)
            lo, hi = keeps[k] - 8, keeps[k] + g_real + 14
            takes = []
            for t, path in zip(rep["takes"], got["takes"]):
                yuv = post.yuv_profile(path)
                mo = post.motion_profile(path)
                takes.append({"take": t["take"], "seed": t["seed"], "seam_jump": t["seam_jump"], "path": path,
                              "flashes": flash_frames(yuv, lo, hi), "jumps": jump_frames(mo, lo, hi), "gen": g_real})
            log(f"{label} window {length}f context {c} gen {g_real}: " + "; ".join(
                f"take {t['take']} seam {t['seam_jump']:.2f} flashes {t['flashes'] or '-'} jumps {t['jumps'] or '-'}"
                for t in takes))
            tried += takes
            chosen = pick_bridge_take(takes)
            if chosen:
                break
        if not chosen:
            raise RuntimeError(f"{label}: no clean take in {len(tried)} (seam/flash/jump checks)")
        outputs.append(chosen["path"])
        gens.append(chosen["gen"])
        info.append({"between": [name, nxt], "through": anchor, "generated_frames": chosen["gen"],
                     "seam_jump": round(chosen["seam_jump"], 2), "take": chosen["take"], "seed": chosen["seed"],
                     "takes_tried": len(tried)})
    return {"pieces": [(outputs[b], s, e) for b, s, e in bridge_layout(keeps, gens)], "bridges": info,
            "keeps": keeps, "gens": gens}

def finish_bridged(bridged, order, seg_out, seams, defects, pl, o, api, log, timing, canvas):
    """Assemble the bridged loop and RIFE it to the output fps (no remap: the speed is the model's own)."""
    work = api.work_dir
    api.phase("joining bridged rotation", 87)
    joined = post.concat_trimmed(bridged["pieces"], os.path.join(work, "joined_24fps.mp4"))
    m = post.motion_profile(joined)
    plateau = post.plateau_of(m)
    wrap = post.frame_diff(post.last_frame_png(joined, os.path.join(work, "edge_joined_last.png")),
                           post.extract_frame(joined, 0, os.path.join(work, "edge_joined_first.png")))
    log(f"bridged loop: {len(m) + 1} frames, plateau {plateau:.2f}, wrap {wrap:.2f}; "
        + ", ".join(f"{b['through']} {b['generated_frames']}f seam {b['seam_jump']}" for b in bridged["bridges"]))
    if plateau and wrap > 2.5 * plateau:
        defects.append(f"loop seam: motion {wrap:.1f} vs plateau {plateau:.1f} (visible pop when looping)")
    api.phase("interpolating", 90)
    frames_dir = os.path.join(work, "frames")
    n_out = int(round((len(m) + 1) * int(o["fps"]) / float(graphs.FPS)))
    # the model's own pacing, with H3's one-in-four long step spaced evenly (post.judder_weights)
    post.retime_frames(joined, frames_dir, post.dejudder_times(m, n_out), density=int(o["density"]), tmp_root=work,
                       log=log)
    api.phase("encoding", 93)
    out = os.path.join(work, f"turntable_{o['shorter_size']}p{o['fps']}.mp4")
    audio_src = next((seg_out[n]["local"] for n in order if not seg_out[n]["reused"]), seg_out[order[0]]["local"])
    post.encode_frames(frames_dir, out, int(o["fps"]), o["shorter_size"], audio_src=audio_src, loop_audio=True)
    inf = post.info(out)
    for s in seams:
        s["bridged"] = True
    return {"video": out, "joined": joined, "segments": seg_out, "order": order, "variant": pl["variant"],
            "complete": pl["complete"], "loops": pl["loops"], "seams": seams, "defects": defects,
            "frames": inf["frames"], "fps": int(o["fps"]), "width": inf["width"], "height": inf["height"],
            "canvas": list(canvas), "plateau": plateau, "profile_before": m, "timing": timing, "joins": "bridge",
            "bridges": bridged["bridges"]}


def run_orbit(pl, opts, api, log):
    """The orbit method (module docstring): one pinned 360 generation, checked, cut into the usual
    segments and RIFE'd to the old rotation time. Returns the same dict as run()."""
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if v not in (None, "")}}
    work = api.work_dir
    os.makedirs(work, exist_ok=True)
    timing, defects = {}, []
    w, h = graphs.dims(o["resolution"], o["ratio"])
    length = graphs.frames_for(pl["orbit_seconds"])
    rotation_s = pl["seconds"] * len(pl["sequence"])

    api.phase("padding photos", 3)
    photos = {a: opts["anchor_local"][a] for a in pl["anchors"]}
    angles = {a: float(ANCHOR_ANGLE[a]) for a in pl["anchors"]}
    view_local = dict(opts.get("view_local") or [])
    for angle, _ in pl["views"]:
        key = f"view_{angle:g}"
        photos[key], angles[key] = view_local[angle], angle
    padded = post.pad_photos_common(photos, w, h, work)
    pins, refs = [], {}
    for name in sorted(padded, key=lambda n: angles[n]):
        info = padded[name]
        x0, y0, x1, y1 = info["box"]
        idx = orbit_frame(angles[name], length)
        log(f"view {name} ({angles[name]:g} deg -> frame {idx}): subject {x0:.2f}-{x1:.2f} x, {y0:.2f}-{y1:.2f} y")
        refs[name] = (idx, info["local"])
        if name != "front":
            pins.append((api.upload(info["local"])["name"], idx))
    if len({i for _, i in pins}) != len(pins):
        raise RuntimeError("two views land on the same frame of the orbit; space the extra angles further apart")
    front = api.upload(padded["front"]["local"])["name"]
    if pl["variant"] == "four" and {"front", "left"} <= set(padded):
        fw = padded["front"]["box"][2] - padded["front"]["box"][0]
        lw = padded["left"]["box"][2] - padded["left"]["box"][0]
        if lw < fw:
            defects.append(f"the left photo's subject ({lw:.2f} of the width) is narrower than the front "
                           f"photo's ({fw:.2f}); a side view of a car should be wider. Check the photo roles")

    def check(path):
        """(problems, view diffs, jumps) of one take: drift off the backdrop, a pinned view the model
        did not land on, a one-frame colour flash; motion jumps are reported, not retried."""
        drifted, share, _ = post.background_drift(path)
        diffs = {}
        for name, (idx, ref) in refs.items():
            fr = post.extract_frame(path, idx, os.path.join(work, f"pin_{name}.png"))
            diffs[name] = round(post.frame_diff(ref, fr), 2)
        last = post.last_frame_png(path, os.path.join(work, "pin_last.png"))
        diffs["front_end"] = round(post.frame_diff(refs["front"][1], last), 2)
        n = post.info(path)["frames"]
        flashes = flash_frames(post.yuv_profile(path), 1, n - 1)
        jumps = jump_frames(post.motion_profile(path), 1, n - 1)
        problems = []
        if drifted:
            problems.append(f"backdrop drift in {share:.0%} of frames")
        bad = {k: v for k, v in diffs.items() if v > ORBIT_PIN_MAX}
        if bad:
            problems.append("missed views " + ", ".join(f"{k} {v:.0f}" for k, v in bad.items()))
        if flashes:
            problems.append(f"colour flash at frames {flashes[:6]}")
        return problems, diffs, jumps

    takes = []
    for attempt in range(ORBIT_TRIES):
        api.check_cancel()
        label = "orbit 360" + (f" try {attempt + 1}" if attempt else "")
        seed = int(o["seed"]) + 300 * attempt
        if attempt == 0:
            strength = ORBIT_STRENGTH["many" if len(refs) >= ORBIT_FEW_VIEWS else "few"]
            settings = {"turbo": o.get("turbo", True), "steps": o.get("steps"),
                        "orbit_strength": float(o.get("orbit_strength") or strength)}
        else:
            settings = dict(ORBIT_RETRY)
            label += " (full steps)"
        spec = {"resolution": o["resolution"], "ratio": o["ratio"], "length": length, "seed": seed, **settings,
                "front": front, "pins": pins,
                "label": label, "prange": (8, 80) if attempt == 0 else (80, 86),
                "after_seconds": estimate.TURNTABLE_POST_S, "after_text": f" - then {int(o['fps'])} fps"}
        t0 = time.monotonic()
        path = api.orbit(spec)
        timing[label] = round(time.monotonic() - t0, 1)
        api.phase(f"{label}: checking views", 81)
        problems, diffs, jumps = check(path)
        log(f"{label}: seed {seed}, {settings}, view diffs {diffs}, jumps {jumps or '-'}"
            + (f" -> {'; '.join(problems)}" if problems else " -> clean"))
        takes.append({"path": path, "seed": seed, "problems": problems, "diffs": diffs, "jumps": jumps,
                      "worst": max(diffs.values()), "settings": settings})
        if not problems:
            break
    best = min(takes, key=lambda t: (len(t["problems"]), t["worst"]))
    if best["problems"]:
        defects.append(f"orbit: best of {len(takes)} takes still has: {'; '.join(best['problems'])}")
    if best["jumps"]:
        defects.append(f"orbit: motion jumps at frames {best['jumps'][:8]}")
    raw = best["path"]

    # ---- the usual segments, cut at the anchor frames (they share their anchor frame, like quarters)
    api.phase("cutting segments", 87)
    seg_out = {}
    order = list(pl["sequence"])
    for name, lo, hi in orbit_cuts(pl["variant"], length):
        n = hi - lo + 1
        clip = post.trim_copy(raw, os.path.join(work, f"seg_{name}_24.mp4"), lo, hi + 1)
        seg_out[name] = {"local": clip, "frames": n, "pieces": 1, "attempts": len(takes), "repairs": 0,
                         "reused": False, "seconds": round(n / float(graphs.FPS), 3)}
    seams = [{"between": [a, b], "frame_diff": 0.0, "loop_seam": (a, b) == (order[-1], order[0]), "orbit": True}
             for a, b in list(zip(order, order[1:])) + [(order[-1], order[0])]]
    joined = post.trim_copy(raw, os.path.join(work, "joined_24fps.mp4"), 0, length - 1)   # last frame = first

    # ---- RIFE to the rotation time the quarters method had, keeping the model's own pacing
    api.phase("interpolating", 90)
    m = post.motion_profile(raw)
    plateau = post.plateau_of(m)
    n_out = int(round(rotation_s * int(o["fps"])))
    frames_dir = os.path.join(work, "frames")
    # endpoint excluded: the loop wraps onto frame 0, one step after the last output frame
    post.retime_frames(raw, frames_dir, post.dejudder_times(m, n_out + 1)[:-1], density=int(o["density"]),
                       tmp_root=work, log=log)
    api.phase("encoding", 93)
    out = os.path.join(work, f"turntable_{o['shorter_size']}p{o['fps']}.mp4")
    post.encode_frames(frames_dir, out, int(o["fps"]), o["shorter_size"])
    inf = post.info(out)
    return {"video": out, "joined": joined, "segments": seg_out, "order": order, "variant": pl["variant"],
            "complete": True, "loops": True, "seams": seams, "defects": defects, "frames": inf["frames"],
            "fps": int(o["fps"]), "width": inf["width"], "height": inf["height"], "canvas": [w, h],
            "plateau": plateau, "profile_before": m, "timing": timing, "joins": "orbit", "method": "orbit",
            "orbit": {"length": length, "rotation_s": rotation_s, "seed": best["seed"], "takes": len(takes),
                      "settings": best["settings"],
                      "pins": {n: i for n, (i, _) in refs.items()}, "view_diffs": best["diffs"]}}
