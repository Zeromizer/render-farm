"""Car turntable flow: a seamless, constant-speed 360 of a product from two
photos (straight-on front and straight-on rear), for background removal.

Proven 2026-09-06 on the Proton e.MAS 7 (Videos/H3/car_emas7_360_smooth_1080p60.mp4):

1. Pad both photos to the generation canvas (no crop, letterboxed in the
   photo's own background colour) so each anchor frame IS the photo.
2. Half 1: image-to-video with first_frame = front, last_frame = rear.
   Half 2: first_frame = rear, last_frame = front. Both seams are then pixel
   exact by construction, and the clip loops.
   PROMPT RULE: never name what you do not want in detail. A version of the
   prompt that said "no overhead view, no top-down view, no grey studio floor"
   produced an aerial shot over a town (2026-09-06). The short positive
   wording below yielded clean halves 3 times out of 4; the drift check
   below catches the 4th.
3. Repair: the model occasionally 'teleports' the car mid-clip to reach the
   last frame (one frame with 4-6x the normal motion). When a half has such a
   cut, keep the clean part, take its last clean frame as a new anchor and
   generate the remainder from that frame to the target photo.
4. Join the pieces on their shared frames, then time-remap to constant
   angular speed (the model eases in/out at every anchor) and RIFE to the
   output fps.

A single 10 s prompt-only 360 is 3x cheaper but drifts in size and speed and
crops the car; this flow costs ~23 GPU-minutes at 768p on the 4080.
"""
import os

from videogen import graphs
from studio import post

BASE_PROMPT = (
    "{car} on a seamless plain light studio background, exactly as in the given frames. The car rotates in place "
    "about its own vertical axis at a perfectly constant, slow angular speed, no easing in or out, no pauses, no speed "
    "changes, {motion}. The car stays the same size and stays centred; the whole car including all wheels stays fully "
    "inside the frame with empty space on both sides. Fixed camera at eye level, no camera movement, no zoom. No "
    "turntable, no floor, no ground shadow, no reflection, no horizon, just the plain background. {details} Soft even "
    "studio lighting, no people, no text overlays. Quiet studio ambience."
)
MOTION = {
    "half1": ("turning exactly 180 degrees clockwise as seen from above so that its left side sweeps past the camera, "
              "from the straight-on front view to the straight-on rear view"),
    "half2": ("continuing the same clockwise turn as seen from above for exactly 180 degrees so that its right side "
              "sweeps past the camera, from the straight-on rear view back to the straight-on front view"),
    "repair": ("continuing the same turn from the exact view in the first frame until it reaches the straight-on "
               "{target} view shown in the last frame; the same side of the car stays visible throughout and it never "
               "turns the other way"),
}
DEFAULTS = {"resolution": "768p", "ratio": "16:9", "seconds_per_half": 10, "seed": 21, "fps": 60,
            "shorter_size": 1080, "density": 10, "details": ""}


def run(opts, api, log):
    """opts: front_local, rear_local, car (short description, e.g. 'A light teal metallic Proton e.MAS 7 SUV'),
             details (extra sentence: badge/plate text), resolution, ratio, seconds_per_half, seed, fps, shorter_size, density
       api:  work_dir; upload(local_path) -> an opaque anchor ref (the api decides: a bucket object for the
             farm queue, a ComfyUI input name for the worker); submit(vg, label) -> handle, where vg carries
             first_frame / last_frame as those refs; wait(handle, label) -> local mp4 (raises on fail/cancel);
             phase(text, progress); check_cancel()
       returns {video, pieces, profile_before, plateau, frames, fps}"""
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if v not in (None, "")}}
    w, h = graphs.dims(o["resolution"], o["ratio"])
    work = api.work_dir
    os.makedirs(work, exist_ok=True)

    api.phase("padding photos", 2)
    anchors = {}
    for name in ("front", "rear"):
        local = post.pad_photo(o[f"{name}_local"], os.path.join(work, f"{name}_{w}x{h}.png"), w, h)
        anchors[name] = dict(api.upload(local), local=local)

    def prompt(motion_key, **fmt):
        return BASE_PROMPT.format(car=o["car"], motion=MOTION[motion_key].format(**fmt), details=o["details"]).replace("  ", " ")

    def gen(first, last, text, seconds, label, seed_offset=0):
        api.check_cancel()
        vg = {"mode": "i2v", "prompt": text, "duration_s": seconds, "resolution": o["resolution"], "ratio": o["ratio"],
              "seed": int(o["seed"]) + seed_offset, "first_frame": first, "last_frame": last}
        jid = api.submit(vg, label)
        log(f"{label}: job {jid}")
        return api.wait(jid, label)

    spf = float(o["seconds_per_half"])

    def gen_checked(first, last, key, label, seed_offset):
        """Generate a half; reject and reseed when the scene drifts away from the flat backdrop
        (grey floor / overhead camera), which the cut detector cannot see because it is gradual."""
        for attempt in range(3):
            path = gen(first, last, prompt(key), spf, label + (f" try {attempt + 1}" if attempt else ""),
                       seed_offset=seed_offset + 100 * attempt)
            drifted, share, worst = post.background_drift(path)
            log(f"{label}: backdrop drift {share:.0%} of frames (worst {worst:.0f})" + (" -> regenerating" if drifted else ""))
            if not drifted:
                return path
        raise RuntimeError(f"{label}: the scene kept drifting off the plain backdrop after 3 seeds")

    half1 = gen_checked(anchors["front"], anchors["rear"], "half1", "half 1 (front to rear)", 0)
    half2 = gen_checked(anchors["rear"], anchors["front"], "half2", "half 2 (rear to front)", 1)

    def repair(path, target, target_name, depth=0):
        """Return the list of clean clips that together cover path's intended span."""
        m = post.motion_profile(path)
        cut, plateau = post.find_cut(m)
        if cut is None:
            log(f"{os.path.basename(path)}: clean ({len(m)} frames, plateau {plateau:.1f})")
            return [path]
        if depth >= 2:
            raise RuntimeError(f"{target_name} half keeps cutting mid-turn (frame {cut}); try another seed")
        keep_end = max(cut - 4, 12)      # last clean frame; back off from the cut
        log(f"{os.path.basename(path)}: cut at frame {cut} (motion {m[cut]:.1f} vs plateau {plateau:.1f}), keeping 0..{keep_end}")
        anchor_png = post.extract_frame(path, keep_end, os.path.join(work, f"anchor_{target_name}_{depth}.png"))
        anchor = api.upload(anchor_png)
        kept = post.trim_copy(path, os.path.join(work, f"kept_{target_name}_{depth}.mp4"), 0, keep_end + 1)
        # Seconds left is roughly the unfinished share of the half, never under 3 s.
        remaining = max(3.0, round(spf * (1 - keep_end / len(m)) + 1.0))
        piece = gen(anchor, anchors[target_name], prompt("repair", target=target_name), remaining,
                    f"repair {target_name} ({remaining:.0f}s)", seed_offset=10 + depth)
        return [kept] + repair(piece, target, target_name, depth + 1)

    api.phase("checking halves for cuts", 60)
    clips = repair(half1, anchors["rear"], "rear") + repair(half2, anchors["front"], "front")

    api.phase("joining pieces", 70)
    pieces = []
    for i, c in enumerate(clips):
        n = post.info(c)["frames"]
        a = 0 if i == 0 else 1                       # drop the shared anchor frame
        b = n - 1 if i == len(clips) - 1 else n      # drop the final front frame so it loops
        pieces.append((c, a, b))
    joined = post.concat_trimmed(pieces, os.path.join(work, "joined_24fps.mp4"))

    api.phase("remapping to constant speed + interpolating", 80)
    frames_dir = os.path.join(work, "frames")
    n_out, before, plateau = post.remap_constant_speed(joined, frames_dir, fps=int(o["fps"]), density=int(o["density"]),
                                                       tmp_root=work, log=log)
    api.phase("encoding", 92)
    out = os.path.join(work, f"turntable_{o['shorter_size']}p{o['fps']}.mp4")
    post.encode_frames(frames_dir, out, int(o["fps"]), o["shorter_size"], audio_src=half1, loop_audio=True)
    return {"video": out, "pieces": clips, "joined": joined, "profile_before": before, "plateau": plateau,
            "frames": n_out, "fps": int(o["fps"])}
