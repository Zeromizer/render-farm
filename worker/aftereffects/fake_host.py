"""A stand-in for After Effects so the pipeline, runner and tests run on a
machine without AE. It interprets the same job object the recipes get,
writes a clearly labelled stub "project" (a JSON model, not an .aep the real
AE could open), and renders the model to RGBA PNG frames (or a QuickTime
Animation .mov through ffmpeg) with real fades and slides, so every
downstream step - ProRes/VP9 encode, alpha statistics, frame counts,
contact sheet, bundle, manifest - is exercised for real.

Anything produced through this host carries host.kind == "fake" in its
manifest and the runner refuses it unless AE_FAKE_HOST=1 is set. It is a
test fixture, never a delivery path.
"""
import json
import os
import struct
import subprocess
import sys
import time
import zlib

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

import proc  # noqa: E402
from aftereffects.errors import AEError  # noqa: E402

STUB_HEADER = "AIMOTION FAKE AE PROJECT - test fixture, not an After Effects file\n"


def _png(width, height, rgba_rows):
    """Encode RGBA rows (bytes objects of width*4) as a PNG."""
    raw = b"".join(b"\x00" + r for r in rgba_rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")


def _hex(c):
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _layer_alpha(spec, t):
    if t < spec["in_s"] or t >= spec["out_s"]:
        return 0.0
    a = spec.get("opacity", 100) / 100.0
    fi, fo = spec.get("fade_in_s", 0), spec.get("fade_out_s", 0)
    if fi and t < spec["in_s"] + fi:
        a *= (t - spec["in_s"]) / fi
    if fo and t > spec["out_s"] - fo:
        a *= (spec["out_s"] - t) / fo
    return max(0.0, min(1.0, a))


def _layer_pos(spec, t):
    x, y = spec["position"]
    sf = spec.get("slide_from")
    if sf:
        d = spec.get("slide_s", 0.4) or 0.4
        k = min(1.0, max(0.0, (t - spec["in_s"]) / d))
        k = k * k * (3 - 2 * k)
        x += sf[0] * (1 - k)
        y += sf[1] * (1 - k)
    return x, y


def _v2_as_v1(settings):
    """Flatten a v2 settings object into the v1-shaped lists the box renderer
    draws (fades approximated from opacity keys, slides ignored)."""
    if "layers" not in settings:
        return settings
    texts, shapes, images = [], [], []
    for L in settings["layers"]:
        spec = {"id": L["id"], "position": L["position"], "in_s": L["in_s"], "out_s": L["out_s"],
                "opacity": L["opacity"], "scale": L["scale"][0], "fade_in_s": 0, "fade_out_s": 0}
        ok = L["keyframes"].get("opacity")
        if ok and len(ok) >= 2 and ok[0]["v"] == 0:
            spec["fade_in_s"] = ok[1]["t"] - ok[0]["t"]
        if ok and len(ok) >= 2 and ok[-1]["v"] == 0:
            spec["fade_out_s"] = ok[-1]["t"] - ok[-2]["t"]
        if L["kind"] == "text":
            spec.update(text=L["text"], size=L["size"] * L["stretch"][1], color=L["color"])
            texts.append(spec)
        elif L["kind"] == "shape":
            spec.update(size=L["size"], color=L["color"] or "#888888")
            shapes.append(spec)
        else:
            spec.update(asset=L["asset"])
            images.append(spec)
    return {"composition": settings["composition"], "texts": texts, "shapes": shapes, "images": images}


def _layer_boxes(settings):
    """Every layer as a (spec, w, h, color) box in z order bottom-up."""
    settings = _v2_as_v1(settings)
    boxes = []
    for im in settings["images"]:
        boxes.append((im, 200, 200, "#8888FF"))
    for sh in settings["shapes"]:
        boxes.append((sh, sh["size"][0], sh["size"][1], sh["color"]))
    for tx in settings["texts"]:
        boxes.append((tx, min(len(tx["text"]) * tx["size"] * 0.55, 4000), tx["size"] * 1.2, tx["color"]))
    return boxes


def render_frame(settings, t):
    W, H = settings["composition"]["width"], settings["composition"]["height"]
    rows = [bytearray(W * 4) for _ in range(H)]
    for spec, bw, bh, color in _layer_boxes(settings):
        a = _layer_alpha(spec, t)
        if a <= 0:
            continue
        s = spec.get("scale", 100) / 100.0
        sf = spec.get("scale_from")
        if sf is not None:
            d = spec.get("scale_s", 0.4) or 0.4
            k = min(1.0, max(0.0, (t - spec["in_s"]) / d))
            s = (sf + (spec.get("scale", 100) - sf) * k) / 100.0
        bw, bh = int(bw * s), int(bh * s)
        cx, cy = _layer_pos(spec, t)
        x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
        x1, y1 = min(W, x0 + bw), min(H, y0 + bh)
        x0, y0 = max(0, x0), max(0, y0)
        if x1 <= x0 or y1 <= y0:
            continue
        r, g, b = _hex(color)
        al = int(round(a * 255))
        pix = bytes((r, g, b, al)) * (x1 - x0)
        for y in range(y0, y1):
            rows[y][x0 * 4:x1 * 4] = pix
    return rows


class FakeHost:
    kind = "fake"

    def __init__(self, log=print, om_kind="sequence", fail_author=None, hang_stage=None,
                 no_manifest_stage=None, frame_delay_s=0.0, drop_frames=0, ffmpeg=None, version="fake-25.0.0"):
        self.log = log
        self.om_kind = om_kind
        self.om_ext = "png" if om_kind == "sequence" else "mov"
        self.om_template = "Lossless with Alpha"
        self.rs_template = "Best Settings"
        self.fail_author = fail_author          # (code, message, line) -> author writes an error manifest
        self.hang_stage = hang_stage            # "author" | "render": block until cancel/timeout
        self.no_manifest_stage = no_manifest_stage
        self.frame_delay_s = frame_delay_s
        self.drop_frames = drop_frames          # render fewer frames than expected (incomplete output)
        self.ffmpeg = ffmpeg
        self.version = version
        self.launched = []

    def info(self):
        return {"kind": self.kind, "version": self.version, "om_template": self.om_template,
                "om_kind": self.om_kind, "om_ext": self.om_ext, "rs_template": self.rs_template,
                "note": "FAKE HOST - test fixture, no After Effects involved"}

    def _hang(self, stage, timeout_s, cancel_check):
        started = time.monotonic()
        while True:
            time.sleep(0.05)
            if cancel_check():
                raise proc.Canceled()
            if time.monotonic() - started > timeout_s:
                raise proc.TimedOut(f"fake {stage} exceeded {timeout_s:.0f}s")

    def _write(self, path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)

    @staticmethod
    def _read_stub(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if not text.startswith(STUB_HEADER):
            raise AEError("VERIFY_FAILED", f"{path} is not a fake project stub")
        return json.loads(text[len(STUB_HEADER):])

    def _write_stub(self, path, model):
        with open(path, "w", encoding="utf-8") as f:
            f.write(STUB_HEADER + json.dumps(model, indent=1))

    @staticmethod
    def _comp_summary(model, name):
        s = model["settings"]
        c = s["composition"]
        if "layers" in s:
            return FakeHost._comp_summary_v2(model, name)
        layers = []
        idx = 1
        for tx in reversed(s["texts"]):
            layers.append({"index": idx, "name": tx["id"], "kind": "text", "text": tx["text"], "font": tx["font"],
                           "font_size": tx["size"], "in_s": tx["in_s"], "out_s": tx["out_s"], "enabled": True,
                           "position_keys": 2 if tx.get("slide_from") else 0,
                           "opacity_keys": (2 if tx["fade_in_s"] else 0) + (2 if tx["fade_out_s"] else 0)})
            idx += 1
        for sh in reversed(s["shapes"]):
            layers.append({"index": idx, "name": sh["id"], "kind": "shape", "in_s": sh["in_s"], "out_s": sh["out_s"],
                           "enabled": True, "position_keys": 2 if sh.get("slide_from") else 0,
                           "opacity_keys": (2 if sh["fade_in_s"] else 0) + (2 if sh["fade_out_s"] else 0)})
            idx += 1
        for im in reversed(s["images"]):
            layers.append({"index": idx, "name": im["id"], "kind": "footage", "source": model["assets"].get(im["asset"]),
                           "footage_missing": not os.path.exists(model["assets"].get(im["asset"], "")),
                           "in_s": im["in_s"], "out_s": im["out_s"], "enabled": True})
            idx += 1
        return {"name": name, "width": c["width"], "height": c["height"], "fps": c["fps"],
                "duration_s": c["duration_s"], "frames": c["frames"], "layers": layers}

    @staticmethod
    def _comp_summary_v2(model, name):
        s = model["settings"]
        c = s["composition"]
        out = []
        for idx, L in enumerate(reversed(s["layers"]), start=1):
            keys = {k: len(v) for k, v in L["keyframes"].items()}
            fx = [e["match_name"] for e in L["effects"]] + (["ADBE Exposure2"] if "brightness" in keys else [])
            d = {"index": idx, "name": L["id"], "kind": {"image": "footage"}[L["kind"]] if L["kind"] == "image" else L["kind"],
                 "in_s": L["in_s"], "out_s": L["out_s"], "enabled": True, "motion_blur": L["motion_blur"],
                 "effects": fx, "keys": keys, "masks": 0}
            if L["kind"] == "text":
                d.update(text=L["text"], font=L["font"], font_size=L["size"], stretch=list(L["stretch"]),
                         stroke={"width": L["stroke"]["width"]} if L.get("stroke") else None)
            if L["kind"] == "image":
                d["source"] = model["assets"].get(L["asset"])
                d["footage_missing"] = not os.path.exists(model["assets"].get(L["asset"], ""))
            if L.get("matte"):
                d["matte"] = {"layer": L["matte"]["layer"], "type": 5013}
            if L.get("clip"):
                inner = dict(d)
                d = {"index": idx, "name": L["id"], "kind": "precomp", "masks": 1, "inner": inner,
                     "in_s": L["in_s"], "out_s": L["out_s"], "enabled": True, "motion_blur": L["motion_blur"],
                     "effects": [], "keys": {}, "matte": d.get("matte")}
            out.append(d)
        mb = c.get("motion_blur") or {}
        return {"name": name, "width": c["width"], "height": c["height"], "fps": c["fps"], "duration_s": c["duration_s"],
                "frames": c["frames"], "layers": out, "motion_blur": bool(mb.get("enabled")),
                "shutter_angle": mb.get("shutter_angle")}

    def run_script(self, stage, job, jsx_path, manifest_path, timeout_s, cancel_check, pid_sink):
        self.launched.append((stage, jsx_path))
        pid_sink(os.getpid(), "fake-AfterFX.exe")
        if not job.get("token"):
            raise AEError("AE_NOT_ISOLATED", "fake host: job carries no AE_JOB_TOKEN")
        if self.hang_stage == stage:
            self._hang(stage, timeout_s, cancel_check)
        if self.no_manifest_stage == stage:
            raise AEError("AE_NO_MANIFEST", f"fake After Effects exited without writing {os.path.basename(manifest_path)}")
        t0 = time.time()
        m = {"ok": False, "stage": stage, "ae": {"app_version": self.version, "host": "fake"}, "warnings": []}
        try:
            if stage == "author":
                if self.fail_author:
                    code, msg, line = self.fail_author
                    raise _Jsx(code, msg, line)
                imgs = job["settings"].get("images") or [L for L in job["settings"].get("layers", []) if L["kind"] == "image"]
                for im in imgs:
                    p = job["assets"].get(im["asset"], "")
                    if not os.path.exists(p):
                        raise _Jsx("ASSET_MISSING", f"asset '{im['asset']}' not found at {p}", 121)
                model = {"settings": job["settings"], "assets": job["assets"], "comp": job["settings"]["composition"],
                         "name": job["composition"]}
                self._write_stub(job["project_path"], model)
                texts = job["settings"].get("texts") or [L for L in job["settings"].get("layers", []) if L["kind"] == "text"]
                m.update(ok=True, project_file=job["project_path"], om_templates=[self.om_template],
                         om_template=job["om_template"],
                         fonts={t["font"]: {"available": True, "method": "fake"} for t in texts},
                         comp=self._comp_summary(model, job["composition"]), easing_resolved={}, anchors_resolved={})
            elif stage == "inspect":
                model = self._read_stub(job["project_path"])
                m.update(ok=True, comp=self._comp_summary(model, job["composition"]), compositions=[model["name"]])
            elif stage == "revise":
                model = self._read_stub(job["source_project_path"])
                applied = []
                by_id = {}
                for group in ("texts", "shapes", "images"):
                    for spec in model["settings"][group]:
                        by_id[spec["id"]] = spec
                for ch in job["changes"]:
                    spec = by_id.get(ch["id"])
                    if spec is None:
                        raise _Jsx("VERIFY_FAILED", f"layer '{ch['id']}' not found in composition", 30)
                    a = {"id": ch["id"]}
                    if "text" in ch:
                        spec["text"] = ch["text"]
                        a["text"] = ch["text"]
                    if "in_s" in ch or "out_s" in ch:
                        spec["in_s"] = ch.get("in_s", spec["in_s"])
                        spec["out_s"] = ch.get("out_s", spec["out_s"])
                        a["in_s"], a["out_s"] = spec["in_s"], spec["out_s"]
                    applied.append(a)
                self._write_stub(job["project_path"], model)
                m.update(ok=True, applied=applied, project_file=job["project_path"],
                         comp=self._comp_summary(model, job["composition"]))
            else:
                raise _Jsx("JSX_ERROR", f"unknown stage {stage}", 1)
        except _Jsx as e:
            m["error"] = {"code": e.code, "message": e.msg, "line": e.line, "file": os.path.basename(jsx_path)}
        m["elapsed_ms"] = int((time.time() - t0) * 1000)
        self._write(manifest_path, m)
        return m

    def render(self, project_path, comp_name, render_dir, expected_frames, timeout_s, cancel_check,
               on_progress, pid_sink):
        pid_sink(os.getpid(), "fake-aerender.exe")
        if self.hang_stage == "render":
            self._hang("render", timeout_s, cancel_check)
        model = self._read_stub(project_path)
        s = model["settings"]
        for im in (s.get("images") or [L for L in s.get("layers", []) if L["kind"] == "image"]):
            if not os.path.exists(model["assets"].get(im["asset"], "")):
                raise AEError("RENDER_FAILED", f"aerender ERROR: footage missing for layer {im['id']}")
        c = s["composition"]
        os.makedirs(render_dir, exist_ok=True)
        n = c["frames"] - self.drop_frames
        started = time.monotonic()
        for i in range(n):
            if time.monotonic() - started > timeout_s:
                raise proc.TimedOut(f"fake render exceeded {timeout_s:.0f}s")
            if i % 10 == 0 and cancel_check():
                raise proc.Canceled()
            rows = render_frame(s, i / c["fps"])
            with open(os.path.join(render_dir, f"frame_{i:05d}.png"), "wb") as f:
                f.write(_png(c["width"], c["height"], rows))
            on_progress((i + 1) / c["frames"])
            if self.frame_delay_s:
                time.sleep(self.frame_delay_s)
        pattern = os.path.join(render_dir, "frame_*.png")
        info = {"om_template": self.om_template, "rs_template": self.rs_template, "channels": "RGB + Alpha",
                "alpha": "straight", "format": "PNG Sequence" if self.om_kind == "sequence" else "QuickTime"}
        if self.om_kind == "sequence":
            return pattern, info
        # QuickTime Animation RGBA like AE's "Lossless with Alpha" template.
        out = os.path.join(render_dir, "render.mov")
        cmd = [self.ffmpeg or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(c["fps"]),
               "-i", os.path.join(render_dir, "frame_%05d.png"), "-c:v", "qtrle", "-pix_fmt", "argb", out]
        subprocess.run(cmd, check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        for i in range(n):
            os.remove(os.path.join(render_dir, f"frame_{i:05d}.png"))
        return out, info


class _Jsx(Exception):
    def __init__(self, code, msg, line):
        super().__init__(msg)
        self.code, self.msg, self.line = code, msg, line
