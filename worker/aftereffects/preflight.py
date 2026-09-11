"""Preflight report for the After Effects engine on this host.

  ..\\.venv\\Scripts\\python.exe -m aftereffects.preflight [--json]

Reports AE install (AfterFX.exe / aerender.exe / version), ffmpeg/ffprobe,
RAM/disk, the fonts the default smoke recipe needs, and the ProRes/VP9
encoders. Reads nothing but those; no environment dump, no credentials.
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from aftereffects import ae_host, fonts, media  # noqa: E402

DEFAULT_FONTS = ("ArialMT", "Arial-BoldMT")


def _memory():
    class M(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in ("total", "avail", "pt", "pa", "vt", "va", "ext")]
    m = M()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"total_gb": round(m.total / 1024 ** 3, 1), "available_gb": round(m.avail / 1024 ** 3, 1)}


def _encoders():
    try:
        out = subprocess.run([media.tool("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except Exception:  # noqa: BLE001
        return {}
    return {name: (f" {name} " in out) for name in ("prores_ks", "libvpx-vp9", "png", "qtrle")}


def report():
    ae = ae_host.find_after_effects()
    ff = media.ffmpeg_dir()
    disk = shutil.disk_usage(os.environ.get("LOCALAPPDATA", "C:\\"))
    r = {
        "after_effects": ae or {"installed": False,
                                "action": "Install Adobe After Effects from Creative Cloud on this Windows host, "
                                          "launch it once and sign in, then re-run this preflight. "
                                          "Set AE_DIR if it lives outside Program Files."},
        "ffmpeg": {"dir": ff, "version": media.version(), "ffprobe": bool(ff and os.path.exists(os.path.join(ff, "ffprobe.exe"))),
                   "encoders": _encoders()},
        "memory": _memory(),
        "disk_localappdata": {"free_gb": round(disk.free / 1024 ** 3, 1), "total_gb": round(disk.total / 1024 ** 3, 1)},
        "fonts": {f: (f not in fonts.missing([f])) for f in DEFAULT_FONTS},
        "font_dirs": fonts.font_dirs(),
        "om_template": os.environ.get("AE_OM_TEMPLATE") or ae_host.DEFAULT_OM_TEMPLATE,
        "om_kind": os.environ.get("AE_OM_KIND") or "mov",
        "fake_host": os.environ.get("AE_FAKE_HOST") == "1",
    }
    if ae:
        r["after_effects"]["installed"] = True
    r["ready"] = bool(ae) and bool(ff) and all(r["fonts"].values()) and r["ffmpeg"]["encoders"].get("prores_ks", False)
    return r


def main():
    r = report()
    if "--json" in sys.argv:
        print(json.dumps(r, indent=1))
        return
    ae = r["after_effects"]
    print("After Effects:", f"{ae['version']} at {ae['dir']}" if ae.get("installed") else "NOT INSTALLED")
    if not ae.get("installed"):
        print("  ->", ae["action"])
    print("ffmpeg:", r["ffmpeg"]["version"], "at", r["ffmpeg"]["dir"], "| encoders", r["ffmpeg"]["encoders"])
    print("memory:", r["memory"], "| disk:", r["disk_localappdata"])
    print("fonts:", r["fonts"])
    print("output module template:", r["om_template"], f"({r['om_kind']})")
    print("READY" if r["ready"] else "NOT READY")


if __name__ == "__main__":
    main()
