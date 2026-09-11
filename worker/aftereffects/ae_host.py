"""The native Windows After Effects host: locate the install, run a trusted
recipe in an isolated AE instance, render a saved project with aerender, and
serialize AE work to a single slot.

Scripts run as   AfterFX.exe -m -noui -r <job.jsx>
  -m      a NEW instance, so the script never lands in an operator's open
          project (the recipe's AE.assertIsolated() double-checks)
  -noui   no panels
  -r      run the script; the recipe writes a manifest and calls app.quit()
AfterFX.exe has no useful exit code or stdout: completion is the manifest
file, and its absence after exit is an explicit AE_NO_MANIFEST failure.

Renders run as   aerender.exe -project X.aep -comp Main -RStemplate ...
                 -OMtemplate ... -output <pattern>
with stdout streamed for PROGRESS lines. No -continueOnMissingFootage: a
missing asset must fail, never render as a placeholder.

Every process this module launches is recorded through pid_sink so a retry
can kill exactly the stale attempt's AfterFX/aerender and nothing else.
"""
import ctypes
import glob
import os
import re
import subprocess
import sys
import time

_WORKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

import proc  # noqa: E402  (worker/proc.py: Canceled, TimedOut, _kill_tree)
from aftereffects.errors import AEError  # noqa: E402

_PROGRESS_RE = re.compile(r"PROGRESS:\s+\d+:\d+:\d+:\d+\s+\((\d+)\)")
_ERROR_RE = re.compile(r"aerender (?:ERROR|Error)|After Effects error", re.IGNORECASE)
_TOTAL_RE = re.compile(r"PROGRESS:\s+Duration:\s+(\d+):(\d+):(\d+):(\d+)")

DEFAULT_RS_TEMPLATE = "Best Settings"
DEFAULT_OM_TEMPLATE = "Lossless with Alpha"


def find_after_effects():
    """Return {dir, afterfx, aerender, version} for the newest install, or None.

    AE_DIR in the environment wins (the folder holding AfterFX.exe)."""
    candidates = []
    env_dir = os.environ.get("AE_DIR")
    if env_dir:
        candidates.append(env_dir)
    for root in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        candidates += sorted(glob.glob(os.path.join(root, "Adobe", "Adobe After Effects*", "Support Files")), reverse=True)
    for d in candidates:
        afterfx = os.path.join(d, "AfterFX.exe")
        aerender = os.path.join(d, "aerender.exe")
        if os.path.exists(afterfx) and os.path.exists(aerender):
            return {"dir": d, "afterfx": afterfx, "aerender": aerender,
                    "version": file_version(afterfx) or "unknown"}
    return None


def file_version(path):
    """FileVersion string of a Windows executable via the version resource."""
    try:
        size = ctypes.windll.version.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ctypes.windll.version.GetFileVersionInfoW(path, 0, size, buf):
            return None
        ptr = ctypes.c_void_p()
        length = ctypes.c_uint()
        if not ctypes.windll.version.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
            return None
        # VS_FIXEDFILEINFO: dwFileVersionMS at +8, dwFileVersionLS at +12
        ms = ctypes.c_uint.from_address(ptr.value + 8).value
        ls = ctypes.c_uint.from_address(ptr.value + 12).value
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    except Exception:  # noqa: BLE001
        return None


class Slot:
    """One AE job at a time on this machine, across worker processes.

    A named mutex, so a second worker lane (or a studio tool) that also drives
    AE waits instead of racing on the single AE licence and the RAM budget."""
    NAME = "Global\\render-farm-aftereffects-slot"

    def __init__(self, wait_seconds=600, cancel_check=None):
        self.wait_seconds = wait_seconds
        self.cancel_check = cancel_check or (lambda: False)
        self._h = None

    def __enter__(self):
        k32 = ctypes.windll.kernel32
        self._h = k32.CreateMutexW(None, False, self.NAME)
        if not self._h:
            raise AEError("AE_SLOT_TIMEOUT", "could not create the AE slot mutex")
        deadline = time.monotonic() + self.wait_seconds
        while True:
            rc = k32.WaitForSingleObject(self._h, 1000)
            if rc in (0, 0x80):   # WAIT_OBJECT_0 / WAIT_ABANDONED (holder died: fine)
                return self
            if self.cancel_check():
                raise proc.Canceled()
            if time.monotonic() > deadline:
                raise AEError("AE_SLOT_TIMEOUT", f"another After Effects job held the slot for over {self.wait_seconds}s")

    def __exit__(self, *exc):
        k32 = ctypes.windll.kernel32
        if self._h:
            k32.ReleaseMutex(self._h)
            k32.CloseHandle(self._h)
            self._h = None


def _read_manifest(path):
    import json
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


class RealHost:
    kind = "aftereffects"

    def __init__(self, install=None, log=print, om_template=None, om_kind=None, om_ext=None,
                 rs_template=None):
        self.install = install or find_after_effects()
        if not self.install:
            raise AEError("AE_NOT_INSTALLED", "After Effects (AfterFX.exe + aerender.exe) not found on this host")
        self.log = log
        self.om_template = om_template or os.environ.get("AE_OM_TEMPLATE") or DEFAULT_OM_TEMPLATE
        # What the template writes: a QuickTime "mov" or an image "sequence".
        self.om_kind = om_kind or os.environ.get("AE_OM_KIND") or "mov"
        self.om_ext = om_ext or os.environ.get("AE_OM_EXT") or ("mov" if self.om_kind == "mov" else "png")
        self.rs_template = rs_template or os.environ.get("AE_RS_TEMPLATE") or DEFAULT_RS_TEMPLATE
        if self.om_kind not in ("mov", "sequence"):
            raise AEError("INVALID_REQUEST", f"AE_OM_KIND must be mov|sequence, got {self.om_kind!r}")

    def info(self):
        return {"kind": self.kind, "dir": self.install["dir"], "version": self.install["version"],
                "om_template": self.om_template, "om_kind": self.om_kind, "om_ext": self.om_ext,
                "rs_template": self.rs_template}

    # ---------------------------------------------------------------- scripts
    def run_script(self, stage, job, jsx_path, manifest_path, timeout_s, cancel_check, pid_sink):
        """Launch an isolated AE instance on jsx_path and wait for its manifest."""
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        cmd = [self.install["afterfx"], "-m", "-noui", "-r", jsx_path]
        env = dict(os.environ)
        env["AE_JOB_TOKEN"] = str(job.get("token") or "")
        self.log(f"afterfx ({stage}): {' '.join(cmd)}")
        p = subprocess.Popen(cmd, cwd=os.path.dirname(jsx_path), stdout=subprocess.DEVNULL, env=env,
                             stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        pid_sink(p.pid, "AfterFX.exe")
        started = time.monotonic()
        last_cancel = started
        grace_started = None
        try:
            while True:
                exited = p.poll() is not None
                have = os.path.exists(manifest_path)
                if have and exited:
                    break
                if have and grace_started is None:
                    grace_started = time.monotonic()   # manifest first, app.quit() next
                if have and time.monotonic() - grace_started > 90:
                    self.log("afterfx wrote its manifest but did not quit within 90 s; killing it")
                    proc._kill_tree(p)
                    break
                if exited and not have:
                    # AE sometimes flushes the file a moment after the process is gone.
                    time.sleep(2)
                    if os.path.exists(manifest_path):
                        break
                    raise AEError("AE_NO_MANIFEST",
                                  f"After Effects exited (code {p.returncode}) without writing {os.path.basename(manifest_path)}; "
                                  "check that 'Allow Scripts to Write Files and Access Network' is enabled and that AE opens for this user")
                now = time.monotonic()
                if now - started > timeout_s:
                    proc._kill_tree(p)
                    raise proc.TimedOut(f"afterfx {stage} exceeded {timeout_s:.0f}s")
                if now - last_cancel >= 5:
                    last_cancel = now
                    if cancel_check():
                        proc._kill_tree(p)
                        raise proc.Canceled()
                time.sleep(1)
        finally:
            if p.poll() is None:
                proc._kill_tree(p)
        return _read_manifest(manifest_path)

    # ---------------------------------------------------------------- render
    def render(self, project_path, comp_name, render_dir, expected_frames, timeout_s, cancel_check,
               on_progress, pid_sink):
        """aerender the saved comp. Returns (output, info): output is the file AE
        wrote (mov/avi - AE picks the extension the template's format needs,
        whatever -output said) or a glob for a sequence; info carries what the
        aerender log reported (format, channels, alpha mode, output path)."""
        os.makedirs(render_dir, exist_ok=True)
        if self.om_kind == "mov":
            output = os.path.join(render_dir, f"render.{self.om_ext}")
        else:
            output = os.path.join(render_dir, f"frame_[#####].{self.om_ext}")
        cmd = [self.install["aerender"], "-project", project_path, "-comp", comp_name,
               "-RStemplate", self.rs_template, "-OMtemplate", self.om_template,
               "-output", output, "-sound", "OFF", "-close", "DO_NOT_SAVE_CHANGES"]
        self.log(f"aerender: {' '.join(cmd)}")
        errors = []
        info = {"om_template": self.om_template, "rs_template": self.rs_template}

        def on_line(line):
            m = _PROGRESS_RE.search(line)
            if m and expected_frames:
                on_progress(min(1.0, (int(m.group(1)) + 1) / expected_frames))
            elif _ERROR_RE.search(line):
                errors.append(line.strip())
            if not m:
                self.log(f"  {line[:300]}")
            for key, field in (("Output To:", "output_to"), ("Format:", "format"), ("Channels:", "channels"),
                               ("Color:", "alpha"), ("Depth:", "depth"), ("Final Size:", "final_size")):
                if key in line:
                    info[field] = line.split(key, 1)[1].strip()

        _run_streaming_with_pid(cmd, render_dir, on_line, cancel_check, timeout_s, pid_sink, errors)
        raw_alpha = (info.get("alpha") or "").lower()
        info["alpha"] = "premultiplied" if "premult" in raw_alpha else ("straight" if "straight" in raw_alpha else info.get("alpha"))
        if "alpha" not in (info.get("channels") or "").lower():
            raise AEError("ALPHA_MISSING", f"output module template '{self.om_template}' rendered channels "
                          f"'{info.get('channels')}', not RGB + Alpha")
        if self.om_kind == "mov":
            written = sorted(f for f in os.listdir(render_dir)
                             if f.startswith("render.") and os.path.isfile(os.path.join(render_dir, f)))
            if len(written) != 1:
                raise AEError("RENDER_FAILED", f"aerender left {len(written)} output file(s) in the render dir: {written}")
            return os.path.join(render_dir, written[0]), info
        return os.path.join(render_dir, f"frame_*.{self.om_ext}"), info


def _run_streaming_with_pid(cmd, cwd, on_line, cancel_check, timeout_seconds, pid_sink, errors):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
    pid_sink(p.pid, os.path.basename(cmd[0]))
    tail = []
    started = time.monotonic()
    last_cancel = started
    try:
        for line in p.stdout:
            line = line.rstrip("\r\n")
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
            on_line(line)
            now = time.monotonic()
            if now - started > timeout_seconds:
                proc._kill_tree(p)
                raise proc.TimedOut(f"exceeded {timeout_seconds:.0f}s")
            if now - last_cancel >= 5:
                last_cancel = now
                if cancel_check():
                    proc._kill_tree(p)
                    raise proc.Canceled()
        rc = p.wait(timeout=60)
    finally:
        if p.poll() is None:
            proc._kill_tree(p)
    if rc != 0 or errors:
        raise AEError("RENDER_FAILED", f"aerender exit code {rc}: " + " | ".join((errors or tail[-8:])[:8]),
                      {"exit_code": rc, "tail": tail[-15:]})


def kill_recorded(pids, log):
    """Kill exactly the AfterFX/aerender processes a stale attempt recorded.

    Pids are recycled, so the image name is checked before anything is killed."""
    for pid, image in pids:
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout
        except OSError:
            continue
        if image.lower() in out.lower():
            log(f"killing stale {image} pid {pid} from a previous attempt")
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(int(pid))], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
