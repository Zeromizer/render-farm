"""Installed-font check by PostScript name, read straight out of the TrueType /
OpenType `name` tables in the Windows font folders (no fontTools).

After Effects addresses fonts by PostScript name (ArialMT, Arial-BoldMT,
SegoeUI). The recipe re-checks inside AE, which is authoritative; this
preflight makes a missing font fail before AE is even launched.
"""
import os
import struct

_cache = None


def font_dirs():
    dirs = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    extra = os.environ.get("AE_EXTRA_FONT_DIRS")
    if extra:
        dirs += [d for d in extra.split(os.pathsep) if d]
    return [d for d in dirs if os.path.isdir(d)]


def _name_records(data, base):
    """PostScript names (name ID 6) of one sfnt starting at `base`."""
    out = set()
    if base + 12 > len(data):
        return out
    num_tables = struct.unpack_from(">H", data, base + 4)[0]
    for i in range(num_tables):
        rec = base + 12 + 16 * i
        if rec + 16 > len(data):
            break
        tag, _, off, length = struct.unpack_from(">4sIII", data, rec)
        if tag != b"name":
            continue
        if off + 6 > len(data):
            break
        _, count, str_off = struct.unpack_from(">HHH", data, off)
        for j in range(count):
            r = off + 6 + 12 * j
            if r + 12 > len(data):
                break
            plat, enc, _lang, nid, slen, soff = struct.unpack_from(">HHHHHH", data, r)
            if nid != 6:
                continue
            s = off + str_off + soff
            raw = data[s:s + slen]
            try:
                if plat in (0, 3):
                    out.add(raw.decode("utf-16-be"))
                else:
                    out.add(raw.decode("mac-roman", errors="replace"))
            except UnicodeDecodeError:
                pass
    return out


def postscript_names(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12:
        return set()
    if data[:4] == b"ttcf":
        n = struct.unpack_from(">I", data, 8)[0]
        bases = [struct.unpack_from(">I", data, 12 + 4 * i)[0] for i in range(min(n, 64))]
    else:
        bases = [0]
    names = set()
    for b in bases:
        names |= _name_records(data, b)
    return names


def installed_postscript_names(refresh=False):
    global _cache
    if _cache is not None and not refresh:
        return _cache
    names = set()
    for d in font_dirs():
        for fn in os.listdir(d):
            if os.path.splitext(fn)[1].lower() not in (".ttf", ".otf", ".ttc"):
                continue
            try:
                names |= postscript_names(os.path.join(d, fn))
            except (OSError, struct.error):
                continue
    _cache = names
    return names


def missing(font_names):
    have = {n.lower() for n in installed_postscript_names()}
    return sorted({f for f in font_names if f.lower() not in have})
