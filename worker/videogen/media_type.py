"""Recognise a media file by its content, not its name.

The platform stores assets content-addressed (assets/sha256/<hex>, no
extension), so a downloaded input arrives as "front" with no suffix and the
old extension check rejected it ("unsupported type ''"). ComfyUI's Load*
nodes and ffmpeg both want a sensible suffix, so we sniff the magic bytes,
verify the file decodes, and rename it with the matching extension.

Stdlib only; no Pillow in the worker venv.
"""
import os

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
KIND_EXT = {"image": IMAGE_EXT, "video": VIDEO_EXT, "audio": AUDIO_EXT}


def sniff(path):
    """(kind, ext) from the first bytes, or (None, None) when unrecognised.
    kind is image | video | audio; ext includes the dot."""
    with open(path, "rb") as f:
        head = f.read(64)
    return sniff_bytes(head)


def sniff_bytes(head):
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image", ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image", ".webp"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio", ".wav"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B "):
            return "audio", ".m4a"
        if brand == b"qt  ":
            return "video", ".mov"
        return "video", ".mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video", ".webm" if b"webm" in head else ".mkv"
    if head.startswith(b"fLaC"):
        return "audio", ".flac"
    if head.startswith(b"OggS"):
        return "audio", ".ogg"
    if head.startswith(b"ID3") or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE6) == 0xE2):
        return "audio", ".mp3"
    return None, None


def kind_of_ext(ext):
    ext = (ext or "").lower()
    for kind, exts in KIND_EXT.items():
        if ext in exts:
            return kind
    return None


def ensure_extension(local, allowed_kinds, name=None, probe=True):
    """Return a path for `local` whose suffix matches its content and is one of
    allowed_kinds (subset of image/video/audio). Renames the file when the
    suffix was missing or wrong. Raises RuntimeError with a useful message for
    unsupported or corrupt files. `probe` runs ffprobe to confirm the file
    really decodes (a sniffed header is not proof of a whole file)."""
    name = name or os.path.basename(local)
    allowed_kinds = tuple(allowed_kinds)
    kind, ext = sniff(local)
    cur_ext = os.path.splitext(local)[1].lower()
    if kind is None:
        # Unknown magic: trust a supported suffix (e.g. an odd jpeg variant), else refuse.
        kind = kind_of_ext(cur_ext)
        ext = cur_ext
        if kind is None:
            raise RuntimeError(f"video_gen input {name}: unsupported or corrupt file "
                               f"(no recognised image/video/audio signature, suffix {cur_ext!r})")
    if kind not in allowed_kinds:
        raise RuntimeError(f"video_gen input {name}: {kind} ({ext}) given where "
                           f"{' or '.join(allowed_kinds)} is expected")
    if cur_ext not in (ext, ".jpeg" if ext == ".jpg" else ext):
        target = os.path.splitext(local)[0] + ext
        if os.path.abspath(target) != os.path.abspath(local):
            if os.path.exists(target):
                os.remove(target)
            os.replace(local, target)
        local = target
    if probe:
        _probe(local, kind, name)
    return local


def _probe(local, kind, name):
    """ffprobe must see a matching stream; a truncated or mislabeled object fails here."""
    import json
    import subprocess

    from videogen import segments

    sel = "a:0" if kind == "audio" else "v:0"
    r = subprocess.run([segments._tool("ffprobe"), "-v", "error", "-select_streams", sel, "-show_entries",
                        "stream=codec_type,width,height", "-of", "json", local],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    streams = []
    try:
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except ValueError:
        pass
    if r.returncode != 0 or not streams:
        raise RuntimeError(f"video_gen input {name}: file does not decode as {kind} "
                           f"({(r.stderr or '').strip()[-200:] or 'no stream found'})")
    if kind in ("image", "video") and not (streams[0].get("width") and streams[0].get("height")):
        raise RuntimeError(f"video_gen input {name}: {kind} has no dimensions")
