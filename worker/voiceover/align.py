"""Map spoken word timings back onto the WRITTEN script. Pure; no audio, no models.

Why this exists: what a voice says and what the screen shows are different
strings. The script says "e.MAS" and the synthesiser is handed "ee-Mars"; the
script says "S$126,000" and Whisper hears "126,000 dollars". Captions and cue
landings key off the written words, so every written token needs a time even
when nothing spoken matches it character for character.

Method: difflib opcodes over normalised tokens.
  equal    -> copy the spoken time
  replace  -> spread the spoken span over the written tokens by length
              (this is where "e.MAS" <- "ee-Mars" lands, correctly)
  delete   -> a written token nothing spoke: filled from its neighbours
  insert   -> a spoken word the script does not have: ignored
"""
import difflib
import re

_NORM = re.compile(r"[^0-9a-z]+")
_EDGE_PUNCT = "\"'“”‘’()[]{},;:!?."
_EOS = re.compile(r"(\.\.\.|[.!?])[\"'”’)\]]*$")


def norm(token):
    return _NORM.sub("", token.lower())


def written_tokens(text):
    """Whitespace tokens of the written line, with display form and sentence-end flag.

    The display form strips surrounding punctuation but keeps internal marks,
    so "e.MAS," shows as "e.MAS" and "$126,000." as "$126,000".
    """
    out = []
    for raw in text.split():
        display = raw.strip(_EDGE_PUNCT) or raw
        out.append({"w": display, "raw": raw, "eos": bool(_EOS.search(raw))})
    return out


def split_spoken(words):
    """A synthesiser token can hold several words ("6.4 seconds" is ONE edge-tts
    WordBoundary). Split those, sharing the span by character length, so the
    matcher compares word with word."""
    out = []
    for start, end, text in words:
        parts = text.split()
        if len(parts) <= 1:
            out.append((start, end, text.strip()))
            continue
        total = sum(max(1, len(p)) for p in parts)
        t = start
        for p in parts:
            d = (end - start) * max(1, len(p)) / total
            out.append((t, t + d, p))
            t += d
    return [w for w in out if w[2]]


def _spread(tokens, start, end):
    total = sum(max(1, len(norm(t["w"]))) for t in tokens)
    t = start
    for tok in tokens:
        d = (end - start) * max(1, len(norm(tok["w"]))) / total
        tok["start"], tok["end"] = t, t + d
        t += d


def align(text, spoken, line_start=0.0, line_end=None):
    """Written tokens of `text`, each with start/end in seconds.

    spoken      [(start_s, end_s, token)] relative to the line's own audio
    line_start  offset added to every time (the line's place in the full VO)
    line_end    the line audio's length; bounds the fill for trailing tokens
    """
    toks = written_tokens(text)
    if not toks:
        return []
    sp = split_spoken(spoken)
    if not sp:
        # Nothing heard at all: spread evenly over the line so the caller still
        # gets monotonic times rather than a crash. Flagged by `matched` below.
        _spread(toks, 0.0, line_end or 0.0)
        for t in toks:
            t["matched"] = False
        return _finish(toks, line_start)

    a = [norm(t["w"]) for t in toks]
    b = [norm(w[2]) for w in sp]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                toks[i1 + k]["start"], toks[i1 + k]["end"] = sp[j1 + k][0], sp[j1 + k][1]
                toks[i1 + k]["matched"] = True
        elif tag == "replace":
            _spread(toks[i1:i2], sp[j1][0], sp[j2 - 1][1])
            for t in toks[i1:i2]:
                t["matched"] = False
        elif tag == "delete":
            for t in toks[i1:i2]:
                t["matched"] = False
        # insert: spoken filler the script does not have - nothing to time.

    _fill_gaps(toks, line_end if line_end is not None else sp[-1][1])
    return _finish(toks, line_start)


def _fill_gaps(toks, line_end):
    """Give every untimed (deleted) token a slot between its timed neighbours."""
    i = 0
    n = len(toks)
    while i < n:
        if "start" in toks[i]:
            i += 1
            continue
        j = i
        while j < n and "start" not in toks[j]:
            j += 1
        lo = toks[i - 1]["end"] if i > 0 else 0.0
        hi = toks[j]["start"] if j < n else max(lo, line_end)
        if hi < lo:
            hi = lo
        _spread(toks[i:j], lo, hi)
        i = j


def _finish(toks, line_start):
    out = []
    prev = 0.0
    for t in toks:
        # Monotonic and non-negative whatever the matcher did.
        s = max(prev, float(t["start"]))
        e = max(s, float(t["end"]))
        prev = s
        out.append({"w": t["w"], "start": round(line_start + s, 3),
                    "end": round(line_start + e, 3), "eos": t["eos"],
                    "matched": bool(t.get("matched"))})
    return out


def match_rate(words):
    """Share of written tokens that matched a spoken word exactly. A low rate on a
    line means the voice did not say what was written (or said it unintelligibly)."""
    if not words:
        return 1.0
    return sum(1 for w in words if w["matched"]) / len(words)
