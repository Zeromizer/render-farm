"""Progress and time-left for a YuE2 generation, from ComfyUI's tqdm lines.

comfy_client.wait() was built for one fixed run of diffusion steps: it reads
"N/M [.. X it/s]", calls N/M the fraction and (M-N) x rate the time left. A YuE2
prompt is three different runs and breaks each assumption in turn. Measured on the
first queue job (15 s bed, 151 s wall), the row said "~2828 min left":

  YuE2 ABC sampling:   2280/8192 [.., 22 token/s]   the total is max_abc_tokens, a CAP;
                                                    a real score ends at 1300-2300
  YuE2 music sampling: 161/425   [.., 21 token/s]   the total is real (25 x max_duration,
                                                    and the model always runs to it)
  (KSampler)           25/32     [.., 4 it/s]       the only stage wait() understood

and "21 token/s" was being read as 21 SECONDS per token, because wait()'s caller
only inverts the rate for the literal unit "it/s".

So this module reads the newest line itself (the description says which stage it
is), works out an honest label, fraction and seconds-left per stage, and hands
wait() a synthetic (step, total, seconds_per_step) whose arithmetic reproduces
exactly those numbers. wait() keeps owning the queue position, the cancel check,
the timeout and the crash detection; the runner swaps this in for
comfy_client.sampling_progress only for the length of its own prompt and puts the
original back, because the same worker process runs video_gen next.
"""
import re
import time

_LINE = re.compile(
    r"(?:YuE2\s+(?P<desc>[A-Za-z]+)\s+sampling:?)?[^\[\]]*?"
    r"(?P<i>\d+)/(?P<n>\d+)\s*\[[^\]<]*<[^,\]]*,\s*(?P<rate>[\d.]+)\s*(?P<unit>s/it|it/s|s/token|token/s)\]")

# A score that finishes does so by about here; past it the seed is probably a runaway
# heading for the cap (audiogen/graphs.abc_token_cap), which is what bounds the damage.
_TYPICAL_ABC_TOKENS = 2200
_MUSIC_TOKENS_PER_SECOND = 25
_DIFFUSION_SECONDS = 10.0
_VIRTUAL_STEPS = 1000


def parse(line):
    """(stage, i, n, seconds_per_unit) from one log line, or None. stage: abc | music | diffusion."""
    m = _LINE.search(line or "")
    if not m:
        return None
    rate = float(m.group("rate"))
    if m.group("unit") in ("it/s", "token/s"):
        rate = 1.0 / rate if rate else 0.0
    desc = (m.group("desc") or "").lower()
    stage = "abc" if desc == "abc" else "music" if desc == "music" else "diffusion"
    return stage, int(m.group("i")), int(m.group("n")), rate


def estimate(stage, i, n, rate, duration_s, headroom=2.0):
    """(label, seconds_left) for the whole prompt, given the newest line.

    Later stages are costed from what this box measured, at the rate the current
    stage is actually running where that is a token stage (both run ~21 token/s)."""
    music_tokens = _MUSIC_TOKENS_PER_SECOND * (float(duration_s) + headroom)
    if stage == "abc":
        per_token = rate or 1.0 / 21
        # The cap is not the expected count. Expect a normal score; once past that,
        # expect the cap (n), because that is where a runaway is going.
        expected = _TYPICAL_ABC_TOKENS if i < _TYPICAL_ABC_TOKENS else n
        left = max(0, expected - i) * per_token + music_tokens * per_token + _DIFFUSION_SECONDS
        return f"writing the score ({i} tokens)", left
    if stage == "music":
        left = max(0, n - i) * (rate or 1.0 / 21) + _DIFFUSION_SECONDS
        return f"writing the music {i}/{n}", left
    return f"rendering audio {i}/{n}", max(0, n - i) * (rate or 0.3) + 2.0


class Reader:
    """Drop-in for comfy_client.sampling_progress(since_iso) while one audio prompt runs.

    `fetch_entries` returns ComfyUI's /internal/logs/raw entries ([{t, m}], oldest first);
    injected so the arithmetic is testable without a server. `label` is what the runner
    shows instead of wait()'s "sampling N/M", which would be the synthetic numbers."""

    def __init__(self, duration_s, fetch_entries, headroom=2.0, clock=time.monotonic):
        self.duration_s = float(duration_s)
        self.headroom = headroom
        self.fetch_entries = fetch_entries
        self.clock = clock
        self.started = None
        self.label = None

    def __call__(self, since_iso):
        if self.started is None:
            self.started = self.clock()      # first call = the prompt left the queue
        found = None
        for e in reversed(self.fetch_entries() or []):
            if e.get("t", "") < since_iso:
                break
            found = parse(e.get("m"))
            if found:
                break
        if not found:
            return None                       # wait() says "loading model" from the hint
        self.label, left = estimate(*found, duration_s=self.duration_s, headroom=self.headroom)
        elapsed = max(0.0, self.clock() - self.started)
        frac = elapsed / (elapsed + left) if elapsed + left > 0 else 0.0
        # wait(): frac' = 0.1 + 0.8 * i/n, eta' = (n - i) * rate + tail. Solve for i and rate.
        step = min(_VIRTUAL_STEPS - 1, int(frac * _VIRTUAL_STEPS))
        # wait() adds its own tail (decode + save) on top, which is the right thing to add.
        rate = left / (_VIRTUAL_STEPS - step)
        return step, _VIRTUAL_STEPS, rate
