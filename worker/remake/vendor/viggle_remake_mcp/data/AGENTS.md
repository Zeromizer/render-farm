# Viggle Remake agent rules (the `viggle-remake` MCP)

You remake video clips with the `viggle-remake` MCP tools, one **session** per video. Two kinds:
- **swap** - whole-person replacement: the people of a span are replaced by other characters (invented people,
  anthropomorphic animals, a mascot), the original soundtrack kept.
- **edit** - an object is removed or replaced, or the background is swapped (no ball, the bike becomes an ostrich, the
  airport becomes a barn); the people stay.

The task gives you the session id, the kind, the source clip with its span, the cast or the edit, and your image-call
budget. You do not download or search for sources. When you run unattended (e.g. `codex exec`), nobody answers
questions mid-run: decide, do, and write down what you decided. A clip of a minute or more takes hours (dozens of
paints and windows); there is no time or token limit. Never finish before every window with a person to replace has a
reviewed take: a source fallback is not a delivery.

Sessions live under `$REMAKE_HOME/sessions/<sid>`: you may read their files directly. Do everything else through the
tools: never run the package's modules yourself, never touch GPUs or workers, never call an image or LLM API, never
touch another session's directory. **Look at every image a tool returns** before the next step; never judge from a
file name.

## Tools, in workflow order
| tool | what | cost |
|---|---|---|
| `new_session(sid, kind)` | creates `$REMAKE_HOME/sessions/<sid>`; kind `swap`/`edit` is fixed | free |
| `status(sid)` | where the session stands; call it first when you resume | free |
| `peek(sid, src, start, dur)` | contact sheet of a source span + its cut times | free |
| `ingest(sid, src, start, dur, crop, force)` | the span -> `case/` (24 fps), then strips | free |
| `strips` / `frames(sid, frames)` | one row per detector shot / up to 12 exact frames, large | free |
| `save_case(sid, plan_grid, case, where, cuts)` | writes the case files you pass, runs the case check | free |
| `windows(sid)` | render windows + each window's keyframes -> `case/refs.json` | free |
| `set_refs(sid, window, refs, why)` | a window's keyframes by hand (kept by later `windows`) | free |
| `sheet(sid, name, source=lib/photo/text)` | swap: character sheet | lib free, else 1 image call |
| `paint(sid, frames, quality)` | swap: paint 1-2 keyframes | 1 image call |
| `paint_many(sid, groups)` | swap, pool image mode only: several `paint`s at once | 1 image call per group |
| `epaint(sid, frame, edit, refs)` | edit: paint one keyframe | 1 image call |
| `take_keyframe(sid, window, frame, fix, box)` | swap: a shot's second keyframe cut from its take, repaired inside `box` | 1 image call (free without `fix`) |
| `write_prompts(sid, windows)` | swap: a brief per window (or, with a server LLM, the prompts themselves) | free (server LLM: one chat call per window) |
| `set_prompt(sid, window, text)` | store the prompt you wrote for a window and bind it | free |
| `cut_windows(sid, windows)` | edit: cut each window, return its contact sheet | free |
| `edit_prompt(sid, window, text)` | replace a prompt (edit: same as `set_prompt`) | free |
| `submit(sid, seeds, windows)` | queue renders (720, seed 0) | GPU time |
| `wait(sid, max_s)` | block until this session's renders are done | free |
| `strip(sid, windows)` | review strip: src row vs each take | free |
| `finish(sid, picks)` | `case/final/full_720.mp4` (cut flashes fixed) | free |
| `budget(sid)` / `view(sid, path)` / `save_result(sid, result)` | spend / look at a file / deliverable | free |

Paths you pass (`src`, `photo`, `refs`, `view`) are absolute or relative to the session dir; pictures from outside the
session are copied into its `photos/`. `paint`, `epaint` and `sheet` can take a few minutes. If one times out, `status`
shows whether the keyframe got painted before you call it again.

## Budget (hard)
- At most **24 image calls per session** (sheets + paints + repaints; enforced; the operator may set another limit), and
  your task's total across videos. Typical swap: 0-3 sheets + 3-6 paint calls + 1-2 repaints; typical edit: 1-3
  epaints. Plan keyframes before painting; do not repaint for small flaws.
- An error saying quota/credit, "PAINT STOPPED", "budget reached" or "used its N image calls": stop painting at once,
  never retry or work around it, finish with what is painted if you can, and report.
- A moderation refusal was already retried once. Do not reword around it. Pick a different frame of that shot once; if
  that is refused too, leave the shot without a keyframe and note it.
- Paint quality: `medium` when a character in the grid comes with an identity photo, `low` for everything else
  (invented people, animals, mascots, stylised characters) and every epaint. Never higher.

## Codex image mode
With `REMAKE_PAINT=codex` (the codex config in `examples/` sets it) `sheet` (photo/text), `paint` and `epaint` call no
image API and cost nothing. Each picture takes two calls of the same tool:
1. The first call returns `CODEX IMAGE STEP` with `referenced_image_paths`, a `copy_to` path and a PROMPT. Nothing is
   painted yet.
2. Call your `image_gen` tool once with exactly those paths and the PROMPT verbatim: no rewording, nothing added or
   dropped (it ends with the output size the import checks). Copy the PNG it saves to `copy_to`.
3. Call the same tool again with the same arguments. It imports the picture and returns the comparison to judge as usual.
   If it says the aspect is wrong, nothing was imported: start again from step 1.

When the step carries a `BRIEF_JSON` line, the import finds the PNG by itself (no copy). Then do the whole picture in
**one exec cell** - every extra cell is a full-context request - with no `yield_time_ms` pragma (the config lets a cell
run to its end; a pragma only makes you poll):
```js
const A = {sid: "S", frames: [36, 112], quality: "medium"};   // the tool's arguments
const T = tools.mcp__viggle_remake__paint;                       // or sheet / epaint
let r = await T(A);
const j = (r.content || []).map(c => c.text || "").join("\n").match(/^BRIEF_JSON (.+)$/m);
if (j) {
  const b = JSON.parse(j[1]);
  try { await tools.image_gen__imagegen(b); } catch (e) { text("refused once, same request again: " + e); await tools.image_gen__imagegen(b); }
  r = await T(A);
}
for (const c of r.content || []) c.type === "image" ? image(c) : text(c.text);
```
Several pictures may go in one cell this way (one after another, never in parallel); judge every comparison it
returns before the next cell. Likewise call `wait` and `finish` without a pragma.

In codex mode nothing retries a refusal for you. An image_gen `moderation_blocked` on a cast member is usually a
transient false positive (the same character passes in other calls): the recipe sends the identical request once more;
if that is refused too, run the cell again later in the job, unchanged. A refused **sheet** never turns a cast person
into `none`: retry it, or use `source="lib"` when the library has that character. Only after a refusal of the same
request later in the job does the rule in Budget apply.

`image_gen` takes at most 5 reference pictures. A paint of 3+ characters would need more (a sheet and a photo each,
plus the frame), so the brief packs each character's sheet and photo into one card (two characters a card past 4) and
its PROMPT already describes the cards: pass its paths as they are. A step that starts with `NOTE:` says why the
previous picture was not imported.

In this mode the per-session limit is **80** image calls unless the operator set another, and `quality` is ignored.
Judge and repaint by the same rules as above. Never paint a keyframe outside these tools: a PNG that `paint`/`epaint`
did not import is not used.

## Pool image mode
With `REMAKE_PAINT=pool` there is no CODEX IMAGE STEP: `sheet`, `paint` and `epaint` paint in **one call** (the server
runs a helper codex that calls image_gen and imports the picture) and return the comparison as usual. Do not call
`image_gen` yourself and skip the exec-cell recipe above. `paint_many(sid, groups)` runs up to 8 groups in parallel (the server
instructions give the number; a group = the 1-2 frames one `paint` call takes): paint each batch of up to 8 groups in
**one** `paint_many` call - 8 groups take about 1.5-2 min, the same as one. Call it in a plain cell with no
`yield_time_ms` pragma: the cell runs to its end; a pragma only makes you poll it with `wait`, every poll a
full-context request. Judge every comparison it returns, then repaint the bad ones together in one
more `paint_many` with `force=true` and `notes`. The server already retried a refusal once; a group that still failed
says so in its row while the others land: call it again later in the job, unchanged (a refused sheet never turns a cast
person into `none`). The limit is 80 image calls as in codex mode, and `quality` is ignored.

## Case files (`save_case`; `ingest` creates empty ones)
- `plan_grid`: `{"people": [{"id": "P1", "desc": "how to recognise them in the SOURCE (clothes, hair, position)",
  "replace_with": "Vela" | "none"}], "shots": [{"start": 0, "end": 135, "people": ["P1", "P3"], "desc": "what happens"}]}`.
  Shots tile [0, nf) exactly on `cuts` (fix `cuts` and the shots together if the detector missed or invented a cut).
  A shot's `people` = everyone in it, replaced or not. One id per source person across the clip; one character may
  replace several people (clones). Replace everyone the task's cast names, however many share a shot: the cast
  mapping overrides the next rule. With no cast given, replace at most 4 people per shot. Extras and crowds stay
  `"none"` unless cast.
- `case` (swap): `{"chars": {"Vela": "identity sentence: who, age, build, hair, face, plus the outfit"},
  "outfits": {"Vela": "teal flight jacket, grey turtleneck, cargo trousers, brown boots"}}` - the whole outfit down to
  the shoes; make outfits clearly different from each other or people swap identities. Names: letters, digits, space,
  `-`, `_`.
- `where`: one entry per keyframe, every person visible in that frame (replaced or not) and where:
  `{"135": {"P1": "centre, facing camera, mid-dance", "P3": "left edge, only an arm visible", "_note": "for the painter"}}`.
  Someone seen only as an arm stays only that arm. Write each entry from that frame, not from the shot's cast: the
  painter puts in everyone listed, so a person not in the frame is left out, and one text copied to more than 3
  keyframes is refused by the case check.

## Swap sessions
1. **Ingest** the task's span; look at every strip, `frames` where people are small. If the span is clearly worse than
   the task says (strobe cuts, tiny faces, subtitles over faces) you may shift/shorten it a few seconds (`force=true`) -
   say so in result.json.
2. **Plan** `plan_grid`, then **cast**: `case` + `sheet` per character. The operator's character library is
   `$REMAKE_HOME/chars/<Name>_sheet.png` (+ an optional `<Name>_photo.png`) and `lib_*.json`; `sheet(source="lib")`
   copies one and fills `case` with its library strings. A character not in the library: `sheet(source="photo",
   photo=...)` from a picture the task gives (kind `person`/`anthro`/`creature`) or `sheet(source="text", text=...)` for
   an invented animal or creature. Look at every sheet you use: it is the face you must recognise later. Costume
   variant: write the new outfit in `case`, then `sheet(source="photo")` from the character's photo (an animal/mascot:
   its library sheet as the photo, kind `anthro`/`creature`). Keep faces visible: no sunglasses, masks or helmets on a
   replacement character. Animals replacing humans are **anthropomorphic**: photorealistic animal head and fur on an
   upright, human-proportioned, clothed body.
3. **Keyframes** (strict):
   - ONE keyframe per shot that contains a replaced person: the frame where the replaced people are largest and sharp,
     whole face visible (not behind a mic or hand), >= 3 frames from a cut, and showing **every** replaced person of the
     shot - a replaced person absent from the keyframe renders as the source person.
   - A SECOND keyframe in the shot only if (a) a replaced face flips side (frontal <-> pure profile or back;
     three-quarter does not count) or (b) a replaced person enters who is not in the first one. The new state lasts
     >= 2 s and the two keyframes are >= 3 s apart; give the reason as `set_refs(..., why)`. A push-in/pull-out or an
     occlusion does NOT qualify. Never two keyframes of nearly the same framing (paint-to-paint differences snap).
   - Anything else is fixed AFTER the first render, never by painting another keyframe from the source: a part the
     keyframe does not show (legs, shoes, a hem, a carried bag) that the take gets wrong gets a `take_keyframe` - a
     frame of the take itself, repaired only around that part (see the fix round).
   - **Rapid cuts** (a window cutting back and forth between a few camera setups; a setup = the same camera position,
     framing and people): one keyframe per SETUP per window instead of per shot, chosen as above. A shot repeating a
     setup already keyframed in the same window counts as covered; `windows` still lists it under `!! no keyframe`, which
     is expected for those shots. Never two keyframes of the same setup in one window, and never leave such a span as
     source because of the per-shot rule.
   - No `!!` line from `windows` is a reason to stop or to ask. `the renderer takes 4 keyframes` means a window holds
     more setups than that: keyframe the setups with the most screen time (`set_refs`), note the weak window in the
     result and go on (a run that stopped here delivered nothing; runs that went on came out fine).
   - Faces under ~50 px at 480p drift to strangers; a window with 5+ fast cuts follows none of them well.
   - `save_case(where=...)`, then `windows` until `status`'s check reports no shot without a keyframe ("not painted yet"
     and rapid-cut repeats are expected). A window takes at most 4 keyframes. Windows end on cuts and are at most what
     the live workers render (`windows` sizes them; <= 345 frames, 14.4 s); a longer one-take becomes a chain of
     **hops**, each continuing the previous window's take (shown as "hop of w<j>"). Start the workers before `windows`,
     or run `windows` again once they are up.
4. **Paint**: two PAINTED keyframes of one shot (flip / new person) always go in ONE call - painted apart, an
   outfit's colours can come out different and the render switches between them mid-shot. Other frames: two per
   call when they show the same characters, else one. A frame lit unlike the rest - a backlit silhouette, deep
   shadow, a strobe - goes alone, with a `note` saying it stays that dark and only the outline changes (ears, muzzle,
   tail): without it the painter lights the figure up and the render follows, and a note reaches every frame of its
   call. Read every comparison: right character in the right place, its
   outfit fully replacing the original (no hat/sunglasses/costume left), composition and people count unchanged,
   nobody added, unreplaced people
   untouched. Then compare across calls: each replaced character's outfit colours and patterns (base colour, stripes,
   prints) must match its sheet and be the same in every keyframe; repaint the one that differs, naming the part and
   colour in `note`. Repaint real failures only (`force=true, note=...`). A result's `kept_source_face` is one: the
   face under the new hair is still the original actor's (a measured face match, easy to miss by eye); repaint that
   frame alone with the note it asks for - without one, a bald-on-bald or same-hair swap comes back as the source again.
5. **Prompts**: `windows` again, then `write_prompts`, which cuts each window and checks its keyframes.
   - **You write them (the default).** `write_prompts` returns one brief per window: its frames, fps and canvas, who
     becomes whom, its shots with who is on screen, its `<Picture k>` paths in order, its contact sheet, and the rules;
     the images it returns are, per window, the pictures in order and then the contact sheet. The short
     prompt-writing guide and the format examples are at the end of this file when `scripts/codex_exec.sh` started
     you; otherwise the session's first `write_prompts` result opens with them (later calls do not repeat them; pass
     `guide=true` once if they are no longer in your context). Follow the guide, copy the example's structure. Write each
     window's six sections from its brief and its images, then `set_prompt(sid, window, text)`. `set_prompt` refuses a
     text whose sections are missing or out of order or that leaves a `<Picture k>` unnamed: fix it and call again; act
     on its warnings ("dim", non-ASCII). Speech: no transcript is known - visible talking or singing is
     `<d>[unclear]</d>` with the speaker's ID, never guessed words.
   - **A server LLM writes them** (when the operator set `REMAKE_LLM_MODEL`): `write_prompts` returns the warnings and
     every prompt; read each one. In **PROMPT POOL** mode (`REMAKE_LLM_MODEL=codex-pool`, the server instructions say
     so) one helper codex per window writes and stores them, and you get only each window's lint result: act on it, and
     to fix a prompt (a lint line, a review's repair, a timed action) call `write_prompts(windows=[w], notes="...")` with
     what to change instead of writing the text yourself.
   - Either way act on the warnings (someone "in no ref frame" = move that keyframe). Timed actions: when a replaced
     person must open/close eyes or mouth or start any visible action at a given moment, the prompt must say WHEN
     ("only at about 00:11, two seconds before the cut"); fix a stored prompt with `edit_prompt`. Any later repaint /
     keyframe / window change: `windows` + `write_prompts` + `set_prompt` again before submit (submit refuses a stale
     prompt).

## Edit sessions
1. `new_session(sid, "edit")` first. `ingest` as above.
2. The edited thing is a **pseudo-person**: real people `"replace_with": "none"`, the object
   `{"id": "X1", "desc": "the orange basketball the player dribbles", "replace_with": "NoBall"}` (or "Ostrich", "Barn"),
   in every shot where it is visible. `case`: `{"chars": {"NoBall": "removed: no ball anywhere"}, "outfits":
   {"NoBall": "none"}, "edit": "one sentence of the edit"}`. `where`: the people AND `"X1": "where the object is"`.
3. Keyframes: one per shot containing X1, where the object is clearly visible and large, >= 3 frames from a cut. A second
   one only if the shot is > 8 s or the object's look changes a lot, >= 3 s apart (`set_refs` with why).
4. `epaint(frame, edit="...")` per keyframe. The instruction says what changes and what fills the space: "Remove the
   basketball completely: his right hand is open in the same dribbling pose with nothing in or under it, and the floor
   and crowd behind where the ball was are visible." `refs` = an optional picture of what the edit brings in. Read each
   comparison: the edit done, nothing else changed (people, pose, background, framing).
5. `windows`, `cut_windows`, then write each window's prompt by hand and `edit_prompt` (or `set_prompt`) it. Format
   (window time MM:SS.mmm, 00:00.000 = the window's first frame): six sections in order, a blank line between, nothing
   before `subject_definitions:`; `<Picture k>` = the window's k-th keyframe in refs order, every one named; never write
   "dim". The edit-session format example (a background swap, two keyframes; end of this file, or atop `cut_windows`): copy its structure:
   - `subject_definitions:` `<Video 1> is the source video for the target video edit.` / `<Picture 1> is a frame of the
     target video, exactly as the target video shows it at that moment.` / one `<Subject k>` per person, the object
     (and its replacement, with looks/size/colour) and the setting.
   - `summary:` `[video editing] The target video is an edited version of <Video 1>. ...` - same moves, framing, timing.
   - `retention_analysis:` `<Video 1> (source video editing): fully_preserved - ...`, `<Picture k> (target keyframe):
     fully_preserved - ...`, each subject `fully_preserved` / `removed - ...` / `replaced - ... by <Subject k>`.
   - `detailed_description:` style line, then `[Shot 1] At 00:00.000 ...` per shot with times.
   - `overall_soundscape:` as in the source (music described here); `non_diegetic_music:` `N/A`.
   Describe the target video only; keep the people's descriptions identical to the source.
   Any later repaint / keyframe change: `windows` + `cut_windows` + `edit_prompt` again.

## Render, review, finish (both kinds)
- **Render while you paint.** Painting is most of the wall-clock and the workers sit idle through it. Paint in time
  order (pool image mode: one `paint_many` per batch); as soon as every keyframe of the next few windows (about 4) is
  painted and checked, run `windows`, then
  `write_prompts(windows=[those])`, their `set_prompt`s and `submit(windows=[those])`, and go on painting. A hop goes in
  its window's batch or a later one. `wait` only after the last batch is in: the renders do not need you. The cross-call
  outfit check still covers every keyframe; a later repaint of a keyframe a submitted window uses makes only that
  window stale - write its prompt and submit it again.
- `submit(sid)` = canvas 720, seed 0. It refuses the 720 canvas when no live worker renders it, and windows longer than
  the live workers take (run `windows` again). `wait` repeatedly until nothing is queued or running (minutes per window,
  and the queue is shared between sessions). If nothing of yours finishes for 40 min while `all_queue` does not shrink,
  or `wait` reports no `workers_alive`, stop and report. `orphaned` jobs (their worker died) are put back by a live
  worker; if they stay, report. Hops are queued with their window: a worker holds a hop until the take it continues
  lands (`hops_waiting`); `hops_stuck` means that take will not come (its window failed or was not submitted): submit
  that window. A hop must be picked with the same seed as its window.
- **Fix round, at most one per window per user request**: `submit(seeds=[1], windows=[w])`, or seed 0 again after a
  repaint/new keyframe (changed inputs make a new job). **A part the keyframe does not show is wrong** on a main
  figure for >= 2 s (legs, shoes, skirt length, a bag off the body or on the wrong side): not seed 1 but
  `take_keyframe(sid, w, frame, fix, box)` - `frame` = a frame of the seed-0 take where that part is large and the rest
  is right, >= 72 frames from the shot's keyframe; `fix` = what is wrong and what it should be, from the source and the
  sheet ("her skirt ends at the knee and shows bare legs; it is a floor-length navy skirt"); `box` = a tight
  [x0, y0, x1, y1] (fractions) around that part. Outside the box the keyframe is the take's own pixels, so it agrees
  with the render and does not snap. Read its take|edit|keyframe comparison (the fix done, no seam at the box edge),
  then `write_prompts(windows=[w])`, `set_prompt`, `submit(windows=[w])` - seed 0 again, this window's fix round.
  Measured on 8 shots x 2 seeds: it fixed the 3 shots with such a defect in both seeds and added no snap in the 5
  others. Do not wait for seed 1 elsewhere. A fix round that changed the inputs (pin, repaint, take_keyframe) and whose
  seed 0 came out worse - a cut now `dropped`, a new early cut, a worse face - gets one follow-up in the same round:
  `submit(seeds=[1], windows=[w])` and pick the better take; if seed 1 is worse too, `unpin(w)` (after a pin) puts the
  window back so its pre-fix take counts again (seen once: the pin's seed 0 dropped a short insert shot, seeds 1 and 2
  still cut 5-6 frames early). Then deliver: the user reviews
  and a follow-up message with their feedback opens a new round for the windows it names (an automatic "continue"
  after an error does not).
- `strip` for all windows; compare every column of every take with `src` (`KEY` = keyframe frame).
  Swap: wrong identity (wrong character, two of the same, a face unlike the sheet) - not replaced / leak (an original
  face or costume item) - wrong count or composition (someone missing/added, a missed or extra cut) - broken image
  (melted faces, extra limbs, smears) - the source's light lost (a backlit silhouette or a figure in shadow comes out
  lit, with face and outfit detail: repaint that keyframe alone with a `note` that it stays dark, then seed 0 again;
  a main figure's is BAD). Edit: the object must be gone/replaced in EVERY column, not only at keyframes;
  people, motion and background otherwise identical; no smear where it was. If it comes back between keyframes, add a
  keyframe mid-span (note it) or strengthen the prompt, re-render once.
  `cut timing` under a strip line is measured, not seen: a cut the take puts 8+ frames off the source (`cut f2393 at
  f2376 (-17)`) or adds (`extra cut f803`, often a one-frame glitch). A main cut that lands 8+ frames off is off the
  beat: BAD. A cut 8+ frames LATE: seed 1 in the fix round, and pick the take whose cuts land. A cut 8+ frames EARLY
  (`early on every seed: [5042]`, listed from seed 0 alone): spend the fix round on the pin, not on seed 1 - another
  seed will not move it (5 of 6 such cuts were early again on seed 1): in a dialogue the next speaker's voice starts
  under the outgoing listener and the model cuts to the voice. `pin_before_cuts(window, [those cuts])` paints a keyframe of
  the outgoing shot 6 frames before each cut; then `write_prompts([window])` and seed 0 again (measured on 4 cuts x 2
  seeds: mean 14 frames early -> 2.75, all 8 within 5 frames). `mute_before_cuts` (silence the second before the
  cut, mean 14 -> 7) is the fallback when the outgoing shot is under 9 frames or a pinned cut is still early. An extra cut close to a source cut
  (`extra cut f2380 (-13 from the source cut f2393)`) is the same fault: the take leaves the shot early and cuts again
  on the beat, or dissolves the people into half-transparent ghosts first. Rapid cuts of one setup (the same face re-framed) and flash frames may be listed; those are MINOR.
  `cut f1529 dropped` means the take plays the previous shot on through that cut: a missed shot, BAD. It is the seed
  (a 6-frame montage shot: 1 take in 10 dropped it, with or without a keyframe in that shot): seed 1.
  `edge bands: s0 bottom edge band f285-308` is measured too: along that edge the take grew a black bar or a strip
  of another picture that the source does not have. Another seed will not fix it (5 of 5 seeds kept it). The cause is a
  keyframe of that window carrying another frame's edge: open its paint files, find the one with the strip on that
  edge, repaint that frame alone (`paint` with only that frame, `force=true`), then seed 0 again. BAD if it lasts 1 s+.
  Grades: BAD = any of that on a main/large figure or a missed shot; MINOR = only on a small/edge/background figure or
  mild drift; OK = clean. Be strict: "slightly weaker likeness" often hides a wrong identity. A window where every take
  is BAD: find the cause before spending (a shot without its own keyframe, a bad paint, or bad luck).
- `finish(sid, picks={"0": 0, ...})` -> `case/final/full_720.mp4` (cut flashes are fixed inside; it prints what).
  `rest_source=true` keeps source footage only for windows you rendered and could not fix (say which in `problems`).

## Content rules (each one costs a failed run when broken)
1. Whole-person replacement: face, hair, build and outfit all change. Never write "only the face" or "keep the
   original clothes" anywhere.
2. The video model gets only painted keyframes, never sheets or reference photos.
3. Every shot with a replaced person (or the edited object) needs its own keyframe (in rapid cuts: its setup's keyframe
   in the same window); more seeds do not fix a missing one.
4. Keep a video short - about 8-20 s unless the task says otherwise: every window costs GPU time and every seam is a
   risk.

## Deliverable: `save_result`
```json
{"status": "done | partial | failed", "final": "$REMAKE_HOME/sessions/<sid>/case/final/full_720.mp4 (absolute)",
 "seconds": 12.0, "cast": {"P1": "Vela"}, "caption": "...", "picks": {"0": 0},
 "windows": [{"w": 0, "grade": "OK|MINOR|BAD", "note": "..."}], "overall": "OK|MINOR|BAD",
 "image_calls": 7, "changed": "anything you changed vs the task and why", "problems": "what is still weak"}
```
`sid` and `kind` are added for you. `overall` = the worst grade among the picks.
Final message: <= 80 words - status, overall grade, image calls used, the biggest remaining problem.
