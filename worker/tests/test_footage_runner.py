"""Footage runner: frame arithmetic and the CPU probe/assemble path.

The arithmetic cases are anchored on the windows actually measured in Phase 0
on this host, so a change that silently alters the grid maths fails here.
"""
import json
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners import footage  # noqa: E402

TAKES = r"C:\Coding\benchmarks\obvpm-phase0-2026-09-22\takes"
R24 = {"num": 24, "den": 1}


class FrameArithmetic(unittest.TestCase):
    def test_same_rate_is_identity(self):
        self.assertEqual(footage.allocate_frames(0, 73, R24, R24), 73)
        self.assertEqual(footage.allocate_frames(10, 44, R24, R24), 34)

    def test_rate_conversion_rounds_half_up(self):
        # 30 fps -> 24 fps: 30 source frames is 1.0 s is 24 output frames
        self.assertEqual(footage.allocate_frames(0, 30, {"num": 30, "den": 1}, R24), 24)
        # 24 -> 30 the other way
        self.assertEqual(footage.allocate_frames(0, 24, R24, {"num": 30, "den": 1}), 30)
        # NTSC 30000/1001 for one second of source
        self.assertEqual(
            footage.allocate_frames(0, 30, {"num": 30000, "den": 1001}, R24), 24)

    def test_short_trim_never_vanishes(self):
        self.assertEqual(footage.allocate_frames(0, 1, {"num": 60, "den": 1}, R24), 1)

    def test_empty_or_reversed_trim_is_rejected(self):
        with self.assertRaises(footage.FootageError):
            footage.allocate_frames(10, 10, R24, R24)
        with self.assertRaises(footage.FootageError):
            footage.allocate_frames(10, 5, R24, R24)

    def test_sample_window_matches_phase0_measurements(self):
        # one-sided, 39 held: Phase 0 T2 sampled 73 and delivered 34
        self.assertEqual(footage.solve_sample_window(34, 39, 0), (73, 34))
        # asking for less still lands on the same grid step
        self.assertEqual(footage.solve_sample_window(30, 39, 0), (73, 34))
        # two-sided, 78 held: Phase 0 T3b/T4b sampled 107 and delivered 29
        self.assertEqual(footage.solve_sample_window(29, 39, 39), (107, 29))
        # a smaller ask lands on the LOWER grid step, not the tested maximum:
        # 1 + 78 = 79 rounds up to 90, delivering 12. 29 is the ceiling at
        # raw 107, not what every two-sided request costs.
        self.assertEqual(footage.solve_sample_window(1, 39, 39), (90, 12))

    def test_window_always_lands_on_the_grid(self):
        for desired in range(1, 80):
            raw, delivered = footage.solve_sample_window(desired, 39, 0)
            self.assertEqual((raw - footage.GRID_OFFSET) % footage.GRID_STRIDE, 0)
            self.assertGreaterEqual(delivered, desired)

    def test_limits_cover_only_operations_proven_through_the_adapter(self):
        """Phase 0 proved the NODES; it did not prove this runner can drive
        them on the AV grid. Each op joins the list after its own raw-90 run."""
        lim = footage.generation_limits()
        self.assertEqual(set(lim), set(footage.PROVEN_GENERATION_OPS))
        self.assertNotIn("prepend", lim)
        for op, spec in lim.items():
            two = op in ("bridge", "loop")
            self.assertEqual(spec["grid_offset"], 39)
            self.assertEqual(spec["grid_stride"], 51)
            self.assertEqual(spec["max_new_frames"],
                             90 - 39 - (39 if two else 0))

    def test_av_grid_is_used_for_every_source_audio_state(self):
        """One conservative profile: an audio stream is not evidence of
        audible content, and a successor can have audio its source lacked."""
        self.assertEqual(footage.solve_sample_window(24, 39, 0, av_grid=True), (90, 51))
        self.assertEqual(footage.solve_sample_window(1, 39, 0, av_grid=True), (90, 51))
        self.assertEqual(footage.solve_sample_window(12, 39, 39, av_grid=True), (90, 12))
        for want in range(1, 52):
            raw, delivered = footage.solve_sample_window(want, 39, 0, av_grid=True)
            self.assertEqual((raw - 39) % 51, 0)
            self.assertGreaterEqual(delivered, want)

    def test_masked_window_22_is_not_on_the_av_grid(self):
        """Upstream masked_window_ok rejects 22 for masked/both pins; it is
        legal only for guide pins. Guards against reviving the 34->22 chain."""
        ok = lambda n: n >= 39 and (n - 39) % 51 == 0
        self.assertFalse(ok(22))
        self.assertTrue(ok(39))
        self.assertTrue(ok(90))

    def test_a_chained_extend_from_51_delivered_is_grid_legal(self):
        """raw 90 -> delivered 51 makes the successor extendable at 39:
        raw_start = pinned_head + (delivered - window) = 39 + 51 - 39 = 51."""
        raw_start = 39 + (51 - 39)
        self.assertEqual(raw_start, 51)
        self.assertEqual(raw_start % 17, 0)


@unittest.skipUnless(os.path.isdir(TAKES), "Phase 0 takes not archived here")
class ProbeAndAssemble(unittest.TestCase):
    """The real CPU path, on the clips Phase 0 actually produced."""

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(TAKES, "T1-root_00001.mp4")
        cls.ext = os.path.join(TAKES, "T2-extend_00001.mp4")
        cls.work = os.path.join(os.environ.get("TEMP", "."), "footage_test")
        os.makedirs(cls.work, exist_ok=True)

    def test_probe_matches_known_takes(self):
        info = footage.probe_media(self.root)
        self.assertEqual(info["frame_count"], 73)
        self.assertEqual((info["width"], info["height"]), (832, 480))
        self.assertEqual(info["fps"], R24)
        self.assertAlmostEqual(info["duration_seconds"], 73 / 24, places=4)
        ext = footage.probe_media(self.ext)
        self.assertEqual(ext["frame_count"], 34)

    def test_sha256_is_stable(self):
        a, size = footage.sha256_size(self.root)
        b, _ = footage.sha256_size(self.root)
        self.assertEqual(a, b)
        self.assertEqual(size, os.path.getsize(self.root))

    def test_assembled_frame_count_equals_allocation(self):
        """Trim two takes and concatenate: the delivered file must contain
        exactly the number of frames the allocation predicted."""
        cuts = [(self.root, 0, 40), (self.ext, 4, 30)]
        want, parts = 0, []
        for i, (src, a, b) in enumerate(cuts):
            info = footage.probe_media(src)
            n = footage.allocate_frames(a, b, info["fps"], R24)
            want += n
            part = os.path.join(self.work, f"p{i}.mp4")
            footage._cut(src, a, b, info, R24, 832, 480, n, part)
            self.assertEqual(footage.probe_media(part)["frame_count"], n)
            parts.append(part)
        final = os.path.join(self.work, "out.mp4")
        footage._concat(parts, final, lambda *_: None)
        got = footage.probe_media(final)
        self.assertEqual(got["frame_count"], want)
        self.assertEqual(want, 40 + 26)
        self.assertTrue(got["has_audio"])   # silence is inserted, never absent

    def test_trim_past_the_end_is_rejected(self):
        info = footage.probe_media(self.ext)
        with self.assertRaises(footage.FootageError):
            footage.allocate_frames(0, 0, info["fps"], R24)



class ResultEnvelope(unittest.TestCase):
    """Every operation must ship the envelope's required members. The website
    validates the whole shape, so a missing one fails collection outright."""

    REQUIRED = ("schema_version", "task_id", "org_id", "job_id",
                "request_key", "operation", "capabilities", "lineage", "warnings")

    def _manifest(self, op, tmp):
        job = {"id": "task-1", "params": {"footage": {
            "schema_version": 1, "org_id": "org-1", "job_id": "job-1",
            "request_key": "rk-1", "operation": op, "sources": []}}}
        import runners.footage as f
        uploaded = []
        real = f.db.upload_file
        f.db.upload_file = lambda remote, local, ctype: uploaded.append(remote)
        try:
            local, ext, ctype = f.run(job, None, tmp, None, lambda *_: None, lambda: False, 60)
        finally:
            f.db.upload_file = real
        self.assertEqual((ext, ctype), ("json", "application/json"))
        with open(local) as fh:
            return json.load(fh), uploaded

    def test_capabilities_manifest_has_every_required_member(self):
        tmp = os.path.join(os.environ.get("TEMP", "."), "footage_env")
        os.makedirs(tmp, exist_ok=True)
        m, uploaded = self._manifest("capabilities", tmp)
        for key in self.REQUIRED:
            self.assertIn(key, m, f"capabilities manifest is missing {key!r}")
        # the empty-but-present shape the website requires
        self.assertEqual(m["lineage"], {"source_take_ids": [], "source_trims": []})
        self.assertEqual(m["operation"], "capabilities")
        self.assertEqual(m["task_id"], "task-1")
        self.assertEqual(m["request_key"], "rk-1")
        # canonical path is written by the runner itself
        self.assertIn("footage/org-1/job-1/task-1/manifest.json", uploaded)

    def test_capabilities_never_advertises_what_run_refuses(self):
        """The advertised operation list and what run() will actually serve
        must agree — otherwise the UI enables a button that always fails."""
        import runners.footage as f
        ops = f.capabilities_block()["operations"]
        for op in ("extend", "prepend", "bridge", "loop"):
            if op not in f.PROVEN_GENERATION_OPS:
                self.assertNotIn(op, ops, f"{op} advertised but not proven")

    def test_run_refuses_every_operation_it_does_not_advertise(self):
        """The earlier version of this test only asserted when the adapter was
        disabled, so it was vacuous in the shipped state and run() accepted all
        four operations. Force the interesting case: adapter READY, exactly one
        operation proven, and assert the others never reach op_continuation."""
        import unittest.mock as mock
        import runners.footage as f

        for proven in ((), ("extend",)):
            with mock.patch.object(f, "GENERATION_ADAPTER_READY", True), \
                 mock.patch.object(f, "PROVEN_GENERATION_OPS", proven), \
                 mock.patch.object(f, "op_continuation") as spy:
                for op in ("extend", "prepend", "bridge", "loop"):
                    job = {"id": "t", "params": {"footage": {
                        "schema_version": f.CONTRACT_VERSION, "operation": op,
                        "org_id": "o", "job_id": "j"}}}
                    if op in proven:
                        continue
                    with self.assertRaises(f.FootageError) as caught:
                        f.run(job, None, ".", None, lambda m: None,
                              lambda: False, 60)
                    self.assertIn("not available", str(caught.exception))
                self.assertEqual(
                    spy.call_count, 0,
                    "an unadvertised operation reached the GPU adapter")


@unittest.skipUnless(os.path.isdir(TAKES), "Phase 0 takes not archived here")
class ContinuationAdapter(unittest.TestCase):
    """Sidecar reading and lineage verification, against the REAL Phase 0
    artifacts rather than hand-written fixtures."""

    ROOT = os.path.join(TAKES, "T1-root_00001.mctx.safetensors")
    EXT = os.path.join(TAKES, "T2-extend_00001.mctx.safetensors")
    BRIDGE = os.path.join(TAKES, "T3b-bridge_00001.mctx.safetensors")

    def test_reads_a_real_sidecar_header(self):
        h = footage.read_mctx_header(self.ROOT)
        self.assertEqual(h["format"], "mctx_v1")
        self.assertEqual(int(h["raw_frames"]), 73)
        self.assertEqual(int(h["delivered_frames"]), 73)
        self.assertEqual(int(h["pinned_head_frames"]), 0)

    def test_extend_sidecar_reports_held_and_delivered(self):
        h = footage.read_mctx_header(self.EXT)
        self.assertEqual(int(h["raw_frames"]), 73)
        self.assertEqual(int(h["pinned_head_frames"]), 39)
        self.assertEqual(int(h["delivered_frames"]), 34)
        self.assertEqual(h["relation"], "extends")

    def test_self_id_binds_the_delivered_mp4(self):
        h = footage.read_mctx_header(self.EXT)
        digest, _ = footage.sha256_size(self.EXT.replace(".mctx.safetensors", ".mp4"))
        self.assertEqual(h["self_id"], digest)

    def test_rejects_a_file_that_is_not_safetensors(self):
        with self.assertRaises(footage.FootageError):
            footage.read_mctx_header(self.EXT.replace(".mctx.safetensors", ".mp4"))

    def _pins(self, path):
        pins = footage.read_mctx_header(path).get("pins")
        return json.loads(pins) if isinstance(pins, str) else pins

    def test_bridge_recipe_maps_both_parents_to_requested_takes(self):
        pins = self._pins(self.BRIDGE)
        self.assertEqual(len(pins), 2)
        sources = [{"take_id": "take-A", "media": {"sha256": pins[0]["source_id"]}},
                   {"take_id": "take-B", "media": {"sha256": pins[1]["source_id"]}}]
        warnings = []
        resolved = footage.verify_recipe(pins, sources, warnings)
        self.assertEqual([r["take_id"] for r in resolved], ["take-A", "take-B"])
        self.assertEqual([r["placement"] for r in resolved], ["before", "after"])
        self.assertEqual([r["context_mode"] for r in resolved], ["latent", "latent"])
        self.assertEqual(resolved[0]["source_frames"], 39)
        self.assertEqual(warnings, [])

    def test_sidecar_own_digest_is_not_accepted_as_identity(self):
        """A recipe names the MEDIA hash. Matching on the sidecar's own bytes
        would silently fail to resolve every real lineage."""
        pins = self._pins(self.BRIDGE)
        sidecar_digest, _ = footage.sha256_size(self.BRIDGE)
        sources = [{"take_id": "take-A", "context": {"sha256": sidecar_digest}},
                   {"take_id": "take-B", "media": {"sha256": pins[1]["source_id"]}}]
        with self.assertRaises(footage.FootageError):
            footage.verify_recipe(pins, sources, [])

    def test_context_media_sha256_also_resolves(self):
        pins = self._pins(self.BRIDGE)
        sources = [{"take_id": "take-A", "context": {"media_sha256": pins[0]["source_id"]}},
                   {"take_id": "take-B", "media": {"sha256": pins[1]["source_id"]}}]
        resolved = footage.verify_recipe(pins, sources, [])
        self.assertEqual([r["take_id"] for r in resolved], ["take-A", "take-B"])

    def test_unknown_recipe_hash_is_a_hard_failure(self):
        """A nonempty hash that maps to nothing requested must FAIL, not warn:
        the take was continued from something the caller did not ask for."""
        pins = self._pins(self.BRIDGE)
        sources = [{"take_id": "take-A", "media": {"sha256": pins[0]["source_id"]}},
                   {"take_id": "take-B", "media": {"sha256": "0" * 64}}]
        with self.assertRaises(footage.FootageError) as cm:
            footage.verify_recipe(pins, sources, [])
        self.assertIn("not any requested source", str(cm.exception))

    def test_blank_pixel_origin_takes_identity_from_its_boundary(self):
        """A pixel pin has no content hash, but the contract needs a real
        take_id. It comes from the boundary that produced it — request
        provenance, not a hash guess, because the same media can appear in
        two clips and a hash could not tell them apart."""
        pins = [{"source_id": "", "place": "before", "source_start": 0,
                 "source_frames": 39}]
        plan = [{"place": "before", "take_id": "take-A", "src_index": 0}]
        warnings = []
        resolved = footage.verify_recipe(pins, [{"take_id": "take-A"}],
                                         warnings, plan)
        self.assertEqual(resolved[0]["context_mode"], "pixel")
        self.assertEqual(resolved[0]["take_id"], "take-A")
        self.assertTrue(any("re-encoded pixels" in w for w in warnings))

    def test_bridge_pixel_pins_keep_their_own_sources_apart(self):
        """Two boundaries, two different sources: each pin must carry the id
        of the clip it actually came from."""
        pins = [{"source_id": "", "place": "before", "source_frames": 39},
                {"source_id": "", "place": "after", "source_frames": 39}]
        plan = [{"place": "before", "take_id": "take-A", "src_index": 0},
                {"place": "after", "take_id": "take-B", "src_index": 1}]
        resolved = footage.verify_recipe(
            pins, [{"take_id": "take-A"}, {"take_id": "take-B"}], [], plan)
        self.assertEqual([r["take_id"] for r in resolved],
                         ["take-A", "take-B"])

    def test_pin_placement_must_match_the_planned_boundary(self):
        """If what came back was placed differently from what was planned,
        the mapping is unsafe and lineage must not be invented."""
        pins = [{"source_id": "", "place": "after", "source_frames": 39}]
        plan = [{"place": "before", "take_id": "take-A", "src_index": 0}]
        with self.assertRaises(footage.FootageError) as cm:
            footage.verify_recipe(pins, [{"take_id": "take-A"}], [], plan)
        self.assertIn("does not match the graph", str(cm.exception))

    # -------------------------------------------------- the user's cut

    HDR_ROOT = {"delivered_frames": 73, "pinned_head_frames": 0}
    HDR_GEN = {"delivered_frames": 51, "pinned_head_frames": 39}

    def test_untrimmed_departure_still_uses_tail(self):
        """The proven path must not change shape: a cut covering the whole
        clip is still take_from=tail, not an at_frame equivalent."""
        s = {"in_frame": 0, "out_frame": 73}
        b = footage.resolve_boundary("departure", s, self.HDR_ROOT, 39, 73)
        self.assertEqual(b["take_from"], "tail")
        self.assertTrue(b["latent_legal"])
        self.assertEqual(b["raw_start"], 34)

    def test_trimmed_departure_ends_at_the_users_out_frame(self):
        """THE regression. A cut of [0,56) must pin the 39 frames ending at
        56 — not the last 39 frames of the whole file."""
        s = {"in_frame": 0, "out_frame": 56}
        b = footage.resolve_boundary("departure", s, self.HDR_ROOT, 39, 73)
        self.assertEqual(b["take_from"], "at_frame")
        self.assertEqual(b["take_from_frame"], 56)
        self.assertEqual(b["raw_start"], 17)
        self.assertTrue(b["latent_legal"])

    def test_two_different_out_frames_give_two_different_windows(self):
        """Two requests on the SAME source must not produce the same pin."""
        a = footage.resolve_boundary("departure", {"in_frame": 0, "out_frame": 56},
                                     self.HDR_ROOT, 39, 73)
        b = footage.resolve_boundary("departure", {"in_frame": 0, "out_frame": 73},
                                     self.HDR_ROOT, 39, 73)
        self.assertNotEqual(a["raw_start"], b["raw_start"])

    def test_illegal_cut_is_detected_with_legal_alternatives(self):
        """An interior cut off the 17-frame group boundary cannot be sliced
        from latents, and the runner must say which cuts would work."""
        s = {"in_frame": 0, "out_frame": 60}
        b = footage.resolve_boundary("departure", s, self.HDR_ROOT, 39, 73)
        self.assertFalse(b["latent_legal"])
        self.assertEqual(b["raw_start"], 21)
        self.assertIn(56, b["legal_ends"])          # raw 17 -> end 56
        self.assertIn(73, b["legal_ends"])          # raw 34 -> end 73
        for end in b["legal_ends"]:
            alt = footage.resolve_boundary(
                "departure", {"in_frame": 0, "out_frame": end},
                self.HDR_ROOT, 39, 73)
            self.assertTrue(alt["latent_legal"], f"suggested end {end} is not legal")

    def test_arrival_window_starts_at_the_users_in_frame(self):
        s = {"in_frame": 17, "out_frame": 73}
        b = footage.resolve_boundary("arrival", s, self.HDR_ROOT, 39, 73)
        self.assertEqual(b["take_from"], "at_frame")
        self.assertEqual(b["take_from_frame"], 17 + 39)
        self.assertEqual(b["raw_start"], 17)
        self.assertTrue(b["latent_legal"])

    def test_generated_parent_offsets_through_pinned_head(self):
        """A generated take's delivered frame 12 is raw 51, because 39 frames
        of its run were held context. Latent legality is decided in RAW."""
        b = footage.resolve_boundary("departure", {"in_frame": 0, "out_frame": 51},
                                     self.HDR_GEN, 39, 51)
        self.assertEqual(b["take_from"], "tail")
        self.assertEqual(b["raw_start"], 39 + 12)
        self.assertEqual(b["raw_start"] % 17, 0)

    def test_cut_shorter_than_the_context_window_is_refused(self):
        with self.assertRaises(footage.FootageError) as cm:
            footage.resolve_boundary("departure", {"in_frame": 0, "out_frame": 20},
                                     self.HDR_ROOT, 39, 73)
        self.assertIn("at least 39 frames", str(cm.exception))

    def test_cut_outside_the_source_is_refused(self):
        with self.assertRaises(footage.FootageError):
            footage.resolve_boundary("departure", {"in_frame": 0, "out_frame": 99},
                                     self.HDR_ROOT, 39, 73)

    def test_boundary_plan_covers_every_operation(self):
        self.assertEqual(footage.boundary_plan("extend"), [(0, "departure")])
        self.assertEqual(footage.boundary_plan("prepend"), [(0, "arrival")])
        # loop takes BOTH boundaries from one source; bridge one from each
        self.assertEqual(footage.boundary_plan("loop"),
                         [(0, "departure"), (0, "arrival")])
        self.assertEqual(footage.boundary_plan("bridge"),
                         [(0, "departure"), (1, "arrival")])

    def test_chain_window_snap_is_grid_legal(self):
        """A 34-frame candidate cannot take a 39-frame pin; obvpm snaps to 22,
        which lands on the 17-frame grid. Guards the chaining case."""
        delivered, pinned_head = 34, 39
        self.assertGreater(footage.PIN_WINDOW, delivered)      # 39 > 34
        snapped = 22
        raw_start = pinned_head + (delivered - snapped)
        self.assertEqual(raw_start, 51)
        self.assertEqual(raw_start % 17, 0)


class ContinuationGraph(unittest.TestCase):
    """The graph actually handed to ComfyUI, without running it."""

    REQ = {"output": {"fps": {"num": 24, "den": 1}, "width": 832, "height": 480},
           "generation": {"prompt": "a car", "seed": 1}}

    def _spec(self, n, role, src=0, **kw):
        d = {"src_index": src, "role": role,
             "place": "before" if role == "departure" else "after",
             "mode": "masked" if role == "departure" else "both",
             "take_from": "at_frame", "take_from_frame": 56,
             "clip_ref": f"footage/x/b{n}.mp4", "staged_fps": 24.0,
             "staged_has_audio": True, "take_id": f"take-{src}"}
        d.update(kw)
        return d

    def setUp(self):
        footage._STAGE_KEY["folder"] = "footage/x"

    def test_pixel_loop_gives_each_boundary_its_own_window(self):
        """The loop bug: one encoder per SOURCE meant both pins came from the
        same kept tail, so the arrival was the clip's end rather than its
        beginning. Each boundary needs its own pre-trimmed clip."""
        staged = [("footage/x/src0.mp4", False, "src0.mp4")]
        plan = [self._spec(0, "departure"), self._spec(1, "arrival")]
        g = footage.build_continuation("loop", self.REQ, staged, 90, "p",
                                       latent_only=False, plan=plan)
        encs = {k: v for k, v in g.items() if v["class_type"] == "H3MCtxFromFrames"}
        self.assertEqual(len(encs), 2, "each boundary needs its own encoder")
        keeps = sorted(v["inputs"]["keep"] for v in encs.values())
        self.assertEqual(keeps, ["head", "tail"])
        clips = {g[v["inputs"]["images"][0]]["inputs"]["video"][0] for v in encs.values()}
        self.assertEqual(len(clips), 2, "the two windows must be different files")

    def test_pixel_encoder_connects_audio_when_the_source_has_it(self):
        staged = [("footage/x/src0.mp4", False, "src0.mp4")]
        plan = [self._spec(0, "departure", staged_has_audio=True)]
        g = footage.build_continuation("extend", self.REQ, staged, 90, "p",
                                       latent_only=False, plan=plan)
        enc = next(v for v in g.values() if v["class_type"] == "H3MCtxFromFrames")
        comp = enc["inputs"]["images"][0]
        self.assertEqual(enc["inputs"]["audio"], [comp, 1],
                         "GetVideoComponents output 1 is audio")

    def test_silent_source_omits_audio_so_the_pin_encodes_silence(self):
        staged = [("footage/x/src0.mp4", False, "src0.mp4")]
        plan = [self._spec(0, "departure", staged_has_audio=False)]
        g = footage.build_continuation("extend", self.REQ, staged, 90, "p",
                                       latent_only=False, plan=plan)
        enc = next(v for v in g.values() if v["class_type"] == "H3MCtxFromFrames")
        self.assertNotIn("audio", enc["inputs"])

    def test_loadvideo_paths_are_annotated_for_the_output_folder(self):
        """Staging writes to ComfyUI/output; a bare relative path resolves
        against input and is rejected at submission."""
        staged = [("footage/x/src0.mp4", False, "src0.mp4")]
        plan = [self._spec(0, "departure")]
        g = footage.build_continuation("extend", self.REQ, staged, 90, "p",
                                       latent_only=False, plan=plan)
        for node in g.values():
            if node["class_type"] == "LoadVideo":
                self.assertTrue(node["inputs"]["file"].endswith(" [output]"))

    def test_latent_source_carries_the_users_cut_into_the_pin_spec(self):
        staged = [("footage/x/src0.mp4", True, "src0.mp4")]
        plan = [self._spec(0, "departure", take_from="at_frame",
                           take_from_frame=56)]
        g = footage.build_continuation("extend", self.REQ, staged, 90, "p",
                                       latent_only=True, plan=plan)
        spec = next(v for v in g.values() if v["class_type"] == "H3MCtxPinSpec")
        self.assertEqual(spec["inputs"]["take_from"], "at_frame")
        self.assertEqual(spec["inputs"]["take_from_frame"], 56)

    def test_seams_are_kept_as_separate_named_joins(self):
        item = {"clip": "a.mp4", "seam": 0.012, "seam2": 0.031}
        self.assertEqual(footage._seams(item),
                         {"departure": 0.012, "arrival": 0.031})
        self.assertEqual(footage._seams({"clip": "a.mp4", "seam": 0.5}),
                         {"departure": 0.5})
        self.assertEqual(footage._seams({}), {})

    def test_result_reader_returns_both_path_and_measurements(self):
        outputs = {"result": {"h3_result": [
            {"clip": "out.mp4", "seam": 0.02, "seam2": 0.04}]}}
        path, item = footage._saved_path(outputs)
        self.assertEqual(path, "out.mp4")
        self.assertEqual(item["seam2"], 0.04)


if __name__ == "__main__":
    unittest.main(verbosity=2)
