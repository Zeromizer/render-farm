"""Repair join (profile continuation-gap-v1): the capability gate and every
fail-closed check that must run before the GPU is booked.

CPU only, no ComfyUI, no network. The GPU path is never reached: tests that
drive op_continuation stop at build_continuation.
"""
import copy
import os
import sys
import tempfile
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners import footage as f  # noqa: E402

A_SHA, B_SHA = "a" * 64, "b" * 64
A_CTX, B_CTX = "c" * 64, "d" * 64


def request():
    return {
        "schema_version": f.CONTRACT_VERSION, "operation": f.GAP_OPERATION,
        "org_id": "o", "job_id": "j", "request_key": "rk",
        "sources": [
            {"take_id": "take-A",
             "media": {"bucket": "renders", "path": "a.mp4", "sha256": A_SHA},
             "context": {"bucket": "renders", "path": "a.mctx.safetensors",
                         "sha256": A_CTX, "media_sha256": A_SHA},
             "in_frame": 0, "out_frame": 51},
            {"take_id": "take-B",
             "media": {"bucket": "renders", "path": "b.mp4", "sha256": B_SHA},
             "context": {"bucket": "renders", "path": "b.mctx.safetensors",
                         "sha256": B_CTX, "media_sha256": B_SHA},
             "in_frame": 12, "out_frame": 51}],
        "output": {"fps": {"num": 24, "den": 1}, "width": 832, "height": 480},
        "generation": {"profile": f.GAP_PROFILE, "prompt": "continue the rotation",
                       "negative_prompt": "", "seed": 77, "new_frames": 12,
                       "context_mode": "latent", "resolution": "480p"},
        "repair_join": {"sequence_id": "seq-1", "sequence_revision": 3,
                        "clip_instance_ids": ["ci-A", "ci-B"],
                        "original_trims": [{"in_frame": 0, "out_frame": 51},
                                           {"in_frame": 0, "out_frame": 51}]}}


def header(self_id, relation="extends", parent_id=None, **over):
    h = {"format": "mctx_v1", "self_id": self_id, "relation": relation,
         "parent_id": parent_id, "raw_frames": "90", "pinned_head_frames": "39",
         "pinned_tail_frames": "0", "delivered_frames": "51"}
    h.update(over)
    return h


def info(**over):
    i = {"frame_count": 51, "fps": {"num": 24, "den": 1}, "width": 832,
         "height": 480, "has_audio": True}
    i.update(over)
    return i


def plan():
    return [f.resolve_boundary("departure", request()["sources"][0], header(A_SHA),
                               f.PIN_WINDOW, 51, f.GAP_FPS, f.GAP_FPS),
            f.resolve_boundary("arrival", request()["sources"][1],
                               header(B_SHA, parent_id=A_SHA), f.PIN_WINDOW, 51,
                               f.GAP_FPS, f.GAP_FPS)]


DIGESTS = {"a.mp4": A_SHA, "b.mp4": B_SHA, "a.mctx": A_CTX, "b.mctx": B_CTX}


def check(req=None, headers=None, infos=None, pl=None, digests=None):
    req = req or request()
    d = dict(DIGESTS, **(digests or {}))
    f.check_gap_sources(
        req["sources"], ["a.mp4", "b.mp4"], ["a.mctx", "b.mctx"],
        headers or [header(A_SHA, relation="extends", parent_id="root"),
                    header(B_SHA, parent_id=A_SHA)],
        infos or [info(), info()], pl or plan(), digest=lambda p: d[p])


class Capability(unittest.TestCase):
    def test_shipped_state_offers_exactly_this_one_profile(self):
        """Released: the profile is offered, and ONLY it — general operations,
        limits and 768p are exactly what they were before."""
        self.assertEqual(f.OFFERED_GENERATION_PROFILES, (f.GAP_PROFILE,))
        self.assertEqual(f.PROVEN_GENERATION_OPS, ("extend", "prepend"))
        with mock.patch.object(f, "_comfy_identity", return_value=("0.37.0", "c", "o")):
            cap = f.capabilities_block()
        self.assertEqual(cap["operations"],
                         ["capabilities", "probe", "assemble", "extend", "prepend"])
        self.assertEqual(list(cap["generation_profiles"]), [f.GAP_PROFILE])
        self.assertEqual(cap["max_generation_frames"], {"480p": 90, "768p": 0})
        self.assertEqual(sorted(cap["generation_limits"]), ["extend", "prepend"])

    def test_withdrawn_state_offers_no_profile_and_no_new_key(self):
        """Emptying the flag restores the pre-release manifest byte-for-byte."""
        with mock.patch.object(f, "_comfy_identity", return_value=("0.37.0", "c", "o")), \
             mock.patch.object(f, "OFFERED_GENERATION_PROFILES", ()):
            cap = f.capabilities_block()
        self.assertNotIn("generation_profiles", cap)
        self.assertEqual(cap["operations"],
                         ["capabilities", "probe", "assemble", "extend", "prepend"])

    def test_offered_profile_is_advertised_without_widening_operations(self):
        with mock.patch.object(f, "_comfy_identity", return_value=("0.37.0", "c", "o")), \
             mock.patch.object(f, "OFFERED_GENERATION_PROFILES", (f.GAP_PROFILE,)):
            cap = f.capabilities_block()
        self.assertEqual(cap["operations"],
                         ["capabilities", "probe", "assemble", "extend", "prepend"])
        p = cap["generation_profiles"][f.GAP_PROFILE]
        self.assertEqual(p["operation"], "repair_join")
        self.assertEqual((p["sampled_frames"], p["held_prefix_frames"],
                          p["new_frames"], p["held_suffix_frames"]), (90, 39, 12, 39))
        self.assertEqual(p["worker_cuts"], [{"in_frame": 0, "out_frame": 51},
                                            {"in_frame": 12, "out_frame": 51}])
        self.assertEqual(p["replaces"], {"source": 1, "in_frame": 0, "out_frame": 12})
        self.assertEqual(p["pin_modes"], {"before": "both", "after": "both"})
        self.assertEqual(p["seed"], 77)

    def test_no_profile_without_obvpm(self):
        with mock.patch.object(f, "_comfy_identity", return_value=("0.37.0", "c", None)), \
             mock.patch.object(f, "OFFERED_GENERATION_PROFILES", (f.GAP_PROFILE,)):
            self.assertNotIn("generation_profiles", f.capabilities_block())

    def _run(self, req):
        job = {"id": "t", "params": {"footage": req}}
        return f.run(job, None, ".", None, lambda m: None, lambda: False, 60)

    def test_run_refuses_repair_join_while_withdrawn(self):
        with mock.patch.object(f, "op_continuation") as spy, \
             mock.patch.object(f, "OFFERED_GENERATION_PROFILES", ()), \
             mock.patch.object(f, "_comfy_identity", return_value=("0.37.0", "c", "o")):
            with self.assertRaises(f.FootageError) as cm:
                self._run(request())
        self.assertIn("not available", str(cm.exception))
        self.assertEqual(spy.call_count, 0)

    def test_bridge_with_a_profile_field_is_still_general_bridge_and_refused(self):
        req = request()
        req["operation"] = "bridge"
        with mock.patch.object(f, "op_continuation") as spy, \
             mock.patch.object(f, "OFFERED_GENERATION_PROFILES", (f.GAP_PROFILE,)):
            with self.assertRaises(f.FootageError):
                self._run(req)
        self.assertEqual(spy.call_count, 0)

    def test_offered_and_valid_reaches_the_adapter_with_the_profile(self):
        with mock.patch.object(f, "op_continuation", side_effect=RuntimeError("stop")) as spy, \
             mock.patch.object(f, "_comfy_identity", return_value=("0.37.0", "c", "o")), \
             mock.patch.object(f, "OFFERED_GENERATION_PROFILES", (f.GAP_PROFILE,)):
            with self.assertRaises(RuntimeError):
                self._run(request())
        self.assertEqual(spy.call_args.args[0], "bridge")
        self.assertEqual(spy.call_args.kwargs["profile"], f.GAP_PROFILE)

    def test_offered_but_invalid_never_reaches_the_adapter(self):
        req = request()
        req["generation"]["context_mode"] = "auto"
        with mock.patch.object(f, "op_continuation") as spy, \
             mock.patch.object(f, "_comfy_identity", return_value=("0.37.0", "c", "o")), \
             mock.patch.object(f, "OFFERED_GENERATION_PROFILES", (f.GAP_PROFILE,)):
            with self.assertRaises(f.FootageError):
                self._run(req)
        self.assertEqual(spy.call_count, 0)


class RequestValidation(unittest.TestCase):
    def test_the_profile_request_passes(self):
        f.validate_gap_request(request())

    def _refused(self, mutate, needle):
        req = request()
        mutate(req)
        with self.assertRaises(f.FootageError) as cm:
            f.validate_gap_request(req)
        self.assertIn(needle, str(cm.exception))

    def test_every_departure_from_the_profile_is_refused(self):
        S, G, O, R = "sources", "generation", "output", "repair_join"
        cases = [
            (lambda r: r[G].pop("profile"), "generation.profile"),
            (lambda r: r[G].__setitem__("profile", "continuation-gap-v2"), "generation.profile"),
            (lambda r: r[G].__setitem__("new_frames", 13), "new_frames"),
            (lambda r: r[G].__setitem__("new_frames", "12"), "new_frames"),
            (lambda r: r[G].__setitem__("context_mode", "auto"), "context_mode"),
            (lambda r: r[G].__setitem__("context_mode", "pixel"), "context_mode"),
            (lambda r: r[G].pop("resolution"), "resolution"),
            (lambda r: r[G].__setitem__("resolution", "768p"), "resolution"),
            (lambda r: r[G].__setitem__("seed", None), "seed"),
            (lambda r: r[G].__setitem__("seed", True), "seed"),
            (lambda r: r[G].__setitem__("seed", 78), "seed"),
            (lambda r: r[G].__setitem__("seed", 77.0), "seed"),
            (lambda r: r[G].pop("seed"), "seed"),
            (lambda r: r[G].__setitem__("prompt", "  "), "prompt"),
            (lambda r: r[O].__setitem__("fps", {"num": 30, "den": 1}), "output.fps"),
            (lambda r: r[O].pop("fps"), "output.fps"),
            (lambda r: r[O].__setitem__("width", 768), "832x480"),
            (lambda r: r.pop(O), "output"),
            (lambda r: r[S].pop(), "exactly 2"),
            (lambda r: r[S].append(copy.deepcopy(r[S][1])), "exactly 2"),
            (lambda r: r[S][0].__setitem__("out_frame", 50), "source 0 worker cut"),
            (lambda r: r[S][0].__setitem__("in_frame", 12), "source 0 worker cut"),
            (lambda r: r[S][1].__setitem__("in_frame", 0), "source 1 worker cut"),
            (lambda r: r[S][1].__setitem__("in_frame", 11), "source 1 worker cut"),
            (lambda r: r[S][1].__setitem__("out_frame", 50), "source 1 worker cut"),
            (lambda r: r[S][1].pop("out_frame"), "source 1 worker cut"),
            (lambda r: r[S][1].pop("context"), "paired context"),
            (lambda r: r[S][1]["context"].__setitem__("media_sha256", A_SHA), "does not name"),
            (lambda r: r[S][0]["media"].__setitem__("sha256", "xyz"), "sha256 hex"),
            (lambda r: r[S][1].pop("take_id"), "take_id"),
            (lambda r: r[S][1]["media"].__setitem__("sha256", A_SHA), "same media"),
            (lambda r: r.pop(R), "sequence_id"),
            (lambda r: r[R].__setitem__("sequence_revision", 0), "sequence_revision"),
            (lambda r: r[R].__setitem__("sequence_revision", "3"), "sequence_revision"),
            (lambda r: r[R].__setitem__("clip_instance_ids", ["ci", "ci"]), "clip_instance_ids"),
            (lambda r: r[R].__setitem__("clip_instance_ids", ["ci-A"]), "clip_instance_ids"),
            # B already destructively trimmed in the saved sequence
            (lambda r: r[R]["original_trims"][1].__setitem__("in_frame", 12), "original_trims"),
            (lambda r: r[R]["original_trims"].pop(), "original_trims"),
        ]
        for mutate, needle in cases:
            with self.subTest(needle=needle):
                self._refused(mutate, needle)


class SourceChecks(unittest.TestCase):
    def test_the_reviewed_pair_passes_and_both_pins_sit_at_raw_51(self):
        pl = plan()
        self.assertEqual([(p["take_from"], p["take_from_frame"], p["raw_start"])
                          for p in pl], [("tail", 51, 51), ("at_frame", 51, 51)])
        check(pl=pl)

    def _refused(self, needle, **kw):
        with self.assertRaises(f.FootageError) as cm:
            check(**kw)
        self.assertIn(needle, str(cm.exception))

    def test_bytes_must_match_the_declared_hashes(self):
        self._refused("media bytes", digests={"b.mp4": "e" * 64})
        self._refused("context bytes", digests={"a.mctx": "e" * 64})

    def test_lineage_must_be_a_direct_extend_of_A(self):
        A = header(A_SHA)
        self._refused("must be an extend", headers=[A, header(B_SHA, relation="bridges", parent_id=A_SHA)])
        self._refused("parent is not A", headers=[A, header(B_SHA, parent_id="f" * 64)])
        self._refused("parent is not A", headers=[A, header(B_SHA, parent_id=None)])
        self._refused("self_id", headers=[header("f" * 64), header(B_SHA, parent_id=A_SHA)])

    def test_context_geometry_must_be_raw_90_head_39_delivered_51(self):
        B = header(B_SHA, parent_id=A_SHA)
        for k, v in (("raw_frames", "73"), ("pinned_head_frames", "0"),
                     ("pinned_tail_frames", "39"), ("delivered_frames", "50"),
                     ("raw_frames", None)):
            with self.subTest(k=k, v=v):
                self._refused(k, headers=[header(A_SHA, **{k: v}), B])

    def test_media_must_be_native(self):
        for over, needle in (({"frame_count": 50}, "decodes 50"),
                             ({"fps": {"num": 30, "den": 1}}, "fps"),
                             ({"width": 1344, "height": 768}, "832x480"),
                             ({"has_audio": False}, "no audio")):
            with self.subTest(needle=needle):
                self._refused(needle, infos=[info(), info(**over)])

    def test_an_off_grid_window_is_refused_not_demoted_to_pixels(self):
        pl = plan()
        pl[1] = dict(pl[1], raw_start=50, latent_legal=False)
        self._refused("latent-legal", pl=pl)


class AdapterWiring(unittest.TestCase):
    """op_continuation with the profile, stopped at build_continuation."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        for i in (0, 1):
            open(os.path.join(self.tmp, f"stage{i:02d}.mctx.safetensors"), "wb").close()
        self.headers = {0: header(A_SHA), 1: header(B_SHA, parent_id=A_SHA)}

    def _drive(self, headers=None):
        hs = headers or self.headers
        locs = {0: "a.mp4", 1: "b.mp4"}
        digests = {"a.mp4": A_SHA, "b.mp4": B_SHA,
                   os.path.join(self.tmp, "stage00.mctx.safetensors"): A_CTX,
                   os.path.join(self.tmp, "stage01.mctx.safetensors"): B_CTX}
        stop = RuntimeError("stop before the GPU")
        with mock.patch.object(f, "_journal_last", return_value=(None, False)), \
             mock.patch.object(f, "_stage_pair",
                               side_effect=lambda s, w, i, log: (f"x/src{i}.mp4", True, locs[i])), \
             mock.patch.object(f, "probe_media", return_value=info()), \
             mock.patch.object(f, "read_mctx_header",
                               side_effect=lambda p: hs[int(p[-18])]), \
             mock.patch.object(f, "sha256_size", side_effect=lambda p: (digests[p], 1)), \
             mock.patch.object(f, "build_continuation", side_effect=stop) as build:
            with self.assertRaises(Exception) as cm:
                f.op_continuation("bridge", request(), "task-rj", self.tmp, None,
                                  lambda m: None, lambda: False, 60,
                                  profile=f.GAP_PROFILE)
        return cm.exception, build

    def test_valid_sources_reach_the_graph_with_both_pins_mode_both(self):
        exc, build = self._drive()
        self.assertEqual(str(exc), "stop before the GPU")
        op, _req, staged, length = build.call_args.args[:4]
        pl = build.call_args.kwargs["plan"]
        self.assertEqual((op, length), ("bridge", 90))
        self.assertTrue(all(has for _r, has, _l in staged), "a source was demoted to pixels")
        self.assertEqual([(p["place"], p["mode"], p["take_from"], p["take_from_frame"])
                          for p in pl],
                         [("before", "both", "tail", 51), ("after", "both", "at_frame", 51)])
        self.assertTrue(build.call_args.kwargs["latent_only"])

    def test_a_bad_lineage_never_builds_a_graph(self):
        exc, build = self._drive({0: header(A_SHA), 1: header(B_SHA, parent_id="f" * 64)})
        self.assertIsInstance(exc, f.FootageError)
        self.assertEqual(build.call_count, 0)

    def test_general_bridge_still_pins_its_departure_masked(self):
        """The profile's mode override must not leak into general bridge."""
        self.assertEqual(f.resolve_boundary("departure", request()["sources"][0],
                                            header(A_SHA), 39, 51)["mode"], "masked")


REAL_RUN = "C:/ComfyUI/output/footage/diag-moving-gap-seed77-both-run-1790130336"


def raw_pins():
    return [{"source_id": A_SHA, "source_kind": "clip", "source_start": 51,
             "source_frames": 39, "place": "before", "audio_window": 0, "mode": "both"},
            {"source_id": B_SHA, "source_kind": "clip", "source_start": 51,
             "source_frames": 39, "place": "after", "audio_window": 0, "mode": "both"}]


class Result(unittest.TestCase):
    PINS = [{"take_id": "take-A", "placement": "before", "context_mode": "latent"},
            {"take_id": "take-B", "placement": "after", "context_mode": "latent"}]
    HDR = {"raw_frames": "90", "pinned_head_frames": "39",
           "pinned_tail_frames": "39", "delivered_frames": "12",
           "width": "832", "height": "480", "fps": "24"}

    def _refused(self, hdr=None, raw=(), pins=None):
        with self.assertRaises(f.FootageError):
            f.verify_gap_result(hdr or self.HDR, raw_pins() if raw == () else raw,
                                pins or self.PINS, request()["sources"])

    def test_the_profile_candidate_is_accepted(self):
        f.verify_gap_result(self.HDR, raw_pins(), self.PINS, request()["sources"])

    def test_a_candidate_of_the_wrong_shape_is_not_returned(self):
        for hdr, pins in ((dict(self.HDR, delivered_frames="13"), self.PINS),
                          (dict(self.HDR, pinned_tail_frames="0"), self.PINS),
                          (dict(self.HDR, width="1344"), self.PINS),
                          (dict(self.HDR, fps="30"), self.PINS),
                          ({k: v for k, v in self.HDR.items() if k != "fps"}, self.PINS),
                          (self.HDR, self.PINS[::-1]),
                          (self.HDR, self.PINS[:1]),
                          (self.HDR, [self.PINS[0], dict(self.PINS[1], context_mode="pixel")])):
            with self.subTest(hdr=hdr, pins=pins):
                self._refused(hdr=hdr, pins=pins)

    @staticmethod
    def _mutated(i, k, v):
        raw = raw_pins()
        if v is None:
            raw[i].pop(k)
        else:
            raw[i][k] = v
        return raw

    def test_root_reported_mutations_are_refused_for_either_pin(self):
        """The 12 variants 24a7604 accepted, for A or B independently."""
        for i in (0, 1):
            for k, v in (("source_start", 0), ("source_frames", 1),
                         ("source_start", None), ("source_frames", None),
                         ("mode", "masked"), ("mode", None)):
                with self.subTest(pin=i, field=k, value=v):
                    self._refused(raw=self._mutated(i, k, v))

    def test_every_other_pin_departure_is_refused(self):
        for i, k, v in ((0, "source_kind", "image"), (1, "source_kind", None),
                        (0, "audio_window", 39), (1, "audio_window", None),
                        (0, "audio_window", False), (0, "source_start", "51"),
                        (1, "source_frames", 39.0), (1, "place", "before"),
                        (0, "place", None), (0, "source_id", B_SHA),
                        (1, "source_id", ""), (0, "mask_ramp_frames", 0)):
            with self.subTest(pin=i, field=k, value=v):
                self._refused(raw=self._mutated(i, k, v))
        for bad in ([], raw_pins()[:1], raw_pins() + raw_pins()[:1],
                    raw_pins()[::-1], None, "[]", [raw_pins()[0], "x"]):
            with self.subTest(pins=bad):
                self._refused(raw=bad)

    @unittest.skipUnless(os.path.isdir(REAL_RUN), "retained moving-gap run not on this host")
    def test_the_retained_reviewed_candidate_passes_and_a_moved_window_fails(self):
        import glob
        import json
        side = sorted(glob.glob(os.path.join(REAL_RUN, "bridge-*_00001.mctx.safetensors")))[0]
        hdr = f.read_mctx_header(side)
        pins = json.loads(hdr["pins"]) if isinstance(hdr["pins"], str) else hdr["pins"]
        src = request()["sources"]
        src[0]["media"]["sha256"] = "6c32485171204f87692351c73609e76ec78ff0e8466396776dd473c10441e793"
        src[1]["media"]["sha256"] = "8c5ddb9f32336736c9423ee733e7e061c8e0eb932c6ffda973d0edc5372c1f1c"
        warnings = []
        f.verify_gap_result(hdr, pins, f.verify_recipe(pins, src, warnings), src)
        self.assertEqual(warnings, [])
        for i in (0, 1):
            for k, v in (("source_start", 0), ("mode", "masked")):
                moved = [dict(p) for p in pins]
                moved[i][k] = v
                with self.subTest(pin=i, field=k):
                    with self.assertRaises(f.FootageError):
                        f.verify_gap_result(hdr, moved, f.verify_recipe(moved, src, []), src)

    def test_lineage_reports_saved_sequence_and_actual_worker_cuts(self):
        lin = f.gap_lineage(request())
        self.assertEqual(lin, {
            "profile": "continuation-gap-v1",
            "sequence_id": "seq-1", "sequence_revision": 3,
            "clip_instance_ids": ["ci-A", "ci-B"],
            "original_trims": [{"in_frame": 0, "out_frame": 51},
                               {"in_frame": 0, "out_frame": 51}],
            "worker_cuts": [{"clip_instance_id": "ci-A", "take_id": "take-A",
                             "in_frame": 0, "out_frame": 51},
                            {"clip_instance_id": "ci-B", "take_id": "take-B",
                             "in_frame": 12, "out_frame": 51}],
            "replaces": {"clip_instance_id": "ci-B", "in_frame": 0, "out_frame": 12}})


if __name__ == "__main__":
    unittest.main()
