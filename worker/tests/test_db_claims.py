"""worker/db.py under network trouble: a lost claim reply, a dropped upload, one client per
lane. Loads db.py on its own with fakes for supabase and friends, so it runs on a bare
Python and next to tests that stub the db module."""
import importlib.util
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

WORKER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TransportError(Exception):
    pass


class ProtocolError(Exception):
    pass


def load_db(projects=None):
    """projects: {name: (url, key)}; the default is the single Aimotion project, which is
    what a worker .env without the NBD pair gives (worker/config.py)."""
    projects = projects or {"aimotion": ("u", "k")}
    fakes = {
        "supabase": types.SimpleNamespace(ClientOptions=lambda **kw: kw, create_client=mock.Mock(side_effect=lambda *a, **kw: object())),
        "httpx": types.SimpleNamespace(TransportError=TransportError),
        "httpcore": types.SimpleNamespace(ProtocolError=ProtocolError),
        "config": types.SimpleNamespace(SUPABASE_URL="u", SUPABASE_SERVICE_KEY="k", BUCKET="renders", SIGNED_URL_SECONDS=60,
                                        STALE_MINUTES=5, DEFAULT_PROJECT="aimotion", PROJECTS=dict(projects)),
    }
    spec = importlib.util.spec_from_file_location("db_under_test", os.path.join(WORKER, "db.py"))
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fakes):
        spec.loader.exec_module(module)
    return module, fakes["supabase"]


class Table:
    """Enough of the query builder: the row lives here, and execute() can lose its reply."""

    def __init__(self, box):
        self.box = box
        self.fields = None

    def select(self, *a):
        return self

    def update(self, fields):
        self.fields = fields
        return self

    def eq(self, key, value):
        return self

    def __getattr__(self, name):
        return lambda *a, **kw: self

    @property
    def not_(self):
        return self

    def execute(self):
        box = self.box
        if self.fields is None:
            box["reads"] += 1
            if box["read_fails"] > 0:
                box["read_fails"] -= 1
                raise TransportError("read")
            rows = [box["row"]] if box["row"] else []
            return types.SimpleNamespace(data=rows)
        if box["landed"]:
            box["row"].update(self.fields, status="processing")
        if box["lost_reply"]:
            raise TransportError("Server disconnected")
        return types.SimpleNamespace(data=[box["row"]] if box["landed"] else [])


class Claim(unittest.TestCase):
    def setUp(self):
        self.db, _ = load_db()
        self.box = {"row": {"id": "j", "attempts": 0, "status": "pending"}, "landed": True, "lost_reply": False, "reads": 0, "read_fails": 0}
        self.db.sb = types.SimpleNamespace(table=lambda name: Table(self.box))
        p = mock.patch.object(self.db.time, "sleep")
        p.start()
        self.addCleanup(p.stop)

    def test_a_normal_claim(self):
        job = self.db.claim_conditional(include={"hyperframes"})
        self.assertEqual(job["id"], "j")

    def test_the_fetch_lane_names_its_own_first_phase(self):
        job = self.db.claim_conditional(include={"url_fetch"}, phase="downloading")
        self.assertEqual(job["phase"], "downloading")

    def test_a_lost_reply_on_a_landed_claim_is_still_ours(self):
        # 5af15c24: the row went to processing, no slot ran it, and reclaim failed it
        self.box["lost_reply"] = True
        job = self.db.claim_conditional(include={"hyperframes"})
        self.assertEqual((job["id"], job["status"]), ("j", "processing"))

    def test_a_lost_reply_when_the_claim_never_landed_claims_nothing(self):
        self.box.update(lost_reply=True, landed=False)
        self.assertIsNone(self.db.claim_conditional(include={"hyperframes"}))

    def test_another_workers_claim_is_not_ours(self):
        self.box["lost_reply"] = True
        real = Table.execute

        def theirs(table):
            if table.fields is not None:
                self.box["row"].update(status="processing", claimed_at="2026-10-31T05:49:11.840774+00:00")
                raise TransportError("Server disconnected")
            return real(table)
        with mock.patch.object(Table, "execute", theirs):
            self.assertIsNone(self.db.claim_conditional(include={"hyperframes"}))

    def test_the_time_compares_as_a_moment(self):
        self.box["row"].update(status="processing", claimed_at="2026-10-31T05:49:11.840774Z")
        self.assertEqual(self.db._claimed_at("j", "2026-10-31T05:49:11.840774+00:00")["id"], "j")


class Upload(unittest.TestCase):
    def setUp(self):
        self.db, _ = load_db()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(self.db.time, "sleep").start()
        handle, self.path = tempfile.mkstemp()
        os.write(handle, b"video")
        os.close(handle)
        self.addCleanup(os.remove, self.path)

    def bucket(self, upload):
        return types.SimpleNamespace(storage=types.SimpleNamespace(from_=lambda name: types.SimpleNamespace(upload=upload)))

    def test_a_timeout_is_retried_from_the_start_of_the_file(self):
        bodies = []

        def upload(remote, f, headers):
            bodies.append(f.read())
            if len(bodies) == 1:
                raise TransportError("ReadTimeout")
        self.db.sb = self.bucket(upload)
        self.assertEqual(self.db.upload_output("j", self.path, "mp4", "video/mp4"), "outputs/j.mp4")
        self.assertEqual(bodies, [b"video", b"video"])

    def test_it_gives_up_after_three(self):
        calls = []

        def upload(remote, f, headers):
            calls.append(1)
            raise TransportError("ReadTimeout")
        self.db.sb = self.bucket(upload)
        with self.assertRaises(TransportError):
            self.db.upload_file("r", self.path, "video/mp4")
        self.assertEqual(len(calls), 3)


class Clients(unittest.TestCase):
    def client_in(self, db, name):
        got = []
        t = threading.Thread(target=lambda: got.append(db.sb._client()), name=name)
        t.start()
        t.join()
        return got[0]

    def test_each_lane_has_its_own(self):
        db, _ = load_db()
        with mock.patch.object(db, "create_client", side_effect=lambda *a, **kw: object()):
            a, b, c = (self.client_in(db, n) for n in ("light-lane-0", "light-lane-1", "music-lane"))
        self.assertEqual(len({id(a), id(b), id(c)}), 3)

    def test_a_lane_keeps_its_client(self):
        db, _ = load_db()
        with mock.patch.object(db, "create_client", side_effect=lambda *a, **kw: object()):
            self.assertIs(self.client_in(db, "light-lane-0"), self.client_in(db, "light-lane-0"))

    def test_short_lived_threads_share_one(self):
        db, _ = load_db()
        with mock.patch.object(db, "create_client", side_effect=lambda *a, **kw: object()) as make:
            self.assertIs(self.client_in(db, "Thread-1"), self.client_in(db, "Thread-2"))
        self.assertEqual(make.call_count, 1)

    def test_clients_get_the_long_storage_timeout(self):
        db, _ = load_db()
        with mock.patch.object(db, "create_client", return_value=object()) as make:
            self.client_in(db, "music-lane")
        self.assertEqual(make.call_args.kwargs["options"]["storage_client_timeout"], db.STORAGE_TIMEOUT_SECONDS)


TWO = {"aimotion": ("https://aim.supabase.co", "ka"), "nbd": ("https://nbd.supabase.co", "kn")}


class Rpc:
    """Per-project fake client: claim_farm_job hands out that project's queue in order,
    and records which project each call reached."""

    def __init__(self, name, queue, calls, fail=False):
        self.name, self.queue, self.calls, self.fail = name, queue, calls, fail

    def rpc(self, fn, params=None):
        def execute():
            self.calls.append((self.name, fn))
            if self.fail:
                raise TransportError("down")
            if fn == "reclaim_stale_farm_jobs":
                return types.SimpleNamespace(data=2)
            return types.SimpleNamespace(data=[self.queue.pop(0)] if self.queue else [])
        return types.SimpleNamespace(execute=execute)


class Projects(unittest.TestCase):
    def wire(self, db, queues, down=()):
        """Point db.sb at a fake client per project, chosen by the thread's current project."""
        self.calls = []
        clients = {n: Rpc(n, list(q), self.calls, fail=n in down) for n, q in queues.items()}

        class Sb:
            def __getattr__(_, name):
                return getattr(clients[db.current_project()], name)
        db.sb = Sb()
        return clients

    def test_one_project_is_what_a_bare_env_gives(self):
        db, _ = load_db()
        self.assertEqual(db.projects(), ["aimotion"])
        self.wire(db, {"aimotion": [{"id": "a1"}]})
        job = db.claim_job()
        self.assertEqual(job, {"id": "a1", "_project": "aimotion"})
        self.assertEqual(db.claim_job(), None)
        self.assertEqual(self.calls, [("aimotion", "claim_farm_job")] * 2)

    def test_one_project_errors_propagate_as_before(self):
        db, _ = load_db()
        self.wire(db, {"aimotion": []}, down={"aimotion"})
        with self.assertRaises(TransportError):
            db.claim_job()

    def test_round_robin_alternates_while_both_have_work(self):
        db, _ = load_db(TWO)
        self.wire(db, {"aimotion": [{"id": f"a{i}"} for i in range(3)], "nbd": [{"id": f"n{i}"} for i in range(3)]})
        got = [db.claim_job()["_project"] for _ in range(6)]
        self.assertEqual(got, ["aimotion", "nbd"] * 3)

    def test_a_busy_project_does_not_starve_the_other(self):
        db, _ = load_db(TWO)
        self.wire(db, {"aimotion": [{"id": f"a{i}"} for i in range(10)], "nbd": [{"id": "n0"}]})
        first_two = {db.claim_job()["_project"], db.claim_job()["_project"]}
        self.assertEqual(first_two, {"aimotion", "nbd"})

    def test_an_empty_project_is_skipped(self):
        db, _ = load_db(TWO)
        self.wire(db, {"aimotion": [], "nbd": [{"id": "n0"}]})
        self.assertEqual(db.claim_job(), {"id": "n0", "_project": "nbd"})

    def test_one_project_down_does_not_stop_the_other(self):
        db, _ = load_db(TWO)
        self.wire(db, {"aimotion": [{"id": "a0"}], "nbd": []}, down={"nbd"})
        with mock.patch("builtins.print"):
            got = [db.claim_job() for _ in range(2)]
        self.assertIn({"id": "a0", "_project": "aimotion"}, got)

    def test_every_project_down_raises(self):
        db, _ = load_db(TWO)
        self.wire(db, {"aimotion": [], "nbd": []}, down={"aimotion", "nbd"})
        with mock.patch("builtins.print"), self.assertRaises(TransportError):
            db.claim_job()

    def test_a_claim_leaves_the_thread_on_its_project(self):
        db, _ = load_db(TWO)
        self.wire(db, {"aimotion": [], "nbd": [{"id": "n0"}]})
        db.claim_job()
        self.assertEqual(db.current_project(), "aimotion")

    def test_reclaim_runs_in_every_project(self):
        db, _ = load_db(TWO)
        self.wire(db, {"aimotion": [], "nbd": []})
        self.assertEqual(db.reclaim_stale(), 4)
        self.assertEqual(sorted(p for p, fn in self.calls if fn == "reclaim_stale_farm_jobs"), ["aimotion", "nbd"])

    def test_reclaim_survives_one_project_down(self):
        db, _ = load_db(TWO)
        self.wire(db, {"aimotion": [], "nbd": []}, down={"nbd"})
        self.assertEqual(db.reclaim_stale(), 2)

    def test_include_per_project_skips_a_project_with_nothing(self):
        db, _ = load_db(TWO)
        seen = []
        with mock.patch.object(db, "_claim_conditional",
                               side_effect=lambda inc, exc, *a: seen.append((db.current_project(), inc, exc))):
            db.claim_conditional(include={"aimotion": {"video_gen"}, "nbd": set()}, exclude={"aimotion": {"x"}})
        self.assertEqual(seen, [("aimotion", {"video_gen"}, {"x"})])

    def test_a_project_missing_from_the_exclude_dict_is_not_claimed(self):
        db, _ = load_db(TWO)
        seen = []
        with mock.patch.object(db, "_claim_conditional",
                               side_effect=lambda inc, exc, *a: seen.append(db.current_project())):
            db.claim_conditional(exclude={"aimotion": {"remotion"}})
        self.assertEqual(seen, ["aimotion"])

    def test_gating_survives_one_project_down(self):
        db, _ = load_db(TWO)

        def gated(capability=None):
            if db.current_project() == "nbd":
                raise TransportError("down")
            return {"url_fetch"}
        with mock.patch.object(db, "gated_engines", side_effect=gated):
            self.assertEqual(db.gated_engines_by_project(), {"aimotion": {"url_fetch"}})
            with mock.patch.object(db, "projects", return_value=["nbd"]), self.assertRaises(TransportError):
                db.gated_engines_by_project()

    def test_a_plain_set_applies_to_every_project(self):
        db, _ = load_db(TWO)
        seen = []
        with mock.patch.object(db, "_claim_conditional",
                               side_effect=lambda inc, exc, *a: seen.append((db.current_project(), inc, exc))):
            db.claim_conditional(include={"remotion"})
        self.assertEqual(seen, [("aimotion", {"remotion"}, ()), ("nbd", {"remotion"}, ())])

    def test_a_pending_check_does_not_move_the_rotation(self):
        db, _ = load_db(TWO)
        before = db._rotation()
        db._each(lambda name: {"x": 1} if name == "aimotion" else None, claim=False)
        self.assertEqual(db._rotation(), before)

    def test_unknown_project_fails_loudly(self):
        db, _ = load_db()
        with self.assertRaises(RuntimeError):
            db.use_project("nbd")
        self.assertEqual(db.job_project({"id": "x"}), "aimotion")


class ProjectClients(unittest.TestCase):
    def test_each_project_gets_its_own_client_and_credentials(self):
        db, _ = load_db(TWO)
        made = []
        with mock.patch.object(db, "create_client", side_effect=lambda url, key, **kw: made.append(url) or object()):
            a = db.sb._client()
            with db.project("nbd"):
                n = db.sb._client()
            again = db.sb._client()
        self.assertIsNot(a, n)
        self.assertIs(a, again)
        self.assertEqual(made, ["https://aim.supabase.co", "https://nbd.supabase.co"])

    def test_threads_do_not_share_a_current_project(self):
        db, _ = load_db(TWO)
        db.use_project("nbd")
        seen = []
        t = threading.Thread(target=lambda: seen.append(db.current_project()))
        t.start()
        t.join()
        self.assertEqual(seen, ["aimotion"])
        self.assertEqual(db.current_project(), "nbd")
        db.use_project("aimotion")


class HeartbeatProject(unittest.TestCase):
    def test_the_beat_thread_writes_to_the_jobs_project(self):
        db, _ = load_db(TWO)
        beats = []
        db.update_job = lambda jid, fields: beats.append(db.current_project())
        fake_config = types.SimpleNamespace(HEARTBEAT_SECONDS=0.01)
        spec = importlib.util.spec_from_file_location("heartbeat_under_test", os.path.join(WORKER, "heartbeat.py"))
        hb_mod = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"db": db, "config": fake_config,
                                           "queue_status": types.SimpleNamespace(annotate=lambda log: None)}):
            spec.loader.exec_module(hb_mod)
            with db.project("nbd"):
                with hb_mod.Heartbeat("j1"):
                    deadline = time.time() + 2
                    while not beats and time.time() < deadline:
                        time.sleep(0.01)
        self.assertTrue(beats)
        self.assertEqual(set(beats), {"nbd"})


if __name__ == "__main__":
    unittest.main()
