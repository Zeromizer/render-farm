"""worker/db.py under network trouble: a lost claim reply, a dropped upload, one client per
lane. Loads db.py on its own with fakes for supabase and friends, so it runs on a bare
Python and next to tests that stub the db module."""
import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

WORKER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TransportError(Exception):
    pass


class ProtocolError(Exception):
    pass


def load_db():
    fakes = {
        "supabase": types.SimpleNamespace(ClientOptions=lambda **kw: kw, create_client=mock.Mock(side_effect=lambda *a, **kw: object())),
        "httpx": types.SimpleNamespace(TransportError=TransportError),
        "httpcore": types.SimpleNamespace(ProtocolError=ProtocolError),
        "config": types.SimpleNamespace(SUPABASE_URL="u", SUPABASE_SERVICE_KEY="k", BUCKET="renders", SIGNED_URL_SECONDS=60),
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


if __name__ == "__main__":
    unittest.main()
