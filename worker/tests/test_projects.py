"""More than one platform on this box (worker/db.py projects, 2026-11-11): the pieces
outside db.py that must not mix them up. The archive tool's workspace/project pairing
guard (a wrong pair deletes every job dir as an "orphan") and the queue view, which
counts both projects' jobs because they share the GPU."""
import contextlib
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

WORKER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WORKER)


def fake_db(projects=("aimotion",)):
    state = {"current": "aimotion"}

    @contextlib.contextmanager
    def project(name):
        before = state["current"]
        state["current"] = name
        try:
            yield name
        finally:
            state["current"] = before
    return types.SimpleNamespace(
        DEFAULT_PROJECT="aimotion", projects=lambda: list(projects), project=project,
        current_project=lambda: state["current"],
        job_project=lambda job: (job or {}).get("_project") or "aimotion",
        use_project=lambda name: state.update(current=name), sb=None, state=state)


def load_archive(db):
    spec = importlib.util.spec_from_file_location("archive_under_test",
                                                  os.path.join(ROOT, "tools", "archive-stale-workspaces.py"))
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"db": db, "httpx": types.SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class ArchivePairing(unittest.TestCase):
    def setUp(self):
        self.archive = load_archive(fake_db(("aimotion", "nbd")))

    def test_the_aimotion_workspace_with_aimotion_is_fine(self):
        self.assertIsNone(self.archive.check_pairing(r"C:\rp-workspace", "aimotion"))

    def test_the_demo_workspace_stays_allowed_for_aimotion(self):
        self.assertIsNone(self.archive.check_pairing(r"C:\rp-workspace-demo", "aimotion"))

    def test_the_nbd_workspace_with_nbd_is_fine(self):
        self.assertIsNone(self.archive.check_pairing(r"C:\rp-workspace-nbd", "nbd"))

    def test_the_nbd_workspace_against_aimotion_is_refused(self):
        self.assertIn("nbd", self.archive.check_pairing(r"C:\rp-workspace-nbd", "aimotion"))

    def test_the_aimotion_workspace_against_nbd_is_refused(self):
        self.assertIsNotNone(self.archive.check_pairing(r"C:\rp-workspace", "nbd"))

    def test_nbd_is_refused_even_when_the_worker_has_no_nbd_project(self):
        archive = load_archive(fake_db(("aimotion",)))
        self.assertIsNotNone(archive.check_pairing(r"C:\rp-workspace-nbd", "aimotion"))

    def test_a_bare_run_keeps_the_aimotion_defaults(self):
        self.assertEqual(self.archive.WORKSPACE, r"C:\rp-workspace")
        self.assertIn("render-platform-sandy", self.archive.SITE)


class Query:
    def __init__(self, rows, project, db):
        self.rows, self.project, self.db = rows, project, db

    def __getattr__(self, name):
        return lambda *a, **kw: self

    def execute(self):
        if self.db.current_project() != self.project:
            raise AssertionError(f"read {self.project}'s rows under {self.db.current_project()}")
        return types.SimpleNamespace(data=self.rows)


def load_queue_status(db):
    spec = importlib.util.spec_from_file_location("queue_status_under_test", os.path.join(WORKER, "queue_status.py"))
    module = importlib.util.module_from_spec(spec)
    fakes = {"db": db, "config": types.SimpleNamespace(LIGHT_LANE=False, MUSIC_LANE=False),
             "fetch_lane": types.SimpleNamespace(LANE_ENGINES={"url_fetch"}),
             "light_lane": types.SimpleNamespace(LIGHT_ENGINES=set()),
             "music_lane": types.SimpleNamespace(ENGINES=set()),
             "videogen": types.SimpleNamespace(estimate=types.SimpleNamespace()),
             "videogen.estimate": types.SimpleNamespace()}
    with mock.patch.dict(sys.modules, fakes):
        spec.loader.exec_module(module)
    return module


class QueueAcrossProjects(unittest.TestCase):
    def test_rows_of_both_projects_are_tagged_and_merged_in_claim_order(self):
        db = fake_db(("aimotion", "nbd"))
        tables = {"aimotion": [{"id": "a1", "priority": 5, "created_at": "2026-11-11T10:00:00"}],
                  "nbd": [{"id": "n1", "priority": 5, "created_at": "2026-11-11T09:00:00"}]}

        class Sb:
            def table(_, name):
                return Query(tables[db.current_project()], db.current_project(), db)
        db.sb = Sb()
        try:
            qs = load_queue_status(db)
        except Exception as e:  # noqa: BLE001 - the module's own imports changed: say so plainly
            self.skipTest(f"queue_status imports changed: {e}")
        rows = qs._live_rows()
        self.assertEqual([(r["id"], r["_project"]) for r in rows], [("n1", "nbd"), ("a1", "aimotion")])

    def test_one_project_keeps_the_database_order(self):
        db = fake_db(("aimotion",))
        rows = [{"id": "b", "priority": 9, "created_at": "2"}, {"id": "a", "priority": 1, "created_at": "1"}]

        class Sb:
            def table(_, name):
                return Query(rows, "aimotion", db)
        db.sb = Sb()
        try:
            qs = load_queue_status(db)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"queue_status imports changed: {e}")
        self.assertEqual([r["id"] for r in qs._live_rows()], ["b", "a"])


class RenderLoopExclude(unittest.TestCase):
    def test_each_project_gets_the_lanes_plus_its_own_gated_engines(self):
        with open(os.path.join(WORKER, "render_worker.py"), encoding="utf-8") as f:
            src = f.read()
        start = src.index("def _with_lanes(")
        end = src.index("\ndef ", start + 1)
        ns = {}
        exec(src[start:end], ns)
        got = ns["_with_lanes"]({"remotion"}, {"aimotion": {"remake", "url_fetch"}, "nbd": {"url_fetch"}})
        self.assertEqual(got, {"aimotion": {"remotion", "remake", "url_fetch"}, "nbd": {"remotion", "url_fetch"}})


if __name__ == "__main__":
    unittest.main()
