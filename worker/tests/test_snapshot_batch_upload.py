"""Storyboard batch regressions: slide/timestamp binding from the CLI's file
names, and the upload step that must never let a half-uploaded batch end up
done. Storage and the database are faked; ffmpeg/Chrome are not involved.

    cd worker && ..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -t .
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners import hyperframes  # noqa: E402
import proc  # noqa: E402

try:
    import render_worker  # noqa: E402  (opens worker.log, needs ../.env for the db client)
except Exception as exc:  # noqa: BLE001
    render_worker = None
    _RW_ERR = exc


class OrderSnapshots(unittest.TestCase):
    NAMES = ["frame-00-at-0s.png", "frame-01-at-1s.png", "frame-02-at-2.5s.png"]

    def test_binds_by_frame_index_and_checks_timestamp(self):
        out = hyperframes.order_snapshots(list(reversed(self.NAMES)), [0, 1, 2.5])
        self.assertEqual(out, self.NAMES)

    def test_decimal_names_do_not_confuse_order(self):
        names = ["frame-00-at-1.25s.png", "frame-01-at-1.5s.png", "frame-02-at-10s.png"]
        self.assertEqual(hyperframes.order_snapshots(sorted(names, reverse=True), [1.25, 1.5, 10]), names)

    def test_wrong_timestamp_is_an_error_not_a_silent_shift(self):
        with self.assertRaises(RuntimeError) as cm:
            hyperframes.order_snapshots(self.NAMES, [0, 1, 3])
        self.assertIn("slide 2", str(cm.exception))

    def test_count_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            hyperframes.order_snapshots(self.NAMES[:2], [0, 1, 2.5])
        with self.assertRaises(RuntimeError):
            hyperframes.order_snapshots(self.NAMES + ["frame-03-at-4s.png"], [0, 1, 2.5])

    def test_gap_in_indexes_rejected(self):
        with self.assertRaises(RuntimeError):
            hyperframes.order_snapshots(["frame-00-at-0s.png", "frame-02-at-2s.png"], [0, 2])

    def test_unknown_naming_falls_back_to_sorted_order(self):
        self.assertEqual(hyperframes.order_snapshots(["b.png", "a.png"], [0, 1]), ["b.png", "a.png"])

    def test_snapshot_manifest_carries_index_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text("original")

            def run(cmd, **kwargs):
                output = Path(cmd[cmd.index("--output") + 1])
                output.mkdir()
                for name in self.NAMES:
                    (output / name).write_bytes(b"png")
                (output / "contact-sheet.jpg").write_bytes(b"jpg")

            with patch.object(hyperframes.proc, "run_streaming", run):
                manifest, ext, ctype = hyperframes._snapshot(["hyperframes"], directory, "index.html",
                                                             {"snapshot_times": [0, 1, 2.5]}, directory,
                                                             type("HB", (), {})(), lambda _: None, {})
            self.assertEqual((ext, ctype), ("json", "application/json"))
            data = json.loads(Path(manifest).read_text())
            self.assertEqual(data["count"], 3)
            self.assertEqual([(s["index"], s["at"], Path(s["file"]).name) for s in data["snapshots"]],
                             [(0, 0, self.NAMES[0]), (1, 1, self.NAMES[1]), (2, 2.5, self.NAMES[2])])


@unittest.skipIf(render_worker is None, "render_worker not importable here")
class UploadBatch(unittest.TestCase):
    """render_worker.upload_snapshot_batch with a fake db.upload_file."""

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="batch-")
        self.files = []
        for i in range(3):
            p = os.path.join(self.td, f"frame-0{i}-at-{i}s.png")
            with open(p, "wb") as f:
                f.write(b"\x89PNG fake %d" % i)
            self.files.append(p)
        self.manifest = os.path.join(self.td, "snapshot-batch.json")
        with open(self.manifest, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "count": 3,
                       "snapshots": [{"index": i, "at": i, "file": p} for i, p in enumerate(self.files)]}, f)
        self.uploaded = []

    def _upload(self, remote, local, ctype):
        self.uploaded.append((remote, os.path.basename(local), ctype))
        return remote

    def test_success_rewrites_manifest_with_bucket_paths_only(self):
        with patch.object(render_worker.db, "upload_file", self._upload):
            batch = render_worker.upload_snapshot_batch("JID", self.manifest, [0, 1, 2], lambda: False)
        self.assertEqual([u[0] for u in self.uploaded],
                         ["outputs/JID-slide-0.png", "outputs/JID-slide-1.png", "outputs/JID-slide-2.png"])
        self.assertTrue(all(u[2] == "image/png" for u in self.uploaded))
        with open(self.manifest, encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk, batch)
        self.assertEqual(batch["job_id"], "JID")
        self.assertEqual(batch["count"], 3)
        self.assertEqual([s["at"] for s in batch["snapshots"]], [0, 1, 2])
        self.assertEqual([s["path"] for s in batch["snapshots"]],
                         ["outputs/JID-slide-0.png", "outputs/JID-slide-1.png", "outputs/JID-slide-2.png"])
        self.assertTrue(all("file" not in s and s["bucket"] == render_worker.config.BUCKET for s in batch["snapshots"]))
        self.assertNotIn(self.td.replace("\\", "\\\\"), json.dumps(batch))   # never a local path

    def test_upload_failure_raises_before_done(self):
        def flaky(remote, local, ctype):
            if remote.endswith("slide-1.png"):
                raise RuntimeError("storage 503")
            return self._upload(remote, local, ctype)
        with patch.object(render_worker.db, "upload_file", flaky):
            with self.assertRaises(RuntimeError):
                render_worker.upload_snapshot_batch("JID", self.manifest, [0, 1, 2], lambda: False)
        self.assertEqual(len(self.uploaded), 1)
        # the manifest on disk is untouched, so nothing downstream can mistake it for complete
        with open(self.manifest, encoding="utf-8") as f:
            self.assertIn("file", json.load(f)["snapshots"][0])

    def test_cancel_between_slides(self):
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] == 2          # canceled after slide 0 went up
        with patch.object(render_worker.db, "upload_file", self._upload):
            with self.assertRaises(proc.Canceled):
                render_worker.upload_snapshot_batch("JID", self.manifest, [0, 1, 2], cancel)
        self.assertEqual(len(self.uploaded), 1)

    def test_incomplete_batch_rejected(self):
        with patch.object(render_worker.db, "upload_file", self._upload):
            with self.assertRaises(RuntimeError) as cm:
                render_worker.upload_snapshot_batch("JID", self.manifest, [0, 1, 2, 3], lambda: False)
        self.assertIn("Incomplete", str(cm.exception))
        self.assertEqual(self.uploaded, [])

    def test_missing_slide_file_rejected(self):
        os.remove(self.files[2])
        with patch.object(render_worker.db, "upload_file", self._upload):
            with self.assertRaises(RuntimeError):
                render_worker.upload_snapshot_batch("JID", self.manifest, [0, 1, 2], lambda: False)
        self.assertEqual(len(self.uploaded), 2)

    def test_order_mismatch_rejected(self):
        with patch.object(render_worker.db, "upload_file", self._upload):
            with self.assertRaises(RuntimeError):
                render_worker.upload_snapshot_batch("JID", self.manifest, [0, 2, 1], lambda: False)


@unittest.skipIf(render_worker is None, "render_worker not importable here")
class RunJobNeverDoneOnPartialBatch(unittest.TestCase):
    """run_job end to end with the runner, clone and db faked: an upload failure or a
    cancel leaves the row failed/canceled through the main loop's handlers, never done."""

    def _run(self, upload_file, cancel_flags):
        job = {"id": "JID", "engine": "hyperframes", "repo_url": "-", "git_ref": "main",
               "params": {"output_kind": "still", "snapshot_times": [0, 1, 2]}}
        updates = []
        cancels = iter(cancel_flags)

        def runner(job, repo, work_dir, hb, log, cancel_check, timeout):
            files = []
            for i in range(3):
                p = os.path.join(work_dir, f"frame-0{i}-at-{i}s.png")
                with open(p, "wb") as f:
                    f.write(b"png")
                files.append(p)
            m = os.path.join(work_dir, "snapshot-batch.json")
            with open(m, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "snapshots": [{"index": i, "at": i, "file": p} for i, p in enumerate(files)]}, f)
            return m, "json", "application/json"

        class HB:
            progress = 0

            def __init__(self, jid): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with patch.dict(render_worker.RUNNERS, {"hyperframes": runner}), \
                patch.object(render_worker, "Heartbeat", HB), \
                patch.object(render_worker.config, "WORK_DIR", tempfile.mkdtemp(prefix="rw-")), \
                patch.object(render_worker.git_cache, "checkout", lambda *a, **k: tempfile.mkdtemp()), \
                patch.object(render_worker.db, "set_phase", lambda *a, **k: None), \
                patch.object(render_worker.db, "cancel_requested", lambda jid: next(cancels, False)), \
                patch.object(render_worker.db, "upload_file", upload_file), \
                patch.object(render_worker.db, "upload_output", lambda jid, p, ext, ct: f"outputs/{jid}.{ext}"), \
                patch.object(render_worker.db, "create_signed_url", lambda r: "https://signed"), \
                patch.object(render_worker.db, "update_job", lambda jid, fields: updates.append(fields)):
            try:
                render_worker.run_job(job)
            except (RuntimeError, proc.Canceled) as exc:
                return exc, updates
        return None, updates

    def test_success_marks_done_with_json_output(self):
        exc, updates = self._run(lambda r, l, c: r, [False, False, False])
        self.assertIsNone(exc)
        self.assertEqual(updates[-1]["status"], "done")
        self.assertEqual(updates[-1]["output_path"], "outputs/JID.json")

    def test_failed_slide_upload_never_done(self):
        def bad(remote, local, ctype):
            if remote.endswith("slide-2.png"):
                raise RuntimeError("upload failed")
            return remote
        exc, updates = self._run(bad, [False, False, False])
        self.assertIsInstance(exc, RuntimeError)
        self.assertFalse(any(u.get("status") == "done" for u in updates))

    def test_cancel_mid_batch_never_done(self):
        exc, updates = self._run(lambda r, l, c: r, [False, True])
        self.assertIsInstance(exc, proc.Canceled)
        self.assertFalse(any(u.get("status") == "done" for u in updates))


if __name__ == "__main__":
    unittest.main()
