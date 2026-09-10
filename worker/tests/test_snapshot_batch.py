import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners import hyperframes


class BatchSnapshots(unittest.TestCase):
    def test_one_command_and_ordered_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text("original")
            (root / "boards.html").write_text("boards")
            seen = []
            def run(cmd, **kwargs):
                seen.append(cmd)
                self.assertEqual((root / "index.html").read_text(), "boards")
                output = Path(cmd[cmd.index("--output") + 1]); output.mkdir()
                for name in ["frame-10.png", "frame-2.png", "frame-1.png"]:
                    (output / name).write_bytes(b"png")
            with patch.object(hyperframes.proc, "run_streaming", run):
                manifest, ext, _ = hyperframes._snapshot(["hyperframes"], directory, "boards.html", {"snapshot_times": [0, 1, 2]}, directory, type("HB", (), {})(), lambda _: None, {})
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0][seen[0].index("--at") + 1], "0,1,2")
            self.assertEqual(ext, "json")
            self.assertEqual([Path(s["file"]).name for s in json.loads(Path(manifest).read_text())["snapshots"]], ["frame-1.png", "frame-2.png", "frame-10.png"])
            self.assertEqual((root / "index.html").read_text(), "original")

    def test_rejects_ambiguous_times(self):
        for times in [[1, 0], [0, 0], [0, float("nan")], [0, True], [0]]:
            with self.assertRaises(ValueError):
                hyperframes._snapshot([], "", "", {"snapshot_times": times}, "", None, None, {})

    def test_restores_entry_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text("original")
            (root / "boards.html").write_text("boards")
            with patch.object(hyperframes.proc, "run_streaming", side_effect=RuntimeError("cancelled")):
                with self.assertRaises(RuntimeError):
                    hyperframes._snapshot([], directory, "boards.html", {"snapshot_times": [0, 1]}, directory, None, lambda _: None, {})
            self.assertEqual((root / "index.html").read_text(), "original")


if __name__ == "__main__":
    unittest.main()
