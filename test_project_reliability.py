import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project_paths


class AtomicJsonTests(unittest.TestCase):
    def test_atomic_write_keeps_previous_value_as_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            project_paths.atomic_write_json(path, {"milestone_index": 3})
            project_paths.atomic_write_json(path, {"milestone_index": 4})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"milestone_index": 4})
            self.assertEqual(
                json.loads(path.with_name("progress.json.bak").read_text(encoding="utf-8")),
                {"milestone_index": 3},
            )

    def test_read_json_uses_backup_after_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text("{broken", encoding="utf-8")
            path.with_name("progress.json.bak").write_text(
                '{"milestone_index": 7}', encoding="utf-8"
            )
            self.assertEqual(project_paths.read_json_with_backup(path)["milestone_index"], 7)

    def test_atomic_write_does_not_overwrite_good_backup_with_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            backup = path.with_name("progress.json.bak")
            path.write_text("{broken", encoding="utf-8")
            backup.write_text('{"milestone_index": 7}', encoding="utf-8")

            project_paths.atomic_write_json(path, {"milestone_index": 8})

            self.assertEqual(json.loads(backup.read_text(encoding="utf-8"))["milestone_index"], 7)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["milestone_index"], 8)

    def test_validation_failure_also_uses_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text("[]", encoding="utf-8")
            path.with_name("progress.json.bak").write_text(
                '{"milestone_index": 7}', encoding="utf-8"
            )

            def require_object(value):
                if not isinstance(value, dict):
                    raise ValueError("expected object")
                return value

            self.assertEqual(
                project_paths.read_json_with_backup(path, validator=require_object)["milestone_index"],
                7,
            )


class CheckpointSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import train
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"training dependencies are not installed: {exc}") from exc
        cls.train = train

    def test_checkpoint_selection_uses_step_not_mtime(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            self.train, "CHECKPOINTS_DIR", Path(directory)
        ):
            older_steps = Path(directory) / "poke_stage_name_100_steps.zip"
            newer_steps = Path(directory) / "poke_stage_name_200_steps.zip"
            older_steps.touch()
            newer_steps.touch()
            os.utime(older_steps, (2_000_000_000, 2_000_000_000))

            self.assertEqual(self.train.latest_stage_checkpoint("stage_name"), os.fspath(newer_steps))

    def test_checkpoint_vecnormalize_path_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            self.train.checkpoint_vecnormalize_path("model.zip")


if __name__ == "__main__":
    unittest.main()
