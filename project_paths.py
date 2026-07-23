import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
ROM_PATH = PROJECT_ROOT / "yellow.gb"
START_STATE_PATH = PROJECT_ROOT / "start.state"
MILESTONES_DIR = PROJECT_ROOT / "milestones"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
PROGRESS_FILE = MILESTONES_DIR / "progress.json"


def ensure_runtime_dirs() -> None:
    for path in (MILESTONES_DIR, CHECKPOINTS_DIR, SCREENSHOTS_DIR, RECORDINGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def milestone_state_path(name: str) -> Path:
    return MILESTONES_DIR / f"{name}.state"


def latest_hit_state_path(milestone_name: str, env_id: int) -> Path:
    return MILESTONES_DIR / f"_{milestone_name}_latest_hit_{env_id}.state"


def latest_hit_glob(milestone_name: str) -> str:
    return str(MILESTONES_DIR / f"_{milestone_name}_latest_hit_*.state")


def atomic_write_json(path: Path, value: Any, *, keep_backup: bool = True) -> None:
    """Durably replace a JSON file without exposing a partially written file.

    The previous valid file is retained as ``<name>.bak``. This matters for the
    curriculum progress file: restarting from stage zero after a torn write can
    waste days of training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if keep_backup and path.is_file():
            # Never replace a known-good backup with a corrupt primary file.
            # A prior interrupted/manual write may have damaged the primary,
            # while read_json_with_backup still allowed the process to recover.
            try:
                with path.open("r", encoding="utf-8") as existing:
                    json.load(existing)
            except (OSError, json.JSONDecodeError):
                pass
            else:
                shutil.copy2(path, path.with_name(f"{path.name}.bak"))
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def read_json_with_backup(path: Path, validator: Callable[[Any], Any] | None = None) -> Any:
    """Read and optionally validate JSON, falling back to the backup on error."""
    path = Path(path)
    errors: list[tuple[Path, Exception]] = []
    for candidate in (path, path.with_name(f"{path.name}.bak")):
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return validator(value) if validator is not None else value
        except (OSError, TypeError, ValueError) as exc:
            errors.append((candidate, exc))

    details = "; ".join(f"{candidate}: {error}" for candidate, error in errors)
    raise ValueError(f"Could not read valid JSON from {path} or its backup ({details})")
