"""Repair the broken forest milestone saves with the best validated replacements.

This does three things:
1. Backs up the current forest milestone files.
2. Installs the recovered Stage 9/10 forest states.
3. Removes the stale Stage 11 fixed state so training uses a fresh 05e hit state.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from project_paths import MILESTONES_DIR, PROGRESS_FILE, milestone_state_path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKUP_ROOT = MILESTONES_DIR / "_repair_backups"

REPLACEMENTS = {
    "05d_forest_south": PROJECT_ROOT / "forest_entrance_from_gate_fixed.state",
    "05e_forest_north": PROJECT_ROOT / "teleport_main_body.state",
}

BACKUP_ONLY = [
    "05f_gate_transition",
]

RESET_PROGRESS_INDEX = 10


def backup_path(run_dir: Path, source: Path) -> Path:
    return run_dir / source.name


def ensure_source_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required replacement state is missing: {path}")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BACKUP_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    for replacement in REPLACEMENTS.values():
        ensure_source_exists(replacement)

    manifest: list[str] = []

    tracked_targets = sorted(set(REPLACEMENTS) | set(BACKUP_ONLY))
    for name in tracked_targets:
        target = milestone_state_path(name)
        if target.exists():
            backup = backup_path(run_dir, target)
            shutil.copy2(target, backup)
            manifest.append(f"backup {target.name} -> {backup.name}")

    if PROGRESS_FILE.exists():
        progress_backup = backup_path(run_dir, PROGRESS_FILE)
        shutil.copy2(PROGRESS_FILE, progress_backup)
        manifest.append(f"backup {PROGRESS_FILE.name} -> {progress_backup.name}")

    for name, source in REPLACEMENTS.items():
        target = milestone_state_path(name)
        shutil.copy2(source, target)
        manifest.append(f"install {source.name} -> {target.name}")

    for name in BACKUP_ONLY:
        target = milestone_state_path(name)
        if target.exists():
            target.unlink()
            manifest.append(f"remove {target.name} (force fresh hit-state handoff)")

    PROGRESS_FILE.write_text(json.dumps({"milestone_index": RESET_PROGRESS_INDEX}))
    manifest.append(f"set progress.json milestone_index={RESET_PROGRESS_INDEX}")

    manifest_path = run_dir / "repair_manifest.txt"
    manifest_path.write_text("\n".join(manifest) + "\n")

    print(f"Forest milestone repair applied.")
    print(f"Backup: {run_dir}")
    for line in manifest:
        print(f"  {line}")


if __name__ == "__main__":
    main()
