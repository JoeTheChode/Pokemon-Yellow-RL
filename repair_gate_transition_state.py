"""Promote the best recent 05e forest hit into the 05f gate-transition state.

This avoids getting stuck with the newest hit state when that state lands on the
frozen `(1, 0)` forest tile. We score recent hit states by a tiny mobility probe
and copy the best candidate into `milestones/05f_gate_transition.state`.
"""

from __future__ import annotations

import glob
import shutil
from dataclasses import dataclass
from pathlib import Path

from pyboy import PyBoy

from project_paths import ROM_PATH, latest_hit_glob, milestone_state_path


ADDR_MAP_ID = 0xD35D
ADDR_POS_A = 0xD361
ADDR_POS_B = 0xD362
ADDR_BATTLE_FLAG = 0xD057

ACTIONS = ["up", "down", "left", "right"]
SETTLE_TICKS = 30
TARGET_STATE = milestone_state_path("05f_gate_transition")


@dataclass
class Candidate:
    path: Path
    map_id: int
    pos_a: int
    pos_b: int
    battle_flag: int
    mobility: int

    @property
    def score(self) -> tuple[int, int, int]:
        preferred = 1 if (self.pos_a, self.pos_b) in {(2, 1), (1, 1)} else 0
        return (preferred, self.mobility, -int(self.path.stat().st_mtime))


def settle(pyboy: PyBoy) -> None:
    for _ in range(SETTLE_TICKS):
        pyboy.tick()


def probe_coords(state_path: Path) -> tuple[int, int, int, int]:
    pyboy = PyBoy(str(ROM_PATH), window="null")
    with open(state_path, "rb") as handle:
        pyboy.load_state(handle)
    settle(pyboy)
    result = (
        pyboy.memory[ADDR_MAP_ID],
        pyboy.memory[ADDR_POS_A],
        pyboy.memory[ADDR_POS_B],
        pyboy.memory[ADDR_BATTLE_FLAG],
    )
    pyboy.stop()
    return result


def probe_mobility(state_path: Path) -> int:
    coords = set()
    for action in ACTIONS:
        pyboy = PyBoy(str(ROM_PATH), window="null")
        with open(state_path, "rb") as handle:
            pyboy.load_state(handle)
        settle(pyboy)
        pyboy.button_press(action)
        pyboy.tick(8)
        pyboy.button_release(action)
        pyboy.tick(16)
        coords.add(
            (
                pyboy.memory[ADDR_MAP_ID],
                pyboy.memory[ADDR_POS_A],
                pyboy.memory[ADDR_POS_B],
            )
        )
        pyboy.stop()
    return len(coords)


def iter_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for raw_path in sorted(glob.glob(latest_hit_glob("05e_forest_north")), reverse=True):
        path = Path(raw_path)
        map_id, pos_a, pos_b, battle_flag = probe_coords(path)
        if map_id != 51 or battle_flag != 0:
            continue
        mobility = probe_mobility(path)
        if mobility <= 1:
            continue
        candidates.append(
            Candidate(
                path=path,
                map_id=map_id,
                pos_a=pos_a,
                pos_b=pos_b,
                battle_flag=battle_flag,
                mobility=mobility,
            )
        )
    return candidates


def main() -> None:
    candidates = iter_candidates()
    if not candidates:
        raise SystemExit("No viable 05e hit states were found.")

    best = max(candidates, key=lambda candidate: candidate.score)
    shutil.copy2(best.path, TARGET_STATE)
    print(
        "Installed best 05f gate-transition state:",
        f"{best.path.name} -> {TARGET_STATE.name}",
        f"map={best.map_id} a={best.pos_a} b={best.pos_b} mobility={best.mobility}",
    )


if __name__ == "__main__":
    main()
