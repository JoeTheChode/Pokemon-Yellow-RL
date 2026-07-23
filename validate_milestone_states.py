"""Lightweight reachability check for curriculum save states.

This catches obviously broken milestone states before we spend GPU time on them.
It intentionally uses a cheap coordinate-state search, so it is a sanity check,
not a proof of global reachability.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

from pyboy import PyBoy

from project_paths import ROM_PATH, milestone_state_path


ADDR_MAP_ID = 0xD35D
ADDR_POS_A = 0xD361
ADDR_POS_B = 0xD362
ADDR_REPEL = 0xD078

FOREST_TARGETS = {
    "05d_forest_south": {50, 51},
    "05e_forest_north": {47},
    "05f_gate_transition": {47},
}


@dataclass(frozen=True)
class Node:
    map_id: int
    pos_a: int
    pos_b: int


def apply_action(pyboy: PyBoy, action: str, use_repel: bool) -> None:
    if use_repel:
        pyboy.memory[ADDR_REPEL] = 255
    pyboy.button_press(action)
    pyboy.tick(8)
    pyboy.button_release(action)
    pyboy.tick(16)
    if use_repel:
        pyboy.memory[ADDR_REPEL] = 255


def snapshot(pyboy: PyBoy) -> Node:
    mem = pyboy.memory
    return Node(mem[ADDR_MAP_ID], mem[ADDR_POS_A], mem[ADDR_POS_B])


def settle(pyboy: PyBoy, settle_ticks: int, use_repel: bool) -> None:
    for _ in range(settle_ticks):
        if use_repel:
            pyboy.memory[ADDR_REPEL] = 255
        pyboy.tick()


def search_state(name: str, max_depth: int, use_repel: bool, settle_ticks: int) -> None:
    state_path = milestone_state_path(name)
    pyboy = PyBoy(str(ROM_PATH), window="null")
    actions = ("up", "down", "left", "right")

    with open(state_path, "rb") as f:
        seed_bytes = f.read()

    start = None
    seen: dict[Node, tuple[str, ...]] = {}
    queue: deque[tuple[str, ...]] = deque([()])
    targets = FOREST_TARGETS.get(name, set())
    hits: list[tuple[Node, tuple[str, ...]]] = []

    while queue:
        seq = queue.popleft()
        if len(seq) > max_depth:
            continue

        with open(state_path, "rb") as f:
            pyboy.load_state(f)
        settle(pyboy, settle_ticks, use_repel)
        for step in seq:
            apply_action(pyboy, step, use_repel)

        node = snapshot(pyboy)
        if start is None:
            start = node

        if node in seen:
            continue

        seen[node] = seq

        if targets and node.map_id in targets:
            hits.append((node, seq))
            # One hit is enough for a sanity check.
            break

        if len(seq) == max_depth:
            continue

        for action in actions:
            queue.append(seq + (action,))

    assert start is not None

    maps_seen = sorted({node.map_id for node in seen})
    pos_as = [node.pos_a for node in seen]
    pos_bs = [node.pos_b for node in seen]

    print(f"{name}")
    print(f"  start: map={start.map_id} a={start.pos_a} b={start.pos_b}")
    print(f"  reachable coordinate states: {len(seen)}")
    print(f"  maps seen: {maps_seen}")
    print(f"  coord range: a=[{min(pos_as)}..{max(pos_as)}] b=[{min(pos_bs)}..{max(pos_bs)}]")

    if targets:
        print(f"  target maps: {sorted(targets)}")
        if hits:
            node, seq = hits[0]
            print(f"  target hit: map={node.map_id} a={node.pos_a} b={node.pos_b} via {seq}")
        else:
            print("  target hit: none within sanity depth")

    pyboy.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=["05d_forest_south", "05e_forest_north", "05f_gate_transition"])
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--no-repel", action="store_true")
    parser.add_argument("--settle-ticks", type=int, default=0)
    args = parser.parse_args()

    for name in args.names:
        search_state(
            name,
            max_depth=args.max_depth,
            use_repel=not args.no_repel,
            settle_ticks=args.settle_ticks,
        )


if __name__ == "__main__":
    main()
