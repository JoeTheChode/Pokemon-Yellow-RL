"""Navigate from 05d state into Viridian Forest and map it."""

import random

from pyboy import PyBoy

from project_paths import ROM_PATH, milestone_state_path
from yellow_navigation import map_labels


ADDR_MAP_ID = 0xD35D
ADDR_POS_A = 0xD361
ADDR_POS_B = 0xD362
ADDR_BATTLE_FLAG = 0xD057
ADDR_REPEL = 0xD078

ACTIONS = ['up', 'down', 'left', 'right', 'a', 'b']


def get_pos(pyboy):
    return (pyboy.memory[ADDR_MAP_ID], pyboy.memory[ADDR_POS_A], pyboy.memory[ADDR_POS_B])


def main():
    state_file = milestone_state_path('05d_viridian_forest')
    pyboy = PyBoy(str(ROM_PATH), window='null')

    with open(state_file, 'rb') as f:
        pyboy.load_state(f)

    random.seed(42)
    tiles_by_map = {}
    map_transitions = []

    last_map = None
    steps = 50000

    print(f"Running {steps} random steps...")
    for step in range(steps):
        pyboy.memory[ADDR_REPEL] = 255

        action = random.choice(ACTIONS)
        pyboy.button_press(action)
        pyboy.tick(8)
        pyboy.button_release(action)
        pyboy.tick(16)

        mid, y, x = get_pos(pyboy)

        tiles_by_map.setdefault(mid, set()).add((y, x))

        if mid != last_map:
            if last_map is not None:
                map_transitions.append((step, last_map, mid))
            last_map = mid

        if step % 10000 == 0:
            total = sum(len(t) for t in tiles_by_map.values())
            print(f"  step {step}: {total} tiles, maps={sorted(tiles_by_map.keys())}")

        if pyboy.memory[ADDR_BATTLE_FLAG] != 0:
            for _ in range(200):
                pyboy.button_press('b')
                pyboy.tick(8)
                pyboy.button_release('b')
                pyboy.tick(16)

    total = sum(len(t) for t in tiles_by_map.values())
    print(f"\nDone: {total} tiles across maps {sorted(tiles_by_map.keys())}")

    print(f"\nMap transitions ({len(map_transitions)}):")
    for step, from_map, to_map in map_transitions[:30]:
        print(f"  step {step:5d}: map {from_map} -> {to_map}")
    if len(map_transitions) > 30:
        print(f"  ... and {len(map_transitions) - 30} more")

    map_names = map_labels()

    for mid in sorted(tiles_by_map.keys()):
        tiles = tiles_by_map[mid]
        ys = [t[0] for t in tiles]
        xs = [t[1] for t in tiles]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)

        name = map_names.get(mid, f"Map {mid}")
        print(f"\n=== Map {mid}: {name} ({len(tiles)} tiles) ===")
        print(f"    Y(D361): [{min_y}, {max_y}], X(D362): [{min_x}, {max_x}]")

        x_range = range(min_x, max_x + 1)
        if max_x - min_x > 30:
            print("    (map too wide, showing tile list)")
            for y, x in sorted(tiles):
                print(f"    y={y:2d}, x={x:2d}")
            continue

        print("     ", end="")
        for xv in x_range:
            print(f"{xv:3d}", end="")
        print()
        for yv in range(min_y, max_y + 1):
            print(f"  {yv:2d} ", end="")
            for xv in x_range:
                print("  ." if (yv, xv) in tiles else "   ", end="")
            print()

    pyboy.stop()


if __name__ == '__main__':
    main()
