"""Teleport around the forest to map walkable tiles and find the path to exit."""
import os, io
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

from pyboy import PyBoy
from collections import deque

ROM = os.path.join(os.path.dirname(__file__), 'yellow.gb')
STATE = os.path.join(os.path.dirname(__file__), 'milestones', '05e_forest_north.state')

ADDR_MAP_ID = 0xD35D
ADDR_POS_A = 0xD361  # Y
ADDR_POS_B = 0xD362  # X

def bfs_from(pyboy, state_bytes, start_y, start_x, max_tiles=1000):
    """BFS from a specific y,x position in map 51."""
    pyboy.load_state(io.BytesIO(state_bytes))
    mem = pyboy.memory
    # Teleport to start_y, start_x
    mem[ADDR_POS_A] = start_y
    mem[ADDR_POS_B] = start_x
    # Tick a few frames to settle
    for _ in range(5):
        pyboy.tick()

    actual_y = mem[ADDR_POS_A]
    actual_x = mem[ADDR_POS_B]
    actual_map = mem[ADDR_MAP_ID]

    if actual_map != 51:
        return set(), []

    start = (51, actual_y, actual_x)
    visited = {start}
    queue = deque([start])
    exits = []
    states = {}

    buf = io.BytesIO()
    pyboy.save_state(buf)
    states[start] = buf.getvalue()

    while queue and len(visited) < max_tiles:
        pos = queue.popleft()
        pyboy.load_state(io.BytesIO(states[pos]))

        for d in ['up', 'down', 'left', 'right']:
            save_buf = io.BytesIO()
            pyboy.save_state(save_buf)
            save_bytes = save_buf.getvalue()

            pyboy.button_press(d)
            pyboy.tick(8)
            pyboy.button_release(d)
            pyboy.tick(16)

            new_pos = (mem[ADDR_MAP_ID], mem[ADDR_POS_A], mem[ADDR_POS_B])
            if new_pos not in visited:
                visited.add(new_pos)
                new_buf = io.BytesIO()
                pyboy.save_state(new_buf)
                states[new_pos] = new_buf.getvalue()
                if new_pos[0] != 51:
                    exits.append((pos, d, new_pos))
                else:
                    queue.append(new_pos)

            pyboy.load_state(io.BytesIO(save_bytes))

    return visited, exits

pyboy = PyBoy(ROM, window='null')
with open(STATE, 'rb') as f:
    state_bytes = f.read()

print('Scanning forest for connected regions...', flush=True)
print('Map is 68 tall x 152 wide tiles', flush=True)

# Try teleporting to grid positions across the forest
all_forest_tiles = set()
all_exits = []
regions = []

# Scan a grid of starting positions
for test_y in range(0, 40, 4):
    for test_x in range(0, 30, 4):
        # Skip if already explored
        if (51, test_y, test_x) in all_forest_tiles:
            continue

        visited, exits = bfs_from(pyboy, state_bytes, test_y, test_x)
        forest_tiles = {(y, x) for m, y, x in visited if m == 51}

        if len(forest_tiles) > 1:
            new_tiles = forest_tiles - {(y,x) for _, y, x in all_forest_tiles if _ == 51}
            if new_tiles:
                regions.append((test_y, test_x, forest_tiles, exits))
                for m, y, x in visited:
                    all_forest_tiles.add((m, y, x))
                all_exits.extend(exits)
                min_y = min(y for y, x in forest_tiles)
                max_y = max(y for y, x in forest_tiles)
                min_x = min(x for y, x in forest_tiles)
                max_x = max(x for y, x in forest_tiles)
                print(f'  Region from ({test_y},{test_x}): {len(forest_tiles)} tiles, '
                      f'y=[{min_y}..{max_y}], x=[{min_x}..{max_x}]', flush=True)
                if exits:
                    for from_pos, direction, to_pos in exits:
                        print(f'    EXIT: {direction} from y={from_pos[1]},x={from_pos[2]} -> '
                              f'map={to_pos[0]},y={to_pos[1]},x={to_pos[2]}', flush=True)

# Summary
print(f'\n=== SUMMARY ===', flush=True)
print(f'Total regions found: {len(regions)}', flush=True)
print(f'Total forest tiles: {len([(m,y,x) for m,y,x in all_forest_tiles if m==51])}', flush=True)

if all_exits:
    print(f'\nAll exits from forest:', flush=True)
    for from_pos, direction, to_pos in all_exits:
        print(f'  {direction} from (y={from_pos[1]},x={from_pos[2]}) -> map={to_pos[0]},y={to_pos[1]},x={to_pos[2]}', flush=True)
else:
    print(f'\nNO EXITS FOUND from any region!', flush=True)

# Show the region with exits (if any) as a visual map
for start_y, start_x, tiles, exits in regions:
    if exits:
        print(f'\n=== Region with exit (from {start_y},{start_x}) ===', flush=True)
        by_y = {}
        for y, x in tiles:
            by_y.setdefault(y, []).append(x)
        for y in sorted(by_y.keys()):
            xs = sorted(by_y[y])
            row = ''
            if xs:
                for x in range(min(xs), max(xs) + 1):
                    row += '#' if x in xs else '.'
            print(f'  y={y:2d} x={min(xs):2d}-{max(xs):2d}: {row}', flush=True)

pyboy.stop()
