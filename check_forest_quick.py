"""Walk the real Viridian Forest path from the south gate entrance."""
import os, io
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

from pyboy import PyBoy

ROM = os.path.join(os.path.dirname(__file__), 'yellow.gb')
MILESTONES = os.path.join(os.path.dirname(__file__), 'milestones')

ADDR_MAP_ID = 0xD35D
ADDR_POS_A = 0xD361
ADDR_POS_B = 0xD362
ADDR_BATTLE_FLAG = 0xD057

pyboy = PyBoy(ROM, window='null')

# Try 05d state first (should be at forest gate south entrance)
for state_name in ['05d_forest_south', '05c_viridian_north', '05b_pokecenter']:
    state_file = os.path.join(MILESTONES, f'{state_name}.state')
    if not os.path.exists(state_file):
        continue
    with open(state_file, 'rb') as f:
        pyboy.load_state(f)
    mem = pyboy.memory
    m = mem[ADDR_MAP_ID]
    y = mem[ADDR_POS_A]
    x = mem[ADDR_POS_B]
    print(f'{state_name}: map={m}, y={y}, x={x}', flush=True)

# Use 05c state and walk north until we reach/pass through the forest
# Or use 05d state
state_file = os.path.join(MILESTONES, '05d_forest_south.state')
with open(state_file, 'rb') as f:
    pyboy.load_state(f)

mem = pyboy.memory
print(f'\nStarting walk from: map={mem[ADDR_MAP_ID]}, y={mem[ADDR_POS_A]}, x={mem[ADDR_POS_B]}', flush=True)

# Walk up for a long time, logging position changes
last = (mem[ADDR_MAP_ID], mem[ADDR_POS_A], mem[ADDR_POS_B])
path = [last]
stuck_count = 0
direction_idx = 0
directions = ['up', 'up', 'up', 'up', 'right', 'up', 'up', 'left', 'up']

for step in range(5000):
    # Simple policy: walk up, if stuck try right then left
    d = 'up'
    if stuck_count > 5:
        d = 'right'
    if stuck_count > 10:
        d = 'left'
    if stuck_count > 15:
        d = 'down'
    if stuck_count > 20:
        stuck_count = 0  # reset

    # Handle battles - mash A/B
    if mem[ADDR_BATTLE_FLAG] != 0:
        for _ in range(500):
            pyboy.button_press('a')
            pyboy.tick(8)
            pyboy.button_release('a')
            pyboy.tick(8)
            pyboy.button_press('b')
            pyboy.tick(8)
            pyboy.button_release('b')
            pyboy.tick(8)
            if mem[ADDR_BATTLE_FLAG] == 0:
                break
        continue

    pyboy.button_press(d)
    pyboy.tick(8)
    pyboy.button_release(d)
    pyboy.tick(16)

    cur = (mem[ADDR_MAP_ID], mem[ADDR_POS_A], mem[ADDR_POS_B])
    if cur != last:
        print(f'  step {step}: map={cur[0]}, y={cur[1]}, x={cur[2]} (moved {d})', flush=True)
        last = cur
        stuck_count = 0
        if cur[0] == 47:  # forest gate north!
            print(f'  *** REACHED FOREST GATE NORTH ***', flush=True)
            break
        if cur[0] not in (51, 50, 13, 1):  # unexpected map
            print(f'  *** REACHED NEW MAP {cur[0]} ***', flush=True)
    else:
        stuck_count += 1

print(f'\nFinal: map={mem[ADDR_MAP_ID]}, y={mem[ADDR_POS_A]}, x={mem[ADDR_POS_B]}', flush=True)
pyboy.stop()
