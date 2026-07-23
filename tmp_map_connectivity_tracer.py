"""Empirically trace real map connectivity from Pewter City onward, logging
every (from_map -> to_map) transition the FIRST time it happens, with exact
position and a screenshot -- ground truth from actual observed traversal,
not from labels/assumptions or reverse-engineered ROM header parsing.
"""
import json
import random
from io import BytesIO

from project_paths import ROM_PATH, MILESTONES_DIR, PROJECT_ROOT
from pyboy import PyBoy
from train import (
    ADDR_MAP_ID, ADDR_POS_A, ADDR_POS_B, ADDR_BATTLE_FLAG, ADDR_TEXT_BOX,
    ADDR_REPEL, ADDR_BADGES,
    _ensure_lead_attack_memory,
)

OUT_DIR = PROJECT_ROOT / "debug_state_screens" / "map_trace"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pyboy = PyBoy(str(ROM_PATH), window="null")
with open(MILESTONES_DIR / "07_beat_brock.state", "rb") as f:
    pyboy.load_state(BytesIO(f.read()))

mem = pyboy.memory


def pos():
    return (int(mem[ADDR_MAP_ID]), int(mem[ADDR_POS_A]), int(mem[ADDR_POS_B]))


def press(d):
    pyboy.button_press(d)
    pyboy.tick(10)
    pyboy.button_release(d)
    pyboy.tick(6)


transitions = []  # list of dicts: from_map, from_pos, to_map, to_pos, step
seen_maps = set()
seen_transitions = set()  # (from_map, to_map) pairs already logged

cur_map, cur_y, cur_x = pos()
seen_maps.add(cur_map)
print(f"start: map={cur_map} pos=({cur_y},{cur_x}) badges={int(mem[ADDR_BADGES])}")
pyboy.screen.image.save(OUT_DIR / f"map_{cur_map:03d}_first_seen.png")

directions = ["right", "up", "down", "left"]
dir_idx = 0
stuck = 0
last_pos = pos()
window_start = 0
step = 0
MAX_STEPS = 60000
log_lines = []

while step < MAX_STEPS:
    step += 1
    battle_flag = int(mem[ADDR_BATTLE_FLAG])
    text_box = int(mem[ADDR_TEXT_BOX])

    if battle_flag in (1, 2):
        _ensure_lead_attack_memory(mem)
        press("a")
    elif text_box != 0:
        press("a")
    else:
        mem[ADDR_REPEL] = 0
        prev_map, prev_y, prev_x = pos()

        press(directions[dir_idx])

        new_map, new_y, new_x = pos()
        if new_map != prev_map:
            pair = (prev_map, new_map)
            if pair not in seen_transitions:
                seen_transitions.add(pair)
                entry = {
                    "step": step,
                    "from_map": prev_map,
                    "from_pos": [prev_y, prev_x],
                    "to_map": new_map,
                    "to_pos": [new_y, new_x],
                }
                transitions.append(entry)
                line = f"TRANSITION #{len(transitions)}: map{prev_map}({prev_y},{prev_x}) -> map{new_map}({new_y},{new_x}) at step {step}"
                print(line)
                log_lines.append(line)
                pyboy.screen.image.save(OUT_DIR / f"transition_{len(transitions):02d}_map{prev_map}_to_map{new_map}.png")
            if new_map not in seen_maps:
                seen_maps.add(new_map)
                pyboy.screen.image.save(OUT_DIR / f"map_{new_map:03d}_first_seen.png")

        # Rolling-window stuck detection -> escalate direction changes.
        if step - window_start >= 20:
            if pos() == last_pos:
                stuck += 1
                if stuck > 8:
                    # Deeply stuck (likely a dead-end pocket) -- go random
                    # for a while instead of cycling deterministically,
                    # which can trap us in a 2-direction bounce loop.
                    dir_idx = random.randrange(len(directions))
                else:
                    dir_idx = (dir_idx + 1) % len(directions)
            else:
                stuck = 0
            last_pos = pos()
            window_start = step

    if step % 2000 == 0:
        cm, cy, cx = pos()
        print(f"step {step}: map={cm} pos=({cy},{cx}) maps_seen={sorted(seen_maps)} dir={directions[dir_idx]}")

with open(OUT_DIR / "transitions.json", "w") as f:
    json.dump(transitions, f, indent=2)
with open(OUT_DIR / "log.txt", "w") as f:
    f.write("\n".join(log_lines))

print(f"\nFinal maps seen: {sorted(seen_maps)}")
print(f"Total transitions logged: {len(transitions)}")
pyboy.stop()
