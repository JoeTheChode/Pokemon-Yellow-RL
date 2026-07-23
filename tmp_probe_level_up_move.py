"""Probe: find the RAM address holding the pending "new move" candidate when
Pikachu levels up with 4 moves already known. Greedy walker toward the
project's real Route3 waypoints, with stuck-detection that tries perpendicular
directions when the primary direction is blocked (Route3 has boulder-maze
dead-end pockets that a single fixed direction can wander into).
"""
from io import BytesIO

from project_paths import ROM_PATH, MILESTONES_DIR, PROJECT_ROOT
from pyboy import PyBoy
from train import (
    ADDR_MAP_ID, ADDR_POS_A, ADDR_POS_B, ADDR_BATTLE_FLAG, ADDR_TEXT_BOX,
    ADDR_ENEMY_HP, ADDR_LEVEL, ADDR_REPEL,
    PARTY_MOVE_ID_ADDRS, PARTY_MOVE_PP_ADDRS, GEN1_MOVE_TABLE,
    _ensure_lead_attack_memory,
)

OUT_DIR = PROJECT_ROOT / "debug_state_screens" / "level_up_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Real Route3 waypoints (map 14) from POST_BROCK_ROUTE_FRONTIER_PATH.
route3_waypoints = [
    (10, 0), (9, 8), (9, 11), (6, 11), (6, 14), (4, 19),
    (6, 21), (6, 27), (8, 33), (5, 47), (11, 56), (8, 59), (0, 59),
]

pyboy = PyBoy(str(ROM_PATH), window="null")
with open(MILESTONES_DIR / "07_beat_brock.state", "rb") as f:
    pyboy.load_state(BytesIO(f.read()))

mem = pyboy.memory


def moves():
    return [int(mem[a]) for a in PARTY_MOVE_ID_ADDRS[0]], [int(mem[a]) & 0x3F for a in PARTY_MOVE_PP_ADDRS[0]]


def wram_snapshot():
    return bytes(mem[addr] for addr in range(0xC000, 0xE000))


def pos():
    return (int(mem[ADDR_MAP_ID]), int(mem[ADDR_POS_A]), int(mem[ADDR_POS_B]))


def press(d, n=1):
    for _ in range(n):
        pyboy.button_press(d)
        pyboy.tick(10)
        pyboy.button_release(d)
        pyboy.tick(6)


start_level = int(mem[ADDR_LEVEL])
print("start level:", start_level, "moves:", moves())
print("start pos:", pos())

# Get from Pewter into Route3 first (proven: right works).
press("right", 40)
print("after entering route3:", pos())

wp_idx = 0
last_pos = pos()
stuck_count = 0
window_pos = pos()
window_start_step = 0
prev_wram = None
level_changed = False
xp_boosted = False
step = 0
MAX_STEPS = 15000

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
        cur_map, cur_y, cur_x = pos()

        if wp_idx < len(route3_waypoints):
            ty, tx = route3_waypoints[wp_idx]
            if abs(cur_y - ty) <= 1 and abs(cur_x - tx) <= 1:
                wp_idx += 1
                print(f"  reached waypoint {wp_idx}/{len(route3_waypoints)}: {(ty, tx)} at step {step}")
        if wp_idx >= len(route3_waypoints):
            ty, tx = route3_waypoints[-1]
        else:
            ty, tx = route3_waypoints[wp_idx]

        dy, dx = ty - cur_y, tx - cur_x
        primary = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy > 0 else "up")
        secondary = ("down" if dy > 0 else "up") if primary in ("left", "right") else ("right" if dx > 0 else "left")

        # Rolling-window stuck check: compare position against where we were
        # ~24 steps ago, not the immediately preceding step (single-step
        # comparisons are too noisy -- sub-tile animation frames make
        # "moved since last iteration" flicker even while net progress is
        # zero over a real window).
        if step - window_start_step >= 24:
            if pos() == window_pos:
                stuck_count += 1
            else:
                stuck_count = 0
            window_pos = pos()
            window_start_step = step

        if stuck_count == 0:
            d = primary
        elif stuck_count == 1:
            d = secondary
        elif stuck_count == 2:
            d = "left"  # the escape that worked out of the first dead-end pocket
        else:
            # Cycle through everything, including retreat, and periodically
            # reset to let primary/secondary try again from a new spot.
            d = ["up", "down", "left", "right"][stuck_count % 4]
            if stuck_count > 10:
                stuck_count = 0

        press(d, 4)

    prev_wram_snapshot = prev_wram
    cur_level = int(mem[ADDR_LEVEL])
    if cur_level > start_level:
        level_changed = True
        print(f"\nLEVEL UP detected at step {step}: {start_level} -> {cur_level}")
        break
    prev_wram = wram_snapshot()

    if step % 500 == 0:
        m_ids, m_pp = moves()
        xp_now = (int(mem[0xD179]) << 16) | (int(mem[0xD17A]) << 8) | int(mem[0xD17B])
        print(f"step {step}: wp={wp_idx}/{len(route3_waypoints)} pos={pos()} stuck={stuck_count} "
              f"level={cur_level} xp={xp_now} moves={m_ids} pp={m_pp}")
        pyboy.screen.image.save(OUT_DIR / f"progress_{step:06d}.png")

    if not xp_boosted and wp_idx >= 2:
        xp_boosted = True
        xp_now = (int(mem[0xD179]) << 16) | (int(mem[0xD17A]) << 8) | int(mem[0xD17B])
        # First boost undershot (1.5M -> 1.64M via a real battle win, still
        # not enough to cross the level-19 threshold) -- go much bigger this
        # time so any single subsequent battle win is guaranteed to cross it.
        boosted = min(xp_now + 800000, 0xFFFFFF)
        mem[0xD179] = (boosted >> 16) & 0xFF
        mem[0xD17A] = (boosted >> 8) & 0xFF
        mem[0xD17B] = boosted & 0xFF
        print(f"  [XP BOOST] {xp_now} -> {boosted}")

if not level_changed:
    print(f"No level-up within step budget (reached waypoint {wp_idx}/{len(route3_waypoints)}); aborting.")
    pyboy.stop()
    raise SystemExit(0)

pyboy.screen.image.save(OUT_DIR / "00_level_up_moment.png")
new_moves, new_pp = moves()
print("moves right at level-up tick:", new_moves, new_pp)

cur_wram = wram_snapshot()
if prev_wram_snapshot is not None:
    diffs = [
        (0xC000 + i, prev_wram_snapshot[i], cur_wram[i])
        for i in range(len(cur_wram))
        if prev_wram_snapshot[i] != cur_wram[i]
    ]
    print(f"\n{len(diffs)} byte(s) changed between the tick before and the level-up tick:")
    for addr, old, new in diffs:
        note = ""
        if 1 <= new <= 165 and new in GEN1_MOVE_TABLE:
            note = f"  <- valid move ID: {GEN1_MOVE_TABLE[new]}"
        print(f"  0x{addr:04X}: {old} -> {new}{note}")

for i in range(150):
    pyboy.tick(1)
    if i % 10 == 0:
        tb = int(mem[ADDR_TEXT_BOX])
        bf = int(mem[ADDR_BATTLE_FLAG])
        m_ids, m_pp = moves()
        print(f"  +{i} ticks: battle={bf} text_box={tb} moves={m_ids} pp={m_pp}")
        pyboy.screen.image.save(OUT_DIR / f"after_{i:04d}.png")

pyboy.screen.image.save(OUT_DIR / "99_final.png")
print("final moves:", moves())
pyboy.stop()
