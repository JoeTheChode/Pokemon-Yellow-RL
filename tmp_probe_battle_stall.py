"""Probe: reproduce the 07_beat_brock force-A trainer-battle stall headlessly,
using the now-repaired milestones/07_beat_brock.state, and watch exactly what
happens to the lead move slot / PP turn by turn.
"""
from io import BytesIO

from project_paths import ROM_PATH, MILESTONES_DIR, PROJECT_ROOT
from pyboy import PyBoy
from train import (
    ADDR_MAP_ID, ADDR_POS_A, ADDR_POS_B, ADDR_BATTLE_FLAG, ADDR_TEXT_BOX,
    ADDR_ENEMY_HP, ADDR_LEVEL, PARTY_MOVE_ID_ADDRS, PARTY_MOVE_PP_ADDRS,
)

OUT_DIR = PROJECT_ROOT / "debug_state_screens" / "battle_stall_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pyboy = PyBoy(str(ROM_PATH), window="null")
with open(MILESTONES_DIR / "07_beat_brock.state", "rb") as f:
    pyboy.load_state(BytesIO(f.read()))

mem = pyboy.memory


def info():
    return {
        "map": int(mem[ADDR_MAP_ID]),
        "y": int(mem[ADDR_POS_A]),
        "x": int(mem[ADDR_POS_B]),
        "battle": int(mem[ADDR_BATTLE_FLAG]),
        "text_box": int(mem[ADDR_TEXT_BOX]),
        "enemy_hp": int(mem[ADDR_ENEMY_HP]),
        "level": int(mem[ADDR_LEVEL]),
        "moves": [int(mem[a]) for a in PARTY_MOVE_ID_ADDRS[0]],
        "pps": [int(mem[a]) & 0x3F for a in PARTY_MOVE_PP_ADDRS[0]],
    }


print("start:", info())
pyboy.screen.image.save(OUT_DIR / "00_start.png")

directions = (["down", "a", "right", "a"] * 2 + ["down"] * 4 + ["right"] * 4) * 60
battle_hit = False
for i, d in enumerate(directions):
    pyboy.button_press(d)
    pyboy.tick(10)
    pyboy.button_release(d)
    pyboy.tick(6)
    cur = info()
    if cur["battle"] != 0:
        battle_hit = True
        print(f"battle triggered at step {i}: {cur}")
        break
    if i % 40 == 0:
        print(f"step {i}: {cur}")

if not battle_hit:
    print("No trainer battle triggered by scripted walk; aborting probe.")
    pyboy.stop()
    raise SystemExit(0)

pyboy.screen.image.save(OUT_DIR / "01_battle_start.png")
for _ in range(60):
    pyboy.tick(1)
pyboy.screen.image.save(OUT_DIR / "02_before_force_a.png")
print("before force-a loop:", info())

log_lines = []
last_pps = None
for step in range(800):
    pyboy.button_press("a")
    pyboy.tick(8)
    pyboy.button_release("a")
    pyboy.tick(4)
    cur = info()
    if cur["pps"] != last_pps:
        print(f"step {step}: PP CHANGE moves={cur['moves']} pps={cur['pps']} enemy_hp={cur['enemy_hp']} text_box={cur['text_box']}")
        last_pps = cur["pps"]
    log_lines.append(f"{step}: {cur}")
    if step % 40 == 0:
        pyboy.screen.image.save(OUT_DIR / f"step_{step:03d}.png")
        print(f"step {step}: {cur}")
    if cur["battle"] == 0:
        print(f"battle ended at step {step}: {cur}")
        pyboy.screen.image.save(OUT_DIR / f"step_{step:03d}_battle_end.png")
        break

with open(OUT_DIR / "log.txt", "w") as f:
    f.write("\n".join(log_lines))

print("final:", info())
pyboy.stop()
