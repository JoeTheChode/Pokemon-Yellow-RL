from io import BytesIO
from project_paths import ROM_PATH, MILESTONES_DIR, PROJECT_ROOT
from pyboy import PyBoy
from train import ADDR_MAP_ID, ADDR_POS_A, ADDR_POS_B, ADDR_REPEL

OUT_DIR = PROJECT_ROOT / "debug_state_screens" / "level_up_probe"

pyboy = PyBoy(str(ROM_PATH), window="null")
with open(MILESTONES_DIR / "07_beat_brock.state", "rb") as f:
    pyboy.load_state(BytesIO(f.read()))
mem = pyboy.memory
mem[ADDR_REPEL] = 0

for _ in range(40):
    pyboy.button_press("right")
    pyboy.tick(10)
    pyboy.button_release("right")
    pyboy.tick(6)

print("stuck spot reached:", int(mem[ADDR_MAP_ID]), int(mem[ADDR_POS_A]), int(mem[ADDR_POS_B]))
pyboy.screen.image.save(OUT_DIR / "diag_00_stuck_spot.png")

for d in ["up", "down", "left", "right"]:
    for _ in range(15):
        pyboy.button_press(d)
        pyboy.tick(10)
        pyboy.button_release(d)
        pyboy.tick(6)
    pos = (int(mem[ADDR_MAP_ID]), int(mem[ADDR_POS_A]), int(mem[ADDR_POS_B]))
    print(f"after 15x {d}: {pos}")
    pyboy.screen.image.save(OUT_DIR / f"diag_dir_{d}.png")

pyboy.stop()
