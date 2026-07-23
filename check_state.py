"""Quick check: what state does each milestone start from?"""

from pyboy import PyBoy

from project_paths import PROJECT_ROOT, ROM_PATH, milestone_state_path


ADDR_MAP_ID = 0xD35D
ADDR_POS_A = 0xD361
ADDR_POS_B = 0xD362
ADDR_PARTY_SIZE = 0xD162
ADDR_BADGES = 0xD356
ADDR_LEVEL = 0xD18B


pyboy = PyBoy(str(ROM_PATH), window='null')

for name in ['05d_viridian_forest', '06_pewter_city']:
    path = milestone_state_path(name)
    if path.exists():
        with open(path, 'rb') as f:
            pyboy.load_state(f)
        mid = pyboy.memory[ADDR_MAP_ID]
        y = pyboy.memory[ADDR_POS_A]
        x = pyboy.memory[ADDR_POS_B]
        party = pyboy.memory[ADDR_PARTY_SIZE]
        badges = pyboy.memory[ADDR_BADGES]
        level = pyboy.memory[ADDR_LEVEL]
        pyboy.screen.image.save(PROJECT_ROOT / f'state_{name}.png')
        print(f"{name}: map={mid}, y={y}, x={x}, party={party}, badges={badges}, level={level}")
    else:
        print(f"{name}: NO STATE FILE")

pyboy.stop()
