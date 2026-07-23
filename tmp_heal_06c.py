import shutil
from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

SRC = 'milestones/06c_pewter_lvl13.state'
BACKUP = 'milestones/06c_pewter_lvl13.20260416_230000.pre_heal_backup.state'
shutil.copy(SRC, BACKUP)

pyboy = PyBoy(str(ROM_PATH), window='null')
with open(SRC, 'rb') as f:
    pyboy.load_state(BytesIO(f.read()))
m = pyboy.memory
party_size = int(m[0xD162])
CUR = [(0xD16B, 0xD16C)]
MAX = [(0xD18C, 0xD18D)]
max_hp = (int(m[MAX[0][0]]) << 8) | int(m[MAX[0][1]])
m[CUR[0][0]] = (max_hp >> 8) & 0xFF
m[CUR[0][1]] = max_hp & 0xFF
m[0xD014] = m[CUR[0][0]]
m[0xD015] = m[CUR[0][1]]
print(f'healed to {max_hp} HP, level={int(m[0xD18B])}, map={int(m[0xD35D])}, y={int(m[0xD361])}, x={int(m[0xD362])}')
with open(SRC, 'wb') as f:
    pyboy.save_state(f)
pyboy.stop()
