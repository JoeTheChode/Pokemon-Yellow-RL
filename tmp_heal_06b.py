import shutil
from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

SRC = 'milestones/06b_pewter_lvl10.state'
BACKUP = 'milestones/06b_pewter_lvl10.20260416_122400.pre_heal_backup.state'
shutil.copy(SRC, BACKUP)

pyboy = PyBoy(str(ROM_PATH), window='null')
with open(SRC, 'rb') as f:
    pyboy.load_state(BytesIO(f.read()))
m = pyboy.memory
party_size = int(m[0xD162])
CUR = [(0xD16B, 0xD16C), (0xD197, 0xD198), (0xD1C3, 0xD1C4), (0xD1EF, 0xD1F0), (0xD21B, 0xD21C), (0xD247, 0xD248)]
MAX = [(0xD18C, 0xD18D), (0xD1B8, 0xD1B9), (0xD1E4, 0xD1E5), (0xD210, 0xD211), (0xD23C, 0xD23D), (0xD268, 0xD269)]
for i in range(party_size):
    ch, cl = CUR[i]; mh, ml = MAX[i]
    max_hp = (int(m[mh]) << 8) | int(m[ml])
    m[ch] = (max_hp >> 8) & 0xFF
    m[cl] = max_hp & 0xFF
    print(f'  mon {i}: healed to {max_hp} HP')
m[0xD014] = m[0xD16B]
m[0xD015] = m[0xD16C]
with open(SRC, 'wb') as f:
    pyboy.save_state(f)
print('06b state healed')
pyboy.stop()
