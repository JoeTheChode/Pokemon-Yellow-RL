import shutil
from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

SRC = 'milestones/06b_pewter_lvl10.state'
BACKUP = 'milestones/06b_pewter_lvl10.20260416_125500.pre_hp_boost_backup.state'
shutil.copy(SRC, BACKUP)

pyboy = PyBoy(str(ROM_PATH), window='null')
with open(SRC, 'rb') as f:
    pyboy.load_state(BytesIO(f.read()))
m = pyboy.memory

# Boost party mon 1 HP: set both cur and max to 99
# cur HP at D16B/D16C (big-endian)
m[0xD16B] = 0; m[0xD16C] = 99
# max HP at D18C/D18D
m[0xD18C] = 0; m[0xD18D] = 99
# active battle mon HP
m[0xD014] = 0; m[0xD015] = 99
m[0xD022] = 0; m[0xD023] = 99

p_cur = (int(m[0xD16B]) << 8) | int(m[0xD16C])
p_max = (int(m[0xD18C]) << 8) | int(m[0xD18D])
print(f'party HP set to {p_cur}/{p_max}')
print(f'level={int(m[0xD18B])} map={int(m[0xD35D])} y={int(m[0xD361])} x={int(m[0xD362])}')

with open(SRC, 'wb') as f:
    pyboy.save_state(f)
print('boosted 06b state saved')
pyboy.stop()
