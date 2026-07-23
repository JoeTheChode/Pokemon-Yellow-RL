from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

pyboy = PyBoy(str(ROM_PATH), window='null')
with open('milestones/06_pewter_city.state', 'rb') as f:
    pyboy.load_state(BytesIO(f.read()))
m = pyboy.memory
level = int(m[0xD18B])
# party mon 1 HP (actual stored, not battle active)
p_cur = (int(m[0xD16B]) << 8) | int(m[0xD16C])
p_max = (int(m[0xD18C]) << 8) | int(m[0xD18D])
# active battle mon HP
a_cur = (int(m[0xD014]) << 8) | int(m[0xD015])
a_max = (int(m[0xD022]) << 8) | int(m[0xD023])
print('level', level, 'party_HP', f'{p_cur}/{p_max}', 'active_HP', f'{a_cur}/{a_max}')
print('map', int(m[0xD35D]), 'y', int(m[0xD361]), 'x', int(m[0xD362]), 'battle', int(m[0xD057]), 'textbox', int(m[0xCF13]))
pyboy.stop()
