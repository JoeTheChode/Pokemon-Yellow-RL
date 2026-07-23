from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

pyboy = PyBoy(str(ROM_PATH), window='null')
for name in ['milestones/06_pewter_city.state', 'milestones/05g_to_pewter.state']:
    with open(name, 'rb') as f:
        pyboy.load_state(BytesIO(f.read()))
    m = pyboy.memory
    level = int(m[0xD18B])
    hp = (int(m[0xD014]) << 8) | int(m[0xD015])
    maxhp = (int(m[0xD022]) << 8) | int(m[0xD023])
    xp_hi = int(m[0xD179]); xp_mid = int(m[0xD17A]); xp_lo = int(m[0xD17B])
    xp = (xp_hi << 16) | (xp_mid << 8) | xp_lo
    print(name, 'map', int(m[0xD35D]), 'y', int(m[0xD361]), 'x', int(m[0xD362]), 'level', level, 'hp', f'{hp}/{maxhp}', 'xp', xp, 'battle', int(m[0xD057]), 'text', int(m[0xCF13]), 'party', int(m[0xD162]))
pyboy.stop()
