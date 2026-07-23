from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

pyboy = PyBoy(str(ROM_PATH), window='null')
for name in ['milestones/06b_pewter_lvl10.state', 'milestones/06_pewter_city.state']:
    try:
        with open(name, 'rb') as f:
            pyboy.load_state(BytesIO(f.read()))
    except FileNotFoundError:
        print(f'{name} - not found')
        continue
    m = pyboy.memory
    level = int(m[0xD18B])
    p_cur = (int(m[0xD16B]) << 8) | int(m[0xD16C])
    p_max = (int(m[0xD18C]) << 8) | int(m[0xD18D])
    print(f'{name}: map={int(m[0xD35D])} y={int(m[0xD361])} x={int(m[0xD362])} level={level} HP={p_cur}/{p_max} battle={int(m[0xD057])}')
pyboy.stop()
