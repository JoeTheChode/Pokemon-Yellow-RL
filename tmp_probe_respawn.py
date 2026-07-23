from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

pyboy = PyBoy(str(ROM_PATH), window='null')

# Probe likely respawn-map addresses. Red/Blue has wLastBlackoutMap at D71A.
# Yellow is typically -1 shift, so try D719 as well.
candidates = [0xD719, 0xD71A, 0xD71B, 0xD71C, 0xD71D, 0xD71E]

for name in ['milestones/06_pewter_city.state', 'milestones/05a_exit_lab.state', 'milestones/05b_pokecenter.state']:
    try:
        with open(name, 'rb') as f:
            pyboy.load_state(BytesIO(f.read()))
        m = pyboy.memory
        print(name, 'map', int(m[0xD35D]), 'blackout_candidates:', [(f'{a:04x}', int(m[a])) for a in candidates])
    except FileNotFoundError:
        print(f'{name} - not found, skipping')
pyboy.stop()
