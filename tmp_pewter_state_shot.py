
from pathlib import Path
from project_paths import ROM_PATH
from pyboy import PyBoy
pyboy = PyBoy(str(ROM_PATH), window='null')
with open('milestones/06_pewter_city.state', 'rb') as f:
    pyboy.load_state(f)
Path('screenshots').mkdir(exist_ok=True)
pyboy.screen.image.save('screenshots/06_pewter_city_state.png')
mem = pyboy.memory
print({'map': int(mem[0xD35D]), 'y': int(mem[0xD361]), 'x': int(mem[0xD362]), 'battle': int(mem[0xD057])})
pyboy.stop()
