
from pathlib import Path
from project_paths import ROM_PATH
from pyboy import PyBoy
seq0 = ['up','left','left','up','left','left','left','left','left','left','left','left','left','left','up']
seq1 = ['down','down','right','up']
pyboy = PyBoy(str(ROM_PATH), window='null')
with open('milestones/06_pewter_city.state', 'rb') as f:
    pyboy.load_state(f)
for action in seq0:
    pyboy.button_press(action); pyboy.tick(8); pyboy.button_release(action); pyboy.tick(16)
for action in seq1:
    pyboy.button_press(action); pyboy.tick(96); pyboy.button_release(action); pyboy.tick(8)
Path('screenshots').mkdir(exist_ok=True)
pyboy.screen.image.save('screenshots/pewter_map58.png')
mem = pyboy.memory
print({'map': int(mem[0xD35D]), 'y': int(mem[0xD361]), 'x': int(mem[0xD362]), 'battle': int(mem[0xD057]), 'text': int(mem[0xCF13])})
pyboy.stop()
