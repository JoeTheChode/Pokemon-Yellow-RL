
from io import BytesIO
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
base = BytesIO(); pyboy.save_state(base); data = base.getvalue()
for action in ['up','down','left','right','a']:
    pyboy.load_state(BytesIO(data))
    if action in ['up','down','left','right']:
        for press in [8,24,48,96,160]:
            pyboy.load_state(BytesIO(data))
            pyboy.button_press(action); pyboy.tick(press); pyboy.button_release(action); pyboy.tick(8)
            mem = pyboy.memory
            print(action, press, {'map': int(mem[0xD35D]), 'y': int(mem[0xD361]), 'x': int(mem[0xD362]), 'battle': int(mem[0xD057]), 'text': int(mem[0xCF13])})
    else:
        pyboy.button_press('a'); pyboy.tick(8); pyboy.button_release('a'); pyboy.tick(16)
        mem = pyboy.memory
        print('a', {'map': int(mem[0xD35D]), 'y': int(mem[0xD361]), 'x': int(mem[0xD362]), 'battle': int(mem[0xD057]), 'text': int(mem[0xCF13])})
pyboy.stop()
