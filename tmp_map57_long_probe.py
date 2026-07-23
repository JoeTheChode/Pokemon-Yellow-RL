
from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy
seq = ['up','left','left','up','left','left','left','left','left','left','left','left','left','left','up']
pyboy = PyBoy(str(ROM_PATH), window='null')
with open('milestones/06_pewter_city.state', 'rb') as f:
    pyboy.load_state(f)
for action in seq:
    pyboy.button_press(action); pyboy.tick(8); pyboy.button_release(action); pyboy.tick(16)
base = BytesIO(); pyboy.save_state(base); data = base.getvalue()
for action in ['up','down','left','right']:
    for press in [8,24,48,96,160,256]:
        pyboy.load_state(BytesIO(data))
        pyboy.button_press(action); pyboy.tick(press); pyboy.button_release(action); pyboy.tick(8)
        mem = pyboy.memory
        print(action, press, {'map': int(mem[0xD35D]), 'y': int(mem[0xD361]), 'x': int(mem[0xD362]), 'battle': int(mem[0xD057]), 'text': int(mem[0xCF13])})
pyboy.stop()
