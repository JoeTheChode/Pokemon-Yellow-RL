import shutil
from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

SRC = 'milestones/06c_pewter_lvl13.state'
BACKUP = 'milestones/06c_pewter_lvl13.20260417_002000.pre_dialog_dismiss_backup.state'
shutil.copy(SRC, BACKUP)

pyboy = PyBoy(str(ROM_PATH), window='null')
with open(SRC, 'rb') as f:
    pyboy.load_state(BytesIO(f.read()))

def press_a(ticks_per_press=32):
    pyboy.button_press('a')
    for _ in range(8): pyboy.tick()
    pyboy.button_release('a')
    for _ in range(ticks_per_press): pyboy.tick()

m = pyboy.memory
print(f'START: textbox={int(m[0xCF13])} map={int(m[0xD35D])} y={int(m[0xD361])} x={int(m[0xD362])} battle={int(m[0xD057])}')

# Press A up to 40 times to clear all text boxes
for i in range(40):
    press_a()
    tb = int(m[0xCF13])
    if tb == 0:
        print(f'  dialog cleared at press {i+1}')
        break
    if i % 5 == 0:
        print(f'  press {i+1}: textbox={tb}')

# Verify movement works
for _ in range(8): pyboy.tick()  # settle
print(f'AFTER: textbox={int(m[0xCF13])} map={int(m[0xD35D])} y={int(m[0xD361])} x={int(m[0xD362])} battle={int(m[0xD057])}')

# Now heal (may have been overwritten by natural processes)
max_hp = (int(m[0xD18C]) << 8) | int(m[0xD18D])
m[0xD16B] = (max_hp >> 8) & 0xFF
m[0xD16C] = max_hp & 0xFF
print(f'HEAL: max_HP={max_hp}, set cur_HP={max_hp}')

with open(SRC, 'wb') as f:
    pyboy.save_state(f)
print('saved dialog-dismissed + healed state')
pyboy.stop()
