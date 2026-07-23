"""
Walk the agent into Pewter PC and talk to Nurse Joy to register Pewter PC
as the whiteout respawn point. Save the new state.

Starting state: 06_pewter_city.state (map 58, y=13, x=1)
PewterPC layout: Joy is at approximately y=3, x=3-4.
"""
import shutil
from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy

SRC = 'milestones/06_pewter_city.state'
BACKUP = 'milestones/06_pewter_city.20260416_120000.pre_heal_respawn_backup.state'
shutil.copy(SRC, BACKUP)

pyboy = PyBoy(str(ROM_PATH), window='null')
with open(SRC, 'rb') as f:
    pyboy.load_state(BytesIO(f.read()))

def snap():
    m = pyboy.memory
    return (int(m[0xD35D]), int(m[0xD361]), int(m[0xD362]), int(m[0xD057]), int(m[0xCF13]))

def press(action, press_ticks=16, release_ticks=16):
    pyboy.button_press(action)
    for _ in range(press_ticks): pyboy.tick()
    pyboy.button_release(action)
    for _ in range(release_ticks): pyboy.tick()

print('start:', snap())

# Walk up to the Joy counter: from y=13,x=1, need to go up ~10 tiles, right ~3 tiles
# Pokecenter tiles: Joy counter at roughly y=3, x=3 (one tile below Joy, in front of her)
for _ in range(10):
    press('up')
    print('after up:', snap())
for _ in range(3):
    press('right')
    print('after right:', snap())

# Press A repeatedly to talk, accept heal dialog. Heal sequence takes many ticks.
for i in range(40):
    press('a', press_ticks=8, release_ticks=32)
    s = snap()
    if i % 8 == 0:
        print(f'a#{i}:', s)

# Final position after heal
print('final:', snap())

# Verify Pikachu is healed
m = pyboy.memory
p_cur = (int(m[0xD16B]) << 8) | int(m[0xD16C])
p_max = (int(m[0xD18C]) << 8) | int(m[0xD18D])
print(f'pikachu HP after heal: {p_cur}/{p_max}')

# Save state
with open(SRC, 'wb') as f:
    pyboy.save_state(f)
print(f'saved new state; backup at {BACKUP}')

pyboy.stop()
