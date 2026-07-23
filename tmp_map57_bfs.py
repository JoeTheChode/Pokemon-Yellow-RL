
from collections import deque
from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy
# build map57 root from 06 state with default 8/16 path
seq = ['up','left','left','up','left','left','left','left','left','left','left','left','left','left','up']
ACTIONS=['up','down','left','right']
pyboy = PyBoy(str(ROM_PATH), window='null')
with open('milestones/06_pewter_city.state', 'rb') as f:
    pyboy.load_state(f)
for action in seq:
    pyboy.button_press(action); pyboy.tick(8); pyboy.button_release(action); pyboy.tick(16)
root_io = BytesIO(); pyboy.save_state(root_io); root = root_io.getvalue()

def load(data):
    pyboy.load_state(BytesIO(data))

def save():
    bio=BytesIO(); pyboy.save_state(bio); return bio.getvalue()

def snap():
    m=pyboy.memory
    return (int(m[0xD35D]), int(m[0xD361]), int(m[0xD362]), int(m[0xD057]), int(m[0xCF13]))

seen={}; q=deque()
load(root); s0=snap(); seen[s0]=[]; q.append((root, []))
max_depth=12
while q and len(seen) < 5000:
    data, seq = q.popleft()
    if len(seq) >= max_depth:
        continue
    for action in ACTIONS:
        load(data)
        pyboy.button_press(action); pyboy.tick(96); pyboy.button_release(action); pyboy.tick(8)
        cur=snap()
        if cur not in seen:
            next_seq=seq+[action]
            seen[cur]=next_seq
            q.append((save(), next_seq))
for mid in sorted({s[0] for s in seen}):
    states=sorted(s for s in seen if s[0]==mid)
    ys=[s[1] for s in states]; xs=[s[2] for s in states]
    print('map', mid, 'count', len(states), 'y', (min(ys), max(ys)), 'x', (min(xs), max(xs)))
    print(' sample', [(s, seen[s]) for s in states[:20]])
pyboy.stop()
