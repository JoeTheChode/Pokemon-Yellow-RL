
from collections import deque
from io import BytesIO
from project_paths import ROM_PATH
from pyboy import PyBoy
STATE='milestones/06_pewter_city.state'
ACTIONS=['up','down','left','right']
pyboy=PyBoy(str(ROM_PATH), window='null')
with open(STATE,'rb') as f:
    root=f.read()

def load(data):
    pyboy.load_state(BytesIO(data))

def save():
    bio=BytesIO(); pyboy.save_state(bio); return bio.getvalue()

def snap():
    m=pyboy.memory
    return (int(m[0xD35D]), int(m[0xD361]), int(m[0xD362]), int(m[0xD057]), int(m[0xCF13]))
seen=set(); q=deque()
load(root); s0=snap(); seen.add(s0); q.append((root, []))
max_depth=60
while q and len(seen) < 30000:
    data, seq=q.popleft()
    if len(seq) >= max_depth:
        continue
    for action in ACTIONS:
        load(data)
        pyboy.button_press(action); pyboy.tick(96); pyboy.button_release(action); pyboy.tick(8)
        cur=snap()
        if cur not in seen:
            seen.add(cur)
            q.append((save(), seq+[action]))
for mid in sorted({s[0] for s in seen}):
    states=sorted(s for s in seen if s[0]==mid)
    ys=[s[1] for s in states]; xs=[s[2] for s in states]
    print('map', mid, 'count', len(states), 'y', (min(ys), max(ys)), 'x', (min(xs), max(xs)))
    print(' sample', states[:25])
pyboy.stop()
