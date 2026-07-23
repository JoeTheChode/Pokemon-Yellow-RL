
from collections import deque
from io import BytesIO
from itertools import product
from project_paths import ROM_PATH
from pyboy import PyBoy
STATE='milestones/06_pewter_city.state'
ACTIONS=['up','down','left','right']
configs=[(8,16),(24,8),(48,8),(96,8),(160,8)]
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
for press, release in configs:
    seen=set(); q=deque(); load(root); s0=snap(); seen.add(s0); q.append((root, []))
    max_depth=10
    while q and len(seen) < 5000:
        data, seq=q.popleft()
        if len(seq) >= max_depth:
            continue
        for action in ACTIONS:
            load(data)
            pyboy.button_press(action); pyboy.tick(press); pyboy.button_release(action); pyboy.tick(release)
            cur=snap()
            if cur not in seen:
                seen.add(cur)
                q.append((save(), seq+[action]))
    by_map={}
    for s in seen:
        by_map[s[0]] = by_map.get(s[0],0)+1
    print('config', (press,release), 'maps', by_map)
pyboy.stop()
