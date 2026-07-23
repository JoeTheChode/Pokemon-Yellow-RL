
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
seen={}; data_by_state={}; q=deque()
load(root); s0=snap(); seen[s0]=[]; data_by_state[s0]=root; q.append((root, []))
max_depth=24
while q and len(seen) < 20000:
    data, seq=q.popleft()
    if len(seq) >= max_depth:
        continue
    for action in ACTIONS:
        load(data)
        pyboy.button_press(action); pyboy.tick(96); pyboy.button_release(action); pyboy.tick(8)
        cur=snap()
        if cur not in seen:
            next_seq=seq+[action]
            seen[cur]=next_seq
            saved=save(); data_by_state[cur]=saved
            q.append((saved, next_seq))
for target_map in [52,54,57,58]:
    matches=[s for s in seen if s[0]==target_map]
    print('target', target_map, 'count', len(matches))
    for s in sorted(matches)[:10]:
        print(' ', s, seen[s])
pyboy.stop()
