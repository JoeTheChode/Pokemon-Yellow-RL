
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

def act(action):
    pyboy.button_press(action); pyboy.tick(8); pyboy.button_release(action); pyboy.tick(16)
seen={}; data_by_state={}; q=deque()
load(root); s0=snap(); seen[s0]=[]; data_by_state[s0]=root; q.append((root, []))
max_depth=120
while q and len(seen) < 20000:
    data, seq=q.popleft()
    if len(seq) >= max_depth:
        continue
    for action in ACTIONS:
        load(data); act(action); cur=snap()
        if cur not in seen:
            next_seq=seq+[action]
            seen[cur]=next_seq
            saved=save()
            data_by_state[cur]=saved
            q.append((saved, next_seq))
for state, seq in sorted(seen.items(), key=lambda kv:(kv[0][0], kv[0][1], kv[0][2])):
    if state[0] != 2:
        continue
    data = data_by_state[state]
    exits=[]
    for action in ACTIONS:
        load(data); act(action); cur=snap()
        if cur[0] != 2:
            exits.append((action, cur))
    if exits:
        print('state', state, 'path', seq, 'exits', exits)
pyboy.stop()
