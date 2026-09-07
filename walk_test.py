"""Read-only: load restart_start.state and walk north/around for a while,
logging map/position/level/battle_flag each time something changes, to see
whether the rival (Eevee) battle trigger is actually reachable from here."""
from pyboy import PyBoy
from pyboy.utils import WindowEvent

ADDR_MAP_ID = 0xD35D
ADDR_POS_A = 0xD360
ADDR_POS_B = 0xD361
ADDR_LEVEL = 0xD18B
ADDR_BATTLE_FLAG = 0xD057
ADDR_EVENT_FLAGS_START = 0xD747
ADDR_EVENT_FLAGS_END = 0xD887


def bit_set(flags, bit):
    byte_offset, bit_in_byte = divmod(bit, 8)
    return bool(flags[byte_offset] & (1 << bit_in_byte))


def snapshot(pyboy, label):
    mem = pyboy.memory
    flags = bytes(mem[ADDR_EVENT_FLAGS_START:ADDR_EVENT_FLAGS_END])
    print(
        f"{label}: map={int(mem[ADDR_MAP_ID])} x={int(mem[ADDR_POS_B])} "
        f"y={int(mem[ADDR_POS_A])} level={int(mem[ADDR_LEVEL])} "
        f"battle_flag={int(mem[ADDR_BATTLE_FLAG])} "
        f"rival_lab_flag={bit_set(flags, 35)}"
    )


pyboy = PyBoy("yellow.gb", window="null")
with open("restart_start.state", "rb") as f:
    pyboy.load_state(f)

snapshot(pyboy, "start")

# Pallet Town's exit to Route 1 is north in vanilla Red/Blue/Yellow.
# Walk up repeatedly, checking state periodically, for a generous budget.
DIRECTIONS = [
    (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
]

last_map = None
for step in range(600):
    press, release = DIRECTIONS[0]
    pyboy.send_input(press)
    pyboy.tick(8, False)
    pyboy.send_input(release)
    pyboy.tick(16, False)
    mem = pyboy.memory
    cur_map = int(mem[ADDR_MAP_ID])
    battle_flag = int(mem[ADDR_BATTLE_FLAG])
    if cur_map != last_map or battle_flag != 0 or step % 100 == 0:
        snapshot(pyboy, f"step {step}")
        last_map = cur_map
    if battle_flag == 2:
        print("TRAINER BATTLE ENCOUNTERED")
        break

snapshot(pyboy, "end")
pyboy.stop()
