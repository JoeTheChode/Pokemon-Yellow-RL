import os

_requested_hip_device = os.environ.get("POKEMON_HIP_VISIBLE_DEVICES")
if _requested_hip_device is not None:
    os.environ["HIP_VISIBLE_DEVICES"] = _requested_hip_device

from pyboy import PyBoy

from project_paths import ROM_PATH, START_STATE_PATH


pyboy = PyBoy(str(ROM_PATH), window='SDL2')

# Verify ROM identity.
rom_title = bytes(pyboy.memory[0x0134 + i] for i in range(15)).decode('ascii', errors='replace').strip('\x00')
print(f"ROM Title: '{rom_title}'")
print()

# Load save state.
with open(START_STATE_PATH, 'rb') as f:
    pyboy.load_state(f)

# Scan for map ID address.
# In the bedroom, the correct map address should read 0x26 (38) = REDS_HOUSE_2F
# or 0x25 (37) = REDS_HOUSE_1F.
print("=== Scanning WRAM for expected bedroom map ID (38 or 37) ===")
for addr in range(0xD340, 0xD380):
    val = pyboy.memory[addr]
    if val in [0x25, 0x26]:
        marker = '<-- REDS_HOUSE_2F!' if val == 0x26 else '<-- REDS_HOUSE_1F!'
        print(f"  0x{addr:04X} = {val} (0x{val:02X}) {marker}")

print()
print("=== Values at commonly used addresses ===")
for addr, name in [
    (0xD35E, "wCurMap (Red/Blue)"),
    (0xD361, "wYCoord (Red/Blue)"),
    (0xD362, "wXCoord (Red/Blue)"),
    (0xD356, "wBadges (Red/Blue)"),
]:
    val = pyboy.memory[addr]
    print(f"  0x{addr:04X} ({name}) = {val} (0x{val:02X})")

print()
print("=== Full dump D340-D380 ===")
for addr in range(0xD340, 0xD380):
    val = pyboy.memory[addr]
    marker = ""
    if addr == 0xD35E:
        marker = " <-- D35E (expected map)"
    if addr == 0xD361:
        marker = " <-- D361 (expected Y)"
    if addr == 0xD362:
        marker = " <-- D362 (expected X)"
    print(f"  0x{addr:04X} = {val:>3} (0x{val:02X}){marker}")

print()
print("Walk around in the game window, then press Ctrl+C.")
print("After you move to a new room, the scan will repeat.")
print()

frame = 0
last_map = pyboy.memory[0xD35E]
try:
    while pyboy.tick():
        frame += 1
        cur_map = pyboy.memory[0xD35E]

        if frame % 60 == 0:
            vals = {addr: pyboy.memory[addr] for addr in range(0xD350, 0xD370)}
            print(f"D350-D36F: {' '.join(f'{val:>3}' for val in vals.values())}", end='\r')

        if cur_map != last_map:
            last_map = cur_map
            print(f"\n\n=== MAP CHANGED! Rescanning ===")
            print(f"0xD35E now = {cur_map}")
            print("Scanning for known map IDs (0=Pallet, 12=Route1, 37=House1F, 38=House2F, 40=OaksLab)...")
            for addr in range(0xD340, 0xD380):
                val = pyboy.memory[addr]
                known = {
                    0: "PALLET_TOWN",
                    12: "ROUTE_1",
                    37: "REDS_HOUSE_1F",
                    38: "REDS_HOUSE_2F",
                    40: "OAKS_LAB",
                }
                if val in known:
                    print(f"  0x{addr:04X} = {val} -> {known[val]}")
            print()

except KeyboardInterrupt:
    pass

pyboy.stop()
