"""Empirically locate Gen1's move-data table in yellow.gb by brute-force
searching for a table_base offset where all known (move_id, max_pp) pairs
line up, using the standard 5-byte record layout: effect, power, type,
accuracy, pp (move_id is 1-indexed).
"""
from project_paths import ROM_PATH

KNOWN_PP = {
    21: 20,   # Slam
    39: 30,   # Tail Whip
    45: 40,   # Growl
    84: 30,   # ThunderShock
    85: 15,   # Thunderbolt
    86: 20,   # Thunder Wave
    98: 30,   # Quick Attack
    129: 20,  # Swift
}
rom = ROM_PATH.read_bytes()
print(f"ROM size: {len(rom)} bytes")

found_any = False
for stride in range(4, 8):
    for pp_offset in range(stride):
        max_base = len(rom) - (max(KNOWN_PP) - 1) * stride - stride
        for base in range(0, max_base):
            ok = True
            for move_id, pp in KNOWN_PP.items():
                idx = base + (move_id - 1) * stride + pp_offset
                if rom[idx] != pp:
                    ok = False
                    break
            if ok:
                found_any = True
                print(f"MATCH stride={stride} pp_offset={pp_offset} base=0x{base:06X} ({base})")
    print(f"...checked stride={stride}")

if not found_any:
    print("No candidates found for stride 4-7. Trying 0-indexed move_id instead of 1-indexed...")
    for stride in range(4, 8):
        for pp_offset in range(stride):
            max_base = len(rom) - (max(KNOWN_PP)) * stride - stride
            for base in range(0, max_base):
                ok = True
                for move_id, pp in KNOWN_PP.items():
                    idx = base + move_id * stride + pp_offset
                    if rom[idx] != pp:
                        ok = False
                        break
                if ok:
                    found_any = True
                    print(f"MATCH(0-idx) stride={stride} pp_offset={pp_offset} base=0x{base:06X} ({base})")
    if not found_any:
        print("Still nothing. Table likely isn't a flat linear array of raw PP bytes at a fixed stride visible this way.")
