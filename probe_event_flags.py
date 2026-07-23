"""Sanity-check ADDR_EVENT_FLAGS_START/END against a real save state.

Parses pret/pokeyellow's constants/event_constants.asm (fetched fresh over
the network, same source used to derive the address) to build a bit-index ->
event-name map, then loads milestones/07_beat_brock.state and reports which
named events are currently set. Confirms the derived address is plausible:
early-game events (got starter, beat Brock) should be set; late-game events
should not.
"""
import re
import urllib.request

from pyboy import PyBoy

import train
from project_paths import ROM_PATH, milestone_state_path

ADDR_EVENT_FLAGS_START = train.ADDR_EVENT_FLAGS_START
ADDR_EVENT_FLAGS_END = train.ADDR_EVENT_FLAGS_END


def fetch_event_constants():
    req = urllib.request.Request(
        "https://raw.githubusercontent.com/pret/pokeyellow/master/constants/event_constants.asm",
        headers={"User-Agent": "PokemonYellowTrainer/1.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", "replace")


def build_index_to_name(text):
    index_to_name = {}
    current = 0
    in_const_def = False
    for raw in text.splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        m = re.match(r"const_def\s*(\$?[0-9A-Fa-f]*)", line)
        if m and line.startswith("const_def"):
            val = m.group(1)
            current = int(val.lstrip("$"), 16 if val.startswith("$") else 10) if val else 0
            in_const_def = True
            continue
        m = re.match(r"const_next\s+\$([0-9A-Fa-f]+)", line)
        if m:
            current = int(m.group(1), 16)
            continue
        m = re.match(r"const_skip\s+(\d+)", line)
        if m:
            current += int(m.group(1))
            continue
        if line == "const_skip":
            current += 1
            continue
        m = re.match(r"const\s+(\w+)", line)
        if m and in_const_def:
            index_to_name[current] = m.group(1)
            current += 1
            continue
    return index_to_name


def main():
    print("Fetching event_constants.asm ...")
    text = fetch_event_constants()
    index_to_name = build_index_to_name(text)
    print(f"Parsed {len(index_to_name)} named event flags "
          f"(range 0..{max(index_to_name)})")

    state_file = milestone_state_path("07_beat_brock")
    pyboy = PyBoy(str(ROM_PATH), window="null")
    with open(state_file, "rb") as f:
        pyboy.load_state(f)
    memory = pyboy.memory

    set_bits = []
    for addr in range(ADDR_EVENT_FLAGS_START, ADDR_EVENT_FLAGS_END):
        byte_val = int(memory[addr])
        if byte_val == 0:
            continue
        base_bit = (addr - ADDR_EVENT_FLAGS_START) * 8
        for bit in range(8):
            if byte_val & (1 << bit):
                set_bits.append(base_bit + bit)

    print(f"\nTotal set bits: {len(set_bits)}")
    print("\nSet events with known names:")
    for idx in sorted(set_bits):
        name = index_to_name.get(idx)
        if name:
            print(f"  [{idx}] {name}")

    unnamed = [idx for idx in set_bits if idx not in index_to_name]
    print(f"\nSet bits with no matching name in this parse: {len(unnamed)} {unnamed[:20]}")

    print("\nSanity checks:")
    checks = [
        "EVENT_GOT_STARTER",
        "EVENT_BEAT_BROCK" if "EVENT_BEAT_BROCK" in text else None,
    ]
    name_to_index = {v: k for k, v in index_to_name.items()}
    for name in checks:
        if not name:
            continue
        idx = name_to_index.get(name)
        if idx is None:
            print(f"  {name}: not found in parsed constants")
            continue
        is_set = idx in set_bits
        print(f"  {name} (bit {idx}): {'SET' if is_set else 'not set'}")

    # A handful of far-future/late-game events that should NOT be set yet.
    late_game_candidates = [
        n for n in name_to_index
        if any(tag in n for tag in ("SEAFOAM", "ARTICUNO", "ELITE_FOUR", "CHAMPION", "MEWTWO"))
    ]
    print("\n  Late-game events (should be unset):")
    for name in late_game_candidates[:8]:
        idx = name_to_index[name]
        print(f"    {name} (bit {idx}): {'SET (unexpected!)' if idx in set_bits else 'not set (expected)'}")

    pyboy.stop()


if __name__ == "__main__":
    main()
