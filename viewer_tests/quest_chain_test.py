#!/usr/bin/env python3
"""Quest-chain export + the viewer server's use of it.

Covers the two halves of the objective-card fix that the DOM test cannot see:
the descriptions themselves (each clause must mirror a real check in train.py's
`quest_waypoint_matches`) and the server's staleness detection (a chain that no
longer matches train.py describes the wrong objective, not merely an old one).

Runs under the trainer venv or system python3 -- nothing here imports train.py.

    ./.venv/bin/python viewer_tests/quest_chain_test.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import export_quest_chain as exporter  # noqa: E402
import map_viewer_server as server  # noqa: E402

FAILS = []


def ok(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(label)


MAPS = {1: "Viridian City", 64: "Cerulean Pokémon Center"}
EVENTS = {152: "EVENT_BEAT_CERULEAN_RIVAL"}


def goal_for(waypoint):
    return exporter.sentence(exporter.describe(waypoint, MAPS, EVENTS))


def test_descriptions():
    print("-- descriptions")

    goal = goal_for({'name': 'beat_cerulean_rival', 'event_bit': 152})
    ok("event objective names the flag and its constant",
       "flag 152" in goal and "EVENT_BEAT_CERULEAN_RIVAL" in goal, goal)

    goal = goal_for({'name': 'reach_viridian', 'map': 1})
    ok("travel objective resolves the map name", "Viridian City (map 1)" in goal, goal)

    goal = goal_for({'name': 'x', 'map': 999})
    ok("unknown map id renders as a number, not a guess",
       "map 999" in goal and "(map 999)" not in goal, goal)

    # Map and coordinate box are one condition on where the agent stands.
    goal = goal_for({'name': 'heal_before_misty', 'map': 64, 'max_y': 3,
                     'require_full_hp': True})
    ok("map and bounds read as one location clause",
       "Cerulean Pokémon Center (map 64) at y ≤ 3" in goal, goal)
    ok("full-HP requirement stated", "full HP" in goal, goal)

    # The matcher refuses a level objective mid-battle or mid-dialogue, because
    # XP lands on the faint frame and that savestate is not a durable
    # checkpoint. Omitting it would describe a gate the trainer does not apply.
    goal = goal_for({'name': 'train_pikachu_level_31', 'min_level': 31})
    ok("grind states the level bar", "level 31+" in goal, goal)
    ok("grind states the out-of-battle requirement",
       "no battle or dialogue is on screen" in goal, goal)

    goal = goal_for({'name': 'buy_guard_drink', 'any_bag_item_ids': (60, 61, 62)})
    ok("any-of item list names each item",
       all(n in goal for n in ("Fresh Water", "Soda Pop", "Lemonade")), goal)

    goal = goal_for({'name': 'give_saffron_guards_drink',
                     'status_flag_addr': 0xD727, 'status_flag_mask': 0x40})
    ok("raw WRAM condition rendered in hex", "0xD727" in goal and "0x40" in goal, goal)

    ok("a waypoint with no conditions yields no sentence rather than a stub",
       goal_for({'name': 'nothing'}) == "", repr(goal_for({'name': 'nothing'})))

    kinds = {
        'grind': {'min_level': 5}, 'travel': {'map': 1}, 'event': {'event_bit': 152},
        'heal': {'require_full_hp': True}, 'item': {'bag_item_id': 74},
        'teach': {'party_move_id': 15}, 'badge': {'min_badges': 2},
        'party': {'party_species_id': 180}, 'flag': {'status_flag_addr': 0xD727},
    }
    bad = [k for k, w in kinds.items() if exporter.kind_of(w) != k]
    ok("every objective bucket classifies", not bad, f"wrong: {bad}")


def test_chain_loading():
    print("-- chain loading")
    with tempfile.TemporaryDirectory() as tmp:
        train_path = os.path.join(tmp, "train.py")
        chain_path = os.path.join(tmp, "quest_chain.json")
        with open(train_path, "w", encoding="utf-8") as handle:
            handle.write("# stand-in for train.py\n")
        stat = os.stat(train_path)

        def write_chain(source):
            with open(chain_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "schema": 1, "generated": "2026-08-16T00:00:00+00:00",
                    "source": source, "count": 1,
                    "phases": [{"phase": 56, "name": "beat_cerulean_rival",
                                "kind": "event", "criteria": ["x"],
                                "goal": "Clears when x."}],
                }, handle)

        server.TRAIN_PY_PATH = train_path
        server.QUEST_CHAIN_PATH = chain_path

        def reload_chain():
            server._CHAIN_CACHE["mtime"] = None
            server._CHAIN_CACHE["chain"] = None
            return server.load_quest_chain()

        write_chain({"mtime_ns": stat.st_mtime_ns, "size": stat.st_size})
        chain = reload_chain()
        ok("chain phases keyed by int phase", 56 in chain["phases"],
           list(chain["phases"])[:3])
        ok("matching fingerprint is not stale", chain["stale"] is False)

        # A quest edit shifts every later phase index, so an out-of-date chain
        # does not merely age -- it starts describing a different objective.
        write_chain({"mtime_ns": stat.st_mtime_ns, "size": stat.st_size + 1})
        ok("size drift marks the chain stale", reload_chain()["stale"] is True)

        write_chain({"mtime_ns": stat.st_mtime_ns + 1, "size": stat.st_size})
        ok("mtime drift marks the chain stale", reload_chain()["stale"] is True)

        # Missing or corrupt must degrade to the old name-only card, never
        # raise inside a request thread.
        with open(chain_path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        chain = reload_chain()
        ok("corrupt chain degrades quietly", chain["phases"] == {} and chain["stale"] is None)

        os.remove(chain_path)
        chain = reload_chain()
        ok("missing chain degrades quietly", chain["phases"] == {} and chain["stale"] is None)


def main():
    test_descriptions()
    test_chain_loading()
    print(f"\n{len(FAILS)} FAILED" if FAILS else "\nall quest chain checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
