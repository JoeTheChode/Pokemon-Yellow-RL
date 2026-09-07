#!/usr/bin/env python3
"""Dump train.py's quest chain to navigation_data/quest_chain.json.

The viewer's objective card took its label from the *splits ledger*, which is
built by parsing `[QUEST]` completion lines out of train.log. An objective that
has never been completed therefore has no row -- and the objective being
drilled right now is, by definition, exactly that objective. So the card fell
back to the literal string "phase 56": it named the phase index and never said
what the agents were trying to do.

The chain itself lives in train.py as FULLGAME_QUEST_WAYPOINTS, where each
waypoint is a dict of clear conditions. Importing train.py pulls in torch,
pyboy and stable-baselines3, which has no business happening inside a viewer
HTTP request, so the chain is exported here once and read back as plain JSON.

Every clause below mirrors a check in train.py's `quest_waypoint_matches`, in
the same order. If a condition is added there and not here it will simply be
omitted from the description rather than described wrongly -- but the two are
meant to stay in step.

Run from the trainer venv (system python3 has no gymnasium):

    ./.venv/bin/python export_quest_chain.py

start_viewer.sh runs it on every viewer start, so a viewer restart is enough to
pick up a chain edit.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

TRAIN_PY_PATH = os.path.join(ROOT, "train.py")
CATALOG_PATH = os.path.join(ROOT, "navigation_data", "yellow_catalog.json")
EVENT_FLAGS_PATH = os.path.join(ROOT, "navigation_data", "yellow_event_flags.json")
OUT_PATH = os.path.join(ROOT, "navigation_data", "quest_chain.json")
SCHEMA = 1

# Only the IDs the chain actually references are named, and each was checked
# against the objective's own name (bag_item_id 74 on `collect_lift_key`, and
# so on). An unknown ID renders as a bare number rather than a guess.
MOVE_NAMES = {15: "Cut", 19: "Fly", 57: "Surf", 70: "Strength"}
SPECIES_NAMES = {176: "Charmander", 178: "Charmeleon", 180: "Charizard"}
ITEM_NAMES = {
    43: "Secret Key",
    48: "Card Key",
    60: "Fresh Water",
    61: "Soda Pop",
    62: "Lemonade",
    64: "Gold Teeth",
    72: "Silph Scope",
    74: "Lift Key",
}


def load_map_names():
    """Map id -> in-game name, from the ROM-derived catalog."""
    try:
        with open(CATALOG_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    names = {}
    for entry in data.get("game_maps") or []:
        if isinstance(entry, dict) and entry.get("id") is not None:
            names[int(entry["id"])] = str(entry.get("name") or "").strip()
    return {k: v for k, v in names.items() if v}


def load_event_names():
    """Event-flag bit -> EVENT_* constant name."""
    try:
        with open(EVENT_FLAGS_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    names = {}
    for entry in data.get("events") or []:
        if isinstance(entry, dict) and entry.get("bit") is not None:
            names[int(entry["bit"])] = str(entry.get("name") or "").strip()
    return {k: v for k, v in names.items() if v}


def map_label(map_id, map_names):
    name = map_names.get(int(map_id))
    return f"{name} (map {int(map_id)})" if name else f"map {int(map_id)}"


def event_label(bit, event_names):
    name = event_names.get(int(bit))
    return f"flag {int(bit)} ({name})" if name else f"flag {int(bit)}"


def id_label(value, table):
    name = table.get(int(value))
    return f"{name} (#{int(value)})" if name else f"#{int(value)}"


def bounds_clause(waypoint):
    """The x/y box, if any. pos_a is y and pos_b is x in the matcher."""
    parts = []
    for axis, lo_key, hi_key in (("y", "min_y", "max_y"), ("x", "min_x", "max_x")):
        lo = waypoint.get(lo_key)
        hi = waypoint.get(hi_key)
        if lo is not None and hi is not None:
            parts.append(f"{axis} {int(lo)}–{int(hi)}" if lo != hi
                         else f"{axis} = {int(lo)}")
        elif lo is not None:
            parts.append(f"{axis} ≥ {int(lo)}")
        elif hi is not None:
            parts.append(f"{axis} ≤ {int(hi)}")
    return f"at {', '.join(parts)}" if parts else None


def describe(waypoint, map_names, event_names):
    """Clear conditions for one waypoint, in `quest_waypoint_matches` order.

    Each clause carries its own subject and verb so they read as a sentence
    after "Clears when " no matter which subset a waypoint uses.
    """
    clauses = []

    # Map and coordinate box are one condition on where the agent stands, so
    # they read as one clause rather than two disconnected ones.
    bounds = bounds_clause(waypoint)
    if "map" in waypoint:
        where = f"the agent is in {map_label(waypoint['map'], map_names)}"
        clauses.append(f"{where} {bounds}" if bounds else where)
    elif bounds:
        clauses.append(f"the agent is {bounds}")
    if "min_named_events" in waypoint:
        clauses.append(
            f"{int(waypoint['min_named_events'])} named events have been reached")
    if "event_bit" in waypoint:
        clauses.append(
            f"event {event_label(waypoint['event_bit'], event_names)} is set")
    if "any_event_bits" in waypoint:
        joined = " / ".join(
            event_label(bit, event_names) for bit in waypoint["any_event_bits"])
        clauses.append(f"any of event {joined} is set")
    if "min_level" in waypoint:
        clauses.append(
            f"the lead Pokémon is level {int(waypoint['min_level'])}+")
    if "party_move_id" in waypoint:
        clauses.append(
            f"a party member knows {id_label(waypoint['party_move_id'], MOVE_NAMES)}")
    if "bag_item_id" in waypoint:
        clauses.append(
            f"the bag holds {id_label(waypoint['bag_item_id'], ITEM_NAMES)}")
    if "any_bag_item_ids" in waypoint:
        joined = " / ".join(
            id_label(item, ITEM_NAMES) for item in waypoint["any_bag_item_ids"])
        clauses.append(f"the bag holds any of {joined}")
    if "party_species_id" in waypoint:
        clauses.append(
            f"the party includes {id_label(waypoint['party_species_id'], SPECIES_NAMES)}")
    if "min_party_species_level" in waypoint:
        level = int(waypoint["min_party_species_level"])
        species = waypoint.get("party_species_ids")
        if species:
            joined = " / ".join(id_label(s, SPECIES_NAMES) for s in species)
            clauses.append(f"the party has {joined} at level {level}+")
        else:
            clauses.append(f"a party member is level {level}+")
    if "status_flag_addr" in waypoint:
        mask = int(waypoint.get("status_flag_mask", 0xFF))
        clauses.append(
            f"WRAM 0x{int(waypoint['status_flag_addr']):04X} "
            f"has bits 0x{mask:02X} set")
    if "min_badges" in waypoint:
        clauses.append(f"the badge count is {int(waypoint['min_badges'])}+")
    if "blackout_map" in waypoint:
        clauses.append(
            "the respawn point is "
            f"{map_label(waypoint['blackout_map'], map_names)}")
    if waypoint.get("require_full_hp"):
        clauses.append("the whole party is at full HP")
    if waypoint.get("require_dialogue_closed"):
        clauses.append("no dialogue box is open")
    # The matcher rejects any level objective while a battle or text box is up,
    # because XP lands on the faint frame and a mid-battle savestate is not a
    # durable checkpoint. That is a real part of the clear condition.
    if "min_level" in waypoint or "min_party_species_level" in waypoint:
        clauses.append("no battle or dialogue is on screen")

    return clauses


def kind_of(waypoint):
    """Coarse bucket for the objective, for grouping and styling."""
    if "min_level" in waypoint or "min_party_species_level" in waypoint:
        return "grind"
    if "party_move_id" in waypoint:
        return "teach"
    if "bag_item_id" in waypoint or "any_bag_item_ids" in waypoint:
        return "item"
    if "party_species_id" in waypoint:
        return "party"
    if waypoint.get("require_full_hp") or "blackout_map" in waypoint:
        return "heal"
    if "min_badges" in waypoint:
        return "badge"
    if "event_bit" in waypoint or "any_event_bits" in waypoint:
        return "event"
    if "status_flag_addr" in waypoint:
        return "flag"
    if "map" in waypoint:
        return "travel"
    return "other"


def sentence(clauses):
    if not clauses:
        return ""
    if len(clauses) == 1:
        body = clauses[0]
    else:
        body = ", ".join(clauses[:-1]) + " and " + clauses[-1]
    return f"Clears when {body}."


def source_fingerprint(path):
    stat = os.stat(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": os.path.basename(path),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def build():
    import train  # noqa: E402 -- heavy import, deliberately not at module scope

    map_names = load_map_names()
    event_names = load_event_names()
    phases = []
    for phase, waypoint in enumerate(train.FULLGAME_QUEST_WAYPOINTS):
        clauses = describe(waypoint, map_names, event_names)
        phases.append({
            "phase": phase,
            "name": waypoint.get("name") or "",
            "kind": kind_of(waypoint),
            "criteria": clauses,
            "goal": sentence(clauses),
            "reward": waypoint.get("reward"),
        })
    return {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_fingerprint(TRAIN_PY_PATH),
        "count": len(phases),
        "phases": phases,
    }


def main():
    payload = build()
    # Written atomically: the viewer server reads this file from request
    # threads and must never see a half-written chain.
    tmp_path = f"{OUT_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    os.replace(tmp_path, OUT_PATH)
    print(f"wrote {OUT_PATH}: {payload['count']} phases "
          f"from train.py sha {payload['source']['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
