"""Offline, validated navigation metadata for the Pokémon Yellow trainer."""

from __future__ import annotations

import argparse
import json
from collections import deque
from functools import lru_cache
from pathlib import Path

from project_paths import PROJECT_ROOT


CATALOG_PATH = PROJECT_ROOT / "navigation_data" / "yellow_catalog.json"
EVENT_FLAGS_CATALOG_PATH = PROJECT_ROOT / "navigation_data" / "yellow_event_flags.json"

# Macro-level topology only. It selects the next area; local collision/warp
# pathfinding still needs ROM tile data and live RAM coordinates.
WORLD_EDGES = (
    ("Pallet Town", "Route 1"),
    ("Route 1", "Viridian City"),
    ("Viridian City", "Route 2"),
    ("Viridian City", "Route 22"),
    ("Route 2", "Viridian Forest"),
    ("Route 2", "Pewter City"),
    ("Route 2", "Diglett's Cave"),
    ("Pewter City", "Route 3"),
    ("Route 3", "Mt. Moon"),
    ("Mt. Moon", "Route 4"),
    ("Route 4", "Cerulean City"),
    ("Cerulean City", "Route 5"),
    ("Cerulean City", "Route 9"),
    ("Cerulean City", "Route 24"),
    ("Cerulean City", "Cerulean Cave"),
    ("Route 24", "Route 25"),
    ("Route 5", "Underground Path (Rte 5-6)"),
    ("Underground Path (Rte 5-6)", "Route 6"),
    ("Route 6", "Vermilion City"),
    ("Vermilion City", "S.S. Anne"),
    ("Vermilion City", "Route 11"),
    ("Route 11", "Diglett's Cave"),
    ("Route 9", "Route 10"),
    ("Route 10", "Power Plant"),
    ("Route 10", "Rock Tunnel"),
    ("Rock Tunnel", "Lavender Town"),
    ("Lavender Town", "Pokémon Tower"),
    ("Lavender Town", "Route 8"),
    ("Lavender Town", "Route 12"),
    ("Route 8", "Underground Path (Rte 7-8)"),
    ("Underground Path (Rte 7-8)", "Route 7"),
    ("Route 7", "Celadon City"),
    ("Celadon City", "Team Rocket Hideout"),
    ("Route 5", "Saffron City"),
    ("Route 6", "Saffron City"),
    ("Route 7", "Saffron City"),
    ("Route 8", "Saffron City"),
    ("Saffron City", "Silph Co."),
    ("Celadon City", "Route 16"),
    ("Route 16", "Route 17"),
    ("Route 17", "Route 18"),
    ("Route 18", "Fuchsia City"),
    ("Fuchsia City", "Safari Zone"),
    ("Fuchsia City", "Route 15"),
    ("Route 15", "Route 14"),
    ("Route 14", "Route 13"),
    ("Route 13", "Route 12"),
    ("Fuchsia City", "Route 19"),
    ("Route 19", "Seafoam Islands"),
    ("Seafoam Islands", "Route 20"),
    ("Route 20", "Cinnabar Island"),
    ("Cinnabar Island", "Pokémon Mansion"),
    ("Cinnabar Island", "Route 21"),
    ("Route 21", "Pallet Town"),
    ("Viridian City", "Route 22"),
    ("Route 22", "Route 23"),
    ("Route 23", "Victory Road"),
    ("Victory Road", "Indigo Plateau"),
)


@lru_cache(maxsize=1)
def load_catalog(path: Path = CATALOG_PATH) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    validate_catalog(catalog)
    return catalog


def validate_catalog(catalog: dict) -> None:
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported navigation catalog schema")
    maps = catalog.get("game_maps")
    locations = catalog.get("completion_locations")
    if not isinstance(maps, list) or not isinstance(locations, list):
        raise ValueError("Navigation catalog must contain map and location lists")
    ids = [record["id"] for record in maps]
    if len(ids) != len(set(ids)):
        raise ValueError("Navigation catalog contains duplicate map IDs")
    by_id = {record["id"]: record for record in maps}
    expected = {0: "PALLET_TOWN", 12: "ROUTE_1", 51: "VIRIDIAN_FOREST", 54: "PEWTER_GYM"}
    for map_id, constant in expected.items():
        if by_id.get(map_id, {}).get("constant") != constant:
            raise ValueError(f"Map {map_id} must be {constant}")


def map_record(map_id: int) -> dict | None:
    return next(
        (record for record in load_catalog()["game_maps"] if record["id"] == int(map_id)),
        None,
    )


def map_label(map_id: int) -> str:
    record = map_record(map_id)
    return record["name"] if record else f"Map {int(map_id)}"


def map_labels() -> dict[int, str]:
    return {record["id"]: record["name"] for record in load_catalog()["game_maps"]}


def completion_location_for_map(map_id: int) -> str | None:
    record = map_record(map_id)
    return record.get("completion_location") if record else None


@lru_cache(maxsize=1)
def load_event_flags_catalog(path: Path = EVENT_FLAGS_CATALOG_PATH) -> dict:
    """wEventFlags bit -> name catalog (see sync_event_flags.py). Bit-to-byte
    mapping is byte_offset = bit // 8, bit_in_byte = bit % 8, LSB = bit 0 --
    verified against pret/pokeyellow's engine/flag_action.asm FlagAction
    routine, not assumed."""
    with Path(path).open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("events"), list):
        raise ValueError("Unsupported event-flags catalog schema")
    return catalog


@lru_cache(maxsize=1)
def event_flags_by_bit() -> dict[int, dict]:
    return {event["bit"]: event for event in load_event_flags_catalog()["events"]}


def event_flag_name(bit: int) -> str:
    record = event_flags_by_bit().get(int(bit))
    return record["name"] if record else f"EVENT_UNNAMED_{int(bit)}"


def named_event_bits_set(flag_bytes: bytes) -> set[int]:
    """Given the raw wEventFlags byte range (e.g. pyboy.memory[ADDR_EVENT_FLAGS_START:
    ADDR_EVENT_FLAGS_END]), return the set of *named* bit indices currently set.
    Named-only by design -- most of the 2560-bit range has no assigned constant."""
    named_bits = event_flags_by_bit()
    set_bits = set()
    for bit in named_bits:
        byte_offset, bit_in_byte = divmod(bit, 8)
        if byte_offset < len(flag_bytes) and (flag_bytes[byte_offset] >> bit_in_byte) & 1:
            set_bits.add(bit)
    return set_bits


def map_neighbors(map_id: int) -> list[int]:
    """Return maps connected by an outdoor edge or explicit warp."""
    record = map_record(map_id)
    if record is None:
        return []
    neighbors = {
        edge["destination_map_id"]
        for field in ("connections", "warps")
        for edge in record.get(field, [])
        if edge.get("destination_map_id") is not None
    }
    return sorted(neighbors)


def warp_entry_action(map_id: int, launch_position: tuple[int, int], destination_map_id: int) -> int:
    """Return the trainer action that steps from a launch tile onto a warp."""
    record = map_record(map_id)
    if record is None:
        raise ValueError(f"Unknown map ID: {map_id}")
    destination_map_id = int(destination_map_id)
    launch_y, launch_x = map(int, launch_position)
    for warp in record.get("warps", []):
        if warp.get("destination_map_id") != destination_map_id:
            continue
        warp_y, warp_x = warp["position"]
        delta = (warp_y - launch_y, warp_x - launch_x)
        actions = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
        if delta in actions:
            return actions[delta]
    raise ValueError(
        f"No adjacent warp from map {map_id} at {launch_position} to map {destination_map_id}"
    )


def map_route(start_map_id: int, destination_map_id: int) -> list[int]:
    """Find a map-level route; local reachability still requires tile routing."""
    start_map_id = int(start_map_id)
    destination_map_id = int(destination_map_id)
    if map_record(start_map_id) is None or map_record(destination_map_id) is None:
        raise ValueError("Unknown start or destination map ID")
    queue = deque([(start_map_id, [start_map_id])])
    seen = {start_map_id}
    while queue:
        node, path = queue.popleft()
        if node == destination_map_id:
            return path
        for neighbor in map_neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
    raise ValueError(f"No map route from {start_map_id} to {destination_map_id}")


def macro_route(start: str, destination: str) -> list[str]:
    graph: dict[str, set[str]] = {}
    for left, right in WORLD_EDGES:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    if start not in graph:
        raise ValueError(f"Unknown start location: {start}")
    if destination not in graph:
        raise ValueError(f"Unknown destination: {destination}")

    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == destination:
            return path
        for neighbor in sorted(graph[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
    raise ValueError(f"No route from {start} to {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    identify = subparsers.add_parser("identify", help="Resolve an internal map ID")
    identify.add_argument("map_id", type=lambda value: int(value, 0))
    route = subparsers.add_parser("route", help="Find a macro route between locations")
    route.add_argument("start")
    route.add_argument("destination")
    map_route_parser = subparsers.add_parser("map-route", help="Find a ROM map-level route")
    map_route_parser.add_argument("start_map_id", type=lambda value: int(value, 0))
    map_route_parser.add_argument("destination_map_id", type=lambda value: int(value, 0))
    subparsers.add_parser("validate", help="Validate the generated catalog")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        catalog = load_catalog()
        print(
            f"Valid: {len(catalog['game_maps'])} maps, "
            f"{len(catalog['completion_locations'])} completion locations"
        )
    elif args.command == "identify":
        record = map_record(args.map_id)
        if record is None:
            raise SystemExit(f"Unknown map ID: {args.map_id}")
        print(json.dumps(record, indent=2, ensure_ascii=False))
    elif args.command == "route":
        print(" -> ".join(macro_route(args.start, args.destination)))
    else:
        route = map_route(args.start_map_id, args.destination_map_id)
        print(" -> ".join(f"{map_id}:{map_label(map_id)}" for map_id in route))


if __name__ == "__main__":
    main()
