"""Build the local Yellow navigation catalog from authoritative/source data.

The trainer never depends on the network at runtime. Run this script explicitly
to refresh map constants from pret/pokeyellow and map labels from the saved
Pokemon Completion page.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

from project_paths import PROJECT_ROOT, atomic_write_json


MAP_CONSTANTS_URL = (
    "https://raw.githubusercontent.com/pret/pokeyellow/master/constants/map_constants.asm"
)
COMPLETION_URL = "https://pokemoncompletion.com/completion/Yellow"
REPOSITORY_ARCHIVE_URL = "https://github.com/pret/pokeyellow/archive/refs/heads/master.zip"
DEFAULT_HTML = PROJECT_ROOT / "Pokémon Yellow Interactive Map & 100% Checklist Challenge.html"
DEFAULT_OUTPUT = PROJECT_ROOT / "navigation_data" / "yellow_catalog.json"

MAP_CONSTANT_RE = re.compile(
    r"map_const\s+([A-Z0-9_]+),\s*(\d+),\s*(\d+)\s*;\s*\$([0-9A-Fa-f]{2})"
)
MAP_LABEL_RE = re.compile(
    r'<div class="[^"]*\bmapLabel\b[^"]*"[^>]*style="[^"]*'
    r'translate3d\((-?\d+)px,\s*(-?\d+)px[^\"]*"[^>]*>\s*'
    r"<span[^>]*>(.*?)</span>",
    re.DOTALL,
)
MAP_LINK_RE = re.compile(
    r'<path[^>]+d="M(-?\d+) (-?\d+)L(-?\d+) (-?\d+)"', re.DOTALL
)
CONNECTION_RE = re.compile(
    r"connection\s+(north|south|east|west),\s*[A-Za-z0-9_]+,\s*([A-Z0-9_]+),\s*(-?\d+)"
)
WARP_RE = re.compile(
    r"warp_event\s+(-?\d+),\s*(-?\d+),\s*([A-Z0-9_]+),\s*(\d+)"
)


def display_name(constant: str) -> str:
    special = {
        "POKECENTER": "Pokémon Center",
        "POKEMON": "Pokémon",
        "OAKS": "Oak's",
        "REDS": "Red's",
        "BLUES": "Blue's",
        "BILLS": "Bill's",
        "MR_FUJIS": "Mr. Fuji's",
        "MR_PSYCHICS": "Mr. Psychic's",
        "SS_ANNE": "S.S. Anne",
        "MT_MOON": "Mt. Moon",
    }
    value = constant
    for source, replacement in special.items():
        value = value.replace(source, replacement.replace(" ", "_"))
    words = []
    for word in value.split("_"):
        if re.fullmatch(r"\d+[FB]", word):
            words.append(word)
        elif word in {"GYM", "LAB", "MART", "HQ"}:
            words.append(word.title() if word != "HQ" else word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def completion_parent(constant: str, completion_names: set[str]) -> str | None:
    direct = display_name(constant)
    if direct in completion_names:
        return direct

    prefixes = (
        ("VIRIDIAN_FOREST", "Viridian Forest"),
        ("MT_MOON", "Mt. Moon"),
        ("DIGLETTS_CAVE", "Diglett's Cave"),
        ("ROCK_TUNNEL", "Rock Tunnel"),
        ("POWER_PLANT", "Power Plant"),
        ("SS_ANNE", "S.S. Anne"),
        ("ROCKET_HIDEOUT", "Team Rocket Hideout"),
        ("SILPH_CO", "Silph Co."),
        ("POKEMON_TOWER", "Pokémon Tower"),
        ("SAFARI_ZONE", "Safari Zone"),
        ("SEAFOAM_ISLANDS", "Seafoam Islands"),
        ("POKEMON_MANSION", "Pokémon Mansion"),
        ("VICTORY_ROAD", "Victory Road"),
        ("CERULEAN_CAVE", "Cerulean Cave"),
    )
    for prefix, parent in prefixes:
        if constant.startswith(prefix):
            return parent

    exact_parents = {
        "REDS_HOUSE_1F": "Pallet Town",
        "REDS_HOUSE_2F": "Pallet Town",
        "BLUES_HOUSE": "Pallet Town",
        "OAKS_LAB": "Pallet Town",
        "MUSEUM_1F": "Pewter City",
        "MUSEUM_2F": "Pewter City",
        "BIKE_SHOP": "Cerulean City",
        "UNDERGROUND_PATH_ROUTE_5": "Underground Path (Rte 5-6)",
        "UNDERGROUND_PATH_ROUTE_6": "Underground Path (Rte 5-6)",
        "UNDERGROUND_PATH_ROUTE_6_COPY": "Underground Path (Rte 5-6)",
        "UNDERGROUND_PATH_NORTH_SOUTH": "Underground Path (Rte 5-6)",
        "UNDERGROUND_PATH_ROUTE_7": "Underground Path (Rte 7-8)",
        "UNDERGROUND_PATH_ROUTE_7_COPY": "Underground Path (Rte 7-8)",
        "UNDERGROUND_PATH_ROUTE_8": "Underground Path (Rte 7-8)",
        "UNDERGROUND_PATH_WEST_EAST": "Underground Path (Rte 7-8)",
        "DAYCARE": "Route 5",
        "BILLS_HOUSE": "Route 25",
        "POKEMON_FAN_CLUB": "Vermilion City",
        "LANCES_ROOM": "Indigo Plateau",
        "HALL_OF_FAME": "Indigo Plateau",
        "CHAMPIONS_ROOM": "Indigo Plateau",
        "LORELEIS_ROOM": "Indigo Plateau",
        "BRUNOS_ROOM": "Indigo Plateau",
        "AGATHAS_ROOM": "Indigo Plateau",
        "GAME_CORNER": "Celadon City",
        "GAME_CORNER_PRIZE_ROOM": "Celadon City",
        "MR_FUJIS_HOUSE": "Lavender Town",
        "NAME_RATERS_HOUSE": "Lavender Town",
        "WARDENS_HOUSE": "Fuchsia City",
        "COPYCATS_HOUSE_1F": "Saffron City",
        "COPYCATS_HOUSE_2F": "Saffron City",
        "FIGHTING_DOJO": "Saffron City",
        "MR_PSYCHICS_HOUSE": "Saffron City",
        "SUMMER_BEACH_HOUSE": "Route 19",
    }
    if constant in exact_parents:
        return exact_parents[constant]

    city_names = (
        "PALLET_TOWN",
        "VIRIDIAN_CITY",
        "PEWTER_CITY",
        "CERULEAN_CITY",
        "LAVENDER_TOWN",
        "VERMILION_CITY",
        "CELADON_CITY",
        "FUCHSIA_CITY",
        "CINNABAR_ISLAND",
        "INDIGO_PLATEAU",
        "SAFFRON_CITY",
    )
    for city in city_names:
        city_display = display_name(city)
        if constant == city or constant.startswith(city.removesuffix("_CITY")):
            return city_display if city_display in completion_names else None

    for city_prefix, parent in (
        ("LAVENDER_", "Lavender Town"),
        ("CINNABAR_", "Cinnabar Island"),
    ):
        if constant.startswith(city_prefix):
            return parent

    route_match = re.match(r"ROUTE_(\d+)(?:_|$)", constant)
    if route_match:
        parent = f"Route {int(route_match.group(1))}"
        return parent if parent in completion_names else None
    return None


def parse_completion_html(text: str) -> tuple[list[dict], list[dict]]:
    locations = []
    seen = set()
    for x, y, raw_name in MAP_LABEL_RE.findall(text):
        name = html.unescape(re.sub(r"<[^>]+>", "", raw_name)).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        locations.append({"name": name, "map_pixel": [int(x), int(y)]})

    links = []
    for x1, y1, x2, y2 in MAP_LINK_RE.findall(text):
        endpoints = [[int(x1), int(y1)], [int(x2), int(y2)]]
        if endpoints == [[0, 0], [0, 0]]:
            continue
        links.append({"endpoints": endpoints})
    return locations, links


def parse_map_constants(text: str, completion_names: set[str]) -> list[dict]:
    records = []
    for constant, width, height, hex_id in MAP_CONSTANT_RE.findall(text):
        map_id = int(hex_id, 16)
        records.append(
            {
                "id": map_id,
                "hex_id": f"0x{map_id:02X}",
                "constant": constant,
                "name": display_name(constant),
                "size_blocks": [int(width), int(height)],
                "completion_location": completion_parent(constant, completion_names),
            }
        )
    return records


def parse_repository_archive(archive_bytes: bytes, maps: list[dict]) -> None:
    """Attach inter-map connections and warp endpoints from the disassembly."""
    by_constant = {record["constant"]: record for record in maps}
    for record in maps:
        record["connections"] = []
        record["warps"] = []

    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        for name in archive.namelist():
            if not name.endswith(".asm"):
                continue
            if "/data/maps/headers/" in name:
                text = archive.read(name).decode("utf-8", "replace")
                header = re.search(
                    r"map_header\s+[A-Za-z0-9_]+,\s*([A-Z0-9_]+),", text
                )
                if not header or header.group(1) not in by_constant:
                    continue
                record = by_constant[header.group(1)]
                for direction, destination, offset in CONNECTION_RE.findall(text):
                    if destination not in by_constant:
                        continue
                    record["connections"].append(
                        {
                            "direction": direction,
                            "destination_map_id": by_constant[destination]["id"],
                            "destination_constant": destination,
                            "offset": int(offset),
                        }
                    )
            elif "/data/maps/objects/" in name:
                text = archive.read(name).decode("utf-8", "replace")
                source = re.search(r"def_warps_to\s+([A-Z0-9_]+)", text)
                if not source or source.group(1) not in by_constant:
                    continue
                record = by_constant[source.group(1)]
                for x, y, destination, destination_warp in WARP_RE.findall(text):
                    if destination != "LAST_MAP" and destination not in by_constant:
                        continue
                    record["warps"].append(
                        {
                            "position": [int(y), int(x)],
                            "destination_map_id": (
                                by_constant[destination]["id"] if destination in by_constant else None
                            ),
                            "destination_constant": destination,
                            "destination_warp": int(destination_warp),
                        }
                    )


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "PokemonYellowTrainer/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def fetch_text(url: str) -> str:
    return fetch_bytes(url).decode("utf-8", "replace")


def build_catalog(html_path: Path, constants_text: str, archive_bytes: bytes) -> dict:
    html_bytes = html_path.read_bytes()
    html_text = html_bytes.decode("utf-8", "replace")
    locations, rendered_links = parse_completion_html(html_text)
    completion_names = {location["name"] for location in locations}
    maps = parse_map_constants(constants_text, completion_names)
    if len(maps) < 200:
        raise ValueError(f"Expected at least 200 Yellow maps, extracted {len(maps)}")
    if len(locations) < 50:
        raise ValueError(f"Expected at least 50 completion locations, extracted {len(locations)}")
    parse_repository_archive(archive_bytes, maps)
    return {
        "schema_version": 1,
        "game": "Pokémon Yellow",
        "sources": {
            "map_constants": MAP_CONSTANTS_URL,
            "map_topology": REPOSITORY_ARCHIVE_URL,
            "completion_page": COMPLETION_URL,
            "completion_snapshot": html_path.name,
            "completion_snapshot_sha256": hashlib.sha256(html_bytes).hexdigest(),
        },
        "game_maps": maps,
        "completion_locations": locations,
        # These are retained as source evidence only. A browser snapshot renders
        # only currently instantiated Leaflet links, so they are not treated as
        # a complete or action-level navigation graph.
        "rendered_map_links": rendered_links,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--constants-file", type=Path, help="Use a local map_constants.asm")
    parser.add_argument("--repo-archive", type=Path, help="Use a local pokeyellow repository ZIP")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    constants_text = (
        args.constants_file.read_text(encoding="utf-8")
        if args.constants_file
        else fetch_text(MAP_CONSTANTS_URL)
    )
    archive_bytes = (
        args.repo_archive.read_bytes()
        if args.repo_archive
        else fetch_bytes(REPOSITORY_ARCHIVE_URL)
    )
    catalog = build_catalog(args.html, constants_text, archive_bytes)
    atomic_write_json(args.output, catalog, keep_backup=False)
    print(
        f"Wrote {args.output}: {len(catalog['game_maps'])} game maps, "
        f"{len(catalog['completion_locations'])} completion locations"
    )


if __name__ == "__main__":
    main()
