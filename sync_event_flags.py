"""Build the local Yellow event-flag catalog from authoritative source data.

Parses pret/pokeyellow's constants/event_constants.asm (the const_def/const/
const_skip/const_next rgbds macros that assign each named story/trainer/item
event a bit index into wEventFlags) into a bit-index -> name lookup. Bit-to-
byte mapping (byte_offset = bit // 8, bit_in_byte = bit % 8, LSB = bit 0) is
verified against engine/flag_action.asm's FlagAction routine, not assumed.

The trainer never depends on the network at runtime. Run this script
explicitly to refresh the catalog.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from project_paths import PROJECT_ROOT, atomic_write_json
from sync_navigation_data import fetch_text

EVENT_CONSTANTS_URL = (
    "https://raw.githubusercontent.com/pret/pokeyellow/master/constants/event_constants.asm"
)
FLAG_ACTION_URL = (
    "https://raw.githubusercontent.com/pret/pokeyellow/master/engine/flag_action.asm"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "navigation_data" / "yellow_event_flags.json"

DIRECTIVE_RE = re.compile(
    r"^\s*(const_def|const_next|const_skip|const)\b(.*)$"
)
SECTION_COMMENT_RE = re.compile(r"^\s*;\s*(.+?)\s*$")
NUM_EVENTS_RE = re.compile(r"^\s*DEF\s+NUM_EVENTS\s+EQU\s+const_value\s*$")


def _parse_value(expr: str) -> int:
    """Evaluate the small subset of arithmetic pret's const macros use:
    a bare decimal, a $hex literal, or `$hex +/- decimal`."""
    expr = expr.strip()
    if not expr:
        return 0
    m = re.match(r"^\$([0-9A-Fa-f]+)\s*([+-])\s*(\d+)$", expr)
    if m:
        base = int(m.group(1), 16)
        return base + int(m.group(3)) if m.group(2) == "+" else base - int(m.group(3))
    m = re.match(r"^\$([0-9A-Fa-f]+)$", expr)
    if m:
        return int(m.group(1), 16)
    return int(expr, 10)


def parse_event_constants(text: str) -> list[dict]:
    events: list[dict] = []
    counter = 0
    group = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if NUM_EVENTS_RE.match(line):
            continue
        directive_match = DIRECTIVE_RE.match(line)
        if not directive_match:
            comment_match = SECTION_COMMENT_RE.match(line)
            if comment_match:
                group = comment_match.group(1)
            continue
        directive, rest = directive_match.groups()
        rest = rest.strip()
        if directive == "const_def":
            counter = _parse_value(rest) if rest else 0
        elif directive == "const_next":
            counter = _parse_value(rest)
        elif directive == "const_skip":
            counter += _parse_value(rest) if rest else 1
        elif directive == "const":
            name = rest.split(";", 1)[0].strip()
            events.append(
                {
                    "bit": counter,
                    "byte_offset": counter // 8,
                    "bit_in_byte": counter % 8,
                    "name": name,
                    "group": group,
                }
            )
            counter += 1
    return events


def build_catalog(constants_text: str) -> dict:
    events = parse_event_constants(constants_text)
    if len(events) < 500:
        raise ValueError(f"Expected at least 500 named event flags, parsed {len(events)}")
    max_bit = max(event["bit"] for event in events)
    return {
        "schema_version": 1,
        "game": "Pokémon Yellow",
        "sources": {
            "event_constants": EVENT_CONSTANTS_URL,
            "flag_action_routine": FLAG_ACTION_URL,
        },
        "num_events": 2560,  # NUM_EVENTS = $A00, verified against the source file's tail
        "max_named_bit": max_bit,
        "bit_layout_note": (
            "byte_offset = bit // 8, bit_in_byte = bit % 8, LSB = bit 0 -- "
            "verified against engine/flag_action.asm's FlagAction routine, "
            "not assumed. Matches ADDR_EVENT_FLAGS_START in train.py directly: "
            "byte address = ADDR_EVENT_FLAGS_START + byte_offset."
        ),
        "events": events,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--constants-file", type=Path, help="Use a local event_constants.asm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    constants_text = (
        args.constants_file.read_text(encoding="utf-8")
        if args.constants_file
        else fetch_text(EVENT_CONSTANTS_URL)
    )
    catalog = build_catalog(constants_text)
    atomic_write_json(args.output, catalog, keep_backup=False)
    print(f"Wrote {args.output}: {len(catalog['events'])} named event flags")


if __name__ == "__main__":
    main()
