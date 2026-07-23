"""Manual play station for inspecting and mapping milestone states."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import sdl2
import sdl2.ext
from pyboy import PyBoy

from project_paths import (
    MILESTONES_DIR,
    PROJECT_ROOT,
    PROGRESS_FILE,
    RECORDINGS_DIR,
    ROM_PATH,
    SCREENSHOTS_DIR,
    ensure_runtime_dirs,
    latest_hit_glob,
    milestone_state_path,
)
from yellow_navigation import map_labels


ADDR_MAP_ID = 0xD35D
# Pokemon Yellow shifts the Red/Blue coordinate bytes down by one:
# D360 = y, D361 = x.
ADDR_POS_A = 0xD360
ADDR_POS_B = 0xD361
ADDR_PARTY_SIZE = 0xD162
ADDR_BADGES = 0xD356
ADDR_LEVEL = 0xD18B
ADDR_BATTLE_FLAG = 0xD057
ADDR_ENEMY_HP = 0xCFE6
ADDR_TEXT_BOX = 0xCF13

MAP_LABELS = map_labels()

MILESTONE_NAMES = [
    "00_start",
    "01_got_pikachu",
    "02_viridian_city",
    "03_got_parcel",
    "04_return_pallet",
    "05_deliver_parcel",
    "05a_exit_lab",
    "05b_pokecenter",
    "05c_viridian_north",
    "05d_forest_south",
    "05e_forest_north",
    "05f_gate_transition",
    "05g_to_pewter",
    "06_pewter_city",
    "07_beat_brock",
    "08_cerulean_city",
    "09_beat_misty",
    "10_vermilion_city",
    "11_beat_surge",
    "12_lavender_town",
    "13_celadon_city",
    "14_beat_erika",
    "15_beat_sabrina",
    "16_beat_koga",
    "17_beat_blaine",
    "18_beat_giovanni",
]

KEY_MAP = {
    sdl2.SDL_SCANCODE_UP: "up",
    sdl2.SDL_SCANCODE_DOWN: "down",
    sdl2.SDL_SCANCODE_LEFT: "left",
    sdl2.SDL_SCANCODE_RIGHT: "right",
    sdl2.SDL_SCANCODE_Z: "a",
    sdl2.SDL_SCANCODE_X: "b",
    sdl2.SDL_SCANCODE_RETURN: "start",
    sdl2.SDL_SCANCODE_BACKSPACE: "select",
}

STATE_CHANGE_FIELDS = (
    "map_id",
    "y",
    "x",
    "battle",
    "enemy_hp",
    "party",
    "badges",
    "level",
    "text_box",
)


def map_label(map_id: int) -> str:
    return MAP_LABELS.get(map_id, f"map{map_id}")


def battle_label(battle: int) -> str:
    return {
        0: "overworld",
        1: "wild",
        2: "trainer",
    }.get(battle, f"battle{battle}")


def read_progress_index() -> int:
    if not PROGRESS_FILE.exists():
        return 0
    with open(PROGRESS_FILE, "r", encoding="utf-8") as handle:
        return json.load(handle).get("milestone_index", 0)


def current_milestone_name() -> str:
    idx = max(0, min(read_progress_index(), len(MILESTONE_NAMES) - 1))
    return MILESTONE_NAMES[idx]


def latest_hit_state(previous_name: str) -> Path | None:
    matches = sorted(
        PROJECT_ROOT.glob(Path(latest_hit_glob(previous_name)).name),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def resolve_state_file(args: argparse.Namespace) -> tuple[Path, str]:
    if args.state:
        return Path(args.state), "explicit path"

    if args.current_stage:
        stage_name = current_milestone_name()
        state_path = milestone_state_path(stage_name)
        if state_path.exists():
            return state_path, f"current milestone {stage_name}"
        stage_index = MILESTONE_NAMES.index(stage_name)
        if stage_index > 0:
            previous = MILESTONE_NAMES[stage_index - 1]
            fallback = latest_hit_state(previous)
            if fallback is not None:
                return fallback, f"latest hit from {previous}"
        raise FileNotFoundError(f"No state found for current milestone {stage_name}")

    default_stage = milestone_state_path("05f_gate_transition")
    if default_stage.exists():
        return default_stage, "default 05f_gate_transition"
    return milestone_state_path("02_viridian_city"), "fallback 02_viridian_city"


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def snapshot_info(pyboy: PyBoy) -> dict[str, object]:
    mem = pyboy.memory
    map_id = int(mem[ADDR_MAP_ID])
    battle = int(mem[ADDR_BATTLE_FLAG])
    text_box = int(mem[ADDR_TEXT_BOX])
    return {
        "map_id": map_id,
        "map_name": map_label(map_id),
        "y": int(mem[ADDR_POS_A]),
        "x": int(mem[ADDR_POS_B]),
        "party": int(mem[ADDR_PARTY_SIZE]),
        "badges": int(mem[ADDR_BADGES]),
        "level": int(mem[ADDR_LEVEL]),
        "battle": battle,
        "battle_name": battle_label(battle),
        "enemy_hp": int(mem[ADDR_ENEMY_HP]),
        "text_box": text_box,
        "text_box_active": bool(text_box),
    }


def format_info(info: dict[str, object]) -> str:
    return (
        f"map={info['map_id']} ({info['map_name']}), "
        f"y={info['y']}, x={info['x']}, party={info['party']}, badges={info['badges']}, "
        f"level={info['level']}, battle={info['battle']} ({info['battle_name']}), "
        f"enemy_hp={info['enemy_hp']}, text_box={info['text_box']}"
    )


def print_info(pyboy: PyBoy, prefix: str = "  ") -> dict[str, object]:
    info = snapshot_info(pyboy)
    print(f"{prefix}{format_info(info)}")
    return info


def write_mapping_line(log_path: Path, tag: str, info: dict[str, object], note: str = "") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = f" | {note}" if note else ""
    line = (
        f"[{timestamp}] {tag} map={info['map_id']}({info['map_name']}) "
        f"y={info['y']} x={info['x']} battle={info['battle']}({info['battle_name']}) "
        f"party={info['party']} level={info['level']} enemy_hp={info['enemy_hp']} "
        f"text_box={info['text_box']}{suffix}"
    )
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(f"  {line}")


def current_buttons(held_keys: set[int]) -> list[str]:
    return [KEY_MAP[scan] for scan in sorted(held_keys)]


def state_changes(previous: dict[str, object], current: dict[str, object]) -> dict[str, dict[str, object]]:
    changes: dict[str, dict[str, object]] = {}
    for field in STATE_CHANGE_FIELDS:
        if previous[field] == current[field]:
            continue
        changes[field] = {
            "from": previous[field],
            "to": current[field],
        }
    return changes


class DemoRecorder:
    def __init__(self, record_path: Path, state_file: Path, source_label: str):
        self.record_path = record_path
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state_file = state_file
        self.source_label = source_label
        self.event_index = 0

    def write_event(
        self,
        event_type: str,
        pyboy: PyBoy,
        tick_count: int,
        held_keys: set[int],
        *,
        state: dict[str, object] | None = None,
        **extra: object,
    ) -> None:
        event = {
            "session_id": self.session_id,
            "event_index": self.event_index,
            "timestamp": iso_timestamp(),
            "event_type": event_type,
            "tick": tick_count,
            "frame": int(pyboy.frame_count),
            "held_buttons": current_buttons(held_keys),
            "state": state if state is not None else snapshot_info(pyboy),
        }
        event.update(extra)
        with open(self.record_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        self.event_index += 1


def save_state(pyboy: PyBoy, save_name: str | None = None) -> None:
    raw_name = save_name or input("  Save name (e.g. 05f_gate_transition_manual): ").strip()
    if not raw_name:
        print("  Cancelled.")
        return

    candidate = Path(raw_name)
    path = candidate.with_suffix(".state") if candidate.is_absolute() else milestone_state_path(candidate.stem)
    with open(path, "wb") as handle:
        pyboy.save_state(handle)
    print_info(pyboy)
    print(f"  State saved: {path}")


def save_checkpoint_state(pyboy: PyBoy, base_name: str | None = None) -> Path:
    info = snapshot_info(pyboy)
    raw_base = base_name or "manual_checkpoint"
    stem = Path(raw_base).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_name = str(info["map_name"]).lower()
    safe_map_name = "".join(ch if ch.isalnum() else "_" for ch in map_name).strip("_")
    path = milestone_state_path(f"{stem}_{safe_map_name}_{timestamp}")
    with open(path, "wb") as handle:
        pyboy.save_state(handle)
    print_info(pyboy)
    print(f"  Checkpoint saved: {path}")
    return path


def save_screenshot(pyboy: PyBoy, stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCREENSHOTS_DIR / f"{stem}_{timestamp}.png"
    pyboy.screen.image.save(path)
    print(f"  Screenshot saved: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual play station for milestone mapping.")
    parser.add_argument("state", nargs="?", help="Optional state file path to load.")
    parser.add_argument("save_name", nargs="?", help="Optional default save-state name for F5.")
    parser.add_argument("--current-stage", action="store_true", help="Load the current curriculum state's save file.")
    parser.add_argument("--log-file", help="Optional mapping log file path.")
    parser.add_argument("--record-file", help="Optional JSONL demonstration record path.")
    return parser


def main() -> None:
    ensure_runtime_dirs()
    parser = build_parser()
    args = parser.parse_args()

    state_file, source_label = resolve_state_file(args)
    if not state_file.exists():
        raise FileNotFoundError(f"State file does not exist: {state_file}")

    default_log = MILESTONES_DIR / f"manual_map_{state_file.stem}.log"
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_record = RECORDINGS_DIR / f"manual_demo_{state_file.stem}_{session_stamp}.jsonl"
    log_path = Path(args.log_file) if args.log_file else default_log
    record_path = Path(args.record_file) if args.record_file else default_record

    print(f"Loading state: {state_file}")
    print(f"Source: {source_label}")
    print(f"Mapping log: {log_path}")
    print(f"Demo record: {record_path}")
    print()
    print("Controls:")
    print("  Arrow keys = D-pad")
    print("  Z = A button, X = B button")
    print("  Enter = Start, Backspace = Select")
    print("  F1 or Space = print current position info")
    print("  F2 = toggle auto-log on coordinate changes")
    print("  F3 = write a quick manual waypoint marker to the mapping log")
    print("  F4 = save a screenshot to screenshots/")
    print("  F5 = save/overwrite named state")
    print("  F6 = save timestamped checkpoint state")
    print("  Tab = dump battle RAM")
    print("  Esc = quit")
    print()

    pyboy = PyBoy(str(ROM_PATH), window="SDL2")
    with open(state_file, "rb") as handle:
        pyboy.load_state(handle)

    tick_count = 0
    current_info = print_info(pyboy)
    recorder = DemoRecorder(record_path, state_file, source_label)
    write_mapping_line(log_path, "START", current_info, note=state_file.name)
    recorder.write_event(
        "session_start",
        pyboy,
        tick_count,
        set(),
        state=current_info,
        state_file=str(state_file),
        state_source=source_label,
        mapping_log=str(log_path),
        demo_record=str(record_path),
        controls={
            "up": "ArrowUp",
            "down": "ArrowDown",
            "left": "ArrowLeft",
            "right": "ArrowRight",
            "a": "Z",
            "b": "X",
            "start": "Enter",
            "select": "Backspace",
        },
    )
    print()

    held_keys: set[int] = set()
    running = True
    auto_log = True
    last_info = current_info
    last_position = (
        current_info["map_id"],
        current_info["y"],
        current_info["x"],
        current_info["battle"],
    )

    while running and pyboy.tick():
        tick_count += 1
        info = snapshot_info(pyboy)
        changes = state_changes(last_info, info)
        if changes:
            recorder.write_event(
                "state_change",
                pyboy,
                tick_count,
                held_keys,
                state=info,
                previous_state=last_info,
                changes=changes,
            )
            last_info = info
        position = (info["map_id"], info["y"], info["x"], info["battle"])
        if auto_log and position != last_position:
            write_mapping_line(log_path, "MOVE", info)
            last_position = position

        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break

            if event.type == sdl2.SDL_KEYDOWN:
                scan = event.key.keysym.scancode

                if scan == sdl2.SDL_SCANCODE_ESCAPE:
                    recorder.write_event("session_quit_requested", pyboy, tick_count, held_keys, state=info)
                    running = False
                    break
                if scan in (sdl2.SDL_SCANCODE_F1, sdl2.SDL_SCANCODE_SPACE):
                    info = print_info(pyboy)
                    write_mapping_line(log_path, "INFO", info)
                    recorder.write_event("info", pyboy, tick_count, held_keys, state=info)
                elif scan == sdl2.SDL_SCANCODE_F2:
                    auto_log = not auto_log
                    print(f"  Auto-log {'enabled' if auto_log else 'disabled'}")
                    recorder.write_event(
                        "auto_log_toggled",
                        pyboy,
                        tick_count,
                        held_keys,
                        state=info,
                        auto_log=auto_log,
                    )
                elif scan == sdl2.SDL_SCANCODE_F3:
                    info = snapshot_info(pyboy)
                    write_mapping_line(log_path, "MARK", info, note="quick_marker")
                    recorder.write_event(
                        "marker",
                        pyboy,
                        tick_count,
                        held_keys,
                        state=info,
                        note="quick_marker",
                    )
                elif scan == sdl2.SDL_SCANCODE_F4:
                    screenshot_path = save_screenshot(pyboy, f"manual_{state_file.stem}")
                    recorder.write_event(
                        "screenshot_saved",
                        pyboy,
                        tick_count,
                        held_keys,
                        state=info,
                        screenshot_path=str(screenshot_path),
                    )
                elif scan == sdl2.SDL_SCANCODE_TAB:
                    mem = pyboy.memory
                    print("  === Battle RAM dump (0xCFD0-0xD000) ===")
                    dump_entries: list[dict[str, int]] = []
                    for addr in range(0xCFD0, 0xD000):
                        value = mem[addr]
                        if value != 0:
                            print(f"    0x{addr:04X} = {value} (0x{value:02X})")
                            dump_entries.append({"address": addr, "value": int(value)})
                    print("  === End dump ===")
                    recorder.write_event(
                        "battle_ram_dump",
                        pyboy,
                        tick_count,
                        held_keys,
                        state=info,
                        entries=dump_entries,
                    )
                elif scan == sdl2.SDL_SCANCODE_F5:
                    save_state(pyboy, args.save_name)
                    recorder.write_event(
                        "state_saved",
                        pyboy,
                        tick_count,
                        held_keys,
                        state=info,
                        save_name=args.save_name,
                    )
                elif scan == sdl2.SDL_SCANCODE_F6:
                    checkpoint_path = save_checkpoint_state(pyboy, args.save_name)
                    recorder.write_event(
                        "checkpoint_state_saved",
                        pyboy,
                        tick_count,
                        held_keys,
                        state=info,
                        checkpoint_path=str(checkpoint_path),
                        save_name=args.save_name,
                    )
                elif scan in KEY_MAP and scan not in held_keys:
                    button_name = KEY_MAP[scan]
                    pyboy.button_press(button_name)
                    held_keys.add(scan)
                    recorder.write_event(
                        "button_press",
                        pyboy,
                        tick_count,
                        held_keys,
                        state=info,
                        button=button_name,
                        scancode=int(scan),
                    )

            elif event.type == sdl2.SDL_KEYUP:
                scan = event.key.keysym.scancode
                if scan in KEY_MAP and scan in held_keys:
                    button_name = KEY_MAP[scan]
                    pyboy.button_release(button_name)
                    held_keys.discard(scan)
                    recorder.write_event(
                        "button_release",
                        pyboy,
                        tick_count,
                        held_keys,
                        state=info,
                        button=button_name,
                        scancode=int(scan),
                    )

    final_info = print_info(pyboy)
    write_mapping_line(log_path, "END", final_info)
    recorder.write_event(
        "session_end",
        pyboy,
        tick_count,
        held_keys,
        state=final_info,
        mapping_log=str(log_path),
        demo_record=str(record_path),
        total_events=recorder.event_index,
    )
    print(f"  Demonstration saved: {record_path}")
    pyboy.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
