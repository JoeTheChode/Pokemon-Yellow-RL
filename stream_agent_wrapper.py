"""Broadcasts live agent position to PWhiddy's shared community map viewer
(https://pwhiddy.github.io/pokerl-map-viz/), adapted from
PWhiddy/PokemonRedExperiments' baselines/stream_agent_wrapper.py.

Sends (x, y, map_id) coordinates plus live Pikachu level and a user-chosen
display name/color/env_id every `upload_interval` steps to a third-party WebSocket
server (wss://transdimensional.xyz/broadcast) that PWhiddy's project runs for
this purpose -- this is outbound network traffic to infrastructure outside
this project, by design (that's what makes the shared map viewer work).

Adapted from the original for two things specific to this project:
- Yellow's map/position addresses are shifted -1 from Red/Blue's (see
  ADDR_MAP_ID/ADDR_POS_A/ADDR_POS_B in train.py) -- the original wrapper
  hardcodes Red's addresses, which would read the wrong bytes for Yellow.
- The original references `env.seen_coords`, an attribute PWhiddy's own env
  has that ours doesn't; adapted to use this project's own
  `episode_coord_visits` (falls back gracefully if absent).
"""
import asyncio
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import websockets

# Yellow's addresses (verified in train.py) -- NOT the same as Red/Blue's.
MAP_ID_ADDRESS = 0xD35D
POS_A_ADDRESS = 0xD360  # y
POS_B_ADDRESS = 0xD361  # x
LEVEL_ADDRESS = 0xD18B  # first party Pokemon's actual level (Pikachu in this run)
# These three were originally copied at Red/Blue offsets, which are shifted
# relative to Yellow's, so all three read the wrong byte. Confirmed against
# pret/pokeyellow's own symbol map:
#   $d056 = wIsInBattle        <- correct; 0xD057 is wPartyGainExpFlags, a
#                                 party-EXP bitfield that sits at 1 more or
#                                 less permanently once Pikachu has battled.
#                                 That is why every agent's dot was pinned to
#                                 orange ("wild") regardless of what it was
#                                 actually doing.
#   $cfe5 = wEnemyMonHP        <- correct; big-endian u16. 0xCFE6 is only its
#                                 low byte, so any HP that is an exact
#                                 multiple of 256 read as 0.
#   $d124 = wTextBoxID         <- correct; 0xCD6B is wJoyIgnore, unrelated.
# train.py already uses 0xD056 (ADDR_BATTLE_FLAG), so training was never
# affected -- this was a viewer-only misread.
BATTLE_FLAG_ADDRESS = 0xD056  # wIsInBattle: 0=overworld, 1=wild, 2=trainer
ENEMY_HP_ADDRESS = 0xCFE5  # wEnemyMonHP, big-endian u16
TEXT_BOX_ADDRESS = 0xD124  # wTextBoxID

# Trainer-card fields, resolved from this ROM's own pokeyellow.sym -- NOT from
# Red/Blue offsets, two of which are wrong for Yellow: wPlayerName is D158 and
# wPlayerMoney is D347 in Red/Blue, one byte later than the values below.
PLAYER_NAME_ADDRESS = 0xD157  # wPlayerName, 0x50-terminated Gen-1 text
PLAYER_MONEY_ADDRESS = 0xD346  # wPlayerMoney, 3 bytes, BCD
BADGES_ADDRESS = 0xD355  # wObtainedBadges, one bit per badge
PLAY_TIME_HOURS_ADDRESS = 0xDA40  # then Maxed, Minutes, Seconds
PLAYER_NAME_MAX_LENGTH = 11


def _decode_gen1_text(memory, address, max_length):
    """Decode a 0x50-terminated Gen-1 string (letters/digits/space only)."""
    out = []
    for offset in range(max_length):
        code = int(memory[address + offset])
        if code == 0x50:  # terminator
            break
        if code == 0x7F:
            out.append(" ")
        elif 0x80 <= code <= 0x99:
            out.append(chr(ord("A") + code - 0x80))
        elif 0xA0 <= code <= 0xB9:
            out.append(chr(ord("a") + code - 0xA0))
        elif 0xF6 <= code <= 0xFF:
            out.append(chr(ord("0") + code - 0xF6))
    return "".join(out).strip()


def _read_bcd(memory, address, length):
    """Gen-1 money is packed BCD: each byte is two decimal digits."""
    total = 0
    for offset in range(length):
        byte = int(memory[address + offset])
        total = total * 100 + (byte >> 4) * 10 + (byte & 0x0F)
    return total


def _read_play_time(memory):
    """In-game clock as (seconds, maxed). Yellow caps at 255:59:59."""
    hours = int(memory[PLAY_TIME_HOURS_ADDRESS])
    maxed = int(memory[PLAY_TIME_HOURS_ADDRESS + 1]) != 0
    minutes = int(memory[PLAY_TIME_HOURS_ADDRESS + 2])
    seconds = int(memory[PLAY_TIME_HOURS_ADDRESS + 3])
    # A save state caught mid-tick can hold out-of-range values; clamp rather
    # than publish a nonsense clock to the dashboard.
    minutes = minutes if 0 <= minutes < 60 else 0
    seconds = seconds if 0 <= seconds < 60 else 0
    return hours * 3600 + minutes * 60 + seconds, maxed
BATTLE_STATUS_BY_FLAG = {0: "overworld", 1: "wild", 2: "trainer"}
# Bumped whenever the meaning of the battle_* fields changes. The viewer only
# trusts battle_status when this is present and current, so a dashboard loaded
# against a trainer still running the pre-fix wrapper shows "unavailable"
# rather than confidently wrong colours.
BATTLE_SOURCE_REV = "wIsInBattle@D056"


class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata=None):
        super().__init__(env)
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = dict(stream_metadata or {})
        self.local_stream_path = self._resolve_local_stream_path()
        self.env_label = f"env_id={self.stream_metadata.get('env_id')}"
        self.remote_stream_enabled = os.environ.get(
            "POKEMON_STREAM_REMOTE_ENABLED", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        try:
            sprite_id = int(os.environ.get("POKEMON_STREAM_SPRITE_ID", "0"))
        except ValueError:
            sprite_id = 0
        self.stream_metadata.setdefault("sprite_id", max(0, min(49, sprite_id)))
        self.local_upload_interval = max(
            1, int(os.environ.get("POKEMON_STREAM_UPLOAD_INTERVAL", "10"))
        )
        self.remote_upload_interval = max(
            1, int(os.environ.get("POKEMON_REMOTE_STREAM_UPLOAD_INTERVAL", "300"))
        )
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.broadcast_attempts = 0
        self.broadcast_successes = 0
        if self.remote_stream_enabled:
            self.loop.run_until_complete(self.establish_wc_connection())
            print(
                f"  [Stream {self.env_label}] initial connection "
                f"{'OK' if self.websocket is not None else 'FAILED (will retry on first send)'}",
                flush=True,
            )
        else:
            print(f"  [Stream {self.env_label}] remote stream disabled", flush=True)
        self.local_stream_step_counter = 0
        self.remote_stream_step_counter = 0
        self.coord_list = []
        self.whiteout_count = 0
        self.completed_episodes = 0
        self.whiteout_episodes = 0
        self._episode_wipes_seen = 0
        self._episode_had_whiteout = False
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

    def _resolve_local_stream_path(self):
        configured = self.stream_metadata.get("local_stream_path") or os.environ.get("POKEMON_LOCAL_STREAM_PATH")
        if configured:
            path = Path(configured)
        else:
            path = Path("live_agent_positions.json")
        return path if path.is_absolute() else (Path.cwd() / path)

    def _plausible_lead_level(self):
        """Lead level, holding the last good value through a torn read.

        The old `min(read, 100)` clamp published a garbage read as a clean
        "level 100": on 2026-08-23 every one of the 96 workers hit that at
        least once while the real lead was level 54, and the dashboard's
        historical high (which backfills every level between two readings)
        recorded all of 55..100 as reached. Clamping hid the corruption
        instead of dropping it -- see `party_read_is_plausible` in train.py
        for the same read on the trainer side.
        """
        level = int(self.emulator.memory[LEVEL_ADDRESS])
        if 1 <= level <= 100:
            self._last_plausible_level = level
            return level
        return getattr(self, "_last_plausible_level", 0)

    def _write_local_stream_state(self, x_pos, y_pos, map_n):
        if self.local_stream_path is None:
            return
        # Every worker writes ONLY its own per-env_id file -- no shared
        # read-modify-write. The earlier "one combined file" design had every
        # worker read the current list, merge in its own entry, and write the
        # whole list back; with N concurrent workers doing that with no
        # locking, worker B's write routinely clobbered worker A's just-added
        # entry with the stale snapshot B read before A's write landed
        # (classic lost-update race). That surfaced as entries missing
        # entirely and old ghost entries reappearing even after being
        # "replaced". Per-env files make the race structurally impossible --
        # each worker only ever touches its own file. The viewer merges
        # across files at read time instead.
        env_id = self.stream_metadata.get("env_id", self.env_label)
        base = self.local_stream_path
        payload_path = base.with_name(f"{base.stem}.env{env_id}{base.suffix}")
        payload_path.parent.mkdir(parents=True, exist_ok=True)

        effective_battle_flag = int(self.emulator.memory[BATTLE_FLAG_ADDRESS])
        battle_flag_reader = getattr(self.env, "_effective_battle_flag", None)
        if callable(battle_flag_reader):
            effective_battle_flag = int(battle_flag_reader(effective_battle_flag))
        quest_phase = max(0, int(getattr(self.env, "quest_phase", 0)))
        quest_waypoints = getattr(self.env, "quest_waypoints", ())
        quest_total = len(quest_waypoints)
        quest_name_reader = getattr(self.env, "_quest_phase_name", None)
        quest_next = quest_name_reader() if callable(quest_name_reader) else None
        play_time_seconds, play_time_maxed = _read_play_time(self.emulator.memory)
        entry = {
            "env_id": env_id,
            "user": self.stream_metadata.get("user") or "agent",
            "color": self.stream_metadata.get("color") or "#4cc9f0",
            "extra": self.stream_metadata.get("extra", ""),
            "last_position": [int(x_pos), int(y_pos), int(map_n)],
            "map_id": int(map_n),
            "pikachu_level": self._plausible_lead_level(),
            "battle_status": BATTLE_STATUS_BY_FLAG.get(
                effective_battle_flag, "unknown"
            ),
            "battle_source": BATTLE_SOURCE_REV,
            "debug_raw_battle_flag": int(self.emulator.memory[BATTLE_FLAG_ADDRESS]),
            "debug_enemy_hp": (
                int(self.emulator.memory[ENEMY_HP_ADDRESS]) * 256
                + int(self.emulator.memory[ENEMY_HP_ADDRESS + 1])
            ),
            "debug_text_box": int(self.emulator.memory[TEXT_BOX_ADDRESS]),
            "quest_phase": quest_phase,
            "quest_total": quest_total,
            "quest_next": quest_next,
            # Trainer-card fields for the viewer's Splits tab.
            "player_name": _decode_gen1_text(
                self.emulator.memory, PLAYER_NAME_ADDRESS, PLAYER_NAME_MAX_LENGTH
            ),
            "money": _read_bcd(self.emulator.memory, PLAYER_MONEY_ADDRESS, 3),
            "badge_flags": int(self.emulator.memory[BADGES_ADDRESS]),
            "badge_count": int(self.emulator.memory[BADGES_ADDRESS]).bit_count(),
            "play_time_seconds": play_time_seconds,
            "play_time_maxed": play_time_maxed,
            "whiteout_count": self.whiteout_count,
            "completed_episodes": self.completed_episodes,
            "whiteout_episodes": self.whiteout_episodes,
            "episode_whiteouts": max(
                0, int(getattr(self.env, "party_wipes", 0))
            ),
            "last_seen": time.time(),
        }
        temp_path = payload_path.with_name(f"{payload_path.stem}.tmp.pid{os.getpid()}{payload_path.suffix}")
        try:
            temp_path.write_text(json.dumps(entry), encoding="utf-8")
            os.replace(temp_path, payload_path)
        except OSError:
            pass

    def _update_whiteout_stats(self, episode_finished=False):
        """Roll the env's exact positive-HP -> zero-HP edges into run totals."""
        episode_wipes = max(0, int(getattr(self.env, "party_wipes", 0)))
        if episode_wipes < self._episode_wipes_seen:
            # A reset happened outside our reset() path; start a new baseline
            # instead of turning the counter decrease into a negative wipe.
            self._episode_wipes_seen = episode_wipes
            self._episode_had_whiteout = episode_wipes > 0
        elif episode_wipes > self._episode_wipes_seen:
            self.whiteout_count += episode_wipes - self._episode_wipes_seen
            self._episode_wipes_seen = episode_wipes
            self._episode_had_whiteout = True
        if episode_finished:
            self.completed_episodes += 1
            if self._episode_had_whiteout:
                self.whiteout_episodes += 1

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        self._episode_wipes_seen = max(
            0, int(getattr(self.env, "party_wipes", 0))
        )
        self._episode_had_whiteout = self._episode_wipes_seen > 0
        return result

    def step(self, action):
        y_pos = self.emulator.memory[POS_A_ADDRESS]
        x_pos = self.emulator.memory[POS_B_ADDRESS]
        map_n = self.emulator.memory[MAP_ID_ADDRESS]
        self.coord_list.append([x_pos, y_pos, map_n])
        self.local_stream_step_counter += 1
        self.remote_stream_step_counter += 1

        seen = getattr(self.env, "episode_coord_visits", None)
        if seen is None:
            seen = getattr(self.env, "visited_positions", None)

        if self.local_stream_step_counter >= self.local_upload_interval:
            self.stream_metadata["extra"] = f"coords: {len(seen)}" if seen is not None else ""
            self._write_local_stream_state(x_pos, y_pos, map_n)
            self.local_stream_step_counter = 0

        if self.remote_stream_step_counter >= self.remote_upload_interval:
            self.stream_metadata["extra"] = f"coords: {len(seen)}" if seen is not None else ""
            if self.remote_stream_enabled:
                self.broadcast_attempts += 1
                self.loop.run_until_complete(
                    self.broadcast_ws_message(
                        json.dumps(
                            {
                                "metadata": self.stream_metadata,
                                "coords": self.coord_list,
                            }
                        )
                    )
                )
                if self.broadcast_attempts <= 3 or self.broadcast_attempts % 20 == 0:
                    print(
                        f"  [Stream {self.env_label}] broadcast attempt "
                        f"{self.broadcast_attempts}: "
                        f"{self.broadcast_successes} succeeded so far, "
                        f"websocket={'connected' if self.websocket is not None else 'None'}, "
                        f"last_batch_size={len(self.coord_list)}",
                        flush=True,
                    )
            self.remote_stream_step_counter = 0
            self.coord_list = []

        result = self.env.step(action)
        episode_finished = bool(result[2] or result[3])
        self._update_whiteout_stats(episode_finished=episode_finished)
        return result

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
                self.broadcast_successes += 1
            except Exception as e:
                print(f"  [Stream {self.env_label}] send failed: {e!r}", flush=True)
                self.websocket = None
        else:
            print(f"  [Stream {self.env_label}] no websocket, skipped send", flush=True)

    async def establish_wc_connection(self):
        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.ws_address), timeout=10
            )
        except Exception as e:
            print(f"  [Stream {self.env_label}] connection failed: {e!r}", flush=True)
            self.websocket = None
