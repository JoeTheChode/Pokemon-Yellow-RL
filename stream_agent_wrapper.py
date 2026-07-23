"""Broadcasts live agent position to PWhiddy's shared community map viewer
(https://pwhiddy.github.io/pokerl-map-viz/), adapted from
PWhiddy/PokemonRedExperiments' baselines/stream_agent_wrapper.py.

Sends only (x, y, map_id) coordinates plus a user-chosen display
name/color/env_id every `upload_interval` steps to a third-party WebSocket
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


class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata=None):
        super().__init__(env)
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = dict(stream_metadata or {})
        self.local_stream_path = self._resolve_local_stream_path()
        self.env_label = f"env_id={self.stream_metadata.get('env_id')}"
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.broadcast_attempts = 0
        self.broadcast_successes = 0
        self.loop.run_until_complete(self.establish_wc_connection())
        print(
            f"  [Stream {self.env_label}] initial connection "
            f"{'OK' if self.websocket is not None else 'FAILED (will retry on first send)'}",
            flush=True,
        )
        self.upload_interval = 300
        self.stream_step_counter = 0
        self.coord_list = []
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

        entry = {
            "env_id": env_id,
            "user": self.stream_metadata.get("user") or "agent",
            "color": self.stream_metadata.get("color") or "#4cc9f0",
            "extra": self.stream_metadata.get("extra", ""),
            "last_position": [int(x_pos), int(y_pos), int(map_n)],
            "map_id": int(map_n),
            "last_seen": time.time(),
        }
        temp_path = payload_path.with_name(f"{payload_path.stem}.tmp.pid{os.getpid()}{payload_path.suffix}")
        try:
            temp_path.write_text(json.dumps(entry), encoding="utf-8")
            os.replace(temp_path, payload_path)
        except OSError:
            pass

    def step(self, action):
        y_pos = self.emulator.memory[POS_A_ADDRESS]
        x_pos = self.emulator.memory[POS_B_ADDRESS]
        map_n = self.emulator.memory[MAP_ID_ADDRESS]
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.stream_step_counter >= self.upload_interval:
            seen = getattr(self.env, "episode_coord_visits", None)
            if seen is None:
                seen = getattr(self.env, "visited_positions", None)
            self.stream_metadata["extra"] = f"coords: {len(seen)}" if seen is not None else ""
            self._write_local_stream_state(x_pos, y_pos, map_n)
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
            self.stream_step_counter = 0
            self.coord_list = []

        self.stream_step_counter += 1

        return self.env.step(action)

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
