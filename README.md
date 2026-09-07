# Pokémon Yellow PPO Trainer

> **Status: archived / sunset.** This repository is a final snapshot of an
> experimental reinforcement-learning project that trained an agent to play
> Pokémon Yellow end to end. It is no longer developed or run. The code is
> published for reference; expect rough edges and hard-coded paths.

An experimental RL trainer for Pokémon Yellow built with
[PyBoy](https://github.com/Baekalfen/PyBoy), Gymnasium,
Stable-Baselines3 PPO, and PyTorch. Training runs a large vectorised swarm of
emulator workers through a save-state milestone curriculum and then a single
continuous full-game "quest phase" curriculum that carries the agent from
Pallet Town to the Elite Four.

## What is here

| Path | Purpose |
| --- | --- |
| `train.py` | The whole trainer: env, reward shaping, quest-phase curriculum, forced-route guidance tables, swarm frontier logic, callbacks, PPO loop. Large single module. |
| `project_paths.py` | Shared paths and durable JSON persistence helpers. |
| `yellow_navigation.py` | Offline map-ID resolver and macro route graph over the `pret/pokeyellow` disassembly. |
| `collision.py` | Gen-1 tile / tile-pair collision model used for offline route derivation. |
| `play.py` | Manual play station — load a state, drive it by keyboard, log coordinates and demonstration paths. |
| `watch.py`, `savestate.py`, `debug_pos.py` | Small play/record and ROM-identity helpers. |
| `map_viewer_server.py` + `pokemon_yellow_map_viewer.html` | Live training dashboard: per-worker positions, quest chain, badges, splits, heatmap. |
| `export_quest_chain.py` | Renders the quest chain the viewer's objective card reads. |
| `sync_navigation_data.py`, `sync_event_flags.py` | Regenerate the catalogs under `navigation_data/`. |
| `stream_agent_wrapper.py` | Optional broadcast of each worker's `(x, y, map_id)` to PWhiddy's shared community map (adapted from PokemonRedExperiments). |
| `run_test_suite.py` | Test runner (no pytest dependency). |
| `test_*.py`, `viewer_tests/` | Regression tests for route guidance, safety gates, and the viewer. |
| `navigation_data/` | Generated catalogs: map/location catalog, event-flag names, quest waypoints, leveling plan, quest chain. |
| `pokerl_map_assets/` | Kanto overview image, tile map data, agent sprite sheet for the viewer. |
| `deploy/systemd/` | The systemd units the production trainer, viewer, and liveness watchdog ran under. Reference only. |
| `AGENTS.md` | The live-trainer restart protocol that governed deploys. |

## You must supply the ROM

The ROM and save states are intentionally **not** in this repo. Place a legally
obtained `yellow.gb` (and a `start.state`) in the repo root before running
anything. ROM, save-state, checkpoint, model, log, and screenshot artifacts are
all ignored by `.gitignore`.

## Runtime

The final training runs used Ubuntu 24.04, Python 3.12, and a ROCm PyTorch
build, as a 96-worker `SubprocVecEnv` swarm under systemd. Core dependencies:

- `gymnasium`, `numpy`, `Pillow`, `pyboy`, `stable-baselines3`
- a PyTorch build appropriate for the machine
- `websockets` (only if `POKEMON_STREAM_ENABLED=1`)

`requirements.txt` pins the lighter set used by `play.py` / the navigation
tooling on a workstation.

### Train

```bash
# direct
python train.py
# or the swarm launcher used in production (sets POKEMON_N_ENVS / vec env)
./run_train_ovh.sh
```

Production never restarted the service by hand — see `AGENTS.md` and
`safe_restart_train.sh` for the PID-audited, frontier-preserving protocol, and
`train_watchdog.sh` for the log-staleness liveness check.

### Viewer

```bash
./start_viewer.sh      # exports the quest chain, then serves map_viewer_server.py
```

### Tests

```bash
python run_test_suite.py
python -m unittest -v test_project_reliability.py test_yellow_navigation.py
```

## Navigation data

`yellow_navigation.py` resolves all internal map IDs from
`navigation_data/yellow_catalog.json`, which combines map IDs and dimensions
from the `pret/pokeyellow` disassembly with named locations and display
coordinates. Refreshing the catalogs is an explicit dev operation:

```bash
python -X utf8 sync_navigation_data.py
python -X utf8 sync_event_flags.py
python -X utf8 yellow_navigation.py validate
python -X utf8 yellow_navigation.py identify 0x36
python -X utf8 yellow_navigation.py route "Pallet Town" "Cerulean City"
```

Map-level routes describe topology only. Action-level pathfinding still needs
collision blocks and live object positions from the emulator; webpage pixels
are never treated as walkable tiles.

## Live map streaming (optional)

`stream_agent_wrapper.py` can broadcast each worker's live `(x, y, map_id)` to
PWhiddy's shared community map viewer
(<https://pwhiddy.github.io/pokerl-map-viz/>) over a third-party WebSocket
(`wss://transdimensional.xyz/broadcast`). It is outbound traffic to an
external server and is **off by default**:

```bash
export POKEMON_STREAM_ENABLED=1
export POKEMON_STREAM_USER="your-display-name"   # optional
python train.py
```

## Credits

- ROM disassembly data: [`pret/pokeyellow`](https://github.com/pret/pokeyellow)
- Streaming wrapper and map assets adapted from
  [`PWhiddy/PokemonRedExperiments`](https://github.com/PWhiddy/PokemonRedExperiments)
- Emulation: [PyBoy](https://github.com/Baekalfen/PyBoy)

## License

No license is granted for this archived snapshot. Third-party material
(`pret/pokeyellow` data, PokemonRedExperiments-derived code and assets) remains
under its respective upstream terms. Pokémon and Pokémon Yellow are trademarks
of Nintendo / Creatures Inc. / GAME FREAK inc.; this project is an unaffiliated
research experiment and ships no copyrighted ROM data.
