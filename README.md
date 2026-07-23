# Pokémon Yellow PPO Trainer

An experimental reinforcement-learning trainer for Pokémon Yellow, built with
PyBoy, Gymnasium, Stable-Baselines3 PPO, and PyTorch. Training progresses through
save-state milestones before entering a full-game curriculum.

## Runtime

The actively used setup is Ubuntu 24.04 under WSL with Python 3.12 and a ROCm
PyTorch build. The ROM and save states are intentionally local artifacts and are
not source dependencies.

Required Python packages:

- `gymnasium`
- `numpy`
- `Pillow`
- `pyboy`
- `stable-baselines3`
- a PyTorch build appropriate for the machine (ROCm in the current WSL setup)
- `websockets` (optional — only needed if `POKEMON_STREAM_ENABLED=1`, see below)

Place a legally obtained `yellow.gb` and `start.state` in this directory. Then
launch GPU training from PowerShell with:

```powershell
.\start_train_wsl.ps1
```

For manual state inspection and mapping:

```powershell
python play.py --current-stage
```

## Checks

The pure reliability tests run with the standard library. Checkpoint tests run
when the training dependencies are installed:

```powershell
python -m unittest -v test_project_reliability.py
python -m compileall -q train.py play.py project_paths.py test_project_reliability.py
```

## Data layout

- `train.py` — environment, curriculum, callbacks, and training loop
- `project_paths.py` — shared paths and durable JSON persistence
- `yellow_navigation.py` — offline map-ID resolver and macro route graph
- `navigation_data/yellow_catalog.json` — generated Yellow map/location catalog
- `sync_navigation_data.py` — explicit catalog refresh tool
- `milestones/` — curriculum states and progress
- `checkpoints/` — PPO and VecNormalize checkpoints
- `screenshots/`, `recordings/` — runtime diagnostics

Checkpoint, ROM, state, model, log, and screenshot artifacts are ignored by the
included `.gitignore`; they can be very large and should be backed up separately.

## Navigation data

The trainer resolves all 249 internal map IDs using the generated catalog rather
than maintaining conflicting handwritten dictionaries. The catalog combines the
map IDs and dimensions from the `pret/pokeyellow` disassembly with the 52 named
locations and display coordinates found in the saved Pokémon Completion page.
It is an offline runtime dependency; refreshing it is an explicit development
operation:

```powershell
python -X utf8 sync_navigation_data.py
python -X utf8 yellow_navigation.py validate
python -X utf8 yellow_navigation.py identify 0x36
python -X utf8 yellow_navigation.py route "Pallet Town" "Cerulean City"
```

The catalog also includes outdoor connections and exact warp source coordinates
from the disassembly. Its map-level route command describes topology, but may
cross disconnected portions of one outdoor map (for example, Route 2 around
Viridian Forest). The curated macro graph preserves required major areas such as
Mt. Moon. Action-level pathfinding still requires collision blocks and live
object positions from the ROM/emulator; webpage pixels are never treated as
walkable tiles.

## Live map streaming (optional)

`stream_agent_wrapper.py` (adapted from PWhiddy/PokemonRedExperiments) can
broadcast each env's live (x, y, map_id) to PWhiddy's shared community map
viewer at https://pwhiddy.github.io/pokerl-map-viz/. This is genuinely
outbound network traffic to a third-party server
(`wss://transdimensional.xyz/broadcast`), not something local — enable it
deliberately:

```powershell
$env:POKEMON_STREAM_ENABLED = "1"
$env:POKEMON_STREAM_USER = "your-display-name"   # optional, defaults to SVER-PokemonYellow
$env:POKEMON_STREAM_COLOR = "#FFD700"            # optional hex color
.\start_train_wsl.ps1
```

Off by default. Requires `websockets` (`pip install --break-system-packages websockets`
in the training WSL distro).
