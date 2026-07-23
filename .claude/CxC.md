# CxC - Claude x Codex Changelog

Shared log for coordinating changes between Claude Code (interactive) and Codex (async).
**Both agents: read this file before making changes. Log all changes here after.**

---

## Current State (2026-04-04)

- **Project**: Pokemon Yellow RL agent using PPO (stable-baselines3) + PyBoy emulator
- **Main file**: `train.py` - contains env (`PokemonEnv`), milestones curriculum (00-09+), reward shaping, training loop
- **Supporting files**: `play.py`, `watch.py`, `check_state.py`, `debug_pos.py`, `map_forest.py`, `savestate.py`
- **Milestone progress**: Curriculum spans `00_start` through `09_beat_misty` and beyond
- **Key addresses**: All memory addresses are defined at the top of `train.py`
- **Shaping system**: Per-milestone Y-coordinate shaping with waypoint-list support on complex maps like Viridian Forest
- **No git repo**: changes are tracked here manually

---

## Change Log

### 2026-04-04 - Init
- Created this file for Claude/Codex coordination.

### 2026-04-04 - Codex reliability pass
- Added `project_paths.py` so ROMs, states, milestones, checkpoints, screenshots, and progress files resolve relative to this repo instead of `C:\Users\Admin\pokemon`.
- Updated `train.py` to use shared repo-relative paths throughout the training flow.
- Fixed milestone promotion so `ProgressionCallback` now copies the exact hit-state file saved by the environment that reached the goal, instead of guessing from the vector-env slot index.
- Scoped temporary hit-state files by milestone name so stale `_latest_hit_*.state` files from unrelated milestones are not reused.
- Tightened milestone fallback loading so missing milestone states prefer recent hit states from the immediately previous milestone.
- Reset curriculum `step_penalty` to `0.0` for staged milestone training before the full-game phase.
- Changed training device selection to configuration:
  - `POKEMON_TRAIN_DEVICE` now defaults to `cpu`
  - `POKEMON_HIP_VISIBLE_DEVICES` is only applied when explicitly set
- Reworked helper scripts to use shared paths:
  - `play.py`
  - `watch.py`
  - `check_state.py`
  - `debug_pos.py`
  - `map_forest.py`
  - `savestate.py`
- Added retry handling in `watch.py` so partially written screenshots do not crash the viewer.
- Verified syntax with `python -m compileall train.py play.py watch.py check_state.py debug_pos.py map_forest.py savestate.py project_paths.py`.
- Verified the saved PPO model loads on CPU after the HIP crash path was avoided.

### 2026-04-04 - Codex rollout visibility follow-up
- Added `RolloutProgressCallback` in `train.py` so long PPO rollout collection prints progress every 512 vectorized steps instead of appearing frozen.
- Added a startup note in `train.py` showing when the first PPO log should appear for the current rollout size.
- Recompiled `train.py` successfully after the logging-only change.

### 2026-04-04 - Codex GPU support follow-up
- Verified locally that native Windows ROCm sees the GPUs but fails for compute:
  - `AMD Radeon RX 9060 XT` can be enumerated, but the first real CUDA/HIP tensor allocation hangs.
  - `Radeon 7` crashes on the first tensor allocation with `amdhip64_7.dll`.
- Confirmed against AMD's current ROCm Radeon docs that native Windows currently lists `No ML training support`, while WSL2 is the supported production path for PyTorch training on Radeon 9000 series GPUs.
- Added a native-Windows GPU guard in `train.py`:
  - Native Windows + `POKEMON_TRAIN_DEVICE!=cpu` now fails fast with a clear message unless `POKEMON_ALLOW_UNSUPPORTED_WINDOWS_GPU=1` is set.
- Replaced `checkgpu.py` with a real probe utility that can test actual tensor allocation and forward/backward passes via `python checkgpu.py --probe`.
- Added `run_train_wsl.sh` for launching GPU training inside WSL with `POKEMON_TRAIN_DEVICE=cuda`.
- Added `run_train_wsl.ps1` as a Windows-side wrapper to invoke the WSL training script from this repo.

### 2026-04-04 - Codex WSL GPU activation
- Imported a clean scriptable WSL distro from the official Ubuntu 24.04 rootfs as `Ubuntu2404GPU` after the Store-installed `Ubuntu-24.04` distro got stuck in first-launch setup.
- Installed AMD's supported WSL ROCm stack in `Ubuntu2404GPU` using:
  - `amdgpu-install -y --usecase=wsl,rocm --no-dkms`
- Verified GPU visibility inside WSL with `rocminfo`:
  - `gfx1200`
  - `AMD Radeon RX 9060 XT`
- Created a dedicated WSL venv at `/root/.venvs/pokemon-gpu`.
- Installed the ROCm PyTorch stack in WSL:
  - `torch 2.9.1+rocm7.2.0`
  - `torchvision 0.24.0+rocm7.2.0`
  - `torchaudio 2.9.0+rocm7.2.0`
  - `triton 3.5.1+rocm7.2.0`
- Removed bundled `libhsa-runtime64.so*` from the WSL Torch package so it uses the system ROCm runtime from the AMD WSL install path.
- Installed repo runtime dependencies in WSL:
  - `stable-baselines3`
  - `gymnasium`
  - `pyboy`
  - `pysdl2`
  - `pillow`
- Upgraded WSL NumPy to `2.4.3` to match the Windows training env and allow the existing unified PPO model zip to deserialize correctly.
- Updated `run_train_wsl.sh` to:
  - activate the dedicated WSL venv automatically
  - `cd` to the repo directory before running
  - run with `PYTHONUNBUFFERED=1` for live logs
- Updated `run_train_wsl.ps1` to default to the imported `Ubuntu2404GPU` distro.
- Launched detached GPU training successfully in WSL.
  - Repo log: `wsl_gpu_train.log`
  - Repo PID file: `wsl_gpu_train.pid`
  - Confirmed live startup log reaches:
    - `device[0]=AMD Radeon RX 9060 XT`
    - `Pokemon Yellow RL - Auto Progression`
    - `Resuming unified model from /mnt/c/Users/Admin/Pokemon/milestones/pokemon_unified_model.zip`
- Current active WSL trainer PID at handoff: `481`

### 2026-04-05 - Codex forest-stage tuning + checkpoint-aware GPU restart
- Tuned the north-forest milestones at runtime in `train.py` without rewriting the milestone table directly:
  - `05e_forest_north`
    - `ep_length` reduced to `16384`
    - `step_penalty=0.01`
    - `map_discovery_bonus=25`
    - added per-step detour penalties for maps `50`, `13`, `1`, `41`, and `33`
    - added `hit_bonus=1000`
    - PPO overrides now use `n_steps=1024`, `n_epochs=2`, `ent_coef=0.05`, `target_kl=0.03`
  - `05f_to_pewter`
    - `ep_length` reduced to `32768`
    - matching step/map penalties and `hit_bonus=1000`
    - PPO overrides now use `n_steps=2048`, `n_epochs=2`, `ent_coef=0.05`, `target_kl=0.03`
- Added env-level knobs in `train.py` for:
  - configurable `map_discovery_bonus`
  - configurable per-map dwell penalties via `map_step_penalties`
  - configurable one-time `milestone_hit_bonus`
- Milestone hits now award the configured hit bonus on first clean success, so the agent gets a direct positive signal for actually reaching the stage goal.
- Tightened the value-head auto-reset guard so it now requires both a tiny `value_loss` and a bad/negative `explained_variance` before resetting.
- Added checkpoint-aware resume in `train.py`:
  - stage startup now prefers the latest matching checkpoint from `checkpoints/` before falling back to `pokemon_unified_model.zip`
  - checkpoint frequency for curriculum milestones was increased from `100000` to `50000` callback steps to reduce restart loss
- Added startup logging in `train.py` for the active per-stage PPO and reward tuning settings.
- Verified syntax with `python -m compileall train.py`.

### 2026-04-07 - Codex Stage 11 rollout-buffer handoff fix
- Fixed the crash that appeared immediately after advancing from `05e_forest_north` to `05f_gate_transition`.
- Root cause in `train.py`:
  - SB3 wraps the image vec env with `VecTransposeImage` when it is attached to the model.
  - Our `apply_ppo_overrides()` helper rebuilt the rollout buffer against the pre-wrap env object.
  - That left the buffer expecting stacked channel-last observations while the live env was returning channel-first observations.
  - Result: `ValueError: could not broadcast input array from shape (4,4,84,84) into shape (4,84,84,4)`.
- Updated `apply_ppo_overrides()` to rebuild buffers against the model's live attached env / observation space instead of the stale local env reference.
- Verified syntax with `python -m compileall train.py`.
- Let the prior WSL GPU trainer reach checkpoint `poke_05e_forest_north_43435712_steps.zip`, then restarted training on GPU from that checkpoint with the new settings.
- Confirmed post-restart log shows:
  - `device[0]=AMD Radeon RX 9060 XT`
  - `Stage 10: 05e_forest_north`
  - `Episode length: 16384 steps`
  - `PPO: device=cuda n_steps=1024 n_epochs=2 ent_coef=0.05 target_kl=0.03`
  - `Reward tuning: step_penalty=0.01 map_discovery_bonus=25 hit_bonus=1000`
  - `Resuming milestone checkpoint from /mnt/c/Users/Admin/Pokemon/checkpoints/poke_05e_forest_north_43435712_steps.zip`
- Current active WSL trainer PID at handoff: `1875`

### 2026-04-05 - Codex background trainer monitor
- Added `monitor_train.ps1` to watch the detached WSL GPU trainer and summarize its health from Windows.
- The monitor checks:
  - whether `python train.py` is still running inside `Ubuntu2404GPU`
  - how stale `wsl_gpu_train.log` is
  - recent rollout metrics from the tail of the training log
- The monitor writes:
  - rolling log: `wsl_gpu_monitor.log`
  - latest status snapshot: `wsl_gpu_monitor.status.txt`
  - monitor PID: `wsl_gpu_monitor.pid`
- Launched the monitor in the background and verified its first status entry recorded:
  - trainer PID `1875`
  - fresh log activity
  - recent `fps`, `iterations`, and `total_timesteps`
- Current active monitor PID at handoff: `54532`

### 2026-04-05 - Codex monitor parser fix + stalled-run recovery
- Tightened `monitor_train.ps1` log parsing so it no longer treats `[DEBUG] milestone_check(...)` lines as milestone events:
  - metric matching now looks specifically for the boxed `fps`, `iterations`, and `total_timesteps` lines
  - milestone matching now uses a word-boundary pattern instead of the broad `MILESTONE` substring
- Tightened `monitor_train.ps1` process detection so it matches only an exact `python train.py` process from WSL instead of occasionally reporting wrapper-shell PIDs.
- The monitor correctly detected a stale trainer log while the WSL process was still alive:
  - stale snapshot at `2026-04-05 06:18:55 -07:00`
  - last good checkpoint already existed at `checkpoints/poke_05e_forest_north_43635712_steps.zip`
- Recycled the stalled WSL trainer and relaunched GPU training from that fresh checkpoint.
- Confirmed the restarted run is again on the RX 9060 XT and advancing:
  - `Resuming milestone checkpoint from /mnt/c/Users/Admin/Pokemon/checkpoints/poke_05e_forest_north_43635712_steps.zip`
  - live trainer PID now `7599`
  - live monitor PID now `93504`
- Verified current monitor status shows a healthy running trainer with fresh log age and live metrics.

### 2026-04-04 - Claude Code: GPU fix & WSL2 PyTorch correction
- **Root cause of GPU failure**: Codex installed PyTorch `rocm6.3` wheels into a WSL2 distro running system ROCm 7.2. The `rocm6.3` build lacks gfx1200 (RDNA 4) kernels, causing `hipErrorNotFound` on any compute op. Native Windows ROCm segfaults in `amdhip64_7.dll` on both GPUs (Radeon 7 and 9060 XT) — confirmed independently.
- **Fix**: Replaced the WSL2 PyTorch with AMD's official ROCm 7.2.1 wheels from `repo.radeon.com`:
  - `torch-2.9.1+rocm7.2.1` (system-wide, not in venv)
  - `triton-3.5.1+rocm7.2.1`
  - Also installed `numpy`, `stable-baselines3`, `gymnasium`, `pyboy`, `Pillow` system-wide
- Changed `TRAIN_DEVICE` default in `train.py` from hardcoded `"cpu"` to auto-detect: uses `cuda` when `torch.cuda.is_available()`, falls back to `cpu`
- Simplified `run_train_wsl.sh` to use system-wide packages (removed venv activation, added GPU sanity check)
- **Verified**: 9060 XT runs full CNN forward+backward in WSL2. PPO training smoke test hit **167 FPS on GPU vs 7-12 FPS on CPU** (~20x speedup).
- **NOTE**: Running `python train.py` on native Windows will still use CPU (GPUs segfault). For GPU training, launch via WSL2:
  ```
  wsl -d Ubuntu2404GPU -- bash -c "cd /mnt/c/Users/Admin/Pokemon && python3 -u train.py"
  ```

### 2026-04-05 - Codex directed frontier reward + VecNormalize persistence
- Added directed forest progress shaping in `train.py` so `05e_forest_north` and `05f_to_pewter` reward first-time route progress instead of generic tile novelty:
  - new per-milestone `tile_exploration_bonus`
  - new per-milestone `frontier_path`
  - new per-milestone `frontier_reward_scale`
- Set `tile_exploration_bonus=0.0` for both forest milestones so random wandering no longer pays by itself.
- Implemented a monotonic `best_frontier_score` in `PokemonYellowEnv`:
  - progress is computed from explicit ordered segments and waypoint chains
  - reward only increases when the agent beats its prior best route progress within the episode
  - end-of-episode summaries now include `frontier=...`
- Kept the existing local distance shaping and waypoint bonuses, but added the new frontier reward on top so the agent is nudged toward irreversible northbound progress instead of loopable novelty.
- Added VecNormalize persistence in `train.py`:
  - startup now tries to load stats from the matching checkpoint vecnormalize file first
  - then the latest stage vecnormalize file
  - then `milestones/pokemon_unified_vecnormalize.pkl`
  - checkpoint callbacks now save vecnormalize stats with `save_vecnormalize=True`
  - unified vecnormalize stats are also saved at stage end and after each full-game round
- Updated startup logging to print `tile_bonus` and `frontier_scale`.
- Verified syntax with `python -m compileall train.py`.
- Restarted the active WSL GPU trainer from `checkpoints/poke_05e_forest_north_47235712_steps.zip` so the new reward logic takes effect immediately.
- Confirmed the restarted log shows:
  - `Reward tuning: step_penalty=0.01 tile_bonus=0.0 map_discovery_bonus=25 frontier_scale=1.5 hit_bonus=1000`
  - `Resuming milestone checkpoint from /mnt/c/Users/Admin/Pokemon/checkpoints/poke_05e_forest_north_47235712_steps.zip`
- Updated `monitor_train.ps1` again so it recognizes both `python train.py` and `python3 train.py`.
- Current active WSL trainer PID at handoff: `15073`
- Current active monitor PID at handoff: `17084`

### 2026-04-05 - Codex WSL cleanup
- Removed the old unused WSL venv at `/root/.venvs/pokemon-gpu` to avoid confusion between the previous venv-based ROCm install and the current system Python setup.
- Re-verified after deletion that the live trainer is still the system interpreter process:
  - `15073 python3 train.py`
- Re-verified that `/root/.venvs/pokemon-gpu` is gone after cleanup.
- Confirmed the background monitor still reports the live GPU trainer healthy after the cleanup.

### 2026-04-05 - Codex forest coordinate memory sync
- Loaded the external project memory file at:
  - `C:\Users\Admin\.claude\projects\c--Users-Admin-Pokemon\memory\project_map_coordinates.md`
- Used its verified Viridian Forest notes to correct the forest milestone route definition in `train.py`.
- Important adjustment from the memory notes:
  - the full forest path begins at south-gate entry `y=16`, but milestone `05e_forest_north` actually starts around `y=23`, so the milestone-specific shaping should begin at the southward push toward `y=26`, not at `y=16`
- Updated both forest milestones so `shaping` and `frontier_path` now reflect the milestone-relevant route:
  - `26 -> 25 -> 17 -> 12 -> 11 -> 7 -> 1`
- Tightened gate coordinates using the memory file:
  - `05e_forest_north` now uses north-gate target `(4, 1)` on map `47`
  - `05f_to_pewter` now uses north-gate progression `[(4, 1), (5, 1)]` on map `47`
- Restarted the live WSL GPU trainer from the latest checkpoint/vecnormalize pair:
  - `checkpoints/poke_05e_forest_north_51835712_steps.zip`
  - `checkpoints/poke_05e_forest_north_vecnormalize_51835712_steps.pkl`
- Confirmed the restarted log reaches stage startup and resumes cleanly on GPU.
- Current active WSL trainer PID at handoff: `28121`
- Current active monitor PID at handoff: `52460`

### 2026-04-06 - Codex forest north-gate shaping fix
- Tightened the final Viridian Forest route for both `05e_forest_north` and `05f_to_pewter` in `train.py`:
  - changed the last forest waypoints from `(1, 0)` to an explicit gate approach through `(2, 1)` then `(1, 1)`
  - kept the existing forest route prefix `26 -> 25 -> 17 -> 12 -> 11 -> 7`
- Added reusable per-milestone `zone_bonuses` support to `PokemonYellowEnv`:
  - one-time bonuses now fire when the agent steps onto configured gate-approach tiles
  - this is generic and can be reused for later doors, transitions, and puzzle chokepoints
- Configured forest gate bonuses:
  - `05e_forest_north`: forest top corridor bonuses on map `51` at `(2,1)`, `(1,0)`, and `(1,1)`, plus a small confirmation bonus on gate map `47` at `(4,1)`
  - `05f_to_pewter`: same forest-top bonuses plus north-gate walk-through bonuses on map `47` at `(4,1)` and `(5,1)`
- Added `terminate_on_hit` support and enabled it for both forest stages:
  - once the milestone is actually hit, the episode ends immediately instead of burning the rest of the step budget and dragging `ep_rew_mean` back down
- Expanded startup logging so the trainer prints:
  - `zone_bonus_total=...`
  - `terminate_on_hit=True/False`
- Verified syntax with `python -m compileall train.py`.
- Restarted the live WSL GPU trainer from the newest checkpoint/VecNormalize pair:
  - `checkpoints/poke_05e_forest_north_57635712_steps.zip`
  - `checkpoints/poke_05e_forest_north_vecnormalize_57635712_steps.pkl`
- Confirmed the patched startup log now shows:
  - `zone_bonus_total=1050`
  - `terminate_on_hit=True`
- Current active WSL trainer PID after restart: `436`
- Current hidden Windows WSL host PID after restart: `102260`

### 2026-04-06 - Codex plain-English training logs
- Added a `SimpleStatusCallback` to `train.py` so each rollout now prints a short human-readable status line instead of forcing the user to interpret SB3's metric table.
- The new status line summarizes:
  - current target milestone
  - total steps
  - average reward with plain-language interpretation
  - milestone hits in the current rolling window
  - average episode length
  - critic health from explained variance
  - approximate speed in FPS
- Added `configure_model_logger(model)` and made simple logs the default:
  - raw SB3 stdout tables are now suppressed by default
  - CSV logging is still preserved under `checkpoints/sb3_logs`
  - raw tables can be restored by setting `POKEMON_SHOW_RAW_SB3_TABLES=1`
- Added a startup line showing whether the run is using `plain-English status lines` or `raw SB3 tables`.
- Verified syntax with `python -m compileall train.py`.
- Fixed the plain-English `speed=` field to be session-local after resume:
  - it now measures steps since the current run started, instead of dividing the lifetime total timesteps by a few seconds and printing nonsense.

### 2026-04-06 - Codex guided metrics log format
- Reworked the rollout summary in `train.py` again so it keeps the familiar metric names but adds short explanations beside each one instead of only printing a single ELI5 sentence.
- The guided block now includes:
  - `ep_rew_mean`
  - `ep_len_mean`
  - `goal_hits`
  - `explained_variance`
  - `value_loss`
  - `approx_kl`
  - `clip_fraction`
  - `entropy_loss`
  - `fps`
  - `total_timesteps`
- Each metric now carries a short interpretation, for example:
  - reward: higher is better
  - explained variance: near `1` good, negative bad
  - goal hits: real progression signal
  - FPS: training speed
- Updated the startup banner text from `plain-English status lines` to `guided metrics with explanations`.
- Verified syntax with `python -m compileall train.py`.

### 2026-04-06 - Codex compact metric layout
- Tightened the guided metric output in `train.py` so it is easier to scan in a terminal window:
  - shortened explanations to avoid hard wrapping
  - switched to an aligned two-column metric layout
  - moved `steps` and `fps` into the `[METRICS]` header line
  - removed the redundant `total_timesteps` row since the header already carries it
  - added a blank line after each metric block so consecutive updates are visually separated
- Reduced rollout spam in `train.py`:
  - the half-rollout progress line still prints
  - the full-rollout `1024/1024` line is now suppressed because the metric block follows immediately after it
- Updated `monitor_train.ps1` so the background monitor recognizes the new compact `[METRICS]` format and the key metric rows (`ep_rew_mean`, `goal_hits`, `explained_variance`).
- Verified syntax with `python -m compileall train.py`.

### 2026-04-06 - Codex forest diagnosis telemetry
- Added episode-level diagnosis telemetry to `train.py` so we can answer whether the agent is getting lost, blacking out, or burning time in battles:
  - counts wild battle entries and trainer battle entries separately
  - tracks battle step counts for wild and trainer fights
  - tracks enemy faint counts separately for wild and trainer battles
  - tracks party wipe events (`party_wipes`) when the whole party HP hits zero
  - tracks minimum total party HP reached during the episode
  - tracks end-of-episode party HP for each party slot
- Added verified party HP address handling for this repo's Yellow memory layout:
  - active battle mon current/max HP
  - party slot current/max HP for up to 6 party members
  - verified locally against the project's milestone state before wiring into the trainer
- Added frontier HP snapshots:
  - when the agent reaches a new best route frontier, the log now records map/position, total party HP, active HP, and whether it was in overworld, wild battle, or trainer battle
  - this is printed as `[HP PATH] ...` at episode end
- Expanded episode-end logging:
  - `[EP]` now prints map labels instead of raw map IDs when known
  - `[BATTLE]` prints wild/trainer counts, enemy faint counts, wipes, min total HP, and ending party HP
  - `[HP PATH]` prints the HP snapshots at each frontier improvement
- Verified syntax with `python -m compileall train.py`.

### 2026-04-07 - Codex forest curriculum split
- Split the old forest bottleneck into two curriculum steps in `train.py` so the agent no longer has to solve “reach the top of the forest” and “thread the north-gate transition under encounter pressure” in one stage:
  - `05e_forest_north` now means: reach the forest pre-gate overworld tile with usable HP
  - new `05f_gate_transition`: step from that pre-gate forest tile into `ForestGateN` (map `47`)
  - old `05f_to_pewter` was renamed to `05g_to_pewter` and now assumes the agent already starts from the north gate
- Tightened the `05e_forest_north` success condition:
  - must be on map `51`
  - must be in overworld (`battle_flag == 0`)
  - must be on the top exit approach tiles `[(1,0), (1,1), (2,1)]`
  - must have at least `8` HP on the active mon so the saved next-stage state is not a nearly-dead dud
- Retuned `05e_forest_north` to focus only on forest approach progress:
  - removed map `47` from its shaping/frontier path
  - kept the explicit route up to `(1,1)`
  - kept terminate-on-hit so it cashes out as soon as it reaches a good pre-gate state
- Added a dedicated `05f_gate_transition` micro-stage:
  - short episodes (`4096`)
  - stronger time pressure
  - shaping from forest pre-gate tiles to map `47`
  - stronger gate-entry bonus and `terminate_on_hit=True`
  - PPO tuned tighter for a short, local transition (`n_steps=512`, lower entropy, lower KL target)
- Simplified the renamed `05g_to_pewter` stage:
  - it now starts from the north gate state and focuses on map `47 -> 54 -> 2`
  - removed the old forest-waypoint path because that responsibility now belongs to `05e` and `05f`
- Verified syntax with `python -m compileall train.py`.

### 2026-04-07 - Codex Stage 11 handoff stabilization
- Fixed the Stage 11 crash in `train.py` after the curriculum advanced from `05e_forest_north` to `05f_gate_transition`.
- Root cause:
  - SB3 attaches an image vec env through `VecTransposeImage`, which changes the effective observation layout.
  - `apply_ppo_overrides()` was rebuilding the rollout buffer from the stale pre-wrap env object instead of the model's live attached env.
  - That produced the Stage 11 mismatch: channel-first observations being written into a channel-last rollout buffer (`(4,4,84,84)` vs `(4,84,84,4)`).
- Updated `apply_ppo_overrides()` to rebuild against `model.get_env()` / `model.observation_space`, so stage-to-stage PPO override changes keep the buffer aligned with the real env.
- Recompiled with `python -m compileall train.py`.
- Relaunched the WSL GPU trainer and verified it now survives multiple Stage 11 rollout updates on the `AMD Radeon RX 9060 XT` without reproducing the broadcast error.

### 2026-04-07 - Codex metrics header labeling cleanup
- Fixed the guided metrics header in `train.py` so it reports the current stage name instead of the next stage name.
- This was purely a logging clarity fix on disk; I left the active trainer running, so it will take effect on the next restart.
- Recompiled with `python -m compileall train.py`.

### 2026-04-07 - Codex episode log cleanup
- Cleaned up the per-episode diagnostics in `train.py` so they are easier to scan during live runs.
- Replaced the old three-line block:
  - `[EP] maps=... y_range=...`
  - `[BATTLE] ...`
  - `[HP PATH] ...`
  with a compact summary line plus one short best-frontier line.
- New episode format keeps the same signal but is denser:
  - route through maps
  - Y range
  - tiles explored
  - frontier score
  - wild/trainer encounter counts
  - faint counts
  - party wipes
  - min HP -> end HP
  - best frontier snapshot
- Disabled the noisy per-episode `milestone_check(...)` debug prints by default and gated them behind `POKEMON_SHOW_MILESTONE_DEBUG=1` for when low-level milestone debugging is actually needed.
- Also cleaned up the startup metrics header logic on disk so the first rollout after a restart shows `fps=warmup` instead of a bogus giant number caused by near-zero elapsed time.
- Recompiled with `python -m compileall train.py`.

### 2026-04-07 - Codex Stage 11 survivability and auto-continue pass
- Investigated why Stage 11 kept ending without progress and confirmed the stop was a deliberate 10M-step chunk boundary, not a crash.
- Updated `train.py` so unfinished milestones no longer require a manual restart after each 10M-step chunk:
  - after saving the unified model and VecNormalize stats, the trainer now loops back into the same milestone automatically instead of breaking out
- Added explicit survival / battle-pressure controls to `PokemonYellowEnv`:
  - `force_repel`
  - `wild_battle_step_penalty`
  - `wild_battle_entry_penalty`
  - `hp_loss_penalty_scale`
  - `wipe_penalty`
  - `terminate_on_party_wipe`
- Wired those controls into step/reset logic so stage-specific curriculum slices can cleanly separate navigation from survival pressure when needed.
- Retuned `05f_gate_transition` based on emulator probing of the actual saved state:
  - the saved stage starts at map `51`, coord `(2,1)`, not directly inside the gate
  - local movement from that state showed the useful progress direction is along the top corridor to the right, not the old fake short hop target
  - updated shaping/frontier/zone bonuses to reward that observed corridor progression
  - shortened the stage to `2048` steps and increased time pressure
  - enabled `force_repel=True` for this micro-stage so it can focus on learning the gate approach instead of drowning in wild-battle RNG
  - added HP-loss and wipe penalties plus terminate-on-wipe for faster resets when the run goes bad
- Added `ADDR_REPEL` to `train.py` for the new repel-based micro-stage control.
- Verified syntax with `python -m compileall train.py`.
- Verified runtime with a WSL GPU smoke test:
  - milestone `11` loaded cleanly from `poke_05f_gate_transition_84435904_steps.zip`
  - new survival tuning banner printed as expected
  - first sampled episode showed much lower battle pressure than the pre-patch Stage 11 run

### 2026-04-07 - Codex next-restart log readability pass
- Cleaned up the on-disk log formatting in `train.py` for the next trainer restart without interrupting the current run.
- Episode summaries:
  - repeated identical episode summaries are now collapsed into a counted form like `[EP x3] ...`
  - zero-value clutter is dropped from quiet episodes, so `wild=0/0`, `trainer=0/0`, `faints=0/0`, and `wipes=0` no longer spam every line
  - the best-frontier follow-up line is shortened from `best_frontier=...` to `best=...`
  - frontier snapshots were compacted to remove redundant active-HP text when it was not adding signal
- Metrics block:
  - shortened row labels from long internal names (`ep_rew_mean`, `goal_hits`, `explained_variance`, etc.) to compact names (`reward`, `hits`, `critic`, `value`, `kl`, `clip`, `entropy`, `ep_len`)
  - shortened the interpretation text so the block stays readable in a narrow terminal
  - queued episode summaries are flushed right before the metrics block so the output groups cleanly
- Added safety flushes so queued episode summaries are not stranded if a milestone advances or a training chunk ends mid-rollout.
- Verified syntax with `python -m compileall train.py`.

### 2026-04-08 - Codex Stage 11 training resume
- Resumed Stage 11 (`05f_gate_transition`) from the latest checkpoint on the `AMD Radeon RX 9060 XT`.
- Rotated the stale `wsl_gpu_train.log` from the previous run before restarting.
- Detected and cleaned up an accidental duplicate-launch situation where two `python3 train.py` processes were writing to the same log.
- Relaunched a single clean WSL trainer process and synchronized:
  - `wsl_gpu_train.pid`
  - `wsl_gpu_host.pid`
  - `wsl_gpu_monitor.pid`
- Restarted `monitor_train.ps1` so `wsl_gpu_monitor.status.txt` reflects the live trainer again.
- Confirmed the resumed run is active on milestone `11` with the cleaner post-restart metric format in `wsl_gpu_train.log`.

### 2026-04-08 - Codex screenshot viewer launcher
- Reused the existing Tk/Pillow live screenshot viewer in `watch.py`.
- Added a Windows launcher script `watch_screenshots.ps1` that opens the viewer in its own `cmd` window via `ml-env`:
  - activates the Conda env
  - changes into the repo
  - runs `python watch.py`
- Verified the viewer script still compiles with `python -m compileall watch.py`.

### 2026-04-08 - Codex forest milestone sanity check
- Verified the broken-forest-state diagnosis locally against the actual milestone saves:
  - `05d_forest_south.state` starts on Route 2 (`map=13`) and only reaches a tiny local graph that leads back into Viridian, not toward the forest gate
  - `05e_forest_north.state` starts at `map=51 a=24 b=0` in a trapped left-edge forest branch
  - `05f_gate_transition.state` starts at `map=51 a=2 b=1` in another trapped top-left branch
- Added `validate_milestone_states.py`, a lightweight coordinate-state reachability probe for sanity-checking curriculum save states before training.
- Ran `python3 validate_milestone_states.py --max-depth 12` in WSL and confirmed:
  - `05d_forest_south`: only maps `[1, 13]`, never hits target maps `[50, 51]`
  - `05e_forest_north`: only map `[51]`, reachable coord range `a=[14..31] b=[0..1]`, never hits target map `[47]`
  - `05f_gate_transition`: only map `[51]`, reachable coord range `a=[1..14] b=[0..1]`, never hits target map `[47]`
- Stopped continuing the current forest run once the milestone-state issue was confirmed, since it was burning GPU on impossible handoff states.

### 2026-04-09 - Codex forest repair and restart prep
- Added a stage-loader guard in `train.py` that ignores stale per-stage checkpoints whenever the current milestone state file is newer than that checkpoint.
- Added `repair_forest_milestones.py` to back up the broken forest saves, install the recovered replacements, remove the stale fixed `05f_gate_transition.state`, and reset progress to Stage 10.
- Applied the repair:
  - `05d_forest_south.state` now comes from `forest_entrance_from_gate_fixed.state` (`map=51 y=16 x=1`)
  - `05e_forest_north.state` now comes from `teleport_main_body.state` (`map=51 y=20 x=24`)
  - `05f_gate_transition.state` was removed so Stage 11 must start from a fresh 05e hit state instead of the trapped old save
  - `progress.json` was reset to `{"milestone_index": 10}`
- Backed up the replaced files under `milestones/_repair_backups/20260409_014037/`.
- Restarted GPU training from the repaired Stage 10 forest setup:
  - host PID `28724`
  - trainer PID `318`
  - monitor PID `41180`
- Confirmed in `wsl_gpu_train.log` that the run starts at `05e_forest_north`, loads the repaired `05e_forest_north.state`, ignores the stale per-stage 05e checkpoint, and falls back to the unified model plus unified `VecNormalize`.

### 2026-04-09 - Codex forest state-load and launcher stabilization pass
- Added per-milestone `load_settle_ticks` support to `train.py` and wired it through `PokemonYellowEnv.reset()`.
- Applied `load_settle_ticks=30` to the forest milestones:
  - `05d_forest_south`
  - `05e_forest_north`
  - `05f_gate_transition`
  - `05g_to_pewter`
- Added a startup banner line in `train.py` so the active settle value is visible in the log.
- Updated `validate_milestone_states.py` with `--settle-ticks` so save-state sanity checks can match the real training reset behavior.
- Confirmed locally that the repaired `05d_forest_south.state` needs post-load idle ticks before movement works at all; without that settle window it falsely behaves like a dead state.
- Probed the repaired forest states with both coordinate-state and full-state searches:
  - `05e_forest_north.state` is now definitely navigable within the forest body
  - but none of the currently recovered top-corridor / latest-hit states have produced a real `map=47` north-gate transition in local search
  - so the project is no longer blocked by the old trapped left-edge state, but it still lacks a truly validated post-forest handoff
- Updated WSL launch/monitor tooling:
  - `monitor_train.ps1` now uses `wsl --exec /bin/sh` plus `pgrep -af "train.py"` instead of the noisier `bash -lc` path
  - `run_train_wsl.ps1` now launches through `wsl --exec /bin/sh -lc`
  - added `start_train_wsl.ps1` as the new WSL trainer launcher
- The hidden detached-host approach did not stay alive reliably, so the working launcher path now uses a dedicated PowerShell host window running the WSL trainer in the foreground and teeing output into `wsl_gpu_train.log`.
- Verified syntax with:
  - `python -m compileall train.py validate_milestone_states.py`
- Relaunched Stage 10 successfully with the stabilized host path:
  - WSL trainer PID `450`
  - Windows host PID `35232`
  - milestone remains `10`
  - live startup now shows `State load settle: ticks=30`
  - first fresh Stage 10 metrics returned at roughly `226-245 fps` on the `AMD Radeon RX 9060 XT`

### 2026-04-09 — Claude Code: Forest state file confirmed broken (CRITICAL)
- BFS exploration confirmed `05e_forest_north.state` drops agent at (map=51, y=24, x=0) in a **dead-end corridor** with only 34 reachable tiles (y=14-31, x=0-1). **No exit exists from this section.**
- Full forest scan (teleport BFS across 26 regions) found 1,412+ tiles total, but the agent's starting region is completely isolated.
- The forest is 68×152 tiles. The agent is on the far-left edge (x=0-1) in a narrow dead-end strip.
- Despite Codex's Apr-08 state recovery efforts (moving from left-edge to wider corridor), the state still lacks a validated path to map=47.
- **All training on milestones 05e/05f is wasted while this state is broken.**
- **Action needed**: Create a proper forest state by manually playing through the south gate entrance, or skip forest milestones entirely.

### 2026-04-09 — Architecture discussion: PokemonRedExperiments v2 port
- Ref: https://github.com/PWhiddy/PokemonRedExperiments
- Agreed plan to port v2 design to Yellow:
  1. MultiInputPolicy with dict observations (screens + RAM channels)
  2. RAM side channels: map_id, y, x, HP, level, badges, events, recent_actions
  3. Coordinate/global exploration map instead of frame-KNN
  4. Event-flag rewards + stuck penalty + reward decomposition
- This replaces current pure-image CnnPolicy approach
- Priority: fix state file first, then port architecture

### 2026-04-09 - Codex Red-v2 bridge pass for Yellow
- Dug deeper into `PWhiddy/PokemonRedExperiments` `v2` and pulled over the parts that help our current Yellow trainer without breaking `CnnPolicy` checkpoint compatibility:
  - Red `v2` uses `MultiInputPolicy` plus RAM/state channels (`health`, `level`, `badges`, `events`, `map`, `recent_actions`) and coordinate-based exploration instead of frame-KNN.
  - Rather than hard-switching architectures mid-run, I added a checkpoint-compatible bridge inside `train.py`: the grayscale observation now stamps in a compact RAM/action/minimap overlay so the current CNN can see map/coord/HP/badge/battle/stuck context plus recent actions and local visited tiles.
- Added Red-inspired anti-loop logic in `train.py`:
  - recent action history is now tracked in-env
  - a small overworld oscillation penalty fires on repeated back-and-forth movement patterns
  - applied stage-specific oscillation penalties to the forest milestones
- Tightened the Stage 10 promotion rule in `train.py` so `05e_forest_north` no longer treats the bad `(1,0)` forest tile as a valid handoff. Future 05e hits must now land on `(1,1)` or `(2,1)`.
- Added `repair_gate_transition_state.py`:
  - scans recent `05e_forest_north` hit states
  - probes each candidate's post-load mobility
  - promotes the best viable hit into `milestones/05f_gate_transition.state`
- Ran the repair script and replaced the broken Stage 11 handoff:
  - selected `_05e_forest_north_latest_hit_1.state`
  - installed it as `05f_gate_transition.state`
  - direct probe now shows `map=51 a=2 b=1 battle=0`
  - validator now sees real mobility from Stage 11 (`13` coordinate states within shallow sanity depth), instead of the previous frozen `(1,0)` handoff
- Verified syntax with:
  - `python -m compileall train.py repair_gate_transition_state.py`
- Relaunched training the simple/reliable way:
  - started a dedicated PowerShell window running `run_train_wsl.ps1`
  - confirmed live WSL trainer process `python3 train.py` as PID `441`
  - synchronized `wsl_gpu_train.pid` and `wsl_gpu_host.pid`
  - monitor now sees the trainer again (`running pid=441 log_missing`)
- Detached logging is still flaky through the wrapper launchers; the live trainer is currently tied to the dedicated PowerShell host window rather than the old background log pipeline.

### 2026-04-10 - Codex trainer restart and launcher bug fix
- Restarted WSL and relaunched the GPU trainer after the overnight stop.
- Cleaned up a duplicate-trainer situation so only one live `python3 train.py` process remains.
- Current live trainer after cleanup:
  - milestone `11`
  - WSL trainer PID `428`
  - monitor status `running pid=428 log_missing`
- Fixed a launcher bug in `start_train_wsl.ps1`:
  - replaced the variable name `$host` with `$hostWindowProc`
  - this avoids colliding with PowerShell's built-in read-only `$Host`
  - future restarts will now write the correct Windows host PID to `wsl_gpu_host.pid`

### 2026-04-11 - Codex manual play station for coordinate mapping
- Rebuilt `play.py` into a proper mapping station for inspecting the bot's current start state.
- New manual-play behavior:
  - `--current-stage` now loads the current curriculum state's save file automatically from `progress.json`
  - coordinate/mode changes are auto-logged to a per-state mapping log under `milestones/`
  - map labels and battle labels are printed in a more readable format
  - added manual tools:
    - `F1` / `Space` prints current state info
    - `F2` toggles auto-log on movement
    - `F3` writes a manual waypoint marker with an optional note
    - `F4` saves a screenshot
    - `F5` saves a state
    - `Tab` dumps battle RAM
- Added `play_station.ps1` so the mapping station can be launched from Windows with one command while auto-activating `ml-env`.
- Verified syntax with `python -m compileall play.py`.

### 2026-04-11 - Codex manual player quick-marker fix
- Removed the blocking text prompt from `F3` in `play.py`.
- `F3` now writes an immediate `quick_marker` entry to the mapping log instead of calling `input(...)`, so it no longer pauses the emulator loop waiting for terminal text.
- Left `F5` prompting behavior unchanged, since save-state naming is intentionally interactive.
- Re-verified `play.py` with `python -m compileall play.py`.

### 2026-04-11 - Codex Stage 11 gate-hit logic fix from manual mapping
- Used the manual mapping station findings to correct the Stage 11 success logic in `train.py`.
- Key finding from the manual run:
  - entering `ForestGateN` (`map=47`) can happen while `battle_flag` is still stale/non-zero from a just-fled wild battle
  - the old `05f_gate_transition` check required `map=47` **and** `battle_flag==0`, which meant valid gate transitions could be missed entirely
- Updated `05f_gate_transition` to count the milestone hit on `map=47` regardless of the stale battle flag.
- Added deferred safe hit-state capture:
  - once a milestone is hit, the env now waits for the first safe overworld step on configured maps before saving the next-stage state
  - this avoids dropping the handoff because the first valid hit happened during a stale battle-flag transition
- Added two new per-stage controls in `train.py`:
  - `safe_hit_save_maps`
  - `terminate_after_hit_state_saved`
- Configured Stage 11 to:
  - save the next-stage handoff only on safe overworld `map=47`
  - terminate only after that safe hit-state save succeeds
- Verified syntax with `python -m compileall train.py`.

### 2026-04-12 - Codex parallel-attempts pass for Stage 11
- Evaluated whether multiple simultaneous trainers would help. Conclusion:
  - separate `train.py` processes would fight over the same checkpoints / unified model and dilute coherence
  - the safer parallelism is more vectorized envs inside a single PPO trainer
- Added per-stage `n_envs` support to `train.py`, defaulting to `POKEMON_N_ENVS` / `4` when unspecified.
- Updated Stage 11 (`05f_gate_transition`) to use:
  - `n_envs = 8`
  - `n_steps = 256`
  - this keeps each PPO rollout at roughly the same total batch size (`256 * 8 = 2048` env steps) while doubling the number of simultaneous attempts versus the old `4 x 512` setup
- Added `overworld_non_movement_penalty` to `train.py` and enabled it for Stage 11 to lightly discourage wasted `A/B` presses while still allowing battle exits when `battle_flag != 0`.
- Updated the startup/status text so env-count and rollout totals are printed correctly instead of assuming `4` envs.
- Re-verified syntax with `python -m compileall train.py`.
- Restarted the live trainer cleanly after the change:
  - current WSL trainer PID `470`
  - current monitor status `running pid=470 log_missing`

### 2026-04-12 - Codex true shadow-clone rollout workers
- Refactored `train.py` so stage settings are passed as per-env config instead of mutable `PokemonYellowEnv` class globals.
  - this makes the env safe to launch in subprocess workers
  - each worker now gets a stable `env_id`, so latest-hit states do not collide under parallel rollout workers
- Added a real vec-env builder in `train.py`:
  - `build_env_config(...)`
  - `resolve_vec_env_kind(...)`
  - `build_vec_env(...)`
- Switched WSL training to explicit vec-env backend selection:
  - `dummy` for single-env or unsupported cases
  - `subproc` for true parallel “shadow clone” rollouts
- Updated Stage 11 (`05f_gate_transition`) to explicitly request:
  - `n_envs = 8`
  - `vec_env_kind = subproc`
  - existing `n_steps = 256` kept, so overall PPO rollout batch size stays near the previous budget while attempts per wall-clock minute increase
- Updated callback compatibility for subprocess workers:
  - `ScreenshotCallback` now uses `env_method("save_screenshot", ...)`
  - `ProgressionCallback` now flushes episode summaries through `env_method(...)` when available
- Updated `run_train_wsl.sh` to export:
  - `POKEMON_VEC_ENV=subproc`
  - `POKEMON_SUBPROC_START_METHOD=forkserver`
- Verification:
  - `python -m compileall train.py`
  - WSL subprocess smoke test from a real temp script: `vec_env=subproc smoke_test=ok`
- Relaunched live training after the refactor:
  - main learner process: WSL PID `1865` (`python3 train.py`)
  - subprocess rollout workers: `8`
  - forkserver parent present in process tree
  - monitor resynced to `running pid=1865 log_missing`

### 2026-04-12 - Codex Stage 11 gate-path correction and fast-fail resets
- Identified the real Stage 11 bug from the live metrics:
  - the agent was repeatedly hitting `frontier=700` at `Forest:15,0`
  - that meant Stage 11 was still rewarding the wrong southward corridor instead of the actual north-gate entry
  - result: high reward, `0/50` hits, and huge `stuck` counts near the learned detour
- Corrected `05f_gate_transition` in `train.py`:
  - reduced the stage to the actual local forest exit path:
    - `Forest (2,1) -> (1,1) -> (1,0) -> ForestGateN`
  - removed the old southward waypoint chain to `(15,0)`
  - shortened the episode cap from `2048` to `512`
  - increased off-path penalties for south-gate / Route 2 / Viridian detours
  - increased anti-stuck pressure:
    - `coord_stuck_threshold = 12`
    - `coord_stuck_penalty = 0.35`
    - `action_oscillation_penalty = 0.20`
    - `overworld_non_movement_penalty = 1.0`
- Added generic frontier-stall fail-fast controls to `train.py`:
  - `frontier_stall_steps`
  - `frontier_stall_penalty`
  - Stage 11 now terminates early after `64` overworld steps with no frontier improvement, instead of burning the whole episode in a loop
- Kept the single-learner / 8-subprocess-worker shadow-clone setup intact.
- Verified syntax with:
  - `python -m compileall train.py`
- Restarted the trainer onto the corrected Stage 11:
  - current WSL learner PID `3437`
  - worker processes still `8`
  - monitor/status resynced to `running pid=3437 log_missing`

### 2026-04-12 - Codex Stage 11 exact-action guidance and fail-on-leave
- Re-evaluated the corrected Stage 11 after several more million steps:
  - the agent was now reliably reaching the right pre-gate forest tile (`Forest:1,0`)
  - but still not converting that into `map=47` hits
  - this meant the remaining blocker was the final local action sequence, not navigation or survival
- Added generic local action-guidance support to `train.py`:
  - `action_guidance`
  - `allowed_maps`
  - `disallowed_map_penalty`
- Added reward shaping on the *pre-action* state so the agent can be taught exact local moves on overworld without changing the policy head size.
- Tightened Stage 11 (`05f_gate_transition`) again:
  - explicit local guidance for:
    - `Forest (2,1)` -> prefer `up`
    - `Forest (1,1)` -> prefer `left`
    - `Forest (1,0)` -> prefer `up` into the gate
    - `ForestGateN (1,0)` -> prefer `right` to stay in the gate hallway after entry
  - immediate fail if the episode leaves the intended local maps:
    - allowed maps now only `[51, 47]`
    - disallowed-map penalty `1500`
- Kept the earlier fast-fail structure:
  - short `512`-step episode cap
  - frontier stall timeout
  - strong anti-stuck penalties
- Verified syntax with:
  - `python -m compileall train.py`
- Restarted live training again after the patch:
  - current WSL learner PID `12775`
  - current monitor/status resynced to `running pid=12775 log_missing`

### 2026-07-21 - Codex project reliability audit
- Added atomic JSON persistence in `project_paths.py` with fsync, atomic replace,
  cleanup, and a last-known-good backup. Curriculum startup can now recover from
  a corrupt `progress.json` instead of crashing or silently losing all progress.
- Added progress index validation before writes and defensive handling for
  malformed JSON field values on reads.
- Fixed checkpoint selection in `train.py` to use SB3's embedded timestep rather
  than filesystem modification time, which can change during copy/restore.
- Added strict checkpoint-name parsing and stopped pairing a PPO checkpoint with
  VecNormalize statistics from a different training step.
- Added `test_project_reliability.py` covering persistence recovery, backup
  preservation, checkpoint ordering, and invalid checkpoint names.
- Added a project `README.md` and `.gitignore` to document operation and keep the
  very large ROM/checkpoint/runtime artifacts out of future source control.
- Corrected misleading indentation in `build_env_config`.
- Verification:
  - Windows Python 3.14 compile pass
  - dependency-free Windows tests pass (training-dependent tests skip cleanly)
  - WSL Python 3.12 full suite passes with all training dependencies
- The live WSL trainer was deliberately not restarted; changes take effect on
  its next normal restart.

### 2026-07-21 - Codex navigation foundation
- Added `sync_navigation_data.py` to generate an offline navigation catalog from
  authoritative `pret/pokeyellow` map constants and the saved Pokémon Completion
  Yellow page. The generated data contains all 249 internal maps, dimensions,
  parent completion locations, 52 page locations, and their display coordinates.
- Added `yellow_navigation.py` with catalog validation, map-ID resolution, parent
  location resolution, macro-level breadth-first routing across Kanto, and ROM
  topology queries.
- Extended the catalog with outdoor connections and exact source coordinates for
  warp events extracted from the `pret/pokeyellow` repository archive.
- Centralized map labels used by `train.py`, `play.py`, and `map_forest.py` on the
  generated catalog instead of three handwritten partial dictionaries.
- Found and fixed a concrete shaping error: map ID 54 is `PEWTER_GYM`, not a
  northern copy of Route 2. Stage `05g_to_pewter` no longer rewards entering it.
- Added navigation extraction, identity, parent mapping, macro-route, and trainer
  integration regression tests.
- The running trainer was not restarted; the navigation changes apply on its
  next normal restart.

### 2026-07-21 - Codex Mt. Moon transition-stall fix
- Diagnosed the live `07_beat_brock` route stall from episode telemetry: workers
  repeatedly reached the Route 4 pre-warp waypoint (`frontier≈4400`) and then
  ended with `checkpoint_stall` without entering Mt. Moon.
- Found the concrete action error: at Route 4 `(y=6,x=18)`, the configured rule
  preferred `right`, but the authoritative Mt. Moon warp is directly `up` at
  `(y=5,x=18)`.
- Added `warp_entry_action()` so final transition actions are derived from ROM
  warp coordinates and rejected if the configured launch tile is not adjacent.
- Added exact entry actions for all required cave transitions:
  - Route 4 → Mt. Moon 1F: up
  - Mt. Moon 1F → B1F: left
  - Mt. Moon B1F → B2F: right
  - Mt. Moon B2F → B1F exit branch: right
- Extended generated warp data to preserve `LAST_MAP` exit coordinates.
- Added regression tests for each derived action and trainer integration; the
  full WSL suite passes.
- Restarted the trainer from the paired 714,782,576-step checkpoint. New WSL PID
  `23241` started successfully and reports `action_guidance=13` (fixed config).
