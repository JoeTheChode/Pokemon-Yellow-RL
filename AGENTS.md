# Live trainer restart protocol

The shared production trainer must not be stopped, started, or restarted with
direct `systemctl` commands. Multiple agents can audit the same stale PID and
otherwise reset a newly started attempt twice.

Before any live restart:

1. Record the PID you actually audited.
2. Re-check the live frontier and confirm the diagnosed condition still belongs
   to that PID.
3. Deploy code without restarting, then invoke `safe_restart_train.sh` with
   that recorded PID and a short reason slug.
4. **Activate every code change with `--resume`.** The swarm proves the fix by
   clearing the affected phase on its next mastery rotation -- it does NOT need
   to re-run the earlier phases. There is no "restart the pass to prove a fix"
   step; that policy cost a week of progress (CxC 2026-09-03/04, ~42 resets).
5. `--new-attempt <id>` starts over from quest 0 and is ONLY for an explicit
   user request to restart the whole pass, or a genuinely unrecoverable run.
   It is currently disabled at two layers:
   - `POKEMON_KEEP_EXISTING_SWARM_FRONTIER=1` (drop-in
     `pokemon-train.service.d/frontier-durability.conf`) makes train.py ignore
     any reset_id. `safe_restart_train.sh` refuses `--new-attempt` while it is
     set.
   - Past `POKEMON_QUEST_ZERO_MAX_RESET_PHASE` (40) a reset also needs
     `--force-reset`, not just `--allow-rapid`.
   To genuinely start over: get the user's approval, clear
   `POKEMON_KEEP_EXISTING_SWARM_FRONTIER`, then
   `--new-attempt <id> --force-reset`.
6. Never delete, rename, or quarantine `restart_start.swarm_mastery.json`.
   Mastery, proven routes, and weights are durable curriculum evidence.
7. Log every deploy/restart and its reasoning in `.claude/CxC.md`.

The script serializes restart ownership, refuses a stale PID, refuses another
restart during the ten-minute startup cooldown, backs up all durable state,
and records the operation. `--allow-rapid` skips only the cooldown (for
concurrent-auditor races) and must be justified to the user.

For an explicitly authorized permanent retirement of the trainer, use
`safe_retire_train.sh --expected-pid <audited-pid> --reason <slug>`. It uses the
same restart lock and stale-PID refusal, backs up the durable frontier/mastery
state, stops the service, verifies MainPID=0/inactive, and disables the unit.
Do not replace that path with direct `systemctl stop`/`disable` commands.
