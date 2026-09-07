#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --expected-pid PID --reason SLUG (--resume | --new-attempt ID) [--force-reset] [--allow-rapid]" >&2
  exit 64
}

EXPECTED_PID=""
REASON=""
MODE=""
RESET_ID=""
ALLOW_RAPID=0
FORCE_RESET=0
MANAGER_ENV_ARMED=0

cleanup_manager_env() {
  if (( MANAGER_ENV_ARMED == 1 )); then
    # These values are only inputs to the process started by this invocation.
    # Never let a failed restart leak them into an unrelated future restart.
    if ! sudo systemctl unset-environment \
      POKEMON_QUEST_ZERO_RESET_ID \
      POKEMON_FORCE_QUEST_ZERO_RESET \
      POKEMON_ALLOW_RAPID_QUEST_ZERO_RESET; then
      echo "WARNING: could not clear one-shot systemd environment" >&2
      return 1
    fi
    MANAGER_ENV_ARMED=0
  fi
}

trap 'cleanup_manager_env || true' EXIT

while (($#)); do
  case "$1" in
    --expected-pid) (($# >= 2)) || usage; EXPECTED_PID="$2"; shift 2 ;;
    --reason) (($# >= 2)) || usage; REASON="$2"; shift 2 ;;
    --resume) [[ -z "$MODE" ]] || usage; MODE="resume"; shift ;;
    --new-attempt)
      (($# >= 2)) || usage
      [[ -z "$MODE" ]] || usage
      MODE="new"
      RESET_ID="$2"
      shift 2
      ;;
    --allow-rapid) ALLOW_RAPID=1; shift ;;
    --force-reset) FORCE_RESET=1; shift ;;
    *) usage ;;
  esac
done

[[ "$EXPECTED_PID" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$REASON" =~ ^[A-Za-z0-9._-]+$ ]] || usage
[[ "$MODE" == "resume" || ("$MODE" == "new" && "$RESET_ID" =~ ^[A-Za-z0-9._:-]+$) ]] || usage
[[ "$FORCE_RESET" == "0" || "$MODE" == "new" ]] || usage

SERVICE="pokemon-train.service"
ROOT="/home/ubuntu/pokemon-rl"
LOCK="$ROOT/.pokemon-train-restart.lock"
COOLDOWN_SECONDS="${POKEMON_RESTART_COOLDOWN_SECONDS:-600}"
[[ "$COOLDOWN_SECONDS" =~ ^[0-9]+$ ]] || {
  echo "REFUSED: cooldown must be a non-negative integer" >&2
  exit 64
}
exec 9>"$LOCK"
flock -n 9 || { echo "REFUSED: another restart/deployment owns $LOCK" >&2; exit 75; }

CURRENT_PID="$(systemctl show "$SERVICE" -p MainPID --value)"
if [[ "$CURRENT_PID" != "$EXPECTED_PID" ]]; then
  echo "REFUSED: audited PID $EXPECTED_PID is stale; live PID is $CURRENT_PID" >&2
  exit 73
fi
AGE_SECONDS="$(ps -o etimes= -p "$CURRENT_PID" | tr -d " ")"
[[ "$AGE_SECONDS" =~ ^[0-9]+$ ]] || { echo "REFUSED: cannot determine live process age" >&2; exit 74; }
if (( AGE_SECONDS < COOLDOWN_SECONDS && ALLOW_RAPID == 0 )); then
  echo "REFUSED: live PID $CURRENT_PID started ${AGE_SECONDS}s ago; cooldown is ${COOLDOWN_SECONDS}s" >&2
  exit 75
fi

# A routine code activation must not quarantine an advanced durable frontier.
# A deliberate fresh pass remains available through the explicit override.
MAX_RESET_PHASE="${POKEMON_QUEST_ZERO_MAX_RESET_PHASE:-40}"
[[ "$MAX_RESET_PHASE" =~ ^[0-9]+$ ]] || MAX_RESET_PHASE=40
if [[ "$MODE" == "new" && "$ALLOW_RAPID" == "0" && "$FORCE_RESET" == "0" && "$MAX_RESET_PHASE" -gt 0 ]]; then
  FRONTIER_JSON="$ROOT/restart_start.swarm_frontier.json"
  LIVE_PHASE="$(python3 -c 'import json,sys
try:
    print(int(json.load(open(sys.argv[1])).get("quest_phase", -1)))
except Exception:
    print(-1)' "$FRONTIER_JSON" 2>/dev/null || echo -1)"
  if [[ "$LIVE_PHASE" =~ ^[0-9]+$ ]] && (( LIVE_PHASE >= MAX_RESET_PHASE )); then
    echo "REFUSED: live frontier is at quest_phase $LIVE_PHASE (>= $MAX_RESET_PHASE)." >&2
    echo "         Use --resume for code activation, or --force-reset for an explicit fresh pass." >&2
    exit 75
  fi
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="$ROOT/deployment_backups/safe_restart_${STAMP}_${REASON}"
mkdir -p "$BACKUP"
for NAME in train.py restart_start.quest_zero_mastery.json restart_start.swarm_frontier.json restart_start.swarm_frontier.state restart_start.swarm_mastery.json restart_start.swarm_proven_routes.json; do
  [[ -e "$ROOT/$NAME" ]] && cp -a "$ROOT/$NAME" "$BACKUP/"
done

if [[ "$MODE" == "new" ]]; then
  MANAGER_ENV_ARMED=1
  sudo systemctl set-environment "POKEMON_QUEST_ZERO_RESET_ID=$RESET_ID"
  if (( FORCE_RESET == 1 )); then
    sudo systemctl set-environment POKEMON_FORCE_QUEST_ZERO_RESET=1
  else
    sudo systemctl unset-environment POKEMON_FORCE_QUEST_ZERO_RESET
  fi
  if (( ALLOW_RAPID == 1 )); then
    sudo systemctl set-environment POKEMON_ALLOW_RAPID_QUEST_ZERO_RESET=1
  else
    sudo systemctl unset-environment POKEMON_ALLOW_RAPID_QUEST_ZERO_RESET
  fi
else
  # A resume must never inherit one-shot intent left by an interrupted or
  # manually performed new-attempt restart.
  sudo systemctl unset-environment \
    POKEMON_QUEST_ZERO_RESET_ID \
    POKEMON_FORCE_QUEST_ZERO_RESET \
    POKEMON_ALLOW_RAPID_QUEST_ZERO_RESET
fi
sudo systemctl restart "$SERVICE"
if [[ "$MODE" == "new" ]]; then
  # The new process has already inherited the one-shot values. Remove both
  # from the manager environment before checking the new service PID.
  cleanup_manager_env
fi
NEW_PID="$(systemctl show "$SERVICE" -p MainPID --value)"
[[ "$NEW_PID" =~ ^[1-9][0-9]*$ && "$NEW_PID" != "$CURRENT_PID" ]] || {
  echo "FAILED: service did not acquire a new PID" >&2
  exit 70
}
printf '%s old_pid=%s new_pid=%s mode=%s reset_id=%s reason=%s backup=%s\n' \
  "$STAMP" "$CURRENT_PID" "$NEW_PID" "$MODE" "$RESET_ID" "$REASON" "$BACKUP" \
  >> "$ROOT/restart_audit.log"
echo "OK: $SERVICE old_pid=$CURRENT_PID new_pid=$NEW_PID mode=$MODE backup=$BACKUP"
