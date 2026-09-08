#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --expected-pid PID --reason SLUG" >&2
  exit 64
}

EXPECTED_PID=""
REASON=""

while (($#)); do
  case "$1" in
    --expected-pid) (($# >= 2)) || usage; EXPECTED_PID="$2"; shift 2 ;;
    --reason) (($# >= 2)) || usage; REASON="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$EXPECTED_PID" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$REASON" =~ ^[A-Za-z0-9._-]+$ ]] || usage

SERVICE="pokemon-train.service"
ROOT="/home/ubuntu/pokemon-rl"
LOCK="$ROOT/.pokemon-train-restart.lock"

exec 9>"$LOCK"
flock -n 9 || { echo "REFUSED: another trainer deployment owns $LOCK" >&2; exit 75; }

CURRENT_PID="$(systemctl show "$SERVICE" -p MainPID --value)"
if [[ "$CURRENT_PID" != "$EXPECTED_PID" ]]; then
  echo "REFUSED: audited PID $EXPECTED_PID is stale; live PID is $CURRENT_PID" >&2
  exit 73
fi

ACTIVE_STATE="$(systemctl is-active "$SERVICE" || true)"
if [[ "$ACTIVE_STATE" != "active" ]]; then
  echo "REFUSED: $SERVICE is not active (state=$ACTIVE_STATE)" >&2
  exit 74
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="$ROOT/deployment_backups/safe_retire_${STAMP}_${REASON}"
mkdir -p "$BACKUP"

for NAME in \
  train.py \
  restart_start.quest_zero_mastery.json \
  restart_start.swarm_frontier.json \
  restart_start.swarm_frontier.state \
  restart_start.swarm_mastery.json \
  restart_start.swarm_proven_routes.json \
  restart_audit.log; do
  [[ -e "$ROOT/$NAME" ]] && cp -a "$ROOT/$NAME" "$BACKUP/"
done

systemctl cat "$SERVICE" > "$BACKUP/${SERVICE}.unit.txt"
systemctl status "$SERVICE" --no-pager > "$BACKUP/${SERVICE}.status-before.txt" || true

# The repository policy forbids direct operator systemctl stop/start/restart.
# This script is the serialized, PID-checked retirement path.
sudo systemctl stop "$SERVICE"

POST_PID="$(systemctl show "$SERVICE" -p MainPID --value)"
POST_STATE="$(systemctl is-active "$SERVICE" || true)"
if [[ "$POST_PID" != "0" || "$POST_STATE" != "inactive" ]]; then
  echo "FAILED: $SERVICE did not stop cleanly (pid=$POST_PID state=$POST_STATE)" >&2
  exit 70
fi

sudo systemctl disable "$SERVICE" >/dev/null
ENABLED_STATE="$(systemctl is-enabled "$SERVICE" 2>/dev/null || true)"
if [[ "$ENABLED_STATE" == "enabled" ]]; then
  echo "FAILED: $SERVICE is still enabled" >&2
  exit 71
fi

printf '%s old_pid=%s action=retire reason=%s backup=%s enabled_state=%s\n' \
  "$STAMP" "$CURRENT_PID" "$REASON" "$BACKUP" "$ENABLED_STATE" \
  >> "$ROOT/restart_audit.log"

echo "OK: $SERVICE retired old_pid=$CURRENT_PID state=$POST_STATE enabled=$ENABLED_STATE backup=$BACKUP"
