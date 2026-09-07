#!/usr/bin/env bash
# Restart pokemon-train.service when it is alive but not making progress.
#
# 2026-08-07: one env worker hung inside pyboy.tick(), which blocked
# SubprocVecEnv.step_wait() and deadlocked all 96 envs. The process stayed
# systemd-"active" the whole time, so Restart=always never fired and training
# was silently down for ~2 hours. Liveness has to be measured, not assumed.
#
# train.log is appended by the trainer's stdout and never goes more than ~10s
# without a write while healthy, so staleness is a reliable progress proxy.
set -uo pipefail

LOG=/home/ubuntu/pokemon-rl/train.log
WLOG=/home/ubuntu/pokemon-rl/watchdog.log
MAX_STALE=600

systemctl is-active --quiet pokemon-train.service || exit 0

now=$(date +%s)
mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
age=$(( now - mtime ))

if [ "$age" -ge "$MAX_STALE" ]; then
  echo "$(date -u +%FT%TZ) [WATCHDOG] train.log stale ${age}s (>= ${MAX_STALE}s) - restarting pokemon-train.service" >> "$WLOG"
  systemctl restart pokemon-train.service
  echo "$(date -u +%FT%TZ) [WATCHDOG] restart issued" >> "$WLOG"
fi
