#!/usr/bin/env python3
"""Fixture smoke test for the viewer server's freshness plumbing.

Runs against a throwaway directory containing a synthetic train.log and a
couple of per-env position files. Asserts the behaviour the "log stale" fix
depends on:

  1. log_updated_at is published at all (the field the client now keys off).
  2. It advances when train.log grows, even with no [METRICS] block in sight
     -- that is the exact case the old metrics-based check got wrong.
  3. It does NOT advance while the log is idle.
  4. metrics_updated_at still tracks [METRICS] blocks independently.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# Works both from the repo (viewer_tests/ next to the sources' parent) and
# from a scratchpad copy sitting alongside the sources.
SRC = HERE if os.path.exists(os.path.join(HERE, "map_viewer_server.py")) \
    else os.path.dirname(HERE)
ROOT = HERE
FIXTURE = os.path.join(ROOT, "_fixture")
PORT = int(os.environ.get("SMOKE_PORT", "8731"))

EP_LINE = (
    "  [EP] route(maps_visited)=Celadon City | unique_tiles_visited=72 | "
    "quest_phase=126/156 | end_reason=swarm_catchup\n"
)
METRICS_BLOCK = (
    "  [METRICS] hall_of_fame | steps=1,745,030,400 (total, all sessions) | "
    "session_steps=172,032 (since this process launched) | fps=806\n"
    "    avg_ep_reward    284        good\n"
    "    milestone_hits   0/50       no clears yet (of last 50 episodes)\n"
    "\n"
)

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def fetch():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/live_agent_positions.json", timeout=5) as r:
        return json.load(r)


def append(text):
    with open(os.path.join(FIXTURE, "train.log"), "a", encoding="utf-8") as h:
        h.write(text)


def main():
    os.makedirs(FIXTURE, exist_ok=True)
    for name in ("map_viewer_server.py", "pokemon_yellow_map_viewer.html"):
        src = os.path.join(SRC, name)
        dst = os.path.join(FIXTURE, name)
        with open(src, "rb") as a, open(dst, "wb") as b:
            b.write(a.read())
    with open(os.path.join(FIXTURE, "train.log"), "w", encoding="utf-8") as h:
        h.write(EP_LINE)
    for env_id in (0, 1):
        entry = {
            "env_id": env_id,
            "user": "SVER-YV",
            "color": "#FFD700",
            "last_position": [3, 4, 6],
            "map_id": 6,
            "pikachu_level": 5,
            "battle_status": "overworld",
            "battle_source": "wIsInBattle@D056",
            "visited_count": 100,
            "progress_count": 3,
            "last_seen": time.time(),
        }
        path = os.path.join(FIXTURE, f"live_agent_positions.env{env_id}.json")
        with open(path, "w", encoding="utf-8") as h:
            json.dump(entry, h)

    env = dict(
        os.environ,
        POKEMON_VIEWER_MAX_ENVS="4",
        POKEMON_VIEWER_FEED_HZ="10",
        POKEMON_VIEWER_PORT=str(PORT),
    )
    proc = subprocess.Popen(
        [sys.executable, "map_viewer_server.py"],
        cwd=FIXTURE, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        payload = None
        for _ in range(40):
            time.sleep(0.5)
            try:
                payload = fetch()
                break
            except Exception:
                continue
        if payload is None:
            out = ""
            if proc.poll() is not None:
                out = proc.stdout.read()
            print("  server never came up:", out[:2000])
            return 1

        time.sleep(1.5)
        payload = fetch()
        training = payload.get("training") or {}
        check("training.log_updated_at is published",
              "log_updated_at" in training, repr(training.get("log_updated_at")))
        first_log = training.get("log_updated_at")
        check("log_updated_at is set after the initial read",
              isinstance(first_log, (int, float)), repr(first_log))

        # Idle: no new bytes -> must not advance.
        time.sleep(2.0)
        idle = (fetch().get("training") or {}).get("log_updated_at")
        check("log_updated_at does NOT advance while the log is idle",
              idle == first_log, f"{first_log} -> {idle}")

        # Growth with no METRICS block at all -> must advance.
        metrics_before = (fetch().get("training") or {}).get("metrics_updated_at")
        append(EP_LINE * 3)
        time.sleep(1.5)
        training = fetch().get("training") or {}
        grown = training.get("log_updated_at")
        check("log_updated_at advances on [EP]-only growth",
              isinstance(grown, (int, float)) and grown > first_log,
              f"{first_log} -> {grown}")
        check("metrics_updated_at unchanged by [EP]-only growth",
              training.get("metrics_updated_at") == metrics_before,
              f"{metrics_before} -> {training.get('metrics_updated_at')}")

        # A real METRICS block -> metrics_updated_at moves and rows parse.
        append(METRICS_BLOCK + EP_LINE)
        time.sleep(1.5)
        training = fetch().get("training") or {}
        metrics = training.get("metrics") or {}
        check("metrics_updated_at advances on a [METRICS] block",
              isinstance(training.get("metrics_updated_at"), (int, float))
              and training.get("metrics_updated_at") != metrics_before,
              repr(training.get("metrics_updated_at")))
        check("metrics block parsed (phase + steps + rows)",
              metrics.get("phase") == "hall_of_fame"
              and metrics.get("total_steps") == 1745030400
              and "avg_ep_reward" in (metrics.get("rows") or {}),
              json.dumps(metrics)[:200])
        check("agents still served alongside training payload",
              len(payload.get("agents") or []) >= 1
              or len(fetch().get("agents") or []) >= 1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return 1
    print("all smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
