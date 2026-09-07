#!/usr/bin/env python3
"""Fixture test for [EVENT] parsing and event-progress persistence.

Pins two defects found live on 2026-08-12, both in the swarm's "named events
reached" metric:

  1. EVENT_LINE_RE's trailing capture was `(.+?)\\s*$`. train.py's 96 workers
     share one stdout, so records interleave without a newline and two land on
     one physical line; the `$` anchor made the capture swallow the following
     record as part of the event name. Every interleaving minted a unique
     bogus "name", so the distinct-event set grew without bound -- live it read
     3,698 names against a 522-entry catalog (a 709% coverage bar).
  2. events_by_env was never persisted, while event_log_offset was. A restart
     therefore resumed reading train.log near EOF with an empty event set, so
     the metric fell to zero and could never rebuild.

Run against the pre-fix map_viewer_server.py and checks 2, 3, 4, 6 and 7 fail;
that inversion is what makes this a regression test rather than a tautology.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE if os.path.exists(os.path.join(HERE, "map_viewer_server.py")) \
    else os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "_fixture_events")
PORT = int(os.environ.get("EVENT_PARSE_PORT", "8733"))

# Two records on one physical line with no newline between them -- exactly the
# shape train.log takes when workers interleave (252 such lines in a 400MB
# live sample).
INTERLEAVED = (
    "  [EVENT] env_id=0 step=768 EVENT_BEAT_BROCK"
    "  [EVENT] env_id=1 step=768 EVENT_GOT_HM01\n"
)
# Each interleaved line mints one *distinct* debris name under the old parser
# (the step number differs, so the swallowed tail differs). Enough of them and
# the debris alone overruns the catalog denominator -- which is precisely how
# the live dashboard reached 3,698 / 522.
DEBRIS_FLOOD = "".join(
    f"  [EVENT] env_id=0 step={step} EVENT_BEAT_BROCK"
    f"  [EVENT] env_id=1 step={step} EVENT_GOT_HM01\n"
    for step in range(1000, 1008)
)
# An [EVENT] followed by the head of an [EP] block, the other observed shape.
EVENT_THEN_EP = (
    "  [EVENT] env_id=0 step=128 EVENT_GOT_HM02"
    "  [EP] route(maps_visited)=Viridian Forest | quest_phase=12/156\n"
)
CATALOG = {
    "num_events": 2560,
    "events": [
        {"bit": i, "name": n}
        for i, n in enumerate([
            "EVENT_BEAT_BROCK", "EVENT_GOT_HM01", "EVENT_GOT_HM02",
            "EVENT_GOT_HM03", "EVENT_BEAT_MISTY",
        ])
    ],
}

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def fetch():
    url = f"http://127.0.0.1:{PORT}/live_agent_positions.json"
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def append(text):
    with open(os.path.join(FIXTURE, "train.log"), "a", encoding="utf-8") as h:
        h.write(text)


def boot():
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
    for _ in range(40):
        time.sleep(0.5)
        try:
            return proc, fetch()
        except Exception:
            if proc.poll() is not None:
                print("  server died:", proc.stdout.read()[:2000])
                return proc, None
    return proc, None


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def setup():
    os.makedirs(os.path.join(FIXTURE, "navigation_data"), exist_ok=True)
    for name in ("map_viewer_server.py", "pokemon_yellow_map_viewer.html"):
        src = os.path.join(SRC, name)
        if os.path.exists(src):
            with open(src, "rb") as a, open(os.path.join(FIXTURE, name), "wb") as b:
                b.write(a.read())
    with open(os.path.join(FIXTURE, "navigation_data", "yellow_event_flags.json"), "w") as h:
        json.dump(CATALOG, h)
    for stale in ("viewer_heatmap.json", "train.log"):
        path = os.path.join(FIXTURE, stale)
        if os.path.exists(path):
            os.unlink(path)
    with open(os.path.join(FIXTURE, "train.log"), "w", encoding="utf-8") as h:
        h.write("  [EVENT] env_id=0 step=64 EVENT_BEAT_MISTY\n")
    for env_id in (0, 1):
        entry = {
            "env_id": env_id, "user": "SVER-YV", "color": "#FFD700",
            "last_position": [3, 4, 6], "map_id": 6, "pikachu_level": 5,
            "battle_status": "overworld", "battle_source": "wIsInBattle@D056",
            "visited_count": 100, "progress_count": 0, "last_seen": time.time(),
        }
        with open(os.path.join(FIXTURE, f"live_agent_positions.env{env_id}.json"), "w") as h:
            json.dump(entry, h)


def main():
    setup()
    proc, payload = boot()
    if payload is None:
        return 1
    try:
        append(INTERLEAVED + EVENT_THEN_EP + DEBRIS_FLOOD)
        time.sleep(1.5)
        payload = fetch()
        agg = ((payload.get("training") or {}).get("aggregate")) or {}
        agents = {a["env_id"]: a for a in payload.get("agents") or []}
        reached = agg.get("named_events_reached")
        total = agg.get("named_events_total")

        check("catalog denominator loaded", total == len(CATALOG["events"]), repr(total))
        check("numerator never exceeds the catalog denominator",
              isinstance(reached, int) and reached <= total, f"{reached} / {total}")
        # 4 distinct real events: MISTY, BROCK, HM01, HM02.
        check("all interleaved records parsed (not just the first)",
              reached == 4, f"reached={reached}, expected 4")
        check("env 1's record on an interleaved line is attributed to env 1",
              agents.get(1, {}).get("progress_count") == 1,
              f"env1 progress_count={agents.get(1, {}).get('progress_count')}")
        check("env 0 got its 3 events",
              agents.get(0, {}).get("progress_count") == 3,
              f"env0 progress_count={agents.get(0, {}).get('progress_count')}")
        # Real names are bare [A-Z0-9_] tokens, so a bracket anywhere in a
        # reported event name means a following log record was swallowed.
        latest_all = " | ".join(
            str(a.get("latest_event") or "") for a in agents.values()
        )
        check("no log debris glued onto an event name",
              "[" not in latest_all, latest_all[:160])

        # An uncatalogued name must not inflate the coverage ratio. Asserted
        # as a *delta* against the count taken immediately before, so this
        # cannot accidentally pass by cancelling out an unrelated miscount.
        before_unknown = reached
        append("  [EVENT] env_id=0 step=900 EVENT_NOT_IN_CATALOG\n")
        time.sleep(1.5)
        agg = ((fetch().get("training") or {}).get("aggregate")) or {}
        check("uncatalogued name does not move the coverage count",
              agg.get("named_events_reached") == before_unknown,
              f"{before_unknown} -> {agg.get('named_events_reached')}")

        # The restart check below is only meaningful once state has actually
        # been written with the offset at EOF -- otherwise the restart just
        # re-reads the whole fixture log from byte 0 and every build "passes".
        state_path = os.path.join(FIXTURE, "viewer_heatmap.json")
        log_size = os.path.getsize(os.path.join(FIXTURE, "train.log"))
        saved_at_eof = False
        for _ in range(90):
            time.sleep(1.0)
            try:
                with open(state_path) as h:
                    if json.load(h).get("event_log_offset") == log_size:
                        saved_at_eof = True
                        break
            except (OSError, ValueError):
                continue
        check("state saved with the log offset at EOF (restart precondition)",
              saved_at_eof, f"log_size={log_size}")
    finally:
        stop(proc)

    # Persistence: a restart must not lose event progress. The saved offset
    # resumes near EOF, so anything not persisted is gone for good.
    time.sleep(0.5)
    proc, payload = boot()
    if payload is None:
        return 1
    try:
        time.sleep(1.5)
        agg = ((fetch().get("training") or {}).get("aggregate")) or {}
        check("event progress survives a restart",
              agg.get("named_events_reached") == 4,
              f"reached={agg.get('named_events_reached')} after restart, expected 4")
        check("restart did not resurrect debris from the old state file",
              (agg.get("named_events_reached") or 0) <= (agg.get("named_events_total") or 0),
              f"{agg.get('named_events_reached')} / {agg.get('named_events_total')}")
    finally:
        stop(proc)

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return 1
    print("all event-parse checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
