#!/usr/bin/env python3
"""Per-objective elapsed time: frontier dwell + per-worker segment clocks.

Driven in-process with an injected clock rather than through the HTTP server,
because every property worth asserting here is a timing rule and a sleep-based
test would assert "roughly, if the box is not busy".

Covered, each of which is a way the feature can silently produce a wrong
number rather than no number:

  1. Dwell accrues to the phase the frontier is actually on, and the
     currently-open interval is included in reads.
  2. Dwell stops accruing while train.log is stale (a dead trainer must not
     bill its downtime to the current objective).
  3. A catch-up tail batch records the clear but NO time -- a viewer restart
     replays a backlog in one read and would otherwise stamp thousands of
     historical clears with "now".
  4. A worker's second clear in one episode is bracketed from its previous
     clear; its first clear is bracketed from the episode start.
  5. A stale mark from a previous episode is never used as a bracket, even
     if reset detection missed the reset (run_step alone would allow it).
  6. A new quest-zero attempt clears the timing ledger but keeps the
     all-time step splits.
"""
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE if os.path.exists(os.path.join(HERE, "map_viewer_server.py")) \
    else os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "_fixture_elapsed")

failures = []


def check(label, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                           (" -- " + str(detail)) if detail else ""))
    if not ok:
        failures.append(label)


def build_fixture():
    if os.path.isdir(FIXTURE):
        shutil.rmtree(FIXTURE)
    os.makedirs(FIXTURE)
    shutil.copy(os.path.join(SRC, "map_viewer_server.py"), FIXTURE)
    open(os.path.join(FIXTURE, "train.log"), "w", encoding="utf-8").close()


def quest_line(env_id, step, phase, name, total=302):
    return ("  [QUEST] env_id=%d step=%d phase_steps=%d phase=%d/%d "
            "completed=%s next=x map=1 y=1 x=1"
            % (env_id, step, step, phase, total, name))


def main():
    build_fixture()
    sys.path.insert(0, FIXTURE)
    import map_viewer_server as mvs

    feed = mvs.AgentFeed()

    def clear(env_id, step, phase, name, at, live=True):
        """Feed one [QUEST] line as if observed at wall time `at`."""
        feed._last_tail_wall = at
        feed._tail_live = live
        match = mvs.QUEST_LINE_RE.search(quest_line(env_id, step, phase, name))
        assert match is not None, "fixture quest line did not parse"
        feed._record_quest_split(match)

    # ---------------------------------------------------------- 1 & 2: dwell
    t = 1_000_000.0
    phase_now = {"value": 5}
    feed._current_quest_phase = lambda: phase_now["value"]

    feed.log_updated_at = t
    feed._update_phase_dwell(t)              # enters phase 5
    feed.log_updated_at = t + 30
    feed._update_phase_dwell(t + 30)         # still phase 5, interval open
    check("open dwell interval is included in a read",
          feed._phase_dwell_seconds(5, t + 30) == 30.0,
          feed._phase_dwell_seconds(5, t + 30))

    phase_now["value"] = 6
    feed.log_updated_at = t + 40
    feed._update_phase_dwell(t + 40)         # banks 40s to phase 5
    check("dwell banks to the phase it was spent on",
          feed.phase_dwell.get(5) == 40.0, feed.phase_dwell)
    check("advancing starts a fresh clock on the new phase",
          feed._phase_dwell_seconds(6, t + 40) is None,
          feed._phase_dwell_seconds(6, t + 40))
    check("first_reached recorded per phase",
          feed.phase_first_reached.get(6) == t + 40,
          feed.phase_first_reached)

    # 30s of genuinely live phase-6 time, then the trainer dies at t+70.
    feed.log_updated_at = t + 70
    feed._update_phase_dwell(t + 70)
    # The silence is only *detected* DWELL_STALE_LOG_SECONDS later. Only the
    # 30s up to the last log write may be billed, not the detection window.
    noticed = t + 70 + mvs.DWELL_STALE_LOG_SECONDS + 10
    feed._update_phase_dwell(noticed)
    check("dwell is billed only up to the trainer's last sign of life",
          feed.phase_dwell.get(6) == 30.0, feed.phase_dwell)
    feed._update_phase_dwell(noticed + 600)  # 10 more minutes of nothing
    check("dwell does not accrue at all while train.log is stale",
          feed.phase_dwell.get(6) == 30.0, feed.phase_dwell)

    # Trainer comes back on the same objective: the clock re-opens, the
    # downtime is not back-billed, and it is not counted as a second visit.
    visits_before = feed.phase_visits.get(6)
    # +601, not +600: the previous call already consumed this second's poll
    # slot, and real wall time never repeats a timestamp the way a synthetic
    # clock can.
    feed.log_updated_at = noticed + 601
    feed._update_phase_dwell(noticed + 601)
    check("recovery re-opens the clock without back-billing downtime",
          feed._phase_dwell_seconds(6, noticed + 602) == 31.0,
          feed._phase_dwell_seconds(6, noticed + 602))
    check("resuming the same objective is not a second visit",
          feed.phase_visits.get(6) == visits_before,
          "%s -> %s" % (visits_before, feed.phase_visits.get(6)))

    # ------------------------- 7: a fast frontier does not bill one phase
    # Under load the feed loop runs at seconds per iteration, so the frontier
    # can cross several short objectives between two polls. Whatever the
    # split rule is, it must not hand phase A the time its successors spent.
    feed2 = mvs.AgentFeed()
    hop = {"value": 10}
    feed2._current_quest_phase = lambda: hop["value"]
    feed2.log_updated_at = 6_000_000.0
    feed2._update_phase_dwell(6_000_000.0)     # enters phase 10
    hop["value"] = 14                          # 11, 12, 13 crossed unobserved
    feed2.log_updated_at = 6_000_040.0
    feed2._update_phase_dwell(6_000_040.0)
    banked = {p: feed2.phase_dwell.get(p) for p in (10, 11, 12, 13, 14)}
    check("a skipped-phase span is shared, not banked entirely on the last "
          "phase actually sampled",
          banked[10] == 10.0 and banked[11] == 10.0
          and banked[12] == 10.0 and banked[13] == 10.0,
          banked)
    check("the phase newly entered is not pre-billed",
          banked[14] is None, banked)
    check("split dwell is flagged as an estimate, measured dwell is not",
          feed2.phase_dwell_estimated == {10, 11, 12, 13},
          feed2.phase_dwell_estimated)

    # A normal single-step advance stays an exact measurement.
    hop["value"] = 15
    feed2.log_updated_at = 6_000_070.0
    feed2._update_phase_dwell(6_000_070.0)
    check("a single-phase advance is measured, not estimated",
          feed2.phase_dwell.get(14) == 30.0
          and 14 not in feed2.phase_dwell_estimated,
          "%s %s" % (feed2.phase_dwell.get(14), feed2.phase_dwell_estimated))

    # -------------------- 8: a save must not lose the open dwell interval
    feed3 = mvs.AgentFeed()
    feed3._dwell_phase = 42
    feed3._dwell_since = time.time() - 20
    feed3.phase_dwell[42] = 100.0
    snap = feed3._phase_dwell_snapshot()
    check("the open interval is folded into what gets persisted",
          119.0 <= snap[42] <= 121.0, snap)
    check("the in-process banked ledger is NOT mutated by taking a snapshot",
          feed3.phase_dwell[42] == 100.0, feed3.phase_dwell)
    check("a second snapshot does not compound the open interval",
          abs(feed3._phase_dwell_snapshot()[42] - snap[42]) < 1.0,
          feed3._phase_dwell_snapshot()[42])

    # --------------------------------------------- 3: backlog records no time
    feed.episode_start_wall_by_env[7] = 2_000_000.0
    clear(7, 100, 1, "reach_viridian", at=2_000_050.0, live=False)
    entry = feed.quest_splits[0]
    check("a catch-up batch still counts the clear",
          entry["clears"] == 1, entry)
    check("a catch-up batch records NO segment time",
          entry.get("best_secs") is None and entry.get("last_secs") is None,
          entry)

    # ------------------------------------- 4: episode-start and chained marks
    feed.episode_start_wall_by_env[8] = 3_000_000.0
    clear(8, 120, 1, "reach_viridian", at=3_000_012.0)
    first = feed.quest_splits[0]
    check("first clear of an episode is bracketed from the episode start",
          first.get("last_secs") == 12.0, first.get("last_secs"))

    clear(8, 300, 2, "enter_viridian_mart", at=3_000_020.0)
    second = feed.quest_splits[1]
    check("a later clear in the same episode brackets from the previous clear",
          second.get("last_secs") == 8.0, second.get("last_secs"))

    # A faster attempt must move `best_secs`; a slower one must not.
    feed.episode_start_wall_by_env[9] = 3_000_100.0
    clear(9, 90, 1, "reach_viridian", at=3_000_104.0)
    clear(9, 95, 1, "reach_viridian", at=3_000_199.0)
    check("best_secs keeps the fastest attempt",
          feed.quest_splits[0].get("best_secs") == 4.0,
          feed.quest_splits[0].get("best_secs"))
    check("last_secs reflects the most recent attempt",
          feed.quest_splits[0].get("last_secs") == 95.0,
          feed.quest_splits[0].get("last_secs"))

    # ---------------------------------- 5: a stale mark never spans a reset
    # env 10 clears, then its episode restarts (higher run_step, so run_step
    # alone would wrongly accept the old mark as this attempt's start).
    feed.episode_start_wall_by_env[10] = 4_000_000.0
    clear(10, 100, 1, "reach_viridian", at=4_000_010.0)
    feed.episode_start_wall_by_env[10] = 4_000_500.0   # reset, mark not popped
    clear(10, 900, 1, "reach_viridian", at=4_000_505.0)
    check("a mark from before the episode start is not used as a bracket",
          feed.quest_splits[0].get("last_secs") == 5.0,
          feed.quest_splits[0].get("last_secs"))

    # --------------------------------------------- 6: attempt reset scoping
    feed.quest_splits[0]["best"] = 105
    steps_before = feed.quest_splits[0]["best"]
    clears_before = feed.quest_splits[0]["clears"]
    marker = {"reset_id": "fullgame-TEST-2", "started_ts": 5_000_000.0,
              "schema": 2}
    import json
    with open(mvs.ATTEMPT_MARKER_PATH, "w", encoding="utf-8") as handle:
        json.dump(marker, handle)
    feed._next_attempt_poll = 0.0
    feed.attempt_reset_id = "fullgame-TEST-1"       # a previous attempt
    feed._poll_attempt_marker(5_000_001.0)
    check("a new attempt id is adopted",
          feed.attempt_reset_id == "fullgame-TEST-2", feed.attempt_reset_id)
    check("a new attempt clears frontier dwell",
          feed.phase_dwell == {} and feed.phase_visits == {},
          "%s %s" % (feed.phase_dwell, feed.phase_visits))
    check("a new attempt clears per-worker segment times",
          feed.quest_splits[0].get("best_secs") is None
          and feed.quest_splits[0].get("secs_samples") is None,
          feed.quest_splits[0])
    check("a new attempt KEEPS the all-time step splits",
          feed.quest_splits[0]["best"] == steps_before
          and feed.quest_splits[0]["clears"] == clears_before,
          feed.quest_splits[0])

    # Same id again must not wipe anything a second time.
    feed.phase_dwell[3] = 12.0
    feed.attempt_marker_mtime = None
    feed._next_attempt_poll = 0.0
    feed._poll_attempt_marker(5_000_100.0)
    check("re-reading the same attempt id is not a reset",
          feed.phase_dwell.get(3) == 12.0, feed.phase_dwell)

    # ------------------- 9: the sink shortlist must stay a SHORTLIST
    # A fixed top-N is not a shortlist: with 12 objectives cleared, a top-8
    # reproduced the whole table one row shorter, which is what prompted this.
    def sink_names(rows):
        total = sum(r['dwell_secs'] for r in rows if r.get('dwell_secs'))
        return [s['name'] for s in mvs.AgentFeed.quest_time_sinks(rows, total)]

    # One objective past the coverage bar on its own: nothing else is a sink.
    dominated = [
        {'phase': 0, 'name': 'a', 'dwell_secs': 900.0},
        {'phase': 1, 'name': 'b', 'dwell_secs': 60.0},
        {'phase': 2, 'name': 'c', 'dwell_secs': 50.0},
        {'phase': 3, 'name': 'd', 'dwell_secs': 40.0},
        {'phase': 4, 'name': 'e', 'dwell_secs': 30.0},
        {'phase': 5, 'name': 'f', 'dwell_secs': 20.0},
        {'phase': 6, 'name': 'g', 'dwell_secs': 10.0},
    ]
    check("one dominant objective produces a one-row shortlist, not a top-N",
          sink_names(dominated) == ['a'], sink_names(dominated))

    # The shape that prompted this: the live 12-objective run, where a top-8
    # listed all but four rows. Two objectives explain three quarters of it.
    live_shape = [
        {'phase': 8, 'name': 'heal_at_viridian', 'dwell_secs': 528.0},
        {'phase': 10, 'name': 'train_pikachu_level_7', 'dwell_secs': 93.0},
        {'phase': 4, 'name': 'reenter_oaks_lab', 'dwell_secs': 55.0},
        {'phase': 12, 'name': 'train_pikachu_level_9', 'dwell_secs': 44.0},
        {'phase': 11, 'name': 'train_pikachu_level_8', 'dwell_secs': 40.0},
        {'phase': 3, 'name': 'return_to_pallet', 'dwell_secs': 29.0},
        {'phase': 9, 'name': 'train_pikachu_level_6', 'dwell_secs': 27.0},
    ]
    check("the live 12-objective shape shortlists 2, not 8",
          sink_names(live_shape) == ['heal_at_viridian', 'train_pikachu_level_7'],
          sink_names(live_shape))

    # Genuinely flat: it should still stop well short of listing everything.
    flat = [
        {'phase': i, 'name': chr(ord('a') + i), 'dwell_secs': 100.0}
        for i in range(12)
    ]
    flat_names = sink_names(flat)
    check("a flat distribution is capped rather than listing every objective",
          len(flat_names) <= mvs.QUEST_TIME_SINK_ROWS < len(flat),
          "%d of %d" % (len(flat_names), len(flat)))

    # A trailing row that is neither a real share nor a real duration is not
    # a "sink" and must not pad the list out.
    noise = [
        {'phase': 0, 'name': 'big', 'dwell_secs': 300.0},
        {'phase': 1, 'name': 'also_big', 'dwell_secs': 200.0},
        {'phase': 2, 'name': 'noise', 'dwell_secs': 2.0},
    ]
    check("a trivial objective is never admitted as a time sink",
          'noise' not in sink_names(noise), sink_names(noise))
    check("no dwell at all yields no shortlist",
          mvs.AgentFeed.quest_time_sinks([], 0) == [], 'expected []')

    print()
    if failures:
        print("FAILED (%d): %s" % (len(failures), failures))
        return 1
    print("all quest elapsed-time checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
