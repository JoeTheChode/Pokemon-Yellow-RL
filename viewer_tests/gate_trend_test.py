#!/usr/bin/env python3
"""Gate trend: telling "improving", "converging" and "hung" apart.

The promotion counter alone cannot distinguish them -- a plateau gate reading
"0 of 20" is equally consistent with the swarm finding a faster route every
few seconds (counter keeps resetting), with it having just started, and with
nothing clearing at all. Each of those wants a different reaction from the
operator, so each is asserted here.

Driven in-process with an injected clock: every property is a timing rule, and
a sleep-based test would assert "roughly, if the box is not busy".
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE if os.path.exists(os.path.join(HERE, "map_viewer_server.py")) \
    else os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "_fixture_gate_trend")

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


def main():
    build_fixture()
    sys.path.insert(0, FIXTURE)
    import map_viewer_server as mvs

    PHASE = 37

    def feed_for(script, phase=PHASE, start=1_000_000.0):
        """Replay (offset_seconds, no_improve, best) against a fresh feed."""
        feed = mvs.AgentFeed()
        feed._drilled_phase = lambda: phase
        for offset, no_improve, best in script:
            feed._mastery_counters = {
                'no_improve': {str(phase): no_improve},
                'best': {str(phase): best},
            }
            feed._next_gate_trend_poll = 0.0
            feed._update_gate_trend(start + offset)
        return feed

    # ------------------------------------------------------- improving
    # A faster route lands every ~20s, so the counter never gets far from 0.
    # This is the case that reads identically to a hang on the old card.
    improving = feed_for([
        (0, 0, 1600), (20, 3, 1600), (40, 0, 1540),
        (60, 2, 1540), (80, 0, 1495), (100, 1, 1495),
    ])
    t = improving.gate_trend_payload(1_000_100.0 + 5)
    check("a resetting counter reads as improving, not stalled",
          t['verdict'] == 'improving', t['verdict'])
    check("improvements are counted", t['improvements'] == 2, t)
    check("the last improvement's size is reported",
          t['improvement_delta'] == 45, t['improvement_delta'])
    check("the best route's starting point is kept for comparison",
          t['first_best'] == 1600 and t['best'] == 1495,
          (t['first_best'], t['best']))

    # ------------------------------------------------------ converging
    converging = feed_for([
        (0, 0, 1495), (20, 4, 1495), (40, 9, 1495),
        (60, 13, 1495), (80, 17, 1495),
    ])
    t = converging.gate_trend_payload(1_000_080.0 + 5)
    check("a climbing counter with no new best reads as converging",
          t['verdict'] == 'converging', t['verdict'])
    check("clears are counted from counter jumps, not sample count",
          t['clears'] == 17, t['clears'])
    check("no improvement is reported when the best never moved",
          t['improvements'] == 0 and t['improvement_age_s'] is None, t)

    # --------------------------------------------------------- stalled
    stalled = feed_for([(0, 0, 1495), (4, 2, 1495), (8, 4, 1495)])
    fresh = stalled.gate_trend_payload(1_000_008.0 + 30)
    check("a recent gap is not called a hang yet",
          fresh['verdict'] == 'converging', fresh['verdict'])
    late = stalled.gate_trend_payload(1_000_008.0 + 400)
    check("a long silence after real clears reads as stalled",
          late['verdict'] == 'stalled', late['verdict'])
    check("the stall reports how long it has been quiet",
          390 <= late['idle_s'] <= 410, late['idle_s'])
    check("the stall reports this objective's normal cadence",
          late['typical_clear_s'] == 4.0, late['typical_clear_s'])

    # The threshold has to scale: a slow objective must not be called hung
    # merely for taking its usual time between clears.
    slow = feed_for([(0, 0, 900), (300, 1, 900), (600, 2, 900)])
    slow_t = slow.gate_trend_payload(1_000_600.0 + 400)
    check("a slow objective is not called hung inside its own cadence",
          slow_t['verdict'] == 'converging',
          "%s (stall_after=%s)" % (slow_t['verdict'], slow_t['stall_after_s']))
    check("the stall threshold scales off the observed interval",
          slow_t['stall_after_s'] == 1200.0, slow_t['stall_after_s'])

    # -------------------------------------------------------- watching
    quiet = feed_for([(0, 0, None)])
    early = quiet.gate_trend_payload(1_000_030.0)
    check("nothing seen yet, and not for long, is 'watching' not 'stalled'",
          early['verdict'] == 'watching', early['verdict'])
    check("watching still reports how long it has been watching",
          25 <= early['watched_s'] <= 35, early['watched_s'])
    # ...but silence with no clears at all eventually IS the hang. This is the
    # case that survives a viewer restart with no history to go on.
    late_quiet = quiet.gate_trend_payload(1_000_000.0 + 400)
    check("prolonged silence with no clears at all becomes stalled",
          late_quiet['verdict'] == 'stalled', late_quiet['verdict'])

    # ------------------------------------------- bookkeeping guarantees
    unchanged = feed_for([(0, 5, 900), (10, 5, 900), (20, 5, 900)])
    check("unchanged counters are not recorded as events",
          len(unchanged.gate_trend) == 1, len(unchanged.gate_trend))

    switched = feed_for([(0, 9, 900)])
    switched._drilled_phase = lambda: PHASE + 1
    switched._mastery_counters = {
        'no_improve': {str(PHASE + 1): 0}, 'best': {str(PHASE + 1): 400},
    }
    switched._next_gate_trend_poll = 0.0
    switched._update_gate_trend(1_000_050.0)
    check("moving to a new objective starts its trend from scratch",
          switched.gate_trend_phase == PHASE + 1
          and len(switched.gate_trend) == 1, switched.gate_trend)
    check("the trend names the phase it describes",
          switched.gate_trend_payload(1_000_051.0)['phase'] == PHASE + 1)

    check("no trend at all before a phase is known",
          mvs.AgentFeed().gate_trend_payload(1_000_000.0) is None)

    print()
    if failures:
        print("FAILED (%d): %s" % (len(failures), failures))
        return 1
    print("all gate-trend checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
