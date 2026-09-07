#!/usr/bin/env python3
"""Fixture test for the mastery-gate progress counter.

The gate has two modes and the counter has to report whichever is actually in
force, or it becomes a confident readout of a rule nobody applies:

  plateau      -- deterministic objectives advance after `no_improve_rotations`
                  consecutive clears that set no new best. Clearing faster
                  restarts the streak.
  reliability  -- `train_*` level grinds advance on `grind_successes` clears
                  regardless of step count, because their duration is
                  dominated by encounter RNG.

The plateau streak is read from the trainer's durable `no_improve_counts`
rather than re-derived from the rolling window: a window-derived streak can
only reach window_size - 1, so for targets near the window size it
under-reports and the tab disagrees with the gate the trainer applies.

Rewritten 2026-08-16: the previous version asserted the retired hit-rate rule
(80% of the window under budget) and failed once promotion moved to plateau.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE if os.path.exists(os.path.join(HERE, 'map_viewer_server.py')) \
    else os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, '_fixture_gate')
PORT = int(os.environ.get('GATE_PORT', '8751'))

failures = []


def check(label, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def fetch():
    with urllib.request.urlopen(
            f'http://127.0.0.1:{PORT}/quest_splits.json', timeout=5) as r:
        return json.load(r)


def write(name, payload):
    with open(os.path.join(FIXTURE, name), 'w', encoding='utf-8') as h:
        json.dump(payload, h)


def quest(env, step, phase, name, total=296):
    return (f'  [QUEST] env_id={env} step={step} '
            f'phase={phase}/{total} completed={name} next=x map=1 y=2 x=3\n')


def gate_for(phase):
    for entry in (fetch().get('mastery') or {}).get('phases', []):
        if entry.get('phase') == phase:
            return entry
    return None


def main():
    os.makedirs(os.path.join(FIXTURE, 'navigation_data'), exist_ok=True)
    for name in ('map_viewer_server.py', 'pokemon_yellow_map_viewer.html'):
        src = os.path.join(SRC, name)
        if os.path.exists(src):
            with open(src, 'rb') as a, open(os.path.join(FIXTURE, name), 'wb') as b:
                b.write(a.read())
    with open(os.path.join(FIXTURE, 'navigation_data', 'yellow_event_flags.json'), 'w') as h:
        json.dump({'num_events': 2560, 'events': [{'bit': 0, 'name': 'EVENT_X'}]}, h)
    for stale in ('viewer_heatmap.json',):
        p = os.path.join(FIXTURE, stale)
        if os.path.exists(p):
            os.unlink(p)

    # Non-default thresholds: a hardcoding server would report 20/5 instead.
    with open(os.path.join(FIXTURE, 'train.py'), 'w', encoding='utf-8') as h:
        h.write(
            "fullgame_milestone = {\n"
            "    'swarm_drill_window_size': 32,\n"
            "    'swarm_drill_min_samples': 8,\n"
            "    'swarm_drill_hit_rate': 0.8,\n"
            "    'swarm_drill_max_steps': 4000,\n"
            "    'swarm_drill_early_max_steps': 500,\n"
            "    'swarm_mastery_no_improve_rotations': 6,\n"
            "    'swarm_grind_mastery_successes': 3,\n"
            "}\n"
            "fullgame_env_config = build_env_config(x, y)\n"
        )

    # Names come from cleared-objective log lines; phase 11 is a grind.
    with open(os.path.join(FIXTURE, 'train.log'), 'w', encoding='utf-8') as h:
        h.write(quest(0, 100, 1, 'reach_viridian'))
        h.write(quest(0, 100, 11, 'enter_viridian_forest'))
        h.write(quest(0, 100, 12, 'train_pikachu_level_9'))
        h.write(quest(0, 100, 21, 'reach_route_2'))

    # phase 10: plateau objective, 4 of 6 non-improving clears banked.
    # phase 11: grind, 2 of 3 clears.
    write('restart_start.swarm_mastery.json', {
        'windows': {
            '10': [400, 350, 360, 370, 380, 390],
            '11': [9000, 9500],
            '20': [100, 200],
        },
        'no_improve_counts': {'10': 4},
    })
    write('restart_start.swarm_frontier.json', {'quest_phase': 10})

    env = dict(os.environ, POKEMON_VIEWER_MAX_ENVS='4',
               POKEMON_VIEWER_FEED_HZ='10', POKEMON_VIEWER_PORT=str(PORT))
    proc = subprocess.Popen([sys.executable, 'map_viewer_server.py'], cwd=FIXTURE,
                            env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        payload = None
        for _ in range(40):
            time.sleep(0.5)
            try:
                payload = fetch()
                break
            except Exception:
                if proc.poll() is not None:
                    print('  server died:', proc.stdout.read()[:1500])
                    return 1
        if payload is None:
            return 1
        time.sleep(1.5)
        m = fetch().get('mastery') or {}
        cfg = m.get('config') or {}
        g = m.get('frontier') or {}

        check('thresholds parsed from train.py, not hardcoded',
              cfg.get('no_improve_rotations') == 6 and cfg.get('grind_successes') == 3,
              json.dumps({k: cfg.get(k) for k in
                          ('no_improve_rotations', 'grind_successes', 'window_size')}))
        check('frontier phase read from the frontier meta',
              m.get('frontier_phase') == 10, str(m.get('frontier_phase')))

        # --- plateau mode ---
        check('deterministic objective uses the plateau rule',
              g.get('mode') == 'plateau', str(g.get('mode')))
        check('plateau target is no_improve_rotations',
              g.get('target') == 6, str(g.get('target')))
        check('streak taken from the durable counter, not the window',
              g.get('progress') == 4, str(g.get('progress')))
        check('shortfall is a countable number of clears',
              g.get('short_by') == 2, str(g.get('short_by')))
        check('gate shut while short of the plateau', g.get('allowed') is False)
        check('early budget still reported for context',
              g.get('budget') == 500, str(g.get('budget')))

        # Reaching the target opens it.
        write('restart_start.swarm_mastery.json', {
            'windows': {'10': [400, 350, 360, 370, 380, 390],
                        '11': [9000, 9500], '20': [100, 200]},
            'no_improve_counts': {'10': 6},
        })
        time.sleep(1.5)
        g = (fetch().get('mastery') or {}).get('frontier') or {}
        check('gate opens once the plateau streak is reached',
              g.get('allowed') is True and g.get('short_by') == 0,
              f"progress={g.get('progress')}/{g.get('target')} allowed={g.get('allowed')}")

        # A faster clear restarts the streak -- progress must fall back.
        write('restart_start.swarm_mastery.json', {
            'windows': {'10': [400, 350, 360, 370, 380, 390],
                        '11': [9000, 9500], '20': [100, 200]},
            'no_improve_counts': {'10': 1},
        })
        time.sleep(1.5)
        g = (fetch().get('mastery') or {}).get('frontier') or {}
        check('a new best restarts the streak and shuts the gate',
              g.get('allowed') is False and g.get('progress') == 1,
              f"progress={g.get('progress')} allowed={g.get('allowed')}")

        # --- reliability mode ---
        grind = gate_for(11)
        check('a train_* objective uses the reliability rule',
              grind and grind.get('mode') == 'reliability', str(grind and grind.get('mode')))
        check('reliability target is grind_successes',
              grind.get('target') == 3, str(grind.get('target')))
        check('reliability counts clears regardless of step cost',
              grind.get('progress') == 2 and grind.get('short_by') == 1,
              f"progress={grind.get('progress')} short={grind.get('short_by')}")
        check('a grind far over budget can still be progressing',
              grind.get('under_budget') == 0 and grind.get('allowed') is False,
              f"under_budget={grind.get('under_budget')}")

        write('restart_start.swarm_mastery.json', {
            'windows': {'10': [400], '11': [9000, 9500, 9900], '20': [100, 200]},
            'no_improve_counts': {'10': 1},
        })
        time.sleep(1.5)
        grind = gate_for(11)
        check('grind gate opens on the third clear',
              grind.get('allowed') is True and grind.get('short_by') == 0,
              f"progress={grind.get('progress')}/{grind.get('target')}")

        # Late phases keep the larger budget for their reported context.
        late = gate_for(20)
        check('late phase uses the late budget',
              late and late.get('budget') == 4000, str(late and late.get('budget')))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print()
    if failures:
        print(f'FAILED ({len(failures)}): {failures}')
        return 1
    print('all mastery-gate checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
