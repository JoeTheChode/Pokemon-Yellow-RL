#!/usr/bin/env python3
"""Fixture test for the quest-splits ledger.

Splits answer "how fast has each objective ever been cleared, and how does
that compare to where we are now". The parsing has three traps, all pinned
here:

  1. `phase=` in a [QUEST] line is the phase moved *to*, so the objective just
     finished is phase-1. Off by one and every split is filed against the
     wrong objective.
  2. `phase_steps=` is optional -- lines written before it existed must still
     parse and simply carry no split time, or historical log content breaks
     the parser.
  3. Two records can share one physical line (96 workers, one stdout), so
     finditer must pick up both.

Also checks the ledger survives a restart: the log offset resumes near EOF,
so anything not persisted is gone for good.
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
FIXTURE = os.path.join(HERE, '_fixture_splits')
PORT = int(os.environ.get('SPLITS_PORT', '8742'))

failures = []


def check(label, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def fetch(path='/quest_splits.json'):
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}{path}', timeout=5) as r:
        return json.load(r)


def append(text):
    with open(os.path.join(FIXTURE, 'train.log'), 'a', encoding='utf-8') as h:
        h.write(text)


def quest(env, step, phase_steps, phase, name, total=259):
    ps = f'phase_steps={phase_steps} ' if phase_steps is not None else ''
    return (f'  [QUEST] env_id={env} step={step} {ps}'
            f'phase={phase}/{total} completed={name} next=x map=1 y=2 x=3\n')


def boot():
    env = dict(os.environ, POKEMON_VIEWER_MAX_ENVS='4',
               POKEMON_VIEWER_FEED_HZ='10', POKEMON_VIEWER_PORT=str(PORT),
               POKEMON_SWARM_MASTERY_PATH=os.path.join(
                   FIXTURE, 'restart_start.swarm_mastery.json'))
    proc = subprocess.Popen([sys.executable, 'map_viewer_server.py'], cwd=FIXTURE,
                            env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for _ in range(40):
        time.sleep(0.5)
        try:
            return proc, fetch()
        except Exception:
            if proc.poll() is not None:
                print('  server died:', proc.stdout.read()[:1500])
                return proc, None
    return proc, None


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    os.makedirs(os.path.join(FIXTURE, 'navigation_data'), exist_ok=True)
    for name in ('map_viewer_server.py', 'pokemon_yellow_map_viewer.html'):
        src = os.path.join(SRC, name)
        if os.path.exists(src):
            with open(src, 'rb') as a, open(os.path.join(FIXTURE, name), 'wb') as b:
                b.write(a.read())
    with open(os.path.join(FIXTURE, 'navigation_data', 'yellow_event_flags.json'), 'w') as h:
        json.dump({'num_events': 2560, 'events': [{'bit': 0, 'name': 'EVENT_X'}]}, h)
    for stale in ('viewer_heatmap.json', 'train.log',
                  'restart_start.swarm_mastery.json'):
        p = os.path.join(FIXTURE, stale)
        if os.path.exists(p):
            os.unlink(p)
    open(os.path.join(FIXTURE, 'train.log'), 'w').close()

    proc, payload = boot()
    if payload is None:
        return 1
    try:
        # Two clears of phase 0 (fast then slow) and one of phase 1.
        append(quest(0, 120, 120, 1, 'reach_viridian'))
        append(quest(1, 400, 380, 1, 'reach_viridian'))
        append(quest(2, 700, 210, 2, 'enter_viridian_mart'))
        time.sleep(1.5)
        p = fetch()
        rows = {r['phase']: r for r in p['rows']}

        check('completed objective filed against phase-1, not phase',
              set(rows) == {0, 1}, f'phases seen: {sorted(rows)}')
        check('objective name recorded',
              rows.get(0, {}).get('name') == 'reach_viridian',
              str(rows.get(0, {}).get('name')))
        check('best split is the fastest clear, not the latest',
              rows[0]['best'] == 120 and rows[0]['last'] == 380,
              f"best={rows[0]['best']} last={rows[0]['last']}")
        check('clears counted', rows[0]['clears'] == 2, str(rows[0]['clears']))
        check('total_phases read from the line', p['total_phases'] == 259,
              str(p['total_phases']))
        check('sum_of_bests adds the per-objective bests',
              p['sum_of_bests'] == 120 + 210,
              f"{p['sum_of_bests']} vs {120 + 210}")
        check('timed_phases counts only objectives with a split',
              p['timed_phases'] == 2, str(p['timed_phases']))

        # A faster clear must lower best; a slower one must not raise it.
        append(quest(3, 90, 90, 1, 'reach_viridian'))
        append(quest(3, 900, 900, 1, 'reach_viridian'))
        time.sleep(1.5)
        rows = {r['phase']: r for r in fetch()['rows']}
        check('a faster clear lowers best', rows[0]['best'] == 90, str(rows[0]['best']))
        check('a slower clear does not raise best',
              rows[0]['best'] == 90 and rows[0]['last'] == 900,
              f"best={rows[0]['best']} last={rows[0]['last']}")

        # Legacy line with no phase_steps: must parse, must not invent a time.
        append('  [QUEST] env_id=1 step=50 phase=3/259 completed=return_to_pallet'
               ' next=x map=1 y=2 x=3\n')
        time.sleep(1.5)
        p = fetch()
        rows = {r['phase']: r for r in p['rows']}
        check('legacy line without phase_steps still parses', 2 in rows)
        check('legacy line contributes no split time',
              rows.get(2, {}).get('best') is None
              and rows.get(2, {}).get('clears') == 1,
              str(rows.get(2)))
        check('untimed objective excluded from sum_of_bests',
              p['sum_of_bests'] == 90 + 210 and p['timed_phases'] == 2,
              f"sum={p['sum_of_bests']} timed={p['timed_phases']}")

        # The trainer's mastery ledger is durable even when the viewer's log
        # cursor skipped the original timed [QUEST] records. It must fill
        # blank completed rows without overwriting a newer live `last`.
        mastery = os.path.join(FIXTURE, 'restart_start.swarm_mastery.json')
        with open(mastery, 'w', encoding='utf-8') as h:
            json.dump({
                'schema': 2,
                'windows': {'2': [700, 640], '3': [86, 120]},
                'best_steps': {'2': 600, '3': 86},
            }, h)
        time.sleep(1.5)
        p = fetch()
        rows = {r['phase']: r for r in p['rows']}
        check('mastery ledger backfills a blank legacy split',
              rows[2]['last'] == 640 and rows[2]['best'] == 600,
              str(rows[2]))
        check('mastery ledger creates missing completed split rows',
              rows[3]['last'] == 120 and rows[3]['best'] == 86,
              str(rows[3]))
        check('mastery backfill contributes to timed totals',
              p['timed_phases'] == 4,
              str(p['timed_phases']))

        # Two records interleaved onto one physical line.
        append(quest(0, 10, 10, 5, 'a').rstrip('\n') + quest(1, 20, 20, 6, 'b'))
        time.sleep(1.5)
        rows = {r['phase']: r for r in fetch()['rows']}
        check('both records on an interleaved line are captured',
              4 in rows and 5 in rows, f'phases: {sorted(rows)}')

        # Force a state save so the restart check is meaningful.
        state = os.path.join(FIXTURE, 'viewer_heatmap.json')
        saved = False
        for _ in range(45):
            time.sleep(1.0)
            if os.path.exists(state):
                try:
                    if json.load(open(state)).get('quest_splits'):
                        saved = True
                        break
                except (OSError, ValueError):
                    continue
        check('splits written to the state file', saved)
    finally:
        stop(proc)

    proc, payload = boot()
    if payload is None:
        return 1
    try:
        time.sleep(1.0)
        p = fetch()
        rows = {r['phase']: r for r in p['rows']}
        check('splits survive a viewer restart',
              rows.get(0, {}).get('best') == 90 and rows.get(0, {}).get('clears') == 4,
              str(rows.get(0)))
    finally:
        stop(proc)

    print()
    if failures:
        print(f'FAILED ({len(failures)}): {failures}')
        return 1
    print('all quest-splits checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
