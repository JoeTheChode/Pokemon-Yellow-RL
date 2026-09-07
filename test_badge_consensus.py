#!/usr/bin/env python3
"""The trainer card must not invent badges from a single bad read.

Bug (2026-08-16): the card OR-ed `badge_flags` across all 96 workers. Workers
that read WRAM mid-transition report a garbage byte, and OR is the worst
possible aggregate for that -- one bad reader permanently lights a badge.

Live case: 62 workers reported 3 (Boulder+Cascade, correct at that point in
the run) while 34 read 57 during a map transition. 57|3 = 59, lighting badges
1,2,4,5,6 -- Rainbow/Soul/Marsh without Thunder, which is impossible because
Vermilion Gym sits behind a Cut tree. Twenty seconds later all 96 read 3.

Consensus fixes it: the majority value wins and a minority of bad readers
cannot outvote the run's real state.
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
FIXTURE = os.path.join(HERE, '_fixture_badges')
PORT = int(os.environ.get('BADGE_PORT', '8761'))

failures = []


def check(label, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def fetch():
    # The trainer card is served with the splits payload, not the agent feed.
    with urllib.request.urlopen(
            f'http://127.0.0.1:{PORT}/quest_splits.json', timeout=5) as r:
        return json.load(r)


def write_agents(flag_by_env):
    for env_id, flags in flag_by_env.items():
        entry = {
            'env_id': env_id, 'user': 'SVER-YV', 'color': '#FFD700',
            'last_position': [3, 4, 6], 'map_id': 6, 'pikachu_level': 5,
            'badge_flags': flags, 'player_name': 'SVER', 'money': 6194,
            'play_time_seconds': 449, 'last_seen': time.time(),
        }
        with open(os.path.join(FIXTURE, f'live_agent_positions.env{env_id}.json'), 'w') as h:
            json.dump(entry, h)


def card():
    for _ in range(12):
        time.sleep(0.6)
        c = fetch().get('player_card')
        if c:
            return c
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
    for stale in ('viewer_heatmap.json', 'train.log'):
        p = os.path.join(FIXTURE, stale)
        if os.path.exists(p):
            os.unlink(p)
    open(os.path.join(FIXTURE, 'train.log'), 'w').close()

    # The exact live split: 8 correct readers, 4 mid-transition readers.
    write_agents({**{i: 3 for i in range(8)}, **{i: 57 for i in range(8, 12)}})

    env = dict(os.environ, POKEMON_VIEWER_MAX_ENVS='16',
               POKEMON_VIEWER_FEED_HZ='10', POKEMON_VIEWER_PORT=str(PORT))
    proc = subprocess.Popen([sys.executable, 'map_viewer_server.py'], cwd=FIXTURE,
                            env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        c = card()
        if c is None:
            out = proc.stdout.read()[:1500] if proc.poll() is not None else ''
            print('  no player card served:', out)
            return 1

        check('majority badge byte wins over mid-transition readers',
              c['badge_flags'] == 3, f"got {c['badge_flags']}")
        check('badge count is the real one, not the OR',
              c['badge_count'] == 2, f"got {c['badge_count']} (OR would give 5)")
        earned = [b['name'] for b in c['badges'] if b['earned']]
        check('only Boulder and Cascade lit', earned == ['Boulder', 'Cascade'], str(earned))
        check('the impossible badge set is gone',
              not any(b['earned'] for b in c['badges']
                      if b['name'] in ('Rainbow', 'Soul', 'Marsh')), str(earned))
        check('disagreement is reported, not hidden',
              c.get('badge_agreement') is not None and c['badge_agreement'] < 1.0,
              str(c.get('badge_agreement')))

        # A real badge gain: the whole swarm moves together.
        write_agents({i: 7 for i in range(12)})
        time.sleep(2.0)
        c = card()
        check('a genuine badge gain is reflected', c['badge_flags'] == 7 and c['badge_count'] == 3,
              f"flags={c['badge_flags']} count={c['badge_count']}")
        check('full agreement reported when the swarm agrees',
              c.get('badge_agreement') == 1.0, str(c.get('badge_agreement')))

        # A single stray reader must not be able to light anything.
        write_agents({**{i: 7 for i in range(11)}, 11: 255})
        time.sleep(2.0)
        c = card()
        check('one stray worker cannot light a badge',
              c['badge_flags'] == 7 and c['badge_count'] == 3,
              f"flags={c['badge_flags']} (OR would give 255)")

        # Exact tie must not silently prefer the larger/garbage value.
        write_agents({**{i: 3 for i in range(6)}, **{i: 57 for i in range(6, 12)}})
        time.sleep(2.0)
        c = card()
        check('a tie resolves to one of the reported values, never their OR',
              c['badge_flags'] in (3, 57), f"got {c['badge_flags']}")
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
    print('all badge consensus checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
