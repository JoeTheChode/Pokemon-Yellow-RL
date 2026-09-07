#!/usr/bin/env python3
"""Tests for the mastery ladder: env 0-23 climb their own rung chain.

Behaviour being pinned:
  - The cohort is a contiguous block (env 0..N-1), and it *replaces* the
    modulo drill slice rather than overlapping it.
  - A rung is drilled until it stops improving: a new best resets patience, a
    non-improving clear spends it, and the rung advances only after N
    consecutive clears set no new best.
  - Advancing resets best/patience so the next rung is judged on its own.
  - Ladder workers never inherit the shared frontier (that would drop them at
    the swarm's current objective, the opposite of drilling the opening).
"""
import os
import sys

REPO = os.environ.get('REPO_DIR', '/home/ubuntu/pokemon-rl')
TRAIN_DIR = os.environ.get('TRAIN_DIR', REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, TRAIN_DIR)
os.chdir(REPO)

import train  # noqa: E402

assert os.path.dirname(os.path.abspath(train.__file__)) == os.path.abspath(TRAIN_DIR), (
    f'imported the wrong train.py: {train.__file__}')

failures = []


def check(label, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def drill_sequence(times, limit=5, start=None):
    """Feed clear times through the ladder, returning (meta, advance count)."""
    meta = start or train.ladder_default_meta()
    advances = 0
    for t in times:
        meta, adv = train.ladder_update_after_clear(meta, t, limit)
        advances += 1 if adv else 0
    return meta, advances


def main():
    # --- cohort selection ---
    check('env 0 is on the ladder', train.swarm_worker_is_ladder(0, 24))
    check('env 23 is on the ladder', train.swarm_worker_is_ladder(23, 24))
    check('env 24 is not on the ladder', not train.swarm_worker_is_ladder(24, 24))
    check('env 95 is not on the ladder', not train.swarm_worker_is_ladder(95, 24))
    check('ladder disabled by size 0', not train.swarm_worker_is_ladder(0, 0))
    on_ladder = [e for e in range(96) if train.swarm_worker_is_ladder(e, 24)]
    check('exactly 24 ladder workers', len(on_ladder) == 24, str(len(on_ladder)))
    # The old modulo cohort was 0,4,8..92 -- the ladder must not be that set.
    modulo = [e for e in range(96) if train.swarm_worker_is_drill(e, 0.25)]
    check('ladder is contiguous, not the modulo slice',
          on_ladder != modulo and on_ladder == list(range(24)),
          f'ladder head {on_ladder[:5]} vs modulo head {modulo[:5]}')

    # --- patience / advancement ---
    meta, adv = drill_sequence([500], limit=5)
    check('first clear sets the best', meta['best'] == 500 and adv == 0, str(meta))

    meta, adv = drill_sequence([500, 400, 300], limit=5)
    check('improving clears keep resetting patience',
          meta['best'] == 300 and meta['no_improve'] == 0 and adv == 0, str(meta))

    # 1 improving + 4 non-improving = patience 4 of 5, no advance yet.
    meta, adv = drill_sequence([300, 400, 400, 400, 400], limit=5)
    check('patience spent but not exhausted does not advance',
          adv == 0 and meta['rung'] == 0 and meta['no_improve'] == 4, str(meta))

    meta, adv = drill_sequence([300, 400, 400, 400, 400, 400], limit=5)
    check('advances after N consecutive non-improving clears',
          adv == 1 and meta['rung'] == 1, str(meta))
    check('advancing resets best and patience for the new rung',
          meta['best'] is None and meta['no_improve'] == 0, str(meta))

    # A new best midway must restart the count, not merely pause it.
    meta, adv = drill_sequence([300, 400, 400, 250, 400, 400, 400], limit=5)
    check('a new best mid-window restarts patience (no advance)',
          adv == 0 and meta['rung'] == 0 and meta['best'] == 250, str(meta))

    # Equal-to-best is not an improvement -- otherwise a plateau never ends.
    meta, adv = drill_sequence([300, 300, 300, 300, 300, 300], limit=5)
    check('equalling the best counts as no improvement',
          adv == 1 and meta['rung'] == 1, str(meta))

    # Patience of 1 advances on the first non-improving clear.
    meta, adv = drill_sequence([300, 301], limit=1)
    check('limit=1 advances immediately on a non-improving clear',
          adv == 1 and meta['rung'] == 1, str(meta))

    # Clears accumulate across rungs.
    meta, _ = drill_sequence([300, 400, 400, 400, 400, 400], limit=5)
    check('clears counted across the rung change', meta['clears'] == 6, str(meta))

    # --- rung state paths ---
    base = '/tmp/example.state'
    check('rung 0 uses the game start state',
          str(train.ladder_rung_state_path(base, 0)) == base)
    p1 = str(train.ladder_rung_state_path(base, 1))
    check('rung 1 has its own state file', p1 != base and 'ladder_rung_1' in p1, p1)
    check('rung paths are distinct per rung',
          train.ladder_rung_state_path(base, 1) != train.ladder_rung_state_path(base, 2))

    # --- integration points in the source ---
    src = open(os.path.join(TRAIN_DIR, 'train.py'), encoding='utf-8').read()
    check('ladder workers bypass the frontier load',
          'if self.is_swarm_ladder_worker:' in src
          and src.index('if self.is_swarm_ladder_worker:')
          < src.index('elif swarm_worker_should_load_frontier('),
          'ladder branch must precede the frontier branch')
    check('ladder replaces the modulo drill cohort when enabled',
          'if self.swarm_ladder_size > 0' in src)
    check('clearing a rung records against the ladder',
          'record_ladder_clear(' in src)
    check('the next rung state is captured on advance',
          'save_state(handle)' in src and 'ladder_advanced' in src)
    check('ladder is wired into the milestone config',
          "'swarm_ladder_size': 24" in src)

    print()
    if failures:
        print(f'FAILED ({len(failures)}): {failures}')
        return 1
    print('all mastery-ladder checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
