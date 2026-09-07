#!/usr/bin/env python3
"""Regression test: per-episode gates must actually reset each episode.

The bug (2026-08-13): 19 attributes latch to make something happen "once" and
are read back through a defaulted getattr, but nothing ever cleared them and
reset() did not touch them. "Once per fight" and "once per run" were really
once per *process*: the first episode an env played consumed the gate and
every later episode silently skipped the behaviour -- boss heals, the Routes
16/17/18 bicycle, and the Silph 5F pad routing.

The last check re-derives the sticky-gate set from train.py itself, so adding
a new latched gate without listing it fails the suite. That is the check that
stops this recurring -- the original bug was purely "new marker, forgot it".

Run against a pre-fix train.py and checks 1, 2, 4, 5 and 6 fail.
"""
import os
import re
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


class Fake:
    """Bare stand-in -- the reset helper only assigns attributes on self."""


def function_body(src, header):
    i = src.index(header)
    rest = src[i + len(header):]
    m = re.search(r'\n    def ', rest)
    return rest[:m.start()] if m else rest


def main():
    src = open(os.path.join(TRAIN_DIR, 'train.py'), encoding='utf-8').read()

    has = hasattr(train.PokemonYellowEnv, '_reset_per_episode_gates')
    check('env exposes _reset_per_episode_gates', has)
    if not has:
        print('\nFAILED (pre-fix build)')
        return 1

    reset_body = function_body(src, '    def reset(self, seed=None, options=None):')
    init_body = function_body(src, '    def __init__(self, config=None):')
    check('reset() calls _reset_per_episode_gates',
          '_reset_per_episode_gates()' in reset_body)
    check('__init__ does not call it in place of reset()',
          '_reset_per_episode_gates()' not in init_body
          or '_reset_per_episode_gates()' in reset_body)

    groups = (train.BATTLE_ASSIST_ONE_SHOT_FLAGS
              + train.PER_EPISODE_PROGRESSION_GATES
              + train.BATTLE_ASSIST_HEAL_BUDGETS)
    obj = Fake()
    for marker in train.BATTLE_ASSIST_ONE_SHOT_FLAGS + train.PER_EPISODE_PROGRESSION_GATES:
        setattr(obj, marker, True)
    for marker in train.BATTLE_ASSIST_HEAL_BUDGETS:
        setattr(obj, marker, 999)
    train.PokemonYellowEnv._reset_per_episode_gates(obj)

    stuck = [m for m in train.BATTLE_ASSIST_ONE_SHOT_FLAGS
             + train.PER_EPISODE_PROGRESSION_GATES if getattr(obj, m) is not False]
    spent = [m for m in train.BATTLE_ASSIST_HEAL_BUDGETS if getattr(obj, m) != 0]
    check('all latched flags cleared', not stuck, str(stuck))
    check('all heal budgets zeroed', not spent, str(spent))

    # The specific behaviours a speedrun loses when these stay latched.
    check('bicycle can be remounted on the next run',
          obj._bicycle_selected_for_part11 is False)
    check('champion heal budget restored for the next run',
          obj._champion_emergency_heals_used == 0)
    check('silph 5F pad routing starts fresh',
          obj._silph_5f_pad_warp_seen is False)

    # Meta-check: re-derive sticky gates from source and require coverage.
    gates = set(re.findall(r"getattr\(\s*self,\s*'(_[a-z0-9_]+)'", src))
    latched = set(re.findall(r"self\.(_[a-z0-9_]+)\s*=\s*True", src))
    latched |= set(re.findall(r"self\.(_[a-z0-9_]+)\s*=\s*(?:used|count|n)\s*\+\s*1", src))
    sticky = gates & latched
    covered = set(groups) | set(train.ONCE_PER_PROCESS_LOG_GATES)
    uncovered = sorted(sticky - covered)
    check('every latched gate is reset or explicitly allow-listed',
          not uncovered, f'uncovered: {uncovered}')

    # Heal budgets reached via a variable marker name (getattr(self, marker, 0))
    # are invisible to the regex above, so assert them by name too.
    for marker in ('_blaine_emergency_heals_used', '_earth_emergency_heals_used'):
        check(f'{marker} is covered', marker in covered)

    print()
    if failures:
        print(f'FAILED ({len(failures)}): {failures}')
        return 1
    print('all per-episode gate checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
