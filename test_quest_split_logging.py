#!/usr/bin/env python3
"""Guard: the [QUEST] line must carry phase_steps, in the viewer's format.

This exists because the emit was silently lost. It was added at 14:11 on
2026-08-13, overwritten by an unrelated train.py redeploy at 16:04, and the
only visible symptom was that the splits ledger kept counting clears while
every split time stayed empty -- progression with no logging.

Three things are asserted, because each has its own failure mode:
  1. phase_steps is emitted at all.
  2. It is captured BEFORE _advance_quest_phase, which zeroes the counter --
     capture it after and it always prints 0, which looks like a working
     feature reporting instant clears.
  3. The field order matches the viewer's QUEST_LINE_RE, so the parser keeps
     matching.
"""
import os
import re
import sys

REPO = os.environ.get('REPO_DIR', '/home/ubuntu/pokemon-rl')
TRAIN_DIR = os.environ.get('TRAIN_DIR', REPO)
TRAIN_PY = os.path.join(TRAIN_DIR, 'train.py')
VIEWER_PY = os.path.join(REPO, 'map_viewer_server.py')

failures = []


def check(label, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main():
    src = open(TRAIN_PY, encoding='utf-8').read()

    check('train.py emits phase_steps on the [QUEST] line',
          'phase_steps={' in src, 'the splits ledger has no times without it')

    # The emit block, from the [QUEST] f-string to its flush.
    m = re.search(r'f"  \[QUEST\] env_id=.*?flush=True', src, re.S)
    check('[QUEST] emit block found', m is not None)
    if not m:
        print('\nFAILED')
        return 1
    emit = m.group(0)
    check('phase_steps is inside the [QUEST] emit', 'phase_steps=' in emit, emit[:120])

    # Capture must precede the advance, or the counter is already zeroed.
    cap = src.find('completed_phase_steps = int(self.quest_phase_step_count)')
    adv = src.find('self._advance_quest_phase(')
    check('per-phase counter captured before _advance_quest_phase',
          cap != -1 and adv != -1 and cap < adv,
          f'capture@{cap} advance@{adv}')

    # The viewer's regex must still match what the trainer prints.
    viewer = open(VIEWER_PY, encoding='utf-8').read()
    vm = re.search(r'QUEST_LINE_RE = re\.compile\(\s*(.*?)\)\n', viewer, re.S)
    check('viewer defines QUEST_LINE_RE', vm is not None)
    if vm:
        pattern = ''.join(re.findall(r'r"([^"]*)"', vm.group(1)))
        sample = ('  [QUEST] env_id=7 step=402 phase_steps=118 phase=10/259 '
                  'completed=enter_forest_gate next=x map=50 y=43 x=3 drill')
        got = re.compile(pattern).search(sample)
        check('viewer regex matches a real trainer line', got is not None, sample[:70])
        if got:
            check('regex captures the split time', got.group(3) == '118',
                  f'captured {got.group(3)!r}')
            check('regex captures the completed objective',
                  got.group(6) == 'enter_forest_gate', got.group(6))
        # And the pre-fix format must still parse, so history is not lost.
        legacy = ('  [QUEST] env_id=7 step=402 phase=10/259 '
                  'completed=enter_forest_gate next=x map=50 y=43 x=3')
        lg = re.compile(pattern).search(legacy)
        check('regex still parses lines written without phase_steps',
              lg is not None and lg.group(3) is None)

    print()
    if failures:
        print(f'FAILED ({len(failures)}): {failures}')
        return 1
    print('all quest split-logging checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
