#!/usr/bin/env python3
"""Seed the viewer's quest-splits ledger from historical [QUEST] lines.

The ledger accumulates forward from `event_log_offset`, which resumes near
EOF -- so objectives cleared before the ledger existed are invisible to it
even though the log records them. This scans train.log and rebuilds the
ledger from every completion on the *current* quest chain.

Chain versions matter, and matching on the phase *index* throws most of the
history away: the chain grew over months (/29 -> /76 -> /156 -> /215 -> /259),
so an index means a different objective in each era. Filtering to the current
chain length kept only 49 objectives out of the 258 that have actually been
cleared.

Objective *names* are version-independent, so completions are resolved by name
against the current FULLGAME_QUEST_WAYPOINTS and filed under today's index.
Names that no longer exist in the chain are counted and reported rather than
guessed at.

Lines written before `phase_steps=` existed carry no split time; they still
count as clears and set best_run_step. Run with the viewer stopped, then
restart it -- the running viewer rewrites this file every 30s.
"""
import argparse
import glob
import gzip
import json
import os
import re

RX = re.compile(
    rb"\[QUEST\] env_id=(\d+) step=(\d+) (?:phase_steps=(\d+) )?"
    rb"phase=(\d+)/(\d+) completed=([A-Za-z0-9_]+)"
)
SAMPLES = 64
CHUNK = 64 * 1024 * 1024

parser = argparse.ArgumentParser()
parser.add_argument('--log', default='/home/ubuntu/pokemon-rl/train.log')
parser.add_argument('--state', default='/home/ubuntu/pokemon-rl/viewer_heatmap.json')
parser.add_argument('--repo', default='/home/ubuntu/pokemon-rl',
                    help='repo supplying the current quest chain')
parser.add_argument('--apply', action='store_true',
                    help='write the ledger into the state file')
args = parser.parse_args()

import sys  # noqa: E402  (path set from --repo below)
sys.path.insert(0, args.repo)
os.chdir(args.repo)
import train  # noqa: E402

INDEX_BY_NAME = {}
for index, waypoint in enumerate(train.FULLGAME_QUEST_WAYPOINTS):
    name = waypoint.get('name')
    if name:
        INDEX_BY_NAME.setdefault(str(name), index)
CHAIN_LENGTH = len(train.FULLGAME_QUEST_WAYPOINTS)
print(f'current chain: {CHAIN_LENGTH} objectives')

def log_sources(path):
    """Oldest-first list of log files, including rotated/compressed ones.

    logrotate moves history into train.log.1.gz, so reading only the live file
    silently loses everything before the last rotation -- which is most of the
    run. Ordered oldest first so later clears overwrite earlier ones.
    """
    rotated = sorted(
        glob.glob(path + '.*'),
        key=lambda p: int(re.search(r'\.(\d+)', os.path.basename(p)).group(1))
        if re.search(r'\.(\d+)', os.path.basename(p)) else 0,
        reverse=True,
    )
    return [p for p in rotated if os.path.exists(p)] + [path]


def open_log(path):
    return gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb')


splits = {}
unmatched = {}
sources = log_sources(args.log)
print('log sources (oldest first):')
for p in sources:
    print(f'  {os.path.basename(p)}  {os.path.getsize(p) / 1e6:.1f} MB'
          + ('  [compressed]' if p.endswith('.gz') else ''))
size = sum(os.path.getsize(p) for p in sources)
scanned = 0
for source in sources:
  tail = b''
  with open_log(source) as handle:
    while True:
        chunk = handle.read(CHUNK)
        if not chunk:
            break
        scanned += len(chunk)
        buf = tail + chunk
        for m in RX.finditer(buf):
            name = m.group(6).decode()
            phase = INDEX_BY_NAME.get(name)
            if phase is None:
                # Objective renamed or removed since; counted, never guessed.
                unmatched[name] = unmatched.get(name, 0) + 1
                continue
            run_step = int(m.group(2))
            phase_steps = int(m.group(3)) if m.group(3) is not None else None
            entry = splits.setdefault(phase, {
                'name': '', 'clears': 0, 'best': None, 'last': None,
                'samples': [], 'best_run_step': None,
            })
            entry['name'] = name
            entry['clears'] += 1
            if entry['best_run_step'] is None or run_step < entry['best_run_step']:
                entry['best_run_step'] = run_step
            if phase_steps is not None:
                entry['last'] = phase_steps
                if entry['best'] is None or phase_steps < entry['best']:
                    entry['best'] = phase_steps
                entry['samples'].append(phase_steps)
                if len(entry['samples']) > SAMPLES:
                    del entry['samples'][:-SAMPLES]
        # Keep a tail so a record split across chunks is not lost.
        tail = buf[-512:]

timed = [e for e in splits.values() if e['best'] is not None]
print(f'scanned {scanned/1e6:.1f} MB of {size/1e6:.1f} MB')
print(f'objectives matched into the current chain: {len(splits)} / {CHAIN_LENGTH}')
never = [
    (i, w.get('name')) for i, w in enumerate(train.FULLGAME_QUEST_WAYPOINTS)
    if i not in splits
]
print(f'  never cleared in this log: {len(never)}'
      + (f'  (first: {never[0][0]} {never[0][1]})' if never else ''))
if unmatched:
    top = sorted(unmatched.items(), key=lambda kv: -kv[1])[:5]
    print(f'  log names not in the current chain: {len(unmatched)} '
          f'-- {", ".join(f"{n}×{c}" for n, c in top)}')
print(f'  with split times: {len(timed)}   (older lines carry no phase_steps)')
print(f'  total clears: {sum(e["clears"] for e in splits.values())}')
for phase in sorted(splits)[:12]:
    e = splits[phase]
    print(f'    {phase:>3} {e["name"]:<26} clears={e["clears"]:>4} '
          f'best={e["best"]} best_run_step={e["best_run_step"]}')
if len(splits) > 12:
    print(f'    … and {len(splits) - 12} more')

if not args.apply:
    print('\ndry run — pass --apply to write')
    raise SystemExit(0)

with open(args.state, encoding='utf-8') as handle:
    state = json.load(handle)

# Merge, never replace. The live ledger carries split *times* reconciled from
# the trainer's mastery windows, and the log no longer emits phase_steps -- a
# wholesale write would trade real timings for None and silently lose them.
existing = state.get('quest_splits') or {}
merged = dict(existing)
for phase, entry in splits.items():
    key = str(phase)
    prior = merged.get(key)
    if not isinstance(prior, dict):
        merged[key] = entry
        continue
    combined = dict(prior)
    combined['name'] = entry['name'] or prior.get('name') or ''
    combined['clears'] = max(int(prior.get('clears') or 0), entry['clears'])
    for field in ('best', 'best_run_step'):
        a, b = prior.get(field), entry.get(field)
        candidates = [v for v in (a, b) if isinstance(v, (int, float)) and v >= 0]
        combined[field] = min(candidates) if candidates else None
    if not prior.get('samples') and entry['samples']:
        combined['samples'] = entry['samples']
    merged[key] = combined
state['quest_splits'] = merged
state["quest_total_phases"] = max(CHAIN_LENGTH, int(state.get("quest_total_phases") or 0))
tmp = args.state + '.tmp'
with open(tmp, 'w', encoding='utf-8') as handle:
    json.dump(state, handle, separators=(',', ':'))
os.replace(tmp, args.state)
print(f'\nwrote {len(splits)} objectives into {args.state}')
