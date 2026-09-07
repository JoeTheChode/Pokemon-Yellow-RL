#!/usr/bin/env python3
"""Tests for the Tail Whip -> Slam swap on the lead Pikachu.

Slam replaces the earlier Toxic swap as the single owner of that moveslot --
two helpers both claiming the Tail Whip slot raced, so the lead's fourth move
depended on which ran first.

Also pins the interaction with _protect_lead_moveset: that guard reverts a
change only when a damaging move became non-damaging, so Tail Whip (power 0)
-> Slam (power 80) must survive it. If it did not, the swap would be undone
one step later and the whole thing would be a silent no-op.
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


class FakeEnv:
    """Minimal stand-in: the helper only touches .pyboy.memory and .quest_phase."""

    def __init__(self, level=70, moves=None, party_size=1, species=None):
        self.memory = [0] * 0x10000
        self.pyboy = type('P', (), {'memory': self.memory})()
        self.quest_phase = train.FULLGAME_BEAT_LORELEI_QUEST_PHASE
        moves = moves if moves is not None else [
            train.GEN1_THUNDERBOLT_MOVE_ID, 129,
            train.GEN1_TAIL_WHIP_MOVE_ID, 86,
        ]
        self.memory[train.ADDR_PARTY_SIZE] = party_size
        self.memory[train.PARTY_SPECIES_ADDRS[0]] = (
            species if species is not None else train.GEN1_PIKACHU_SPECIES_ID)
        self.memory[train.ADDR_LEVEL] = level
        for addr, move in zip(train.PARTY_MOVE_ID_ADDRS[0], moves):
            self.memory[addr] = move
        for addr, move in zip(train.PARTY_MOVE_PP_ADDRS[0], moves):
            info = train.GEN1_MOVE_TABLE.get(move) or {}
            self.memory[addr] = min(int(info.get('max_pp', 0)), 0x3F)

    def moves(self):
        return [self.memory[a] for a in train.PARTY_MOVE_ID_ADDRS[0]]

    def pp(self):
        return [self.memory[a] for a in train.PARTY_MOVE_PP_ADDRS[0]]


def main():
    swap = train.PokemonYellowEnv._ensure_pikachu_knows_slam

    # ROM cross-check: the constants must match what the ROM actually says.
    slam = train.GEN1_MOVE_TABLE.get(train.GEN1_SLAM_MOVE_ID) or {}
    whip = train.GEN1_MOVE_TABLE.get(train.GEN1_TAIL_WHIP_MOVE_ID) or {}
    check('ROM agrees Slam is damaging', slam.get('power', 0) == 80, str(slam))
    check('ROM agrees Slam PP constant', slam.get('max_pp') == train.GEN1_SLAM_MAX_PP,
          f"rom={slam.get('max_pp')} const={train.GEN1_SLAM_MAX_PP}")
    check('Tail Whip is non-damaging (the slot is genuinely dead)',
          whip.get('power', 1) == 0, str(whip))

    # Core swap.
    env = FakeEnv()
    before = env.moves()
    check('swap performed', swap(env) is True)
    check('Tail Whip gone, Slam installed',
          train.GEN1_TAIL_WHIP_MOVE_ID not in env.moves()
          and train.GEN1_SLAM_MOVE_ID in env.moves(),
          f'{before} -> {env.moves()}')
    check('Slam landed in the Tail Whip slot only',
          env.moves()[0] == before[0] and env.moves()[1] == before[1]
          and env.moves()[3] == before[3],
          str(env.moves()))
    check('Slam PP set to its maximum',
          env.pp()[2] == train.GEN1_SLAM_MAX_PP, str(env.pp()))

    # Idempotent.
    env2 = FakeEnv()
    swap(env2)
    snapshot = env2.moves()
    check('second call is a no-op', swap(env2) is True and env2.moves() == snapshot)

    # Level gate -- must not grant a move Pikachu has not earned.
    low = FakeEnv(level=train.GEN1_SLAM_LEARN_LEVEL - 1)
    check('below the learn level, no swap',
          swap(low) is False and train.GEN1_TAIL_WHIP_MOVE_ID in low.moves(),
          f'L{train.GEN1_SLAM_LEARN_LEVEL - 1}: {low.moves()}')
    at = FakeEnv(level=train.GEN1_SLAM_LEARN_LEVEL)
    check('at the learn level, swap happens',
          swap(at) is True and train.GEN1_SLAM_MOVE_ID in at.moves())

    # No Tail Whip -> nothing to do.
    none = FakeEnv(moves=[train.GEN1_THUNDERBOLT_MOVE_ID, 129, 21, 86])
    check('no-op when Tail Whip is absent', swap(none) is True)

    # Not Pikachu -> untouched.
    other = FakeEnv(species=180)
    check('non-Pikachu lead untouched',
          swap(other) is False and train.GEN1_TAIL_WHIP_MOVE_ID in other.moves())

    # Phase gate.
    early = FakeEnv()
    early.quest_phase = train.FULLGAME_BEAT_LORELEI_QUEST_PHASE - 1
    check('before the Elite Four phase, no swap',
          swap(early) is False and train.GEN1_TAIL_WHIP_MOVE_ID in early.moves())

    # The recovery guard must not undo the swap on the following step.
    env3 = FakeEnv()
    env3._lead_moveset_snapshot = (tuple(env3.moves()), tuple(env3.pp()))
    env3.lead_moveset_recoveries = 0
    swap(env3)
    train.PokemonYellowEnv._protect_lead_moveset(env3)
    check('recovery guard does not revert Tail Whip -> Slam',
          train.GEN1_SLAM_MOVE_ID in env3.moves()
          and env3.lead_moveset_recoveries == 0,
          f'{env3.moves()}, recoveries={env3.lead_moveset_recoveries}')

    # The old Toxic owner must be gone, or the two race again.
    src = open(os.path.join(TRAIN_DIR, 'train.py'), encoding='utf-8').read()
    check('no second helper still claims the Tail Whip slot',
          src.count('_ensure_pikachu_knows_toxic') == 0
          and src.count('memory[move_addrs[slot]] = GEN1_TOXIC_MOVE_ID') == 0)
    check('slam helper is actually called', '_ensure_pikachu_knows_slam()' in src)

    print()
    if failures:
        print(f'FAILED ({len(failures)}): {failures}')
        return 1
    print('all Pikachu Slam checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
