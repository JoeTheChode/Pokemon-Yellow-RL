#!/usr/bin/env python3
"""Torn party reads must not clear min_level waypoints.

2026-08-23: the live Route 14 (9,12) wedge published a party block striped
with 0x99 -- level 153, HP 39198/39198 in every slot -- and the level ladder
`train_pikachu_soul_marsh_level_55..58` cleared four phases on consecutive
steps for 22,000 reward while the real lead was a level-54 Pikachu.
"""
import unittest

import train


class FakeMemory(dict):
    """Byte-addressable stand-in; unset addresses read 0 like fresh WRAM."""

    def __getitem__(self, address):
        if isinstance(address, slice):
            return bytes(self.get(a, 0) for a in range(address.start, address.stop))
        return self.get(address, 0)


def healthy_memory(level=54, party_size=5):
    mem = FakeMemory()
    mem[train.ADDR_PARTY_SIZE] = party_size
    for index in range(party_size):
        mem[train.ADDR_LEVEL + index * train.PARTY_DATA_STRIDE] = level if index == 0 else 10
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[index]
        max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[index]
        mem[cur_hi], mem[cur_lo] = 0, 131 if index == 0 else 30
        mem[max_hi], mem[max_lo] = 0, 131 if index == 0 else 30
    return mem


def striped_memory():
    """The live corruption: 0x99 across the party block."""
    mem = healthy_memory()
    for address in range(train.ADDR_PARTY_SIZE, 0xD26B):
        mem[address] = 0x99
    mem[train.ADDR_PARTY_SIZE] = 5
    return mem


class PartyReadPlausibilityTests(unittest.TestCase):
    def test_healthy_party_is_plausible(self):
        self.assertTrue(train.party_read_is_plausible(healthy_memory()))

    def test_striped_party_is_not_plausible(self):
        mem = striped_memory()
        self.assertEqual(int(mem[train.ADDR_LEVEL]), 153)
        self.assertFalse(train.party_read_is_plausible(mem))

    def test_level_above_hundred_is_not_plausible(self):
        mem = healthy_memory(level=153)
        self.assertFalse(train.party_read_is_plausible(mem))

    def test_level_one_hundred_is_plausible(self):
        self.assertTrue(train.party_read_is_plausible(healthy_memory(level=100)))

    def test_empty_party_is_not_plausible(self):
        self.assertFalse(train.party_read_is_plausible(healthy_memory(party_size=0)))

    def test_current_hp_above_max_is_not_plausible(self):
        mem = healthy_memory()
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
        mem[cur_hi], mem[cur_lo] = 1, 0  # 256 > 131
        self.assertFalse(train.party_read_is_plausible(mem))

    def test_unreadable_memory_does_not_veto(self):
        # "Cannot read" is not evidence of corruption. A guard that fails
        # closed on a partial view blocks callers that never had a party
        # block to show -- the failure mode that made whole waypoints
        # permanently unclaimable once already.
        self.assertTrue(train.party_read_is_plausible(None))
        self.assertTrue(train.party_read_is_plausible({}))

    def test_partial_view_still_clears_a_met_level_gate(self):
        # The exact fixture test_swarm_drill uses: three flag bytes, no party.
        memory = {
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_FONT_LOADED: 0,
            train.ADDR_TEXT_BOX: 0,
        }
        self.assertTrue(train.quest_waypoint_matches(
            {'name': 'level_6', 'min_level': 6}, 12, 10, 10, (),
            level=6, memory=memory, battle_active=False,
        ))


class MinLevelGateTests(unittest.TestCase):
    WAYPOINT = {'name': 'train_pikachu_soul_marsh_level_55', 'min_level': 55}

    def matches(self, memory, level):
        return train.quest_waypoint_matches(
            self.WAYPOINT, 25, 9, 12, set(),
            level=level, memory=memory, battle_active=False,
        )

    def test_striped_read_does_not_clear_the_level_gate(self):
        # 153 >= 55 on the raw compare; only the party check stops it.
        self.assertFalse(self.matches(striped_memory(), 153))

    def test_real_level_still_clears(self):
        self.assertTrue(self.matches(healthy_memory(level=55), 55))

    def test_real_level_below_target_still_blocked(self):
        self.assertFalse(self.matches(healthy_memory(level=54), 54))

    def test_non_level_waypoint_is_unaffected_by_a_torn_read(self):
        # Only level-gated waypoints consult the party block.
        self.assertTrue(train.quest_waypoint_matches(
            {'name': 'reach_route_14', 'map': 25}, 25, 9, 12, set(),
            memory=striped_memory(), battle_active=False,
        ))


if __name__ == '__main__':
    unittest.main()
