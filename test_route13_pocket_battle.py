#!/usr/bin/env python3
"""The Route 13/14 pocket-escape rule must not steer inside a battle.

2026-08-23 live: 96/96 workers entered one wild battle on Route 14 (9,14) and
burned the whole 2,048-step budget there, `executed_actions a:0, b:2240`.
Coordinates cannot change during a battle, so `same_position_step_count`
climbs every step and this helper's stuck-nudge returned B forever, over the
top of the force-A-in-wild-battles block that runs before it.
"""
import unittest

import train

PHASE = min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES)


class PocketBattleGateTests(unittest.TestCase):
    def call(self, pos, stuck, battle_flag, map_id=25):
        return train.route13_water_pocket_action(
            map_id, pos[0], pos[1], PHASE, stuck, battle_flag=battle_flag,
        )

    def test_wild_battle_silences_the_helper(self):
        for position in ((9, 14), (30, 13)):
            for stuck in (0, 2, 3, 4, 200, 2048):
                self.assertIsNone(self.call(position, stuck, battle_flag=1),
                                  (position, stuck))

    def test_trainer_battle_silences_the_helper(self):
        self.assertIsNone(self.call((30, 13), 4, battle_flag=2))

    def test_overworld_still_escapes_the_pocket(self):
        self.assertEqual(self.call((30, 13), 0, battle_flag=0), 'down')

    def test_overworld_stuck_still_nudges_with_b(self):
        self.assertEqual(self.call((30, 13), 2, battle_flag=0), 'b')

    def test_route13_side_still_works_outside_battle(self):
        self.assertIsNotNone(self.call((10, 51), 0, battle_flag=0, map_id=24))

    def test_route13_side_silent_in_battle(self):
        self.assertIsNone(self.call((10, 51), 4, battle_flag=1, map_id=24))

    def test_default_battle_flag_keeps_old_callers_working(self):
        self.assertEqual(
            train.route13_water_pocket_action(25, 30, 13, PHASE, 0), 'down'
        )

    def test_soul_marsh_grind_sees_real_wild_battles(self):
        # The gate above only helps if the phase reports battles at all.
        self.assertIn(PHASE, train.FULLGAME_SOUL_MARSH_GRIND_PHASES)


if __name__ == '__main__':
    unittest.main()

