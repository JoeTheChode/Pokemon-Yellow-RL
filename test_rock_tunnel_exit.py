#!/usr/bin/env python3
import unittest
from pathlib import Path

import train


def first_rule(rules, map_id, y, x):
    for rule in rules:
        if rule["map"] == map_id and tuple(rule["target"]) == (y, x):
            return rule
    return None


class RockTunnelExitTests(unittest.TestCase):
    def test_jrf_vision_weaves_west(self):
        safe = first_rule(train.ROCK_TUNNEL_EXIT_ACTION_GUIDANCE, 82, 21, 32)
        self.assertIsNotNone(safe)
        self.assertEqual(int(safe["action"]), 1)

    def test_exit_goes_around_jrf3_instead_of_walking_onto_her(self):
        """Live pile (23,32) walked DOWN onto JrF at pret (32,24)."""
        if not hasattr(train, "ROCK_TUNNEL_EXIT_1F_ACTION_GUIDANCE"):
            self.skipTest("EXIT 1F flow lives on the remote trainer")
        pile = first_rule(train.ROCK_TUNNEL_EXIT_ACTION_GUIDANCE, 82, 23, 32)
        self.assertIsNotNone(pile)
        self.assertEqual(int(pile["action"]), 3)  # RIGHT around the sprite
        self.assertIsNone(first_rule(train.ROCK_TUNNEL_EXIT_ACTION_GUIDANCE, 82, 24, 32))
        self.assertIsNone(first_rule(train.ROCK_TUNNEL_EXIT_ACTION_GUIDANCE, 82, 21, 33))
        stuck = first_rule(train.ROCK_TUNNEL_EXIT_ACTION_GUIDANCE, 82, 27, 23)
        self.assertIsNotNone(stuck)
        self.assertEqual(int(stuck["action"]), 2)  # LEFT toward the south warp, not DOWN

    def test_exit_1f_never_steps_onto_jrf_cones_or_sprites(self):
        if not hasattr(train, "ROCK_TUNNEL_EXIT_1F_ACTION_GUIDANCE"):
            self.skipTest("EXIT 1F flow lives on the remote trainer")
        vision = (
            {(21, 37 - i) for i in range(1, 5)}
            | {(24, 32 + i) for i in range(1, 5)}
            | {(21, 37), (24, 22), (24, 32)}
        )
        delta = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for rule in train.ROCK_TUNNEL_EXIT_1F_ACTION_GUIDANCE:
            tile = tuple(rule["target"])
            dy, dx = delta[int(rule["action"])]
            nxt = (tile[0] + dy, tile[1] + dx)
            self.assertNotIn(tile, vision, msg="owns %s" % (tile,))
            self.assertNotIn(nxt, vision, msg="%s -> %s" % (tile, nxt))

    def test_exit_phase_has_trainer_a(self):
        src = Path(train.__file__).read_text(encoding="utf-8")
        block = src[src.index("'trainer_battle_quest_phases': sorted(") :]
        block = block[: block.index("'trainer_battle_entry_bonus'")]
        self.assertTrue(
            "FULLGAME_ROCK_TUNNEL_INTERIOR_PHASES" in block
            or "FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE + 1" in block,
            block[:500],
        )
        maps = src[src.rindex("'trainer_battle_maps': sorted(") :]
        maps = maps[: maps.index("'trainer_battle_quest_phases'")]
        self.assertIn("82, 157, 232", maps)
        self.assertIn("'force_a_on_zero_enemy_hp_text': True", src)


if __name__ == "__main__":
    unittest.main()
