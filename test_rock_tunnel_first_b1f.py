#!/usr/bin/env python3
import unittest
import collections
from pathlib import Path
from unittest import mock

import train


def first_rule(rules, map_id, y, x):
    for rule in rules:
        if rule["map"] == map_id and tuple(rule["target"]) == (y, x):
            return rule
    return None


class RockTunnelFirstB1FTests(unittest.TestCase):
    def make_trainer_env(self, status=4):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_FIRST_B1F_QUEST_PHASE
        env._rock_tunnel_first_1f_emergency_heals_used = 0
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 82,
            train.ADDR_POS_A: 8,
            train.ADDR_POS_B: 22,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 48,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 107,
            train.PARTY_CUR_HP_ADDRS[0][1]: 48,
            train.ADDR_ACTIVE_MON_STATUS: status,
            train.PARTY_STATUS_ADDRS[0]: status,
        })
        env.pyboy = mock.Mock(memory=memory)
        return env, memory

    def test_phase(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(
            names[train.FULLGAME_ROCK_TUNNEL_FIRST_B1F_QUEST_PHASE],
            "reach_rock_tunnel_first_b1f",
        )

    def test_entrance_walks_off_the_exit_mat(self):
        mat = first_rule(train.ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE, 82, 3, 15)
        self.assertEqual(int(mat["action"]), 3)
        west = first_rule(train.ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE, 82, 3, 14)
        self.assertEqual(int(west["action"]), 1)

    def test_spawn_and_south_pocket_go_to_ladder(self):
        spawn = first_rule(train.ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE, 82, 2, 14)
        self.assertEqual(int(spawn["action"]), 3)
        pile = first_rule(train.ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE, 82, 8, 20)
        self.assertEqual(int(pile["action"]), 3)
        hall = first_rule(train.ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE, 82, 4, 20)
        self.assertEqual(int(hall["action"]), 3)

    def test_ladder_warps_to_b1f(self):
        ladder = first_rule(train.ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE, 82, 3, 37)
        self.assertEqual(int(ladder["action"]), 0)
        self.assertEqual(ladder["destination_map"], 232)

    def test_joy_ignore_cleared_on_first_b1f(self):
        import inspect
        src = inspect.getsource(
            train.PokemonYellowEnv._clear_rock_tunnel_overworld_joy_ignore
        )
        self.assertIn("ADDR_TEXT_BOX] = 0", src)
        self.assertIn("FULLGAME_ROCK_TUNNEL_INTERIOR_PHASES", src)

    def test_phase98_reregisters_route10_return(self):
        src = Path(train.__file__).read_text(encoding="utf-8")
        block = src[src.index("list(ROUTE10_TO_TUNNEL_ACTION_GUIDANCE)"):]
        block = block[: block.index("ROCK_TUNNEL_EXIT_ACTION_GUIDANCE")]
        self.assertIn("FULLGAME_ROCK_TUNNEL_FIRST_B1F_QUEST_PHASE", block)
        self.assertIn("list(ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE)", block)
        door = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 18, 8)
        self.assertEqual(door["destination_map"], 82)

    def test_sleep_trainer_recovery_shortens_sleep_and_applies_twice(self):
        env, memory = self.make_trainer_env(status=4)
        self.assertTrue(env._use_rock_tunnel_first_1f_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 107)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 107)
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 1)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 1)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 48
        self.assertTrue(env._use_rock_tunnel_first_1f_emergency_heal())
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 48
        self.assertFalse(env._use_rock_tunnel_first_1f_emergency_heal())

    def test_recovery_is_exactly_scoped_and_clears_non_sleep_status(self):
        env, memory = self.make_trainer_env(status=8)
        self.assertTrue(env._use_rock_tunnel_first_1f_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)

        env, _ = self.make_trainer_env()
        env.quest_phase += 1
        self.assertFalse(env._use_rock_tunnel_first_1f_emergency_heal())

        env, memory = self.make_trainer_env()
        memory[train.ADDR_POS_B] = 21
        self.assertFalse(env._use_rock_tunnel_first_1f_emergency_heal())


if __name__ == "__main__":
    unittest.main()
