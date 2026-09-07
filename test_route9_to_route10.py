#!/usr/bin/env python3
import inspect
import unittest
from types import SimpleNamespace

import train


def first_rule(rules, map_id, y, x):
    for rule in rules:
        if rule["map"] == map_id and tuple(rule["target"]) == (y, x):
            return rule
    return None


class Route9ToRoute10Tests(unittest.TestCase):
    def test_first_route9_trainer_recovery_is_roster_bounded_and_scoped(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_ROUTE10_QUEST_PHASE
        env._route9_trainer0_emergency_heals_used = 0
        memory[train.ADDR_MAP_ID] = 20
        memory[train.ADDR_POS_A] = 10
        memory[train.ADDR_POS_B] = 11
        memory[train.ADDR_BATTLE_FLAG] = 2
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 20
        memory[train.ADDR_ACTIVE_MON_MAX_HP_LO] = 82
        memory[train.ADDR_ACTIVE_MON_STATUS] = 8
        lead_hi, lead_lo = train.PARTY_CUR_HP_ADDRS[0]
        memory[lead_lo] = 20
        memory[train.PARTY_STATUS_ADDRS[0]] = 8

        self.assertTrue(env._use_route9_trainer0_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 82)
        self.assertEqual(memory[lead_lo], 82)
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        self.assertFalse(env._use_route9_trainer0_emergency_heal())

        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 42
        self.assertFalse(env._use_route9_trainer0_emergency_heal())

        # A hit can cross the threshold directly to zero between env steps.
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 0
        memory[lead_lo] = 0
        self.assertTrue(env._use_route9_trainer0_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 82)

        # The four-opponent roster permits at most four recoveries total.
        for _ in range(2):
            memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 20
            self.assertTrue(env._use_route9_trainer0_emergency_heal())
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 20
        self.assertFalse(env._use_route9_trainer0_emergency_heal())

        # A different callout tile is not eligible in a new episode.
        env._route9_trainer0_emergency_heals_used = 0
        memory[train.ADDR_POS_B] = 12
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 20
        self.assertFalse(env._use_route9_trainer0_emergency_heal())

    def test_phase_order(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[train.FULLGAME_ROUTE9_QUEST_PHASE], "reach_route_9")
        self.assertEqual(
            names[train.FULLGAME_ROUTE9_CUT_QUEST_PHASE], "cross_route_9_cut_tree"
        )
        self.assertEqual(names[train.FULLGAME_ROUTE10_QUEST_PHASE], "reach_route_10")
        self.assertLess(
            train.FULLGAME_ROUTE9_CUT_QUEST_PHASE,
            train.FULLGAME_ROUTE10_QUEST_PHASE,
        )

    def test_hiker_pocket_escapes_via_x19(self):
        pile = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 14, 15)
        self.assertEqual(int(pile["action"]), 3)
        hiker_row = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 15, 14)
        self.assertEqual(int(hiker_row["action"]), 0)
        climb = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 13, 19)
        self.assertEqual(int(climb["action"]), 0)
        mouth = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 14, 19)
        self.assertEqual(int(mouth["action"]), 0)

    def test_jrf_platform_jumps_south(self):
        spawn = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 9, 13)
        self.assertEqual(int(spawn["action"]), 2)
        talk = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 10, 13)
        self.assertEqual(int(talk["action"]), 2)
        jump = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 10, 11)
        self.assertEqual(int(jump["action"]), 1)
        west = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 10, 10)
        self.assertEqual(int(west["action"]), 3)

    def test_east_corridor_drops_around_jrf2(self):
        before = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 8, 47)
        self.assertEqual(int(before["action"]), 1)
        after = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 8, 49)
        self.assertEqual(int(after["action"]), 3)

    def test_y12_never_hops_into_ledge_trap(self):
        for rule in train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE:
            if rule["map"] != 20:
                continue
            y, x = rule["target"]
            if y == 12:
                self.assertNotEqual(
                    int(rule["action"]),
                    1,
                    msg=f"y=12 x={x} must not hop south into the ledge pocket",
                )

    def test_east_connection_targets_route10(self):
        for y in (8, 9):
            edge = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, y, 59)
            self.assertEqual(int(edge["action"]), 3)
            self.assertEqual(edge["destination_map"], 21)

    def test_cut_tree_row_walks_east(self):
        tree = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 8, 4)
        self.assertEqual(int(tree["action"]), 3)
        past = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 20, 8, 5)
        self.assertEqual(int(past["action"]), 3)

    def test_cerulean_reenters_on_cut_row(self):
        door = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 3, 12, 39)
        self.assertEqual(int(door["action"]), 3)
        self.assertEqual(door["destination_map"], 20)
        north = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 3, 8, 39)
        self.assertEqual(int(north["action"]), 1)
        south = first_rule(train.ROUTE9_TO_ROUTE10_ACTION_GUIDANCE, 3, 16, 39)
        self.assertEqual(int(south["action"]), 0)

    def test_phase95_registers_route_and_cut(self):
        from pathlib import Path
        src = Path(train.__file__).read_text(encoding="utf-8")
        self.assertIn("for item in ROUTE9_TO_ROUTE10_ACTION_GUIDANCE", src)
        hook = inspect.getsource(train.PokemonYellowEnv._try_cut_progression_tree)
        self.assertIn("FULLGAME_ROUTE10_QUEST_PHASE: (20, (8, 4), 'right')", hook)


if __name__ == "__main__":
    unittest.main()
