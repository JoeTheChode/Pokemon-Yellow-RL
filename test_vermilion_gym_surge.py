#!/usr/bin/env python3
import unittest
from unittest import mock
from types import SimpleNamespace

import train


def first_rule(map_id, y, x):
    for rule in train.VERMILION_GYM_ACTION_GUIDANCE:
        if rule["map"] == map_id and tuple(rule["target"]) == (y, x):
            return rule
    return None


class VermilionGymSurgeTests(unittest.TestCase):
    def test_outdoor_pile_returns_to_door(self):
        door = first_rule(5, 20, 12)
        self.assertEqual(door["destination_map"], 92)
        self.assertEqual(int(door["action"]), 0)
        self.assertEqual(int(first_rule(5, 23, 9)["action"]), 3)
        self.assertEqual(int(first_rule(5, 22, 14)["action"]), 2)
        self.assertEqual(int(first_rule(5, 21, 12)["action"]), 0)

    def test_phase_scope_includes_surge(self):
        self.assertEqual(
            train.FULLGAME_QUEST_WAYPOINTS[train.FULLGAME_LT_SURGE_QUEST_PHASE]["name"],
            "beat_lt_surge",
        )
        self.assertEqual(train.EVENT_1ST_LOCK_OPENED, 353)
        self.assertEqual(train.EVENT_2ND_LOCK_OPENED, 352)
        self.assertEqual(train.FULLGAME_QUEST_WAYPOINTS[86]["event_bit"], 359)

    def test_trash_can_coords_match_hidden_events(self):
        self.assertEqual(train.VERMILION_GYM_TRASH_CANS[0], (7, 1))
        self.assertEqual(train.VERMILION_GYM_TRASH_CANS[10], (9, 7))
        self.assertEqual(train.VERMILION_GYM_TRASH_CANS[14], (11, 9))
        for can in train.VERMILION_GYM_TRASH_CANS:
            self.assertNotIn(can, train.VERMILION_GYM_WALKABLE)

    def test_stand_tiles_face_the_can(self):
        stands = train.vermilion_gym_stand_tiles(
            9, 7, train.VERMILION_GYM_WALKABLE,
        )
        tiles = {(y, x) for y, x, _ in stands}
        self.assertIn((10, 7), tiles)
        self.assertIn((8, 7), tiles)
        face = { (y, x): f for y, x, f in stands }
        self.assertEqual(face[(10, 7)], 0)
        self.assertEqual(face[(8, 7)], 1)

    def test_bfs_from_mat_to_can_ten(self):
        action = train.gym_bfs_first_action(
            train.VERMILION_GYM_WALKABLE,
            (17, 4),
            {(10, 7), (8, 7), (9, 6), (9, 8)},
        )
        self.assertIn(action, (0, 2, 3))
        self.assertIsNone(
            train.gym_bfs_first_action(
                train.VERMILION_GYM_WALKABLE, (10, 7), {(10, 7)},
            )
        )

    def test_surge_resources_are_restored_once_on_interaction_tile(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_LT_SURGE_QUEST_PHASE
        env._surge_battle_resources_prepared = False
        env._event_flag_is_set = lambda bit: False
        memory[train.ADDR_MAP_ID] = 92
        memory[train.ADDR_POS_A] = 2
        memory[train.ADDR_POS_B] = 5
        memory[train.ADDR_BATTLE_FLAG] = 0

        with (
            mock.patch.object(train, "_heal_party_memory") as heal,
            mock.patch.object(train, "_restore_party_pp_memory") as pp,
        ):
            self.assertTrue(env._prepare_surge_battle_resources())
            self.assertFalse(env._prepare_surge_battle_resources())

        heal.assert_called_once_with(memory)
        pp.assert_called_once_with(memory)
        self.assertTrue(env._surge_battle_resources_prepared)

    def test_surge_in_battle_recovery_is_scoped_and_bounded(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_LT_SURGE_QUEST_PHASE
        env._surge_emergency_heals_used = 0
        env._event_flag_is_set = lambda bit: False
        memory[train.ADDR_MAP_ID] = 92
        memory[train.ADDR_POS_A] = 2
        memory[train.ADDR_POS_B] = 5
        memory[train.ADDR_BATTLE_FLAG] = 2
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 39
        memory[train.ADDR_ACTIVE_MON_MAX_HP_LO] = 79
        memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 39
        memory[train.ADDR_ACTIVE_MON_STATUS] = 8
        memory[train.PARTY_STATUS_ADDRS[0]] = 8

        for expected in (1, 2):
            memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 39
            self.assertTrue(env._use_surge_emergency_heal())
            self.assertEqual(env._surge_emergency_heals_used, expected)
            self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 79)
            self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 79)
            self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)
            self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)

        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 39
        self.assertFalse(env._use_surge_emergency_heal())

        env._surge_emergency_heals_used = 0
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 54
        self.assertTrue(env._use_surge_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 79)

        env._surge_emergency_heals_used = 0
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 79
        self.assertFalse(env._use_surge_emergency_heal())

        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 54
        self.assertFalse(env._use_surge_emergency_heal())

    def test_verified_surge_win_restores_only_the_thunder_badge_bit(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        won = False
        env._event_flag_is_set = lambda bit: won and bit == 359
        memory[train.ADDR_BADGES] = 0x43

        self.assertFalse(env._restore_surge_badge_after_verified_win())
        self.assertEqual(memory[train.ADDR_BADGES], 0x43)

        won = True
        self.assertTrue(env._restore_surge_badge_after_verified_win())
        self.assertEqual(memory[train.ADDR_BADGES], 0x47)
        self.assertFalse(env._restore_surge_badge_after_verified_win())

    def test_door_mat_is_walkable(self):
        self.assertIn((17, 4), train.VERMILION_GYM_WALKABLE)
        self.assertIn((17, 5), train.VERMILION_GYM_WALKABLE)
        self.assertNotIn((2, 5), train.VERMILION_GYM_WALKABLE)
        self.assertIn((2, 5), train.VERMILION_GYM_NORTH_WALKABLE)

    def test_trash_can_route_avoids_undefeated_trainer_sightlines(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_LT_SURGE_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        memory[train.ADDR_MAP_ID] = 92
        memory[train.ADDR_POS_A] = 17
        memory[train.ADDR_POS_B] = 4
        env._vermilion_gym_target_can = lambda: (9, 1)

        hazards = set().union(*train.VERMILION_GYM_TRAINER_SIGHTLINES.values())
        walkable = set(train.VERMILION_GYM_WALKABLE) - hazards
        goals = {
            (y, x)
            for y, x, _ in train.vermilion_gym_stand_tiles(9, 1, walkable)
        }
        expected = train.gym_bfs_first_action(walkable, (17, 4), goals)
        self.assertEqual(env._vermilion_gym_forced_action(), expected)

    def test_every_can_remains_reachable_without_optional_trainers(self):
        hazards = set().union(*train.VERMILION_GYM_TRAINER_SIGHTLINES.values())
        walkable = set(train.VERMILION_GYM_WALKABLE) - hazards
        for can in train.VERMILION_GYM_TRASH_CANS:
            goals = {
                (y, x)
                for y, x, _ in train.vermilion_gym_stand_tiles(*can, walkable)
            }
            self.assertTrue(goals, can)
            self.assertIsNotNone(
                train.gym_bfs_first_action(walkable, (17, 4), goals), can,
            )

    def test_bicycle_frontier_dismisses_clerk_text_before_movement_probe(self):
        actions = train.swarm_frontier_validation_pre_actions(
            train.FULLGAME_RETURN_ROUTE24_CHARMANDER_QUEST_PHASE,
            66,
            {train.EVENT_GOT_BICYCLE},
        )
        self.assertEqual(actions, ('b',) * 8)
        self.assertEqual(
            train.swarm_frontier_validation_pre_actions(
                train.FULLGAME_BICYCLE_QUEST_PHASE,
                66,
                {train.EVENT_GOT_BICYCLE},
            ),
            (),
        )

        import inspect
        startup_validation = inspect.getsource(
            train.validate_or_quarantine_swarm_frontier
        )
        self.assertIn("pre_actions=startup_pre_actions", startup_validation)


if __name__ == "__main__":
    unittest.main()
