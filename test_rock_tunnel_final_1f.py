#!/usr/bin/env python3
"""Rock Tunnel B1F final-1F leg: JrF sight tile must walk into the trainer."""
import inspect
import unittest
from unittest import mock

import train


class _Mem(dict):
    def __getitem__(self, key):
        return dict.get(self, key, 0)


def _env_with_left_rule():
    env = object.__new__(train.PokemonYellowEnv)
    env.actions = ["up", "down", "left", "right", "a", "b"]
    env.action_guidance_requires_overworld = True
    env.action_guidance = [{
        "map": 232,
        "target": (14, 11),
        "action": 2,
        "force_action": True,
        "quest_phases": frozenset({
            train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE,
        }),
    }]
    env.completed_run_recovery_after_steps = 0
    env.quest_phase = train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE
    env._forced_interaction_face_key = None
    env._action_guidance_event_allowed = lambda rule: True
    env.same_position_step_count = 0
    env._event_flag_is_set = lambda bit: False
    env.pyboy = mock.Mock()
    env.pyboy.memory = _Mem({
        train.ADDR_MAP_ID: 232,
        train.ADDR_POS_A: 14,
        train.ADDR_POS_B: 11,
        train.ADDR_BATTLE_FLAG: 0,
        train.ADDR_TEXT_BOX: 1,
    })
    return env


class RockTunnelFinal1FTests(unittest.TestCase):
    def test_jrf_recovery_is_single_use_after_two_genuine_faints(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE
        env.trainer_enemy_faints = 2
        env._rock_tunnel_final_jrf_emergency_heals_used = 0
        memory = _Mem({
            train.ADDR_MAP_ID: 232,
            train.ADDR_POS_A: 14,
            train.ADDR_POS_B: 11,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_HI: 0,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 35,
            train.ADDR_ACTIVE_MON_MAX_HP_HI: 0,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 107,
        })
        env.pyboy = mock.Mock()
        env.pyboy.memory = memory

        self.assertTrue(env._use_rock_tunnel_final_jrf_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 107)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 107)
        self.assertFalse(env._use_rock_tunnel_final_jrf_emergency_heal())

    def test_jrf_recovery_does_not_apply_before_second_faint(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE
        env.trainer_enemy_faints = 1
        env._rock_tunnel_final_jrf_emergency_heals_used = 0
        env.pyboy = mock.Mock()
        env.pyboy.memory = _Mem({
            train.ADDR_MAP_ID: 232,
            train.ADDR_POS_A: 14,
            train.ADDR_POS_B: 11,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 35,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 107,
        })

        self.assertFalse(env._use_rock_tunnel_final_jrf_emergency_heal())

    def test_jrf_recovery_does_not_apply_to_neighboring_battle(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE
        env.trainer_enemy_faints = 2
        env._rock_tunnel_final_jrf_emergency_heals_used = 0
        env.pyboy = mock.Mock()
        env.pyboy.memory = _Mem({
            train.ADDR_MAP_ID: 232,
            train.ADDR_POS_A: 13,
            train.ADDR_POS_B: 11,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 35,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 107,
        })

        self.assertFalse(env._use_rock_tunnel_final_jrf_emergency_heal())

    def test_hiker_recovery_is_twice_bounded_and_exactly_scoped(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE
        env._rock_tunnel_hiker_emergency_heals_used = 0
        env._event_flag_is_set = lambda bit: int(bit) == train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_0
        memory = _Mem({
            train.ADDR_MAP_ID: 232,
            train.ADDR_POS_A: 13,
            train.ADDR_POS_B: 6,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_HI: 0,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 40,
            train.ADDR_ACTIVE_MON_MAX_HP_HI: 0,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 107,
        })
        env.pyboy = mock.Mock()
        env.pyboy.memory = memory

        self.assertTrue(env._use_rock_tunnel_hiker_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 107)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 107)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 40
        memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 40
        self.assertTrue(env._use_rock_tunnel_hiker_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 107)
        self.assertFalse(env._use_rock_tunnel_hiker_emergency_heal())

    def test_hiker_recovery_does_not_apply_to_neighboring_ground_fights(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE
        env._rock_tunnel_hiker_emergency_heals_used = 0
        env._event_flag_is_set = lambda bit: int(bit) == train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_0
        env.pyboy = mock.Mock()
        env.pyboy.memory = _Mem({
            train.ADDR_MAP_ID: 232,
            train.ADDR_POS_A: 12,
            train.ADDR_POS_B: 6,
            train.ADDR_BATTLE_FLAG: 2,
        })
        self.assertFalse(env._use_rock_tunnel_hiker_emergency_heal())

    def test_jrf_battle_tile_allows_electric_without_widening_floor_policy(self):
        phase = train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE
        self.assertTrue(
            train.trainer_battle_allows_electric(232, phase, 14, 11)
        )
        self.assertFalse(
            train.trainer_battle_allows_electric(232, phase, 14, 12)
        )
        self.assertFalse(
            train.trainer_battle_allows_electric(232, phase)
        )
        self.assertFalse(
            train.trainer_battle_allows_electric(82, phase, 14, 11)
        )

    def test_jrf_sight_tile_is_on_the_forced_left_corridor(self):
        """JrF at pret (11,13) faces DOWN. (14,11) is the first vision tile."""
        guidance = getattr(train, "ROCK_TUNNEL_FINAL_1F_ACTION_GUIDANCE", None)
        if guidance is None:
            self.skipTest("FINAL_1F guidance lives on the remote trainer")
        hits = [
            rule
            for rule in guidance
            if tuple(rule["target"]) == (14, 11)
        ]
        self.assertEqual(len(hits), 1)
        self.assertEqual(int(hits[0]["action"]), 2)
        self.assertEqual(int(hits[0]["map"]), 232)

    def test_undefeated_sight_tile_walks_into_the_trainer(self):
        env = _env_with_left_rule()
        self.assertEqual(
            env._matching_forced_action_guidance(232, 14, 11, 0),
            0,
        )

    def test_beaten_jrf_walks_left_off_the_sight_tile(self):
        env = _env_with_left_rule()
        env._event_flag_is_set = (
            lambda bit: int(bit) == train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_0
        )
        self.assertEqual(
            env._matching_forced_action_guidance(232, 14, 11, 0),
            2,
        )

    def test_neighbor_tile_still_walks_left(self):
        env = _env_with_left_rule()
        env.action_guidance = [{
            "map": 232,
            "target": (14, 12),
            "action": 2,
            "force_action": True,
            "quest_phases": frozenset({
                train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE,
            }),
        }]
        self.assertEqual(
            env._matching_forced_action_guidance(232, 14, 12, 0),
            2,
        )

    def test_eight_step_fallback_does_not_mash_a_on_cd6b_one(self):
        env = _env_with_left_rule()
        env.action_guidance = []
        env.same_position_step_count = 8
        self.assertIsNone(
            env._matching_forced_action_guidance(232, 14, 11, 0),
        )

    def test_jrf_cone_is_registered(self):
        self.assertEqual(
            train.ROCK_TUNNEL_TRAINER_SIGHT_TILES[(232, 14, 11)],
            (train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_0, 0),
        )
        self.assertEqual(
            train.ROCK_TUNNEL_TRAINER_SIGHT_TILES[(232, 17, 11)],
            (train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_0, 0),
        )

    def test_real_dialog_still_mashes_a(self):
        env = _env_with_left_rule()
        env.pyboy.memory[train.ADDR_TEXT_BOX] = 0xFF
        self.assertEqual(
            env._matching_forced_action_guidance(232, 14, 11, 0),
            4,
        )

    def test_trainer_battle_byte_skips_the_cardinal(self):
        env = _env_with_left_rule()
        env.action_guidance_requires_overworld = False
        env.pyboy.memory[train.ADDR_TEXT_BOX] = 0xFF
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 2
        self.assertEqual(
            env._matching_forced_action_guidance(232, 14, 11, 2),
            4,
        )

    def test_stale_clear_does_not_abort_a_live_callout(self):
        src = inspect.getsource(
            train.PokemonYellowEnv._clear_stale_overworld_joy_ignore
        )
        self.assertIn("0xFF", src)

    def test_stale_clear_does_not_zero_tunnel_cd6b(self):
        env = _env_with_left_rule()
        env.same_position_step_count = 8
        self.assertFalse(env._clear_stale_overworld_joy_ignore())
        self.assertEqual(int(env.pyboy.memory[train.ADDR_TEXT_BOX]), 1)

    def test_skip_block_does_not_treat_cd6b_one_as_dialog(self):
        src = inspect.getsource(
            train.PokemonYellowEnv._matching_forced_action_guidance
        )
        self.assertIn("joy != 1", src)


if __name__ == "__main__":
    unittest.main()
