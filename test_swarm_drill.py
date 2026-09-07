import collections
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import train


class SwarmDrillHelperTests(unittest.TestCase):

    def test_route9_trainer0_handoff_restores_party_and_pp_once(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_ROUTE10_QUEST_PHASE
        env._route9_trainer0_handoff_prepared = False
        env._event_flag_is_set = lambda bit: bit == 1089
        memory[train.ADDR_MAP_ID] = 20
        memory[train.ADDR_BATTLE_FLAG] = 0

        with (
            mock.patch.object(train, '_heal_party_memory') as heal,
            mock.patch.object(train, '_restore_party_pp_memory') as pp,
        ):
            self.assertTrue(env._prepare_route9_trainer0_handoff_resources())
            self.assertFalse(env._prepare_route9_trainer0_handoff_resources())

        heal.assert_called_once_with(memory)
        pp.assert_called_once_with(memory)

    def test_route9_trainer0_handoff_requires_victory_and_overworld(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_ROUTE10_QUEST_PHASE
        env._route9_trainer0_handoff_prepared = False
        memory[train.ADDR_MAP_ID] = 20
        memory[train.ADDR_BATTLE_FLAG] = 0
        env._event_flag_is_set = lambda _bit: False
        self.assertFalse(env._prepare_route9_trainer0_handoff_resources())

        env._event_flag_is_set = lambda bit: bit == 1089
        memory[train.ADDR_BATTLE_FLAG] = 2
        self.assertFalse(env._prepare_route9_trainer0_handoff_resources())

    def test_misty_resources_restore_once_on_leader_approach(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_MISTY_QUEST_PHASE
        env._misty_battle_resources_prepared = False
        memory[train.ADDR_MAP_ID] = 65
        memory[train.ADDR_POS_A] = 3
        memory[train.ADDR_POS_B] = 5
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_BADGES] = 1

        with (
            mock.patch.object(train, '_heal_party_memory') as heal,
            mock.patch.object(train, '_restore_party_pp_memory') as pp,
        ):
            self.assertTrue(env._prepare_misty_battle_resources())
            self.assertFalse(env._prepare_misty_battle_resources())

        heal.assert_called_once_with(memory)
        pp.assert_called_once_with(memory)
        self.assertTrue(env._misty_battle_resources_prepared)

    def test_misty_resources_do_not_restore_in_battle_or_after_badge(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_MISTY_QUEST_PHASE
        env._misty_battle_resources_prepared = False
        memory[train.ADDR_MAP_ID] = 65
        memory[train.ADDR_POS_A] = 2
        memory[train.ADDR_POS_B] = 5
        memory[train.ADDR_BATTLE_FLAG] = 2
        memory[train.ADDR_BADGES] = 1
        self.assertFalse(env._prepare_misty_battle_resources())
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_BADGES] = 3
        self.assertFalse(env._prepare_misty_battle_resources())

    def test_misty_survivor_publishes_as_healed_rival_frontier(self):
        phase = train.FULLGAME_CERULEAN_RIVAL_QUEST_PHASE
        self.assertTrue(train.should_restore_misty_frontier_resources(phase))
        self.assertFalse(train.should_restore_misty_frontier_resources(phase + 1))
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase), 0.0)
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase), 0.0)
    def test_swarm_route_probe_does_not_consume_forced_npc_facing(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.actions = ['up', 'down', 'left', 'right', 'a', 'b']
        env.action_guidance_requires_overworld = True
        env.action_guidance = [{
            'map': 51,
            'target': (41, 1),
            'action': 4,
            'face_action': 3,
            'force_action': True,
            'quest_phases': frozenset({17}),
        }]
        env.quest_phase = 17
        env._forced_interaction_face_key = None
        env._action_guidance_event_allowed = lambda rule: True
        env.pyboy = mock.Mock()
        env.pyboy.memory = mock.MagicMock()
        env.pyboy.memory.__getitem__.side_effect = lambda key: 0
        env.pyboy.memory.__setitem__.side_effect = lambda key, value: None

        self.assertEqual(env._forced_direction_name(51, 41, 1), 'right')
        self.assertIsNone(env._forced_interaction_face_key)
        self.assertEqual(env._matching_forced_action_guidance(51, 41, 1, 0), 3)
        self.assertEqual(env._forced_interaction_face_key, (51, (41, 1)))
        self.assertEqual(env._matching_forced_action_guidance(51, 41, 1, 0), 4)

    def test_joy_ignore_does_not_consume_forced_npc_facing(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.actions = ['up', 'down', 'left', 'right', 'a', 'b']
        env.action_guidance_requires_overworld = True
        env.action_guidance = [{
            'map': 134,
            'target': (4, 4),
            'action': 4,
            'face_action': 0,
            'force_action': True,
            'quest_phases': frozenset({train.FULLGAME_ERIKA_QUEST_PHASE}),
        }]
        env.completed_run_recovery_after_steps = 0
        env.quest_phase = train.FULLGAME_ERIKA_QUEST_PHASE
        env.same_position_step_count = 0
        env._forced_interaction_face_key = None
        env._action_guidance_event_allowed = lambda rule: True
        env.pyboy = mock.Mock()
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 134,
            train.ADDR_POS_A: 4,
            train.ADDR_POS_B: 4,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_TEXT_BOX: 1,
        })
        env.pyboy.memory = memory

        self.assertEqual(env._matching_forced_action_guidance(134, 4, 4, 0), 0)
        self.assertIsNone(env._forced_interaction_face_key)
        self.assertEqual(env._matching_forced_action_guidance(134, 4, 4, 0), 0)
        self.assertIsNone(env._forced_interaction_face_key)

        memory[train.ADDR_TEXT_BOX] = 0
        self.assertEqual(env._matching_forced_action_guidance(134, 4, 4, 0), 0)
        self.assertEqual(env._forced_interaction_face_key, (134, (4, 4)))
        self.assertEqual(env._matching_forced_action_guidance(134, 4, 4, 0), 4)

    def test_erika_recovery_budget_covers_the_three_mon_battle(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ERIKA_QUEST_PHASE
        env._erika_emergency_heals_used = 0
        env._event_flag_is_set = lambda bit: False
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 134,
            train.ADDR_POS_A: 4,
            train.ADDR_POS_B: 4,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 40,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 100,
            train.PARTY_CUR_HP_ADDRS[0][1]: 40,
            train.PARTY_STATUS_ADDRS[0]: 8,
            train.ADDR_ACTIVE_MON_STATUS: 8,
        })
        env.pyboy = mock.Mock(memory=memory)

        for expected in range(1, 6):
            memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 40
            self.assertTrue(env._use_erika_emergency_heal())
            self.assertEqual(env._erika_emergency_heals_used, expected)
            self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 100)
            self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)
            self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 40
        self.assertFalse(env._use_erika_emergency_heal())

    def test_erika_recovery_preserves_live_sleep_turn_counter(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ERIKA_QUEST_PHASE
        env._erika_emergency_heals_used = 0
        env._event_flag_is_set = lambda bit: False
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 134,
            train.ADDR_POS_A: 4,
            train.ADDR_POS_B: 4,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 40,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 100,
            train.PARTY_CUR_HP_ADDRS[0][1]: 40,
            train.ADDR_ACTIVE_MON_STATUS: 3,
            train.PARTY_STATUS_ADDRS[0]: 3,
        })
        env.pyboy = mock.Mock(memory=memory)

        self.assertTrue(env._use_erika_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 100)
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 3)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 3)

    def test_erika_gym_trainer_recovery_has_bounded_low_hp_followups(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ERIKA_QUEST_PHASE
        env.trainer_enemy_faints = 0
        env._erika_gym_trainer_emergency_heals_used = 0
        env._event_flag_is_set = lambda bit: False
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 134,
            train.ADDR_POS_A: 11,
            train.ADDR_POS_B: 4,
            train.ADDR_BATTLE_FLAG: 2,
            # Deliberately above half: live crossed from this range to zero,
            # so a generic low-HP threshold cannot catch the failure.
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 80,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 112,
            train.PARTY_CUR_HP_ADDRS[0][1]: 80,
            train.PARTY_STATUS_ADDRS[0]: 8,
            train.ADDR_ACTIVE_MON_STATUS: 8,
        })
        env.pyboy = mock.Mock(memory=memory)

        self.assertFalse(env._use_erika_gym_trainer_emergency_heal())
        env.trainer_enemy_faints = 1
        self.assertTrue(env._use_erika_gym_trainer_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 112)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 112)
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        # Follow-ups are reserved for the production poison/lock sequence.
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 80
        self.assertFalse(env._use_erika_gym_trainer_emergency_heal())
        for expected in range(2, 6):
            memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 50
            self.assertTrue(env._use_erika_gym_trainer_emergency_heal())
            self.assertEqual(
                env._erika_gym_trainer_emergency_heals_used, expected
            )
            self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 112)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 50
        self.assertFalse(env._use_erika_gym_trainer_emergency_heal())

    def test_pewter_heal_frontier_accepts_alive_low_hp_clear(self):
        phase = train.FULLGAME_PEWTER_HEAL_QUEST_PHASE
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase), 0.0)
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase), 0.0)
        # The next phase is an RNG grind whose saved snapshot is healed.
        self.assertIn(phase + 1, train.FULLGAME_RNG_GRIND_QUEST_PHASES)
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase + 1), 0.0)
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase + 1), 0.0)
        # An ordinary story objective retains the configured safety floor.
        story_phase = train.FULLGAME_BROCK_QUEST_PHASE
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, story_phase), 0.4)
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, story_phase), 0.4)

    def test_post_surge_heal_frontier_accepts_verified_alive_win(self):
        phase = train.FULLGAME_POST_SURGE_HEAL_QUEST_PHASE
        # Live independent clears consistently finish at 44/111 HP (39.64%).
        # The next objective is the forced Vermilion Center heal, so the
        # frontier may retain an alive winner instead of refighting Surge.
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase), 0.0)
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase), 0.0)
        # The exemption is scoped to the heal leg.
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase + 1), 0.4)

    def test_post_erika_heal_frontier_accepts_low_lead_not_low_party(self):
        phase = train.FULLGAME_POST_ERIKA_HEAL_QUEST_PHASE
        # Live Erika clears finish with a 36% lead but a healthy overall
        # party.  The next forced route goes directly to Celadon Nurse Joy.
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase), 0.0)
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase), 0.4)
        self.assertIn(phase + 1, train.FULLGAME_RNG_GRIND_QUEST_PHASES)
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase + 1), 0.0)

    def test_rock_tunnel_exit_handoff_accepts_low_lead_not_low_party(self):
        phase = train.FULLGAME_LAVENDER_QUEST_PHASE
        # The live clear arrived with 26% lead HP but 59.5% whole-party HP.
        # Preserve the clear without weakening the party-wide safety floor.
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase), 0.0)
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase), 0.4)
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase + 1), 0.4)
        self.assertTrue(
            train.should_restore_rock_tunnel_traverse_frontier_resources(phase)
        )
        self.assertFalse(
            train.should_restore_rock_tunnel_traverse_frontier_resources(phase + 1)
        )

    def test_rock_tunnel_exit_handoff_steps_off_return_warp(self):
        phase = train.FULLGAME_LAVENDER_QUEST_PHASE
        self.assertEqual(
            train.completed_rock_tunnel_exit_handoff_action(21, 53, 8, phase),
            'down',
        )
        self.assertTrue(
            train.completed_rock_tunnel_exit_handoff_pending(21, 53, 8, phase)
        )
        self.assertIsNone(
            train.completed_rock_tunnel_exit_handoff_action(21, 54, 8, phase)
        )
        self.assertFalse(
            train.completed_rock_tunnel_exit_handoff_pending(21, 54, 8, phase)
        )
        self.assertIsNone(
            train.completed_rock_tunnel_exit_handoff_action(82, 53, 8, phase)
        )

    def test_lower_route10_trainer_gets_bounded_active_lead_recovery(self):
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 21,
            train.ADDR_POS_A: 64,
            train.ADDR_POS_B: 10,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 50,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 110,
            train.ADDR_ACTIVE_MON_STATUS: 8,
            train.PARTY_CUR_HP_ADDRS[0][1]: 50,
            train.PARTY_STATUS_ADDRS[0]: 8,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_LAVENDER_QUEST_PHASE
        env._route10_lower_trainer_emergency_heals_used = 0

        self.assertTrue(env._use_route10_lower_trainer_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 110)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 110)
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        self.assertFalse(env._use_route10_lower_trainer_emergency_heal())
        env._route10_lower_trainer_emergency_heals_used = 4
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 50
        self.assertFalse(env._use_route10_lower_trainer_emergency_heal())

    def test_final_route3_trainer_gets_two_scoped_low_hp_recoveries(self):
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 14,
            train.ADDR_POS_A: 8,
            train.ADDR_POS_B: 33,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 23,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 52,
            train.ADDR_ACTIVE_MON_STATUS: 8,
            train.PARTY_CUR_HP_ADDRS[0][1]: 23,
            train.PARTY_STATUS_ADDRS[0]: 8,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_ROUTE3_QUEST_PHASE
        env.trainer_enemy_faints = 11
        env._route3_trainer_emergency_heals_used = 0

        for expected in (1, 2):
            memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 23
            self.assertTrue(env._use_route3_trainer_emergency_heal())
            self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 52)
            self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 52)
            self.assertEqual(env._route3_trainer_emergency_heals_used, expected)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 23
        self.assertFalse(env._use_route3_trainer_emergency_heal())

        env._route3_trainer_emergency_heals_used = 0
        env.trainer_enemy_faints = 10
        self.assertFalse(env._use_route3_trainer_emergency_heal())
        env.trainer_enemy_faints = 11
        memory[train.ADDR_POS_B] = 32
        self.assertFalse(env._use_route3_trainer_emergency_heal())

    def test_final_route6_trainer_gets_one_scoped_low_hp_recovery(self):
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 17,
            train.ADDR_POS_A: 31,
            train.ADDR_POS_B: 8,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 34,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 79,
            train.ADDR_ACTIVE_MON_STATUS: 8,
            train.PARTY_CUR_HP_ADDRS[0][1]: 34,
            train.PARTY_STATUS_ADDRS[0]: 8,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_REACH_VERMILION_QUEST_PHASE
        env.trainer_enemy_faints = 5
        env._route6_trainer_emergency_heals_used = 0

        self.assertTrue(env._use_route6_trainer_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 79)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 79)
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 34
        self.assertFalse(env._use_route6_trainer_emergency_heal())

        # Earlier fights, nearby tiles, and other phases are not assisted.
        env._route6_trainer_emergency_heals_used = 0
        env.trainer_enemy_faints = 4
        self.assertFalse(env._use_route6_trainer_emergency_heal())
        env.trainer_enemy_faints = 5
        memory[train.ADDR_POS_A] = 30
        self.assertFalse(env._use_route6_trainer_emergency_heal())
        memory[train.ADDR_POS_A] = 31
        env.quest_phase += 1
        self.assertFalse(env._use_route6_trainer_emergency_heal())

    def test_completed_grind_frontier_is_healed_before_it_is_shared(self):
        completed_phase = 21  # train_pikachu_level_13
        candidate_phase = completed_phase + 1
        self.assertIn(completed_phase, train.FULLGAME_RNG_GRIND_QUEST_PHASES)
        self.assertEqual(
            train.swarm_required_party_hp_fraction(0.4, candidate_phase), 0.0
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(0.4, candidate_phase), 0.0
        )
        self.assertTrue(
            train.should_restore_rock_tunnel_grind_frontier_resources(
                candidate_phase, completed_phase, 0
            )
        )
        # An ordinary story transition retains the safety floor.
        story_phase = train.FULLGAME_ROUTE3_QUEST_PHASE
        self.assertNotIn(story_phase - 1, train.FULLGAME_RNG_GRIND_QUEST_PHASES)
        self.assertEqual(
            train.swarm_required_party_hp_fraction(0.4, story_phase), 0.4
        )

    def test_grind_frontier_accepts_either_patrol_direction(self):
        self.assertIsNone(
            train.swarm_frontier_required_validation_direction('up', True, 26)
        )
        self.assertEqual(
            train.swarm_frontier_required_validation_direction(
                'right', False, 25,
            ),
            'right',
        )
        self.assertEqual(
            train.swarm_frontier_required_validation_direction('left', True, 25),
            'left',
        )

    def test_soul_marsh_completion_waits_for_route15_handoff(self):
        phase = 224
        self.assertIn(phase, train.FULLGAME_SOUL_MARSH_GRIND_PHASES)
        self.assertTrue(
            train.completed_soul_marsh_grind_handoff_pending(25, phase)
        )
        self.assertTrue(
            train.completed_soul_marsh_grind_handoff_pending(157, phase)
        )
        self.assertFalse(
            train.completed_soul_marsh_grind_handoff_pending(26, phase)
        )
        self.assertFalse(
            train.completed_soul_marsh_grind_handoff_pending(
                25, train.FULLGAME_BROCK_QUEST_PHASE,
            )
        )

    def test_level_60_handoff_keeps_rotation_alive_until_fuchsia(self):
        phase = train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE
        self.assertTrue(
            train.completed_soul_marsh_grind_handoff_pending(26, phase)
        )
        self.assertTrue(
            train.completed_soul_marsh_grind_handoff_pending(184, phase)
        )
        for map_id in (7, 154, 157):
            self.assertFalse(
                train.completed_soul_marsh_grind_handoff_pending(map_id, phase)
            )

    def test_level_60_wild_battle_retains_grind_focus(self):
        phase = train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE
        self.assertTrue(
            train.soul_marsh_final_wild_handoff_active(26, phase, 1)
        )
        self.assertFalse(
            train.soul_marsh_final_wild_handoff_active(26, phase, 0)
        )
        self.assertFalse(
            train.soul_marsh_final_wild_handoff_active(7, phase, 1)
        )

    def test_level_60_route15_stale_joy_ignore_is_narrowly_clearable(self):
        phase = train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE
        call = train.soul_marsh_route15_joy_ignore_is_stale
        self.assertTrue(call(26, phase, 0, 0, 0xFF))
        self.assertTrue(call(26, phase, 0, 0, 0x01))
        self.assertFalse(call(26, phase, 1, 0, 0x01))
        self.assertFalse(call(26, phase, 0, 1, 0x01))
        self.assertFalse(call(184, phase, 0, 0, 0x01))
        self.assertFalse(call(26, phase - 1, 0, 0, 0x01))

    def test_level_60_route15_exit_uses_proven_gate_tail(self):
        phase = train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE
        call = lambda map_id, y, x, stuck=1, battle=0: (
            train.soul_marsh_to_fuchsia_action(
                map_id, y, x, phase, stuck, battle_flag=battle,
            )
        )
        self.assertEqual(call(26, 10, 18), 'up')
        self.assertEqual(call(26, 9, 18), 'left')
        self.assertEqual(call(26, 9, 20), 'left')
        self.assertEqual(call(26, 9, 19), 'left')
        self.assertEqual(call(26, 9, 16, stuck=1), 'left')
        self.assertEqual(call(26, 9, 16, stuck=2), 'left')
        self.assertEqual(call(26, 9, 14), 'left')
        self.assertEqual(call(184, 8, 14), 'up')
        self.assertIsNone(call(26, 9, 19, battle=1))

    def test_soul_marsh_grind_starts_after_fuchsia_center_checkpoint(self):
        snorlax = train.FULLGAME_ROUTE12_SNORLAX_QUEST_PHASE
        fuchsia = train.FULLGAME_FUCHSIA_QUEST_PHASE
        heal = train.FULLGAME_FUCHSIA_PRE_GRIND_HEAL_QUEST_PHASE
        first_grind = min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES)
        self.assertEqual((snorlax, fuchsia, heal, first_grind), (218, 219, 220, 221))
        self.assertEqual(
            train.reconcile_frontier_quest_phase(
                224, 'train_pikachu_soul_marsh_level_57',
                train.FULLGAME_QUEST_WAYPOINTS,
            ),
            227,
        )
        waypoint = train.FULLGAME_QUEST_WAYPOINTS[heal]
        self.assertEqual(waypoint['map'], 154)
        self.assertEqual(waypoint['blackout_map'], 7)
        self.assertTrue(waypoint['require_full_hp'])

    def test_fuchsia_arrival_accepts_low_lead_for_forced_heal_checkpoint(self):
        phase = train.FULLGAME_FUCHSIA_PRE_GRIND_HEAL_QUEST_PHASE
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase), 0.0)
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase), 0.4)
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_FUCHSIA_QUEST_PHASE,
            ),
            0.4,
        )

    def test_post_koga_safari_arrival_accepts_low_lead_not_low_party(self):
        # After beat_koga the enter_safari_zone candidate is written with the
        # lead at 0-8% while the reserves keep whole-party HP above its floor.
        # The next scripted action is the HP-gated Fuchsia Center heal and the
        # Safari Zone has no damaging encounters, so waive only the lead floor.
        phase = train.FULLGAME_SAFARI_QUEST_PHASE
        self.assertEqual(train.FULLGAME_KOGA_QUEST_PHASE + 1, phase)
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, phase), 0.0)
        # The independent whole-party floor still guards a fragile frontier.
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, phase), 0.4)
        # beat_koga is published from a healthy gym-entry state, and an
        # ordinary story objective both keep the configured lead floor.
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_KOGA_QUEST_PHASE,
            ),
            0.4,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_BROCK_QUEST_PHASE,
            ),
            0.4,
        )

    def test_post_koga_hurt_safari_phase_routes_through_center_before_gate(self):
        phase = train.FULLGAME_SAFARI_QUEST_PHASE

        def first_action(target, hp_fraction):
            for rule in train.PART11_ACTION_GUIDANCE:
                if int(rule.get('map', -1)) != 7:
                    continue
                if tuple(rule.get('target') or ()) != tuple(target):
                    continue
                phases = rule.get('quest_phases')
                if phases is not None and phase not in phases:
                    continue
                required = tuple(rule.get('require_event_bits') or ())
                excluded = tuple(rule.get('unless_event_bits') or ())
                if required and train.EVENT_BEAT_KOGA not in required:
                    continue
                if train.EVENT_BEAT_KOGA in excluded:
                    continue
                if hp_fraction < float(rule.get('min_party_hp_fraction', 0.0)):
                    continue
                if hp_fraction > float(rule.get('max_party_hp_fraction', 1.0)):
                    continue
                return rule['action'], rule.get('destination_map')
            return None, None

        # Koga's live survivor is ~53% whole-party HP. From the first safe tile
        # after the Gym door, hurt workers now move east into the existing
        # Center route. Once healed, the HP-gated bridge switches off and the
        # Safari flow resumes south from the same tile.
        self.assertEqual(first_action((28, 6), 0.53), (3, None))
        self.assertEqual(first_action((28, 7), 0.53), (3, None))
        self.assertEqual(first_action((28, 6), 1.0), (1, None))
        self.assertEqual(first_action((28, 8), 0.53), (3, None))

    def test_post_koga_safari_frontier_uses_verified_gym_exit_bridge(self):
        phase = train.FULLGAME_SAFARI_QUEST_PHASE
        expected = {
            (map_id, y, x): action
            for map_id, y, x, action, _destination
            in train.FUCHSIA_GYM_POST_KOGA_BRIDGE_STEPS
        }
        actual = {}
        for rule in train.PART11_ACTION_GUIDANCE:
            phases = rule.get('quest_phases')
            if phases is None or phase not in phases:
                continue
            if train.EVENT_BEAT_KOGA not in tuple(rule.get('require_event_bits') or ()):
                continue
            key = (rule.get('map'), *tuple(rule.get('target') or ()))
            if key in expected and rule.get('force_action'):
                actual[key] = rule['action']

        self.assertEqual(actual, expected)
        self.assertEqual(actual[(157, 11, 4)], 2)

    def test_fuchsia_center_returns_soul_marsh_grind_to_route15_grass(self):
        route = {
            (map_id, y, x): (action, destination)
            for map_id, y, x, action, destination
            in train.SOUL_MARSH_FUCHSIA_CENTER_TO_ROUTE15_STEPS
        }
        self.assertEqual(route[(154, 4, 3)], (1, None))
        self.assertEqual(route[(154, 6, 3)], (1, 7))
        self.assertEqual(route[(7, 28, 19)], (2, None))
        self.assertEqual(route[(184, 4, 7)], (3, 26))
        self.assertEqual(route[(26, 8, 18)], (3, None))
        self.assertEqual(route[(26, 9, 18)], (3, None))
        self.assertIn(154, train.FULLGAME_SOUL_MARSH_RETURN_MAPS)

        checkpoint_rule = next(
            rule for rule in train.FUCHSIA_PRE_GRIND_CHECKPOINT_ACTION_GUIDANCE
            if rule['map'] == 7 and rule['target'] == (28, 8)
        )
        return_rule = next(
            rule for rule in train.PART11_ACTION_GUIDANCE
            if rule['map'] == 7 and rule['target'] == (28, 8)
            and rule.get('require_blackout_map') == 7
        )
        self.assertEqual(checkpoint_rule['unless_blackout_map'], 7)
        self.assertEqual(checkpoint_rule['action'], 3)
        self.assertEqual(return_rule['action'], 1)

    def test_fuchsia_grind_direction_switches_after_recovery_anchor(self):
        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = collections.defaultdict(int)
        memory[train.ADDR_LEVEL] = 57
        env.pyboy = mock.Mock(memory=memory)
        env.is_swarm_explorer = False
        checkpoint = {'unless_blackout_map': 7}
        return_to_grass = {'require_blackout_map': 7}

        memory[train.ADDR_LAST_BLACKOUT_MAP] = 4
        self.assertTrue(env._action_guidance_event_allowed(checkpoint))
        self.assertFalse(env._action_guidance_event_allowed(return_to_grass))
        memory[train.ADDR_LAST_BLACKOUT_MAP] = 7
        self.assertFalse(env._action_guidance_event_allowed(checkpoint))
        self.assertTrue(env._action_guidance_event_allowed(return_to_grass))

    def test_rng_grinds_persist_incremental_xp_without_advancing_mastery(self):
        phase = min(train.FULLGAME_SILPH_GRIND_PHASES)
        self.assertEqual(
            train.fullgame_swarm_subphase_value(phase, 17, 123456),
            123456,
        )
        self.assertTrue(train.swarm_incremental_grind_frontier_candidate(
            phase, phase, 123456, 123000,
        ))
        # The serialized grind snapshot is healed, so low HP must not reject
        # the partial XP before that repair can run. A zero-HP party is still
        # rejected separately by the party_fainted frontier blocker.
        self.assertEqual(
            train.swarm_required_party_hp_fraction(0.4, phase), 0.0
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(0.4, phase), 0.0
        )
        self.assertTrue(
            train.should_restore_rock_tunnel_grind_frontier_resources(
                phase, phase, 123000,
            )
        )
        self.assertTrue(train.swarm_grind_battle_teardown_sync_due(
            phase, previous_battle_flag=1, battle_flag=0, total_party_hp=1,
        ))
        self.assertTrue(train.swarm_grind_post_battle_sync_due(
            phase, battle_flag=0, total_party_hp=1, grace_steps_remaining=32,
        ))
        self.assertFalse(train.swarm_grind_post_battle_sync_due(
            phase, battle_flag=1, total_party_hp=1, grace_steps_remaining=32,
        ))
        self.assertTrue(train.swarm_grind_frontier_text_latch_is_stale(
            phase, battle_flag=0, text_box_active=True,
            font_loaded_active=False,
        ))
        self.assertFalse(train.swarm_grind_frontier_text_latch_is_stale(
            phase, battle_flag=0, text_box_active=True,
            font_loaded_active=True,
        ))
        self.assertFalse(train.swarm_grind_battle_teardown_sync_due(
            phase, previous_battle_flag=1, battle_flag=0, total_party_hp=0,
        ))
        self.assertFalse(train.swarm_grind_battle_teardown_sync_due(
            phase, previous_battle_flag=0, battle_flag=0, total_party_hp=1,
        ))
        self.assertFalse(train.swarm_grind_battle_teardown_sync_due(
            train.FULLGAME_BROCK_QUEST_PHASE,
            previous_battle_flag=1,
            battle_flag=0,
            total_party_hp=1,
        ))
        self.assertFalse(train.swarm_incremental_grind_frontier_candidate(
            phase, phase, 123000, 123000,
        ))
        self.assertFalse(train.swarm_incremental_grind_frontier_candidate(
            phase + 1, phase, 124000, 123000,
        ))
        story_phase = train.FULLGAME_BROCK_QUEST_PHASE
        self.assertFalse(train.swarm_incremental_grind_frontier_candidate(
            story_phase, story_phase, 10, 0,
        ))

    def test_charmander_grind_uses_family_xp_and_repairs_lead_xp_frontier(self):
        phase = min(train.FULLGAME_CHARMANDER_GRIND_PHASES)
        memory = bytearray(0x10000)
        memory[train.ADDR_PARTY_SIZE] = 2
        memory[train.PARTY_SPECIES_ADDRS[0]] = train.GEN1_PIKACHU_SPECIES_ID
        memory[train.PARTY_SPECIES_ADDRS[1]] = min(
            train.GEN1_CHARMANDER_SPECIES_IDS
        )
        family_xp = 560
        xp_addr = train.PARTY_SPECIES_ADDRS[1] + train.PARTY_EXPERIENCE_OFFSET
        memory[xp_addr:xp_addr + 3] = family_xp.to_bytes(3, 'big')

        self.assertEqual(
            train.party_species_experience(
                memory, train.GEN1_CHARMANDER_SPECIES_IDS,
            ),
            family_xp,
        )
        poisoned = train.fullgame_swarm_subphase_value(phase, 17, 136218)
        self.assertEqual(
            train.repair_charmander_grind_frontier_value(
                phase, poisoned, family_xp, target_level=11,
            ),
            family_xp * 1000 + 17,
        )
        legitimate = train.fullgame_swarm_subphase_value(phase, 17, 600)
        self.assertEqual(
            train.repair_charmander_grind_frontier_value(
                phase, legitimate, family_xp, target_level=11,
            ),
            legitimate,
        )

    def test_full_health_charmander_exits_cerulean_center(self):
        rules = train.CHARMANDER_CERULEAN_CENTER_EXIT_RECOVERY_ACTION_GUIDANCE
        by_target = {tuple(rule['target']): rule for rule in rules}
        self.assertEqual(by_target[(7, 2)]['action'], 3)
        self.assertEqual(by_target[(6, 5)]['action'], 2)
        self.assertEqual(by_target[(7, 3)]['action'], 1)
        self.assertEqual(by_target[(7, 3)]['destination_map'], 3)

    def test_silph_grind_route16_keeps_celadon_frontier_tier(self):
        phase = min(train.FULLGAME_SILPH_GRIND_PHASES)
        self.assertEqual(
            train.fullgame_swarm_route_progress(phase, 27, -1, 0),
            (8, 0),
        )
        # Celadon and its Center are the two other maps in the exact approach
        # trace and must compare in the same tier throughout the grind.
        for map_id in (6, 133):
            self.assertEqual(
                train.fullgame_swarm_route_progress(phase, map_id, 8, 0),
                (8, 0),
            )

    def test_coverage_report_flags_only_unaccounted_grinds(self):
        waypoints = [
            {'name': 'walk', 'map': 1},
            {'name': 'train', 'min_level': 6},
        ]
        with mock.patch.object(
            train, 'FULLGAME_RNG_GRIND_QUEST_PHASES', frozenset({1})
        ):
            report = train.fullgame_automation_coverage_report(
                waypoints,
                [{
                    'target': (1, 1), 'force_action': True,
                    'quest_phases': frozenset({0}),
                }],
                {'routes': {}},
            )
        self.assertEqual(report['missing_rng_grind_phases'], [1])
        self.assertTrue(report['objectives'][0]['accounted'])
        self.assertFalse(report['objectives'][1]['accounted'])

    def test_completed_run_recovery_uses_only_walked_unambiguous_moves(self):
        def event(before, after, held):
            return json.dumps({
                'event_type': 'state_change',
                'held_buttons': [held],
                'previous_state': {
                    'map_id': 3, 'y': before[0], 'x': before[1],
                    'battle': 0, 'text_box': 0,
                },
                'state': {
                    'map_id': 3, 'y': after[0], 'x': after[1],
                    'battle': 0, 'text_box': 0,
                },
            })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.jsonl'
            path.write_text('\n'.join((
                event((5, 5), (5, 6), 'right'),
                event((5, 5), (4, 5), 'up'),  # conflicting origin: discarded
                event((7, 7), (7, 8), 'right'),
                event((9, 9), (9, 9), 'left'),  # blocked: discarded
            )), encoding='utf-8')
            rules = train.completed_run_recording_guidance(path, {55})
        self.assertEqual(
            [(rule['target'], rule['action']) for rule in rules],
            [((7, 7), 3)],
        )
        self.assertEqual(rules[0]['quest_phases'], frozenset({55}))

    def test_completed_run_recovery_trusts_movement_over_stale_flags(self):
        event = {
            'event_type': 'state_change',
            'held_buttons': ['down'],
            'previous_state': {
                'map_id': 88, 'y': 5, 'x': 4,
                'battle': 1, 'text_box': 212,
            },
            'state': {
                'map_id': 88, 'y': 6, 'x': 4,
                'battle': 1, 'text_box': 212,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.jsonl'
            path.write_text(json.dumps(event), encoding='utf-8')
            rules = train.completed_run_recording_guidance(path, {63})
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]['target'], (5, 4))
        self.assertEqual(rules[0]['action'], 1)

    def test_bill_guidance_covers_separator_ticket_and_exit(self):
        phases = train.FULLGAME_BILL_INTERIOR_QUEST_PHASES
        self.assertEqual(
            phases,
            frozenset({
                train.FULLGAME_MEET_BILL_QUEST_PHASE,
                train.FULLGAME_MEET_BILL_QUEST_PHASE + 1,
                train.FULLGAME_MEET_BILL_QUEST_PHASE + 2,
            }),
        )
        post_separator = [
            rule for rule in train.BILLS_HOUSE_ACTION_GUIDANCE
            if rule.get('require_event_bits') == (1371,)
        ]
        exit_rules = [
            rule for rule in train.BILLS_HOUSE_ACTION_GUIDANCE
            if rule.get('require_event_bits') == (1372,)
        ]
        self.assertTrue(post_separator)
        self.assertTrue(exit_rules)

    def test_completed_run_recovery_activates_only_after_stall(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.action_guidance_requires_overworld = True
        env.action_guidance = []
        env.completed_run_recovery_guidance = [{
            'map': 3, 'target': (7, 7), 'action': 3,
            'force_action': True, 'quest_phases': frozenset({55}),
        }]
        env.completed_run_recovery_after_steps = 768
        env.quest_phase = 55
        env.pyboy = mock.Mock()
        env.pyboy.memory = collections.defaultdict(int, {train.ADDR_LEVEL: 30})
        env._event_flag_is_set = lambda _bit: False
        env._bag_contains_item = lambda _item: False
        env.quest_phase_step_count = 767
        self.assertIsNone(env._matching_forced_action_guidance(3, 7, 7, 0))
        env.quest_phase_step_count = 768
        self.assertEqual(env._matching_forced_action_guidance(3, 7, 7, 0), 3)

    def test_cerulean_gym_guidance_uses_known_city_and_route4_tiles(self):
        forced = {
            (rule['map'], rule['target'], rule['action'])
            for rule in train.CERULEAN_GYM_ACTION_GUIDANCE
            if rule.get('force_action')
        }
        # Route 4 east warp into Cerulean.
        self.assertIn((15, (10, 89), 3), forced)
        # Center spawn joins the y=18 street, then walks east past the Center.
        self.assertIn((3, (17, 19), 1), forced)
        self.assertIn((3, (18, 19), 3), forced)
        # Reverse of the documented gym-exit: x=22 down, y=20 east, door Up.
        self.assertIn((3, (18, 22), 1), forced)
        self.assertIn((3, (20, 29), 3), forced)
        self.assertIn((3, (20, 30), 0), forced)

    def test_cerulean_rival_route_exits_gym_and_walks_to_bridge_trigger(self):
        exit_forced = {
            (rule['map'], rule['target'], rule['action'])
            for rule in train.CERULEAN_RIVAL_ACTION_GUIDANCE
            if rule.get('force_action')
        }
        # Heatmap BFS spine: Misty talk tile south/east then down the x=5 column.
        self.assertIn((65, (2, 5), 1), exit_forced)
        self.assertIn((65, (3, 7), 1), exit_forced)
        self.assertIn((65, (13, 5), 1), exit_forced)
        self.assertIn((65, (13, 4), 1), exit_forced)
        # Gym door street -> y=18 west around the Center -> x=20 north.
        # The first BFS climbed the gym ledge at (22,17); that tile is
        # one-way down, which pinned the live swarm on (22,18).
        self.assertIn((3, (20, 30), 2), exit_forced)
        self.assertIn((3, (18, 22), 2), exit_forced)
        self.assertIn((3, (18, 17), 0), exit_forced)
        self.assertIn((3, (16, 11), 2), exit_forced)
        self.assertIn((3, (12, 8), 3), exit_forced)
        self.assertIn((3, (12, 20), 0), exit_forced)
        self.assertIn((3, (6, 21), 4), exit_forced)
        self.assertIn((3, (6, 20), 4), exit_forced)
        warp = next(
            rule for rule in train.CERULEAN_GYM_EXIT_ACTION_GUIDANCE
            if rule['map'] == 65 and rule['target'] == (13, 5)
        )
        self.assertEqual(warp.get('destination_map'), 3)
        north = {
            (rule['target'], rule['action'], rule.get('destination_map'))
            for rule in train.CERULEAN_ROUTE24_ACTION_GUIDANCE
            if rule.get('map') == 3 and rule['target'] in {(0, 20), (0, 21)}
        }
        self.assertIn(((0, 20), 0, 35), north)
        self.assertIn(((0, 21), 0, 35), north)

    def test_nugget_bridge_and_bill_routes_use_heatmap_spines(self):
        bridge = {
            (rule['target'], rule['action'], rule.get('destination_map'))
            for rule in train.NUGGET_BRIDGE_ACTION_GUIDANCE
            if rule['map'] == 35
        }
        self.assertIn(((35, 10), 0, None), bridge)
        self.assertIn(((32, 11), 2, None), bridge)
        self.assertIn(((4, 14), 1, None), bridge)
        self.assertIn(((5, 14), 3, None), bridge)
        self.assertIn(((8, 14), 3, None), bridge)
        self.assertIn(((8, 19), 3, 36), bridge)
        forced = {}
        for rule in train.NUGGET_BRIDGE_ACTION_GUIDANCE:
            if not rule.get('force_action'):
                continue
            key = (rule['map'], rule['target'])
            action = (rule['action'], rule.get('face_action'))
            self.assertNotIn(key, forced, f'conflicting force at {key}')
            forced[key] = action
        self.assertFalse(
            train.fullgame_frontier_position_allowed(
                train.FULLGAME_ROUTE25_QUEST_PHASE, 35, 4, 14,
            )
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(
                train.FULLGAME_ROUTE25_QUEST_PHASE, 35, 8, 14,
            )
        )
        door = {
            (rule['target'], rule['action'], rule.get('destination_map'))
            for rule in train.ROUTE25_TO_BILL_ACTION_GUIDANCE
            if rule['map'] == 36
        }
        self.assertIn(((4, 0), 3, None), door)
        self.assertIn(((4, 45), 0, 88), door)
        first = next(
            (rule['action'], rule.get('destination_map'))
            for rule in train.ROUTE25_TO_BILL_ACTION_GUIDANCE
            if rule['map'] == 36 and rule['target'] == (8, 17)
        )
        self.assertEqual(first, (0, None))
        exit_rules = {
            (rule['target'], rule.get('destination_map'))
            for rule in train.BILL_HOUSE_EXIT_ACTION_GUIDANCE
            if rule['map'] == 88 and rule.get('destination_map') == 36
        }
        self.assertIn(((7, 2), 36), exit_rules)
        self.assertIn(((7, 3), 36), exit_rules)

    def test_pewter_grind_uses_emulator_replayed_route_to_viewer_67_179(self):
        steps = train.PEWTER_CENTER_TO_ROUTE2_GRASS_STEPS
        self.assertEqual(steps[0], (58, 3, 3, 1, None))
        self.assertIn((58, 7, 3, 1, 2), steps)
        self.assertIn((2, 35, 18, 1, 13), steps)
        self.assertEqual(steps[-1], (13, 4, 3, 1, None))
        # Route 2's viewer offset is (64,174): local (x=3,y=5).
        self.assertEqual((64 + 3, 174 + 5), (67, 179))
        pacing = train.PEWTER_GRIND_ROUTE_ACTION_GUIDANCE[-2:]
        self.assertEqual(
            [(rule['target'], rule['action']) for rule in pacing],
            [((5, 3), 1), ((6, 3), 0)],
        )
        self.assertTrue(all(rule['force_action'] for rule in pacing))

    def test_pewter_grind_retreats_to_center_when_hurt(self):
        """Hurt workers must leave the Route 2 pair; healthy ones stay."""

        def first_action(map_id, target, hp_fraction):
            for rule in train.PEWTER_GRIND_ROUTE_ACTION_GUIDANCE:
                if int(rule.get('map', -1)) != int(map_id):
                    continue
                if tuple(rule.get('target') or ()) != tuple(target):
                    continue
                if hp_fraction < float(rule.get('min_party_hp_fraction', 0.0)):
                    continue
                if hp_fraction > float(rule.get('max_party_hp_fraction', 1.0)):
                    continue
                return rule['action'], rule.get('destination_map')
            return None, None

        heal_frac = train.PEWTER_GRIND_HEAL_HP_FRACTION
        self.assertEqual(heal_frac, 0.9)
        # (5,3) is the conflicting tile: DOWN patrol vs UP toward Pewter.
        self.assertEqual(first_action(13, (5, 3), 1.0), (1, None))
        self.assertEqual(first_action(13, (5, 3), 0.5), (0, None))
        # (6,3) UP is on both the patrol and the retreat; keep it ungated.
        self.assertEqual(first_action(13, (6, 3), 1.0), (0, None))
        self.assertEqual(first_action(13, (6, 3), 0.5), (0, None))
        # Nurse tile: mash A while hurt, walk out once healed.
        self.assertEqual(first_action(58, (3, 3), 0.5), (4, None))
        self.assertEqual(first_action(58, (3, 3), 1.0), (1, None))
        self.assertEqual(first_action(2, (26, 13), 0.5), (0, 58))
        # Healthy workers at the Center door keep walking to Route 2, not in.
        door_healthy_action, door_healthy_dest = first_action(2, (26, 13), 1.0)
        self.assertIsNotNone(door_healthy_action)
        self.assertNotEqual(door_healthy_dest, 58)
        outbound = train.PEWTER_GRIND_ROUTE_ACTION_GUIDANCE[
            len(train.PEWTER_GRIND_HEAL_ACTION_GUIDANCE)
        ]
        self.assertEqual(outbound['target'], (3, 3))
        self.assertEqual(outbound['map'], 58)
        self.assertEqual(
            outbound.get('min_party_hp_fraction'), heal_frac
        )
        self.assertEqual(
            train.PEWTER_GRIND_ROUTE_ACTION_GUIDANCE[-2].get(
                'min_party_hp_fraction'
            ),
            heal_frac,
        )
        self.assertNotIn(
            'min_party_hp_fraction',
            train.PEWTER_GRIND_ROUTE_ACTION_GUIDANCE[-1],
        )

    def test_completed_run_route_carries_level_twenty_into_pewter_gym(self):
        steps = train.PEWTER_ROUTE2_GRASS_TO_CENTER_STEPS
        self.assertEqual(steps[0], (13, 5, 3, 0, None))
        self.assertEqual(steps[-1], (2, 27, 13, 0, None))
        self.assertFalse(any(step[0] == 58 for step in steps))
        gym = train.PEWTER_CENTER_TO_GYM_STEPS
        self.assertEqual(gym[0], (58, 3, 3, 1, None))
        self.assertEqual(gym[-1], (2, 18, 16, 0, 54))
        self.assertTrue(all(
            rule.get('force_action')
            for rule in train.PEWTER_GYM_APPROACH_ACTION_GUIDANCE
        ))

    def test_completed_run_route_leaves_brock_for_route_three(self):
        gym_exit = train.PEWTER_GYM_TO_CENTER_STEPS
        self.assertEqual(gym_exit[0], (54, 2, 4, 1, None))
        self.assertEqual(gym_exit[10], (54, 12, 4, 1, 2))
        route3 = train.PEWTER_CENTER_TO_ROUTE3_STEPS
        self.assertEqual(route3[0], (58, 4, 3, 1, None))
        self.assertEqual(route3[-1], (2, 18, 38, 3, 14))
        to_center = train.PEWTER_POST_BROCK_ACTION_GUIDANCE[
            :len(train.PEWTER_GYM_TO_CENTER_STEPS)
        ]
        self.assertTrue(all(
            rule.get('max_party_hp_fraction') == 0.9
            for rule in to_center
        ))
        nurse_exit = [
            rule for rule in train.PEWTER_POST_BROCK_ACTION_GUIDANCE
            if rule.get('map') == 58
            and rule.get('target') == (3, 3)
            and rule.get('min_party_hp_fraction') == 0.9
        ]
        self.assertEqual(len(nurse_exit), 1)
        self.assertEqual(nurse_exit[0]['action'], 1)
        self.assertTrue(nurse_exit[0]['force_action'])

    def test_proven_trace_replays_only_recorded_states_and_rejoins_forward(self):
        trace = [
            [1, 35, 21, 0, 0, 'up'],
            [1, 34, 21, 0, 0, 'up'],
            [1, 33, 21, 0, 0, 'left'],
        ]
        action, cursor = train.proven_quest_trace_action(
            trace, 0, (1, 35, 21, 0, 0)
        )
        self.assertEqual((action, cursor), ('up', 1))
        self.assertEqual(
            train.proven_quest_trace_action(trace, cursor, (1, 99, 99, 0, 0)),
            (None, cursor),
        )
        self.assertEqual(
            train.proven_quest_trace_action(trace, cursor, (1, 33, 21, 0, 0)),
            ('left', 3),
        )
        # A recovery detour can revisit a known tile behind the cursor.
        self.assertEqual(
            train.proven_quest_trace_action(trace, 3, (1, 35, 21, 0, 0)),
            ('up', 1),
        )

    def test_proven_trace_normalization_erases_route10_replay_loop(self):
        noisy = [
            [21, 54, 9, 0, 0, 'right'],
            [21, 54, 10, 0, 0, 'right'],
            [21, 54, 11, 0, 0, 'up'],
            [21, 54, 11, 0, 0, 'left'],
            [21, 54, 10, 0, 0, 'right'],
            [21, 54, 11, 0, 0, 'right'],
            [21, 54, 12, 0, 0, 'right'],
        ]
        normalized = train._normalize_proven_navigation_trace(noisy)
        self.assertEqual(normalized, [
            [21, 54, 9, 0, 0, 'right'],
            [21, 54, 10, 0, 0, 'right'],
            [21, 54, 11, 0, 0, 'right'],
            [21, 54, 12, 0, 0, 'right'],
        ])
        cursor = 0
        for x in range(9, 13):
            action, cursor = train.proven_quest_trace_action(
                normalized, cursor, (21, 54, x, 0, 0)
            )
            self.assertEqual(action, 'right')

    def test_proven_trace_store_keeps_only_shortest_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'proven.json'
            slow = [[1, y, 21, 0, 0, 'up'] for y in range(35, 30, -1)]
            fast = slow[:3]
            self.assertTrue(train.record_swarm_proven_quest_trace(path, 1, slow))
            self.assertFalse(train.record_swarm_proven_quest_trace(path, 1, slow))
            self.assertTrue(
                train.record_swarm_proven_quest_trace(path, 1, fast, source='drill')
            )
            self.assertEqual(train.load_swarm_proven_quest_trace(path, 1), fast)
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['routes']['1']['steps'], 3)
            self.assertEqual(payload['routes']['1']['source'], 'drill')

            payload['quest_layout'] = 'level_bands_v1'
            path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertTrue(train.record_swarm_proven_quest_trace(
                path, 1, fast[:2], source='drill'
            ))
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['quest_layout'], 'level_bands_v1')

    def test_proven_trace_excludes_dialogue_and_compares_completion_time(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'proven.json'
            noisy = [
                [40, 5, 5, 0, 0, 'up'],
                [40, 4, 5, 0, 0, 'up'],
                [40, 3, 5, 0, 1, 'b'],
                [40, 3, 5, 0, 1, 'a'],
                [40, 3, 5, 1, 0, 'left'],
            ]
            self.assertTrue(train.record_swarm_proven_quest_trace(
                path, 6, noisy, completion_steps=155
            ))
            self.assertEqual(train.load_swarm_proven_quest_trace(path, 6), [
                [40, 5, 5, 0, 0, 'up'],
                [40, 4, 5, 0, 0, 'up'],
            ])
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['routes']['6']['steps'], 155)
            self.assertEqual(payload['routes']['6']['navigation_steps'], 2)
            # Fewer stored directions do not replace a faster completed run.
            self.assertFalse(train.record_swarm_proven_quest_trace(
                path,
                6,
                [[40, 5, 5, 0, 0, 'up']],
                completion_steps=200,
            ))

    def test_verified_guidance_seeds_missing_completed_routes(self):
        guidance = [
            {
                'map': 1, 'target': (35, 21), 'action': 0,
                'verified_route': True, 'quest_phases': frozenset({0, 1}),
            },
            {
                'map': 1, 'target': (34, 21), 'action': 2,
                'verified_route': True, 'quest_phases': frozenset({1}),
            },
            # Conditional interactions remain with action guidance.
            {
                'map': 51, 'target': (41, 1), 'action': 0,
                'verified_route': True, 'quest_phases': frozenset({11}),
                'unless_event_bits': (1381,),
            },
            {
                'map': 2, 'target': (35, 18), 'action': 0,
                'force_action': True, 'radius': 0,
                'quest_phases': frozenset({12}),
            },
            {
                'map': 2, 'target': (34, 18), 'action': 0,
                'quest_phases': frozenset({13}),
            },
        ]
        atlas = train.verified_guidance_route_atlas(guidance)
        self.assertEqual(atlas[0], [[1, 35, 21, 0, 0, 'up']])
        self.assertEqual(atlas[1], [
            [1, 35, 21, 0, 0, 'up'],
            [1, 34, 21, 0, 0, 'left'],
        ])
        self.assertNotIn(11, atlas)
        self.assertEqual(atlas[12], [[2, 35, 18, 0, 0, 'up']])
        self.assertNotIn(13, atlas)
        self.assertNotIn(
            12,
            train.verified_guidance_route_atlas(guidance, quest_count=12),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'proven.json'
            path.write_text(json.dumps({
                'schema': 2, 'quest_layout': 'level_bands_v1', 'routes': {},
            }), encoding='utf-8')
            self.assertEqual(
                train.seed_swarm_proven_routes_from_guidance(path, guidance),
                [0, 1, 12],
            )
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['quest_layout'], 'level_bands_v1')
            self.assertEqual(payload['routes']['0']['source'], 'route_atlas_seed')
            self.assertEqual(train.load_swarm_proven_quest_trace(path, 1), atlas[1])
            # A measured completion replaces the seed and is never overwritten.
            self.assertTrue(train.record_swarm_proven_quest_trace(
                path, 0, [[1, 35, 21, 0, 0, 'up']],
                source='drill', completion_steps=12,
            ))
            self.assertEqual(
                train.seed_swarm_proven_routes_from_guidance(path, guidance), []
            )
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['routes']['0']['source'], 'drill')

    def test_oak_dialogue_fast_forward_is_exact_and_phase_bounded(self):
        self.assertEqual(train.oak_parcel_dialogue_completion_bit(
            train.FULLGAME_DELIVER_OAKS_PARCEL_QUEST_PHASE,
            40, 3, 5, 'a',
        ), 56)
        self.assertEqual(train.oak_parcel_dialogue_completion_bit(
            train.FULLGAME_RECEIVE_POKEDEX_QUEST_PHASE,
            40, 3, 5, 'a',
        ), 37)
        self.assertIsNone(train.oak_parcel_dialogue_completion_bit(
            train.FULLGAME_DELIVER_OAKS_PARCEL_QUEST_PHASE,
            40, 3, 5, 'b',
        ))
        self.assertIsNone(train.oak_parcel_dialogue_completion_bit(
            train.FULLGAME_DELIVER_OAKS_PARCEL_QUEST_PHASE,
            40, 4, 5, 'a',
        ))
        self.assertEqual(train.allowed_atomic_frontier_event_bits(
            train.FULLGAME_RECEIVE_POKEDEX_QUEST_PHASE, {56, 37}
        ), {37})
        self.assertEqual(train.allowed_atomic_frontier_event_bits(
            train.FULLGAME_DELIVER_OAKS_PARCEL_QUEST_PHASE, {56, 37}
        ), set())
        self.assertEqual(train.allowed_atomic_frontier_event_bits(
            train.FULLGAME_RECEIVE_POKEDEX_QUEST_PHASE, {37}
        ), set())

    def test_drill_fraction_pins_a_quarter_of_workers(self):
        drill_ids = [
            env_id for env_id in range(96)
            if train.swarm_worker_is_drill(env_id, 0.25)
        ]
        self.assertEqual(len(drill_ids), 24)
        self.assertTrue(all(env_id % 4 == 0 for env_id in drill_ids))
        self.assertFalse(train.swarm_worker_is_drill(1, 0.25))

    def test_production_topology_has_twenty_four_drills(self):
        drills = [
            env_id for env_id in range(48)
            if train.swarm_worker_is_drill(env_id, 0.5)
        ]
        self.assertEqual(len(drills), 24)
        self.assertEqual(drills, list(range(0, 48, 2)))

    def test_worker_topology_prunes_only_retired_viewer_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for env_id in (0, 47, 48, 95):
                (root / f'live_agent_positions.env{env_id}.json').write_text(
                    '{}', encoding='utf-8'
                )
            unrelated = root / 'live_agent_positions.envx.json'
            unrelated.write_text('{}', encoding='utf-8')

            removed = train.prune_inactive_stream_position_files(48, root)

            self.assertEqual(removed, [
                'live_agent_positions.env48.json',
                'live_agent_positions.env95.json',
            ])
            self.assertTrue((root / 'live_agent_positions.env0.json').exists())
            self.assertTrue((root / 'live_agent_positions.env47.json').exists())
            self.assertTrue(unrelated.exists())
        self.assertFalse(train.swarm_worker_is_drill(0, 0.0))

    def test_mastery_gate_requires_fast_drill_clears_without_plateau_rule(self):
        with mock.patch.dict(os.environ, {"POKEMON_ADVANCE_IF_NO_IMPROVEMENT_EPISODES": "0"}):
            self.assertFalse(
                train.swarm_mastery_allows_promotion([100, 200], 768, 8, 0.8)
            )
            self.assertFalse(
                train.swarm_mastery_allows_promotion([900] * 8, 768, 8, 0.8)
            )
            self.assertTrue(
                train.swarm_mastery_allows_promotion([100] * 7 + [900], 768, 8, 0.8)
            )

    def test_mastery_plateau_requires_five_rotations_without_improvement(self):
        # Need an establishing best plus 5 non-improving clears.
        self.assertFalse(
            train.swarm_mastery_allows_promotion(
                [120, 110, 110, 110, 110], 768, 8, 0.8, no_improve_rotations=5
            )
        )
        self.assertFalse(
            train.swarm_mastery_allows_promotion(
                [120, 100, 100, 100, 100, 100], 768, 8, 0.8, no_improve_rotations=5
            )
        )
        self.assertTrue(
            train.swarm_mastery_allows_promotion(
                [120, 110, 110, 110, 110, 110, 110],
                768,
                8,
                0.8,
                no_improve_rotations=5,
            )
        )
        # Still improving inside the last window blocks promotion.
        self.assertFalse(
            train.swarm_mastery_allows_promotion(
                [120, 110, 109, 109, 109, 109, 109],
                768,
                8,
                0.8,
                no_improve_rotations=5,
            )
        )
        self.assertTrue(
            train.swarm_mastery_no_improve_plateau(
                [200, 150, 150, 150, 150, 150, 150], 5
            )
        )

    def test_plateau_rule_accepts_five_non_improving_fast_rotations(self):
        # Eight identical fast clears contain an establishing best followed
        # by at least five rotations that do not improve it.
        fast = [100] * 8
        self.assertTrue(
            train.swarm_mastery_allows_promotion(
                fast, 768, 8, 0.8, no_improve_rotations=5
            )
        )

    def test_mastery_frontier_advances_exactly_one_quest_per_rotation(self):
        self.assertEqual(train.swarm_mastery_candidate_relation(1, 0), 'next')
        self.assertEqual(
            train.swarm_mastery_candidate_relation(0, 0), 'same_or_behind'
        )
        self.assertEqual(train.swarm_mastery_candidate_relation(2, 0), 'skipped')
        self.assertEqual(
            train.swarm_mastery_candidate_relation(
                train.FULLGAME_ROUTE25_QUEST_PHASE,
                train.FULLGAME_CERULEAN_RIVAL_QUEST_PHASE,
                35,
            ),
            'next',
        )
        self.assertEqual(
            train.swarm_mastery_candidate_relation(
                train.FULLGAME_ROUTE25_QUEST_PHASE,
                train.FULLGAME_CERULEAN_RIVAL_QUEST_PHASE,
                3,
            ),
            'skipped',
        )
        self.assertTrue(train.swarm_mastery_rotation_completed(True, 4, 4))
        self.assertFalse(train.swarm_mastery_rotation_completed(True, 5, 4))
        self.assertFalse(train.swarm_mastery_rotation_completed(False, 4, 4))

    def test_mastery_completion_is_latched_until_episode_reset(self):
        self.assertIsNone(
            train.latch_swarm_mastery_completed_phase(None, None)
        )
        self.assertEqual(
            train.latch_swarm_mastery_completed_phase(None, 113), 113
        )
        # Later steps do not report a fresh completion, but the original
        # phase must remain visible while the ROM settles for publication.
        self.assertEqual(
            train.latch_swarm_mastery_completed_phase(113, None), 113
        )
        # Defensive: a later phase cannot replace this rotation's identity.
        self.assertEqual(
            train.latch_swarm_mastery_completed_phase(113, 114), 113
        )

    def test_early_budget_is_tighter_than_default(self):
        early = train.swarm_drill_step_budget(
            0, 2048, 768, train.FULLGAME_EARLY_GAME_QUEST_PHASES
        )
        later = train.swarm_drill_step_budget(
            train.FULLGAME_BROCK_QUEST_PHASE,
            2048,
            768,
            train.FULLGAME_EARLY_GAME_QUEST_PHASES,
        )
        self.assertEqual(early, 768)
        self.assertEqual(later, 2048)

    def test_drill_completion_window_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "restart_start.swarm_mastery.json"
            samples = None
            for step in range(20):
                samples = train.record_swarm_drill_completion(path, 0, 100 + step, 16)
            self.assertEqual(len(samples), 16)
            self.assertEqual(samples[0], 104)
            self.assertEqual(
                train.load_swarm_mastery_window(path, 0),
                samples,
            )
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['schema'], 2)
            self.assertEqual(payload['best_steps']['0'], 100)

    def test_mastery_writer_waits_through_completion_burst(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "restart_start.swarm_mastery.json"
            acquire = mock.Mock(side_effect=[False] * 8 + [True])
            with mock.patch.object(
                train, 'try_acquire_swarm_frontier_write_lock', acquire
            ), mock.patch.object(train.time, 'sleep') as sleep:
                samples = train.record_swarm_drill_completion(path, 120, 6)

            self.assertEqual(samples, [6])
            self.assertEqual(acquire.call_count, 9)
            self.assertEqual(sleep.call_count, 8)

    def test_tree_cut_attempts_reset_with_each_frontier_episode(self):
        env = object.__new__(train.PokemonYellowEnv)
        env._progression_tree_cut_attempts = {(129, 134, (4, 4))}
        env._forced_interaction_face_key = (134, (4, 4))
        env._reset_per_episode_gates()
        self.assertEqual(env._progression_tree_cut_attempts, set())
        self.assertIsNone(env._forced_interaction_face_key)

    def test_cut_helper_unwinds_residual_party_menu(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.env_id = 1
        env.quest_phase = train.FULLGAME_POST_ERIKA_HEAL_QUEST_PHASE
        env.same_position_step_count = 0
        env._progression_tree_cut_attempts = set()

        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 134,
            train.ADDR_POS_A: 4,
            train.ADDR_POS_B: 2,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_TEXT_BOX: 0,
            train.ADDR_FONT_LOADED: 0,
        })
        env.pyboy = mock.Mock(memory=memory)
        env._party_index_with_move = mock.Mock(return_value=1)
        taps = []

        def tap(button, press_ticks=8, release_ticks=32):
            taps.append((button, press_ticks, release_ticks))
            if len(taps) == 7:
                # The helper starts from a clear overworld gate, but the live
                # second-tree residue sets the text byte during its final A.
                memory[train.ADDR_TEXT_BOX] = 1

        env._tap_scripted_button = tap

        self.assertTrue(env._try_cut_progression_tree())
        self.assertEqual(taps[-4:], [('b', 8, 120)] * 4)
        self.assertIn((env.quest_phase, 134, (4, 2)), env._progression_tree_cut_attempts)

    def test_silph_grind_cuts_respawned_route16_tree(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.env_id = 1
        env.quest_phase = min(train.FULLGAME_SILPH_GRIND_PHASES)
        env.same_position_step_count = 8
        env._progression_tree_cut_attempts = set()

        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 27,
            train.ADDR_POS_A: 10,
            train.ADDR_POS_B: 34,
            train.ADDR_BATTLE_FLAG: 0,
            # Reproduces the Route 16 joy-ignore latch after bumping the tree.
            train.ADDR_TEXT_BOX: 1,
            train.ADDR_FONT_LOADED: 0,
        })
        env.pyboy = mock.Mock(memory=memory)
        env._party_index_with_move = mock.Mock(return_value=1)
        env._tap_scripted_button = mock.Mock()

        self.assertTrue(env._try_cut_progression_tree())
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 0)
        self.assertIn(
            (env.quest_phase, 27, (10, 34)),
            env._progression_tree_cut_attempts,
        )

    def test_fly_house_hm_giver_tile_is_interaction_only(self):
        matching = [
            rule for rule in train.ROUTE16_FLY_HOUSE_ACTION_GUIDANCE
            if rule.get('map') == 188 and rule.get('target') == (4, 2)
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]['action'], 4)
        self.assertEqual(matching[0]['face_action'], 0)
        self.assertIn(1230, matching[0]['unless_event_bits'])

    def test_post_hm02_route_cuts_tree_from_north_and_is_contiguous(self):
        phase = train.FULLGAME_POST_HM02_HEAL_QUEST_PHASE
        env = object.__new__(train.PokemonYellowEnv)
        env.env_id = 1
        env.quest_phase = phase
        env.same_position_step_count = 8
        env._progression_tree_cut_attempts = set()

        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 27,
            train.ADDR_POS_A: 8,
            train.ADDR_POS_B: 34,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_TEXT_BOX: 1,
            train.ADDR_FONT_LOADED: 0,
        })
        env.pyboy = mock.Mock(memory=memory)
        env._party_index_with_move = mock.Mock(return_value=1)
        env._tap_scripted_button = mock.Mock()
        self.assertTrue(env._try_cut_progression_tree())

        route16 = [
            step for step in train.FLY_HOUSE_TO_CELADON_CENTER_STEPS
            if step[0] == 27 and step[2] >= 24
        ]
        actions = {(step[1], step[2], step[3]) for step in route16}
        self.assertIn((8, 34, 1), actions)
        self.assertIn((9, 34, 1), actions)
        self.assertIn((10, 34, 3), actions)

    def test_post_hm02_heal_uses_celadon_nurse_guidance(self):
        self.assertIn(
            train.FULLGAME_POST_HM02_HEAL_QUEST_PHASE,
            train.CELADON_CENTER_HEAL_QUEST_PHASES,
        )
        phase_rules = [
            rule for rule in train.CELADON_CENTER_ACTION_GUIDANCE
            if rule.get('map') == 133
        ]
        self.assertTrue(any(rule.get('target') == (3, 3) for rule in phase_rules))

    def test_post_charizard_uses_fly_to_return_to_lavender(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_POST_CHARIZARD_LAVENDER_QUEST_PHASE

        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            # Live training completes on Route 4; this origin must not be
            # rejected by the legacy Route 7/Celadon recovery allowlist.
            train.ADDR_MAP_ID: 15,
            train.ADDR_BATTLE_FLAG: 0,
        })
        env.pyboy = mock.Mock(memory=memory)
        env._party_index_with_move = mock.Mock(return_value=2)
        taps = []

        def tap(button, press_ticks=8, release_ticks=32):
            taps.append((button, press_ticks, release_ticks))
            if len(taps) == 10:
                memory[train.ADDR_MAP_ID] = 4

        env._tap_scripted_button = tap

        self.assertTrue(env._try_fly_to_lavender_after_charizard())
        self.assertEqual(memory[train.ADDR_START_SAVED_MENU_ITEM], 1)
        self.assertEqual(memory[train.ADDR_PARTY_SAVED_MENU_ITEM], 2)
        self.assertEqual([button for button, _, _ in taps], [
            'b', 'start', 'a', 'a', 'a',
            'up', 'up', 'up', 'up', 'a',
        ])

    def test_post_charizard_nurse_frontier_routes_to_pokemon_tower(self):
        self.assertEqual(
            train.LAVENDER_NURSE_TO_TOWER_STEPS[:5],
            train.LAVENDER_NURSE_TO_CENTER_DOOR_STEPS,
        )
        self.assertEqual(
            train.LAVENDER_NURSE_TO_TOWER_STEPS[-1],
            (4, 6, 14, 0, 142),
        )
        nurse_rules = [
            rule for rule in train.LAVENDER_CENTER_TO_TOWER_ACTION_GUIDANCE
            if rule.get('map') == 141 and rule.get('target') == (3, 3)
        ]
        self.assertEqual(len(nurse_rules), 1)
        self.assertEqual(nurse_rules[0]['action'], 1)

    def test_pokemon_tower_first_floor_route_reaches_second_floor(self):
        steps = train.POKEMON_TOWER_1F_TO_2F_STEPS
        self.assertEqual(steps[0][:4], (142, 17, 10, 0))
        self.assertEqual(steps[-1], (142, 10, 18, 0, 143))
        self.assertEqual(len(steps), 16)
        phase = train.FULLGAME_POKEMON_TOWER_2F_QUEST_PHASE
        rules = [
            rule for rule in train.POKEMON_TOWER_1F_TO_2F_ACTION_GUIDANCE
            if rule.get('target') == (17, 10)
        ]
        self.assertEqual(len(rules), 1)
        self.assertTrue(rules[0]['force_action'])
        self.assertTrue(rules[0]['verified_route'])
        self.assertEqual(phase, 193)

    def test_pokemon_tower_rival_approach_triggers_from_north(self):
        steps = train.POKEMON_TOWER_RIVAL_APPROACH_STEPS
        self.assertEqual(steps[0][:4], (143, 9, 18, 0))
        self.assertEqual(steps[-1][:4], (143, 5, 15, 0))
        self.assertEqual(len(steps), 8)
        trigger = train.POKEMON_TOWER_RIVAL_APPROACH_ACTION_GUIDANCE[-1]
        self.assertEqual(trigger['target'], (5, 15))
        self.assertEqual(trigger['action'], 0)
        self.assertEqual(trigger['unless_event_bits'], (239,))
        post_rival_phase = train.FULLGAME_POKEMON_TOWER_3F_QUEST_PHASE
        self.assertEqual(
            train.swarm_required_party_hp_fraction(0.4, post_rival_phase),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_6F_QUEST_PHASE,
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_6F_QUEST_PHASE,
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_5F_QUEST_PHASE,
            ),
            0.4,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_7F_QUEST_PHASE,
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_7F_QUEST_PHASE,
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_ROCKET_QUEST_PHASE,
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_ROCKET_QUEST_PHASE + 1,
            ),
            0.4,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_ROCKET_QUEST_PHASE,
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_POKEMON_TOWER_ROCKET_QUEST_PHASE + 1,
            ),
            0.4,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(0.4, post_rival_phase),
            0.0,
        )

    def test_fainted_active_mon_switch_recovers_wild_marowak_battle(self):
        post_rival_phase = train.FULLGAME_POKEMON_TOWER_3F_QUEST_PHASE
        self.assertIsNone(train.fainted_active_mon_switch_action(False, 0, 0))
        self.assertIsNone(train.fainted_active_mon_switch_action(True, 1, 0))
        self.assertEqual(train.fainted_active_mon_switch_action(True, 0, 220), 'down')
        self.assertEqual(train.fainted_active_mon_switch_action(True, 0, 221), 'a')
        phase = train.FULLGAME_POKEMON_TOWER_7F_QUEST_PHASE
        self.assertTrue(train.pokemon_tower_marowak_battle_active(
            147, phase, 16, 11, True, False,
        ))
        self.assertTrue(train.pokemon_tower_marowak_loss_overworld(
            147, phase, 16, 11, 0, False,
        ))
        self.assertFalse(train.pokemon_tower_marowak_loss_overworld(
            147, phase, 16, 11, 1, False,
        ))
        self.assertFalse(
            train.trainer_battle_allows_electric(147, phase, 16, 11)
        )
        retry = train.POKEMON_TOWER_6F_MAROWAK_TO_7F_ACTION_GUIDANCE[-3]
        self.assertEqual(
            (retry['map'], retry['target'], retry['action']),
            (147, (16, 11), 0),
        )
        self.assertEqual(retry['unless_event_bits'], (271,))
        self.assertEqual(
            [rule['target'] for rule in train.POKEMON_TOWER_6F_MAROWAK_TO_7F_ACTION_GUIDANCE[-3:]],
            [(16, 11), (15, 11), (15, 10)],
        )
        self.assertTrue(
            train.should_restore_tower_rival_frontier_resources(post_rival_phase)
        )
        self.assertFalse(
            train.should_restore_tower_rival_frontier_resources(
                post_rival_phase + 1
            )
        )

        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = post_rival_phase
        memory = Memory({
            train.ADDR_MAP_ID: 143,
            train.ADDR_TEXT_BOX: 0,
            train.ADDR_PARTY_SIZE: 2,
            train.ADDR_PARTY_SPECIES_LIST: train.GEN1_PIKACHU_SPECIES_ID,
            train.ADDR_PARTY_SPECIES_LIST + 1: train.GEN1_CHARIZARD_SPECIES_ID,
            train.PARTY_SPECIES_ADDRS[0]: train.GEN1_PIKACHU_SPECIES_ID,
            train.PARTY_SPECIES_ADDRS[1]: train.GEN1_CHARIZARD_SPECIES_ID,
            train.PARTY_CUR_HP_ADDRS[0][0]: 0,
            train.PARTY_CUR_HP_ADDRS[0][1]: 0,
            train.PARTY_CUR_HP_ADDRS[1][0]: 0,
            train.PARTY_CUR_HP_ADDRS[1][1]: 121,
        })
        env.pyboy = mock.Mock(memory=memory)
        env._effective_battle_flag = mock.Mock(return_value=0)
        env._lead_moveset_snapshot = object()
        self.assertTrue(env._restore_pikachu_lead_after_charizard())
        self.assertEqual(
            memory[train.ADDR_PARTY_SPECIES_LIST],
            train.GEN1_CHARIZARD_SPECIES_ID,
        )
        self.assertIsNone(env._lead_moveset_snapshot)

        # Entering 6F advances to the 7F objective before the Marowak leg.
        # Keep the same healthy survivor in front instead of swapping the
        # fainted Pikachu back into slot zero on the next environment step.
        env.quest_phase = train.FULLGAME_POKEMON_TOWER_7F_QUEST_PHASE
        self.assertTrue(env._restore_pikachu_lead_after_charizard())
        self.assertEqual(
            memory[train.ADDR_PARTY_SPECIES_LIST],
            train.GEN1_CHARIZARD_SPECIES_ID,
        )
        env.quest_phase = train.FULLGAME_POKEMON_TOWER_ROCKET_QUEST_PHASE
        self.assertTrue(env._restore_pikachu_lead_after_charizard())
        self.assertEqual(
            memory[train.ADDR_PARTY_SPECIES_LIST],
            train.GEN1_CHARIZARD_SPECIES_ID,
        )

    def test_pokemon_tower_purified_zone_restores_party_resources(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_POKEMON_TOWER_6F_QUEST_PHASE
        memory = Memory({
            train.ADDR_MAP_ID: 146,
            train.ADDR_POS_A: 9,
            train.ADDR_POS_B: 10,
            train.ADDR_PARTY_SIZE: 2,
            train.PARTY_CUR_HP_ADDRS[0][1]: 1,
            train.PARTY_MAX_HP_ADDRS[0][1]: 30,
            train.PARTY_CUR_HP_ADDRS[1][1]: 40,
            train.PARTY_MAX_HP_ADDRS[1][1]: 121,
            train.FIRST_PARTY_SPECIES_ADDR + 4: 8,
            train.FIRST_PARTY_SPECIES_ADDR + train.PARTY_DATA_STRIDE + 4: 64,
        })
        env.pyboy = mock.Mock(memory=memory)
        env._effective_battle_flag = mock.Mock(return_value=0)

        self.assertTrue(env._restore_tower_purification_heal())
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 30)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[1][1]], 121)
        self.assertEqual(memory[train.FIRST_PARTY_SPECIES_ADDR + 4], 0)
        self.assertEqual(
            memory[train.FIRST_PARTY_SPECIES_ADDR + train.PARTY_DATA_STRIDE + 4],
            0,
        )
        self.assertTrue(
            train.should_restore_tower_purification_frontier_resources(
                train.FULLGAME_POKEMON_TOWER_ROCKET_QUEST_PHASE
            )
        )

        # Re-entering 5F after reaching 6F is the second pass. Its adjacent
        # confirmed tile at viewer (398,209), local (y=8,x=11), must honor the
        # same full HealParty contract.
        env.quest_phase = train.FULLGAME_POKEMON_TOWER_7F_QUEST_PHASE
        memory[train.ADDR_POS_A] = 8
        memory[train.ADDR_POS_B] = 11
        memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 2
        memory[train.FIRST_PARTY_SPECIES_ADDR + 4] = 8
        self.assertTrue(env._restore_tower_purification_heal())
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 30)
        self.assertEqual(memory[train.FIRST_PARTY_SPECIES_ADDR + 4], 0)
        self.assertTrue(env._tower_5f_second_pass_healed)

    def test_pokemon_tower_5f_both_passes_route_through_confirmed_heal_pad(self):
        # The normal ascent now steps on viewer (397,210), local (y=9,x=10),
        # rather than merely skirting the southeast corner of the pad.
        steps = train.POKEMON_TOWER_5F_TO_6F_STEPS
        self.assertIn((146, 9, 10), [step[:3] for step in steps])
        pad_index = [step[:3] for step in steps].index((146, 9, 10))
        self.assertLess(pad_index, len(steps) - 1)
        self.assertEqual(steps[-1], (146, 8, 18, 1, 147))

        phase = train.FULLGAME_POKEMON_TOWER_7F_QUEST_PHASE
        inbound = {
            (8, 18): 'up',
            (7, 18): 'left',
            (7, 13): 'down',
            (8, 13): 'down',
            (9, 13): 'left',
            (9, 11): 'left',
        }
        for position, expected in inbound.items():
            self.assertEqual(
                train.pokemon_tower_5f_post_heal_action(
                    146, phase, *position, 0, second_pass_healed=False,
                ),
                expected,
            )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 9, 10, 0, second_pass_healed=True,
            ),
            'right',
        )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 8, 18, 0, second_pass_healed=True,
            ),
            'down',
        )

    def test_pokemon_tower_7f_rocket_trigger_clears_stair_latch(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_POKEMON_TOWER_ROCKET_QUEST_PHASE
        memory = Memory({
            train.ADDR_MAP_ID: 148,
            train.ADDR_POS_A: 16,
            train.ADDR_POS_B: 9,
            train.ADDR_TEXT_BOX: 0,
        })
        env.pyboy = mock.Mock(memory=memory)
        env._effective_battle_flag = mock.Mock(return_value=0)
        env._event_flag_is_set = mock.Mock(return_value=False)
        taps = []

        def tap(button, press_ticks=8, release_ticks=32):
            taps.append((button, press_ticks, release_ticks))
            if len(taps) == 6:
                memory[train.ADDR_POS_A] = 12
                memory[train.ADDR_POS_B] = 10
                memory[train.ADDR_TEXT_BOX] = 252

        env._tap_scripted_button = tap
        self.assertTrue(env._try_start_tower_7f_rocket_battle())
        self.assertEqual(
            [button for button, _, _ in taps],
            ['b', 'right', 'up', 'up', 'up', 'up'],
        )
        self.assertEqual(taps[0], ('b', 8, 64))

    def test_post_rival_frontier_joins_existing_third_floor_route(self):
        steps = train.POKEMON_TOWER_POST_RIVAL_TO_3F_STEPS
        self.assertEqual(steps[0][:4], (143, 5, 14, 1))
        self.assertEqual(steps[3][:4], (143, 7, 15, 3))
        self.assertEqual(steps[4][:4], (143, 7, 16, 0))
        self.assertEqual(steps[5][:4], (143, 6, 16, 0))
        self.assertIn((143, 9, 16, 0, None), steps)
        self.assertEqual(steps[-1], (143, 9, 4, 2, 144))
        phase = train.FULLGAME_POKEMON_TOWER_3F_QUEST_PHASE
        self.assertEqual(
            train.pokemon_tower_post_rival_forced_action(143, phase, 7, 16),
            0,
        )
        self.assertEqual(
            train.pokemon_tower_post_rival_forced_action(143, phase, 7, 15),
            3,
        )
        self.assertIsNone(
            train.pokemon_tower_post_rival_forced_action(143, phase + 1, 7, 16)
        )

        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = phase
        memory = Memory({
            train.ADDR_MAP_ID: 143,
            train.ADDR_POS_A: 5,
            train.ADDR_POS_B: 14,
            train.ADDR_BATTLE_FLAG: 0,
        })
        env.pyboy = mock.Mock(memory=memory)
        env._event_flag_is_set = mock.Mock(side_effect=lambda bit: bit == 239)
        taps = []

        def tap(button, press_ticks=8, release_ticks=32):
            taps.append((button, press_ticks, release_ticks))
            if len(taps) == 24:
                memory[train.ADDR_MAP_ID] = 144

        env._tap_scripted_button = tap
        self.assertTrue(env._try_cross_tower_post_rival_route())
        self.assertEqual(taps[0], ('b', 8, 64))
        self.assertEqual(len(taps), 24)
        self.assertEqual([button for button, _, _ in taps[1:3]], ['down', 'down'])
        self.assertEqual([button for button, _, _ in taps[-2:]], ['left', 'left'])

    def test_game_corner_hideout_entry_uses_open_row_and_map_completion(self):
        steps = train.GAME_CORNER_HIDEOUT_ENTRY_STEPS
        self.assertEqual(steps[0][:4], (135, 5, 9, 3))
        self.assertTrue(all(step[1] == 5 for step in steps))
        self.assertEqual(steps[-1], (135, 5, 17, 0, 199))
        waypoint = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_ENTER_HIDEOUT_QUEST_PHASE
        ]
        self.assertEqual(waypoint.get('map'), 199)
        self.assertNotIn('event_bit', waypoint)
        phase = train.FULLGAME_ENTER_HIDEOUT_QUEST_PHASE
        self.assertTrue(train.fullgame_frontier_position_allowed(
            phase, 135, 5, 17
        ))
        self.assertTrue(train.fullgame_frontier_position_allowed(
            phase, 199, 2, 21
        ))

    def test_rocket_hideout_descent_restores_verified_spinner_route(self):
        steps = train.ROCKET_HIDEOUT_B3F_WEST_APPROACH_STEPS
        self.assertEqual(steps[0], (201, 6, 25, 3, None))
        self.assertEqual(steps[-1], (201, 19, 19, 0, 202))
        self.assertIn((201, 11, 10, 1, None), steps)
        self.assertIn((201, 13, 14, 2, None), steps)
        self.assertIn((201, 18, 10, 3, None), steps)
        self.assertIn((201, 22, 15, 2, None), steps)
        self.assertEqual(len(steps), 58)

        b3f_rules = [
            rule for rule in train.ROCKET_HIDEOUT_B3F_WEST_ACTION_GUIDANCE
            if rule.get('map') == 201
        ]
        self.assertEqual(len(b3f_rules), len(steps))
        self.assertTrue(all(rule.get('force_action') for rule in b3f_rules))
        self.assertTrue(all(rule.get('verified_route') for rule in b3f_rules))
        self.assertEqual(b3f_rules[-1].get('wait_ticks'), 512)
        self.assertEqual(b3f_rules[-1].get('destination_map'), 202)
        self.assertIn(
            train.FULLGAME_HIDEOUT_B4F_QUEST_PHASE,
            train.FULLGAME_HIDEOUT_DESCENT_PHASES,
        )

    def test_rocket_hideout_lift_key_return_is_forward_only_and_recovers_softlocks(self):
        phase = train.FULLGAME_ENTER_HIDEOUT_LIFT_QUEST_PHASE
        self.assertEqual(train.ROCKET_HIDEOUT_B4F_KEY_RETURN_STEPS[0][:4], (202, 3, 10, 3))
        self.assertEqual(train.ROCKET_HIDEOUT_B4F_KEY_RETURN_STEPS[-1][-1], 201)
        self.assertEqual(train.ROCKET_HIDEOUT_B3F_KEY_RETURN_STEPS[0][:4], (201, 19, 19, 2))
        self.assertEqual(train.ROCKET_HIDEOUT_B3F_KEY_RETURN_STEPS[-1], (201, 6, 24, 3, 200))
        self.assertEqual(train.ROCKET_HIDEOUT_B2F_KEY_RETURN_STEPS[-1][-1], 199)
        self.assertEqual(train.ROCKET_HIDEOUT_B1F_LIFT_APPROACH_STEPS[0][:4], (199, 3, 23, 1))
        self.assertEqual(train.ROCKET_HIDEOUT_B1F_LIFT_APPROACH_STEPS[-1][:4], (199, 7, 25, 1))
        self.assertEqual(train.ROCKET_HIDEOUT_B1F_TO_LIFT_STEPS[0][:4], (199, 8, 25, 1))
        self.assertEqual(train.ROCKET_HIDEOUT_B1F_TO_LIFT_STEPS[-2], (199, 18, 25, 1, None))
        self.assertEqual(train.ROCKET_HIDEOUT_B1F_TO_LIFT_STEPS[-1], (199, 19, 25, 1, 203))
        self.assertTrue(all(
            rule.get('require_event_bits') == (train.EVENT_BEAT_ROCKET_HIDEOUT_1_TRAINER_0,)
            for rule in train.ROCKET_HIDEOUT_B1F_TO_LIFT_ACTION_GUIDANCE
        ))
        self.assertEqual(train.FULLGAME_HIDEOUT_LIFT_RETURN_PHASES, frozenset({phase}))

        self.assertFalse(train.fullgame_frontier_position_allowed(phase, 202, 3, 10))
        self.assertFalse(train.fullgame_frontier_position_allowed(phase, 202, 2, 10))
        self.assertTrue(train.fullgame_frontier_position_allowed(phase, 202, 3, 11))

        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = phase
        memory = {
            train.ADDR_MAP_ID: 202,
            train.ADDR_POS_A: 3,
            train.ADDR_POS_B: 10,
            train.ADDR_BATTLE_FLAG: 0,
        }
        for base in (
            train.ADDR_PIKACHU_SPRITE_STATE_DATA_1,
            train.ADDR_PIKACHU_SPRITE_STATE_DATA_2,
        ):
            memory.update({base + offset: 0xFF for offset in range(16)})
        env.pyboy = mock.Mock(memory=memory)
        env._bag_contains_item = lambda item_id: item_id == 0x4A
        self.assertTrue(env._clear_pikachu_follower_route_softlock())
        self.assertTrue(all(
            memory[base + offset] == 0
            for base in (
                train.ADDR_PIKACHU_SPRITE_STATE_DATA_1,
                train.ADDR_PIKACHU_SPRITE_STATE_DATA_2,
            )
            for offset in range(16)
        ))

        memory[train.ADDR_MAP_ID] = 201
        memory[train.ADDR_POS_A] = 19
        memory[train.ADDR_POS_B] = 19
        event_addr = train.ADDR_EVENT_FLAGS_START + train.EVENT_ENTERED_ROCKET_HIDEOUT // 8
        memory[event_addr] = 0
        trainer4_addr = train.ADDR_EVENT_FLAGS_START + train.EVENT_BEAT_ROCKET_HIDEOUT_1_TRAINER_4 // 8
        memory[trainer4_addr] = 0
        self.assertTrue(env._recover_rocket_hideout_b1f_door())
        self.assertEqual((memory[train.ADDR_POS_A], memory[train.ADDR_POS_B]), (19, 19))
        self.assertTrue(memory[event_addr] & (1 << (train.EVENT_ENTERED_ROCKET_HIDEOUT % 8)))
        self.assertTrue(memory[trainer4_addr] & (1 << (train.EVENT_BEAT_ROCKET_HIDEOUT_1_TRAINER_4 % 8)))

        selector = object.__new__(train.PokemonYellowEnv)
        selector.quest_phase = train.FULLGAME_ROCKET_JESSIE_JAMES_QUEST_PHASE
        selector.same_position_step_count = 1
        selector.steps = 100
        selector.pyboy = mock.Mock(memory={
            train.ADDR_MAP_ID: 203,
            train.ADDR_POS_A: 2,
            train.ADDR_POS_B: 1,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_TEXT_BOX: 13,
        })
        tapped = []
        selector._tap_scripted_button = lambda button, *args: tapped.append(button)
        self.assertTrue(selector._try_select_rocket_hideout_b4f())
        self.assertEqual(
            tapped,
            ['up', 'a', 'down', 'down', 'a', 'down', 'right', 'up', 'up'],
        )
        self.assertEqual(selector.pyboy.memory[train.ADDR_TEXT_BOX], 0)

        # A failed menu interaction retries by elapsed steps even when the
        # player oscillates and same_position_step_count never grows.
        selector.pyboy.memory[train.ADDR_POS_A] = 2
        selector.pyboy.memory[train.ADDR_POS_B] = 1
        selector.steps = 110
        self.assertFalse(selector._try_select_rocket_hideout_b4f())
        selector.steps = 132
        self.assertTrue(selector._try_select_rocket_hideout_b4f())

        selector._reset_per_episode_gates()
        self.assertEqual(selector._rocket_elevator_b4f_attempts, set())
        self.assertEqual(selector._rocket_elevator_b4f_last_attempt_step, -1000000)

        # Both post-Scope approaches enter B4F's elevator through its actual
        # doorway at (15, 25), then step Down. The prior Up at (15, 24)
        # oscillated every live worker between y=14 and y=15 forever.
        scope_steps = train.ROCKET_SCOPE_TO_ELEVATOR_STEPS
        recovery_start = next(
            i for i, item in enumerate(scope_steps) if item[:3] == (202, 22, 9)
        )
        self.assertEqual(scope_steps[recovery_start - 2], (202, 15, 24, 3, None))
        self.assertEqual(scope_steps[recovery_start - 1], (202, 15, 25, 1, 203))
        self.assertEqual(scope_steps[-2], (202, 15, 24, 3, None))
        self.assertEqual(scope_steps[-1], (202, 15, 25, 1, 203))

    def test_charmander_training_recovers_route7_gate_toward_cerulean(self):
        steps = train.CHARMANDER_ROUTE7_GATE_RECOVERY_STEPS
        gate_rules = {
            (item[1], item[2]): item for item in steps if item[0] == 76
        }
        for x in range(1, 6):
            self.assertEqual(gate_rules[(3, x)][3], 2)
        self.assertEqual(gate_rules[(3, 0)][3], 1)
        self.assertEqual(gate_rules[(4, 0)][3], 3)
        self.assertEqual(gate_rules[(4, 4)], (76, 4, 4, 3, 18))
        self.assertEqual(steps[-2], (18, 10, 18, 3, None))
        self.assertEqual(steps[-1], (18, 10, 19, 3, 10))

        saffron_steps = train.SAFFRON_TO_CERULEAN_CENTER_STEPS
        self.assertEqual(
            saffron_steps[:8],
            tuple((10, y, 0, 0, None) for y in range(18, 10, -1)),
        )
        self.assertEqual(saffron_steps[8], (10, 10, 0, 3, None))

        # Explorer guidance must never be attached to training workers; that
        # duplicate previously won before the northbound Cerulean rules.
        source = Path(train.__file__).read_text(encoding='utf-8')
        assembly = source[source.index("for item in CELADON_VENDING_APPROACH_ACTION_GUIDANCE"):]
        assembly = assembly[:assembly.index("for item in INDIGO_PLATEAU_CENTER_ACTION_GUIDANCE")]
        self.assertEqual(
            assembly.count("for item in SAFFRON_EXPLORER_ENTRY_ACTION_GUIDANCE"),
            1,
        )
        self.assertIn(
            "for item in CHARMANDER_ROUTE7_GATE_RECOVERY_ACTION_GUIDANCE",
            assembly,
        )

    def test_celadon_vending_route_survives_rooftop_phase_handoff(self):
        phase = train.FULLGAME_GUARD_DRINK_QUEST_PHASE
        self.assertEqual(
            train.CELADON_VENDING_APPROACH_STEPS,
            (
                (126, 1, 12, 1, None),
                (126, 3, 15, 2, None),
                (126, 3, 14, 2, None),
                (126, 3, 13, 2, None),
            ),
        )
        self.assertIsNone(train.celadon_vending_machine_action(
            126, 2, 12, phase, 0, False,
        ))
        self.assertEqual(train.celadon_vending_machine_action(
            126, 3, 12, phase, 0, False,
        ), 'up')
        self.assertEqual(train.celadon_vending_machine_action(
            126, 3, 12, phase, 0, True,
        ), 'a')
        self.assertIsNone(train.celadon_vending_machine_action(
            126, 3, 12, phase, 1, True,
        ))

    def test_quest_completion_log_includes_segment_steps(self):
        line = train.fullgame_quest_completion_log_line(
            7, 274, 31, 146, 302, 'find_rocket_hideout',
            'enter_rocket_hideout', 135, 5, 9, True,
        )
        self.assertIn('step=274 phase_steps=31 phase=146/302', line)
        self.assertIn('completed=find_rocket_hideout', line)
        self.assertTrue(line.endswith(' drill'))

    def test_erika_exit_route_steps_onto_city_warp(self):
        gym_steps = [
            step for step in train.ERIKA_TO_CELADON_CENTER_STEPS
            if step[0] == 134
        ]
        self.assertEqual(gym_steps[-1], (134, 17, 4, 1, 6))
        coordinates = [step[:3] for step in gym_steps]
        self.assertEqual(len(coordinates), len(set(coordinates)))
        city_steps = [
            step for step in train.ERIKA_TO_CELADON_CENTER_STEPS
            if step[0] == 6
        ]
        self.assertIn((6, 34, 35, 0, None), city_steps)
        city_coordinates = [step[:3] for step in city_steps]
        self.assertEqual(len(city_coordinates), len(set(city_coordinates)))
        self.assertEqual(city_steps[-1], (6, 10, 41, 0, 133))
        phase = train.FULLGAME_POST_ERIKA_HEAL_QUEST_PHASE
        self.assertEqual(
            train.post_erika_city_dialogue_action(6, 23, 36, phase, 1), 'a'
        )
        self.assertIsNone(
            train.post_erika_city_dialogue_action(6, 23, 36, phase, 0)
        )
        self.assertIn(phase, train.CELADON_CENTER_HEAL_QUEST_PHASES)

    def test_twenty_rotation_plateau_resets_after_improvement(self):
        established = [100] + [100] * 20
        improving_late = [100] + [100] * 19 + [99]
        settled_after_improvement = improving_late + [99] * 20
        self.assertTrue(train.swarm_mastery_no_improve_plateau(established, 20))
        self.assertFalse(train.swarm_mastery_no_improve_plateau(improving_late, 20))
        self.assertTrue(
            train.swarm_mastery_no_improve_plateau(settled_after_improvement, 20)
        )

    def test_rng_grinds_use_success_count_instead_of_timing_plateau(self):
        slow_rng_clears = [9000, 12000, 7000, 15000, 11000]
        self.assertTrue(train.swarm_mastery_allows_promotion(
            slow_rng_clears,
            max_steps=768,
            min_samples=8,
            hit_rate=0.8,
            no_improve_rotations=20,
            reliability_only=True,
            reliability_min_samples=5,
        ))
        self.assertFalse(train.swarm_mastery_allows_promotion(
            slow_rng_clears[:4],
            max_steps=768,
            min_samples=8,
            hit_rate=0.8,
            no_improve_rotations=20,
            reliability_only=True,
            reliability_min_samples=5,
        ))
        self.assertTrue(train.FULLGAME_PRE_FOREST_GRIND_PHASES <=
                        train.FULLGAME_RNG_GRIND_QUEST_PHASES)
        self.assertTrue(train.FULLGAME_PEWTER_GRIND_PHASES <=
                        train.FULLGAME_RNG_GRIND_QUEST_PHASES)
        self.assertNotIn(train.FULLGAME_VIRIDIAN_HEAL_QUEST_PHASE,
                         train.FULLGAME_RNG_GRIND_QUEST_PHASES)
        self.assertNotIn(train.FULLGAME_EXIT_VIRIDIAN_FOREST_QUEST_PHASE,
                         train.FULLGAME_RNG_GRIND_QUEST_PHASES)
        self.assertNotIn(train.FULLGAME_BROCK_QUEST_PHASE,
                         train.FULLGAME_RNG_GRIND_QUEST_PHASES)

    def test_final_trainer_battles_use_reliability_mastery(self):
        expected = {
            train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE,
            train.FULLGAME_BEAT_LORELEI_QUEST_PHASE,
            train.FULLGAME_BEAT_BRUNO_QUEST_PHASE,
            train.FULLGAME_BEAT_AGATHA_QUEST_PHASE,
            train.FULLGAME_BEAT_LANCE_QUEST_PHASE,
            train.FULLGAME_BEAT_CHAMPION_QUEST_PHASE,
        }
        self.assertEqual(
            train.FULLGAME_BATTLE_RELIABILITY_MASTERY_PHASES,
            expected,
        )
        # Five successful clears are enough even when a fresh record would
        # otherwise reset the timing plateau to zero.
        samples = [798, 798, 798, 798, 429]
        for phase in expected:
            with self.subTest(phase=phase):
                self.assertTrue(train.swarm_mastery_allows_promotion(
                    samples,
                    max_steps=768,
                    min_samples=8,
                    hit_rate=0.8,
                    no_improve_rotations=20,
                    reliability_only=(
                        phase in train.FULLGAME_BATTLE_RELIABILITY_MASTERY_PHASES
                    ),
                    reliability_min_samples=5,
                    no_improve_count=0,
                ))

    def test_level_grind_completes_only_after_battle_teardown(self):
        waypoint = {'name': 'level_6', 'min_level': 6}
        memory = {
            train.ADDR_BATTLE_FLAG: 1,
            train.ADDR_FONT_LOADED: 1,
            train.ADDR_TEXT_BOX: 1,
        }
        self.assertFalse(train.quest_waypoint_matches(
            waypoint, 12, 10, 10, (), level=6, memory=memory,
            battle_active=True,
        ))
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_FONT_LOADED] = 0
        memory[train.ADDR_TEXT_BOX] = 0
        self.assertTrue(train.quest_waypoint_matches(
            waypoint, 12, 10, 10, (), level=6, memory=memory,
            battle_active=False,
        ))

    def test_quest_zero_mastery_session_quarantines_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"state")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"frontier")
            frontier_meta.write_text(
                json.dumps({"schema": 1, "quest_phase": 250}),
                encoding="utf-8",
            )
            mastery = state.with_name("restart_start.swarm_mastery.json")
            mastery.write_text(json.dumps({"schema": 1, "windows": {}}), encoding="utf-8")
            proven = state.with_name("restart_start.swarm_proven_routes.json")
            proven.write_text(
                json.dumps({"schema": 1, "routes": {"0": {"steps": 10}}}),
                encoding="utf-8",
            )

            moved = train.ensure_quest_zero_mastery_session(state)
            self.assertEqual(len(moved), 2)
            self.assertFalse(frontier_state.exists())
            self.assertFalse(frontier_meta.exists())
            self.assertTrue(mastery.exists())
            self.assertEqual(
                json.loads(mastery.read_text(encoding="utf-8")),
                {"schema": 1, "windows": {}},
            )
            self.assertTrue(proven.exists())
            marker = state.with_name("restart_start.quest_zero_mastery.json")
            self.assertTrue(marker.exists())

            # Rebuild a mid-session frontier; a second ensure must keep it.
            frontier_state.write_bytes(b"phase1")
            frontier_meta.write_text(
                json.dumps({"schema": 1, "quest_phase": 1}),
                encoding="utf-8",
            )
            self.assertEqual(train.ensure_quest_zero_mastery_session(state), [])
            self.assertTrue(frontier_state.exists())
            self.assertEqual(frontier_state.read_bytes(), b"phase1")

    def test_archived_mastery_merge_keeps_prior_proof_and_new_improvements(self):
        older = {
            "schema": 2,
            "windows": {"8": [202, 315, 288, 416], "9": [300, 310]},
            "best_steps": {"8": 202, "9": 300},
            "no_improve_counts": {"8": 21, "9": 1},
        }
        newer = {
            "schema": 2,
            "windows": {"8": [265, 532], "9": [250, 270, 280]},
            "best_steps": {"8": 265, "9": 250},
            "no_improve_counts": {"8": 1, "9": 2},
        }

        merged = train.merge_swarm_mastery_evidence(older, newer)

        self.assertEqual(merged["best_steps"], {"8": 202, "9": 250})
        self.assertEqual(merged["no_improve_counts"], {"8": 23, "9": 2})
        self.assertEqual(merged["windows"]["8"], [202, 315, 288, 416, 265, 532])
        self.assertEqual(merged["windows"]["9"], [300, 310, 250, 270, 280])

    def test_archived_mastery_merge_preserves_plateau_beyond_rolling_window(self):
        older = {
            "windows": {"30": [412] * 24},
            "best_steps": {"30": 412},
            "no_improve_counts": {"30": 23},
        }
        newer = {
            "windows": {"30": [442] * 32},
            "best_steps": {"30": 442},
            "no_improve_counts": {"30": 4487},
        }

        merged = train.merge_swarm_mastery_evidence(older, newer)

        self.assertEqual(merged["best_steps"]["30"], 412)
        self.assertEqual(merged["no_improve_counts"]["30"], 4511)
        self.assertEqual(len(merged["windows"]["30"]), 32)

    def test_quest_zero_reset_id_starts_exactly_one_new_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"state")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"old-frontier")
            frontier_meta.write_text(
                json.dumps({"schema": 4, "quest_phase": 301}),
                encoding="utf-8",
            )

            first = train.ensure_quest_zero_mastery_session(
                state, reset_id="attempt-20260826-a", now=1000
            )
            self.assertEqual(len(first), 2)
            marker_path = state.with_name("restart_start.quest_zero_mastery.json")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["schema"], 3)
            self.assertEqual(marker["reset_id"], "attempt-20260826-a")

            # A service restart during the same attempt must retain progress.
            frontier_state.write_bytes(b"phase17")
            frontier_meta.write_text(
                json.dumps({"schema": 4, "quest_phase": 17}),
                encoding="utf-8",
            )
            self.assertEqual(
                train.ensure_quest_zero_mastery_session(
                    state, reset_id="attempt-20260826-a", now=1010
                ),
                [],
            )
            self.assertEqual(frontier_state.read_bytes(), b"phase17")

            # A distinct operator-supplied ID intentionally begins a new run.
            second = train.ensure_quest_zero_mastery_session(
                state, reset_id="attempt-20260826-b", now=2000
            )
            self.assertEqual(len(second), 2)
            self.assertFalse(frontier_state.exists())
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["reset_id"], "attempt-20260826-b")

    def test_rapid_duplicate_quest_zero_reset_is_consumed_without_data_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"state")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"fresh-frontier")
            frontier_meta.write_text(
                json.dumps({"schema": 4, "quest_phase": 2}), encoding="utf-8"
            )
            marker = state.with_name("restart_start.quest_zero_mastery.json")
            marker.write_text(json.dumps({
                "schema": 3,
                "reset_id": "attempt-a",
                "started_ts": 1000,
                "quarantined": [],
            }), encoding="utf-8")

            self.assertEqual(train.ensure_quest_zero_mastery_session(
                state,
                reset_id="attempt-b",
                now=1100,
                cooldown_seconds=600,
            ), [])
            self.assertEqual(frontier_state.read_bytes(), b"fresh-frontier")
            saved = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(saved["reset_id"], "attempt-a")
            self.assertEqual(saved["ignored_reset_ids"][0]["reset_id"], "attempt-b")

            # The manager environment can retain the rejected ID forever;
            # later restarts must not turn it into a delayed reset.
            self.assertEqual(train.ensure_quest_zero_mastery_session(
                state,
                reset_id="attempt-b",
                now=5000,
                cooldown_seconds=600,
            ), [])
            self.assertTrue(frontier_state.exists())

    def test_explicit_override_allows_a_rapid_quest_zero_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"state")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"fresh-frontier")
            frontier_meta.write_text("{}", encoding="utf-8")
            marker = state.with_name("restart_start.quest_zero_mastery.json")
            marker.write_text(json.dumps({
                "schema": 3, "reset_id": "attempt-a", "started_ts": 1000,
            }), encoding="utf-8")

            moved = train.ensure_quest_zero_mastery_session(
                state,
                reset_id="attempt-b",
                now=1100,
                cooldown_seconds=600,
                allow_rapid_reset=True,
            )
            self.assertEqual(len(moved), 2)
            self.assertFalse(frontier_state.exists())
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["reset_id"],
                "attempt-b",
            )

    def test_new_reset_id_is_declined_when_frontier_is_past_the_phase_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"state")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"advanced-frontier")
            frontier_meta.write_text(
                json.dumps({"schema": 4, "quest_phase": 162}), encoding="utf-8"
            )
            marker = state.with_name("restart_start.quest_zero_mastery.json")
            marker.write_text(json.dumps({
                "schema": 3, "reset_id": "attempt-a", "started_ts": 1000,
            }), encoding="utf-8")

            self.assertEqual(train.ensure_quest_zero_mastery_session(
                state, reset_id="attempt-b", now=100000, cooldown_seconds=600,
            ), [])
            self.assertEqual(frontier_state.read_bytes(), b"advanced-frontier")
            saved = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(saved["reset_id"], "attempt-a")
            self.assertEqual(
                saved["ignored_reset_ids"][-1]["declined"],
                "frontier_phase_floor",
            )

    def test_early_frontier_still_accepts_a_new_attempt_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"state")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"early-frontier")
            frontier_meta.write_text(
                json.dumps({"schema": 4, "quest_phase": 12}), encoding="utf-8"
            )
            marker = state.with_name("restart_start.quest_zero_mastery.json")
            marker.write_text(json.dumps({
                "schema": 3, "reset_id": "attempt-a", "started_ts": 1000,
            }), encoding="utf-8")

            moved = train.ensure_quest_zero_mastery_session(
                state, reset_id="attempt-b", now=100000, cooldown_seconds=600,
            )
            self.assertEqual(len(moved), 2)
            self.assertFalse(frontier_state.exists())

    def test_phase_floor_zero_restores_unconditional_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"state")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"advanced-frontier")
            frontier_meta.write_text(
                json.dumps({"schema": 4, "quest_phase": 250}), encoding="utf-8"
            )
            marker = state.with_name("restart_start.quest_zero_mastery.json")
            marker.write_text(json.dumps({
                "schema": 3, "reset_id": "attempt-a", "started_ts": 1000,
            }), encoding="utf-8")

            os.environ["POKEMON_QUEST_ZERO_MAX_RESET_PHASE"] = "0"
            try:
                moved = train.ensure_quest_zero_mastery_session(
                    state, reset_id="attempt-b", now=100000,
                    cooldown_seconds=600,
                )
            finally:
                os.environ.pop("POKEMON_QUEST_ZERO_MAX_RESET_PHASE", None)
            self.assertEqual(len(moved), 2)
            self.assertFalse(frontier_state.exists())

    def test_explicit_force_reset_bypasses_advanced_frontier_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"state")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"advanced-frontier")
            frontier_meta.write_text(
                json.dumps({"schema": 4, "quest_phase": 301}), encoding="utf-8"
            )
            mastery = state.with_name("restart_start.swarm_mastery.json")
            mastery.write_text(
                json.dumps({"schema": 2, "windows": {"301": [10]}}),
                encoding="utf-8",
            )
            proven = state.with_name("restart_start.swarm_proven_routes.json")
            proven.write_text(
                json.dumps({"schema": 1, "routes": {"301": {"steps": 10}}}),
                encoding="utf-8",
            )
            marker = state.with_name("restart_start.quest_zero_mastery.json")
            marker.write_text(json.dumps({
                "schema": 3, "reset_id": "attempt-a", "started_ts": 1000,
            }), encoding="utf-8")

            moved = train.ensure_quest_zero_mastery_session(
                state,
                force=True,
                reset_id="attempt-b",
                now=1100,
                cooldown_seconds=600,
            )

            self.assertEqual(len(moved), 2)
            self.assertFalse(frontier_state.exists())
            self.assertFalse(frontier_meta.exists())
            self.assertTrue(mastery.exists())
            self.assertTrue(proven.exists())
            saved = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(saved["reset_id"], "attempt-b")
            self.assertTrue(saved["forced"])

    def test_preforest_layout_migration_rolls_back_and_reindexes_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "restart_start.state"
            state.write_bytes(b"base")
            frontier_state, frontier_meta = train.swarm_frontier_paths(state)
            frontier_state.write_bytes(b"forest")
            frontier_meta.write_text(
                json.dumps({"schema": 4, "quest_phase": 11}),
                encoding="utf-8",
            )
            mastery = state.with_name("restart_start.swarm_mastery.json")
            mastery.write_text(json.dumps({
                "schema": 2,
                "windows": {"0": [10], "7": [20], "8": [30], "10": [40]},
                "best_steps": {"0": 10, "7": 20, "8": 30, "10": 40},
            }), encoding="utf-8")
            proven = state.with_name("restart_start.swarm_proven_routes.json")
            proven.write_text(json.dumps({
                "schema": 1,
                "routes": {
                    "7": {"source": "route", "trace": [7]},
                    "8": {"source": "route", "trace": [8]},
                    "9": {"source": "drill", "trace": [9]},
                    "10": {"source": "route_atlas_seed", "trace": [10]},
                    "11": {"source": "route", "trace": [11]},
                },
            }), encoding="utf-8")

            backups = train.migrate_preforest_level10_quest_layout(state)
            self.assertEqual(
                json.loads(frontier_meta.read_text(encoding="utf-8"))["quest_phase"],
                train.FULLGAME_VIRIDIAN_HEAL_QUEST_PHASE,
            )
            migrated_mastery = json.loads(mastery.read_text(encoding="utf-8"))
            self.assertEqual(set(migrated_mastery["windows"]), {"0", "7"})
            self.assertEqual(set(migrated_mastery["best_steps"]), {"0", "7"})
            migrated_routes = json.loads(proven.read_text(encoding="utf-8"))["routes"]
            self.assertEqual(set(migrated_routes), {"7", "14", "15"})
            self.assertTrue(backups)
            self.assertTrue(all((state.parent / name).exists() for name in backups))
            self.assertEqual(
                train.migrate_preforest_level10_quest_layout(state), []
            )

    def test_early_guidance_stays_on_proven_waypoints(self):
        names = [waypoint['name'] for waypoint in train.FULLGAME_QUEST_WAYPOINTS[:19]]
        self.assertEqual(
            names,
            [
                'reach_viridian',
                'enter_viridian_mart',
                'collect_oaks_parcel',
                'return_to_pallet',
                'reenter_oaks_lab',
                'approach_oak',
                'deliver_oaks_parcel',
                'receive_pokedex',
                'heal_at_viridian',
                'train_pikachu_level_6',
                'train_pikachu_level_7',
                'train_pikachu_level_8',
                'train_pikachu_level_9',
                'train_pikachu_level_10',
                'reach_route_2',
                'enter_forest_gate',
                'enter_viridian_forest',
                'exit_viridian_forest',
                'heal_at_pewter',
            ],
        )
        self.assertEqual(train.FULLGAME_EARLY_GAME_QUEST_PHASES, frozenset(range(17)))
        self.assertGreater(len(train.EARLY_GAME_ACTION_GUIDANCE), 0)
        for rule in train.EARLY_GAME_ACTION_GUIDANCE:
            phases = rule.get('quest_phases')
            self.assertTrue(phases)
            self.assertTrue(set(phases) <= set(train.FULLGAME_EARLY_GAME_QUEST_PHASES))
        self.assertIn(
            train.FULLGAME_REACH_ROUTE_2_QUEST_PHASE,
            train.EARLY_GAME_NORTHBOUND_PHASES,
        )
        leave_lab = [
            rule for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if rule['map'] == 40
            and train.FULLGAME_VIRIDIAN_HEAL_QUEST_PHASE in rule['quest_phases']
        ]
        self.assertTrue(any(rule['target'] == (4, 7) for rule in leave_lab))
        self.assertTrue(any(rule['target'] == (5, 7) for rule in leave_lab))
        self.assertTrue(any(rule.get('destination_map') == 0 for rule in leave_lab))
        current_lab_exit = next(
            rule for rule in leave_lab if rule['target'] == (3, 5)
        )
        self.assertEqual(current_lab_exit['action'], 2)
        self.assertFalse(current_lab_exit.get('route_workers_only'))
        pallet_launch = [
            rule for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if rule['map'] == 0
            and rule['target'] in {(12, 12), (12, 11), (12, 10)}
            and train.FULLGAME_REACH_VIRIDIAN_QUEST_PHASE in rule['quest_phases']
            and rule.get('force_action')
        ]
        self.assertEqual({rule['target'] for rule in pallet_launch}, {
            (12, 12), (12, 11), (12, 10),
        })
        self.assertTrue(all(rule['action'] == 2 for rule in pallet_launch))
        self.assertTrue(all(rule.get('route_workers_only') for rule in pallet_launch))
        route1_north = [
            rule for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if rule['map'] == 12
            and train.FULLGAME_REACH_ROUTE_2_QUEST_PHASE in rule['quest_phases']
            and rule.get('force_action')
        ]
        self.assertTrue(any(rule['target'] == (35, 10) for rule in route1_north))
        self.assertTrue(any(rule.get('destination_map') == 1 for rule in route1_north))
        self.assertTrue(any(
            rule['target'] == (20, 4) and rule['action'] == 1
            for rule in route1_north
        ))
        viridian_north = [
            rule for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if rule['map'] == 1
            and train.FULLGAME_REACH_ROUTE_2_QUEST_PHASE in rule['quest_phases']
            and rule.get('force_action')
        ]
        self.assertTrue(any(
            rule['target'] == (30, 21) and rule['action'] == 2 and rule.get('route_workers_only')
            for rule in viridian_north
        ))
        self.assertTrue(any(rule.get('destination_map') == 13 for rule in viridian_north))
        self.assertTrue(any(
            rule['target'] == (1, 17) and rule['action'] == 0 and rule.get('route_workers_only')
            for rule in viridian_north
        ))
        mart_route = [
            rule for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if rule['map'] == 1
            and train.FULLGAME_ENTER_VIRIDIAN_MART_QUEST_PHASE
            in rule['quest_phases']
            and rule.get('route_workers_only')
        ]
        mart_actions = {rule['target']: rule['action'] for rule in mart_route}
        self.assertEqual(mart_actions[(35, 21)], 0)
        self.assertEqual(mart_actions[(30, 21)], 2)
        self.assertEqual(mart_actions[(28, 20)], 2)
        self.assertEqual(mart_actions[(28, 19)], 0)
        self.assertEqual(mart_actions[(20, 19)], 3)
        self.assertEqual(mart_actions[(20, 28)], 3)
        self.assertNotIn((20, 29), mart_actions)
        return_route = {
            (rule['map'], *rule['target']): rule['action']
            for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if train.FULLGAME_RETURN_TO_PALLET_QUEST_PHASE
            in rule['quest_phases']
            and rule.get('route_workers_only')
        }
        self.assertEqual(return_route[(42, 5, 2)], 1)
        self.assertEqual(return_route[(42, 7, 2)], 3)
        self.assertEqual(return_route[(42, 7, 3)], 1)
        self.assertEqual(return_route[(1, 20, 29)], 1)
        self.assertEqual(return_route[(1, 27, 29)], 1)
        self.assertEqual(return_route[(1, 35, 21)], 1)
        self.assertEqual(return_route[(12, 6, 11)], 1)
        self.assertEqual(return_route[(12, 13, 9)], 1)
        self.assertEqual(return_route[(12, 27, 12)], 1)
        self.assertEqual(return_route[(12, 35, 11)], 1)
        forced_route1 = [
            rule for rule in route1_north
            if rule['target'] in {(35, 10), (28, 10), (20, 12)}
        ]
        self.assertTrue(forced_route1)
        self.assertTrue(all(rule.get('route_workers_only') for rule in forced_route1))
        oak_parcel = [
            rule for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if rule['map'] == 40
            and train.FULLGAME_DELIVER_OAKS_PARCEL_QUEST_PHASE
            in rule['quest_phases']
            and rule.get('force_action')
        ]
        # Guidance is first-match-wins; verify the verified route owns these
        # coordinates ahead of broader legacy corridor hints.
        for target, action in [((5, 5), 0), ((4, 5), 0), ((3, 5), 4)]:
            first = next(rule for rule in oak_parcel if rule['target'] == target)
            self.assertEqual(first['action'], action)
        corridor = [
            rule for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if rule.get('route_workers_only')
        ]
        self.assertTrue(corridor)

    def test_route_runners_are_the_exploit_cohort(self):
        self.assertTrue(train.swarm_route_runner_skips_novelty(False, 0.25))
        self.assertFalse(train.swarm_route_runner_skips_novelty(True, 0.25))
        self.assertFalse(train.swarm_route_runner_skips_novelty(False, 0.0))
        # Mastery now gates the whole swarm, not only drill workers.
        self.assertTrue(train.swarm_mastery_blocks_frontier_publish(False, True))
        self.assertTrue(train.swarm_mastery_blocks_frontier_publish(True, True))
        self.assertFalse(train.swarm_mastery_blocks_frontier_publish(True, False))
        self.assertTrue(train.swarm_worker_obeys_quest_catchup(False))
        self.assertFalse(train.swarm_worker_obeys_quest_catchup(True))
        self.assertTrue(train.swarm_worker_records_mastery(True, True))
        self.assertFalse(train.swarm_worker_records_mastery(False, True))
        self.assertFalse(train.swarm_worker_records_mastery(True, False))
        self.assertTrue(train.swarm_worker_uses_verified_route_guidance(
            True, True
        ))
        self.assertTrue(train.swarm_worker_uses_verified_route_guidance(
            False, True
        ))
        self.assertFalse(train.swarm_worker_uses_verified_route_guidance(
            True, False
        ))
        self.assertFalse(train.swarm_worker_uses_verified_route_guidance(
            True, True, verified_route=False
        ))
        self.assertTrue(train.swarm_worker_replays_proven_trace(
            True, False, True, True
        ))
        self.assertTrue(train.swarm_worker_replays_proven_trace(
            False, False, True, True
        ))
        self.assertFalse(train.swarm_worker_replays_proven_trace(
            True, True, True, True
        ))
        self.assertFalse(train.swarm_worker_replays_proven_trace(
            True, False, True, False
        ))

    def test_route2_forest_gate_uses_verified_npc_detour(self):
        steps = train.ROUTE2_FOREST_GATE_STEPS
        actions = {(map_id, y, x): action for map_id, y, x, action, _ in steps}
        self.assertEqual(actions[(13, 57, 7)], 2)
        self.assertEqual(actions[(13, 57, 6)], 2)
        self.assertEqual(actions[(13, 57, 5)], 0)
        self.assertEqual(actions[(13, 44, 3)], 0)
        self.assertEqual(steps[-1][-1], 50)
        forest_rules = [
            rule for rule in train.EARLY_GAME_ACTION_GUIDANCE
            if rule.get('quest_phases')
            and train.FULLGAME_ENTER_FOREST_GATE_QUEST_PHASE
            in rule['quest_phases']
            and rule.get('verified_route')
        ]
        first_at_wall = next(
            rule for rule in forest_rules if rule['target'] == (57, 7)
        )
        self.assertEqual(first_at_wall['action'], 2)

    def test_pewter_heal_replays_completed_run_end_to_end(self):
        forest = train.VIRIDIAN_FOREST_LASS_APPROACH_STEPS
        self.assertEqual(forest[0][:4], (51, 47, 16, 0))
        self.assertEqual(forest[-1][:4], (51, 42, 1, 0))
        self.assertEqual(len(forest), 21)
        post_lass = train.VIRIDIAN_FOREST_POST_LASS_STEPS
        self.assertEqual(post_lass[0][:4], (51, 41, 1, 0))
        self.assertEqual(post_lass[-1], (51, 0, 1, 0, 47))
        self.assertEqual(len(post_lass), 151)
        # Trainer sight lines consume Up without moving. The next verified
        # leg must therefore restart from the same coordinate.
        for target in ((33, 26), (19, 26), (18, 1)):
            matching = [
                step for step in post_lass
                if step[:3] == (51, *target) and step[3] == 0
            ]
            self.assertGreaterEqual(len(matching), 2)
        self.assertEqual(train.FOREST_NORTH_GATE_EXIT_STEPS[-1][-1], 13)
        self.assertEqual(train.ROUTE2_NORTH_TO_PEWTER_STEPS[-1][-1], 2)
        self.assertEqual(train.PEWTER_CENTER_APPROACH_STEPS[-1][-1], 58)
        forest_rules = [
            rule for rule in train.PEWTER_HEAL_ROUTE_ACTION_GUIDANCE
            if train.FULLGAME_EXIT_VIRIDIAN_FOREST_QUEST_PHASE
            in rule.get('quest_phases', ())
            and rule.get('verified_route')
        ]
        pewter_rules = [
            rule for rule in train.PEWTER_HEAL_ROUTE_ACTION_GUIDANCE
            if train.FULLGAME_PEWTER_HEAL_QUEST_PHASE
            in rule.get('quest_phases', ())
            and rule.get('verified_route')
        ]
        self.assertTrue(forest_rules)
        self.assertTrue(pewter_rules)
        self.assertTrue(all(
            not rule.get('route_workers_only')
            for rule in forest_rules + pewter_rules
        ))
        post_lass_rules = [
            rule for rule in forest_rules
            if rule['target'] in {(33, 26), (19, 26), (18, 1)}
        ]
        self.assertTrue(post_lass_rules)
        self.assertTrue(all(rule.get('force_action') for rule in post_lass_rules))

    def test_preforest_grind_and_pewter_curriculum_boundaries(self):
        self.assertEqual(train.FULLGAME_PRE_FOREST_GRIND_PHASES, frozenset(range(9, 14)))
        self.assertEqual(train.FULLGAME_EXIT_VIRIDIAN_FOREST_QUEST_PHASE, 17)
        self.assertEqual(train.FULLGAME_PEWTER_HEAL_QUEST_PHASE, 18)
        self.assertEqual(train.FULLGAME_PEWTER_GRIND_PHASES, frozenset(range(19, 29)))
        self.assertEqual(train.FULLGAME_BROCK_QUEST_PHASE, 30)
        self.assertEqual(train.VIRIDIAN_CENTER_APPROACH_STEPS[-1][-1], 41)
        self.assertEqual(train.VIRIDIAN_CENTER_EXIT_STEPS[-1][-1], 1)
        self.assertEqual(train.VIRIDIAN_TO_ROUTE1_GRIND_STEPS[-1][-1], 12)
        self.assertEqual(
            train.FOREST_TO_VIRIDIAN_HEAL_RECOVERY_STEPS[-1][-1], 41
        )
        route2_tiles = {
            (m, y, x): action
            for m, y, x, action, _ in train.ROUTE2_SOUTH_TO_VIRIDIAN_STEPS
        }
        self.assertNotIn((13, 47, 3), route2_tiles)
        self.assertNotIn((13, 61, 3), route2_tiles)
        self.assertEqual(route2_tiles[(13, 68, 3)], 1)
        self.assertEqual(route2_tiles[(13, 69, 3)], 3)
        self.assertEqual(train.ROUTE2_SOUTH_TO_VIRIDIAN_STEPS[-1][-1], 1)
        nurse_exit = next(
            rule for rule in train.PRE_FOREST_GRIND_ACTION_GUIDANCE
            if rule['map'] == 41 and rule['target'] == (3, 3)
        )
        self.assertEqual(nurse_exit['action'], 1)
        self.assertTrue(nurse_exit['force_action'])

    def test_level_band_curriculum_has_one_mastered_objective_per_level(self):
        bands = (
            (train.FULLGAME_PEWTER_GRIND_PHASES, range(11, 21)),
            (train.FULLGAME_CERULEAN_GRIND_PHASES, range(21, 31)),
            (train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES, range(31, 46)),
            (train.FULLGAME_SILPH_GRIND_PHASES, range(41, 51)),
            (train.FULLGAME_SOUL_MARSH_GRIND_PHASES, range(51, 61)),
            (train.FULLGAME_VOLCANO_EARTH_GRIND_PHASES, range(61, 71)),
            (train.FULLGAME_ELITE_FOUR_GRIND_PHASES, range(71, 81)),
        )
        for phases, levels in bands:
            self.assertEqual(
                [train.FULLGAME_QUEST_WAYPOINTS[phase]['min_level']
                 for phase in sorted(phases)],
                list(levels),
            )
        self.assertEqual(
            train.FULLGAME_QUEST_WAYPOINTS[21]['name'],
            'train_pikachu_level_13',
        )

    def test_heatmap_proven_grind_patrols_are_exact_and_publishable(self):
        rock_tunnel = {
            (rule['map'], *rule['target']): rule['action']
            for rule in train.ROCK_TUNNEL_GRIND_ACTION_GUIDANCE
        }
        self.assertEqual(rock_tunnel[(21, 8, 8)], 3)
        self.assertEqual(rock_tunnel[(21, 8, 9)], 1)
        self.assertEqual(rock_tunnel[(82, 3, 15)], 0)
        route16 = {
            (rule['map'], *rule['target']): rule['action']
            for rule in train.ROUTE16_GRIND_ACTION_GUIDANCE
            if rule.get('verified_route')
        }
        self.assertEqual(route16[(27, 5, 29)], 0)
        self.assertEqual(route16[(27, 4, 29)], 1)
        route15 = {
            (rule['map'], *rule['target']): rule['action']
            for rule in train.ROUTE15_GRIND_LOOP_ACTION_GUIDANCE
        }
        self.assertEqual(route15[(26, 8, 19)], 1)
        self.assertEqual(route15[(26, 9, 19)], 0)
        mansion = {
            (rule['map'], *rule['target']): rule['action']
            for rule in train.MANSION_GRIND_ACTION_GUIDANCE
        }
        self.assertEqual(mansion[(216, 26, 17)], 3)
        self.assertEqual(mansion[(216, 26, 18)], 2)
        self.assertEqual(mansion[(165, 5, 3)], 1)
        self.assertEqual(mansion[(165, 6, 3)], 0)
        for phase in train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES:
            self.assertTrue(train.fullgame_frontier_position_allowed(
                phase, 82, 13, 36
            ))
        for phase in train.FULLGAME_SILPH_GRIND_PHASES:
            self.assertTrue(train.fullgame_frontier_position_allowed(
                phase, 27, 5, 29
            ))
        # The level-36 completion state remains on Route 4 while the next two
        # automatic checks evolve Charizard and confirm Fly.  Both boundaries
        # must be durable or every rotation reloads the level-36 grind.
        for phase in (
            train.FULLGAME_CHARIZARD_QUEST_PHASE,
            train.FULLGAME_TEACH_FLY_QUEST_PHASE,
        ):
            self.assertTrue(train.fullgame_frontier_position_allowed(
                phase, 15, 12, 69
            ))
            self.assertFalse(train.fullgame_frontier_position_allowed(
                phase, 232, 12, 69
            ))
        for phase in train.FULLGAME_SOUL_MARSH_GRIND_PHASES:
            self.assertTrue(train.fullgame_frontier_position_allowed(
                phase, 26, 8, 19
            ))
            self.assertTrue(train.fullgame_frontier_position_allowed(
                phase, 157, 15, 1
            ))

    def test_proven_routes_follow_objectives_when_phase_numbers_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'proven.json'
            path.write_text(json.dumps({
                'schema': 1,
                'routes': {
                    '21': {
                        'objective': 'train_pikachu_level_13',
                        'source': 'drill', 'steps': 10, 'trace': [],
                    },
                    '27': {
                        'objective': 'enter_pewter_gym',
                        'source': 'route', 'steps': 20, 'trace': [],
                    },
                    '187': {
                        'objective': 'train_pikachu_level_55',
                        'source': 'route_atlas_seed', 'steps': 30, 'trace': [],
                    },
                },
            }), encoding='utf-8')

            changes = train.remap_swarm_proven_routes_by_objective(path)
            payload = json.loads(path.read_text(encoding='utf-8'))
            brock_gym_phase = next(
                index for index, waypoint in enumerate(train.FULLGAME_QUEST_WAYPOINTS)
                if waypoint['name'] == 'enter_pewter_gym'
            )
            self.assertIn((27, brock_gym_phase, 'enter_pewter_gym'), changes)
            self.assertIn('21', payload['routes'])
            self.assertIn(str(brock_gym_phase), payload['routes'])
            self.assertNotIn('27', payload['routes'])
            soul_55_phase = next(
                index for index, waypoint in enumerate(train.FULLGAME_QUEST_WAYPOINTS)
                if waypoint['name'] == 'train_pikachu_soul_marsh_level_55'
            )
            self.assertEqual(
                payload['routes'][str(soul_55_phase)]['objective'],
                'train_pikachu_soul_marsh_level_55',
            )
            self.assertEqual(payload['quest_layout'], 'level_bands_v1')

    def test_viridian_heal_route_precedes_generic_northbound_route(self):
        matching = [
            rule for rule in train.FULLGAME_OPENING_ACTION_GUIDANCE
            if rule.get('map') == 1
            and rule.get('target') == (0, 17)
            and train.FULLGAME_VIRIDIAN_HEAL_QUEST_PHASE
            in rule.get('quest_phases', ())
            and rule.get('force_action')
        ]
        self.assertGreaterEqual(len(matching), 2)
        self.assertEqual(matching[0]['action'], 1)  # south toward the Center
        self.assertEqual(matching[0].get('destination_map'), None)
        self.assertEqual(matching[-1]['action'], 0)  # generic Route 2 transit

    def test_heal_waypoint_waits_for_nurse_dialogue_to_close(self):
        waypoint = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_VIRIDIAN_HEAL_QUEST_PHASE
        ]
        memory = {train.ADDR_FONT_LOADED: 1}
        self.assertFalse(train.quest_waypoint_matches(
            waypoint, 41, 3, 3, (), last_blackout_map=1, memory=memory,
        ))
        memory[train.ADDR_FONT_LOADED] = 0
        self.assertTrue(train.quest_waypoint_matches(
            waypoint, 41, 3, 3, (), last_blackout_map=1, memory=memory,
        ))

    def test_pewter_post_heal_exit_releases_dialogue_before_gym(self):
        phase = train.FULLGAME_PEWTER_GYM_ENTRY_QUEST_PHASE
        for y in range(3, 8):
            self.assertEqual(
                train.pewter_post_heal_gym_exit_action(
                    58, y, 3, phase, 1, 100
                ),
                'b',
            )
            self.assertEqual(
                train.pewter_post_heal_gym_exit_action(
                    58, y, 3, phase, 1, 101
                ),
                'down',
            )
        self.assertIsNone(train.pewter_post_heal_gym_exit_action(
            58, 3, 3, phase, 0, 100
        ))
        self.assertIsNone(train.pewter_post_heal_gym_exit_action(
            58, 3, 3, phase + 1, 1, 100
        ))
        self.assertTrue(train.pewter_center_exit_text_is_stale(
            58, 4, 3, phase, 0, 1
        ))
        self.assertTrue(train.pewter_center_exit_text_is_stale(
            58, 3, 3, phase, 0, 1
        ))
        self.assertFalse(train.pewter_center_exit_text_is_stale(
            58, 4, 3, phase, 1, 1
        ))
        post_brock = train.FULLGAME_ROUTE3_QUEST_PHASE - 1
        self.assertTrue(train.pewter_center_exit_text_is_stale(
            58, 3, 3, post_brock, 0, 1, 1.0
        ))
        self.assertFalse(train.pewter_center_exit_text_is_stale(
            58, 3, 3, post_brock, 0, 1, 0.5
        ))

    def test_every_explicit_heal_has_a_shared_center_exit_handoff(self):
        heal_phases = [
            index + 1
            for index, waypoint in enumerate(train.FULLGAME_QUEST_WAYPOINTS)
            if str(waypoint.get('name', '')).startswith('heal_at_')
            and index + 1 < len(train.FULLGAME_QUEST_WAYPOINTS)
        ]
        self.assertGreaterEqual(len(heal_phases), 8)
        for phase in heal_phases:
            for center_map in train.POKEMON_CENTER_MAP_IDS:
                self.assertEqual(
                    train.completed_heal_center_exit_action(
                        center_map, 3, 3, phase, 1
                    ),
                    'b',
                )
                self.assertEqual(
                    train.completed_heal_center_exit_action(
                        center_map, 3, 3, phase, 0
                    ),
                    'down',
                )
                self.assertEqual(
                    train.completed_heal_center_exit_action(
                        center_map, 6, 3, phase, 0
                    ),
                    'down',
                )
        # Current live handoff: heal_at_mt_moon_center -> enter_mt_moon.
        mt_moon_exit_phase = train.FULLGAME_MT_MOON_CENTER_QUEST_PHASE + 1
        self.assertEqual(
            train.completed_heal_center_exit_action(
                68, 3, 3, mt_moon_exit_phase, 1
            ),
            'b',
        )
        self.assertIsNone(train.completed_heal_center_exit_action(
            68, 3, 3, train.FULLGAME_MT_MOON_CENTER_QUEST_PHASE, 0
        ))
        self.assertTrue(train.completed_heal_center_text_is_stale(
            68, 3, 3, mt_moon_exit_phase, 0, 1, 64
        ))
        self.assertFalse(train.completed_heal_center_text_is_stale(
            68, 3, 3, mt_moon_exit_phase, 0, 1, 63
        ))
        self.assertTrue(train.completed_heal_center_handoff_pending(
            68, 3, 3, mt_moon_exit_phase
        ))
        self.assertTrue(train.completed_heal_center_handoff_pending(
            68, 7, 3, mt_moon_exit_phase
        ))
        self.assertFalse(train.completed_heal_center_handoff_pending(
            15, 6, 11, mt_moon_exit_phase
        ))
        route4_rejoin = [
            rule for rule in train.MT_MOON_ENTRY_ACTION_GUIDANCE
            if rule.get('map') == 15 and rule.get('target') == (5, 11)
        ]
        self.assertEqual(len(route4_rejoin), 1)
        self.assertEqual(route4_rejoin[0]['action'], 1)
        self.assertTrue(route4_rejoin[0]['force_action'])

    def test_mt_moon_fossil_traversal_selection_rocket_and_exit_are_split(self):
        names = [waypoint['name'] for waypoint in train.FULLGAME_QUEST_WAYPOINTS]
        sequence = [
            'enter_mt_moon',
            'reach_mt_moon_fossils',
            'choose_mt_moon_fossil',
            'beat_mt_moon_jessie_james',
            'exit_mt_moon',
            'reach_cerulean',
        ]
        start = names.index(sequence[0])
        self.assertEqual(names[start:start + len(sequence)], sequence)

        reach = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_FOSSIL_REACH_QUEST_PHASE
        ]
        self.assertEqual(reach['event_bit'], train.MT_MOON_SUPER_NERD_EVENT_BIT)
        self.assertTrue(reach['require_dialogue_closed'])
        memory = {train.ADDR_FONT_LOADED: 0}
        self.assertTrue(train.quest_waypoint_matches(
            reach, 61, 7, 13, {train.MT_MOON_SUPER_NERD_EVENT_BIT},
            memory=memory,
        ))
        memory[train.ADDR_FONT_LOADED] = 1
        self.assertFalse(train.quest_waypoint_matches(
            reach, 61, 7, 13, {train.MT_MOON_SUPER_NERD_EVENT_BIT},
            memory=memory,
        ))

        interactions = train.MT_MOON_FOSSIL_SELECTION_ACTION_GUIDANCE
        self.assertEqual(
            {(rule['target'], rule['action']) for rule in interactions},
            {((7, 12), 4), ((7, 13), 4)},
        )
        self.assertTrue(all(rule['force_action'] for rule in interactions))
        self.assertTrue(all('face_action' not in rule for rule in interactions))
        self.assertTrue(all(
            rule['require_event_bits'] == (train.MT_MOON_SUPER_NERD_EVENT_BIT,)
            for rule in interactions
        ))
        cave_exit = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_MT_MOON_EXIT_QUEST_PHASE
        ]
        self.assertEqual((cave_exit['map'], cave_exit['min_x']), (15, 24))

    def test_pewter_gym_route_has_no_center_direction_conflicts(self):
        phase = train.FULLGAME_PEWTER_GYM_ENTRY_QUEST_PHASE
        for y in range(3, 8):
            actions = {
                rule['action']
                for rule in train.PEWTER_GYM_APPROACH_ACTION_GUIDANCE
                if rule.get('map') == 58
                and rule.get('target') == (y, 3)
                and phase in rule.get('quest_phases', ())
                and rule.get('force_action')
            }
            self.assertEqual(actions, {1}, f'conflict at Pewter Center {(y, 3)}')
        all_actions = {}
        for rule in train.PEWTER_GYM_APPROACH_ACTION_GUIDANCE:
            if phase not in rule.get('quest_phases', ()) or not rule.get('force_action'):
                continue
            key = (rule['map'], tuple(rule['target']))
            all_actions.setdefault(key, set()).add(rule['action'])
        self.assertFalse({
            key: actions for key, actions in all_actions.items()
            if len(actions) > 1
        })

    def test_silph_1f_route_uses_open_west_lane_without_backtracking(self):
        steps = train.SILPH_CO_1F_TO_ELEVATOR_STEPS
        self.assertEqual(len(steps), 31)
        self.assertEqual(steps[0][:4], (181, 17, 10, 0))
        self.assertEqual(steps[-1], (181, 1, 20, 0, 236))
        self.assertNotIn((181, 12, 11), {step[:3] for step in steps})

        y, x = 17, 10
        delta = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        visited = []
        for map_id, step_y, step_x, action, _destination in steps:
            self.assertEqual((map_id, step_y, step_x), (181, y, x))
            visited.append((y, x))
            dy, dx = delta[action]
            y, x = y + dy, x + dx
        self.assertEqual((y, x), (0, 20))
        self.assertEqual(len(visited), len(set(visited)))

    def test_silph_1f_follower_release_only_alternates_after_failed_move(self):
        phase = train.FULLGAME_SILPH_CO_5F_QUEST_PHASE
        self.assertIsNone(train.silph_co_1f_follower_release_action(
            181, phase, 1,
        ))
        self.assertEqual(train.silph_co_1f_follower_release_action(
            181, phase, 2,
        ), 'b')
        self.assertIsNone(train.silph_co_1f_follower_release_action(
            181, phase, 3,
        ))
        self.assertEqual(train.silph_co_1f_follower_release_action(
            181, phase, 4,
        ), 'b')
        self.assertIsNone(train.silph_co_1f_follower_release_action(
            181, phase - 1, 2,
        ))
        self.assertIsNone(train.silph_co_1f_follower_release_action(
            10, phase, 2,
        ))

    def test_silph_card_key_return_owns_the_final_elevator_warp_tile(self):
        steps = train.SILPH_CO_CARD_KEY_TO_3F_ELEVATOR_STEPS
        self.assertEqual(steps[-2], (210, 2, 20, 0, None))
        self.assertEqual(steps[-1], (210, 1, 20, 0, 236))

    def test_silph_floor_selector_accepts_normal_live_joy_ignore_values(self):
        for joy_ignore in (1, 13):
            env = object.__new__(train.PokemonYellowEnv)
            env.quest_phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
            env.same_position_step_count = 1
            env._silph_elevator_attempts = {
                (env.quest_phase, 236, 1, 3, 2),
            }
            env.pyboy = mock.Mock()
            env.pyboy.memory = {
                train.ADDR_NUM_BAG_ITEMS: 1,
                train.ADDR_BAG_ITEMS: 0x30,
                train.ADDR_MAP_ID: 236,
                train.ADDR_POS_A: 1,
                train.ADDR_POS_B: 3,
                train.ADDR_BATTLE_FLAG: 0,
                train.ADDR_TEXT_BOX: joy_ignore,
            }
            env._tap_scripted_button = mock.Mock()
            self.assertTrue(env._try_select_silph_story_floor())
            self.assertEqual(env._tap_scripted_button.call_count, 10)
            self.assertEqual(env.pyboy.memory[train.ADDR_TEXT_BOX], 0)

    def test_silph_rival_exit_release_requires_the_beaten_event(self):
        phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        self.assertIsNone(train.silph_co_7f_dialogue_exit_action(
            212, 3, 3, phase, 100, False,
        ))
        self.assertEqual(train.silph_co_7f_dialogue_exit_action(
            212, 3, 3, phase, 100, True,
        ), 'b')
        self.assertEqual(train.silph_co_7f_dialogue_exit_action(
            212, 3, 3, phase, 101, True,
        ), 'down')
        self.assertIsNone(train.silph_co_7f_dialogue_exit_action(
            212, 3, 4, phase, 100, True,
        ))

    def test_silph_rival_win_is_healed_before_frontier_validation(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 212,
            train.ADDR_POS_A: 3,
            train.ADDR_POS_B: 3,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_PARTY_SIZE: 2,
            train.PARTY_CUR_HP_ADDRS[0][1]: 1,
            train.PARTY_MAX_HP_ADDRS[0][1]: 100,
            train.PARTY_CUR_HP_ADDRS[1][1]: 2,
            train.PARTY_MAX_HP_ADDRS[1][1]: 80,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        env.pyboy = mock.Mock(memory=memory)
        env._event_flag_is_set = mock.Mock(return_value=True)
        self.assertTrue(env._restore_silph_post_rival_resources())
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 100)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[1][1]], 80)
        env._event_flag_is_set.return_value = False
        self.assertFalse(env._restore_silph_post_rival_resources())

    def test_silph_rival_heal_budget_is_bounded_and_resets_by_contract(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 212,
            train.ADDR_POS_A: 3,
            train.ADDR_POS_B: 3,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_PARTY_SIZE: 1,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 100,
            train.PARTY_MAX_HP_ADDRS[0][1]: 100,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        env.pyboy = mock.Mock(memory=memory)
        env._event_flag_is_set = mock.Mock(return_value=False)
        env._silph_rival_emergency_heals_used = 0
        for expected in range(1, train.SILPH_RIVAL_EMERGENCY_HEAL_LIMIT + 1):
            memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 50
            memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 50
            self.assertTrue(env._use_silph_rival_emergency_heal())
            self.assertEqual(env._silph_rival_emergency_heals_used, expected)
            self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 100)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 1
        self.assertFalse(env._use_silph_rival_emergency_heal())

    def test_silph_giovanni_win_is_healed_before_frontier_validation(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 235,
            train.ADDR_POS_A: 13,
            train.ADDR_POS_B: 6,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_PARTY_SIZE: 2,
            train.PARTY_CUR_HP_ADDRS[0][1]: 1,
            train.PARTY_MAX_HP_ADDRS[0][1]: 100,
            train.PARTY_CUR_HP_ADDRS[1][1]: 2,
            train.PARTY_MAX_HP_ADDRS[1][1]: 80,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_MASTER_BALL_QUEST_PHASE
        env.pyboy = mock.Mock(memory=memory)
        env._event_flag_is_set = mock.Mock(return_value=True)
        self.assertTrue(env._restore_silph_post_giovanni_resources())
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 100)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[1][1]], 80)
        env._event_flag_is_set.return_value = False
        self.assertFalse(env._restore_silph_post_giovanni_resources())
        self.assertTrue(
            train.should_restore_silph_giovanni_frontier_resources(
                train.FULLGAME_MASTER_BALL_QUEST_PHASE,
            )
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_MASTER_BALL_QUEST_PHASE,
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(
                0.4, train.FULLGAME_MASTER_BALL_QUEST_PHASE,
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_MASTER_BALL_QUEST_PHASE + 1,
            ),
            0.4,
        )

    def test_rocket_giovanni_win_is_healed_before_frontier_validation(self):
        phase = train.FULLGAME_SILPH_SCOPE_QUEST_PHASE

        self.assertTrue(
            train.should_restore_rocket_giovanni_frontier_resources(phase)
        )
        self.assertFalse(
            train.should_restore_rocket_giovanni_frontier_resources(phase + 1)
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(0.4, phase),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(0.4, phase),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(0.4, phase + 1),
            0.4,
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(0.4, phase + 1),
            0.4,
        )

    def test_silph_giovanni_heal_budget_is_bounded_and_resets_by_contract(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 235,
            train.ADDR_POS_A: 13,
            train.ADDR_POS_B: 6,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_PARTY_SIZE: 1,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 100,
            train.PARTY_MAX_HP_ADDRS[0][1]: 100,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_SILPH_CO_GIOVANNI_QUEST_PHASE
        env.pyboy = mock.Mock(memory=memory)
        env._event_flag_is_set = mock.Mock(return_value=False)
        env._silph_giovanni_emergency_heals_used = 0
        for expected in range(1, train.SILPH_GIOVANNI_EMERGENCY_HEAL_LIMIT + 1):
            memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 50
            memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 50
            self.assertTrue(env._use_silph_giovanni_emergency_heal())
            self.assertEqual(env._silph_giovanni_emergency_heals_used, expected)
            self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 100)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 1
        self.assertFalse(env._use_silph_giovanni_emergency_heal())

    def test_silph_president_route_owns_giovanni_tile_without_conflicts(self):
        owned = {}
        for rule in train.SILPH_CO_11F_PRESIDENT_ACTION_GUIDANCE:
            key = (rule['map'], rule['target'])
            action = rule['action']
            if key in owned and owned[key] != action:
                self.fail(f'conflicting actions on {key}: {owned[key]} vs {action}')
            owned[key] = action
        self.assertEqual(owned[(235, (13, 6))], 0)
        self.assertEqual(owned[(235, (12, 7))], 2)
        self.assertEqual(owned[(235, (5, 6))], 4)
        talk = [
            rule for rule in train.SILPH_CO_11F_PRESIDENT_ACTION_GUIDANCE
            if rule['target'] == (5, 6)
        ][0]
        self.assertEqual(talk['face_action'], 3)
        self.assertEqual(
            talk['unless_event_bits'], (train.EVENT_GOT_MASTER_BALL,),
        )

    def test_silph_exit_publishes_from_11f_and_owns_the_pad_chain(self):
        phase = train.FULLGAME_RETURN_SAFFRON_AFTER_SILPH_QUEST_PHASE
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 235, 5, 6)
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 181, 17, 10)
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 10, 21, 18)
        )
        self.assertFalse(
            train.fullgame_frontier_position_allowed(phase, 4, 3, 3)
        )
        owned = {}
        for rule in train.SILPH_CO_11F_EXIT_ACTION_GUIDANCE:
            key = (rule['map'], rule['target'])
            action = rule['action']
            if key in owned and owned[key] != action:
                self.fail(f'conflicting actions on {key}: {owned[key]} vs {action}')
            owned[key] = action
        self.assertEqual(owned[(235, (5, 6))], 2)
        self.assertEqual(owned[(212, (7, 5))], 0)
        self.assertEqual(owned[(208, (11, 11))], 3)
        self.assertEqual(owned[(181, (1, 20))], 2)
        self.assertEqual(owned[(181, (17, 10))], 1)

    def test_silph_rival_forced_attack_is_ground_safe(self):
        phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        self.assertFalse(train.trainer_battle_allows_electric(212, phase))
        # The exception must not disable Thunderbolt for the other Silph
        # trainers or for later non-battle routing on the shared 7F map.
        self.assertTrue(train.trainer_battle_allows_electric(208, phase))
        self.assertTrue(train.trainer_battle_allows_electric(212, phase + 1))

    def test_sabrina_gym_exit_owns_door_mats_and_releases_inbound(self):
        phase = train.FULLGAME_LAVENDER_FUCHSIA_QUEST_PHASE
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 178, 17, 8)
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 10, 4, 34)
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 4, 8, 0)
        )
        post = {}
        post_rules = {}
        for rule in train.SABRINA_RETURN_ACTION_GUIDANCE:
            unless = tuple(rule.get('unless_event_bits') or ())
            if train.EVENT_BEAT_SABRINA in unless:
                continue
            key = (rule['map'], rule['target'])
            action = rule['action']
            if key in post and post[key] != action:
                self.fail(
                    f'conflicting post-Sabrina actions on {key}: '
                    f'{post[key]} vs {action}'
                )
            post[key] = action
            post_rules[key] = rule
        self.assertEqual(post[(178, (17, 8))], 1)
        self.assertEqual(post[(178, (17, 9))], 1)
        door = post_rules[(178, (17, 8))]
        self.assertEqual(door.get('destination_map'), 10)
        self.assertEqual(
            tuple(door.get('require_event_bits') or ()),
            (train.EVENT_BEAT_SABRINA,),
        )
        gym_entry = [
            rule for rule in train.SABRINA_RETURN_ACTION_GUIDANCE
            if rule['map'] == 10
            and rule['target'] == (4, 34)
            and train.EVENT_BEAT_SABRINA in tuple(
                rule.get('unless_event_bits') or ()
            )
        ]
        self.assertTrue(gym_entry)
        self.assertEqual(gym_entry[0]['action'], 0)
        self.assertEqual(gym_entry[0].get('destination_map'), 178)
        self.assertEqual(post[(10, (4, 34))], 3)
        self.assertEqual(post[(10, (18, 36))], 3)

    def test_sabrina_post_battle_repairs_only_exact_one_hp_overflow(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 178,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_PARTY_SIZE: 2,
            train.PARTY_CUR_HP_ADDRS[0][1]: 132,
            train.PARTY_MAX_HP_ADDRS[0][1]: 131,
            train.PARTY_CUR_HP_ADDRS[1][1]: 82,
            train.PARTY_MAX_HP_ADDRS[1][1]: 80,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_LAVENDER_FUCHSIA_QUEST_PHASE
        env.pyboy = mock.Mock(memory=memory)
        env._event_flag_is_set = mock.Mock(return_value=True)

        self.assertTrue(env._repair_sabrina_post_battle_hp_overflow())
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 131)
        # A larger mismatch is not this known artifact and must remain for the
        # general corruption guard to reject.
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[1][1]], 82)

        env.quest_phase -= 1
        memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 132
        self.assertFalse(env._repair_sabrina_post_battle_hp_overflow())
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 132)

    def test_route12_snorlax_coast_leaves_the_fishing_sign(self):
        phase = train.FULLGAME_LAVENDER_FUCHSIA_QUEST_PHASE
        self.assertEqual(
            train.route12_post_snorlax_coast_action(
                23, 63, 11, phase, True, 10,
            ),
            'b',
        )
        self.assertEqual(
            train.route12_post_snorlax_coast_action(
                23, 63, 11, phase, True, 11,
            ),
            'down',
        )
        self.assertIsNone(train.route12_post_snorlax_coast_action(
            23, 63, 11, phase, False, 11,
        ))
        self.assertIsNone(train.rock_tunnel_dialogue_recovery_action(
            23, phase, 0, 4, 63, 11,
        ))
        self.assertIsNone(train.rock_tunnel_dialogue_recovery_action(
            23, phase, 0, 4, 65, 14,
        ))
        self.assertEqual(
            train.rock_tunnel_dialogue_recovery_action(
                23, phase, 0, 4, 10, 9,
            ),
            'a',
        )
        self.assertEqual(
            train.route12_post_snorlax_coast_action(
                23, 65, 14, phase, True, 11, 2,
            ),
            'b',
        )
        self.assertIsNone(train.route12_post_snorlax_coast_action(
            23, 65, 14, phase, True, 11, 1,
        ))
        self.assertEqual(train.ROUTE12_COAST_NEXT_TILE[(65, 14)], (66, 14))
        self.assertEqual(train.ROUTE12_COAST_NEXT_TILE[(63, 10)], (64, 10))
        self.assertEqual(train.ROUTE12_COAST_NEXT_TILE[(67, 14)], (68, 14))
        self.assertEqual(train.ROUTE12_COAST_NEXT_TILE[(69, 14)], (69, 13))
        self.assertIn(
            train.EVENT_BEAT_ROUTE12_SNORLAX,
            train.allowed_atomic_frontier_event_bits(
                min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES),
                {train.EVENT_BEAT_ROUTE12_SNORLAX},
            ),
        )

    def test_route13_water_pocket_escapes_east_dead_end(self):
        phase = train.FULLGAME_FUCHSIA_QUEST_PHASE
        # Confirmed coherent-ROM turns, including both formerly solid edges.
        turns = {(24, 10, 51): 'down', (24, 11, 51): 'left',
                 (24, 8, 26): 'left', (24, 8, 10): 'left',
                 (24, 10, 0): 'left', (25, 10, 19): 'left'}
        names = ('up', 'down', 'left', 'right', 'a', 'b', 'start')
        for key, expected in turns.items():
            with self.subTest(position=key):
                rules = [rule for rule in train.PART11_ACTION_GUIDANCE
                         if (rule.get('map'), *rule.get('target', ())) == key]
                self.assertTrue(rules)
                self.assertEqual(names[rules[0]['action']], expected)
                self.assertEqual(train.route13_water_pocket_action(*key, phase), expected)
                self.assertEqual(train.route13_water_pocket_action(*key, phase, 2), 'b')
                self.assertIsNone(train.route13_water_pocket_action(
                    *key, phase, 2048, battle_flag=2))
        self.assertEqual(train.ROUTE13_EAST_ESCAPE_NEXT_TILE[(10, 51)], (11, 51))
        self.assertEqual(train.ROUTE13_WEST_EXIT_NEXT_TILE[(8, 10)], (8, 9))

    def test_route14_descent_stays_below_blocked_post_trainer_junctions(self):
        actions = {(y, x): action for map_id, y, x, action, _ in
                   train.ROUTE14_DESCENT_STEPS if map_id == 25}
        for y in range(10, 44):
            self.assertEqual(actions[(y, 13)], 1)
        self.assertEqual(actions[(44, 13)], 2)
        self.assertEqual(actions[(44, 0)], 2)
        for old_junction in ((32, 11), (33, 12), (40, 11), (41, 4), (43, 3)):
            self.assertNotIn(old_junction, actions)
        self.assertEqual(train.ROUTE14_DESCENT_STEPS[-1][-1], 26)

    def test_soul_marsh_repel_is_transit_scoped_to_route15(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        phase = min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES)
        memory = Memory({train.ADDR_MAP_ID: 24})
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = phase
        env.wild_battle_quest_phases = train.FULLGAME_SOUL_MARSH_GRIND_PHASES
        env.force_repel = True
        env._quest_phase_forces_repel = mock.Mock(return_value=False)

        for map_id in (23, 24, 25, 184):
            memory[train.ADDR_MAP_ID] = map_id
            self.assertTrue(env._force_repel_is_active(), map_id)
        memory[train.ADDR_MAP_ID] = 26
        self.assertFalse(env._force_repel_is_active())

    def test_saffron_gym_exit_door_clears_joy_and_last_map(self):
        class Memory(dict):
            def __getitem__(self, key):
                return self.get(key, 0)

        memory = Memory({
            train.ADDR_MAP_ID: 178,
            train.ADDR_POS_A: 17,
            train.ADDR_POS_B: 8,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_LAST_MAP: 6,
            train.ADDR_TEXT_BOX: 1,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env._event_flag_is_set = mock.Mock(return_value=True)
        self.assertTrue(env._settle_saffron_gym_exit_door())
        self.assertEqual(memory[train.ADDR_LAST_MAP], 10)
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 0)
        env._event_flag_is_set.return_value = False
        memory[train.ADDR_LAST_MAP] = 6
        memory[train.ADDR_TEXT_BOX] = 1
        self.assertFalse(env._settle_saffron_gym_exit_door())
        self.assertEqual(memory[train.ADDR_LAST_MAP], 6)
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 1)

    def test_soul_marsh_grind_keeps_real_wild_battles(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.force_repel = True
        env.force_repel_quest_phases = frozenset(
            train.FULLGAME_SOUL_MARSH_GRIND_PHASES
        )
        env.force_repel_min_lead_levels = {}
        env.wild_battle_quest_phases = frozenset(
            train.FULLGAME_SOUL_MARSH_GRIND_PHASES
        )
        env.quest_phase = min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES)
        env.suppress_stale_battle_reassertion = False
        env.action_guidance_requires_overworld = True
        env.action_guidance = [{
            'map': 26, 'target': (8, 19), 'action': 1, 'force_action': True,
        }]
        env._action_guidance_event_allowed = lambda rule: True
        env._forced_interaction_face_key = None
        env.pyboy = mock.Mock()
        env.pyboy.memory = {
            train.ADDR_BATTLE_FLAG: 1,
            train.ADDR_LEVEL: 54,
            train.ADDR_MAP_ID: 26,
        }
        self.assertFalse(env._force_repel_is_active())
        self.assertEqual(env._effective_battle_flag(1), 1)
        self.assertIsNone(env._matching_forced_action_guidance(26, 8, 19, 0))

    def test_force_repel_suppresses_high_level_encounters(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.force_repel = True
        env.force_repel_quest_phases = frozenset()
        env.force_repel_min_lead_levels = {}
        env.quest_phase = train.FULLGAME_PEWTER_HEAL_QUEST_PHASE
        env.pyboy = mock.Mock()
        env.pyboy.memory = {}
        self.assertTrue(env._maintain_force_repel_memory())
        self.assertEqual(env.pyboy.memory[train.ADDR_REPEL], 255)
        self.assertEqual(env.pyboy.memory[train.ADDR_GRASS_RATE], 0)
        self.assertEqual(env.pyboy.memory[train.ADDR_WATER_RATE], 0)
        env.force_repel = False
        env.force_repel_quest_phases = frozenset({env.quest_phase})
        env.force_repel_min_lead_levels = {env.quest_phase: 7}
        env.pyboy.memory[train.ADDR_LEVEL] = 5
        self.assertFalse(env._quest_phase_forces_repel())
        env.pyboy.memory[train.ADDR_LEVEL] = 7
        self.assertTrue(env._quest_phase_forces_repel())
        config = train.build_env_config('start.state', {
            'name': 'test', 'check': lambda mem, mid: False, 'ep_length': 1,
            'force_repel_min_lead_levels': {env.quest_phase: 7},
        })
        self.assertEqual(
            config['force_repel_min_lead_levels'], {env.quest_phase: 7}
        )

if __name__ == "__main__":
    unittest.main()
