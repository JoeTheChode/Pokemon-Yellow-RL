#!/usr/bin/env python3
import unittest

import train


def _set_event(memory, event_bit):
    address = train.ADDR_EVENT_FLAGS_START + event_bit // 8
    memory[address] |= 1 << (event_bit % 8)


class Phase287Route22RivalBridgeTests(unittest.TestCase):
    def test_live_frontier_bridge_joins_verified_final_rival_route(self):
        self.assertEqual(
            train.ROUTE22_FINAL_RIVAL_FRONTIER_BRIDGE_STEPS,
            (
                (33, 6, 39, 1, None),
                (33, 7, 39, 1, None),
                (33, 8, 39, 1, None),
            ),
        )
        self.assertEqual(train.ROUTE22_TO_FINAL_RIVAL_STEPS[0][:4], (33, 9, 39, 2))

        rules = [
            rule for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
            if rule.get('map') == 33
            and rule.get('target') in {(6, 39), (7, 39), (8, 39)}
            and train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE
            in rule.get('quest_phases', ())
        ]
        self.assertEqual(
            [(rule['target'], rule['action']) for rule in rules],
            [((6, 39), 1), ((7, 39), 1), ((8, 39), 1)],
        )

    def test_current_frontier_forced_bridge_preempts_stale_proven_trace(self):
        stale_trace = [[33, 6, 39, 0, 0, 'left']]
        proven_action, _ = train.proven_quest_trace_action(
            stale_trace, 0, (33, 6, 39, 0, 0)
        )
        self.assertEqual(proven_action, 'left')

        memory = bytearray(0x10000)
        memory[train.ADDR_LEVEL] = 70
        memory[train.ADDR_MAP_ID] = 33
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_TEXT_BOX] = 0
        memory[train.ADDR_FONT_LOADED] = 0
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = type('FakePyBoy', (), {'memory': memory})()
        env.quest_phase = train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE
        env.action_guidance = train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
        env.action_guidance_requires_overworld = True
        env.completed_run_recovery_after_steps = 0
        env.same_position_step_count = 0
        env.is_swarm_drill_worker = True
        env.swarm_mastery_enabled = True
        env._forced_interaction_face_key = None
        env._action_guidance_event_allowed = lambda _rule: True

        def noop(*_args, **_kwargs):
            return False

        env._clear_stale_overworld_joy_ignore = noop
        env._settle_bills_house_pending_warp = noop
        env._settle_cerulean_trash_house_pending_warp = noop
        env._settle_cerulean_bike_ledge_pocket = noop
        env._restore_glitched_map57_identity = noop
        env._settle_bike_shop_pending_warp = noop
        env._settle_vermilion_center_pending_warp = noop
        env._settle_vermilion_gym_pending_warp = noop
        env._settle_vermilion_side_house_pending_warp = noop
        env._settle_rock_tunnel_center_pending_warp = noop
        env._settle_victory_road_1f_pending_warp = noop
        env._settle_indigo_lobby_pending_warp = noop
        env._clear_rock_tunnel_overworld_joy_ignore = noop
        env._settle_saffron_gym_exit_door = noop
        env._heal_if_exiting_rock_tunnel = noop
        env._vermilion_gym_forced_action = lambda *_args, **_kwargs: None

        for y in (6, 7, 8):
            memory[train.ADDR_POS_A] = y
            memory[train.ADDR_POS_B] = 39
            self.assertEqual(
                env._matching_forced_action_guidance(33, y, 39, 0), 1
            )
        memory[train.ADDR_POS_A] = 9
        self.assertEqual(env._matching_forced_action_guidance(33, 9, 39, 0), 2)

    def _make_trigger_env(self, position=(5, 29)):
        memory = bytearray(0x10000)
        memory[train.ADDR_MAP_ID] = 33
        memory[train.ADDR_POS_A] = position[0]
        memory[train.ADDR_POS_B] = position[1]
        memory[train.ADDR_BATTLE_FLAG] = 0
        _set_event(memory, 81)
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = type('FakePyBoy', (), {'memory': memory})()
        env.quest_phase = train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE
        return env

    def test_missing_giovanni_side_effects_are_restored_on_real_trigger(self):
        env = self._make_trigger_env()

        self.assertTrue(env._restore_route22_final_rival_availability())
        self.assertTrue(env._event_flag_is_set(train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE))
        self.assertTrue(env._event_flag_is_set(train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE))
        self.assertFalse(env._restore_route22_final_rival_availability())

    def test_final_rival_availability_repair_is_trigger_scoped(self):
        env = self._make_trigger_env(position=(7, 39))

        self.assertFalse(env._restore_route22_final_rival_availability())
        self.assertFalse(env._event_flag_is_set(train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE))
        self.assertFalse(env._event_flag_is_set(train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE))

    def test_direct_battle_script_cannot_start_off_trigger(self):
        env = self._make_trigger_env(position=(7, 39))
        _set_event(env.pyboy.memory, train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE)
        _set_event(env.pyboy.memory, train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE)
        env.pyboy.memory[train.ADDR_STATUS_FLAGS_5] = 0b10100101
        env._tap_scripted_button = lambda *_args: self.fail('unexpected scripted input')

        self.assertFalse(env._try_start_route22_rival_battle())
        self.assertEqual(env.pyboy.memory[train.ADDR_STATUS_FLAGS_5], 0b10100101)

    def test_trigger_clears_only_stale_scripted_npc_movement_latch(self):
        env = self._make_trigger_env()
        _set_event(env.pyboy.memory, train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE)
        _set_event(env.pyboy.memory, train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE)
        env.pyboy.memory[train.ADDR_STATUS_FLAGS_5] = 0b10100101
        taps = []

        def enter_battle(*args):
            taps.append(args)
            env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 2

        env._tap_scripted_button = enter_battle

        self.assertTrue(env._try_start_route22_rival_battle())
        self.assertEqual(env.pyboy.memory[train.ADDR_STATUS_FLAGS_5], 0b10100100)
        self.assertEqual(
            env.pyboy.memory[train.ADDR_ROUTE22_CUR_SCRIPT],
            train.ROUTE22_RIVAL_BATTLE_SCRIPT,
        )
        self.assertEqual(env.pyboy.memory[train.ADDR_TEXT_BOX], 0)
        self.assertEqual(taps, [('a', 8, 32)])


if __name__ == '__main__':
    unittest.main()
