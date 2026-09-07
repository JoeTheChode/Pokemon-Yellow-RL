import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import train
from yellow_navigation import event_flags_by_bit


class _AlwaysMovesStubPyBoy:
    """Stand-in for train.PyBoy that reports every direction as working.

    Used to isolate _swarm_sync's promotion-guard logic in tests that don't
    care about frontier_candidate_accepts_movement's own behavior (that gets
    its own dedicated tests) from real PyBoy state-file loading, which the
    fake `save_state` lambda in make_env() doesn't produce a valid file for.
    """

    def __init__(self, rom_path, window=None, **kwargs):
        self.memory = bytearray(0x10000)
        self._pressed = None

    def load_state(self, handle):
        pass

    def button_press(self, name):
        self._pressed = name

    def button_release(self, name):
        self._pressed = None

    def tick(self, n):
        if self._pressed == 'down':
            self.memory[train.ADDR_POS_A] += 1

    def stop(self, **kwargs):
        pass


class FullgameSafetyTests(unittest.TestCase):
    def make_env(self):
        env = object.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = type(
            "FakePyBoy",
            (),
            {"memory": memory, "save_state": lambda self, handle: handle.write(b"fake-state")},
        )()
        env.stale_battle_movement_clear_steps = 1
        env.battle_flag_movement_steps = 0
        env.last_battle_flag_position = None
        env.last_battle_flag = 1
        env.current_battle_step_count = 12
        env.zero_hp_battle_step_count = 0
        env.stale_battle_flag_clears = 0
        env.stale_battle_movement_clears = 0
        env.suppress_stale_battle_reassertion = False
        env.stale_battle_reassertion_suppressions = 0
        env.stale_battle_enemy_hp = 0
        env.stale_battle_stationary_evidence_steps = 0
        env._reported_unverified_frontier_bits = set()
        env.reward_ordered_quest_event_flags_only = False
        env.reward_named_event_flags_only = False
        env.quest_waypoints = ()
        env.quest_phase = 0
        env.force_repel = False
        env.force_repel_quest_phases = frozenset()
        env.quarantined_event_flag_bits = frozenset()
        env.rewarded_named_event_flags = set()
        env.base_named_event_flags = set()
        env.swarm_rank_named_event_bits = False
        env.post_brock_route_phase = 0
        env.env_id = 0
        env._lead_moveset_snapshot = None
        env.lead_moveset_recoveries = 0
        env.direction_press_ticks = 8
        env.direction_release_ticks = 16
        return env

    def test_fullgame_action_space_includes_start(self):
        self.assertEqual(train.FULLGAME_ACTIONS[-1], "start")
        self.assertEqual(len(train.FULLGAME_ACTIONS), 7)

    def test_optional_moltres_interaction_is_blocked_only_beside_its_sprite(self):
        for position in ((4, 11), (6, 11), (5, 10), (5, 12)):
            self.assertTrue(train.optional_moltres_interaction_should_be_blocked(
                'a', 194, *position, False,
            ))
        self.assertFalse(train.optional_moltres_interaction_should_be_blocked(
            'a', 194, 5, 11, False,
        ))
        self.assertFalse(train.optional_moltres_interaction_should_be_blocked(
            'a', 194, 4, 11, True,
        ))
        self.assertFalse(train.optional_moltres_interaction_should_be_blocked(
            'right', 194, 4, 11, False,
        ))
        self.assertFalse(train.optional_moltres_interaction_should_be_blocked(
            'a', 198, 4, 11, False,
        ))

    def test_mt_moon_wild_battle_is_not_reclassified_as_trainer(self):
        env = self.make_env()
        env.trainer_battle_maps = {61}
        env.enemy_hp_battle_proxy_maps = set()

        self.assertFalse(env._trainer_battle_is_active(1, 1, 61, 30))
        self.assertTrue(env._trainer_battle_is_active(2, 2, 61, 30))

    def test_trainer_map_fallback_only_applies_in_enabled_hp_proxy_window(self):
        env = self.make_env()
        env.trainer_battle_maps = {61}
        env.enemy_hp_battle_proxy_maps = {61}

        self.assertTrue(env._trainer_battle_is_active(0, 0, 61, 30))
        env.enemy_hp_battle_proxy_maps = set()
        self.assertFalse(env._trainer_battle_is_active(0, 0, 61, 30))

    def test_checkpoint_frequency_is_aggregate_env_steps(self):
        self.assertEqual(
            train.callback_calls_for_env_steps(12_000_000, 96),
            125_000,
        )

    def test_swarm_explorer_uses_configured_post_brock_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'pallet.state'
            post_brock = root / 'mtmoon_b2f.state'
            original.write_bytes(b'original')
            post_brock.write_bytes(b'post-brock')
            self.assertEqual(
                train.swarm_episode_start_path(original, True, post_brock),
                post_brock,
            )
            self.assertEqual(
                train.swarm_episode_start_path(original, False, post_brock),
                original,
            )
            post_brock.unlink()
            self.assertEqual(
                train.swarm_episode_start_path(original, True, post_brock),
                original,
            )

    def test_swarm_explorers_load_current_frontier_during_normal_training(self):
        self.assertTrue(
            train.swarm_worker_should_load_frontier(True, False, True)
        )
        self.assertTrue(
            train.swarm_worker_should_load_frontier(True, True, True)
        )
        self.assertFalse(
            train.swarm_worker_should_load_frontier(True, True, False)
        )

    def test_swarm_explorer_loads_the_moving_frontier_when_available(self):
        self.assertTrue(
            train.swarm_worker_should_load_frontier(True, True, True)
        )
        self.assertTrue(
            train.swarm_worker_should_load_frontier(True, False, True)
        )
        self.assertFalse(
            train.swarm_worker_should_load_frontier(True, True, False)
        )

    def test_checkpoint_episode_metrics_are_cleared_without_touching_timesteps(self):
        model = SimpleNamespace(
            _stats_window_size=7,
            ep_info_buffer=['stale'],
            ep_success_buffer=['stale'],
            num_timesteps=12_000_000,
        )
        train.reset_episode_reporting_buffers(model)
        self.assertEqual(len(model.ep_info_buffer), 0)
        self.assertEqual(model.ep_info_buffer.maxlen, 7)
        self.assertEqual(len(model.ep_success_buffer), 0)
        self.assertEqual(model.num_timesteps, 12_000_000)

    def test_movement_clears_a_stale_positive_hp_battle_flag(self):
        env = self.make_env()
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 1
        env.pyboy.memory[train.ADDR_ENEMY_HP] = 25

        self.assertFalse(env._clear_stale_battle_flag_on_overworld_movement(13, 10, 5))
        self.assertTrue(env._clear_stale_battle_flag_on_overworld_movement(13, 10, 6))
        self.assertEqual(env.pyboy.memory[train.ADDR_BATTLE_FLAG], 0)
        self.assertEqual(env._effective_battle_flag(), 0)
        self.assertEqual(env.stale_battle_movement_clears, 1)

        self.assertFalse(env._suppress_stale_battle_before_action())
        self.assertEqual(env.pyboy.memory[train.ADDR_BATTLE_FLAG], 0)
        self.assertEqual(env._effective_battle_flag(), 0)
        self.assertEqual(env.stale_battle_reassertion_suppressions, 0)

    def test_stationary_real_battle_is_not_cleared(self):
        env = self.make_env()
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 1
        env.pyboy.memory[train.ADDR_ENEMY_HP] = 25

        for _ in range(10):
            self.assertFalse(env._clear_stale_battle_flag_on_overworld_movement(13, 10, 5))
        self.assertEqual(env.pyboy.memory[train.ADDR_BATTLE_FLAG], 1)
        self.assertEqual(env.stale_battle_movement_clears, 0)

    def test_dialogue_does_not_release_stale_battle_suppression(self):
        env = self.make_env()
        env.suppress_stale_battle_reassertion = True
        env.stale_battle_enemy_hp = 25
        env.last_battle_flag_position = (13, 10, 5)
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 1
        env.pyboy.memory[train.ADDR_ENEMY_HP] = 25
        env.pyboy.memory[train.ADDR_TEXT_BOX] = 1

        self.assertTrue(env._clear_stale_battle_flag_on_overworld_movement(13, 10, 5))
        self.assertTrue(env._clear_stale_battle_flag_on_overworld_movement(13, 10, 5))
        self.assertTrue(env.suppress_stale_battle_reassertion)
        self.assertEqual(env.pyboy.memory[train.ADDR_BATTLE_FLAG], 1)

    def test_changed_enemy_hp_releases_stale_suppression(self):
        env = self.make_env()
        env.suppress_stale_battle_reassertion = True
        env.stale_battle_enemy_hp = 25
        env.last_battle_flag_position = (13, 10, 5)
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 1
        env.pyboy.memory[train.ADDR_ENEMY_HP] = 30

        self.assertTrue(env._clear_stale_battle_flag_on_overworld_movement(13, 10, 5))
        self.assertFalse(env._clear_stale_battle_flag_on_overworld_movement(13, 10, 5))
        self.assertFalse(env.suppress_stale_battle_reassertion)

    def test_quest_requires_the_actual_parcel_event(self):
        waypoint = {'map': 42, 'event_bit': 57}
        self.assertFalse(train.quest_waypoint_matches(waypoint, 42, 5, 3, set()))
        self.assertFalse(train.quest_waypoint_matches(waypoint, 1, 5, 3, {57}))
        self.assertTrue(train.quest_waypoint_matches(waypoint, 42, 5, 3, {57}))

    def test_fullgame_quest_orders_parcel_before_mt_moon(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertLess(names.index('collect_oaks_parcel'), names.index('deliver_oaks_parcel'))
        self.assertLess(names.index('deliver_oaks_parcel'), names.index('enter_mt_moon'))

    def test_pewter_waypoint_requires_the_real_blackout_destination(self):
        waypoint = next(
            item for item in train.FULLGAME_QUEST_WAYPOINTS
            if item['name'] == 'heal_at_pewter'
        )
        self.assertFalse(
            train.quest_waypoint_matches(
                waypoint, 2, 33, 19, set(), last_blackout_map=0
            )
        )
        self.assertTrue(
            train.quest_waypoint_matches(
                waypoint, 58, 3, 3, set(), last_blackout_map=2
            )
        )

    def test_pewter_nurse_guidance_forces_a_only_in_heal_phase(self):
        env = self.make_env()
        env.action_guidance = tuple(
            {
                **dict(item),
                'quest_phases': frozenset({train.FULLGAME_PEWTER_HEAL_QUEST_PHASE}),
            }
            for item in train.PEWTER_HEAL_ACTION_GUIDANCE
        )
        env.action_guidance_requires_overworld = True
        env.quest_phase = train.FULLGAME_PEWTER_HEAL_QUEST_PHASE
        self.assertEqual(env._matching_forced_action_guidance(58, 3, 3, 0), 4)
        env.quest_phase += 1
        self.assertIsNone(env._matching_forced_action_guidance(58, 3, 3, 0))

    def test_pewter_frontier_persists_each_required_level_before_brock(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[12], 'train_pikachu_level_7')
        self.assertEqual(
            [item['min_level'] for item in train.FULLGAME_PEWTER_GRIND_WAYPOINTS],
            list(range(7, 19)),
        )
        self.assertLess(names.index('train_pikachu_level_18'), names.index('enter_pewter_gym'))
        self.assertLess(names.index('enter_pewter_gym'), names.index('beat_brock'))
        self.assertLess(names.index('beat_brock'), names.index('reach_route_3'))
        self.assertEqual(
            train.FULLGAME_BROCK_QUEST_PHASE,
            names.index('beat_brock'),
        )
        self.assertEqual(
            train.FULLGAME_PEWTER_GRIND_PHASES,
            frozenset(range(12, 24)),
        )

    def test_post_brock_phase_ids_and_route3_guidance_match_quest_order(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(train.FULLGAME_ROUTE3_QUEST_PHASE, 27)
        self.assertEqual(train.FULLGAME_MT_MOON_CENTER_QUEST_PHASE, 28)
        self.assertEqual(train.FULLGAME_MT_MOON_QUEST_PHASE, 29)
        self.assertEqual(train.FULLGAME_FOSSIL_QUEST_PHASE, 30)
        self.assertEqual(train.FULLGAME_JESSIE_JAMES_QUEST_PHASE, 31)
        self.assertEqual(train.FULLGAME_CERULEAN_QUEST_PHASE, 32)
        self.assertEqual(
            train.FULLGAME_CAVE_EXPLORATION_PHASE,
            len(train.FULLGAME_QUEST_WAYPOINTS),
        )
        self.assertEqual(
            names[train.FULLGAME_ROUTE3_QUEST_PHASE],
            'reach_mt_moon_approach',
        )
        self.assertEqual(
            names[train.FULLGAME_MT_MOON_CENTER_QUEST_PHASE],
            'heal_at_mt_moon_center',
        )
        self.assertEqual(
            names[train.FULLGAME_CERULEAN_QUEST_PHASE],
            'reach_cerulean',
        )
        self.assertEqual(
            train.FULLGAME_POST_BROCK_QUEST_PHASES,
            frozenset({
                27, 28, 29, 30, 31, 32,
                len(train.FULLGAME_QUEST_WAYPOINTS),
            }),
        )
        self.assertIn(
            ((5, 27), 1),
            train.ROUTE3_MANUAL_TURNS,
        )
        self.assertTrue(
            any(rule['target'] == (12, 22) and rule['action'] == 0
                for rule in train.ROUTE3_ACTION_GUIDANCE)
        )

    def test_mt_moon_center_requires_route4_respawn_and_forces_nurse_a(self):
        waypoint = next(
            item for item in train.FULLGAME_QUEST_WAYPOINTS
            if item['name'] == 'heal_at_mt_moon_center'
        )
        self.assertFalse(
            train.quest_waypoint_matches(
                waypoint, 68, 3, 3, set(), last_blackout_map=2
            )
        )
        self.assertTrue(
            train.quest_waypoint_matches(
                waypoint, 68, 3, 3, set(), last_blackout_map=15
            )
        )
        env = self.make_env()
        env.action_guidance = tuple(
            {
                **dict(item),
                'quest_phases': frozenset({
                    train.FULLGAME_MT_MOON_CENTER_QUEST_PHASE
                }),
            }
            for item in train.MT_MOON_CENTER_ACTION_GUIDANCE
        )
        env.action_guidance_requires_overworld = True
        env.quest_phase = train.FULLGAME_MT_MOON_CENTER_QUEST_PHASE
        self.assertEqual(env._matching_forced_action_guidance(68, 3, 3, 0), 4)

    def test_mt_moon_b1f_exit_branch_requires_exit_coordinates(self):
        self.assertEqual(
            train.post_brock_route_progress_key(60, 27, 11, previous_phase=5),
            (3, 0),
        )
        self.assertEqual(
            train.post_brock_route_progress_key(60, 3, 23, previous_phase=4),
            (5, 23),
        )

    def test_environment_wrapper_does_not_repromote_ordinary_b1f(self):
        env = self.make_env()
        env.post_brock_route_phase = 4
        self.assertEqual(env._post_brock_route_progress_key(27, 11, 60), (3, 0))
        self.assertEqual(env.post_brock_route_phase, 4)
        self.assertEqual(env._post_brock_route_progress_key(3, 23, 60), (5, 23))

    def test_mt_moon_b2f_subprogress_follows_demonstrated_route(self):
        route_to_fossil = [
            (17, 21),  # entry
            (14, 35),  # east wall
            (30, 34),  # southeast corner
            (31, 11),  # bottom-west corner
            (19, 10),  # west climb
            (8, 13),   # north stem
            (7, 12),   # fossil interaction
        ]
        values = [train.mt_moon_b2f_progress_value(y, x) for y, x in route_to_fossil]
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(values), len(set(values)))
        self.assertLess(
            train.mt_moon_b2f_progress_value(6, 9),
            train.mt_moon_b2f_progress_value(7, 12),
        )
        self.assertGreater(
            train.mt_moon_b2f_progress_value(7, 12, fossil_chosen=True),
            train.mt_moon_b2f_progress_value(7, 12),
        )
        self.assertGreater(
            train.mt_moon_b2f_progress_value(4, 4, fossil_chosen=True),
            train.mt_moon_b2f_progress_value(7, 12, fossil_chosen=True),
        )
        self.assertEqual(
            train.post_brock_route_progress_key(61, 26, 15, previous_phase=4),
            (4, 0),
        )
        self.assertGreater(
            train.post_brock_route_progress_key(61, 30, 34, previous_phase=4)[1],
            200,
        )
        self.assertEqual(
            train.post_brock_route_progress_key(61, 27, 13, previous_phase=4),
            (4, 0),
        )
        self.assertEqual(
            train.post_brock_route_progress_key(61, 27, 11, previous_phase=4),
            (4, 304),
        )

    def test_fossil_waypoint_accepts_either_fossil_event(self):
        waypoint = train.FULLGAME_QUEST_WAYPOINTS[train.FULLGAME_FOSSIL_QUEST_PHASE]
        self.assertFalse(train.quest_waypoint_matches(waypoint, 61, 7, 12, {1401}))
        self.assertTrue(train.quest_waypoint_matches(waypoint, 61, 7, 12, {1400, 1401}))
        self.assertTrue(train.quest_waypoint_matches(waypoint, 61, 7, 12, {1401, 1407}))

    def test_jessie_james_is_required_after_the_fossil(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[30], 'choose_mt_moon_fossil')
        self.assertEqual(names[31], 'beat_mt_moon_jessie_james')
        waypoint = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_JESSIE_JAMES_QUEST_PHASE
        ]
        self.assertFalse(
            train.quest_waypoint_matches(waypoint, 61, 4, 3, {1400})
        )
        self.assertTrue(
            train.quest_waypoint_matches(waypoint, 61, 4, 3, {1400, 1402})
        )

    def test_reach_cerulean_is_required_after_jessie_and_james(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[31], 'beat_mt_moon_jessie_james')
        self.assertEqual(names[32], 'reach_cerulean')
        waypoint = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_CERULEAN_QUEST_PHASE
        ]
        self.assertFalse(
            train.quest_waypoint_matches(waypoint, 15, 6, 24, set())
        )
        self.assertTrue(
            train.quest_waypoint_matches(waypoint, 3, 0, 0, set())
        )

    def test_cerulean_grind_heal_and_misty_are_ordered(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[33], 'heal_at_cerulean')
        self.assertEqual(
            [item['min_level'] for item in train.FULLGAME_CERULEAN_GRIND_WAYPOINTS],
            [23, 24, 25, 26],
        )
        # The post-Sabrina curriculum reuses the Cerulean grind machinery
        # for its level-55 Pikachu waypoint, so that phase joins the set.
        level_grind_phase = next(
            index for index, waypoint in enumerate(train.FULLGAME_QUEST_WAYPOINTS)
            if waypoint['name'] == 'train_pikachu_level_55'
        )
        self.assertEqual(
            train.FULLGAME_CERULEAN_GRIND_PHASES,
            frozenset({34, 35, 37, 39, level_grind_phase}),
        )
        self.assertEqual(
            train.FULLGAME_CERULEAN_INTERMEDIATE_HEAL_PHASES,
            frozenset({36, 38}),
        )
        self.assertEqual(names[36], 'heal_after_level_24')
        self.assertEqual(names[38], 'heal_after_level_25')
        self.assertEqual(names[40], 'heal_before_misty')
        self.assertEqual(names[41], 'enter_cerulean_gym')
        self.assertEqual(names[42], 'beat_misty')

        pre_misty_heal = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_PRE_MISTY_HEAL_QUEST_PHASE
        ]
        self.assertFalse(train.quest_waypoint_matches(
            pre_misty_heal, 64, 3, 3, set(),
            total_party_hp=70, total_party_max_hp=72,
        ))
        self.assertTrue(train.quest_waypoint_matches(
            pre_misty_heal, 64, 3, 3, set(),
            total_party_hp=72, total_party_max_hp=72,
        ))

        misty = train.FULLGAME_QUEST_WAYPOINTS[train.FULLGAME_MISTY_QUEST_PHASE]
        self.assertFalse(train.quest_waypoint_matches(
            misty, 65, 2, 5, set(), badge_count=1
        ))
        self.assertTrue(train.quest_waypoint_matches(
            misty, 65, 2, 5, set(), badge_count=2
        ))

        self.assertEqual(names[43], 'beat_cerulean_rival')
        self.assertEqual(names[44], 'reach_route_24')
        self.assertEqual(names[45], 'reach_route_25')
        self.assertEqual(names[46], 'enter_bills_house')
        self.assertEqual(names[47], 'heal_after_route_25')
        self.assertEqual(names[48], 'return_to_bills_house')
        self.assertEqual(names[51], 'receive_ss_ticket')
        self.assertEqual(names[53], 'beat_cerulean_rocket_thief')
        self.assertEqual(names[59], 'reach_vermilion')
        self.assertEqual(names[62], 'board_ss_anne')
        self.assertEqual(names[66], 'receive_hm01')
        self.assertEqual(names[68], 'return_to_cerulean_for_bulbasaur')
        self.assertEqual(names[69], 'enter_melanies_house')
        self.assertEqual(names[70], 'receive_bulbasaur')
        self.assertEqual(names[71], 'teach_cut_to_bulbasaur')
        self.assertEqual(names[72], 'return_to_vermilion_with_cut')
        self.assertEqual(names[73], 'enter_vermilion_gym')
        self.assertEqual(names[74], 'beat_lt_surge')
        self.assertEqual(names[75], 'heal_after_lt_surge')
        self.assertEqual(names[76], 'receive_squirtle')
        self.assertEqual(names[77], 'return_to_route_24_for_charmander')
        self.assertEqual(names[78], 'receive_charmander')
        self.assertEqual(names[79], 'reach_route_9')
        self.assertEqual(names[80], 'cross_route_9_cut_tree')
        self.assertEqual(names[81], 'reach_route_10')
        self.assertEqual(names[82], 'heal_at_rock_tunnel_center')
        self.assertEqual(names[83], 'enter_rock_tunnel')
        self.assertEqual(names[84], 'reach_rock_tunnel_first_b1f')
        self.assertEqual(names[85], 'reach_rock_tunnel_middle_1f')
        self.assertEqual(names[86], 'reach_rock_tunnel_second_b1f')
        self.assertEqual(names[87], 'reach_rock_tunnel_final_1f')
        self.assertEqual(
            names[88:93],
            [f'train_pikachu_rock_tunnel_level_{level}' for level in range(36, 41)],
        )
        self.assertEqual(
            train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES,
            frozenset(range(88, 93)),
        )
        self.assertEqual(names[93], 'exit_rock_tunnel_to_lower_route_10')
        self.assertEqual(names[94], 'reach_lavender')
        self.assertEqual(names[95], 'heal_at_lavender_center')
        self.assertEqual(names[96], 'reach_route_8')
        self.assertEqual(names[97], 'enter_route_8_underground_path')
        self.assertEqual(names[98], 'cross_route_7_8_underground_path')
        self.assertEqual(names[99], 'exit_route_7_underground_path')
        self.assertEqual(names[100], 'reach_route_7')
        self.assertEqual(names[101], 'reach_celadon')
        self.assertEqual(names[102], 'heal_at_celadon_center')
        self.assertEqual(
            train.FULLGAME_POST_MISTY_QUEST_PHASES,
            frozenset(range(43, 103)),
        )

        rock_heal = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_ROCK_TUNNEL_HEAL_QUEST_PHASE
        ]
        self.assertFalse(train.quest_waypoint_matches(
            rock_heal, 81, 3, 3, set(),
            total_party_hp=180, total_party_max_hp=185,
        ))
        self.assertTrue(train.quest_waypoint_matches(
            rock_heal, 81, 3, 3, set(),
            total_party_hp=185, total_party_max_hp=185,
        ))

        gift_cases = (
            (train.FULLGAME_SQUIRTLE_QUEST_PHASE, 327),
            (train.FULLGAME_BULBASAUR_QUEST_PHASE, 168),
            (train.FULLGAME_CHARMANDER_QUEST_PHASE, 1359),
        )
        for phase, event_bit in gift_cases:
            waypoint = train.FULLGAME_QUEST_WAYPOINTS[phase]
            self.assertFalse(train.quest_waypoint_matches(
                waypoint, 0, 0, 0, set()
            ))
            self.assertTrue(train.quest_waypoint_matches(
                waypoint, 0, 0, 0, {event_bit}
            ))

        teach_cut = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_TEACH_CUT_QUEST_PHASE
        ]
        self.assertFalse(train.quest_waypoint_matches(
            teach_cut, 63, 2, 3, {168}, party_move_ids={33, 45}
        ))
        self.assertTrue(train.quest_waypoint_matches(
            teach_cut, 63, 2, 3, {168}, party_move_ids={15, 33, 45}
        ))

    def test_cut_is_taught_only_to_real_gift_bulbasaur_after_hm01(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_TEACH_CUT_QUEST_PHASE
        memory[train.ADDR_PARTY_SIZE] = 3
        memory[train.PARTY_SPECIES_ADDRS[0]] = 84   # Pikachu
        memory[train.PARTY_SPECIES_ADDRS[1]] = 123  # Caterpie
        memory[train.PARTY_SPECIES_ADDRS[2]] = train.GEN1_BULBASAUR_SPECIES_ID
        bulba_moves = train.PARTY_MOVE_ID_ADDRS[2]
        bulba_pp = train.PARTY_MOVE_PP_ADDRS[2]
        for addr, move_id in zip(bulba_moves, (33, 45, 73, 22)):
            memory[addr] = move_id
        for addr in bulba_pp:
            memory[addr] = 20

        self.assertFalse(env._ensure_bulbasaur_knows_cut())
        for bit in (1504, 168):
            addr = train.ADDR_EVENT_FLAGS_START + bit // 8
            memory[addr] |= 1 << (bit % 8)
        self.assertTrue(env._ensure_bulbasaur_knows_cut())
        self.assertEqual(
            [memory[addr] for addr in bulba_moves],
            [33, train.GEN1_CUT_MOVE_ID, 73, 22],
        )
        self.assertEqual(memory[bulba_pp[1]], train.GEN1_CUT_MAX_PP)
        self.assertNotIn(
            train.GEN1_CUT_MOVE_ID,
            [memory[addr] for addr in train.PARTY_MOVE_ID_ADDRS[0]],
        )

    def test_celadon_objectives_are_ordered_and_use_yellow_saffron_map(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        ordered = [
            'beat_erika', 'receive_hm02', 'find_rocket_hideout',
            'beat_rocket_hideout_giovanni', 'buy_guard_drink',
            'give_saffron_guards_drink', 'reach_saffron',
            'evolve_charizard', 'teach_fly_to_charizard',
        ]
        self.assertEqual([names.index(name) for name in ordered], sorted(
            names.index(name) for name in ordered
        ))
        saffron = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_REACH_SAFFRON_QUEST_PHASE
        ]
        self.assertEqual(saffron['map'], 10)

        gym_entry = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_CELADON_GYM_QUEST_PHASE
        ]
        self.assertFalse(train.quest_waypoint_matches(
            gym_entry, 134, 27, 12, set()
        ))
        self.assertTrue(train.quest_waypoint_matches(
            gym_entry, 134, 17, 4, set()
        ))

    def test_charmander_training_swap_keeps_all_parallel_party_records_aligned(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = min(train.FULLGAME_CHARMANDER_GRIND_PHASES)
        # The live post-Saffron frontier retains D057=1 even on overworld
        # Route 7. Effective battle classification must not block the swap.
        env.force_repel_quest_phases = frozenset({env.quest_phase})
        memory[train.ADDR_MAP_ID] = 18
        memory[train.ADDR_BATTLE_FLAG] = 1
        memory[train.ADDR_PARTY_SIZE] = 2
        species = (train.GEN1_PIKACHU_SPECIES_ID, 176)
        for index, species_id in enumerate(species):
            memory[train.ADDR_PARTY_SPECIES_LIST + index] = species_id
            memory[train.PARTY_SPECIES_ADDRS[index]] = species_id
            memory[train.PARTY_SPECIES_ADDRS[index] + 1] = 20 + index
            for offset in range(train.GEN1_PARTY_OT_NAME_LENGTH):
                memory[train.ADDR_PARTY_OT + index * 11 + offset] = 40 + index
                memory[train.ADDR_PARTY_NICKS + index * 11 + offset] = 60 + index

        self.assertTrue(env._promote_charmander_family_for_training())
        self.assertEqual(memory[train.ADDR_PARTY_SPECIES_LIST], 176)
        self.assertEqual(memory[train.PARTY_SPECIES_ADDRS[0]], 176)
        self.assertEqual(memory[train.PARTY_SPECIES_ADDRS[1]], 84)
        self.assertEqual(memory[train.PARTY_SPECIES_ADDRS[0] + 1], 21)
        self.assertEqual(memory[train.ADDR_PARTY_OT], 41)
        self.assertEqual(memory[train.ADDR_PARTY_NICKS], 61)

    def test_charmander_training_promotes_ember_for_forced_a_battles(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = min(train.FULLGAME_CHARMANDER_GRIND_PHASES)
        memory[train.ADDR_PARTY_SIZE] = 1
        memory[train.ADDR_PARTY_SPECIES_LIST] = 176
        memory[train.PARTY_SPECIES_ADDRS[0]] = 176
        move_addrs = train.PARTY_MOVE_ID_ADDRS[0]
        pp_addrs = train.PARTY_MOVE_PP_ADDRS[0]
        for addr, value in zip(move_addrs, (10, 45, 52, 0)):
            memory[addr] = value
        for addr, value in zip(pp_addrs, (31, 40, 24, 0)):
            memory[addr] = value

        self.assertTrue(env._promote_charmander_family_for_training())
        self.assertEqual([memory[a] for a in move_addrs], [52, 45, 10, 0])
        self.assertEqual([memory[a] for a in pp_addrs], [24, 40, 31, 0])

    def test_charmander_grind_recovers_health_and_ember_between_battles(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = min(train.FULLGAME_CHARMANDER_GRIND_PHASES)
        memory[train.ADDR_PARTY_SIZE] = 1
        memory[train.PARTY_SPECIES_ADDRS[0]] = 176
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
        max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[0]
        memory[cur_hi], memory[cur_lo] = 0, 2
        memory[max_hi], memory[max_lo] = 0, 29
        memory[train.PARTY_MOVE_ID_ADDRS[0][0]] = train.GEN1_EMBER_MOVE_ID
        memory[train.PARTY_MOVE_PP_ADDRS[0][0]] = 2

        self.assertTrue(env._maintain_charmander_training_resources())
        self.assertEqual((memory[cur_hi] << 8) | memory[cur_lo], 29)
        self.assertEqual(
            memory[train.PARTY_MOVE_PP_ADDRS[0][0]] & 0x3F,
            train.GEN1_MOVE_TABLE[train.GEN1_EMBER_MOVE_ID]['max_pp'],
        )

    def test_charmander_grind_does_not_heal_during_battle(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = min(train.FULLGAME_CHARMANDER_GRIND_PHASES)
        memory[train.ADDR_PARTY_SIZE] = 1
        memory[train.PARTY_SPECIES_ADDRS[0]] = 176
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
        max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[0]
        memory[cur_hi], memory[cur_lo] = 0, 2
        memory[max_hi], memory[max_lo] = 0, 29
        memory[train.ADDR_BATTLE_FLAG] = 1

        self.assertFalse(env._maintain_charmander_training_resources())
        self.assertEqual((memory[cur_hi] << 8) | memory[cur_lo], 2)

    def test_mansion_grind_recovers_pikachu_between_honest_encounters(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = min(train.FULLGAME_MANSION_GRIND_PHASES)
        memory[train.ADDR_MAP_ID] = 165
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_PARTY_SIZE] = 1
        memory[train.PARTY_SPECIES_ADDRS[0]] = train.GEN1_PIKACHU_SPECIES_ID
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
        max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[0]
        memory[cur_hi], memory[cur_lo] = 0, 81
        memory[max_hi], memory[max_lo] = 0, 121
        memory[train.PARTY_MOVE_ID_ADDRS[0][0]] = train.GEN1_THUNDERBOLT_MOVE_ID
        memory[train.PARTY_MOVE_PP_ADDRS[0][0]] = 3

        self.assertTrue(env._maintain_mansion_training_resources())
        self.assertEqual((memory[cur_hi] << 8) | memory[cur_lo], 121)
        self.assertEqual(
            memory[train.PARTY_MOVE_PP_ADDRS[0][0]] & 0x3F,
            train.GEN1_THUNDERBOLT_MAX_PP,
        )

    def test_mansion_grind_recovery_is_map_battle_and_phase_scoped(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = min(train.FULLGAME_MANSION_GRIND_PHASES)
        memory[train.ADDR_MAP_ID] = 165
        memory[train.ADDR_PARTY_SIZE] = 1
        memory[train.PARTY_SPECIES_ADDRS[0]] = train.GEN1_PIKACHU_SPECIES_ID
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
        max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[0]
        memory[cur_hi], memory[cur_lo] = 0, 1
        memory[max_hi], memory[max_lo] = 0, 121
        memory[train.ADDR_BATTLE_FLAG] = 1
        self.assertFalse(env._maintain_mansion_training_resources())
        self.assertEqual((memory[cur_hi] << 8) | memory[cur_lo], 1)

        memory[train.ADDR_BATTLE_FLAG] = 0
        env.quest_phase = train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE
        self.assertFalse(env._maintain_mansion_training_resources())
        self.assertEqual((memory[cur_hi] << 8) | memory[cur_lo], 1)

    def test_mansion_level_checkpoints_get_a_full_episode_horizon(self):
        phase = min(train.FULLGAME_MANSION_GRIND_PHASES)
        self.assertEqual(
            train.quest_phase_stall_limit_for_phase(4096, phase), 16384,
        )
        self.assertEqual(
            train.quest_phase_stall_limit_for_phase(
                4096, train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE
            ),
            4096,
        )

    def test_post_strength_mansion_exit_selects_dig_not_strength(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_MANSION_TRAINING_EXIT_QUEST_PHASE
        memory[train.ADDR_MAP_ID] = 214
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_TEXT_BOX] = 0
        env._party_index_with_move = lambda move_id: 3
        taps = []
        env._tap_scripted_button = lambda button, *args: taps.append(button)

        self.assertFalse(env._try_dig_out_after_mansion_training())
        self.assertEqual(
            train.GEN1_POST_STRENGTH_DIG_FIELD_MENU_INDEX, 1,
        )
        # START, party selection, and field menu produce three A presses;
        # the sole Down must occur before the fourth A that activates Dig.
        self.assertEqual(taps, [
            'b', 'start', 'a', 'a', 'down', 'a', 'a',
        ])

    def test_pikachu_returns_to_lead_after_charizard_fly_objective(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_TEACH_FLY_QUEST_PHASE + 1
        memory[train.ADDR_PARTY_SIZE] = 2
        for index, species_id in enumerate((180, train.GEN1_PIKACHU_SPECIES_ID)):
            memory[train.ADDR_PARTY_SPECIES_LIST + index] = species_id
            memory[train.PARTY_SPECIES_ADDRS[index]] = species_id
            memory[train.PARTY_SPECIES_ADDRS[index] + 1] = 30 + index
            for offset in range(train.GEN1_PARTY_OT_NAME_LENGTH):
                memory[train.ADDR_PARTY_OT + index * 11 + offset] = 50 + index
                memory[train.ADDR_PARTY_NICKS + index * 11 + offset] = 70 + index

        self.assertTrue(env._restore_pikachu_lead_after_charizard())
        self.assertEqual(
            memory[train.ADDR_PARTY_SPECIES_LIST],
            train.GEN1_PIKACHU_SPECIES_ID,
        )
        self.assertEqual(
            memory[train.PARTY_SPECIES_ADDRS[0]],
            train.GEN1_PIKACHU_SPECIES_ID,
        )
        self.assertEqual(memory[train.PARTY_SPECIES_ADDRS[1]], 180)
        self.assertEqual(memory[train.PARTY_SPECIES_ADDRS[0] + 1], 31)
        self.assertEqual(memory[train.ADDR_PARTY_OT], 51)
        self.assertEqual(memory[train.ADDR_PARTY_NICKS], 71)

    def test_post_charizard_curriculum_targets_lavender_center(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        start = names.index('return_to_lavender_after_charizard')
        self.assertEqual(names[start:start + 3], [
            'return_to_lavender_after_charizard',
            'heal_before_pokemon_tower',
            'enter_pokemon_tower',
        ])
        self.assertEqual(train.FULLGAME_QUEST_WAYPOINTS[start]['map'], 4)
        self.assertEqual(train.FULLGAME_QUEST_WAYPOINTS[start + 1]['map'], 141)

    def test_tower_silph_and_sabrina_objectives_use_exact_yellow_events(self):
        by_name = {
            item['name']: item for item in train.FULLGAME_QUEST_WAYPOINTS
        }
        self.assertEqual(by_name['beat_pokemon_tower_rival']['event_bit'], 239)
        self.assertEqual(
            by_name['beat_pokemon_tower_jessie_james']['event_bit'], 273
        )
        self.assertEqual(by_name['receive_poke_flute']['event_bit'], 296)
        self.assertEqual(by_name['enter_silph_co']['map'], 181)
        self.assertEqual(by_name['collect_silph_card_key']['bag_item_id'], 0x30)
        self.assertEqual(by_name['beat_silph_co_rival']['event_bit'], 1856)
        self.assertEqual(by_name['receive_lapras']['status_flag_addr'], 0xD72D)
        self.assertEqual(by_name['receive_lapras']['status_flag_mask'], 0x01)
        self.assertEqual(
            by_name['beat_silph_co_11f_jessie_james']['event_bit'], 1924
        )
        self.assertEqual(by_name['beat_silph_co_giovanni']['event_bit'], 1935)
        self.assertEqual(by_name['receive_master_ball']['event_bit'], 1933)
        self.assertEqual(by_name['enter_saffron_gym']['map'], 178)
        self.assertEqual(by_name['beat_sabrina']['event_bit'], 865)

    def test_tower_and_silph_maps_receive_trainer_battle_assistance(self):
        tower_silph_maps = set(range(142, 149)) | {
            178, 181, 207, 208, 209, 210, 211, 212, 213,
            233, 234, 235, 236,
        }
        # Assert the canonical set itself; milestone wiring is constructed
        # from these exact IDs immediately before environment creation.
        self.assertIn(142, tower_silph_maps)
        self.assertIn(235, tower_silph_maps)
        self.assertIn(178, tower_silph_maps)

    def test_saffron_to_lavender_route_has_verified_transitions(self):
        steps = train.SAFFRON_TO_LAVENDER_STEPS
        self.assertEqual(steps[0], (10, 18, 36, 3, None))
        transitions = [
            (map_id, y, x, destination)
            for map_id, y, x, _, destination in steps
            if destination is not None
        ]
        self.assertEqual(transitions, [
            (10, 18, 39, 19),
            (19, 10, 1, 79),
            (79, 3, 5, 19),
            (19, 8, 59, 4),
        ])
        self.assertEqual(len(steps), 86)

    def test_post_charizard_lavender_center_approach_is_complete(self):
        self.assertEqual(
            train.POST_CHARIZARD_LAVENDER_CENTER_STEPS,
            (
                (4, 8, 0, 0, None),
                (4, 7, 0, 0, None),
                (4, 6, 0, 3, None),
                (4, 6, 1, 3, None),
                (4, 6, 2, 3, None),
                (4, 6, 3, 0, 141),
            ),
        )

    def test_lavender_center_frontier_routes_directly_to_pokemon_tower(self):
        steps = train.LAVENDER_CENTER_TO_TOWER_STEPS
        self.assertEqual(steps[0], (141, 4, 6, 2, None))
        self.assertEqual(steps[-1], (4, 6, 14, 0, 142))
        self.assertEqual(
            [
                (map_id, y, x, destination)
                for map_id, y, x, _, destination in steps
                if destination is not None
            ],
            [(141, 7, 4, 4), (4, 6, 14, 142)],
        )

    def test_tower_3f_route_reaches_viewer_tile_405_229_and_4f(self):
        steps = train.POKEMON_TOWER_3F_TO_4F_STEPS
        self.assertEqual(steps[0], (143, 10, 6, 2, None))
        self.assertEqual(steps[-1], (144, 9, 17, 3, 145))
        self.assertEqual(len(steps), 33)
        self.assertEqual(
            [
                (map_id, y, x, destination)
                for map_id, y, x, _, destination in steps
                if destination is not None
            ],
            [(143, 9, 4, 144), (144, 9, 17, 145)],
        )

    def test_tower_4f_route_reaches_viewer_tile_369_229_and_5f(self):
        steps = train.POKEMON_TOWER_4F_TO_5F_STEPS
        self.assertEqual(steps[0], (145, 7, 16, 1, None))
        self.assertEqual(steps[-1], (145, 10, 3, 0, 146))
        self.assertEqual(len(steps), 21)
        self.assertEqual(
            [
                (map_id, y, x, destination)
                for map_id, y, x, _, destination in steps
                if destination is not None
            ],
            [(145, 10, 3, 146)],
        )

    def test_tower_2f_post_rival_route_reaches_3f(self):
        steps = train.POKEMON_TOWER_2F_TO_3F_STEPS
        self.assertEqual(steps[0], (143, 9, 16, 0, None))
        self.assertEqual(steps[-1], (143, 9, 4, 2, 144))
        self.assertEqual(len(steps), 21)
        self.assertIn((143, 5, 7), [step[:3] for step in steps])
        self.assertIn((143, 7, 5), [step[:3] for step in steps])

    def test_tower_5f_route_heals_at_398_210_then_reaches_405_210(self):
        steps = train.POKEMON_TOWER_5F_TO_6F_STEPS
        self.assertEqual(steps[0], (146, 9, 3, 3, None))
        self.assertEqual(steps[-1], (146, 8, 18, 1, 147))
        self.assertEqual(len(steps), 29)
        pad_index = next(
            index for index, step in enumerate(steps)
            if step[:3] == (146, 9, 11)
        )
        self.assertGreater(pad_index, 0)
        self.assertLess(pad_index, len(steps) - 1)
        self.assertNotIn((146, 8, 12), [step[:3] for step in steps])
        self.assertNotIn((146, 7, 11), [step[:3] for step in steps])

    def test_repel_preserves_real_tower_trainers_but_not_wild_battles(self):
        phase = train.FULLGAME_POKEMON_TOWER_5F_QUEST_PHASE
        self.assertTrue(train.force_repel_preserves_trainer_battle(145, phase, 2))
        self.assertFalse(train.force_repel_preserves_trainer_battle(145, phase, 1))
        self.assertFalse(train.force_repel_preserves_trainer_battle(4, phase, 2))

    def test_tower_5f_post_heal_ignores_reasserted_text_and_exits_east(self):
        phase = train.FULLGAME_POKEMON_TOWER_6F_QUEST_PHASE
        expected = {
            (9, 12): 'right',
            (9, 13): 'up',
            (8, 13): 'up',
            (7, 13): 'right',
            (8, 18): 'down',
        }
        for position, action in expected.items():
            self.assertEqual(
                train.pokemon_tower_5f_post_heal_action(
                    146, phase, *position, 0,
                ),
                action,
            )
        self.assertIsNone(
            train.pokemon_tower_5f_post_heal_action(146, phase, 9, 11, 0)
        )
        self.assertIsNone(
            train.pokemon_tower_5f_post_heal_action(146, phase, 9, 12, 2)
        )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 7, 16, 0, trainer_defeated=True,
            ),
            'right',
        )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 7, 16, 0, trainer_defeated=False,
                same_position_steps=1,
            ),
            'a',
        )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 7, 16, 0, trainer_defeated=False,
            ),
            'a',
        )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 6, 17, 0, trainer_defeated=True,
            ),
            'left',
        )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 6, 16, 0, trainer_defeated=True,
            ),
            'down',
        )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 8, 16, 0, trainer_defeated=True,
            ),
            'right',
        )
        self.assertEqual(
            train.pokemon_tower_5f_post_heal_action(
                146, phase, 8, 17, 0, trainer_defeated=True,
            ),
            'right',
        )

    def test_tower_5f_clears_stale_text_only_after_defeated_channeler(self):
        phase = train.FULLGAME_POKEMON_TOWER_6F_QUEST_PHASE
        self.assertTrue(train.pokemon_tower_5f_defeated_trainer_text_is_stale(
            146, phase, 6, 17, 0, True,
        ))
        self.assertFalse(train.pokemon_tower_5f_defeated_trainer_text_is_stale(
            146, phase, 6, 17, 0, False,
        ))
        self.assertFalse(train.pokemon_tower_5f_defeated_trainer_text_is_stale(
            146, phase, 6, 17, 2, True,
        ))
        self.assertFalse(train.pokemon_tower_5f_defeated_trainer_text_is_stale(
            145, phase, 6, 17, 0, True,
        ))
        self.assertTrue(
            train.pokemon_tower_5f_gate_trainer_collision_is_stale(
                146, phase, True,
            )
        )
        self.assertFalse(
            train.pokemon_tower_5f_gate_trainer_collision_is_stale(
                146, phase, False,
            )
        )

    def test_tower_6f_route_triggers_marowak_then_uses_375_217_stairs(self):
        steps = train.POKEMON_TOWER_6F_MAROWAK_TO_7F_STEPS
        self.assertEqual(steps[0], (147, 9, 18, 0, None))
        self.assertEqual(steps[-2], (147, 15, 10, 1, None))
        self.assertEqual(steps[-1], (147, 16, 10, 2, 148))
        self.assertEqual(len(steps), 36)
        # The direct route crosses the mandatory Channeler sightlines but
        # never enters the optional X Accuracy cul-de-sac at (14,14).
        self.assertIn((147, 5, 15), [step[:3] for step in steps])
        self.assertIn((147, 7, 9), [step[:3] for step in steps])
        self.assertNotIn((147, 14, 14), [step[:3] for step in steps])
        self.assertIn((147, 7, 6), [step[:3] for step in steps])

    def test_only_scripted_6f_marowak_gets_ground_safe_move_selection(self):
        phase = train.FULLGAME_POKEMON_TOWER_7F_QUEST_PHASE
        self.assertTrue(
            train.pokemon_tower_marowak_battle_active(
                147, phase, 16, 10, True, False,
            )
        )
        self.assertFalse(
            train.pokemon_tower_marowak_battle_active(
                147, phase, 16, 10, True, True,
            )
        )
        self.assertFalse(
            train.pokemon_tower_marowak_battle_active(
                147, phase, 7, 9, True, False,
            )
        )

    def test_late_game_worker_roles_split_training_and_exploration(self):
        env = self.make_env()
        training_rule = {'training_workers_only': True}
        explorer_rule = {'explorer_workers_only': True}
        env.is_swarm_explorer = False
        self.assertTrue(env._action_guidance_event_allowed(training_rule))
        self.assertFalse(env._action_guidance_event_allowed(explorer_rule))
        env.is_swarm_explorer = True
        self.assertFalse(env._action_guidance_event_allowed(training_rule))
        self.assertTrue(env._action_guidance_event_allowed(explorer_rule))

        phase = min(train.FULLGAME_CHARMANDER_GRIND_PHASES)
        self.assertFalse(train.late_game_explorer_should_reset(True, phase, 10))
        self.assertFalse(train.late_game_explorer_should_reset(True, phase, 165))
        self.assertTrue(train.late_game_explorer_should_reset(True, phase, 6))
        self.assertFalse(train.late_game_explorer_should_reset(False, phase, 6))

    def test_explorers_enter_saffron_from_route7_frontier(self):
        self.assertEqual(
            train.SAFFRON_EXPLORER_ENTRY_STEPS,
            ((18, 2, 18, 3, None), (18, 2, 19, 3, 10)),
        )

    def test_charmander_workers_take_north_gate_to_cerulean_center(self):
        actions = {
            (map_id, y, x): (action, destination)
            for map_id, y, x, action, destination
            in train.SAFFRON_TO_CERULEAN_CENTER_STEPS
        }
        self.assertEqual(actions[(10, 10, 0)][0], 3)
        self.assertEqual(actions[(10, 0, 18)], (0, 16))
        self.assertEqual(actions[(16, 35, 8)][0], 3)
        self.assertEqual(actions[(16, 34, 10)], (0, 70))
        self.assertEqual(actions[(70, 5, 3)][0], 0)
        self.assertEqual(actions[(70, 0, 4)], (0, 16))
        self.assertEqual(actions[(16, 29, 10)][0], 0)
        self.assertEqual(actions[(16, 24, 10)][0], 3)
        self.assertEqual(actions[(16, 0, 15)], (0, 3))
        self.assertEqual(actions[(3, 18, 32)][0], 1)
        self.assertEqual(actions[(3, 18, 19)], (0, 64))

        phase = min(train.FULLGAME_CHARMANDER_GRIND_PHASES)
        for map_id in (3, 10, 15, 16, 18, 64, 70):
            self.assertTrue(
                train.fullgame_frontier_position_allowed(
                    phase, map_id, 20, 20
                ),
                map_id,
            )

    def test_charmander_route4_patrol_starts_at_requested_viewer_tile(self):
        def first_action(map_id, y, x):
            return next(
                action
                for step_map, step_y, step_x, action, _
                in train.CHARMANDER_ROUTE4_TARGET_STEPS
                if (step_map, step_y, step_x) == (map_id, y, x)
            )

        # Route matching is ordered, so verify the first rule at repeated tiles.
        self.assertEqual(first_action(15, 13, 69), 0)
        self.assertEqual(first_action(15, 12, 69), 2)
        self.assertEqual(first_action(15, 12, 68), 3)

    def test_fly_is_only_taught_to_charizard_after_real_hm02_event(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_TEACH_FLY_QUEST_PHASE
        memory[train.ADDR_PARTY_SIZE] = 2
        memory[train.PARTY_SPECIES_ADDRS[0]] = train.GEN1_PIKACHU_SPECIES_ID
        memory[train.PARTY_SPECIES_ADDRS[1]] = train.GEN1_CHARIZARD_SPECIES_ID
        for addr, move_id in zip(train.PARTY_MOVE_ID_ADDRS[1], (10, 45, 52, 0)):
            memory[addr] = move_id

        self.assertFalse(env._ensure_charizard_knows_fly())
        bit = 1230
        memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= 1 << (bit % 8)
        self.assertTrue(env._ensure_charizard_knows_fly())
        self.assertIn(
            train.GEN1_FLY_MOVE_ID,
            [memory[addr] for addr in train.PARTY_MOVE_ID_ADDRS[1]],
        )
        self.assertNotIn(
            train.GEN1_FLY_MOVE_ID,
            [memory[addr] for addr in train.PARTY_MOVE_ID_ADDRS[0]],
        )

    def test_surf_is_only_taught_to_squirtle_after_real_hm03_event(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_TEACH_SURF_QUEST_PHASE
        memory[train.ADDR_PARTY_SIZE] = 2
        memory[train.PARTY_SPECIES_ADDRS[0]] = train.GEN1_PIKACHU_SPECIES_ID
        memory[train.PARTY_SPECIES_ADDRS[1]] = 177
        for addr, move_id in zip(
            train.PARTY_MOVE_ID_ADDRS[1], (91, 39, 145, 61)
        ):
            memory[addr] = move_id

        self.assertFalse(env._ensure_squirtle_knows_surf())
        bit = train.EVENT_GOT_HM03
        memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= 1 << (bit % 8)
        self.assertTrue(env._ensure_squirtle_knows_surf())
        moves = [memory[addr] for addr in train.PARTY_MOVE_ID_ADDRS[1]]
        self.assertEqual(moves, [91, train.GEN1_SURF_MOVE_ID, 145, 61])
        self.assertNotIn(
            train.GEN1_SURF_MOVE_ID,
            [memory[addr] for addr in train.PARTY_MOVE_ID_ADDRS[0]],
        )

    def test_vermilion_gym_guidance_matches_verified_post_cut_route(self):
        route = {
            (rule['map'], *tuple(rule['target'])): rule['action']
            for rule in train.VERMILION_GYM_ACTION_GUIDANCE
        }
        self.assertEqual(route[(17, 28, 8)], 1)
        self.assertEqual(route[(17, 35, 8)], 1)
        self.assertEqual(route[(5, 0, 18)], 1)
        self.assertEqual(route[(5, 17, 18)], 2)
        self.assertEqual(route[(5, 17, 14)], 1)
        self.assertEqual(route[(5, 18, 14)], 3)
        self.assertEqual(route[(5, 18, 15)], 1)
        self.assertEqual(route[(5, 19, 15)], 1)
        self.assertEqual(route[(5, 20, 15)], 2)
        self.assertEqual(route[(5, 20, 14)], 2)
        self.assertEqual(route[(5, 20, 13)], 2)
        self.assertEqual(route[(5, 20, 12)], 0)
        final_rule = next(
            rule for rule in train.VERMILION_GYM_ACTION_GUIDANCE
            if rule['map'] == 5 and tuple(rule['target']) == (20, 12)
        )
        self.assertEqual(final_rule['destination_map'], 92)
        route6_transition = next(
            rule for rule in train.VERMILION_GYM_ACTION_GUIDANCE
            if rule['map'] == 17 and tuple(rule['target']) == (35, 8)
        )
        self.assertEqual(route6_transition['destination_map'], 5)

    def test_post_surge_heal_and_squirtle_guidance_are_phase_safe(self):
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_POST_SURGE_HEAL_QUEST_PHASE
            ),
            0.0,
        )
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(
                0.4, train.FULLGAME_SQUIRTLE_QUEST_PHASE
            ),
            0.4,
        )
        heal_route = {
            (rule['map'], *tuple(rule['target'])): rule
            for rule in train.VERMILION_POST_SURGE_HEAL_ACTION_GUIDANCE
        }
        self.assertEqual(heal_route[(92, 2, 5)]['action'], 1)
        self.assertEqual(heal_route[(92, 6, 5)]['action'], 2)
        self.assertEqual(heal_route[(92, 13, 4)]['action'], 3)
        self.assertEqual(heal_route[(92, 17, 5)]['destination_map'], 5)
        self.assertEqual(heal_route[(5, 20, 12)]['action'], 3)
        self.assertEqual(heal_route[(5, 20, 15)]['action'], 0)
        self.assertEqual(heal_route[(5, 19, 15)]['action'], 0)
        self.assertEqual(heal_route[(5, 17, 14)]['action'], 0)
        self.assertEqual(heal_route[(5, 14, 14)]['action'], 3)
        self.assertEqual(heal_route[(5, 8, 18)]['action'], 2)
        self.assertEqual(heal_route[(5, 4, 11)]['destination_map'], 89)
        self.assertEqual(heal_route[(89, 3, 3)]['face_action'], 0)
        jenny = {
            tuple(rule['target']): rule
            for rule in train.VERMILION_SQUIRTLE_ACTION_GUIDANCE
        }
        self.assertEqual(set(jenny), {
            (15, 18), (15, 20), (14, 19), (16, 19)
        })
        self.assertTrue(all(
            rule['unless_event_bits'] == (327,) for rule in jenny.values()
        ))

    def test_route9_approach_and_cut_guidance_match_live_frontier_replay(self):
        rules = train.ROUTE9_APPROACH_ACTION_GUIDANCE
        self.assertEqual(len(rules), 85)
        targets = [
            (rule['map'], *tuple(rule['target']))
            for rule in rules
        ]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(targets[0], (35, 8, 5))
        self.assertEqual(rules[0]['action'], 1)
        self.assertEqual(targets[-1], (3, 16, 39))
        self.assertEqual(rules[-1]['action'], 3)
        transitions = {
            (rule['map'], *tuple(rule['target'])): rule['destination_map']
            for rule in rules
            if 'destination_map' in rule
        }
        self.assertEqual(transitions, train.ROUTE9_APPROACH_TRANSITIONS)
        cut_route = {
            tuple(rule['target']): rule['action']
            for rule in train.ROUTE9_CUT_ACTION_GUIDANCE
        }
        self.assertEqual(cut_route, {
            (8, 0): 3,
            (8, 1): 3,
            (8, 2): 3,
            (8, 3): 3,
            (8, 4): 3,
        })

    def test_cerulean_to_vermilion_return_route_is_complete_and_unambiguous(self):
        rules = train.VERMILION_RETURN_ACTION_GUIDANCE
        self.assertEqual(len(rules), 191)
        targets = [
            (rule['map'], *tuple(rule['target']))
            for rule in rules
        ]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(targets[0], (3, 18, 7))
        self.assertEqual(rules[0]['action'], 3)
        self.assertEqual(targets[-1], (17, 35, 9))
        self.assertEqual(rules[-1]['action'], 1)
        transitions = {
            (rule['map'], *tuple(rule['target'])): rule['destination_map']
            for rule in rules
            if 'destination_map' in rule
        }
        self.assertEqual(transitions, train.VERMILION_RETURN_TRANSITIONS)
        self.assertEqual(transitions[(17, 35, 9)], 5)

    def test_cerulean_center_route5_and_gym_do_not_regress_route_rank(self):
        key = train.post_brock_route_progress_key
        self.assertEqual(key(3, 19, 4, 7), (7, 0))
        self.assertEqual(key(64, 7, 3, 7), (7, 0))
        self.assertEqual(key(65, 13, 4, 7), (7, 0))
        self.assertEqual(key(16, 0, 17, 7), (7, 0))
        self.assertEqual(key(15, 13, 69, 7), (7, 0))
        self.assertEqual(key(15, 13, 69, 6), (6, 69))
        self.assertEqual(key(35, 10, 31, 7), (8, 0))
        self.assertEqual(key(36, 8, 1, 8), (8, 0))
        self.assertEqual(key(88, 7, 2, 8), (8, 0))
        self.assertEqual(key(3, 6, 20, 8), (8, 0))
        self.assertEqual(key(16, 1, 10, 8), (8, 0))
        self.assertEqual(key(5, 0, 10, 8), (8, 0))

    def test_late_charmander_route4_approach_advances_same_story_tier(self):
        key = train.post_brock_route_progress_key
        self.assertEqual(key(10, 2, 2, 8), (8, 0))
        self.assertEqual(key(70, 5, 4, 8), (8, 0))
        self.assertEqual(key(15, 10, 89, 8), (8, 101))
        self.assertEqual(key(15, 13, 69, 8), (8, 121))

    def test_route25_heal_precedes_bill_interaction(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[46:50], [
            'enter_bills_house',
            'heal_after_route_25',
            'return_to_bills_house',
            'meet_bill',
        ])
        heal = train.FULLGAME_QUEST_WAYPOINTS[
            train.FULLGAME_ROUTE25_HEAL_QUEST_PHASE
        ]
        self.assertFalse(train.quest_waypoint_matches(
            heal, 64, 3, 3, set(), total_party_hp=84, total_party_max_hp=85
        ))
        self.assertTrue(train.quest_waypoint_matches(
            heal, 64, 3, 3, set(), total_party_hp=85, total_party_max_hp=85
        ))

    def test_repel_covers_recovery_legs_and_rock_tunnel(self):
        env = self.make_env()
        env.force_repel_quest_phases = frozenset({
            train.FULLGAME_ROUTE25_HEAL_QUEST_PHASE,
            train.FULLGAME_RETURN_TO_BILLS_HOUSE_QUEST_PHASE,
            train.FULLGAME_MEET_BILL_QUEST_PHASE,
        } | set(range(
            train.FULLGAME_LAVENDER_QUEST_PHASE,
            train.FULLGAME_CELADON_HEAL_QUEST_PHASE + 1,
        )) | (set(range(
            train.FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE,
            train.FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE + 1,
        )) - set(train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES)))

        env.quest_phase = train.FULLGAME_ROUTE25_HEAL_QUEST_PHASE
        self.assertTrue(env._force_repel_is_active())
        self.assertEqual(env._effective_battle_flag(1), 0)
        env.enemy_hp_battle_proxy_maps = None
        self.assertFalse(env._enemy_hp_can_indicate_battle(36))
        env.quest_phase = train.FULLGAME_RETURN_TO_BILLS_HOUSE_QUEST_PHASE
        self.assertTrue(env._force_repel_is_active())
        env.quest_phase = train.FULLGAME_MEET_BILL_QUEST_PHASE
        self.assertTrue(env._force_repel_is_active())
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE
        self.assertTrue(env._force_repel_is_active())
        env.quest_phase = next(iter(train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES))
        self.assertFalse(env._force_repel_is_active())
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE
        self.assertTrue(env._force_repel_is_active())
        env.quest_phase = train.FULLGAME_LAVENDER_QUEST_PHASE
        env.pyboy.memory[train.ADDR_MAP_ID] = 21
        self.assertTrue(env._force_repel_is_active())
        self.assertEqual(env._effective_battle_flag(1), 0)
        self.assertEqual(env._effective_battle_flag(2), 2)
        env.quest_phase = train.FULLGAME_LAVENDER_HEAL_QUEST_PHASE
        self.assertTrue(env._force_repel_is_active())
        env.quest_phase = train.FULLGAME_ROUTE8_QUEST_PHASE
        self.assertTrue(env._force_repel_is_active())
        env.quest_phase = train.FULLGAME_CELADON_HEAL_QUEST_PHASE
        self.assertTrue(env._force_repel_is_active())
        env.quest_phase = next(iter(train.FULLGAME_CERULEAN_GRIND_PHASES))
        self.assertFalse(env._force_repel_is_active())
        self.assertEqual(env._effective_battle_flag(1), 1)
        self.assertTrue(env._enemy_hp_can_indicate_battle(36))
        env.force_repel = True
        self.assertTrue(env._force_repel_is_active())

    def test_rock_tunnel_grind_subphase_uses_lead_xp_only_in_grind(self):
        phase = next(iter(train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES))
        self.assertEqual(
            train.fullgame_swarm_subphase_value(phase, 17, 46865),
            46865,
        )
        self.assertEqual(
            train.fullgame_swarm_subphase_value(
                train.FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE, 17, 46865
            ),
            17,
        )
        charmander_phase = next(iter(train.FULLGAME_CHARMANDER_GRIND_PHASES))
        self.assertEqual(
            train.fullgame_swarm_subphase_value(
                charmander_phase, 121, 1000
            ),
            1000121,
        )
        mansion_phase = next(iter(train.FULLGAME_MANSION_GRIND_PHASES))
        self.assertEqual(
            train.fullgame_swarm_subphase_value(
                mansion_phase, 17, 132846
            ),
            132846,
        )

    def test_rock_tunnel_grind_resources_restore_on_every_serialized_xp_frontier(self):
        phase = next(iter(train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES))
        self.assertTrue(
            train.should_restore_rock_tunnel_grind_frontier_resources(
                phase, phase - 1, 47000
            )
        )
        self.assertTrue(
            train.should_restore_rock_tunnel_grind_frontier_resources(
                phase, phase, 0
            )
        )
        self.assertTrue(
            train.should_restore_rock_tunnel_grind_frontier_resources(
                phase, phase, 47000
            )
        )
        self.assertFalse(
            train.should_restore_rock_tunnel_grind_frontier_resources(
                train.FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE, phase, 0
            )
        )
        charmander_phase = next(iter(train.FULLGAME_CHARMANDER_GRIND_PHASES))
        self.assertTrue(
            train.should_restore_rock_tunnel_grind_frontier_resources(
                charmander_phase, charmander_phase, 0
            )
        )

    def test_swarm_frontier_directory_lock_is_exclusive_and_releasable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / 'frontier.write_lock'
            self.assertTrue(
                train.try_acquire_swarm_frontier_write_lock(lock_path)
            )
            self.assertFalse(
                train.try_acquire_swarm_frontier_write_lock(lock_path)
            )
            train.release_swarm_frontier_write_lock(lock_path)
            self.assertTrue(
                train.try_acquire_swarm_frontier_write_lock(lock_path)
            )
            train.release_swarm_frontier_write_lock(lock_path)

    def test_bill_guidance_switches_from_bill_to_pc_by_event(self):
        env = self.make_env()
        env.action_guidance = tuple(train.BILLS_HOUSE_ACTION_GUIDANCE)
        env.action_guidance_requires_overworld = True
        env.quest_phase = train.FULLGAME_MEET_BILL_QUEST_PHASE
        env._forced_interaction_face_key = None

        self.assertEqual(env._matching_forced_action_guidance(36, 7, 49, 0), 2)
        self.assertEqual(env._matching_forced_action_guidance(36, 7, 48, 0), 0)
        self.assertEqual(env._matching_forced_action_guidance(36, 4, 48, 0), 2)
        self.assertEqual(env._matching_forced_action_guidance(36, 4, 45, 0), 0)
        self.assertEqual(env._matching_forced_action_guidance(88, 7, 2, 0), 0)
        self.assertEqual(env._matching_forced_action_guidance(88, 6, 2, 0), 3)
        self.assertEqual(env._matching_forced_action_guidance(88, 6, 5, 0), 0)
        self.assertEqual(env._matching_forced_action_guidance(88, 5, 5, 0), 3)
        self.assertEqual(env._matching_forced_action_guidance(88, 5, 5, 0), 4)

        bill_said_addr = train.ADDR_EVENT_FLAGS_START + 1374 // 8
        env.pyboy.memory[bill_said_addr] |= 1 << (1374 % 8)
        env._forced_interaction_face_key = None
        self.assertEqual(env._matching_forced_action_guidance(88, 6, 6, 0), 2)
        self.assertEqual(env._matching_forced_action_guidance(88, 5, 5, 0), 2)
        self.assertEqual(env._matching_forced_action_guidance(88, 5, 1, 0), 0)
        self.assertEqual(env._matching_forced_action_guidance(88, 5, 1, 0), 4)

        separator_addr = train.ADDR_EVENT_FLAGS_START + 1371 // 8
        env.pyboy.memory[separator_addr] |= 1 << (1371 % 8)
        self.assertIsNone(env._matching_forced_action_guidance(88, 5, 1, 0))

    def test_cerulean_guidance_targets_center_route4_grass_and_misty(self):
        self.assertTrue(any(
            rule['map'] == 3 and rule['target'] == (18, 19)
            and rule.get('force_action')
            for rule in train.CERULEAN_CENTER_ACTION_GUIDANCE
        ))
        self.assertTrue(any(
            rule['map'] == 15 and rule['target'] == (13, 69)
            and rule['action'] == 3 and rule.get('force_action')
            for rule in train.CERULEAN_GRIND_ACTION_GUIDANCE
        ))
        self.assertTrue(any(
            rule['map'] == 3 and rule['target'] == (20, 30)
            and rule['action'] != 0 and rule.get('force_action')
            for rule in train.CERULEAN_GYM_REJECTION_GUIDANCE
        ))
        self.assertTrue(any(
            rule['map'] == 65 and rule['target'] == (7, 2)
            and rule['action'] == 1 and rule.get('force_action')
            for rule in train.CERULEAN_GYM_REJECTION_GUIDANCE
        ))
        self.assertTrue(any(
            rule['map'] == 65 and rule['target'] == (2, 5)
            and rule['action'] == 4 and rule['face_action'] == 2
            for rule in train.CERULEAN_GYM_ACTION_GUIDANCE
        ))
        gym_actions = {
            tuple(rule['target']): rule['action']
            for rule in train.CERULEAN_GYM_ACTION_GUIDANCE
            if rule['map'] == 65
        }
        self.assertEqual(gym_actions[(6, 5)], 0)
        self.assertEqual(gym_actions[(5, 5)], 3)
        self.assertEqual(gym_actions[(5, 6)], 3)
        self.assertEqual(gym_actions[(5, 7)], 0)
        self.assertEqual(gym_actions[(4, 7)], 0)
        self.assertEqual(gym_actions[(3, 7)], 2)

    def test_rock_tunnel_center_guidance_heals_before_cave_entry(self):
        aisle = {
            tuple(rule['target']): rule
            for rule in train.ROCK_TUNNEL_CENTER_ACTION_GUIDANCE
        }
        self.assertEqual(aisle[(7, 3)]['action'], 0)
        self.assertEqual(aisle[(4, 3)]['action'], 0)
        self.assertEqual(aisle[(3, 3)]['action'], 4)
        self.assertEqual(aisle[(3, 3)]['face_action'], 0)
        self.assertTrue(all(rule.get('force_action') for rule in aisle.values()))

    def test_rock_tunnel_ladder_waypoints_reject_backward_landings(self):
        cases = (
            (train.FULLGAME_ROCK_TUNNEL_FIRST_B1F_QUEST_PHASE,
             (232, 25, 33), (232, 3, 27)),
            (train.FULLGAME_ROCK_TUNNEL_MIDDLE_1F_QUEST_PHASE,
             (82, 3, 5), (82, 17, 37)),
            (train.FULLGAME_ROCK_TUNNEL_SECOND_B1F_QUEST_PHASE,
             (232, 11, 23), (232, 25, 33)),
            (train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE,
             (82, 17, 37), (82, 3, 5)),
        )
        for phase, forward, backward in cases:
            waypoint = train.FULLGAME_QUEST_WAYPOINTS[phase]
            self.assertTrue(train.quest_waypoint_matches(
                waypoint, *forward, set()
            ))
            self.assertFalse(train.quest_waypoint_matches(
                waypoint, *backward, set()
            ))

    def test_frontier_phase_maps_reject_rock_tunnel_backtracking(self):
        allowed = train.fullgame_frontier_position_allowed
        self.assertTrue(allowed(
            train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE, 232, 3, 3
        ))
        self.assertFalse(allowed(
            train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE, 82, 17, 37
        ))
        self.assertTrue(allowed(
            train.FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE, 82, 33, 15
        ))
        self.assertFalse(allowed(
            train.FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE, 232, 29, 37
        ))
        self.assertTrue(allowed(
            train.FULLGAME_LAVENDER_QUEST_PHASE, 21, 53, 8
        ))
        self.assertFalse(allowed(
            train.FULLGAME_LAVENDER_QUEST_PHASE, 21, 29, 8
        ))
        self.assertFalse(allowed(
            train.FULLGAME_LAVENDER_QUEST_PHASE, 82, 33, 15
        ))

    def test_tower_frontiers_cannot_regress_to_lavender_center(self):
        allowed = train.fullgame_frontier_position_allowed
        cases = (
            (train.FULLGAME_POKEMON_TOWER_ENTRY_QUEST_PHASE, 141),
            (train.FULLGAME_POKEMON_TOWER_2F_QUEST_PHASE, 142),
            (train.FULLGAME_POKEMON_TOWER_RIVAL_QUEST_PHASE, 143),
            (train.FULLGAME_POKEMON_TOWER_3F_QUEST_PHASE, 143),
            (train.FULLGAME_POKEMON_TOWER_4F_QUEST_PHASE, 144),
            (train.FULLGAME_POKEMON_TOWER_5F_QUEST_PHASE, 145),
            (train.FULLGAME_POKEMON_TOWER_6F_QUEST_PHASE, 146),
            (train.FULLGAME_POKEMON_TOWER_7F_QUEST_PHASE, 147),
            (train.FULLGAME_POKEMON_TOWER_ROCKET_QUEST_PHASE, 148),
            (train.FULLGAME_RESCUE_MR_FUJI_QUEST_PHASE, 148),
            (train.FULLGAME_ENTER_MR_FUJIS_HOUSE_QUEST_PHASE, 149),
            (train.FULLGAME_POKE_FLUTE_QUEST_PHASE, 149),
        )
        for phase, map_id in cases:
            self.assertTrue(allowed(phase, map_id, 2, 2), (phase, map_id))
            if map_id != 141:
                self.assertFalse(allowed(phase, 141, 4, 6), phase)

    def test_tower_5f_frontier_rejects_unfinished_channeler_callout(self):
        phase = train.FULLGAME_POKEMON_TOWER_6F_QUEST_PHASE
        allowed = train.fullgame_frontier_position_allowed
        self.assertTrue(allowed(phase, 146, 9, 3, set()))
        self.assertFalse(allowed(phase, 146, 7, 16, set()))
        self.assertTrue(allowed(phase, 146, 7, 16, {258}))

    def test_fuji_and_flute_guidance_are_event_gated(self):
        tower = {
            tuple(rule['target']): rule
            for rule in train.POKEMON_TOWER_7F_FUJI_ACTION_GUIDANCE
        }
        self.assertEqual(tower[(3, 9)]['face_action'], 3)
        self.assertEqual(tower[(4, 10)]['face_action'], 0)
        self.assertEqual(tower[(3, 9)]['unless_event_bits'], (279, 1231))
        house = {
            tuple(rule['target']): rule
            for rule in train.MR_FUJIS_HOUSE_FLUTE_ACTION_GUIDANCE
        }
        self.assertEqual(house[(2, 3)]['face_action'], 0)
        self.assertEqual(house[(2, 3)]['unless_event_bits'], (296,))

    def test_post_flute_route_recovers_route10_then_reverses_to_saffron(self):
        steps = train.ROUTE10_TO_SAFFRON_WITH_FLUTE_STEPS
        self.assertEqual(steps[0], (21, 58, 17, 2, None))
        self.assertIn((21, 61, 16, 0, None), steps)
        transitions = [
            (map_id, y, x, destination)
            for map_id, y, x, _, destination in steps
            if destination is not None
        ]
        self.assertEqual(transitions[0], (21, 71, 9, 4))
        self.assertEqual(transitions[1], (4, 8, 0, 19))
        self.assertEqual(transitions[-1][-1], 10)
        self.assertEqual(steps[-1][:3], (10, 18, 37))

    def test_post_flute_frontier_rejects_cave_and_eastward_maps(self):
        phase = train.FULLGAME_RETURN_SAFFRON_WITH_FLUTE_QUEST_PHASE
        allowed = train.fullgame_frontier_position_allowed
        for map_id in (149, 21, 4, 19, 79, 10):
            self.assertTrue(allowed(phase, map_id, 2, 2), map_id)
        for map_id in (23, 82, 232):
            self.assertFalse(allowed(phase, map_id, 2, 2), map_id)

    def test_silph_recovery_reaches_requested_viewer_tile_and_enters(self):
        steps = train.SILPH_CO_OUTDOOR_RECOVERY_STEPS
        self.assertEqual(steps[0], (141, 7, 2, 3, None))
        self.assertIn((4, 8, 0, 2, 19), steps)
        self.assertIn((10, 22, 18, 0, 181), steps)
        self.assertEqual(steps[-1], (10, 22, 18, 0, 181))
        rules = {
            (rule['map'], tuple(rule['target'])): rule
            for rule in train.SILPH_CO_OUTDOOR_RECOVERY_ACTION_GUIDANCE
        }
        self.assertEqual(rules[(10, (22, 18))]['action'], 0)
        self.assertEqual(rules[(10, (22, 18))]['destination_map'], 181)

    def test_silph_frontiers_cannot_regress_outdoors(self):
        allowed = train.fullgame_frontier_position_allowed
        self.assertTrue(allowed(
            train.FULLGAME_SILPH_CO_ENTRY_QUEST_PHASE, 10, 22, 18
        ))
        self.assertFalse(allowed(
            train.FULLGAME_SILPH_CO_ENTRY_QUEST_PHASE, 141, 7, 2
        ))
        for phase in range(
            train.FULLGAME_SILPH_CO_5F_QUEST_PHASE,
            train.FULLGAME_MASTER_BALL_QUEST_PHASE + 1,
        ):
            for map_id in (181, 207, 210, 212, 235, 236):
                self.assertTrue(allowed(phase, map_id, 2, 2), (phase, map_id))
            for map_id in (10, 21, 19, 4, 141, 82, 232):
                self.assertFalse(allowed(phase, map_id, 2, 2), (phase, map_id))

    def test_silph_interior_outranks_same_phase_route8_recovery(self):
        phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        self.assertEqual(
            train.fullgame_swarm_route_progress(phase, 19, 8, 0),
            (8, 0),
        )
        self.assertEqual(
            train.fullgame_swarm_route_progress(phase, 181, -1, 0),
            (9, 0),
        )
        self.assertGreater(
            train.swarm_progress_key(False, 133, phase, 9, 0),
            train.swarm_progress_key(False, 133, phase, 8, 0),
        )

    def test_silph_story_route_uses_elevator_card_key_and_rival_warp(self):
        self.assertEqual(
            train.SILPH_CO_1F_TO_ELEVATOR_STEPS[-1],
            (181, 1, 20, 0, 236),
        )
        self.assertEqual(
            train.SILPH_CO_ELEVATOR_PANEL_STEPS[-1][:3],
            (236, 1, 2),
        )
        card_steps = train.SILPH_CO_5F_TO_CARD_KEY_STEPS
        self.assertIn((210, 14, 9, 1, 233), card_steps)
        self.assertIn((233, 14, 17, 1, 210), card_steps)
        self.assertEqual(card_steps[-1][:3], (210, 16, 19))
        card_rule = train.SILPH_CO_5F_TO_CARD_KEY_ACTION_GUIDANCE[-1]
        self.assertEqual(card_rule['target'], (16, 20))
        self.assertEqual(card_rule['face_action'], 3)
        door_rules = [
            rule for rule in train.SILPH_CO_3F_RIVAL_ACTION_GUIDANCE
            if rule['target'] == (9, 18) and rule['action'] == 4
        ]
        self.assertEqual(door_rules[0]['unless_event_bits'], (1801,))
        self.assertEqual(
            train.SILPH_CO_3F_DOOR_TO_7F_STEPS[-1],
            (208, 11, 12, 2, 212),
        )

    def test_silph_post_rival_route_collects_lapras_then_reaches_11f(self):
        lapras_rule = train.SILPH_CO_7F_LAPRAS_ACTION_GUIDANCE[-1]
        self.assertEqual(lapras_rule['target'], (5, 2))
        self.assertEqual(lapras_rule['face_action'], 2)
        # 7F row y=5 is walled from x=4 eastward, so the Lapras alcove's only
        # exit is south down the x=2 column and then east along y=7 onto the
        # 11F pad at (7,5). The old right-then-down shape walked into (5,4).
        self.assertEqual(
            train.SILPH_CO_LAPRAS_TO_11F_STEPS[-1],
            (212, 7, 4, 3, 235),
        )
        self.assertNotIn(
            (212, 5, 4),
            {step[:3] for step in train.SILPH_CO_LAPRAS_TO_11F_STEPS},
        )
        self.assertEqual(
            train.SILPH_CO_11F_JESSIE_JAMES_ACTION_GUIDANCE[0]['target'],
            (2, 3),
        )
        giovanni_rules = train.SILPH_CO_11F_GIOVANNI_ACTION_GUIDANCE
        self.assertEqual(giovanni_rules[-2]['target'], (14, 6))
        self.assertEqual(giovanni_rules[-2]['unless_event_bits'], (1928,))
        self.assertEqual(giovanni_rules[-1]['require_event_bits'], (1928,))

    def test_silph_elevator_selects_5f_then_3f_from_story_phase(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 236
        memory[train.ADDR_POS_A] = 1
        memory[train.ADDR_POS_B] = 3
        env.same_position_step_count = 0
        calls = []
        env._tap_scripted_button = lambda button, press_ticks=8, release_ticks=32: (
            calls.append((button, press_ticks, release_ticks))
        )
        env.quest_phase = train.FULLGAME_SILPH_CARD_KEY_QUEST_PHASE
        self.assertTrue(env._try_select_silph_story_floor())
        self.assertEqual([button for button, _, _ in calls].count('down'), 7)
        self.assertEqual(
            [button for button, _, _ in calls[:7]],
            ['up', 'a', 'down', 'down', 'down', 'down', 'a'],
        )

        calls.clear()
        env._silph_elevator_attempts = set()
        memory[train.ADDR_NUM_BAG_ITEMS] = 1
        memory[train.ADDR_BAG_ITEMS] = 0x30
        memory[train.ADDR_BAG_ITEMS + 1] = 1
        env.quest_phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        self.assertTrue(env._try_select_silph_story_floor())
        self.assertEqual(
            [button for button, _, _ in calls[:5]],
            ['up', 'a', 'down', 'down', 'a'],
        )

    def test_silph_pad_return_false_text_latch_is_narrowly_scoped(self):
        phase = train.FULLGAME_SILPH_CARD_KEY_QUEST_PHASE
        self.assertFalse(train.silph_card_key_pad_text_is_stale(
            210, phase, 16, 9, 0, 0, 1
        ))
        self.assertTrue(train.silph_card_key_pad_text_is_stale(
            210, train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE, 15, 9, 0, 0
        ))
        self.assertFalse(train.silph_card_key_pad_text_is_stale(
            210, phase, 16, 20, 0, 0
        ))
        self.assertFalse(train.silph_card_key_pad_text_is_stale(
            210, phase, 16, 9, 2, 20
        ))
        self.assertFalse(train.silph_card_key_pad_text_is_stale(
            210, phase, 15, 9, 0, 0, 255
        ))

    def test_silph_pad_return_consumes_card_corridor_movement_directly(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 210
        memory[train.ADDR_POS_A] = 16
        memory[train.ADDR_POS_B] = 9
        memory[train.ADDR_TEXT_BOX] = 1
        env.quest_phase = train.FULLGAME_SILPH_CARD_KEY_QUEST_PHASE
        calls = []
        env._tap_scripted_button = lambda button, press_ticks=8, release_ticks=32: (
            calls.append((button, press_ticks, release_ticks))
        )
        ignored_calls = []
        env._tap_scripted_button_through_joy_ignore = (
            lambda button, press_ticks=24, release_ticks=96:
            ignored_calls.append((button, press_ticks, release_ticks))
        )
        self.assertTrue(env._try_cross_silph_pad_return())
        self.assertEqual(ignored_calls, [('a', 24, 96)])
        self.assertEqual(calls, [('left', 8, 32)])

        ignored_calls.clear()
        env._event_flag_is_set = lambda bit: (
            bit == train.EVENT_BEAT_SILPH_CO_5F_TRAINER_0
        )
        self.assertTrue(env._try_cross_silph_pad_return())
        self.assertEqual(calls, [
            ('left', 8, 32), ('right', 24, 96), ('right', 24, 96),
        ])

        memory[train.ADDR_TEXT_BOX] = 0
        self.assertTrue(env._try_cross_silph_pad_return())
        self.assertEqual(calls, [
            ('left', 8, 32),
            ('right', 24, 96), ('right', 24, 96),
            ('right', 24, 96), ('right', 24, 96),
        ])
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 0)

        calls.clear()
        memory[train.ADDR_POS_A] = 15
        env.quest_phase = train.FULLGAME_SILPH_CARD_KEY_QUEST_PHASE
        self.assertTrue(env._try_cross_silph_pad_return())
        self.assertEqual(calls, [('down', 24, 96), ('down', 24, 96)])

        calls.clear()
        memory[train.ADDR_TEXT_BOX] = 255
        env._silph_5f_pad_warp_seen = False
        self.assertTrue(env._try_cross_silph_pad_return())
        self.assertEqual(calls, [('up', 24, 96), ('up', 24, 96)])

        calls.clear()
        env._silph_5f_pad_warp_seen = True
        self.assertTrue(env._try_cross_silph_pad_return())
        self.assertEqual(calls, [('down', 24, 96), ('down', 24, 96)])

        calls.clear()
        memory[train.ADDR_TEXT_BOX] = 0
        env.quest_phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        self.assertTrue(env._try_cross_silph_pad_return())
        self.assertEqual(calls, [('up', 24, 96), ('up', 24, 96)])

    def test_silph_3f_route_trainer_is_faced_before_interaction(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 208
        memory[train.ADDR_POS_A] = 7
        memory[train.ADDR_POS_B] = 19
        memory[train.ADDR_TEXT_BOX] = 1
        env.quest_phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        calls = []
        env._tap_scripted_button = lambda button, press_ticks=8, release_ticks=32: (
            calls.append((button, press_ticks, release_ticks))
        )
        env._tap_scripted_button_through_joy_ignore = (
            lambda button, press_ticks=24, release_ticks=96:
            calls.append((button, press_ticks, release_ticks))
        )
        self.assertTrue(env._try_start_silph_3f_route_trainer())
        self.assertEqual(calls, [('right', 8, 32), ('a', 24, 96)])

        env._event_flag_is_set = lambda bit: (
            bit == train.EVENT_BEAT_SILPH_CO_3F_TRAINER_0
        )
        self.assertFalse(env._try_start_silph_3f_route_trainer())

    def test_silph_7f_dialogue_exit_alternates_b_with_the_route_move(self):
        """Both 7F story beats end on a text latch that eats D-pad input.

        The predecessor helper tapped A while zeroing wJoyIgnore, which
        desynchronised the rival's exit cutscene and made its own
        `wJoyIgnore != 0` gate false forever -- so it fired once and the
        phase-175 route then forced Down into a frozen sprite for the rest of
        the episode (4,100+ Downs on one tile, measured live).
        """
        action = train.silph_co_7f_dialogue_exit_action
        lapras_phase = train.FULLGAME_LAPRAS_QUEST_PHASE
        floor11_phase = train.FULLGAME_SILPH_CO_11F_QUEST_PHASE

        # Post-rival latch, at both of the ROM's rival trigger coordinates.
        for position in ((3, 3), (2, 3)):
            self.assertEqual(action(212, *position, lapras_phase, 0), 'b')
            self.assertEqual(action(212, *position, lapras_phase, 1), 'down')

        # Post-Lapras-gift latch on the employee's talk tile.
        self.assertEqual(action(212, 5, 2, floor11_phase, 0), 'b')
        self.assertEqual(action(212, 5, 2, floor11_phase, 1), 'down')

        # Never before the rival is beaten, off-tile, off-phase, or off-map.
        self.assertIsNone(
            action(212, 3, 3, train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE, 0)
        )
        self.assertIsNone(action(212, 5, 3, lapras_phase, 0))
        self.assertIsNone(action(212, 5, 2, lapras_phase, 0))
        self.assertIsNone(action(212, 3, 3, floor11_phase, 0))
        self.assertIsNone(action(235, 3, 3, floor11_phase, 0))

    def test_sabrina_return_route_reaches_the_gym_door_from_lavender(self):
        """Viewer (268,195) = Saffron local (y=3,x=34) = warp 3 -> SAFFRON_GYM."""
        self.assertEqual(
            train.SAFFRON_TO_SABRINA_GYM_STEPS[0][:3], (10, 18, 36),
        )
        self.assertEqual(
            train.SAFFRON_TO_SABRINA_GYM_STEPS[-1], (10, 4, 34, 0, 178),
        )
        # The post-Silph frontier saves on the healed nurse tile, four north
        # of where LAVENDER_CENTER_TO_SAFFRON_STEPS begins.
        self.assertEqual(
            train.LAVENDER_NURSE_TO_CENTER_DOOR_STEPS[0][:3], (141, 3, 3),
        )
        self.assertEqual(
            train.LAVENDER_NURSE_TO_CENTER_DOOR_STEPS[-1], (141, 7, 3, 1, 4),
        )
        # Whole-journey scoping, not one phase per leg.
        self.assertEqual(
            sorted(train.FULLGAME_SABRINA_RETURN_PHASES),
            [
                train.FULLGAME_MASTER_BALL_QUEST_PHASE,
                train.FULLGAME_RETURN_SAFFRON_AFTER_SILPH_QUEST_PHASE,
                train.FULLGAME_SAFFRON_GYM_QUEST_PHASE,
                train.FULLGAME_SABRINA_QUEST_PHASE,
                train.FULLGAME_LAVENDER_FUCHSIA_QUEST_PHASE,
            ],
        )
        # 141 Centre -> 4 Lavender -> 19 Route 8 -> 79 gatehouse -> 10 Saffron
        # -> 178 the gym interior.
        self.assertEqual(
            {rule['map'] for rule in train.SABRINA_RETURN_ACTION_GUIDANCE},
            {141, 4, 19, 79, 10, 178},
        )

    def test_saffron_gym_maze_routes_every_cell_and_releases_after_sabrina(self):
        """Saffron Gym is a 9-cell teleporter maze; nothing on map 178 was
        guided, so workers reached Sabrina by luck and wandered her room."""
        rules = train.SABRINA_ROOM_ACTION_GUIDANCE
        approach = [r for r in rules if r.get('unless_event_bits')]
        exit_rules = [r for r in rules if r.get('require_event_bits')]
        self.assertTrue(approach and exit_rules)
        # First-match-wins: a tile owned twice in one direction is a loop.
        for group in (approach, exit_rules):
            targets = [tuple(r['target']) for r in group]
            self.assertEqual(len(targets), len(set(targets)))
            self.assertEqual({r['map'] for r in group}, {178})
        # Sabrina stands on (8,9) (ROM object_event 9,8 = viewer 274,180) and
        # must never be given a rule -- nothing can stand on her.
        self.assertNotIn(
            (8, 9), {tuple(r['target']) for r in rules},
        )
        # Both directions gate on the same event, in opposite senses.
        for rule in approach:
            self.assertEqual(
                rule['unless_event_bits'], (train.EVENT_BEAT_SABRINA,)
            )
        for rule in exit_rules:
            self.assertEqual(
                rule['require_event_bits'], (train.EVENT_BEAT_SABRINA,)
            )
        # The talk tile is directly below her, faced up, and released on win.
        talk = [
            r for r in approach
            if tuple(r['target']) == (9, 9) and r['action'] == 4
        ]
        self.assertEqual(len(talk), 1)
        self.assertEqual(talk[0]['face_action'], 0)
        # The exit chain must terminate by stepping onto the door tile.
        door = [r for r in exit_rules if tuple(r['target']) == (16, 8)]
        self.assertEqual(len(door), 1)
        self.assertEqual(door[0]['action'], 1)

    def test_electric_promotion_allowed_outside_ground_rock_fights(self):
        """The Electric ban existed for Brock/Rock Tunnel but applied to every
        assisted battle, so a Pikachu lead fought the whole back half of the
        game with its second-best move. Probe: against Saffron Gym's Slowbro,
        Swift (60, neutral) took 354 A presses, Thunderbolt (95, 2x) took 139.
        """
        safe = (
            {178}
            | set(range(142, 149))
            | (train.FULLGAME_SILPH_INTERIOR_MAPS - {235})
        )
        # Saffron Gym and the Tower/Silph interiors are Psychic/Ghost/Poison.
        self.assertIn(178, safe)
        self.assertIn(212, safe)
        self.assertIn(145, safe)
        # Giovanni's Ground team on Silph 11F must keep the ban.
        self.assertNotIn(235, safe)
        # Brock's and Rock Tunnel's maps were never in the electric-safe set.
        self.assertNotIn(54, safe)
        self.assertNotIn(82, safe)
        self.assertNotIn(232, safe)
        # And the phases those maps are fought on are the Tower/Silph arc.
        self.assertIn(
            train.FULLGAME_SABRINA_QUEST_PHASE,
            train.FULLGAME_TOWER_SILPH_PHASES,
        )

    def test_switch_handoff_covers_the_late_story_battles(self):
        """The switch-prompt assist stopped at Misty, so every later story
        fight could loop in the party menu: measured live at ~2,130 battle
        steps / a:2,025 / exactly one enemy faint at full HP, on Silph 3F
        (map 208) and on all 96 workers in Saffron Gym (map 178)."""
        handoff = train.trainer_switch_handoff_action
        for map_id, phase in (
            (178, train.FULLGAME_SABRINA_QUEST_PHASE),
            (208, train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE),
            (212, train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE),
            (145, train.FULLGAME_POKEMON_TOWER_5F_QUEST_PHASE),
            (199, train.FULLGAME_HIDEOUT_B2F_QUEST_PHASE),
        ):
            self.assertEqual(
                handoff(map_id, phase, True, 0, 1, 0, 0xFF), 'b',
                msg='no switch-prompt assist on map %d phase %d'
                    % (map_id, phase),
            )
        # Still inert before an opposing Pokemon has actually fainted, and
        # outside a trainer battle.
        self.assertIsNone(
            handoff(178, train.FULLGAME_SABRINA_QUEST_PHASE, True, 0, 0, 0, 0xFF)
        )
        self.assertIsNone(
            handoff(178, train.FULLGAME_SABRINA_QUEST_PHASE, False, 0, 1, 0, 0xFF)
        )
        # Enemy still alive: A must keep attacking rather than declining.
        self.assertIsNone(
            handoff(178, train.FULLGAME_SABRINA_QUEST_PHASE, True, 40, 1, 0, 0xFF)
        )

    def test_saffron_gym_callout_recovery_pulses_a_when_stationary(self):
        """The whole 96-worker swarm froze on (16,17), three tiles below the
        Youngster at (13,17), pressing Left into an unadvanced callout."""
        recover = train.rock_tunnel_dialogue_recovery_action
        phase = train.FULLGAME_SABRINA_QUEST_PHASE
        # Sparse pulse once stationary, in the proven 4-per-8 pattern.
        self.assertEqual(recover(178, phase, 0, 4), 'a')
        self.assertEqual(recover(178, phase, 0, 7), 'a')
        self.assertIsNone(recover(178, phase, 0, 0))
        self.assertIsNone(recover(178, phase, 0, 8))
        # Never during a battle, and never off-map. Map 10 (Saffron) is no
        # longer a valid off-map example: the Part 11 coast pulse now covers
        # it deliberately, so use a map outside every scoped set.
        self.assertIsNone(recover(178, phase, 1, 4))
        self.assertIsNone(recover(2, phase, 0, 4))

    def test_post_silph_frontier_cannot_regress_to_lavender(self):
        """A post-Giovanni blackout published a healed Lavender Center state
        as the phase-180 frontier, stranding the swarm 93 forced steps from
        Saffron. The Silph guard stopped one phase short of covering it."""
        allowed = train.fullgame_frontier_position_allowed
        return_phase = train.FULLGAME_RETURN_SAFFRON_AFTER_SILPH_QUEST_PHASE
        self.assertFalse(allowed(return_phase, 141, 3, 3))
        self.assertFalse(allowed(return_phase, 4, 6, 3))
        self.assertFalse(
            allowed(train.FULLGAME_SAFFRON_GYM_QUEST_PHASE, 141, 3, 3)
        )
        self.assertTrue(
            allowed(train.FULLGAME_SAFFRON_GYM_QUEST_PHASE, 10, 18, 36)
        )
        self.assertTrue(
            allowed(train.FULLGAME_SABRINA_QUEST_PHASE, 178, 17, 8)
        )
        self.assertFalse(
            allowed(train.FULLGAME_SABRINA_QUEST_PHASE, 10, 4, 34)
        )

    def test_sabrina_battle_resources_restore_once_before_her_battle(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 178
        memory[train.ADDR_POS_A] = 17
        memory[train.ADDR_POS_B] = 8
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_PARTY_SIZE] = 2
        for index in range(2):
            cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[index]
            max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[index]
            memory[cur_hi], memory[cur_lo] = 0, 7
            memory[max_hi], memory[max_lo] = 0, 140
        env.quest_phase = train.FULLGAME_SABRINA_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        env._sabrina_resources_prepared = False

        self.assertTrue(env._prepare_sabrina_battle_resources())
        for index in range(2):
            cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[index]
            self.assertEqual(
                (int(memory[cur_hi]) << 8) | int(memory[cur_lo]), 140
            )
        self.assertFalse(env._prepare_sabrina_battle_resources())
        env._sabrina_resources_prepared = False
        env._event_flag_is_set = lambda bit: bit == train.EVENT_BEAT_SABRINA
        self.assertFalse(env._prepare_sabrina_battle_resources())

    def test_post_sabrina_exit_pulses_b_off_her_talk_tile(self):
        """Third instance of the Silph 7F class: her badge/TM text latches on
        the talk tile while the exit rule forces Right into it. Live: 288
        wins, every worker parked on (9,9) at quest_phase_stall."""
        action = train.saffron_gym_post_sabrina_exit_action
        phase = train.FULLGAME_SABRINA_QUEST_PHASE
        self.assertEqual(action(178, 9, 9, phase, True, 0), 'b')
        self.assertEqual(action(178, 9, 9, phase, True, 1), 'right')
        # Inert until she is actually beaten, and only on her tile.
        self.assertIsNone(action(178, 9, 9, phase, False, 0))
        self.assertIsNone(action(178, 9, 10, phase, True, 0))
        self.assertIsNone(action(10, 9, 9, phase, True, 0))
        # Still live on the follow-on phase that walks out to the bike shop.
        later = max(train.FULLGAME_SABRINA_RETURN_PHASES)
        self.assertEqual(action(178, 9, 9, later, True, 0), 'b')

    def test_cerulean_bike_shop_route_uses_the_only_gap_in_the_fence(self):
        """Cerulean's south band is fenced off: row 28 is solid except
        x=0-3, 16, 17, 19, 36, 37, and the x=16/17 gap sits above ledge tiles
        that only allow a southward hop. The old route walked up x=25 into
        (28,25) and pinned 79 of 96 workers on (29,25)."""
        steps = train.CERULEAN_TO_BIKE_SHOP_STEPS
        tiles = {(y, x) for _m, y, x, _a, _d in steps}
        self.assertEqual(steps[0][:3], (3, 35, 25))
        # ends by stepping onto the bike shop door, warp 5 of Cerulean City
        self.assertEqual(steps[-1], (3, 26, 13, 0, 66))
        # crosses the fence row only in the east corridor, which is the only
        # gap the emulator will actually let a worker walk north through
        fence = {x for (y, x) in tiles if y == 28}
        self.assertTrue(fence.issubset({36, 37}),
                        'fence row entered at %s' % fence)
        # and never tries the (28,25) column that pinned the swarm
        self.assertNotIn((28, 25), tiles)
        # and never tries to climb the ledge columns
        self.assertNotIn((29, 16), tiles)
        self.assertNotIn((29, 17), tiles)
        # First-match-wins forcing: a tile owned twice is an instant loop.
        # (Consecutive tiles are NOT always adjacent -- the route was derived
        # from real emulator movement and includes a ledge hop, which moves
        # two tiles at once.)
        self.assertEqual(len(tiles), len(steps))
        # Only the final step carries the warp into the shop.
        self.assertTrue(all(s[4] is None for s in steps[:-1]))

    def test_charizard_leads_saffron_gym_and_pikachu_restore_stands_down(self):
        """Pikachu's Thunderbolt is a SPECIAL move into Gen 1 Psychics' best
        stat; Charizard's Slash is PHYSICAL into their paper Defense. Probe:
        the Pikachu lead lost/timed out under every bounded heal policy over
        38,304 identical live episodes, while the Charizard lead beat her in
        894 steps using one heal."""
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 178
        memory[train.ADDR_TEXT_BOX] = 0
        memory[train.ADDR_PARTY_SIZE] = 2
        charizard = sorted(train.GEN1_CHARMANDER_SPECIES_IDS)[-1]
        memory[train.PARTY_SPECIES_ADDRS[0]] = train.GEN1_PIKACHU_SPECIES_ID
        memory[train.PARTY_SPECIES_ADDRS[1]] = charizard
        memory[train.ADDR_PARTY_SPECIES_LIST] = train.GEN1_PIKACHU_SPECIES_ID
        memory[train.ADDR_PARTY_SPECIES_LIST + 1] = charizard
        env.quest_phase = train.FULLGAME_SABRINA_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        env._effective_battle_flag = lambda: 0
        env._lead_moveset_snapshot = None

        self.assertTrue(env._lead_charizard_for_saffron_gym())
        self.assertEqual(
            int(memory[train.PARTY_SPECIES_ADDRS[0]]), charizard
        )
        # The every-step Pikachu restore must stand down here, or it swaps
        # straight back and the swap accomplishes nothing.
        self.assertFalse(env._restore_pikachu_lead_after_charizard())
        self.assertEqual(
            int(memory[train.PARTY_SPECIES_ADDRS[0]]), charizard
        )

        # Released once she is beaten, and inert off-map / off-phase.
        env._event_flag_is_set = lambda bit: bit == train.EVENT_BEAT_SABRINA
        self.assertFalse(env._lead_charizard_for_saffron_gym())
        env._event_flag_is_set = lambda bit: False
        memory[train.ADDR_MAP_ID] = 10
        self.assertFalse(env._lead_charizard_for_saffron_gym())
        memory[train.ADDR_MAP_ID] = 178
        env.quest_phase = train.FULLGAME_SAFFRON_GYM_QUEST_PHASE
        self.assertFalse(env._lead_charizard_for_saffron_gym())

    def test_sabrina_emergency_heals_are_gym_wide_and_bounded(self):
        """Three gym trainers sit in unavoidable sight lines, so the heal has
        to cover the gauntlet, not just her tile -- but stay bounded."""
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 178
        memory[train.ADDR_POS_A] = 4
        memory[train.ADDR_POS_B] = 17
        memory[train.ADDR_BATTLE_FLAG] = 2
        memory[train.ADDR_PARTY_SIZE] = 1
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
        max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[0]
        memory[train.ADDR_ACTIVE_MON_MAX_HP_HI] = 0
        memory[train.ADDR_ACTIVE_MON_MAX_HP_LO] = 150
        memory[max_hi], memory[max_lo] = 0, 150
        env.quest_phase = train.FULLGAME_SABRINA_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        env._sabrina_emergency_heals_used = 0

        for expected in range(train.SABRINA_GYM_EMERGENCY_HEAL_LIMIT):
            memory[train.ADDR_ACTIVE_MON_CUR_HP_HI] = 0
            memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 10
            memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 10
            self.assertTrue(env._use_sabrina_emergency_heal())
            self.assertEqual(
                (int(memory[train.ADDR_ACTIVE_MON_CUR_HP_HI]) << 8)
                | int(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO]),
                150,
            )
        # Bounded: no fifth heal.
        memory[train.ADDR_ACTIVE_MON_CUR_HP_HI] = 0
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 10
        self.assertFalse(env._use_sabrina_emergency_heal())
        # And never once she is beaten.
        env._sabrina_emergency_heals_used = 0
        env._event_flag_is_set = lambda bit: bit == train.EVENT_BEAT_SABRINA
        self.assertFalse(env._use_sabrina_emergency_heal())

    def test_silph_rival_battle_resources_restore_once_before_his_battle(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 212
        memory[train.ADDR_POS_A] = 3
        memory[train.ADDR_POS_B] = 3
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_PARTY_SIZE] = 2
        for index in range(2):
            cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[index]
            max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[index]
            memory[cur_hi], memory[cur_lo] = 0, 5
            memory[max_hi], memory[max_lo] = 0, 120
        env.quest_phase = train.FULLGAME_SILPH_CO_RIVAL_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        env._silph_rival_resources_prepared = False

        self.assertTrue(env._prepare_silph_rival_battle_resources())
        for index in range(2):
            cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[index]
            self.assertEqual(
                (int(memory[cur_hi]) << 8) | int(memory[cur_lo]), 120
            )
        # Once only, and never after he is already beaten.
        self.assertFalse(env._prepare_silph_rival_battle_resources())
        env._silph_rival_resources_prepared = False
        env._event_flag_is_set = (
            lambda bit: bit == train.EVENT_BEAT_SILPH_CO_RIVAL
        )
        self.assertFalse(env._prepare_silph_rival_battle_resources())

    def test_lavender_and_celadon_center_guidance_force_nurse_interaction(self):
        for map_id, guidance in (
            (141, train.LAVENDER_CENTER_ACTION_GUIDANCE),
            (133, train.CELADON_CENTER_ACTION_GUIDANCE),
        ):
            aisle = {tuple(rule['target']): rule for rule in guidance}
            self.assertEqual(aisle[(7, 3)]['map'], map_id)
            self.assertEqual(aisle[(7, 3)]['action'], 0)
            self.assertEqual(aisle[(3, 3)]['action'], 4)
            self.assertEqual(aisle[(3, 3)]['face_action'], 0)
            self.assertTrue(all(
                rule.get('force_action') for rule in aisle.values()
            ))

    def test_lavender_route8_guidance_exits_center_and_town_west(self):
        rules = {
            (rule['map'], tuple(rule['target'])): rule
            for rule in train.LAVENDER_ROUTE8_ACTION_GUIDANCE
        }
        self.assertEqual(rules[(141, (3, 3))]['action'], 1)
        self.assertEqual(rules[(141, (7, 3))]['destination_map'], 4)
        self.assertEqual(rules[(4, (17, 9))]['action'], 0)
        self.assertEqual(rules[(4, (9, 9))]['action'], 2)
        self.assertEqual(rules[(4, (9, 0))]['destination_map'], 19)
        self.assertTrue(all(
            rule.get('force_action')
            for rule in train.LAVENDER_ROUTE8_ACTION_GUIDANCE
        ))

    def test_route8_underground_guidance_finishes_authoritative_warp(self):
        rules = {
            tuple(rule['target']): rule
            for rule in train.ROUTE8_UNDERGROUND_ENTRY_ACTION_GUIDANCE
        }
        self.assertEqual(len(train.ROUTE8_UNDERGROUND_ENTRY_ROUTE_STEPS), 56)
        self.assertEqual(rules[(15, 45)]['action'], 2)
        self.assertEqual(rules[(15, 44)]['action'], 0)
        self.assertEqual(rules[(6, 41)]['action'], 2)
        self.assertEqual(rules[(4, 16)]['action'], 2)
        self.assertEqual(rules[(4, 13)]['action'], 0)
        self.assertEqual(rules[(4, 13)]['destination_map'], 80)
        self.assertTrue(all(
            rule.get('force_action')
            for rule in train.ROUTE8_UNDERGROUND_ENTRY_ACTION_GUIDANCE
        ))

    def test_underground_route7_and_celadon_routes_match_replayed_geometry(self):
        cases = (
            (
                train.ROUTE8_GATEHOUSE_UNDERGROUND_ROUTE_STEPS,
                4, (80, 7, 3), 121,
            ),
            (
                train.UNDERGROUND_WESTBOUND_ROUTE_STEPS,
                48, (121, 2, 47), 77,
            ),
            (
                train.ROUTE7_GATEHOUSE_EXIT_ROUTE_STEPS,
                4, (77, 4, 4), 18,
            ),
            (
                train.ROUTE7_CELADON_ROUTE_STEPS,
                23, (18, 14, 5), 6,
            ),
            (
                train.CELADON_CENTER_ENTRY_ROUTE_STEPS,
                10, (6, 11, 49), 133,
            ),
        )
        for route, expected_length, start, destination in cases:
            self.assertEqual(len(route), expected_length)
            self.assertEqual(route[0][:3], start)
            self.assertEqual(route[-1][-1], destination)

        self.assertEqual(
            train.UNDERGROUND_WESTBOUND_ROUTE_STEPS[0][3], 1
        )
        self.assertEqual(
            train.UNDERGROUND_WESTBOUND_ROUTE_STEPS[3][3], 2
        )
        self.assertEqual(
            train.CELADON_CENTER_ENTRY_ROUTE_STEPS[-1][:4],
            (6, 10, 41, 0),
        )

    def test_rock_tunnel_stationary_callouts_receive_sparse_a_pulses(self):
        action = train.rock_tunnel_dialogue_recovery_action
        phase = train.FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE
        self.assertIsNone(action(232, phase, 0, 7))
        self.assertEqual(action(232, phase, 0, 8), 'a')
        self.assertIsNone(action(232, phase, 0, 9))
        self.assertEqual(action(82, phase, 0, 16), 'a')
        self.assertIsNone(action(232, phase, 2, 16))
        self.assertIsNone(action(21, phase, 0, 16))
        self.assertIsNone(action(
            232, train.FULLGAME_LAVENDER_QUEST_PHASE, 0, 16
        ))

    def test_force_repel_keeps_rock_tunnel_trainer_flag_authoritative(self):
        preserve = train.force_repel_preserves_trainer_battle
        phase = train.FULLGAME_ROCK_TUNNEL_EXIT_QUEST_PHASE
        self.assertTrue(preserve(232, phase, 2))
        self.assertTrue(preserve(82, phase, 2))
        self.assertFalse(preserve(232, phase, 1))
        self.assertTrue(preserve(
            21, train.FULLGAME_LAVENDER_QUEST_PHASE, 2
        ))
        self.assertFalse(preserve(
            21, train.FULLGAME_LAVENDER_QUEST_PHASE, 1
        ))
        self.assertTrue(preserve(
            19, train.FULLGAME_ROUTE8_UNDERGROUND_ENTRY_QUEST_PHASE, 2
        ))
        self.assertFalse(preserve(36, phase, 2))
        self.assertFalse(preserve(
            232, train.FULLGAME_LAVENDER_QUEST_PHASE, 2
        ))

    def test_post_surge_pikachu_retains_swift_and_receives_thunderbolt(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_SQUIRTLE_QUEST_PHASE + 1
        env.pyboy = SimpleNamespace(memory=bytearray(0x10000))
        env._event_flag_is_set = lambda bit: bit == 327
        memory = env.pyboy.memory
        memory[train.ADDR_PARTY_SIZE] = 1
        memory[train.PARTY_SPECIES_ADDRS[0]] = train.GEN1_PIKACHU_SPECIES_ID
        moves = (129, 45, 39, 86)  # Swift plus three status moves.
        for address, move_id in zip(train.PARTY_MOVE_ID_ADDRS[0], moves):
            memory[address] = move_id
        self.assertTrue(env._ensure_pikachu_knows_thunderbolt())
        updated = [memory[address] for address in train.PARTY_MOVE_ID_ADDRS[0]]
        self.assertEqual(updated[0], 129)
        self.assertIn(train.GEN1_THUNDERBOLT_MOVE_ID, updated)
        thunderbolt_slot = updated.index(train.GEN1_THUNDERBOLT_MOVE_ID)
        self.assertEqual(
            memory[train.PARTY_MOVE_PP_ADDRS[0][thunderbolt_slot]],
            train.GEN1_THUNDERBOLT_MAX_PP,
        )

    def test_phase88_route_guidance_matches_emulator_replay(self):
        rules = train.ROCK_TUNNEL_EXIT_ACTION_GUIDANCE
        self.assertEqual(len(rules), 186)
        self.assertEqual(rules[0]['map'], 232)
        self.assertEqual(rules[0]['target'], (29, 37))
        self.assertEqual(rules[0]['action'], 2)
        transitions = [rule for rule in rules if 'destination_map' in rule]
        self.assertEqual(
            [(rule['map'], rule['target'], rule['destination_map'])
             for rule in transitions],
            [
                (232, (3, 28), 82),
                (82, (11, 16), 232),
                (232, (3, 4), 82),
                (82, (33, 16), 21),
            ],
        )
        self.assertTrue(all(rule.get('force_action') for rule in rules))

    def test_trainer_switch_handoff_declines_each_post_faint_prompt(self):
        action = train.trainer_switch_handoff_action
        phase = train.FULLGAME_MISTY_QUEST_PHASE
        self.assertEqual(action(65, phase, True, 0, 2, 1, 0x01), 'b')
        self.assertIsNone(action(65, phase, True, 59, 2, 1, 0x01))
        self.assertEqual(action(65, phase, True, 0, 3, 1, 0x01), 'b')
        self.assertEqual(action(65, phase, True, 0, 2, 1, 0x03), 'b')
        self.assertIsNone(action(65, phase, False, 0, 2, 1, 0x01))
        self.assertIsNone(action(14, phase, True, 0, 2, 1, 0x01))
        post_misty = train.FULLGAME_CERULEAN_RIVAL_QUEST_PHASE
        self.assertEqual(action(35, post_misty, True, 0, 1, 0, 0x03), 'b')
        self.assertEqual(action(
            166, train.FULLGAME_BLAINE_QUEST_PHASE,
            True, 0, 2, 1, 0x3F,
        ), 'b')
        self.assertEqual(action(
            45, train.FULLGAME_VIRIDIAN_GIOVANNI_QUEST_PHASE,
            True, 0, 2, 1, 0x7F,
        ), 'b')

    def test_trainer_battle_recovery_switches_reserves_and_breaks_menu_loops(self):
        action = train.trainer_battle_recovery_action
        self.assertIsNone(action(False, 0, 128))
        self.assertEqual(action(True, 0, 20), 'down')
        self.assertEqual(action(True, 0, 21), 'a')
        self.assertIsNone(action(True, 25, 127))
        self.assertEqual(action(True, 25, 128), 'b')
        self.assertEqual(action(True, 25, 129), 'b')
        self.assertIsNone(action(True, 25, 136))
        self.assertIsNone(action(True, 25, 144))
        self.assertEqual(action(True, 25, 160), 'b')

    def test_post_misty_exit_closes_dialogue_then_walks_down(self):
        phase = train.FULLGAME_MISTY_QUEST_PHASE + 1
        action = train.cerulean_post_misty_exit_action
        self.assertEqual(action(65, 2, 5, phase, 0x03, 2), 'b')
        self.assertEqual(action(65, 2, 5, phase, 0x03, 3), 'down')
        self.assertIsNone(action(65, 2, 5, phase, 0x01, 2))
        self.assertIsNone(action(65, 3, 5, phase, 0x03, 2))
        self.assertIsNone(action(65, 2, 5, phase - 1, 0x03, 2))

    def test_cerulean_post_heal_exit_closes_text_then_walks_down(self):
        phase = next(iter(train.FULLGAME_CERULEAN_GRIND_PHASES))
        self.assertEqual(
            train.cerulean_post_heal_exit_action(64, 3, 3, phase, 2),
            'b',
        )
        self.assertEqual(
            train.cerulean_post_heal_exit_action(64, 3, 3, phase, 3),
            'down',
        )
        self.assertIsNone(
            train.cerulean_post_heal_exit_action(15, 13, 69, phase, 1)
        )

    def test_celadon_post_rocket_heal_exit_closes_text_then_walks_down(self):
        phase = train.FULLGAME_MART_ROOF_QUEST_PHASE
        self.assertEqual(
            train.celadon_post_rocket_heal_exit_action(
                133, 3, 3, phase, 42
            ),
            'b',
        )
        self.assertEqual(
            train.celadon_post_rocket_heal_exit_action(
                133, 3, 3, phase, 43
            ),
            'down',
        )
        self.assertIsNone(
            train.celadon_post_rocket_heal_exit_action(
                133, 4, 3, phase, 44
            )
        )

    def test_celadon_center_clears_impossible_stale_wild_battle_flag(self):
        env = self.make_env()
        env.quest_phase = train.FULLGAME_MART_ROOF_QUEST_PHASE
        env.pyboy.memory[train.ADDR_MAP_ID] = 133
        env.pyboy.memory[train.ADDR_POS_A] = 3
        env.pyboy.memory[train.ADDR_POS_B] = 3
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 1
        env.battle_flag_movement_steps = 4
        env.stale_battle_stationary_evidence_steps = 2

        self.assertTrue(env._clear_impossible_celadon_center_battle_flag())
        self.assertEqual(env.pyboy.memory[train.ADDR_BATTLE_FLAG], 0)

        env.pyboy.memory[train.ADDR_MAP_ID] = 6
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 1
        self.assertTrue(env._clear_impossible_celadon_center_battle_flag())
        self.assertEqual(env.pyboy.memory[train.ADDR_BATTLE_FLAG], 0)

        env.pyboy.memory[train.ADDR_MAP_ID] = 7
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 1
        self.assertFalse(env._clear_impossible_celadon_center_battle_flag())
        self.assertEqual(env.pyboy.memory[train.ADDR_BATTLE_FLAG], 1)

        env.quest_phase = train.FULLGAME_MART_ROOF_QUEST_PHASE + 1
        env.pyboy.memory[train.ADDR_MAP_ID] = 6
        self.assertFalse(env._clear_impossible_celadon_center_battle_flag())
        self.assertEqual(env.pyboy.memory[train.ADDR_BATTLE_FLAG], 1)

    def test_celadon_mart_route_detours_at_live_collision_tile(self):
        actions = {
            (map_id, y, x): action
            for map_id, y, x, action, _ in train.CELADON_MART_ROOF_STEPS
        }
        self.assertEqual(actions[(6, 14, 26)], 0)
        self.assertEqual(actions[(6, 13, 26)], 2)
        self.assertEqual(actions[(6, 13, 24)], 2)
        self.assertEqual(actions[(6, 13, 14)], 1)
        self.assertEqual(actions[(6, 14, 14)], 2)
        self.assertEqual(actions[(6, 14, 10)], 1)
        self.assertEqual(actions[(122, 7, 2)], 0)
        self.assertEqual(actions[(122, 5, 2)], 3)
        self.assertEqual(actions[(122, 5, 12)], 0)
        self.assertEqual(actions[(122, 2, 12)], 0)
        self.assertEqual(actions[(123, 2, 12)], 3)
        self.assertEqual(actions[(123, 2, 16)], 0)
        self.assertEqual(actions[(124, 2, 16)], 2)
        self.assertEqual(actions[(124, 2, 12)], 0)
        self.assertEqual(actions[(125, 2, 12)], 3)
        self.assertEqual(actions[(136, 2, 16)], 2)

    def test_celadon_mart_route_maps_are_valid_frontier_locations(self):
        phase = train.FULLGAME_MART_ROOF_QUEST_PHASE
        for map_id in (6, 122, 123, 124, 125, 126, 133, 136):
            self.assertTrue(
                train.fullgame_frontier_position_allowed(phase, map_id, 2, 2),
                map_id,
            )

        drink_phase = train.FULLGAME_GIVE_GUARD_DRINK_QUEST_PHASE
        for map_id in (6, 18, 76, 122, 123, 124, 125, 126, 136):
            self.assertTrue(
                train.fullgame_frontier_position_allowed(
                    drink_phase, map_id, 2, 2
                ),
                map_id,
            )

    def test_celadon_mart_descent_uses_open_top_corridors(self):
        actions = {
            (map_id, y, x): action
            for map_id, y, x, action, _ in train.CELADON_ROOF_TO_SAFFRON_STEPS
        }
        self.assertEqual(actions[(136, 2, 12)], 3)
        self.assertEqual(actions[(136, 2, 16)], 0)
        self.assertEqual(actions[(125, 2, 16)], 2)
        self.assertEqual(actions[(125, 2, 12)], 0)
        self.assertEqual(actions[(124, 2, 12)], 3)
        self.assertEqual(actions[(123, 2, 16)], 2)

    def test_guard_route_targets_requested_route7_viewer_tile(self):
        actions = {
            (map_id, y, x): (action, destination)
            for map_id, y, x, action, destination
            in train.CELADON_ROOF_TO_SAFFRON_STEPS
        }
        self.assertEqual(actions[(6, 15, 10)][0], 3)
        self.assertEqual(actions[(6, 15, 18)][0], 0)
        self.assertEqual(actions[(6, 10, 18)][0], 3)
        self.assertEqual(actions[(18, 3, 0)][0], 3)
        self.assertEqual(actions[(18, 6, 2)][0], 1)
        self.assertEqual(actions[(18, 9, 2)][0], 3)
        # Raw (x11,y10) is viewer (226,211); Right crosses into map 76.
        self.assertEqual(actions[(18, 10, 11)], (3, 76))

    def test_phase126_frontier_recovery_matches_emulator_replay(self):
        actions = {
            (map_id, y, x): (action, destination)
            for map_id, y, x, action, destination
            in train.CELADON_PHASE126_FRONTIER_RECOVERY_STEPS
        }
        self.assertEqual(actions[(6, 7, 2)][0], 0)
        self.assertEqual(actions[(6, 13, 8)][0], 0)
        self.assertEqual(actions[(6, 14, 8)][0], 3)
        self.assertEqual(actions[(6, 11, 49)], (3, 18))

    def test_ordered_event_claim_advances_either_fossil_choice(self):
        env = self.make_env()
        waypoint = train.FULLGAME_QUEST_WAYPOINTS[train.FULLGAME_FOSSIL_QUEST_PHASE]
        env.quest_waypoints = (waypoint,)
        env.quest_phase_hits = []
        env.quest_phase_step_count = 0
        env.reward_ordered_quest_event_flags_only = True
        bit = train.MT_MOON_DOME_FOSSIL_EVENT_BIT
        env.pyboy.memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= 1 << (bit % 8)
        self.assertEqual(env._claim_new_named_event_flags(61, 7, 12), {bit})
        name, reward = env._advance_quest_phase(61, 7, 12)
        self.assertEqual(name, 'choose_mt_moon_fossil')
        self.assertEqual(reward, 7000.0)
        self.assertEqual(env.quest_phase, 1)

    def test_fossil_waypoint_reconciles_a_flag_already_in_reset_baseline(self):
        env = self.make_env()
        waypoint = train.FULLGAME_QUEST_WAYPOINTS[train.FULLGAME_FOSSIL_QUEST_PHASE]
        env.quest_waypoints = (waypoint,)
        env.quest_phase_hits = []
        env.quest_phase_step_count = 0
        bit = train.MT_MOON_DOME_FOSSIL_EVENT_BIT
        env.pyboy.memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= 1 << (bit % 8)
        env.base_named_event_flags = {bit}
        env.rewarded_named_event_flags = set()
        name, _ = env._advance_quest_phase(61, 4, 3)
        self.assertEqual(name, 'choose_mt_moon_fossil')
        self.assertEqual(env.quest_phase, 1)

    def test_single_event_waypoint_reconciles_a_durable_baseline_flag(self):
        env = self.make_env()
        bit = 239  # EVENT_BEAT_POKEMON_TOWER_RIVAL
        env.quest_waypoints = ({
            'name': 'beat_pokemon_tower_rival',
            'event_bit': bit,
            'reward': 9000.0,
        },)
        env.quest_phase_hits = []
        env.quest_phase_step_count = 0
        env.pyboy.memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= (
            1 << (bit % 8)
        )
        env.base_named_event_flags = {bit}
        env.rewarded_named_event_flags = set()
        name, reward = env._advance_quest_phase(143, 9, 16)
        self.assertEqual(name, 'beat_pokemon_tower_rival')
        self.assertEqual(reward, 9000.0)
        self.assertEqual(env.quest_phase, 1)

    def test_fossil_interaction_confirms_until_event_sets(self):
        # Fossil selection lives in its own guidance list, not in the cave
        # traversal rules, and deliberately carries no `face_action`: the
        # proven route arrives on the tile already facing the fossil, and a
        # re-fired turn resets the interaction across the yes/no prompt.
        env = self.make_env()
        rules = tuple(train.MT_MOON_FOSSIL_SELECTION_ACTION_GUIDANCE)
        self.assertTrue(rules)
        for rule in rules:
            self.assertNotIn('face_action', rule)
            self.assertEqual(rule['map'], 61)
            self.assertEqual(rule['action'], train.FULLGAME_ACTIONS.index('a'))
            self.assertEqual(
                rule['unless_event_bits'], train.MT_MOON_FOSSIL_EVENT_BITS
            )
            self.assertEqual(
                tuple(rule['require_event_bits']),
                (train.MT_MOON_SUPER_NERD_EVENT_BIT,),
            )
        # Both fossils are legal choices; cover the tile below each.
        self.assertEqual(
            sorted(rule['target'] for rule in rules), [(7, 12), (7, 13)]
        )
        env.action_guidance = rules
        env.action_guidance_requires_overworld = True
        env.quest_phase = train.FULLGAME_FOSSIL_QUEST_PHASE
        # Guidance carries an implicit min_lead_level of 1, so a zeroed
        # fixture would veto every rule before the event gates are reached.
        env.pyboy.memory[train.ADDR_LEVEL] = 21

        # Super Nerd still blocks the fossils until he is beaten.
        self.assertIsNone(env._matching_forced_action_guidance(61, 7, 12, 0))
        nerd = train.MT_MOON_SUPER_NERD_EVENT_BIT
        env.pyboy.memory[train.ADDR_EVENT_FLAGS_START + nerd // 8] |= (
            1 << (nerd % 8)
        )
        # A is forced, and stays forced across the prompt (no facing step).
        self.assertEqual(env._matching_forced_action_guidance(61, 7, 12, 0), 4)
        self.assertEqual(env._matching_forced_action_guidance(61, 7, 12, 0), 4)
        self.assertEqual(env._matching_forced_action_guidance(61, 7, 13, 0), 4)
        # Once either mutually exclusive fossil is taken the rule disengages.
        bit = train.MT_MOON_DOME_FOSSIL_EVENT_BIT
        env.pyboy.memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= 1 << (bit % 8)
        self.assertIsNone(env._matching_forced_action_guidance(61, 7, 12, 0))
        self.assertIsNone(env._matching_forced_action_guidance(61, 7, 13, 0))

    def test_create_pyboy_uses_private_in_memory_battery_ram(self):
        captured = {}

        class CapturePyBoy:
            def __init__(self, rom_path, **kwargs):
                captured.update(kwargs)

            def stop(self, **kwargs):
                captured['stop_kwargs'] = kwargs

        original_pyboy = train.PyBoy
        train.PyBoy = CapturePyBoy
        try:
            session = train.create_pyboy()
            session.stop()
        finally:
            train.PyBoy = original_pyboy

        ram_file = captured['ram_file']
        self.assertEqual(len(ram_file.getvalue()), train.PYBOY_RAM_SIZE)
        self.assertEqual(ram_file.getvalue(), bytes(train.PYBOY_RAM_SIZE))
        self.assertIs(captured['stop_kwargs']['ram_file'], ram_file)

    def test_level_and_badge_waypoints_require_real_memory_progress(self):
        level_waypoint = {'min_level': 7}
        badge_waypoint = {'min_badges': 1}
        self.assertFalse(
            train.quest_waypoint_matches(level_waypoint, 2, 22, 7, set(), level=6)
        )
        self.assertTrue(
            train.quest_waypoint_matches(level_waypoint, 2, 22, 7, set(), level=7)
        )
        self.assertFalse(
            train.quest_waypoint_matches(badge_waypoint, 54, 2, 4, set(), badge_count=0)
        )
        self.assertTrue(
            train.quest_waypoint_matches(badge_waypoint, 54, 2, 4, set(), badge_count=1)
        )

    def test_phase_gated_brock_controls_turn_off_after_badge_phase(self):
        env = self.make_env()
        env.quest_phase = train.FULLGAME_BROCK_QUEST_PHASE
        phases = frozenset({train.FULLGAME_BROCK_QUEST_PHASE})
        self.assertTrue(env._quest_phase_enabled(phases))
        env.quest_phase += 1
        self.assertFalse(env._quest_phase_enabled(phases))
        self.assertTrue(env._quest_phase_enabled(None))

    def test_action_guidance_rules_can_have_distinct_phase_scopes(self):
        env = self.make_env()
        env.action_guidance = [{
            'map': 2,
            'target': 19,
            'radius': 0,
            'action': 1,
            'bonus': 125.0,
            'penalty': 20.0,
            'quest_phases': train.FULLGAME_PEWTER_GRIND_PHASES,
        }]
        env.action_guidance_quest_phases = frozenset(
            train.FULLGAME_PEWTER_GRIND_PHASES
            | {train.FULLGAME_BROCK_QUEST_PHASE}
        )
        env.action_guidance_requires_overworld = True
        env.frontier_action_guidance_bonus = 0.0
        env.frontier_action_guidance_penalty = 0.0

        env.quest_phase = 21
        self.assertEqual(
            env._action_guidance_reward(2, 19, 5, 0, 1),
            125.0,
        )
        env.quest_phase = train.FULLGAME_BROCK_QUEST_PHASE
        self.assertEqual(
            env._action_guidance_reward(2, 19, 5, 0, 1),
            0.0,
        )

    def test_forced_route_bonus_stops_paying_on_a_blocked_tile(self):
        env = self.make_env()
        env.action_guidance = [{
            'map': 23, 'target': (63, 10), 'radius': 0,
            'action': 1, 'bonus': 150.0, 'penalty': 20.0,
        }]
        env.action_guidance_quest_phases = None
        env.action_guidance_requires_overworld = True
        env.frontier_action_guidance_bonus = 0.0
        env.frontier_action_guidance_penalty = 0.0
        env.quest_phase = train.FULLGAME_FUCHSIA_QUEST_PHASE

        env.same_position_step_count = 0
        self.assertEqual(env._action_guidance_reward(23, 63, 10, 0, 1), 150.0)
        env.same_position_step_count = (
            train.ACTION_GUIDANCE_STATIONARY_BONUS_STEPS
        )
        self.assertEqual(env._action_guidance_reward(23, 63, 10, 0, 1), 150.0)

        # Past the grace window the worker is demonstrably stuck. This is the
        # farm that paid 150 x 4096 = 614k per episode on the Route 12 wall.
        env.same_position_step_count = (
            train.ACTION_GUIDANCE_STATIONARY_BONUS_STEPS + 1
        )
        self.assertEqual(env._action_guidance_reward(23, 63, 10, 0, 1), 0.0)
        # The wrong-action penalty must still apply, so the shaping gradient
        # toward the guided direction is unchanged.
        self.assertEqual(env._action_guidance_reward(23, 63, 10, 0, 0), -20.0)

    def _forced_moves(self, maps, cleared_bits=()):
        """First-match-wins forced action per tile, as the env resolves it.

        `cleared_bits` models event flags that are already set live, which
        switches off any rule gated behind `unless_event_bits`.
        """
        moves = {}
        for rule in train.PART11_ACTION_GUIDANCE:
            if rule.get('map') not in maps or not rule.get('force_action'):
                continue
            if any(bit in cleared_bits
                   for bit in (rule.get('unless_event_bits') or ())):
                continue
            key = (rule['map'],) + tuple(rule['target'])
            if key not in moves:
                moves[key] = (rule['action'], rule.get('destination_map'))
        return moves

    def _walk_forced(self, moves, start, limit=200):
        """Follow the forced walk; returns the destination_map it ends on."""
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        tile = start
        seen = set()
        for _ in range(limit):
            self.assertIn(tile, moves, 'unguided at %s from %s' % (tile, start))
            self.assertNotIn(tile, seen, 'loop at %s from %s' % (tile, start))
            seen.add(tile)
            action, destination = moves[tile]
            if destination is not None:
                return destination
            delta_y, delta_x = deltas[action]
            tile = (tile[0], tile[1] + delta_y, tile[2] + delta_x)
        self.fail('no map transition from %s' % (start,))

    # Route 13 tiles the live swarm was standing on when the leg was rebuilt,
    # plus the two piles that were stalled: 49 workers on (10,25) forced Left
    # into the wall at (10,24), and 20 unguided on (11,48).
    ROUTE13_LIVE_TILES = (
        (0, 50), (4, 50), (5, 50), (8, 50), (10, 49), (10, 50), (10, 51),
        (11, 48), (11, 51), (10, 25), (10, 27), (10, 35), (10, 47), (10, 7),
        (10, 22), (11, 23), (11, 25), (12, 17), (12, 25), (12, 35), (13, 17),
        (8, 30), (9, 32), (6, 50), (7, 51),
    )

    def test_route13_forced_walk_always_reaches_route14(self):
        """The whole swarm stalled on Route 13: the leg started at (0,30) but
        workers arrive at (0,50), so the arrival half of the map owned no rule,
        and the part that did own rules forced Left along row 10 across the
        wall at (10,24). Every tile the swarm actually occupies must now walk
        to the Route 14 hand-off without a gap or a loop."""
        moves = self._forced_moves({24})
        for tile in self.ROUTE13_LIVE_TILES:
            self.assertEqual(
                self._walk_forced(moves, (24,) + tile, limit=200), 25,
                'Route 13 %s does not reach Route 14' % (tile,))

    def test_route13_never_forces_into_the_row10_break(self):
        """Row 10 is severed at x=24 (wall) and x=23 (trainer F3). (10,25) is
        the tile 49 workers were pinned on; it can only leave to the east."""
        moves = self._forced_moves({24})
        self.assertNotIn((24, 10, 24), moves)
        self.assertNotIn((24, 10, 23), moves)
        self.assertEqual(moves[(24, 10, 25)][0], 3)
        # (5,50) is trainer F4, so the x=50 descent has to jog to x=51 at y=4.
        self.assertEqual(moves[(24, 4, 50)][0], 3)

    def test_coast_legs_have_no_off_map_or_dead_transition_tiles(self):
        """Route 13's hand-off sat at (10,-9) and 49 of Route 14's 50 tiles
        were at negative x, so neither leg could ever match."""
        sizes = {24: (18, 60), 25: (54, 20), 26: (18, 60), 184: (10, 8)}
        transitions = {}
        for map_id, y, x, _action, destination in (
                train.NO_BIKE_COAST_TO_FUCHSIA_STEPS):
            if map_id not in sizes:
                continue
            height, width = sizes[map_id]
            self.assertTrue(
                0 <= y < height and 0 <= x < width,
                'off-map guidance tile (%s,%s,%s)' % (map_id, y, x))
            if destination is not None:
                transitions.setdefault(map_id, []).append((y, x, destination))
        self.assertEqual(transitions[24], [(8, 0, 25)])
        self.assertEqual(transitions[25], [(44, 1, 26)])
        self.assertEqual(transitions[184], [(4, 1, 26)])
        # Route 15 no longer claims the gate warp from row 8: that approach
        # runs into the (8,16) trap. Only the two post-gate rows hand over.
        self.assertEqual(
            sorted(transitions[26]), [(8, 0, 7), (9, 0, 7)])

    def test_route14_hands_over_on_row_8_not_row_6(self):
        """Trainer M2 at (6,15) seals Route 14's row 6, so a Route 13 exit at
        row 6 lands the swarm in a dead pocket it can never leave."""
        r13 = self._forced_moves({24})
        self.assertEqual(self._walk_forced(r13, (24, 0, 50), limit=200), 25)
        exit_tiles = [key[1:] for key, value in r13.items()
                      if value[1] == 25]
        self.assertEqual(exit_tiles, [(8, 0)])
        r14 = self._forced_moves({25})
        self.assertIn((25, 8, 19), r14)
        self.assertEqual(self._walk_forced(r14, (25, 8, 19), limit=200), 26)

    def test_route15_walks_through_the_gate_to_fuchsia(self):
        """The old legs started at x=40 (arrival is x=59), stopped the gate
        walk two tiles short of the (4,0) exit warp, and put the Fuchsia
        hand-off on (8,7) -- the tile you arrive on leaving the gate."""
        r15 = self._forced_moves({26})
        # The row-8 approach stops short of the ambiguous tiles by design.
        self.assertNotIn((26, 8, 19), r15)
        self.assertEqual(self._walk_forced(r15, (26, 8, 7), limit=200), 7)
        gate = self._forced_moves({184})
        self.assertEqual(self._walk_forced(gate, (184, 4, 7), limit=60), 26)

    def _fuchsia_moves(self, hp_fraction, quest_phase):
        """First-match-wins forced action as the env resolves it at this HP."""
        moves = {}
        for rule in train.PART11_ACTION_GUIDANCE:
            if not rule.get('force_action'):
                continue
            if hp_fraction > float(rule.get('max_party_hp_fraction', 1.0)):
                continue
            if hp_fraction < float(rule.get('min_party_hp_fraction', 0.0)):
                continue
            phases = rule.get('quest_phases')
            if phases is not None and quest_phase not in phases:
                continue
            key = (rule['map'],) + tuple(rule['target'])
            if key not in moves:
                moves[key] = (rule['action'], rule.get('destination_map'))
        return moves

    def test_coast_rank_is_ordered_so_the_frontier_can_advance(self):
        """A FLAT coast rank ties every worker with the frontier, and because
        the fullgame config disables named-event ranking the whole key was
        equal -- promotion needs strictly `>`, so the frontier sat on Route 12
        for 12 hours while workers reached Fuchsia."""
        phase = train.FULLGAME_FUCHSIA_QUEST_PHASE
        ranks = [
            train.fullgame_swarm_route_progress(phase, map_id, -1, 0)[0]
            for map_id in (4, 23, 87, 24, 25, 26, 184, 7, 154, 157)
        ]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(len(set(ranks)), len(ranks))
        # forward progress must never rank lower than where it came from
        self.assertGreater(
            train.fullgame_swarm_route_progress(phase, 7, -1, 0),
            train.fullgame_swarm_route_progress(phase, 23, -1, 0),
        )

    def test_fullgame_scope_preserves_safari_route_exclusions(self):
        """The production wrapper must not reactivate pre-Koga routes.

        FUCHSIA_ENTRY_STEPS owns (30,8) with Up before the Safari flow owns
        it with Down. Widening every nested scope to all Part 11 phases makes
        first-match-wins bounce the swarm back to row 28 forever.
        """
        rules = train.scope_action_guidance(
            train.PART11_ACTION_GUIDANCE,
            train.FULLGAME_PART11_PHASES,
        )
        phase = train.FULLGAME_SAFARI_QUEST_PHASE
        matches = [
            rule for rule in rules
            if rule.get('force_action')
            and rule.get('map') == 7
            and tuple(rule.get('target', ())) == (30, 8)
            and phase in rule['quest_phases']
        ]
        self.assertTrue(matches)
        self.assertEqual(matches[0]['action'], 1)  # Down toward row 32.
        self.assertNotIn(0, [rule['action'] for rule in matches])

    def test_safari_gate_guidance_reaches_and_accepts_admission(self):
        moves = self._fuchsia_moves(1.0, train.FULLGAME_SAFARI_QUEST_PHASE)
        self.assertEqual(moves[(156, 5, 2)][0], 3)
        self.assertEqual(moves[(156, 5, 3)][0], 0)
        self.assertEqual(moves[(156, 4, 3)][0], 0)
        self.assertEqual(moves[(156, 3, 3)][0], 0)
        self.assertEqual(moves[(156, 2, 3)][0], 4)
        self.assertEqual(moves[(156, 2, 4)][0], 4)

    def test_late_game_explorers_are_allowed_into_safari(self):
        phase = train.FULLGAME_SAFARI_QUEST_PHASE
        for map_id in (156, 217, 218, 219, 220, 221, 222, 223, 224, 225):
            self.assertFalse(
                train.late_game_explorer_should_reset(True, phase, map_id),
                map_id,
            )

    def test_hurt_worker_reaches_the_fuchsia_pokemon_center(self):
        moves = self._fuchsia_moves(0.5, train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE)
        self.assertEqual(self._walk_forced(moves, (7, 16, 39), limit=250), 154)
        self.assertEqual(self._walk_forced(moves, (7, 28, 8), limit=250), 154)
        # inside the Center the aisle leads to the nurse talk tile
        self.assertEqual(moves[(154, 7, 3)][0], 0)
        self.assertEqual(moves[(154, 3, 3)][0], 4)

    def test_healed_worker_goes_to_the_gym_and_never_re_enters_the_center(self):
        """Gen 1 drops you on the tile below the door, so an ungated Center
        rule on (28,19) would walk a healed worker straight back inside."""
        moves = self._fuchsia_moves(1.0, train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE)
        self.assertEqual(self._walk_forced(moves, (7, 28, 19), limit=250), 157)
        self.assertEqual(self._walk_forced(moves, (7, 16, 39), limit=250), 157)
        # 2026-08-08: the descent now starts at (4,3), not the nurse tile.
        # This used to assert a forced walk from (154,3,3), and that invariant
        # was actively harmful: the instant the party crosses 0.9 the heal rule
        # stops matching, but Nurse Joy's closing text is still up, and a
        # movement action does nothing against an open text box. All 96 workers
        # pressed Down 922 times on (3,3) and sat there 942 steps. (3,3) is now
        # deliberately unforced once healed so the policy can dismiss the text.
        self.assertEqual(self._walk_forced(moves, (154, 4, 3), limit=60), 7)

    def test_healed_worker_is_not_forced_on_the_fuchsia_nurse_tile(self):
        """The nurse tile must have no forced rule once the party is healed.

        Regression guard for the 942-step pin above: whatever owns (154,3,3)
        at full HP must not be a forced action, or the worker is trapped
        against the heal dialogue it has not been given a chance to close.
        """
        healed = self._fuchsia_moves(1.0, train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE)
        self.assertNotIn((154, 3, 3), healed)
        # ...while a hurt worker must still be forced to talk to her (A).
        hurt = self._fuchsia_moves(0.5, train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE)
        self.assertEqual(hurt[(154, 3, 3)][0], 4)

    def test_fuchsia_gym_door_is_row_28_not_the_walled_row_11(self):
        """The old leg walked row 11 west from x=19 and claimed the Gym at
        (11,5); row 11 is solid from x=6 to x=17 and the door is (27,5)."""
        moves = self._fuchsia_moves(1.0, train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE)
        gym_doors = [k[1:] for k, v in moves.items()
                     if k[0] == 7 and v[1] == 157]
        self.assertEqual(gym_doors, [(28, 5)])
        for x in range(6, 18):
            self.assertNotIn((7, 11, x), moves)

    def test_safari_legs_are_scoped_off_outside_the_safari_phases(self):
        """Unscoped, they marched Gym-phase workers to the Warden; picking up
        HM04 out of order permanently disqualifies a worker from promotion."""
        gym = self._fuchsia_moves(1.0, train.FULLGAME_FUCHSIA_GYM_QUEST_PHASE)
        self.assertEqual(
            [k for k in gym if k[0] in (156, 219, 220, 222)], [])
        safari = self._fuchsia_moves(1.0, train.FULLGAME_SAFARI_QUEST_PHASE)
        self.assertTrue([k for k in safari if k[0] in (156, 219, 220, 222)])

    def test_route15_does_not_force_into_the_816_trap(self):
        """(8,16) refuses movement in all four directions -- including back
        east -- with a clean game state, and the forced Left across it pinned
        78 of 96 workers for 3,488 steps an episode. The leg must hand over
        before it, and must not claim the gate warp from row 8."""
        moves = self._forced_moves({26})
        # (8,16) and (9,16) are both one-way from the west: nothing may be
        # forced onto them, or onto a tile whose forced action steps into one.
        for tile in ((8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19),
                     (9, 16), (9, 17)):
            self.assertNotIn((26,) + tile, moves,
                             'nothing may be forced at %s' % (tile,))
        self.assertEqual(moves[(26, 8, 20)][0], 2)
        self.assertIsNone(moves[(26, 8, 20)][1])
        # the post-gate legs still hand over to Fuchsia from both door rows
        self.assertEqual(self._walk_forced(moves, (26, 8, 7), limit=60), 7)
        self.assertEqual(self._walk_forced(moves, (26, 9, 7), limit=60), 7)

    def test_frontier_movement_check_uses_the_routes_own_direction(self):
        """'Any direction moves' is too weak on a forced-route tile. A live
        candidate at Route 12 (63,11) accepted Down/Up/Left but not the Right
        the route forces there, passed the old check, and froze all 96
        workers for a whole episode each."""
        env = self.make_env()
        env.actions = ['down', 'left', 'right', 'up', 'a', 'b', 'start']
        env.action_guidance = [{
            'map': 23, 'target': (63, 11), 'radius': 0,
            'action': 2, 'force_action': True, 'bonus': 150.0,
        }]
        env.action_guidance_quest_phases = None
        env.action_guidance_requires_overworld = True
        env.quest_phase = train.FULLGAME_FUCHSIA_QUEST_PHASE
        env._forced_interaction_face_key = None

        self.assertEqual(env._forced_direction_name(23, 63, 11), 'right')
        # An unguided tile has no required direction, so the candidate falls
        # back to the original any-direction check.
        self.assertIsNone(env._forced_direction_name(23, 63, 12))
        # A forced A-press tile is an interaction, not a movement direction.
        env.action_guidance = [{
            'map': 23, 'target': (61, 10), 'radius': 0,
            'action': 4, 'force_action': True, 'bonus': 150.0,
        }]
        self.assertIsNone(env._forced_direction_name(23, 61, 10))

    def test_coast_progress_does_not_rank_below_lavender(self):
        """Post-Brock routing predates every map past Route 8, so it scored
        Lavender at phase 8 and the whole coast at the unknown-map phase -1.
        A worker that had actually advanced therefore ranked BELOW the
        Lavender frontier and was truncated as swarm_catchup on the next
        128-step check -- 1,500+ consecutive episodes, avg_ep_length pinned at
        128 against a leg needing ~200 steps.

        The requirement is "never below", not "equal". Asserting equality here
        is what pinned the coast to a flat rank, and with named-event ranking
        disabled that made every worker's key identical to the frontier's, so
        the frontier could not advance at all -- see
        test_coast_rank_is_ordered_so_the_frontier_can_advance."""
        progress = train.fullgame_swarm_route_progress
        phase = train.FULLGAME_FUCHSIA_QUEST_PHASE
        lavender = progress(phase, 4, 8, 0)
        for map_id in (23, 87, 24, 25, 26, 184, 7):
            self.assertGreaterEqual(
                progress(phase, map_id, -1, 0), lavender,
                'map %d ranks below Lavender' % map_id,
            )
        # Maps outside the coast keep their ordinary rank, and earlier
        # curriculum phases are untouched.
        self.assertEqual(progress(phase, 19, 8, 0), (8, 0))
        self.assertEqual(progress(10, 14, 0, 5), (0, 5))

    def test_route12_entry_walks_the_x9_corridor_not_into_a_wall(self):
        """The Lavender->Route 12 entry rules forced Right at (0,9), which is
        a wall, and being listed before ROUTE12_TO_SNORLAX_TILE_STEPS they beat
        the correct Down. Live: all 96 workers pinned on (0,9), avg_ep_length
        collapsed 4096 -> 128 with max_consecutive_same_tile_steps=112."""
        moves = self._forced_moves(
            {23}, cleared_bits={train.EVENT_BEAT_ROUTE12_SNORLAX},
        )
        self.assertEqual(moves[(23, 0, 9)][0], 1)
        self.assertEqual(self._walk_forced(moves, (23, 0, 9)), 87)
        self.assertEqual(self._walk_forced(moves, (23, 0, 8)), 87)
        # and the post-flute coast leg still reaches Route 13
        self.assertEqual(self._walk_forced(moves, (23, 63, 10)), 24)

    def test_flute_helper_recovers_post_snorlax_workers_at_the_gate_exit(self):
        """With the old y>=55 window a worker that had already beaten Snorlax
        left the Route 12 gate at (22,10), walked to (27,10) and was stranded:
        the ledges at y=31 and y=52 make the coast start unwalkable."""
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 23
        memory[train.ADDR_BATTLE_FLAG] = 0
        bit = train.EVENT_BEAT_ROUTE12_SNORLAX
        memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= 1 << (bit % 8)

        memory[train.ADDR_POS_A] = 27
        memory[train.ADDR_POS_B] = 10
        self.assertTrue(env._force_route12_snorlax_flute_tile())
        self.assertEqual(
            (memory[train.ADDR_POS_A], memory[train.ADDR_POS_B]), (63, 10),
        )
        # Already on the coast start: must not keep re-poking every step.
        self.assertFalse(env._force_route12_snorlax_flute_tile())
        # North of the gate is still out of scope.
        memory[train.ADDR_POS_A] = 15
        self.assertFalse(env._force_route12_snorlax_flute_tile())

    def _lavender_forced_moves(self):
        """First-match-wins map-4 forced action per tile, as the env sees it."""
        return {
            key[1:]: value
            for key, value in self._forced_moves({4}).items()
        }

    def test_lavender_forced_walk_always_reaches_route12(self):
        """90 of 96 workers sat unguided on (11,3) for 4.5h: the x=0 column
        leg starts at x=0 and the next leg started at x=7, so the whole
        west-centre band owned no rule and the policy collapsed there."""
        moves = self._lavender_forced_moves()
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        starts = [
            (11, 1), (11, 3), (11, 9), (11, 0), (8, 0),
            (12, 13), (15, 13), (13, 10), (16, 9), (15, 0),
        ]
        for start in starts:
            tile = start
            seen = set()
            for _ in range(80):
                self.assertIn(tile, moves, 'unguided at %s from %s' % (tile, start))
                self.assertNotIn(tile, seen, 'loop at %s from %s' % (tile, start))
                seen.add(tile)
                action, destination = moves[tile]
                if destination is not None:
                    self.assertEqual(destination, 23)
                    break
                delta_y, delta_x = deltas[action]
                tile = (tile[0] + delta_y, tile[1] + delta_x)
            else:
                self.fail('no Route 12 transition from %s' % (start,))

    def test_lavender_dead_x8_exit_column_is_not_forced(self):
        """(16,8) is walkable but (17,8) is NOT, so the old x=8 tail forced
        Down into a wall one tile short of Route 12. The live exit is x=9."""
        moves = self._lavender_forced_moves()
        self.assertNotIn((17, 8), moves)
        self.assertNotIn((16, 8), moves)
        # (12,7) is a wall and (9,7) is unreachable, so the whole x=7 column
        # leg was bogus -- and it owned (11,7), which the row-11 recovery
        # must cross going east.
        self.assertNotIn((12, 7), moves)
        self.assertEqual(moves[(11, 7)][0], 3)

    def test_route12_coast_leg_matches_emulator_verified_path(self):
        steps = [
            step for step in train.NO_BIKE_COAST_TO_FUCHSIA_STEPS
            if step[0] == 23 and step[1] >= 61
        ]
        self.assertEqual(steps[0][:3], (23, 61, 10))
        self.assertEqual(steps[-1][:3], (23, 107, 10))
        self.assertEqual(steps[-1][4], 24)

        # Every tile is forced exactly once. A repeated tile is the
        # self-intersection class that loops a swarm under first-match-wins.
        tiles = [step[:3] for step in steps]
        self.assertEqual(len(tiles), len(set(tiles)))

        # The path is contiguous: each forced action lands on the next tile.
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for current, following in zip(steps, steps[1:]):
            delta_y, delta_x = deltas[current[3]]
            self.assertEqual(
                (current[1] + delta_y, current[2] + delta_x),
                (following[1], following[2]),
            )

        # The straight x=10 column the original route walked is walled at
        # (64,10); that wall pinned all 96 workers on (63,10) for 8 hours.
        self.assertNotIn((23, 64, 10), tiles)

    def test_route12_coast_leg_avoids_flute_helper_teleport_window(self):
        # _force_route12_snorlax_flute_tile re-parks any post-Snorlax worker
        # with 55 <= y < 63 onto (63,10). Only the two-tile approach may sit
        # in that window, or the helper teleports workers off their own route.
        inside = [
            step[:3] for step in train.NO_BIKE_COAST_TO_FUCHSIA_STEPS
            if step[0] == 23 and 55 <= step[1] < 63
        ]
        self.assertEqual(inside, [(23, 61, 10), (23, 62, 10)])

    def test_brock_attack_readiness_promotes_non_electric_move(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_PARTY_SIZE] = 1
        for addresses in (train.PARTY_MOVE_ID_ADDRS[0], train.ADDR_BATTLE_MON_MOVE_IDS):
            memory[addresses[0]] = 84  # ThunderShock: ineffective against Ground
            memory[addresses[1]] = 98  # Quick Attack
            memory[addresses[2]] = 45  # Growl
            memory[addresses[3]] = 0
        for addresses in (train.PARTY_MOVE_PP_ADDRS[0], train.ADDR_BATTLE_MON_PP):
            memory[addresses[0]] = 25
            memory[addresses[1]] = 30
            memory[addresses[2]] = 40
            memory[addresses[3]] = 0

        env._ensure_brock_attack_ready()

        self.assertEqual(memory[train.PARTY_MOVE_ID_ADDRS[0][0]], 98)
        self.assertEqual(memory[train.ADDR_BATTLE_MON_MOVE_IDS[0]], 98)

    def test_only_named_non_quarantined_events_can_be_claimed(self):
        env = self.make_env()
        env.quarantined_event_flag_bits = train.QUARANTINED_EVENT_FLAG_BITS
        env.base_named_event_flags = set()
        env.rewarded_named_event_flags = set()
        catalogued = set(event_flags_by_bit())
        named_bit = next(bit for bit in catalogued if bit not in env.quarantined_event_flag_bits)
        unnamed_bit = next(
            bit
            for bit in range((train.ADDR_EVENT_FLAGS_END - train.ADDR_EVENT_FLAGS_START) * 8)
            if bit not in catalogued and bit not in env.quarantined_event_flag_bits
        )
        for bit in (named_bit, unnamed_bit):
            addr = train.ADDR_EVENT_FLAGS_START + bit // 8
            env.pyboy.memory[addr] |= 1 << (bit % 8)

        self.assertEqual(env._claim_new_named_event_flags(), {named_bit})
        self.assertEqual(env._claim_new_named_event_flags(), set())

    def test_ordered_event_reward_rejects_impossible_pokedex_flag(self):
        env = self.make_env()
        env.quarantined_event_flag_bits = train.QUARANTINED_EVENT_FLAG_BITS
        env.base_named_event_flags = set()
        env.rewarded_named_event_flags = set()
        env.reward_ordered_quest_event_flags_only = True
        env.quest_waypoints = train.FULLGAME_QUEST_WAYPOINTS
        env.quest_phase = 2
        pokedex_bit = 37
        env.pyboy.memory[train.ADDR_EVENT_FLAGS_START + pokedex_bit // 8] |= (
            1 << (pokedex_bit % 8)
        )

        self.assertEqual(env._claim_new_named_event_flags(42, 19, 29), set())
        self.assertEqual(env.rewarded_named_event_flags, set())

    def test_ordered_parcel_reward_requires_viridian_mart(self):
        env = self.make_env()
        env.quarantined_event_flag_bits = train.QUARANTINED_EVENT_FLAG_BITS
        env.base_named_event_flags = set()
        env.rewarded_named_event_flags = set()
        env.reward_ordered_quest_event_flags_only = True
        env.quest_waypoints = train.FULLGAME_QUEST_WAYPOINTS
        env.quest_phase = 2
        parcel_bit = 57
        env.pyboy.memory[train.ADDR_EVENT_FLAGS_START + parcel_bit // 8] |= (
            1 << (parcel_bit % 8)
        )

        self.assertEqual(env._claim_new_named_event_flags(1, 19, 29), set())
        self.assertEqual(env._claim_new_named_event_flags(42, 19, 29), {57})

    def test_ordered_swarm_rank_ignores_arbitrary_named_event_count(self):
        clean = train.swarm_progress_key(False, 3, 2, -1, 0)
        poisoned = train.swarm_progress_key(False, 99, 2, -1, 0)
        advanced = train.swarm_progress_key(False, 0, 3, -1, 0)
        self.assertEqual(clean, poisoned)
        self.assertGreater(advanced, clean)

    def test_fullgame_rejects_legacy_frontier_metadata(self):
        self.assertFalse(
            train.swarm_frontier_meta_is_compatible(
                {'event_bits': 6, 'quest_phase': 2},
                train.FULLGAME_SWARM_FRONTIER_SCHEMA,
            )
        )
        self.assertTrue(
            train.swarm_frontier_meta_is_compatible(
                {'schema': train.FULLGAME_SWARM_FRONTIER_SCHEMA},
                train.FULLGAME_SWARM_FRONTIER_SCHEMA,
            )
        )

    def test_frontier_state_checksum_rejects_interleaved_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / 'frontier.state'
            state_path.write_bytes(b'worker-a')
            meta = {'state_sha256': train.file_sha256(state_path)}
            self.assertTrue(
                train.swarm_frontier_state_matches_meta(state_path, meta)
            )
            state_path.write_bytes(b'worker-b')
            self.assertFalse(
                train.swarm_frontier_state_matches_meta(state_path, meta)
            )

    def test_startup_quarantines_a_checksum_mismatched_frontier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            start_path = Path(temp_dir) / 'restart_start.state'
            state_path, meta_path = train.swarm_frontier_paths(start_path)
            state_path.write_bytes(b'new-worker-state')
            meta_path.write_text(
                json.dumps({
                    'schema': train.FULLGAME_SWARM_FRONTIER_SCHEMA,
                    'state_sha256': train.file_sha256(state_path),
                }),
                encoding='utf-8',
            )
            state_path.write_bytes(b'interleaved-other-worker-state')

            self.assertFalse(
                train.validate_or_quarantine_swarm_frontier(
                    start_path,
                    train.FULLGAME_SWARM_FRONTIER_SCHEMA,
                )
            )
            self.assertFalse(state_path.exists())
            self.assertFalse(meta_path.exists())
            self.assertEqual(
                len(list(Path(temp_dir).glob('*.invalid_startup.*'))),
                2,
            )

    def test_startup_keeps_a_checksum_matched_movable_frontier(self):
        original_validator = train.frontier_candidate_accepts_movement
        train.frontier_candidate_accepts_movement = lambda *args: True
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                start_path = Path(temp_dir) / 'restart_start.state'
                state_path, meta_path = train.swarm_frontier_paths(start_path)
                state_path.write_bytes(b'movable-state')
                meta_path.write_text(
                    json.dumps({
                        'schema': train.FULLGAME_SWARM_FRONTIER_SCHEMA,
                        'state_sha256': train.file_sha256(state_path),
                    }),
                    encoding='utf-8',
                )

                self.assertTrue(
                    train.validate_or_quarantine_swarm_frontier(
                        start_path,
                        train.FULLGAME_SWARM_FRONTIER_SCHEMA,
                    )
                )
                self.assertTrue(state_path.exists())
                self.assertTrue(meta_path.exists())
        finally:
            train.frontier_candidate_accepts_movement = original_validator

    def test_protect_lead_moveset_restores_move_deleted_by_level_up(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_PARTY_SIZE] = 1
        move_addrs = train.PARTY_MOVE_ID_ADDRS[0]
        pp_addrs = train.PARTY_MOVE_PP_ADDRS[0]
        memory[move_addrs[0]] = 84   # ThunderShock (damaging)
        memory[move_addrs[1]] = 45   # Growl (status)
        memory[move_addrs[2]] = 39   # Tail Whip (status)
        memory[move_addrs[3]] = 0
        memory[pp_addrs[0]] = 30
        memory[pp_addrs[1]] = 40
        memory[pp_addrs[2]] = 30
        memory[pp_addrs[3]] = 0

        env._protect_lead_moveset()  # establishes the baseline snapshot
        self.assertEqual(env.lead_moveset_recoveries, 0)

        # The in-game "delete a move?" menu just traded ThunderShock away
        # for another status move (Thunder Wave) -- the exact failure
        # confirmed live on the 2026-07-25 shared frontier.
        memory[move_addrs[0]] = 86   # Thunder Wave (status)
        memory[pp_addrs[0]] = 20

        env._protect_lead_moveset()

        self.assertEqual(memory[move_addrs[0]], 84)
        self.assertEqual(memory[pp_addrs[0]], 30)
        self.assertEqual(env.lead_moveset_recoveries, 1)

    def test_protect_lead_moveset_allows_a_real_attack_upgrade(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_PARTY_SIZE] = 1
        move_addrs = train.PARTY_MOVE_ID_ADDRS[0]
        pp_addrs = train.PARTY_MOVE_PP_ADDRS[0]
        memory[move_addrs[0]] = 84   # ThunderShock (damaging)
        memory[move_addrs[1]] = 45
        memory[move_addrs[2]] = 39
        memory[move_addrs[3]] = 0
        memory[pp_addrs[0]] = 30
        memory[pp_addrs[1]] = 40
        memory[pp_addrs[2]] = 30
        memory[pp_addrs[3]] = 0

        env._protect_lead_moveset()

        # Learning Quick Attack over ThunderShock keeps the damaging-move
        # count the same -- a real improvement, not a downgrade, so it must
        # be left alone.
        memory[move_addrs[0]] = 98   # Quick Attack (damaging)
        memory[pp_addrs[0]] = 30

        env._protect_lead_moveset()

        self.assertEqual(memory[move_addrs[0]], 98)
        self.assertEqual(env.lead_moveset_recoveries, 0)

    def test_protect_lead_moveset_ignores_multi_slot_changes(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_PARTY_SIZE] = 1
        move_addrs = train.PARTY_MOVE_ID_ADDRS[0]
        pp_addrs = train.PARTY_MOVE_PP_ADDRS[0]
        memory[move_addrs[0]] = 84
        memory[move_addrs[1]] = 45
        memory[move_addrs[2]] = 39
        memory[move_addrs[3]] = 0
        for addr in pp_addrs:
            memory[addr] = 20

        env._protect_lead_moveset()

        # A savestate load (not a level-up) can change several slots at
        # once -- not the menu-deletion failure mode, so it must be left
        # alone rather than "corrected" against a now-irrelevant snapshot.
        memory[move_addrs[0]] = 86
        memory[move_addrs[1]] = 33

        env._protect_lead_moveset()

        self.assertEqual(memory[move_addrs[0]], 86)
        self.assertEqual(memory[move_addrs[1]], 33)
        self.assertEqual(env.lead_moveset_recoveries, 0)

    def _swarm_env(self, temp_dir, quest_phase=1, min_hp_fraction=0.0):
        env = self.make_env()
        env.quest_phase = quest_phase
        env.swarm_frontier_schema = train.FULLGAME_SWARM_FRONTIER_SCHEMA
        env.swarm_frontier_min_hp_fraction = min_hp_fraction
        env.swarm_frontier_min_lead_hp_fraction = 0.0
        env.swarm_catchup_behind_bits = 1
        env.swarm_catchup_behind_route4_x = 20
        env.swarm_frontier_state_path = Path(temp_dir) / 'frontier.state'
        env.swarm_frontier_meta_path = Path(temp_dir) / 'frontier.json'
        env.pyboy.memory[train.ADDR_PARTY_SIZE] = 1
        return env

    def test_swarm_sync_withholds_a_critically_low_hp_frontier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._swarm_env(temp_dir, min_hp_fraction=0.4)
            memory = env.pyboy.memory
            memory[train.PARTY_CUR_HP_ADDRS[0][0]] = 0
            memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 3
            memory[train.PARTY_MAX_HP_ADDRS[0][0]] = 0
            memory[train.PARTY_MAX_HP_ADDRS[0][1]] = 38

            # 2026-07-25: this exact 3/38 shared frontier froze the whole
            # swarm on quest_phase for 2+ hours; below-threshold HP must
            # never be written out as the new frontier for other workers.
            env._swarm_sync()

            self.assertFalse(env.swarm_frontier_meta_path.exists())

    def test_swarm_sync_promotes_a_healthy_frontier(self):
        original_pyboy = train.PyBoy
        train.PyBoy = _AlwaysMovesStubPyBoy
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                env = self._swarm_env(temp_dir, min_hp_fraction=0.4)
                memory = env.pyboy.memory
                memory[train.PARTY_CUR_HP_ADDRS[0][0]] = 0
                memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 30
                memory[train.PARTY_MAX_HP_ADDRS[0][0]] = 0
                memory[train.PARTY_MAX_HP_ADDRS[0][1]] = 38

                env._swarm_sync()

                self.assertTrue(env.swarm_frontier_meta_path.exists())
                meta = json.loads(env.swarm_frontier_meta_path.read_text(encoding='utf-8'))
                self.assertEqual(meta['quest_phase'], 1)
        finally:
            train.PyBoy = original_pyboy

    def test_swarm_sync_persists_xp_and_restores_first_grind_checkpoint(self):
        original_pyboy = train.PyBoy
        train.PyBoy = _AlwaysMovesStubPyBoy
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                phase = min(train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES)
                env = self._swarm_env(
                    temp_dir, quest_phase=phase, min_hp_fraction=0.4
                )
                env.swarm_frontier_min_lead_hp_fraction = 0.4
                memory = env.pyboy.memory
                memory[train.ADDR_MAP_ID] = 232
                memory[train.ADDR_POS_A] = 13
                memory[train.ADDR_POS_B] = 31
                memory[train.PARTY_CUR_HP_ADDRS[0][0]] = 0
                memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 50
                memory[train.PARTY_MAX_HP_ADDRS[0][0]] = 0
                memory[train.PARTY_MAX_HP_ADDRS[0][1]] = 100
                memory[train.PARTY_MOVE_ID_ADDRS[0][0]] = 85
                memory[train.PARTY_MOVE_PP_ADDRS[0][0]] = 1
                xp = 46865
                memory[train.ADDR_XP] = (xp >> 16) & 0xFF
                memory[train.ADDR_XP + 1] = (xp >> 8) & 0xFF
                memory[train.ADDR_XP + 2] = xp & 0xFF
                saved = {}

                def capture_save(handle):
                    saved['hp'] = (
                        (memory[train.PARTY_CUR_HP_ADDRS[0][0]] << 8)
                        | memory[train.PARTY_CUR_HP_ADDRS[0][1]]
                    )
                    saved['pp'] = memory[train.PARTY_MOVE_PP_ADDRS[0][0]] & 0x3F
                    handle.write(b'fake-state')

                env.pyboy.save_state = capture_save
                env._swarm_sync()

                meta = json.loads(
                    env.swarm_frontier_meta_path.read_text(encoding='utf-8')
                )
                self.assertEqual(meta['route4_value'], xp)
                self.assertEqual(saved, {'hp': 100, 'pp': 15})
                self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 50)
                self.assertEqual(memory[train.PARTY_MOVE_PP_ADDRS[0][0]] & 0x3F, 1)
        finally:
            train.PyBoy = original_pyboy

    def test_swarm_sync_rejects_fainted_lead_despite_healthy_reserve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._swarm_env(temp_dir, min_hp_fraction=0.0)
            env.swarm_frontier_min_lead_hp_fraction = 0.4
            memory = env.pyboy.memory
            memory[train.ADDR_PARTY_SIZE] = 2
            for index, (current, maximum) in enumerate(((0, 54), (18, 18))):
                cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[index]
                max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[index]
                memory[cur_hi], memory[cur_lo] = divmod(current, 256)
                memory[max_hi], memory[max_lo] = divmod(maximum, 256)

            env._swarm_sync()

            self.assertFalse(env.swarm_frontier_meta_path.exists())

    def test_swarm_sync_default_threshold_preserves_alive_only_behavior(self):
        original_pyboy = train.PyBoy
        train.PyBoy = _AlwaysMovesStubPyBoy
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                env = self._swarm_env(temp_dir, min_hp_fraction=0.0)
                memory = env.pyboy.memory
                memory[train.PARTY_CUR_HP_ADDRS[0][0]] = 0
                memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 3
                memory[train.PARTY_MAX_HP_ADDRS[0][0]] = 0
                memory[train.PARTY_MAX_HP_ADDRS[0][1]] = 38

                env._swarm_sync()

                self.assertTrue(env.swarm_frontier_meta_path.exists())
        finally:
            train.PyBoy = original_pyboy

    def test_frontier_candidate_accepts_movement_detects_a_working_direction(self):
        class RespondingStubPyBoy:
            def __init__(self, rom_path, window=None, **kwargs):
                self.memory = bytearray(0x10000)
                self.memory[train.ADDR_MAP_ID] = 51
                self.memory[train.ADDR_POS_A] = 10
                self.memory[train.ADDR_POS_B] = 10
                self._pressed = None

            def load_state(self, handle):
                pass

            def button_press(self, name):
                self._pressed = name

            def button_release(self, name):
                self._pressed = None

            def tick(self, n):
                if self._pressed == 'up':
                    self.memory[train.ADDR_POS_A] = 9

            def stop(self, **kwargs):
                pass

        original_pyboy = train.PyBoy
        train.PyBoy = RespondingStubPyBoy
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                state_path = Path(temp_dir) / 'candidate.state'
                state_path.write_bytes(b'fake')
                self.assertTrue(
                    train.frontier_candidate_accepts_movement(state_path, 8, 16)
                )
        finally:
            train.PyBoy = original_pyboy

    def test_frontier_candidate_accepts_movement_rejects_a_frozen_state(self):
        class FrozenStubPyBoy:
            def __init__(self, rom_path, window=None, **kwargs):
                self.memory = bytearray(0x10000)
                self.memory[train.ADDR_MAP_ID] = 51
                self.memory[train.ADDR_POS_A] = 33
                self.memory[train.ADDR_POS_B] = 26

            def load_state(self, handle):
                pass

            def button_press(self, name):
                pass

            def button_release(self, name):
                pass

            def tick(self, n):
                pass  # nothing ever moves -- e.g. an unadvanceable dialogue

            def stop(self, **kwargs):
                pass

        original_pyboy = train.PyBoy
        train.PyBoy = FrozenStubPyBoy
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                state_path = Path(temp_dir) / 'candidate.state'
                state_path.write_bytes(b'fake')
                self.assertFalse(
                    train.frontier_candidate_accepts_movement(state_path, 8, 16)
                )
        finally:
            train.PyBoy = original_pyboy

    def test_rocket_curriculum_does_not_wait_for_unused_door_event(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertNotIn('unlock_rocket_hideout_door', names)
        self.assertEqual(
            train.FULLGAME_QUEST_WAYPOINTS[
                train.FULLGAME_ROCKET_JESSIE_JAMES_QUEST_PHASE
            ]['event_bit'],
            1698,
        )

    def test_rocket_elevator_guidance_reaches_b4f_panel(self):
        rules = {
            (rule['map'], *rule['target']): rule['action']
            for rule in train.ROCKET_ELEVATOR_PANEL_ACTION_GUIDANCE
        }
        self.assertEqual(rules[(203, 4, 2)], 2)  # left
        self.assertEqual(rules[(203, 4, 1)], 0)  # up
        self.assertEqual(rules[(203, 3, 1)], 0)  # up
        self.assertEqual(rules[(203, 2, 2)], 2)  # left to panel

    def test_rocket_b4f_story_approaches_are_contiguous(self):
        self.assertEqual(
            train.ROCKET_JESSIE_JAMES_APPROACH_ACTION_GUIDANCE[0]['target'],
            (15, 25),
        )
        giovanni_targets = [
            rule['target']
            for rule in train.ROCKET_GIOVANNI_APPROACH_ACTION_GUIDANCE
        ]
        self.assertEqual(giovanni_targets[0], (14, 25))
        self.assertEqual(giovanni_targets[-1], (3, 23))
        self.assertEqual(
            train.ROCKET_SILPH_SCOPE_APPROACH_ACTION_GUIDANCE[0]['target'],
            (3, 24),
        )

    def test_rocket_scope_escape_uses_b2f_and_reaches_celadon_center(self):
        transitions = [
            (rule['map'], rule.get('destination_map'))
            for rule in train.ROCKET_B2_ESCAPE_TO_CELADON_CENTER_ACTION_GUIDANCE
            if rule.get('destination_map') is not None
        ]
        self.assertEqual(
            transitions,
            [(200, 199), (199, 135), (135, 6), (6, 133)],
        )
        self.assertEqual(
            train.ROCKET_SCOPE_TO_ELEVATOR_ACTION_GUIDANCE[0]['target'],
            (2, 24),
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(
                train.FULLGAME_POST_ROCKET_HEAL_QUEST_PHASE,
                133,
                7,
                3,
            )
        )

    def test_swarm_sync_rejects_a_candidate_frozen_by_dialogue(self):
        class FrozenStubPyBoy:
            def __init__(self, rom_path, window=None, **kwargs):
                self.memory = bytearray(0x10000)

            def load_state(self, handle):
                pass

            def button_press(self, name):
                pass

            def button_release(self, name):
                pass

            def tick(self, n):
                pass

            def stop(self, **kwargs):
                pass

        original_pyboy = train.PyBoy
        train.PyBoy = FrozenStubPyBoy
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Confirmed live 2026-07-25: this exact scenario (healthy HP,
                # battle_flag==0, no proxy textbox) still got shared as the
                # frontier while mid a trainer's unadvanceable callout
                # screen -- every follower then froze on the same tile.
                env = self._swarm_env(temp_dir, min_hp_fraction=0.0)
                memory = env.pyboy.memory
                memory[train.PARTY_CUR_HP_ADDRS[0][0]] = 0
                memory[train.PARTY_CUR_HP_ADDRS[0][1]] = 30
                memory[train.PARTY_MAX_HP_ADDRS[0][0]] = 0
                memory[train.PARTY_MAX_HP_ADDRS[0][1]] = 38

                env._swarm_sync()

                self.assertFalse(env.swarm_frontier_meta_path.exists())
                leftover_temp_files = [
                    path for path in Path(temp_dir).iterdir()
                    if '.tmp.' in path.name
                ]
                self.assertEqual(leftover_temp_files, [])
        finally:
            train.PyBoy = original_pyboy


    def test_part11_waypoints_and_frontier_guards(self):
        """Post-Sabrina Part 11 ladder: bike, L55, R16, Fuchsia, Koga, Safari."""
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        expected = [
            "beat_sabrina",
            "reach_lavender_for_fuchsia",
            "train_pikachu_level_55",
            "beat_route12_snorlax",
            "reach_fuchsia_city",
            "enter_fuchsia_gym",
            "beat_koga",
            "enter_safari_zone",
            "collect_gold_teeth",
            "get_hm03_surf",
            "get_hm04_strength",
        ]
        self.assertEqual(
            names[train.FULLGAME_SABRINA_QUEST_PHASE:
                  train.FULLGAME_HM04_STRENGTH_QUEST_PHASE + 1],
            expected,
        )
        self.assertEqual(
            train.FULLGAME_QUEST_WAYPOINTS[
                train.FULLGAME_PIKACHU_55_GRIND_QUEST_PHASE
            ]["min_level"],
            55,
        )
        self.assertEqual(train.EVENT_GOT_BICYCLE, 192)
        self.assertEqual(train.EVENT_BEAT_ROUTE12_SNORLAX, 1167)
        self.assertEqual(
            train.FULLGAME_ROUTE12_SNORLAX_QUEST_PHASE,
            train.FULLGAME_SABRINA_QUEST_PHASE + 3,
        )
        self.assertEqual(train.EVENT_BEAT_KOGA, 601)
        self.assertEqual(train.EVENT_GOT_HM03, 2176)
        self.assertEqual(train.EVENT_GOT_HM04, 568)
        # Bike shop door viewer (247,145) -> local (x=13,y=25); approach from y=26.
        self.assertEqual(
            train.CERULEAN_TO_BIKE_SHOP_STEPS[-1][:4], (3, 26, 13, 0),
        )

    def test_pallet_route21_cinnabar_continuation_is_ordered_and_guarded(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[
            train.FULLGAME_FLY_TO_PALLET_QUEST_PHASE:
            train.FULLGAME_CINNABAR_QUEST_PHASE + 1
        ], [
            "fly_to_pallet_town",
            "teach_surf_to_squirtle",
            "enter_route_21",
            "reach_cinnabar_island",
        ])
        self.assertEqual(
            train.PALLET_TO_SURF_SHORE_STEPS[0][:4], (0, 6, 5, 1),
        )
        self.assertEqual(
            train.PALLET_TO_SURF_SHORE_STEPS[-1][:4], (0, 16, 3, 1),
        )
        self.assertEqual(
            train.PALLET_SURF_TO_ROUTE21_STEPS[-1], (0, 17, 4, 1, 32),
        )
        self.assertEqual(
            train.ROUTE21_TO_CINNABAR_STEPS[-1], (32, 89, 3, 1, 8),
        )
        self.assertTrue(train.fullgame_frontier_position_allowed(
            train.FULLGAME_TEACH_SURF_QUEST_PHASE, 0, 6, 5,
        ))
        self.assertFalse(train.fullgame_frontier_position_allowed(
            train.FULLGAME_TEACH_SURF_QUEST_PHASE, 7, 28, 29,
        ))
        self.assertEqual(
            train.fullgame_swarm_route_progress(
                train.FULLGAME_CINNABAR_QUEST_PHASE, 8, -1, 0,
            ),
            (27, 0),
        )
        self.assertEqual(
            train.BIKE_SHOP_DOOR_ENTER_ACTION_GUIDANCE[0]["target"], (25, 13),
        )
        # Coast path: Lavender -> Route 12 Snorlax -> Fuchsia. The guidance
        # target is the tile the player STANDS on to use the flute, which is
        # (61,10) -- the Snorlax itself occupies (62,10) (pret object_event
        # x=10,y=62) and is not walkable until it has been beaten.
        self.assertEqual(
            train.ROUTE12_SNORLAX_ACTION_GUIDANCE[0]["target"], (61, 10),
        )
        self.assertIn(23, {r["map"] for r in train.PART11_ACTION_GUIDANCE})
        self.assertIn(4, {r["map"] for r in train.PART11_ACTION_GUIDANCE})
        # Merchant viewer (229,149) -> Bike Shop local (x=4,y=2) => (y,x)=(2,4).
        self.assertIn(
            (2, 4),
            {tuple(r["target"]) for r in train.BIKE_SHOP_VENDOR_ACTION_GUIDANCE},
        )
        self.assertIn(
            (2, 4),
            {tuple(r["target"]) for r in train.BIKE_SHOP_VENDOR_ACTION_GUIDANCE},
        )
        # R7 corridor + R16 flute tile (viewer 151,210 -> y=10,x=27).
        self.assertIn(
            (11, 11),
            {(s[1], s[2]) for s in train.SAFFRON_TO_ROUTE7_SNORLAX_APPROACH_STEPS
             if s[0] == 18},
        )
        self.assertEqual(
            train.ROUTE16_SNORLAX_ACTION_GUIDANCE[0]["target"], (10, 27),
        )
        self.assertTrue(
            any(r["map"] == 23 for r in train.ROUTE12_SNORLAX_ACTION_GUIDANCE)
        )
        self.assertTrue(
            any(r["map"] == 4 for r in train.PART11_ACTION_GUIDANCE)
        )
        # Safari teeth + Secret House + Warden Strength.
        self.assertEqual(
            train.SAFARI_GOLD_TEETH_ACTION_GUIDANCE[0]["target"], (7, 20),
        )
        self.assertEqual(
            train.SAFARI_GOLD_TEETH_ACTION_GUIDANCE[0]["unless_bag_item_ids"],
            (train.ITEM_GOLD_TEETH,),
        )
        self.assertEqual(
            train.SAFARI_SECRET_HOUSE_TALK_GUIDANCE[0]["map"], 222,
        )
        self.assertEqual(
            train.WARDEN_TEETH_ACTION_GUIDANCE[0]["require_event_bits"],
            (train.EVENT_GOT_HM03,),
        )
        self.assertEqual(
            train.WARDEN_TEETH_ACTION_GUIDANCE[0]["target"], (3, 3),
        )
        self.assertEqual(
            train.WARDEN_TEETH_ACTION_GUIDANCE[0]["face_action"], 2,
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(
                train.FULLGAME_LAVENDER_FUCHSIA_QUEST_PHASE, 178, 9, 9,
            )
        )
        self.assertFalse(
            train.fullgame_frontier_position_allowed(
                train.FULLGAME_HM03_SURF_QUEST_PHASE, 4, 5, 5,
            )
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(
                train.FULLGAME_HM03_SURF_QUEST_PHASE, 222, 4, 3,
            )
        )
        self.assertIn(
            train.FULLGAME_LAVENDER_FUCHSIA_QUEST_PHASE,
            train.FULLGAME_SABRINA_RETURN_PHASES,
        )
        self.assertIn(
            train.FULLGAME_KOGA_QUEST_PHASE,
            train.FULLGAME_PART11_PHASES,
        )

    def test_safari_route_uses_honest_four_map_circuit_with_budget(self):
        self.assertEqual(train.ADDR_SAFARI_STEPS_HI, 0xD70C)
        self.assertEqual(train.ADDR_SAFARI_STEPS_LO, 0xD70D)
        self.assertEqual(train.ADDR_NUM_SAFARI_BALLS, 0xDA46)
        self.assertEqual(
            train.SAFARI_CENTER_TO_EAST_STEPS[-1][4], 217,
        )
        self.assertEqual(train.SAFARI_EAST_TO_NORTH_STEPS[-1][4], 218)
        self.assertEqual(train.SAFARI_NORTH_TO_WEST_STEPS[-1][4], 219)
        self.assertEqual(train.FULLGAME_SAFARI_ROUTE_ORDER[218], 23)
        route_steps = (
            len(train.SAFARI_CENTER_TO_EAST_STEPS)
            + len(train.SAFARI_EAST_TO_NORTH_STEPS)
            + len(train.SAFARI_NORTH_TO_WEST_STEPS)
            + len(train.SAFARI_WEST_GOLD_TEETH_STEPS)
        )
        self.assertEqual(route_steps, 264)
        self.assertLess(route_steps, 500)
        west_secret_actions = {
            (step[1], step[2]): step[3]
            for step in train.SAFARI_WEST_TO_SECRET_HOUSE_STEPS
        }
        self.assertEqual(west_secret_actions[(7, 18)], 0)
        self.assertEqual(train.SAFARI_WEST_TO_SECRET_HOUSE_STEPS[-1][4], 222)
        self.assertEqual(
            train.SAFARI_SECRET_HOUSE_ENTRY_STEPS[0][:4], (222, 7, 3, 0),
        )

    def test_gold_teeth_guidance_releases_when_item_is_in_bag(self):
        env = self.make_env()
        rule = train.SAFARI_GOLD_TEETH_ACTION_GUIDANCE[0]
        env.quest_phase = train.FULLGAME_GOLD_TEETH_QUEST_PHASE
        self.assertTrue(env._action_guidance_event_allowed(rule))
        env.pyboy.memory[train.ADDR_NUM_BAG_ITEMS] = 1
        env.pyboy.memory[train.ADDR_BAG_ITEMS] = train.ITEM_GOLD_TEETH
        env.pyboy.memory[train.ADDR_BAG_ITEMS + 1] = 1
        self.assertFalse(env._action_guidance_event_allowed(rule))

    def test_safari_resource_counters_are_read_without_observation_change(self):
        env = self.make_env()
        env.pyboy.memory[train.ADDR_SAFARI_STEPS_HI] = 1
        env.pyboy.memory[train.ADDR_SAFARI_STEPS_LO] = 223
        env.pyboy.memory[train.ADDR_NUM_SAFARI_BALLS] = 30
        self.assertEqual(env._safari_resources(), (479, 30))

    def test_post_surf_route_exits_safari_and_reaches_warden(self):
        self.assertEqual(
            train.SAFARI_CENTER_POST_SURF_EXIT_STEPS[0][:4],
            (220, 11, 26, 1),
        )
        self.assertEqual(
            train.SAFARI_CENTER_POST_SURF_EXIT_STEPS[-1][4], 156,
        )
        self.assertEqual(
            train.FUCHSIA_TO_WARDEN_STEPS[0][:4], (7, 3, 18, 1),
        )
        self.assertEqual(train.FUCHSIA_TO_WARDEN_STEPS[-1][4], 155)
        self.assertEqual(len(train.FUCHSIA_TO_WARDEN_STEPS) - 1, 78)
        gate_entry = train.SAFARI_GATE_ENTRY_ACTION_GUIDANCE[0]
        self.assertNotIn(
            train.FULLGAME_HM04_STRENGTH_QUEST_PHASE,
            gate_entry['quest_phases'],
        )

    def test_post_surf_production_guidance_wins_first_match(self):
        env = self.make_env()
        env.action_guidance = train.PART11_ACTION_GUIDANCE
        env.action_guidance_requires_overworld = True
        env.quest_phase = train.FULLGAME_HM04_STRENGTH_QUEST_PHASE
        bit = train.EVENT_GOT_HM03
        env.pyboy.memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= (
            1 << (bit % 8)
        )
        self.assertEqual(
            env._matching_forced_action_guidance(220, 11, 26, 0), 1,
        )
        self.assertEqual(
            env._matching_forced_action_guidance(7, 4, 18, 0), 3,
        )
        self.assertEqual(
            env._matching_forced_action_guidance(155, 3, 3, 0), 2,
        )
        self.assertEqual(
            env._matching_forced_action_guidance(155, 3, 3, 0), 4,
        )

    def test_part11_helpers_are_scoped_to_story_maps(self):
        """Poké Flute / Bicycle / Safari-RUN helpers stay on Part 11 tiles."""
        self.assertTrue(hasattr(train.PokemonYellowEnv, "_try_use_poke_flute_on_snorlax"))
        self.assertTrue(hasattr(train.PokemonYellowEnv, "_try_select_bicycle_for_cycling_road"))
        self.assertTrue(hasattr(train.PokemonYellowEnv, "_try_force_safari_run"))
        # Fuchsia Gym callout recovery includes the Koga fight phases.
        recover = train.rock_tunnel_dialogue_recovery_action
        self.assertEqual(
            recover(157, train.FULLGAME_KOGA_QUEST_PHASE, 0, 4), "a",
        )
        self.assertIsNone(
            recover(157, train.FULLGAME_SABRINA_QUEST_PHASE, 0, 4),
        )

    def test_cinnabar_and_viridian_waypoints_follow_the_live_frontier(self):
        names = [item['name'] for item in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[196], 'reach_cinnabar_island')
        self.assertEqual(names[197], 'heal_at_cinnabar')
        self.assertEqual(names[202], 'collect_secret_key')
        self.assertEqual(names[211], 'beat_blaine')
        self.assertEqual(names[214], 'beat_viridian_gym_giovanni')
        self.assertEqual(names[215], 'leave_viridian_gym')
        self.assertEqual(names[216], 'teach_strength_to_squirtle')
        self.assertEqual(names[218], 'fly_to_cinnabar_for_training')
        self.assertEqual(names[219], 'enter_mansion_for_training')
        self.assertEqual(names[220], 'train_pikachu_mansion_level_51')
        self.assertEqual(names[239], 'train_pikachu_mansion_level_70')
        self.assertEqual(names[240], 'exit_mansion_after_training')
        self.assertEqual(names[241], 'fly_to_viridian_after_training')
        self.assertEqual(names[242], 'return_to_route_22_for_league')
        self.assertEqual(names[245], 'enter_victory_road_1f')
        self.assertEqual(names[-1], 'enter_hall_of_fame')
        self.assertEqual(
            len(train.FULLGAME_MANSION_GRIND_PHASES), 20,
        )
        self.assertEqual(
            train.FULLGAME_QUEST_WAYPOINTS[
                train.FULLGAME_TEACH_STRENGTH_QUEST_PHASE
            ]['party_move_id'],
            train.GEN1_STRENGTH_MOVE_ID,
        )
        self.assertEqual(
            train.FULLGAME_CINNABAR_QUIZ_PHASES,
            tuple(range(205, 211)),
        )

    def test_mansion_routes_use_north_stairs_and_all_three_switches(self):
        self.assertEqual(train.MANSION_2F_TO_3F_NORTH_STEPS[0][:4], (214, 11, 5, 3))
        self.assertEqual(train.MANSION_2F_TO_3F_NORTH_STEPS[-1][4], 215)
        switch_origins = {
            (step[1], step[2]) for step in train.MANSION_B1F_SECRET_KEY_STEPS
        }
        self.assertIn((26, 18), switch_origins)
        self.assertIn((4, 20), switch_origins)
        self.assertEqual(train.MANSION_B1F_SECRET_KEY_STEPS[-1][:3], (216, 11, 5))

    def test_final_gym_routes_and_helpers_are_production_scoped(self):
        self.assertTrue(hasattr(train.PokemonYellowEnv, '_try_answer_cinnabar_quiz'))
        self.assertTrue(hasattr(train.PokemonYellowEnv, '_try_dig_out_after_secret_key'))
        self.assertTrue(hasattr(train.PokemonYellowEnv, '_try_fly_to_viridian'))
        self.assertTrue(hasattr(train.PokemonYellowEnv, '_lead_squirtle_for_fire_and_ground_gyms'))
        self.assertEqual(train.VIRIDIAN_CITY_TO_GYM_STEPS[-1][4], 45)
        self.assertEqual(train.VIRIDIAN_GYM_TO_GIOVANNI_STEPS[-1][:3], (45, 2, 3))
        phases = {
            phase
            for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
            for phase in rule.get('quest_phases', ())
        }
        self.assertIn(train.FULLGAME_BLAINE_QUEST_PHASE, phases)
        self.assertIn(train.FULLGAME_VIRIDIAN_GIOVANNI_QUEST_PHASE, phases)

    def test_route22_final_rival_route_uses_the_open_lower_passage(self):
        self.assertEqual(
            train.ROUTE22_TO_FINAL_RIVAL_STEPS[0][:4],
            (33, 9, 39, 2),
        )
        self.assertEqual(
            train.ROUTE22_TO_FINAL_RIVAL_STEPS[-1][:4],
            (33, 5, 30, 2),
        )
        phases = {
            phase
            for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
            for phase in rule.get('quest_phases', ())
        }
        self.assertIn(train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE, phases)
        self.assertFalse(train.late_game_explorer_should_reset(
            True, train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE, 33,
        ))
        self.assertFalse(train.late_game_explorer_should_reset(
            True, train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE, 108,
        ))
        self.assertEqual(
            train.ROUTE22_TO_LEAGUE_GATE_STEPS[0][:4], (33, 9, 35, 1),
        )
        self.assertEqual(
            train.ROUTE22_TO_LEAGUE_GATE_STEPS[4][:4], (33, 12, 34, 2),
        )
        self.assertEqual(train.ROUTE22_TO_LEAGUE_GATE_STEPS[-1][4], 193)
        self.assertEqual(train.LEAGUE_GATE_TO_ROUTE23_STEPS[-1][4], 34)
        self.assertIn(
            train.FULLGAME_ROUTE23_QUEST_PHASE,
            {
                phase
                for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
                for phase in rule.get('quest_phases', ())
            },
        )
        env = self.make_env()
        env.quest_phase = train.FULLGAME_ROUTE23_QUEST_PHASE
        env.pyboy.memory[train.ADDR_MAP_ID] = 33
        env.pyboy.memory[train.ADDR_TEXT_BOX] = 1
        env._event_flag_is_set = lambda bit: bit == train.EVENT_BEAT_ROUTE22_RIVAL
        self.assertTrue(env._clear_route22_league_transit_joy_ignore())
        self.assertEqual(env.pyboy.memory[train.ADDR_TEXT_BOX], 0)

    def test_route22_rival_uses_real_battle_script_with_charizard_slash(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE
        memory[train.ADDR_MAP_ID] = 33
        memory[train.ADDR_PARTY_SIZE] = 2
        memory[train.ADDR_PARTY_SPECIES_LIST] = train.GEN1_PIKACHU_SPECIES_ID
        memory[train.ADDR_PARTY_SPECIES_LIST + 1] = train.GEN1_CHARIZARD_SPECIES_ID
        memory[train.PARTY_SPECIES_ADDRS[0]] = train.GEN1_PIKACHU_SPECIES_ID
        memory[train.PARTY_SPECIES_ADDRS[1]] = train.GEN1_CHARIZARD_SPECIES_ID
        for addr, value in zip(train.PARTY_MOVE_ID_ADDRS[1], (19, 163, 10, 43)):
            memory[addr] = value
        env._event_flag_is_set = lambda bit: bit in {
            train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE,
            train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE,
        }

        self.assertTrue(env._prepare_route22_rival_battle())
        self.assertEqual(
            memory[train.PARTY_SPECIES_ADDRS[0]],
            train.GEN1_CHARIZARD_SPECIES_ID,
        )
        self.assertEqual(memory[train.PARTY_MOVE_ID_ADDRS[0][0]], 163)

        def tap(button, press_ticks, release_ticks):
            memory[train.ADDR_BATTLE_FLAG] = 2

        env._tap_scripted_button = tap
        self.assertTrue(env._try_start_route22_rival_battle())
        self.assertEqual(
            memory[train.ADDR_ROUTE22_CUR_SCRIPT],
            train.ROUTE22_RIVAL_BATTLE_SCRIPT,
        )

    def test_route23_guidance_reaches_the_validated_surf_shore(self):
        self.assertEqual(
            train.ROUTE23_TO_FIRST_BADGE_GUARD_STEPS[0][:4], (34, 135, 9, 0),
        )
        self.assertEqual(
            train.ROUTE23_TO_FIRST_BADGE_GUARD_STEPS[-1][:3], (34, 120, 9),
        )
        self.assertEqual(
            train.ROUTE23_BETWEEN_BADGE_GUARDS_STEPS[0][:4], (34, 117, 9, 0),
        )
        self.assertEqual(
            train.ROUTE23_TO_SURF_SHORE_STEPS[0][:4], (34, 104, 10, 3),
        )
        phases = {
            phase
            for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
            for phase in rule.get('quest_phases', ())
        }
        self.assertIn(train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE, phases)
        self.assertFalse(train.late_game_explorer_should_reset(
            True, train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE, 193,
        ))

    def test_route23_badge_guard_and_surf_helpers_are_strictly_scoped(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE
        memory[train.ADDR_MAP_ID] = 34
        memory[train.ADDR_POS_A] = 119
        memory[train.ADDR_POS_B] = 9
        taps = []

        def clear_guard_tap(button, press_ticks, release_ticks):
            taps.append(button)
            if button == 'up' and taps.count('up') == 3:
                memory[train.ADDR_POS_A] = 117

        env._tap_scripted_button = clear_guard_tap
        self.assertTrue(env._try_clear_route23_badge_guard())
        self.assertEqual(taps, ['a', 'b', 'up'] * 3)

        memory[train.ADDR_POS_A] = 104
        memory[train.ADDR_POS_B] = 11
        memory[train.ADDR_WALK_BIKE_SURF_STATE] = 0
        flags = {train.EVENT_GOT_HM03, train.EVENT_GOT_HM04}
        env._event_flag_is_set = lambda bit: bit in flags
        env._party_index_with_move = lambda move: (
            3 if move == train.GEN1_SURF_MOVE_ID else None
        )
        taps.clear()

        def surf_tap(button, press_ticks, release_ticks):
            taps.append(button)
            if taps.count('a') == 4:
                memory[train.ADDR_WALK_BIKE_SURF_STATE] = 2

        env._tap_scripted_button = surf_tap
        self.assertTrue(env._try_start_route23_surf())
        self.assertEqual(memory[train.ADDR_START_SAVED_MENU_ITEM], 1)
        self.assertEqual(memory[train.ADDR_PARTY_SAVED_MENU_ITEM], 3)
        self.assertNotIn('down', taps)
        self.assertEqual(
            taps, ['up', 'start', 'a', 'a', 'a', 'a'],
        )

        memory[train.ADDR_MAP_ID] = 33
        memory[train.ADDR_WALK_BIKE_SURF_STATE] = 0
        taps.clear()
        self.assertFalse(env._try_start_route23_surf())
        self.assertEqual(taps, [])

    def test_victory_road_1f_route_and_strength_solution_are_scoped(self):
        self.assertEqual(
            train.VICTORY_ROAD_1F_TO_BOULDER_STEPS[0][:4], (108, 17, 8, 0),
        )
        self.assertEqual(
            train.VICTORY_ROAD_1F_TO_BOULDER_STEPS[-1][:4], (108, 14, 4, 3),
        )
        env = self.make_env()
        memory = env.pyboy.memory
        env.pyboy.tick = lambda ticks: None
        env.quest_phase = train.FULLGAME_VICTORY_ROAD_2F_QUEST_PHASE
        memory[train.ADDR_MAP_ID] = 108
        memory[train.ADDR_POS_A] = 14
        memory[train.ADDR_POS_B] = 5
        memory[train.ADDR_VICTORY_ROAD_1F_BOULDER1_Y] = 19
        memory[train.ADDR_VICTORY_ROAD_1F_BOULDER1_X] = 9
        flags = {train.EVENT_GOT_HM04}
        env._event_flag_is_set = lambda bit: bit in flags
        env._party_index_with_move = lambda move: (
            3 if move == train.GEN1_STRENGTH_MOVE_ID else None
        )
        taps = []

        def tap(button, press_ticks, release_ticks):
            taps.append(button)
            if taps.count('a') == 4:
                memory[train.ADDR_STATUS_FLAGS_1] |= 1 << train.BIT_STRENGTH_ACTIVE
            if len(taps) == 47:
                flags.add(train.EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH)

        env._tap_scripted_button = tap
        self.assertTrue(env._try_solve_victory_road_1f_boulder())
        self.assertEqual(memory[train.ADDR_PARTY_SAVED_MENU_ITEM], 3)
        self.assertEqual(len(taps), 47)
        self.assertEqual(taps[-4:], ['up', 'right', 'down', 'down'])

        flags.discard(train.EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH)
        memory[train.ADDR_MAP_ID] = 34
        taps.clear()
        self.assertFalse(env._try_solve_victory_road_1f_boulder())
        self.assertEqual(taps, [])

    def test_route22_rival_exit_recovery_requires_the_won_battle(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.pyboy.tick = lambda ticks: None
        env.quest_phase = train.FULLGAME_ROUTE23_QUEST_PHASE
        memory[train.ADDR_MAP_ID] = 33
        memory[train.ADDR_ROUTE22_CUR_SCRIPT] = train.ROUTE22_RIVAL_EXIT_WAIT_SCRIPT
        memory[train.ADDR_TEXT_BOX] = 0xFF
        memory[train.ADDR_FONT_LOADED] = 0
        flags = {
            train.EVENT_BEAT_ROUTE22_RIVAL,
            train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE,
            train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE,
        }
        env._event_flag_is_set = lambda bit: bit in flags
        env._set_event_flag_bit = lambda bit, value=True: (
            flags.add(bit) if value else flags.discard(bit)
        ) is None

        self.assertTrue(env._finish_route22_rival_exit())
        self.assertEqual(
            memory[train.ADDR_ROUTE22_CUR_SCRIPT], train.ROUTE22_RIVAL_DONE_SCRIPT,
        )
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 0)
        self.assertIn(train.EVENT_BEAT_ROUTE22_RIVAL, flags)
        self.assertNotIn(train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE, flags)
        self.assertNotIn(train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE, flags)

    def test_final_gym_emergency_heal_is_low_hp_battle_only_and_once(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_BLAINE_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        memory[train.ADDR_MAP_ID] = 166
        memory[train.ADDR_BATTLE_FLAG] = 2
        memory[train.ADDR_ACTIVE_MON_CUR_HP_HI] = 0
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 30
        memory[train.ADDR_ACTIVE_MON_MAX_HP_HI] = 0
        memory[train.ADDR_ACTIVE_MON_MAX_HP_LO] = 120
        lead_hi, lead_lo = train.PARTY_CUR_HP_ADDRS[0]
        memory[lead_hi], memory[lead_lo] = 0, 30

        self.assertTrue(env._use_final_gym_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 120)
        self.assertEqual(memory[lead_lo], 120)

        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 30
        self.assertFalse(env._use_final_gym_emergency_heal())
        del env._blaine_emergency_heals_used
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 0
        memory[lead_lo] = 0
        self.assertTrue(env._use_final_gym_emergency_heal())
        self.assertEqual(memory[lead_lo], 120)
        del env._blaine_emergency_heals_used
        memory[train.ADDR_BATTLE_FLAG] = 0
        self.assertFalse(env._use_final_gym_emergency_heal())

    def test_final_gym_lead_uses_trained_carries_when_squirtle_is_low_level(self):
        env = self.make_env()
        memory = env.pyboy.memory
        memory[train.ADDR_MAP_ID] = 166
        memory[train.ADDR_PARTY_SIZE] = 3
        memory[train.ADDR_PARTY_SPECIES_LIST] = next(iter(train.GEN1_SQUIRTLE_SPECIES_IDS))
        memory[train.ADDR_PARTY_SPECIES_LIST + 1] = train.GEN1_PIKACHU_SPECIES_ID
        memory[train.ADDR_PARTY_SPECIES_LIST + 2] = train.GEN1_CHARIZARD_SPECIES_ID
        species = [
            next(iter(train.GEN1_SQUIRTLE_SPECIES_IDS)),
            train.GEN1_PIKACHU_SPECIES_ID,
            train.GEN1_CHARIZARD_SPECIES_ID,
        ]
        for index, (species_id, level) in enumerate(zip(species, (11, 49, 41))):
            memory[train.PARTY_SPECIES_ADDRS[index]] = species_id
            memory[train.FIRST_PARTY_SPECIES_ADDR + index * train.PARTY_DATA_STRIDE + 0x21] = level
        memory[train.PARTY_MOVE_ID_ADDRS[1][0]] = train.GEN1_THUNDERBOLT_MOVE_ID
        memory[train.PARTY_MOVE_ID_ADDRS[2][0]] = train.GEN1_FLY_MOVE_ID
        env.quest_phase = train.FULLGAME_BLAINE_QUEST_PHASE
        env._effective_battle_flag = lambda: 0

        self.assertTrue(env._lead_squirtle_for_fire_and_ground_gyms())
        self.assertEqual(memory[train.PARTY_SPECIES_ADDRS[0]], train.GEN1_PIKACHU_SPECIES_ID)

        memory[train.ADDR_MAP_ID] = 45
        env.quest_phase = train.FULLGAME_VIRIDIAN_GIOVANNI_QUEST_PHASE
        self.assertTrue(env._lead_squirtle_for_fire_and_ground_gyms())
        self.assertEqual(memory[train.PARTY_SPECIES_ADDRS[0]], train.GEN1_CHARIZARD_SPECIES_ID)

    def test_viridian_trainer_trap_recovery_requires_both_honest_wins(self):
        env = self.make_env()
        memory = env.pyboy.memory
        env.quest_phase = train.FULLGAME_VIRIDIAN_GIOVANNI_QUEST_PHASE
        memory[train.ADDR_MAP_ID] = 45
        memory[train.ADDR_POS_A], memory[train.ADDR_POS_B] = 4, 10
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_TEXT_BOX] = 0
        events = {86}
        env._event_flag_is_set = lambda bit: bit in events
        self.assertFalse(env._clear_viridian_gym_trainer_trap())
        events.add(87)
        self.assertTrue(env._clear_viridian_gym_trainer_trap())
        self.assertEqual(
            (memory[train.ADDR_POS_A], memory[train.ADDR_POS_B]), (2, 3),
        )


if __name__ == "__main__":
    unittest.main()
