#!/usr/bin/env python3
import unittest
from unittest import mock

import train


class Memory(dict):
    def __getitem__(self, key):
        return self.get(key, 0)


class FuchsiaTransitSafetyTests(unittest.TestCase):
    def test_route12_post_snorlax_never_rewrites_coordinates(self):
        for position in ((31, 10), (63, 10), (63, 11), (65, 14),
                         (67, 14), (90, 8), (107, 10)):
            with self.subTest(position=position):
                memory = Memory({
                    train.ADDR_MAP_ID: 23,
                    train.ADDR_POS_A: position[0],
                    train.ADDR_POS_B: position[1],
                    train.ADDR_BATTLE_FLAG: 0,
                })
                before = dict(memory)
                env = object.__new__(train.PokemonYellowEnv)
                env.pyboy = mock.Mock(memory=memory)
                env.same_position_step_count = 200
                env._event_flag_is_set = mock.Mock(return_value=True)
                self.assertFalse(env._force_route12_snorlax_flute_tile())
                self.assertEqual(memory, before)

    def test_route12_recovery_yields_to_battle_inputs(self):
        for battle in (1, 2):
            for position in ((63, 11), (65, 14), (90, 8)):
                for stuck in (0, 2, 4, 200, 2048):
                    with self.subTest(battle=battle, position=position, stuck=stuck):
                        self.assertIsNone(train.route12_post_snorlax_coast_action(
                            23, *position, train.FULLGAME_FUCHSIA_QUEST_PHASE,
                            True, 10, stuck, battle_flag=battle,
                        ))

    def test_soul_marsh_repel_stays_active_until_route15(self):
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

    def test_pocket_recovery_never_rewrites_coordinates(self):
        phase = min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES)
        for map_id, position in (
            (24, (10, 51)),
            (24, (11, 25)),
            (24, (8, 10)),
            (25, (8, 15)),
            (25, (9, 12)),
        ):
            with self.subTest(map_id=map_id, position=position):
                memory = Memory({
                    train.ADDR_MAP_ID: map_id,
                    train.ADDR_POS_A: position[0],
                    train.ADDR_POS_B: position[1],
                    train.ADDR_BATTLE_FLAG: 0,
                })
                env = object.__new__(train.PokemonYellowEnv)
                env.pyboy = mock.Mock(memory=memory)
                env.quest_phase = phase
                env.same_position_step_count = 200

                self.assertFalse(env._force_route13_water_pocket())
                self.assertEqual(
                    (memory[train.ADDR_POS_A], memory[train.ADDR_POS_B]),
                    position,
                )

    def test_input_based_route14_escape_remains_available(self):
        phase = min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES)
        self.assertEqual(train.route13_water_pocket_action(25, 30, 13, phase), 'down')
        self.assertEqual(train.route13_water_pocket_action(25, 44, 3, phase), 'left')

    def test_route13_connection_recovery_rejoins_east_escape(self):
        owned = {(m, y, x): a for m, y, x, a, _ in
                 train.ROUTE13_CROSSING_STEPS + train.ROUTE13_RECOVERY_STEPS}
        for y in range(4):
            self.assertEqual(owned[(24, y, 51)], 1)
        for y in range(5, 10):
            self.assertEqual(owned[(24, y, 50)], 3)
        self.assertEqual(owned[(24, 10, 50)], 1)
        # The old y=9 west edge and y=5 west exit are solid tiles.
        self.assertNotEqual(owned.get((24, 9, 50)), 2)
        self.assertNotIn((24, 5, 0), owned)
        self.assertEqual(train.ROUTE13_CROSSING_STEPS[-1], (24, 10, 0, 2, 25))

    def test_route14_transit_releases_stale_trainer_text(self):
        phase = train.FULLGAME_FUCHSIA_QUEST_PHASE
        for stuck, expected in ((0, 'down'), (2, 'b'), (3, 'down'), (4, 'b')):
            self.assertEqual(train.route13_water_pocket_action(
                25, 30, 13, phase, stuck), expected)
        for battle in (1, 2):
            self.assertIsNone(train.route13_water_pocket_action(
                25, 30, 13, phase, 2048, battle_flag=battle))

    def test_route14_exit_preserves_ram_and_normal_action_accounting(self):
        memory = Memory({train.ADDR_MAP_ID: 25, train.ADDR_POS_A: 44,
                         train.ADDR_POS_B: 3, train.ADDR_BATTLE_FLAG: 0,
                         train.ADDR_TEXT_BOX: 1, train.ADDR_FONT_LOADED: 1})
        for base in (train.ADDR_PIKACHU_SPRITE_STATE_DATA_1,
                     train.ADDR_PIKACHU_SPRITE_STATE_DATA_2):
            for offset in range(train.PIKACHU_SPRITE_STATE_DATA_SIZE):
                memory[base + offset] = 0xA5
        before = dict(memory)
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_FUCHSIA_QUEST_PHASE
        env.same_position_step_count = 200
        self.assertFalse(env._clear_route14_exit_follower_softlock())
        self.assertEqual(memory, before)
        env.pyboy.button_press.assert_not_called()
        env.pyboy.tick.assert_not_called()

    def test_fuchsia_gym_level_progress_can_become_durable_frontier(self):
        for phase in train.FULLGAME_SOUL_MARSH_GRIND_PHASES:
            with self.subTest(phase=phase):
                self.assertTrue(
                    train.fullgame_frontier_position_allowed(
                        phase, 157, 15, 1
                    )
                )

    def test_koga_win_is_atomic_during_soul_marsh_grind(self):
        for phase in train.FULLGAME_SOUL_MARSH_GRIND_PHASES:
            with self.subTest(phase=phase):
                self.assertIn(
                    train.EVENT_BEAT_KOGA,
                    train.allowed_atomic_frontier_event_bits(
                        phase, {train.EVENT_BEAT_KOGA}
                    ),
                )
        self.assertNotIn(
            train.EVENT_BEAT_KOGA,
            train.allowed_atomic_frontier_event_bits(
                min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES) - 1,
                {train.EVENT_BEAT_KOGA},
            ),
        )

    def test_post_koga_route_returns_to_route15_grass(self):
        rules = {
            (rule['map'], *rule['target']): rule
            for rule in train.PART11_ACTION_GUIDANCE
            if rule.get('quest_phases') == train.FULLGAME_SOUL_MARSH_GRIND_PHASES
            and rule.get('require_event_bits') == (train.EVENT_BEAT_KOGA,)
        }
        expected_actions = {
            (157, 11, 4): 2,
            (157, 11, 3): 0,
            (7, 28, 5): 3,
            (7, 28, 8): 1,
            (7, 16, 39): 3,
            (184, 4, 7): 3,
            (26, 8, 14): 3,
            (26, 8, 18): 3,
            (26, 9, 14): 3,
            (26, 9, 18): 3,
        }
        for key, action in expected_actions.items():
            with self.subTest(position=key):
                self.assertEqual(rules[key]['action'], action)
        self.assertEqual(rules[(7, 16, 39)]['destination_map'], 26)
        self.assertEqual(rules[(184, 4, 7)]['destination_map'], 26)

    def test_post_koga_transit_does_not_look_behind_gym_frontier(self):
        phase = min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES)
        for map_id in train.FULLGAME_SOUL_MARSH_RETURN_MAPS:
            with self.subTest(map_id=map_id):
                self.assertEqual(
                    train.fullgame_swarm_route_progress(
                        phase, map_id, -1, 0
                    ),
                    (19, 0),
                )

    def test_secret_house_route_covers_actual_left_door_landing(self):
        actions = {
            (map_id, y, x): action
            for map_id, y, x, action, _destination
            in train.SAFARI_SECRET_HOUSE_ENTRY_STEPS
        }
        self.assertEqual(actions[(222, 7, 2)], 0)
        self.assertEqual(actions[(222, 6, 2)], 0)
        self.assertEqual(actions[(222, 5, 2)], 3)
        self.assertEqual(actions[(222, 5, 3)], 0)
        self.assertEqual(actions[(222, 7, 3)], 0)

    def test_secret_house_talk_does_not_rearm_face_action(self):
        self.assertNotIn(
            'face_action', train.SAFARI_SECRET_HOUSE_TALK_GUIDANCE[0]
        )

    def test_post_hm03_secret_house_state_can_become_durable(self):
        self.assertTrue(train.fullgame_frontier_position_allowed(
            train.FULLGAME_HM04_STRENGTH_QUEST_PHASE, 222, 4, 3,
            {train.EVENT_GOT_HM03},
        ))

    def test_hm04_return_corridor_never_ranks_behind_secret_house(self):
        phase = train.FULLGAME_HM04_STRENGTH_QUEST_PHASE
        for map_id in train.FULLGAME_HM04_RETURN_MAPS:
            with self.subTest(map_id=map_id):
                self.assertEqual(
                    train.fullgame_swarm_route_progress(
                        phase, map_id, -1, 0
                    ),
                    (24, 0),
                )

    def test_secret_house_reserves_hm03_slot_by_discarding_only_a_tm(self):
        phase = min(train.FULLGAME_SAFARI_COLLECTION_PHASES)
        items = [train.ITEM_HM03 - 2, train.ITEM_HM03 - 1, train.ITEM_TM01]
        items.extend(range(1, 18))
        memory = Memory({
            train.ADDR_MAP_ID: 222,
            train.ADDR_NUM_BAG_ITEMS: 20,
        })
        for index, item_id in enumerate(items):
            memory[train.ADDR_BAG_ITEMS + index * 2] = item_id
            memory[train.ADDR_BAG_ITEMS + index * 2 + 1] = 1
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = phase
        env.env_id = 1

        self.assertTrue(env._reserve_safari_hm03_bag_slot())
        self.assertEqual(memory[train.ADDR_NUM_BAG_ITEMS], 19)
        remaining = [
            memory[train.ADDR_BAG_ITEMS + index * 2]
            for index in range(19)
        ]
        self.assertIn(train.ITEM_HM03 - 2, remaining)
        self.assertIn(train.ITEM_HM03 - 1, remaining)
        self.assertNotIn(train.ITEM_TM01, remaining)
        self.assertEqual(memory[train.ADDR_BAG_ITEMS + 38], 0xFF)


if __name__ == "__main__":
    unittest.main()

