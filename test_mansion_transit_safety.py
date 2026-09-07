#!/usr/bin/env python3
import unittest
from unittest import mock

import train


class Memory(dict):
    def __getitem__(self, key):
        return self.get(key, 0)


class SettlingCinnabarDoorPyBoy:
    def __init__(self, memory):
        self.memory = memory
        self.ticks = 0

    def tick(self, ticks):
        self.ticks += int(ticks)
        if self.ticks >= 16:
            self.memory[train.ADDR_POS_A] = 3
            self.memory[train.ADDR_POS_B] = 18
            self.memory[train.ADDR_TEXT_BOX] = 0xFF
        if self.ticks >= 30:
            self.memory[train.ADDR_TEXT_BOX] = 0


class MansionTransitSafetyTests(unittest.TestCase):
    def test_volcano_grind_stale_joy_latch_requires_idle_settle(self):
        phase = min(train.FULLGAME_VOLCANO_EARTH_GRIND_PHASES)
        self.assertTrue(train.mansion_grind_stale_joy_requires_settle(
            216, phase, 0, 0xFF, 0,
        ))
        self.assertEqual(train.MANSION_GRIND_STALE_JOY_SETTLE_NOOPS, 3)

        # A real dialogue, active battle, another map, or another phase keeps
        # its existing input handling; this recovery is only for the observed
        # B1F post-battle latch.
        self.assertFalse(train.mansion_grind_stale_joy_requires_settle(
            216, phase, 0, 0xFF, 1,
        ))
        self.assertFalse(train.mansion_grind_stale_joy_requires_settle(
            216, phase, 1, 0xFF, 0,
        ))
        self.assertFalse(train.mansion_grind_stale_joy_requires_settle(
            165, phase, 0, 0xFF, 0,
        ))
        self.assertFalse(train.mansion_grind_stale_joy_requires_settle(
            216, train.FULLGAME_SECRET_KEY_QUEST_PHASE, 0, 0xFF, 0,
        ))

    def test_volcano_grind_invalid_proven_start_escapes_down(self):
        phase = min(train.FULLGAME_VOLCANO_EARTH_GRIND_PHASES)
        for position in ((2, 7), (1, 8)):
            with self.subTest(position=position):
                self.assertEqual(
                    train.mansion_volcano_grind_frontier_escape_action(
                        216, position[0], position[1], phase, 0, 0, 0,
                    ),
                    'down',
                )
        # A real latch/dialogue or any other tile/phase keeps ownership of the
        # action. This override only repairs the exact invalid durable traces.
        self.assertIsNone(train.mansion_volcano_grind_frontier_escape_action(
            216, 2, 7, phase, 0, 0xFF, 0,
        ))
        self.assertIsNone(train.mansion_volcano_grind_frontier_escape_action(
            216, 2, 7, phase, 0, 0, 1,
        ))
        self.assertIsNone(train.mansion_volcano_grind_frontier_escape_action(
            216, 3, 7, phase, 0, 0, 0,
        ))
        self.assertIsNone(train.mansion_volcano_grind_frontier_escape_action(
            216, 2, 7, train.FULLGAME_SECRET_KEY_QUEST_PHASE, 0, 0, 0,
        ))

    def test_volcano_grind_is_not_generic_mansion_dialogue_recovery(self):
        phase = min(train.FULLGAME_VOLCANO_EARTH_GRIND_PHASES)
        for stationary in range(4, 16):
            with self.subTest(stationary=stationary):
                self.assertIsNone(train.rock_tunnel_dialogue_recovery_action(
                    216, phase, 0, stationary, 2, 7,
                ))

    def test_b1f_descent_does_not_regress_on_intermediate_1f(self):
        phase = train.FULLGAME_MANSION_B1F_QUEST_PHASE
        for map_id in train.FULLGAME_MANSION_B1F_DESCENT_MAPS:
            with self.subTest(map_id=map_id):
                self.assertEqual(
                    train.fullgame_swarm_route_progress(
                        phase, map_id, -1, 0
                    ),
                    (31, 0),
                )

    def test_b1f_still_outranks_the_descent(self):
        self.assertEqual(
            train.fullgame_swarm_route_progress(
                train.FULLGAME_MANSION_B1F_QUEST_PHASE, 216, -1, 0
            ),
            (32, 0),
        )

    def test_secret_key_reserves_bag_slot_by_discarding_only_a_tm(self):
        protected_items = [train.ITEM_HM03, train.ITEM_SECRET_KEY - 1]
        items = protected_items + [train.ITEM_TM01] + list(range(1, 18))
        memory = Memory({
            train.ADDR_MAP_ID: 216,
            train.ADDR_NUM_BAG_ITEMS: 20,
        })
        for index, item_id in enumerate(items):
            memory[train.ADDR_BAG_ITEMS + index * 2] = item_id
            memory[train.ADDR_BAG_ITEMS + index * 2 + 1] = 1
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_SECRET_KEY_QUEST_PHASE
        env.env_id = 1

        self.assertTrue(env._reserve_mansion_secret_key_bag_slot())
        self.assertEqual(memory[train.ADDR_NUM_BAG_ITEMS], 19)
        remaining = [
            memory[train.ADDR_BAG_ITEMS + index * 2]
            for index in range(19)
        ]
        for item_id in protected_items:
            self.assertIn(item_id, remaining)
        self.assertNotIn(train.ITEM_TM01, remaining)
        self.assertEqual(memory[train.ADDR_BAG_ITEMS + 38], 0xFF)

    def test_owned_tm28_restores_dig_without_replacing_surf(self):
        memory = Memory({
            train.ADDR_PARTY_SIZE: 1,
            train.PARTY_SPECIES_ADDRS[0]: next(iter(train.GEN1_SQUIRTLE_SPECIES_IDS)),
            train.ADDR_NUM_BAG_ITEMS: 1,
            train.ADDR_BAG_ITEMS: train.ITEM_TM28_DIG,
            train.ADDR_BAG_ITEMS + 1: 1,
        })
        starting_moves = [33, 39, 145, train.GEN1_SURF_MOVE_ID]
        for addr, move_id in zip(train.PARTY_MOVE_ID_ADDRS[0], starting_moves):
            memory[addr] = move_id
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_MANSION_ESCAPE_QUEST_PHASE

        self.assertTrue(env._ensure_squirtle_knows_dig_for_mansion_escape())
        moves = [memory[addr] for addr in train.PARTY_MOVE_ID_ADDRS[0]]
        self.assertEqual(
            moves,
            [33, train.GEN1_DIG_MOVE_ID, 145, train.GEN1_SURF_MOVE_ID],
        )
        self.assertEqual(memory[train.PARTY_MOVE_PP_ADDRS[0][1]], 10)

    def test_dig_restore_is_scoped_and_requires_owned_tm28(self):
        memory = Memory({
            train.ADDR_PARTY_SIZE: 1,
            train.PARTY_SPECIES_ADDRS[0]: next(iter(train.GEN1_SQUIRTLE_SPECIES_IDS)),
            train.ADDR_NUM_BAG_ITEMS: 0,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_MANSION_ESCAPE_QUEST_PHASE
        self.assertFalse(env._ensure_squirtle_knows_dig_for_mansion_escape())

        memory[train.ADDR_NUM_BAG_ITEMS] = 1
        memory[train.ADDR_BAG_ITEMS] = train.ITEM_TM28_DIG
        env.quest_phase -= 1
        self.assertFalse(env._ensure_squirtle_knows_dig_for_mansion_escape())

    def test_live_mansion_moveset_replaces_weakest_attack_and_preserves_surf(self):
        memory = Memory({
            train.ADDR_PARTY_SIZE: 1,
            train.PARTY_SPECIES_ADDRS[0]: next(iter(train.GEN1_SQUIRTLE_SPECIES_IDS)),
            train.ADDR_NUM_BAG_ITEMS: 1,
            train.ADDR_BAG_ITEMS: train.ITEM_TM28_DIG,
            train.ADDR_BAG_ITEMS + 1: 1,
        })
        # Exact phase-257 live frontier moves: Swift, Tackle, Bubble, Surf.
        starting_moves = [129, 33, 145, train.GEN1_SURF_MOVE_ID]
        for addr, move_id in zip(train.PARTY_MOVE_ID_ADDRS[0], starting_moves):
            memory[addr] = move_id
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_MANSION_ESCAPE_QUEST_PHASE

        self.assertTrue(env._ensure_squirtle_knows_dig_for_mansion_escape())
        moves = [memory[addr] for addr in train.PARTY_MOVE_ID_ADDRS[0]]
        self.assertEqual(
            moves,
            [129, 33, train.GEN1_DIG_MOVE_ID, train.GEN1_SURF_MOVE_ID],
        )
        self.assertEqual(memory[train.PARTY_MOVE_PP_ADDRS[0][2]], 10)

    def test_post_strength_dig_follows_strength_in_live_field_menu(self):
        self.assertEqual(train.GEN1_POST_STRENGTH_SURF_FIELD_MENU_INDEX, 2)
        self.assertEqual(train.GEN1_STRENGTH_FIELD_MENU_INDEX, 3)
        self.assertEqual(
            train.GEN1_POST_STRENGTH_DIG_FIELD_MENU_INDEX,
            train.GEN1_STRENGTH_FIELD_MENU_INDEX + 1,
        )

    def test_post_blaine_handoff_restores_party_for_safe_publication(self):
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
        max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[0]
        memory = Memory({
            train.ADDR_MAP_ID: 1,
            train.ADDR_BATTLE_FLAG: 0,
            train.ADDR_PARTY_SIZE: 1,
            cur_hi: 0,
            cur_lo: 34,
            max_hi: 0,
            max_lo: 100,
            train.PARTY_MOVE_ID_ADDRS[0][0]: 33,
            train.PARTY_MOVE_PP_ADDRS[0][0]: 1,
        })
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_FLY_TO_VIRIDIAN_QUEST_PHASE
        env._effective_battle_flag = mock.Mock(return_value=0)
        env._event_flag_is_set = mock.Mock(return_value=True)

        self.assertTrue(env._restore_blaine_post_battle_resources())
        self.assertEqual((memory[cur_hi] << 8) | memory[cur_lo], 100)
        self.assertEqual(
            memory[train.PARTY_MOVE_PP_ADDRS[0][0]],
            train.GEN1_MOVE_TABLE[33]['max_pp'],
        )
        self.assertFalse(env._restore_blaine_post_battle_resources())

    def test_post_blaine_restore_requires_victory_phase(self):
        memory = Memory({train.ADDR_MAP_ID: 1, train.ADDR_PARTY_SIZE: 1})
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=memory)
        env.quest_phase = train.FULLGAME_BLAINE_QUEST_PHASE
        env._effective_battle_flag = mock.Mock(return_value=0)
        env._event_flag_is_set = mock.Mock(return_value=True)
        self.assertFalse(env._restore_blaine_post_battle_resources())

    def test_post_blaine_frontier_bridge_joins_verified_gym_exit(self):
        self.assertEqual(
            train.CINNABAR_GYM_POST_BLAINE_BRIDGE_STEPS,
            (
                (166, 2, 3, 3, None),
                (166, 2, 4, 1, None),
                (166, 3, 4, 3, None),
            ),
        )
        self.assertEqual(
            train.CINNABAR_GYM_EXIT_AFTER_BLAINE_STEPS[0][:3],
            (166, 3, 5),
        )
        bridge_rules = [
            item for item in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
            if item.get('map') == 166
            and item.get('target') in {(2, 3), (2, 4), (3, 4)}
            and train.FULLGAME_FLY_TO_VIRIDIAN_QUEST_PHASE
            in item.get('quest_phases', ())
        ]
        self.assertEqual(
            [(item['target'], item['action']) for item in bridge_rules],
            [((2, 3), 3), ((2, 4), 1), ((3, 4), 3)],
        )

    def test_post_blaine_door_settles_before_fly_can_open_start(self):
        memory = Memory({
            train.ADDR_MAP_ID: 8,
            train.ADDR_POS_A: 17,
            train.ADDR_POS_B: 17,
            train.ADDR_TEXT_BOX: 0,
            train.ADDR_FONT_LOADED: 0,
            train.ADDR_BATTLE_FLAG: 0,
        })
        pyboy = SettlingCinnabarDoorPyBoy(memory)
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = pyboy
        env.quest_phase = train.FULLGAME_FLY_TO_VIRIDIAN_QUEST_PHASE

        self.assertTrue(env._settle_cinnabar_gym_exit_for_fly())
        self.assertGreaterEqual(pyboy.ticks, 30)
        self.assertLessEqual(pyboy.ticks, 64)
        self.assertEqual(
            (memory[train.ADDR_MAP_ID], memory[train.ADDR_POS_A], memory[train.ADDR_POS_B]),
            (8, 3, 18),
        )
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 0)


if __name__ == "__main__":
    unittest.main()
