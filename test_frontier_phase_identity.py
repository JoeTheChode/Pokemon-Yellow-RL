import inspect
import unittest

import train


class FrontierPhaseIdentityTests(unittest.TestCase):
    def test_hall_of_fame_objective_is_publishable_before_credits(self):
        phase = train.FULLGAME_ENTER_HALL_OF_FAME_QUEST_PHASE

        self.assertFalse(train.fullgame_frontier_is_postcredits_phase(phase))
        self.assertTrue(train.fullgame_frontier_is_postcredits_phase(phase + 1))
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 120, 3, 4)
        )
        self.assertTrue(
            train.is_hall_of_fame_frontier_candidate(
                phase, 120, {train.EVENT_BEAT_CHAMPION_RIVAL}
            )
        )
        self.assertFalse(
            train.is_hall_of_fame_frontier_candidate(
                phase, 118, {train.EVENT_BEAT_CHAMPION_RIVAL}
            )
        )

    def test_hall_of_fame_frontier_uses_cutscene_proof_at_startup(self):
        source = inspect.getsource(train.validate_or_quarantine_swarm_frontier)
        self.assertIn('frontier_candidate_reaches_map_with_actions', source)
        self.assertIn("destination_map=118", source)
        self.assertIn("actions=('a', 'up')", source)

    def test_mastery_gate_diagnostic_uses_defined_reliability_flag(self):
        source = inspect.getsource(train.PokemonYellowEnv._swarm_sync)
        self.assertNotIn('rng_grind_phase', source)
        self.assertIn(
            "mode={'reliability' if reliability_mastery_phase else 'plateau'}",
            source,
        )

    def test_objective_name_wins_after_waypoint_insertion(self):
        waypoints = [
            {'name': 'before'},
            {'name': 'inserted_one'},
            {'name': 'inserted_two'},
            {'name': 'durable_objective'},
        ]

        self.assertEqual(
            train.reconcile_frontier_quest_phase(
                1, 'durable_objective', waypoints
            ),
            3,
        )

    def test_rock_tunnel_b1f_cannot_publish_as_grind_or_heal(self):
        grind_phase = min(train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES)

        self.assertFalse(
            train.fullgame_frontier_position_allowed(grind_phase, 232, 9, 4)
        )
        self.assertFalse(
            train.fullgame_frontier_position_allowed(
                train.FULLGAME_ROCK_TUNNEL_HEAL_QUEST_PHASE, 232, 9, 4
            )
        )

    def test_route10_recovery_locations_remain_publishable(self):
        grind_phase = min(train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES)

        for map_id in (21, 81, 82):
            with self.subTest(map_id=map_id):
                self.assertTrue(
                    train.fullgame_frontier_position_allowed(
                        grind_phase, map_id, 8, 8
                    )
                )

    def test_pre_forest_grind_rejects_route1_south_of_ledges(self):
        phase = train.FULLGAME_QUEST_WAYPOINTS.index(
            next(
                waypoint for waypoint in train.FULLGAME_QUEST_WAYPOINTS
                if waypoint['name'] == 'train_pikachu_level_10'
            )
        )
        self.assertIn(phase, train.FULLGAME_PRE_FOREST_GRIND_PHASES)
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 12, 9, 13)
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(phase, 12, 16, 12)
        )
        self.assertFalse(
            train.fullgame_frontier_position_allowed(phase, 12, 23, 12)
        )
        self.assertFalse(
            train.fullgame_frontier_position_allowed(phase, 12, 18, 10)
        )

    def test_pre_forest_grind_recovery_forces_up_from_poisoned_tile(self):
        matches = [
            (map_id, y, x, action)
            for map_id, y, x, action, _dest
            in train.ROUTE1_PRE_FOREST_GRIND_RECOVERY_STEPS
            if (map_id, y, x) == (12, 23, 12)
        ]
        self.assertEqual(matches, [(12, 23, 12, 0)])
        self.assertTrue(
            any(
                (map_id, y, x) == (12, 22, 12)
                for map_id, y, x, _action, _dest
                in train.ROUTE1_PRE_FOREST_GRIND_RECOVERY_STEPS
            )
        )

    def test_same_phase_grind_catchup_is_xp_only(self):
        phase = max(train.FULLGAME_PRE_FOREST_GRIND_PHASES)
        self.assertIs(
            train.swarm_grind_phase_catchup_needed(phase, phase, 884, 884, 20),
            False,
        )
        self.assertIs(
            train.swarm_grind_phase_catchup_needed(phase, phase, 884, 904, 20),
            True,
        )
        self.assertIs(
            train.swarm_grind_phase_catchup_needed(phase, phase, 900, 884, 20),
            False,
        )
        self.assertIsNone(
            train.swarm_grind_phase_catchup_needed(phase, phase - 1, 100, 884, 20)
        )

    def test_brock_bans_electric_in_pewter_gym(self):
        self.assertFalse(
            train.trainer_battle_allows_electric(
                54, train.FULLGAME_BROCK_QUEST_PHASE
            )
        )
        memory = bytearray(0x10000)
        memory[train.ADDR_PARTY_SIZE] = 1
        memory[train.ADDR_MAP_ID] = 54
        moves = [84, 98, 21, 86]
        for idx, move_id in enumerate(moves):
            memory[train.PARTY_MOVE_ID_ADDRS[0][idx]] = move_id
            memory[train.PARTY_MOVE_PP_ADDRS[0][idx]] = 20
            memory[train.ADDR_BATTLE_MON_MOVE_IDS[idx]] = move_id
            memory[train.ADDR_BATTLE_MON_PP[idx]] = 20
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = type('FakePyBoy', (), {'memory': memory})()
        env.quest_phase = train.FULLGAME_BROCK_QUEST_PHASE
        env._ensure_forced_attack_ready()
        slot0 = memory[train.PARTY_MOVE_ID_ADDRS[0][0]]
        self.assertNotEqual(slot0, train.GEN1_THUNDERSHOCK_MOVE_ID)
        self.assertGreater(
            (train.GEN1_MOVE_TABLE.get(slot0) or {}).get('power', 0), 0
        )
        self.assertNotEqual(
            (train.GEN1_MOVE_TABLE.get(slot0) or {}).get('type'),
            train.GEN1_ELECTRIC_TYPE,
        )

    def test_forced_attack_clears_player_disable_latch(self):
        memory = bytearray(0x10000)
        memory[train.ADDR_PARTY_SIZE] = 1
        memory[train.ADDR_MAP_ID] = 3
        memory[train.ADDR_PLAYER_DISABLED_MOVE] = 0x16
        for idx, move_id in enumerate((129, 84, 98, 21)):
            memory[train.PARTY_MOVE_ID_ADDRS[0][idx]] = move_id
            memory[train.PARTY_MOVE_PP_ADDRS[0][idx]] = 15
            memory[train.ADDR_BATTLE_MON_MOVE_IDS[idx]] = move_id
            memory[train.ADDR_BATTLE_MON_PP[idx]] = 15
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = type('FakePyBoy', (), {'memory': memory})()
        env.quest_phase = train.FULLGAME_CERULEAN_ROCKET_THIEF_QUEST_PHASE

        env._ensure_forced_attack_ready()

        self.assertEqual(memory[train.ADDR_PLAYER_DISABLED_MOVE], 0)

    def test_pewter_grind_restores_thundershock_to_slot_0(self):
        memory = bytearray(0x10000)
        memory[train.ADDR_PARTY_SIZE] = 1
        moves = [98, 45, 39, 86]  # Quick Attack, Growl, Tail Whip, Thunder Wave
        for idx, move_id in enumerate(moves):
            memory[train.PARTY_MOVE_ID_ADDRS[0][idx]] = move_id
            memory[train.PARTY_MOVE_PP_ADDRS[0][idx]] = 10
        self.assertTrue(train.ensure_lead_grind_electric_attack(memory))
        self.assertEqual(
            memory[train.PARTY_MOVE_ID_ADDRS[0][0]],
            train.GEN1_THUNDERSHOCK_MOVE_ID,
        )
        self.assertEqual(
            memory[train.ADDR_BATTLE_MON_MOVE_IDS[0]],
            train.GEN1_THUNDERSHOCK_MOVE_ID,
        )
        self.assertFalse(train.ensure_lead_grind_electric_attack(memory))

    def test_mt_moon_jj_win_is_publishable_at_low_hp(self):
        exit_phase = train.FULLGAME_MT_MOON_EXIT_QUEST_PHASE
        fight_phase = train.FULLGAME_JESSIE_JAMES_QUEST_PHASE
        self.assertEqual(train.swarm_required_lead_hp_fraction(0.4, exit_phase), 0.0)
        self.assertEqual(train.swarm_required_party_hp_fraction(0.4, exit_phase), 0.0)
        self.assertTrue(train.should_restore_mt_moon_jj_frontier_resources(exit_phase))
        self.assertTrue(train.should_restore_mt_moon_jj_frontier_resources(fight_phase))
        self.assertFalse(
            train.should_restore_mt_moon_jj_frontier_resources(
                train.FULLGAME_CERULEAN_QUEST_PHASE
            )
        )

    def test_brock_win_is_publishable_for_the_safe_center_route(self):
        post_brock = train.FULLGAME_BROCK_QUEST_PHASE + 1
        self.assertEqual(post_brock, train.FULLGAME_ROUTE3_QUEST_PHASE - 1)
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(0.4, post_brock), 0.0
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(0.4, post_brock), 0.0
        )
        # The exemption ends after the encounter-free Pewter Center handoff.
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(0.4, post_brock + 1), 0.4
        )
        self.assertEqual(
            train.swarm_required_party_hp_fraction(0.4, post_brock + 1), 0.4
        )

    def test_enter_bills_house_ignores_pending_warp_frame(self):
        waypoint = next(
            item for item in train.FULLGAME_QUEST_WAYPOINTS
            if item['name'] == 'enter_bills_house'
        )
        self.assertFalse(
            train.quest_waypoint_matches(waypoint, 88, 3, 45, set())
        )
        self.assertTrue(
            train.quest_waypoint_matches(waypoint, 88, 7, 2, set())
        )
        heal = train.FULLGAME_ROUTE25_HEAL_QUEST_PHASE
        self.assertFalse(
            train.fullgame_frontier_position_allowed(heal, 88, 3, 45)
        )
        self.assertTrue(
            train.fullgame_frontier_position_allowed(heal, 88, 7, 2)
        )

    def test_route1_grind_patrol_and_corridor_target_grass_pair(self):
        self.assertEqual(
            train.ROUTE1_PRE_FOREST_GRIND_PATROL_STEPS,
            ((12, 9, 13, 3, None), (12, 9, 14, 2, None)),
        )
        hits = [
            (rule['target'], rule['action'])
            for rule in train.PRE_FOREST_GRIND_ACTION_GUIDANCE
            if rule.get('map') == 12 and rule.get('target') == (15, 9)
        ]
        self.assertTrue(hits)
        self.assertEqual(hits[0][1], 0)


if __name__ == '__main__':
    unittest.main()
