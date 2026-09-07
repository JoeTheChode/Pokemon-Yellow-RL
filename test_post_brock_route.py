import unittest

import train


class PostBrockRouteProgressTests(unittest.TestCase):
    def test_route_orders_cave_between_west_and_east_route4(self):
        key = train.post_brock_route_progress_key
        route = [
            key(14, 0, 59),
            key(15, 6, 18),
            key(59, 35, 14),
            key(60, 5, 5, 2),
            key(61, 17, 21, 3),
            key(60, 7, 5, 4),
            key(15, 5, 24, 5),
            key(3, 35, 25, 6),
        ]
        self.assertEqual(route, sorted(route))
        self.assertEqual(len(route), len(set(route)))

    def test_entrance_center_does_not_regress_entrance_progress(self):
        self.assertEqual(
            train.post_brock_route_progress_key(68, 7, 3, 1),
            (1, 18),
        )

    def test_pending_cave_warp_frame_is_not_serialized(self):
        self.assertFalse(train.is_safe_post_brock_swarm_frontier(15, 6, 18))
        self.assertTrue(train.is_safe_post_brock_swarm_frontier(15, 6, 17))
        self.assertTrue(train.is_safe_post_brock_swarm_frontier(59, 35, 14))

    def test_unresolved_jessie_james_pocket_is_not_serialized(self):
        self.assertFalse(train.is_safe_post_brock_swarm_frontier(
            61, 3, 3, fossil_chosen=True, jessie_james_defeated=False
        ))
        self.assertTrue(train.is_safe_post_brock_swarm_frontier(
            61, 3, 3, fossil_chosen=True, jessie_james_defeated=True
        ))
        self.assertTrue(train.is_safe_post_brock_swarm_frontier(
            61, 7, 4, fossil_chosen=True, jessie_james_defeated=False
        ))

    def test_b2f_progress_rejects_across_wall_x_shortcuts(self):
        progress = train.mt_moon_b2f_progress_value
        self.assertEqual(progress(27, 29), 0)
        self.assertEqual(progress(11, 35), 0)
        self.assertEqual(progress(17, 29), 8)

    def test_unresolved_upper_east_b2f_pocket_is_not_serialized(self):
        self.assertFalse(train.is_safe_post_brock_swarm_frontier(
            61, 11, 35, fossil_chosen=False
        ))
        self.assertTrue(train.is_safe_post_brock_swarm_frontier(
            61, 17, 35, fossil_chosen=False
        ))

    def test_optional_mt_moon_ladder_pockets_are_not_serialized(self):
        self.assertFalse(train.is_safe_post_brock_swarm_frontier(
            60, 18, 27, fossil_chosen=False
        ))
        self.assertTrue(train.is_safe_post_brock_swarm_frontier(
            60, 17, 11, fossil_chosen=False
        ))
        self.assertFalse(train.is_safe_post_brock_swarm_frontier(
            61, 24, 23, fossil_chosen=False
        ))
        self.assertFalse(train.is_safe_post_brock_swarm_frontier(
            61, 27, 29, fossil_chosen=False
        ))
        self.assertTrue(train.is_safe_post_brock_swarm_frontier(
            61, 17, 21, fossil_chosen=False
        ))

    def test_stage_neutralizes_map_ladder_and_guides_cave_warps(self):
        stage = train.MILESTONES[train.MILESTONE_INDEX['07_beat_brock']]
        self.assertEqual(stage['reward_cap'], 550)
        rules = {
            (rule['map'], rule['target']): rule['action']
            for rule in stage['action_guidance']
        }
        self.assertEqual(rules[(15, (6, 18))], 0)
        self.assertEqual(rules[(59, (5, 6))], 2)
        self.assertEqual(rules[(60, (17, 20))], 3)
        self.assertEqual(rules[(61, (7, 4))], 3)
        destinations = {
            (rule['map'], rule['target']): rule['destination_map']
            for rule in stage['action_guidance']
        }
        self.assertEqual(destinations[(15, (6, 18))], 59)
        self.assertEqual(destinations[(59, (5, 6))], 60)
        self.assertEqual(destinations[(60, (17, 20))], 61)
        self.assertEqual(destinations[(61, (7, 4))], 60)
        self.assertEqual(stage['same_position_step_limit'], 16)
        self.assertEqual(stage['ep_length'], 16384)
        self.assertFalse(stage['swarm_enabled'])
        self.assertFalse(stage['action_guidance_requires_overworld'])
        entrance = next(
            rule for rule in stage['action_guidance']
            if rule['map'] == 15 and rule['target'] == (6, 18)
        )
        self.assertTrue(entrance['force_action'])
        self.assertEqual(entrance['transition_bonus'], 0.0)

    def test_transition_guidance_is_not_paid_for_an_attempt(self):
        env = object.__new__(train.PokemonYellowEnv)
        env.action_guidance = train.POST_BROCK_ACTION_GUIDANCE
        env.action_guidance_requires_overworld = False
        env.frontier_action_guidance_bonus = 0.0
        env.claimed_transition_guidance = set()

        match = env._matching_transition_guidance(15, 6, 18, 1, 3)
        self.assertIsNotNone(match)
        self.assertEqual(env._matching_transition_guidance(15, 6, 17, 0, 0), None)
        self.assertEqual(env._action_guidance_reward(15, 6, 18, 0, 0), 0.0)
        self.assertEqual(env._claim_transition_guidance_reward(match, 15), 0.0)
        self.assertEqual(env._claim_transition_guidance_reward(match, 59), 0.0)
        self.assertEqual(env._claim_transition_guidance_reward(match, 59), 0.0)


if __name__ == '__main__':
    unittest.main()
