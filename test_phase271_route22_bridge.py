#!/usr/bin/env python3
import unittest

import train


class Phase271Route22BridgeTests(unittest.TestCase):
    def test_post_strength_live_frontier_bridge_joins_route22_guidance(self):
        self.assertEqual(
            train.VIRIDIAN_POST_STRENGTH_ROUTE22_BRIDGE_STEPS,
            (
                (1, 29, 25, 0, None),
                (1, 28, 25, 2, None),
                (1, 28, 24, 2, None),
                (1, 28, 23, 2, None),
                (1, 28, 22, 2, None),
                (1, 28, 21, 2, None),
                (1, 28, 20, 2, None),
                (1, 28, 19, 0, None),
                (1, 27, 19, 0, None),
                (1, 26, 19, 3, None),
                (1, 26, 20, 3, None),
            ),
        )
        self.assertTrue(any(
            step[:3] == (1, 26, 21) and step[3] == 0
            for step in train.VIRIDIAN_CITY_TO_ROUTE22_STEPS
        ))

        bridge_targets = {
            (29, 25), (28, 25), (28, 24), (28, 23), (28, 22), (28, 21),
            (28, 20), (28, 19), (27, 19), (26, 19), (26, 20),
        }
        bridge_rules = [
            rule for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
            if rule.get('map') == 1
            and rule.get('target') in bridge_targets
            and train.FULLGAME_ROUTE22_LEAGUE_QUEST_PHASE
            in rule.get('quest_phases', ())
        ]
        self.assertEqual(
            [(rule['target'], rule['action']) for rule in bridge_rules],
            [
                ((29, 25), 0),
                ((28, 25), 2), ((28, 24), 2), ((28, 23), 2),
                ((28, 22), 2), ((28, 21), 2), ((28, 20), 2),
                ((28, 19), 0), ((27, 19), 0),
                ((26, 19), 3), ((26, 20), 3),
            ],
        )


if __name__ == '__main__':
    unittest.main()
