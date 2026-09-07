#!/usr/bin/env python3
import unittest

import train


def first_rule(map_id, y, x):
    for rule in train.RETURN_TO_CERULEAN_ACTION_GUIDANCE:
        if rule["map"] == map_id and tuple(rule["target"]) == (y, x):
            return rule
    return None


class ReturnToCeruleanGuidanceTests(unittest.TestCase):
    def test_harbor_pile_goes_north_not_back_to_dock(self):
        self.assertEqual(int(first_rule(5, 26, 30)["action"]), 0)
        self.assertEqual(int(first_rule(5, 27, 31)["action"]), 2)
        self.assertEqual(int(first_rule(5, 14, 30)["action"]), 2)
        self.assertEqual(int(first_rule(5, 8, 18)["action"]), 0)
        north = first_rule(5, 0, 18)
        self.assertEqual(int(north["action"]), 0)
        self.assertEqual(north["destination_map"], 17)
        dock = first_rule(5, 31, 18)
        self.assertEqual(int(dock["action"]), 0)
        self.assertNotIn("destination_map", dock)

    def test_core_reaches_cerulean_via_underground(self):
        dests = [
            (rule["map"], tuple(rule["target"]), rule["destination_map"])
            for rule in train.RETURN_TO_CERULEAN_ACTION_GUIDANCE
            if "destination_map" in rule
        ]
        self.assertIn((5, (0, 18), 17), dests)
        self.assertIn((17, (14, 17), 74), dests)
        self.assertIn((16, (0, 18), 3), dests)
        self.assertIn((3, (16, 13), 63), dests)
        self.assertNotIn(62, [d[2] for d in dests])

    def test_cerulean_south_uses_x36_seam_not_void(self):
        pile = first_rule(3, 29, 28)
        self.assertEqual(int(pile["action"]), 3)
        self.assertEqual(int(first_rule(3, 29, 19)["action"]), 3)
        self.assertEqual(int(first_rule(3, 28, 36)["action"]), 0)
        self.assertEqual(int(first_rule(3, 19, 36)["action"]), 2)
        self.assertEqual(int(first_rule(3, 19, 35)["action"]), 0)
        door = first_rule(3, 16, 13)
        self.assertEqual(door["destination_map"], 63)
        self.assertEqual(int(door["action"]), 0)

    def test_phase_constant(self):
        self.assertEqual(
            train.FULLGAME_QUEST_WAYPOINTS[
                train.FULLGAME_RETURN_FOR_BULBASAUR_QUEST_PHASE
            ]["name"],
            "return_to_cerulean_for_bulbasaur",
        )


if __name__ == "__main__":
    unittest.main()
