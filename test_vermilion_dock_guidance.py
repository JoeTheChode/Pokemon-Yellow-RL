#!/usr/bin/env python3
import unittest

import train


def first_rule(y, x, map_id=5):
    for rule in train.VERMILION_DOCK_ACTION_GUIDANCE:
        if rule["map"] == map_id and tuple(rule["target"]) == (y, x):
            return rule
    return None


class VermilionDockGuidanceTests(unittest.TestCase):
    def test_dock_warps_use_pret_xy_not_swapped(self):
        dests = [
            tuple(rule["target"])
            for rule in train.VERMILION_DOCK_ACTION_GUIDANCE
            if rule.get("destination_map") == 94
        ]
        self.assertEqual(set(dests), {(31, 18), (31, 19)})
        self.assertNotIn((18, 31), dests)
        self.assertNotIn((19, 31), dests)

    def test_live_pile_and_harbor_first_actions(self):
        expected = {
            (4, 20): 1,
            (8, 20): 1,
            (4, 11): 3,
            (14, 20): 3,
            (14, 30): 1,
            (20, 30): 1,
            (26, 30): 2,
            (26, 18): 1,
            (30, 18): 1,
            (31, 18): 1,
            (12, 30): 2,
            (4, 28): 2,
        }
        for (y, x), action in expected.items():
            rule = first_rule(y, x)
            self.assertIsNotNone(rule, f"missing ({y},{x})")
            self.assertEqual(int(rule["action"]), action, f"({y},{x})")

    def test_ticket_tile_does_not_wait_for_warp(self):
        ticket = first_rule(30, 18)
        self.assertIsNotNone(ticket)
        self.assertNotIn("destination_map", ticket)
        warp = first_rule(31, 18)
        self.assertEqual(warp["destination_map"], 94)
        self.assertGreaterEqual(int(warp.get("wait_ticks") or 0), 128)

    def test_phase_constant_is_enter_dock(self):
        self.assertEqual(
            train.FULLGAME_QUEST_WAYPOINTS[
                train.FULLGAME_ENTER_VERMILION_DOCK_QUEST_PHASE
            ]["name"],
            "enter_vermilion_dock",
        )


if __name__ == "__main__":
    unittest.main()
