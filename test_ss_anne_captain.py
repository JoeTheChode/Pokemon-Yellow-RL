#!/usr/bin/env python3
import unittest

import train


def first_rule(map_id, y, x):
    for rule in train.SS_ANNE_CAPTAIN_ACTION_GUIDANCE:
        if rule["map"] == map_id and tuple(rule["target"]) == (y, x):
            return rule
    return None


class SSAnneCaptainGuidanceTests(unittest.TestCase):
    def test_captain_door_is_2f_not_bow(self):
        dests = [
            (rule["map"], tuple(rule["target"]))
            for rule in train.SS_ANNE_CAPTAIN_ACTION_GUIDANCE
            if rule.get("destination_map") == 101
        ]
        self.assertEqual(dests, [(96, (4, 36))])
        door = next(
            rule
            for rule in train.SS_ANNE_CAPTAIN_ACTION_GUIDANCE
            if rule.get("destination_map") == 101
        )
        self.assertEqual(int(door["action"]), 0)
        self.assertFalse(door.get("require_event_bits"))

    def test_2f_avoids_3f_stairs_and_walks_east(self):
        inbound = [
            rule
            for rule in train.SS_ANNE_CAPTAIN_ACTION_GUIDANCE
            if rule["map"] == 96 and not rule.get("require_event_bits")
        ]

        def inbound_first(y, x):
            for rule in inbound:
                if tuple(rule["target"]) == (y, x):
                    return int(rule["action"])
            self.fail(f"missing inbound (96,{y},{x})")

        self.assertEqual(inbound_first(12, 2), 3)
        self.assertEqual(inbound_first(4, 2), 3)
        self.assertEqual(inbound_first(4, 3), 1)
        self.assertEqual(inbound_first(13, 10), 3)
        self.assertEqual(inbound_first(8, 36), 0)

    def test_after_hm01_leaves_cabin_for_city(self):
        leave_door = first_rule(101, 7, 0)
        self.assertEqual(leave_door["destination_map"], 96)
        self.assertEqual(leave_door.get("require_event_bits"), (1504,))
        stairs = first_rule(96, 4, 2)
        self.assertEqual(stairs["destination_map"], 95)
        self.assertEqual(stairs.get("require_event_bits"), (1504,))
        gangway = first_rule(95, 0, 26)
        self.assertEqual(gangway["destination_map"], 94)
        city = first_rule(94, 0, 14)
        self.assertEqual(city["destination_map"], 5)

    def test_3f_and_bow_return_to_2f(self):
        bow = first_rule(99, 6, 13)
        self.assertEqual(bow["destination_map"], 97)
        hall = first_rule(97, 3, 19)
        self.assertEqual(hall["destination_map"], 96)
        self.assertEqual(int(first_rule(97, 3, 0)["action"]), 3)
        self.assertEqual(int(first_rule(99, 4, 4)["action"]), 3)

    def test_1f_stairs_to_2f(self):
        stairs = next(
            rule
            for rule in train.SS_ANNE_CAPTAIN_ACTION_GUIDANCE
            if rule["map"] == 95
            and tuple(rule["target"]) == (6, 2)
            and rule.get("destination_map") == 96
        )
        self.assertEqual(int(stairs["action"]), 0)
        inbound = next(
            rule
            for rule in train.SS_ANNE_CAPTAIN_ACTION_GUIDANCE
            if rule["map"] == 95
            and tuple(rule["target"]) == (6, 20)
            and not rule.get("require_event_bits")
        )
        self.assertEqual(int(inbound["action"]), 2)

    def test_phase_constants(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(
            names[train.FULLGAME_REACH_SS_ANNE_CAPTAIN_QUEST_PHASE],
            "reach_ss_anne_captain",
        )
        self.assertEqual(
            names[train.FULLGAME_RUB_CAPTAINS_BACK_QUEST_PHASE],
            "rub_captains_back",
        )

    def test_captain_cutscene_is_a_free_skip(self):
        waypoints = train.FULLGAME_QUEST_WAYPOINTS
        rub = train.FULLGAME_RUB_CAPTAINS_BACK_QUEST_PHASE
        self.assertTrue(
            train.swarm_skipped_phases_were_already_satisfied(
                waypoints, rub, rub + 2, 34
            )
        )
        self.assertFalse(
            train.swarm_skipped_phases_were_already_satisfied(
                waypoints, rub, rub + 3, 34
            )
        )


if __name__ == "__main__":
    unittest.main()
