#!/usr/bin/env python3
import unittest

import train


def first_rule(map_id, y, x, require=None):
    for rule in train.MELANIES_HOUSE_ACTION_GUIDANCE:
        if rule["map"] != map_id or tuple(rule["target"]) != (y, x):
            continue
        bits = rule.get("require_event_bits")
        if require is None and not bits:
            return rule
        if require is not None and bits == require:
            return rule
    return None


class MelanieLeaveGuidanceTests(unittest.TestCase):
    def test_inbound_door_stops_after_gift(self):
        door = first_rule(3, 16, 13)
        self.assertIsNotNone(door)
        self.assertEqual(int(door["action"]), 0)
        self.assertEqual(door["destination_map"], 63)
        self.assertEqual(door.get("unless_event_bits"), (168,))

    def test_glitched_frontier_tile_settles_onto_warp(self):
        settle = first_rule(63, 15, 13)
        self.assertIsNotNone(settle)
        self.assertEqual(int(settle["action"]), 1)
        self.assertNotIn("destination_map", settle)

    def test_talk_then_leave_aisle(self):
        talk = first_rule(63, 2, 3)
        self.assertEqual(int(talk["action"]), 4)
        self.assertEqual(talk.get("unless_event_bits"), (168,))
        leave = first_rule(63, 2, 3, require=(168,))
        self.assertEqual(int(leave["action"]), 2)
        door = first_rule(63, 7, 2, require=(168,))
        self.assertEqual(door["destination_map"], 3)
        self.assertEqual(int(door["action"]), 1)
        self.assertIsNone(first_rule(63, 7, 3, require=(168,)))
        aisle = first_rule(63, 2, 2, require=(168,))
        self.assertEqual(int(aisle["action"]), 1)
        pile = first_rule(63, 1, 2, require=(168,))
        self.assertEqual(int(pile["action"]), 1)

    def test_outdoor_connector_joins_cut_return(self):
        spawn = first_rule(3, 16, 13, require=(168,))
        self.assertEqual(int(spawn["action"]), 1)
        self.assertNotIn("destination_map", spawn)
        street = first_rule(3, 18, 13, require=(168,))
        self.assertEqual(int(street["action"]), 2)
        self.assertIsNone(first_rule(3, 18, 8, require=(168,)))

    def test_phase_scope_covers_cut_return(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[train.FULLGAME_MELANIES_HOUSE_QUEST_PHASE], "enter_melanies_house")
        self.assertEqual(names[train.FULLGAME_BULBASAUR_QUEST_PHASE], "receive_bulbasaur")
        self.assertEqual(names[train.FULLGAME_TEACH_CUT_QUEST_PHASE], "teach_cut_to_bulbasaur")
        self.assertEqual(
            names[train.FULLGAME_RETURN_WITH_CUT_QUEST_PHASE],
            "return_to_vermilion_with_cut",
        )
        self.assertNotEqual(
            train.FULLGAME_RETURN_WITH_CUT_QUEST_PHASE,
            train.FULLGAME_RETURN_FOR_BULBASAUR_QUEST_PHASE,
        )


if __name__ == "__main__":
    unittest.main()
