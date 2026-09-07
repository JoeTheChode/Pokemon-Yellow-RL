#!/usr/bin/env python3
import inspect
import unittest
from pathlib import Path

import train


def first_rule(rules, map_id, y, x):
    for rule in rules:
        if rule["map"] == map_id and tuple(rule["target"]) == (y, x):
            return rule
    return None


class Route10ToTunnelTests(unittest.TestCase):
    def test_phase_is_enter_rock_tunnel(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(
            names[train.FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE],
            "enter_rock_tunnel",
        )
        self.assertGreater(
            train.FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE,
            train.FULLGAME_ROCK_TUNNEL_HEAL_QUEST_PHASE,
        )

    def test_center_door_walks_south(self):
        door = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 19, 11)
        self.assertEqual(int(door["action"]), 1)
        self.assertNotIn("destination_map", door)
        south = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 20, 11)
        self.assertEqual(int(south["action"]), 2)
        tree = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 20, 9)
        self.assertEqual(int(tree["action"]), 2)
        mouth = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 20, 8)
        self.assertEqual(int(mouth["action"]), 2)
        west = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 20, 3)
        self.assertEqual(int(west["action"]), 0)
        approach = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 18, 4)
        self.assertEqual(int(approach["action"]), 3)

    def test_entry_phase_forces_trainer_a(self):
        self.assertIn(
            train.FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE,
            train.FULLGAME_POST_MISTY_QUEST_PHASES
            | frozenset({train.FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE}),
        )
        src = Path(train.__file__).read_text(encoding="utf-8")
        block = src[src.index("'trainer_battle_quest_phases': sorted(") :]
        block = block[: block.index("force_a_in_trainer_battles")]
        self.assertIn("FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE", block)
        hook = inspect.getsource(train.PokemonYellowEnv._try_cut_progression_tree)
        self.assertIn(
            "FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE: (21, (20, 10), 'left')",
            hook,
        )

    def test_tunnel_door_faces_up(self):
        for y in (17, 18):
            rule = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, y, 8)
            self.assertEqual(int(rule["action"]), 0)
            self.assertEqual(rule["destination_map"], 82)

    def test_south_wander_returns_north(self):
        wrap = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 26, 14)
        self.assertEqual(int(wrap["action"]), 2)
        tail = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 28, 12)
        self.assertEqual(int(tail["action"]), 0)
        entry = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 21, 7, 3)
        self.assertEqual(int(entry["action"]), 3)

    def test_center_leave_uses_door_mats(self):
        mat = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 81, 7, 3)
        self.assertEqual(int(mat["action"]), 1)
        self.assertEqual(mat["destination_map"], 21)
        aisle = first_rule(train.ROUTE10_TO_TUNNEL_ACTION_GUIDANCE, 81, 4, 3)
        self.assertEqual(int(aisle["action"]), 1)

    def test_phase97_registered_and_snaps_outdoor_coords(self):
        src = Path(train.__file__).read_text(encoding="utf-8")
        # Registered alongside ROUTE10_TO_TUNNEL_DOOR_ACTION_GUIDANCE, which
        # shadows its broken half on map 21, so this is a tuple now.
        self.assertIn("list(ROUTE10_TO_TUNNEL_ACTION_GUIDANCE)", src)
        # The registration that uses the table must be the entry phase one:
        # look back from the use, not forward from the first mention of the
        # constant (which appears in unrelated phase-set arithmetic).
        use = src.index("list(ROUTE10_TO_TUNNEL_ACTION_GUIDANCE)")
        self.assertIn("FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE", src[use - 600:use])
        hook = inspect.getsource(
            train.PokemonYellowEnv._settle_rock_tunnel_center_pending_warp
        )
        self.assertIn("ADDR_POS_A] = 7", hook)
        self.assertIn("ADDR_POS_B] = 3", hook)
        self.assertIn("ADDR_LAST_MAP] = 21", hook)
        match = inspect.getsource(
            train.PokemonYellowEnv._matching_forced_action_guidance
        )
        self.assertIn("_settle_rock_tunnel_center_pending_warp", match)


if __name__ == "__main__":
    unittest.main()
