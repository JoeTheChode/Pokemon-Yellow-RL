#!/usr/bin/env python3
"""The Rock Tunnel level grind, moved ahead of the tunnel onto Route 10 grass.

Fixtures are generated from pret/pokeyellow: WALKABLE_21 is one bit per tile
(row-major), SPRITES_21 are trainer tiles, WARPS_21 warp tiles, GRASS_21 the
encounter tiles, and BLOCKED_21 the edges refused by a ledge or a tile-pair
collision. Regenerate with the scratch script r10_emit.py.
"""
import unittest

import train

WALKABLE_21 = (
    "00000000000000000000ffffc00000ff7ff0ff07ff7ff0ff07fc7f000002fc7fc0ff07007c00c0070040001004fc41c0a004fc7dc0ef07fc7dc0ef07fc7dc0ef07fc7fc0ff07fc7fc0ff0300c000000c00c000000c00c000000c00c000000c00c000040cfcc3c03d0cfcc3c03f0cfcffc0ff0f00000000000000000000000000000000c003102cfcffc3df3ffcc3c33f080cc003003c0cc0c30020fcffc3ff3f00fc03c03f000c008000c00f00fc00c003003c00"
)
MAP_SIZE_21 = (72, 20)  # height, width
SPRITES_21 = frozenset([(25, 7), (44, 10), (54, 7), (57, 3), (61, 3), (64, 14)])
WARPS_21 = frozenset([(17, 8), (19, 11), (39, 6), (53, 8)])
GRASS_21 = frozenset([(6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11), (9, 12), (9, 13)])
BLOCKED_21 = frozenset((
    
))

# --- test body ---------------------------------------------------------------

DELTA = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
MOVE = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
PATROL = ((8, 9), (9, 9))          # viewer global (342,137) and (343,137)
TUNNEL_DOOR = (17, 8)              # Route 10 warp 1 -> Rock Tunnel 1F
CENTER_DOOR = (19, 11)             # Route 10 warp -> Rock Tunnel Pokemon Center
ENTRANCE_MAT = (3, 15)             # Rock Tunnel 1F warp -> Route 10


def walkable_21(y, x):
    height, width = MAP_SIZE_21
    if not (0 <= y < height and 0 <= x < width):
        return False
    index = y * width + x
    return bool(bytes.fromhex(WALKABLE_21)[index // 8] >> (index % 8) & 1)


def rules_by_tile(guidance, map_id):
    out = {}
    for rule in guidance:
        if int(rule["map"]) == map_id:
            out.setdefault(tuple(rule["target"]), rule)
    return out


def action_of(rule):
    action = rule["action"]
    if isinstance(action, (list, tuple, set)):
        action = next(iter(action))
    return int(action)


class Route10GrindTests(unittest.TestCase):
    """The level grind runs before Rock Tunnel, on the Route 10 grass.

    Grinding after the traverse was backwards: the party had to survive 15
    trainers across both floors, with no Center inside, to earn the levels that
    were meant to make surviving it possible. On 2026-08-19 that was 951 of 951
    episodes ending in party_wipe with a L35 Pikachu carrying three L10
    starters.
    """

    def test_grind_runs_between_route_10_and_the_heal(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        grind = [i for i, n in enumerate(names)
                 if n.startswith("train_pikachu_rock_tunnel_level_")]
        self.assertTrue(grind)
        self.assertEqual(names[min(grind) - 1], "reach_route_10")
        self.assertEqual(names[max(grind) + 1], "heal_at_rock_tunnel_center")
        self.assertEqual(names[max(grind) + 2], "enter_rock_tunnel")
        # The traverse must still follow the heal, not precede the grind.
        self.assertGreater(names.index("reach_rock_tunnel_first_b1f"), max(grind))
        self.assertGreater(names.index("exit_rock_tunnel_to_lower_route_10"),
                           names.index("reach_rock_tunnel_final_1f"))

    def test_the_patrol_sits_on_real_grass(self):
        """A patrol off the grass rolls no encounters and the grind never ends."""
        for tile in PATROL:
            self.assertIn(tile, GRASS_21, msg="%s is not a grass tile" % (tile,))
        tiles = rules_by_tile(train.ROCK_TUNNEL_GRIND_ACTION_GUIDANCE, 21)
        for tile in PATROL:
            self.assertIn(tile, tiles)
        # ... and that the two tiles point at each other, so it is a real patrol.
        a, b = PATROL
        dy, dx = MOVE[action_of(tiles[a])]
        self.assertEqual((a[0] + dy, a[1] + dx), b)
        dy, dx = MOVE[action_of(tiles[b])]
        self.assertEqual((b[0] + dy, b[1] + dx), a)

    def test_no_route_10_rule_walks_into_terrain(self):
        for name in ("ROCK_TUNNEL_GRIND_ACTION_GUIDANCE",
                     "ROUTE10_CENTER_APPROACH_ACTION_GUIDANCE",
                     "ROUTE10_TO_TUNNEL_DOOR_ACTION_GUIDANCE"):
            for tile, rule in rules_by_tile(getattr(train, name), 21).items():
                action = action_of(rule)
                dy, dx = MOVE[action]
                nxt = (tile[0] + dy, tile[1] + dx)
                context = "%s: %s %s -> %s" % (name, tile, DELTA[action], nxt)
                self.assertTrue(walkable_21(*nxt), msg=context + " (wall)")
                self.assertNotIn(nxt, SPRITES_21, msg=context + " (trainer)")
                self.assertNotIn((tile, nxt), BLOCKED_21,
                                 msg=context + " (ledge or tile-pair)")

    def test_grinding_never_wanders_into_a_warp(self):
        """A warp ends the grind: the tunnel at (17,8), the Center at (19,11)."""
        tiles = rules_by_tile(train.ROCK_TUNNEL_GRIND_ACTION_GUIDANCE, 21)
        for tile, rule in tiles.items():
            dy, dx = MOVE[action_of(rule)]
            nxt = (tile[0] + dy, tile[1] + dx)
            self.assertNotIn(nxt, WARPS_21,
                             msg="grind at %s steps onto warp %s" % (tile, nxt))

    def _walks_to(self, guidance, start, goal, limit=400):
        tiles = rules_by_tile(guidance, 21)
        cur, seen = start, set()
        for _ in range(limit):
            if cur == goal:
                return True
            if cur not in tiles or cur in seen:
                return False
            seen.add(cur)
            dy, dx = MOVE[action_of(tiles[cur])]
            cur = (cur[0] + dy, cur[1] + dx)
        return False

    def test_every_owned_tile_reaches_its_goal(self):
        legs = (
            ("ROUTE10_CENTER_APPROACH_ACTION_GUIDANCE", CENTER_DOOR),
            ("ROUTE10_TO_TUNNEL_DOOR_ACTION_GUIDANCE", TUNNEL_DOOR),
        )
        for name, goal in legs:
            guidance = getattr(train, name)
            for tile in rules_by_tile(guidance, 21):
                self.assertTrue(self._walks_to(guidance, tile, goal),
                                msg="%s: %s never reaches %s" % (name, tile, goal))

    def test_grind_flows_into_the_patrol_from_anywhere_on_route_10(self):
        guidance = train.ROCK_TUNNEL_GRIND_ACTION_GUIDANCE
        for tile in rules_by_tile(guidance, 21):
            self.assertTrue(
                self._walks_to(guidance, tile, PATROL[0])
                or self._walks_to(guidance, tile, PATROL[1]),
                msg="grind: %s never reaches the grass" % (tile,))

    def test_a_worker_that_resets_inside_the_tunnel_walks_back_out(self):
        """The durable frontier is the entrance mat, so this is the normal case."""
        tiles = rules_by_tile(train.ROCK_TUNNEL_GRIND_ACTION_GUIDANCE, 82)
        self.assertTrue(tiles, "grind guidance owns no tunnel tiles")
        self.assertIn(ENTRANCE_MAT, tiles)
        self.assertEqual(tiles[ENTRANCE_MAT]["destination_map"], 21)
        # Stepping off and back onto the mat is what fires the warp.
        self.assertEqual(action_of(tiles[ENTRANCE_MAT]), 0)
        self.assertEqual(action_of(tiles[(2, 15)]), 1)

    def test_the_frontier_may_be_published_on_route_10(self):
        """Leaving this at the tunnel floors refuses every frontier the grind earns."""
        for phase in train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES:
            self.assertTrue(
                train.fullgame_frontier_position_allowed(phase, 21, 8, 8),
                msg="phase %d cannot publish on the Route 10 grass" % phase)
            self.assertTrue(
                train.fullgame_frontier_position_allowed(phase, 81, 3, 3),
                msg="phase %d cannot publish in the Center" % phase)

    def test_grind_phases_still_count_as_rng_grinds(self):
        """That is what heals the saved snapshot and drops the lead-HP floor."""
        grind = set(train.FULLGAME_ROCK_TUNNEL_GRIND_PHASES)
        self.assertTrue(grind <= set(train.FULLGAME_RNG_GRIND_QUEST_PHASES))
        for phase in grind:
            self.assertTrue(train.should_restore_rock_tunnel_grind_frontier_resources(
                phase, phase - 1, 0))
        after = max(grind) + 1
        self.assertEqual(
            train.swarm_required_lead_hp_fraction(0.40, after), 0.0,
            "the post-grind heal candidate still faces the lead-HP floor")


if __name__ == "__main__":
    unittest.main()
