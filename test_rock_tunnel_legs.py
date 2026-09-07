#!/usr/bin/env python3
"""Rock Tunnel route legs, checked against ROM collision physics.

Fixtures below are generated from pret/pokeyellow: WALKABLE_* is one bit
per tile (row-major), SPRITES_* are trainer tiles (solid), WARPS_* are warp
tiles, and PAIR_BLOCKED_* are Gen 1 tile-pair collisions -- adjacent tiles
that are both walkable but that the game refuses to let the player cross.
Regenerate with scratch script rt_emit.py against the pret checkout.
"""
import unittest
from pathlib import Path

import train

WALKABLE_82 = (
    "e69efbe7a9061ef8e701f6defbe7bdf6de03003cf6deffffbef0c0ffffbcfeefffffbefeefffff3cfeaffbbabffc0ff000bcbeeffbffbfbeeffbff3fbeeffbffbf3ccff3ffbfbeffebeeaebeff030000beffbfbfbf3cffbfbfbffeffbfbfbffeff3f3f3ffeffbfbfbffcffbfbfbfbabbbbbfbf0200003f3ffeffffbfbffcffffbfbffeffffbfbffcffff3f3ffefdeffdbffef5effdbffefdeffdbffefdeffd3ffefdeff97ffefdeff97ffe01e0f97f0000000000"
)
SPRITES_82 = frozenset([(5, 7), (8, 23), (15, 17), (16, 5), (21, 37), (24, 22), (24, 32)])
PAIR_BLOCKED_82 = frozenset((
    ((4, 34), (4, 33)),
    ((6, 2), (6, 1)),
    ((6, 14), (6, 13)),
    ((6, 34), (6, 33)),
    ((7, 2), (7, 1)),
    ((7, 14), (7, 13)),
    ((7, 15), (8, 15)),
    ((7, 16), (8, 16)),
    ((7, 17), (8, 17)),
    ((7, 19), (8, 19)),
    ((7, 25), (8, 25)),
    ((7, 27), (8, 27)),
    ((7, 28), (8, 28)),
    ((7, 29), (8, 29)),
    ((7, 31), (8, 31)),
    ((8, 2), (8, 1)),
    ((8, 20), (8, 19)),
    ((8, 34), (8, 33)),
    ((9, 7), (10, 7)),
    ((10, 2), (10, 1)),
    ((10, 8), (10, 7)),
    ((10, 14), (10, 13)),
    ((10, 20), (10, 19)),
    ((11, 2), (11, 1)),
    ((11, 8), (11, 7)),
    ((11, 14), (11, 13)),
    ((11, 20), (11, 19)),
    ((12, 2), (12, 1)),
    ((12, 8), (12, 7)),
    ((12, 14), (12, 13)),
    ((12, 20), (12, 19)),
    ((13, 21), (14, 21)),
    ((13, 22), (14, 22)),
    ((13, 23), (14, 23)),
    ((13, 25), (14, 25)),
    ((13, 26), (14, 26)),
    ((13, 27), (14, 27)),
    ((13, 29), (14, 29)),
    ((13, 30), (14, 30)),
    ((13, 31), (14, 31)),
    ((13, 33), (14, 33)),
    ((13, 34), (14, 34)),
    ((13, 35), (14, 35)),
    ((13, 37), (14, 37)),
    ((14, 2), (14, 1)),
    ((14, 8), (14, 7)),
    ((15, 2), (15, 1)),
    ((15, 8), (15, 7)),
    ((16, 2), (16, 1)),
    ((16, 8), (16, 7)),
    ((18, 2), (18, 1)),
    ((19, 2), (19, 1)),
    ((20, 2), (20, 1)),
    ((21, 3), (22, 3)),
    ((21, 4), (22, 4)),
    ((21, 5), (22, 5)),
    ((21, 7), (22, 7)),
    ((21, 8), (22, 8)),
    ((21, 9), (22, 9)),
    ((21, 11), (22, 11)),
    ((21, 12), (22, 12)),
    ((21, 13), (22, 13)),
    ((21, 15), (22, 15)),
    ((21, 16), (22, 16)),
    ((21, 17), (22, 17)),
    ((21, 19), (22, 19)),
    ((21, 20), (22, 20)),
    ((21, 21), (22, 21)),
))
WARPS_82 = frozenset([(0, 15), (3, 5), (3, 15), (3, 37), (11, 17), (17, 37), (33, 15), (35, 15)])
MAP_SIZE_82 = (36, 40)  # height, width

WALKABLE_232 = (
    "eaeeeeeeee0200000000beeffffebfbceffffe3fbeeffffebf3ecffffcbfbeeffbbebebceffbbe3ebeefabbebe3ecf033cbcfeeffffebffceffffe3ffeeffffebffecffffcbfbeffebeebfbcff0b003fbefffbffbf3efff3ffbfbeeffbffbfbccffbff3fbeeffbbbbb3ecff30380aaeffefbbf000ff0fb3ffeeffffbbffeeffff3bffeeffffbbffccffffb3fbafbbbfbbf02c003f0bffeffffffbffcffffff3ffeffffffbffcffffffbfeeeeeeeeae0000000000"
)
SPRITES_232 = frozenset([(5, 3), (5, 33), (10, 6), (10, 30), (13, 11), (21, 20), (28, 14), (30, 26)])
PAIR_BLOCKED_232 = frozenset((
    ((2, 2), (2, 1)),
    ((2, 8), (2, 7)),
    ((2, 14), (2, 13)),
    ((2, 26), (2, 25)),
    ((3, 8), (3, 7)),
    ((3, 14), (3, 13)),
    ((3, 26), (3, 25)),
    ((4, 2), (4, 1)),
    ((4, 8), (4, 7)),
    ((4, 14), (4, 13)),
    ((4, 26), (4, 25)),
    ((5, 2), (5, 1)),
    ((5, 19), (6, 19)),
    ((5, 31), (6, 31)),
    ((6, 2), (6, 1)),
    ((6, 8), (6, 7)),
    ((6, 14), (6, 13)),
    ((6, 20), (6, 19)),
    ((6, 26), (6, 25)),
    ((6, 34), (6, 33)),
    ((7, 8), (7, 7)),
    ((7, 14), (7, 13)),
    ((7, 20), (7, 19)),
    ((7, 21), (8, 21)),
    ((7, 23), (8, 23)),
    ((7, 26), (7, 25)),
    ((7, 34), (7, 33)),
    ((8, 2), (8, 1)),
    ((8, 8), (8, 7)),
    ((8, 14), (8, 13)),
    ((8, 26), (8, 25)),
    ((8, 34), (8, 33)),
    ((9, 2), (9, 1)),
    ((10, 2), (10, 1)),
    ((10, 14), (10, 13)),
    ((10, 26), (10, 25)),
    ((11, 14), (11, 13)),
    ((11, 26), (11, 25)),
    ((12, 2), (12, 1)),
    ((12, 14), (12, 13)),
    ((12, 26), (12, 25)),
    ((13, 2), (13, 1)),
    ((13, 7), (14, 7)),
    ((13, 19), (14, 19)),
    ((13, 21), (14, 21)),
    ((13, 22), (14, 22)),
    ((13, 23), (14, 23)),
    ((13, 26), (14, 26)),
    ((13, 27), (14, 27)),
    ((13, 29), (14, 29)),
    ((13, 30), (14, 30)),
    ((13, 31), (14, 31)),
    ((14, 2), (14, 1)),
    ((14, 8), (14, 7)),
    ((14, 32), (14, 31)),
    ((15, 8), (15, 7)),
    ((16, 2), (16, 1)),
    ((16, 8), (16, 7)),
    ((16, 20), (16, 19)),
    ((17, 2), (17, 1)),
    ((17, 13), (18, 13)),
    ((18, 2), (18, 1)),
    ((18, 8), (18, 7)),
    ((18, 14), (18, 13)),
    ((18, 20), (18, 19)),
    ((19, 8), (19, 7)),
    ((19, 20), (19, 19)),
    ((19, 27), (20, 27)),
    ((19, 28), (20, 28)),
    ((19, 29), (20, 29)),
    ((19, 31), (20, 31)),
    ((19, 32), (20, 32)),
    ((19, 33), (20, 33)),
    ((19, 35), (20, 35)),
    ((19, 36), (20, 36)),
    ((19, 37), (20, 37)),
    ((20, 2), (20, 1)),
    ((20, 8), (20, 7)),
    ((20, 14), (20, 13)),
    ((20, 20), (20, 19)),
    ((21, 2), (21, 1)),
    ((21, 3), (22, 3)),
    ((21, 5), (22, 5)),
    ((21, 14), (22, 14)),
    ((21, 15), (22, 15)),
    ((21, 17), (22, 17)),
    ((22, 8), (22, 7)),
    ((22, 20), (22, 19)),
    ((22, 28), (22, 27)),
    ((23, 28), (23, 27)),
    ((24, 2), (24, 1)),
    ((24, 14), (24, 13)),
    ((24, 28), (24, 27)),
    ((25, 2), (25, 1)),
    ((25, 14), (25, 13)),
    ((26, 2), (26, 1)),
    ((26, 14), (26, 13)),
    ((26, 28), (26, 27)),
    ((27, 3), (28, 3)),
    ((27, 4), (28, 4)),
    ((27, 5), (28, 5)),
    ((27, 7), (28, 7)),
    ((27, 8), (28, 8)),
    ((27, 9), (28, 9)),
    ((27, 11), (28, 11)),
    ((27, 19), (28, 19)),
    ((27, 20), (28, 20)),
    ((27, 21), (28, 21)),
    ((27, 23), (28, 23)),
    ((27, 24), (28, 24)),
    ((27, 25), (28, 25)),
    ((27, 28), (27, 27)),
    ((28, 28), (28, 27)),
    ((30, 2), (30, 1)),
    ((32, 2), (32, 1)),
    ((33, 2), (34, 2)),
    ((33, 3), (34, 3)),
    ((33, 5), (34, 5)),
    ((33, 6), (34, 6)),
    ((33, 7), (34, 7)),
    ((33, 9), (34, 9)),
    ((33, 10), (34, 10)),
    ((33, 11), (34, 11)),
    ((33, 13), (34, 13)),
    ((33, 14), (34, 14)),
    ((33, 15), (34, 15)),
    ((33, 17), (34, 17)),
    ((33, 18), (34, 18)),
    ((33, 19), (34, 19)),
    ((33, 21), (34, 21)),
    ((33, 22), (34, 22)),
    ((33, 23), (34, 23)),
    ((33, 25), (34, 25)),
    ((33, 26), (34, 26)),
    ((33, 27), (34, 27)),
    ((33, 29), (34, 29)),
    ((33, 30), (34, 30)),
    ((33, 31), (34, 31)),
    ((33, 33), (34, 33)),
    ((33, 34), (34, 34)),
    ((33, 35), (34, 35)),
    ((33, 37), (34, 37)),
))
WARPS_232 = frozenset([(3, 3), (3, 27), (11, 23), (25, 33)])
MAP_SIZE_232 = (36, 40)  # height, width

# --- test body ---------------------------------------------------------------

DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

GRIDS = {
    82: (WALKABLE_82, MAP_SIZE_82, SPRITES_82, PAIR_BLOCKED_82, WARPS_82),
    232: (WALKABLE_232, MAP_SIZE_232, SPRITES_232, PAIR_BLOCKED_232, WARPS_232),
}

# leg -> (guidance attribute, map, ladder tile, destination map)
LEGS = {
    "first_b1f": ("ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE", 82, (3, 37), 232),
    "middle_1f": ("ROCK_TUNNEL_MIDDLE_1F_ACTION_GUIDANCE", 232, (3, 27), 82),
    "second_b1f": ("ROCK_TUNNEL_SECOND_B1F_ACTION_GUIDANCE", 82, (11, 17), 232),
    "final_1f": ("ROCK_TUNNEL_FINAL_1F_ACTION_GUIDANCE", 232, (3, 3), 82),
}


def walkable(map_id, y, x):
    bits, (height, width), _, _, _ = GRIDS[map_id]
    if not (0 <= y < height and 0 <= x < width):
        return False
    index = y * width + x
    return bool(bytes.fromhex(bits)[index // 8] >> (index % 8) & 1)


def rules_by_tile(guidance):
    out = {}
    for rule in guidance:
        out.setdefault(tuple(rule["target"]), rule)
    return out


class RockTunnelLegTests(unittest.TestCase):
    """Every Rock Tunnel leg is a flow field over real Gen 1 cave physics.

    The 2026-08-19 stall: the first-B1F route was traced with a plain collision
    BFS, which does not know about tile-pair collisions. (7,31)->(8,31) is two
    walkable tiles the game refuses to let you cross, so 96/96 workers pressed
    DOWN into it 882 times per episode and every phase-98 episode ended in
    progressless_stall. These tests replay each leg against collision, sprite,
    warp and tile-pair data lifted from the ROM.
    """

    def test_phases_are_named_as_expected(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[train.FULLGAME_ROCK_TUNNEL_FIRST_B1F_QUEST_PHASE],
                         "reach_rock_tunnel_first_b1f")
        self.assertEqual(names[train.FULLGAME_ROCK_TUNNEL_MIDDLE_1F_QUEST_PHASE],
                         "reach_rock_tunnel_middle_1f")
        self.assertEqual(names[train.FULLGAME_ROCK_TUNNEL_SECOND_B1F_QUEST_PHASE],
                         "reach_rock_tunnel_second_b1f")
        self.assertEqual(names[train.FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE],
                         "reach_rock_tunnel_final_1f")

    def test_the_live_stall_tile_no_longer_forces_down(self):
        """(7,31) DOWN is the elevation change the game refuses."""
        tiles = rules_by_tile(train.ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE)
        self.assertIn((7, 31), tiles)
        self.assertNotEqual(int(tiles[(7, 31)]["action"]), 1)
        self.assertEqual(int(tiles[(7, 31)]["action"]), 2)  # west, the long way round
        self.assertIn(((7, 31), (8, 31)), PAIR_BLOCKED_82)

    def test_no_forced_action_crosses_impassable_terrain(self):
        for leg, (attr, map_id, ladder, _) in LEGS.items():
            _, _, sprites, blocked, _ = GRIDS[map_id]
            for tile, rule in rules_by_tile(getattr(train, attr)).items():
                action = int(rule["action"])
                self.assertIn(action, DELTA, msg="%s %s" % (leg, tile))
                dy, dx = DELTA[action]
                nxt = (tile[0] + dy, tile[1] + dx)
                context = "%s: %s %s -> %s" % (leg, tile, NAMES[action], nxt)
                self.assertTrue(walkable(map_id, *tile), msg=context + " (source is a wall)")
                self.assertTrue(walkable(map_id, *nxt), msg=context + " (wall)")
                self.assertNotIn(nxt, sprites, msg=context + " (trainer sprite)")
                self.assertNotIn((tile, nxt), blocked, msg=context + " (tile-pair collision)")

    def test_second_b1f_does_not_walk_through_1f_hikers(self):
        """RIGHT through (7,5) DOWN vision was a free Ground fight every episode."""
        vision = (
            {(5 + i, 7) for i in range(1, 5)}
            | {(16 + i, 5) for i in range(1, 5)}
            | {(15, 17 - i) for i in range(1, 4)}
        )
        tiles = rules_by_tile(train.ROCK_TUNNEL_SECOND_B1F_ACTION_GUIDANCE)
        for tile, rule in tiles.items():
            dy, dx = DELTA[int(rule["action"])]
            nxt = (tile[0] + dy, tile[1] + dx)
            self.assertNotIn(tile, vision, msg="owns vision %s" % (tile,))
            self.assertNotIn(nxt, vision, msg="%s steps onto %s" % (tile, nxt))

    def test_no_leg_routes_through_a_foreign_warp(self):
        """Stepping on any other warp teleports the worker off the leg."""
        for leg, (attr, map_id, ladder, _) in LEGS.items():
            _, _, _, _, warps = GRIDS[map_id]
            for tile, rule in rules_by_tile(getattr(train, attr)).items():
                dy, dx = DELTA[int(rule["action"])]
                nxt = (tile[0] + dy, tile[1] + dx)
                if nxt == ladder:
                    continue
                self.assertNotIn(nxt, warps,
                                 msg="%s: %s steps onto warp %s" % (leg, tile, nxt))

    def test_every_owned_tile_walks_to_its_ladder(self):
        for leg, (attr, map_id, ladder, dest) in LEGS.items():
            tiles = rules_by_tile(getattr(train, attr))
            for start in tiles:
                seen = set()
                cur = start
                for _ in range(len(tiles) + 4):
                    if cur == ladder:
                        break
                    self.assertIn(cur, tiles, msg="%s: %s left the flow field" % (leg, start))
                    self.assertNotIn(cur, seen, msg="%s: %s loops at %s" % (leg, start, cur))
                    seen.add(cur)
                    dy, dx = DELTA[int(tiles[cur]["action"])]
                    cur = (cur[0] + dy, cur[1] + dx)
                self.assertEqual(cur, ladder,
                                 msg="%s: %s never reaches the ladder" % (leg, start))

    def test_each_ladder_carries_its_transition(self):
        for leg, (attr, map_id, ladder, dest) in LEGS.items():
            rule = rules_by_tile(getattr(train, attr))[ladder]
            self.assertEqual(rule["destination_map"], dest, msg=leg)
            self.assertEqual(int(rule["wait_ticks"]), 256, msg=leg)
            self.assertTrue(rule["force_action"], msg=leg)
            self.assertTrue(rule["verified_route"], msg=leg)

    def test_every_leg_is_registered_with_its_own_phase_and_a_recovery_floor(self):
        """Each phase must own both floors: its own, and the way back onto it."""
        source = Path(train.__file__).read_text(encoding="utf-8")
        expected = {
            "FULLGAME_ROCK_TUNNEL_MIDDLE_1F_QUEST_PHASE": (
                "ROCK_TUNNEL_MIDDLE_1F_ACTION_GUIDANCE",
                "ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE",
            ),
            "FULLGAME_ROCK_TUNNEL_SECOND_B1F_QUEST_PHASE": (
                "ROCK_TUNNEL_SECOND_B1F_ACTION_GUIDANCE",
                "ROCK_TUNNEL_MIDDLE_1F_ACTION_GUIDANCE",
            ),
            "FULLGAME_ROCK_TUNNEL_FINAL_1F_QUEST_PHASE": (
                "ROCK_TUNNEL_FINAL_1F_ACTION_GUIDANCE",
                "ROCK_TUNNEL_SECOND_B1F_ACTION_GUIDANCE",
            ),
        }
        for phase_const, guidance_names in expected.items():
            index = source.index(phase_const + ",\n")
            block = source[index:index + 900]
            for name in guidance_names:
                self.assertIn(name, block,
                              msg="%s is not registered for %s" % (name, phase_const))

    def test_recovery_floors_never_collide_with_the_leg_they_recover(self):
        """Same-phase lists must be on different maps, or first-match-wins bites."""
        pairs = (
            ("ROCK_TUNNEL_MIDDLE_1F_ACTION_GUIDANCE", "ROCK_TUNNEL_FIRST_B1F_ACTION_GUIDANCE"),
            ("ROCK_TUNNEL_SECOND_B1F_ACTION_GUIDANCE", "ROCK_TUNNEL_MIDDLE_1F_ACTION_GUIDANCE"),
            ("ROCK_TUNNEL_FINAL_1F_ACTION_GUIDANCE", "ROCK_TUNNEL_SECOND_B1F_ACTION_GUIDANCE"),
        )
        for leg_name, recovery_name in pairs:
            leg_maps = {r["map"] for r in getattr(train, leg_name)}
            recovery_maps = {r["map"] for r in getattr(train, recovery_name)}
            self.assertEqual(len(leg_maps), 1)
            self.assertFalse(leg_maps & recovery_maps,
                             msg="%s and %s share a map" % (leg_name, recovery_name))

    def test_rock_tunnel_never_promotes_electric_in_trainer_battles(self):
        """Ground-heavy floor: Thunderbolt deals nothing to 4 of 7 trainers.

        Rock Tunnel 1F was the original (and for a while only) map on the
        allow-Electric list. A later widening for the Tower/Silph/Saffron arc
        left it in place while its own note said Rock Tunnel stayed banned, so
        the assisted lead kept picking Thunderbolt against Geodude, Onix,
        Graveler and Cubone.
        """
        source = Path(train.__file__).read_text(encoding="utf-8")
        start = source.index("allow_electric = bool(")
        block = source[start:start + 1400]
        self.assertNotIn("int(prev_map_id) == 82", block)
        self.assertNotIn("FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE", block)
        # The rest of the allowlist must survive: those opponents are not Ground.
        self.assertIn("electric_safe_maps", block)
        self.assertIn("FULLGAME_ELITE_FOUR_MAPS", block)

    def test_ground_safe_promotion_still_defaults_to_banning_electric(self):
        import inspect
        signature = inspect.signature(train.PokemonYellowEnv._ensure_brock_attack_ready)
        self.assertIs(signature.parameters["allow_electric"].default, False)

    def test_interior_phases_are_exactly_the_legs_fought_inside_the_cave(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        inside = sorted(train.FULLGAME_ROCK_TUNNEL_INTERIOR_PHASES)
        self.assertEqual([names[p] for p in inside], [
            "reach_rock_tunnel_first_b1f",
            "reach_rock_tunnel_middle_1f",
            "reach_rock_tunnel_second_b1f",
            "reach_rock_tunnel_final_1f",
            "exit_rock_tunnel_to_lower_route_10",
        ])
        # enter_rock_tunnel is still outside: that worker is stood on Route 10
        # with the Center two dozen steps away, so the ordinary floor applies.
        self.assertNotIn(train.FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE, inside)

    def test_the_hp_floors_stand_down_only_inside_the_tunnel(self):
        """2,016 wipes and two publishes: every ladder crossing was refused."""
        for phase in train.FULLGAME_ROCK_TUNNEL_INTERIOR_PHASES:
            self.assertEqual(train.swarm_required_lead_hp_fraction(0.40, phase), 0.0)
            self.assertEqual(train.swarm_required_party_hp_fraction(0.40, phase), 0.0)
            self.assertTrue(
                train.should_restore_rock_tunnel_traverse_frontier_resources(phase))
        outside = (
            train.FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE,
            train.FULLGAME_LAVENDER_QUEST_PHASE,
        )
        for phase in outside:
            self.assertEqual(train.swarm_required_lead_hp_fraction(0.40, phase), 0.40)
            self.assertEqual(train.swarm_required_party_hp_fraction(0.40, phase), 0.40)
            self.assertFalse(
                train.should_restore_rock_tunnel_traverse_frontier_resources(phase))

    def test_the_traverse_keeps_its_directional_movement_proof(self):
        """The grind predicate also waives the direction check; these are one-way."""
        for phase in train.FULLGAME_ROCK_TUNNEL_INTERIOR_PHASES:
            grind = train.should_restore_rock_tunnel_grind_frontier_resources(
                phase, phase - 1, 0)
            self.assertFalse(grind, "phase %d would waive the movement proof" % phase)
            self.assertEqual(
                train.swarm_frontier_required_validation_direction("up", grind), "up")

    def test_only_the_saved_snapshot_is_healed(self):
        """The live worker must not get a free mid-cave heal."""
        source = Path(train.__file__).read_text(encoding="utf-8")
        block = source[source.index("restore_resources = ("):]
        block = block[:block.index("state_sha256 = file_sha256")]
        self.assertIn("if restore_resources:\n                            _heal_party_memory", block)
        # The finally-clause that puts the live bytes back must still be there.
        self.assertIn("for address, value in resource_snapshot.items():", block)

    def test_a_fainted_party_is_still_never_promoted(self):
        source = Path(train.__file__).read_text(encoding="utf-8")
        self.assertIn("_blockers.append('party_fainted')", source)

    def test_the_tunnel_text_box_latch_cannot_veto_a_candidate(self):
        """CD6B=1 is Rock Tunnel's normal overworld, not a dialogue.

        _clear_rock_tunnel_overworld_joy_ignore zeroes it, but only on 1F and
        only for the first-B1F phase, so every later leg reported `text_box`
        forever: 96/96 workers read CD6B=1 in 30 consecutive overworld samples
        and the run produced zero FRONTIER_WROTE lines.
        """
        source = Path(train.__file__).read_text(encoding="utf-8")
        gate = source[source.index("_blockers.append('text_box')") - 2000:]
        gate = gate[:gate.index("if total_party_hp < 1:")]
        self.assertIn("FULLGAME_ROCK_TUNNEL_INTERIOR_PHASES", gate)
        self.assertIn("(82, 232)", gate)
        # The movement probe is what actually protects against a frozen
        # candidate, so it must still guard the write.
        self.assertIn("frontier_candidate_accepts_movement", source)

    def test_the_clear_hook_covers_both_tunnel_floors_during_interior(self):
        import inspect
        src = inspect.getsource(
            train.PokemonYellowEnv._clear_rock_tunnel_overworld_joy_ignore)
        self.assertIn("FULLGAME_ROCK_TUNNEL_INTERIOR_PHASES", src)
        self.assertIn("(82, 232)", src)
        self.assertIn("ADDR_TEXT_BOX] = 0", src)


if __name__ == "__main__":
    unittest.main()
