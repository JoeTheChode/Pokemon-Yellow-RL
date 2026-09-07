"""Coast boundary, action-ownership, and curriculum regressions."""
import ast
from pathlib import Path
import unittest
from unittest import mock
import train


class CoastCheckpointRouteTests(unittest.TestCase):
    def test_production_arrival_repels_wilds_but_preserves_coast_trainers(self):
        # Read the production configuration expression, rather than assume
        # global force_repel=True (production actually leaves it False).
        tree = ast.parse(Path(train.__file__).read_text())
        milestone = next(n.value for n in ast.walk(tree)
                         if isinstance(n, ast.Assign) and any(
                             isinstance(t, ast.Name) and t.id == 'fullgame_milestone'
                             for t in n.targets))
        phase_expr = next(v for k, v in zip(milestone.keys, milestone.values)
                          if isinstance(k, ast.Constant)
                          and k.value == 'force_repel_quest_phases')
        phases = eval(compile(ast.Expression(phase_expr), '<production-repel>', 'eval'),
                      vars(train))
        self.assertIn(train.FULLGAME_FUCHSIA_QUEST_PHASE, phases)
        self.assertTrue(set(phases).isdisjoint(train.FULLGAME_SOUL_MARSH_GRIND_PHASES))
        env = object.__new__(train.PokemonYellowEnv)
        env.pyboy = mock.Mock(memory=bytearray(0x10000))
        env.quest_phase = train.FULLGAME_FUCHSIA_QUEST_PHASE
        env.force_repel = False
        env.force_repel_quest_phases = frozenset(phases)
        env.force_repel_min_lead_levels = {}
        env.wild_battle_quest_phases = train.FULLGAME_SOUL_MARSH_GRIND_PHASES
        env.suppress_stale_battle_reassertion = False
        env.pyboy.memory[train.ADDR_LEVEL] = 50
        for map_id in (23, 24, 25, 26):
            with self.subTest(map_id=map_id):
                memory = env.pyboy.memory
                memory[train.ADDR_MAP_ID] = map_id
                memory[train.ADDR_BATTLE_FLAG] = 2
                memory[train.ADDR_GRASS_RATE] = 25
                memory[train.ADDR_WATER_RATE] = 25
                self.assertTrue(env._maintain_force_repel_memory())
                self.assertEqual(memory[train.ADDR_GRASS_RATE], 0)
                self.assertEqual(memory[train.ADDR_WATER_RATE], 0)
                self.assertEqual(memory[train.ADDR_BATTLE_FLAG], 2)
                self.assertEqual(env._effective_battle_flag(), 2)
        env.quest_phase = min(train.FULLGAME_SOUL_MARSH_GRIND_PHASES)
        env.pyboy.memory[train.ADDR_MAP_ID] = 26
        env.pyboy.memory[train.ADDR_BATTLE_FLAG] = 1
        env.pyboy.memory[train.ADDR_GRASS_RATE] = 25
        self.assertFalse(env._maintain_force_repel_memory())
        self.assertEqual(env.pyboy.memory[train.ADDR_GRASS_RATE], 25)
        self.assertEqual(env._effective_battle_flag(), 1)

    def test_checkpoint_route_is_continuous_across_actual_map_connections(self):
        # These transition coordinates come from the exact checkpoint replay.
        transitions = {
            (23, 107, 10, 1): (24, 0, 50),
            (24, 10, 0, 2): (25, 10, 19),
            (25, 44, 0, 2): (26, 8, 59),
            (26, 8, 15, 2): (184, 4, 7),
            (184, 4, 1, 2): (26, 8, 7),
            (26, 8, 0, 2): (7, 16, 39),
        }
        route = {}
        for m, y, x, action, destination in train.NO_BIKE_COAST_TO_FUCHSIA_STEPS:
            key = m, y, x
            self.assertTrue(key not in route or route[key] == (action, destination), key)
            route[key] = action, destination
        position = 23, 61, 10
        visited, crossed = set(), set()
        while position != (7, 16, 39):
            self.assertNotIn(position, visited, 'route loops')
            visited.add(position)
            self.assertLess(len(visited), 400)
            self.assertIn(position, route, 'missing connector')
            action, destination = route[position]
            edge = (*position, action)
            if edge in transitions:
                position = transitions[edge]
                self.assertEqual(destination, position[0])
                crossed.add(edge)
            else:
                self.assertIsNone(destination, 'destination flag occurs before the warp')
                dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
                position = position[0], position[1] + dy, position[2] + dx
        self.assertEqual(crossed, set(transitions))

    def test_late_route_override_yields_on_every_battle_tile(self):
        for m, y, x in train.COAST_TRANSIT_ACTIONS:
            for battle in (1, 2):
                self.assertIsNone(train.route13_water_pocket_action(
                    m, y, x, train.FULLGAME_FUCHSIA_QUEST_PHASE,
                    2048, battle_flag=battle), (m, y, x, battle))

    def test_route15_arrival_override_does_not_capture_grind_workers(self):
        for phase in train.FULLGAME_SOUL_MARSH_GRIND_PHASES:
            for position in ((26, 8, 19), (26, 8, 18), (184, 4, 7)):
                self.assertIsNone(train.route13_water_pocket_action(*position, phase, 200))
        self.assertEqual(train.route13_water_pocket_action(
            26, 8, 16, train.FULLGAME_FUCHSIA_QUEST_PHASE), 'left')

    def test_gate_reverse_is_independent_of_coastal_approach_length(self):
        steps = train.ROUTE15_GATE_TO_FUCHSIA_STEPS
        self.assertEqual(steps[0], (184, 4, 7, 2, None))
        self.assertEqual(steps[-1], (26, 8, 0, 2, 7))
        self.assertTrue(all(m == 184 or (m == 26 and y == 8 and x <= 7)
                            for m, y, x, _, _ in steps))


if __name__ == '__main__':
    unittest.main()

