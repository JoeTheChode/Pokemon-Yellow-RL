#!/usr/bin/env python3
"""Quest tracker must follow the live phase, not historical split rows."""
import json
import unittest
from unittest import mock

import map_viewer_server as mvs


class QuestProgressTests(unittest.TestCase):
    def test_live_phase_not_ledger_width(self):
        progress = mvs.AgentFeed.quest_progress_counts(98, 297, 250)
        self.assertEqual(progress["live_phase"], 98)
        self.assertEqual(progress["phases_cleared"], 98)
        self.assertEqual(progress["remaining"], 199)
        self.assertEqual(progress["percent_complete"], 33.0)
        self.assertEqual(progress["ledger_rows"], 250)

    def test_no_live_phase_falls_back_to_ledger(self):
        progress = mvs.AgentFeed.quest_progress_counts(None, 297, 12)
        self.assertIsNone(progress["live_phase"])
        self.assertEqual(progress["phases_cleared"], 12)
        self.assertEqual(progress["remaining"], 285)

    def test_payload_prefers_live_agents_over_old_hof_rows(self):
        feed = mvs.AgentFeed()
        feed.quest_total_phases = 297
        feed.quest_splits = {
            phase: {
                "name": f"obj_{phase}",
                "clears": 1,
                "best": 10,
                "last": 10,
                "samples": [10],
                "best_run_step": 10,
            }
            for phase in list(range(0, 98)) + [250, 295]
        }
        feed.entries = {
            i: {
                "quest_phase": 98,
                "quest_next": "reach_rock_tunnel_first_b1f",
                "last_position": [8, 17, 82],
                "last_seen": 1.0,
            }
            for i in range(4)
        }
        payload = json.loads(feed.quest_splits_payload())
        self.assertEqual(payload["live_phase"], 98)
        self.assertEqual(payload["phases_cleared"], 98)
        self.assertEqual(payload["remaining"], 199)
        self.assertEqual(payload["percent_complete"], 33.0)
        self.assertEqual(payload["ledger_rows"], 100)
        self.assertEqual(payload["total_phases"], 297)
        phases = [row["phase"] for row in payload["rows"]]
        self.assertEqual(phases[0], 0)
        self.assertEqual(phases[-1], 98)
        self.assertNotIn(250, phases)
        self.assertNotIn(295, phases)
        self.assertEqual(len(payload["rows"]), 99)
        self.assertTrue(all(row["phase"] <= 98 for row in payload["rows"]))

    def test_display_rows_pad_previous_and_drop_empty_future(self):
        feed = mvs.AgentFeed()
        feed.quest_splits = {
            5: {
                "name": "heal_at_pewter",
                "clears": 12,
                "best": None,
                "last": None,
                "samples": [],
                "best_run_step": 80,
            },
            250: {
                "name": "enter_hall_of_fame",
                "clears": 3,
                "best": None,
                "last": None,
                "samples": [],
                "best_run_step": 9,
            },
        }
        rows = feed._splits_rows_for_display(7)
        phases = [row["phase"] for row in rows]
        self.assertEqual(phases, list(range(0, 8)))
        by_phase = {row["phase"]: row for row in rows}
        self.assertEqual(by_phase[5]["clears"], 12)
        self.assertIsNone(by_phase[5]["best"])
        self.assertNotIn(250, by_phase)

    def test_display_rows_follow_objective_identity_after_phase_insert(self):
        feed = mvs.AgentFeed()
        feed.quest_splits = {
            10: {
                "name": "enter_game_corner", "clears": 4,
                "best": 35, "last": 40, "samples": [35, 40],
                "best_run_step": 35,
            },
        }
        names = [f"obj_{phase}" for phase in range(16)]
        names[15] = "enter_game_corner"
        with mock.patch.object(mvs, 'load_quest_waypoint_names', return_value=names):
            rows = feed._splits_rows_for_display(15)
        by_phase = {row['phase']: row for row in rows}
        self.assertEqual(by_phase[15]['name'], 'enter_game_corner')
        self.assertEqual(by_phase[15]['best'], 35)
        self.assertEqual(by_phase[15]['clears'], 4)

    def test_persisted_splits_remap_before_current_phase_mastery(self):
        feed = mvs.AgentFeed()
        feed.quest_splits = {
            10: {
                "name": "enter_game_corner", "clears": 4,
                "best": 35, "last": 40, "samples": [35, 40],
                "best_run_step": 35,
            },
            15: {
                "name": "old_neighbor", "clears": 2,
                "best": None, "last": None, "samples": [],
                "best_run_step": 70,
            },
        }
        names = [f"obj_{phase}" for phase in range(21)]
        names[15] = "enter_game_corner"
        names[20] = "old_neighbor"
        with mock.patch.object(mvs, 'load_quest_waypoint_names', return_value=names):
            self.assertTrue(feed._remap_quest_splits_by_objective())
        self.assertEqual(feed.quest_splits[15]['name'], 'enter_game_corner')
        self.assertEqual(feed.quest_splits[15]['best'], 35)
        self.assertEqual(feed.quest_splits[20]['name'], 'old_neighbor')

    def test_timed_count_excludes_active_historical_split(self):
        feed = mvs.AgentFeed()
        feed.quest_total_phases = 10
        feed.entries = {
            0: {
                'quest_phase': 2, 'quest_next': 'active',
                'last_position': [1, 1, 1], 'last_seen': 1.0,
            },
        }
        feed.quest_splits = {
            phase: {
                'name': f'obj_{phase}', 'clears': 1, 'best': 10,
                'last': 10, 'samples': [10], 'best_run_step': 10,
            }
            for phase in range(3)
        }
        with mock.patch.object(
            mvs, 'load_quest_waypoint_names',
            return_value=[f'obj_{phase}' for phase in range(10)],
        ):
            payload = json.loads(feed.quest_splits_payload())
        self.assertEqual(payload['phases_cleared'], 2)
        self.assertEqual(payload['timed_phases'], 2)


if __name__ == "__main__":
    unittest.main()
