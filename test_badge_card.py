#!/usr/bin/env python3
"""Player-card badge consensus: do not light 4/5/6 from garbage byte 57."""
import unittest

import map_viewer_server as mvs


def agent(**kwargs):
    entry = {
        'env_id': 0,
        'last_position': [20, 12, 5],
        'map_id': 5,
        'pikachu_level': 24,
        'badge_flags': 7,
        'player_name': 'SVER',
        'money': 13366,
        'play_time_seconds': 4000,
        'play_time_maxed': False,
    }
    entry.update(kwargs)
    return entry


class BadgeCardTests(unittest.TestCase):
    def setUp(self):
        self.feed = mvs.AgentFeed()

    def payload(self, agents):
        self.feed.entries = {i: a for i, a in enumerate(agents)}
        return self.feed._player_card_payload()

    def test_prefix_bytes_are_plausible(self):
        for flags in (0, 1, 3, 7, 15):
            self.assertTrue(
                mvs.AgentFeed._badge_read_is_plausible(agent(badge_flags=flags))
            )

    def test_out_of_order_late_gyms_are_plausible(self):
        # Gyms 4-8 can be cleared in any order. 47 = Boulder/Cascade/Thunder
        # + Rainbow + Marsh: this run beats Sabrina before Koga, and the live
        # card showed 3/8 for it because the old filter demanded a prefix.
        for flags in (0b00101111, 0b00011111, 0b01001111, 0b11101111, 255):
            self.assertTrue(
                mvs.AgentFeed._badge_read_is_plausible(agent(badge_flags=flags)),
                flags,
            )

    def test_late_badge_without_the_early_three_is_not_plausible(self):
        # Rainbow/Soul/Marsh cannot be earned before Cascade and Thunder.
        for flags in (0b00001000, 0b00111001, 0b10000011, 0b01000101):
            self.assertFalse(
                mvs.AgentFeed._badge_read_is_plausible(agent(badge_flags=flags)),
                flags,
            )

    def test_unanimous_out_of_order_byte_wins_the_card(self):
        # The live 2026-08-23 state: all 96 workers read 47.
        agents = [agent(badge_flags=47, money=56529, play_time_seconds=921599)
                  for _ in range(96)]
        card = self.payload(agents)
        self.assertEqual(card['badge_flags'], 47)
        self.assertEqual(card['badge_count'], 5)
        self.assertEqual(card['badge_source'], 'workers')
        self.assertEqual(card['badge_agreement'], 1.0)
        self.assertEqual(
            [b['name'] for b in card['badges'] if b['earned']],
            ['Boulder', 'Cascade', 'Thunder', 'Rainbow', 'Marsh'],
        )
        # Money and play time ride the same plausible pool, so rejecting the
        # badge byte zeroed them too.
        self.assertEqual(card['money'], 56529)
        self.assertEqual(card['play_time_seconds'], 921599)

    def test_garbage_57_is_not_plausible(self):
        self.assertFalse(
            mvs.AgentFeed._badge_read_is_plausible(
                agent(
                    badge_flags=57,
                    map_id=57,
                    last_position=[57, 0, 57],
                    player_name='',
                    pikachu_level=57,
                )
            )
        )
        # Even on a normal map, 57 is not a gym-prefix byte.
        self.assertFalse(
            mvs.AgentFeed._badge_read_is_plausible(agent(badge_flags=57))
        )

    def test_majority_garbage_does_not_hide_cascade_thunder(self):
        agents = [
            agent(
                badge_flags=57,
                map_id=57,
                last_position=[57, 0, 57],
                player_name='',
                pikachu_level=57,
                play_time_seconds=57,
                play_time_maxed=True,
            )
            for _ in range(95)
        ]
        agents.append(agent(badge_flags=7, map_id=66, last_position=[4, 2, 66]))
        card = self.payload(agents)
        self.assertEqual(card['badge_flags'], 7)
        self.assertEqual(card['badge_count'], 3)
        earned = [b['name'] for b in card['badges'] if b['earned']]
        self.assertEqual(earned, ['Boulder', 'Cascade', 'Thunder'])
        self.assertNotIn('Rainbow', earned)
        self.assertNotIn('Soul', earned)
        self.assertNotIn('Marsh', earned)

    def test_minority_garbage_still_loses_to_real_prefix(self):
        agents = [agent(badge_flags=3) for _ in range(8)]
        agents.extend(agent(badge_flags=57, player_name='') for _ in range(4))
        card = self.payload(agents)
        self.assertEqual(card['badge_flags'], 3)
        self.assertEqual(
            [b['name'] for b in card['badges'] if b['earned']],
            ['Boulder', 'Cascade'],
        )

    def test_tie_prefers_prefix_over_57(self):
        agents = [agent(badge_flags=3) for _ in range(6)]
        agents.extend(agent(badge_flags=57, player_name='') for _ in range(6))
        card = self.payload(agents)
        self.assertEqual(card['badge_flags'], 3)

    def test_all_garbage_infers_thunder_from_quest_phase(self):
        agents = [
            agent(
                badge_flags=57,
                map_id=57,
                last_position=[57, 0, 57],
                player_name='',
                pikachu_level=57,
                play_time_seconds=57,
                play_time_maxed=True,
                quest_phase=91,
                quest_next='return_to_route_24_for_charmander',
            )
            for _ in range(96)
        ]
        card = self.payload(agents)
        self.assertEqual(card['badge_flags'], 7)
        self.assertEqual(
            [b['name'] for b in card['badges'] if b['earned']],
            ['Boulder', 'Cascade', 'Thunder'],
        )
        self.assertEqual(card['play_time_seconds'], 0)
        self.assertEqual(card['badge_source'], 'quest_chain')

    def test_all_garbage_infers_out_of_order_badges_from_quest_phase(self):
        # Past `beat_sabrina` but before `beat_koga`: Marsh is earned and Soul
        # is not, which a prefix cannot express.
        names = mvs.load_quest_waypoint_names()
        if 'beat_sabrina' not in names or 'beat_koga' not in names:
            self.skipTest('quest waypoint names unavailable')
        phase = names.index('beat_sabrina') + 1
        self.assertLess(phase, names.index('beat_koga'))
        agents = [
            agent(
                badge_flags=57,
                map_id=57,
                last_position=[57, 0, 57],
                player_name='',
                quest_phase=phase,
            )
            for _ in range(96)
        ]
        card = self.payload(agents)
        self.assertEqual(card['badge_source'], 'quest_chain')
        self.assertEqual(
            [b['name'] for b in card['badges'] if b['earned']],
            ['Boulder', 'Cascade', 'Thunder', 'Rainbow', 'Marsh'],
        )


if __name__ == '__main__':
    unittest.main()
