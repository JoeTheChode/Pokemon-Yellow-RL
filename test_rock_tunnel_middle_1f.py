import collections
import unittest
from unittest import mock

import train


class RockTunnelMiddle1FRecoveryTests(unittest.TestCase):
    def test_trainer5_engagement_allows_electric_only_at_exact_tile(self):
        phase = train.FULLGAME_ROCK_TUNNEL_MIDDLE_1F_QUEST_PHASE

        self.assertTrue(train.trainer_battle_allows_electric(232, phase, 28, 17))
        self.assertFalse(train.trainer_battle_allows_electric(232, phase, 28, 18))
        self.assertFalse(train.trainer_battle_allows_electric(232, phase, 27, 17))
        self.assertFalse(train.trainer_battle_allows_electric(232, phase))
        self.assertFalse(train.trainer_battle_allows_electric(82, phase, 28, 17))

    def make_env(self, beaten):
        env = object.__new__(train.PokemonYellowEnv)
        env.quest_phase = train.FULLGAME_ROCK_TUNNEL_MIDDLE_1F_QUEST_PHASE
        env._rock_tunnel_middle_hiker_emergency_heals_used = 0
        env._event_flag_is_set = lambda bit: int(bit) in set(beaten)
        memory = collections.defaultdict(int, {
            train.ADDR_MAP_ID: 232,
            train.ADDR_POS_A: 13,
            train.ADDR_POS_B: 30,
            train.ADDR_BATTLE_FLAG: 2,
            train.ADDR_ACTIVE_MON_CUR_HP_LO: 54,
            train.ADDR_ACTIVE_MON_MAX_HP_LO: 108,
            train.PARTY_CUR_HP_ADDRS[0][1]: 54,
            train.ADDR_ACTIVE_MON_STATUS: 8,
            train.PARTY_STATUS_ADDRS[0]: 8,
        })
        env.pyboy = mock.Mock(memory=memory)
        return env, memory

    def test_recovery_requires_both_prior_trainers_and_applies_once(self):
        prior = {
            train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_7,
            train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_5,
        }
        env, memory = self.make_env(prior)

        self.assertTrue(env._use_rock_tunnel_middle_hiker_emergency_heal())
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_CUR_HP_LO], 108)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 108)
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 54
        self.assertFalse(env._use_rock_tunnel_middle_hiker_emergency_heal())

    def test_recovery_does_not_apply_before_prerequisites_or_after_win(self):
        env, _ = self.make_env({train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_7})
        self.assertFalse(env._use_rock_tunnel_middle_hiker_emergency_heal())

        env, _ = self.make_env({
            train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_7,
            train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_5,
            train.EVENT_BEAT_ROCK_TUNNEL_2_TRAINER_4,
        })
        self.assertFalse(env._use_rock_tunnel_middle_hiker_emergency_heal())


if __name__ == '__main__':
    unittest.main()
