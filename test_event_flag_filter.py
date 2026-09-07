import unittest

import train


class EventFlagLayoutTests(unittest.TestCase):
    def make_env(self, bit):
        env = object.__new__(train.PokemonYellowEnv)
        memory = bytearray(train.ADDR_EVENT_FLAGS_END + 1)
        memory[train.ADDR_EVENT_FLAGS_START + bit // 8] |= 1 << (bit % 8)
        env.pyboy = type("FakePyBoy", (), {"memory": memory})()
        env.quarantined_event_flag_bits = train.QUARANTINED_EVENT_FLAG_BITS
        return env

    def test_exact_rom_symbol_addresses_are_used(self):
        self.assertEqual(train.ADDR_BADGES, 0xD355)
        self.assertEqual(train.ADDR_XP, 0xD178)
        self.assertEqual(train.ADDR_BATTLE_FLAG, 0xD056)
        self.assertEqual(train.ADDR_TEXT_BOX, 0xCD6B)
        self.assertEqual(train.ADDR_EVENT_FLAGS_START, 0xD746)
        self.assertEqual(train.ADDR_EVENT_FLAGS_END, 0xD886)
        self.assertEqual(train.ADDR_REPEL, 0xD0DA)

    def test_parcel_event_uses_d74d_bit1(self):
        parcel = self.make_env(57)
        self.assertEqual(train.ADDR_EVENT_FLAGS_START + 57 // 8, 0xD74D)
        self.assertEqual(parcel._count_event_flag_bits(), 1)
        self.assertIn(57, parcel._current_rewardable_named_event_bits())

    def test_no_real_event_is_quarantined_after_address_fix(self):
        route21_trainer7 = self.make_env(1304)
        route22_rival = self.make_env(1312)
        self.assertEqual(train.QUARANTINED_EVENT_FLAG_BITS, frozenset())
        self.assertEqual(route21_trainer7._count_event_flag_bits(), 1)
        self.assertEqual(route22_rival._count_event_flag_bits(), 1)


if __name__ == "__main__":
    unittest.main()
