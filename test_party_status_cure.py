#!/usr/bin/env python3
"""Regression tests for the party status cure on serialized frontier snapshots.

2026-08-24: the soul-marsh grind frontier was published at 131/131 HP with
Pikachu still poisoned. `should_restore_rock_tunnel_grind_frontier_resources`
heals the snapshot before serialization, but the heal only rewrote HP and PP --
never Gen 1's status bitfield -- so every one of 96 workers loaded a lead on a
524-step poison clock and the frontier sat frozen for 19.8 hours.
"""
import unittest

import train


class Memory(dict):
    def __getitem__(self, key):
        return self.get(key, 0)


class PartyStatusAddressTest(unittest.TestCase):
    def test_addresses_match_pokeyellow_sym(self):
        # Yellow is not Red/Blue here; these come from pokeyellow.sym.
        self.assertEqual(train.FIRST_PARTY_STATUS_ADDR, 0xD16E)
        self.assertEqual(train.PARTY_STATUS_ADDRS[0], 0xD16E)
        self.assertEqual(train.PARTY_STATUS_ADDRS[1], 0xD19A)
        self.assertEqual(train.ADDR_ACTIVE_MON_STATUS, 0xD017)

    def test_table_covers_every_party_slot_on_the_party_stride(self):
        self.assertEqual(
            len(train.PARTY_STATUS_ADDRS), len(train.PARTY_CUR_HP_ADDRS)
        )
        for idx, addr in enumerate(train.PARTY_STATUS_ADDRS):
            self.assertEqual(
                addr, 0xD16E + idx * train.PARTY_DATA_STRIDE
            )
        # The status byte sits between species and the HP pair of the same
        # slot, so a stride slip would collide with a neighbouring field.
        for idx, addr in enumerate(train.PARTY_STATUS_ADDRS):
            self.assertGreater(addr, train.PARTY_SPECIES_ADDRS[idx])
            self.assertLess(addr, train.PARTY_MAX_HP_ADDRS[idx][0])


class CurePartyStatusTest(unittest.TestCase):
    def test_clears_every_occupied_slot_and_the_battle_copy(self):
        memory = Memory({
            train.ADDR_PARTY_SIZE: 3,
            train.PARTY_STATUS_ADDRS[0]: 0x08,  # poison
            train.PARTY_STATUS_ADDRS[1]: 0x40,  # paralysis
            train.PARTY_STATUS_ADDRS[2]: 0x03,  # sleep, 3 turns
            train.ADDR_ACTIVE_MON_STATUS: 0x08,
        })
        train._cure_party_status_memory(memory)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[1]], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[2]], 0)
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0)

    def test_does_not_touch_slots_past_party_size(self):
        memory = Memory({
            train.ADDR_PARTY_SIZE: 2,
            train.PARTY_STATUS_ADDRS[0]: 0x08,
            train.PARTY_STATUS_ADDRS[1]: 0x08,
            train.PARTY_STATUS_ADDRS[4]: 0x77,  # stale bytes past the party
        })
        train._cure_party_status_memory(memory)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[1]], 0)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[4]], 0x77)

    def test_empty_party_is_a_no_op(self):
        memory = Memory({
            train.ADDR_PARTY_SIZE: 0,
            train.ADDR_ACTIVE_MON_STATUS: 0x08,
        })
        train._cure_party_status_memory(memory)
        # Nothing to cure, and the battle copy is meaningless without a party.
        self.assertEqual(memory[train.ADDR_ACTIVE_MON_STATUS], 0x08)

    def test_oversized_party_size_is_clamped(self):
        memory = Memory({train.ADDR_PARTY_SIZE: 200})
        train._cure_party_status_memory(memory)  # must not IndexError

    def test_heal_party_memory_alone_leaves_status_set(self):
        """The hole this fix closes -- keep it documented as behaviour."""
        memory = Memory({
            train.ADDR_PARTY_SIZE: 1,
            train.PARTY_CUR_HP_ADDRS[0][1]: 1,
            train.PARTY_MAX_HP_ADDRS[0][1]: 131,
            train.PARTY_STATUS_ADDRS[0]: 0x08,
        })
        train._heal_party_memory(memory)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 131)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0x08)
        train._cure_party_status_memory(memory)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)


class GrindSnapshotContractTest(unittest.TestCase):
    def test_soul_marsh_grind_phases_restore_snapshot_resources(self):
        # Phase 222 train_pikachu_soul_marsh_level_55 is the live stall.
        for phase in (218, 222, 227):
            self.assertIn(phase, train.FULLGAME_RNG_GRIND_QUEST_PHASES)
            self.assertTrue(
                train.should_restore_rock_tunnel_grind_frontier_resources(
                    phase, phase, 0
                ),
                'phase %d must heal its serialized snapshot' % phase,
            )

    def test_grind_snapshot_is_healed_and_cured_together(self):
        """Whatever the snapshot heal restores, the cure must run beside it."""
        memory = Memory({
            train.ADDR_PARTY_SIZE: 2,
            train.PARTY_CUR_HP_ADDRS[0][1]: 12,
            train.PARTY_MAX_HP_ADDRS[0][1]: 131,
            train.PARTY_CUR_HP_ADDRS[1][1]: 0,
            train.PARTY_MAX_HP_ADDRS[1][1]: 30,
            train.PARTY_STATUS_ADDRS[0]: 0x08,
            train.PARTY_MOVE_ID_ADDRS[0][0]: 85,   # Thunderbolt
            train.PARTY_MOVE_PP_ADDRS[0][0]: 0,
        })
        train._heal_party_memory(memory)
        train._restore_party_pp_memory(memory)
        train._cure_party_status_memory(memory)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[0][1]], 131)
        self.assertEqual(memory[train.PARTY_CUR_HP_ADDRS[1][1]], 30)
        self.assertEqual(memory[train.PARTY_STATUS_ADDRS[0]], 0)
        self.assertEqual(
            memory[train.PARTY_MOVE_PP_ADDRS[0][0]],
            train.GEN1_MOVE_TABLE[85]['max_pp'],
        )

    def test_live_worker_status_is_restored_after_serialization(self):
        """The snapshot is cured; the live emulator must keep its poison.

        The publish path snapshots `resource_addresses` before healing and
        writes them back in a `finally`. If the status bytes were cured but
        not listed there, publishing a frontier would silently full-heal the
        running worker -- free progress the rest of the gates assume is
        impossible.
        """
        import inspect
        source = inspect.getsource(train.PokemonYellowEnv)
        self.assertIn('resource_addresses.update(PARTY_STATUS_ADDRS)', source)
        self.assertIn('ADDR_ACTIVE_MON_STATUS,', source)
        self.assertIn(
            '_cure_party_status_memory(self.pyboy.memory)', source
        )


if __name__ == '__main__':
    unittest.main()
