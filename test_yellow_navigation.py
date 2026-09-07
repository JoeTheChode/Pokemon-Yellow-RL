import copy
import io
import unittest
import zipfile

import sync_navigation_data
import yellow_navigation


class NavigationCatalogTests(unittest.TestCase):
    def test_catalog_is_valid_and_uses_authoritative_map_identity(self):
        catalog = yellow_navigation.load_catalog()
        yellow_navigation.validate_catalog(catalog)
        self.assertEqual(yellow_navigation.map_record(54)["constant"], "PEWTER_GYM")
        self.assertEqual(yellow_navigation.map_record(13)["constant"], "ROUTE_2")

    def test_internal_map_resolves_to_completion_location(self):
        self.assertEqual(yellow_navigation.completion_location_for_map(51), "Viridian Forest")
        self.assertEqual(yellow_navigation.completion_location_for_map(59), "Mt. Moon")
        self.assertEqual(yellow_navigation.completion_location_for_map(40), "Pallet Town")

    def test_macro_route_reaches_cerulean(self):
        route = yellow_navigation.macro_route("Pallet Town", "Cerulean City")
        self.assertEqual(route[0], "Pallet Town")
        self.assertEqual(route[-1], "Cerulean City")
        self.assertIn("Mt. Moon", route)

    def test_catalog_contains_rom_connections_and_warps(self):
        pallet = yellow_navigation.map_record(0)
        self.assertIn(12, [edge["destination_map_id"] for edge in pallet["connections"]])
        oak_warp = next(warp for warp in pallet["warps"] if warp["destination_map_id"] == 40)
        self.assertEqual(oak_warp["position"], [11, 12])

    def test_map_route_uses_rom_topology(self):
        self.assertEqual(yellow_navigation.map_route(0, 40), [0, 40])
        self.assertIn(12, yellow_navigation.map_neighbors(0))

    def test_warp_entry_actions_are_derived_from_rom_coordinates(self):
        self.assertEqual(yellow_navigation.warp_entry_action(15, (6, 18), 59), 0)
        self.assertEqual(yellow_navigation.warp_entry_action(59, (5, 6), 60), 2)
        self.assertEqual(yellow_navigation.warp_entry_action(60, (17, 20), 61), 3)
        self.assertEqual(yellow_navigation.warp_entry_action(61, (7, 4), 60), 3)

    def test_early_game_warps_stay_on_the_proven_pallet_forest_chain(self):
        self.assertEqual(yellow_navigation.map_route(0, 1), [0, 12, 1])
        self.assertEqual(yellow_navigation.map_route(1, 42), [1, 42])
        self.assertEqual(yellow_navigation.map_route(0, 40), [0, 40])
        self.assertEqual(yellow_navigation.warp_entry_action(0, (12, 12), 40), 0)
        self.assertEqual(yellow_navigation.warp_entry_action(1, (20, 29), 42), 0)
        self.assertEqual(yellow_navigation.warp_entry_action(13, (44, 3), 50), 0)

    def test_saved_page_exposes_location_coordinates(self):
        text = sync_navigation_data.DEFAULT_HTML.read_text(encoding="utf-8")
        locations, _ = sync_navigation_data.parse_completion_html(text)
        by_name = {record["name"]: record for record in locations}
        self.assertEqual(by_name["Pallet Town"]["map_pixel"], [470, 1911])
        self.assertIn("Victory Road", by_name)

    def test_repository_parser_keeps_exact_trainer_and_item_coordinates(self):
        maps = [{"id": 61, "constant": "MT_MOON_B2F"}]
        source = b"""\
def_warp_events
warp_event 5, 7, LAST_MAP, 7
def_object_events
object_event 12, 8, SPRITE_SUPER_NERD, STAY, RIGHT, TEXT_NERD, OPP_SUPER_NERD, 2
object_event 25, 21, SPRITE_POKE_BALL, STAY, NONE, TEXT_HP_UP, HP_UP
object_event 12, 6, SPRITE_FOSSIL, STAY, NONE, TEXT_DOME_FOSSIL
def_warps_to MT_MOON_B2F
"""
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("pokeyellow/data/maps/objects/MtMoonB2F.asm", source)
        sync_navigation_data.parse_repository_archive(archive_bytes.getvalue(), maps)
        self.assertEqual(maps[0]["trainers"][0]["position"], [8, 12])
        self.assertEqual(maps[0]["trainers"][0]["opponent"], "OPP_SUPER_NERD")
        self.assertEqual(maps[0]["items"][0]["position"], [21, 25])
        self.assertEqual(maps[0]["items"][0]["item"], "HP_UP")


class EventFlagCatalogTests(unittest.TestCase):
    def test_catalog_loads_and_covers_the_whole_game(self):
        catalog = yellow_navigation.load_event_flags_catalog()
        self.assertEqual(catalog["num_events"], 2560)
        self.assertGreater(len(catalog["events"]), 500)

    def test_known_story_beats_resolve_to_the_right_bit(self):
        self.assertEqual(yellow_navigation.event_flag_name(119), "EVENT_BEAT_BROCK")
        self.assertEqual(yellow_navigation.event_flag_name(191), "EVENT_BEAT_MISTY")

    def test_unnamed_bit_falls_back_gracefully(self):
        self.assertEqual(yellow_navigation.event_flag_name(2559), "EVENT_UNNAMED_2559")

    def test_out_of_range_bit_is_rejected(self):
        with self.assertRaises(ValueError):
            yellow_navigation.event_flag_name(-1)
        with self.assertRaises(ValueError):
            yellow_navigation.event_flag_name(2560)

    def test_catalog_rejects_duplicate_or_inconsistent_bits(self):
        catalog = copy.deepcopy(yellow_navigation.load_event_flags_catalog())
        catalog["events"][1]["bit"] = catalog["events"][0]["bit"]
        with self.assertRaisesRegex(ValueError, "duplicate bit"):
            yellow_navigation.validate_event_flags_catalog(catalog)

        catalog = copy.deepcopy(yellow_navigation.load_event_flags_catalog())
        catalog["events"][0]["byte_offset"] += 1
        with self.assertRaisesRegex(ValueError, "inconsistent byte/bit"):
            yellow_navigation.validate_event_flags_catalog(catalog)

    def test_named_event_bits_set_matches_byte_layout_from_flag_action(self):
        # bit 119 -> byte_offset 14, bit_in_byte 7 (LSB=bit0), per
        # engine/flag_action.asm's FlagAction routine.
        flag_bytes = bytearray(320)
        flag_bytes[14] |= 1 << 7
        self.assertEqual(
            yellow_navigation.named_event_bits_set(bytes(flag_bytes)), {119}
        )

    def test_sync_event_flags_parser_handles_const_directives(self):
        import sync_event_flags

        sample = (
            "; Sample section\n"
            "\tconst_def\n"
            "\tconst EVENT_A\n"
            "\tconst_skip 2\n"
            "\tconst EVENT_B\n"
            "\tconst_next $10\n"
            "\tconst EVENT_C\n"
        )
        events = sync_event_flags.parse_event_constants(sample)
        self.assertEqual(events, [
            {"bit": 0, "byte_offset": 0, "bit_in_byte": 0, "name": "EVENT_A", "group": "Sample section"},
            {"bit": 3, "byte_offset": 0, "bit_in_byte": 3, "name": "EVENT_B", "group": "Sample section"},
            {"bit": 16, "byte_offset": 2, "bit_in_byte": 0, "name": "EVENT_C", "group": "Sample section"},
        ])

    def test_sync_event_flags_parser_handles_const_next_arithmetic(self):
        import sync_event_flags

        self.assertEqual(sync_event_flags._parse_value("$F0 - 2"), 0xF0 - 2)
        self.assertEqual(sync_event_flags._parse_value("$28"), 0x28)

    def test_sync_event_flags_parser_derives_num_events(self):
        import sync_event_flags

        _, num_events = sync_event_flags._parse_event_constants(
            "const_def $20\nconst EVENT_A\nconst_skip 3\nDEF NUM_EVENTS EQU const_value\n"
        )
        self.assertEqual(num_events, 0x24)


if __name__ == "__main__":
    unittest.main()
