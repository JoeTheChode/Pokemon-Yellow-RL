import unittest

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

    def test_saved_page_exposes_location_coordinates(self):
        text = sync_navigation_data.DEFAULT_HTML.read_text(encoding="utf-8")
        locations, _ = sync_navigation_data.parse_completion_html(text)
        by_name = {record["name"]: record for record in locations}
        self.assertEqual(by_name["Pallet Town"]["map_pixel"], [470, 1911])
        self.assertIn("Victory Road", by_name)


class TrainerNavigationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import train
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"training dependencies are not installed: {exc}") from exc
        cls.train = train

    def test_trainer_uses_catalog_labels(self):
        self.assertEqual(self.train.MAP_LABELS[54], "Pewter Gym")

    def test_route_to_pewter_does_not_reward_entering_pewter_gym(self):
        milestone = next(ms for ms in self.train.MILESTONES if ms["name"] == "05g_to_pewter")
        self.assertNotIn(54, milestone["shaping"])
        self.assertIn(13, milestone["shaping"])

    def test_post_brock_route_has_exact_required_warp_actions(self):
        rules = {
            (rule["map"], tuple(rule["target"])): rule["action"]
            for rule in self.train.POST_BROCK_ROUTE_ACTION_GUIDANCE
            if rule.get("radius") == 0
        }
        self.assertEqual(rules[(15, (6, 18))], 0)
        self.assertEqual(rules[(59, (5, 6))], 2)
        self.assertEqual(rules[(60, (17, 20))], 3)
        self.assertEqual(rules[(61, (7, 4))], 3)


if __name__ == "__main__":
    unittest.main()
