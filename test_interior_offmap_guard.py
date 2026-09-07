import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name(".audit_remote_train_current.py")
SPEC = importlib.util.spec_from_file_location("deployed_train_candidate", SOURCE)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class InteriorOffmapGuardTests(unittest.TestCase):
    def test_catalog_dimensions_are_width_then_height(self):
        catalog = {
            "game_maps": [{
                "id": 40,
                "connections": [],
                "size_blocks": [5, 6],
                "warps": [{"position": [11, 4]}],
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            geometry = train.load_interior_map_geometry(str(path))

        self.assertEqual(geometry[40][0], (12, 10))
        self.assertEqual(geometry[40][1], frozenset({(11, 4)}))

    def test_oaks_lab_south_aisle_and_warp_are_legal(self):
        self.assertFalse(
            train.interior_offmap_move_is_illegal(40, 9, 4, "down")
        )
        self.assertFalse(
            train.interior_offmap_move_is_illegal(40, 10, 4, "down")
        )
        self.assertFalse(
            train.interior_offmap_move_is_illegal(40, 11, 4, "down")
        )

    def test_nonwarp_edge_step_remains_blocked(self):
        self.assertTrue(
            train.interior_offmap_move_is_illegal(40, 11, 3, "down")
        )


if __name__ == "__main__":
    unittest.main()
