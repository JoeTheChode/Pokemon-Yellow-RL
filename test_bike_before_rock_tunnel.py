#!/usr/bin/env python3
import unittest

import train


def first_rule(rules, map_id, y, x, require=None):
    for rule in rules:
        if rule["map"] != map_id or tuple(rule["target"]) != (y, x):
            continue
        bits = rule.get("require_event_bits")
        if require is None and not bits:
            return rule
        if require is not None and bits == require:
            return rule
    return None


class BikeBeforeRockTunnelTests(unittest.TestCase):
    def test_voucher_and_bike_are_before_rock_tunnel(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        voucher = names.index("get_bike_voucher")
        bicycle = names.index("get_bicycle")
        charmander = names.index("receive_charmander")
        route9 = names.index("reach_route_9")
        tunnel = names.index("enter_rock_tunnel")
        squirtle = names.index("receive_squirtle")
        self.assertEqual(names[voucher - 1], "receive_squirtle")
        self.assertLess(squirtle, voucher)
        self.assertLess(voucher, bicycle)
        self.assertLess(bicycle, charmander)
        self.assertLess(charmander, route9)
        self.assertLess(route9, tunnel)

    def test_event_bits_match_rom(self):
        names = {w["name"]: w for w in train.FULLGAME_QUEST_WAYPOINTS}
        self.assertEqual(int(names["get_bike_voucher"]["event_bit"]), 337)
        self.assertEqual(int(names["get_bicycle"]["event_bit"]), 192)
        self.assertEqual(train.EVENT_GOT_BIKE_VOUCHER, 337)
        self.assertEqual(train.EVENT_GOT_BICYCLE, 192)

    def test_fan_club_door_and_chairman(self):
        door = first_rule(train.POST_SURGE_FAN_CLUB_APPROACH_ACTION_GUIDANCE, 5, 13, 9)
        self.assertEqual(door["destination_map"], 90)
        self.assertEqual(int(door["action"]), 0)
        talk = first_rule(train.FAN_CLUB_CHAIRMAN_ACTION_GUIDANCE, 90, 2, 3)
        self.assertEqual(int(talk["action"]), 4)
        self.assertEqual(talk.get("unless_event_bits"), (337,))
        # Viewer 219,274 = club origin 216,273 + chairman (x=3,y=1).
        self.assertEqual(talk["target"], (2, 3))
        leave = first_rule(
            train.POST_SURGE_FAN_CLUB_LEAVE_ACTION_GUIDANCE,
            90, 7, 2, require=(337,),
        )
        self.assertEqual(leave["destination_map"], 5)

    def test_bike_shop_requires_voucher(self):
        door = first_rule(
            train.BIKE_SHOP_DOOR_ENTER_ACTION_GUIDANCE, 3, 25, 13, require=(337,),
        )
        self.assertIsNotNone(door)
        self.assertEqual(door["destination_map"], 66)
        self.assertEqual(door.get("require_event_bits"), (337,))
        self.assertEqual(door.get("unless_event_bits"), (192,))
        clerk = first_rule(
            train.BIKE_SHOP_VENDOR_ACTION_GUIDANCE, 66, 3, 6, require=(337,),
        )
        self.assertIsNotNone(clerk)
        self.assertEqual(int(clerk["action"]), 4)
        self.assertEqual(int(clerk["face_action"]), 0)
        leave = first_rule(
            train.POST_SURGE_BIKE_SHOP_LEAVE_ACTION_GUIDANCE,
            66, 7, 2, require=(192,),
        )
        self.assertEqual(leave["destination_map"], 3)
        aisle = first_rule(
            train.POST_SURGE_BIKE_SHOP_LEAVE_ACTION_GUIDANCE,
            66, 3, 6, require=(192,),
        )
        self.assertEqual(int(aisle["action"]), 2)
        col = first_rule(
            train.POST_SURGE_BIKE_SHOP_LEAVE_ACTION_GUIDANCE,
            66, 3, 3, require=(192,),
        )
        self.assertEqual(int(col["action"]), 1)
        west = first_rule(
            train.POST_SURGE_BIKE_SHOP_LEAVE_ACTION_GUIDANCE,
            66, 6, 3, require=(192,),
        )
        self.assertEqual(int(west["action"]), 1)

    def test_phase91_does_not_skip_shop_leave(self):
        import inspect
        src = inspect.getsource(train.PokemonYellowEnv._matching_forced_action_guidance)
        shop = src[src.index("Bike shop door mats") : src.index("Fan Club door mat")]
        self.assertNotIn("EVENT_GOT_BICYCLE):\n                return None", shop)
        self.assertIn("ADDR_LAST_MAP] = 3", shop)
        self.assertIn("shop_pos != (3, 6)", shop)
        self.assertIn("ADDR_FONT_LOADED] = 0", shop)
        self.assertIn("return 1", shop)
        door = first_rule(
            train.POST_SURGE_BIKE_TO_ROUTE24_ACTION_GUIDANCE,
            3, 25, 13, require=(192,),
        )
        self.assertEqual(int(door["action"]), 1)
        street = first_rule(
            train.POST_SURGE_BIKE_TO_ROUTE24_ACTION_GUIDANCE,
            3, 26, 13, require=(192,),
        )
        self.assertEqual(int(street["action"]), 3)
        turn_north = first_rule(
            train.POST_SURGE_BIKE_TO_ROUTE24_ACTION_GUIDANCE,
            3, 26, 16, require=(192,),
        )
        self.assertEqual(int(turn_north["action"]), 0)
        recover_wall = first_rule(
            train.POST_SURGE_BIKE_TO_ROUTE24_ACTION_GUIDANCE,
            3, 26, 20, require=(192,),
        )
        self.assertEqual(int(recover_wall["action"]), 2)
        north_city = first_rule(
            train.POST_SURGE_BIKE_TO_ROUTE24_ACTION_GUIDANCE,
            3, 12, 20, require=(192,),
        )
        self.assertEqual(int(north_city["action"]), 0)
        jenny = first_rule(
            train.POST_SURGE_BIKE_TO_ROUTE24_ACTION_GUIDANCE,
            3, 15, 27, require=(192,),
        )
        self.assertEqual(int(jenny["action"]), 0)
        wall = first_rule(
            train.POST_SURGE_BIKE_TO_ROUTE24_ACTION_GUIDANCE,
            3, 15, 22, require=(192,),
        )
        self.assertEqual(int(wall["action"]), 0)
        street = first_rule(
            train.POST_SURGE_BIKE_TO_ROUTE24_ACTION_GUIDANCE,
            3, 13, 27, require=(192,),
        )
        self.assertEqual(int(street["action"]), 2)

    def test_phase_constants(self):
        names = [w["name"] for w in train.FULLGAME_QUEST_WAYPOINTS]
        self.assertEqual(names[train.FULLGAME_BIKE_VOUCHER_QUEST_PHASE], "get_bike_voucher")
        self.assertEqual(names[train.FULLGAME_BICYCLE_QUEST_PHASE], "get_bicycle")
        self.assertEqual(
            names[train.FULLGAME_RETURN_ROUTE24_CHARMANDER_QUEST_PHASE],
            "return_to_route_24_for_charmander",
        )
        self.assertLess(
            train.FULLGAME_BICYCLE_QUEST_PHASE,
            train.FULLGAME_ROCK_TUNNEL_ENTRY_QUEST_PHASE,
        )

    def test_damian_talk_tile(self):
        talk = first_rule(train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 6, 6)
        self.assertEqual(int(talk["action"]), 4)
        self.assertEqual(talk.get("unless_event_bits"), (1359,))

    def test_damian_approach_uses_passable_x6_gap(self):
        ledge_stall = first_rule(
            train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 8, 7,
        )
        self.assertEqual(int(ledge_stall["action"]), 2)
        gap = first_rule(
            train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 8, 6,
        )
        self.assertEqual(int(gap["action"]), 0)
        climb = first_rule(
            train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 7, 6,
        )
        self.assertEqual(int(climb["action"]), 0)

    def test_nugget_bridge_weaves_around_trainers(self):
        stall = first_rule(train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 32, 11)
        self.assertEqual(int(stall["action"]), 2)
        bypass = first_rule(train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 31, 10)
        self.assertEqual(int(bypass["action"]), 0)
        lass = first_rule(train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 29, 10)
        self.assertEqual(int(lass["action"]), 3)
        gap = first_rule(train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 8, 10)
        self.assertEqual(int(gap["action"]), 2)
        north = first_rule(train.POST_SURGE_DAMIAN_ACTION_GUIDANCE, 35, 7, 6)
        self.assertEqual(int(north["action"]), 0)

    def test_y13_street_avoids_trade_house_warp(self):
        trade = first_rule(train.POST_SURGE_FAN_CLUB_APPROACH_ACTION_GUIDANCE, 5, 13, 15)
        self.assertEqual(int(trade["action"]), 1)
        self.assertNotIn("destination_map", trade)
        door = first_rule(train.POST_SURGE_FAN_CLUB_APPROACH_ACTION_GUIDANCE, 5, 13, 9)
        self.assertEqual(door["destination_map"], 90)
        self.assertEqual(int(first_rule(train.POST_SURGE_FAN_CLUB_APPROACH_ACTION_GUIDANCE, 5, 14, 15)["action"]), 2)
        self.assertEqual(int(first_rule(train.POST_SURGE_FAN_CLUB_APPROACH_ACTION_GUIDANCE, 5, 14, 9)["action"]), 0)

    def test_jenny_tile_leaves_for_fan_club_after_gift(self):
        rule = first_rule(
            train.POST_SURGE_FAN_CLUB_APPROACH_ACTION_GUIDANCE,
            5, 15, 20, require=(327,),
        )
        self.assertIsNotNone(rule)
        self.assertEqual(int(rule["action"]), 0)
        self.assertEqual(rule.get("unless_event_bits"), (337,))
        west = first_rule(
            train.POST_SURGE_FAN_CLUB_APPROACH_ACTION_GUIDANCE,
            5, 14, 20, require=(327,),
        )
        self.assertEqual(int(west["action"]), 2)

    def test_north_cerulean_walks_south_to_bike_shop(self):
        req = (337,)
        col = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 18, 13, require=req,
        )
        self.assertIsNotNone(col)
        self.assertEqual(int(col["action"]), 1)
        self.assertEqual(col.get("unless_event_bits"), (192,))
        west = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 18, 12, require=req,
        )
        self.assertEqual(int(west["action"]), 3)
        east = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 18, 35, require=req,
        )
        self.assertEqual(int(east["action"]), 2)
        gym_wall = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 18, 32, require=req,
        )
        self.assertEqual(int(gym_wall["action"]), 3)
        ledge = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 18, 34, require=req,
        )
        self.assertEqual(int(ledge["action"]), 1)
        land = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 20, 34, require=req,
        )
        self.assertEqual(int(land["action"]), 1)
        corridor = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 32, 36, require=req,
        )
        self.assertEqual(int(corridor["action"]), 0)
        drop = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 26, 36, require=req,
        )
        self.assertEqual(int(drop["action"]), 0)
        jump = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 18, 36, require=req,
        )
        self.assertEqual(int(jump["action"]), 2)
        wall36 = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 25, 36, require=req,
        )
        self.assertEqual(int(wall36["action"]), 0)
        cutover = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 29, 36, require=req,
        )
        self.assertEqual(int(cutover["action"]), 0)
        stall = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 29, 19, require=req,
        )
        self.assertEqual(int(stall["action"]), 3)
        stand = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 26, 13, require=req,
        )
        self.assertEqual(stand["destination_map"], 66)
        self.assertEqual(int(stand["action"]), 0)
        plaza = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 26, 16, require=req,
        )
        self.assertEqual(int(plaza["action"]), 2)
        npc = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 26, 30, require=req,
        )
        self.assertEqual(int(npc["action"]), 1)
        building = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 22, 13, require=req,
        )
        self.assertEqual(int(building["action"]), 3)
        ledge_north = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 32, 16, require=req,
        )
        self.assertEqual(int(ledge_north["action"]), 2)
        self.assertIsNone(
            first_rule(
                train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 34, 16, require=req,
            )
        )
        self.assertIsNone(
            first_rule(
                train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 35, 16, require=req,
            )
        )
        west_gap = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 34, 13, require=req,
        )
        self.assertEqual(int(west_gap["action"]), 0)
        south = first_rule(
            train._forced_route_action_guidance(
                train.CERULEAN_TO_BIKE_SHOP_STEPS,
                require_event_bits=(337,),
                unless_event_bits=(192,),
            ),
            3, 29, 15, require=req,
        )
        self.assertEqual(int(south["action"]), 0)
        door = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 25, 13, require=req,
        )
        self.assertEqual(door["destination_map"], 66)
        self.assertEqual(int(door["action"]), 0)
        center = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 3, 17, 19, require=req,
        )
        self.assertEqual(int(center["action"]), 1)
        mat = first_rule(
            train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE, 66, 7, 3, require=req,
        )
        self.assertEqual(int(mat["action"]), 0)

    def test_north_to_shop_beats_return_first_match(self):
        combined = (
            list(train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE)
            + list(train.BIKE_SHOP_VENDOR_ACTION_GUIDANCE)
            + list(train.RETURN_TO_CERULEAN_ACTION_GUIDANCE)
        )
        pile = first_rule(combined, 3, 18, 13, require=(337,))
        self.assertIsNotNone(pile)
        self.assertEqual(int(pile["action"]), 1)
        self.assertNotIn("destination_map", pile)
        clerk = first_rule(combined, 66, 3, 6, require=(337,))
        self.assertEqual(int(clerk["action"]), 4)
        east_mat = first_rule(combined, 66, 7, 3, require=(337,))
        self.assertEqual(int(east_mat["action"]), 0)
        west_mat = first_rule(combined, 66, 7, 2, require=(337,))
        self.assertEqual(int(west_mat["action"]), 3)

    def test_redeem_hook_is_live_on_get_bicycle(self):
        import inspect
        from pathlib import Path
        src = Path(train.__file__).read_text(encoding="utf-8")
        start = src.index("def _try_redeem_bike_voucher_at_clerk")
        chunk = src[start:start + 700]
        self.assertNotIn("return False  # coast path: disabled", chunk)
        self.assertIn("FULLGAME_BICYCLE_QUEST_PHASE", chunk)
        helper = inspect.getsource(train.rock_tunnel_dialogue_recovery_action)
        self.assertIn("bike_shop_redeem_text_phase", helper)
        self.assertIn("bike_shop_redeem_text_phase or fan_club_text_phase", helper)
        self.assertIsNone(train.rock_tunnel_dialogue_recovery_action(
            66, train.FULLGAME_BICYCLE_QUEST_PHASE, 0, 8, 7, 2,
        ))
        self.assertEqual(train.rock_tunnel_dialogue_recovery_action(
            66, train.FULLGAME_BICYCLE_QUEST_PHASE, 0, 8, 3, 6,
        ), "a")
        redeem = inspect.getsource(
            train.PokemonYellowEnv._try_redeem_bike_voucher_at_clerk
        )
        self.assertIn("pos != (3, 6)", redeem)
        self.assertIn('"up"', redeem)

    def test_bike_shop_pending_warp_is_settled_without_coordinate_pokes(self):
        import inspect
        src = inspect.getsource(train.PokemonYellowEnv._settle_bike_shop_pending_warp)
        self.assertIn("self.pyboy.tick()", src)
        self.assertNotIn("memory[ADDR_POS_A] =", src)
        self.assertNotIn("memory[ADDR_POS_B] =", src)

    def test_bike_shop_customer_aisle_clears_and_moves_atomically(self):
        from types import SimpleNamespace

        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_BICYCLE_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        env._bag_contains_item = lambda item: item == train.ITEM_BIKE_VOUCHER
        taps = []
        env._tap_scripted_button = lambda button, press, release: taps.append(
            (button, press, release)
        )
        memory[train.ADDR_MAP_ID] = 66
        memory[train.ADDR_POS_A] = 3
        memory[train.ADDR_POS_B] = 3
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_TEXT_BOX] = 1
        memory[train.ADDR_FONT_LOADED] = 1

        self.assertTrue(env._advance_bike_shop_customer_aisle())
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 0)
        self.assertEqual(memory[train.ADDR_FONT_LOADED], 0)
        self.assertEqual(taps, [("right", 8, 24)])

    def test_bike_shop_customer_aisle_clears_blocking_pikachu_at_x4(self):
        from types import SimpleNamespace

        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_BICYCLE_QUEST_PHASE
        # Recovered phase-90 frontiers can own the item without retaining the
        # received-voucher event byte; inventory must still unlock the aisle.
        env._event_flag_is_set = lambda bit: False
        env._bag_contains_item = lambda item: item == train.ITEM_BIKE_VOUCHER
        taps = []
        env._tap_scripted_button = lambda button, press, release: taps.append(
            (button, press, release)
        )
        memory[train.ADDR_MAP_ID] = 66
        memory[train.ADDR_POS_A] = 3
        memory[train.ADDR_POS_B] = 4
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_TEXT_BOX] = 1
        memory[train.ADDR_FONT_LOADED] = 1
        for base in (
            train.ADDR_PIKACHU_SPRITE_STATE_DATA_1,
            train.ADDR_PIKACHU_SPRITE_STATE_DATA_2,
        ):
            memory[base:base + train.PIKACHU_SPRITE_STATE_DATA_SIZE] = b"\xff" * 16

        self.assertTrue(env._advance_bike_shop_customer_aisle())
        for base in (
            train.ADDR_PIKACHU_SPRITE_STATE_DATA_1,
            train.ADDR_PIKACHU_SPRITE_STATE_DATA_2,
        ):
            self.assertEqual(
                memory[base:base + train.PIKACHU_SPRITE_STATE_DATA_SIZE],
                b"\x00" * 16,
            )
        self.assertEqual(memory[train.ADDR_POS_B], 6)
        self.assertEqual(memory[train.ADDR_PLAYER_DIRECTION], 8)
        self.assertEqual(memory[train.ADDR_PLAYER_SPRITE_FACING], 0x08)
        self.assertEqual(taps, [])

    def test_bike_shop_exit_clears_blocking_pikachu_at_x4(self):
        from types import SimpleNamespace

        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_RETURN_ROUTE24_CHARMANDER_QUEST_PHASE
        # Quest metadata is authoritative even if a recovered state's event
        # byte is stale or absent.
        env._event_flag_is_set = lambda bit: False
        env._bag_contains_item = lambda item: False
        restored = []
        env._set_event_flag_bit = lambda bit, value: restored.append((bit, value))
        env._tap_scripted_button = lambda *args: self.fail("unexpected tap")
        memory[train.ADDR_MAP_ID] = 66
        memory[train.ADDR_POS_A] = 3
        memory[train.ADDR_POS_B] = 4
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_TEXT_BOX] = 1
        memory[train.ADDR_FONT_LOADED] = 1

        self.assertTrue(env._advance_bike_shop_customer_aisle())
        self.assertEqual(memory[train.ADDR_POS_B], 3)
        self.assertEqual(memory[train.ADDR_PLAYER_DIRECTION], 4)
        self.assertEqual(memory[train.ADDR_PLAYER_SPRITE_FACING], 0x00)
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 0)
        self.assertEqual(memory[train.ADDR_FONT_LOADED], 0)
        self.assertEqual(restored, [(train.EVENT_GOT_BICYCLE, True)])

    def test_bike_shop_exit_crosses_latched_door_atomically(self):
        from types import SimpleNamespace

        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_RETURN_ROUTE24_CHARMANDER_QUEST_PHASE
        env._event_flag_is_set = lambda bit: False
        env._bag_contains_item = lambda item: False
        restored = []
        env._set_event_flag_bit = lambda bit, value: restored.append((bit, value))
        taps = []
        env._tap_scripted_button = lambda button, press, release: taps.append(
            (button, press, release)
        )
        memory[train.ADDR_MAP_ID] = 66
        memory[train.ADDR_POS_A] = 6
        memory[train.ADDR_POS_B] = 3
        memory[train.ADDR_BATTLE_FLAG] = 0
        memory[train.ADDR_TEXT_BOX] = 1
        memory[train.ADDR_FONT_LOADED] = 1

        self.assertTrue(env._advance_bike_shop_customer_aisle())
        self.assertEqual(memory[train.ADDR_LAST_MAP], 3)
        self.assertEqual(memory[train.ADDR_POS_A], 7)
        self.assertEqual(memory[train.ADDR_POS_B], 3)
        self.assertEqual(memory[train.ADDR_TEXT_BOX], 0)
        self.assertEqual(memory[train.ADDR_FONT_LOADED], 0)
        self.assertEqual(restored, [(train.EVENT_GOT_BICYCLE, True)])
        self.assertEqual(taps, [("down", 8, 256)])

    def test_bike_shop_phase91_closes_clerk_dialogue_with_inputs_before_left(self):
        from types import SimpleNamespace

        env = train.PokemonYellowEnv.__new__(train.PokemonYellowEnv)
        memory = bytearray(0x10000)
        env.pyboy = SimpleNamespace(memory=memory)
        env.quest_phase = train.FULLGAME_RETURN_ROUTE24_CHARMANDER_QUEST_PHASE
        env._event_flag_is_set = lambda bit: bit == train.EVENT_GOT_BICYCLE
        env._bag_contains_item = lambda item: False
        env._bike_shop_walkout_disabled = False
        taps = []

        def tap(button, press, release):
            taps.append((button, press, release))
            if button == "down":
                memory[train.ADDR_POS_A] = 4

        env._tap_scripted_button = tap
        memory[train.ADDR_MAP_ID] = 66
        memory[train.ADDR_POS_A] = 3
        memory[train.ADDR_POS_B] = 6
        memory[train.ADDR_BATTLE_FLAG] = 0

        self.assertTrue(env._advance_bike_shop_customer_aisle())
        self.assertEqual(taps, [("b", 8, 24)] * 8 + [("down", 8, 24)])
        self.assertEqual(memory[train.ADDR_POS_A], 4)

    def test_ledge_pocket_snaps_to_shop_stand(self):
        import inspect

        src = inspect.getsource(train.PokemonYellowEnv._settle_cerulean_bike_ledge_pocket)
        self.assertIn("pos_a >= 33", src)
        self.assertIn("16 <= pos_b <= 23", src)
        self.assertIn("ADDR_POS_A] = 26", src)
        self.assertIn("ADDR_POS_B] = 13", src)
        self.assertIn("FULLGAME_BICYCLE_QUEST_PHASE", src)
        combined = (
            list(train.CERULEAN_NORTH_TO_BIKE_SHOP_ACTION_GUIDANCE)
            + list(train.RETURN_TO_CERULEAN_ACTION_GUIDANCE)
        )
        stand = first_rule(combined, 3, 26, 13, require=(337,))
        self.assertEqual(int(stand["action"]), 0)
        self.assertEqual(stand["destination_map"], 66)


if __name__ == "__main__":
    unittest.main()
