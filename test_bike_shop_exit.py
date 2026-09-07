"""Bike Shop exit route + interior off-map guard.

Locks in the 2026-08-28 audit findings, all measured on the live phase-91
frontier state with PyBoy:

  * Map 66 has exactly two warp tiles, (7,2) and (7,3). Down from (7,4),
    (7,5) or (7,6) walks off the map into the garbage "map 57" state.
    13 of 16 random episodes ended there before the guard existed.
  * Poking the player onto the mat instead of walking there skips Yellow's
    warp check and freezes the worker forever -- the live phase-91 stall
    (1,018 identical steps, 5 executed actions out of 1,024, all 96 workers).
  * The verified 8-step exit route only works once the emulator has settled:
    it corrupts at load_settle_ticks<=65 and exits cleanly at >=70.
"""
import inspect
import re

import train


BIKE_SHOP = 66
CORRUPTING_DOWN_TILES = ((7, 4), (7, 5), (7, 6))
WARP_TILES = ((7, 2), (7, 3))
# Pikachu's follower blocks their descent depending on the approach, so no
# static action is correct for these and they must stay out of the table.
FOLLOWER_BLOCKED_POCKET = ((2, 2), (2, 3), (2, 4), (2, 5), (3, 4))


def test_catalog_geometry_loaded_for_the_bike_shop():
    geometry = train.INTERIOR_MAP_GEOMETRY[BIKE_SHOP]
    (height, width), warps = geometry
    assert (height, width) == (8, 8)
    assert warps == frozenset(WARP_TILES)


def test_only_warp_tiles_may_leave_an_interior():
    for y, x in WARP_TILES:
        assert not train.interior_offmap_move_is_illegal(BIKE_SHOP, y, x, 'down')
    for y, x in CORRUPTING_DOWN_TILES:
        assert train.interior_offmap_move_is_illegal(BIKE_SHOP, y, x, 'down')
    # East wall: Right off x=7 is the same class of off-map step.
    for y in (3, 4, 6):
        assert train.interior_offmap_move_is_illegal(BIKE_SHOP, y, 7, 'right')


def test_guard_ignores_in_bounds_moves_and_non_movement_actions():
    assert not train.interior_offmap_move_is_illegal(BIKE_SHOP, 4, 4, 'down')
    assert not train.interior_offmap_move_is_illegal(BIKE_SHOP, 7, 6, 'left')
    for action in ('a', 'b', 'start', 'noop'):
        assert not train.interior_offmap_move_is_illegal(BIKE_SHOP, 7, 6, action)


def test_guard_leaves_outdoor_maps_alone():
    # Maps with connections legitimately walk off their own edges; the guard
    # must never see them or it would wall the swarm inside every route.
    outdoor = [
        map_id for map_id in (1, 2, 3, 12, 13)
        if map_id not in train.INTERIOR_MAP_GEOMETRY
    ]
    assert outdoor, "expected at least one connected overworld map to be exempt"
    for map_id in outdoor:
        assert not train.interior_offmap_move_is_illegal(map_id, 0, 0, 'up')


def test_exit_route_never_emits_a_move_that_leaves_the_map():
    for (y, x), action in train.BIKE_SHOP_EXIT_ROUTE.items():
        if (y, x) in WARP_TILES:
            continue
        assert not train.interior_offmap_move_is_illegal(BIKE_SHOP, y, x, action)


def test_exit_route_never_presses_down_on_a_corrupting_tile():
    for tile in CORRUPTING_DOWN_TILES:
        assert train.BIKE_SHOP_EXIT_ROUTE[tile] == 'left'


def test_follower_blocked_pocket_is_left_to_the_policy():
    for tile in FOLLOWER_BLOCKED_POCKET:
        assert tile not in train.BIKE_SHOP_EXIT_ROUTE


def test_route_table_reaches_a_warp_tile_from_every_entry():
    deltas = train._OFFMAP_STEP_DELTAS
    for start in train.BIKE_SHOP_EXIT_ROUTE:
        tile = start
        for _ in range(12):
            if tile in WARP_TILES:
                break
            action = train.BIKE_SHOP_EXIT_ROUTE.get(tile)
            assert action is not None, f"{start} left the table at {tile}"
            dy, dx = deltas[action]
            tile = (tile[0] + dy, tile[1] + dx)
        else:
            raise AssertionError(f"{start} never reached a warp tile")


def test_live_frontier_tile_exits_in_eight_moves():
    # The phase-91 frontier state starts every worker on (3,6).
    deltas = train._OFFMAP_STEP_DELTAS
    tile, moves = (3, 6), 0
    while tile not in WARP_TILES:
        dy, dx = deltas[train.BIKE_SHOP_EXIT_ROUTE[tile]]
        tile = (tile[0] + dy, tile[1] + dx)
        moves += 1
    assert (moves, tile) == (7, (7, 3))  # + the final Down through the door


def test_walk_out_never_pokes_coordinates():
    source = inspect.getsource(
        train.PokemonYellowEnv._walk_out_of_bike_shop
    )
    for addr in ('ADDR_POS_A', 'ADDR_POS_B', 'ADDR_MAP_ID'):
        assert f'memory[{addr}] =' not in source
        assert f'memory[{addr}]=' not in source


def test_leave_path_delegates_to_the_walker():
    source = inspect.getsource(
        train.PokemonYellowEnv._advance_bike_shop_customer_aisle
    )
    assert 'return self._walk_out_of_bike_shop()' in source
    # The old poke-onto-the-mat behaviour must not come back.
    assert 'memory[ADDR_POS_A] = 7' not in source


def test_softlock_exit_follows_the_verified_route():
    source = inspect.getsource(
        train.PokemonYellowEnv._force_exit_bike_shop_softlock
    )
    assert 'BIKE_SHOP_EXIT_ROUTE' in source
    # Blind Down-mashing is what walked workers off the bottom row.
    assert '_tap_scripted_button("down", 24, 48)' not in source


def test_walkout_latches_reset_every_episode():
    for gate in ('_bike_shop_walkout_disabled', '_bike_shop_walkout_misses'):
        assert gate in train.PER_EPISODE_PROGRESSION_GATES


def test_fullgame_settle_ticks_clear_the_measured_threshold():
    # The exit route corrupts into map 57 at settle<=65 and works at >=70.
    # `fullgame_milestone` is built inside train.py's __main__ block, so this
    # has to assert on the source rather than on an importable object.
    with open(train.__file__, encoding='utf-8') as handle:
        source = handle.read()
    block = source[source.index('fullgame_milestone = {'):]
    block = block[:block.index("'battle_step_limit'")]
    found = re.search(r"'load_settle_ticks': (\d+)", block)
    assert found, "fullgame_milestone no longer sets load_settle_ticks"
    assert int(found.group(1)) >= 70, found.group(1)


def test_scripted_noop_stall_limit_is_below_the_episode_length():
    assert 0 < train.SCRIPTED_NOOP_STALL_LIMIT < 1024
    assert train.BIKE_SHOP_WALKOUT_STRIKE_LIMIT >= 1
