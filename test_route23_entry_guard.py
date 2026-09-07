import train


def _set_event(memory, event_bit):
    address = train.ADDR_EVENT_FLAGS_START + event_bit // 8
    memory[address] |= 1 << (event_bit % 8)


def test_route23_entry_guidance_joins_recorded_approach():
    approach = train.ROUTE23_ENTRY_GUARD_APPROACH_STEPS
    exit_steps = train.ROUTE23_ENTRY_GUARD_EXIT_STEPS
    recorded = train.ROUTE23_TO_FIRST_BADGE_GUARD_STEPS

    assert approach[0][:3] == (34, 139, 7)
    assert approach[-1][:4] == (34, 137, 7, 0)
    assert exit_steps[0][:3] == (34, 135, 7)
    assert exit_steps[-1][:4] == (34, 135, 8, 3)
    assert recorded[0][:3] == (34, 135, 9)
    assert train.GEN1_POST_STRENGTH_SURF_FIELD_MENU_INDEX == 2


def test_route23_surf_guidance_reaches_victory_road():
    route = train.ROUTE23_SURF_TO_VICTORY_ROAD_STEPS

    assert route[0][:4] == (34, 103, 11, 0)
    assert len(route) == 89
    assert route[-1] == (34, 32, 4, 0, 108)


def test_victory_road_frontier_joins_saved_strength_route():
    connector = train.VICTORY_ROAD_1F_ENTRY_TO_ROUTE_STEPS
    route = train.VICTORY_ROAD_1F_TO_BOULDER_STEPS

    assert connector == ((108, 31, 4, 0, None),)
    assert route[0][:3] == (108, 17, 8)


def test_all_five_league_battles_have_resource_assists():
    assert '_lorelei_agatha_resources_prepared_phase' in (
        train.BATTLE_ASSIST_ONE_SHOT_FLAGS
    )
    assert '_lorelei_agatha_emergency_heals_used' in (
        train.BATTLE_ASSIST_HEAL_BUDGETS
    )
    assert '_bruno_battle_resources_prepared' in train.BATTLE_ASSIST_ONE_SHOT_FLAGS
    assert '_lance_battle_resources_prepared' in train.BATTLE_ASSIST_ONE_SHOT_FLAGS
    assert '_champion_battle_resources_prepared' in train.BATTLE_ASSIST_ONE_SHOT_FLAGS


def test_route23_entry_guard_advances_dialogue_then_moves_once():
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 34
    memory[train.ADDR_POS_A] = 136
    memory[train.ADDR_POS_B] = 7
    memory[train.ADDR_FONT_LOADED] = 1
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE
    a_presses = 0
    up_presses = 0

    def tap(button, *_args):
        nonlocal a_presses, up_presses
        if button == 'a':
            a_presses += 1
            if a_presses == 3:
                memory[train.ADDR_FONT_LOADED] = 0
        elif button == 'up':
            up_presses += 1
            memory[train.ADDR_POS_A] = 135

    env._tap_scripted_button = tap

    assert env._try_clear_route23_entry_guard() is True
    assert a_presses == 3
    assert up_presses == 1
    assert (memory[train.ADDR_POS_A], memory[train.ADDR_POS_B]) == (135, 7)


def test_route23_entry_guard_is_strictly_scoped():
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 34
    memory[train.ADDR_POS_A] = 135
    memory[train.ADDR_POS_B] = 7
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE

    assert env._try_clear_route23_entry_guard() is False


def test_route23_repairs_only_legacy_missing_earth_badge_frontier():
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 34
    memory[train.ADDR_POS_A] = 139
    memory[train.ADDR_POS_B] = 7
    memory[train.ADDR_BADGES] = 0x7F
    _set_event(memory, 81)
    _set_event(memory, train.EVENT_BEAT_ROUTE22_RIVAL)
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE

    assert env._restore_legacy_earth_badge_after_viridian_recovery() is True
    assert memory[train.ADDR_BADGES] == 0xFF
    assert env._restore_legacy_earth_badge_after_viridian_recovery() is False


def test_route23_does_not_invent_earth_badge_without_legacy_evidence():
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 34
    memory[train.ADDR_BADGES] = 0x7F
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE

    assert env._restore_legacy_earth_badge_after_viridian_recovery() is False
    assert memory[train.ADDR_BADGES] == 0x7F


def test_route23_later_badge_guard_clears_four_pages_and_moves():
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 34
    memory[train.ADDR_POS_A] = 96
    memory[train.ADDR_POS_B] = 10
    memory[train.ADDR_FONT_LOADED] = 1
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = train.FULLGAME_VICTORY_ROAD_1F_QUEST_PHASE
    a_presses = 0

    def tap(button, *_args):
        nonlocal a_presses
        if button == 'a':
            a_presses += 1
            if a_presses == 4:
                memory[train.ADDR_FONT_LOADED] = 0
        elif button == 'up' and memory[train.ADDR_FONT_LOADED] == 0:
            memory[train.ADDR_POS_A] = 95

    env._tap_scripted_button = tap

    assert env._try_clear_route23_badge_guard() is True
    assert a_presses == 4
    assert (memory[train.ADDR_POS_A], memory[train.ADDR_POS_B]) == (95, 10)
