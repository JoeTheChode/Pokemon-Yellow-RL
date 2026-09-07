"""Victory Road 1F uses the stored atlas route; CD6B=1 must not block it."""
import train


def _vr_env(memory, phase=None):
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = (
        train.FULLGAME_VICTORY_ROAD_2F_QUEST_PHASE if phase is None else phase
    )
    return env


def _set_flag(memory, bit):
    addr = train.ADDR_EVENT_FLAGS_START + int(bit) // 8
    memory[addr] = int(memory[addr]) | (1 << (int(bit) % 8))


def _solver_memory(*, joy=1, font=0, battle=0, y=14, x=5, boulder=(15, 5)):
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 108
    memory[train.ADDR_POS_A] = y
    memory[train.ADDR_POS_B] = x
    memory[train.ADDR_FONT_LOADED] = font
    memory[train.ADDR_TEXT_BOX] = joy
    memory[train.ADDR_BATTLE_FLAG] = battle
    memory[train.ADDR_VICTORY_ROAD_1F_BOULDER1_Y] = boulder[0] + 4
    memory[train.ADDR_VICTORY_ROAD_1F_BOULDER1_X] = boulder[1] + 4
    _set_flag(memory, train.EVENT_GOT_HM04)
    return memory


def test_strength_field_menu_is_after_surf():
    assert train.GEN1_POST_STRENGTH_SURF_FIELD_MENU_INDEX == 2
    assert train.GEN1_STRENGTH_FIELD_MENU_INDEX == 3


def test_victory_road_1f_atlas_is_the_stored_entrance_to_boulder_route():
    route = train.VICTORY_ROAD_1F_TO_BOULDER_STEPS
    assert route[0][:4] == (108, 17, 8, 0)
    assert route[-1][:4] == (108, 14, 4, 3)
    y, x = 17, 8
    deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    for map_id, tile_y, tile_x, action, dest in route:
        assert map_id == 108
        assert (tile_y, tile_x) == (y, x)
        assert dest is None
        dy, dx = deltas[action]
        y += dy
        x += dx
    assert (y, x) == (14, 5)


def test_victory_road_input_locked_treats_cd6b_one_as_overworld():
    memory = bytearray(0x10000)
    env = _vr_env(memory)
    memory[train.ADDR_FONT_LOADED] = 0
    memory[train.ADDR_TEXT_BOX] = 1
    assert env._victory_road_input_locked(memory) is False
    memory[train.ADDR_TEXT_BOX] = 0xFF
    assert env._victory_road_input_locked(memory) is True
    memory[train.ADDR_TEXT_BOX] = 0
    memory[train.ADDR_FONT_LOADED] = 1
    assert env._victory_road_input_locked(memory) is True


def test_victory_road_1f_solver_starts_on_live_idle_joy_ignore():
    memory = _solver_memory(joy=1)
    env = _vr_env(memory)
    taps = []

    def tap(button, *_args):
        taps.append(button)
        raise RuntimeError('stop-after-gate')

    env._tap_scripted_button = tap
    try:
        env._try_solve_victory_road_1f_boulder()
    except RuntimeError:
        pass
    assert taps == ['down']
    assert memory[train.ADDR_TEXT_BOX] == 0


def test_victory_road_1f_solver_rejects_script_lock_and_text():
    for kwargs in ({'joy': 0xFF}, {'font': 1}, {'battle': 1}, {'y': 17, 'x': 8}):
        memory = _solver_memory(**kwargs)
        env = _vr_env(memory)
        env._party_index_with_move = lambda _move_id: 1
        assert env._try_solve_victory_road_1f_boulder() is False


def test_victory_road_1f_solver_clears_idle_joy_ignore_before_down():
    memory = _solver_memory(joy=1)
    env = _vr_env(memory)
    env._party_index_with_move = lambda _move_id: 0
    taps = []

    def tap(button, *_args):
        taps.append((button, int(memory[train.ADDR_TEXT_BOX])))
        raise RuntimeError('stop-after-first-tap')

    env._tap_scripted_button = tap
    try:
        env._try_solve_victory_road_1f_boulder()
    except RuntimeError:
        pass
    assert taps[0] == ('down', 0)
    assert memory[train.ADDR_TEXT_BOX] == 0


def test_victory_road_1f_post_switch_route_reloads_then_walks_to_2f():
    back = train.VICTORY_ROAD_1F_SWITCH_TO_ENTRANCE_STEPS
    ahead = train.VICTORY_ROAD_1F_ENTRANCE_TO_2F_STEPS
    assert back[0][:4] == (108, 12, 17, 2)
    assert ahead[0][:4] == (108, 17, 8, 0)
    assert ahead[-1] == (108, 1, 2, 2, 194)
    rules = [
        rule for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
        if rule.get('map') == 108
        and rule.get('require_event_bits') == (
            train.EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH,
        )
    ]
    assert any(rule.get('target') == (12, 17) for rule in rules)
    assert any(
        rule.get('target') == (1, 2) and rule.get('destination_map') == 194
        for rule in rules
    )
