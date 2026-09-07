import json

import train


class FakePyBoy:
    def __init__(self, memory, base_memory=None):
        self.memory = memory
        self.base_memory = base_memory
        self.ticks = 0
        self.loads = 0

    def tick(self, ticks):
        self.ticks += ticks

    def load_state(self, _handle):
        assert self.base_memory is not None
        self.memory[:] = self.base_memory
        self.loads += 1


def _set_event(memory, event_bit):
    address = train.ADDR_EVENT_FLAGS_START + event_bit // 8
    memory[address] |= 1 << (event_bit % 8)


def _make_env(*, phase=None, event=True, battle=0):
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 33
    memory[train.ADDR_BATTLE_FLAG] = battle
    memory[train.ADDR_PARTY_SIZE] = 2
    for index, maximum in enumerate((300, 170)):
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[index]
        max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[index]
        memory[cur_hi], memory[cur_lo] = divmod(0 if index == 0 else 169, 256)
        memory[max_hi], memory[max_lo] = divmod(maximum, 256)
        memory[train.PARTY_MOVE_ID_ADDRS[index][0]] = 85
        memory[train.PARTY_MOVE_PP_ADDRS[index][0]] = 1
    if event:
        _set_event(memory, train.EVENT_BEAT_ROUTE22_RIVAL)
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = FakePyBoy(memory)
    env.quest_phase = (
        train.FULLGAME_ROUTE23_QUEST_PHASE if phase is None else phase
    )
    return env


def test_route22_post_rival_win_is_restored_for_safe_frontier():
    env = _make_env()

    assert env._restore_route22_post_rival_resources() is True
    for index, maximum in enumerate((300, 170)):
        cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[index]
        assert (env.pyboy.memory[cur_hi] << 8) | env.pyboy.memory[cur_lo] == maximum
        assert env.pyboy.memory[train.PARTY_MOVE_PP_ADDRS[index][0]] == 15


def test_route22_post_rival_restore_is_strictly_scoped():
    assert _make_env(phase=0)._restore_route22_post_rival_resources() is False
    assert _make_env(event=False)._restore_route22_post_rival_resources() is False
    assert _make_env(battle=2)._restore_route22_post_rival_resources() is False


def test_route22_post_rival_rebuilds_legacy_frontier_runtime(tmp_path):
    env = _make_env()
    current = env.pyboy.memory
    base = bytearray(current)
    event_address = (
        train.ADDR_EVENT_FLAGS_START + train.EVENT_BEAT_ROUTE22_RIVAL // 8
    )
    base[event_address] &= ~(1 << (train.EVENT_BEAT_ROUTE22_RIVAL % 8))
    base[train.ADDR_ROUTE22_CUR_SCRIPT] = train.ROUTE22_RIVAL_BATTLE_SCRIPT
    env.pyboy = FakePyBoy(current, base)
    env.swarm_frontier_meta_path = tmp_path / 'frontier.json'
    env.swarm_frontier_state_path = tmp_path / 'frontier.state'
    env.swarm_frontier_meta_path.write_text(json.dumps({
        'quest_phase': train.FULLGAME_ROUTE22_RIVAL_QUEST_PHASE,
    }))
    env.swarm_frontier_state_path.write_bytes(b'fixture')

    assert env._restore_route22_post_rival_resources() is True
    assert env.pyboy.loads == 1
    assert env._route22_post_rival_runtime_recovered is True
    assert env._event_flag_is_set(train.EVENT_BEAT_ROUTE22_RIVAL)
    assert tuple(current[address] for address in train.ADDR_TOWN_VISITED_FLAGS) == (
        0,
        0,
    )


def test_route22_post_rival_connector_joins_existing_league_route():
    connector = train.ROUTE22_POST_RIVAL_TO_LEAGUE_ROUTE_STEPS
    league_route = train.ROUTE22_TO_LEAGUE_GATE_STEPS

    assert connector[0][:3] == (33, 6, 39)
    assert connector[-1][:4] == (33, 12, 36, 2)
    assert league_route[3][:3] == (33, 12, 35)
    connector_origins = {tuple(step[:3]) for step in connector}
    assert not connector_origins.intersection(
        tuple(step[:3]) for step in league_route
    )


def test_route22_tree_wall_uses_verified_north_detour():
    route = train.ROUTE22_TO_LEAGUE_GATE_STEPS
    start = next(
        index for index, step in enumerate(route)
        if step[:3] == (33, 5, 26)
    )

    assert tuple(step[:4] for step in route[start:start + 5]) == (
        (33, 5, 26, 0),
        (33, 4, 26, 2),
        (33, 4, 25, 2),
        (33, 4, 24, 1),
        (33, 5, 24, 2),
    )


def test_route22_post_rival_exit_clears_any_residual_script_substate():
    env = _make_env()
    memory = env.pyboy.memory
    memory[train.ADDR_ROUTE22_CUR_SCRIPT] = train.ROUTE22_RIVAL_BATTLE_SCRIPT
    memory[train.ADDR_TEXT_BOX] = 1
    memory[train.ADDR_FONT_LOADED] = 1
    _set_event(memory, train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE)
    _set_event(memory, train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE)
    down_presses = 0

    def tap(button, *_args):
        nonlocal down_presses
        assert button == 'down'
        down_presses += 1
        if down_presses == 2:
            memory[train.ADDR_POS_A] += 1

    env._tap_scripted_button = tap

    assert env._finish_route22_rival_exit() is True
    assert memory[train.ADDR_ROUTE22_CUR_SCRIPT] == train.ROUTE22_RIVAL_DONE_SCRIPT
    assert memory[train.ADDR_TEXT_BOX] == 0
    assert memory[train.ADDR_FONT_LOADED] == 0
    assert not env._event_flag_is_set(train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE)
    assert not env._event_flag_is_set(train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE)
    assert env.pyboy.ticks == 120
    assert down_presses == 2
    assert memory[train.ADDR_POS_A] == 1
    assert env._finish_route22_rival_exit() is False
