from pathlib import Path

import train


class FakePyBoy:
    def __init__(self, current_memory, base_memory):
        self.memory = current_memory
        self._base_memory = base_memory
        self.loads = 0

    def load_state(self, _handle):
        self.memory[:] = self._base_memory
        self.loads += 1


def _set_event(memory, event_bit):
    address = train.ADDR_EVENT_FLAGS_START + event_bit // 8
    memory[address] |= 1 << (event_bit % 8)


def _make_env(
    tmp_path, monkeypatch, *, phase=None, event=True, badges=0xFF,
    route22_ready=True,
):
    base = bytearray(0x10000)
    base[train.ADDR_MAP_ID] = 1
    base[train.ADDR_POS_A] = 26
    base[train.ADDR_POS_B] = 23
    current = bytearray(0x10000)
    current[train.ADDR_MAP_ID] = 45
    current[train.ADDR_BATTLE_FLAG] = 0
    current[train.ADDR_PARTY_SIZE] = 5
    current[train.ADDR_BADGES] = badges
    current[train.ADDR_BAG_ITEMS] = 0x2B
    current[train.ADDR_LAST_BLACKOUT_MAP] = 8
    current[train.ADDR_TOWN_VISITED_FLAGS[0]] = 0xFF
    current[train.ADDR_TOWN_VISITED_FLAGS[1]] = 0x05
    if event:
        _set_event(current, 81)
        if route22_ready:
            _set_event(current, train.EVENT_ROUTE22_RIVAL_BATTLE_AVAILABLE)
            _set_event(current, train.EVENT_ROUTE22_RIVAL_WANTS_BATTLE)
    state_path = tmp_path / '05c_viridian_north.state'
    state_path.write_bytes(b'fixture')
    monkeypatch.setattr(train, 'PROJECT_ROOT', Path(tmp_path))
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = FakePyBoy(current, base)
    env.quest_phase = (
        train.FULLGAME_VIRIDIAN_GYM_EXIT_QUEST_PHASE if phase is None else phase
    )
    return env


def test_viridian_gym_exit_recovery_preserves_progress(tmp_path, monkeypatch):
    env = _make_env(tmp_path, monkeypatch)

    assert env._recover_viridian_gym_exit_softlock() is True
    memory = env.pyboy.memory
    assert env.pyboy.loads == 1
    assert (
        memory[train.ADDR_MAP_ID],
        memory[train.ADDR_POS_A],
        memory[train.ADDR_POS_B],
    ) == (1, 26, 23)
    assert memory[train.ADDR_PARTY_SIZE] == 5
    assert memory[train.ADDR_BADGES] == 0xFF
    assert memory[train.ADDR_BAG_ITEMS] == 0x2B
    assert memory[train.ADDR_LAST_BLACKOUT_MAP] == 8
    assert tuple(memory[address] for address in train.ADDR_TOWN_VISITED_FLAGS) == (
        0xFF,
        0x05,
    )
    assert env._event_flag_is_set(81)


def test_viridian_gym_exit_recovery_is_strictly_scoped(tmp_path, monkeypatch):
    env = _make_env(tmp_path, monkeypatch, phase=0)
    assert env._recover_viridian_gym_exit_softlock() is False
    assert env.pyboy.loads == 0

    env = _make_env(tmp_path, monkeypatch, event=False)
    assert env._recover_viridian_gym_exit_softlock() is False
    assert env.pyboy.loads == 0


def test_viridian_gym_exit_waits_for_authentic_post_battle_tail(
    tmp_path, monkeypatch,
):
    env = _make_env(tmp_path, monkeypatch, badges=0x7F)
    assert env._recover_viridian_gym_exit_softlock() is False
    assert env.pyboy.loads == 0

    env = _make_env(tmp_path, monkeypatch, route22_ready=False)
    assert env._recover_viridian_gym_exit_softlock() is False
    assert env.pyboy.loads == 0


def test_viridian_gym_exit_recovery_gate_resets_each_episode(
    tmp_path, monkeypatch,
):
    env = _make_env(tmp_path, monkeypatch)
    env._viridian_gym_exit_recovered = True

    env._reset_per_episode_gates()

    assert env._viridian_gym_exit_recovered is False
    assert env._recover_viridian_gym_exit_softlock() is True
    assert env.pyboy.loads == 1


def test_training_fly_repairs_legacy_visited_town_flags():
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 33
    memory[train.ADDR_BATTLE_FLAG] = 0
    memory[train.ADDR_TEXT_BOX] = 0
    memory[train.ADDR_PARTY_SIZE] = 1
    memory[train.PARTY_MOVE_ID_ADDRS[0][0]] = train.GEN1_FLY_MOVE_ID
    memory[train.ADDR_TOWN_VISITED_FLAGS[0]] = 0x03
    memory[train.ADDR_TOWN_VISITED_FLAGS[1]] = 0x00
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = train.FULLGAME_CINNABAR_TRAINING_FLY_QUEST_PHASE
    a_presses = 0

    def tap(button, *_args):
        nonlocal a_presses
        if button == 'a':
            a_presses += 1
            if a_presses == 4:
                memory[train.ADDR_MAP_ID] = 8

    env._tap_scripted_button = tap

    assert env._try_fly_to_cinnabar_for_training() is True
    assert tuple(memory[address] for address in train.ADDR_TOWN_VISITED_FLAGS) == (
        0xFF,
        0x05,
    )
