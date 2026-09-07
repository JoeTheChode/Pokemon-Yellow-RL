import train


class FakePyBoy:
    def __init__(self, memory):
        self.memory = memory


def _set_event(memory, event_bit):
    address = train.ADDR_EVENT_FLAGS_START + event_bit // 8
    memory[address] |= 1 << (event_bit % 8)


def _make_env(*, phase=None, map_id=61, battle=0, event=False):
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = map_id
    memory[train.ADDR_BATTLE_FLAG] = battle
    memory[train.ADDR_PARTY_SIZE] = 1
    cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
    max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[0]
    memory[cur_hi], memory[cur_lo] = divmod(26, 256)
    memory[max_hi], memory[max_lo] = divmod(54, 256)
    memory[train.PARTY_MOVE_ID_ADDRS[0][0]] = 21  # Slam
    memory[train.PARTY_MOVE_PP_ADDRS[0][0]] = 6
    if event:
        _set_event(memory, train.MT_MOON_JESSIE_JAMES_EVENT_BIT)
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = FakePyBoy(memory)
    env.quest_phase = (
        train.FULLGAME_JESSIE_JAMES_QUEST_PHASE if phase is None else phase
    )
    return env


def test_jessie_james_preparation_restores_hp_and_pp_once_per_episode():
    env = _make_env()

    assert env._prepare_mt_moon_jessie_james_resources() is True
    memory = env.pyboy.memory
    cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
    assert (memory[cur_hi] << 8) | memory[cur_lo] == 54
    assert memory[train.PARTY_MOVE_PP_ADDRS[0][0]] == 20
    assert env._prepare_mt_moon_jessie_james_resources() is False


def test_jessie_james_preparation_is_strictly_scoped():
    assert _make_env(phase=0)._prepare_mt_moon_jessie_james_resources() is False
    assert _make_env(map_id=60)._prepare_mt_moon_jessie_james_resources() is False
    assert _make_env(battle=2)._prepare_mt_moon_jessie_james_resources() is False
    assert _make_env(event=True)._prepare_mt_moon_jessie_james_resources() is False
