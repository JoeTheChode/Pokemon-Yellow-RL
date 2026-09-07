import train


class FakePyBoy:
    def __init__(self, memory):
        self.memory = memory


def _set_moves(memory, party_moves, party_pp, battle_moves=None, battle_pp=None):
    memory[train.ADDR_PARTY_SIZE] = 1
    for idx, move_id in enumerate(party_moves):
        memory[train.PARTY_MOVE_ID_ADDRS[0][idx]] = move_id
        memory[train.PARTY_MOVE_PP_ADDRS[0][idx]] = party_pp[idx]
    battle_moves = list(party_moves if battle_moves is None else battle_moves)
    battle_pp = list(party_pp if battle_pp is None else battle_pp)
    for idx, move_id in enumerate(battle_moves):
        memory[train.ADDR_BATTLE_MON_MOVE_IDS[idx]] = move_id
        memory[train.ADDR_BATTLE_MON_PP[idx]] = battle_pp[idx]


def _make_env(party_moves, party_pp, battle_moves=None, battle_pp=None):
    memory = bytearray(0x10000)
    _set_moves(memory, party_moves, party_pp, battle_moves, battle_pp)
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = FakePyBoy(memory)
    env.quest_phase = train.FULLGAME_BROCK_QUEST_PHASE
    return env


def _party_moves(memory):
    return [int(memory[addr]) for addr in train.PARTY_MOVE_ID_ADDRS[0]]


def _battle_moves(memory):
    return [int(memory[addr]) for addr in train.ADDR_BATTLE_MON_MOVE_IDS]


def test_live_brock_moveset_promotes_swift_without_deleting_thundershock():
    # Frontier dump 2026-08-27: party TS/QA/Slam/TW, battle TS/QA/TailWhip/TW.
    env = _make_env(
        [84, 98, 21, 86],
        [30, 30, 20, 20],
        battle_moves=[84, 98, 39, 86],
        battle_pp=[30, 30, 30, 20],
    )
    env._ensure_brock_attack_ready(allow_electric=False)
    memory = env.pyboy.memory
    assert memory[train.PARTY_MOVE_ID_ADDRS[0][0]] == train.PEWTER_FALLBACK_ATTACK_MOVE_ID
    assert memory[train.ADDR_BATTLE_MON_MOVE_IDS[0]] == train.PEWTER_FALLBACK_ATTACK_MOVE_ID
    assert train.GEN1_THUNDERSHOCK_MOVE_ID in _party_moves(memory)
    assert 21 in _party_moves(memory)
    assert train.GEN1_THUNDERSHOCK_MOVE_ID in _battle_moves(memory)


def test_ground_safe_prefers_ice_beam_over_swift():
    env = _make_env([84, 129, 58, 86], [15, 20, 10, 20])
    env._ensure_brock_attack_ready(allow_electric=False)
    assert env.pyboy.memory[train.PARTY_MOVE_ID_ADDRS[0][0]] == 58
    assert env.pyboy.memory[train.ADDR_BATTLE_MON_MOVE_IDS[0]] == 58


def test_electric_allowed_still_promotes_thunderbolt():
    env = _make_env([129, 85, 98, 86], [20, 15, 30, 20])
    env._ensure_brock_attack_ready(allow_electric=True)
    assert env.pyboy.memory[train.PARTY_MOVE_ID_ADDRS[0][0]] == 85
    assert env.pyboy.memory[train.ADDR_BATTLE_MON_MOVE_IDS[0]] == 85


def test_trainer_battle_bans_electric_in_pewter_gym():
    assert train.trainer_battle_allows_electric(54, train.FULLGAME_BROCK_QUEST_PHASE) is False


def test_brock_recovery_applies_once_after_geodude_faints():
    env = _make_env([129, 84, 98, 86], [20, 30, 30, 20])
    memory = env.pyboy.memory
    env._brock_emergency_heals_used = 0
    env.trainer_enemy_faints = 1
    memory[train.ADDR_MAP_ID] = 54
    memory[train.ADDR_POS_A] = 2
    memory[train.ADDR_POS_B] = 4
    memory[train.ADDR_BATTLE_FLAG] = 2
    memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 38
    memory[train.ADDR_ACTIVE_MON_MAX_HP_LO] = 50

    assert env._use_brock_emergency_heal() is True
    assert memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] == 50
    assert memory[train.PARTY_CUR_HP_ADDRS[0][1]] == 50
    assert env._use_brock_emergency_heal() is False


def test_brock_recovery_does_not_apply_before_first_enemy_faint():
    env = _make_env([129, 84, 98, 86], [20, 30, 30, 20])
    memory = env.pyboy.memory
    env._brock_emergency_heals_used = 0
    env.trainer_enemy_faints = 0
    memory[train.ADDR_MAP_ID] = 54
    memory[train.ADDR_POS_A] = 2
    memory[train.ADDR_POS_B] = 4
    memory[train.ADDR_BATTLE_FLAG] = 2
    memory[train.ADDR_ACTIVE_MON_CUR_HP_LO] = 38
    memory[train.ADDR_ACTIVE_MON_MAX_HP_LO] = 50

    assert env._use_brock_emergency_heal() is False
