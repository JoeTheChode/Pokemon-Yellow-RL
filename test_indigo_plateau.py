import train


def _blank_indigo_env(phase, map_id=174):
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = map_id
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = phase
    env._indigo_plateau_resources_prepared = False
    return env, memory


def test_route23_north_to_indigo_enters_map_9():
    route = train.ROUTE23_NORTH_TO_INDIGO_STEPS
    assert route[0][:4] == (34, 30, 18, 0)
    assert route[-1] == (34, 0, 10, 0, 9)


def test_indigo_plateau_door_is_up_from_y6():
    nines = [step for step in train.INDIGO_PLATEAU_TO_LOBBY_STEPS if step[0] == 9]
    tens = [step for step in nines if step[1:][:2] == (17, 10)]
    nines_col = [step for step in nines if step[1:][:2] == (17, 9)]
    assert tens and tens[0][3] == 0
    assert nines_col and nines_col[0][3] == 0
    assert nines[-1][3] == 0
    assert nines[-1][4] == 174
    assert nines[-1][1] == 6


def test_indigo_lobby_mat_cuts_over_to_nurse_aisle():
    route = train.INDIGO_LOBBY_MAT_TO_AISLE_STEPS
    assert (174, 11, 7, 0, None) in route or route[0][:4] == (174, 11, 7, 0)
    assert any(step[:4] == (174, 7, 4, 2) for step in route)


def test_saved_lobby_route_enters_loreleis_room():
    route = train.INDIGO_LOBBY_TO_LORELEI_STEPS
    assert route[0][:4] == (174, 11, 7, 0)
    assert len(route) == 20
    assert any(step[:4] == (174, 3, 3, 0) for step in route)
    assert route[-1] == (174, 1, 8, 0, 245)


def test_indigo_arrival_skips_lead_hp_floor():
    assert train.swarm_required_lead_hp_fraction(
        0.4, train.FULLGAME_INDIGO_PLATEAU_QUEST_PHASE
    ) == 0.0
    assert train.swarm_required_lead_hp_fraction(
        0.4, train.FULLGAME_INDIGO_LOBBY_QUEST_PHASE
    ) == 0.0
    assert train.swarm_required_lead_hp_fraction(
        0.4, train.FULLGAME_HEAL_BEFORE_ELITE_FOUR_QUEST_PHASE
    ) == 0.0
    assert train.swarm_required_lead_hp_fraction(0.4, 0) == 0.4


def test_final_league_handoffs_skip_only_the_lead_hp_floor():
    for phase in (
        train.FULLGAME_BEAT_CHAMPION_QUEST_PHASE,
        train.FULLGAME_ENTER_HALL_OF_FAME_QUEST_PHASE,
    ):
        assert train.swarm_required_lead_hp_fraction(0.4, phase) == 0.0
        assert train.swarm_required_party_hp_fraction(0.4, phase) == 0.4


def test_indigo_center_pulls_mart_band_left_to_aisle():
    lefts = [
        rule for rule in train.INDIGO_PLATEAU_CENTER_ACTION_GUIDANCE
        if rule.get('map') == 174
        and rule.get('target') == (5, 10)
        and rule.get('action') == 2
    ]
    assert lefts
    assert lefts[0].get('max_party_hp_fraction') == 0.99


def test_indigo_nurse_interaction_faces_joy_before_talking():
    nurse = [
        rule for rule in train.INDIGO_PLATEAU_CENTER_ACTION_GUIDANCE
        if rule.get('map') == 174
        and rule.get('target') == (3, 3)
        and rule.get('action') == 4
    ]
    assert nurse
    assert nurse[0].get('face_action') == 0
    assert nurse[0].get('force_action') is True


def test_indigo_checkpoint_restores_hp_without_risking_one_way_door():
    env, memory = _blank_indigo_env(
        train.FULLGAME_HEAL_BEFORE_ELITE_FOUR_QUEST_PHASE
    )
    memory[train.ADDR_PARTY_SIZE] = 1
    cur_hi, cur_lo = train.PARTY_CUR_HP_ADDRS[0]
    max_hi, max_lo = train.PARTY_MAX_HP_ADDRS[0]
    memory[cur_hi], memory[cur_lo] = 0, 1
    memory[max_hi], memory[max_lo] = 0, 80

    assert env._prepare_indigo_plateau_resources() is True
    assert (memory[cur_hi] << 8) | memory[cur_lo] == 80
    assert env._indigo_plateau_resources_prepared is True
    assert env._prepare_indigo_plateau_resources() is False


def test_indigo_checkpoint_restore_is_strictly_scoped():
    wrong_phase, _ = _blank_indigo_env(
        train.FULLGAME_ENTER_LORELEIS_ROOM_QUEST_PHASE
    )
    wrong_map, _ = _blank_indigo_env(
        train.FULLGAME_HEAL_BEFORE_ELITE_FOUR_QUEST_PHASE, map_id=245
    )
    assert wrong_phase._prepare_indigo_plateau_resources() is False
    assert wrong_map._prepare_indigo_plateau_resources() is False


def test_lorelei_and_agatha_have_deterministic_battle_handoffs():
    rules = train.LORELEI_AGATHA_ROOM_ACTION_GUIDANCE
    for map_id, event_bit in (
        (245, train.EVENT_BEAT_LORELEI),
        (247, train.EVENT_BEAT_AGATHA),
    ):
        assert any(
            rule['map'] == map_id
            and rule['target'] == (11, 4)
            and rule['action'] == 0
            and rule['unless_event_bits'] == (event_bit,)
            for rule in rules
        )
        talk = [
            rule for rule in rules
            if rule['map'] == map_id and rule['target'] == (3, 5)
        ]
        assert talk
        assert talk[0]['action'] == 4
        assert talk[0]['face_action'] == 0


def test_all_league_room_exits_join_the_next_saved_battle():
    expected = (
        (train.LORELEI_TO_BRUNO_STEPS, 245, 246),
        (train.BRUNO_TO_AGATHA_STEPS, 246, 247),
        (train.AGATHA_TO_LANCE_STEPS, 247, 113),
        (train.LANCE_TO_CHAMPION_STEPS, 113, 120),
    )
    for route, source, destination in expected:
        assert route[0][0] == source
        assert route[-1][0] == source
        assert route[-1][4] == destination
    assert train.LORELEI_TO_BRUNO_STEPS[0][:4] == (245, 4, 2, 3)
    assert (245, 2, 6, 1, None) in train.LORELEI_TO_BRUNO_STEPS
    assert (245, 3, 5, 2, None) in train.LORELEI_TO_BRUNO_STEPS


def test_champion_completion_holds_a_clean_hof_frontier():
    rules = train.CHAMPION_FRONTIER_HOLD_ACTION_GUIDANCE
    assert {rule['target'] for rule in rules} == {(3, 4), (2, 4)}
    actions = {rule['target']: rule['action'] for rule in rules}
    assert actions == {(3, 4): 4, (2, 4): 5}
    for rule in rules:
        assert rule['require_event_bits'] == (
            train.EVENT_BEAT_CHAMPION_RIVAL,
        )
        assert rule['episode_start_phases'] == frozenset({
            train.FULLGAME_BEAT_CHAMPION_QUEST_PHASE,
        })


def test_indigo_guidance_covers_route23_plateau_and_lobby():
    rules = train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
    r23 = [
        rule for rule in rules
        if rule.get('map') == 34
        and rule.get('target') == (30, 18)
        and train.FULLGAME_INDIGO_PLATEAU_QUEST_PHASE
        in (rule.get('quest_phases') or ())
    ]
    plateau = [
        rule for rule in rules
        if rule.get('map') == 9
        and rule.get('destination_map') == 174
    ]
    mat = [
        rule for rule in rules
        if rule.get('map') == 174
        and rule.get('target') == (11, 7)
    ]
    assert r23 and r23[0]['action'] == 0
    assert plateau
    assert mat and mat[0]['action'] == 0
