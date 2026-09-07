import train


def _rule_at(map_id, y, x):
    return next(
        rule
        for rule in train.SCOPED_ROUTE5_TO_UNDERGROUND_ACTION_GUIDANCE
        if int(rule["map"]) == map_id and tuple(rule["target"]) == (y, x)
    )


def _first_return_rule_at(map_id, y, x):
    return next(
        rule
        for rule in train.RETURN_TO_CERULEAN_ACTION_GUIDANCE
        if int(rule["map"]) == map_id and tuple(rule["target"]) == (y, x)
    )


def test_route5_underground_guidance_is_scoped_to_phase_67():
    assert train.FULLGAME_ROUTE5_UNDERGROUND_QUEST_PHASE == 67
    assert train.FULLGAME_QUEST_WAYPOINTS[67]["name"] == (
        "enter_route_5_underground_path"
    )
    assert train.SCOPED_ROUTE5_TO_UNDERGROUND_ACTION_GUIDANCE
    assert {
        rule["quest_phases"]
        for rule in train.SCOPED_ROUTE5_TO_UNDERGROUND_ACTION_GUIDANCE
    } == {frozenset({67})}


def test_route5_underground_guidance_covers_exact_frontier_to_door():
    # The phase-66 frontier is the north connection at (0,18).
    assert int(_rule_at(16, 0, 18)["action"]) == 1
    assert int(_rule_at(16, 23, 18)["action"]) == 2
    assert int(_rule_at(16, 23, 15)["action"]) == 1
    assert int(_rule_at(16, 28, 15)["action"]) == 3
    door = _rule_at(16, 28, 17)
    assert int(door["action"]) == 0
    assert int(door["destination_map"]) == 71


def test_return_to_cerulean_prioritizes_verified_route5_reverse_leg():
    # Forced guidance is first-match-wins.  The broad recovery fan also owns
    # these coordinates, but pointing it east at (28,17) sends workers onto
    # the Underground Path entrance latch at (28,18), where UP cannot move.
    assert int(_first_return_rule_at(16, 28, 18)["action"]) == 2
    assert int(_first_return_rule_at(16, 28, 17)["action"]) == 2
    assert int(_first_return_rule_at(16, 28, 16)["action"]) == 2
    assert int(_first_return_rule_at(16, 28, 15)["action"]) == 0
    assert int(_first_return_rule_at(16, 27, 15)["action"]) == 0
    assert int(_first_return_rule_at(16, 23, 15)["action"]) == 3
