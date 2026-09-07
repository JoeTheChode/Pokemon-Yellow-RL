import train


def _first_inbound_action(y, x):
    for rule in train.SS_ANNE_CAPTAIN_ACTION_GUIDANCE:
        if (
            int(rule["map"]) == 95
            and tuple(rule["target"]) == (y, x)
            and not rule.get("require_event_bits")
        ):
            return int(rule["action"])
    raise AssertionError(f"missing inbound S.S. Anne 1F guidance at {(y, x)}")


def test_ss_anne_1f_gangway_connector_avoids_south_wall():
    # Emulator replay from the live phase-75 frontier proved this connector:
    # descend the east column, sidestep the wall, then reach the cross-corridor.
    assert [_first_inbound_action(y, 27) for y in range(4)] == [1, 1, 1, 1]
    assert _first_inbound_action(4, 27) == 2
    assert _first_inbound_action(4, 26) == 1
    assert _first_inbound_action(5, 26) == 1
    assert _first_inbound_action(6, 26) == 1


def test_ss_anne_1f_cross_corridor_reaches_2f_stairs():
    assert _first_inbound_action(7, 26) == 2
    assert _first_inbound_action(7, 8) == 2
    assert _first_inbound_action(7, 7) == 0
    assert _first_inbound_action(6, 7) == 2
    assert _first_inbound_action(6, 3) == 2
    stairs = next(
        rule
        for rule in train.SS_ANNE_CAPTAIN_ACTION_GUIDANCE
        if int(rule["map"]) == 95
        and tuple(rule["target"]) == (6, 2)
        and int(rule.get("destination_map", -1)) == 96
    )
    assert int(stairs["action"]) == 0
