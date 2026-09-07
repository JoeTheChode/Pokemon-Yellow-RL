import train


def test_victory_road_1f_pending_warp_detects_route23_cave_door():
    assert train.victory_road_1f_pending_warp(108, 31, 4) is True
    assert train.victory_road_1f_pending_warp(108, 17, 8) is False
    assert train.victory_road_1f_pending_warp(108, 14, 5) is False
    assert train.victory_road_1f_pending_warp(34, 31, 4) is False


def test_victory_road_1f_pending_warp_is_not_a_safe_frontier():
    assert train.is_safe_post_brock_swarm_frontier(108, 31, 4) is False
    assert train.is_safe_post_brock_swarm_frontier(108, 17, 8) is True


def test_victory_road_1f_boulder_approach_ends_on_solver_tile():
    route = train.VICTORY_ROAD_1F_TO_BOULDER_STEPS
    assert route[0][:4] == (108, 17, 8, 0)
    assert route[-1][:4] == (108, 14, 4, 3)


def test_victory_road_1f_switch_walk_returns_to_entrance():
    route = train.VICTORY_ROAD_1F_SWITCH_TO_ENTRANCE_STEPS
    assert route[0][:4] == (108, 12, 17, 2)
    assert route[-1][:4] == (108, 17, 9, 2)


def test_victory_road_1f_stairs_are_left_from_1_2_not_up_from_1_1():
    route = train.VICTORY_ROAD_1F_ENTRANCE_TO_2F_STEPS
    assert route[0][:4] == (108, 17, 8, 0)
    assert route[-1] == (108, 1, 2, 2, 194)


def test_victory_road_1f_guidance_is_split_on_the_switch_event():
    boulder_rules = [
        rule for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
        if rule.get('map') == 108
        and rule.get('target') == (17, 8)
        and train.EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH
        in rule.get('unless_event_bits', ())
    ]
    opened_rules = [
        rule for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
        if rule.get('map') == 108
        and rule.get('target') == (1, 2)
        and train.EVENT_VICTORY_ROAD_1_BOULDER_ON_SWITCH
        in rule.get('require_event_bits', ())
    ]
    stale_up_on_warp = [
        rule for rule in train.CINNABAR_MANSION_GYM_ACTION_GUIDANCE
        if rule.get('map') == 108
        and rule.get('target') == (1, 1)
        and rule.get('action') == 0
    ]
    assert boulder_rules
    assert opened_rules
    assert opened_rules[0]['action'] == 2
    assert opened_rules[0].get('destination_map') == 194
    assert not stale_up_on_warp


def test_pending_warp_does_not_force_a_on_joy_ignore():
    memory = bytearray(0x10000)
    memory[train.ADDR_MAP_ID] = 108
    memory[train.ADDR_POS_A] = 31
    memory[train.ADDR_POS_B] = 4
    memory[train.ADDR_TEXT_BOX] = 0xFF
    memory[train.ADDR_BATTLE_FLAG] = 0
    env = object.__new__(train.PokemonYellowEnv)
    env.pyboy = type('FakePyBoy', (), {'memory': memory})()
    env.quest_phase = train.FULLGAME_VICTORY_ROAD_2F_QUEST_PHASE
    env.action_guidance = []
    env.action_guidance_requires_overworld = True
    env.completed_run_recovery_after_steps = 0
    env.same_position_step_count = 16

    def noop(*_args, **_kwargs):
        return False

    env._clear_stale_overworld_joy_ignore = noop
    env._settle_bills_house_pending_warp = noop
    env._settle_cerulean_trash_house_pending_warp = noop
    env._settle_cerulean_bike_ledge_pocket = noop
    env._restore_glitched_map57_identity = noop
    env._settle_bike_shop_pending_warp = noop
    env._settle_vermilion_center_pending_warp = noop
    env._settle_vermilion_gym_pending_warp = noop
    env._settle_vermilion_side_house_pending_warp = noop
    env._settle_rock_tunnel_center_pending_warp = noop
    env._settle_victory_road_1f_pending_warp = noop
    env._clear_rock_tunnel_overworld_joy_ignore = noop
    env._settle_saffron_gym_exit_door = noop
    env._heal_if_exiting_rock_tunnel = noop
    env._vermilion_gym_forced_action = lambda *_args, **_kwargs: None

    assert env._matching_forced_action_guidance(108, 31, 4, 0) is None
