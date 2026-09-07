import inspect

import train


def test_normal_battle_damage_resets_recovery_stall_age():
    assert train.next_trainer_battle_stall_steps(2, 24, 32, 29, 29, 211) == 0
    assert train.next_trainer_battle_stall_steps(2, 24, 24, 25, 29, 211) == 0


def test_only_consecutive_unchanged_trainer_steps_age_toward_recovery():
    assert train.next_trainer_battle_stall_steps(2, 24, 24, 29, 29, 17) == 18
    assert train.next_trainer_battle_stall_steps(0, 24, 24, 29, 29, 17) == 0


def test_recovery_burst_still_unwinds_a_genuine_menu_stall():
    assert train.trainer_battle_recovery_action(True, 29, 127) is None
    assert train.trainer_battle_recovery_action(True, 29, 128) == "b"
    assert train.trainer_battle_recovery_action(True, 29, 136) is None


def test_trainer_recovery_never_overrides_a_wild_battle_action():
    # The step loop passes trainer_battle_active_before_action. A wild battle
    # must therefore be False here even when the generic battle flag is live.
    assert train.trainer_battle_recovery_action(False, 0, 0) is None
    step_source = inspect.getsource(train.PokemonYellowEnv.step)
    assert (
        "trainer_battle_recovery_action(\n"
        "            trainer_battle_active_before_action,"
    ) in step_source
