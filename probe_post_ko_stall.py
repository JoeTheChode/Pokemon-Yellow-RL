"""Empirically probe what actually happens when battle_flag stays active with
enemy_hp==0 and text_box==0 (the state _clear_stale_battle_flag_if_ready force-clears).

Loads the live 07_beat_brock.state read-only, mirrors the exact force-A battle
policy train.py uses for this stage, and looks for a real trainer/wild win.
Once the enemy faints, it captures a fine-grained memory trace across the
"stuck" window instead of guessing from disassembly alone.
"""
import random
import sys

from pyboy import PyBoy

import train
from project_paths import ROM_PATH, milestone_state_path

ADDR_MAP_ID = train.ADDR_MAP_ID
ADDR_POS_A = train.ADDR_POS_A
ADDR_POS_B = train.ADDR_POS_B
ADDR_BATTLE_FLAG = train.ADDR_BATTLE_FLAG
ADDR_ENEMY_HP = train.ADDR_ENEMY_HP
ADDR_TEXT_BOX = train.ADDR_TEXT_BOX
ADDR_REPEL = train.ADDR_REPEL
ADDR_LEVEL = train.ADDR_LEVEL

DUMP_LO = 0xCC00
DUMP_HI = 0xD080  # exclusive


def snapshot(memory):
    return bytes(memory[addr] for addr in range(DUMP_LO, DUMP_HI))


def diff(prev, cur):
    return [(DUMP_LO + i, prev[i], cur[i]) for i in range(len(cur)) if prev[i] != cur[i]]


def ensure_attack_ready(memory):
    move_addrs = train.PARTY_MOVE_ID_ADDRS[0]
    pp_addrs = train.PARTY_MOVE_PP_ADDRS[0]
    move_id = int(memory[move_addrs[0]])
    move_info = train.GEN1_MOVE_TABLE.get(move_id)
    if move_info is None or move_info['power'] <= 0:
        train._ensure_lead_attack_memory(memory)
        move_id = int(memory[move_addrs[0]])
        move_info = train.GEN1_MOVE_TABLE.get(move_id, train.GEN1_MOVE_TABLE[train.PEWTER_FALLBACK_ATTACK_MOVE_ID])
    pp = int(memory[pp_addrs[0]]) & 0x3F
    if pp <= 0:
        pp_ups = int(memory[pp_addrs[0]]) & 0xC0
        memory[pp_addrs[0]] = pp_ups | min(move_info['max_pp'], 0x3F)


def press(pyboy, button, press_ticks=8, release_ticks=16):
    pyboy.button_press(button)
    pyboy.tick(press_ticks)
    pyboy.button_release(button)
    pyboy.tick(release_ticks)


def main():
    random.seed(1234)
    state_file = milestone_state_path('07_beat_brock')
    pyboy = PyBoy(str(ROM_PATH), window='null')
    with open(state_file, 'rb') as f:
        pyboy.load_state(f)
    memory = pyboy.memory

    print(f"Loaded state: map={memory[ADDR_MAP_ID]} pos=({memory[ADDR_POS_A]},{memory[ADDR_POS_B]}) "
          f"level={memory[ADDR_LEVEL]} battle_flag={memory[ADDR_BATTLE_FLAG]}")

    reverse = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    directions = ['up', 'down', 'left', 'right']
    current_dir = random.choice(directions)
    steps_in_dir = 0
    home_map = int(memory[ADDR_MAP_ID])
    last_pos = (int(memory[ADDR_POS_A]), int(memory[ADDR_POS_B]))
    stuck_count = 0
    last_action_dir = None
    banned_dir = None
    banned_cooldown = 0
    retreating = 0
    retreat_dir = None

    search_budget = 40000
    found_battle = False
    for step in range(search_budget):
        memory[ADDR_REPEL] = 255
        battle_flag = int(memory[ADDR_BATTLE_FLAG])
        if battle_flag in (1, 2):
            found_battle = True
            print(f"\nBattle started at search step {step}: battle_flag={battle_flag} "
                  f"map={memory[ADDR_MAP_ID]} enemy_hp={memory[ADDR_ENEMY_HP]}")
            break

        cur_map = int(memory[ADDR_MAP_ID])
        cur_pos = (int(memory[ADDR_POS_A]), int(memory[ADDR_POS_B]))

        if cur_map != home_map and retreating == 0:
            # Left the intended map (e.g. wandered back toward the Route 3
            # boulder maze) -- ban whichever direction caused this for a long
            # cooldown and retreat the way we came instead of trying to solve
            # the maze (already known-hard per project memory).
            banned_dir = last_action_dir
            banned_cooldown = 400
            retreating = 6
            retreat_dir = reverse.get(last_action_dir, random.choice(directions))
            print(f"  left home_map={home_map} into map={cur_map} via '{last_action_dir}' "
                  f"at pos={cur_pos} -- retreating via '{retreat_dir}' and banning '{last_action_dir}' for a while")

        if retreating > 0:
            current_dir = retreat_dir
            steps_in_dir = 0
            retreating -= 1
        elif cur_pos == last_pos:
            stuck_count += 1
            if stuck_count > 6:
                choices = [d for d in directions if d != current_dir]
                if banned_cooldown > 0 and banned_dir in choices and len(choices) > 1:
                    choices = [d for d in choices if d != banned_dir]
                current_dir = random.choice(choices)
                steps_in_dir = 0
                stuck_count = 0
        else:
            stuck_count = 0
        last_pos = cur_pos

        if banned_cooldown > 0:
            banned_cooldown -= 1

        steps_in_dir += 1
        if steps_in_dir > random.randint(8, 24) and retreating == 0:
            choices = directions
            if banned_cooldown > 0 and banned_dir is not None:
                choices = [d for d in directions if d != banned_dir]
            current_dir = random.choice(choices)
            steps_in_dir = 0
        last_action_dir = current_dir
        press(pyboy, current_dir)
        if step % 5000 == 0 and step:
            print(f"  ... still searching for a battle, step {step}, "
                  f"map={memory[ADDR_MAP_ID]} pos=({memory[ADDR_POS_A]},{memory[ADDR_POS_B]})")

    if not found_battle:
        print(f"No battle triggered within {search_budget} search steps. Giving up this run.")
        pyboy.stop()
        sys.exit(1)

    # Fight it out with the exact same force-A policy train.py uses.
    battle_step_cap = 4000
    saw_zero_hp = False
    post_ko_snapshots = []
    prev_snap = None
    for bstep in range(battle_step_cap):
        memory[ADDR_REPEL] = 255
        battle_flag = int(memory[ADDR_BATTLE_FLAG])
        enemy_hp = int(memory[ADDR_ENEMY_HP])
        text_box = int(memory[ADDR_TEXT_BOX])

        if battle_flag == 0:
            print(f"\nBattle fully resolved naturally at battle-step {bstep} "
                  f"(battle_flag back to 0). Post-KO steps captured: {len(post_ko_snapshots)}")
            break

        if enemy_hp == 0 and not saw_zero_hp:
            saw_zero_hp = True
            print(f"\nEnemy HP hit 0 at battle-step {bstep}. Entering fine-grained capture phase.")

        if saw_zero_hp:
            cur_snap = snapshot(memory)
            changed = diff(prev_snap, cur_snap) if prev_snap is not None else []
            post_ko_snapshots.append((bstep, battle_flag, enemy_hp, text_box, changed))
            prev_snap = cur_snap

            if len(post_ko_snapshots) in (65, 120, 200, 300):
                stuck_window = post_ko_snapshots[-40:]
                all_changed_addrs = sorted({addr for *_, ch in stuck_window for addr, _, _ in ch})
                print(f"  [post-KO step {len(post_ko_snapshots)}] battle_flag={battle_flag} "
                      f"enemy_hp={enemy_hp} text_box={text_box} "
                      f"addrs_changed_in_last_40_steps={[hex(a) for a in all_changed_addrs]}")

            if len(post_ko_snapshots) == 200:
                print("\n>>> Trying an UP press + A to see if a menu cursor is what's stuck <<<")
                press(pyboy, 'up')
                press(pyboy, 'a')
                after = snapshot(memory)
                changed_after_input = diff(cur_snap, after)
                print(f"    battle_flag={memory[ADDR_BATTLE_FLAG]} enemy_hp={memory[ADDR_ENEMY_HP]} "
                      f"text_box={memory[ADDR_TEXT_BOX]}")
                print(f"    bytes changed by UP+A: {[(hex(a), p, n) for a, p, n in changed_after_input]}")

        ensure_attack_ready(memory)
        press(pyboy, 'a')

    else:
        print(f"\nHit battle_step_cap ({battle_step_cap}) still in battle. "
              f"Total post-KO steps observed: {len(post_ko_snapshots)}")

    if post_ko_snapshots:
        first = post_ko_snapshots[0]
        last = post_ko_snapshots[-1]
        print(f"\nSummary: first post-KO snapshot at bstep={first[0]}, last at bstep={last[0]} "
              f"({len(post_ko_snapshots)} total). Final battle_flag={last[1]} enemy_hp={last[2]} "
              f"text_box={last[3]}")

    pyboy.stop()


if __name__ == '__main__':
    main()
