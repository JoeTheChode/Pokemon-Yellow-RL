import glob
import json
import os
import platform
import re
import shutil
import time
from collections import deque
from io import BytesIO
from pathlib import Path

_requested_hip_device = os.environ.get("POKEMON_HIP_VISIBLE_DEVICES")
if _requested_hip_device is not None:
    os.environ["HIP_VISIBLE_DEVICES"] = _requested_hip_device

import gymnasium as gym
import numpy as np
from PIL import Image
from gymnasium import spaces
from pyboy import PyBoy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
)

from project_paths import (
    CHECKPOINTS_DIR,
    MILESTONES_DIR as PROJECT_MILESTONES_DIR,
    PROGRESS_FILE as PROJECT_PROGRESS_FILE,
    ROM_PATH,
    SCREENSHOTS_DIR,
    START_STATE_PATH,
    atomic_write_json,
    ensure_runtime_dirs,
    latest_hit_glob,
    latest_hit_state_path,
    milestone_state_path,
    read_json_with_backup,
)
from yellow_navigation import map_labels, warp_entry_action
from stream_agent_wrapper import StreamWrapper

import torch as _torch
def _default_device():
    explicit = os.environ.get("POKEMON_TRAIN_DEVICE")
    if explicit:
        return explicit
    if _torch.cuda.is_available():
        return "cuda"
    return "cpu"

TRAIN_DEVICE = _default_device()
ALLOW_UNSUPPORTED_WINDOWS_GPU = os.environ.get("POKEMON_ALLOW_UNSUPPORTED_WINDOWS_GPU") == "1"
SHOW_RAW_SB3_TABLES = os.environ.get("POKEMON_SHOW_RAW_SB3_TABLES") == "1"
SHOW_MILESTONE_DEBUG = os.environ.get("POKEMON_SHOW_MILESTONE_DEBUG") == "1"
DEFAULT_N_ENVS = max(1, int(os.environ.get("POKEMON_N_ENVS", "4")))
# Optional live broadcast to PWhiddy's shared community map viewer
# (https://pwhiddy.github.io/pokerl-map-viz/) -- sends only (x,y,map_id) plus
# a display name/color to a third-party WebSocket server every ~300 steps
# per env. Off by default; opt in with POKEMON_STREAM_ENABLED=1.
STREAM_ENABLED = os.environ.get("POKEMON_STREAM_ENABLED") == "1"
STREAM_USER = os.environ.get("POKEMON_STREAM_USER", "SVER-PokemonYellow")
STREAM_COLOR = os.environ.get("POKEMON_STREAM_COLOR", "#FFD700")
CURRICULUM_VERSION = 2
PRE_PEWTER_BROCK_INDEX = 14
PEWTER_CURRICULUM_INSERTIONS = 3
# Resume from the freshest stage checkpoint by default. 06c/06d/07 previously
# forced unified-model restarts to escape a bad policy loop, but that discards
# the most recent stage-specific routing progress on every restart. 07_beat_brock
# spent steps 532M->708M+ stuck at 0/50 hits with this still enabled -- leaving
# it on would have thrown that progress away on the next resume.
FORCE_UNIFIED_RESUME_STAGES = set()
DEFAULT_VEC_ENV_KIND = os.environ.get(
    "POKEMON_VEC_ENV",
    "subproc" if DEFAULT_N_ENVS > 1 and platform.system() != "Windows" else "dummy",
).strip().lower()
DEFAULT_SUBPROC_START_METHOD = os.environ.get(
    "POKEMON_SUBPROC_START_METHOD",
    "forkserver" if platform.system() != "Windows" else "spawn",
).strip().lower()

OPPOSITE_ACTIONS = {
    0: 1,  # up <-> down
    1: 0,
    2: 3,  # left <-> right
    3: 2,
}

# Memory addresses for Pokemon Yellow (shifted -1 from Red/Blue)
ADDR_MAP_ID = 0xD35D
# Yellow shifts the Red/Blue coordinate bytes down by one: D360=y, D361=x.
ADDR_POS_A = 0xD360
ADDR_POS_B = 0xD361
ADDR_BADGES = 0xD356
ADDR_XP = 0xD179
ADDR_LEVEL = 0xD18B  # first Pokemon's actual level
ADDR_BATTLE_FLAG = 0xD057  # 0=overworld, 1=wild, 2=trainer
ADDR_ENEMY_HP = 0xCFE6     # enemy current HP (Yellow offset, -1 from Red/Blue)
ADDR_ENEMY_MAX_HP = 0xCFF4 # enemy max HP (same as Red/Blue)
ADDR_PARTY_SIZE = 0xD162
ADDR_TEXT_BOX = 0xCF13  # non-zero when dialogue/text box is active
ADDR_REPEL = 0xD078
ADDR_ACTIVE_MON_CUR_HP_HI = 0xD014
ADDR_ACTIVE_MON_CUR_HP_LO = 0xD015
ADDR_ACTIVE_MON_MAX_HP_HI = 0xD022
ADDR_ACTIVE_MON_MAX_HP_LO = 0xD023
# wEventFlags: derived by anchoring on ADDR_BADGES (0xD356, same "Main Data"
# WRAM section) and byte-counting forward through pret/pokeyellow's
# ram/wram.asm (same method used for ADDR_BATTLE_MON_PP above). Lands at
# 0xD747 -- identical to Red's well-known wEventFlags address (PWhiddy's
# PokemonRedExperiments uses the same 0xD747), which is a strong independent
# check that this derivation is correct. ~2560 bits / 320 bytes, one bit per
# story/quest/trainer-beaten flag, including per-trainer flags for our
# current stuck stretch (EVENT_BEAT_ROUTE_3_TRAINER_0..7,
# EVENT_BEAT_MT_MOON_1_TRAINER_0..6, EVENT_BEAT_ROUTE_4_TRAINER_0, etc. --
# see constants/event_constants.asm).
ADDR_EVENT_FLAGS_START = 0xD747
ADDR_EVENT_FLAGS_END = 0xD887  # exclusive
# wBattleMon (pret/pokeyellow's battle_struct, 29 bytes) is the in-battle
# working copy of the active Pokemon -- separate from the party struct below
# and only re-synced from it at battle start or on a mid-battle move-learn.
# ADDR_ACTIVE_MON_CUR_HP_HI/ADDR_ACTIVE_MON_MAX_HP_HI above already anchor
# this struct (offsets 1 and 15 from its base), which puts the base at
# 0xD013 and its 4-byte Moves/PP fields at 0xD01B and 0xD02C respectively --
# confirmed against a real screenshot of the FIGHT menu showing a move stuck
# at 0/10 PP mid-battle while this project's PP refill was only ever applied
# to the party struct copy (FIRST_PARTY_MOVE_PP_ADDR below), never this one.
ADDR_BATTLE_MON_MOVE_IDS = tuple(0xD01B + offset for offset in range(4))
ADDR_BATTLE_MON_PP = tuple(0xD02C + offset for offset in range(4))

# Yellow uses shifted party-data addresses versus many Red/Blue references.
# These pairs were verified against the project's actual save states.
PARTY_CUR_HP_ADDRS = [
    (0xD16B, 0xD16C),
    (0xD197, 0xD198),
    (0xD1C3, 0xD1C4),
    (0xD1EF, 0xD1F0),
    (0xD21B, 0xD21C),
    (0xD247, 0xD248),
]
PARTY_MAX_HP_ADDRS = [
    (0xD18C, 0xD18D),
    (0xD1B8, 0xD1B9),
    (0xD1E4, 0xD1E5),
    (0xD210, 0xD211),
    (0xD23C, 0xD23D),
    (0xD268, 0xD269),
]
PARTY_DATA_STRIDE = 0x2C
FIRST_PARTY_MOVE_ID_ADDR = 0xD172
FIRST_PARTY_MOVE_PP_ADDR = 0xD187
PARTY_MOVE_ID_ADDRS = [
    tuple(FIRST_PARTY_MOVE_ID_ADDR + (idx * PARTY_DATA_STRIDE) + offset for offset in range(4))
    for idx in range(len(PARTY_CUR_HP_ADDRS))
]
PARTY_MOVE_PP_ADDRS = [
    tuple(FIRST_PARTY_MOVE_PP_ADDR + (idx * PARTY_DATA_STRIDE) + offset for offset in range(4))
    for idx in range(len(PARTY_CUR_HP_ADDRS))
]
def _load_gen1_move_table():
    # Gen1's move-data table, empirically located in yellow.gb (2026-07-20) by
    # brute-force searching for the byte offset where every already-known
    # (move_id, max_pp) pair lined up, then cross-checked against real Gen1
    # move data (move 1=Pound power=40/Normal/35pp, move 87=Thunder
    # power=120/Electric/70%acc/10pp, 165 total moves ending at Struggle --
    # all match exactly). Each record is 6 bytes, 1-indexed by move ID:
    # [move_id_self_check, effect, power, type, accuracy, max_pp].
    # Reading real ROM data here means this table is automatically correct
    # for every move Pikachu ever learns, instead of a hand-curated dict that
    # silently doesn't cover a move nobody remembered to add (the exact gap
    # that caused GEN1_DAMAGING_MOVE_IDS to only know about 5 of 165 moves).
    table = {}
    try:
        rom_bytes = ROM_PATH.read_bytes()
        base = 0x038000
        stride = 6
        move_id = 1
        while True:
            idx = base + (move_id - 1) * stride
            if idx + stride > len(rom_bytes):
                break
            record = rom_bytes[idx:idx + stride]
            if record[0] != move_id:
                break
            table[move_id] = {
                'effect': record[1],
                'power': record[2],
                'type': record[3],
                'accuracy': record[4],
                'max_pp': record[5],
            }
            move_id += 1
    except OSError:
        pass
    return table


GEN1_MOVE_TABLE = _load_gen1_move_table()
PEWTER_FALLBACK_ATTACK_MOVE_ID = 129  # Swift -- always known-safe, taught via TM28 well before Brock

MAP_LABELS = map_labels()

# === Milestone definitions ===
MILESTONES = [
    {
        'name': '00_start',
        'state': START_STATE_PATH,
        'check': lambda mem, mid: mem[ADDR_PARTY_SIZE] >= 1 and mid != 40,
        'next_name': '01_got_pikachu',
        'ep_length': 8192,
        'reward_threshold': 2500,
        'desc': 'Escape bedroom, get Pikachu, beat Gary',
        'shaping': {38: 7, 37: 3},  # bedroom stairs Y, then door Y
    },
    {
        'name': '01_got_pikachu',
        'state': None,
        'check': lambda mem, mid: mid == 1,
        'next_name': '02_viridian_city',
        'ep_length': 16384,
        'reward_threshold': None,
        'desc': 'Travel Route 1 (map 12) to Viridian City (map 1)',
        'shaping': {0: 0, 12: 0},  # north (low Y) through Pallet & Route 1
    },
    {
        'name': '02_viridian_city',
        'state': None,
        'check': lambda mem, mid: mid == 42,
        'next_name': '03_got_parcel',
        'ep_length': 8192,
        'desc': 'Enter Viridian Pokemart to get parcel',
        'shaping': {1: 29},  # south in Viridian toward mart
        'ppo': {'ent_coef': 0.05},  # higher entropy to explore around obstacles
    },
    {
        'name': '03_got_parcel',
        'state': None,
        'check': lambda mem, mid: mid == 0,
        'next_name': '04_return_pallet',
        'ep_length': 16384,
        'desc': 'Return through Route 1 to Pallet Town',
        'shaping': {1: 21, 12: 17},  # south exit of Viridian, south on Route 1
        'ppo': {'ent_coef': 0.05},
    },
    {
        'name': '04_return_pallet',
        'state': None,
        'check': lambda mem, mid: mid == 40,
        'next_name': '05_deliver_parcel',
        'ep_length': 8192,
        'desc': 'Enter Oaks Lab to deliver parcel',
        'shaping': {0: 12},  # south in Pallet to Oaks Lab
    },
    {
        'name': '05_deliver_parcel',
        'state': None,
        'check': lambda mem, mid: mid == 0,  # just exit Oak's Lab to Pallet Town
        'next_name': '05a_exit_lab',
        'ep_length': 8192,
        'desc': 'Exit Oaks Lab',
        'shaping': {40: 12},  # Oak Lab exit door at y=12
    },
    {
        'name': '05a_exit_lab',
        'state': None,
        'check': lambda mem, mid: mid == 1,  # reach Viridian City
        'next_name': '05b_pokecenter',
        'ep_length': 16384,
        'desc': 'Travel north through Pallet & Route 1 to Viridian City',
        'shaping': {},  # no shaping — rely on map bonus + tile exploration
    },
    {
        'name': '05b_pokecenter',
        'state': None,
        'check': lambda mem, mid: mid == 41,  # Viridian Pokecenter
        'next_name': '05c_viridian_north',
        'ep_length': 8192,
        'desc': 'Heal at Viridian Pokecenter before forest',
        'shaping': {1: 23},  # Pokecenter entrance at y=23
    },
    {
        'name': '05c_viridian_north',
        'state': None,
        'check': lambda mem, mid: mid == 13,  # Route 2
        'next_name': '05d_viridian_forest',
        'ep_length': 16384,
        'desc': 'North through Viridian City to Route 2',
    },
    {
        'name': '05d_forest_south',
        'state': None,
        'check': lambda mem, mid: mid == 51 and mem[ADDR_POS_A] >= 24,
        'next_name': '05e_forest_north',
        'ep_length': 32768,
        'desc': 'Enter Viridian Forest and reach south end (y>=24)',
        'shaping': {50: 5, 51: 26},  # no Route 2 shaping — agent finds gate naturally, then south in forest
        'ppo': {'ent_coef': 0.1, 'target_kl': 0.05},
    },
    {
        'name': '05e_forest_north',
        'state': None,
        'check': lambda mem, mid: (
            mid == 51
            and mem[ADDR_BATTLE_FLAG] == 0
            and ((mem[ADDR_ACTIVE_MON_CUR_HP_HI] << 8) | mem[ADDR_ACTIVE_MON_CUR_HP_LO]) >= 8
            and (
                (mem[ADDR_POS_A] == 1 and mem[ADDR_POS_B] == 1)
                or (mem[ADDR_POS_A] == 2 and mem[ADDR_POS_B] == 1)
            )
        ),
        'next_name': '05f_gate_transition',
        'ep_length': 32768,
        'desc': 'Reach the north gate approach tile in the forest with enough HP',
        'reward_cap': 350,  # keep the stage focused on the forest itself
        'shaping': {
            51: [(26, 1), (17, 0), (12, 0), (7, 1), (1, 0)],
            # Pull agent back toward forest when it wanders
            50: 5,   # Forest gate south → exit into forest
            13: 0,   # Route 2 → north toward forest gate
            1: 0,    # Viridian City → north toward Route 2
        },
        'ppo': {'ent_coef': 0.1, 'target_kl': 0.05},
    },
    {
        'name': '05f_gate_transition',
        'state': None,
        'check': lambda mem, mid: mid == 47,
        'next_name': '05g_to_pewter',
        'ep_length': 8192,
        'desc': 'Step from the forest exit tile into the north gate',
        'reward_cap': 390,
        'shaping': {
            51: [(2, 1), (1, 1)],
            47: (4, 1),
            50: 5,
            13: 0,
            1: 0,
        },
        'ppo': {'ent_coef': 0.08, 'target_kl': 0.04},
    },
    {
        'name': '05g_to_pewter',
        'state': None,
        'check': lambda mem, mid: mid == 2,  # Pewter City
        'next_name': '06_pewter_city',
        'ep_length': 65536,
        'desc': 'Through the north gate and Route 2 north to Pewter City',
        'reward_cap': 500,
        'shaping': {
            47: 5,   # Forest gate north — walk through
            # Pull back from escape maps
            50: 5,   # Forest gate south → back into forest
            13: 0,   # Route 2 → north toward Pewter
            1: 0,    # Viridian → north
        },
        'ppo': {'ent_coef': 0.1, 'target_kl': None},
    },
    {
        'name': '06_pewter_city',
        'state': None,
        'check': lambda mem, mid: int(mem[ADDR_LEVEL]) >= 8,
        'next_name': '06b_pewter_lvl10',
        'ep_length': 32768,
        'desc': 'Grind in Pewter until Pikachu reaches level 8, then return to the Pokecenter checkpoint',
    },
    {
        'name': '06b_pewter_lvl10',
        'state': None,
        'check': lambda mem, mid: int(mem[ADDR_LEVEL]) >= 10,
        'next_name': '06c_pewter_lvl13',
        'ep_length': 8192,
        'desc': 'Grind in Pewter until Pikachu reaches level 10, then return to the Pokecenter checkpoint',
    },
    {
        'name': '06c_pewter_lvl13',
        'state': None,
        'check': lambda mem, mid: int(mem[ADDR_LEVEL]) >= 13,
        'next_name': '06d_beat_brock',
        'ep_length': 8192,
        'desc': 'Grind in Pewter until Pikachu reaches level 13, then return to the Pokecenter checkpoint',
    },
    {
        'name': '06d_beat_brock',
        'state': None,
        'check': lambda mem, mid: mem[ADDR_BADGES] >= 1,
        'next_name': '07_beat_brock',
        'ep_length': 32768,
        'desc': 'Beat Brock (Boulder Badge)',
        'shaping': {2: (16, 0)},  # gym entrance in Pewter City
        'ppo': {'ent_coef': 0.1, 'target_kl': 0.05},  # high entropy + relaxed KL for gym exploration
    },
    {
        'name': '07_beat_brock',
        'state': None,
        'check': lambda mem, mid: mid == 3,  # map 3 = Cerulean City
        'next_name': '08_cerulean_city',
        'ep_length': 32768,
        'desc': 'Through Mt. Moon to Cerulean City',
    },
    {
        'name': '08_cerulean_city',
        'state': None,
        'check': lambda mem, mid: mem[ADDR_BADGES] >= 2,
        'next_name': '09_beat_misty',
        'ep_length': 16384,
        'desc': 'Beat Misty (Cascade Badge)',
    },
    {
        'name': '09_beat_misty',
        'state': None,
        'check': lambda mem, mid: mid == 5,  # map 5 = Vermilion City
        'next_name': '10_vermilion_city',
        'ep_length': 16384,
        'desc': 'Reach Vermilion City',
    },
    {
        'name': '10_vermilion_city',
        'state': None,
        'check': lambda mem, mid: mem[ADDR_BADGES] >= 3,
        'next_name': '11_beat_surge',
        'ep_length': 32768,
        'desc': 'S.S. Anne + Beat Lt. Surge (Thunder Badge)',
    },
    {
        'name': '11_beat_surge',
        'state': None,
        'check': lambda mem, mid: mid == 4,  # map 4 = Lavender Town
        'next_name': '12_lavender_town',
        'ep_length': 32768,
        'desc': 'Rock Tunnel to Lavender Town',
    },
    {
        'name': '12_lavender_town',
        'state': None,
        'check': lambda mem, mid: mid == 6,  # map 6 = Celadon City
        'next_name': '13_celadon_city',
        'ep_length': 16384,
        'desc': 'Reach Celadon City',
    },
    {
        'name': '13_celadon_city',
        'state': None,
        'check': lambda mem, mid: mem[ADDR_BADGES] >= 4,
        'next_name': '14_beat_erika',
        'ep_length': 32768,
        'desc': 'Beat Erika (Rainbow Badge) + Rocket Hideout',
    },
    {
        'name': '14_beat_erika',
        'state': None,
        'check': lambda mem, mid: mem[ADDR_BADGES] >= 5,
        'next_name': '15_beat_sabrina',
        'ep_length': 32768,
        'desc': 'Pokemon Tower + Silph Co + Beat Sabrina (Marsh Badge)',
    },
    {
        'name': '15_beat_sabrina',
        'state': None,
        'check': lambda mem, mid: mem[ADDR_BADGES] >= 6,
        'next_name': '16_beat_koga',
        'ep_length': 32768,
        'desc': 'Reach Fuchsia, Beat Koga (Soul Badge)',
    },
    {
        'name': '16_beat_koga',
        'state': None,
        'check': lambda mem, mid: mem[ADDR_BADGES] >= 7,
        'next_name': '17_beat_blaine',
        'ep_length': 32768,
        'desc': 'Cinnabar Island, Beat Blaine (Volcano Badge)',
    },
    {
        'name': '17_beat_blaine',
        'state': None,
        'check': lambda mem, mid: mem[ADDR_BADGES] >= 8,
        'next_name': '18_beat_giovanni',
        'ep_length': 16384,
        'desc': 'Beat Giovanni (Earth Badge)',
    },
    {
        'name': '18_beat_giovanni',
        'state': None,
        'check': lambda mem, mid: mid == 118,  # Hall of Fame
        'next_name': '19_hall_of_fame',
        'ep_length': 65536,
        'desc': 'Victory Road + Elite Four + Champion',
    },
]

MILESTONES_DIR = PROJECT_MILESTONES_DIR
PROGRESS_FILE = PROJECT_PROGRESS_FILE
MILESTONE_INDEX = {milestone['name']: idx for idx, milestone in enumerate(MILESTONES)}
PEWTER_STAGE_START_LEVELS = {
    '06_pewter_city': 7,
    '06b_pewter_lvl10': 8,
    '06c_pewter_lvl13': 10,
    '06d_beat_brock': 18,
}
PEWTER_GYM_MAP = 54
PEWTER_STAGE_SAFE_MAPS = {2, 13, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58}
PEWTER_STAGE_SAFE_MAP_LIST = sorted(PEWTER_STAGE_SAFE_MAPS)
PEWTER_STAGE_START_MAPS = {2, 13, 54, 58}
PEWTER_STAGE_START_MAP_LIST = sorted(PEWTER_STAGE_START_MAPS)
PEWTER_BROCK_START_MAPS = {PEWTER_GYM_MAP}
PEWTER_BROCK_ALLOWED_MAPS = {PEWTER_GYM_MAP}
PEWTER_BROCK_ALLOWED_MAP_LIST = sorted(PEWTER_BROCK_ALLOWED_MAPS)
PEWTER_GRIND_BATTLE_PROXY_MAPS = {13, 51}
PEWTER_GRIND_BATTLE_PROXY_MAP_LIST = sorted(PEWTER_GRIND_BATTLE_PROXY_MAPS)
PEWTER_GRIND_ALLOWED_MAP_LIST = PEWTER_STAGE_SAFE_MAP_LIST
PEWTER_PROGRESS_SAVE_MAPS = PEWTER_STAGE_SAFE_MAP_LIST
PEWTER_GRIND_FULL_HP_MAP_STEP_PENALTIES = {
    58: 0.25,
    2: 0.06,
}
PARTY_DATA_COPY_START = ADDR_PARTY_SIZE
PARTY_DATA_COPY_END = 0xD26B
PEWTER_RETREAT_SHAPING = {
    51: (1, 1),
    47: (15, 1),
    54: 8,
    13: [(5, 0), (3, 1), (8, 1), (9, 1)],
    57: (2, 1),
    2: [(19, 1), (13, 1), (3, 1)],
    58: (13, 1),
}
PEWTER_GRIND_SHAPING = {
    2: [(3, 1), (13, 1), (19, 1)],
    13: [(9, 1), (8, 1), (3, 1), (5, 0)],
    47: [(4, 1), (1, 0)],
    51: (11, 1),
}
PEWTER_GRIND_ROUTE_SHAPING = {
    58: (13, 1),
    2: PEWTER_GRIND_SHAPING[2],
    13: [(9, 1), (8, 1)],
    54: 8,
}
PEWTER_GRIND_ROUTE_ZONE_BONUSES = {
    2: [
        {'target': 13, 'radius': 0, 'bonus': 150},
        {'target': 19, 'radius': 0, 'bonus': 250},
    ],
    13: [
        {'target': 9, 'radius': 0, 'bonus': 250},
    ],
    54: [
        {'target': 8, 'radius': 0, 'bonus': 400},
    ],
}
PEWTER_GRIND_ACTION_GUIDANCE = [
    {'map': 58, 'target': 13, 'radius': 0, 'action': 1, 'bonus': 150.0, 'penalty': 15.0},
    {'map': 2, 'target': 19, 'radius': 0, 'action': 1, 'bonus': 125.0, 'penalty': 20.0},
    {'map': 13, 'target': 9, 'radius': 0, 'action': 1, 'bonus': 150.0, 'penalty': 20.0},
    {'map': 54, 'target': 8, 'radius': 1, 'action': 0, 'bonus': 60.0, 'penalty': 10.0},
]
PEWTER_GRIND_FRONTIER_PATH = [
    (2, [(3, 1), (13, 1), (19, 1)]),
    (13, [(9, 1), (8, 1), (3, 1), (5, 0)]),
    (47, [(4, 1), (1, 0)]),
    (51, (11, 1)),
]
PEWTER_BROCK_SHAPING = {
    PEWTER_GYM_MAP: [
        (8, 4), (8, 3), (8, 2), (8, 1),
        (7, 1), (6, 1), (5, 1), (4, 1),
        (4, 2), (4, 3), (4, 4), (3, 4), (2, 4),
    ],
}
PEWTER_BROCK_FRONTIER_PATH = [
    (PEWTER_GYM_MAP, PEWTER_BROCK_SHAPING[PEWTER_GYM_MAP]),
]
PEWTER_BROCK_ZONE_BONUSES = {
    PEWTER_GYM_MAP: [
        {'target': (8, 4), 'radius': 0, 'bonus': 250},
        {'target': (8, 1), 'radius': 0, 'bonus': 350},
        {'target': (4, 1), 'radius': 0, 'bonus': 450},
        {'target': (4, 4), 'radius': 0, 'bonus': 500},
        {'target': (2, 4), 'radius': 0, 'bonus': 900},
    ],
}
PEWTER_BROCK_ACTION_GUIDANCE = [
    {'map': PEWTER_GYM_MAP, 'target': (8, 4), 'radius': 0, 'action': 2, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (8, 3), 'radius': 0, 'action': 2, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (8, 2), 'radius': 0, 'action': 2, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (8, 1), 'radius': 0, 'action': 0, 'bonus': 200.0, 'penalty': 30.0},
    {'map': PEWTER_GYM_MAP, 'target': (7, 1), 'radius': 0, 'action': 0, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (6, 1), 'radius': 0, 'action': 0, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (5, 1), 'radius': 0, 'action': 0, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (4, 1), 'radius': 0, 'action': 3, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (4, 2), 'radius': 0, 'action': 3, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (4, 3), 'radius': 0, 'action': 3, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (4, 4), 'radius': 0, 'action': 0, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (3, 4), 'radius': 0, 'action': 0, 'bonus': 175.0, 'penalty': 25.0},
    {'map': PEWTER_GYM_MAP, 'target': (2, 4), 'radius': 0, 'action': 4, 'bonus': 225.0, 'penalty': 25.0},
]
PEWTER_GRIND_ZONE_BONUSES = {
    47: [
        {'target': (1, 0), 'radius': 0, 'bonus': 250},
    ],
    51: [
        {'target': (7, 1), 'radius': 0, 'bonus': 150},
        {'target': (11, 1), 'radius': 0, 'bonus': 300},
    ],
}

MILESTONES[MILESTONE_INDEX['05d_forest_south']].update({
    'load_settle_ticks': 30,
    'coord_stuck_threshold': 24,
    'coord_stuck_penalty': 0.05,
    'action_oscillation_penalty': 0.02,
})

MILESTONES[MILESTONE_INDEX['05e_forest_north']].update({
    'ep_length': 16384,
    'load_settle_ticks': 30,
    'coord_stuck_threshold': 48,
    'coord_stuck_penalty': 0.05,
    'action_oscillation_penalty': 0.05,
    'step_penalty': 0.01,
    'tile_exploration_bonus': 0.0,
    'map_discovery_bonus': 25,
    'map_step_penalties': {
        50: 0.10,
        13: 0.10,
        1: 0.25,
        41: 0.25,
        33: 0.50,
    },
    'shaping': {
        51: [(26, 1), (25, 1), (17, 0), (12, 0), (11, 1), (7, 1), (2, 1), (1, 1)],
        50: 5,
        13: 0,
        1: 0,
    },
    'frontier_path': [
        (51, [(26, 1), (25, 1), (17, 0), (12, 0), (11, 1), (7, 1), (2, 1), (1, 1)]),
    ],
    'zone_bonuses': {
        51: [
            {'target': (2, 1), 'radius': 0, 'bonus': 150},
            {'target': (1, 0), 'radius': 0, 'bonus': 250},
            {'target': (1, 1), 'radius': 0, 'bonus': 500},
        ],
    },
    'frontier_reward_scale': 1.5,
    'hit_bonus': 1000,
    'terminate_on_hit': True,
    'ppo': {'ent_coef': 0.05, 'target_kl': 0.03, 'n_steps': 1024, 'n_epochs': 2},
})

MILESTONES[MILESTONE_INDEX['05f_gate_transition']].update({
    'n_envs': 8,
    'vec_env_kind': 'subproc',
    'ep_length': 8192,
    'actions': ['up', 'down', 'left', 'right', 'a', 'noop'],
    'direction_press_ticks': 18,
    'direction_release_ticks': 6,
    'button_press_ticks': 8,
    'button_release_ticks': 16,
    'post_transition_settle_ticks': 120,
    'load_settle_ticks': 30,
    'coord_stuck_threshold': 64,
    'coord_stuck_penalty': 0.05,
    'action_oscillation_penalty': 0.0,
    'overworld_non_movement_penalty': 0.25,
    'step_penalty': 0.03,
    'tile_exploration_bonus': 0.0,
    'map_discovery_bonus': 25,
    'map_step_penalties': {
        50: 0.10,
        13: 0.10,
        1: 2.00,
        33: 2.00,
        41: 2.00,
    },
    'shaping': {
        51: [(26, 1), (25, 1), (17, 0), (12, 0), (11, 1), (7, 1), (2, 1), (1, 1), (1, 0)],
        47: [(1, 0), (4, 1)],
        50: 5,
        13: 0,
        1: 0,
    },
    'frontier_path': [
        (51, [(26, 1), (25, 1), (17, 0), (12, 0), (11, 1), (7, 1), (2, 1), (1, 1), (1, 0)]),
        (47, [(1, 0), (4, 1)]),
    ],
    'frontier_soft_stall_score': 850.0,
    'frontier_soft_stall_steps': 48,
    'frontier_soft_stall_penalty': 0.5,
    'frontier_stall_steps': 192,
    'frontier_stall_penalty': 1000.0,
    'frontier_hit_score': 900.0,
    'zone_bonuses': {
        51: [
            {'target': (2, 1), 'radius': 0, 'bonus': 150},
            {'target': (1, 0), 'radius': 0, 'bonus': 250},
            {'target': (1, 1), 'radius': 0, 'bonus': 500},
        ],
        47: [
            {'target': (1, 0), 'radius': 0, 'bonus': 600},
            {'target': (4, 1), 'radius': 0, 'bonus': 800},
        ],
    },
    'action_guidance': None,
    'action_guidance_requires_overworld': True,
    'penalty_exempt_zones': [
        {'map': 51, 'target': (1, 1), 'radius': 1},
        {'map': 47, 'target': (4, 1), 'radius': 1},
    ],
    'allowed_maps': [51, 47, 50, 13, 1],
    'disallowed_map_penalty': 2500.0,
    'frontier_reward_scale': 2.0,
    'hit_bonus': 1500,
    'safe_hit_save_maps': [51, 47],
    'force_repel': True,
    'wild_battle_entry_penalty': 50.0,
    'hp_loss_penalty_scale': 3.0,
    'wipe_penalty': 1000.0,
    'wipe_progress_max_y_scale': 0.0,
    'wipe_progress_tiles_scale': 2.0,
    'wipe_progress_frontier_scale': 0.5,
    'party_size_bonus_scale': 0.0,
    'invalid_party_state_penalty': 2500.0,
    'terminate_on_invalid_party_state': True,
    'terminate_on_party_wipe': True,
    'terminate_after_hit_state_saved': True,
    'ppo': {'ent_coef': 0.05, 'target_kl': 0.04, 'n_steps': 256, 'n_epochs': 2},
})

MILESTONES[MILESTONE_INDEX['05g_to_pewter']].update({
    'ep_length': 1024,
    'load_settle_ticks': 1,
    'coord_stuck_threshold': 12,
    'coord_stuck_penalty': 0.10,
    'action_oscillation_penalty': 0.10,
    'step_penalty': 0.02,
    'tile_exploration_bonus': 0.0,
    'map_discovery_bonus': 50,
    'direction_press_ticks': 24,
    'direction_release_ticks': 8,
    'button_press_ticks': 8,
    'button_release_ticks': 8,
    'force_repel': True,
    'allowed_maps': [47, 13, 2],
    'disallowed_map_penalty': 1000.0,
    'frontier_stall_steps': 48,
    'frontier_stall_penalty': 1000.0,
    'shaping': {
        47: (15, 1),
        13: [(5, 0), (3, 1), (8, 1), (9, 1)],
        2: (19, 1),
    },
    'frontier_path': [
        (47, [(15, 1)]),
        (13, [(5, 0), (3, 1), (8, 1), (9, 1)]),
        (2, [(19, 1)]),
    ],
    'zone_bonuses': {
        13: [
            {'target': (3, 1), 'radius': 0, 'bonus': 150},
            {'target': (8, 1), 'radius': 0, 'bonus': 200},
            {'target': (9, 1), 'radius': 0, 'bonus': 300},
        ],
        2: [
            {'target': (19, 1), 'radius': 0, 'bonus': 500},
        ],
    },
    'action_guidance': [
        {'map': 47, 'target': (15, 1), 'radius': 0, 'action': 0, 'bonus': 100.0, 'penalty': 25.0},
        {'map': 13, 'target': (5, 0), 'radius': 0, 'action': 0, 'bonus': 75.0, 'penalty': 25.0},
        {'map': 13, 'target': (3, 1), 'radius': 0, 'action': 3, 'bonus': 75.0, 'penalty': 25.0},
        {'map': 13, 'target': (8, 1), 'radius': 0, 'action': 0, 'bonus': 100.0, 'penalty': 25.0},
    ],
    'frontier_reward_scale': 2.0,
    'hit_bonus': 1500,
    'terminate_on_hit': True,
    'safe_hit_save_maps': [2],
    'terminate_after_hit_state_saved': True,
    'party_size_bonus_scale': 0.0,
    'terminate_on_party_wipe': True,
    'ppo': {'ent_coef': 0.03, 'target_kl': 0.02, 'n_steps': 256, 'n_epochs': 2},
})

for stage_name, target_level in (
    ('06_pewter_city', 8),
    ('06b_pewter_lvl10', 10),
    ('06c_pewter_lvl13', 18),
):
    MILESTONES[MILESTONE_INDEX[stage_name]].update({
        'check': lambda mem, mid, level_goal=target_level: int(mem[ADDR_LEVEL]) >= level_goal,
        'desc': (
            f'Grind in Pewter until Pikachu reaches level {target_level}, '
            'then return to the Pokecenter checkpoint'
        ),
        'ep_length': 8192,
        'load_settle_ticks': 60,
        'step_penalty': 0.01,
        'tile_exploration_bonus': 0.0,
        'map_discovery_bonus': 25,
        'actions': ['up', 'down', 'left', 'right', 'a', 'b'],
        'direction_press_ticks': 96,
        'direction_release_ticks': 8,
        'button_press_ticks': 8,
        'button_release_ticks': 8,
        'battle_direction_press_ticks': 8,
        'battle_direction_release_ticks': 8,
        'battle_button_press_ticks': 8,
        'battle_button_release_ticks': 8,
        'battle_noop_ticks': 8,
        'coord_stuck_threshold': 32,
        'coord_stuck_penalty': 0.05,
        'action_oscillation_penalty': 0.05,
        'allowed_maps': PEWTER_GRIND_ALLOWED_MAP_LIST,
        'disallowed_map_penalty': 250.0,
        'map_step_penalties': {},
        'shaping': PEWTER_GRIND_ROUTE_SHAPING,
        'post_hit_shaping': PEWTER_RETREAT_SHAPING,
        'low_hp_retreat_threshold': 10,
        'low_hp_retreat_shaping': PEWTER_RETREAT_SHAPING,
        'zone_bonuses': None,
        'action_guidance': None,
        'force_a_in_wild_battles': True,
        'safe_hit_save_maps': PEWTER_STAGE_START_MAP_LIST,
        'safe_hit_min_total_hp': 1,
        'safe_hit_save_bonus': 1500.0,
        'progress_checkpoint_maps': PEWTER_PROGRESS_SAVE_MAPS,
        'progress_checkpoint_bonus': 1000.0,
        'restore_party_resources_on_progress_checkpoint': True,
        'terminate_after_hit_state_saved': True,
        'hit_bonus': 1000,
        'terminate_on_party_wipe': True,
        'party_size_bonus_scale': 0.0,
        'wild_battle_step_penalty': 0.25,
        'wild_battle_entry_penalty': -150.0,
        'wild_faint_bonus': 250.0,
        'trainer_faint_bonus': 250.0,
        'level_up_bonus': 750.0,
        'battle_a_bonus': 0.75,
        'battle_direction_penalty': 0.10,
        'battle_step_limit': 256,
        'battle_step_limit_penalty': 750.0,
        'stale_battle_flag_clear_steps': 96,
        'progressless_step_limit': 768,
        'progressless_step_penalty': 500.0,
        'checkpoint_progress_step_limit': 1024,
        'checkpoint_progress_step_penalty': 750.0,
        'full_hp_map_step_penalties': PEWTER_GRIND_FULL_HP_MAP_STEP_PENALTIES,
        'min_steps_to_advance': 100_000,
        'ppo': {'ent_coef': 0.08, 'target_kl': 0.03, 'n_steps': 512, 'n_epochs': 2},
    })

MILESTONES[MILESTONE_INDEX['06c_pewter_lvl13']].update({
    # 06c regressed into immediate battle stalls in Forest. Give the stage
    # its earlier frontier signal back and avoid terminating fights before they
    # can resolve into XP.
    'ep_length': 8192,
    'shaping': PEWTER_GRIND_SHAPING,
    'frontier_path': PEWTER_GRIND_FRONTIER_PATH,
    'frontier_reward_scale': 1.5,
    'allowed_maps': PEWTER_GRIND_ALLOWED_MAP_LIST,
    'disallowed_map_penalty': 250.0,
    'action_guidance': None,
    'zone_bonuses': PEWTER_GRIND_ZONE_BONUSES,
    'safe_hit_save_maps': PEWTER_STAGE_SAFE_MAP_LIST,
    'map_step_penalties': {
        54: 0.04,
    },
    'low_hp_retreat_threshold': 10,
    'progress_checkpoint_bonus': 1250.0,
    'wild_battle_step_penalty': 0.10,
    'wild_battle_entry_penalty': 35.0,
    'hp_loss_penalty_scale': 4.0,
    'wipe_penalty': 1250.0,
    'wipe_progress_tiles_scale': 1.0,
    'wipe_progress_frontier_scale': 0.25,
    'battle_a_bonus': 1.5,
    'battle_direction_penalty': 0.10,
    'enemy_hp_battle_proxy_maps': PEWTER_GRIND_BATTLE_PROXY_MAP_LIST,
    'battle_step_limit': 512,
    'battle_step_limit_penalty': 750.0,
    'progressless_step_limit': 2048,
    'progressless_step_penalty': 500.0,
    'checkpoint_progress_step_limit': 3072,
    'checkpoint_progress_step_penalty': 400.0,
    'full_hp_map_step_penalties': PEWTER_GRIND_FULL_HP_MAP_STEP_PENALTIES,
})

MILESTONES[MILESTONE_INDEX['06d_beat_brock']].update({
    'load_settle_ticks': 30,
    'step_penalty': 0.01,
    'tile_exploration_bonus': 0.0,
    'map_discovery_bonus': 50,
    'actions': ['up', 'down', 'left', 'right', 'a', 'b'],
    'direction_press_ticks': 8,
    'direction_release_ticks': 24,
    'button_press_ticks': 8,
    'button_release_ticks': 16,
    'battle_direction_press_ticks': 8,
    'battle_direction_release_ticks': 24,
    'battle_button_press_ticks': 8,
    'battle_button_release_ticks': 8,
    'battle_noop_ticks': 8,
    'coord_stuck_threshold': 96,
    'coord_stuck_penalty': 0.05,
    'action_oscillation_penalty': 0.05,
    'overworld_non_movement_penalty': 0.20,
    'allowed_maps': PEWTER_BROCK_ALLOWED_MAP_LIST,
    'disallowed_map_penalty': 1000.0,
    'map_step_penalties': {
        2: 0.02,
        58: 0.05,
    },
    'shaping': PEWTER_BROCK_SHAPING,
    'frontier_path': PEWTER_BROCK_FRONTIER_PATH,
    'frontier_reward_scale': 2.0,
    'frontier_soft_stall_score': 395.0,
    'frontier_soft_stall_steps': 512,
    'frontier_soft_stall_penalty': 0.5,
    'frontier_stall_steps': 3072,
    'frontier_stall_penalty': 750.0,
    'zone_bonuses': PEWTER_BROCK_ZONE_BONUSES,
    'action_guidance': PEWTER_BROCK_ACTION_GUIDANCE,
    'post_hit_shaping': {
        **PEWTER_RETREAT_SHAPING,
        PEWTER_GYM_MAP: [(13, 4), (12, 4)],
    },
    'safe_hit_save_maps': PEWTER_BROCK_ALLOWED_MAP_LIST,
    'safe_hit_min_total_hp': 1,
    'safe_hit_save_bonus': 2000.0,
    'terminate_after_hit_state_saved': True,
    'hit_bonus': 1500,
    'terminate_on_party_wipe': True,
    'hp_loss_penalty_scale': 3.0,
    'wipe_penalty': 2000.0,
    'wipe_progress_frontier_scale': 0.5,
    'party_size_bonus_scale': 0.0,
    'trainer_battle_entry_bonus': 250.0,
    'trainer_faint_bonus': 2000.0,
    'trainer_faint_hit_count': 2,
    'level_up_bonus': 1000.0,
    'trainer_battle_maps': PEWTER_BROCK_ALLOWED_MAP_LIST,
    'force_a_on_zero_enemy_hp_text': True,
    'force_a_in_trainer_battles': True,
    'trainer_battle_action_script': None,
    'battle_a_bonus': 1.0,
    'battle_direction_penalty': 0.25,
    'stale_battle_flag_clear_steps': 192,
    'battle_step_limit': 4096,
    'battle_step_limit_penalty': 750.0,
    'progressless_step_limit': 3072,
    'progressless_step_penalty': 750.0,
    'ppo': {'ent_coef': 0.08, 'target_kl': 0.03, 'n_steps': 512, 'n_epochs': 2},
})

MILESTONES[MILESTONE_INDEX['07_beat_brock']].update({
    # Redesigned 2026-07-22 to match PWhiddy/PokemonRedExperiments' proven
    # approach for this exact milestone (reaching Cerulean): raw button
    # control at all times, including in battle, no hand-authored waypoints
    # or forced actions. See [[reference_external_rl_projects]] memory and
    # the plan this was built from. Reward is event-flags (dense, ROM-native
    # per-trainer/per-story-beat progress -- see ADDR_EVENT_FLAGS_START) +
    # badges (already in the shared _get_reward() base) + per-episode
    # coordinate exploration + healing. Deliberately no level reward (see
    # level_reward_scale=0 below) and no hand-tuned shaping/frontier/
    # action_guidance/force-A/PP machinery -- the previous version of this
    # stage had all of that and still needed a forced clear on ~every single
    # trainer win, which is what motivated trying this instead.
    #
    # Follow-up same day: the first restart converged into a stable
    # battle-avoidance loop (0 wild battles, full HP, ~190 tiles on repeat,
    # every episode ending at max_steps). Checked pokemonred_puffer's actual
    # rewards/environment/wrappers code directly (not just its README) for
    # comparison. Two concrete deltas from that reading:
    #  1. Neither PWhiddy's nor Puffer's exploration reward ever *punishes*
    #     revisiting -- Puffer's DecayWrapper/MaxLengthWrapper let old
    #     exploration credit decay/get evicted so it can be earned again,
    #     but there's no active penalty anywhere. Our revisit_stuck_penalty
    #     was adding one on top of an already-timid, battle-avoidant policy.
    #     Removed (both now 0) to match.
    #  2. Neither project has any force-A/PP logic *because they train from
    #     scratch* -- there's no discontinuity to cause an aversion. We are
    #     fine-tuning a checkpoint that had 700M+ steps of force-A-guaranteed
    #     wins baked in; suddenly removing that can produce a run of
    #     genuinely unwinnable battles right as it's relearning menu input
    #     from scratch, which is enough to teach avoidance instead of
    #     competence. `ensure_battle_move_ready` below reinstates *only* the
    #     "a real move is loaded" guarantee, not which/when button is
    #     pressed -- the policy still has to learn to open FIGHT and pick a
    #     move itself, it just can't lose to its own stale moveset while
    #     doing so. (Checked Puffer's `a_press` reward as a possible fix too
    #     -- it only fires outside battle, on facing-tile interactions, so
    #     it wouldn't have touched this problem; not adopted.)
    'ep_length': 65536,
    'load_settle_ticks': 30,
    'step_penalty': 0.01,
    'tile_exploration_bonus': 0.0,
    'map_discovery_bonus': 0.0,
    'level_reward_scale': 0.0,
    'event_flag_reward_scale': 100.0,
    'explore_coord_reward_scale': 5.0,
    'heal_reward_scale': 1000.0,
    'revisit_stuck_threshold': 0,
    'revisit_stuck_penalty': 0.0,
    'ensure_battle_move_ready': True,
    # Read drubinstein/pokemonred_puffer's own writeup (drubinstein.github.io
    # /pokerl, chapter 2 "Rewards") after the battle-avoidance loop persisted
    # through the first two fixes. It names our exact symptom directly: "If
    # we were to only reward unique coordinates, the agent never interacts
    # with Brock." Their fix was to treat *any* interaction -- signs,
    # objects, warps, and Pokemon encounters, not just new tiles -- as
    # reward-worthy in itself, separate from the sparse story-progress
    # payoff. We had exactly this: event/explore/heal reward but nothing for
    # the moment of engaging a battle, so standing still was exactly as
    # rewarding as risking a fight. Reinstating a small trainer-encounter
    # bonus (not a wild-battle penalty, which we're also dropping) directly
    # counters that -- entering a fight becomes worth something regardless
    # of how it resolves, instead of being pure risk.
    'trainer_battle_entry_bonus': 100.0,
    'wild_battle_entry_penalty': 0.0,
    'force_repel': True,
    # Restrict enemy-HP proxying for this stage; global proxying created
    # false trainer battle loops that ended at trainer=1/1537 battle_stalls.
    'enemy_hp_battle_proxy_maps': [],
    'wipe_penalty': 2500.0,
    'terminate_on_party_wipe': True,
    # Bounded safety net against one battle eating the whole episode budget --
    # not an action override, just an early-exit-with-penalty ceiling.
    'battle_step_limit': 768,
    'battle_step_limit_penalty': 500.0,
    'hit_bonus': 1500,
    'terminate_on_hit': True,
    # 2026-07-22: scaled envs 4->16 (WSL has 16 cores) so more independent
    # rollouts feed the same PPO update -- more chances for stochastic
    # exploration to stumble past the current sticking point, which then
    # gets reinforced via the shared policy gradient like any other
    # experience. n_steps cut 1024->256 (4x) to hold total batch size
    # (n_steps*n_envs) and update cadence roughly constant rather than
    # accidentally quadrupling how much experience accrues between updates.
    'n_envs': 16,
    # Go-Explore-style "swarming" (see _swarm_sync docstring): with 16
    # workers now running, whichever one gets furthest (most event-flag
    # bits) becomes the shared frontier; the other 15 get pulled forward to
    # it instead of independently re-grinding the same solved ground.
    # check_interval=64 keeps the shared-file overhead cheap; a worker more
    # than 3 event-flag-bits behind the frontier truncates to catch up.
    'swarm_enabled': True,
    'swarm_check_interval': 64,
    'swarm_catchup_behind_bits': 3,
    # Minimal directional nudge (see route4_progress_reward_scale docstring
    # on PokemonYellowEnv): the milestone's saved start is already on Route 4
    # (x=12,y=7) and the real objective is walking east to Cerulean, verified
    # against the user's own recorded playthrough
    # (recordings/manual_07_after_brock_clean.jsonl). 3.0/tile over the
    # ~80-tile crossing totals ~240 -- comparable to a couple of event flags,
    # not dominant over hit_bonus=1500.
    'route4_progress_reward_scale': 3.0,
    'ppo': {'ent_coef': 0.06, 'target_kl': 0.04, 'n_steps': 256, 'n_epochs': 2},
})

# Full game reward ladder
MAP_REWARDS = {
    38: 2, 37: 5, 0: 25, 12: 50,
    1: 100, 40: 200, 42: 210,
    13: 300, 50: 310, 51: 350, 47: 390, 2: 400, 58: 410, 57: 425, 54: 575,
    14: 550, 68: 560, 59: 600, 60: 620, 61: 640, 15: 680, 3: 700, 65: 800,
    35: 750, 36: 760, 88: 780,
    16: 850, 17: 860, 5: 900, 94: 920, 95: 940, 101: 960, 92: 1000,
    20: 1050, 21: 1060, 82: 1100, 232: 1120, 4: 1200,
    19: 1250, 18: 1260, 6: 1300, 134: 1400, 135: 1350, 199: 1360, 202: 1380,
    142: 1420, 148: 1480,
    10: 1500, 181: 1550, 235: 1600, 178: 1700,
    27: 1750, 28: 1760, 29: 1770, 7: 1800, 220: 1850, 157: 1900,
    30: 1920, 31: 1930, 8: 1950, 165: 1970, 166: 2000,
    45: 2100,
    33: 2150, 34: 2200, 108: 2300, 194: 2350, 198: 2400,
    9: 2500, 174: 2550, 245: 2600, 246: 2700, 247: 2800, 113: 2900,
    120: 3000, 118: 5000,
}


class PokemonYellowEnv(gym.Env):
    """Pokemon Yellow training env with per-instance stage config."""
    state_file = START_STATE_PATH
    max_steps = 4096
    milestone_name = "bootstrap"
    milestone_check = None  # lambda mem, mid: bool
    step_penalty = 0.0  # starts at 0 during curriculum, increases in full-game phase
    # Per-milestone shaping: dict of {map_id: (target_y, target_x)} or None
    shaping_targets = None
    post_hit_shaping_targets = None
    reward_cap = None  # max location_reward from MAP_REWARDS (prevents off-path maps rewarding)
    tile_exploration_bonus = 5.0
    map_discovery_bonus = 100.0
    map_step_penalties = None
    full_hp_map_step_penalties = None
    frontier_path = None
    frontier_reward_scale = 0.0
    frontier_segment_span = 1000.0
    frontier_waypoint_span = 100.0
    frontier_action_guidance_bonus = 0.0
    frontier_action_guidance_penalty = 0.0
    zone_bonuses = None
    milestone_hit_bonus = 0.0
    frontier_soft_stall_score = None
    frontier_soft_stall_steps = 0
    frontier_soft_stall_penalty = 0.0
    frontier_stall_steps = 0
    frontier_stall_penalty = 0.0
    frontier_hit_score = None
    action_guidance = None
    action_guidance_requires_overworld = True
    penalty_exempt_zones = None
    auto_wait_zones = None
    force_a_in_wild_battles = False
    force_a_in_trainer_battles = False
    # Decoupled from force-A: keeps slot-0 loaded with a damaging move + PP
    # whenever in battle, without overriding *when*/*whether* A gets pressed.
    # Addresses a failure mode specific to fine-tuning an already-shaped
    # checkpoint rather than training from scratch (which is what PWhiddy/
    # pokemonred_puffer actually do, and why neither of them needs this): a
    # policy that always had force-A guaranteeing a good move suddenly losing
    # that guarantee can have a run of genuinely unwinnable battles right as
    # it's relearning menu navigation from raw input, which is enough to
    # teach battle-avoidance instead of battle-competence. This does not
    # choose actions for the policy -- it only prevents "the move you'd want
    # to use doesn't work" from being the reason a battle goes badly.
    ensure_battle_move_ready = False
    trainer_battle_action_script = None
    trainer_battle_maps = None
    enemy_hp_battle_proxy_maps = None
    force_a_on_zero_enemy_hp_text = False
    low_hp_retreat_threshold = 0
    low_hp_retreat_shaping_targets = None
    allowed_maps = None
    disallowed_map_penalty = 0.0
    terminate_on_milestone_hit = False
    terminate_after_hit_state_saved = False
    safe_hit_save_maps = None
    safe_hit_min_total_hp = 0
    safe_hit_save_bonus = 0.0
    progress_checkpoint_maps = None
    progress_checkpoint_bonus = 0.0
    restore_party_resources_on_progress_checkpoint = False
    direction_press_ticks = 8
    direction_release_ticks = 16
    button_press_ticks = 8
    button_release_ticks = 16
    battle_direction_press_ticks = None
    battle_direction_release_ticks = None
    battle_button_press_ticks = None
    battle_button_release_ticks = None
    battle_noop_ticks = None
    post_transition_settle_ticks = 0
    load_settle_ticks = 0
    coord_stuck_threshold = 0
    coord_stuck_penalty = 0.0
    action_oscillation_penalty = 0.0
    overworld_non_movement_penalty = 0.0
    force_repel = False
    wild_battle_step_penalty = 0.0
    wild_battle_entry_penalty = 0.0
    hp_loss_penalty_scale = 0.0
    wipe_penalty = 0.0
    wipe_progress_max_y_scale = 0.0
    wipe_progress_tiles_scale = 0.0
    wipe_progress_frontier_scale = 0.0
    invalid_party_state_penalty = 0.0
    terminate_on_invalid_party_state = False
    terminate_on_party_wipe = False
    party_size_bonus_scale = 2000.0
    wild_faint_bonus = 0.0
    trainer_battle_entry_bonus = 0.0
    trainer_faint_bonus = 0.0
    trainer_faint_hit_count = 0
    level_up_bonus = 0.0
    battle_a_bonus = 0.0
    battle_direction_penalty = 0.0
    battle_step_limit = 0
    battle_step_limit_penalty = 0.0
    stale_battle_flag_clear_steps = 0
    progressless_step_limit = 0
    progressless_step_penalty = 0.0
    checkpoint_progress_step_limit = 0
    checkpoint_progress_step_penalty = 0.0
    actions = ['up', 'down', 'left', 'right', 'a', 'b']
    noop_ticks = 24
    # PWhiddy/PokemonRedExperiments-style reward terms (opt-in per stage; all
    # default off so existing stages are unaffected). See event_constants.asm
    # via ADDR_EVENT_FLAGS_START/END -- rewards *new* event flags set this
    # episode (a ratcheting max, never decreases mid-episode), independent of
    # any hand-authored waypoints or battle-menu handling.
    event_flag_reward_scale = 0.0
    explore_coord_reward_scale = 0.0
    heal_reward_scale = 0.0
    revisit_stuck_threshold = 0
    revisit_stuck_penalty = 0.0
    # Default (50) matches the historical hardcoded level*50 term exactly, so
    # every existing stage is unaffected. 07_beat_brock sets this to 0 to
    # match PWhiddy's deliberately-disabled level reward (stop rewarding the
    # over-leveling-via-grinding pattern).
    level_reward_scale = 50.0

    # "Swarming" (Go-Explore-inspired, see drubinstein/pokerl book ch.3):
    # whenever any parallel worker's event-flag-bit count exceeds the shared
    # frontier's, it becomes the new leader and its state is saved as the
    # frontier for every worker to load on their next reset(). Workers that
    # fall meaningfully behind the frontier get force-truncated so they pick
    # it up immediately rather than grinding through already-solved ground
    # until their own episode happens to end naturally. Opt-in, default off
    # so existing stages are unaffected. Deliberately separate from the older
    # progress_checkpoint_maps/_progress_checkpoint_key machinery above --
    # that key is frontier_path/level/xp based and would barely move for a
    # stage like 07_beat_brock that has neither; this uses the same
    # ROM-native event-flag-bit count already used for reward on that stage.
    swarm_enabled = False
    swarm_check_interval = 64
    swarm_catchup_behind_bits = 3

    # Minimal directional nudge for Route 4 (map_id 15) specifically -- the
    # milestone's saved start state is already ON Route 4 (x=12,y=7, verified
    # against milestones/07_beat_brock.state directly), and the real
    # objective from there is simply walking east (increasing x) to reach
    # Cerulean City. Confirmed against the user's own recorded playthrough
    # (recordings/manual_07_after_brock_clean.jsonl, 2026-05-05): enters
    # Route 4 near (9,17) and walks steadily east to ~x=89 at the Cerulean
    # border. Deliberately much smaller in scope than the old deleted
    # POST_BROCK_ROUTE_FRONTIER_PATH/ACTION_GUIDANCE waypoint machinery --
    # just a small reward for each new x-tile of forward progress on this
    # one map, not a full path/waypoint system. Opt-in, default off.
    route4_progress_reward_scale = 0.0

    _next_env_id = 0  # class-level counter for unique env IDs
    _pending_episode_summary = None
    _pending_episode_count = 0

    def __init__(self, config=None):
        super().__init__()
        config = dict(config or {})
        self.env_id = int(config.get("env_id", PokemonYellowEnv._next_env_id))
        PokemonYellowEnv._next_env_id += 1
        self.state_file = os.fspath(config.get("state_file", START_STATE_PATH))
        self.max_steps = int(config.get("max_steps", 4096))
        self.milestone_name = config.get("milestone_name", "bootstrap")
        self.milestone_check = config.get("milestone_check")
        self.step_penalty = float(config.get("step_penalty", 0.0))
        self.shaping_targets = config.get("shaping_targets")
        self.post_hit_shaping_targets = config.get("post_hit_shaping_targets")
        self.reward_cap = config.get("reward_cap")
        self.tile_exploration_bonus = float(config.get("tile_exploration_bonus", 5.0))
        self.map_discovery_bonus = float(config.get("map_discovery_bonus", 100.0))
        self.map_step_penalties = config.get("map_step_penalties")
        self.full_hp_map_step_penalties = config.get("full_hp_map_step_penalties")
        self.frontier_path = config.get("frontier_path")
        self.frontier_reward_scale = float(config.get("frontier_reward_scale", 0.0))
        self.frontier_segment_span = float(config.get("frontier_segment_span", 1000.0))
        self.frontier_waypoint_span = float(config.get("frontier_waypoint_span", 100.0))
        self.frontier_action_guidance_bonus = float(config.get("frontier_action_guidance_bonus", 0.0))
        self.frontier_action_guidance_penalty = float(config.get("frontier_action_guidance_penalty", 0.0))
        self.zone_bonuses = config.get("zone_bonuses")
        self.milestone_hit_bonus = float(config.get("milestone_hit_bonus", 0.0))
        frontier_soft_stall_score = config.get("frontier_soft_stall_score")
        self.frontier_soft_stall_score = (
            None if frontier_soft_stall_score is None else float(frontier_soft_stall_score)
        )
        self.frontier_soft_stall_steps = int(config.get("frontier_soft_stall_steps", 0))
        self.frontier_soft_stall_penalty = float(config.get("frontier_soft_stall_penalty", 0.0))
        self.frontier_stall_steps = int(config.get("frontier_stall_steps", 0))
        self.frontier_stall_penalty = float(config.get("frontier_stall_penalty", 0.0))
        frontier_hit_score = config.get("frontier_hit_score")
        self.frontier_hit_score = (
            None if frontier_hit_score is None else float(frontier_hit_score)
        )
        self.action_guidance = config.get("action_guidance")
        self.action_guidance_requires_overworld = bool(config.get("action_guidance_requires_overworld", True))
        self.penalty_exempt_zones = config.get("penalty_exempt_zones")
        self.auto_wait_zones = config.get("auto_wait_zones")
        self.force_a_in_wild_battles = bool(config.get("force_a_in_wild_battles", False))
        self.force_a_in_trainer_battles = bool(config.get("force_a_in_trainer_battles", False))
        self.ensure_battle_move_ready = bool(config.get("ensure_battle_move_ready", False))
        trainer_battle_action_script = config.get("trainer_battle_action_script")
        self.trainer_battle_action_script = (
            list(trainer_battle_action_script)
            if trainer_battle_action_script
            else None
        )
        trainer_battle_maps = config.get("trainer_battle_maps")
        self.trainer_battle_maps = set(trainer_battle_maps or [])
        enemy_hp_battle_proxy_maps = config.get("enemy_hp_battle_proxy_maps")
        self.enemy_hp_battle_proxy_maps = (
            None if enemy_hp_battle_proxy_maps is None else set(enemy_hp_battle_proxy_maps)
        )
        self.force_a_on_zero_enemy_hp_text = bool(config.get("force_a_on_zero_enemy_hp_text", False))
        self.low_hp_retreat_threshold = int(config.get("low_hp_retreat_threshold", 0))
        self.low_hp_retreat_shaping_targets = config.get("low_hp_retreat_shaping_targets")
        self.allowed_maps = config.get("allowed_maps")
        self.disallowed_map_penalty = float(config.get("disallowed_map_penalty", 0.0))
        self.terminate_on_milestone_hit = bool(config.get("terminate_on_milestone_hit", False))
        self.terminate_after_hit_state_saved = bool(config.get("terminate_after_hit_state_saved", False))
        self.safe_hit_save_maps = config.get("safe_hit_save_maps")
        self.safe_hit_min_total_hp = int(config.get("safe_hit_min_total_hp", 0))
        self.safe_hit_save_bonus = float(config.get("safe_hit_save_bonus", 0.0))
        self.progress_checkpoint_maps = config.get("progress_checkpoint_maps")
        self.progress_checkpoint_bonus = float(config.get("progress_checkpoint_bonus", 0.0))
        self.restore_party_resources_on_progress_checkpoint = bool(
            config.get("restore_party_resources_on_progress_checkpoint", False)
        )
        self.direction_press_ticks = int(config.get("direction_press_ticks", 8))
        self.direction_release_ticks = int(config.get("direction_release_ticks", 16))
        self.button_press_ticks = int(config.get("button_press_ticks", 8))
        self.button_release_ticks = int(config.get("button_release_ticks", 16))
        battle_direction_press_ticks = config.get("battle_direction_press_ticks")
        self.battle_direction_press_ticks = (
            self.direction_press_ticks
            if battle_direction_press_ticks is None
            else int(battle_direction_press_ticks)
        )
        battle_direction_release_ticks = config.get("battle_direction_release_ticks")
        self.battle_direction_release_ticks = (
            self.direction_release_ticks
            if battle_direction_release_ticks is None
            else int(battle_direction_release_ticks)
        )
        battle_button_press_ticks = config.get("battle_button_press_ticks")
        self.battle_button_press_ticks = (
            self.button_press_ticks
            if battle_button_press_ticks is None
            else int(battle_button_press_ticks)
        )
        battle_button_release_ticks = config.get("battle_button_release_ticks")
        self.battle_button_release_ticks = (
            self.button_release_ticks
            if battle_button_release_ticks is None
            else int(battle_button_release_ticks)
        )
        battle_noop_ticks = config.get("battle_noop_ticks")
        self.battle_noop_ticks = (
            self.noop_ticks
            if battle_noop_ticks is None
            else int(battle_noop_ticks)
        )
        self.post_transition_settle_ticks = int(config.get("post_transition_settle_ticks", 0))
        self.load_settle_ticks = int(config.get("load_settle_ticks", 0))
        self.coord_stuck_threshold = int(config.get("coord_stuck_threshold", 0))
        self.coord_stuck_penalty = float(config.get("coord_stuck_penalty", 0.0))
        self.action_oscillation_penalty = float(config.get("action_oscillation_penalty", 0.0))
        self.overworld_non_movement_penalty = float(config.get("overworld_non_movement_penalty", 0.0))
        self.force_repel = bool(config.get("force_repel", False))
        self.wild_battle_step_penalty = float(config.get("wild_battle_step_penalty", 0.0))
        self.wild_battle_entry_penalty = float(config.get("wild_battle_entry_penalty", 0.0))
        self.hp_loss_penalty_scale = float(config.get("hp_loss_penalty_scale", 0.0))
        self.wipe_penalty = float(config.get("wipe_penalty", 0.0))
        self.wipe_progress_max_y_scale = float(config.get("wipe_progress_max_y_scale", 0.0))
        self.wipe_progress_tiles_scale = float(config.get("wipe_progress_tiles_scale", 0.0))
        self.wipe_progress_frontier_scale = float(config.get("wipe_progress_frontier_scale", 0.0))
        self.invalid_party_state_penalty = float(config.get("invalid_party_state_penalty", 0.0))
        self.terminate_on_invalid_party_state = bool(config.get("terminate_on_invalid_party_state", False))
        self.terminate_on_party_wipe = bool(config.get("terminate_on_party_wipe", False))
        self.party_size_bonus_scale = float(config.get("party_size_bonus_scale", 2000.0))
        self.wild_faint_bonus = float(config.get("wild_faint_bonus", 0.0))
        self.trainer_battle_entry_bonus = float(config.get("trainer_battle_entry_bonus", 0.0))
        self.trainer_faint_bonus = float(config.get("trainer_faint_bonus", 0.0))
        self.trainer_faint_hit_count = int(config.get("trainer_faint_hit_count", 0))
        self.level_up_bonus = float(config.get("level_up_bonus", 0.0))
        self.battle_a_bonus = float(config.get("battle_a_bonus", 0.0))
        self.battle_direction_penalty = float(config.get("battle_direction_penalty", 0.0))
        self.battle_step_limit = int(config.get("battle_step_limit", 0))
        self.battle_step_limit_penalty = float(config.get("battle_step_limit_penalty", 0.0))
        self.stale_battle_flag_clear_steps = int(config.get("stale_battle_flag_clear_steps", 0))
        self.progressless_step_limit = int(config.get("progressless_step_limit", 0))
        self.progressless_step_penalty = float(config.get("progressless_step_penalty", 0.0))
        self.checkpoint_progress_step_limit = int(config.get("checkpoint_progress_step_limit", 0))
        self.checkpoint_progress_step_penalty = float(config.get("checkpoint_progress_step_penalty", 0.0))
        self.required_route_maps = set(config.get("required_route_maps") or [])
        self.actions = list(config.get("actions") or ['up', 'down', 'left', 'right', 'a', 'b'])
        self.noop_ticks = int(config.get("noop_ticks", 24))
        self.event_flag_reward_scale = float(config.get("event_flag_reward_scale", 0.0))
        self.explore_coord_reward_scale = float(config.get("explore_coord_reward_scale", 0.0))
        self.heal_reward_scale = float(config.get("heal_reward_scale", 0.0))
        self.revisit_stuck_threshold = int(config.get("revisit_stuck_threshold", 0))
        self.revisit_stuck_penalty = float(config.get("revisit_stuck_penalty", 0.0))
        self.level_reward_scale = float(config.get("level_reward_scale", 50.0))
        self.swarm_enabled = bool(config.get("swarm_enabled", False))
        self.swarm_check_interval = max(1, int(config.get("swarm_check_interval", 64)))
        self.swarm_catchup_behind_bits = int(config.get("swarm_catchup_behind_bits", 3))
        _state_path = Path(self.state_file)
        self.swarm_frontier_state_path = _state_path.with_name(
            f"{_state_path.stem}.swarm_frontier{_state_path.suffix}"
        )
        self.swarm_frontier_meta_path = _state_path.with_name(
            f"{_state_path.stem}.swarm_frontier.json"
        )
        self.route4_progress_reward_scale = float(config.get("route4_progress_reward_scale", 0.0))
        self.pyboy = PyBoy(str(ROM_PATH), window='null')
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.steps = 0
        self.last_reward = 0
        self.visited_positions = set()
        self.coord_visit_counts = {}
        self.visited_maps = set()
        self.last_dist = None
        self.last_party_size = 0
        self.milestone_hit = False  # tracks if milestone was hit this episode
        self.max_map_reward = 0     # tracks furthest map reached
        self.latest_hit_state = None
        self.best_progress_checkpoint = (0, 0)
        self.best_observed_progress_checkpoint = (0, 0)
        self.best_frontier_score = float("-inf")
        self.frontier_stall_count = 0
        self.claimed_zone_bonuses = set()
        self.frontier_hp_snapshots = []
        self.last_wipe_progress_credit = 0.0
        self.wild_battle_entries = 0
        self.trainer_battle_entries = 0
        self.wild_battle_steps = 0
        self.trainer_battle_steps = 0
        self.wild_enemy_faints = 0
        self.trainer_enemy_faints = 0
        self.transition_settle_events = 0
        self.transition_settle_ticks_used = 0
        self.stale_battle_flag_clears = 0
        self.enemy_hp_proxy_battle_frames = 0
        self.party_wipes = 0
        self.min_total_party_hp = None
        self.last_total_party_hp = 0
        self.last_battle_flag = 0
        self.max_coord_repeat = 0
        self.last_level = 0
        self.last_wild_faints = 0
        self.last_trainer_faints = 0
        self.current_battle_step_count = 0
        self.zero_hp_battle_step_count = 0
        self.progressless_step_count = 0
        self.checkpoint_progress_step_count = 0
        self.recent_actions = deque(maxlen=8)
        self.trainer_battle_action_index = 0
        self.episode_coord_visits = {}
        self.base_event_flags_bits = 0
        self.max_event_flags_progress = 0
        self.last_hp_fraction = None
        self.route4_start_x = None
        self.max_route4_x_progress = 0

    def _save_latest_hit_state(self):
        milestone_state = latest_hit_state_path(self.milestone_name, self.env_id)
        try:
            with open(milestone_state, 'wb') as f:
                self.pyboy.save_state(f)
            self.latest_hit_state = milestone_state
        except Exception as e:
            print(f"  [ERROR] Failed to save milestone state: {e}")

    def _save_progress_checkpoint(self):
        state_path = Path(self.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = state_path.with_name(
            f"{state_path.stem}.tmp.env{self.env_id}.pid{os.getpid()}{state_path.suffix}"
        )
        try:
            if (
                self.restore_party_resources_on_progress_checkpoint
                and self.milestone_name in PEWTER_STAGE_START_LEVELS
            ):
                _save_repaired_pewter_progress_checkpoint(
                    state_path,
                    self.milestone_name,
                    self.pyboy.memory,
                )
                return
            if self.restore_party_resources_on_progress_checkpoint:
                _heal_party_memory(self.pyboy.memory)
                _restore_party_pp_memory(self.pyboy.memory)
            with open(temp_path, 'wb') as handle:
                self.pyboy.save_state(handle)
            os.replace(temp_path, state_path)
        except Exception as e:
            print(f"  [ERROR] Failed to save progress checkpoint: {e}")
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    def _swarm_sync(self):
        """Go-Explore-style leader sync across the whole swarm of workers.

        Progress is measured in event-flag bits (ROM-native, monotonic --
        see ADDR_EVENT_FLAGS_START), read from a small shared JSON file so
        every worker (separate OS process) can compare against it cheaply.
        A worker ahead of the recorded frontier saves its state as the new
        shared frontier. A worker meaningfully behind it signals that it
        should truncate now so its next reset() picks up the frontier state
        instead of continuing to grind through already-solved ground.
        """
        own_progress = self._count_event_flag_bits()
        frontier_progress = 0
        try:
            if self.swarm_frontier_meta_path.exists():
                meta = json.loads(self.swarm_frontier_meta_path.read_text(encoding='utf-8'))
                frontier_progress = int(meta.get('progress', 0))
        except (OSError, ValueError):
            frontier_progress = 0

        if own_progress > frontier_progress:
            battle_flag = int(self.pyboy.memory[ADDR_BATTLE_FLAG])
            text_box_active = int(self.pyboy.memory[ADDR_TEXT_BOX]) != 0
            total_party_hp, _ = self._party_hp_totals(self._party_hp_snapshot())
            # Only promote a "safe" moment -- same guard as
            # _save_progress_checkpoint -- so the saved state doesn't resume
            # mid-battle/mid-dialogue for every other worker.
            if battle_flag == 0 and not text_box_active and total_party_hp >= 1:
                try:
                    self.swarm_frontier_state_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_state = self.swarm_frontier_state_path.with_name(
                        f"{self.swarm_frontier_state_path.stem}.tmp.env{self.env_id}"
                        f".pid{os.getpid()}{self.swarm_frontier_state_path.suffix}"
                    )
                    with open(temp_state, 'wb') as handle:
                        self.pyboy.save_state(handle)
                    os.replace(temp_state, self.swarm_frontier_state_path)

                    temp_meta = self.swarm_frontier_meta_path.with_name(
                        f"{self.swarm_frontier_meta_path.stem}.tmp.env{self.env_id}"
                        f".pid{os.getpid()}.json"
                    )
                    temp_meta.write_text(
                        json.dumps({
                            'progress': own_progress,
                            'env_id': self.env_id,
                            'ts': time.time(),
                        }),
                        encoding='utf-8',
                    )
                    os.replace(temp_meta, self.swarm_frontier_meta_path)
                    print(
                        f"  [Swarm env_id={self.env_id}] new frontier leader: "
                        f"{own_progress} event-flag bits (was {frontier_progress})",
                        flush=True,
                    )
                except OSError as e:
                    print(f"  [Swarm env_id={self.env_id}] failed to save frontier: {e}", flush=True)
            return False

        return (frontier_progress - own_progress) >= self.swarm_catchup_behind_bits

    def _local_visit_panel(self, map_id, pos_a, pos_b, radius=5):
        size = radius * 2 + 1
        panel = np.zeros((size, size), dtype=np.uint8)
        for visit_map, visit_a, visit_b in self.visited_positions:
            if visit_map != map_id:
                continue
            rel_a = visit_a - pos_a + radius
            rel_b = visit_b - pos_b + radius
            if 0 <= rel_a < size and 0 <= rel_b < size:
                panel[rel_a, rel_b] = 150
        panel[radius, radius] = 255
        return np.repeat(np.repeat(panel, 2, axis=0), 2, axis=1)

    def _draw_panel(self, canvas, top, left, panel, alpha=0.8):
        height = min(panel.shape[0], canvas.shape[0] - top)
        width = min(panel.shape[1], canvas.shape[1] - left)
        if height <= 0 or width <= 0:
            return
        region = canvas[top:top + height, left:left + width].astype(np.float32)
        overlay = panel[:height, :width].astype(np.float32)
        canvas[top:top + height, left:left + width] = (
            region * (1.0 - alpha) + overlay * alpha
        ).astype(np.uint8)

    def _state_feature_rows(self, map_id, pos_a, pos_b, battle_flag, party_size, party_snapshot):
        total_cur_hp, total_max_hp = self._party_hp_totals(party_snapshot)
        hp_frac = (total_cur_hp / total_max_hp) if total_max_hp > 0 else 0.0
        badges = self.pyboy.memory[ADDR_BADGES]
        badge_frac = badges.bit_count() / 8.0
        stuck_base = max(1, self.coord_stuck_threshold or 64)
        stuck_frac = min(self.coord_visit_counts.get((map_id, pos_a, pos_b), 0) / stuck_base, 1.0)
        return [
            map_id / 255.0,
            pos_a / 255.0,
            pos_b / 255.0,
            hp_frac,
            min(party_size, 6) / 6.0,
            badge_frac,
            min(battle_flag, 2) / 2.0,
            stuck_frac,
        ]

    def _state_bar_panel(self, map_id, pos_a, pos_b, battle_flag, party_size, party_snapshot):
        rows = self._state_feature_rows(map_id, pos_a, pos_b, battle_flag, party_size, party_snapshot)
        panel = np.zeros((16, 24), dtype=np.uint8)
        for idx, value in enumerate(rows):
            start_row = idx * 2
            panel[start_row:start_row + 2, :] = 20
            fill = max(1, int(round(value * 23)))
            panel[start_row:start_row + 2, 1:1 + fill] = 235
        return panel

    def _recent_actions_panel(self):
        panel = np.zeros((12, 16), dtype=np.uint8)
        actions = list(self.recent_actions)[-8:]
        start_col = panel.shape[1] - (len(actions) * 2)
        for idx, action in enumerate(actions):
            row = max(0, min(int(action), 5)) * 2
            col = start_col + idx * 2
            panel[row:row + 2, col:col + 2] = 255
        return panel

    def _augment_observation(self, gray):
        map_id = self.pyboy.memory[ADDR_MAP_ID]
        pos_a = self.pyboy.memory[ADDR_POS_A]
        pos_b = self.pyboy.memory[ADDR_POS_B]
        battle_flag = self.pyboy.memory[ADDR_BATTLE_FLAG]
        party_size = self.pyboy.memory[ADDR_PARTY_SIZE]
        party_snapshot = self._party_hp_snapshot(party_size)

        augmented = gray.copy()
        self._draw_panel(
            augmented,
            0,
            0,
            self._state_bar_panel(map_id, pos_a, pos_b, battle_flag, party_size, party_snapshot),
            alpha=0.72,
        )
        local_panel = self._local_visit_panel(map_id, pos_a, pos_b)
        self._draw_panel(
            augmented,
            0,
            augmented.shape[1] - local_panel.shape[1],
            local_panel,
            alpha=0.72,
        )
        action_panel = self._recent_actions_panel()
        self._draw_panel(
            augmented,
            augmented.shape[0] - action_panel.shape[0],
            augmented.shape[1] - action_panel.shape[1],
            action_panel,
            alpha=0.80,
        )
        return augmented

    def _is_recent_action_oscillation(self):
        if len(self.recent_actions) < 4:
            return False
        a, b, c, d = list(self.recent_actions)[-4:]
        if a != c or b != d:
            return False
        return OPPOSITE_ACTIONS.get(a) == b

    def _get_obs(self):
        img = np.array(self.pyboy.screen.image)[:, :, :3]
        gray = np.mean(img, axis=2).astype(np.uint8)
        gray_img = Image.fromarray(gray, mode='L').resize((84, 84))
        gray_small = np.array(gray_img)
        gray_small = self._augment_observation(gray_small)
        return gray_small[:, :, np.newaxis]

    def _get_reward(self):
        level = max(1, min(int(self.pyboy.memory[ADDR_LEVEL]), 100))
        badges = min(int(self.pyboy.memory[ADDR_BADGES]).bit_count(), 8)
        map_id = self.pyboy.memory[ADDR_MAP_ID]
        location_reward = MAP_REWARDS.get(map_id, 0)
        if self.reward_cap is not None:
            location_reward = min(location_reward, self.reward_cap)
        return (level * self.level_reward_scale) + (badges * 1000) + location_reward

    def _target_distance(self, target, pos_a, pos_b):
        if isinstance(target, tuple):
            ty, tx = target
            return abs(pos_a - ty) + abs(pos_b - tx)
        return abs(pos_a - target)

    def _read_u16(self, hi_addr, lo_addr):
        return (self.pyboy.memory[hi_addr] << 8) | self.pyboy.memory[lo_addr]

    def _battle_label(self, battle_flag):
        return {
            0: "overworld",
            1: "wild",
            2: "trainer",
        }.get(battle_flag, f"battle{battle_flag}")

    def _progress_checkpoint_key(self):
        level = max(1, min(int(self.pyboy.memory[ADDR_LEVEL]), 100))
        xp = (
            (int(self.pyboy.memory[ADDR_XP]) << 16)
            | (int(self.pyboy.memory[ADDR_XP + 1]) << 8)
            | int(self.pyboy.memory[ADDR_XP + 2])
        )
        frontier_key = -1
        if self.frontier_path and self.best_frontier_score != float("-inf"):
            frontier_key = int(max(self.best_frontier_score, 0.0))
        return (frontier_key, level, xp)

    def _map_label(self, map_id):
        return MAP_LABELS.get(map_id, f"map{map_id}")

    def _party_hp_snapshot(self, party_size=None):
        if party_size is None:
            party_size = self.pyboy.memory[ADDR_PARTY_SIZE]
        party_size = max(0, min(int(party_size), len(PARTY_CUR_HP_ADDRS)))
        snapshot = []
        for idx in range(party_size):
            cur_hp = self._read_u16(*PARTY_CUR_HP_ADDRS[idx])
            max_hp = self._read_u16(*PARTY_MAX_HP_ADDRS[idx])
            if max_hp <= 0 or max_hp > 999:
                continue
            cur_hp = max(0, min(cur_hp, max_hp))
            snapshot.append((cur_hp, max_hp))
        return snapshot

    def _is_invalid_party_state(self, raw_party_size, party_snapshot):
        raw_party_size = int(raw_party_size)
        if raw_party_size < 0 or raw_party_size > len(PARTY_CUR_HP_ADDRS):
            return True
        if raw_party_size > 0 and not party_snapshot:
            return True
        return False

    def _party_hp_totals(self, snapshot):
        return (
            sum(cur_hp for cur_hp, _ in snapshot),
            sum(max_hp for _, max_hp in snapshot),
        )

    def _count_event_flag_bits(self):
        # PWhiddy/PokemonRedExperiments-style progress signal: total set bits
        # across Yellow's wEventFlags bitfield (one bit per story/quest/
        # trainer-beaten event). See ADDR_EVENT_FLAGS_START/END for how the
        # address was derived and cross-checked.
        memory = self.pyboy.memory
        return sum(
            int(memory[addr]).bit_count()
            for addr in range(ADDR_EVENT_FLAGS_START, ADDR_EVENT_FLAGS_END)
        )

    def _format_party_hp(self, snapshot):
        if not snapshot:
            return "none"
        return ",".join(
            f"P{idx + 1}:{cur_hp}/{max_hp}"
            for idx, (cur_hp, max_hp) in enumerate(snapshot)
        )

    def _format_party_hp_totals(self, snapshot):
        if not snapshot:
            return "n/a"
        total_cur_hp, total_max_hp = self._party_hp_totals(snapshot)
        return f"{total_cur_hp}/{total_max_hp}"

    def _format_map_path(self, map_labels):
        if not map_labels:
            return "n/a"
        if len(map_labels) == 1:
            return map_labels[0]
        return " -> ".join(map_labels)

    def _format_frontier_summary(self):
        if not self.frontier_hp_snapshots:
            return None
        best_snapshot = self.frontier_hp_snapshots[-1]
        if best_snapshot.startswith("f"):
            _, summary = best_snapshot.split("@", 1)
            return summary
        return best_snapshot

    @classmethod
    def queue_episode_summary(cls, summary, frontier_summary=None):
        entry = (summary, frontier_summary)
        if cls._pending_episode_summary == entry:
            cls._pending_episode_count += 1
            return
        cls.flush_pending_episode_summaries()
        cls._pending_episode_summary = entry
        cls._pending_episode_count = 1

    @classmethod
    def flush_pending_episode_summaries(cls):
        if cls._pending_episode_summary is None:
            return
        summary, frontier_summary = cls._pending_episode_summary
        label = "[EP]" if cls._pending_episode_count == 1 else f"[EP x{cls._pending_episode_count}]"
        print(f"  {label} {summary}")
        if frontier_summary:
            print(f"       best={frontier_summary}")
        cls._pending_episode_summary = None
        cls._pending_episode_count = 0

    def _record_frontier_hp_snapshot(self, frontier_score, map_id, pos_a, pos_b, battle_flag, party_snapshot):
        total_cur_hp, total_max_hp = self._party_hp_totals(party_snapshot)
        snapshot = f"f{int(frontier_score)}@{self._map_label(map_id)}:{pos_a},{pos_b} hp={total_cur_hp}/{total_max_hp}"
        if battle_flag != 0:
            snapshot += f" {self._battle_label(battle_flag)}"
        if self.frontier_hp_snapshots and self.frontier_hp_snapshots[-1] == snapshot:
            return
        self.frontier_hp_snapshots.append(snapshot)
        self.frontier_hp_snapshots = self.frontier_hp_snapshots[-8:]

    def _wipe_progress_credit(self):
        if (
            self.wipe_progress_max_y_scale == 0.0
            and self.wipe_progress_tiles_scale == 0.0
            and self.wipe_progress_frontier_scale == 0.0
        ):
            return 0.0

        progress_positions = self.visited_positions
        if self.frontier_path:
            frontier_maps = {segment_map for segment_map, _ in self.frontier_path}
            progress_positions = {
                pos for pos in self.visited_positions if pos[0] in frontier_maps
            }

        max_y = max((visit_a for _, visit_a, _ in progress_positions), default=0)
        tiles_explored = len(progress_positions)
        frontier_credit = 0.0
        if self.best_frontier_score != float("-inf"):
            frontier_credit = max(self.best_frontier_score, 0.0)

        return (
            max_y * self.wipe_progress_max_y_scale
            + tiles_explored * self.wipe_progress_tiles_scale
            + frontier_credit * self.wipe_progress_frontier_scale
        )

    def _frontier_score(self, map_id, pos_a, pos_b):
        frontier_path = self.frontier_path
        if not frontier_path:
            return None

        best_score = None
        if self.best_frontier_score == float("-inf"):
            matching_segments = [idx for idx, (segment_map, _) in enumerate(frontier_path) if map_id == segment_map]
            max_segment_index = matching_segments[0] if matching_segments else 0
        else:
            max_segment_index = int(max(self.best_frontier_score, 0.0) // self.frontier_segment_span) + 1
        for segment_index, (segment_map, target) in enumerate(frontier_path):
            if map_id != segment_map:
                continue
            if segment_index > max_segment_index:
                continue

            base_score = segment_index * self.frontier_segment_span
            if isinstance(target, list):
                reached = 0
                for waypoint_index, waypoint in enumerate(target):
                    if self._target_distance(waypoint, pos_a, pos_b) <= 1:
                        reached = max(reached, waypoint_index + 1)
                next_index = min(reached, len(target) - 1)
                next_target = target[next_index]
                distance = self._target_distance(next_target, pos_a, pos_b)
                score = (
                    base_score
                    + reached * self.frontier_waypoint_span
                    - distance
                )
                best_score = score if best_score is None else max(best_score, score)
                continue

            distance = self._target_distance(target, pos_a, pos_b)
            score = base_score + self.frontier_waypoint_span - distance
            best_score = score if best_score is None else max(best_score, score)

        return best_score

    def _frontier_next_target(self, map_id, pos_a, pos_b):
        frontier_path = self.frontier_path
        if not frontier_path:
            return None

        if self.best_frontier_score == float("-inf"):
            matching_segments = [idx for idx, (segment_map, _) in enumerate(frontier_path) if map_id == segment_map]
            max_segment_index = matching_segments[0] if matching_segments else 0
        else:
            max_segment_index = int(max(self.best_frontier_score, 0.0) // self.frontier_segment_span) + 1

        best_candidate = None
        for segment_index, (segment_map, target) in enumerate(frontier_path):
            if map_id != segment_map or segment_index > max_segment_index:
                continue

            if isinstance(target, list):
                reached = 0
                for waypoint_index, waypoint in enumerate(target):
                    if self._target_distance(waypoint, pos_a, pos_b) <= 1:
                        reached = max(reached, waypoint_index + 1)
                next_index = min(reached, len(target) - 1)
                next_target = target[next_index]
                base_score = segment_index * self.frontier_segment_span
                candidate_score = base_score + reached * self.frontier_waypoint_span
            else:
                next_target = target
                candidate_score = segment_index * self.frontier_segment_span

            if best_candidate is None or candidate_score > best_candidate[0]:
                best_candidate = (candidate_score, next_target)

        if best_candidate is None:
            return None
        return best_candidate[1]

    def _claim_zone_bonuses(self, map_id, pos_a, pos_b):
        zone_bonuses = self.zone_bonuses
        if not zone_bonuses:
            return 0.0

        reward = 0.0
        for zone_index, zone in enumerate(zone_bonuses.get(map_id, [])):
            if self._target_distance(zone['target'], pos_a, pos_b) > zone.get('radius', 0):
                continue
            claim_key = (map_id, zone_index)
            if claim_key in self.claimed_zone_bonuses:
                continue
            self.claimed_zone_bonuses.add(claim_key)
            reward += zone['bonus']
        return reward

    def _is_in_named_zone(self, zones, map_id, pos_a, pos_b):
        if not zones:
            return False

        for zone in zones:
            if map_id != zone.get('map'):
                continue
            if self._target_distance(zone['target'], pos_a, pos_b) <= zone.get('radius', 0):
                return True
        return False

    def _action_guidance_reward(self, map_id, pos_a, pos_b, battle_flag, action):
        guidance = self.action_guidance
        if not guidance and self.frontier_action_guidance_bonus <= 0.0:
            return 0.0
        if self.action_guidance_requires_overworld and battle_flag != 0:
            return 0.0

        reward = 0.0
        for rule in guidance or []:
            if map_id != rule['map']:
                continue
            if self._target_distance(rule['target'], pos_a, pos_b) > rule.get('radius', 0):
                continue
            preferred = rule['action']
            if not isinstance(preferred, (list, tuple, set)):
                preferred = [preferred]
            if int(action) in preferred:
                reward += rule.get('bonus', 0.0)
            else:
                reward -= rule.get('penalty', 0.0)

        if self.frontier_action_guidance_bonus > 0.0:
            next_target = self._frontier_next_target(map_id, pos_a, pos_b)
            preferred = []
            if isinstance(next_target, tuple):
                target_y, target_x = next_target
                if pos_a > target_y:
                    preferred.append(0)
                elif pos_a < target_y:
                    preferred.append(1)
                if pos_b > target_x:
                    preferred.append(2)
                elif pos_b < target_x:
                    preferred.append(3)
            elif next_target is not None:
                if pos_a > next_target:
                    preferred.append(0)
                elif pos_a < next_target:
                    preferred.append(1)

            if preferred:
                if int(action) in preferred:
                    reward += self.frontier_action_guidance_bonus
                else:
                    reward -= self.frontier_action_guidance_penalty
        return reward

    def _apply_auto_wait(self):
        zones = self.auto_wait_zones
        if not zones:
            return 0

        map_id = self.pyboy.memory[ADDR_MAP_ID]
        pos_a = self.pyboy.memory[ADDR_POS_A]
        pos_b = self.pyboy.memory[ADDR_POS_B]

        for zone in zones:
            if map_id != zone.get('map'):
                continue
            if self._target_distance(zone['target'], pos_a, pos_b) > zone.get('radius', 0):
                continue

            ticks = int(zone.get('ticks', 0))
            if ticks <= 0:
                return 0

            remaining = ticks
            while remaining > 0:
                chunk = min(remaining, 32)
                self.pyboy.tick(chunk)
                remaining -= chunk
                if self.force_repel:
                    self.pyboy.memory[ADDR_REPEL] = 255
            return ticks

        return 0

    def _settle_after_possible_transition(self, prev_map_id):
        ticks = self.post_transition_settle_ticks
        if ticks <= 0:
            return 0

        map_id = int(self.pyboy.memory[ADDR_MAP_ID])
        should_settle = map_id != int(prev_map_id) or map_id == 255
        if (
            not should_settle
            and self.allowed_maps is not None
            and map_id not in self.allowed_maps
        ):
            should_settle = True
        if not should_settle:
            return 0

        remaining = ticks
        while remaining > 0:
            chunk = min(remaining, 32)
            self.pyboy.tick(chunk)
            remaining -= chunk
            if self.force_repel:
                self.pyboy.memory[ADDR_REPEL] = 255
        return ticks

    def _clear_stale_battle_flag_if_ready(self):
        if self.stale_battle_flag_clear_steps <= 0:
            return False

        battle_flag = int(self.pyboy.memory[ADDR_BATTLE_FLAG])
        enemy_hp = int(self.pyboy.memory[ADDR_ENEMY_HP])
        text_box = int(self.pyboy.memory[ADDR_TEXT_BOX])
        if battle_flag not in (1, 2) or enemy_hp > 0 or text_box != 0:
            if battle_flag == 0 or enemy_hp > 0:
                self.zero_hp_battle_step_count = 0
            return False

        if self.zero_hp_battle_step_count < self.stale_battle_flag_clear_steps:
            return False

        self.pyboy.memory[ADDR_BATTLE_FLAG] = 0
        self.last_battle_flag = 0
        self.current_battle_step_count = 0
        self.zero_hp_battle_step_count = 0
        self.stale_battle_flag_clears += 1
        return True

    def _enemy_hp_can_indicate_battle(self, map_id):
        return self.enemy_hp_battle_proxy_maps is None or int(map_id) in self.enemy_hp_battle_proxy_maps

    def _ensure_forced_attack_ready(self):
        # force_a_in_wild_battles/force_a_in_trainer_battles blindly mash A and
        # rely on move slot 0 being a damaging move with PP left (same assumption
        # ensure_post_brock_route_state's _ensure_lead_attack_memory makes at
        # stage load). That assumption silently breaks mid-episode once slot 0's
        # PP runs dry or a level-up/heal shuffles a status move into slot 0 --
        # the result is exactly the "trainer=1/N battle_stall" loops documented
        # around battle_step_limit: A gets pressed forever with nothing to
        # actually resolve the fight. Re-assert the same guarantee every step
        # instead of only once per stage load.
        memory = self.pyboy.memory
        if int(memory[ADDR_PARTY_SIZE]) <= 0:
            return
        move_addrs = PARTY_MOVE_ID_ADDRS[0]
        pp_addrs = PARTY_MOVE_PP_ADDRS[0]
        move_id = int(memory[move_addrs[0]])
        move_info = GEN1_MOVE_TABLE.get(move_id)
        if move_info is None or move_info['power'] <= 0:
            _ensure_lead_attack_memory(memory)
            move_id = int(memory[move_addrs[0]])
            move_info = GEN1_MOVE_TABLE.get(move_id, GEN1_MOVE_TABLE[PEWTER_FALLBACK_ATTACK_MOVE_ID])
        pp = int(memory[pp_addrs[0]]) & 0x3F
        if pp <= 0:
            pp_ups = int(memory[pp_addrs[0]]) & 0xC0
            memory[pp_addrs[0]] = pp_ups | min(move_info['max_pp'], 0x3F)

        # The FIGHT menu's PP display and "No PP left for this move!" check
        # read wBattleMon's own copy (ADDR_BATTLE_MON_PP), not the party
        # struct refilled above -- it's only copied in from the party struct
        # at battle start or on a mid-battle move-learn, so topping up the
        # party copy alone doesn't stop the in-battle copy from running dry
        # and getting the fight stuck. Keep both in sync the same way.
        battle_pp_addr = ADDR_BATTLE_MON_PP[0]
        battle_move_id = int(memory[ADDR_BATTLE_MON_MOVE_IDS[0]])
        battle_move_info = GEN1_MOVE_TABLE.get(battle_move_id, move_info)
        battle_pp = int(memory[battle_pp_addr]) & 0x3F
        if battle_pp <= 0:
            battle_pp_ups = int(memory[battle_pp_addr]) & 0xC0
            memory[battle_pp_addr] = battle_pp_ups | min(battle_move_info['max_pp'], 0x3F)

    def step(self, action):
        self.recent_actions.append(int(action))
        self._clear_stale_battle_flag_if_ready()
        prev_map_id = self.pyboy.memory[ADDR_MAP_ID]
        prev_pos_a = self.pyboy.memory[ADDR_POS_A]
        prev_pos_b = self.pyboy.memory[ADDR_POS_B]
        prev_overworld_flag = self.pyboy.memory[ADDR_BATTLE_FLAG]
        prev_enemy_hp_for_action = int(self.pyboy.memory[ADDR_ENEMY_HP])
        prev_text_box_for_action = int(self.pyboy.memory[ADDR_TEXT_BOX])
        battle_active_before_action = (
            prev_overworld_flag in (1, 2)
            or (
                prev_enemy_hp_for_action > 0
                and self._enemy_hp_can_indicate_battle(prev_map_id)
            )
        )
        action_index = int(action)
        if 0 <= action_index < len(self.actions):
            action_name = self.actions[action_index]
        else:
            # Backward-compatibility for legacy checkpoints that still emit a
            # now-removed noop branch while the live milestone action list is shorter.
            action_name = 'noop'
        if (
            battle_active_before_action
            and (
                self.force_a_in_wild_battles
                or self.force_a_in_trainer_battles
                or self.ensure_battle_move_ready
            )
        ):
            self._ensure_forced_attack_ready()
        if self.force_a_in_wild_battles and prev_overworld_flag == 1:
            # Wild-grind stages should learn routing, not spend entire
            # episodes drifting around battle menus. Repeated A is the safe
            # default once an encounter has started.
            action_name = 'a'
        trainer_battle_active_before_action = (
            prev_overworld_flag == 2
            or (
                battle_active_before_action
                and int(prev_map_id) in self.trainer_battle_maps
            )
        )
        if self.trainer_battle_action_script and trainer_battle_active_before_action:
            action_name = self.trainer_battle_action_script[
                self.trainer_battle_action_index % len(self.trainer_battle_action_script)
            ]
            self.trainer_battle_action_index += 1
        elif self.force_a_in_trainer_battles and trainer_battle_active_before_action:
            # Brock-stage policy should learn how to reach trainers; once a
            # trainer battle starts, repeated A keeps menu handling deterministic.
            action_name = 'a'
        elif not trainer_battle_active_before_action:
            self.trainer_battle_action_index = 0
        if (
            self.force_a_on_zero_enemy_hp_text
            and prev_text_box_for_action != 0
            and prev_enemy_hp_for_action == 0
        ):
            action_name = 'a'
        in_battle_before_action = battle_active_before_action
        if self.force_repel:
            self.pyboy.memory[ADDR_REPEL] = 255
        if action_name == 'noop':
            noop_ticks = self.battle_noop_ticks if in_battle_before_action else self.noop_ticks
            self.pyboy.tick(noop_ticks)
        else:
            is_direction_action = action_name in {'up', 'down', 'left', 'right'}
            if is_direction_action:
                press_ticks = (
                    self.battle_direction_press_ticks
                    if in_battle_before_action
                    else self.direction_press_ticks
                )
                release_ticks = (
                    self.battle_direction_release_ticks
                    if in_battle_before_action
                    else self.direction_release_ticks
                )
            else:
                press_ticks = (
                    self.battle_button_press_ticks
                    if in_battle_before_action
                    else self.button_press_ticks
                )
                release_ticks = (
                    self.battle_button_release_ticks
                    if in_battle_before_action
                    else self.button_release_ticks
                )
            self.pyboy.button_press(action_name)
            self.pyboy.tick(press_ticks)
            self.pyboy.button_release(action_name)
            self.pyboy.tick(release_ticks)
        if self.force_repel:
            self.pyboy.memory[ADDR_REPEL] = 255
        self._apply_auto_wait()
        if self.force_repel:
            self.pyboy.memory[ADDR_REPEL] = 255
        settled_ticks = self._settle_after_possible_transition(prev_map_id)
        if settled_ticks:
            self.transition_settle_events += 1
            self.transition_settle_ticks_used += settled_ticks
            if self.force_repel:
                self.pyboy.memory[ADDR_REPEL] = 255
        obs = self._get_obs()
        total = self._get_reward()
        reward = total - self.last_reward
        self.last_reward = total

        map_id = self.pyboy.memory[ADDR_MAP_ID]
        pos_a = self.pyboy.memory[ADDR_POS_A]
        pos_b = self.pyboy.memory[ADDR_POS_B]
        prev_battle_flag = self.last_battle_flag
        prev_enemy_hp = self.last_enemy_hp
        progress_event = False

        # Step penalty — incentivizes faster completion
        reward -= self.step_penalty

        # Battle-aware reward — HP delta when in a wild encounter
        # Guard with enemy_hp > 0 since battle_flag can be stale in menus/shops
        battle_flag = int(self.pyboy.memory[ADDR_BATTLE_FLAG])
        enemy_hp = int(self.pyboy.memory[ADDR_ENEMY_HP])
        text_box = int(self.pyboy.memory[ADDR_TEXT_BOX])
        if battle_flag in (1, 2) and enemy_hp == 0 and text_box == 0:
            self.zero_hp_battle_step_count += 1
            if self._clear_stale_battle_flag_if_ready():
                battle_flag = int(self.pyboy.memory[ADDR_BATTLE_FLAG])
        elif battle_flag == 0 or enemy_hp > 0:
            self.zero_hp_battle_step_count = 0

        if (
            battle_flag not in (1, 2)
            and enemy_hp > 0
            and self._enemy_hp_can_indicate_battle(map_id)
        ):
            # Yellow can expose live trainer battle intro/menu states before
            # D057 flips; enemy HP is the reliable signal that battle controls
            # and rewards should already be active.
            battle_flag = 1
            self.enemy_hp_proxy_battle_frames += 1
        if battle_flag == 1 and int(map_id) in self.trainer_battle_maps:
            battle_flag = 2
        if battle_flag in (1, 2) and enemy_hp > 0:
            if prev_enemy_hp > 0:
                hp_delta = prev_enemy_hp - enemy_hp
                if hp_delta > 0:
                    reward += hp_delta * 2  # reward damage dealt
            self.last_enemy_hp = enemy_hp
        else:
            self.last_enemy_hp = 0  # reset when not in battle

        # Tile exploration bonus (bypasses differential)
        pos_key = (map_id, pos_a, pos_b)
        if pos_key not in self.visited_positions:
            self.visited_positions.add(pos_key)
            reward += self.tile_exploration_bonus

        if battle_flag == 0:
            coord_count = self.coord_visit_counts.get(pos_key, 0) + 1
            self.coord_visit_counts[pos_key] = coord_count
            self.max_coord_repeat = max(self.max_coord_repeat, coord_count)
            penalty_exempt = self._is_in_named_zone(self.penalty_exempt_zones, map_id, pos_a, pos_b)
            if not penalty_exempt:
                threshold = self.coord_stuck_threshold
                if threshold and coord_count >= threshold:
                    reward -= self.coord_stuck_penalty
                if self.action_oscillation_penalty and self._is_recent_action_oscillation():
                    reward -= self.action_oscillation_penalty
                if action_name not in {'up', 'down', 'left', 'right'}:
                    reward -= self.overworld_non_movement_penalty

            if self.explore_coord_reward_scale or self.revisit_stuck_threshold:
                # PWhiddy/PokemonRedExperiments-style per-episode coordinate
                # exploration + mild revisit penalty. Separate from
                # coord_visit_counts/max_coord_repeat above (which drive the
                # existing coord_stuck_penalty) so this is purely additive
                # and only active for stages that opt in.
                episode_visits = self.episode_coord_visits.get(pos_key, 0)
                if episode_visits == 0 and self.explore_coord_reward_scale:
                    reward += self.explore_coord_reward_scale
                self.episode_coord_visits[pos_key] = episode_visits + 1
                if (
                    self.revisit_stuck_threshold
                    and episode_visits + 1 >= self.revisit_stuck_threshold
                ):
                    reward -= self.revisit_stuck_penalty

        if self.event_flag_reward_scale:
            event_bits = self._count_event_flag_bits()
            event_progress = max(0, event_bits - self.base_event_flags_bits)
            if event_progress > self.max_event_flags_progress:
                reward += self.event_flag_reward_scale * (
                    event_progress - self.max_event_flags_progress
                )
                self.max_event_flags_progress = event_progress
                progress_event = True

        if (
            self.route4_progress_reward_scale
            and map_id == 15
            and self.route4_start_x is not None
        ):
            route4_progress = max(0, pos_b - self.route4_start_x)
            if route4_progress > self.max_route4_x_progress:
                reward += self.route4_progress_reward_scale * (
                    route4_progress - self.max_route4_x_progress
                )
                self.max_route4_x_progress = route4_progress
                progress_event = True

        reward += self._action_guidance_reward(
            prev_map_id,
            prev_pos_a,
            prev_pos_b,
            prev_overworld_flag,
            action,
        )

        # New map discovery bonus — big reward for finding doors/exits
        if map_id not in self.visited_maps:
            self.visited_maps.add(map_id)
            reward += self.map_discovery_bonus

        map_step_penalties = self.map_step_penalties
        if map_step_penalties:
            reward -= map_step_penalties.get(map_id, 0.0)

        # Party state can become nonsensical after bad memory transitions; clamp
        # before using it for rewards so corruption cannot become a training goal.
        raw_party_size = int(self.pyboy.memory[ADDR_PARTY_SIZE])
        party_size = max(0, min(raw_party_size, len(PARTY_CUR_HP_ADDRS)))
        party_snapshot = self._party_hp_snapshot(party_size)
        invalid_party_state = self._is_invalid_party_state(raw_party_size, party_snapshot)
        party_size_unchanged_for_heal = party_size == self.last_party_size
        if not invalid_party_state and party_size > self.last_party_size:
            reward += self.party_size_bonus_scale * (party_size - self.last_party_size)
            self.last_party_size = party_size
        total_party_hp, total_party_max_hp = self._party_hp_totals(party_snapshot)
        self.min_total_party_hp = (
            total_party_hp
            if self.min_total_party_hp is None
            else min(self.min_total_party_hp, total_party_hp)
        )
        if self.heal_reward_scale and total_party_max_hp > 0:
            # PWhiddy/PokemonRedExperiments-style healing reward: squared
            # increase in party HP fraction, excluding steps where party
            # size just changed (e.g. catching a mon), mirroring
            # update_heal_reward exactly.
            cur_hp_fraction = total_party_hp / total_party_max_hp
            if (
                self.last_hp_fraction is not None
                and party_size_unchanged_for_heal
                and cur_hp_fraction > self.last_hp_fraction
            ):
                heal_amount = cur_hp_fraction - self.last_hp_fraction
                reward += self.heal_reward_scale * (heal_amount ** 2)
            self.last_hp_fraction = cur_hp_fraction
        if (
            battle_flag == 0
            and not self.milestone_hit
            and total_party_max_hp > 0
            and total_party_hp >= total_party_max_hp
            and int(self.pyboy.memory[ADDR_TEXT_BOX]) == 0
        ):
            full_hp_map_step_penalties = self.full_hp_map_step_penalties
            if full_hp_map_step_penalties:
                reward -= full_hp_map_step_penalties.get(map_id, 0.0)

        if battle_flag == 1:
            self.wild_battle_steps += 1
            self.current_battle_step_count += 1
            reward -= self.wild_battle_step_penalty
        elif battle_flag == 2:
            self.trainer_battle_steps += 1
            self.current_battle_step_count += 1
        else:
            self.current_battle_step_count = 0

        if battle_flag in (1, 2) or prev_battle_flag in (1, 2):
            if self.battle_a_bonus and action_name == 'a':
                reward += self.battle_a_bonus
            if self.battle_direction_penalty and action_name in {'up', 'down', 'left', 'right'}:
                reward -= self.battle_direction_penalty

        if battle_flag in (1, 2) and prev_battle_flag not in (1, 2):
            if battle_flag == 1:
                self.wild_battle_entries += 1
                reward -= self.wild_battle_entry_penalty
            else:
                self.trainer_battle_entries += 1
                reward += self.trainer_battle_entry_bonus
            progress_event = True

        if prev_enemy_hp > 0 and enemy_hp == 0 and prev_battle_flag in (1, 2):
            if prev_battle_flag == 1:
                self.wild_enemy_faints += 1
            else:
                self.trainer_enemy_faints += 1
            progress_event = True

        if self.wild_faint_bonus and self.wild_enemy_faints > self.last_wild_faints:
            reward += self.wild_faint_bonus * (self.wild_enemy_faints - self.last_wild_faints)
        self.last_wild_faints = self.wild_enemy_faints
        if self.trainer_faint_bonus and self.trainer_enemy_faints > self.last_trainer_faints:
            reward += self.trainer_faint_bonus * (self.trainer_enemy_faints - self.last_trainer_faints)
        self.last_trainer_faints = self.trainer_enemy_faints
        if (
            self.trainer_faint_hit_count > 0
            and self.trainer_enemy_faints >= self.trainer_faint_hit_count
        ):
            milestone_just_hit = not self.milestone_hit
            if milestone_just_hit:
                reward += self.milestone_hit_bonus
            self.milestone_hit = True
            progress_event = True
            if milestone_just_hit:
                self.last_dist = None

        if self.level_up_bonus:
            cur_level = int(self.pyboy.memory[ADDR_LEVEL])
            if cur_level > self.last_level > 0:
                reward += self.level_up_bonus * (cur_level - self.last_level)
                progress_event = True
            self.last_level = cur_level

        hp_loss = max(0, self.last_total_party_hp - total_party_hp)
        if hp_loss > 0:
            reward -= hp_loss * self.hp_loss_penalty_scale

        party_wiped = (
            not invalid_party_state
            and total_party_hp == 0
            and self.last_total_party_hp > 0
        )
        if party_wiped:
            self.party_wipes += 1
            reward -= self.wipe_penalty
            self.last_wipe_progress_credit = self._wipe_progress_credit()
            reward += self.last_wipe_progress_credit
        else:
            self.last_wipe_progress_credit = 0.0
        if invalid_party_state:
            reward -= self.invalid_party_state_penalty
        self.last_total_party_hp = total_party_hp
        self.last_battle_flag = battle_flag

        # Movement shaping — milestone-aware target positions
        # Supports: int (Y target), tuple (Y,X target), or list of (Y,X) waypoints
        cur_dist = None
        targets = self.shaping_targets
        if self.milestone_hit and self.post_hit_shaping_targets:
            targets = self.post_hit_shaping_targets
        elif (
            battle_flag == 0
            and not self.milestone_hit
            and self.low_hp_retreat_threshold > 0
            and self.low_hp_retreat_shaping_targets
            and total_party_hp <= self.low_hp_retreat_threshold
        ):
            targets = self.low_hp_retreat_shaping_targets
        if targets and map_id in targets:
            t = targets[map_id]
            if isinstance(t, list):
                # Waypoint list — find the furthest reached waypoint, shape toward next
                best_wp = 0
                for i, (wy, wx) in enumerate(t):
                    if abs(pos_a - wy) + abs(pos_b - wx) <= 1:
                        best_wp = max(best_wp, i + 1)
                old_progress = getattr(self, 'waypoint_progress', 0)
                self.waypoint_progress = max(old_progress, best_wp)
                # One-time bonus for each new waypoint reached
                if self.waypoint_progress > old_progress:
                    new_wps = self.waypoint_progress - old_progress
                    reward += new_wps * 200
                    self.last_dist = None  # avoid negative spike from distance jump
                # Shape toward next unreached waypoint
                next_wp = min(self.waypoint_progress, len(t) - 1)
                wy, wx = t[next_wp]
                cur_dist = abs(pos_a - wy) + abs(pos_b - wx)
            elif isinstance(t, tuple):
                ty, tx = t
                cur_dist = abs(pos_a - ty) + abs(pos_b - tx)
            else:
                cur_dist = abs(pos_a - t)

        if cur_dist is not None and self.last_dist is not None:
            dist_delta = self.last_dist - cur_dist
            reward += dist_delta * 10.0
        self.last_dist = cur_dist

        frontier_score = self._frontier_score(map_id, pos_a, pos_b)
        if frontier_score is not None and frontier_score > self.best_frontier_score:
            frontier_delta = frontier_score - self.best_frontier_score
            if self.best_frontier_score == float("-inf"):
                frontier_delta = 0.0
            reward += frontier_delta * self.frontier_reward_scale
            self.best_frontier_score = frontier_score
            self.frontier_stall_count = 0
            self._record_frontier_hp_snapshot(
                frontier_score, map_id, pos_a, pos_b, battle_flag, party_snapshot
            )
        elif frontier_score is not None and battle_flag == 0:
            self.frontier_stall_count += 1

        if (
            battle_flag == 0
            and self.frontier_soft_stall_steps
            and self.frontier_soft_stall_penalty > 0.0
            and self.frontier_soft_stall_score is not None
            and self.best_frontier_score != float("-inf")
            and self.best_frontier_score >= self.frontier_soft_stall_score
            and self.frontier_stall_count >= self.frontier_soft_stall_steps
        ):
            soft_stall_overage = self.frontier_stall_count - self.frontier_soft_stall_steps + 1
            reward -= soft_stall_overage * self.frontier_soft_stall_penalty

        if (
            self.frontier_hit_score is not None
            and frontier_score is not None
            and frontier_score >= self.frontier_hit_score
        ):
            milestone_just_hit = not self.milestone_hit
            if not self.milestone_hit:
                reward += self.milestone_hit_bonus
            self.milestone_hit = True
            if milestone_just_hit:
                self.last_dist = None

        reward += self._claim_zone_bonuses(map_id, pos_a, pos_b)

        # Track max map reward reached this episode
        map_reward = MAP_REWARDS.get(map_id, 0)
        self.max_map_reward = max(self.max_map_reward, map_reward)

        # Track milestone hit using the current milestone's check function
        check_fn = self.milestone_check
        if check_fn is not None and SHOW_MILESTONE_DEBUG:
            if self.steps == 1:  # Debug: log once per episode
                try:
                    result = check_fn(self.pyboy.memory, map_id)
                    print(f"  [DEBUG] milestone_check(map={map_id}, y={pos_a}, x={pos_b}, party={party_size}) = {result}")
                except Exception as e:
                    print(f"  [DEBUG] milestone_check EXCEPTION: {e}")
        route_requirement_met = (
            not self.required_route_maps
            or self.required_route_maps.issubset(self.visited_maps)
        )
        if check_fn is not None and route_requirement_met and check_fn(self.pyboy.memory, map_id):
            milestone_just_hit = not self.milestone_hit
            if not self.milestone_hit:
                reward += self.milestone_hit_bonus
            self.milestone_hit = True
            if milestone_just_hit:
                self.last_dist = None

        progress_checkpoint_maps = self.progress_checkpoint_maps
        current_progress_key = self._progress_checkpoint_key()
        progress_checkpoint_improved = (
            not invalid_party_state
            and current_progress_key > self.best_observed_progress_checkpoint
        )
        if progress_checkpoint_improved:
            self.best_observed_progress_checkpoint = current_progress_key
            progress_event = True
        if (
            progress_checkpoint_maps
            and battle_flag == 0
            and int(self.pyboy.memory[ADDR_TEXT_BOX]) == 0
            and total_party_hp >= 1
            and map_id in progress_checkpoint_maps
            and current_progress_key > self.best_progress_checkpoint
        ):
            reward += self.progress_checkpoint_bonus
            self._save_progress_checkpoint()
            self.best_progress_checkpoint = current_progress_key
            progress_event = True

        if progress_event or battle_flag in (1, 2):
            self.progressless_step_count = 0
        else:
            self.progressless_step_count += 1
        if progress_event and (battle_flag in (1, 2) or prev_battle_flag in (1, 2)):
            # A battle that has already produced real progress (entry, KO, level-up)
            # should get a fresh stall budget for the follow-up text/transition.
            self.current_battle_step_count = 0
        if self.milestone_hit or progress_checkpoint_improved:
            self.checkpoint_progress_step_count = 0
        else:
            self.checkpoint_progress_step_count += 1

        safe_hit_maps = self.safe_hit_save_maps
        can_save_hit_state = (
            self.milestone_hit
            and self.latest_hit_state is None
            and battle_flag == 0
            and int(self.pyboy.memory[ADDR_TEXT_BOX]) == 0
            and total_party_hp >= self.safe_hit_min_total_hp
            and (not safe_hit_maps or map_id in safe_hit_maps)
        )
        if can_save_hit_state:
            reward += self.safe_hit_save_bonus
            self._save_latest_hit_state()

        self.steps += 1
        # Track positions before terminal summaries so the final transition is visible.
        if not hasattr(self, '_ep_positions'):
            self._ep_positions = []
        self._ep_positions.append((map_id, pos_a, pos_b))

        swarm_catchup_needed = False
        if self.swarm_enabled and self.steps % self.swarm_check_interval == 0:
            swarm_catchup_needed = self._swarm_sync()

        terminated = self.steps >= self.max_steps
        termination_reason = "max_steps" if terminated else None
        if not terminated and swarm_catchup_needed:
            # Neutral administrative cutoff, not a failure -- no penalty.
            # Ends this episode now so the next reset() loads whatever the
            # swarm's current frontier state is instead of continuing to
            # re-explore ground another worker has already cleared.
            terminated = True
            termination_reason = "swarm_catchup"
        if (
            not terminated
            and self.allowed_maps is not None
            and map_id not in self.allowed_maps
        ):
            reward -= self.disallowed_map_penalty
            terminated = True
            termination_reason = f"disallowed_map={map_id}"
        if (
            not terminated
            and self.frontier_stall_steps
            and self.best_frontier_score != float("-inf")
            and self.frontier_stall_count >= self.frontier_stall_steps
        ):
            reward -= self.frontier_stall_penalty
            terminated = True
            termination_reason = "frontier_stall"
        if (
            not terminated
            and self.terminate_on_invalid_party_state
            and invalid_party_state
        ):
            terminated = True
            termination_reason = f"invalid_party={raw_party_size}"
        if not terminated and self.terminate_on_party_wipe and party_wiped:
            terminated = True
            termination_reason = "party_wipe"
        if (
            not terminated
            and self.checkpoint_progress_step_limit > 0
            and self.checkpoint_progress_step_count >= self.checkpoint_progress_step_limit
        ):
            reward -= self.checkpoint_progress_step_penalty
            terminated = True
            termination_reason = "checkpoint_stall"
        if (
            not terminated
            and self.progressless_step_limit > 0
            and self.progressless_step_count >= self.progressless_step_limit
        ):
            reward -= self.progressless_step_penalty
            terminated = True
            termination_reason = "progressless_stall"
        if (
            not terminated
            and self.battle_step_limit > 0
            and self.current_battle_step_count >= self.battle_step_limit
        ):
            reward -= self.battle_step_limit_penalty
            terminated = True
            termination_reason = "battle_stall"
        if (
            not terminated
            and self.terminate_after_hit_state_saved
            and self.latest_hit_state is not None
        ):
            terminated = True
            termination_reason = "hit_state_saved"
        elif not terminated and self.terminate_on_milestone_hit and self.milestone_hit:
            terminated = True
            termination_reason = "milestone_hit"
        info = {}
        if terminated:
            info['milestone_hit'] = self.milestone_hit
            info['max_map_reward'] = self.max_map_reward
            info['party_size'] = party_size
            info['raw_party_size'] = raw_party_size
            info['map_id'] = map_id
            info['level'] = int(self.pyboy.memory[ADDR_LEVEL])
            info['battle_flag'] = battle_flag
            info['text_box'] = int(self.pyboy.memory[ADDR_TEXT_BOX])
            info['total_party_hp'] = total_party_hp
            info['termination_reason'] = termination_reason
            info['env_id'] = self.env_id
            info['latest_hit_state'] = (
                os.fspath(self.latest_hit_state) if self.latest_hit_state is not None else None
            )
            # Log min/max Y reached and all maps visited this episode
            if hasattr(self, '_ep_positions'):
                maps_seen = set(m for m, _, _ in self._ep_positions)
                ys = [y for _, y, _ in self._ep_positions]
                frontier_text = (
                    f"{self.best_frontier_score:.1f}"
                    if self.best_frontier_score != float("-inf")
                    else "n/a"
                )
                map_labels = [self._map_label(m) for m in sorted(maps_seen)]
                route_text = self._format_map_path(map_labels)
                min_hp_text = (
                    f"{self.min_total_party_hp}/{total_party_max_hp}"
                    if total_party_max_hp > 0 and self.min_total_party_hp is not None
                    else "n/a"
                )
                end_hp_text = self._format_party_hp_totals(party_snapshot)
                segments = [
                    f"route(maps_visited)={route_text}",
                    f"pos_y_range={min(ys)}..{max(ys)}",
                    f"unique_tiles_visited={len(self.visited_positions)}",
                    f"frontier_score={frontier_text}",
                    f"party_hp(min->end)={min_hp_text}->{end_hp_text}",
                ]
                if termination_reason:
                    segments.append(f"end_reason={termination_reason}")
                if invalid_party_state:
                    segments.append(f"invalid_party_size={raw_party_size}")
                if self.wild_battle_entries or self.wild_battle_steps:
                    segments.append(f"wild_battle_entries={self.wild_battle_entries}")
                    segments.append(f"wild_battle_steps={self.wild_battle_steps}")
                if self.trainer_battle_entries or self.trainer_battle_steps:
                    segments.append(f"trainer_battle_entries={self.trainer_battle_entries}")
                    segments.append(f"trainer_battle_steps={self.trainer_battle_steps}")
                if self.wild_enemy_faints or self.trainer_enemy_faints:
                    segments.append(f"wild_enemy_faints={self.wild_enemy_faints}")
                    segments.append(f"trainer_enemy_faints={self.trainer_enemy_faints}")
                if self.party_wipes:
                    segments.append(f"party_wipes={self.party_wipes}")
                    if self.last_wipe_progress_credit > 0.0:
                        segments.append(f"wipe_progress_credit={self.last_wipe_progress_credit:.0f}")
                if self.transition_settle_events:
                    segments.append(f"map_transition_settle_events={self.transition_settle_events}")
                    segments.append(f"map_transition_settle_ticks={self.transition_settle_ticks_used}")
                if self.stale_battle_flag_clears:
                    # Count of times a battle got stuck with the enemy already
                    # at 0 HP and no textbox up (post-KO hang) long enough that
                    # we forced ADDR_BATTLE_FLAG back to 0 to unstick it.
                    segments.append(f"stuck_post_ko_forced_clears={self.stale_battle_flag_clears}")
                if self.enemy_hp_proxy_battle_frames:
                    segments.append(f"enemy_hp_proxy_battle_frames={self.enemy_hp_proxy_battle_frames}")
                if self.max_coord_repeat > 1:
                    segments.append(f"max_consecutive_same_tile_steps={self.max_coord_repeat}")

                self.__class__.queue_episode_summary(
                    " | ".join(segments),
                    self._format_frontier_summary(),
                )
                frontier_summary = self._format_frontier_summary()
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        swarm_state_loaded = False
        if self.swarm_enabled and self.swarm_frontier_state_path.exists():
            try:
                with open(self.swarm_frontier_state_path, 'rb') as f:
                    self.pyboy.load_state(f)
                swarm_state_loaded = True
            except OSError:
                # Rare race with another worker's concurrent save; fall back
                # to the milestone's own start state below rather than fail.
                swarm_state_loaded = False
        if not swarm_state_loaded:
            with open(self.state_file, 'rb') as f:
                self.pyboy.load_state(f)
        if self.force_repel:
            self.pyboy.memory[ADDR_REPEL] = 255
        for _ in range(self.load_settle_ticks):
            self.pyboy.tick()
            if self.force_repel:
                self.pyboy.memory[ADDR_REPEL] = 255
        self.steps = 0
        self.visited_positions = set()
        self.coord_visit_counts = {}
        self.last_dist = None
        self.waypoint_progress = 0
        self._ep_positions = []
        self.visited_maps = set()
        self.last_party_size = 0
        self.milestone_hit = False
        self.max_map_reward = 0
        self.latest_hit_state = None
        self.best_frontier_score = float("-inf")
        self.frontier_stall_count = 0
        current_progress_key = self._progress_checkpoint_key()
        self.best_progress_checkpoint = current_progress_key
        self.best_observed_progress_checkpoint = current_progress_key
        self.claimed_zone_bonuses = set()
        self.frontier_hp_snapshots = []
        self.wild_battle_entries = 0
        self.trainer_battle_entries = 0
        self.wild_battle_steps = 0
        self.trainer_battle_steps = 0
        self.wild_enemy_faints = 0
        self.trainer_enemy_faints = 0
        self.transition_settle_events = 0
        self.transition_settle_ticks_used = 0
        self.stale_battle_flag_clears = 0
        self.enemy_hp_proxy_battle_frames = 0
        self.party_wipes = 0
        self.last_wipe_progress_credit = 0.0
        self.min_total_party_hp = None
        self.last_total_party_hp = 0
        self.last_battle_flag = 0
        self.last_enemy_hp = 0
        self.max_coord_repeat = 0
        self.last_level = int(self.pyboy.memory[ADDR_LEVEL])
        self.last_wild_faints = 0
        self.last_trainer_faints = 0
        self.recent_actions.clear()
        self.trainer_battle_action_index = 0
        self.current_battle_step_count = 0
        self.zero_hp_battle_step_count = 0
        self.progressless_step_count = 0
        self.checkpoint_progress_step_count = 0
        self.episode_coord_visits = {}
        self.base_event_flags_bits = self._count_event_flag_bits()
        self.max_event_flags_progress = 0
        self.max_route4_x_progress = 0
        self.route4_start_x = (
            int(self.pyboy.memory[ADDR_POS_B])
            if int(self.pyboy.memory[ADDR_MAP_ID]) == 15
            else None
        )
        self.last_reward = self._get_reward()
        starting_party_hp = self._party_hp_snapshot()
        self.last_total_party_hp, starting_party_max_hp = self._party_hp_totals(starting_party_hp)
        self.min_total_party_hp = self.last_total_party_hp
        self.last_hp_fraction = (
            self.last_total_party_hp / starting_party_max_hp
            if starting_party_max_hp > 0
            else None
        )
        return self._get_obs(), {}

    def save_screenshot(self, path):
        self.pyboy.screen.image.save(path)
        return path

    def flush_episode_summaries(self):
        self.__class__.flush_pending_episode_summaries()
        return True

    def close(self):
        self.pyboy.stop()


class ProgressionCallback(BaseCallback):
    """Monitors milestone completion rate and triggers auto-advance."""

    def __init__(self, milestone, milestones_dir, hit_threshold=0.7,
                 window_size=50, verbose=1):
        super().__init__(verbose)
        self.milestone = milestone
        self.milestones_dir = milestones_dir
        self.current_name = milestone['name']
        self.next_name = milestone['next_name']
        self.next_state_path = milestone_state_path(self.next_name)
        self.hit_threshold = hit_threshold
        self.window_size = window_size
        self.episode_hits = deque(maxlen=window_size)
        self.reward_threshold = milestone.get('reward_threshold')
        self.reward_above_count = 0
        self._recent_ep_rewards = deque(maxlen=50)  # last 50 episode rewards
        self.advanced = False
        self.best_state_saved = False
        self._last_value_reset = 0  # timestep of last value head reset
        self._value_reset_check_every = 50_000
        self._value_reset_cooldown = 200_000
        self._next_value_reset_check = None
        self._critic_collapse_checks = 0
        self._stage_start_time = time.time()
        os.makedirs(milestones_dir, exist_ok=True)

    def _next_state_is_valid(self):
        return (
            self.next_state_path.exists()
            and state_file_matches_milestone_requirements(self.next_state_path, self.next_name)
        )

    def _candidate_hit_states(self, preferred_source=None):
        candidates = []
        seen = set()

        if preferred_source:
            source = os.fspath(preferred_source)
            if source not in seen:
                candidates.append(source)
                seen.add(source)

        for source in list_recent_hit_states(self.current_name):
            if source in seen:
                continue
            try:
                if os.path.getmtime(source) + 1 < self._stage_start_time:
                    continue
            except OSError:
                continue
            candidates.append(source)
            seen.add(source)

        return candidates

    def _copy_hit_state(self, preferred_source=None):
        if self.current_name == '05f_gate_transition' and self.next_name == '05g_to_pewter':
            source = repair_05g_to_pewter_state(
                target_state=self.next_state_path,
                source_state=preferred_source,
                backup_existing=True,
            )
            self.best_state_saved = True
            print(
                "  Built repaired 05g_to_pewter.state:",
                f"source={os.path.basename(source)}",
            )
            return os.fspath(self.next_state_path)
        if self.current_name == '05g_to_pewter' and self.next_name == '06_pewter_city':
            source = repair_06_pewter_city_state(
                target_state=self.next_state_path,
                source_state=preferred_source,
                backup_existing=True,
            )
            self.best_state_saved = True
            print(
                "  Built repaired 06_pewter_city.state:",
                f"source={os.path.basename(source)}",
            )
            return os.fspath(self.next_state_path)
        if self.current_name == '06c_pewter_lvl13' and self.next_name == '06d_beat_brock':
            source = repair_pewter_stage_state(
                target_state=self.next_state_path,
                milestone_name=self.next_name,
                source_state=preferred_source,
                backup_existing=True,
            )
            self.best_state_saved = True
            print(
                "  Built repaired 06d_beat_brock.state:",
                f"source={os.path.basename(source)}",
            )
            return os.fspath(self.next_state_path)

        for source in self._candidate_hit_states(preferred_source):
            if not os.path.exists(source):
                continue
            snapshot = probe_state_snapshot(source)
            if not snapshot_matches_milestone_requirements(snapshot, self.next_name):
                continue
            shutil.copy2(source, self.next_state_path)
            self.best_state_saved = True
            return source
        return None

    def _flush_episode_summaries(self):
        try:
            self.training_env.env_method('flush_episode_summaries')
        except Exception:
            PokemonYellowEnv.flush_pending_episode_summaries()

    def _should_reset_value_head(self, value_loss, explained_variance):
        # Mild negative explained variance is common in PPO; only reset on a
        # sustained low-loss collapse or a clearly severe critic failure.
        if value_loss is None or explained_variance is None:
            self._critic_collapse_checks = 0
            return False

        severe_collapse = explained_variance <= -5.0 and value_loss <= 5e-4
        sustained_collapse = explained_variance <= -0.5 and value_loss <= 2e-4

        if severe_collapse:
            self._critic_collapse_checks = 0
            return True

        if sustained_collapse:
            self._critic_collapse_checks += 1
        else:
            self._critic_collapse_checks = 0

        if self._critic_collapse_checks >= 2:
            self._critic_collapse_checks = 0
            return True
        return False

    def _maybe_auto_reset_value_head(self):
        if self._next_value_reset_check is None:
            current_window = self.num_timesteps // self._value_reset_check_every
            self._next_value_reset_check = (current_window + 1) * self._value_reset_check_every

        ran_check = False
        while self.num_timesteps >= self._next_value_reset_check:
            ran_check = True
            self._next_value_reset_check += self._value_reset_check_every

        if not ran_check:
            return

        if self.num_timesteps - self._last_value_reset <= self._value_reset_cooldown:
            self._critic_collapse_checks = 0
            return

        logger = self.model.logger.name_to_value
        value_loss = logger.get('train/value_loss')
        explained_variance = logger.get('train/explained_variance')
        if not self._should_reset_value_head(value_loss, explained_variance):
            return

        print(
            "  [AUTO-RESET]",
            f"value_loss={value_loss:.6f}, explained_variance={explained_variance:.3f}",
            "— resetting value head",
        )
        reset_value_head(self.model)
        self._last_value_reset = self.num_timesteps

    def _on_rollout_start(self):
        if self.advanced:
            return
        # Reset only between rollouts so the collected buffer is never split
        # across two different critics.
        self._maybe_auto_reset_value_head()

    def _on_step(self):
        if self.advanced:
            return False

        # SB3 stores episode info from Monitor wrapper in self.locals
        infos = self.locals.get('infos', [])
        for info in infos:
            # Monitor wraps terminal info in 'terminal_info' on auto-reset
            terminal_info = info.get('terminal_info', info)

            if 'milestone_hit' in terminal_info:
                hit = terminal_info['milestone_hit']
                self.episode_hits.append(hit)

                # Save state on first hit — use per-env state from the env that hit
                if hit and not self.best_state_saved:
                    source = self._copy_hit_state(terminal_info.get('latest_hit_state'))
                    if source is not None:
                        self._flush_episode_summaries()
                        print(f"\n{'='*60}")
                        print(f"  MILESTONE FIRST HIT: {self.next_name}")
                        print(f"  source={os.path.basename(source)}")
                        print(f"  party={terminal_info.get('party_size', '?')} "
                              f"max_map={terminal_info.get('max_map_reward', '?')}")
                        print(f"  Timestep: {self.num_timesteps}")
                        print(f"{'='*60}\n")
                    else:
                        print(f"  WARNING: milestone hit detected for {self.current_name}, "
                              f"but no saved hit state was found yet. "
                              f"map={terminal_info.get('map_id', '?')} "
                              f"level={terminal_info.get('level', '?')} "
                              f"battle={terminal_info.get('battle_flag', '?')} "
                              f"text={terminal_info.get('text_box', '?')} "
                              f"hp={terminal_info.get('total_party_hp', '?')} "
                              f"end={terminal_info.get('termination_reason', '?')}")

        # Check hit rate
        if len(self.episode_hits) >= self.window_size:
            hit_rate = sum(self.episode_hits) / len(self.episode_hits)
            if self.num_timesteps % 50000 < 10:
                print(f"  [Milestone: {self.next_name}] hit_rate={hit_rate:.0%} "
                      f"({sum(self.episode_hits)}/{len(self.episode_hits)})")
            if hit_rate >= self.hit_threshold:
                # Ensure next milestone has a state file before advancing
                if not self._next_state_is_valid():
                    source = self._copy_hit_state()
                    if source is not None:
                        self._flush_episode_summaries()
                        print(f"  Created {self.next_name}.state from {os.path.basename(source)}")
                    else:
                        print(f"  WARNING: No state file for {self.next_name} and no hit states available!")
                        print(f"  Cannot advance without a state file.")
                        return True  # keep training until a hit state is saved
                print(f"\n{'='*60}")
                print(f"  MILESTONE MASTERED: {self.milestone['name']}")
                print(f"  Hit rate: {hit_rate:.0%} over last {self.window_size} episodes")
                print(f"  Advancing to: {self.next_name}")
                print(f"{'='*60}\n")
                self.advanced = True
                return False

        # Advance on first hit + minimum training time
        # Agent proved it can reach the goal; no need to master before moving on.
        # The unified model will improve on earlier sections as it trains later ones.
        min_steps = self.milestone.get('min_steps_to_advance', 200_000)
        if self.best_state_saved and self.num_timesteps >= min_steps and self._next_state_is_valid():
            hit_rate = sum(self.episode_hits) / max(len(self.episode_hits), 1)
            print(f"\n{'='*60}")
            print(f"  MILESTONE ADVANCING (first hit + {min_steps//1000}K steps)")
            print(f"  {self.milestone['name']} -> {self.next_name}")
            print(f"  Current hit rate: {hit_rate:.0%}")
            print(f"{'='*60}\n")
            self.advanced = True
            return False

        # Fallback: advance based on reward threshold from Monitor episode rewards
        if self.reward_threshold is not None:
            # Collect episode rewards from Monitor's 'episode' key in infos
            for info in infos:
                if 'episode' in info:
                    self._recent_ep_rewards.append(info['episode']['r'])
            # Check mean of recent episodes
            ep_rew = None
            if len(self._recent_ep_rewards) >= 20:
                ep_rew = sum(self._recent_ep_rewards) / len(self._recent_ep_rewards)
            if ep_rew is not None and ep_rew >= self.reward_threshold:
                self.reward_above_count += 1
            else:
                self.reward_above_count = 0

            if self.num_timesteps % 50000 < 10 and ep_rew is not None:
                print(f"  [Reward check] mean={ep_rew:.0f} threshold={self.reward_threshold} count={self.reward_above_count}/20")
            if self.reward_above_count >= 20:
                # Ensure we have a state file to advance to
                if not self._next_state_is_valid():
                    self._copy_hit_state()
                if self._next_state_is_valid():
                    print(f"\n{'='*60}")
                    print(f"  MILESTONE MASTERED (reward): {self.milestone['name']}")
                    print(f"  ep_rew_mean={ep_rew:.0f} >= {self.reward_threshold} "
                          f"for {self.reward_above_count} iterations")
                    print(f"  Advancing to: {self.next_name}")
                    print(f"{'='*60}\n")
                    self.advanced = True
                    return False
                else:
                    print(f"  [WARNING] Reward threshold met but no state file found at {self.next_state_path}")

        return True


class ScreenshotCallback(BaseCallback):
    def __init__(self, save_path, save_freq=5000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            screenshot_path = os.path.join(self.save_path, f'step_{self.num_timesteps:08d}.png')
            try:
                self.training_env.env_method('save_screenshot', screenshot_path, indices=0)
            except Exception:
                env = self.training_env.envs[0]
                while hasattr(env, 'env'):
                    env = env.env
                env.pyboy.screen.image.save(screenshot_path)
        return True


class RolloutProgressCallback(BaseCallback):
    """Prints progress while PPO is still collecting a rollout."""

    def __init__(self, report_every_vec_steps=512, verbose=0):
        super().__init__(verbose)
        self.report_every_vec_steps = report_every_vec_steps

    def _on_step(self):
        rollout_len = getattr(self.model, "n_steps", None)
        n_envs = getattr(self.training_env, "num_envs", 1)
        if rollout_len and self.n_calls % self.report_every_vec_steps == 0:
            rollout_step = self.n_calls % rollout_len
            if rollout_step == 0:
                rollout_step = rollout_len
            if rollout_step == rollout_len:
                return True
            total_env_steps = rollout_step * n_envs
            rollout_total_env_steps = rollout_len * n_envs
            print(
                f"  [ROLLOUT] {rollout_step}/{rollout_len} vec steps "
                f"({total_env_steps}/{rollout_total_env_steps} env steps)"
            )
        return True


class SimpleStatusCallback(BaseCallback):
    """Prints guided metrics with short explanations after each rollout."""

    def __init__(self, stage_name, verbose=0):
        super().__init__(verbose)
        self.stage_name = stage_name
        self._recent_ep_rewards = deque(maxlen=50)
        self._recent_hits = deque(maxlen=50)
        self._recent_ep_lengths = deque(maxlen=50)
        self._start_time = time.time()
        self._start_timesteps = 0

    def _on_training_start(self):
        PokemonYellowEnv.flush_pending_episode_summaries()
        self._start_time = time.time()
        self._start_timesteps = self.num_timesteps
        print(f"  Session step baseline: {self._start_timesteps:,} lifetime steps")

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            terminal_info = info.get('terminal_info', info)
            if 'episode' in info:
                self._recent_ep_rewards.append(info['episode']['r'])
                self._recent_ep_lengths.append(info['episode']['l'])
            if 'milestone_hit' in terminal_info:
                self._recent_hits.append(int(bool(terminal_info['milestone_hit'])))
        return True

    def _reward_summary(self, ep_rew):
        if ep_rew is None:
            return "warming up"
        if ep_rew >= 1000:
            return "strong"
        if ep_rew >= 0:
            return "good"
        if ep_rew >= -500:
            return "mixed"
        return "bad"

    def _critic_summary(self, explained_variance):
        if explained_variance is None:
            return "warming up"
        if explained_variance >= 0.7:
            return "healthy"
        if explained_variance >= 0.0:
            return "okay"
        return "weak"

    def _value_loss_summary(self, value_loss, explained_variance):
        if value_loss is None:
            return "warming up"
        if value_loss < 0.001 and explained_variance is not None and explained_variance < 0.0:
            return "watch critic"
        return "lower better"

    def _approx_kl_summary(self, approx_kl):
        if approx_kl is None:
            return "warming up"
        if approx_kl <= 0.02:
            return "normal"
        return "large"

    def _clip_fraction_summary(self, clip_fraction):
        if clip_fraction is None:
            return "warming up"
        if clip_fraction <= 0.15:
            return "normal"
        return "high"

    def _entropy_summary(self, entropy_loss):
        if entropy_loss is None:
            return "warming up"
        return "exploration"

    def _format_metric_row(self, name, value, note):
        return f"    {name:<16} {value:<10} {note}"

    def _on_rollout_end(self):
        PokemonYellowEnv.flush_pending_episode_summaries()

        ep_rew = None
        if self._recent_ep_rewards:
            ep_rew = sum(self._recent_ep_rewards) / len(self._recent_ep_rewards)

        ep_len = None
        if self._recent_ep_lengths:
            ep_len = sum(self._recent_ep_lengths) / len(self._recent_ep_lengths)

        hit_total = sum(self._recent_hits)
        hit_window = len(self._recent_hits)
        hit_text = f"{hit_total}/{hit_window}" if hit_window else "0/0"
        hit_status = "no clears yet" if hit_total == 0 else "stage hits seen"

        logger_values = self.model.logger.name_to_value
        explained_variance = logger_values.get('train/explained_variance')
        value_loss = logger_values.get('train/value_loss')
        approx_kl = logger_values.get('train/approx_kl')
        clip_fraction = logger_values.get('train/clip_fraction')
        entropy_loss = logger_values.get('train/entropy_loss')
        current_level = None
        try:
            levels = self.training_env.get_attr('last_level')
            if levels:
                current_level = max(int(level) for level in levels)
        except Exception:
            current_level = None
        elapsed = max(time.time() - self._start_time, 1e-6)
        session_steps = self.num_timesteps - self._start_timesteps
        fps = int(session_steps / elapsed)
        fps_text = "warmup" if elapsed < 5 or session_steps <= 0 else str(fps)

        reward_text = "n/a" if ep_rew is None else f"{ep_rew:.0f}"
        length_text = "n/a" if ep_len is None else f"{ep_len:.0f}"
        level_text = "n/a" if current_level is None else str(current_level)

        approx_kl_text = "n/a" if approx_kl is None else f"{approx_kl:.4f}"
        clip_fraction_text = "n/a" if clip_fraction is None else f"{clip_fraction:.3f}"
        entropy_text = "n/a" if entropy_loss is None else f"{entropy_loss:.2f}"
        explained_text = "n/a" if explained_variance is None else f"{explained_variance:.3f}"
        value_loss_text = "n/a" if value_loss is None else f"{value_loss:.6f}"

        print(
            f"  [METRICS] {self.stage_name} | steps={self.num_timesteps:,} (total, all sessions) | "
            f"session_steps={session_steps:,} (since this process launched) | fps={fps_text}"
        )
        print(self._format_metric_row("avg_ep_reward", reward_text, self._reward_summary(ep_rew)))
        print(self._format_metric_row("milestone_hits", hit_text, hit_status + " (of last 50 episodes)"))
        print(self._format_metric_row("pikachu_level", level_text, "Pikachu's current level"))
        print(self._format_metric_row("critic_expl_var", explained_text, self._critic_summary(explained_variance)))
        print(self._format_metric_row("value_loss", value_loss_text, self._value_loss_summary(value_loss, explained_variance)))
        print(self._format_metric_row("kl_divergence", approx_kl_text, self._approx_kl_summary(approx_kl)))
        print(self._format_metric_row("clip_fraction", clip_fraction_text, self._clip_fraction_summary(clip_fraction)))
        print(self._format_metric_row("entropy_loss", entropy_text, self._entropy_summary(entropy_loss)))
        print(self._format_metric_row("avg_ep_length", length_text, "steps per episode"))
        print()


class CheckpointPruneCallback(BaseCallback):
    """Keeps checkpoints/ bounded during long runs instead of only cleaning up
    between milestones. 07_beat_brock alone ran 176M+ steps in one stretch
    before ever advancing, which is long enough on its own to reaccumulate
    the same multi-GB bloat this was added to fix (2026-07-20)."""

    def __init__(self, active_stage_name, prune_every_calls, verbose=0):
        super().__init__(verbose)
        self.active_stage_name = active_stage_name
        self.prune_every_calls = max(1, int(prune_every_calls))

    def _on_step(self):
        if self.n_calls % self.prune_every_calls == 0:
            removed = prune_checkpoint_history(self.active_stage_name)
            if removed:
                print(f"  [CheckpointPrune] removed {removed} superseded checkpoint file(s)")
        return True


def list_recent_hit_states(milestone_name):
    return sorted(
        glob.glob(latest_hit_glob(milestone_name)),
        key=os.path.getmtime,
        reverse=True,
    )


def _snapshot_party_hp_totals(memory):
    raw_party_size = int(memory[ADDR_PARTY_SIZE])
    party_size = max(0, min(raw_party_size, len(PARTY_CUR_HP_ADDRS)))
    total_cur_hp = 0
    total_max_hp = 0
    for idx in range(party_size):
        cur_hp = (memory[PARTY_CUR_HP_ADDRS[idx][0]] << 8) | memory[PARTY_CUR_HP_ADDRS[idx][1]]
        max_hp = (memory[PARTY_MAX_HP_ADDRS[idx][0]] << 8) | memory[PARTY_MAX_HP_ADDRS[idx][1]]
        if max_hp <= 0 or max_hp > 999:
            continue
        total_cur_hp += max(0, min(cur_hp, max_hp))
        total_max_hp += max_hp
    return party_size, total_cur_hp, total_max_hp


def _heal_party_memory(memory):
    party_size = max(0, min(int(memory[ADDR_PARTY_SIZE]), len(PARTY_CUR_HP_ADDRS)))
    for idx in range(party_size):
        cur_hi, cur_lo = PARTY_CUR_HP_ADDRS[idx]
        max_hi, max_lo = PARTY_MAX_HP_ADDRS[idx]
        max_hp = (int(memory[max_hi]) << 8) | int(memory[max_lo])
        memory[cur_hi] = (max_hp >> 8) & 0xFF
        memory[cur_lo] = max_hp & 0xFF

    if party_size > 0:
        lead_hi, lead_lo = PARTY_CUR_HP_ADDRS[0]
        memory[ADDR_ACTIVE_MON_CUR_HP_HI] = int(memory[lead_hi])
        memory[ADDR_ACTIVE_MON_CUR_HP_LO] = int(memory[lead_lo])
        memory[ADDR_ACTIVE_MON_MAX_HP_HI] = int(memory[PARTY_MAX_HP_ADDRS[0][0]])
        memory[ADDR_ACTIVE_MON_MAX_HP_LO] = int(memory[PARTY_MAX_HP_ADDRS[0][1]])


def _restore_party_pp_memory(memory):
    party_size = max(0, min(int(memory[ADDR_PARTY_SIZE]), len(PARTY_MOVE_ID_ADDRS)))
    for idx in range(party_size):
        for move_addr, pp_addr in zip(PARTY_MOVE_ID_ADDRS[idx], PARTY_MOVE_PP_ADDRS[idx]):
            move_id = int(memory[move_addr])
            if move_id <= 0:
                memory[pp_addr] = 0
                continue
            move_info = GEN1_MOVE_TABLE.get(move_id)
            if move_info is None:
                continue
            pp_ups = int(memory[pp_addr]) & 0xC0
            memory[pp_addr] = pp_ups | min(move_info['max_pp'], 0x3F)


def _ensure_lead_attack_memory(memory):
    # Only intervenes when slot 0 isn't a real attack at all (status move,
    # unrecognized ID, or empty slot) -- doesn't churn move order just because
    # a marginally stronger option exists elsewhere, so this stays a no-op on
    # the hot path once slot 0 is already fine.
    if int(memory[ADDR_PARTY_SIZE]) <= 0:
        return False

    move_addrs = PARTY_MOVE_ID_ADDRS[0]
    pp_addrs = PARTY_MOVE_PP_ADDRS[0]

    current_move_id = int(memory[move_addrs[0]])
    current_move_info = GEN1_MOVE_TABLE.get(current_move_id)
    if current_move_info is not None and current_move_info['power'] > 0:
        return False

    # Slot 0 has nothing useful -- pull in the strongest damaging move
    # Pikachu currently knows (ranked by real ROM move power, preferring one
    # with PP left over one that's merely stronger on paper but spent).
    best_slot = None
    best_key = None
    for idx, (move_addr, pp_addr) in enumerate(zip(move_addrs, pp_addrs)):
        if idx == 0:
            continue
        move_id = int(memory[move_addr])
        move_info = GEN1_MOVE_TABLE.get(move_id)
        if move_info is None or move_info['power'] <= 0:
            continue
        pp = int(memory[pp_addr]) & 0x3F
        key = (pp > 0, move_info['power'])
        if best_key is None or key > best_key:
            best_key = key
            best_slot = idx

    if best_slot is None:
        memory[move_addrs[0]] = PEWTER_FALLBACK_ATTACK_MOVE_ID
        memory[pp_addrs[0]] = GEN1_MOVE_TABLE[PEWTER_FALLBACK_ATTACK_MOVE_ID]['max_pp']
        return True

    attack_slot = best_slot
    first_move = int(memory[move_addrs[0]])
    first_pp = int(memory[pp_addrs[0]])
    memory[move_addrs[0]] = int(memory[move_addrs[attack_slot]])
    memory[pp_addrs[0]] = int(memory[pp_addrs[attack_slot]])
    memory[move_addrs[attack_slot]] = first_move
    memory[pp_addrs[attack_slot]] = first_pp
    return True


def _snapshot_lead_move_pps(memory):
    if int(memory[ADDR_PARTY_SIZE]) <= 0:
        return []
    return [int(memory[addr]) & 0x3F for addr in PARTY_MOVE_PP_ADDRS[0]]


def _snapshot_lead_move_ids(memory):
    if int(memory[ADDR_PARTY_SIZE]) <= 0:
        return []
    return [int(memory[addr]) for addr in PARTY_MOVE_ID_ADDRS[0]]


def probe_state_snapshot(state_file):
    pyboy = PyBoy(str(ROM_PATH), window='null')
    try:
        with open(state_file, 'rb') as handle:
            pyboy.load_state(handle)
        mem = pyboy.memory
        party_size, total_cur_hp, total_max_hp = _snapshot_party_hp_totals(mem)
        return {
            'map_id': int(mem[ADDR_MAP_ID]),
            'pos_a': int(mem[ADDR_POS_A]),
            'pos_b': int(mem[ADDR_POS_B]),
            'battle_flag': int(mem[ADDR_BATTLE_FLAG]),
            'text_box': int(mem[ADDR_TEXT_BOX]),
            'enemy_hp': int(mem[ADDR_ENEMY_HP]),
            'level': int(mem[ADDR_LEVEL]),
            'badges': int(mem[ADDR_BADGES]).bit_count(),
            'party_size': party_size,
            'total_party_hp': total_cur_hp,
            'total_party_max_hp': total_max_hp,
            'lead_move_ids': _snapshot_lead_move_ids(mem),
            'lead_move_pps': _snapshot_lead_move_pps(mem),
        }
    finally:
        pyboy.stop()


def snapshot_is_overworld_safe(snapshot, min_total_hp=1):
    return (
        snapshot['battle_flag'] == 0
        and snapshot.get('text_box', 0) == 0
        and snapshot.get('total_party_hp', 0) >= min_total_hp
    )


def milestone_state_requirements(milestone_name):
    if milestone_name == '05g_to_pewter':
        return {
            'allowed_maps': {47, 13, 2},
            'min_level': None,
            'min_total_hp': 1,
        }
    if milestone_name == '06d_beat_brock':
        return {
            'allowed_maps': PEWTER_BROCK_START_MAPS,
            'min_level': PEWTER_STAGE_START_LEVELS[milestone_name],
            'min_total_hp': 1,
            'require_zero_enemy_hp': True,
        }
    if milestone_name == '06c_pewter_lvl13':
        return {
            'allowed_maps': PEWTER_STAGE_SAFE_MAPS,
            'min_level': PEWTER_STAGE_START_LEVELS[milestone_name],
            'min_total_hp': 1,
            'require_zero_enemy_hp': True,
        }
    if milestone_name in PEWTER_STAGE_START_LEVELS:
        return {
            'allowed_maps': PEWTER_STAGE_START_MAPS,
            'min_level': PEWTER_STAGE_START_LEVELS[milestone_name],
            'min_total_hp': 1,
            'require_zero_enemy_hp': True,
        }
    return None


def snapshot_matches_milestone_requirements(snapshot, milestone_name):
    requirements = milestone_state_requirements(milestone_name)
    if requirements is None:
        return snapshot_is_overworld_safe(snapshot, min_total_hp=1)

    if not snapshot_is_overworld_safe(snapshot, min_total_hp=requirements['min_total_hp']):
        return False
    allowed_maps = requirements.get('allowed_maps')
    if allowed_maps is not None and snapshot['map_id'] not in allowed_maps:
        return False
    min_level = requirements.get('min_level')
    if min_level is not None and snapshot.get('level', 0) < min_level:
        return False
    if requirements.get('require_zero_enemy_hp') and snapshot.get('enemy_hp', 0) > 0:
        return False
    return True


def state_file_matches_milestone_requirements(state_file, milestone_name):
    try:
        snapshot = probe_state_snapshot(state_file)
    except Exception:
        return False
    return snapshot_matches_milestone_requirements(snapshot, milestone_name)


def _copy_party_progress_memory(dst_memory, src_memory):
    for address in range(PARTY_DATA_COPY_START, PARTY_DATA_COPY_END):
        dst_memory[address] = src_memory[address]
    dst_memory[ADDR_BADGES] = src_memory[ADDR_BADGES]


def _save_repaired_pewter_progress_checkpoint(target_path, milestone_name, source_memory):
    target_path = Path(target_path)
    temp_path = target_path.with_name(f"{target_path.stem}.tmp{target_path.suffix}")
    base_state_path = Path(ensure_06_pewter_city_state(milestone_state_path('06_pewter_city')))

    safe_pyboy = PyBoy(str(ROM_PATH), window='null')
    try:
        with open(base_state_path, 'rb') as handle:
            safe_pyboy.load_state(handle)

        _copy_party_progress_memory(safe_pyboy.memory, source_memory)
        _heal_party_memory(safe_pyboy.memory)
        _restore_party_pp_memory(safe_pyboy.memory)
        _ensure_lead_attack_memory(safe_pyboy.memory)
        safe_pyboy.memory[ADDR_BATTLE_FLAG] = 0
        safe_pyboy.memory[ADDR_TEXT_BOX] = 0
        safe_pyboy.memory[ADDR_ENEMY_HP] = 0
        safe_pyboy.memory[ADDR_REPEL] = 255

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, 'wb') as handle:
            safe_pyboy.save_state(handle)
        os.replace(temp_path, target_path)
    finally:
        safe_pyboy.stop()
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass

    snapshot = probe_state_snapshot(target_path)
    if not snapshot_matches_milestone_requirements(snapshot, milestone_name):
        raise RuntimeError(
            f"Saved {milestone_name} progress checkpoint is invalid: "
            f"map={snapshot['map_id']} level={snapshot.get('level')}"
        )


def _pick_pewter_stage_repair_source(milestone_name, target_path, preferred_source=None):
    min_level = PEWTER_STAGE_START_LEVELS[milestone_name]
    candidate_paths = []
    seen = set()

    def add_candidate(path_like):
        if not path_like:
            return
        path = Path(path_like)
        key = os.fspath(path)
        if key in seen or not path.exists():
            return
        seen.add(key)
        candidate_paths.append(path)

    add_candidate(preferred_source)
    add_candidate(target_path)

    milestone_idx = MILESTONE_INDEX[milestone_name]
    if milestone_idx > 0:
        previous_name = MILESTONES[milestone_idx - 1]['name']
        for hit_path in list_recent_hit_states(previous_name):
            add_candidate(hit_path)
        add_candidate(milestone_state_path(previous_name))

    add_candidate(milestone_state_path('06_pewter_city'))

    fallback = None
    for path in candidate_paths:
        try:
            snapshot = probe_state_snapshot(path)
        except Exception:
            continue
        if not snapshot_is_overworld_safe(snapshot, min_total_hp=1):
            continue
        if fallback is None:
            fallback = (path, snapshot)
        if snapshot.get('level', 0) >= min_level:
            return path, snapshot

    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"No usable Pewter repair source found for {milestone_name}")


def repair_pewter_stage_state(target_state=None, milestone_name=None, source_state=None, backup_existing=True):
    if milestone_name is None:
        raise ValueError("milestone_name is required for Pewter stage repair")

    target_path = Path(target_state or milestone_state_path(milestone_name))
    preferred_source = Path(source_state) if source_state else None
    source_path, source_snapshot = _pick_pewter_stage_repair_source(
        milestone_name,
        target_path,
        preferred_source=preferred_source,
    )
    base_state_path = Path(ensure_06_pewter_city_state(milestone_state_path('06_pewter_city')))

    if backup_existing and target_path.exists():
        current_snapshot = probe_state_snapshot(target_path)
        if not snapshot_matches_milestone_requirements(current_snapshot, milestone_name):
            backup_name = (
                f"{target_path.stem}.{time.strftime('%Y%m%d_%H%M%S')}"
                f".offroute_backup{target_path.suffix}"
            )
            shutil.copy2(target_path, target_path.with_name(backup_name))

    safe_pyboy = PyBoy(str(ROM_PATH), window='null')
    source_pyboy = PyBoy(str(ROM_PATH), window='null')
    try:
        with open(base_state_path, 'rb') as handle:
            safe_pyboy.load_state(handle)
        with open(source_path, 'rb') as handle:
            source_pyboy.load_state(handle)

        _copy_party_progress_memory(safe_pyboy.memory, source_pyboy.memory)
        _heal_party_memory(safe_pyboy.memory)
        _restore_party_pp_memory(safe_pyboy.memory)
        _ensure_lead_attack_memory(safe_pyboy.memory)
        safe_pyboy.memory[ADDR_BATTLE_FLAG] = 0
        safe_pyboy.memory[ADDR_TEXT_BOX] = 0
        safe_pyboy.memory[ADDR_ENEMY_HP] = 0
        safe_pyboy.memory[ADDR_REPEL] = 255
        if milestone_name == '06d_beat_brock':
            _walk_pewter_pc_to_brock_gym(safe_pyboy)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'wb') as handle:
            safe_pyboy.save_state(handle)
    finally:
        safe_pyboy.stop()
        source_pyboy.stop()

    repaired_snapshot = probe_state_snapshot(target_path)
    if not snapshot_matches_milestone_requirements(repaired_snapshot, milestone_name):
        raise RuntimeError(
            f"Repaired {milestone_name} state is still invalid: "
            f"map={repaired_snapshot['map_id']} level={repaired_snapshot.get('level')}"
        )
    return os.fspath(source_path)


def ensure_pewter_stage_state(state_file, milestone_name, backup_existing=True):
    state_path = Path(state_file)
    if not state_path.exists():
        return os.fspath(state_path)

    snapshot = probe_state_snapshot(state_path)
    if not snapshot_matches_milestone_requirements(snapshot, milestone_name):
        source = repair_pewter_stage_state(
            target_state=state_path,
            milestone_name=milestone_name,
            backup_existing=backup_existing,
        )
        snapshot = probe_state_snapshot(state_path)
        print(
            f"  Repaired {milestone_name} state:",
            f"source={os.path.basename(source)}",
            f"map={snapshot['map_id']} a={snapshot['pos_a']} b={snapshot['pos_b']}",
            f"level={snapshot.get('level', 0)}",
        )
    lead_move_pps = snapshot.get('lead_move_pps') or []
    lead_move_ids = snapshot.get('lead_move_ids') or []
    lead_move_needs_restore = bool(lead_move_pps) and lead_move_pps[0] <= 0
    lead_move_info = GEN1_MOVE_TABLE.get(lead_move_ids[0]) if lead_move_ids else None
    lead_move_needs_attack = (
        bool(lead_move_ids)
        and (
            lead_move_info is None
            or lead_move_info['power'] <= 0
            or not lead_move_pps
            or lead_move_pps[0] <= 0
        )
    )
    if (
        snapshot['total_party_hp'] >= snapshot.get('total_party_max_hp', 0)
        and not lead_move_needs_restore
        and not lead_move_needs_attack
    ):
        return os.fspath(state_path)

    if backup_existing:
        backup_name = (
            f"{state_path.stem}.{time.strftime('%Y%m%d_%H%M%S')}"
            f".pre_heal_backup{state_path.suffix}"
        )
        shutil.copy2(state_path, state_path.with_name(backup_name))

    pyboy = PyBoy(str(ROM_PATH), window='null')
    try:
        with open(state_path, 'rb') as handle:
            pyboy.load_state(handle)
        _heal_party_memory(pyboy.memory)
        _restore_party_pp_memory(pyboy.memory)
        _ensure_lead_attack_memory(pyboy.memory)
        with open(state_path, 'wb') as handle:
            pyboy.save_state(handle)
    finally:
        pyboy.stop()

    healed_snapshot = probe_state_snapshot(state_path)
    print(
        f"  Healed {milestone_name} state:",
        f"map={healed_snapshot['map_id']} a={healed_snapshot['pos_a']} b={healed_snapshot['pos_b']}",
        f"hp={healed_snapshot['total_party_hp']}/{healed_snapshot['total_party_max_hp']}",
        f"moves={healed_snapshot.get('lead_move_ids', [])}",
        f"pp={healed_snapshot.get('lead_move_pps', [])}",
    )
    return os.fspath(state_path)


def ensure_post_brock_route_state(state_file, backup_existing=True):
    state_path = Path(state_file)
    if not state_path.exists():
        return os.fspath(state_path)

    snapshot = probe_state_snapshot(state_path)
    lead_move_pps = snapshot.get('lead_move_pps') or []
    needs_repair = (
        snapshot.get('map_id') not in {2, 14, 68, 59, 60, 61, 15}
        or (
            snapshot.get('map_id') == 2
            and (snapshot.get('pos_a'), snapshot.get('pos_b')) != (17, 37)
        )
        or snapshot.get('badges', 0) < 1
        or snapshot.get('total_party_hp', 0) < snapshot.get('total_party_max_hp', 0)
        or not lead_move_pps
        or lead_move_pps[0] <= 0
    )
    if not needs_repair:
        return os.fspath(state_path)

    if backup_existing:
        backup_name = (
            f"{state_path.stem}.{time.strftime('%Y%m%d_%H%M%S')}"
            f".pre_post_brock_repair{state_path.suffix}"
        )
        shutil.copy2(state_path, state_path.with_name(backup_name))

    pyboy = PyBoy(str(ROM_PATH), window='null')
    try:
        load_path = state_path
        route_seed_candidates = [state_path]
        route_seed_candidates.extend(
            sorted(
                state_path.parent.glob(
                    f"{state_path.stem}.*.pre_post_brock_repair{state_path.suffix}"
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
        for candidate in route_seed_candidates:
            if not candidate.exists():
                continue
            try:
                candidate_snapshot = probe_state_snapshot(candidate)
            except Exception:
                continue
            if (
                candidate_snapshot.get('map_id') == 2
                and candidate_snapshot.get('badges', 0) >= 1
                and snapshot_is_overworld_safe(candidate_snapshot, min_total_hp=1)
            ):
                load_path = candidate
                break
        with open(load_path, 'rb') as handle:
            pyboy.load_state(handle)
        pyboy.memory[ADDR_BADGES] = int(pyboy.memory[ADDR_BADGES]) | 0x01
        _heal_party_memory(pyboy.memory)
        _restore_party_pp_memory(pyboy.memory)
        _ensure_lead_attack_memory(pyboy.memory)
        pyboy.memory[ADDR_MAP_ID] = 2
        pyboy.memory[ADDR_POS_A] = 17
        pyboy.memory[ADDR_POS_B] = 37
        pyboy.memory[ADDR_BATTLE_FLAG] = 0
        pyboy.memory[ADDR_TEXT_BOX] = 0
        pyboy.memory[ADDR_ENEMY_HP] = 0
        pyboy.tick(30)
        with open(state_path, 'wb') as handle:
            pyboy.save_state(handle)
    finally:
        pyboy.stop()

    repaired = probe_state_snapshot(state_path)
    print(
        "  Repaired 07_beat_brock route state:",
        f"map={repaired['map_id']} a={repaired['pos_a']} b={repaired['pos_b']}",
        f"badges={repaired.get('badges', 0)}",
        f"hp={repaired['total_party_hp']}/{repaired['total_party_max_hp']}",
        f"moves={repaired.get('lead_move_ids', [])}",
        f"pp={repaired.get('lead_move_pps', [])}",
    )
    return os.fspath(state_path)


def _press_state_builder_action(pyboy, action_name, press_ticks, release_ticks):
    pyboy.memory[ADDR_REPEL] = 255
    pyboy.button_press(action_name)
    pyboy.tick(press_ticks)
    pyboy.button_release(action_name)
    if release_ticks:
        pyboy.tick(release_ticks)
    pyboy.memory[ADDR_REPEL] = 255


def _walk_pewter_pc_to_brock_gym(pyboy):
    pyboy.tick(30)
    _press_state_builder_action(pyboy, 'down', 96, 30)
    pyboy.tick(60)

    # From the Pewter Pokecenter door to the real Pewter Gym warp.
    for action_name in (
        'right', 'right', 'right',
        'up', 'up', 'up', 'up',
        'right', 'right', 'right',
        'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up', 'up',
        'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left',
        'down', 'down', 'down',
        'left',
        'down', 'down',
        'right', 'right', 'right', 'right', 'right', 'right',
        'up',
    ):
        _press_state_builder_action(pyboy, action_name, 8, 24)
    pyboy.tick(120)
    for _ in range(5):
        _press_state_builder_action(pyboy, 'up', 8, 24)
    pyboy.tick(60)

    final_snapshot = (
        int(pyboy.memory[ADDR_MAP_ID]),
        int(pyboy.memory[ADDR_POS_A]),
        int(pyboy.memory[ADDR_POS_B]),
        int(pyboy.memory[ADDR_BATTLE_FLAG]),
    )
    if final_snapshot[:3] != (PEWTER_GYM_MAP, 8, 4) or final_snapshot[3] != 0:
        raise RuntimeError(
            "06d repair expected to reach the Pewter Gym entrance, "
            f"got map={final_snapshot[0]} a={final_snapshot[1]} b={final_snapshot[2]} "
            f"battle={final_snapshot[3]}"
        )


def repair_05g_to_pewter_state(target_state=None, source_state=None, backup_existing=True):
    target_path = Path(target_state or milestone_state_path('05g_to_pewter'))
    preferred_source = Path(source_state) if source_state else None

    if preferred_source is not None and preferred_source.exists():
        preferred_snapshot = probe_state_snapshot(preferred_source)
        if (
            preferred_snapshot['map_id'] in {47, 13, 2}
            and preferred_snapshot['battle_flag'] == 0
        ):
            shutil.copy2(preferred_source, target_path)
            return os.fspath(preferred_source)

    base_source = milestone_state_path('05f_gate_transition')
    if not base_source.exists():
        raise FileNotFoundError(f"Missing source state: {base_source}")

    if backup_existing and target_path.exists():
        current_snapshot = probe_state_snapshot(target_path)
        if current_snapshot['map_id'] not in {47, 13, 2}:
            backup_name = (
                f"{target_path.stem}.{time.strftime('%Y%m%d_%H%M%S')}"
                f".dead_forest_exit_backup{target_path.suffix}"
            )
            shutil.copy2(target_path, target_path.with_name(backup_name))

    pyboy = PyBoy(str(ROM_PATH), window='null')
    try:
        with open(base_source, 'rb') as handle:
            pyboy.load_state(handle)

        # Walk to a valid gate-building state, then swap to the structurally
        # identical north-gate map so the checkpoint sits on a real 05g corridor.
        for action_name, press_ticks in (
            ('right', 96),
            ('right', 128),
            ('down', 96),
        ):
            _press_state_builder_action(pyboy, action_name, press_ticks, 8)

        if int(pyboy.memory[ADDR_MAP_ID]) != 50:
            raise RuntimeError(
                "05g repair expected to reach map 50 before promotion, "
                f"got map {int(pyboy.memory[ADDR_MAP_ID])}"
            )

        pyboy.memory[ADDR_MAP_ID] = 47
        pyboy.tick(1)
        pyboy.memory[ADDR_REPEL] = 255

        root_io = BytesIO()
        pyboy.save_state(root_io)
        root_bytes = root_io.getvalue()

        pyboy.load_state(BytesIO(root_bytes))
        for action_name in ('up', 'up', 'right', 'up', 'up'):
            _press_state_builder_action(pyboy, action_name, 96, 8)
        if int(pyboy.memory[ADDR_MAP_ID]) != 2:
            raise RuntimeError(
                "Repaired 05g checkpoint did not validate to Pewter City; "
                f"ended on map {int(pyboy.memory[ADDR_MAP_ID])}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'wb') as handle:
            handle.write(root_bytes)
    finally:
        pyboy.stop()

    return os.fspath(base_source)


def ensure_05g_to_pewter_state(state_file):
    state_path = Path(state_file)
    if state_path.exists():
        snapshot = probe_state_snapshot(state_path)
        if snapshot['map_id'] in {47, 13, 2} and snapshot['battle_flag'] == 0:
            return os.fspath(state_path)

    source = repair_05g_to_pewter_state(target_state=state_path, backup_existing=True)
    snapshot = probe_state_snapshot(state_path)
    print(
        "  Repaired 05g_to_pewter.state:",
        f"source={os.path.basename(source)}",
        f"map={snapshot['map_id']} a={snapshot['pos_a']} b={snapshot['pos_b']}",
    )
    return os.fspath(state_path)


def repair_06_pewter_city_state(target_state=None, source_state=None, backup_existing=True):
    target_path = Path(target_state or milestone_state_path('06_pewter_city'))
    preferred_source = Path(source_state) if source_state else None

    if preferred_source is not None and preferred_source.exists():
        preferred_snapshot = probe_state_snapshot(preferred_source)
        if preferred_snapshot['map_id'] == 58 and preferred_snapshot['battle_flag'] == 0:
            shutil.copy2(preferred_source, target_path)
            return os.fspath(preferred_source)

    base_source = preferred_source if preferred_source is not None and preferred_source.exists() else target_path
    if not base_source.exists():
        base_source = milestone_state_path('05g_to_pewter')
    if not base_source.exists():
        raise FileNotFoundError(f"Missing source state for 06 repair: {base_source}")

    if backup_existing and target_path.exists():
        current_snapshot = probe_state_snapshot(target_path)
        if current_snapshot['map_id'] != 58:
            backup_name = (
                f"{target_path.stem}.{time.strftime('%Y%m%d_%H%M%S')}"
                f".south_gate_backup{target_path.suffix}"
            )
            shutil.copy2(target_path, target_path.with_name(backup_name))

    pyboy = PyBoy(str(ROM_PATH), window='null')
    try:
        with open(base_source, 'rb') as handle:
            pyboy.load_state(handle)

        # Walk from the south Pewter entrance to the local Pokecenter frontage.
        for action_name in (
            'up',
            'left', 'left',
            'up',
            'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left', 'left',
            'up',
        ):
            _press_state_builder_action(pyboy, action_name, 8, 16)

        for action_name in ('down', 'down', 'right', 'up'):
            _press_state_builder_action(pyboy, action_name, 96, 8)

        final_snapshot = (
            int(pyboy.memory[ADDR_MAP_ID]),
            int(pyboy.memory[ADDR_POS_A]),
            int(pyboy.memory[ADDR_POS_B]),
            int(pyboy.memory[ADDR_BATTLE_FLAG]),
        )
        if final_snapshot[:3] != (58, 13, 1) or final_snapshot[3] != 0:
            raise RuntimeError(
                "06 repair expected map58 y=13 x=1 overworld, "
                f"got map={final_snapshot[0]} a={final_snapshot[1]} b={final_snapshot[2]} "
                f"battle={final_snapshot[3]}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'wb') as handle:
            pyboy.save_state(handle)
    finally:
        pyboy.stop()

    return os.fspath(base_source)


def ensure_06_pewter_city_state(state_file):
    state_path = Path(state_file)
    if state_path.exists():
        snapshot = probe_state_snapshot(state_path)
        if snapshot['map_id'] == 58 and snapshot['battle_flag'] == 0:
            return os.fspath(state_path)

    source = repair_06_pewter_city_state(target_state=state_path, backup_existing=True)
    snapshot = probe_state_snapshot(state_path)
    print(
        "  Repaired 06_pewter_city.state:",
        f"source={os.path.basename(source)}",
        f"map={snapshot['map_id']} a={snapshot['pos_a']} b={snapshot['pos_b']}",
    )
    return os.fspath(state_path)


def latest_stage_checkpoint(milestone_name):
    return _latest_stage_artifact(milestone_name, extension='zip', vecnormalize=False)


def latest_stage_vecnormalize(milestone_name):
    return _latest_stage_artifact(milestone_name, extension='pkl', vecnormalize=True)


def _latest_stage_artifact(milestone_name, extension, vecnormalize):
    """Return the artifact with the greatest recorded training step.

    File modification time is not a reliable ordering: backup/restore and file
    copies routinely change it. The step embedded by SB3 is authoritative.
    """
    marker = '_vecnormalize_' if vecnormalize else '_'
    pattern = re.compile(
        rf'^poke_{re.escape(milestone_name)}{marker}(\d+)_steps\.{re.escape(extension)}$'
    )
    candidates = []
    for path in Path(CHECKPOINTS_DIR).glob(f'poke_{milestone_name}{marker}*_steps.{extension}'):
        match = pattern.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime_ns, path))
    if not candidates:
        return None
    return os.fspath(max(candidates)[2])


def checkpoint_vecnormalize_path(checkpoint_path):
    checkpoint = Path(checkpoint_path)
    match = re.fullmatch(r'(.+)_(\d+)_steps\.zip', checkpoint.name)
    if match is None:
        raise ValueError(f"Not an SB3 checkpoint path: {checkpoint_path}")
    return os.fspath(checkpoint.with_name(f"{match.group(1)}_vecnormalize_{match.group(2)}_steps.pkl"))


# checkpoints/ grew to 7,290 files / 73GB over ~4 months because nothing ever
# pruned old stage checkpoints (2026-07-20). Falling back to an earlier stage
# only ever needs its milestones/*.state file (see load_progress), never the
# full PPO checkpoint history, so a finished stage only needs one snapshot
# kept as a manual-recovery point. The active stage keeps a deeper window
# since that's the one a crash/corruption (see 07_beat_brock.state, same day)
# would actually need to roll back through.
CHECKPOINT_KEEP_ACTIVE = 20
CHECKPOINT_KEEP_INACTIVE = 1
CHECKPOINT_PRUNE_MIN_AGE_SECONDS = 120
CHECKPOINT_FILE_RE = re.compile(r'^poke_(.+?)_(?:vecnormalize_)?(\d+)_steps\.(zip|pkl)$')


def prune_checkpoint_history(active_stage_name,
                              keep_active=CHECKPOINT_KEEP_ACTIVE,
                              keep_inactive=CHECKPOINT_KEEP_INACTIVE):
    by_stage = {}
    for path in glob.glob(os.path.join(CHECKPOINTS_DIR, 'poke_*_steps.*')):
        match = CHECKPOINT_FILE_RE.match(os.path.basename(path))
        if not match:
            continue
        stage, steps, _ext = match.groups()
        by_stage.setdefault(stage, []).append((int(steps), path))

    now = time.time()
    removed = 0
    for stage, entries in by_stage.items():
        keep_n = keep_active if stage == active_stage_name else keep_inactive
        keep_steps = set(sorted({steps for steps, _ in entries}, reverse=True)[:keep_n])
        for steps, path in entries:
            if steps in keep_steps:
                continue
            try:
                if now - os.path.getmtime(path) < CHECKPOINT_PRUNE_MIN_AGE_SECONDS:
                    continue
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


def should_ignore_stage_checkpoint(state_file, checkpoint_path):
    # State files can be repaired or healed after a training run without making
    # the latest stage checkpoint unusable, so mtime alone is not a good reason
    # to throw away the freshest stage weights.
    return False


def load_progress():
    if os.path.exists(PROGRESS_FILE) or os.path.exists(f"{PROGRESS_FILE}.bak"):
        def validate_progress(value):
            if not isinstance(value, dict):
                raise ValueError("progress root must be a JSON object")
            return {
                'milestone_index': int(value.get('milestone_index', 0)),
                'curriculum_version': int(value.get('curriculum_version', 1)),
            }

        try:
            progress = read_json_with_backup(PROGRESS_FILE, validator=validate_progress)
        except (TypeError, ValueError) as exc:
            print(f"  WARNING: Ignoring unreadable curriculum progress: {exc}")
            return 0
        milestone_index = progress['milestone_index']
        progress_version = progress['curriculum_version']
        if progress_version < CURRICULUM_VERSION and milestone_index >= PRE_PEWTER_BROCK_INDEX:
            milestone_index += PEWTER_CURRICULUM_INSERTIONS
        milestone_index = max(0, min(milestone_index, len(MILESTONES)))
        if milestone_index >= len(MILESTONES):
            return milestone_index
        while milestone_index > 0:
            milestone_name = MILESTONES[milestone_index]['name']
            requirements = milestone_state_requirements(milestone_name)
            if requirements is None:
                break
            state_path = milestone_state_path(milestone_name)
            if state_path.exists() and state_file_matches_milestone_requirements(state_path, milestone_name):
                break
            previous_name = MILESTONES[milestone_index - 1]['name']
            print(
                f"  WARNING: {milestone_name}.state is not a usable start state; "
                f"falling back to {previous_name}."
            )
            milestone_index -= 1
        return max(0, min(milestone_index, len(MILESTONES)))
    return 0

def save_progress(index):
    ensure_runtime_dirs()
    index = int(index)
    if not 0 <= index <= len(MILESTONES):
        raise ValueError(f"milestone index {index} is outside 0..{len(MILESTONES)}")
    atomic_write_json(PROGRESS_FILE, {
        'milestone_index': index,
        'curriculum_version': CURRICULUM_VERSION,
    })

def get_state_file(milestone):
    if milestone['state'] is not None:
        return milestone['state']
    state_path = milestone_state_path(milestone['name'])
    if state_path.exists():
        if milestone['name'] == '05g_to_pewter':
            return ensure_05g_to_pewter_state(state_path)
        if milestone['name'] == '06_pewter_city':
            return ensure_06_pewter_city_state(state_path)
        if milestone['name'] in PEWTER_STAGE_START_LEVELS:
            return ensure_pewter_stage_state(state_path, milestone['name'])
        if milestone['name'] == '07_beat_brock':
            return ensure_post_brock_route_state(state_path)
        return state_path
    milestone_idx = MILESTONE_INDEX[milestone['name']]
    if milestone_idx > 0:
        previous_name = MILESTONES[milestone_idx - 1]['name']
        hit_files = list_recent_hit_states(previous_name)
        if hit_files:
            print(f"  WARNING: No state for {milestone['name']}, "
                  f"using recent hit from {previous_name}: {os.path.basename(hit_files[0])}")
            if milestone['name'] == '05g_to_pewter':
                return ensure_05g_to_pewter_state(state_path)
            if milestone['name'] == '06_pewter_city':
                return ensure_06_pewter_city_state(state_path)
            return hit_files[0]
    print(f"  WARNING: No state for {milestone['name']}, falling back to start.state!")
    if milestone['name'] == '05g_to_pewter':
        return ensure_05g_to_pewter_state(state_path)
    if milestone['name'] == '06_pewter_city':
        return ensure_06_pewter_city_state(state_path)
    return START_STATE_PATH


def build_env_config(state_file, milestone):
    return {
        'state_file': os.fspath(state_file),
        'max_steps': milestone['ep_length'],
        'milestone_name': milestone['name'],
        'milestone_check': milestone['check'],
        'step_penalty': milestone.get('step_penalty', 0.0),
        'shaping_targets': milestone.get('shaping'),
        'post_hit_shaping_targets': milestone.get('post_hit_shaping'),
        'reward_cap': milestone.get('reward_cap'),
        'tile_exploration_bonus': milestone.get('tile_exploration_bonus', 5.0),
        'map_discovery_bonus': milestone.get('map_discovery_bonus', 100.0),
        'map_step_penalties': milestone.get('map_step_penalties'),
        'full_hp_map_step_penalties': milestone.get('full_hp_map_step_penalties'),
        'frontier_path': milestone.get('frontier_path'),
        'frontier_reward_scale': milestone.get('frontier_reward_scale', 0.0),
        'frontier_segment_span': milestone.get('frontier_segment_span', 1000.0),
        'frontier_waypoint_span': milestone.get('frontier_waypoint_span', 100.0),
        'frontier_action_guidance_bonus': milestone.get('frontier_action_guidance_bonus', 0.0),
        'frontier_action_guidance_penalty': milestone.get('frontier_action_guidance_penalty', 0.0),
        'frontier_soft_stall_score': milestone.get('frontier_soft_stall_score'),
        'frontier_soft_stall_steps': milestone.get('frontier_soft_stall_steps', 0),
        'frontier_soft_stall_penalty': milestone.get('frontier_soft_stall_penalty', 0.0),
        'frontier_stall_steps': milestone.get('frontier_stall_steps', 0),
        'frontier_stall_penalty': milestone.get('frontier_stall_penalty', 0.0),
        'frontier_hit_score': milestone.get('frontier_hit_score'),
        'action_guidance': milestone.get('action_guidance'),
        'required_route_maps': milestone.get('required_route_maps'),
        'action_guidance_requires_overworld': milestone.get('action_guidance_requires_overworld', True),
        'penalty_exempt_zones': milestone.get('penalty_exempt_zones'),
        'auto_wait_zones': milestone.get('auto_wait_zones'),
        'force_a_in_wild_battles': milestone.get('force_a_in_wild_battles', False),
        'force_a_in_trainer_battles': milestone.get('force_a_in_trainer_battles', False),
        'ensure_battle_move_ready': milestone.get('ensure_battle_move_ready', False),
        'trainer_battle_action_script': milestone.get('trainer_battle_action_script'),
        'trainer_battle_maps': milestone.get('trainer_battle_maps'),
        'enemy_hp_battle_proxy_maps': milestone.get('enemy_hp_battle_proxy_maps'),
        'force_a_on_zero_enemy_hp_text': milestone.get('force_a_on_zero_enemy_hp_text', False),
        'low_hp_retreat_threshold': milestone.get('low_hp_retreat_threshold', 0),
        'low_hp_retreat_shaping_targets': milestone.get('low_hp_retreat_shaping'),
        'allowed_maps': milestone.get('allowed_maps'),
        'disallowed_map_penalty': milestone.get('disallowed_map_penalty', 0.0),
        'zone_bonuses': milestone.get('zone_bonuses'),
        'milestone_hit_bonus': milestone.get('hit_bonus', 0.0),
        'terminate_on_milestone_hit': milestone.get('terminate_on_hit', False),
        'terminate_after_hit_state_saved': milestone.get('terminate_after_hit_state_saved', False),
        'safe_hit_save_maps': milestone.get('safe_hit_save_maps'),
        'safe_hit_min_total_hp': milestone.get('safe_hit_min_total_hp', 0),
        'safe_hit_save_bonus': milestone.get('safe_hit_save_bonus', 0.0),
        'progress_checkpoint_maps': milestone.get('progress_checkpoint_maps'),
        'progress_checkpoint_bonus': milestone.get('progress_checkpoint_bonus', 0.0),
        'restore_party_resources_on_progress_checkpoint': milestone.get(
            'restore_party_resources_on_progress_checkpoint',
            False,
        ),
        'actions': milestone.get('actions'),
        'noop_ticks': milestone.get('noop_ticks', 24),
        'direction_press_ticks': milestone.get('direction_press_ticks', 8),
        'direction_release_ticks': milestone.get('direction_release_ticks', 16),
        'button_press_ticks': milestone.get('button_press_ticks', 8),
        'button_release_ticks': milestone.get('button_release_ticks', 16),
        'battle_direction_press_ticks': milestone.get('battle_direction_press_ticks'),
        'battle_direction_release_ticks': milestone.get('battle_direction_release_ticks'),
        'battle_button_press_ticks': milestone.get('battle_button_press_ticks'),
        'battle_button_release_ticks': milestone.get('battle_button_release_ticks'),
        'battle_noop_ticks': milestone.get('battle_noop_ticks'),
        'post_transition_settle_ticks': milestone.get('post_transition_settle_ticks', 0),
        'load_settle_ticks': milestone.get('load_settle_ticks', 0),
        'coord_stuck_threshold': milestone.get('coord_stuck_threshold', 0),
        'coord_stuck_penalty': milestone.get('coord_stuck_penalty', 0.0),
        'action_oscillation_penalty': milestone.get('action_oscillation_penalty', 0.0),
        'overworld_non_movement_penalty': milestone.get('overworld_non_movement_penalty', 0.0),
        'force_repel': milestone.get('force_repel', False),
        'wild_battle_step_penalty': milestone.get('wild_battle_step_penalty', 0.0),
        'wild_battle_entry_penalty': milestone.get('wild_battle_entry_penalty', 0.0),
        'hp_loss_penalty_scale': milestone.get('hp_loss_penalty_scale', 0.0),
        'wipe_penalty': milestone.get('wipe_penalty', 0.0),
        'wipe_progress_max_y_scale': milestone.get('wipe_progress_max_y_scale', 0.0),
        'wipe_progress_tiles_scale': milestone.get('wipe_progress_tiles_scale', 0.0),
        'wipe_progress_frontier_scale': milestone.get('wipe_progress_frontier_scale', 0.0),
        'party_size_bonus_scale': milestone.get('party_size_bonus_scale', 2000.0),
        'wild_faint_bonus': milestone.get('wild_faint_bonus', 0.0),
        'trainer_battle_entry_bonus': milestone.get('trainer_battle_entry_bonus', 0.0),
        'trainer_faint_bonus': milestone.get('trainer_faint_bonus', 0.0),
        'trainer_faint_hit_count': milestone.get('trainer_faint_hit_count', 0),
        'level_up_bonus': milestone.get('level_up_bonus', 0.0),
        'battle_a_bonus': milestone.get('battle_a_bonus', 0.0),
        'battle_direction_penalty': milestone.get('battle_direction_penalty', 0.0),
        'battle_step_limit': milestone.get('battle_step_limit', 0),
        'battle_step_limit_penalty': milestone.get('battle_step_limit_penalty', 0.0),
        'stale_battle_flag_clear_steps': milestone.get('stale_battle_flag_clear_steps', 0),
        'progressless_step_limit': milestone.get('progressless_step_limit', 0),
        'progressless_step_penalty': milestone.get('progressless_step_penalty', 0.0),
        'checkpoint_progress_step_limit': milestone.get('checkpoint_progress_step_limit', 0),
        'checkpoint_progress_step_penalty': milestone.get('checkpoint_progress_step_penalty', 0.0),
        'invalid_party_state_penalty': milestone.get('invalid_party_state_penalty', 0.0),
        'terminate_on_invalid_party_state': milestone.get('terminate_on_invalid_party_state', False),
        'terminate_on_party_wipe': milestone.get('terminate_on_party_wipe', False),
        'event_flag_reward_scale': milestone.get('event_flag_reward_scale', 0.0),
        'explore_coord_reward_scale': milestone.get('explore_coord_reward_scale', 0.0),
        'heal_reward_scale': milestone.get('heal_reward_scale', 0.0),
        'revisit_stuck_threshold': milestone.get('revisit_stuck_threshold', 0),
        'revisit_stuck_penalty': milestone.get('revisit_stuck_penalty', 0.0),
        'level_reward_scale': milestone.get('level_reward_scale', 50.0),
        'swarm_enabled': milestone.get('swarm_enabled', False),
        'swarm_check_interval': milestone.get('swarm_check_interval', 64),
        'swarm_catchup_behind_bits': milestone.get('swarm_catchup_behind_bits', 3),
        'route4_progress_reward_scale': milestone.get('route4_progress_reward_scale', 0.0),
    }


def resolve_vec_env_kind(n_envs, requested=None):
    kind = (requested or DEFAULT_VEC_ENV_KIND or 'dummy').strip().lower()
    if kind not in {'dummy', 'subproc'}:
        kind = 'dummy'
    if n_envs <= 1:
        return 'dummy'
    if kind == 'subproc' and platform.system() == 'Windows':
        return 'dummy'
    return kind


def build_vec_env(env_cls, env_config, n_envs, vec_env_kind=None):
    resolved_kind = resolve_vec_env_kind(n_envs, vec_env_kind)

    def make_env(rank):
        def _init():
            config = dict(env_config)
            config['env_id'] = rank
            env = env_cls(config=config)
            if STREAM_ENABLED:
                env = StreamWrapper(
                    env,
                    stream_metadata={
                        "user": STREAM_USER,
                        "env_id": rank,
                        "color": STREAM_COLOR,
                        "extra": "",
                    },
                )
            return Monitor(env)
        return _init

    env_fns = [make_env(rank) for rank in range(n_envs)]
    if resolved_kind == 'subproc':
        env = SubprocVecEnv(env_fns, start_method=DEFAULT_SUBPROC_START_METHOD)
    else:
        env = DummyVecEnv(env_fns)
    return env, resolved_kind


def fail_fast_for_unsupported_windows_gpu():
    wants_gpu = TRAIN_DEVICE.lower() != "cpu"
    if platform.system() == "Windows" and wants_gpu and not ALLOW_UNSUPPORTED_WINDOWS_GPU:
        raise RuntimeError(
            "Native Windows ROCm on Radeon is not a supported ML training path in AMD's current docs. "
            "Use WSL/Linux for GPU training, or set POKEMON_ALLOW_UNSUPPORTED_WINDOWS_GPU=1 to override at your own risk."
        )


def configure_model_logger(model):
    if SHOW_RAW_SB3_TABLES:
        model.verbose = 1
        return

    model.verbose = 0
    sb3_log_dir = CHECKPOINTS_DIR / "sb3_logs"
    sb3_log_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(sb3_log_dir), ["csv"]))


def apply_ppo_overrides(model, env, ppo_kwargs):
    # SB3 may wrap image VecEnvs (for example with VecTransposeImage) when the
    # env is attached to the model. Rebuild buffers against the model's live env
    # so stage-to-stage handoffs keep observation layout consistent.
    active_env = model.get_env() or env
    model.ent_coef = ppo_kwargs['ent_coef']
    model.n_epochs = ppo_kwargs['n_epochs']
    if 'target_kl' in ppo_kwargs:
        model.target_kl = ppo_kwargs['target_kl']
    if model.n_steps != ppo_kwargs['n_steps']:
        model.n_steps = ppo_kwargs['n_steps']
        model.rollout_buffer = model.rollout_buffer_class(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=active_env.num_envs,
            **model.rollout_buffer_kwargs,
        )


def reset_value_head(model):
    """Re-initialize the value network while keeping the policy intact.

    Use when the value function has collapsed (value_loss → 0, explained_variance
    deeply negative) — the critic is predicting near-constant values and can't
    recover through normal gradient updates.
    """
    policy = model.policy
    value_params = []
    # Value MLP branch (mlp_extractor.value_net) — hidden layers
    for module in policy.mlp_extractor.value_net:
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        value_params.extend(module.parameters())
    # Final value output layer (value_net) — single linear layer
    policy.value_net.reset_parameters()
    value_params.extend(policy.value_net.parameters())

    optimizer = getattr(policy, 'optimizer', None)
    if optimizer is not None:
        seen = set()
        for param in value_params:
            if id(param) in seen:
                continue
            seen.add(id(param))
            optimizer.state.pop(param, None)
            if param.grad is not None:
                param.grad = None

    print("  [RESET] Value head re-initialized and optimizer state cleared")


if __name__ == '__main__':
    fail_fast_for_unsupported_windows_gpu()
    ensure_runtime_dirs()
    milestone_idx = load_progress()

    print(f"Pokemon Yellow RL - Auto Progression")
    print(f"Starting from milestone {milestone_idx}/{len(MILESTONES)}")
    print()

    # Single unified model that carries through the entire curriculum
    ppo_defaults = dict(
        ent_coef=0.02, n_steps=4096,
        n_epochs=3,
        learning_rate=1e-4,
        clip_range=0.2,
        target_kl=0.03,
    )
    unified_model_path = os.path.join(MILESTONES_DIR, 'pokemon_unified_model.zip')
    unified_vecnormalize_path = os.path.join(MILESTONES_DIR, 'pokemon_unified_vecnormalize.pkl')
    model = None

    while milestone_idx < len(MILESTONES):
        ms = MILESTONES[milestone_idx]
        n_envs = max(1, int(ms.get('n_envs', DEFAULT_N_ENVS)))
        state_file = get_state_file(ms)
        ppo_kwargs = {**ppo_defaults, **ms.get('ppo', {})}
        checkpoint_path = latest_stage_checkpoint(ms['name'])
        force_unified_resume = ms['name'] in FORCE_UNIFIED_RESUME_STAGES
        stage_checkpoint_stale = should_ignore_stage_checkpoint(state_file, checkpoint_path)
        if force_unified_resume and checkpoint_path is not None:
            print(
                f"  WARNING: Ignoring the latest {ms['name']} stage checkpoint "
                "to escape a logged local optimum."
            )
            checkpoint_path = None
        if stage_checkpoint_stale:
            print(f"  WARNING: {ms['name']}.state is newer than its latest stage checkpoint.")
            print(f"  Ignoring stale stage checkpoint and reusing the unified model instead.")
            checkpoint_path = None
        vecnormalize_path = None
        if checkpoint_path is not None and not force_unified_resume:
            checkpoint_stats = checkpoint_vecnormalize_path(checkpoint_path)
            if os.path.exists(checkpoint_stats):
                vecnormalize_path = checkpoint_stats
            else:
                print(
                    f"  WARNING: No matching VecNormalize state for {checkpoint_path}; "
                    "starting fresh reward normalization."
                )
        if checkpoint_path is None and not stage_checkpoint_stale and not force_unified_resume:
            vecnormalize_path = latest_stage_vecnormalize(ms['name'])
        if checkpoint_path is None and vecnormalize_path is None and os.path.exists(unified_vecnormalize_path):
            vecnormalize_path = unified_vecnormalize_path

        env_config = build_env_config(state_file, ms)
        vec_env_kind = resolve_vec_env_kind(n_envs, ms.get('vec_env_kind'))

        print(f"{'='*60}")
        print(f"  Stage {milestone_idx}: {ms['name']}")
        print(f"  Goal: {ms['desc']}")
        print(f"  Episode length: {ms['ep_length']} steps")
        print(f"  Loading state: {state_file}")
        print(f"  PPO: device={TRAIN_DEVICE} n_steps={ppo_kwargs['n_steps']} "
              f"n_epochs={ppo_kwargs['n_epochs']} ent_coef={ppo_kwargs['ent_coef']} "
              f"n_envs={n_envs} "
              f"target_kl={ppo_kwargs.get('target_kl')}")
        print(f"  Clone workers: backend={vec_env_kind} start_method={DEFAULT_SUBPROC_START_METHOD if vec_env_kind == 'subproc' else 'local'}")
        if ms.get('swarm_enabled', False):
            print(f"  Swarm: enabled check_interval={ms.get('swarm_check_interval', 64)} "
                  f"catchup_behind_bits={ms.get('swarm_catchup_behind_bits', 3)}")
        if ms.get('route4_progress_reward_scale', 0.0):
            print(f"  Route4 east-progress nudge: scale={ms.get('route4_progress_reward_scale', 0.0)}")
        print(f"  Reward tuning: step_penalty={ms.get('step_penalty', 0.0)} "
              f"tile_bonus={ms.get('tile_exploration_bonus', 5)} "
              f"map_discovery_bonus={ms.get('map_discovery_bonus', 100)} "
              f"level_reward_scale={ms.get('level_reward_scale', 50.0)} "
              f"event_flag_reward_scale={ms.get('event_flag_reward_scale', 0.0)} "
              f"explore_coord_reward_scale={ms.get('explore_coord_reward_scale', 0.0)} "
              f"heal_reward_scale={ms.get('heal_reward_scale', 0.0)} "
              f"revisit_stuck=({ms.get('revisit_stuck_threshold', 0)}, {ms.get('revisit_stuck_penalty', 0.0)}) "
              f"frontier_scale={ms.get('frontier_reward_scale', 0)} "
              f"frontier_soft_stall=({ms.get('frontier_soft_stall_score')}, "
              f"{ms.get('frontier_soft_stall_steps', 0)}, "
              f"{ms.get('frontier_soft_stall_penalty', 0.0)}) "
              f"frontier_stall=({ms.get('frontier_stall_steps', 0)}, {ms.get('frontier_stall_penalty', 0.0)}) "
              f"frontier_hit_score={ms.get('frontier_hit_score')} "
              f"actions={ms.get('actions', ['up', 'down', 'left', 'right', 'a', 'b'])} "
              f"noop_ticks={ms.get('noop_ticks', 24)} "
              f"auto_wait_zones={ms.get('auto_wait_zones')} "
              f"action_ticks=dir({ms.get('direction_press_ticks', 8)}/{ms.get('direction_release_ticks', 16)}) "
              f"btn({ms.get('button_press_ticks', 8)}/{ms.get('button_release_ticks', 16)}) "
              f"battle_ticks=dir({ms.get('battle_direction_press_ticks', ms.get('direction_press_ticks', 8))}/"
              f"{ms.get('battle_direction_release_ticks', ms.get('direction_release_ticks', 16))}) "
              f"btn({ms.get('battle_button_press_ticks', ms.get('button_press_ticks', 8))}/"
              f"{ms.get('battle_button_release_ticks', ms.get('button_release_ticks', 16))}) "
              f"noop({ms.get('battle_noop_ticks', ms.get('noop_ticks', 24))}) "
              f"post_transition_settle_ticks={ms.get('post_transition_settle_ticks', 0)} "
              f"action_guidance={len(ms.get('action_guidance', []) or [])} "
              f"required_route_maps={ms.get('required_route_maps')} "
              f"allowed_maps={ms.get('allowed_maps')} "
              f"disallowed_map_penalty={ms.get('disallowed_map_penalty', 0.0)} "
              f"coord_stuck=({ms.get('coord_stuck_threshold', 0)}, {ms.get('coord_stuck_penalty', 0.0)}) "
              f"action_oscillation_penalty={ms.get('action_oscillation_penalty', 0.0)} "
              f"overworld_non_movement_penalty={ms.get('overworld_non_movement_penalty', 0.0)} "
              f"party_size_bonus_scale={ms.get('party_size_bonus_scale', 2000.0)} "
              f"hit_bonus={ms.get('hit_bonus', 0)} "
              f"safe_hit_save_maps={ms.get('safe_hit_save_maps')} "
              f"safe_hit_min_total_hp={ms.get('safe_hit_min_total_hp', 0)} "
              f"safe_hit_save_bonus={ms.get('safe_hit_save_bonus', 0.0)} "
              f"low_hp_retreat_threshold={ms.get('low_hp_retreat_threshold', 0)} "
              f"post_hit_shaping={ms.get('post_hit_shaping')} "
              f"zone_bonus_total={sum(zone.get('bonus', 0) for zones in (ms.get('zone_bonuses') or {}).values() for zone in zones)} "
              f"terminate_on_hit={ms.get('terminate_on_hit', False)} "
              f"terminate_after_hit_state_saved={ms.get('terminate_after_hit_state_saved', False)}")
        print(f"  State load settle: ticks={ms.get('load_settle_ticks', 0)}")
        print(f"  Survival tuning: repel={ms.get('force_repel', False)} "
              f"force_a_wild={ms.get('force_a_in_wild_battles', False)} "
              f"force_a_trainer={ms.get('force_a_in_trainer_battles', False)} "
              f"ensure_battle_move_ready={ms.get('ensure_battle_move_ready', False)} "
              f"trainer_script={ms.get('trainer_battle_action_script')} "
              f"wild_step_penalty={ms.get('wild_battle_step_penalty', 0.0)} "
              f"wild_entry_penalty={ms.get('wild_battle_entry_penalty', 0.0)} "
              f"trainer_entry_bonus={ms.get('trainer_battle_entry_bonus', 0.0)} "
              f"hp_loss_scale={ms.get('hp_loss_penalty_scale', 0.0)} "
              f"wipe_penalty={ms.get('wipe_penalty', 0.0)} "
              f"wipe_progress=({ms.get('wipe_progress_max_y_scale', 0.0)}, "
              f"{ms.get('wipe_progress_tiles_scale', 0.0)}, "
              f"{ms.get('wipe_progress_frontier_scale', 0.0)}) "
              f"battle_step_limit=({ms.get('battle_step_limit', 0)}, "
              f"{ms.get('battle_step_limit_penalty', 0.0)}) "
              f"stale_battle_clear={ms.get('stale_battle_flag_clear_steps', 0)} "
              f"progressless_step_limit=({ms.get('progressless_step_limit', 0)}, "
              f"{ms.get('progressless_step_penalty', 0.0)}) "
              f"checkpoint_progress_step_limit=({ms.get('checkpoint_progress_step_limit', 0)}, "
              f"{ms.get('checkpoint_progress_step_penalty', 0.0)}) "
              f"full_hp_map_step_penalties={ms.get('full_hp_map_step_penalties')} "
              f"invalid_party=({ms.get('invalid_party_state_penalty', 0.0)}, "
              f"{ms.get('terminate_on_invalid_party_state', False)}) "
              f"terminate_on_wipe={ms.get('terminate_on_party_wipe', False)}")
        print(f"  Console logs: {'raw SB3 tables' if SHOW_RAW_SB3_TABLES else 'guided metrics with explanations'}")
        print(f"  First PPO log after {ppo_kwargs['n_steps']} vec steps "
              f"({ppo_kwargs['n_steps'] * n_envs:,} env steps across {n_envs} envs)")
        print(f"{'='*60}\n")

        env, vec_env_kind = build_vec_env(PokemonYellowEnv, env_config, n_envs, vec_env_kind)
        if vecnormalize_path is not None:
            print(f"  Loading VecNormalize stats from {vecnormalize_path}")
            env = VecNormalize.load(vecnormalize_path, env)
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        env = VecFrameStack(env, n_stack=4)

        if model is None:
            # Load unified model or per-milestone fallback, or start fresh
            if checkpoint_path is not None:
                print(f"  Resuming milestone checkpoint from {checkpoint_path}")
                model = PPO.load(checkpoint_path, env=env, device=TRAIN_DEVICE,
                                 custom_objects=ppo_kwargs)
            elif os.path.exists(unified_model_path):
                print(f"  Resuming unified model from {unified_model_path}")
                model = PPO.load(unified_model_path, env=env, device=TRAIN_DEVICE,
                                 custom_objects=ppo_kwargs)
            else:
                # Check for legacy per-milestone model
                legacy_path = os.path.join(MILESTONES_DIR, f'{ms["name"]}_model.zip')
                if os.path.exists(legacy_path):
                    print(f"  Resuming from legacy model {legacy_path}")
                    model = PPO.load(legacy_path, env=env, device=TRAIN_DEVICE,
                                     custom_objects=ppo_kwargs)
                else:
                    model = PPO('CnnPolicy', env, verbose=1, device=TRAIN_DEVICE,
                                **ppo_kwargs)
            apply_ppo_overrides(model, env, ppo_kwargs)
        else:
            # Carry forward — update PPO hyperparams for this milestone
            model.set_env(env)
            apply_ppo_overrides(model, env, ppo_kwargs)
            # Reset collapsed value head — reward landscape changes between milestones
            reset_value_head(model)

        configure_model_logger(model)

        checkpoint_cb = CheckpointCallback(
            save_freq=50000,
            save_path=str(CHECKPOINTS_DIR),
            name_prefix=f'poke_{ms["name"]}',
            save_vecnormalize=True,
        )
        screenshot_cb = ScreenshotCallback(
            save_path=str(SCREENSHOTS_DIR),
            save_freq=5000
        )
        progression_cb = ProgressionCallback(
            milestone=ms,
            milestones_dir=MILESTONES_DIR,
            hit_threshold=0.7,
            window_size=50,
        )
        rollout_report_interval = max(128, ppo_kwargs['n_steps'] // 2)
        rollout_progress_cb = RolloutProgressCallback(report_every_vec_steps=rollout_report_interval)
        simple_status_cb = SimpleStatusCallback(stage_name=ms['name'])
        checkpoint_prune_cb = CheckpointPruneCallback(
            active_stage_name=ms['name'],
            prune_every_calls=200_000,
        )

        model.learn(
            total_timesteps=10_000_000,
            callback=[
                checkpoint_cb, screenshot_cb, progression_cb,
                rollout_progress_cb, simple_status_cb, checkpoint_prune_cb,
            ],
            reset_num_timesteps=False,
        )

        PokemonYellowEnv.flush_pending_episode_summaries()

        # Always save the unified model
        vec_normalize_env = model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(unified_vecnormalize_path)
            print(f"  Unified VecNormalize saved: {unified_vecnormalize_path}")
        model.save(unified_model_path)
        print(f"  Unified model saved: {unified_model_path}")

        env.close()

        if progression_cb.advanced:
            milestone_idx += 1
            save_progress(milestone_idx)
            # The stage that just finished is no longer "active" -- collapse
            # its checkpoint history down to one recovery snapshot right away
            # instead of waiting for the next stage's periodic prune to get to it.
            removed = prune_checkpoint_history(active_stage_name=None)
            if removed:
                print(f"  Pruned {removed} checkpoint file(s) from the completed stage.")
            print(f"\n  Progressing to milestone {milestone_idx}...\n")
        else:
            print(f"\n  Training ended at 10M steps without mastering milestone.")
            print(f"  Continuing same milestone from the latest checkpoint.")
            save_progress(milestone_idx)
            continue

    if milestone_idx >= len(MILESTONES):
        print("\n" + "="*60)
        print("  CURRICULUM COMPLETE — Starting full-game training loop")
        print("="*60 + "\n")

        # Phase 2: Full-game loop training
        # Agent plays from start.state with max episode length.
        # Step penalty ramps up each round, pushing the agent to complete faster.
        # Speedrun record is ~1:56 = ~18,000 agent steps at 24 ticks/action.
        fullgame_round = 0
        max_steps = 200_000  # start generous
        step_penalty = 0.1   # initial penalty — gentle pressure

        while True:
            fullgame_round += 1

            fullgame_milestone = {
                'name': 'fullgame',
                'check': lambda mem, mid: mid == 118,
                'next_name': 'hall_of_fame',
                'ep_length': max_steps,
                'desc': 'Complete the game start to finish',
                'step_penalty': step_penalty,
            }
            fullgame_env_config = build_env_config(START_STATE_PATH, fullgame_milestone)
            fullgame_vec_env_kind = resolve_vec_env_kind(DEFAULT_N_ENVS)

            print(f"\n{'='*60}")
            print(f"  Full-game training round {fullgame_round}")
            print(f"  max_steps={max_steps:,}  step_penalty={step_penalty:.2f}")
            print(f"  Clone workers: backend={fullgame_vec_env_kind} start_method={DEFAULT_SUBPROC_START_METHOD if fullgame_vec_env_kind == 'subproc' else 'local'} n_envs={DEFAULT_N_ENVS}")
            print(f"{'='*60}\n")

            fullgame_n_envs = DEFAULT_N_ENVS
            env, fullgame_vec_env_kind = build_vec_env(
                PokemonYellowEnv,
                fullgame_env_config,
                fullgame_n_envs,
                fullgame_vec_env_kind,
            )
            if os.path.exists(unified_vecnormalize_path):
                print(f"  Loading VecNormalize stats from {unified_vecnormalize_path}")
                env = VecNormalize.load(unified_vecnormalize_path, env)
                env.training = True
                env.norm_reward = True
            else:
                env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            env = VecFrameStack(env, n_stack=4)

            if fullgame_round == 1:
                model.set_env(env)
            else:
                model.set_env(env)

            configure_model_logger(model)

            checkpoint_cb = CheckpointCallback(
                save_freq=500000,
                save_path=str(CHECKPOINTS_DIR),
                name_prefix='poke_fullgame',
                save_vecnormalize=True,
            )
            screenshot_cb = ScreenshotCallback(
                save_path=str(SCREENSHOTS_DIR),
                save_freq=10000
            )
            progression_cb = ProgressionCallback(
                milestone=fullgame_milestone,
                milestones_dir=MILESTONES_DIR,
                hit_threshold=0.5,  # 50% completion rate before tightening
                window_size=20,
            )
            rollout_progress_cb = RolloutProgressCallback(report_every_vec_steps=512)
            simple_status_cb = SimpleStatusCallback(stage_name='hall_of_fame')
            checkpoint_prune_cb = CheckpointPruneCallback(
                active_stage_name='fullgame',
                prune_every_calls=1_000_000,
            )

            model.learn(
                total_timesteps=50_000_000,
                callback=[
                    checkpoint_cb, screenshot_cb, progression_cb,
                    rollout_progress_cb, simple_status_cb, checkpoint_prune_cb,
                ],
                reset_num_timesteps=False,
            )

            model.save(unified_model_path)
            print(f"  Unified model saved after full-game round {fullgame_round}")
            vec_normalize_env = model.get_vec_normalize_env()
            if vec_normalize_env is not None:
                vec_normalize_env.save(unified_vecnormalize_path)
                print(f"  Unified VecNormalize saved after full-game round {fullgame_round}")

            env.close()

            # Tighten the screws each round the agent hits 50% completion
            if progression_cb.advanced:
                max_steps = max(25_000, int(max_steps * 0.8))  # shrink toward speedrun range
                step_penalty = min(2.0, step_penalty * 1.5)    # increase time pressure
                print(f"  Tightening: max_steps → {max_steps:,}, step_penalty → {step_penalty:.2f}")
