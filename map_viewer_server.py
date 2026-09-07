#!/usr/bin/env python3
import collections
import csv
import json
import math
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

ROOT = os.path.dirname(os.path.abspath(__file__))
MAX_ENVS = int(os.environ.get("POKEMON_VIEWER_MAX_ENVS", "96"))
FEED_HZ = max(1.0, float(os.environ.get("POKEMON_VIEWER_FEED_HZ", "10")))
HEATMAP_PATH = os.path.join(ROOT, "viewer_heatmap.json")
TRAIN_LOG_PATH = os.path.join(ROOT, "train.log")
SWARM_MASTERY_PATH = os.environ.get(
    "POKEMON_SWARM_MASTERY_PATH",
    os.path.join(ROOT, "restart_start.swarm_mastery.json"),
)
# A quest-zero restart keeps the learned weights, so it never prints
# FRESH_MODEL_LINE -- the only run-boundary signal the viewer had. It
# does rewrite this marker with a new `reset_id`, which is what scopes
# the timing ledger to the current attempt.
ATTEMPT_MARKER_PATH = os.environ.get(
    "POKEMON_ATTEMPT_MARKER_PATH",
    re.sub(
        r"\.swarm_mastery\.json$", ".quest_zero_mastery.json",
        SWARM_MASTERY_PATH,
    ),
)
PROGRESS_CSV_PATH = os.path.join(ROOT, "checkpoints", "sb3_logs", "progress.csv")
EVENT_FLAGS_CATALOG_PATH = os.path.join(ROOT, "navigation_data", "yellow_event_flags.json")
QUEST_WAYPOINTS_PATH = os.path.join(ROOT, "navigation_data", "yellow_quest_waypoints.json")
HEATMAP_SAVE_SECONDS = 30.0
SERVICE_POLL_SECONDS = 5.0
HISTORY_POLL_SECONDS = 10.0
HISTORY_MAX_ROWS = 500
RECENT_WINDOW_SECONDS = 300.0
# --- per-objective elapsed time ------------------------------------
# train.log carries no timestamps, so the only clock for a [QUEST] line
# is when the viewer observed it. That is accurate to the tail poll
# while the tail is keeping up, and worthless for a catch-up batch: a
# viewer restart replays everything written while it was down, and a
# fresh-model rescan replays the whole (multi-GB) file, both in a single
# read that would stamp thousands of historical clears with "now".
# A batch counts as live only if the previous tail was moments ago AND
# the chunk is small enough to be one poll's worth of output.
LIVE_TAIL_MAX_GAP_SECONDS = 5.0
LIVE_TAIL_MAX_BYTES = 2_000_000
# Discard an implausible segment rather than let one bad clock reading
# own the "worst objective" row forever.
QUEST_SEGMENT_MAX_SECONDS = 3600.0
# Frontier dwell is minutes-scale; 1 Hz is ample and keeps the 10 Hz
# feed loop free of the frontier-file fallback read.
DWELL_POLL_SECONDS = 1.0
# Stop the dwell clock when the trainer stops writing, so a crash loop
# or a deploy restart is not billed to whatever objective was current.
DWELL_STALE_LOG_SECONDS = 120.0
ATTEMPT_POLL_SECONDS = 15.0
RESETTING_COUNTER_FIELDS = (
    "completed_episodes",
    "whiteout_count",
    "whiteout_episodes",
)
# 2026-08-12: the trailing capture used to be `(.+?)\s*$` -- non-greedy, but
# the `$` anchor made it run to the end of the physical line anyway. train.py's
# 96 workers share one stdout, so their writes interleave *without* a newline
# and two records routinely land on a single physical line:
#     [EVENT] env_id=1 step=768 EVENT_BEAT_X  [EVENT] env_id=4 step=768 EVENT_...
# The old pattern swallowed that tail as part of the event name, minting a
# brand-new unique "name" for every interleaving, so the swarm's distinct-event
# set grew without bound (live: 3,698 "names" against a 522-entry catalog -- a
# 709% coverage bar). Event names are `[A-Z0-9_]` tokens, so matching them
# explicitly cannot run past a record boundary. Verified over 400MB of live
# train.log: strict parsing gives 227 distinct names, all 227 in the catalog;
# the old pattern gave 470, of which 243 were parse debris.
EVENT_LINE_RE = re.compile(
    r"\[EVENT\]\s+env_id=(\d+)\s+step=(\d+)\s+([A-Z0-9_]+(?:\s*,\s*[A-Z0-9_]+)*)"
)
# Quest splits. `phase=` is the phase the agent moved *to*, so the objective
# just finished is phase-1 -- that is what the split is recorded against.
# `phase_steps=` (steps spent on that objective alone) is optional: lines
# written before it was added still parse, they just carry no split time.
# Interleaving is handled the same way as [EVENT]: match the fields explicitly
# and use finditer so a second record on one physical line is not dropped.
QUEST_LINE_RE = re.compile(
    r"\[QUEST\]\s+env_id=(\d+)\s+step=(\d+)\s+"
    r"(?:phase_steps=(\d+)\s+)?"
    r"phase=(\d+)/(\d+)\s+completed=([A-Za-z0-9_]+)"
)
# How many recent split times to retain per objective for the median.
QUEST_SPLIT_SAMPLES = 64
# Time-sink shortlist. Ranking every dwelt objective is not a shortlist --
# with 12 objectives cleared, a top-8 reproduced the table one row shorter.
# Take ranked objectives only until they explain most of the tracked time,
# and never admit one that is neither a meaningful share nor a meaningful
# duration. The point is "these few cost the run", not "here is a ranking".
QUEST_TIME_SINK_ROWS = 6
QUEST_TIME_SINK_COVERAGE = 75.0
QUEST_TIME_SINK_MIN_SHARE = 3.0
QUEST_TIME_SINK_MIN_SECONDS = 10.0
# Gate trend. A plateau counter sitting at "0 of 20" means one of three
# completely different things -- the swarm keeps finding faster routes (the
# counter resets), the swarm is converging (it climbs), or nothing is clearing
# at all (it never moves). The card could not tell them apart, so the trend is
# sampled from the trainer's own durable counters and reported explicitly.
GATE_TREND_SAMPLES = 96
GATE_TREND_POLL_SECONDS = 2.0
# "No clears for a while" has to scale with how fast the objective normally
# clears: 2 minutes is an eternity for a 3-second warp and nothing at all for
# a long grind. Floor, then a multiple of the observed clear interval.
GATE_STALL_MIN_SECONDS = 120.0
GATE_STALL_INTERVAL_FACTOR = 4.0
# How recently the best route must have improved to still call it improving.
GATE_IMPROVING_SECONDS = 600.0
# Same token shape, used to sanity-check names restored from the state file.
EVENT_NAME_RE = re.compile(r"[A-Z0-9_]+")
# train.py prints this exact line only when the fullgame loop creates a
# genuinely new PPO model (no checkpoint/unified-model resume found) --
# the one unambiguous signal that training has started completely over.
# Used to auto-reset all cumulative viewer state (heatmap, event progress,
# reset counters) so a fresh model's dashboard doesn't stay contaminated
# with a discarded run's history. See CxC.md 2026-07-24 for why this
# matters: a checkpoint got archived specifically because its whole
# history was under broken reward signal, and the viewer needed to stop
# reflecting it the moment the fresh model took over.
FRESH_MODEL_LINE = "Starting a fresh model (full restart, no checkpoint resume)"
METRICS_HEADER_RE = re.compile(
    r"^\s*\[METRICS\]\s+(?P<phase>\S+)\s*\|\s*steps=(?P<total_steps>[\d,]+)\s*\([^)]*\)"
    r"\s*\|\s*session_steps=(?P<session_steps>[\d,]+)\s*\([^)]*\)\s*\|\s*fps=(?P<fps>\d+)\s*$"
)
METRICS_ROW_RE = re.compile(r"^\s{2,}(?P<key>\S+)\s+(?P<value>\S+)\s*(?P<note>.*?)\s*$")
SERVICE_PROPERTY_RE = re.compile(r"^([A-Za-z]+)=(.*)$")
HISTORY_FIELDS = [
    "time/total_timesteps",
    "time/fps",
    "train/explained_variance",
    "train/value_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/entropy_loss",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
]


def load_map_offsets():
    path = os.path.join(ROOT, "pokerl_map_assets", "map_data.json")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            int(region["id"]): tuple(region["coordinates"])
            for region in data.get("regions", [])
            if "id" in region and len(region.get("coordinates", [])) == 2
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def load_event_catalog_total():
    try:
        with open(EVENT_FLAGS_CATALOG_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        # `num_events` (2,560) is the ROM's total event-flag bit-space
        # (NUM_EVENTS = $A00); most of those bits are unnamed. Only the
        # named entries in `events` (522) can ever produce a `[EVENT]` line
        # in train.log, so that -- not the raw bit count -- is the correct
        # denominator for "named events reached".
        events = data.get("events") or []
        return len(events) or int(data.get("num_events") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def load_event_catalog_names():
    """The catalog's named event flags, as a set.

    Used as the membership test for "named events reached" so the numerator
    is drawn from the same 522 names as the denominator.
    """
    try:
        with open(EVENT_FLAGS_CATALOG_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            entry["name"]
            for entry in (data.get("events") or [])
            if isinstance(entry, dict) and entry.get("name")
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return set()


TRAIN_PY_PATH = os.path.join(ROOT, "train.py")
SWARM_FRONTIER_META_PATH = os.environ.get(
    "POKEMON_SWARM_FRONTIER_META_PATH",
    os.path.join(ROOT, "restart_start.swarm_frontier.json"),
)
# Gate defaults mirror train.py's own fallbacks; the real values are parsed
# from the fullgame milestone below so this keeps reporting the truth when
# they are retuned (window_size went 16 -> 32 on 2026-08-13).
DRILL_GATE_DEFAULTS = {
    "window_size": 32,
    "min_samples": 8,
    "hit_rate": 0.8,
    "max_steps": 2048,
    "early_max_steps": 768,
    "no_improve_rotations": 20,
    "grind_successes": 5,
}
_GATE_CACHE = {"mtime": None, "config": dict(DRILL_GATE_DEFAULTS)}


def load_drill_gate_config():
    """Read the drill/mastery thresholds out of train.py's fullgame milestone.

    Parsed rather than hardcoded because these are actively retuned, and a
    progress counter measured against a stale threshold is worse than none.
    Cached on train.py's mtime so this is not re-read every request.
    """
    try:
        mtime = os.stat(TRAIN_PY_PATH).st_mtime_ns
    except OSError:
        return dict(_GATE_CACHE["config"])
    if _GATE_CACHE["mtime"] == mtime:
        return dict(_GATE_CACHE["config"])
    config = dict(DRILL_GATE_DEFAULTS)
    try:
        with open(TRAIN_PY_PATH, encoding="utf-8") as handle:
            src = handle.read()
        start = src.find("fullgame_milestone = {")
        end = src.find("fullgame_env_config = build_env_config", start)
        block = src[start:end] if start != -1 and end > start else ""
        for key, field in (
            ("swarm_drill_window_size", "window_size"),
            ("swarm_drill_min_samples", "min_samples"),
            ("swarm_drill_hit_rate", "hit_rate"),
            ("swarm_drill_max_steps", "max_steps"),
            ("swarm_drill_early_max_steps", "early_max_steps"),
            ("swarm_mastery_no_improve_rotations", "no_improve_rotations"),
            ("swarm_grind_mastery_successes", "grind_successes"),
        ):
            match = re.search(r"['\"]%s['\"]:\s*([0-9.]+)" % key, block)
            if not match:
                continue
            config[field] = (
                float(match.group(1)) if field == "hit_rate"
                else int(float(match.group(1)))
            )
    except (OSError, ValueError, TypeError):
        pass
    _GATE_CACHE["mtime"] = mtime
    _GATE_CACHE["config"] = config
    return dict(config)


MAP_OFFSETS = load_map_offsets()
EVENT_CATALOG_TOTAL = load_event_catalog_total()
EVENT_CATALOG_NAMES = load_event_catalog_names()
_QUEST_NAMES_CACHE = {"mtime": None, "names": []}


def load_quest_waypoint_names():
    """Official FULLGAME_QUEST_WAYPOINTS names, indexed by phase."""
    try:
        mtime = os.stat(QUEST_WAYPOINTS_PATH).st_mtime_ns
    except OSError:
        _QUEST_NAMES_CACHE["mtime"] = None
        _QUEST_NAMES_CACHE["names"] = []
        return []
    if _QUEST_NAMES_CACHE["mtime"] == mtime:
        return _QUEST_NAMES_CACHE["names"]
    try:
        with open(QUEST_WAYPOINTS_PATH, encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        names = [str(name) for name in (payload.get("names") or ())]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        names = []
    _QUEST_NAMES_CACHE["mtime"] = mtime
    _QUEST_NAMES_CACHE["names"] = names
    return names


def accumulate_resetting_counter(total, previous, reported):
    """Accumulate a worker counter that can reset when its process restarts."""
    reported = max(0, int(reported))
    total = max(0, int(total))
    if previous is None:
        delta = reported
    elif reported >= previous:
        delta = reported - previous
    else:
        # A lower value marks a new worker generation. Count from zero again.
        delta = reported
    return total + delta, reported


class AgentFeed:
    def __init__(self):
        self.condition = threading.Condition()
        self.entries = {}
        self.mtimes = {}
        self.payload = (
            b'{"agents":[],"heatmap_delta":[],"heatmap_version":0,'
            b'"training":{"metrics":null,"metrics_updated_at":null,'
            b'"log_updated_at":null,"service":null,"aggregate":null}}'
        )
        self.version = 0
        self.heatmap_version = 0
        self.heatmap_started_at = time.time()
        self.heatmap_by_env = collections.defaultdict(collections.Counter)
        self.heatmap_global = collections.Counter()

        # Last-visit timestamp per tile (never pruned, unlike recent_visits
        # below) -- lets the cumulative heatmap fade a tile's color back
        # toward "cold" once nobody has stepped on it in a while, instead of
        # a tile staying permanently red just because it was hot once, long
        # ago. Persisted the same way as the counts so a service restart
        # doesn't reset every tile back to "just visited".
        self.last_visit_global = {}
        self.last_visit_by_env = collections.defaultdict(dict)

        # Recency ("hot right now") heatmap: a time-ordered queue of visit
        # events plus a live running count, pruned each tick. Unlike the
        # cumulative counters above, this one can legitimately decrease as
        # old events fall out of the window, so it's served by polling
        # (mode=recent on /live_agent_heatmap.json) rather than SSE deltas.
        self.recent_visits = collections.deque()
        self.recent_counter = collections.Counter()

        # Per-episode heatmap + reset detection. Prefers the exact signal
        # (an env's `completed_episodes` field incrementing, once the
        # trainer has restarted with that field live) and falls back to a
        # fixed-start-state fingerprint heuristic otherwise: every fullgame
        # episode reloads the identical save state, so an env returning to
        # the exact (map, x, y) it first appeared at -- after having moved
        # away from it at least once -- is a reasonable stand-in for "this
        # env's episode just reset" until the exact signal is available.
        self.episode_heat_by_env = collections.defaultdict(collections.Counter)
        self.reset_seq_by_env = collections.defaultdict(int)
        self.reset_signal_by_env = {}
        self.last_completed_episodes_by_env = {}
        self.start_fingerprint_by_env = {}
        self.away_since_reset_by_env = collections.defaultdict(bool)

        # Trainer workers are recreated after each 50M-step internal round,
        # resetting the counters in their JSON snapshots. Preserve both the
        # accumulated total and the last raw value so dashboard totals remain
        # monotonic across worker and viewer restarts.
        self.cumulative_agent_counters = {
            field: collections.defaultdict(int)
            for field in RESETTING_COUNTER_FIELDS
        }
        self.last_agent_counters = {
            field: {}
            for field in RESETTING_COUNTER_FIELDS
        }

        self.event_log_inode = None
        self.event_log_offset = 0
        self.event_log_fragment = ""
        # phase -> {'name', 'clears', 'best', 'last', 'last_delta', 'samples',
        #           'best_run_step', 'attempt_clears', 'best_secs',
        #           'last_secs', 'last_delta_secs', 'secs_samples'}
        self.quest_splits = {}
        self.quest_total_phases = 0
        # Wall-clock the frontier has spent on each objective. This is
        # the number an audit wants: 96 workers run in parallel, so one
        # worker's attempt duration says nothing about what an objective
        # costs the run, while dwell is exactly that cost.
        self.phase_dwell = {}
        self.phase_visits = {}
        self.phase_first_reached = {}
        # Phases whose dwell is a split estimate rather than a measurement,
        # because the frontier crossed them between two dwell polls.
        self.phase_dwell_estimated = set()
        self._dwell_phase = None
        self._dwell_since = None
        # Survives the clock being stopped, so a trainer restart in the
        # middle of one objective is not counted as a second visit to it.
        self._dwell_last_phase = None
        self._next_dwell_poll = 0.0
        # Rolling samples of the drilled phase's promotion counters.
        self.gate_trend_phase = None
        self.gate_trend = []
        self.gate_trend_started = None
        self._next_gate_trend_poll = 0.0
        # Last parsed copy of the trainer's mastery counters. Cached because
        # the file is only re-read when its mtime moves, while the trend has
        # to keep evaluating "still nothing" on every tick.
        self._mastery_counters = {}
        self.attempt_reset_id = None
        self.attempt_started_ts = None
        self.attempt_marker_mtime = None
        self._next_attempt_poll = 0.0
        # Wall time each env last began an episode, and its last observed
        # objective clear -- together these bracket one attempt.
        self.episode_start_wall_by_env = {}
        self._quest_clear_mark_by_env = {}
        self._last_tail_wall = None
        self._tail_live = False
        self.mastery_splits_mtime = None
        self.events_by_env = collections.defaultdict(set)
        self.latest_event_by_env = {}
        self.latest_event_step_by_env = {}
        self._metrics_pending = None
        self.latest_metrics = None
        self.metrics_updated_at = None
        # Wall time we last read *any* new bytes out of train.log. This is the
        # honest "is the trainer alive" signal. `metrics_updated_at` is NOT --
        # a [METRICS] block is only emitted once per PPO rollout, which at
        # n_steps=1024 x 96 envs is several minutes apart, so a staleness
        # check built on it fires constantly while training is perfectly
        # healthy and still appending [EP] lines every second.
        self.log_updated_at = None

        self.service_status = None
        self._next_service_poll = 0.0

        self.history = collections.deque(maxlen=HISTORY_MAX_ROWS)
        self.history_inode = None
        self.history_offset = 0
        self.history_header = None
        self.history_fragment = ""
        self.history_signature = None
        self._next_history_poll = 0.0

        self.latest_aggregate = None

        # Persistent, never-decreasing record of the highest Pikachu level
        # any agent has ever reached and how many distinct times each level
        # was newly reached -- unlike the live per-agent `pikachu_level`
        # field, this survives every episode reset (every episode reloads
        # the same level-5 save state, so the live snapshot alone made it
        # look like leveling never happened even when it briefly had).
        self.max_level_ever_by_env = {}
        self.max_level_ever = 0
        self.level_reached_counts = collections.Counter()

        self._load_heatmap()
        self._tail_train_log()
        self._reconcile_mastery_splits()
        self._read_history()
        self._poll_service_status()

    def _load_heatmap(self):
        try:
            with open(HEATMAP_PATH, encoding="utf-8") as handle:
                data = json.load(handle)
            # 2026-07-26: `event_log_inode`/`event_log_offset` used to be
            # in-memory only. `event_log_inode` starts as `None` in
            # `__init__`, which never equals a real (dev, ino) pair, so
            # `_tail_train_log`'s very first call after *any* process
            # restart always took its "log rotated" branch and reset
            # `event_log_offset` to 0 -- re-scanning train.log (append-only
            # across this project's entire multi-day history) from byte
            # zero and re-triggering `_reset_for_fresh_model()` for every
            # historical "Starting a fresh model" line still in the file.
            # Confirmed live: a routine viewer restart (the service hadn't
            # been restarted since 2026-07-24) wiped 2 days of accumulated
            # heatmap coverage this way. Persisting both fields closes it --
            # `_tail_train_log`'s existing inode-mismatch check still
            # correctly forces a real reset on an actual log rotation.
            inode = data.get("event_log_inode")
            if isinstance(inode, list) and len(inode) == 2:
                self.event_log_inode = (int(inode[0]), int(inode[1]))
                self.event_log_offset = max(0, int(data.get("event_log_offset", 0)))
            self.heatmap_started_at = float(data.get("started_at", self.heatmap_started_at))
            for env_key, tiles in data.get("by_env", {}).items():
                env_id = int(env_key)
                if not 0 <= env_id < MAX_ENVS:
                    continue
                counter = self.heatmap_by_env[env_id]
                for entry in tiles:
                    # 4th element (last-visit timestamp) is optional so a
                    # heatmap file saved before decay support still loads --
                    # those tiles just start with no recency data (rendered
                    # as fully "hot", same as before this feature existed).
                    x, y, count = entry[0], entry[1], entry[2]
                    last_seen = float(entry[3]) if len(entry) > 3 else 0.0
                    key = (int(x), int(y))
                    counter[key] = int(count)
                    self.heatmap_global[key] += int(count)
                    if last_seen > 0:
                        self.last_visit_by_env[env_id][key] = last_seen
                        if last_seen > self.last_visit_global.get(key, 0):
                            self.last_visit_global[key] = last_seen
            for phase_key, seconds in (data.get("phase_dwell") or {}).items():
                try:
                    self.phase_dwell[int(phase_key)] = float(seconds)
                except (TypeError, ValueError):
                    continue
            for phase_key, count in (data.get("phase_visits") or {}).items():
                try:
                    self.phase_visits[int(phase_key)] = int(count)
                except (TypeError, ValueError):
                    continue
            for phase_key, stamp in (data.get("phase_first_reached") or {}).items():
                try:
                    self.phase_first_reached[int(phase_key)] = float(stamp)
                except (TypeError, ValueError):
                    continue
            for phase_key in (data.get("phase_dwell_estimated") or []):
                try:
                    self.phase_dwell_estimated.add(int(phase_key))
                except (TypeError, ValueError):
                    continue
            restored_attempt = data.get("attempt_reset_id")
            self.attempt_reset_id = (
                str(restored_attempt) if restored_attempt else None
            )
            try:
                self.attempt_started_ts = float(data.get("attempt_started_ts"))
            except (TypeError, ValueError):
                self.attempt_started_ts = None
            self.heatmap_version = int(data.get("version", 0))
            # Restored through the same token filter the live parser uses, so
            # a state file written by an older build (which could contain
            # interleaved-line debris) is cleaned on load rather than
            # carrying the inflated counts forward.
            self.quest_total_phases = int(data.get("quest_total_phases") or 0)
            for phase_key, entry in (data.get("quest_splits") or {}).items():
                if not isinstance(entry, dict):
                    continue
                samples = [
                    int(v) for v in (entry.get("samples") or [])
                    if isinstance(v, (int, float)) and v >= 0
                ]
                self.quest_splits[int(phase_key)] = {
                    'name': str(entry.get('name') or ''),
                    'clears': max(0, int(entry.get('clears') or 0)),
                    'best': entry.get('best'),
                    'last': entry.get('last'),
                    'last_delta': entry.get('last_delta'),
                    'samples': samples[-QUEST_SPLIT_SAMPLES:],
                    'best_run_step': entry.get('best_run_step'),
                }
            for env_key, names in (data.get("events_by_env") or {}).items():
                env_id = int(env_key)
                if not 0 <= env_id < MAX_ENVS:
                    continue
                self.events_by_env[env_id].update(
                    name for name in names
                    if isinstance(name, str) and EVENT_NAME_RE.fullmatch(name)
                )
            counter_data = data.get("agent_counters") or {}
            for field in RESETTING_COUNTER_FIELDS:
                totals = (counter_data.get("totals") or {}).get(field) or {}
                previous = (counter_data.get("last_reported") or {}).get(field) or {}
                for env_key, value in totals.items():
                    env_id = int(env_key)
                    if 0 <= env_id < MAX_ENVS:
                        self.cumulative_agent_counters[field][env_id] = max(0, int(value))
                for env_key, value in previous.items():
                    env_id = int(env_key)
                    if 0 <= env_id < MAX_ENVS:
                        self.last_agent_counters[field][env_id] = max(0, int(value))
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def _save_heatmap(self):
        with self.condition:
            data = {
                "started_at": self.heatmap_started_at,
                "version": self.heatmap_version,
                "event_log_inode": list(self.event_log_inode) if self.event_log_inode else None,
                "event_log_offset": self.event_log_offset,
                "by_env": {
                    str(env_id): [
                        [x, y, count, self.last_visit_by_env.get(env_id, {}).get((x, y), 0)]
                        for (x, y), count in sorted(counter.items())
                    ]
                    for env_id, counter in self.heatmap_by_env.items()
                },
                # Event progress is only ever rebuilt from train.log *after*
                # `event_log_offset`, and that offset resumes near EOF -- so
                # without persisting this, every restart dropped the swarm's
                # entire event history and "named events reached" restarted
                # from zero with no way to recover it. Same class of bug as
                # the 2026-07-26 offset fix above, one field over.
                "events_by_env": {
                    str(env_id): sorted(names)
                    for env_id, names in self.events_by_env.items()
                    if names
                },
                # Splits are the record of how fast each objective has ever
                # been cleared, so they must outlive a viewer restart -- the
                # log offset resumes near EOF and would never rebuild them.
                "quest_splits": {
                    str(phase): entry
                    for phase, entry in self.quest_splits.items()
                },
                "quest_total_phases": self.quest_total_phases,
                # Dwell is wall-clock, so it cannot be rebuilt from the log
                # after the fact (train.log has no timestamps) -- unlike the
                # step splits, a restart without this loses it permanently.
                "phase_dwell": {
                    str(phase): seconds
                    for phase, seconds in self._phase_dwell_snapshot().items()
                },
                "phase_visits": {
                    str(phase): count
                    for phase, count in self.phase_visits.items()
                },
                "phase_first_reached": {
                    str(phase): stamp
                    for phase, stamp in self.phase_first_reached.items()
                },
                "phase_dwell_estimated": sorted(self.phase_dwell_estimated),
                "attempt_reset_id": self.attempt_reset_id,
                "attempt_started_ts": self.attempt_started_ts,
                "agent_counters": {
                    "totals": {
                        field: {
                            str(env_id): value
                            for env_id, value in counters.items()
                        }
                        for field, counters in self.cumulative_agent_counters.items()
                    },
                    "last_reported": {
                        field: {
                            str(env_id): value
                            for env_id, value in counters.items()
                        }
                        for field, counters in self.last_agent_counters.items()
                    },
                },
            }
        temp_path = HEATMAP_PATH + f".tmp.{os.getpid()}"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, separators=(",", ":"))
            os.replace(temp_path, HEATMAP_PATH)
        except OSError:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def start(self):
        threading.Thread(target=self._run, name="agent-feed", daemon=True).start()

    def _finalize_pending_metrics(self):
        if self._metrics_pending is not None:
            self.latest_metrics = self._metrics_pending
            self.metrics_updated_at = time.time()
            self._metrics_pending = None
            return True
        return False

    def _reset_for_fresh_model(self):
        """Clear every cumulative/session stat so the dashboard reflects
        only the new model going forward, not whatever run was archived.
        """
        now = time.time()
        self.heatmap_version = 0
        self.heatmap_started_at = now
        self.heatmap_by_env = collections.defaultdict(collections.Counter)
        self.heatmap_global = collections.Counter()
        self.last_visit_global = {}
        self.last_visit_by_env = collections.defaultdict(dict)
        self.recent_visits = collections.deque()
        self.recent_counter = collections.Counter()
        self.episode_heat_by_env = collections.defaultdict(collections.Counter)
        self.reset_seq_by_env = collections.defaultdict(int)
        self.reset_signal_by_env = {}
        self.last_completed_episodes_by_env = {}
        self.start_fingerprint_by_env = {}
        self.away_since_reset_by_env = collections.defaultdict(bool)
        self.cumulative_agent_counters = {
            field: collections.defaultdict(int)
            for field in RESETTING_COUNTER_FIELDS
        }
        self.last_agent_counters = {
            field: {}
            for field in RESETTING_COUNTER_FIELDS
        }
        self.quest_splits = {}
        self.quest_total_phases = 0
        self._reset_attempt_timing()
        self.events_by_env = collections.defaultdict(set)
        self.latest_event_by_env = {}
        self.latest_event_step_by_env = {}
        self.max_level_ever_by_env = {}
        self.max_level_ever = 0
        self.level_reached_counts = collections.Counter()
        self._save_heatmap()
        # Force the next history poll to re-read the current CSV revision.
        # SB3 rewrites its header in place when training-only fields first
        # appear, so history ingestion is revision-based rather than
        # append-offset-based now.
        self.history.clear()
        self.history_signature = None

    def _tail_train_log(self):
        """Incrementally consume train.log for both named-event lines and
        the periodic [METRICS] block, in one pass over the same tail read.
        """
        try:
            stat_result = os.stat(TRAIN_LOG_PATH)
            inode = (stat_result.st_dev, stat_result.st_ino)
            if (
                self.event_log_inode != inode
                or stat_result.st_size < self.event_log_offset
            ):
                self.event_log_inode = inode
                self.event_log_offset = 0
                self.event_log_fragment = ""
                self.events_by_env.clear()
                self.latest_event_by_env.clear()
                self.latest_event_step_by_env.clear()
                self._metrics_pending = None
            with open(TRAIN_LOG_PATH, encoding="utf-8", errors="replace") as handle:
                handle.seek(self.event_log_offset)
                chunk = handle.read()
                self.event_log_offset = handle.tell()
        except OSError:
            return False

        # Stamped on every poll, not just the ones that found new bytes, so
        # this measures whether the *viewer* is keeping up rather than
        # whether the trainer happened to be quiet for a moment.
        now = time.time()
        previous_tail = self._last_tail_wall
        self._last_tail_wall = now
        self._tail_live = (
            previous_tail is not None
            and (now - previous_tail) <= LIVE_TAIL_MAX_GAP_SECONDS
            and len(chunk) <= LIVE_TAIL_MAX_BYTES
        )

        if not chunk:
            return False
        self.log_updated_at = now
        text = self.event_log_fragment + chunk
        if text.endswith("\n"):
            lines = text.splitlines()
            self.event_log_fragment = ""
        else:
            lines = text.splitlines()
            self.event_log_fragment = lines.pop() if lines else text

        changed = False
        for line in lines:
            if FRESH_MODEL_LINE in line:
                self._reset_for_fresh_model()
                changed = True
                continue
            header_match = METRICS_HEADER_RE.match(line)
            if header_match:
                changed = self._finalize_pending_metrics() or changed
                self._metrics_pending = {
                    "phase": header_match.group("phase"),
                    "total_steps": int(header_match.group("total_steps").replace(",", "")),
                    "session_steps": int(header_match.group("session_steps").replace(",", "")),
                    "fps": int(header_match.group("fps")),
                    "rows": {},
                }
                continue
            if self._metrics_pending is not None:
                row_match = METRICS_ROW_RE.match(line)
                if row_match:
                    key = row_match.group("key")
                    value = row_match.group("value")
                    value_num = None
                    try:
                        value_num = float(value)
                    except ValueError:
                        pass
                    self._metrics_pending["rows"][key] = {
                        "value": value,
                        "value_num": value_num,
                        "note": row_match.group("note") or "",
                    }
                    continue
                changed = self._finalize_pending_metrics() or changed

            for match in QUEST_LINE_RE.finditer(line):
                changed = self._record_quest_split(match) or changed

            # `finditer`, not `search`: an interleaved physical line can carry
            # more than one record, and only taking the first silently dropped
            # the rest (252 such lines in a 400MB sample).
            for match in EVENT_LINE_RE.finditer(line):
                env_id = int(match.group(1))
                if not 0 <= env_id < MAX_ENVS:
                    continue
                step = int(match.group(2))
                event_names = [
                    name.strip()
                    for name in match.group(3).split(",")
                    if name.strip()
                ]
                if not event_names:
                    continue
                known = self.events_by_env[env_id]
                before = len(known)
                known.update(event_names)
                self.latest_event_by_env[env_id] = ", ".join(event_names)
                self.latest_event_step_by_env[env_id] = step
                changed = changed or len(known) != before
        return changed

    def _record_quest_split(self, match):
        """Fold one [QUEST] completion into the per-objective split ledger."""
        run_step = int(match.group(2))
        phase_steps = int(match.group(3)) if match.group(3) is not None else None
        logged_phase = int(match.group(4))
        total_phases = int(match.group(5))
        name = match.group(6)
        # phase= is the phase moved *to*. The finished objective is usually
        # phase-1, but a skip-ahead (already-set event bits) logs the first
        # completed name with the final phase. File by the official waypoint
        # name when we have it so "receive_squirtle" cannot land on the next
        # slot after a voucher/bike insert.
        names = load_quest_waypoint_names()
        if name in names:
            completed_phase = names.index(name)
        else:
            completed_phase = logged_phase - 1
        if completed_phase < 0 or not 0 < total_phases <= 4096:
            return False
        self.quest_total_phases = max(self.quest_total_phases, total_phases)

        entry = self.quest_splits.get(completed_phase)
        if entry is None:
            entry = {
                'name': name, 'clears': 0, 'best': None, 'last': None,
                'last_delta': None, 'samples': [], 'best_run_step': None,
            }
            self.quest_splits[completed_phase] = entry
        entry['name'] = name
        entry['clears'] += 1
        self._record_quest_clear_timing(
            entry, int(match.group(1)), run_step,
        )
        # `run_step` is cumulative for the episode, so it is only a run time
        # for a worker that started this episode at phase 0. Tracked as the
        # best *observed* cumulative step, and read as such.
        if entry['best_run_step'] is None or run_step < entry['best_run_step']:
            entry['best_run_step'] = run_step
        if phase_steps is None:
            return True
        previous_best = entry.get('best')
        entry['last'] = phase_steps
        entry['last_delta'] = (
            None if previous_best is None else phase_steps - previous_best
        )
        if previous_best is None or phase_steps < previous_best:
            entry['best'] = phase_steps
        entry['samples'].append(phase_steps)
        if len(entry['samples']) > QUEST_SPLIT_SAMPLES:
            del entry['samples'][:-QUEST_SPLIT_SAMPLES]
        return True

    def _record_quest_clear_timing(self, entry, env_id, run_step):
        """Wall-clock one worker spent on the objective it just cleared.

        Two brackets, in order of precedence:
          * the worker's previous clear in this same episode -- that attempt
            ended exactly where this one began;
          * otherwise the episode start, because a first clear means the
            whole episode so far went into this objective (which is also
            what the trainer's own `step=` field means on that line).

        Every sample is gated on `_tail_live`. A dropped sample leaves a
        blank cell; a backlog sample would be a wrong number that looks
        real, and this ledger is persisted, so one would stay wrong.
        """
        entry['attempt_clears'] = int(entry.get('attempt_clears') or 0) + 1
        observed_at = self._last_tail_wall
        if not self._tail_live or observed_at is None:
            # Cannot trust this observation time, so it must not become the
            # start bracket for the next clear either.
            self._quest_clear_mark_by_env.pop(env_id, None)
            return

        episode_start = self.episode_start_wall_by_env.get(env_id)
        mark = self._quest_clear_mark_by_env.get(env_id)
        if (
            mark is not None
            and run_step > mark['run_step']
            and (episode_start is None or mark['wall'] >= episode_start)
        ):
            # `run_step` is cumulative per episode, so a larger one means no
            # reset happened in between. The episode-start check is the
            # belt-and-braces half: reset detection can be missed, and a
            # stale mark would then silently bracket across the reset.
            started_at = mark['wall']
        else:
            started_at = episode_start

        self._quest_clear_mark_by_env[env_id] = {
            'run_step': run_step, 'wall': observed_at,
        }
        if started_at is None:
            return
        secs = observed_at - started_at
        if not 0.0 < secs <= QUEST_SEGMENT_MAX_SECONDS:
            return
        secs = round(secs, 2)
        best = entry.get('best_secs')
        entry['last_secs'] = secs
        entry['last_delta_secs'] = (
            None if best is None else round(secs - best, 2)
        )
        if best is None or secs < best:
            entry['best_secs'] = secs
        samples = entry.setdefault('secs_samples', [])
        samples.append(secs)
        if len(samples) > QUEST_SPLIT_SAMPLES:
            del samples[:-QUEST_SPLIT_SAMPLES]

    def _update_phase_dwell(self, now):
        """Accumulate wall-clock the frontier has spent on each objective.

        This is the metric per-worker segment times cannot give. 96 workers
        run in parallel, so a worker taking 40s on an objective says nothing
        about what that objective costs the run; dwell is exactly that cost,
        and the dwell column sums to the tracked run time.

        The clock stops whenever train.log goes stale, so a crash loop or a
        deploy restart is not billed to whatever objective was current.
        """
        if now < self._next_dwell_poll:
            return False
        self._next_dwell_poll = now + DWELL_POLL_SECONDS
        fresh = (
            self.log_updated_at is not None
            and (now - self.log_updated_at) <= DWELL_STALE_LOG_SECONDS
        )
        if not fresh:
            # Bill up to the trainer's last sign of life, not up to the
            # moment the silence was noticed -- otherwise the whole
            # DWELL_STALE_LOG_SECONDS detection window lands on whichever
            # objective happened to be current when the trainer died.
            self._flush_phase_dwell(
                self.log_updated_at if self.log_updated_at is not None else now
            )
            return False
        phase = self._current_quest_phase()
        if phase is None:
            self._flush_phase_dwell(now)
            return False
        if self._dwell_phase == phase:
            return False
        self._flush_phase_dwell(now, through=phase)
        self._dwell_phase = phase
        self._dwell_since = now
        # A phase can be entered more than once -- a quarantined attempt or a
        # frontier rollback walks back through objectives already dwelt on.
        # Resuming the same phase after a trainer outage is not one of those,
        # so compare against the last phase occupied rather than the (cleared)
        # live one.
        if phase != self._dwell_last_phase:
            self.phase_visits[phase] = int(self.phase_visits.get(phase, 0)) + 1
        self._dwell_last_phase = phase
        self.phase_first_reached.setdefault(phase, now)
        return True

    def _flush_phase_dwell(self, now, through=None):
        """Bank the open dwell interval and stop the clock.

        `through` is the phase newly observed. Under load this feed loop runs
        at seconds per iteration, not the nominal 10 Hz, so the frontier can
        cross several short objectives between two polls. Banking the whole
        interval to the last phase actually sampled made one objective absorb
        its successors' time -- live, phase 4 read 55s while phases 5, 6 and 7
        read nothing at all, which is exactly backwards for an audit. Split
        the interval across the phases the frontier traversed and mark them
        estimated, so a wrong attribution cannot masquerade as a measurement.
        """
        if self._dwell_phase is not None and self._dwell_since is not None:
            elapsed = max(0.0, now - self._dwell_since)
            if elapsed > 0.0:
                start = self._dwell_phase
                spanned = [start]
                if through is not None and through > start + 1:
                    spanned = list(range(start, through))
                share = elapsed / len(spanned)
                for phase in spanned:
                    self.phase_dwell[phase] = round(
                        float(self.phase_dwell.get(phase, 0.0)) + share, 2
                    )
                    if len(spanned) > 1:
                        self.phase_dwell_estimated.add(phase)
                        self.phase_visits[phase] = max(
                            1, int(self.phase_visits.get(phase, 0))
                        )
                        self.phase_first_reached.setdefault(phase, self._dwell_since)
        self._dwell_phase = None
        self._dwell_since = None

    def _phase_dwell_seconds(self, phase, now=None):
        """Banked dwell for `phase`, plus the open interval if it is current."""
        total = float(self.phase_dwell.get(phase, 0.0))
        if phase == self._dwell_phase and self._dwell_since is not None:
            moment = now if now is not None else time.time()
            total += max(0.0, moment - self._dwell_since)
        return round(total, 2) if total > 0.0 else None

    def _phase_dwell_snapshot(self):
        """Banked dwell with the currently-open interval folded in.

        The clock is only banked on a phase transition, so a process that is
        killed mid-objective would lose everything since it entered that one.
        On a multi-hour grind -- the row an audit cares about most -- that is
        the worst possible thing to drop. Saving the open interval bounds the
        loss to one save period instead. Only the serialised copy is folded;
        `self.phase_dwell` keeps its banked-only meaning, so the running
        process cannot double-count.
        """
        snapshot = dict(self.phase_dwell)
        if self._dwell_phase is not None and self._dwell_since is not None:
            open_interval = max(0.0, time.time() - self._dwell_since)
            if open_interval > 0.0:
                snapshot[self._dwell_phase] = round(
                    float(snapshot.get(self._dwell_phase, 0.0)) + open_interval, 2
                )
        return snapshot

    def _poll_attempt_marker(self, now):
        """Scope the timing ledger to the current quest-zero attempt.

        A quest-zero restart keeps the learned weights, so it never prints
        FRESH_MODEL_LINE and the viewer's only run-boundary reset never
        fires. Blending a finished run into a new one is not hypothetical:
        the all-time `clears` column reads 340,561 on phase 6 for exactly
        that reason. Timing is per-attempt so the audit means something;
        the step columns keep their all-time behaviour.
        """
        if now < self._next_attempt_poll:
            return False
        self._next_attempt_poll = now + ATTEMPT_POLL_SECONDS
        try:
            mtime = os.stat(ATTEMPT_MARKER_PATH).st_mtime_ns
            if self.attempt_marker_mtime == mtime:
                return False
            with open(ATTEMPT_MARKER_PATH, encoding="utf-8") as handle:
                marker = json.load(handle) or {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        self.attempt_marker_mtime = mtime
        reset_id = marker.get('reset_id')
        reset_id = str(reset_id) if reset_id else None
        started_ts = marker.get('started_ts')
        try:
            started_ts = float(started_ts)
        except (TypeError, ValueError):
            started_ts = None
        if reset_id == self.attempt_reset_id:
            self.attempt_started_ts = started_ts or self.attempt_started_ts
            return False
        adopting = self.attempt_reset_id is None and not self.phase_dwell
        self.attempt_reset_id = reset_id
        self.attempt_started_ts = started_ts
        if adopting:
            # Nothing recorded yet -- learning the id is not a run boundary.
            return True
        self._reset_attempt_timing()
        return True

    def _reset_attempt_timing(self):
        """Drop every per-attempt timing figure, keeping all-time step data."""
        self._flush_phase_dwell(time.time())
        self.phase_dwell = {}
        self.phase_visits = {}
        self.phase_first_reached = {}
        self.phase_dwell_estimated = set()
        self._quest_clear_mark_by_env = {}
        self.episode_start_wall_by_env = {}
        self.gate_trend = []
        self.gate_trend_phase = None
        self.gate_trend_started = None
        for entry in self.quest_splits.values():
            entry.pop('attempt_clears', None)
            entry.pop('best_secs', None)
            entry.pop('last_secs', None)
            entry.pop('last_delta_secs', None)
            entry.pop('secs_samples', None)

    def _reconcile_mastery_splits(self):
        """Backfill split times from the trainer's durable mastery ledger.

        The viewer tails only new log bytes after a restart. A quest can
        therefore be durably mastered while its historical `[QUEST]` timing
        line is behind the saved log cursor. The trainer's drill windows and
        all-time bests are authoritative fallback timing data for those blank
        rows. Live log observations still own `last` when one exists and may
        contribute a faster all-worker best.
        """
        changed = self._remap_quest_splits_by_objective()
        try:
            mtime = os.stat(SWARM_MASTERY_PATH).st_mtime_ns
            if self.mastery_splits_mtime == mtime:
                return changed
            with open(SWARM_MASTERY_PATH, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            self.mastery_splits_mtime = None
            return False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

        windows = payload.get('windows') or {}
        best_steps = payload.get('best_steps') or {}
        self._mastery_counters = {
            'no_improve': payload.get('no_improve_counts') or {},
            'best': best_steps,
            'windows': windows,
        }
        for phase_key in set(windows) | set(best_steps):
            try:
                phase = int(phase_key)
            except (ValueError, TypeError):
                continue
            if not 0 <= phase < 4096:
                continue
            samples = []
            for value in windows.get(str(phase), ()):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    continue
                if value >= 0:
                    samples.append(value)
            try:
                mastery_best = int(best_steps[str(phase)])
            except (KeyError, ValueError, TypeError):
                mastery_best = min(samples) if samples else None
            if mastery_best is not None and mastery_best < 0:
                mastery_best = None
            if not samples and mastery_best is None:
                continue

            entry = self.quest_splits.get(phase)
            if entry is None:
                entry = {
                    'name': '', 'clears': 0, 'best': None, 'last': None,
                    'last_delta': None, 'samples': [], 'best_run_step': None,
                }
                self.quest_splits[phase] = entry
                changed = True
            mastery_clears = len(samples)
            if mastery_clears > entry['clears']:
                entry['clears'] = mastery_clears
                changed = True
            if mastery_best is not None and (
                entry['best'] is None or mastery_best < entry['best']
            ):
                entry['best'] = mastery_best
                changed = True
            if entry['last'] is None and samples:
                entry['last'] = samples[-1]
                entry['last_delta'] = (
                    entry['last'] - entry['best']
                    if entry.get('best') is not None else None
                )
                changed = True
            if not entry['samples'] and samples:
                entry['samples'] = samples[-QUEST_SPLIT_SAMPLES:]
                changed = True
        self.mastery_splits_mtime = mtime
        return changed

    def _remap_quest_splits_by_objective(self):
        """Move persisted phase-keyed rows to the current objective layout.

        The trainer curriculum can insert objectives. Split rows retain their
        stable objective name, while mastery samples correctly use the new
        numeric phase. Normalize the old rows first so current mastery data
        joins the same objective instead of a displaced neighbor.
        """
        names = load_quest_waypoint_names()
        current_phase = {name: phase for phase, name in enumerate(names)}
        if not current_phase:
            return False
        remapped = {}
        changed = False
        for old_phase, original in self.quest_splits.items():
            entry = dict(original)
            target = current_phase.get(str(entry.get('name')), int(old_phase))
            changed = changed or target != int(old_phase)
            prior = remapped.get(target)
            if prior is None:
                remapped[target] = entry
                continue
            # A row already written under the new phase can collide with its
            # displaced historical row. Merge evidence without double-counting
            # identical swarm bursts or replacing a faster segment.
            prior['clears'] = max(
                int(prior.get('clears') or 0), int(entry.get('clears') or 0)
            )
            bests = [
                value for value in (prior.get('best'), entry.get('best'))
                if value is not None
            ]
            prior['best'] = min(bests) if bests else None
            if entry.get('last') is not None:
                prior['last'] = entry['last']
                prior['last_delta'] = entry.get('last_delta')
            samples = list(prior.get('samples') or ()) + list(
                entry.get('samples') or ()
            )
            prior['samples'] = samples[-QUEST_SPLIT_SAMPLES:]
            run_steps = [
                value for value in (
                    prior.get('best_run_step'), entry.get('best_run_step')
                ) if value is not None
            ]
            prior['best_run_step'] = min(run_steps) if run_steps else None
            changed = True
        if changed:
            self.quest_splits = remapped
        return changed

    def _live_quest_summary(self):
        """Majority quest the live workers are on right now."""
        counts = collections.Counter()
        nexts = collections.Counter()
        maps = collections.Counter()
        newest = 0.0
        try:
            agents = list(self.entries.values())
        except RuntimeError:
            agents = []
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            phase = agent.get("quest_phase")
            if phase is None:
                continue
            try:
                counts[int(phase)] += 1
            except (TypeError, ValueError):
                continue
            nxt = agent.get("quest_next")
            if nxt:
                nexts[str(nxt)] += 1
            pos = agent.get("last_position") or []
            if len(pos) >= 3:
                try:
                    maps[int(pos[2])] += 1
                except (TypeError, ValueError):
                    pass
            seen = agent.get("last_seen")
            try:
                newest = max(newest, float(seen or 0))
            except (TypeError, ValueError):
                pass
        if not counts:
            return None
        phase, n = counts.most_common(1)[0]
        return {
            "phase": int(phase),
            "name": nexts.most_common(1)[0][0] if nexts else "",
            "n": int(sum(counts.values())),
            "majority": int(n),
            "maps": [
                {"id": int(mid), "n": int(count)}
                for mid, count in maps.most_common(4)
            ],
            "age_s": (
                max(0.0, time.time() - newest) if newest > 0 else None
            ),
        }

    def _drilled_phase(self):
        """The phase the gate card is describing.

        Same precedence the card itself uses -- live workers outrank a
        published frontier -- so the trend can never describe a different
        objective than the counter printed above it.
        """
        live = self._live_quest_summary()
        if live and (live.get("age_s") is None or live["age_s"] < 8):
            try:
                return int(live["phase"])
            except (KeyError, TypeError, ValueError):
                pass
        try:
            with open(SWARM_FRONTIER_META_PATH, encoding="utf-8") as handle:
                return int((json.load(handle) or {}).get("quest_phase"))
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _update_gate_trend(self, now):
        """Sample the drilled phase's promotion counters over time.

        Runs on a tick rather than only when the mastery file changes: the
        single most important verdict this produces is "nothing is clearing",
        and that is precisely the case where the file never changes. Tracking
        has to keep running while nothing happens for the silence to be
        measurable.
        """
        if now < self._next_gate_trend_poll:
            return False
        self._next_gate_trend_poll = now + GATE_TREND_POLL_SECONDS
        phase = self._drilled_phase()
        if phase is None:
            return False
        if phase != self.gate_trend_phase:
            self.gate_trend_phase = phase
            self.gate_trend = []
            self.gate_trend_started = now
        counters = self._mastery_counters or {}
        key = str(int(phase))
        try:
            no_improve = int((counters.get('no_improve') or {}).get(key))
        except (TypeError, ValueError):
            no_improve = None
        try:
            best = int((counters.get('best') or {}).get(key))
        except (TypeError, ValueError):
            best = None
        if no_improve is None and best is None:
            return False
        sample = {'t': now, 'no_improve': no_improve or 0, 'best': best}
        previous = self.gate_trend[-1] if self.gate_trend else None
        if previous is not None and (
            previous['no_improve'] == sample['no_improve']
            and previous['best'] == sample['best']
        ):
            # Unchanged counters are not an event. Silence is measured from
            # the last real one, so recording it would erase the signal.
            return False
        self.gate_trend.append(sample)
        if len(self.gate_trend) > GATE_TREND_SAMPLES:
            del self.gate_trend[:-GATE_TREND_SAMPLES]
        return True

    def gate_trend_payload(self, now=None):
        """Is this objective improving, converging, or hung?

        Derived from consecutive samples of the trainer's own counters:
          * `no_improve` going UP is a clear that did not beat the best;
          * `no_improve` resetting, or `best` dropping, is a faster route --
            which is *why* the plateau counter keeps returning to zero;
          * neither moving, for longer than this objective's own clear
            cadence, is the hang the card previously could not show.
        """
        now = time.time() if now is None else now
        if self.gate_trend_phase is None or self.gate_trend_started is None:
            return None
        samples = self.gate_trend
        clears = 0
        improvements = 0
        last_clear_t = None
        last_improvement_t = None
        last_improvement_delta = None
        intervals = []
        for previous, current in zip(samples, samples[1:]):
            improved = (
                previous['best'] is not None and current['best'] is not None
                and current['best'] < previous['best']
            )
            cleared = False
            if current['no_improve'] > previous['no_improve']:
                clears += current['no_improve'] - previous['no_improve']
                cleared = True
            elif current['no_improve'] < previous['no_improve'] or improved:
                clears += 1
                cleared = True
            if improved:
                improvements += 1
                last_improvement_t = current['t']
                last_improvement_delta = previous['best'] - current['best']
            if cleared:
                if last_clear_t is not None:
                    intervals.append(current['t'] - last_clear_t)
                last_clear_t = current['t']

        watched = max(0.0, now - self.gate_trend_started)
        idle_for = (now - last_clear_t) if last_clear_t is not None else watched
        typical = None
        if intervals:
            ordered = sorted(intervals)
            typical = ordered[len(ordered) // 2]
        stall_after = GATE_STALL_MIN_SECONDS
        if typical:
            stall_after = max(stall_after, GATE_STALL_INTERVAL_FACTOR * typical)

        improvement_age = (
            (now - last_improvement_t) if last_improvement_t is not None else None
        )
        if idle_for >= stall_after:
            verdict = 'stalled'
        elif improvement_age is not None and improvement_age <= GATE_IMPROVING_SECONDS:
            verdict = 'improving'
        elif clears > 0:
            verdict = 'converging'
        else:
            # Seen nothing yet, but not for long enough to call it a hang.
            verdict = 'watching'

        return {
            'phase': int(self.gate_trend_phase),
            'verdict': verdict,
            'watched_s': round(watched, 1),
            'clears': clears,
            'clears_per_min': (
                round(60.0 * clears / watched, 1) if watched > 0 and clears else None
            ),
            'idle_s': round(idle_for, 1),
            'stall_after_s': round(stall_after, 1),
            'typical_clear_s': round(typical, 1) if typical else None,
            'improvements': improvements,
            'improvement_age_s': (
                round(improvement_age, 1) if improvement_age is not None else None
            ),
            'improvement_delta': last_improvement_delta,
            'best': samples[-1]['best'] if samples else None,
            'first_best': next(
                (s['best'] for s in samples if s['best'] is not None), None
            ),
            # Evenly spaced for shape only: a sawtooth means faster routes
            # keep landing, a clean ramp means it is converging, a flat line
            # means nothing is happening. Exact timings are in the fields
            # above -- do not read spacing off this.
            'history': [s['no_improve'] for s in samples],
        }

    def _mastery_gate_payload(self):
        """Progress toward the promotion gate for the objective being drilled.

        Level grinds use successful-clear reliability because their duration
        is dominated by encounter and battle RNG. Deterministic objectives
        use the trainer's no-step-improvement plateau. Report the active mode
        directly so the viewer never describes an obsolete hit-rate rule.

        The card must describe what the swarm is *doing*, not a stale
        published index whose name came from a completed split. Live
        majority `quest_next` plus the trainer's durable
        `no_improve_counts` are the same signals train.py uses.
        """
        config = load_drill_gate_config()
        windows = {}
        no_improve_counts = {}
        mastery_ts = None
        try:
            with open(SWARM_MASTERY_PATH, encoding="utf-8") as handle:
                mastery = json.load(handle) or {}
            windows = mastery.get("windows") or {}
            raw_counts = mastery.get("no_improve_counts") or {}
            for key, value in raw_counts.items():
                try:
                    no_improve_counts[str(int(key))] = int(value)
                except (TypeError, ValueError):
                    continue
            if mastery.get("ts") is not None:
                mastery_ts = float(mastery["ts"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        frontier_phase = None
        frontier_objective = ""
        frontier_ts = None
        try:
            with open(SWARM_FRONTIER_META_PATH, encoding="utf-8") as handle:
                frontier_meta = json.load(handle) or {}
            frontier_phase = int(frontier_meta.get("quest_phase"))
            frontier_objective = str(frontier_meta.get("quest_objective") or "")
            if frontier_meta.get("ts") is not None:
                frontier_ts = float(frontier_meta["ts"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            frontier_phase = None

        # Early phases get the tighter budget. The boundary is the
        # reach_viridian .. enter_viridian_forest span, resolved by name from
        # the splits ledger so it tracks the quest chain rather than a
        # hardcoded index.
        name_to_phase = {
            entry.get('name'): phase
            for phase, entry in self.quest_splits.items() if entry.get('name')
        }
        early_lo = name_to_phase.get('reach_viridian', 0)
        early_hi = name_to_phase.get('enter_viridian_forest', 10)

        def budget_for(phase):
            return (config['early_max_steps'] if early_lo <= phase <= early_hi
                    else config['max_steps'])

        reliability_names = {
            'meet_bill', 'use_bills_cell_separator', 'receive_ss_ticket',
            'leave_bills_house',
        }

        def status_for(phase, name_override=""):
            samples = []
            for value in windows.get(str(int(phase)), ()):
                try:
                    samples.append(int(value))
                except (ValueError, TypeError):
                    continue
            budget = budget_for(int(phase))
            under = sum(1 for s in samples if 0 <= s <= budget)
            n = len(samples)
            name = (
                name_override
                or (self.quest_splits.get(int(phase)) or {}).get('name')
                or ''
            )
            is_grind = (
                str(name).startswith('train_')
                or str(name).startswith('heal_after_')
                or str(name) in reliability_names
            )
            if is_grind:
                mode = 'reliability'
                target = max(1, int(config['grind_successes']))
                progress = min(n, target)
                allowed = n >= target
            else:
                mode = 'plateau'
                target = max(1, int(config['no_improve_rotations']))
                durable = no_improve_counts.get(str(int(phase)))
                if durable is not None:
                    progress = min(max(0, durable), target)
                    allowed = durable >= target
                else:
                    best = None
                    streak = 0
                    for sample in samples:
                        if best is None or sample < best:
                            best = sample
                            streak = 0
                        else:
                            streak += 1
                    progress = min(streak, target)
                    allowed = n >= target + 1 and streak >= target
            needed_hits = math.ceil(config['hit_rate'] * n) if n else 0
            return {
                'phase': int(phase),
                'name': name,
                'mode': mode,
                'progress': progress,
                'target': target,
                'samples': n,
                'window_size': config['window_size'],
                'under_budget': under,
                'budget': budget,
                'hit_rate': (under / n) if n else None,
                'needed_hits': needed_hits,
                # How many more under-budget clears are required right now.
                'short_by': max(0, target - progress),
                'allowed': allowed,
            }

        per_phase = {}
        for key in windows:
            try:
                per_phase[int(key)] = status_for(int(key))
            except (ValueError, TypeError):
                continue
        published = None
        if frontier_phase is not None:
            published = status_for(frontier_phase, frontier_objective)
        live = self._live_quest_summary()
        displayed = published
        # Live workers are the truth for "what are we doing". A published
        # plateau on phase 61 while 96 agents fight the rocket is the bug
        # this card used to show.
        if live and (live.get("age_s") is None or live["age_s"] < 8):
            live_name = live.get("name") or ""
            live_status = status_for(live["phase"], live_name)
            if published is None or int(live["phase"]) != int(published["phase"]):
                live_status["allowed"] = False
            displayed = live_status
        now = time.time()
        return {
            'config': config,
            'frontier_phase': frontier_phase,
            'frontier': displayed,
            'published': published,
            'live': live,
            'frontier_age_s': (
                max(0.0, now - frontier_ts) if frontier_ts else None
            ),
            'mastery_age_s': (
                max(0.0, now - mastery_ts) if mastery_ts else None
            ),
            'trend': self.gate_trend_payload(now),
            'phases': [per_phase[k] for k in sorted(per_phase)],
        }

    # Yellow's own badge order, so the card reads like the in-game one.
    BADGE_NAMES = (
        'Boulder', 'Cascade', 'Thunder', 'Rainbow',
        'Soul', 'Marsh', 'Volcano', 'Earth',
    )
    # Only the first three gyms are forced into order: Cut needs Cascade and
    # the Vermilion gym sits behind a Cut tree, so bits 0-2 can only ever be
    # one of these prefixes. Gyms 4-8 are NOT ordered -- this run's chain
    # beats Sabrina (Marsh, bit 5) at phase 216 and Koga (Soul, bit 4) at 231,
    # a perfectly legal 47. Prefix-checking the whole byte called that garbage
    # and hid two earned badges (plus money and play time) behind the
    # quest-phase fallback for the entire post-Erika run.
    EARLY_GYM_MASK = 0b111
    EARLY_GYM_PREFIXES = frozenset({0, 1, 3, 7})

    @staticmethod
    def _badge_read_is_plausible(agent):
        """Drop mid-warp WRAM that majority-vote would otherwise treat as truth."""
        flags = agent.get('badge_flags')
        if not isinstance(flags, int):
            return False
        if not 0 <= flags <= 0xFF:
            return False
        early = flags & AgentFeed.EARLY_GYM_MASK
        if early not in AgentFeed.EARLY_GYM_PREFIXES:
            return False
        # A late badge without all three early ones is a mid-warp garbage read
        # (observed: 57 = bits 0,3,4,5), not Rainbow/Soul/Marsh earned before
        # Cascade/Thunder -- those gyms are unreachable without them.
        if flags >> 3 and early != AgentFeed.EARLY_GYM_MASK:
            return False
        pos = agent.get('last_position') or []
        map_id = agent.get('map_id')
        if len(pos) >= 3:
            try:
                local_x, _local_y, mid = int(pos[0]), int(pos[1]), int(pos[2])
            except (TypeError, ValueError):
                return True
            # Classic glitch: map, X, and badge byte are the same value.
            if flags == mid == local_x:
                return False
            if isinstance(map_id, int) and flags == int(map_id) and local_x == flags:
                return False
        return True

    # Waypoint that awards each badge, in badge-byte bit order. The two
    # non-gym Giovanni fights (Rocket Hideout, Silph Co) award nothing.
    GYM_WAYPOINT_BADGE_BITS = (
        ('beat_brock', 0),
        ('beat_misty', 1),
        ('beat_lt_surge', 2),
        ('beat_erika', 3),
        ('beat_koga', 4),
        ('beat_sabrina', 5),
        ('beat_blaine', 6),
        ('beat_viridian_gym_giovanni', 7),
    )

    def _inferred_gym_badge_mask(self, agents):
        """Badges implied by the current quest phase.

        `quest_phase` is the objective not yet completed, so a phase after
        `beat_lt_surge` means Thunder is already earned. This is a per-gym
        mask and not a prefix: the chain clears Sabrina before Koga, so a
        prefix stops dead at Thunder and under-reports by two badges from
        Celadon onwards.
        """
        names = load_quest_waypoint_names()
        phases = [
            int(agent['quest_phase'])
            for agent in agents
            if isinstance(agent.get('quest_phase'), int)
        ]
        if not phases:
            return 0
        current = max(phases)
        flags = 0
        for name, bit in self.GYM_WAYPOINT_BADGE_BITS:
            try:
                index = names.index(name)
            except ValueError:
                continue
            if current > index:
                flags |= 1 << bit
        return flags

    def _player_card_payload(self):
        """Trainer card aggregated across live workers.

        Every worker resets onto the shared frontier state, so name/money/badges
        are the same run for all of them; play time differs only by how far into
        its own episode each worker is. Take the furthest for a clock that ticks
        instead of sawtoothing 96 ways.

        Badge flags use majority vote among *plausible* gym-prefix bytes, not
        OR and not raw majority. One mid-warp worker reading 57 used to lose
        to 3; when 95 workers inherit that garbage map-57 state, raw majority
        lights badges 4/5/6 and hides Cascade/Thunder. Prefix-filter first.
        """
        agents = [a for a in self.entries.values() if isinstance(a, dict)]
        if not agents:
            return None

        plausible = [a for a in agents if self._badge_read_is_plausible(a)]
        named = [
            a for a in plausible
            if isinstance(a.get('player_name'), str) and str(a.get('player_name')).strip()
        ]
        badge_pool = named or plausible

        def best(key, default=0, pool=None):
            src = pool if pool is not None else agents
            values = [a.get(key) for a in src]
            values = [v for v in values if isinstance(v, (int, float))]
            return max(values) if values else default

        flag_votes = collections.Counter(
            agent['badge_flags'] for agent in badge_pool
            if isinstance(agent.get('badge_flags'), int)
        )
        badge_flags = flag_votes.most_common(1)[0][0] if flag_votes else 0
        badge_source = 'workers'
        if not flag_votes:
            # Every live reader is mid-warp trash (the 57-byte case).
            # Infer the earned gyms from the current quest instead of
            # showing 0 or lighting a set nobody actually read.
            badge_flags = self._inferred_gym_badge_mask(agents)
            badge_source = 'quest_chain'
        all_flag_votes = collections.Counter(
            agent['badge_flags'] for agent in agents
            if isinstance(agent.get('badge_flags'), int)
        )
        badge_flag_agreement = (
            all_flag_votes[badge_flags] / sum(all_flag_votes.values())
            if all_flag_votes and badge_flags in all_flag_votes
            else (
                flag_votes[badge_flags] / sum(flag_votes.values())
                if flag_votes else None
            )
        )
        names = [
            str(a.get('player_name')).strip()
            for a in (named or agents)
            if isinstance(a.get('player_name'), str) and str(a.get('player_name')).strip()
        ]
        time_pool = named or plausible
        play_time_seconds = int(best('play_time_seconds', pool=time_pool)) if time_pool else 0
        if not names and not badge_flags and not play_time_seconds:
            # Trainer predates the trainer-card fields; hide the card
            # rather than showing a wiped save of zeroes.
            return None
        return {
            'name': names[0] if names else None,
            'money': int(best('money', pool=time_pool)),
            'badge_flags': badge_flags,
            'badge_count': int(badge_flags).bit_count(),
            'badge_agreement': badge_flag_agreement,
            # 'quest_chain' means no worker produced a readable badge byte and
            # the set below is inferred, not observed -- the card used to
            # report that case as "all agreeing".
            'badge_source': badge_source,
            'badges': [
                {'name': name, 'earned': bool(badge_flags & (1 << index))}
                for index, name in enumerate(self.BADGE_NAMES)
            ],
            'play_time_seconds': play_time_seconds,
            'play_time_maxed': any(
                bool(a.get('play_time_maxed')) for a in time_pool
            ),
            'workers': len(agents),
        }

    def _current_quest_phase(self):
        """Live swarm objective, then the published frontier.

        `quest_phase` is the waypoint not yet completed. Historical split
        rows from earlier runs (Hall of Fame, etc.) must not outrank it.
        """
        summary = self._live_quest_summary()
        if summary is not None:
            return int(summary["phase"])
        try:
            with open(SWARM_FRONTIER_META_PATH, encoding="utf-8") as handle:
                frontier_meta = json.load(handle) or {}
            return int(frontier_meta.get("quest_phase"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def quest_progress_counts(live_phase, total_phases, ledger_count):
        """Cleared/remaining follow the live objective, not ledger width.

        The splits file keeps every [QUEST] row the viewer has ever seen,
        including later waypoints from discarded runs. `len(rows)` was
        250 with 47 remaining while the swarm was on phase 98.
        """
        if live_phase is not None:
            cleared = max(0, int(live_phase))
        else:
            cleared = max(0, int(ledger_count or 0))
        total = int(total_phases or 0)
        remaining = max(0, total - cleared) if total > 0 else None
        percent = (
            round(100.0 * cleared / total, 1) if total > 0 else None
        )
        return {
            "live_phase": None if live_phase is None else int(live_phase),
            "phases_cleared": cleared,
            "remaining": remaining,
            "percent_complete": percent,
            "ledger_rows": int(ledger_count or 0),
        }

    def _split_display_row(self, phase, entry, official_names, now=None):
        entry = entry or {}
        samples = sorted(entry.get('samples') or [])
        name = (
            official_names[phase]
            if 0 <= phase < len(official_names)
            else (entry.get('name') or '')
        )
        secs_samples = sorted(entry.get('secs_samples') or [])
        return {
            'phase': int(phase),
            'name': name,
            'clears': int(entry.get('clears') or 0),
            'best': entry.get('best'),
            'last': entry.get('last'),
            'last_delta': entry.get('last_delta'),
            'median': samples[len(samples) // 2] if samples else None,
            'best_run_step': entry.get('best_run_step'),
            # Wall-clock the frontier spent here -- the run's real cost for
            # this objective, and the column an audit should sort on.
            'dwell_secs': self._phase_dwell_seconds(phase, now),
            'dwell_estimated': phase in self.phase_dwell_estimated or None,
            'visits': int(self.phase_visits.get(phase, 0)) or None,
            'first_reached': self.phase_first_reached.get(phase),
            # Per-worker attempt durations. Useful for comparing objectives
            # against each other, NOT additive into a run time: 96 workers
            # attempt in parallel, so these overlap heavily.
            'best_secs': entry.get('best_secs'),
            'last_secs': entry.get('last_secs'),
            'last_delta_secs': entry.get('last_delta_secs'),
            'median_secs': (
                secs_samples[len(secs_samples) // 2] if secs_samples else None
            ),
            'timed_clears': len(secs_samples) or None,
            # Clears since this attempt began. The all-time `clears` above
            # spans every run the ledger has ever seen (340k on phase 6).
            'attempt_clears': int(entry.get('attempt_clears') or 0) or None,
        }

    def _splits_rows_for_display(self, live_phase):
        """Previous + current quests, with official names even if untimed.

        The persisted ledger keeps later waypoints from discarded runs.
        Those empty future rows used to fill the table with dashes. Keep a
        later row only when it actually has a segment time.
        """
        # One clock for the whole table, so the live row's ticking dwell
        # cannot disagree with the totals computed from the same rows.
        now = time.time()
        official_names = load_quest_waypoint_names()
        ledger = self.quest_splits
        # Curriculum insertions move phase numbers. Persisted split rows carry
        # objective names, so prefer identity over their old numeric slot.
        # Without this, the five-level insertion before Celadon displayed
        # level-46 timing as level-41 and Game Corner rows five phases early.
        ledger_by_name = {
            str(entry.get('name')): entry
            for entry in ledger.values()
            if entry.get('name')
        }
        if live_phase is None:
            phases = sorted(ledger)
        else:
            phases = list(range(0, int(live_phase) + 1))
        return [
            self._split_display_row(
                phase,
                (
                    ledger_by_name.get(official_names[phase])
                    if phase < len(official_names) else None
                ) or ledger.get(phase),
                official_names,
                now,
            )
            for phase in phases
        ]

    @staticmethod
    def quest_time_sinks(rows, total_secs):
        """The few objectives that actually explain the run's wall-clock.

        Stops as soon as the rows taken cover QUEST_TIME_SINK_COVERAGE of the
        tracked time, so a run dominated by one grind lists one objective
        rather than padding to a fixed length. The top row is always admitted
        -- it is by definition the most expensive thing measured so far, even
        early on when everything is cheap -- but every row after it must clear
        both a share and an absolute-duration floor.
        """
        total = float(total_secs or 0.0)
        if total <= 0:
            return []
        ranked = sorted(
            (row for row in rows if row.get('dwell_secs')),
            key=lambda row: row['dwell_secs'], reverse=True,
        )
        sinks = []
        covered = 0.0
        for row in ranked:
            share = 100.0 * row['dwell_secs'] / total
            if sinks and (
                share < QUEST_TIME_SINK_MIN_SHARE
                or row['dwell_secs'] < QUEST_TIME_SINK_MIN_SECONDS
            ):
                break
            sinks.append({
                'phase': row['phase'],
                'name': row['name'],
                'dwell_secs': row['dwell_secs'],
                'estimated': row.get('dwell_estimated') or None,
                'share': round(share, 1),
            })
            covered += share
            if len(sinks) >= QUEST_TIME_SINK_ROWS or covered >= QUEST_TIME_SINK_COVERAGE:
                break
        return sinks

    def quest_splits_payload(self):
        """Splits ledger plus the two totals worth watching.

        `sum_of_bests` is the theoretical run if every objective were cleared
        at its own best -- the target to chase. `timed_phases` says how much
        of the chain that sum actually covers, so a small sum from three
        cleared objectives can't be mistaken for a fast run.
        """
        live_phase = self._current_quest_phase()
        rows = self._splits_rows_for_display(live_phase)
        total_phases = self.quest_total_phases
        progress = self.quest_progress_counts(
            live_phase, total_phases, len(self.quest_splits),
        )
        # The denominator is cleared objectives, so exclude the active phase
        # from timing coverage even if an earlier run left a historical split
        # for it. Otherwise the card can claim 141 of 146 with six genuinely
        # untimed completed rows.
        timed = [
            row for row in rows
            if row['best'] is not None
            and row['phase'] < progress['phases_cleared']
        ]
        dwelt = [row['dwell_secs'] for row in rows if row['dwell_secs']]
        sinks = self.quest_time_sinks(rows, sum(dwelt))
        # Encoded here, like history_payload -- _send_json writes bytes.
        return json.dumps(
            {
                'total_phases': total_phases,
                'live_phase': progress['live_phase'],
                'phases_cleared': progress['phases_cleared'],
                'remaining': progress['remaining'],
                'percent_complete': progress['percent_complete'],
                'ledger_rows': progress['ledger_rows'],
                'player_card': self._player_card_payload(),
                'timed_phases': len(timed),
                'sum_of_bests': sum(r['best'] for r in timed) if timed else None,
                'sum_of_medians': (
                    sum(r['median'] for r in timed) if timed else None
                ),
                # Wall-clock accounting for the current attempt. `dwell_secs`
                # is disjoint per objective (the frontier is on exactly one
                # at a time), so unlike the per-worker segment times these
                # genuinely add up to elapsed training time.
                'attempt_reset_id': self.attempt_reset_id,
                'attempt_started_ts': self.attempt_started_ts,
                'tracked_run_secs': (
                    round(sum(dwelt), 2) if dwelt else None
                ),
                'current_dwell_secs': (
                    self._phase_dwell_seconds(progress['live_phase'])
                    if progress['live_phase'] is not None else None
                ),
                'dwelt_phases': len(dwelt),
                # A shortlist, not a re-ranked copy of the table.
                'time_sinks': sinks,
                'time_sink_share': (
                    round(sum(s['share'] for s in sinks), 1) if sinks else None
                ),
                'rows': rows,
                'mastery': self._mastery_gate_payload(),
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def _poll_service_status(self):
        try:
            result = subprocess.run(
                [
                    "systemctl", "show", "pokemon-train.service", "-p",
                    "ActiveState,SubState,MainPID,ActiveEnterTimestamp,NRestarts",
                ],
                capture_output=True, text=True, timeout=3, check=False,
            )
        except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
            self.service_status = {"error": "systemctl unavailable", "checked_at": time.time()}
            return
        fields = {}
        for line in result.stdout.splitlines():
            match = SERVICE_PROPERTY_RE.match(line)
            if match:
                fields[match.group(1)] = match.group(2)
        active_enter = fields.get("ActiveEnterTimestamp") or ""
        uptime_seconds = None
        if active_enter and active_enter not in ("", "n/a"):
            for fmt in ("%a %Y-%m-%d %H:%M:%S %Z", "%a %Y-%m-%d %H:%M:%S %z"):
                try:
                    parsed = datetime.strptime(active_enter, fmt)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    uptime_seconds = max(
                        0.0, (datetime.now(timezone.utc) - parsed).total_seconds()
                    )
                    break
                except ValueError:
                    continue
        self.service_status = {
            "active_state": fields.get("ActiveState"),
            "sub_state": fields.get("SubState"),
            "pid": fields.get("MainPID"),
            "since": active_enter or None,
            "uptime_seconds": uptime_seconds,
            "restarts": fields.get("NRestarts"),
            "checked_at": time.time(),
        }

    def _read_history(self):
        """Reload SB3's small CSV when its on-disk revision changes.

        SB3 expands the CSV header after the first PPO update by rewriting
        the file in place. An append-only offset reader lands in the middle
        of that rewritten header and rejects every subsequent wider row.
        Re-reading at most HISTORY_MAX_ROWS every ten seconds is cheap and
        correctly handles truncation, schema expansion, and process restarts.
        """
        try:
            stat_result = os.stat(PROGRESS_CSV_PATH)
        except OSError:
            return False
        inode = (stat_result.st_dev, stat_result.st_ino)
        signature = (inode, stat_result.st_size, stat_result.st_mtime_ns)
        if signature == self.history_signature:
            return False

        parsed = collections.deque(maxlen=HISTORY_MAX_ROWS)
        try:
            with open(
                PROGRESS_CSV_PATH,
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "time/total_timesteps" not in reader.fieldnames:
                    return False
                for row in reader:
                    try:
                        step = int(float(row.get("time/total_timesteps", "nan")))
                    except (TypeError, ValueError):
                        continue
                    point = {"step": step}
                    for field in HISTORY_FIELDS:
                        try:
                            point[field] = float(row[field])
                        except (KeyError, TypeError, ValueError):
                            point[field] = None
                    parsed.append(point)
        except (OSError, csv.Error):
            return False

        self.history = parsed
        self.history_signature = signature
        self.history_inode = inode
        self.history_offset = stat_result.st_size
        self.history_header = list(reader.fieldnames)
        self.history_fragment = ""
        return True

    def _decorate_progress(self, env_id, agent):
        agent["progress_count"] = len(self.events_by_env[env_id])
        agent["latest_event"] = self.latest_event_by_env.get(env_id)
        agent["latest_event_step"] = self.latest_event_step_by_env.get(env_id)
        agent["reset_seq"] = self.reset_seq_by_env.get(env_id, 0)
        agent["reset_signal"] = self.reset_signal_by_env.get(env_id)

    def _update_cumulative_counters(self, env_id, agent):
        for field in RESETTING_COUNTER_FIELDS:
            raw = agent.get(field)
            if raw is None:
                continue
            try:
                total, previous = accumulate_resetting_counter(
                    self.cumulative_agent_counters[field][env_id],
                    self.last_agent_counters[field].get(env_id),
                    raw,
                )
            except (TypeError, ValueError):
                continue
            self.cumulative_agent_counters[field][env_id] = total
            self.last_agent_counters[field][env_id] = previous
            agent[f"cumulative_{field}"] = total

    def _update_episode_tracking(self, env_id, map_id, x, y, agent):
        """Detect episode resets and clear that env's per-episode heat.

        Prefers the exact `completed_episodes` signal (once live); falls
        back to the fixed-start-state fingerprint heuristic described above
        only for envs that have never reported `completed_episodes` at all.
        """
        fingerprint = (map_id, x, y)
        reset_fired = False
        signal = None

        completed_raw = agent.get("completed_episodes")
        has_exact_field = completed_raw is not None
        if has_exact_field:
            try:
                completed = int(completed_raw)
            except (TypeError, ValueError):
                completed = None
            if completed is not None:
                previous = self.last_completed_episodes_by_env.get(env_id)
                if previous is not None and completed > previous:
                    reset_fired = True
                    signal = "exact"
                self.last_completed_episodes_by_env[env_id] = completed
        elif env_id not in self.last_completed_episodes_by_env:
            first = self.start_fingerprint_by_env.get(env_id)
            if first is None:
                self.start_fingerprint_by_env[env_id] = fingerprint
            elif fingerprint != first:
                self.away_since_reset_by_env[env_id] = True
            elif self.away_since_reset_by_env[env_id]:
                reset_fired = True
                signal = "heuristic"
                self.away_since_reset_by_env[env_id] = False

        if reset_fired:
            self.reset_seq_by_env[env_id] += 1
            self.reset_signal_by_env[env_id] = signal
            self.episode_heat_by_env[env_id] = collections.Counter()
            # Opens the timing bracket for this env's next objective clear.
            # Deliberately not seeded on first sighting: an env already
            # mid-episode when the viewer started has an unknown start, and
            # guessing "now" would under-report its first segment.
            self.episode_start_wall_by_env[env_id] = time.time()
            self._quest_clear_mark_by_env.pop(env_id, None)
        return reset_fired

    def _update_level_tracking(self, env_id, agent):
        """Record the highest Pikachu level any agent has ever reached.

        Every episode reloads the same level-5 save state, so the live
        `pikachu_level` field alone makes real leveling look like it never
        happened once that episode resets. This tracks the swarm-wide
        historical high independently of any single live snapshot.
        """
        level = agent.get("pikachu_level")
        try:
            level = int(level)
        except (TypeError, ValueError):
            return
        if level <= 0:
            return
        previous = self.max_level_ever_by_env.get(env_id)
        if previous is None:
            # First observation for this env -- record the baseline without
            # backfilling level_reached_counts. Using 0 as a default here
            # instead would make every agent's very first (normal, level-5)
            # reading look like five separate freshly-earned level-ups.
            self.max_level_ever_by_env[env_id] = level
            if level > self.max_level_ever:
                self.max_level_ever = level
            return
        if level > previous:
            for reached in range(previous + 1, level + 1):
                self.level_reached_counts[reached] += 1
            self.max_level_ever_by_env[env_id] = level
            if level > self.max_level_ever:
                self.max_level_ever = level

    def _prune_recent(self, now):
        cutoff = now - RECENT_WINDOW_SECONDS
        queue = self.recent_visits
        while queue and queue[0][0] < cutoff:
            _, x, y = queue.popleft()
            key = (x, y)
            self.recent_counter[key] -= 1
            if self.recent_counter[key] <= 0:
                del self.recent_counter[key]

    def _record_heat(self, env_id, agent):
        position = agent.get("last_position") or [0, 0, 0]
        if len(position) < 3:
            return None
        try:
            local_x, local_y, map_id = map(int, position[:3])
            offset = MAP_OFFSETS.get(map_id)
            if offset is None:
                return None
            global_x = local_x + int(offset[0])
            global_y = local_y + int(offset[1])
            if not (0 <= global_x < 436 and 0 <= global_y < 444):
                return None
        except (ValueError, TypeError):
            return None
        key = (global_x, global_y)
        env_counter = self.heatmap_by_env[env_id]
        env_counter[key] += 1
        self.heatmap_global[key] += 1
        self.heatmap_version += 1
        self.episode_heat_by_env[env_id][key] += 1
        now = time.time()
        self.recent_visits.append((now, global_x, global_y))
        self.recent_counter[key] += 1
        self.last_visit_by_env[env_id][key] = now
        self.last_visit_global[key] = now
        return [env_id, global_x, global_y, env_counter[key], self.heatmap_global[key], now]

    def _compute_aggregate(self):
        all_named_events = set()
        for events in self.events_by_env.values():
            all_named_events.update(events)
        # Count only catalogued flags, so the numerator can never exceed the
        # denominator regardless of what ends up in the log. Belt-and-braces
        # alongside the strict EVENT_LINE_RE parse.
        if EVENT_CATALOG_NAMES:
            all_named_events &= EVENT_CATALOG_NAMES
        total_completed_episodes = sum(
            self.cumulative_agent_counters["completed_episodes"].values()
        )
        total_whiteouts = sum(
            self.cumulative_agent_counters["whiteout_count"].values()
        )
        total_whiteout_episodes = sum(
            self.cumulative_agent_counters["whiteout_episodes"].values()
        )
        levels = []
        for agent in self.entries.values():
            level = agent.get("pikachu_level")
            if isinstance(level, (int, float)) and level > 0:
                levels.append(level)
        return {
            "agents_online": len(self.entries),
            "named_events_reached": len(all_named_events),
            "named_events_total": EVENT_CATALOG_TOTAL,
            "total_completed_episodes": total_completed_episodes,
            "total_whiteouts": total_whiteouts,
            "total_whiteout_episodes": total_whiteout_episodes,
            "swarm_whiteout_rate": (
                100.0 * total_whiteout_episodes / total_completed_episodes
                if total_completed_episodes else None
            ),
            "max_pikachu_level": max(levels) if levels else None,
            "avg_pikachu_level": (sum(levels) / len(levels)) if levels else None,
            "levels_reported": len(levels),
            "max_pikachu_level_ever": self.max_level_ever or None,
            "level_reached_counts": dict(sorted(self.level_reached_counts.items())),
        }

    def _run(self):
        interval = 1.0 / FEED_HZ
        next_save = time.monotonic() + HEATMAP_SAVE_SECONDS
        while True:
            started = time.monotonic()
            now_wall = time.time()
            # Ahead of the tail: a brand-new attempt must clear the old
            # timing ledger before this tick's clears are recorded into it.
            attempt_changed = self._poll_attempt_marker(now_wall)
            log_changed = self._tail_train_log()
            mastery_changed = self._reconcile_mastery_splits()
            dwell_changed = self._update_phase_dwell(now_wall)
            self._update_gate_trend(now_wall)
            changed = (
                log_changed or mastery_changed
                or attempt_changed or dwell_changed
            )
            heatmap_delta = []
            if log_changed:
                for env_id, agent in self.entries.items():
                    self._decorate_progress(env_id, agent)
            for env_id in range(MAX_ENVS):
                path = os.path.join(ROOT, f"live_agent_positions.env{env_id}.json")
                try:
                    mtime = os.stat(path).st_mtime_ns
                    if self.mtimes.get(env_id) == mtime:
                        continue
                    with open(path, encoding="utf-8") as handle:
                        agent = json.load(handle)
                    if int(agent.get("env_id")) != env_id:
                        continue
                    position = agent.get("last_position") or [0, 0, 0]
                    local_x = local_y = map_id = None
                    if len(position) >= 3:
                        try:
                            local_x, local_y, map_id = map(int, position[:3])
                        except (ValueError, TypeError):
                            local_x = local_y = map_id = None
                    self._update_cumulative_counters(env_id, agent)
                    if local_x is not None:
                        self._update_episode_tracking(env_id, map_id, local_x, local_y, agent)
                    self._update_level_tracking(env_id, agent)
                    heat = self._record_heat(env_id, agent)
                    if heat is not None:
                        heatmap_delta.append(heat)
                    # The trainer's legacy `extra` string reports the size of
                    # episode_coord_visits. That dictionary is reward-gated
                    # and has produced the misleading "coords: 0" label even
                    # while last_position and the viewer heatmap are healthy.
                    # Publish the viewer's own authoritative cumulative tile
                    # coverage as a typed field instead. The browser can then
                    # show both the real x/y/map and a reliable visit count
                    # without parsing a display string from the trainer.
                    visited_count = len(self.heatmap_by_env[env_id])
                    agent["visited_count"] = visited_count
                    self._decorate_progress(env_id, agent)
                    if local_x is not None:
                        # Keep `extra` useful for viewer tabs that loaded
                        # the pre-fix JavaScript and only know this legacy
                        # display field. New tabs use the typed fields.
                        agent["extra"] = (
                            f"x {local_x}, y {local_y}, map {map_id} • "
                            f"{visited_count:,} tiles visited • "
                            f"{agent['progress_count']} named events"
                        )
                    self.entries[env_id] = agent
                    self.mtimes[env_id] = mtime
                    changed = True
                except FileNotFoundError:
                    if env_id in self.entries:
                        self.entries.pop(env_id, None)
                        self.mtimes.pop(env_id, None)
                        changed = True
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue

            self._prune_recent(now_wall)

            if time.monotonic() >= self._next_service_poll:
                self._poll_service_status()
                self._next_service_poll = time.monotonic() + SERVICE_POLL_SECONDS
                changed = True

            if time.monotonic() >= self._next_history_poll:
                if self._read_history():
                    changed = True
                self._next_history_poll = time.monotonic() + HISTORY_POLL_SECONDS

            self.latest_aggregate = self._compute_aggregate()

            if changed:
                payload = json.dumps(
                    {
                        "agents": [self.entries[key] for key in sorted(self.entries)],
                        "heatmap_delta": heatmap_delta,
                        "heatmap_version": self.heatmap_version,
                        "training": {
                            "metrics": self.latest_metrics,
                            "metrics_updated_at": self.metrics_updated_at,
                            "log_updated_at": self.log_updated_at,
                            "service": self.service_status,
                            "aggregate": self.latest_aggregate,
                        },
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                with self.condition:
                    self.payload = payload
                    self.version += 1
                    self.condition.notify_all()
            if time.monotonic() >= next_save:
                self._save_heatmap()
                next_save = time.monotonic() + HEATMAP_SAVE_SECONDS
            elapsed = time.monotonic() - started
            time.sleep(max(0.01, interval - elapsed))

    def snapshot(self):
        with self.condition:
            return self.version, self.payload

    def wait(self, version, timeout=10.0):
        with self.condition:
            if self.version == version:
                self.condition.wait(timeout)
            return self.version, self.payload

    def heatmap_payload(self, env_value, mode="cumulative"):
        last_visit = None
        if mode == "recent":
            counter = self.recent_counter.copy()
        elif mode == "episode":
            if env_value == "all":
                counter = collections.Counter()
                for env_counter in self.episode_heat_by_env.values():
                    counter.update(env_counter)
            else:
                try:
                    env_id = int(env_value)
                except (TypeError, ValueError):
                    env_id = -1
                counter = self.episode_heat_by_env.get(env_id, collections.Counter()).copy()
        else:
            mode = "cumulative"
            if env_value == "all":
                counter = self.heatmap_global.copy()
                last_visit = self.last_visit_global
            else:
                try:
                    env_id = int(env_value)
                except (TypeError, ValueError):
                    env_id = -1
                counter = self.heatmap_by_env.get(env_id, collections.Counter()).copy()
                last_visit = self.last_visit_by_env.get(env_id, {})
        # Cumulative-only: pair each tile's count with its last-visit
        # timestamp so the client can fade a tile's color back toward blue
        # once it's gone cold, instead of a count earned once staying
        # permanently "hot" forever. `recent`/`episode` already have their
        # own built-in recency (a sliding window / per-episode reset), so
        # they don't need this extra field.
        if last_visit is not None:
            tiles = [
                [x, y, count, last_visit.get((x, y), 0)]
                for (x, y), count in sorted(counter.items())
            ]
        else:
            tiles = [[x, y, count] for (x, y), count in sorted(counter.items())]
        return json.dumps(
            {
                "env": env_value,
                "mode": mode,
                "version": self.heatmap_version,
                "started_at": self.heatmap_started_at,
                "recent_window_seconds": RECENT_WINDOW_SECONDS,
                "max_count": max(counter.values(), default=0),
                "tiles": tiles,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def tile_detail_payload(self, x_value, y_value):
        try:
            x = int(x_value)
            y = int(y_value)
        except (TypeError, ValueError):
            return json.dumps({"error": "invalid coordinates"}, separators=(",", ":")).encode("utf-8")
        key = (x, y)
        contributors = []
        for env_id, counter in self.heatmap_by_env.items():
            count = counter.get(key, 0)
            if count:
                contributors.append({"env_id": env_id, "count": count})
        contributors.sort(key=lambda item: -item["count"])
        return json.dumps(
            {
                "x": x,
                "y": y,
                "total": self.heatmap_global.get(key, 0),
                "recent": self.recent_counter.get(key, 0),
                "contributors": contributors[:10],
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def history_payload(self):
        return json.dumps(
            {"points": list(self.history), "fields": HISTORY_FIELDS},
            separators=(",", ":"),
        ).encode("utf-8")


FEED = AgentFeed()


class ViewerHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        if urlsplit(self.path).path.endswith(".html"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def _send_json(self, body, include_body=True):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _send_agents(self, include_body):
        _, body = FEED.snapshot()
        self._send_json(body, include_body)

    def _send_heatmap(self, include_body):
        query = parse_qs(urlsplit(self.path).query)
        env_value = query.get("env", ["all"])[0]
        mode = query.get("mode", ["cumulative"])[0]
        if mode not in ("cumulative", "recent", "episode"):
            mode = "cumulative"
        body = FEED.heatmap_payload(env_value, mode)
        self._send_json(body, include_body)

    def _send_tile_detail(self):
        query = parse_qs(urlsplit(self.path).query)
        x_value = query.get("x", [None])[0]
        y_value = query.get("y", [None])[0]
        self._send_json(FEED.tile_detail_payload(x_value, y_value), True)

    def _send_history(self, include_body):
        self._send_json(FEED.history_payload(), include_body)

    def _send_quest_splits(self, include_body):
        self._send_json(FEED.quest_splits_payload(), include_body)

    def _send_agent_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        version = -1
        try:
            self.wfile.write(b"retry: 1000\n\n")
            self.wfile.flush()
            while True:
                next_version, payload = FEED.wait(version, timeout=10.0)
                if next_version != version:
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    version = next_version
                else:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/live_agent_positions.json":
            self._send_agents(True)
        elif path == "/live_agent_positions.stream":
            self._send_agent_stream()
        elif path == "/live_agent_heatmap.json":
            self._send_heatmap(True)
        elif path == "/tile_detail.json":
            self._send_tile_detail()
        elif path == "/training_history.json":
            self._send_history(True)
        elif path == "/quest_splits.json":
            self._send_quest_splits(True)
        else:
            super().do_GET()

    def do_HEAD(self):
        path = urlsplit(self.path).path
        if path == "/live_agent_positions.json":
            self._send_agents(False)
        elif path == "/live_agent_heatmap.json":
            self._send_heatmap(False)
        elif path == "/training_history.json":
            self._send_history(False)
        else:
            super().do_HEAD()


if __name__ == "__main__":
    FEED.start()
    # Overridable so a fixture copy can be smoke-tested on the same host
    # without colliding with the live instance on 8000.
    port = int(os.environ.get("POKEMON_VIEWER_PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), ViewerHandler)
    server.daemon_threads = True
    server.serve_forever()
