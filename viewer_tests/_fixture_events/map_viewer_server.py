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
PROGRESS_CSV_PATH = os.path.join(ROOT, "checkpoints", "sb3_logs", "progress.csv")
EVENT_FLAGS_CATALOG_PATH = os.path.join(ROOT, "navigation_data", "yellow_event_flags.json")
HEATMAP_SAVE_SECONDS = 30.0
SERVICE_POLL_SECONDS = 5.0
HISTORY_POLL_SECONDS = 10.0
HISTORY_MAX_ROWS = 500
RECENT_WINDOW_SECONDS = 300.0
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


QUEST_CHAIN_PATH = os.path.join(ROOT, "navigation_data", "quest_chain.json")
_CHAIN_CACHE = {"mtime": None, "chain": None}


def load_quest_chain():
    """train.py's objective chain: name and clear condition for every phase.

    The splits ledger is built from `[QUEST]` *completion* lines, so it only
    ever knows objectives that have already been cleared -- and the objective
    being drilled right now is precisely the one that has not been. Reading
    names from the ledger alone left the gate card showing the literal
    "phase 56", and left `status_for` unable to tell a level grind from a
    deterministic objective (it detects grinds by the `train_` name prefix),
    so the frontier's promotion rule was reported as a plateau regardless.

    Exported by export_quest_chain.py, which imports train.py -- too heavy to
    do in a request thread. Staleness against train.py is reported instead of
    hidden: inserting a waypoint shifts every later phase index, so a stale
    chain does not just age, it starts describing the wrong objective.
    """
    try:
        mtime = os.stat(QUEST_CHAIN_PATH).st_mtime_ns
    except OSError:
        return {"phases": {}, "stale": None, "generated": None}
    if _CHAIN_CACHE["mtime"] == mtime and _CHAIN_CACHE["chain"] is not None:
        return _CHAIN_CACHE["chain"]
    chain = {"phases": {}, "stale": None, "generated": None}
    try:
        with open(QUEST_CHAIN_PATH, encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        chain["generated"] = payload.get("generated")
        for entry in payload.get("phases") or []:
            if isinstance(entry, dict) and entry.get("phase") is not None:
                chain["phases"][int(entry["phase"])] = entry
        source = payload.get("source") or {}
        try:
            stat = os.stat(TRAIN_PY_PATH)
            chain["stale"] = (
                int(source.get("mtime_ns") or -1) != stat.st_mtime_ns
                or int(source.get("size") or -1) != stat.st_size
            )
        except (OSError, ValueError, TypeError):
            chain["stale"] = None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        chain = {"phases": {}, "stale": None, "generated": None}
    _CHAIN_CACHE["mtime"] = mtime
    _CHAIN_CACHE["chain"] = chain
    return chain


MAP_OFFSETS = load_map_offsets()
EVENT_CATALOG_TOTAL = load_event_catalog_total()
EVENT_CATALOG_NAMES = load_event_catalog_names()


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
        # phase -> {'name', 'clears', 'best', 'last', 'samples', 'best_run_step'}
        self.quest_splits = {}
        self.quest_total_phases = 0
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

        if not chunk:
            return False
        self.log_updated_at = time.time()
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
        completed_phase = int(match.group(4)) - 1
        total_phases = int(match.group(5))
        name = match.group(6)
        if completed_phase < 0 or not 0 < total_phases <= 4096:
            return False
        self.quest_total_phases = max(self.quest_total_phases, total_phases)

        entry = self.quest_splits.get(completed_phase)
        if entry is None:
            entry = {
                'name': name, 'clears': 0, 'best': None, 'last': None,
                'samples': [], 'best_run_step': None,
            }
            self.quest_splits[completed_phase] = entry
        entry['name'] = name
        entry['clears'] += 1
        # `run_step` is cumulative for the episode, so it is only a run time
        # for a worker that started this episode at phase 0. Tracked as the
        # best *observed* cumulative step, and read as such.
        if entry['best_run_step'] is None or run_step < entry['best_run_step']:
            entry['best_run_step'] = run_step
        if phase_steps is None:
            return True
        entry['last'] = phase_steps
        if entry['best'] is None or phase_steps < entry['best']:
            entry['best'] = phase_steps
        entry['samples'].append(phase_steps)
        if len(entry['samples']) > QUEST_SPLIT_SAMPLES:
            del entry['samples'][:-QUEST_SPLIT_SAMPLES]
        return True

    def _reconcile_mastery_splits(self):
        """Backfill split times from the trainer's durable mastery ledger.

        The viewer tails only new log bytes after a restart. A quest can
        therefore be durably mastered while its historical `[QUEST]` timing
        line is behind the saved log cursor. The trainer's drill windows and
        all-time bests are authoritative fallback timing data for those blank
        rows. Live log observations still own `last` when one exists and may
        contribute a faster all-worker best.
        """
        try:
            mtime = os.stat(SWARM_MASTERY_PATH).st_mtime_ns
            if self.mastery_splits_mtime == mtime:
                return False
            with open(SWARM_MASTERY_PATH, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            self.mastery_splits_mtime = None
            return False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

        windows = payload.get('windows') or {}
        best_steps = payload.get('best_steps') or {}
        changed = False
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
                    'samples': [], 'best_run_step': None,
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
                changed = True
            if not entry['samples'] and samples:
                entry['samples'] = samples[-QUEST_SPLIT_SAMPLES:]
                changed = True
        self.mastery_splits_mtime = mtime
        return changed

    def _mastery_gate_payload(self):
        """Progress toward the promotion gate for the objective being drilled.

        Level grinds use successful-clear reliability because their duration
        is dominated by encounter and battle RNG. Deterministic objectives
        use the trainer's no-step-improvement plateau. Report the active mode
        directly so the viewer never describes an obsolete hit-rate rule.
        """
        config = load_drill_gate_config()
        no_improve_counts = {}
        try:
            with open(SWARM_MASTERY_PATH, encoding="utf-8") as handle:
                mastery_payload = json.load(handle) or {}
            windows = mastery_payload.get("windows") or {}
            no_improve_counts = mastery_payload.get("no_improve_counts") or {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            windows = {}
        chain = load_quest_chain()
        chain_phases = chain.get("phases") or {}
        frontier_phase = None
        frontier_objective = ''
        try:
            with open(SWARM_FRONTIER_META_PATH, encoding="utf-8") as handle:
                frontier_meta = json.load(handle) or {}
            frontier_phase = int(frontier_meta.get("quest_phase"))
            # The swarm writes the objective's name next to its index. This is
            # the one live, always-current label for the in-progress phase.
            frontier_objective = str(frontier_meta.get("quest_objective") or '')
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            frontier_phase = None

        # Early phases get the tighter budget. The boundary is the
        # reach_viridian .. enter_viridian_forest span, resolved by name from
        # the splits ledger so it tracks the quest chain rather than a
        # hardcoded index. reach_viridian is never *completed* -- the ladder
        # resumes envs past it -- so it has no ledger row; fall back to the
        # exported chain, which carries uncleared phases too.
        name_to_phase = {
            entry.get('name'): phase
            for phase, entry in self.quest_splits.items() if entry.get('name')
        }
        for phase, entry in chain_phases.items():
            if entry.get('name'):
                name_to_phase.setdefault(entry['name'], phase)
        early_lo = name_to_phase.get('reach_viridian', 0)
        early_hi = name_to_phase.get('enter_viridian_forest', 10)

        def budget_for(phase):
            return (config['early_max_steps'] if early_lo <= phase <= early_hi
                    else config['max_steps'])

        def status_for(phase):
            samples = []
            for value in windows.get(str(int(phase)), ()):
                try:
                    samples.append(int(value))
                except (ValueError, TypeError):
                    continue
            budget = budget_for(int(phase))
            under = sum(1 for s in samples if 0 <= s <= budget)
            n = len(samples)
            chain_entry = chain_phases.get(int(phase)) or {}
            # Ledger first (it is what the splits table is keyed on), then the
            # exported chain, then the swarm's own live label. Only the last
            # two can name an objective that has never been cleared.
            name = (
                (self.quest_splits.get(int(phase)) or {}).get('name')
                or chain_entry.get('name')
                or (frontier_objective if int(phase) == frontier_phase else '')
                or ''
            )
            is_grind = name.startswith('train_')
            if is_grind:
                mode = 'reliability'
                target = max(1, int(config['grind_successes']))
                progress = min(n, target)
                allowed = n >= target
            else:
                mode = 'plateau'
                target = max(1, int(config['no_improve_rotations']))
                best = None
                streak = 0
                for sample in samples:
                    if best is None or sample < best:
                        best = sample
                        streak = 0
                    else:
                        streak += 1
                # The trainer gates on a durable counter that survives the
                # rolling window (see record_swarm_drill_completion). The
                # window-derived streak below can only ever reach
                # window_size - 1, so for a phase whose plateau target is close
                # to the window it under-reports and the tab disagrees with the
                # gate the trainer actually applies. Prefer the durable value.
                has_durable_streak = False
                try:
                    streak = max(0, int(no_improve_counts[str(int(phase))]))
                    has_durable_streak = True
                except (KeyError, ValueError, TypeError):
                    pass
                progress = min(streak, target)
                allowed = (
                    streak >= target if has_durable_streak
                    # Mirror swarm_mastery_no_improve_plateau's window-only
                    # fallback, which also needs the establishing clear.
                    else (n >= target + 1 and streak >= target)
                )
            needed_hits = math.ceil(config['hit_rate'] * n) if n else 0
            return {
                'phase': int(phase),
                'name': name,
                # What clearing this objective actually requires, spelled out
                # from the waypoint's own conditions rather than its name.
                'kind': chain_entry.get('kind') or '',
                'goal': chain_entry.get('goal') or '',
                'criteria': chain_entry.get('criteria') or [],
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
        return {
            'config': config,
            'frontier_phase': frontier_phase,
            'frontier': status_for(frontier_phase) if frontier_phase is not None else None,
            'phases': [per_phase[k] for k in sorted(per_phase)],
            # True means quest_chain.json no longer matches train.py, so the
            # descriptions may be off by an inserted waypoint. Surfaced rather
            # than suppressed: a confidently wrong objective is the failure
            # this whole path exists to avoid.
            'chain_stale': chain.get('stale'),
            'chain_generated': chain.get('generated'),
        }

    # Yellow's own badge order, so the card reads like the in-game one.
    BADGE_NAMES = (
        'Boulder', 'Cascade', 'Thunder', 'Rainbow',
        'Soul', 'Marsh', 'Volcano', 'Earth',
    )

    def _player_card_payload(self):
        """Trainer card aggregated across live workers.

        Every worker resets onto the shared frontier state, so name/money/badges
        are the same run for all of them; play time differs only by how far into
        its own episode each worker is. Take the furthest for a clock that ticks
        instead of sawtoothing 96 ways, and the highest badge count so a worker
        caught mid-reset cannot blank the card.
        """
        agents = [a for a in self.entries.values() if isinstance(a, dict)]
        if not agents:
            return None

        def best(key, default=0):
            values = [a.get(key) for a in agents]
            values = [v for v in values if isinstance(v, (int, float))]
            return max(values) if values else default

        badge_flags = 0
        for agent in agents:
            flags = agent.get('badge_flags')
            if isinstance(flags, int):
                badge_flags |= flags
        names = [
            str(a.get('player_name')).strip()
            for a in agents
            if isinstance(a.get('player_name'), str) and str(a.get('player_name')).strip()
        ]
        play_time_seconds = int(best('play_time_seconds'))
        if not names and not badge_flags and not play_time_seconds:
            # Trainer predates the trainer-card fields; show nothing rather
            # than a card of zeroes that looks like a wiped save.
            return None
        return {
            'name': names[0] if names else None,
            'money': int(best('money')),
            'badge_flags': badge_flags,
            'badge_count': int(badge_flags).bit_count(),
            'badges': [
                {'name': name, 'earned': bool(badge_flags & (1 << index))}
                for index, name in enumerate(self.BADGE_NAMES)
            ],
            'play_time_seconds': play_time_seconds,
            'play_time_maxed': any(bool(a.get('play_time_maxed')) for a in agents),
            'workers': len(agents),
        }

    def quest_splits_payload(self):
        """Splits ledger plus the two totals worth watching.

        `sum_of_bests` is the theoretical run if every objective were cleared
        at its own best -- the target to chase. `timed_phases` says how much
        of the chain that sum actually covers, so a small sum from three
        cleared objectives can't be mistaken for a fast run.
        """
        rows = []
        for phase in sorted(self.quest_splits):
            entry = self.quest_splits[phase]
            samples = sorted(entry['samples'])
            rows.append({
                'phase': phase,
                'name': entry['name'],
                'clears': entry['clears'],
                'best': entry['best'],
                'last': entry['last'],
                'median': samples[len(samples) // 2] if samples else None,
                'best_run_step': entry['best_run_step'],
            })
        timed = [r for r in rows if r['best'] is not None]
        total_phases = self.quest_total_phases
        percent_complete = (
            round(100.0 * len(rows) / total_phases, 1)
            if isinstance(total_phases, int) and total_phases > 0
            else None
        )
        # Encoded here, like history_payload -- _send_json writes bytes.
        return json.dumps(
            {
                'total_phases': total_phases,
                'phases_cleared': len(rows),
                'percent_complete': percent_complete,
                'player_card': self._player_card_payload(),
                'timed_phases': len(timed),
                'sum_of_bests': sum(r['best'] for r in timed) if timed else None,
                'sum_of_medians': (
                    sum(r['median'] for r in timed) if timed else None
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
            log_changed = self._tail_train_log()
            mastery_changed = self._reconcile_mastery_splits()
            changed = log_changed or mastery_changed
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
