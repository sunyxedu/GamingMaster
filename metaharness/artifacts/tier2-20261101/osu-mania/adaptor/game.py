"""Headless, deterministic osu!mania ScoreV1 adaptor.

The real osu! client has no supported headless input API.  This module therefore
implements the native Mode 3 chart format and the stable mania judgement/score
profile as a small real-time simulator.  It deliberately has no wall-clock
thread: ``tick`` advances the chart clock by exactly ``tick_s``.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from metaharness.games.base import ActionSpec


TIER = 2
DETERMINISTIC = True
SEEDS = {
    "train": [100, 101, 102, 103, 104, 105, 106, 107],
    "heldout": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
}
EVAL_KWARGS: dict[str, Any] = {}

RULESET = "stable-scorev1-native-mania"
TICK_S = 0.016
DEFAULT_LOOKAHEAD_MS = 2000.0
MAX_PLANNING_HORIZON_MS = 4000.0

_RESULT_ORDER = ("MAX", "GREAT", "GOOD", "OK", "MEH", "MISS")
_HIT_VALUES = {"MAX": 320.0, "GREAT": 300.0, "GOOD": 200.0, "OK": 100.0, "MEH": 50.0, "MISS": 0.0}
_ACCURACY_VALUES = {"MAX": 300.0, "GREAT": 300.0, "GOOD": 200.0, "OK": 100.0, "MEH": 50.0, "MISS": 0.0}
_BONUS_VALUES = {"MAX": 32.0, "GREAT": 32.0, "GOOD": 16.0, "OK": 8.0, "MEH": 4.0, "MISS": 0.0}
_BONUS_INCREASE = {"MAX": 2.0, "GREAT": 1.0, "GOOD": 0.0, "OK": 0.0, "MEH": 0.0, "MISS": 0.0}
_BONUS_PENALTY = {"MAX": 0.0, "GREAT": 0.0, "GOOD": 8.0, "OK": 24.0, "MEH": 44.0, "MISS": math.inf}


@dataclass
class Note:
    object_id: int
    lane: int
    start_ms: float
    end_ms: float | None
    kind: str
    pattern: str = "mixed"
    resolved: bool = False

    @property
    def is_hold(self) -> bool:
        return self.end_ms is not None


@dataclass
class Chart:
    key_count: int
    od: float
    bpm: float
    notes: list[Note]
    duration_ms: float
    source: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveHold:
    note: Note
    head_error_ms: int
    broken: bool = False
    release_ms: float | None = None


def _stable_u(seed: int, label: str) -> float:
    """A version-independent [0, 1) value for a scenario component."""
    raw = f"osu-mania-v2\0{int(seed)}\0{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / float(1 << 64)


def _round_ms(value: float) -> int:
    """Round half away from zero, matching the documented integer-error profile."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _finite_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _native_windows(od: float) -> dict[str, int]:
    """Stable native mania half-windows in milliseconds."""
    return {
        "MAX": 16,
        "GREAT": max(1, int(64.0 - 3.0 * od)),
        "GOOD": max(1, int(97.0 - 3.0 * od)),
        "OK": max(1, int(127.0 - 3.0 * od)),
        "MEH": max(1, int(151.0 - 3.0 * od)),
        "MISS": max(1, int(188.0 - 3.0 * od)),
    }


def _lane_for_x(x: float, key_count: int) -> int:
    return max(0, min(key_count - 1, int(math.floor(x * key_count / 512.0))))


def _parse_osu_text(text: str, source: str = "chart.osu") -> Chart:
    """Parse the native Mode 3 fields used by this adaptor."""
    sections: dict[str, list[str]] = {}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            sections.setdefault(section, [])
        elif section:
            sections.setdefault(section, []).append(line)

    def fields(name: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in sections.get(name, []):
            if ":" in line:
                key, value = line.split(":", 1)
                out[key.strip().lower()] = value.strip()
        return out

    general = fields("General")
    if int(float(general.get("mode", "0"))) != 3:
        raise ValueError(f"{source} is not an osu!mania chart (Mode: 3 required)")
    difficulty = fields("Difficulty")
    key_count = int(round(float(difficulty.get("circlesize", "4"))))
    if not 1 <= key_count <= 18:
        raise ValueError(f"{source} has unsupported CircleSize {key_count}")
    od = _clamp(_finite_float(difficulty.get("overalldifficulty", "5"), 5.0), 0.0, 10.0)

    bpm = 120.0
    for line in sections.get("TimingPoints", []):
        try:
            timing_fields = line.split(",")
            beat_length = float(timing_fields[1])
        except (ValueError, IndexError):
            continue
        if beat_length > 0:
            bpm = round(60000.0 / beat_length, 3)
            break

    notes: list[Note] = []
    for raw in sections.get("HitObjects", []):
        parts = raw.split(",")
        if len(parts) < 5:
            continue
        try:
            x = float(parts[0])
            start = float(parts[2])
            object_type = int(parts[3])
        except (ValueError, IndexError):
            continue
        if not math.isfinite(x) or not math.isfinite(start):
            continue
        lane = _lane_for_x(x, key_count)
        end: float | None = None
        kind = "tap"
        if object_type & 128:
            params = parts[5] if len(parts) > 5 else ""
            try:
                end_candidate = float(params.split(":", 1)[0])
            except (ValueError, IndexError):
                end_candidate = start
            if end_candidate > start:
                end = end_candidate
                kind = "hold"
        notes.append(Note(0, lane, max(0.0, start), end, kind))

    if not notes:
        raise ValueError(f"{source} has no usable mania hit objects")
    notes.sort(key=lambda n: (n.start_ms, n.lane, n.end_ms or n.start_ms))
    for object_id, note in enumerate(notes, 1):
        note.object_id = object_id
    duration = max((n.end_ms if n.end_ms is not None else n.start_ms) for n in notes) + 1000.0
    holds = sum(n.is_hold for n in notes)
    metrics = {
        "note_count": len(notes),
        "hold_count": holds,
        "hold_fraction": holds / len(notes),
        "pattern_mix": {"external": len(notes)},
        "release_overlap_count": 0,
    }
    return Chart(key_count, od, bpm, notes, duration, source, metrics)


def _chart_to_native_text(key_count: int, od: float, bpm: float, notes: Iterable[Note]) -> str:
    """Serialize generated notes through the native format before parsing them."""
    beat = 60000.0 / bpm
    lines = [
        "osu file format v14",
        "",
        "[General]",
        "Mode: 3",
        "",
        "[Difficulty]",
        f"CircleSize: {key_count}",
        f"OverallDifficulty: {od:.2f}",
        "",
        "[TimingPoints]",
        f"0,{beat:.6f},4,2,1,100,1,0",
        "",
        "[HitObjects]",
    ]
    for note in notes:
        x = (note.lane + 0.5) * 512.0 / key_count
        if note.is_hold:
            params = f"{note.end_ms:.3f}:0:0:0:0"
            lines.append(f"{x:.3f},192,{note.start_ms:.3f},128,0,{params}")
        else:
            lines.append(f"{x:.3f},192,{note.start_ms:.3f},1,0,0:0:0:0:")
    return "\n".join(lines) + "\n"


def _generate_chart(seed: int) -> Chart:
    """Generate a varied but valid native mania chart from independent seed streams."""
    u = lambda label: _stable_u(seed, label)
    difficulty = u("difficulty")
    key_count = 3 + int(u("key-count") * 6.0)  # 3..8
    od = round(2.0 + u("od") * 8.0, 1)
    bpm_grid = (100, 110, 120, 132, 144, 156, 168, 180, 192, 210, 228, 246, 264)
    bpm = float(bpm_grid[int(u("bpm") * len(bpm_grid)) % len(bpm_grid)])
    subdivision = (1.0, 0.5, 1.0 / 3.0, 0.25, 1.0 / 6.0)[int(u("grid") * 5.0) % 5]
    beat = 60000.0 / bpm
    step = max(42.0, beat * subdivision)
    length_ms = 6000.0 + int(u("length") * 6500.0)
    intro_ms = 650.0 + int(u("intro") * 400.0)
    family_names = ("stream", "jack", "chord", "anchor", "mixed")
    family = family_names[int(u("family") * len(family_names)) % len(family_names)]
    hold_fraction = 0.04 + 0.23 * u("hold-fraction")
    density = 0.42 + 0.55 * difficulty
    lane_free = [0.0] * key_count
    notes: list[Note] = []
    pattern_counts: dict[str, int] = {name: 0 for name in ("tap", "chord", "stream", "jack", "anchor", "hold")}

    def row_unit(index: int, label: str) -> float:
        return _stable_u(seed, f"row-{index}-{label}")

    def choose_lanes(index: int, row: int) -> list[int]:
        local_family = family
        if row_unit(index, "family") > 0.76:
            local_family = family_names[int(row_unit(index, "alternate-family") * len(family_names)) % len(family_names)]
        if local_family == "jack":
            base = int(u("jack-lane") * key_count) % key_count
            lane = (base + (row // max(2, 2 + int(u("jack-run") * 5.0))) % key_count) % key_count
            lanes = [lane]
        elif local_family == "anchor":
            anchor = int(u("anchor-lane") * key_count) % key_count
            moving = (row + int(u("anchor-offset") * key_count)) % key_count
            lanes = [anchor] if moving == anchor else [anchor, moving]
        elif local_family == "chord":
            max_chord = min(key_count, 2 + int(difficulty * max(1, key_count - 1)))
            count = 2 + int(row_unit(index, "chord-size") * max(1, max_chord - 1))
            count = max(2, min(max_chord, count))
            start = int(row_unit(index, "chord-start") * key_count) % key_count
            lanes = [((start + j * (1 + int(row_unit(index, f"chord-step-{j}") * 2.0))) % key_count) for j in range(count)]
            lanes = list(dict.fromkeys(lanes))
        elif local_family == "stream":
            start = int(u("stream-lane") * key_count) % key_count
            direction = -1 if u("stream-direction") < 0.5 else 1
            lanes = [((start + direction * row) % key_count)]
            if row_unit(index, "stream-chord") < 0.10 * difficulty:
                lanes.append((lanes[0] + (1 if direction > 0 else -1)) % key_count)
        else:
            lanes = [int(row_unit(index, "lane") * key_count) % key_count]
            if row_unit(index, "mixed-chord") < 0.08 + 0.20 * difficulty:
                other = (lanes[0] + 1 + int(row_unit(index, "other-lane") * max(1, key_count - 1))) % key_count
                lanes.append(other)
        return list(dict.fromkeys(lanes))

    row = 0
    time_ms = intro_ms
    while time_ms <= intro_ms + length_ms:
        # Independent density stream: harder charts have more occupied rows, not only a faster family.
        occupied = row_unit(row, "density") < min(0.97, density)
        if occupied:
            lanes = choose_lanes(row, row)
            # A row is valid only if every selected lane is free.  This also prevents
            # a hold from overlapping another object in its own lane.
            available = [lane for lane in lanes if time_ms >= lane_free[lane] - 0.001]
            if available:
                is_chord = len(available) > 1
                for lane_index, lane in enumerate(available):
                    hold_test = row_unit(row + lane_index * 17, "hold")
                    make_hold = hold_test < hold_fraction and time_ms + step * 1.5 < intro_ms + length_ms + 1.0
                    end_ms: float | None = None
                    kind = "hold" if make_hold else "tap"
                    if make_hold:
                        duration_steps = 2 + int(row_unit(row + lane_index * 23, "hold-length") * (3 + int(4 * difficulty)))
                        end_ms = time_ms + max(step * 1.5, duration_steps * step)
                        # Keep the generated object within the chart and leave the lane free after it.
                        end_ms = min(end_ms, intro_ms + length_ms)
                        if end_ms <= time_ms:
                            end_ms = None
                            kind = "tap"
                    pattern = "chord" if is_chord else family
                    if pattern == "mixed":
                        pattern = "tap"
                    notes.append(Note(0, lane, time_ms, end_ms, kind, pattern))
                    lane_free[lane] = (end_ms if end_ms is not None else time_ms) + 0.1
                    pattern_counts["hold" if end_ms is not None else "tap"] += 1
                    if is_chord:
                        pattern_counts["chord"] += 1
                    elif pattern in pattern_counts:
                        pattern_counts[pattern] += 1
        row += 1
        time_ms = intro_ms + row * step
        if len(notes) >= 360:
            break

    # Ensure even an unlucky density stream has a meaningful opening and tail.
    # Keep a deterministic hold in the fallback when this scenario's independent
    # hold stream calls for one; do not turn the fallback into an all-tap chart.
    if len(notes) < 20:
        notes.clear()
        lane_free = [0.0] * key_count
        fallback_step = max(step, 80.0)
        hold_index = 3 + int(u("fallback-hold-index") * 12.0)
        for i in range(24):
            t = intro_ms + i * fallback_step
            lane = i % key_count
            make_hold = i == hold_index and hold_fraction >= 0.08
            end = t + 2.0 * fallback_step if make_hold else None
            notes.append(Note(0, lane, t, end, "hold" if make_hold else "tap", "stream"))
    notes.sort(key=lambda n: (n.start_ms, n.lane, n.end_ms or n.start_ms))
    for object_id, note in enumerate(notes, 1):
        note.object_id = object_id
    native_text = _chart_to_native_text(key_count, od, bpm, notes)
    chart = _parse_osu_text(native_text, source=f"generated-seed-{int(seed)}.osu")
    # Retain generator labels (the native parser intentionally knows no pattern family).
    by_key = {(round(n.start_ms, 3), n.lane, round(n.end_ms or n.start_ms, 3)): n.pattern for n in notes}
    for note in chart.notes:
        note.pattern = by_key.get((round(note.start_ms, 3), note.lane, round(note.end_ms or note.start_ms, 3)), "mixed")
    actual_pattern_counts: dict[str, int] = {}
    for note in chart.notes:
        label = "hold" if note.is_hold else note.pattern
        actual_pattern_counts[label] = actual_pattern_counts.get(label, 0) + 1
    chart.metrics.update({
        "difficulty": round(difficulty, 4),
        "family": family,
        "density": round(density, 4),
        "grid_subdivision": subdivision,
        "interval_ms": round(step, 3),
        "length_ms": round(length_ms, 3),
        "hold_fraction_target": round(hold_fraction, 4),
        "pattern_mix": actual_pattern_counts,
    })
    chart.metrics["release_overlap_count"] = _release_overlap_count(chart.notes)
    chart.metrics["note_count"] = len(chart.notes)
    chart.metrics["hold_count"] = sum(note.is_hold for note in chart.notes)
    chart.metrics["hold_fraction"] = chart.metrics["hold_count"] / max(1, chart.metrics["note_count"])
    return chart


def _release_overlap_count(notes: list[Note]) -> int:
    holds = [note for note in notes if note.is_hold]
    count = 0
    for hold in holds:
        for other in notes:
            if other.lane == hold.lane or other.object_id == hold.object_id:
                continue
            if abs(other.start_ms - float(hold.end_ms)) <= 120.0:
                count += 1
                break
    return count


def _load_external_charts(root: str | os.PathLike[str] | None) -> list[Chart]:
    if not root:
        return []
    path = Path(root).expanduser()
    if not path.is_dir():
        return []
    charts: list[Chart] = []
    try:
        paths = sorted(path.rglob("*.osu"))
    except OSError:
        return []
    for chart_path in paths[:256]:
        try:
            charts.append(_parse_osu_text(chart_path.read_text(encoding="utf-8", errors="replace"), str(chart_path)))
        except (OSError, ValueError, UnicodeError):
            continue
    return charts


def _chart_observation_features(chart: Chart) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Decode chart rows, chord groups, and local pattern cues for observations.

    These are derived from the same parsed hit objects that the judge uses; no
    model is expected to reconstruct simultaneous groups or jack/anchor runs
    from a flat note list.
    """
    grouped: dict[float, list[Note]] = {}
    for note in chart.notes:
        grouped.setdefault(round(note.start_ms, 3), []).append(note)
    row_keys = sorted(grouped)
    rows: list[dict[str, Any]] = []
    features: dict[int, dict[str, Any]] = {}
    for index, row_key in enumerate(row_keys):
        group = sorted(grouped[row_key], key=lambda note: (note.lane, note.object_id))
        lanes = sorted({note.lane for note in group})
        previous_interval = None if index == 0 else round(row_key - row_keys[index - 1], 3)
        rows.append({
            "index": index,
            "start_ms": row_key,
            "lanes": lanes,
            "notes": group,
            "interval_from_previous_ms": previous_interval,
            "anchor_lane": None,
            "pattern": "chord" if len(lanes) > 1 else ("hold" if group[0].is_hold else (group[0].pattern if group[0].pattern in {"stream", "jack", "anchor"} else "single")),
            "same_lane_run_index": 0,
            "same_lane_run_length": 1,
            "stream": False,
        })

    # A jack is a run of single-lane rows; an anchor is a repeated lane inside
    # otherwise multi-lane rows. Both are cues, not alternate engine rules.
    index = 0
    while index < len(rows):
        row = rows[index]
        if len(row["lanes"]) == 1:
            lane = row["lanes"][0]
            end = index
            while end + 1 < len(rows) and len(rows[end + 1]["lanes"]) == 1 and rows[end + 1]["lanes"][0] == lane:
                end += 1
            run_length = end - index + 1
            for run_index in range(index, end + 1):
                rows[run_index]["same_lane_run_index"] = run_index - index
                rows[run_index]["same_lane_run_length"] = run_length
            index = end + 1
        else:
            index += 1

    def close_interval(a: float | None, b: float | None) -> bool:
        if a is None or b is None:
            return False
        return abs(a - b) <= max(2.0, 0.15 * max(abs(a), abs(b)))

    for index, row in enumerate(rows):
        if len(row["lanes"]) > 1:
            nearby: dict[int, int] = {}
            for other in rows[max(0, index - 2):min(len(rows), index + 3)]:
                for lane in other["lanes"]:
                    nearby[lane] = nearby.get(lane, 0) + 1
            repeated = [lane for lane, count in nearby.items() if lane in row["lanes"] and count >= 2]
            if repeated:
                row["anchor_lane"] = min(repeated)
                row["pattern"] = "anchor"
        elif row["same_lane_run_length"] < 3:
            previous = rows[index - 1] if index else None
            following = rows[index + 1] if index + 1 < len(rows) else None
            if (previous and following and len(previous["lanes"]) == 1 and len(following["lanes"]) == 1
                    and previous["lanes"] != row["lanes"] != following["lanes"]
                    and close_interval(row["interval_from_previous_ms"], following["interval_from_previous_ms"])):
                row["pattern"] = "stream"
                row["stream"] = True

    previous_lane_time: dict[int, float] = {}
    for row in rows:
        for note in row["notes"]:
            lane_previous = previous_lane_time.get(note.lane)
            row_pattern = row["pattern"]
            if row_pattern == "single":
                row_pattern = "hold" if note.is_hold else (note.pattern if note.pattern in {"stream", "jack", "anchor"} else "single")
            features[note.object_id] = {
                "row_index": row["index"],
                "row_start_ms": row["start_ms"],
                "chord_size": len(row["lanes"]),
                "chord_lanes": list(row["lanes"]),
                "is_chord": len(row["lanes"]) > 1,
                "pattern": row_pattern,
                "anchor_lane": row["anchor_lane"],
                "stream": bool(row["stream"]),
                "same_lane_run_index": row["same_lane_run_index"],
                "same_lane_run_length": row["same_lane_run_length"],
                "interval_from_previous_ms": row["interval_from_previous_ms"],
                "lane_interval_ms": None if lane_previous is None else round(note.start_ms - lane_previous, 3),
            }
            previous_lane_time[note.lane] = note.start_ms

    row_summaries: list[dict[str, Any]] = []
    for row in rows:
        row_summaries.append({
            "row_index": row["index"],
            "start_ms": row["start_ms"],
            "lanes": list(row["lanes"]),
            "object_ids": [note.object_id for note in row["notes"]],
            "is_chord": len(row["lanes"]) > 1,
            "pattern": row["pattern"],
            "anchor_lane": row["anchor_lane"],
            "interval_from_previous_ms": row["interval_from_previous_ms"],
        })
    return features, row_summaries


class OsuManiaGame:
    name = "osu!mania (stable native ScoreV1 simulator)"
    tick_s = TICK_S

    action_spec = ActionSpec(
        names=(
            "noop",
            "event <time_ms> <lane> <down|up>",
            "events <time_ms> <lane> <down|up> ...",
            "press <lane>",
            "release <lane>",
            "tap <lane>",
        ),
        descriptions={
            "noop": "Advance one 16 ms chart tick without submitting a key transition.",
            "event <time_ms> <lane> <down|up>": "Queue one timestamped lane transition; concrete example: event 1325 2 down.",
            "events <time_ms> <lane> <down|up> ...": "Queue a space-separated batch in nondecreasing time order; concrete example: events 1325 2 down 1360 2 up.",
            "press <lane>": "Press a lane at the current chart time (shorthand for a down transition).",
            "release <lane>": "Release a lane at the current chart time (shorthand for an up transition).",
            "tap <lane>": "Press and release a lane at the current chart time; useful for a quick local tap.",
        },
        noop="noop",
        notes="Lane numbers are zero-based.  Event times are chart milliseconds, not wall-clock seconds. "
              "Only future events within the visible planning horizon are accepted; a past event is applied at the current time and is judged late. "
              "A down transition must be near a note head; holding a key does not invent a hit. "
              "Malformed actions are ignored and reported in the next observation.",
    )

    def __init__(self, *, lookahead_ms: float | None = None, chart_root: str | None = None,
                 ruleset: str | None = None) -> None:
        selected_ruleset = ruleset if ruleset is not None else os.environ.get("OSU_MANIA_RULESET", RULESET)
        if selected_ruleset != RULESET:
            raise ValueError(f"unsupported OSU_MANIA_RULESET {selected_ruleset!r}; only {RULESET!r} is implemented")
        if lookahead_ms is None:
            lookahead_ms = _finite_float(os.environ.get("OSU_MANIA_LOOKAHEAD_MS", DEFAULT_LOOKAHEAD_MS), DEFAULT_LOOKAHEAD_MS)
        self.lookahead_ms = _clamp(lookahead_ms, 250.0, 5000.0)
        self.planning_horizon_ms = min(MAX_PLANNING_HORIZON_MS, max(self.lookahead_ms, 1000.0))
        self.chart_root = chart_root if chart_root is not None else os.environ.get("OSU_MANIA_CHART_ROOT")
        self._external_charts = _load_external_charts(self.chart_root)
        self._closed = False
        self._seed = 0
        self._chart: Chart | None = None
        self._time_ms = 0.0
        self._ticks = 0
        self._key_down: list[bool] = []
        self._active_holds: dict[int, ActiveHold] = {}
        self._event_queue: list[tuple[float, int, int, bool]] = []
        self._event_sequence = 0
        self._submitted_event_keys: set[tuple[float, int, bool]] = set()
        self._counts = {result: 0 for result in _RESULT_ORDER}
        self._combo = 0
        self._max_combo = 0
        self._bonus = 100.0
        self._raw_score = 0.0
        self._recent_results: list[dict[str, Any]] = []
        self._timing_errors: list[dict[str, Any]] = []
        self._hold_breaks = 0
        self._holds_seen = 0
        self._last_action_status = "not_started"
        self._action_errors: list[str] = []
        self._reset_generation = 0

    def reset(self, seed: int) -> dict[str, Any]:
        try:
            self._seed = int(seed)
        except (TypeError, ValueError):
            self._seed = 0
        self._closed = False
        if self._external_charts:
            index = int.from_bytes(hashlib.sha256(f"chart\0{self._seed}".encode()).digest()[:8], "big") % len(self._external_charts)
            self._chart = self._clone_chart(self._external_charts[index])
            self._chart.source = f"external:{self._chart.source}"
        else:
            self._chart = _generate_chart(self._seed)
        self._time_ms = 0.0
        self._ticks = 0
        self._key_down = [False] * self._chart.key_count
        self._active_holds.clear()
        self._event_queue.clear()
        self._event_sequence = 0
        self._submitted_event_keys.clear()
        self._counts = {result: 0 for result in _RESULT_ORDER}
        self._combo = 0
        self._max_combo = 0
        self._bonus = 100.0
        self._raw_score = 0.0
        self._recent_results.clear()
        self._timing_errors.clear()
        self._hold_breaks = 0
        self._holds_seen = sum(note.is_hold for note in self._chart.notes)
        self._last_action_status = "reset"
        self._action_errors.clear()
        self._reset_generation += 1
        return self.observe()

    @staticmethod
    def _clone_chart(chart: Chart) -> Chart:
        notes = [Note(n.object_id, n.lane, n.start_ms, n.end_ms, n.kind, n.pattern) for n in chart.notes]
        return Chart(chart.key_count, chart.od, chart.bpm, notes, chart.duration_ms, chart.source, dict(chart.metrics))

    def observe(self) -> dict[str, Any]:
        chart = self._chart
        if chart is None:
            return {"game": "osu!mania", "ruleset": RULESET, "ready": False, "done": True, "score": 0.0}
        windows = _native_windows(chart.od)
        features, row_summaries = _chart_observation_features(chart)
        horizon = self._time_ms + self.lookahead_ms
        visible: list[list[dict[str, Any]]] = [[] for _ in range(chart.key_count)]
        visible_row_indices: set[int] = set()
        visible_note_ids: set[int] = set()
        for note in chart.notes:
            if note.resolved or note.object_id in self._active_holds:
                continue
            if self._time_ms - windows["MISS"] <= note.start_ms <= horizon:
                feature = features[note.object_id]
                view: dict[str, Any] = {
                    "id": note.object_id,
                    "lane": note.lane,
                    "start_ms": _clean_number(note.start_ms),
                    "time_to_start_ms": _clean_number(note.start_ms - self._time_ms),
                    "end_ms": _clean_number(note.end_ms) if note.end_ms is not None else None,
                    "time_to_tail_ms": _clean_number(note.end_ms - self._time_ms) if note.end_ms is not None else None,
                    "kind": note.kind,
                    "motion": "falling_toward_judgement_line",
                    "pattern": feature["pattern"],
                    "row_index": feature["row_index"],
                    "chord_size": feature["chord_size"],
                    "chord_lanes": feature["chord_lanes"],
                    "is_chord": feature["is_chord"],
                    "anchor_lane": feature["anchor_lane"],
                    "stream": feature["stream"],
                    "same_lane_run_index": feature["same_lane_run_index"],
                    "same_lane_run_length": feature["same_lane_run_length"],
                    "interval_from_previous_ms": _clean_number(feature["interval_from_previous_ms"]),
                    "lane_interval_ms": _clean_number(feature["lane_interval_ms"]),
                    "head_window_ms": {
                        "earliest_ms": _clean_number(note.start_ms - windows["MISS"]),
                        "latest_ms": _clean_number(note.start_ms + windows["MISS"]),
                    },
                }
                if note.end_ms is not None:
                    view.update({
                        "hold_duration_ms": _clean_number(note.end_ms - note.start_ms),
                        "release_window_ms": {
                            "earliest_ms": _clean_number(note.end_ms - windows["MEH"]),
                            "latest_ms": _clean_number(note.end_ms + windows["OK"]),
                        },
                        "release_due": self._time_ms >= note.end_ms - windows["MEH"],
                    })
                else:
                    view.update({"hold_duration_ms": None, "release_window_ms": None, "release_due": False})
                visible[note.lane].append(view)
                visible_row_indices.add(feature["row_index"])
                visible_note_ids.add(note.object_id)
        lanes = [
            {"lane": lane, "key_down": bool(self._key_down[lane]), "upcoming": visible[lane]}
            for lane in range(chart.key_count)
        ]
        active = []
        for _, hold in sorted(self._active_holds.items(), key=lambda item: (item[1].note.lane, item[1].note.object_id)):
            lane = hold.note.lane
            tail_ms = float(hold.note.end_ms)
            key_down = bool(self._key_down[lane])
            release_due = self._time_ms >= tail_ms - windows["MEH"]
            risk = "high" if hold.broken or not key_down or self._time_ms > tail_ms + windows["OK"] else ("medium" if release_due else "low")
            active.append({
                "id": hold.note.object_id,
                "lane": lane,
                "head_ms": _clean_number(hold.note.start_ms),
                "tail_ms": _clean_number(tail_ms),
                "time_to_tail_ms": _clean_number(tail_ms - self._time_ms),
                "hold_duration_ms": _clean_number(tail_ms - hold.note.start_ms),
                "motion": "held_until_tail",
                "head_error_ms": hold.head_error_ms,
                "key_down": key_down,
                "hold_broken": bool(hold.broken),
                "release_window_ms": {
                    "earliest_ms": _clean_number(tail_ms - windows["MEH"]),
                    "latest_ms": _clean_number(tail_ms + windows["OK"]),
                },
                "release_due": release_due,
                "hold_break_risk": risk,
            })
        visible_patterns: list[dict[str, Any]] = []
        for row in row_summaries:
            if row["row_index"] not in visible_row_indices:
                continue
            pending_ids = [object_id for object_id in row["object_ids"] if object_id in visible_note_ids]
            pending_lanes = sorted({chart.notes[object_id - 1].lane for object_id in pending_ids})
            visible_patterns.append({
                "row_index": row["row_index"],
                "start_ms": _clean_number(row["start_ms"]),
                "time_to_start_ms": _clean_number(row["start_ms"] - self._time_ms),
                "lanes": row["lanes"],
                "pending_lanes": pending_lanes,
                "object_ids": pending_ids,
                "is_chord": row["is_chord"],
                "pattern": row["pattern"],
                "anchor_lane": row["anchor_lane"],
                "interval_from_previous_ms": _clean_number(row["interval_from_previous_ms"]),
            })
        total = len(chart.notes)
        judged = sum(self._counts.values())
        remaining = sum(not note.resolved for note in chart.notes)
        accuracy = self._accuracy()
        timing_summary = self._timing_summary()
        chart_remaining = max(0.0, chart.duration_ms - self._time_ms)
        return {
            "game": "osu!mania",
            "ruleset": RULESET,
            "seed": self._seed,
            "source": chart.source,
            "time_ms": _clean_number(self._time_ms),
            "current_time_ms": _clean_number(self._time_ms),
            "timers": {
                "current_time_ms": _clean_number(self._time_ms),
                "lookahead_ms": _clean_number(self.lookahead_ms),
                "planning_horizon_ms": _clean_number(self.planning_horizon_ms),
                "chart_remaining_ms": _clean_number(chart_remaining),
                "tick_s": self.tick_s,
            },
            "tick_s": self.tick_s,
            "ticks": self._ticks,
            "key_count": chart.key_count,
            "od": _clean_number(chart.od),
            "bpm": _clean_number(chart.bpm),
            "lookahead_ms": _clean_number(self.lookahead_ms),
            "planning_horizon_ms": _clean_number(self.planning_horizon_ms),
            "chart_duration_ms": _clean_number(chart.duration_ms),
            "lanes": lanes,
            "visible_patterns": visible_patterns,
            "active_holds": active,
            "recent_results": list(self._recent_results[-12:]),
            "judgement_windows_ms": windows,
            "score": self.score,
            "accuracy": accuracy,
            "counts": dict(self._counts),
            "combo": self._combo,
            "final_combo": self._combo,
            "max_combo": self._max_combo,
            "remaining_objects": remaining,
            "total_objects": total,
            "judged_objects": judged,
            "miss_rate": self._counts["MISS"] / max(1, total),
            "judged_miss_rate": self._counts["MISS"] / max(1, judged),
            "hold_break_rate": self._hold_breaks / max(1, self._holds_seen),
            "hold_breaks": self._hold_breaks,
            "resources": {
                "health": None,
                "health_model": "disabled; chart runs to completion",
                "score": self.score,
                "accuracy": accuracy,
                "combo": self._combo,
            },
            "timing_error_summary": timing_summary,
            "timing_errors_ms": list(self._timing_errors[-128:]),
            "chart_metrics": _json_safe(chart.metrics),
            "done": self.done,
            "last_action": self._last_action_status,
            "action_errors": list(self._action_errors[-3:]),
        }

    def describe(self) -> str:
        return (
            "This is a deterministic offline osu!mania Mode 3 simulator using the native .osu chart "
            "format and the stable ScoreV1 profile (not current osu!lazer). The chart clock advances "
            "16 ms for every tick, independent of model latency; there are no opponents, audio, display, "
            "account, health fail, mods, or network. Objective: score each visible note. A tap needs a "
            "down transition in its lane near its start. A hold needs a down near its head, continuous "
            "key-down state, and an up near its tail. Chords are simultaneous notes in different lanes; "
            "jacks repeat a lane and anchors combine a repeated lane with another lane. Stable native "
            "windows are MAX ±16 ms, GREAT ±(64-3*OD), GOOD ±(97-3*OD), OK ±(127-3*OD), MEH "
            "±(151-3*OD), with MISS at ±(188-3*OD), all in chart milliseconds. Hold notes receive one "
            "judgement: head error and tail error are combined; an early release breaks the hold and "
            "caps it at MEH, while a late MEH-window release is a miss. Score is native ScoreV1 out of "
            "1,000,000; MAX and GREAT both weigh 300 for mania accuracy. The observation is structured "
            "JSON: chart time and timers, per-lane upcoming notes with lane, kind, falling motion, head/tail "
            "times and windows, decoded chord lanes, stream/jack/anchor cues and intervals, visible pattern "
            "rows, key state, active-hold release windows and break risk, recent signed head/tail errors, "
            "judgement windows, result counts, score, accuracy, combo, maximum combo, miss/hold-break rates, "
            "and seeded chart metrics. Health is explicitly disabled and there are no other resources. Use "
            "zero-based lanes. Action grammar: `event TIME LANE down|up` or a batch "
            "`events TIME LANE down|up TIME LANE down|up ...`; `press`, `release`, and `tap` use the "
            "current time, and `noop` only advances. Event times must be nondecreasing within a batch and "
            "future events may be scheduled only within the planning horizon. Malformed actions are ignored."
        )

    def tick(self, action: Any) -> None:
        """Submit the current action, then apply one fixed real-time chart tick."""
        if self._closed or self._chart is None or self.done:
            self._last_action_status = "ignored_after_done"
            return
        self._last_action_status = "noop"
        self._action_errors.clear()
        events = self._events_from_action(action)
        if events is not None:
            if self._queue_events(events):
                self._last_action_status = "accepted" if events else "noop"
            else:
                self._last_action_status = "ignored"
        else:
            self._last_action_status = "ignored"
        self._advance_to(self._time_ms + self.tick_s * 1000.0)
        self._ticks += 1

    def act(self, action: Any) -> dict[str, Any]:
        """Optional direct batch submission helper for local integrations."""
        if not self._closed and self._chart is not None and not self.done:
            self._last_action_status = "noop"
            self._action_errors.clear()
            events = self._events_from_action(action)
            if events is not None:
                self._last_action_status = "accepted" if self._queue_events(events) else "ignored"
            else:
                self._last_action_status = "ignored"
        return self.observe()

    def advance(self, to_time_ms: float) -> dict[str, Any]:
        """Optional deterministic clock helper; the evaluator normally calls tick()."""
        if self._closed or self._chart is None or self.done:
            return self.observe()
        target = _finite_float(to_time_ms, self._time_ms)
        self._advance_to(max(self._time_ms, target))
        return self.observe()

    def _events_from_action(self, action: Any) -> list[tuple[float, int, bool]] | None:
        if action is None or action == "" or action == "noop":
            return []
        if isinstance(action, dict):
            raw = action.get("events")
            if raw is None and {"time_ms", "lane", "down"}.issubset(action):
                raw = [action]
            return self._parse_event_records(raw)
        if isinstance(action, (list, tuple)):
            return self._parse_event_records(action)
        if not isinstance(action, str):
            self._action_errors.append(f"unsupported action type {type(action).__name__}")
            return None
        command = action.strip()
        if not command or command == "noop":
            return []
        # Permit a JSON object/list as a convenient structured equivalent.
        if command[:1] in "[{":
            try:
                return self._events_from_action(json.loads(command))
            except (json.JSONDecodeError, TypeError):
                self._action_errors.append("action JSON could not be decoded")
                return None
        parts = command.replace(",", " ").split()
        verb = parts[0].lower() if parts else ""
        if verb in {"press", "release", "tap", "hold"}:
            if len(parts) != 2:
                self._action_errors.append(f"{verb} expects one lane")
                return None
            try:
                lane = int(parts[1])
            except ValueError:
                self._action_errors.append("lane is not an integer")
                return None
            if verb == "press":
                return [(self._time_ms, lane, True)]
            if verb == "release":
                return [(self._time_ms, lane, False)]
            if verb == "tap":
                return [(self._time_ms, lane, True), (self._time_ms, lane, False)]
            return [(self._time_ms, lane, True)]
        if verb in {"event", "schedule", "events"}:
            tokens = parts[1:]
            if verb != "events" and len(tokens) != 3:
                self._action_errors.append("event expects TIME LANE down|up")
                return None
            if verb == "events" and len(tokens) % 3 != 0:
                self._action_errors.append("events expects repeated TIME LANE down|up triples")
                return None
            records: list[tuple[float, int, bool]] = []
            for i in range(0, len(tokens), 3):
                try:
                    time_ms = float(tokens[i])
                    lane = int(tokens[i + 1])
                except (ValueError, IndexError):
                    self._action_errors.append("event time/lane is malformed")
                    return None
                down = _parse_down(tokens[i + 2])
                if down is None or not math.isfinite(time_ms):
                    self._action_errors.append("event transition must be down or up")
                    return None
                records.append((time_ms, lane, down))
            return records
        self._action_errors.append(f"unknown action {command[:80]!r}")
        return None

    def _parse_event_records(self, raw: Any) -> list[tuple[float, int, bool]] | None:
        if raw is None:
            self._action_errors.append("events field is missing")
            return None
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        records: list[tuple[float, int, bool]] = []
        for item in raw:
            if not isinstance(item, dict):
                self._action_errors.append("each event must be an object")
                return None
            try:
                time_ms = float(item.get("time_ms", item.get("time")))
                lane = int(item.get("lane"))
            except (TypeError, ValueError):
                self._action_errors.append("event needs numeric time_ms and lane")
                return None
            down = _parse_down(item.get("down"))
            if not math.isfinite(time_ms) or down is None:
                self._action_errors.append("event has invalid time or transition")
                return None
            records.append((time_ms, lane, down))
        return records

    def _queue_events(self, events: list[tuple[float, int, bool]]) -> bool:
        if self._chart is None:
            return False
        previous: float | None = None
        valid: list[tuple[float, int, bool]] = []
        new_keys: list[tuple[float, int, bool]] = []
        for requested_time, lane, down in events:
            if previous is not None and requested_time < previous:
                self._action_errors.append("event timestamps must be nondecreasing")
                return False
            previous = requested_time
            if lane < 0 or lane >= self._chart.key_count:
                self._action_errors.append(f"lane {lane} is outside 0..{self._chart.key_count - 1}")
                return False
            event_key = (round(requested_time, 6), lane, down)
            # The Tier 2 mailbox repeats its last action on stale ticks.  A
            # timestamped transition is therefore idempotent: replaying the same
            # action must not create a new late event every 16 ms.
            if event_key in self._submitted_event_keys or event_key in new_keys:
                continue
            # A queued event in the past is physically applied now, preserving its
            # lateness rather than silently moving it to the requested timestamp.
            effective_time = max(self._time_ms, requested_time)
            if effective_time > self._time_ms + self.planning_horizon_ms + 1e-6:
                self._action_errors.append(f"event at {requested_time:g} ms exceeds planning horizon")
                return False
            valid.append((effective_time, lane, down))
            new_keys.append(event_key)
        for event_time, lane, down in valid:
            heapq.heappush(self._event_queue, (event_time, self._event_sequence, lane, down))
            self._event_sequence += 1
        self._submitted_event_keys.update(new_keys)
        return True

    def _advance_to(self, target_ms: float) -> None:
        if self._chart is None:
            return
        target_ms = max(self._time_ms, target_ms)
        while self._event_queue and self._event_queue[0][0] <= target_ms + 1e-7:
            event_time = self._event_queue[0][0]
            self._process_deadlines_before(event_time)
            self._time_ms = max(self._time_ms, event_time)
            same_time: list[tuple[int, bool]] = []
            while self._event_queue and abs(self._event_queue[0][0] - event_time) <= 1e-7:
                _, sequence, lane, down = heapq.heappop(self._event_queue)
                same_time.append((lane, down))
            for lane, down in same_time:
                self._apply_transition(event_time, lane, down)
        self._time_ms = target_ms
        self._process_deadlines_before(target_ms)

    def _process_deadlines_before(self, at_time: float) -> None:
        """Resolve objects whose strict deadline is already behind the clock."""
        if self._chart is None:
            return
        windows = _native_windows(self._chart.od)
        for note in self._chart.notes:
            if note.resolved:
                continue
            if note.object_id in self._active_holds:
                hold = self._active_holds[note.object_id]
                if note.end_ms is not None and at_time > note.end_ms + windows["MISS"] + 1e-7:
                    self._hold_breaks += 1
                    self._resolve_hold(hold, _round_ms(at_time - note.end_ms), at_time, forced_miss=True)
            elif at_time > note.start_ms + windows["MISS"] + 1e-7:
                self._record_result(note, "MISS", None, None, False)

    def _apply_transition(self, event_time: float, lane: int, down: bool) -> None:
        if self._chart is None or not (0 <= lane < len(self._key_down)):
            return
        if down:
            if self._key_down[lane]:
                return
            self._key_down[lane] = True
            windows = _native_windows(self._chart.od)
            candidates = [
                note for note in self._chart.notes
                if not note.resolved and note.lane == lane and note.start_ms - windows["MISS"] - 1e-7 <= event_time <= note.start_ms + windows["MISS"] + 1e-7
            ]
            if not candidates:
                return
            note = min(candidates, key=lambda item: (abs(event_time - item.start_ms), item.start_ms, item.object_id))
            error = _round_ms(event_time - note.start_ms)
            if note.is_hold:
                self._active_holds[note.object_id] = ActiveHold(note, error)
            else:
                result = self._tap_result(abs(error), windows)
                self._record_result(note, result, error, None, False)
        else:
            if not self._key_down[lane]:
                return
            self._key_down[lane] = False
            hold = next((h for h in self._active_holds.values() if h.note.lane == lane), None)
            if hold is None:
                return
            self._finish_hold(hold, event_time)

    @staticmethod
    def _tap_result(abs_error: int, windows: dict[str, int]) -> str:
        for result in _RESULT_ORDER:
            if abs_error <= windows[result]:
                return result
        return "MISS"

    def _finish_hold(self, hold: ActiveHold, release_time: float) -> None:
        if self._chart is None or hold.note.end_ms is None:
            return
        windows = _native_windows(self._chart.od)
        hold.release_ms = release_time
        tail_error = _round_ms(release_time - hold.note.end_ms)
        if release_time < hold.note.end_ms - windows["MEH"]:
            hold.broken = True
            self._hold_breaks += 1
            self._resolve_hold(hold, tail_error, release_time, forced_miss=False)
            return
        if release_time > hold.note.end_ms + windows["OK"]:
            hold.broken = True
            self._hold_breaks += 1
            self._resolve_hold(hold, tail_error, release_time, forced_miss=True)
            return
        self._resolve_hold(hold, tail_error, release_time, forced_miss=False)

    def _resolve_hold(self, hold: ActiveHold, tail_error: int | None, when: float, forced_miss: bool) -> None:
        note = hold.note
        self._active_holds.pop(note.object_id, None)
        if forced_miss or tail_error is None:
            result = "MISS"
        else:
            windows = _native_windows(self._chart.od if self._chart else 5.0)
            head_abs = abs(hold.head_error_ms)
            tail_abs = abs(tail_error)
            combined = head_abs + tail_abs
            if hold.broken:
                result = "MEH"
            elif head_abs <= 1.2 * windows["MAX"] and combined <= 2.4 * windows["MAX"]:
                result = "MAX"
            elif head_abs <= 1.1 * windows["GREAT"] and combined <= 2.2 * windows["GREAT"]:
                result = "GREAT"
            elif head_abs <= windows["GOOD"] and combined <= 2.0 * windows["GOOD"]:
                result = "GOOD"
            elif head_abs <= windows["OK"] and combined <= 2.0 * windows["OK"]:
                result = "OK"
            elif head_abs <= windows["MISS"] and combined <= 2.0 * windows["MISS"]:
                result = "MEH"
            else:
                result = "MISS"
        self._record_result(note, result, hold.head_error_ms, tail_error, hold.broken)

    def _record_result(self, note: Note, result: str, head_error: int | None, tail_error: int | None,
                       hold_break: bool) -> None:
        if note.resolved:
            return
        note.resolved = True
        self._counts[result] += 1
        if result == "MISS" or hold_break:
            self._combo = 0
        else:
            self._combo += 1
            self._max_combo = max(self._max_combo, self._combo)
        self._apply_score(result)
        record: dict[str, Any] = {
            "id": note.object_id,
            "lane": note.lane,
            "kind": note.kind,
            "result": result,
            "head_error_ms": head_error,
            "tail_error_ms": tail_error,
            "hold_break": bool(hold_break),
            "time_ms": _clean_number(self._time_ms),
        }
        self._recent_results.append(record)
        self._timing_errors.append({"id": note.object_id, "head_error_ms": head_error, "tail_error_ms": tail_error})
        if len(self._recent_results) > 64:
            del self._recent_results[:-64]
        if len(self._timing_errors) > 2000:
            del self._timing_errors[:-2000]

    def _apply_score(self, result: str) -> None:
        if self._chart is None or not self._chart.notes:
            return
        unit = 1_000_000.0 * 0.5 / len(self._chart.notes)
        # Stable ScoreV1 uses the current bonus for this object's bonus half, then
        # updates the multiplier for the following object.
        self._raw_score += unit * (_HIT_VALUES[result] / 320.0)
        self._raw_score += unit * (_BONUS_VALUES[result] * math.sqrt(max(0.0, self._bonus)) / 320.0)
        penalty = _BONUS_PENALTY[result]
        if math.isinf(penalty):
            self._bonus = 0.0
        else:
            self._bonus = _clamp(self._bonus + _BONUS_INCREASE[result] - penalty, 0.0, 100.0)

    def _accuracy(self) -> float:
        judged = sum(self._counts.values())
        if judged == 0:
            return 1.0
        numerator = sum(_ACCURACY_VALUES[result] * self._counts[result] for result in _RESULT_ORDER)
        return numerator / (300.0 * judged)

    def _timing_summary(self) -> dict[str, Any]:
        heads = [item["head_error_ms"] for item in self._timing_errors if item["head_error_ms"] is not None]
        tails = [item["tail_error_ms"] for item in self._timing_errors if item["tail_error_ms"] is not None]
        def summary(values: list[int]) -> dict[str, Any]:
            if not values:
                return {"count": 0, "mean_ms": 0.0, "min_ms": 0, "max_ms": 0}
            return {
                "count": len(values),
                "mean_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        return {"head": summary(heads), "tail": summary(tails)}

    @property
    def done(self) -> bool:
        if self._chart is None:
            return True
        return all(note.resolved for note in self._chart.notes) and not self._active_holds

    @property
    def score(self) -> float:
        return float(_clamp(round(self._raw_score), 0.0, 1_000_000.0))

    @property
    def ticks(self) -> int:
        return int(self._ticks)

    def score_breakdown(self) -> dict[str, float]:
        judged = sum(self._counts.values())
        total = len(self._chart.notes) if self._chart else 0
        return {
            "native_score_v1": float(self.score),
            "accuracy": float(self._accuracy()),
            "progress": float(judged / max(1, total)),
            "combo_reliability": float(self._max_combo / max(1, total)),
            "miss_rate": float(self._counts["MISS"] / max(1, total)),
            "hold_break_rate": float(self._hold_breaks / max(1, self._holds_seen)),
        }

    def info(self) -> dict[str, Any]:
        chart = self._chart
        return {
            "valid_trial": True,
            "invalid_reason": None,
            "seed": self._seed,
            "ruleset": RULESET,
            "chart_source": chart.source if chart else None,
            "chart_metrics": _json_safe(chart.metrics if chart else {}),
            "counts": dict(self._counts),
            "score": self.score,
            "accuracy": self._accuracy(),
            "combo": self._combo,
            "final_combo": self._combo,
            "max_combo": self._max_combo,
            "timing_error_summary": self._timing_summary(),
            "timing_errors_ms": list(self._timing_errors[-2000:]),
            "hold_breaks": self._hold_breaks,
            "miss_rate": self._counts["MISS"] / max(1, len(chart.notes) if chart else 0),
            "hold_break_rate": self._hold_breaks / max(1, self._holds_seen),
        }

    def close(self) -> None:
        """No child process is owned; clear transient state and remain idempotent."""
        self._event_queue.clear()
        self._submitted_event_keys.clear()
        self._active_holds.clear()
        self._closed = True


def _parse_down(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"down", "press", "true", "1"}:
            return True
        if lowered in {"up", "release", "false", "0"}:
            return False
    return None


def _clean_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if not math.isfinite(float(value)):
        return 0.0
    if abs(float(value) - round(float(value))) < 1e-9:
        return int(round(float(value)))
    return float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return _clean_number(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def make_game(**kwargs: Any) -> OsuManiaGame:
    """Construct one isolated simulator instance for an evaluator episode."""
    allowed = {key: kwargs[key] for key in ("lookahead_ms", "chart_root", "ruleset") if key in kwargs}
    return OsuManiaGame(**allowed)
