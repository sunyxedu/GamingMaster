"""Headless Malody Key-mode adaptor.

This module implements the chosen offline interface from the study: a small,
seeded emulator for native-mode-0 ``.mc`` charts.  It intentionally has no
client, display, audio device, network connection, or account dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
import random
import threading
from typing import Any, Iterable, Mapping

from metaharness.games.base import ActionSpec


TIER = 2
DETERMINISTIC = True
# Both groups contain the same eight scenario families.  Held-out values are
# different chart instances, rather than a separate difficulty family.
SEEDS = {
    "train": [100, 101, 102, 103, 104, 105, 106, 107],
    "heldout": [200, 201, 202, 203, 204, 205, 206, 207],
}
EVAL_KWARGS: dict[str, Any] = {}


_JUDGMENT_ORDER = ("Big P", "p1", "p2", "p3", "G", "M")
_JUDGMENT_WEIGHTS = {
    "Big P": 1.00,
    "p1": 0.90,
    "p2": 0.85,
    "p3": 0.80,
    "G": 0.40,
    "M": 0.00,
}
# The public RM article's desktop timing table.  Key-mode scoring is not
# officially specified here; the table is used as the documented emulator
# profile and is described as such in ADAPTOR.md.
_WINDOWS = {
    "A": (60.0, 105.0, 120.0, 135.0, 200.0),
    "B": (47.0, 63.0, 88.0, 118.0, 200.0),
    "C": (40.0, 56.0, 81.0, 111.0, 200.0),
    "D": (34.0, 50.0, 75.0, 105.0, 200.0),
    "E": (25.0, 50.0, 75.0, 105.0, 200.0),
}
_RM_JUDGE_MULTIPLIER = {"A": 0.78, "B": 0.85, "C": 1.00, "D": 1.10, "E": 1.20}
_RM_MOD_MULTIPLIER = {"none": 1.00, "Flip": 1.00, "Const": 0.95}


@dataclass
class _Note:
    note_id: int
    raw_column: int
    column: int
    start_ms: float
    end_ms: float | None
    kind: str
    start_beat: float = 0.0
    end_beat: float | None = None
    pattern: str = "tap"
    bpm: float = 0.0
    group_id: int = 0
    group_size: int = 1
    group_columns: tuple[int, ...] = ()
    head_judged: bool = False
    head_judgment: str | None = None
    head_offset_ms: float | None = None
    tail_judged: bool = False
    tail_judgment: str | None = None
    tail_offset_ms: float | None = None
    active: bool = False

    @property
    def is_hold(self) -> bool:
        return self.end_ms is not None


@dataclass(frozen=True)
class _TimingPoint:
    beat: float
    bpm: float


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clean_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return float(round(value, 3))


def _beat_value(value: Any) -> float:
    """Decode Malody's [measure, numerator, denominator] beat triple."""
    if _finite(value):
        return float(value)
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"invalid beat {value!r}")
    measure, numerator, denominator = value[:3]
    if not (_finite(measure) and _finite(numerator) and _finite(denominator)):
        raise ValueError(f"non-finite beat {value!r}")
    denominator = float(denominator)
    if denominator == 0:
        raise ValueError(f"zero beat denominator {value!r}")
    beat = float(measure) + float(numerator) / denominator
    if not math.isfinite(beat):
        raise ValueError(f"non-finite beat {value!r}")
    return beat


def _beat_triplet(beat: float) -> list[int]:
    """Encode a beat without losing the fractional snap in a native note."""
    fraction = Fraction(float(beat)).limit_denominator(192)
    measure = fraction.numerator // fraction.denominator
    remainder = fraction - measure
    if remainder == 0:
        return [int(measure), 0, 1]
    return [int(measure), int(remainder.numerator), int(remainder.denominator)]


def _read_timing_point(item: Any) -> _TimingPoint:
    if isinstance(item, Mapping):
        beat = item.get("beat", item.get("at", 0))
        bpm = item.get("bpm")
    elif isinstance(item, (list, tuple)):
        values = list(item)
        if values and isinstance(values[0], (list, tuple)):
            beat = values[0]
            bpm = values[1] if len(values) > 1 else None
        elif len(values) >= 4:
            # Common flattened .mc convention: measure, numerator,
            # denominator, bpm.  Some versions put a sign before bpm.
            beat = values[:3]
            bpm = values[3]
            if len(values) >= 5 and _finite(values[4]) and _finite(values[3]):
                # A value at index three outside a plausible BPM range is
                # usually the time-signature/sign field.
                if float(values[3]) <= 0 or float(values[3]) > 1000:
                    bpm = values[4]
        else:
            raise ValueError(f"invalid timing point {item!r}")
    else:
        raise ValueError(f"invalid timing point {item!r}")
    beat_value = _beat_value(beat)
    if not _finite(bpm) or float(bpm) <= 0:
        raise ValueError(f"invalid BPM in timing point {item!r}")
    return _TimingPoint(beat_value, float(bpm))


def _beat_to_ms(beat: float, points: list[_TimingPoint], offset_ms: float) -> float:
    """Piecewise-linear beat conversion, including the chart offset once."""
    if not points:
        raise ValueError("a chart needs at least one timing point")
    if beat <= points[0].beat:
        return offset_ms + (beat - points[0].beat) * 60000.0 / points[0].bpm
    elapsed = 0.0
    for current, following in zip(points, points[1:]):
        if beat <= following.beat:
            elapsed += (beat - current.beat) * 60000.0 / current.bpm
            return offset_ms + elapsed
        elapsed += (following.beat - current.beat) * 60000.0 / current.bpm
    last = points[-1]
    return offset_ms + elapsed + (beat - last.beat) * 60000.0 / last.bpm


def _parse_native_chart(chart: Mapping[str, Any], mod: str) -> tuple[list[_Note], list[_TimingPoint], float, dict[str, Any]]:
    """Parse and validate the subset of native .mc needed by Key mode."""
    if not isinstance(chart, Mapping):
        raise ValueError("native chart must be an object")
    meta = chart.get("meta")
    if not isinstance(meta, Mapping) or int(meta.get("mode", -1)) != 0:
        raise ValueError("native chart meta.mode must be 0 (Key)")
    mode_ext = meta.get("mode_ext", {})
    columns = mode_ext.get("column") if isinstance(mode_ext, Mapping) else None
    if not isinstance(columns, int) or isinstance(columns, bool) or not 1 <= columns <= 16:
        raise ValueError("native chart meta.mode_ext.column must be an integer from 1 to 16")

    raw_time = chart.get("time")
    if not isinstance(raw_time, list) or not raw_time:
        raise ValueError("native chart time must be a non-empty list")
    points = sorted((_read_timing_point(x) for x in raw_time), key=lambda p: p.beat)
    unique: list[_TimingPoint] = []
    for point in points:
        if unique and abs(point.beat - unique[-1].beat) < 1e-9:
            unique[-1] = point
        else:
            unique.append(point)
    points = unique

    offset = meta.get("offset", chart.get("offset", 0.0))
    if not _finite(offset):
        raise ValueError("chart offset must be finite")
    offset_ms = float(offset)
    raw_notes = chart.get("note")
    if not isinstance(raw_notes, list) or not raw_notes:
        raise ValueError("native chart note must be a non-empty list")

    parsed: list[tuple[float, int, float | None]] = []
    for raw in raw_notes:
        if not isinstance(raw, Mapping):
            raise ValueError("every native note must be an object")
        beat = _beat_value(raw.get("beat"))
        raw_column = raw.get("column")
        if not isinstance(raw_column, int) or isinstance(raw_column, bool) or not 0 <= raw_column < columns:
            raise ValueError(f"illegal note column {raw_column!r}")
        end_raw = raw.get("endbeat")
        end_beat = None if end_raw is None else _beat_value(end_raw)
        if end_beat is not None and end_beat < beat:
            raise ValueError("hold endbeat precedes beat")
        parsed.append((beat, raw_column, end_beat))

    parsed.sort(key=lambda item: (item[0], item[1], item[2] if item[2] is not None else item[0]))
    seen: set[tuple[float, int]] = set()
    notes: list[_Note] = []
    for note_id, (beat, raw_column, end_beat) in enumerate(parsed, start=1):
        key = (round(beat, 9), raw_column)
        if key in seen:
            raise ValueError(f"duplicate same-column note at beat {beat}")
        seen.add(key)
        effective_column = columns - 1 - raw_column if mod == "Flip" else raw_column
        start_ms = _beat_to_ms(beat, points, offset_ms)
        end_ms = None if end_beat is None else _beat_to_ms(end_beat, points, offset_ms)
        if end_ms is not None and end_ms < start_ms:
            raise ValueError("hold end time precedes start time")
        notes.append(_Note(
            note_id=note_id,
            raw_column=raw_column,
            column=effective_column,
            start_ms=start_ms,
            end_ms=end_ms,
            kind="hold" if end_ms is not None else "tap",
            start_beat=beat,
            end_beat=end_beat,
        ))

    _annotate_notes(notes, points, columns)
    summary = {
        "mode": "key",
        "columns": columns,
        "offset_ms": _clean_float(offset_ms),
        "bpm_segments": [
            {"start_beat": _clean_float(p.beat), "bpm": _clean_float(p.bpm)} for p in points
        ],
        "note_count": len(notes),
    }
    return notes, points, offset_ms, summary


def _bpm_at_beat(beat: float, points: list[_TimingPoint]) -> float:
    current = points[0].bpm
    for point in points:
        if point.beat > beat:
            break
        current = point.bpm
    return float(current)


def _annotate_notes(notes: list[_Note], points: list[_TimingPoint], columns: int) -> None:
    """Decode timing groups and the pattern cues a player would read."""
    groups: dict[float, list[_Note]] = {}
    for note in notes:
        note.bpm = _bpm_at_beat(note.start_beat, points)
        groups.setdefault(round(note.start_ms, 6), []).append(note)
    for group_id, key in enumerate(sorted(groups), start=1):
        members = groups[key]
        group_columns = tuple(sorted(note.column for note in members))
        for note in members:
            note.group_id = group_id
            note.group_size = len(members)
            note.group_columns = group_columns

    by_column: dict[int, list[_Note]] = {}
    for note in notes:
        by_column.setdefault(note.column, []).append(note)
    for lane_notes in by_column.values():
        lane_notes.sort(key=lambda note: note.start_beat)
    all_starts = sorted(note.start_beat for note in notes)
    for note in notes:
        if note.is_hold:
            note.pattern = "hold"
            continue
        if note.group_size > 1:
            note.pattern = "chord"
            continue
        lane_notes = by_column.get(note.column, [])
        lane_index = next((i for i, candidate in enumerate(lane_notes) if candidate is note), -1)
        same_lane_neighbor = any(
            abs(lane_notes[i].start_beat - note.start_beat) <= 0.51
            for i in (lane_index - 1, lane_index + 1)
            if 0 <= i < len(lane_notes)
        )
        if same_lane_neighbor:
            note.pattern = "jack"
        elif note.column in {0, columns - 1}:
            note.pattern = "anchor"
        elif any(abs(start - note.start_beat) <= 0.51 and start != note.start_beat for start in all_starts):
            note.pattern = "stream"
        else:
            note.pattern = "tap"


def _add_generated_note(
    notes: list[dict[str, Any]],
    occupied: dict[int, list[tuple[float, float]]],
    beat: float,
    column: int,
    hold_beats: float = 0.0,
) -> bool:
    """Add a valid note while avoiding impossible same-lane overlaps."""
    beat = round(float(beat), 6)
    end = round(beat + float(hold_beats), 6) if hold_beats > 0 else beat
    if column in occupied:
        for old_start, old_end in occupied[column]:
            if beat < old_end - 1e-8 and end > old_start + 1e-8:
                return False
            if abs(beat - old_start) < 1e-8:
                return False
    raw: dict[str, Any] = {"beat": _beat_triplet(beat), "column": int(column)}
    if hold_beats > 0:
        raw["endbeat"] = _beat_triplet(end)
    notes.append(raw)
    occupied.setdefault(column, []).append((beat, end if hold_beats > 0 else beat + 0.001))
    return True


def _make_native_chart(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate one deterministic native-mode-0 chart and its scenario label."""
    rng = random.Random(seed * 1000003 + 97)
    family = seed % 8
    family_names = (
        "isolated_taps", "alternation", "streams", "jacks",
        "chords", "anchors", "holds", "mixed_dense",
    )
    columns = 4 + (family % 4)
    offsets = (-18.0, -8.0, 0.0, 9.0, 17.0, 12.0, -13.0, 5.0)
    mods = ("none", "none", "Flip", "Const", "none", "Flip", "Const", "none")
    snap_denoms = (4, 8, 16, 8, 12, 16, 8, 16)
    base_bpm = 108.0 + (seed % 7) * 7.0 + family * 2.0
    changing_bpm = family in (1, 2, 4, 7)
    time_entries: list[dict[str, Any]] = [{"beat": [0, 0, 1], "bpm": base_bpm}]
    if changing_bpm:
        time_entries.append({"beat": [8, 0, 1], "bpm": base_bpm + (12.0 if family % 2 else -10.0)})
        if family in (2, 7):
            time_entries.append({"beat": [16, 0, 1], "bpm": base_bpm + 6.0})

    raw_notes: list[dict[str, Any]] = []
    occupied: dict[int, list[tuple[float, float]]] = {}
    start = 2.0 + (rng.randrange(0, 3) / 8.0)
    lane_shift = rng.randrange(columns)

    if family == 0:  # isolated, deliberately spacious taps
        for i in range(18 + seed % 4):
            _add_generated_note(raw_notes, occupied, start + i * 0.75, (lane_shift + i * 2) % columns)
    elif family == 1:  # alternating outer/inner stream
        for i in range(28 + seed % 5):
            beat = start + i * 0.5
            lane = (lane_shift + (i % 2) * (columns - 1)) % columns
            _add_generated_note(raw_notes, occupied, beat, lane)
            if i % 7 == 3:
                _add_generated_note(raw_notes, occupied, beat, (lane + 1) % columns)
    elif family == 2:  # dense cyclic stream with occasional chord
        for i in range(38 + seed % 5):
            beat = start + i * 0.25
            lane = (lane_shift + i * (1 + seed % 3)) % columns
            _add_generated_note(raw_notes, occupied, beat, lane)
            if i % 9 == 4:
                _add_generated_note(raw_notes, occupied, beat, (lane + 2) % columns)
    elif family == 3:  # same-lane jacks followed by lane changes
        for i in range(13 + seed % 3):
            beat = start + i * 0.75
            lane = (lane_shift + i // 2) % columns
            _add_generated_note(raw_notes, occupied, beat, lane)
            _add_generated_note(raw_notes, occupied, beat + 0.25, lane)
            if i % 4 == 1:
                _add_generated_note(raw_notes, occupied, beat + 0.5, (lane + 1) % columns)
    elif family == 4:  # two- and three-note chords
        for i in range(20 + seed % 4):
            beat = start + i * 0.625
            left = (lane_shift + i) % columns
            _add_generated_note(raw_notes, occupied, beat, left)
            _add_generated_note(raw_notes, occupied, beat, (left + 1 + i % 2) % columns)
            if i % 5 == 2:
                _add_generated_note(raw_notes, occupied, beat, (left + columns - 1) % columns)
    elif family == 5:  # anchors on outer lanes with inner responses
        for i in range(27 + seed % 4):
            beat = start + i * 0.5
            anchor = 0 if i % 2 == 0 else columns - 1
            _add_generated_note(raw_notes, occupied, beat, anchor)
            if i % 3 == 1:
                _add_generated_note(raw_notes, occupied, beat, (lane_shift + i) % columns)
            if i % 8 == 6:
                _add_generated_note(raw_notes, occupied, beat + 0.25, columns // 2)
    elif family == 6:  # holds interleaved with taps, including a final release
        for i in range(10 + seed % 3):
            beat = start + i * 1.25
            lane = (lane_shift + i * 2) % columns
            length = 0.75 if i % 3 else 1.25
            if not _add_generated_note(raw_notes, occupied, beat, lane, length):
                continue
            for delta, other in ((0.25, (lane + 1) % columns), (0.75, (lane + 2) % columns)):
                _add_generated_note(raw_notes, occupied, beat + delta, other)
        final_beat = start + (11 + seed % 3) * 1.25
        _add_generated_note(raw_notes, occupied, final_beat, (lane_shift + 1) % columns, 1.5)
    else:  # mixed transfer chart: taps, chords, jacks, streams and holds
        for i in range(12):
            beat = start + i * 0.5
            lane = (lane_shift + i) % columns
            _add_generated_note(raw_notes, occupied, beat, lane)
            if i % 4 == 1:
                _add_generated_note(raw_notes, occupied, beat, (lane + 2) % columns)
        hold_beat = start + 6.75
        _add_generated_note(raw_notes, occupied, hold_beat, lane_shift % columns, 1.5)
        for i in range(18):
            beat = hold_beat + 0.5 + i * 0.25
            lane = (lane_shift + 1 + i * 2) % columns
            _add_generated_note(raw_notes, occupied, beat, lane)
            if i % 6 == 3:
                _add_generated_note(raw_notes, occupied, beat, (lane + 1) % columns)
        final_beat = hold_beat + 5.5
        _add_generated_note(raw_notes, occupied, final_beat, (lane_shift + 3) % columns, 1.0)

    raw_notes.sort(key=lambda n: (_beat_value(n["beat"]), int(n["column"]), _beat_value(n.get("endbeat", n["beat"]))))
    chart = {
        "meta": {
            "mode": 0,
            "mode_ext": {"column": columns},
            "version": 1,
            "creator": "local deterministic adaptor",
            "offset": offsets[family],
        },
        "time": time_entries,
        "note": raw_notes,
    }
    scenario = {
        "family": family_names[family],
        "difficulty": round(1.0 + len(raw_notes) / 18.0 + (columns - 4) * 0.35, 2),
        "columns": columns,
        "offset_ms": offsets[family],
        "snap_resolution": f"1/{snap_denoms[family]} beat grid",
        "mod": mods[family],
        "changing_bpm": changing_bpm,
        "chart_id": f"generated-mc-seed-{seed}",
    }
    return chart, scenario


class MalodyKeyGame:
    """A deterministic, in-process RealtimeGame for Malody native Key charts."""

    name = "Malody Key (offline native .mc emulator)"

    def __init__(self, **kwargs: Any) -> None:
        tick_s = kwargs.get("tick_s", 0.04)
        self.tick_s = float(tick_s) if _finite(tick_s) and float(tick_s) > 0 else 0.04
        requested_judge = str(kwargs.get("judge", "C")).upper()
        self._default_judge = requested_judge if requested_judge in _WINDOWS else "C"
        self._lock = threading.RLock()
        names: list[str] = ["noop"]
        descriptions: dict[str, str] = {
            "noop": "Do not press or release a lane; the chart clock advances one 40 ms tick.",
        }
        for column in range(7):
            press = f"press {column}"
            release = f"release {column}"
            names.extend((press, release))
            descriptions[press] = f"Press the displayed Key column {column} at this tick; starts a tap or hold head."
            descriptions[release] = f"Release the displayed Key column {column}; resolves its active hold tail."
        for size in (2, 3):
            for combo in itertools.combinations(range(7), size):
                label = "press " + "+".join(map(str, combo))
                names.append(label)
                descriptions[label] = "Press these columns simultaneously (a chord) at this tick."
        self.action_spec = ActionSpec(
            names=names,
            descriptions=descriptions,
            noop="noop",
            notes=(
                "Grammar also accepts JSON actions {type: press|release, column: N, at_ms: T}, "
                "{type: batch, events: [...]}, or {type: advance, to_ms: T}. Columns are zero-based "
                "after the active Flip option. A press is a tap unless it matches a hold head; a hold "
                "stays down until release. Malformed actions are treated as no-op and reported in the "
                "next observation."
            ),
        )
        self._closed = False
        self._seed = 0
        self._judge = self._default_judge
        self._reset_state()

    def _reset_state(self) -> None:
        self._chart: dict[str, Any] = {}
        self._scenario: dict[str, Any] = {}
        self._notes: list[_Note] = []
        self._timing: list[_TimingPoint] = []
        self._offset_ms = 0.0
        self._mod = "none"
        self._time_ms = 0.0
        self._ticks = 0
        self._done = False
        self._combo = 0
        self._max_combo = 0
        self._misses = 0
        self._broken_holds = 0
        self._judgment_count = {name: 0 for name in _JUDGMENT_ORDER}
        self._tail_count = {name: 0 for name in _JUDGMENT_ORDER}
        self._quality_sum = 0.0
        self._quality_count = 0
        self._rm_proxy = 0.0
        self._last_judgments: list[dict[str, Any]] = []
        self._last_error: str | None = None
        self._last_action = "noop"
        self._future_clock_target: float | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        with self._lock:
            try:
                seed_int = int(seed)
            except (TypeError, ValueError):
                seed_int = 0
            chart, scenario = _make_native_chart(seed_int)
            mod = str(scenario["mod"])
            notes, timing, offset, summary = _parse_native_chart(chart, mod)
            self._reset_state()
            self._closed = False
            self._seed = seed_int
            self._chart = chart
            self._scenario = scenario
            self._notes = notes
            self._timing = timing
            self._offset_ms = offset
            self._mod = mod
            self._judge = self._default_judge
            self._scenario["judge"] = self._judge
            self._scenario["native_summary"] = summary
            return self.observe()

    @property
    def done(self) -> bool:
        return bool(self._done)

    @property
    def score(self) -> float:
        with self._lock:
            return float(self._weighted_accuracy())

    @property
    def ticks(self) -> int:
        return int(self._ticks)

    def _weighted_accuracy(self) -> float:
        if self._quality_count <= 0:
            return 0.0
        return max(0.0, min(1.0, self._quality_sum / self._quality_count))

    def _completion(self) -> float:
        total = len(self._notes) + sum(1 for note in self._notes if note.is_hold)
        if total <= 0:
            return 1.0
        resolved = sum(1 for note in self._notes if note.head_judged)
        resolved += sum(1 for note in self._notes if note.is_hold and note.tail_judged)
        return max(0.0, min(1.0, resolved / total))

    def _rm_perfect_total(self) -> float:
        multiplier = _RM_JUDGE_MULTIPLIER[self._judge] * _RM_MOD_MULTIPLIER.get(self._mod, 1.0)
        total = 0.0
        for combo_after in range(1, len(self._notes) + 1):
            if combo_after <= 20:
                base = 200.0
            elif combo_after <= 50:
                base = 332.0
            elif combo_after <= 100:
                base = 466.0
            else:
                base = 600.0
            total += base * multiplier
        return total

    def _judgment_for_offset(self, offset_ms: float) -> str:
        magnitude = abs(float(offset_ms))
        big_p, p1, p2, p3, good = _WINDOWS[self._judge]
        if magnitude <= big_p:
            return "Big P"
        if magnitude <= p1:
            return "p1"
        if magnitude <= p2:
            return "p2"
        if magnitude <= p3:
            return "p3"
        if magnitude <= good:
            return "G"
        return "M"

    def _append_judgment(
        self,
        *,
        note: _Note,
        judgment: str,
        offset_ms: float,
        at_ms: float,
        kind: str,
        tail: bool = False,
    ) -> None:
        quality = _JUDGMENT_WEIGHTS.get(judgment, 0.0)
        self._quality_sum += quality
        self._quality_count += 1
        item = {
            "note_id": int(note.note_id),
            "column": int(note.column),
            "raw_column": int(note.raw_column),
            "kind": kind,
            "judgment": judgment,
            "offset_ms": _clean_float(offset_ms),
            "at_ms": _clean_float(at_ms),
            "combo_after": int(self._combo),
        }
        if tail:
            self._tail_count[judgment] = self._tail_count.get(judgment, 0) + 1
        else:
            self._judgment_count[judgment] = self._judgment_count.get(judgment, 0) + 1
            if judgment == "M":
                self._misses += 1
                self._combo = 0
                item["combo_after"] = 0
            else:
                self._combo += 1
                self._max_combo = max(self._max_combo, self._combo)
                combo_after = self._combo
                if combo_after <= 20:
                    base = 200.0
                elif combo_after <= 50:
                    base = 332.0
                elif combo_after <= 100:
                    base = 466.0
                else:
                    base = 600.0
                self._rm_proxy += (
                    base
                    * _RM_JUDGE_MULTIPLIER[self._judge]
                    * _RM_MOD_MULTIPLIER.get(self._mod, 1.0)
                    * quality
                )
                item["combo_after"] = int(self._combo)
        self._last_judgments.append(item)
        del self._last_judgments[:-20]

    def _resolve_head(self, note: _Note, at_ms: float) -> None:
        if note.head_judged:
            return
        offset = at_ms - note.start_ms
        judgment = self._judgment_for_offset(offset)
        note.head_judged = True
        note.head_judgment = judgment
        note.head_offset_ms = offset
        self._append_judgment(note=note, judgment=judgment, offset_ms=offset, at_ms=at_ms, kind="hold_head" if note.is_hold else "tap")
        if note.is_hold and judgment != "M":
            note.active = True
        elif note.is_hold:
            # A missed head cannot later receive a release, but it still
            # resolves the hold as a broken object so the chart can finish.
            self._break_hold(note, at_ms, "head_missed")

    def _break_hold(self, note: _Note, at_ms: float, reason: str) -> None:
        if not note.is_hold or note.tail_judged:
            return
        note.tail_judged = True
        note.active = False
        note.tail_judgment = "M"
        note.tail_offset_ms = at_ms - float(note.end_ms or at_ms)
        self._broken_holds += 1
        self._quality_count += 1
        item = {
            "note_id": int(note.note_id),
            "column": int(note.column),
            "raw_column": int(note.raw_column),
            "kind": "hold_tail",
            "judgment": "broken_hold",
            "reason": reason,
            "offset_ms": _clean_float(note.tail_offset_ms),
            "at_ms": _clean_float(at_ms),
            "combo_after": int(self._combo),
        }
        self._tail_count["M"] += 1
        self._last_judgments.append(item)
        del self._last_judgments[:-20]

    def _resolve_tail(self, note: _Note, at_ms: float) -> None:
        if not note.is_hold or note.tail_judged:
            return
        end_ms = float(note.end_ms or at_ms)
        if at_ms < end_ms:
            self._break_hold(note, at_ms, "released_before_end")
            return
        offset = at_ms - end_ms
        judgment = self._judgment_for_offset(offset)
        note.tail_judged = True
        note.active = False
        note.tail_judgment = judgment
        note.tail_offset_ms = offset
        self._append_judgment(note=note, judgment=judgment, offset_ms=offset, at_ms=at_ms, kind="hold_tail", tail=True)

    def _resolve_misses_through(self, at_ms: float) -> None:
        deadline = _WINDOWS[self._judge][-1]
        for note in self._notes:
            if not note.head_judged and at_ms > note.start_ms + deadline:
                self._resolve_head(note, note.start_ms + deadline + 0.001)
            if note.active and note.end_ms is not None and at_ms > note.end_ms + deadline:
                self._break_hold(note, note.end_ms + deadline + 0.001, "release_timeout")

    def _find_head(self, column: int, at_ms: float) -> _Note | None:
        candidates = [
            note for note in self._notes
            if note.column == column and not note.head_judged
            and abs(at_ms - note.start_ms) <= _WINDOWS[self._judge][-1]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda note: (abs(at_ms - note.start_ms), note.start_ms, note.note_id))

    def _active_hold(self, column: int) -> _Note | None:
        active = [note for note in self._notes if note.active and note.column == column]
        return min(active, key=lambda note: note.end_ms or 0.0) if active else None

    def _apply_event(self, event: Mapping[str, Any], default_time: float) -> float:
        requested_time = event.get("at_ms", default_time)
        if not _finite(requested_time):
            self._last_error = "ignored non-finite event timestamp"
            at_ms = default_time
        else:
            at_ms = float(requested_time)
            if at_ms < default_time:
                self._last_error = "late event timestamp was applied at the current clock"
                at_ms = default_time
        self._resolve_misses_through(at_ms)
        event_type = str(event.get("type", "")).lower()
        columns = event.get("column")
        if isinstance(columns, int) and not isinstance(columns, bool):
            column_list = [columns]
        elif isinstance(columns, (list, tuple)):
            column_list = list(columns)
        elif isinstance(columns, str):
            column_list = [part for part in columns.replace(",", "+").split("+") if part]
        else:
            column_list = []
        for column in column_list:
            if isinstance(column, bool):
                continue
            try:
                column_int = int(column)
            except (TypeError, ValueError):
                continue
            if not 0 <= column_int < int(self._scenario.get("columns", 0)):
                self._last_error = f"ignored column {column_int} outside the active chart"
                continue
            if event_type == "press":
                note = self._find_head(column_int, at_ms)
                if note is not None:
                    self._resolve_head(note, at_ms)
            elif event_type == "release":
                note = self._active_hold(column_int)
                if note is not None:
                    self._resolve_tail(note, at_ms)
            else:
                self._last_error = f"ignored unknown event type {event_type!r}"
        return at_ms

    def _parse_string_action(self, action: str, now: float) -> tuple[list[dict[str, Any]], float | None]:
        text = action.strip()
        if not text or text in {"noop", "wait", "advance"}:
            return [], None
        parts = text.replace(",", " ").split()
        verb = parts[0].lower() if parts else ""
        if verb == "advance":
            if len(parts) > 1:
                try:
                    target = float(parts[1])
                except (TypeError, ValueError):
                    target = math.nan
                if math.isfinite(target):
                    return [], target
                self._last_error = f"ignored non-finite advance target in {action!r}"
            return [], None
        if verb not in {"press", "release"} or len(parts) < 2:
            self._last_error = f"ignored malformed action {action!r}"
            return [], None
        raw_columns = parts[1].replace("+", ",").split(",")
        columns: list[int] = []
        for raw in raw_columns:
            try:
                columns.append(int(raw))
            except ValueError:
                self._last_error = f"ignored malformed column in {action!r}"
        if not columns:
            return [], None
        return [{"type": verb, "column": columns, "at_ms": now}], None

    def _parse_action(self, action: Any, now: float) -> tuple[list[dict[str, Any]], float | None]:
        if isinstance(action, str):
            return self._parse_string_action(action, now)
        if isinstance(action, Mapping):
            action_type = str(action.get("type", "")).lower()
            if action_type == "batch":
                raw_events = action.get("events", [])
                if not isinstance(raw_events, (list, tuple)):
                    self._last_error = "ignored batch with non-list events"
                    return [], None
                events: list[dict[str, Any]] = []
                for raw in raw_events:
                    if isinstance(raw, Mapping):
                        events.append(dict(raw))
                    else:
                        self._last_error = "ignored non-object event in batch"
                events.sort(key=lambda event: float(event.get("at_ms", now)) if _finite(event.get("at_ms", now)) else now)
                return events, None
            if action_type == "advance":
                target = action.get("to_ms", action.get("at_ms"))
                return [], float(target) if _finite(target) else None
            if action_type in {"press", "release"}:
                return [dict(action)], None
            self._last_error = f"ignored malformed action type {action_type!r}"
            return [], None
        if isinstance(action, (list, tuple)):
            events: list[dict[str, Any]] = []
            for item in action:
                sub_events, sub_target = self._parse_action(item, now)
                events.extend(sub_events)
                if sub_target is not None:
                    self._future_clock_target = max(self._future_clock_target or now, sub_target)
            return events, None
        self._last_error = f"ignored malformed action of type {type(action).__name__}"
        return [], None

    def tick(self, action: Any) -> None:
        """Apply one current action and advance the logical chart clock."""
        with self._lock:
            if self._closed:
                return
            if self._done:
                self._ticks += 1
                return
            self._last_error = None
            now = float(self._time_ms)
            self._future_clock_target = None
            try:
                events, target = self._parse_action(action, now)
                self._last_action = str(action)[:120]
                for event in events:
                    self._apply_event(event, now)
                if target is not None and _finite(target):
                    self._future_clock_target = float(target)
                next_time = now + self.tick_s * 1000.0
                if self._future_clock_target is not None and self._future_clock_target > next_time:
                    next_time = self._future_clock_target
                self._time_ms = max(now, next_time)
                self._resolve_misses_through(self._time_ms)
                self._update_done()
            except Exception as exc:  # malformed external input must not kill a trial
                self._last_error = f"internal action handling error: {type(exc).__name__}: {exc}"[:300]
                self._time_ms = now + self.tick_s * 1000.0
                self._resolve_misses_through(self._time_ms)
                self._update_done()
            finally:
                self._ticks += 1

    def _update_done(self) -> None:
        if not self._notes:
            self._done = True
            return
        all_heads = all(note.head_judged for note in self._notes)
        all_tails = all(not note.is_hold or note.tail_judged for note in self._notes)
        last_end = max((note.end_ms if note.end_ms is not None else note.start_ms) for note in self._notes)
        self._done = bool(all_heads and all_tails and self._time_ms >= last_end + _WINDOWS[self._judge][-1])

    def observe(self) -> dict[str, Any]:
        with self._lock:
            horizon = self._time_ms + 2200.0
            upcoming: list[dict[str, Any]] = []
            for note in self._notes:
                if note.head_judged or note.start_ms > horizon:
                    continue
                upcoming.append({
                    "id": int(note.note_id),
                    "column": int(note.column),
                    "raw_column": int(note.raw_column),
                    "start_beat": _clean_float(note.start_beat),
                    "end_beat": None if note.end_beat is None else _clean_float(note.end_beat),
                    "start_ms": _clean_float(note.start_ms),
                    "end_ms": None if note.end_ms is None else _clean_float(note.end_ms),
                    "kind": note.kind,
                    "pattern": note.pattern,
                    "motion": "falling_toward_receptor",
                    "bpm": _clean_float(note.bpm),
                    "group_id": int(note.group_id),
                    "group_size": int(note.group_size),
                    "group_columns": list(note.group_columns),
                    "in_ms": _clean_float(note.start_ms - self._time_ms),
                })
                if len(upcoming) >= 64:
                    break
            held = sorted(note.column for note in self._notes if note.active)
            active_holds = [
                {
                    "id": int(note.note_id),
                    "column": int(note.column),
                    "raw_column": int(note.raw_column),
                    "kind": "hold",
                    "pattern": note.pattern,
                    "motion": "held_at_receptor",
                    "head_judgment": note.head_judgment,
                    "end_ms": _clean_float(note.end_ms or self._time_ms),
                    "remaining_ms": _clean_float((note.end_ms or self._time_ms) - self._time_ms),
                }
                for note in self._notes if note.active
            ]
            total_heads = len(self._notes)
            judged_heads = sum(1 for note in self._notes if note.head_judged)
            pending_holds = sum(1 for note in self._notes if note.is_hold and not note.tail_judged)
            clear = bool(self._done and self._misses == 0 and self._broken_holds == 0)
            stats = {
                "score": _clean_float(self._weighted_accuracy()),
                "accuracy": _clean_float(self._weighted_accuracy()),
                "judgment_histogram": dict(self._judgment_count),
                "tail_judgment_histogram": dict(self._tail_count),
                "misses": int(self._misses),
                "broken_holds": int(self._broken_holds),
                "combo": int(self._combo),
                "max_combo": int(self._max_combo),
                "completion": _clean_float(self._completion()),
                "judged_heads": int(judged_heads),
                "total_heads": int(total_heads),
                "pending_heads": int(total_heads - judged_heads),
                "pending_holds": int(pending_holds),
                "clear": clear,
                "terminal_status": "cleared" if clear else ("complete_with_errors" if self._done else "playing"),
                "rm_proxy_2025": _clean_float(self._rm_proxy),
                "rm_proxy_normalized": _clean_float(
                    self._rm_proxy / self._rm_perfect_total() if self._rm_perfect_total() else 0.0
                ),
            }
            chart_summary = {
                "id": self._scenario.get("chart_id", f"generated-mc-seed-{self._seed}"),
                "native_mode": 0,
                "columns": int(self._scenario.get("columns", 0)),
                "difficulty": float(self._scenario.get("difficulty", 0.0)),
                "pattern_family": self._scenario.get("family", "unknown"),
                "bpm_segments": [
                    {"start_beat": _clean_float(point.beat), "bpm": _clean_float(point.bpm)}
                    for point in self._timing
                ],
                "offset_ms": _clean_float(self._offset_ms),
                "snap_resolution": self._scenario.get("snap_resolution", "unknown"),
                "mod": self._mod,
            }
            windows = _WINDOWS[self._judge]
            column_count = int(self._scenario.get("columns", 0))
            result: dict[str, Any] = {
                "episode": {
                    "seed": int(self._seed),
                    "mode": "key",
                    "columns": column_count,
                    "judge": self._judge,
                    "mod": self._mod,
                    "time_ms": _clean_float(self._time_ms),
                    "tick": int(self._ticks),
                    "tick_s": _clean_float(self.tick_s),
                    "judgment_windows_ms": {
                        "Big P": _clean_float(windows[0]),
                        "p1": _clean_float(windows[1]),
                        "p2": _clean_float(windows[2]),
                        "p3": _clean_float(windows[3]),
                        "G": _clean_float(windows[4]),
                        "miss_after": _clean_float(windows[4]),
                    },
                    "done": bool(self._done),
                },
                "controls": {
                    "column_indexing": "zero_based_displayed",
                    "column_mapping": [
                        {"column": column, "press": f"press {column}", "release": f"release {column}"}
                        for column in range(column_count)
                    ],
                    "resources": {"modeled": False, "available": []},
                },
                "chart": chart_summary,
                "upcoming": upcoming,
                "held_columns": held,
                "active_holds": active_holds,
                "last_judgments": list(self._last_judgments),
                "stats": stats,
            }
            if self._last_error:
                result["action_feedback"] = {"error": self._last_error, "last_action": self._last_action}
            return result

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            perfect = self._rm_perfect_total()
            return {
                "score": float(self._weighted_accuracy()),
                "weighted_accuracy": float(self._weighted_accuracy()),
                "completion": float(self._completion()),
                "judged_heads": float(sum(1 for note in self._notes if note.head_judged)),
                "total_heads": float(len(self._notes)),
                "misses": float(self._misses),
                "broken_holds": float(self._broken_holds),
                "max_combo": float(self._max_combo),
                "rm_proxy_2025": float(self._rm_proxy),
                "rm_proxy_normalized": float(self._rm_proxy / perfect if perfect else 0.0),
            }

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "valid_trial": True,
                "invalid_reason": None,
                "interface": "local deterministic native .mc Key emulator",
                "seed": int(self._seed),
                "chart_id": self._scenario.get("chart_id", ""),
                "score_model": "weighted judgment accuracy; rm_proxy_2025 is RM-only reference math",
                "fidelity_limits": [
                    "No official Malody client, rendering, audio, input-device jitter, or account is used.",
                    "Public timing and combo values are an RM-mode proxy, not verified native Key scoring.",
                ],
            }

    def describe(self) -> str:
        return (
            "Malody Key mode, offline native .mc emulator. Objective: clear the generated seeded chart "
            "with accurate presses, simultaneous chord presses, and released holds. The observation is "
            "strict structured JSON: episode clock/seed/columns/judge/mod/tick cadence, explicit judgment "
            "windows, controls.column_mapping, and resources (this Key emulator models no lives or power "
            "items); chart metadata has ID, difficulty, BPM segments, signed offset, snap grid, and mod. "
            "The next 2.2 seconds (up to 64) of notes carries raw/effective lane, fractional beat and "
            "millisecond timers, kind, decoded pattern (tap/stream/jack/chord/anchor/hold), falling motion, "
            "BPM, exact-start group ID/size/columns, and time until the receptor. held_columns and "
            "active_holds carry hold motion, head judgment, tail timer, and remaining time. last_judgments "
            "has signed offsets and raw/effective columns; stats has pending counts, judgment histograms, "
            "combo, misses, broken holds, accuracy, completion, clear/terminal status, and score. The "
            "logical real-time cadence is tick_s seconds (default 40 ms); each tick applies the supplied "
            "current action and then advances the clock, with no wall-clock sleep inside the adaptor. "
            "Action grammar is 'noop', 'press N', 'release N', or 'press N+M' for a chord, where N is "
            "the zero-based displayed column. JSON press/release actions may add at_ms, and JSON batches "
            "contain timestamp-sorted events; advance may use to_ms. A press within the active judge window "
            "resolves the nearest unjudged head. C-judge windows are Big P 40 ms, p1 56, p2 81, p3 111, "
            "G 200, then M; the selected chart judge is reported. Holds must be released at or after their "
            "end, otherwise they are broken. Score is weighted accuracy (Big P 1.00, p1 .90, p2 .85, "
            "p3 .80, G .40, M 0); score_breakdown and observations also report misses, broken holds, max "
            "combo, completion, and the explicitly labelled rm_proxy_2025 using the documented 200/332/466/600 "
            "RM combo bases and multipliers. That proxy is not an officially verified Key score. Seed "
            "families cover 4K-7K, taps, streams, jacks, chords, anchors, BPM changes, offsets, snap grids, "
            "holds/final releases, and none/Flip/Const mods."
        )

    def close(self) -> None:
        """No external resources are owned; make the lifecycle explicitly idempotent."""
        with self._lock:
            self._closed = True


def make_game(**kwargs: Any) -> MalodyKeyGame:
    """Construct a fresh adaptor instance for one evaluator episode."""
    return MalodyKeyGame(**kwargs)
