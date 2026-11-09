"""Headless Quaver Keys adaptor.

This module implements the selected local-simulator interface from the study.  It
uses native ``.qua`` YAML objects and the Standard* scoring constants, but keeps
the song clock virtual so a headless evaluator does not need a display, audio,
Steam, or a network connection.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shlex
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from metaharness.games.base import ActionSpec


TIER = 2
DETERMINISTIC = True
# The two groups deliberately contain the same five scenario families.  A held
# out seed changes the seeded lane/layout stream rather than selecting a new
# family which never appeared during training.
SEEDS = {
    "train": list(range(1001, 1017)),
    "heldout": list(range(2001, 2017)),
}
EVAL_KWARGS: dict[str, Any] = {}

_RULESET = "standard-star-v1"
_JUDGEMENTS = ("Marvelous", "Perfect", "Great", "Good", "Okay", "Miss")
_PRESS_WINDOWS = {
    "Marvelous": 18.0,
    "Perfect": 43.0,
    "Great": 76.0,
    "Good": 106.0,
    "Okay": 127.0,
    "Miss": 164.0,
}
_ACCURACY_WEIGHTS = {
    "Marvelous": 100.0,
    "Perfect": 98.25,
    "Great": 65.0,
    "Good": 25.0,
    "Okay": -100.0,
    "Miss": -50.0,
}
_SCORE_WEIGHTS = {
    "Marvelous": 100,
    "Perfect": 50,
    "Great": 25,
    "Good": 10,
    "Okay": 5,
    "Miss": 0,
}
_HEALTH_WEIGHTS = {
    "Marvelous": 0.5,
    "Perfect": 0.4,
    "Great": 0.2,
    "Good": -3.0,
    "Okay": -4.5,
    "Miss": -6.0,
}


@dataclass
class _Note:
    note_id: int
    lane: int
    start_ms: int
    end_ms: int
    start_judged: bool = False
    start_missed: bool = False
    release_resolved: bool = False
    held: bool = False

    @property
    def is_long(self) -> bool:
        return self.end_ms > self.start_ms

    @property
    def resolved(self) -> bool:
        return self.start_missed or (
            self.start_judged and (not self.is_long or self.release_resolved)
        )


class ChartFormatError(ValueError):
    """Raised when a chart uses a feature this simulator cannot score safely."""


class QuaverLocalGame:
    """A deterministic, virtual-clock implementation of RealtimeGame."""

    name = "Quaver Keys (local .qua simulator)"
    tick_s = 0.018

    def __init__(self, **kwargs: Any) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self._runtime_dir = Path(__file__).resolve().parent / "runtime" / "charts"
        self._runtime_dir.mkdir(parents=True, exist_ok=True)

        self._ruleset = str(
            kwargs.get("ruleset", os.environ.get("QUAVER_RULESET", _RULESET))
        )
        if self._ruleset != _RULESET:
            raise ValueError(
                f"unsupported QUAVER_RULESET {self._ruleset!r}; only {_RULESET!r} is implemented"
            )

        root_value = kwargs.get("chart_root", os.environ.get("QUAVER_CHART_ROOT", ""))
        self._chart_root = self._resolve_chart_root(root_value)
        self._external_charts = self._discover_external_charts(self._chart_root)

        # Modifiers are intentionally opt-in.  Scenario generation has its own
        # seeded BPM/rate and offset, but does not silently apply player mods.
        self._requested_mode = kwargs.get("mode")
        if self._requested_mode is not None and self._requested_mode not in {
            "Keys4",
            "Keys7",
            "4K",
            "7K",
        }:
            raise ValueError("mode must be Keys4/Keys7 (or 4K/7K)")
        self._requested_mirror = bool(kwargs.get("mirror", False))
        self._requested_rate = kwargs.get("rate", None)
        if self._requested_rate is not None:
            self._requested_rate = float(self._requested_rate)
            if not math.isfinite(self._requested_rate) or self._requested_rate <= 0:
                raise ValueError("rate must be a positive finite number")
        self._requested_offset = kwargs.get("timing_offset_ms", None)
        if self._requested_offset is not None:
            self._requested_offset = float(self._requested_offset)
            if not math.isfinite(self._requested_offset):
                raise ValueError("timing_offset_ms must be finite")
        self._requested_windows = self._parse_custom_windows(
            kwargs.get("custom_window_ms", kwargs.get("timing_windows", None))
        )
        self._lookahead_ms = int(kwargs.get("lookahead_ms", 2000))
        if self._lookahead_ms < 500 or self._lookahead_ms > 5000:
            raise ValueError("lookahead_ms must be between 500 and 5000")

        names = ["noop"]
        names.extend(f"press {lane}" for lane in range(1, 8))
        names.extend(f"release {lane}" for lane in range(1, 8))
        self.action_spec = ActionSpec(
            names=names,
            noop="noop",
            descriptions={
                "noop": "Apply no lane event for this 18 ms tick.",
                **{
                    f"press {lane}": f"Press lane {lane} at the current tick time (or use press {lane} <time_ms>)."
                    for lane in range(1, 8)
                },
                **{
                    f"release {lane}": f"Release a held long note in lane {lane} at the current tick time (or use release {lane} <time_ms>)."
                    for lane in range(1, 8)
                },
            },
            notes=(
                "Parameterized grammar: 'press <lane> [time_ms]' or 'release <lane> [time_ms]'; "
                "a JSON event/list or 'events <json>' may batch timestamped events. "
                "Timestamped events are queued ahead of the virtual clock. Lanes outside the "
                "current 4K/7K chart are ignored and reported in the next observation."
            ),
        )

        self._seed: int | None = None
        self._scenario: dict[str, Any] = {}
        self._chart_hash = ""
        self._chart_path: str | None = None
        self._notes: list[_Note] = []
        self._note_patterns: dict[int, dict[str, Any]] = {}
        self._timing_points: list[dict[str, float]] = []
        self._lane_count = 4
        self._mode = "Keys4"
        self._windows = dict(_PRESS_WINDOWS)
        self._release_multiplier = 1.5
        self._ln_miss_judgement = "Good"
        self._now_ms = 0
        self._ticks = 0
        self._event_sequence = 0
        self._event_queue: list[tuple[int, int, str, int]] = []
        self._held_lanes: set[int] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=16)
        self._timing_errors: list[float] = []
        self._auto_miss_count = 0
        self._counts = {name: 0 for name in _JUDGEMENTS}
        self._accuracy_numerator = 0.0
        self._score_count = 0
        self._multiplier_count = 0
        self._multiplier_index = 0
        self._combo = 0
        self._max_combo = 0
        self._health = 100.0
        self._total_judgements = 0
        self._summed_score = 1
        self._done = False
        self._last_action_error: str | None = None
        self._last_action = ""
        self._finish_time_ms = 0

    # ------------------------------------------------------------------
    # Public RealtimeGame surface
    # ------------------------------------------------------------------
    def reset(self, seed: int) -> dict[str, Any]:
        with self._lock:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an int")
            if self._closed:
                # close() owns no external process, but a closed object is still
                # not a usable episode.  This makes lifecycle errors explicit.
                raise RuntimeError("cannot reset a closed QuaverLocalGame")

            self._seed = int(seed)
            if self._external_charts:
                raw_chart, chart_source = self._select_external_chart(self._seed)
                chart, scenario = self._prepare_external_chart(raw_chart, chart_source)
            else:
                chart, scenario = self._generate_seed_chart(self._seed)

            chart = self._apply_requested_modifiers(chart, scenario)
            parsed = self._parse_chart(chart)
            effective = self._canonical_chart(parsed["raw"])
            self._chart_hash = hashlib.sha256(effective).hexdigest()
            self._write_generated_chart_if_needed(chart, self._seed)

            self._mode = parsed["mode"]
            self._lane_count = parsed["lane_count"]
            self._timing_points = parsed["timing_points"]
            self._notes = [
                _Note(
                    note_id=i + 1,
                    lane=int(item["Lane"]),
                    start_ms=int(item["StartTime"]),
                    end_ms=int(item["EndTime"]),
                )
                for i, item in enumerate(parsed["objects"])
            ]
            self._notes.sort(key=lambda n: (n.start_ms, n.lane, n.note_id))
            self._note_patterns = self._classify_note_patterns(self._notes)
            self._scenario = dict(scenario)
            self._scenario.update(
                {
                    "seed": self._seed,
                    "mode": self._mode,
                    "lane_count": self._lane_count,
                    "chart_hash": self._chart_hash,
                    "chart_source": str(chart.get("_source", "generated")),
                    "ruleset": self._ruleset,
                }
            )
            self._configure_windows(chart)

            self._now_ms = 0
            self._ticks = 0
            self._event_sequence = 0
            self._event_queue = []
            self._held_lanes = set()
            self._recent.clear()
            self._timing_errors = []
            self._auto_miss_count = 0
            self._counts = {name: 0 for name in _JUDGEMENTS}
            self._accuracy_numerator = 0.0
            self._score_count = 0
            self._multiplier_count = 0
            self._multiplier_index = 0
            self._combo = 0
            self._max_combo = 0
            self._health = 100.0
            self._total_judgements = sum(2 if n.is_long else 1 for n in self._notes)
            self._summed_score = self._calculate_summed_score(self._total_judgements)
            self._done = False
            self._last_action_error = None
            self._last_action = "reset"
            last_end = max((n.end_ms if n.is_long else n.start_ms for n in self._notes), default=0)
            self._finish_time_ms = int(
                last_end + max(self._windows["Miss"], self._release_window("Okay")) + self.tick_s * 1000
            )
            return self._observe_unlocked()

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return self._observe_unlocked()

    def tick(self, action: Any) -> None:
        """Apply one action at the current virtual time and advance one cadence.

        A plain press/release is timestamped at the beginning of this tick.  A
        timestamped event is placed in a small deterministic queue, allowing a
        scheduler to submit a lookahead batch without asking a model to sleep
        for individual notes.
        """
        with self._lock:
            self._last_action_error = None
            self._last_action = self._action_label(action)
            events = self._parse_action(action)
            for event_type, lane, event_time, explicit_time in events:
                if lane < 1 or lane > self._lane_count:
                    self._record_action_error(
                        f"lane {lane} is outside the current {self._mode} chart"
                    )
                    continue
                if event_time < self._now_ms:
                    self._record_action_error(
                        f"late {event_type} lane {lane} at {event_time} ms; clock is {self._now_ms} ms"
                    )
                    continue
                if not explicit_time:
                    event_time = self._now_ms
                self._event_sequence += 1
                self._event_queue.append(
                    (int(event_time), self._event_sequence, event_type, int(lane))
                )
            self._event_queue.sort(key=lambda item: (item[0], item[1]))

            next_time = self._now_ms + int(round(self.tick_s * 1000.0))
            self._advance_to(next_time)
            self._ticks += 1
            self._update_done()

    # A small convenience for deterministic chart schedulers.  RealtimeGame
    # does not require it, but it makes the timestamped protocol easy to test.
    def step(self, action: Any) -> dict[str, Any]:
        self.tick(action)
        return self.observe()

    def run_until(self, time_ms: int | float) -> dict[str, Any]:
        with self._lock:
            if not self._notes:
                return self._observe_unlocked()
            target = int(time_ms)
            if target < self._now_ms:
                self._record_action_error(
                    f"run_until target {target} ms is before clock {self._now_ms} ms"
                )
                return self._observe_unlocked()
            self._advance_to(target)
            self._update_done()
            return self._observe_unlocked()

    def finish(self) -> dict[str, Any]:
        with self._lock:
            self._advance_to(self._finish_time_ms + 1)
            self._update_done(force=True)
            return self._observe_unlocked()

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done)

    @property
    def score(self) -> float:
        with self._lock:
            return float(self._normalized_score())

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "valid_trial": True,
                "invalid_reason": None,
                "fidelity": "local_sim",
                "fidelity_version": "quaver-local-sim-v1",
                "seed": self._seed,
                "chart_hash": self._chart_hash,
                "chart_path": self._chart_path,
                "ruleset": self._ruleset,
                "mode": self._mode,
                "lane_count": self._lane_count,
                "scenario": dict(self._scenario),
                "modifiers": self._modifiers_for_info(),
                "total_judgements": self._total_judgements,
                "ticks": self._ticks,
            }

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            total = max(1, self._total_judgements)
            return {
                "accuracy_fraction": float(self._accuracy_percent() / 100.0),
                "normalized_score_fraction": float(self._normalized_score() / 1_000_000.0),
                "combo_fraction": float(min(self._max_combo / total, 1.0)),
                "health_fraction": float(self._health / 100.0),
                "survival": 1.0 if self._health > 0 else 0.0,
            }

    def close(self) -> None:
        """Release this instance; no server, process, or connection is started."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._event_queue.clear()
            self._held_lanes.clear()

    def describe(self) -> str:
        return (
            "Quaver Keys is a real-time vertical-scrolling rhythm chart. The objective is to "
            "finish the generated or supplied .qua map with the highest timing accuracy, "
            "normalized score, and combo. Each tap has one judgement; each long note has a "
            "press and a release judgement. The local simulator uses the official Standard* "
            "windows in milliseconds: Marvelous ±18, Perfect ±43, Great ±76, Good ±106, "
            "Okay ±127, and Miss through ±164 (outside that window is an automatic miss). Release windows are 1.5 times those "
            "press windows; an early release outside them is a Miss, while an unreleased long "
            "note is converted to the configured LN-miss judgement (Good by default). "
            "Judgement weights are Marvelous 100, Perfect 98.25, Great 65, Good 25, "
            "Okay -100, Miss -50. Accuracy is the clamped weighted mean; score is the "
            "official-style normalized score out of 1,000,000 and combo breaks on Miss. "
            "Performance rating is difficulty rating * (accuracy / 98)^6.\n\n"
            "This is RealtimeGame tier 2: every tick is 18 ms and the virtual song clock advances "
            "whether a new decision is supplied. Plain actions are 'noop', 'press <lane>', or "
            "'release <lane>'; an optional integer after the lane is a timestamp in milliseconds. "
            "For lookahead scheduling, pass one event dict/list or a string beginning with "
            "'events ' followed by JSON, for example events [{\"type\":\"press\",\"lane\":2,"
            "\"time_ms\":1000}]. Press selects the earliest pending object in that lane; release "
            "selects the earliest held long note. Duplicate or impossible events are ignored and "
            "reported in the next observation.\n\n"
            "Observations expose the fair visible window (clock - 164 ms through clock + 2000 ms), "
            "with note id, lane, kind, vertical motion, start/end time, relative timers, "
            "simultaneous group, decoded tap/stream/jack/chord/chordjack pattern, and "
            "pending/held/judged/missed state. The top-level timing object gives current BPM, "
            "beat period, rate, offset, upcoming timing changes, and the timestamp-authoritative "
            "vertical movement model. lane_actions maps every active lane to its press/release "
            "action, and input_mapping identifies the local virtual lane interface; there are "
            "no physical keys in this headless backend. Observations also expose held lanes, resources, chart progress, recent "
            "judgement errors, aggregate player-action early/late timing-error statistics, variance, "
            "recent-window drift and histogram, automatic expiry count, full "
            "judgement counts, accuracy, score, grade, combo, max combo, health, terminal state, "
            "seed, chart hash, ruleset, scenario metadata, and explicit modifiers. The hidden "
            "remainder of the chart is not included. The built-in charts vary mode, length, BPM/rate, "
            "timing offset, density, streams, jacks, chords, chordjacks, and long-note patterns by seed; external "
            "charts must be valid supported .qua YAML. Mines, timing groups, and unsupported future "
            "chart versions are rejected rather than scored as ordinary notes. This backend is an "
            "offline local_sim: it omits Quaver rendering, audio, OS input latency, Steam/network "
            "validation, and other client-side effects."
        )

    # ------------------------------------------------------------------
    # Chart discovery, generation, and validation
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_chart_root(value: Any) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        root = Path(str(value)).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"QUAVER_CHART_ROOT is not a directory: {root}")
        return root

    @staticmethod
    def _discover_external_charts(root: Path | None) -> list[Path]:
        if root is None:
            return []
        out: list[Path] = []
        for path in sorted(root.rglob("*.qua")):
            resolved = path.resolve()
            try:
                if os.path.commonpath((str(root), str(resolved))) != str(root):
                    continue
            except ValueError:
                continue
            if resolved.is_file():
                out.append(resolved)
        if not out:
            raise FileNotFoundError(f"QUAVER_CHART_ROOT contains no .qua charts: {root}")
        return out

    def _select_external_chart(self, seed: int) -> tuple[dict[str, Any], Path]:
        path = self._external_charts[abs(seed) % len(self._external_charts)]
        try:
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except Exception as exc:  # noqa: BLE001
            raise ChartFormatError(f"cannot parse chart {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ChartFormatError(f"chart {path} must contain a YAML mapping")
        raw = dict(raw)
        raw["_source"] = str(path)
        return raw, path

    def _prepare_external_chart(
        self, raw: dict[str, Any], path: Path
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        objects = raw.get("HitObjects")
        mode = str(raw.get("Mode", ""))
        if mode not in {"Keys4", "Keys7"}:
            raise ChartFormatError(f"unsupported chart mode {mode!r} in {path}")
        if not isinstance(objects, list) or not objects:
            raise ChartFormatError(f"chart {path} has no HitObjects")
        bpm = 120.0
        points = raw.get("TimingPoints", [])
        if not isinstance(points, list):
            raise ChartFormatError("TimingPoints must be a list")
        if points:
            first = points[0]
            if isinstance(first, dict) and isinstance(first.get("Bpm"), (int, float)):
                bpm = float(first["Bpm"])
        note_times = [int(item.get("StartTime", 0)) for item in objects if isinstance(item, dict)]
        end_times = [
            int(item.get("EndTime", 0))
            for item in objects
            if isinstance(item, dict) and int(item.get("EndTime", 0)) > 0
        ]
        last = max(note_times + end_times)
        rating_value = raw.get("DifficultyRating", raw.get("Difficulty", 0.0))
        try:
            rating = float(rating_value)
        except (TypeError, ValueError):
            rating = 0.0
        if not math.isfinite(rating) or rating <= 0:
            rating = self._estimate_difficulty(len(objects), last, mode)
        scenario = {
            "family": "external_chart",
            "difficulty_rating": round(rating, 3),
            "bpm": round(bpm, 3),
            "rate": 1.0,
            "timing_offset_ms": 0,
            "chart_length_ms": last,
            "pattern_mix": self._classify_pattern_mix(objects, int(mode[-1])),
            "source": str(path),
        }
        return raw, scenario

    def _generate_seed_chart(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        rng = random.Random(seed)
        family_index = abs(seed) % 5
        family_names = ("calibration", "mixed", "ln-heavy", "dense-chords", "stream-jack")
        family = family_names[family_index]
        if self._requested_mode in {"Keys4", "4K"}:
            mode = "Keys4"
        elif self._requested_mode in {"Keys7", "7K"}:
            mode = "Keys7"
        else:
            # Use the same mode mix in the corresponding train/held-out
            # seed ranges while leaving the PRNG free to vary every layout.
            mode = "Keys7" if (abs(seed) % 10 + family_index) % 3 == 0 else "Keys4"
        lane_count = 7 if mode == "Keys7" else 4

        base_bpm = {
            "calibration": 112,
            "mixed": 164,
            "ln-heavy": 138,
            "dense-chords": 205,
            "stream-jack": 188,
        }[family]
        rate = (0.85, 1.0, 1.15)[rng.randrange(3)]
        bpm = round(base_bpm * rate + rng.uniform(-5.0, 5.0), 3)
        offset = (-24, -12, 0, 12, 24)[rng.randrange(5)]
        if self._requested_offset is not None:
            offset = int(round(self._requested_offset))
        beat_ms = 60_000.0 / bpm
        family_spacing = {
            "calibration": 2.0,
            "mixed": 1.0,
            "ln-heavy": 1.05,
            "dense-chords": 0.55,
            "stream-jack": 0.65,
        }[family]
        spacing = max(24.0, beat_ms / 2.0 * family_spacing)
        count = {
            "calibration": 24,
            "mixed": 52,
            "ln-heavy": 54,
            "dense-chords": 76,
            "stream-jack": 88,
        }[family] + rng.randrange(0, 9)
        first_time = 680 + offset
        objects: list[dict[str, Any]] = []
        active_until = [0] * lane_count
        previous_lanes: list[int] = []
        feature_counts = {"stream": 0, "jack": 0, "chord": 0, "chordjack": 0, "long_note": 0}

        def available_lanes(at: int) -> list[int]:
            return [lane for lane in range(1, lane_count + 1) if active_until[lane - 1] <= at]

        def add_object(at: int, lane: int, long_length: int = 0) -> None:
            if lane < 1 or lane > lane_count or active_until[lane - 1] > at:
                return
            end = at + long_length if long_length > 0 else 0
            objects.append({"StartTime": int(at), "Lane": int(lane), "EndTime": int(end)})
            if end:
                active_until[lane - 1] = end
                feature_counts["long_note"] += 1

        for i in range(count):
            at = int(round(first_time + i * spacing))
            available = available_lanes(at)
            if not available:
                available = list(range(1, lane_count + 1))
            roll = rng.random()
            if family == "calibration":
                chord_size = 2 if i % 11 == 0 and len(available) >= 2 else 1
                make_ln = i in {7, 18}
            elif family == "mixed":
                chord_size = 2 if roll < 0.25 and len(available) >= 2 else 1
                make_ln = roll > 0.78
            elif family == "ln-heavy":
                chord_size = 2 if roll < 0.22 and len(available) >= 2 else 1
                make_ln = roll < 0.63
            elif family == "dense-chords":
                chord_size = 3 if roll < 0.25 and len(available) >= 3 else 2 if len(available) >= 2 else 1
                make_ln = roll < 0.17
            else:  # stream-jack
                chord_size = 2 if i % 8 == 0 and len(available) >= 2 else 1
                make_ln = roll < 0.10

            if family == "stream-jack" and previous_lanes and rng.random() < 0.37:
                chosen = [previous_lanes[0]] if previous_lanes[0] in available else [rng.choice(available)]
                feature_counts["jack"] += 1
            else:
                chosen = rng.sample(available, min(chord_size, len(available)))
            chosen = sorted(set(chosen))
            if len(chosen) >= 2:
                feature_counts["chord"] += 1
                if previous_lanes and set(chosen) & set(previous_lanes):
                    feature_counts["chordjack"] += 1
            elif previous_lanes and chosen and chosen[0] == previous_lanes[0]:
                feature_counts["jack"] += 1
            elif chosen:
                feature_counts["stream"] += 1
            for lane in chosen:
                length = 0
                if make_ln and (lane + i) % 3 != 0:
                    length = int(round(beat_ms * rng.choice((1.5, 2.0, 3.0, 4.0))))
                add_object(at, lane, length)
            previous_lanes = chosen

        objects.sort(key=lambda item: (item["StartTime"], item["Lane"]))
        # The generated chart itself is valid native YAML; the private source
        # marker is removed by _parse_chart and never enters the .qua payload.
        last_time = max((int(o["EndTime"] or o["StartTime"]) for o in objects), default=0)
        density = len(objects) / max(last_time / 1000.0, 1.0)
        difficulty = 1.2 + density * 0.32 + (bpm - 100) / 100.0
        if family == "dense-chords":
            difficulty += 1.0
        if family == "ln-heavy":
            difficulty += 0.25
        difficulty = round(max(1.0, min(9.5, difficulty)), 3)
        total_features = max(1, sum(feature_counts.values()))
        pattern_mix = {
            key: round(value / total_features, 3)
            for key, value in feature_counts.items()
        }
        chart: dict[str, Any] = {
            "QuaVersion": 1,
            "Mode": mode,
            "Title": f"MetaHarness Seed {seed}",
            "Artist": "Offline chart generator",
            "AudioFile": "generated.ogg",
            "BackgroundFile": "",
            "DifficultyName": family,
            "DifficultyRating": difficulty,
            "TimingPoints": [{"StartTime": 0, "Bpm": bpm, "Signature": 4}],
            "HitObjects": objects,
            "_source": "generated",
        }
        scenario = {
            "family": family,
            "difficulty_rating": difficulty,
            "bpm": bpm,
            "rate": round(rate, 3),
            "timing_offset_ms": int(offset),
            "chart_length_ms": int(last_time),
            "density_objects_per_second": round(density, 3),
            "pattern_mix": pattern_mix,
            "source": "generated",
        }
        return chart, scenario

    def _apply_requested_modifiers(
        self, chart: dict[str, Any], scenario: dict[str, Any]
    ) -> dict[str, Any]:
        source = chart.get("_source", "generated")
        chart = {key: value for key, value in chart.items() if key != "_source"}
        chart["HitObjects"] = [dict(obj) for obj in chart.get("HitObjects", [])]
        mode = str(chart.get("Mode", "Keys4"))
        lane_count = 7 if mode == "Keys7" else 4
        if self._requested_mirror:
            for obj in chart["HitObjects"]:
                if isinstance(obj, dict) and isinstance(obj.get("Lane"), int):
                    obj["Lane"] = lane_count + 1 - int(obj["Lane"])
        if self._requested_rate is not None and self._requested_rate != 1.0:
            rate = self._requested_rate
            for obj in chart["HitObjects"]:
                if isinstance(obj, dict):
                    start = int(obj.get("StartTime", 0))
                    end = int(obj.get("EndTime", 0) or 0)
                    obj["StartTime"] = int(round(start / rate))
                    obj["EndTime"] = int(round(end / rate)) if end else 0
            for point in chart.get("TimingPoints", []) or []:
                if isinstance(point, dict) and isinstance(point.get("Bpm"), (int, float)):
                    point["Bpm"] = float(point["Bpm"]) * rate
            scenario["rate"] = round(float(scenario.get("rate", 1.0)) * rate, 3)
        if self._requested_offset is not None:
            # Explicit offsets are applied to target times, not hidden as a
            # scoring bias.  This keeps the chart's .qua semantics observable.
            desired = int(round(self._requested_offset))
            previous = int(scenario.get("timing_offset_ms", 0))
            delta = desired - previous
            for obj in chart["HitObjects"]:
                obj["StartTime"] = int(obj.get("StartTime", 0)) + delta
                if int(obj.get("EndTime", 0) or 0) > 0:
                    obj["EndTime"] = int(obj["EndTime"]) + delta
            scenario["timing_offset_ms"] = desired
        chart["_source"] = source
        return chart

    def _parse_chart(self, chart: dict[str, Any]) -> dict[str, Any]:
        version_value = chart.get("QuaVersion", chart.get("Version", "1.0"))
        try:
            version = float(version_value)
        except (TypeError, ValueError) as exc:
            raise ChartFormatError(f"unsupported QuaVersion {version_value!r}") from exc
        if not math.isfinite(version) or version > 1.0:
            raise ChartFormatError(
                f"unsupported future chart version {version_value!r}; supported through 1.0"
            )
        mode = str(chart.get("Mode", ""))
        if mode not in {"Keys4", "Keys7"}:
            raise ChartFormatError(f"unsupported .qua Mode {mode!r}")
        lane_count = int(mode[-1])
        # Empty containers are emitted by some native serializers and are
        # harmless; non-empty groups/SVs would change the scroll semantics.
        for key in ("TimingGroups", "TimingGroup"):
            value = chart.get(key)
            if value not in (None, {}, []):
                raise ChartFormatError("TimingGroups are unsupported by the local simulator")
        for key in ("SliderVelocities", "ScrollSpeedFactors"):
            value = chart.get(key)
            if value not in (None, {}, []):
                raise ChartFormatError("scroll-velocity groups are unsupported")
        points = chart.get("TimingPoints", [])
        if not isinstance(points, list):
            raise ChartFormatError("TimingPoints must be a list")
        timing_points: list[dict[str, float]] = []
        for point in points:
            if not isinstance(point, dict):
                raise ChartFormatError("each TimingPoints entry must be a mapping")
            if any(key in point for key in ("TimingGroup", "Group", "ScrollVelocity", "SliderVelocity")):
                raise ChartFormatError("timing groups / scroll-velocity groups are unsupported")
            start_time = point.get("StartTime", 0)
            if (
                isinstance(start_time, bool)
                or not isinstance(start_time, (int, float))
                or not math.isfinite(float(start_time))
                or int(start_time) != float(start_time)
            ):
                raise ChartFormatError("TimingPoints.StartTime must be a finite integer")
            bpm = point.get("Bpm")
            if bpm is not None:
                if not isinstance(bpm, (int, float)) or not math.isfinite(float(bpm)) or float(bpm) <= 0:
                    raise ChartFormatError("TimingPoints.Bpm must be a positive finite number")
                signature = point.get("Signature", 4)
                if (
                    isinstance(signature, bool)
                    or not isinstance(signature, (int, float))
                    or not math.isfinite(float(signature))
                    or float(signature) <= 0
                ):
                    raise ChartFormatError("TimingPoints.Signature must be a positive number")
                timing_points.append(
                    {
                        "start_ms": float(int(start_time)),
                        "bpm": float(bpm),
                        "signature": float(signature),
                    }
                )
        timing_points.sort(key=lambda point: point["start_ms"])
        if not timing_points:
            # Keep a valid chart observable while explicitly marking that no
            # BPM was supplied; no hidden timing data is invented.
            timing_points = [{"start_ms": 0.0, "bpm": 0.0, "signature": 4.0}]
        objects = chart.get("HitObjects")
        if not isinstance(objects, list) or not objects:
            raise ChartFormatError("HitObjects must be a non-empty list")
        normalized: list[dict[str, int]] = []
        for index, item in enumerate(objects):
            if not isinstance(item, dict):
                raise ChartFormatError(f"HitObjects[{index}] must be a mapping")
            if self._is_unsupported_mine(item):
                raise ChartFormatError("mines are unsupported; refusing to score one as a tap")
            timing_group = item.get("TimingGroup")
            if timing_group not in (None, "", "$Default"):
                raise ChartFormatError("hit-object timing groups are unsupported")
            if any(key in item for key in ("Group", "ScrollVelocity", "SliderVelocity")):
                raise ChartFormatError("hit-object timing groups / scroll velocities are unsupported")
            start = self._integer_field(item, "StartTime", index)
            lane = self._integer_field(item, "Lane", index)
            end = self._integer_field(item, "EndTime", index, default=0)
            if start < 0:
                raise ChartFormatError(f"HitObjects[{index}].StartTime must be nonnegative")
            if lane < 1 or lane > lane_count:
                raise ChartFormatError(
                    f"HitObjects[{index}].Lane {lane} outside 1..{lane_count}"
                )
            if end < 0 or (end != 0 and end <= start):
                raise ChartFormatError(
                    f"HitObjects[{index}].EndTime must be 0 or greater than StartTime"
                )
            normalized.append({"StartTime": start, "Lane": lane, "EndTime": end})
        normalized.sort(key=lambda item: (item["StartTime"], item["Lane"]))
        effective = dict(chart)
        effective["HitObjects"] = normalized
        return {
            "raw": effective,
            "mode": mode,
            "lane_count": lane_count,
            "objects": normalized,
            "timing_points": timing_points,
        }

    @staticmethod
    def _integer_field(
        item: dict[str, Any], key: str, index: int, default: int | None = None
    ) -> int:
        value = item.get(key, default)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ChartFormatError(f"HitObjects[{index}].{key} must be numeric")
        if not math.isfinite(float(value)) or int(value) != float(value):
            raise ChartFormatError(f"HitObjects[{index}].{key} must be a finite integer")
        return int(value)

    @staticmethod
    def _is_unsupported_mine(item: dict[str, Any]) -> bool:
        for key in ("Mine", "IsMine", "isMine", "IsMineObject"):
            if item.get(key) is True:
                return True
        for key in ("Type", "HitObjectType", "ObjectType"):
            value = item.get(key, "")
            if str(value).strip().lower() == "mine" or value == 1:
                return True
        return False

    @staticmethod
    def _canonical_chart(chart: dict[str, Any]) -> bytes:
        clean = {key: value for key, value in chart.items() if not key.startswith("_")}
        return json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def _write_generated_chart_if_needed(self, chart: dict[str, Any], seed: int) -> None:
        if chart.get("_source") != "generated":
            self._chart_path = str(chart.get("_source", "external"))
            return
        output = self._runtime_dir / f"seed_{seed}.qua"
        clean = {key: value for key, value in chart.items() if not key.startswith("_")}
        output.write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
        self._chart_path = str(output)

    @staticmethod
    def _classify_note_patterns(notes: list[_Note]) -> dict[int, dict[str, Any]]:
        """Decode compact chart context needed for reading the next pattern."""
        by_start: dict[int, list[_Note]] = {}
        for note in notes:
            by_start.setdefault(note.start_ms, []).append(note)
        contexts: dict[int, dict[str, Any]] = {}
        previous_lanes: set[int] = set()
        for group_id, start_ms in enumerate(sorted(by_start), start=1):
            group = sorted(by_start[start_ms], key=lambda note: (note.lane, note.note_id))
            lanes = {note.lane for note in group}
            if len(lanes) >= 2:
                pattern = "chordjack" if previous_lanes & lanes else "chord"
            elif previous_lanes and lanes and next(iter(lanes)) in previous_lanes:
                pattern = "jack"
            elif previous_lanes:
                pattern = "stream"
            else:
                pattern = "tap"
            for note in group:
                contexts[note.note_id] = {
                    "group_id": group_id,
                    "group_size": len(group),
                    "simultaneous_lanes": sorted(lanes),
                    "pattern": pattern,
                    "note_role": "long_note" if note.is_long else "tap",
                }
            previous_lanes = lanes
        return contexts

    @staticmethod
    def _classify_pattern_mix(objects: list[Any], lane_count: int) -> dict[str, float]:
        if not objects:
            return {key: 0.0 for key in ("stream", "jack", "chord", "chordjack", "long_note")}
        by_time: dict[int, list[dict[str, Any]]] = {}
        long_count = 0
        for item in objects:
            if not isinstance(item, dict):
                continue
            start = int(item.get("StartTime", 0))
            by_time.setdefault(start, []).append(item)
            if int(item.get("EndTime", 0) or 0) > start:
                long_count += 1
        times = sorted(by_time)
        counts = {key: 0 for key in ("stream", "jack", "chord", "chordjack", "long_note")}
        previous: set[int] = set()
        for time in times:
            lanes = {int(x.get("Lane", 0)) for x in by_time[time]}
            if len(lanes) >= 2:
                counts["chord"] += 1
                if previous & lanes:
                    counts["chordjack"] += 1
            elif lanes:
                lane = next(iter(lanes))
                if lane in previous:
                    counts["jack"] += 1
                else:
                    counts["stream"] += 1
            previous = lanes
        counts["long_note"] = long_count
        denominator = max(1, sum(counts.values()))
        return {key: round(value / denominator, 3) for key, value in counts.items()}

    @staticmethod
    def _estimate_difficulty(count: int, length_ms: int, mode: str) -> float:
        density = count / max(length_ms / 1000.0, 1.0)
        return round(max(1.0, min(9.5, density * 0.35 + (1.0 if mode == "Keys7" else 0.5))), 3)

    # ------------------------------------------------------------------
    # Actions and virtual clock
    # ------------------------------------------------------------------
    def _parse_action(self, action: Any) -> list[tuple[str, int, int, bool]]:
        """Return (type, lane, time_ms, explicitly_timestamped) events."""
        now = self._now_ms
        if action is None:
            return []
        if isinstance(action, str):
            text = action.strip()
            if not text or text.lower() in {"noop", "wait", "pass", "none"}:
                return []
            if text.lower().startswith("events "):
                try:
                    return self._parse_action(json.loads(text[7:].strip()))
                except Exception as exc:  # noqa: BLE001
                    self._record_action_error(f"invalid events JSON: {exc}")
                    return []
            if text[0] in "[{":
                try:
                    return self._parse_action(json.loads(text))
                except Exception as exc:  # noqa: BLE001
                    self._record_action_error(f"invalid JSON action: {exc}")
                    return []
            try:
                words = shlex.split(text)
            except ValueError as exc:
                self._record_action_error(f"invalid action quoting: {exc}")
                return []
            if not words or words[0].lower() in {"noop", "wait", "pass"}:
                return []
            kind = words[0].lower()
            if kind not in {"press", "release"} or len(words) < 2:
                self._record_action_error(f"unsupported action grammar {text!r}")
                return []
            try:
                lane = int(words[1])
            except ValueError:
                self._record_action_error(f"lane must be an integer in {text!r}")
                return []
            event_time = now
            explicit = False
            if len(words) >= 3:
                token = words[2].lower()
                if token == "at" and len(words) >= 4:
                    token = words[3]
                try:
                    event_time = int(float(token))
                    explicit = True
                except (TypeError, ValueError, OverflowError):
                    self._record_action_error(f"timestamp must be an integer in {text!r}")
                    return []
            return [(kind, lane, event_time, explicit)]
        if isinstance(action, dict):
            if "events" in action:
                return self._parse_action(action["events"])
            kind = action.get("type", action.get("action", ""))
            if str(kind).lower() in {"noop", "wait", "pass", ""}:
                if str(kind).lower() == "":
                    self._record_action_error("event is missing type")
                return []
            try:
                lane_value = action.get("lane")
                lane = int(lane_value)
                if isinstance(lane_value, bool):
                    raise ValueError
            except (TypeError, ValueError):
                self._record_action_error("event lane must be an integer")
                return []
            event_time = action.get("time_ms", action.get("time", now))
            explicit = "time_ms" in action or "time" in action
            try:
                if isinstance(event_time, bool):
                    raise ValueError
                event_time = int(float(event_time))
            except (TypeError, ValueError, OverflowError):
                self._record_action_error("event time_ms must be numeric")
                return []
            kind = str(kind).lower()
            if kind not in {"press", "release"}:
                self._record_action_error(f"unsupported event type {kind!r}")
                return []
            return [(kind, lane, event_time, explicit)]
        if isinstance(action, (list, tuple)):
            events: list[tuple[str, int, int, bool]] = []
            for item in action:
                events.extend(self._parse_action(item))
            return events
        self._record_action_error(f"malformed action type {type(action).__name__}")
        return []

    @staticmethod
    def _action_label(action: Any) -> str:
        if isinstance(action, str):
            return action[:120]
        try:
            return json.dumps(action, separators=(",", ":"), default=str)[:120]
        except Exception:  # noqa: BLE001
            return type(action).__name__

    def _record_action_error(self, message: str) -> None:
        if self._last_action_error:
            if len(self._last_action_error) < 900:
                self._last_action_error += "; " + message
        else:
            self._last_action_error = message[:900]

    def _advance_to(self, target_ms: int) -> None:
        if target_ms < self._now_ms:
            return
        while self._event_queue and self._event_queue[0][0] <= target_ms:
            event_time, _seq, kind, lane = self._event_queue.pop(0)
            if event_time < self._now_ms:
                self._record_action_error(
                    f"queued {kind} lane {lane} became late at {event_time} ms"
                )
                continue
            self._expire_until(event_time)
            self._now_ms = event_time
            self._apply_event(kind, lane, event_time)
        self._expire_until(target_ms)
        self._now_ms = target_ms

    def _expire_until(self, target_ms: int) -> None:
        # Strict comparison preserves the inclusive timing boundary: an action
        # exactly at the miss deadline is still classified in the Miss window;
        # an unhit object is auto-missed only after that deadline.
        for note in self._notes:
            if not note.start_judged and not note.start_missed:
                if target_ms > note.start_ms + self._windows["Miss"]:
                    note.start_missed = True
                    self._record_judgement(
                        note, "press", "Miss", target_ms - note.start_ms, target_ms, automatic=True
                    )
                    if note.is_long:
                        note.release_resolved = True
                        self._record_judgement(
                            note, "release", "Miss", target_ms - note.end_ms, target_ms, automatic=True
                        )
            elif note.is_long and note.start_judged and not note.release_resolved:
                if target_ms > note.end_ms + self._release_window("Okay"):
                    note.release_resolved = True
                    note.held = False
                    self._held_lanes.discard(note.lane)
                    # Quaver's default LNMissJudgement is Good and does not
                    # break combo.  The configured value is preserved here.
                    self._record_judgement(
                        note,
                        "release",
                        self._ln_miss_judgement,
                        target_ms - note.end_ms,
                        target_ms,
                        automatic=True,
                    )

    def _apply_event(self, kind: str, lane: int, event_time: int) -> None:
        if kind == "press":
            candidates = [
                note
                for note in self._notes
                if note.lane == lane and not note.start_judged and not note.start_missed
            ]
            if not candidates:
                self._record_action_error(f"press lane {lane} has no pending object")
                return
            note = min(candidates, key=lambda item: (item.start_ms, item.note_id))
            error = event_time - note.start_ms
            if abs(error) > self._windows["Miss"]:
                if error < 0:
                    self._record_action_error(
                        f"press lane {lane} was {abs(error)} ms early for note {note.note_id}"
                    )
                else:
                    self._record_action_error(
                        f"press lane {lane} missed note {note.note_id} by {error} ms"
                    )
                return
            judgement = self._classify(abs(error), release=False)
            note.start_judged = True
            if note.is_long:
                note.held = True
                self._held_lanes.add(lane)
            self._record_judgement(note, "press", judgement, error, event_time)
            return

        candidates = [
            note
            for note in self._notes
            if note.lane == lane and note.is_long and note.start_judged and not note.release_resolved
        ]
        if not candidates:
            self._record_action_error(f"release lane {lane} has no held long note")
            return
        note = min(candidates, key=lambda item: (item.end_ms, item.note_id))
        error = event_time - note.end_ms
        release_limit = self._release_window("Okay")
        if abs(error) > release_limit:
            # Outside the release window, an explicit early release is a Miss.
            # A sufficiently late release normally got converted by _expire_until.
            note.release_resolved = True
            note.held = False
            self._held_lanes.discard(lane)
            self._record_judgement(note, "release", "Miss", error, event_time)
            return
        judgement = self._classify(abs(error), release=True, signed_error=error)
        note.release_resolved = True
        note.held = False
        self._held_lanes.discard(lane)
        self._record_judgement(note, "release", judgement, error, event_time)

    # ------------------------------------------------------------------
    # Judgements, score, and observations
    # ------------------------------------------------------------------
    def _classify(
        self, absolute_error: float, *, release: bool, signed_error: float = 0.0
    ) -> str:
        for name in _JUDGEMENTS[:-1]:
            window = self._release_window(name) if release else self._windows[name]
            if absolute_error <= window:
                # ScoreProcessorKeys makes an Okay release a Good when the
                # configured LN-miss judgement is Good, including an early
                # release that is still inside the enlarged release window.
                if release and name == "Okay" and self._ln_miss_judgement == "Good":
                    return "Good"
                # A late release is rounded up to LNMissJudgement.  With the
                # default this only changes an Okay to Good.
                if release and signed_error > 0:
                    rank = {value: index for index, value in enumerate(_JUDGEMENTS)}
                    miss_rank = rank[self._ln_miss_judgement]
                    if rank[name] > miss_rank:
                        return self._ln_miss_judgement
                return name
        return "Miss"

    def _record_judgement(
        self,
        note: _Note,
        part: str,
        judgement: str,
        error_ms: float,
        event_time: int,
        *,
        automatic: bool = False,
    ) -> None:
        judgement = judgement if judgement in _JUDGEMENTS else "Miss"
        self._counts[judgement] += 1
        if not automatic:
            self._timing_errors.append(float(error_ms))
        if automatic:
            self._auto_miss_count += 1
        self._accuracy_numerator += _ACCURACY_WEIGHTS[judgement]
        self._update_score(judgement)
        self._health = max(0.0, min(100.0, self._health + _HEALTH_WEIGHTS[judgement]))
        self._recent.append(
            {
                "id": note.note_id,
                "lane": note.lane,
                "part": part,
                "judgement": judgement,
                "error_ms": int(error_ms) if float(error_ms).is_integer() else round(float(error_ms), 3),
                "time_ms": int(event_time),
                "automatic": bool(automatic),
            }
        )

    def _update_score(self, judgement: str) -> None:
        if judgement == "Good":
            self._multiplier_count -= 10
        elif judgement == "Miss":
            self._multiplier_count -= 20
        else:
            self._multiplier_count += 1
        self._multiplier_count = max(0, min(150, self._multiplier_count))
        self._multiplier_index = self._multiplier_count // 10
        if judgement == "Miss":
            self._combo = 0
        else:
            self._combo += 1
            self._max_combo = max(self._max_combo, self._combo)
        self._score_count += _SCORE_WEIGHTS[judgement] + self._multiplier_index * 10

    @staticmethod
    def _calculate_summed_score(total_judgements: int) -> int:
        if total_judgements <= 0:
            return 1
        max_multiplier_count = 150
        summed = 0
        for i in range(1, total_judgements + 1):
            if i >= max_multiplier_count:
                break
            summed += 100 + 10 * (i // 10)
        if total_judgements >= max_multiplier_count:
            summed += (total_judgements - (max_multiplier_count - 1)) * (100 + max_multiplier_count)
        return max(1, summed)

    def _normalized_score(self) -> float:
        return max(0.0, min(1_000_000.0, 1_000_000.0 * self._score_count / self._summed_score))

    def _accuracy_percent(self) -> float:
        count = sum(self._counts.values())
        if count <= 0:
            return 100.0
        return max(0.0, min(100.0, self._accuracy_numerator / count))

    def _grade(self) -> str:
        accuracy = self._accuracy_percent()
        if self._counts["Miss"] == 0 and self._counts["Okay"] == 0 and self._counts["Good"] == 0 and self._counts["Great"] == 0 and self._counts["Perfect"] == 0:
            return "X"
        if accuracy >= 99.0:
            return "SS"
        if accuracy >= 95.0:
            return "S"
        if accuracy >= 90.0:
            return "A"
        if accuracy >= 80.0:
            return "B"
        if accuracy >= 70.0:
            return "C"
        if accuracy >= 60.0:
            return "D"
        return "F"

    def _release_window(self, judgement: str) -> float:
        return self._windows[judgement] * self._release_multiplier

    def _configure_windows(self, chart: dict[str, Any]) -> None:
        self._windows = dict(_PRESS_WINDOWS)
        if self._requested_windows is not None:
            self._windows.update(self._requested_windows)
        # A chart may request explicit custom windows, but default generated
        # maps never do.  Unknown settings are rejected rather than ignored.
        custom = chart.get("JudgementWindows")
        if custom is not None:
            parsed = self._parse_custom_windows(custom)
            if parsed is None:
                raise ChartFormatError("JudgementWindows must contain six numeric windows")
            self._windows.update(parsed)
        self._ln_miss_judgement = str(chart.get("LNMissJudgement", "Good"))
        if self._ln_miss_judgement not in {"Good", "Okay", "Miss"}:
            raise ChartFormatError("LNMissJudgement must be Good, Okay, or Miss")

    @staticmethod
    def _parse_custom_windows(value: Any) -> dict[str, float] | None:
        if value is None:
            return None
        names = _JUDGEMENTS
        if isinstance(value, dict):
            values = [value.get(name, value.get(name.lower())) for name in names]
        elif isinstance(value, (list, tuple)) and len(value) == 6:
            values = list(value)
        else:
            raise ValueError("custom_window_ms must be a six-item list or judgement mapping")
        result: dict[str, float] = {}
        previous = 0.0
        for name, raw in zip(names, values):
            if raw is None:
                raise ValueError(f"custom window missing {name}")
            number = float(raw)
            if not math.isfinite(number) or number <= previous:
                raise ValueError("custom judgement windows must be finite and increasing")
            result[name] = number
            previous = number
        return result

    def _modifiers_for_info(self) -> dict[str, Any]:
        return {
            "mirror": self._requested_mirror,
            "requested_rate": self._requested_rate if self._requested_rate is not None else 1.0,
            "scenario_rate": self._scenario.get("rate", 1.0),
            "timing_offset_ms": self._scenario.get("timing_offset_ms", 0),
            "custom_window_ms": {key: self._windows[key] for key in _JUDGEMENTS},
            "ln_miss_judgement": self._ln_miss_judgement,
        }

    def _update_done(self, force: bool = False) -> None:
        if not self._notes:
            self._done = True
            return
        all_resolved = all(note.resolved for note in self._notes)
        self._done = bool(force or (all_resolved and self._now_ms >= self._finish_time_ms))
        if self._health <= 0.0 and self._now_ms >= min(self._finish_time_ms, 1):
            # The client can fail on zero health; retain chart state for
            # diagnostics but stop the realtime episode.
            self._done = True

    def _timing_state(self) -> dict[str, Any]:
        current = self._timing_points[0] if self._timing_points else {
            "start_ms": 0.0,
            "bpm": 0.0,
            "signature": 4.0,
        }
        upcoming: list[dict[str, Any]] = []
        next_future_ms = -1
        for point in self._timing_points:
            if point["start_ms"] <= self._now_ms:
                current = point
            else:
                if next_future_ms < 0:
                    next_future_ms = int(point["start_ms"])
                if point["start_ms"] <= self._now_ms + self._lookahead_ms and len(upcoming) < 8:
                    upcoming.append(
                        {
                            "start_ms": int(point["start_ms"]),
                            "bpm": round(float(point["bpm"]), 6),
                            "signature": round(float(point["signature"]), 6),
                            "time_to_change_ms": int(point["start_ms"] - self._now_ms),
                        }
                    )
        next_change = next_future_ms
        bpm = float(current["bpm"])
        return {
            "current_bpm": round(bpm, 6),
            "current_signature": round(float(current["signature"]), 6),
            "current_point_ms": int(current["start_ms"]),
            "beat_period_ms": round(60000.0 / bpm, 6) if bpm > 0 else 0.0,
            "next_change_ms": next_change,
            "rate": round(float(self._scenario.get("rate", 1.0) or 1.0), 6),
            "timing_offset_ms": int(self._scenario.get("timing_offset_ms", 0) or 0),
            "scroll": {
                "axis": "vertical",
                "direction": "toward_receptor",
                "velocity_model": "timestamp_authoritative",
                "velocity_factor": 1.0,
                "timing_groups_supported": False,
            },
            "upcoming_points": upcoming,
        }

    def _timing_error_summary(self) -> dict[str, Any]:
        errors = list(self._timing_errors)
        if errors:
            mean = statistics.fmean(errors)
            median = statistics.median(errors)
            mean_abs = statistics.fmean(abs(error) for error in errors)
            max_abs = max(abs(error) for error in errors)
            variance = statistics.pvariance(errors) if len(errors) > 1 else 0.0
            recent_errors = errors[-8:]
            recent_mean = statistics.fmean(recent_errors)
            recent_abs = statistics.fmean(abs(error) for error in recent_errors)
            recent_variance = statistics.pvariance(recent_errors) if len(recent_errors) > 1 else 0.0
            if len(errors) >= 4:
                half = max(1, len(errors) // 2)
                drift = statistics.fmean(errors[-half:]) - statistics.fmean(errors[:half])
            else:
                drift = 0.0
        else:
            mean = median = mean_abs = max_abs = variance = 0.0
            recent_mean = recent_abs = recent_variance = drift = 0.0
        bins = {
            "early_over_100": 0,
            "early_100_to_43": 0,
            "early_43_to_18": 0,
            "near_zero": 0,
            "late_18_to_43": 0,
            "late_43_to_100": 0,
            "late_over_100": 0,
        }
        for error in errors:
            if error < -100:
                bins["early_over_100"] += 1
            elif error < -43:
                bins["early_100_to_43"] += 1
            elif error < -18:
                bins["early_43_to_18"] += 1
            elif error <= 18:
                bins["near_zero"] += 1
            elif error <= 43:
                bins["late_18_to_43"] += 1
            elif error <= 100:
                bins["late_43_to_100"] += 1
            else:
                bins["late_over_100"] += 1
        return {
            "sample_count": len(errors),
            "automatic_judgement_count": int(self._auto_miss_count),
            "mean_ms": round(float(mean), 6),
            "median_ms": round(float(median), 6),
            "mean_abs_ms": round(float(mean_abs), 6),
            "max_abs_ms": round(float(max_abs), 6),
            "variance_ms2": round(float(variance), 6),
            "recent_mean_ms": round(float(recent_mean), 6),
            "recent_mean_abs_ms": round(float(recent_abs), 6),
            "recent_variance_ms2": round(float(recent_variance), 6),
            "drift_ms": round(float(drift), 6),
            "early_count": sum(1 for error in errors if error < 0),
            "late_count": sum(1 for error in errors if error > 0),
            "on_time_count": sum(1 for error in errors if error == 0),
            "histogram": bins,
        }

    def _chart_progress(self) -> dict[str, Any]:
        chart_end = max(
            (note.end_ms if note.is_long else note.start_ms for note in self._notes),
            default=0,
        )
        resolved_notes = sum(1 for note in self._notes if note.resolved)
        completed = sum(self._counts.values())
        remaining_judgements = max(0, self._total_judgements - completed)
        return {
            "resolved_notes": int(resolved_notes),
            "remaining_notes": int(max(0, len(self._notes) - resolved_notes)),
            "remaining_judgements": int(remaining_judgements),
            "elapsed_ms": int(self._now_ms),
            "remaining_time_ms": int(max(0, chart_end - self._now_ms)),
            "progress_fraction": round(
                max(0.0, min(1.0, self._now_ms / chart_end)) if chart_end else 1.0,
                6,
            ),
        }

    def _visible_notes(self) -> list[dict[str, Any]]:
        low = self._now_ms - int(self._windows["Miss"])
        high = self._now_ms + self._lookahead_ms
        visible: list[dict[str, Any]] = []
        for note in self._notes:
            target_end = note.end_ms if note.is_long else note.start_ms
            if not (note.start_ms <= high and target_end >= low):
                continue
            if note.start_ms < low and note.resolved and target_end < low:
                continue
            if note.start_missed:
                state = "missed"
            elif note.is_long and note.start_judged and not note.release_resolved:
                state = "held"
            elif note.resolved:
                state = "judged"
            else:
                state = "pending"
            visible.append(
                {
                    "id": note.note_id,
                    "lane": note.lane,
                    "kind": "long" if note.is_long else "tap",
                    "motion": "scroll",
                    "motion_detail": "vertical_toward_receptor",
                    "start_ms": note.start_ms,
                    "end_ms": note.end_ms if note.is_long else 0,
                    "time_to_start_ms": note.start_ms - self._now_ms,
                    "time_to_end_ms": (note.end_ms if note.is_long else note.start_ms) - self._now_ms,
                    "state": state,
                    **self._note_patterns.get(
                        note.note_id,
                        {
                            "group_id": 0,
                            "group_size": 1,
                            "simultaneous_lanes": [note.lane],
                            "pattern": "tap",
                            "note_role": "long_note" if note.is_long else "tap",
                        },
                    ),
                }
            )
        return visible

    def _observe_unlocked(self) -> dict[str, Any]:
        accuracy = round(self._accuracy_percent(), 6)
        difficulty = float(self._scenario.get("difficulty_rating", 0.0) or 0.0)
        performance = difficulty * (accuracy / 98.0) ** 6 if difficulty > 0 else 0.0
        performance = max(0.0, float(performance))
        timing = self._timing_state()
        progress = self._chart_progress()
        lane_actions = {
            str(lane): {
                "press": f"press {lane}",
                "release": f"release {lane}",
                "held": lane in self._held_lanes,
            }
            for lane in range(1, self._lane_count + 1)
        }
        return {
            "interface": "quaver-local-sim-v1",
            "fidelity": "local_sim",
            "fidelity_version": "quaver-local-sim-v1",
            "clock_ms": int(self._now_ms),
            "tick_s": float(self.tick_s),
            "ticks": int(self._ticks),
            "mode": self._mode,
            "lane_count": int(self._lane_count),
            "lanes": int(self._lane_count),
            "lane_actions": lane_actions,
            "input_mapping": {
                "kind": "virtual_lane_events",
                "lane_order": list(range(1, self._lane_count + 1)),
                "physical_keys": "not_applicable_in_local_sim",
            },
            "timing": timing,
            "judgement_window": "Standard*" if self._windows == _PRESS_WINDOWS else "custom",
            "judgement_window_ms": {key: float(self._windows[key]) for key in _JUDGEMENTS},
            "release_window_multiplier": float(self._release_multiplier),
            "visible_notes": self._visible_notes(),
            "held_lanes": sorted(self._held_lanes),
            "recent": list(self._recent),
            "timing_error_summary": self._timing_error_summary(),
            "judgements": dict(self._counts),
            "miss_count": int(self._counts["Miss"]),
            "total_judgements": int(self._total_judgements),
            "completed_judgements": int(sum(self._counts.values())),
            "score": round(self._normalized_score(), 3),
            "score_count": int(self._score_count),
            "accuracy_percent": accuracy,
            "grade": self._grade(),
            "combo": int(self._combo),
            "final_combo": int(self._combo),
            "max_combo": int(self._max_combo),
            "full_combo": bool(self._counts["Miss"] == 0 and all(n.resolved for n in self._notes)),
            "health": round(float(self._health), 4),
            "resources": {
                "health": round(float(self._health), 4),
                "combo": int(self._combo),
                "max_combo": int(self._max_combo),
                "held_lanes": sorted(self._held_lanes),
            },
            "chart_progress": progress,
            "difficulty_rating": round(difficulty, 6),
            "performance_rating": round(performance, 6),
            "seed": self._seed,
            "chart_hash": self._chart_hash,
            "ruleset": self._ruleset,
            "scenario": dict(self._scenario),
            "modifiers": self._modifiers_for_info(),
            "lookahead_ms": int(self._lookahead_ms),
            "terminal": bool(self._done),
            "last_action": self._last_action,
            "last_action_error": self._last_action_error,
        }


def make_game(**kwargs: Any) -> QuaverLocalGame:
    """Construct one fresh local Quaver episode."""
    return QuaverLocalGame(**kwargs)
