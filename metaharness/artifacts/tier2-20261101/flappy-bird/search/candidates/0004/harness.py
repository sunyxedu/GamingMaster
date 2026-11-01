"""Async supervisory MPC harness for the real-time Flappy Bird clone.

The game loop never waits for a model.  Code owns the physics rollout and posts one
fresh action after every observed tick.  Astra may select a coarse risk posture in the
background; Jev is only a consensus tie-breaker between two already-computed actions.
"""
from __future__ import annotations

import math
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


# Public game constants, used only for the local model-predictive rollout.  The
# observation's geometry is preferred whenever it supplies the corresponding value.
DEFAULT_W = 288.0
DEFAULT_H = 512.0
DEFAULT_BIRD_X = 57.0
DEFAULT_BIRD_W = 34.0
DEFAULT_BIRD_H = 24.0
DEFAULT_PIPE_W = 52.0
DEFAULT_GROUND = 404.48
DEFAULT_SPEED = 4.0
DEFAULT_GRAVITY = 1.0
DEFAULT_FLAP_V = -9.0
DEFAULT_MAX_DOWN = 10.0


def _finite(value: Any, default: float = 0.0) -> float:
    """Return a finite float without allowing malformed model/game data to escape."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _json_number(value: Any, default: float = 0.0) -> float:
    result = _finite(value, default)
    return result if math.isfinite(result) else default


def _valid_gap(top: float, bottom: float) -> bool:
    return math.isfinite(top) and math.isfinite(bottom) and 40.0 <= bottom - top <= 250.0


def _pipe_from_dict(item: Any, fallback_name: str = "pipe") -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    x = _finite(item.get("x"), DEFAULT_W)
    top = _finite(item.get("gap_top_y", item.get("top_y")), 0.0)
    bottom = _finite(item.get("gap_bottom_y", item.get("bottom_y")), DEFAULT_H)
    if not _valid_gap(top, bottom):
        return None
    return {
        "name": _text(item.get("name")) or fallback_name,
        "x": x,
        "top": top,
        "bottom": bottom,
    }


def _pipe_list(observation: Any) -> list[dict[str, Any]]:
    if not isinstance(observation, dict):
        return []
    raw = observation.get("pipes")
    result: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for index, item in enumerate(raw):
            parsed = _pipe_from_dict(item, "pipe_%d" % index)
            if parsed is not None:
                result.append(parsed)
    target = _pipe_from_dict(observation.get("target_pipe"), "target")
    if target is not None and not any(
        abs(target["x"] - item["x"]) < 1e-6
        and abs(target["top"] - item["top"]) < 1e-6
        and abs(target["bottom"] - item["bottom"]) < 1e-6
        for item in result
    ):
        result.append(target)
    # The adaptor exposes records in spatial order.  Sorting makes the controller
    # tolerate a missing convenience view without trusting a model to find a target.
    result.sort(key=lambda item: item["x"])
    return result


def _view(observation: Any) -> dict[str, Any]:
    """Extract a small, finite, named state and derived geometry from strict JSON."""
    obs = observation if isinstance(observation, dict) else {}
    bird = obs.get("bird") if isinstance(obs.get("bird"), dict) else {}
    geometry = obs.get("geometry") if isinstance(obs.get("geometry"), dict) else {}

    features = obs.get("features")
    if not isinstance(features, list):
        features = []

    def feature(index: int, default: float) -> float:
        return _finite(features[index], default) if index < len(features) else default

    bird_x = _finite(bird.get("x"), DEFAULT_BIRD_X)
    bird_y = _finite(bird.get("y"), feature(9, 244.0))
    bird_v = _finite(bird.get("vertical_velocity"), feature(10, -9.0))
    bird_w = max(1.0, _finite(bird.get("width"), DEFAULT_BIRD_W))
    bird_h = max(1.0, _finite(bird.get("height"), DEFAULT_BIRD_H))
    screen_w = max(1.0, _finite(geometry.get("screen_width"), DEFAULT_W))
    screen_h = max(1.0, _finite(geometry.get("screen_height"), DEFAULT_H))
    pipe_w = max(1.0, _finite(geometry.get("pipe_width"), DEFAULT_PIPE_W))
    ground = _finite(geometry.get("ground_y"), DEFAULT_GROUND)
    speed = abs(_finite(geometry.get("pipe_speed_px_per_frame"), DEFAULT_SPEED)) or DEFAULT_SPEED
    gravity = abs(_finite(geometry.get("gravity_px_per_frame2"), DEFAULT_GRAVITY)) or DEFAULT_GRAVITY
    flap_v = _finite(geometry.get("flap_velocity_px_per_frame"), DEFAULT_FLAP_V)
    max_down = max(1.0, abs(_finite(geometry.get("max_vertical_velocity"), DEFAULT_MAX_DOWN)))
    if max_down < 5.0:
        max_down = DEFAULT_MAX_DOWN

    pipes = _pipe_list(obs)
    active = None
    # A pipe is active until its right edge is behind the bird.  Use the first
    # spatially valid record rather than the raw feature slot named "last".
    for item in pipes:
        if item["x"] + pipe_w >= bird_x - 1.0:
            active = item
            break
    if active is None and pipes:
        active = pipes[-1]
    if active is None:
        active = {
            "name": "fallback",
            "x": screen_w,
            "top": max(0.0, min(screen_h - 100.0, bird_y - 50.0)),
            "bottom": max(100.0, min(screen_h, bird_y + 50.0)),
        }
        if active["bottom"] - active["top"] < 40.0:
            active["top"], active["bottom"] = 0.0, screen_h

    following = [item for item in pipes if item["x"] > active["x"] + 1.0]
    target_center = (active["top"] + active["bottom"] - bird_h) / 2.0
    top_margin = bird_y - active["top"]
    bottom_margin = active["bottom"] - (bird_y + bird_h)
    overlap_now = (
        active["x"] < bird_x + bird_w
        and active["x"] + pipe_w > bird_x
    )
    # Phase is a code-computed label, not a question for Jev.
    if active["x"] + pipe_w < bird_x:
        phase = "passed"
    elif active["x"] > bird_x + bird_w:
        phase = "approach"
    else:
        phase = "crossing"

    frame_value = obs.get("frame", 0)
    try:
        frame = int(float(frame_value))
    except (TypeError, ValueError, OverflowError):
        frame = 0
    score = _finite(obs.get("score", obs.get("pipe_pairs_passed", 0.0)), 0.0)
    done = bool(obs.get("done", False) or obs.get("terminated", False) or obs.get("truncated", False))
    return {
        "frame": frame,
        "score": score,
        "done": done,
        "bird_x": bird_x,
        "bird_y": bird_y,
        "bird_v": bird_v,
        "bird_w": bird_w,
        "bird_h": bird_h,
        "screen_w": screen_w,
        "screen_h": screen_h,
        "pipe_w": pipe_w,
        "ground": ground,
        "speed": speed,
        "gravity": gravity,
        "flap_v": flap_v,
        "max_down": max_down,
        "pipes": pipes,
        "active": active,
        "following": following,
        "target_center": target_center,
        "top_margin": top_margin,
        "bottom_margin": bottom_margin,
        "overlap_now": overlap_now,
        "phase": phase,
    }


def _target_key(view: dict[str, Any]) -> tuple[int, int]:
    active = view["active"]
    # Gap identity is stable while a pipe travels horizontally.  Horizontal
    # freshness is checked separately on a pending Jev answer.
    return (
        int(round(_finite(active.get("top")) / 2.0)),
        int(round(_finite(active.get("bottom")) / 2.0)),
    )


def _step_vertical(y: float, velocity: float, action: str, view: dict[str, Any]) -> tuple[float, float]:
    """One frame in the order verified from the archived observation trace."""
    if action == "FLAP":
        # A flap replaces the velocity and applies that impulse for this frame;
        # the next observed frame still reports -9 when flaps are held.
        velocity = view["flap_v"]
        return y + velocity, velocity
    # In this build gravity is applied before an IDLE displacement.  This gives
    # frame 0 (244,-9) -> frame 5 (214,-4) in the archived all-IDLE prefix.
    velocity = min(velocity + view["gravity"], view["max_down"])
    return y + velocity, velocity


def _pipe_overlap(pipe_x: float, view: dict[str, Any]) -> bool:
    return (
        pipe_x < view["bird_x"] + view["bird_w"]
        and pipe_x + view["pipe_w"] > view["bird_x"]
    )


def _target_index(view: dict[str, Any]) -> int:
    target = view["active"]
    return next(
        (i for i, item in enumerate(view["pipes"])
         if abs(item["x"] - target["x"]) < 1e-6
         and abs(item["top"] - target["top"]) < 1e-6
         and abs(item["bottom"] - target["bottom"]) < 1e-6),
        0,
    )


def _new_node(view: dict[str, Any], since_flap: int = 100) -> dict[str, Any]:
    return {
        "actions": [],
        "y": view["bird_y"],
        "velocity": view["bird_v"],
        "pipe_positions": tuple(item["x"] for item in view["pipes"]),
        "score": 0.0,
        "dead": False,
        "min_margin": 1000.0,
        "closest_error": 1000.0,
        "target_contact": False,
        "flaps": 0,
        "since_flap": max(0, int(since_flap)),
    }


def _advance_node(
    node: dict[str, Any], view: dict[str, Any], action: str, margin: float,
    target_index: int,
) -> dict[str, Any]:
    """Advance one beam node, avoiding the quadratic re-simulation of each prefix."""
    if node.get("dead"):
        return dict(node, actions=list(node["actions"]) + [action])
    y, velocity = _step_vertical(node["y"], node["velocity"], action, view)
    pipe_positions = tuple(x - view["speed"] for x in node["pipe_positions"])
    score = _finite(node.get("score"), 0.0)
    min_margin = _finite(node.get("min_margin"), 1000.0)
    closest_error = _finite(node.get("closest_error"), 1000.0)
    target_contact = bool(node.get("target_contact", False))
    dead = False

    if y + view["bird_h"] >= view["ground"]:
        dead = True
        score -= 10000.0
    if not dead:
        for index, item in enumerate(view["pipes"]):
            if index >= len(pipe_positions) or not _pipe_overlap(pipe_positions[index], view):
                continue
            gap_margin = min(y - item["top"], item["bottom"] - (y + view["bird_h"]))
            min_margin = min(min_margin, gap_margin)
            if index == target_index:
                target_contact = True
            if gap_margin < margin:
                score -= 500.0 + max(0.0, margin - gap_margin) * 20.0
                if gap_margin < 0.0:
                    dead = True
                    score -= 10000.0
                    break
            else:
                score += min(12.0, gap_margin) * 0.35

    # Before contact, weakly reward a trajectory that arrives at the safe centre.
    target = view["active"]
    if target_index < len(pipe_positions):
        target_x = pipe_positions[target_index]
    else:
        target_x = target["x"] - len(node["actions"]) * view["speed"]
    center_x = view["bird_x"] + view["bird_w"] / 2.0
    horizontal_error = abs(target_x + view["pipe_w"] / 2.0 - center_x)
    desired = (target["top"] + target["bottom"] - view["bird_h"]) / 2.0
    vertical_error = abs(y - desired)
    closest_error = min(closest_error, vertical_error)
    if horizontal_error < 100.0:
        score -= vertical_error * 0.008
    if action == "FLAP":
        score -= 0.055
    if y < 0.0:
        score -= min(10.0, -y) * 0.018

    return {
        "actions": list(node["actions"]) + [action],
        "y": y,
        "velocity": velocity,
        "pipe_positions": pipe_positions,
        "score": score,
        "dead": dead,
        "min_margin": min_margin,
        "closest_error": closest_error,
        "target_contact": target_contact,
        "flaps": int(node.get("flaps", 0)) + (1 if action == "FLAP" else 0),
        "since_flap": 0 if action == "FLAP" else int(node.get("since_flap", 100)) + 1,
    }


def _node_result(node: dict[str, Any]) -> dict[str, Any]:
    score = _finite(node.get("score"), 0.0)
    if not node.get("target_contact"):
        score -= _finite(node.get("closest_error"), 1000.0) * 0.004
    if node.get("dead"):
        score -= 100000.0
    return {
        "score": score,
        "dead": bool(node.get("dead")),
        "min_margin": _finite(node.get("min_margin"), -1.0),
        "closest_error": _finite(node.get("closest_error"), 1000.0),
        "flaps": int(node.get("flaps", 0)),
        "final_y": _finite(node.get("y")),
        "final_v": _finite(node.get("velocity")),
        "actions": list(node.get("actions", [])),
    }


def _horizon(view: dict[str, Any]) -> int:
    active = view["active"]
    # Run through the whole horizontal overlap, plus a small re-planning buffer.
    exit_distance = active["x"] + view["pipe_w"] - view["bird_x"]
    frames = int(math.ceil(exit_distance / max(view["speed"], 1.0))) + 4
    return max(12, min(72, frames))


def _beam_plan(
    view: dict[str, Any], mode: str, since_flap: int = 100
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Incremental short-horizon beam MPC; return best and first-action rival."""
    margin = 7.0 if mode == "CAUTIOUS" else 4.0
    target_index = _target_index(view)
    beam: list[dict[str, Any]] = [_new_node(view, since_flap)]
    beam_width = 40
    for _ in range(_horizon(view)):
        expanded: list[dict[str, Any]] = []
        for prefix in beam:
            if prefix.get("dead"):
                continue
            for action in ("IDLE", "FLAP"):
                # A tap is an impulse, not a held state.  Require a short recovery
                # interval unless the current trajectory is genuinely near the
                # ground or already outside the lower edge of the target.
                if action == "FLAP" and int(prefix.get("since_flap", 100)) < 3:
                    urgent = (
                        prefix["y"] + view["bird_h"] >= view["ground"] - 30.0
                        or (prefix["target_contact"] and prefix["y"] + view["bird_h"] > view["active"]["bottom"] - 8.0)
                    )
                    if not urgent:
                        continue
                expanded.append(_advance_node(prefix, view, action, margin, target_index))
        if not expanded:
            break
        expanded.sort(key=lambda item: item["score"], reverse=True)
        # Preserve the best 20 prefixes for each first action, then fill globally.
        kept: list[dict[str, Any]] = []
        for first in ("IDLE", "FLAP"):
            same = [item for item in expanded if item["actions"] and item["actions"][0] == first]
            kept.extend(same[:10])
        seen = {id(item) for item in kept}
        for item in expanded:
            if id(item) not in seen:
                kept.append(item)
            if len(kept) >= beam_width:
                break
        beam = kept[:beam_width]

    if not beam:
        idle = _node_result(_advance_node(_new_node(view, since_flap), view, "IDLE", margin, target_index))
        flap = _node_result(_advance_node(_new_node(view, since_flap), view, "FLAP", margin, target_index))
        return idle, flap
    ranked = sorted((_node_result(item) for item in beam), key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    best_first = best["actions"][0] if best["actions"] else "IDLE"
    rivals = [item for item in ranked if item["actions"] and item["actions"][0] != best_first]
    if rivals:
        return best, rivals[0]
    rival_action = "FLAP" if best_first == "IDLE" else "IDLE"
    rival = _node_result(_advance_node(_new_node(view, since_flap), view, rival_action, margin, target_index))
    return best, rival


def _label(result: dict[str, Any], view: dict[str, Any]) -> str:
    if result.get("dead"):
        return "predicted_collision"
    margin = _finite(result.get("min_margin"), -1.0)
    if margin < 0.0:
        return "no_pipe_contact_in_horizon"
    if margin < 8.0:
        return "edge_grazing"
    if margin < 20.0:
        return "narrow_safe_band"
    return "wide_safe_band"


def _candidate_state(
    view: dict[str, Any],
    best: dict[str, Any],
    rival: dict[str, Any],
    mode: str,
    revision: int,
) -> dict[str, Any]:
    best_action = best["actions"][0] if best.get("actions") else "IDLE"
    rival_action = rival["actions"][0] if rival.get("actions") else ("FLAP" if best_action == "IDLE" else "IDLE")
    return {
        "game": "flappy_bird",
        "frame": int(view["frame"]),
        "plan": {"mode": mode, "revision": int(revision), "owner": "code_mpc"},
        "phase": view["phase"],
        "active_gap": {
            "top": _json_number(view["active"]["top"]),
            "bottom": _json_number(view["active"]["bottom"]),
            "x": _json_number(view["active"]["x"]),
            "width": _json_number(view["pipe_w"]),
        },
        "bird": {
            "y": _json_number(view["bird_y"]),
            "vertical_velocity": _json_number(view["bird_v"]),
            "top_margin": _json_number(view["top_margin"]),
            "bottom_margin": _json_number(view["bottom_margin"]),
        },
        "forecast": {
            "code_action": best_action,
            "code_label": _label(best, view),
            "code_min_margin": _json_number(best.get("min_margin"), -1.0),
            "alternative_action": rival_action,
            "alternative_label": _label(rival, view),
            "alternative_min_margin": _json_number(rival.get("min_margin"), -1.0),
            "code_dead": bool(best.get("dead", False)),
            "alternative_dead": bool(rival.get("dead", False)),
        },
        "score": _json_number(view["score"]),
    }


def _parse_mode(text: Any) -> str | None:
    # Accept only a stand-alone, explicit mode token.  Prose or arbitrary preflight
    # output cannot silently change a controller.
    value = _text(text).strip().upper().strip("`'\" ")
    match = re.fullmatch(r"(BALANCED|CAUTIOUS)[.!]?", value)
    return match.group(1) if match else None


def _answer_value(answer: Any) -> tuple[str | None, float]:
    if answer is None:
        return None, 0.0
    value = getattr(answer, "value", None)
    if not isinstance(value, str):
        value = getattr(answer, "choice", None)
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        probabilities = {}
    probability = _finite(probabilities.get(value), 0.0) if isinstance(value, str) else 0.0
    return value if isinstance(value, str) else None, probability


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.mode = "BALANCED"
        self.pending_mode: str | None = None
        self.revision = 0
        self.astra_future = None
        self.jev_future = None
        self.jev_request: dict[str, Any] | None = None
        self.last_target_key: tuple[int, int] | None = None
        self.last_jev_frame = -1000
        self.astra_ready_frame = -1000
        self.since_flap = 100

    def _astra_prompt(self, initial: dict[str, Any]) -> str:
        return (
            "You are the slow episode supervisor for a symbolic Flappy Bird clone. "
            "The live loop is controlled by a code-side short-horizon ballistic MPC; "
            "you must not invent actions or do arithmetic. Choose one risk posture for "
            "the controller and reply with exactly one token: BALANCED or CAUTIOUS. "
            "BALANCED uses a 4 px interior robustness margin and sparse impulses. "
            "CAUTIOUS uses a 7 px margin and gives up some flap economy for edge clearance. "
            "Use CAUTIOUS only if the supplied opening state suggests that robustness is "
            "more important than impulse economy; otherwise use BALANCED. This is not a "
            "seed-specific memorized opening. Opening state (already derived by code): "
            + repr(initial)
        )

    def _start_astra(self, initial: dict[str, Any]) -> None:
        if self.astra_future is not None or getattr(self.ctx, "system2", None) is None:
            return

        def worker() -> str:
            try:
                result = self.ctx.think(self._astra_prompt(initial), timeout_s=45.0)
                return _text(getattr(result, "text", ""))
            except (TimeoutError, RuntimeError):
                return ""

        try:
            self.astra_future = self.ctx.spawn(worker)
        except (TimeoutError, RuntimeError):
            self.astra_future = None

    def _poll_astra(self, view: dict[str, Any]) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            text = future.result()
        except Exception:
            text = ""
        parsed = _parse_mode(text)
        if parsed is None or parsed == self.mode:
            return
        # Do not change the objective in the middle of a pipe corridor.  It becomes
        # active at the next observed target transition (or after a quiet buffer).
        self.pending_mode = parsed
        self.astra_ready_frame = int(view["frame"])
        self.ctx.log("astra_plan_ready", mode=parsed, frame=view["frame"])

    def _maybe_commit_mode(self, view: dict[str, Any]) -> None:
        if self.pending_mode is None:
            return
        current_key = _target_key(view)
        if (
            self.last_target_key is None
            or current_key != self.last_target_key
            or int(view["frame"]) - self.astra_ready_frame >= 20
        ):
            self.mode = self.pending_mode
            self.pending_mode = None
            self.revision += 1
            self.ctx.log("astra_plan_committed", mode=self.mode, frame=view["frame"], revision=self.revision)

    def _jev_prompt(self, candidate: dict[str, Any]) -> dict[str, Any]:
        # The two choices intentionally reverse option order.  Code accepts an
        # answer only when both independent judgments agree with enough probability.
        code = candidate["forecast"]["code_action"]
        alternate = candidate["forecast"]["alternative_action"]
        code_description = (
            "Use the code MPC candidate `code_action`. It has already simulated the "
            "published gravity/flap rule and is the safer candidate when its forecast "
            "label and margin support that conclusion."
        )
        alternate_description = (
            "Use the other one-frame candidate `alternative_action`. Choose it only "
            "when the supplied forecast says it is materially safer than code_action; "
            "do not calculate or reinterpret the numbers."
        )
        return {
            "state": candidate,
            "questions": {
                "forward": Choice(
                    "Choose exactly one candidate for the next single frame. Use the named forecast labels, phase, and margins in state. Do not do arithmetic. Prefer a candidate marked predicted_collision only if the other candidate is not predicted_collision.",
                    {"code": code_description, "alternate": alternate_description},
                ),
                "reverse": Choice(
                    "Independently choose exactly one candidate for the next single frame. Use the named forecast labels, phase, and margins in state. Do not do arithmetic. Prefer a candidate marked predicted_collision only if the other candidate is not predicted_collision.",
                    {"alternate": alternate_description, "code": code_description},
                ),
            },
            "code_action": code,
            "alternate_action": alternate,
        }

    def _should_ask_jev(self, view: dict[str, Any], best: dict[str, Any], rival: dict[str, Any]) -> bool:
        if getattr(self.ctx, "system1", None) is None or self.jev_future is not None:
            return False
        frame = int(view["frame"])
        if frame - self.last_jev_frame < 8:
            return False
        # Ask on a real boundary: a close MPC contest, or a narrow predicted passage.
        score_gap = abs(_finite(best.get("score")) - _finite(rival.get("score")))
        close = score_gap < 2.0
        urgent = bool(best.get("dead")) != bool(rival.get("dead"))
        crossing = view["phase"] == "crossing"
        return close or urgent or crossing

    def _start_jev(self, candidate: dict[str, Any], view: dict[str, Any]) -> None:
        payload = self._jev_prompt(candidate)
        request = {
            "frame": int(view["frame"]),
            "target_key": _target_key(view),
            "target_x": _finite(view["active"]["x"]),
            "code_action": payload["code_action"],
            "alternate_action": payload["alternate_action"],
            "revision": self.revision,
        }

        def worker() -> Any:
            try:
                return self.ctx.decide(payload["state"], payload["questions"], timeout_s=0.42)
            except (TimeoutError, RuntimeError):
                return None

        try:
            self.jev_future = self.ctx.spawn(worker)
            self.jev_request = request
        except (TimeoutError, RuntimeError):
            self.jev_future = None
            self.jev_request = None

    def _poll_jev(self, view: dict[str, Any]) -> str | None:
        future = self.jev_future
        request = self.jev_request
        if future is None or request is None or not future.done():
            return None
        self.jev_future = None
        self.jev_request = None
        try:
            answers = future.result()
        except Exception:
            return None
        if not isinstance(answers, dict):
            return None
        # A result is a short-lived interpretation of the state that generated it.
        age = int(view["frame"]) - int(request["frame"])
        if age < 0 or age > 8 or self.revision != request["revision"]:
            self.ctx.log("jev_answer_dropped", reason="stale", age=age, frame=view["frame"])
            return None
        if _target_key(view) != request["target_key"]:
            self.ctx.log("jev_answer_dropped", reason="target_changed", frame=view["frame"])
            return None
        if abs(_finite(view["active"]["x"]) - request["target_x"]) > 36.0:
            self.ctx.log("jev_answer_dropped", reason="horizontal_stale", frame=view["frame"])
            return None
        first, first_p = _answer_value(answers.get("forward"))
        second, second_p = _answer_value(answers.get("reverse"))
        if first not in ("code", "alternate") or second != first:
            self.ctx.log("jev_answer_dropped", reason="no_consensus", frame=view["frame"])
            return None
        if min(first_p, second_p) < 0.60:
            self.ctx.log("jev_answer_dropped", reason="low_probability", frame=view["frame"])
            return None
        self.last_jev_frame = int(view["frame"])
        chosen = request["code_action"] if first == "code" else request["alternate_action"]
        self.ctx.log("jev_consensus", option=first, action=chosen, frame=view["frame"])
        return chosen if chosen in ("IDLE", "FLAP") else None

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        if not getattr(game, "tick_s", None):
            return

        try:
            initial_observation = game.observe()
            initial_view = _view(initial_observation)
            self.last_target_key = _target_key(initial_view)
            self._start_astra(initial_view)

            # Post before the first tick; otherwise the runner's initial IDLE hold
            # consumes the first control frame.
            while not ctx.done and not getattr(game, "done", False):
                self._poll_astra(initial_view)
                self._maybe_commit_mode(initial_view)
                best, rival = _beam_plan(initial_view, self.mode, self.since_flap)
                candidate = _candidate_state(initial_view, best, rival, self.mode, self.revision)
                chosen = self._poll_jev(initial_view)
                if chosen not in ("IDLE", "FLAP"):
                    if self._should_ask_jev(initial_view, best, rival):
                        self._start_jev(candidate, initial_view)
                    chosen = best["actions"][0] if best.get("actions") else "IDLE"
                ctx.act(chosen)
                self.since_flap = 0 if chosen == "FLAP" else self.since_flap + 1
                self.last_target_key = _target_key(initial_view)
                ctx.wait_for_tick()
                if ctx.done or getattr(game, "done", False):
                    break
                current_observation = game.observe()
                initial_view = _view(current_observation)
        except (BudgetExceeded, ModelUnavailable, EpisodeOver):
            # Deadline, game closure, and exhausted budgets are normal episode ends.
            return


def build(ctx: Any) -> Harness:
    return Harness(ctx)
