"""Certified route-planning harness for the real-time Flappy Bird clone.

The controller has two different authorities.  A small forward route beam chooses
an economical pulse schedule using the visible current and following corridors.  A
backward integer reachability certificate is the safety gate: a route suggestion is
never posted unless that one-frame action still has a collision-free continuation.
The game loop itself never waits for either model and explicitly revokes a flap on
the next observed frame.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


ACTIONS = ("IDLE", "FLAP")
V_MIN = -9
V_MAX = 10
DEFAULTS = {
    "width": 288.0,
    "height": 512.0,
    "bird_x": 57.0,
    "bird_w": 34.0,
    "bird_h": 24.0,
    "pipe_w": 52.0,
    "ground": 404.48,
    "speed": 4.0,
    "gravity": 1.0,
    "flap_v": -9.0,
    "max_down": 10.0,
}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def valid_gap(top: float, bottom: float, height: float) -> bool:
    return (
        math.isfinite(top)
        and math.isfinite(bottom)
        and 35.0 <= bottom - top <= max(250.0, height)
        and not (top <= 1.0 and bottom >= height - 1.0)
    )


def parse_pipe(item: Any, height: float, name: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    x = finite(item.get("x"), DEFAULTS["width"])
    top = finite(item.get("gap_top_y", item.get("top_y")), 0.0)
    bottom = finite(item.get("gap_bottom_y", item.get("bottom_y")), height)
    if not valid_gap(top, bottom, height):
        return None
    return {
        "name": text(item.get("name")) or name,
        "x": x,
        "top": top,
        "bottom": bottom,
    }


def view(raw: Any) -> dict[str, Any]:
    """Make a finite public-state view; malformed stand-in observations are safe."""
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            raw = {}
    if not isinstance(raw, dict):
        raw = {}

    geometry = raw.get("geometry") if isinstance(raw.get("geometry"), dict) else {}
    bird = raw.get("bird") if isinstance(raw.get("bird"), dict) else {}
    features = raw.get("features")
    if not isinstance(features, (list, tuple)):
        features = []

    def feature(index: int, fallback: float) -> float:
        return finite(features[index], fallback) if index < len(features) else fallback

    width = max(1.0, finite(geometry.get("screen_width"), DEFAULTS["width"]))
    height = max(1.0, finite(geometry.get("screen_height"), DEFAULTS["height"]))
    bird_x = finite(bird.get("x"), DEFAULTS["bird_x"])
    bird_w = max(1.0, finite(bird.get("width"), DEFAULTS["bird_w"]))
    bird_h = max(1.0, finite(bird.get("height"), DEFAULTS["bird_h"]))
    bird_y = finite(bird.get("y"), feature(9, 244.0))
    bird_v = finite(bird.get("vertical_velocity"), feature(10, -9.0))
    pipe_w = max(1.0, finite(geometry.get("pipe_width"), DEFAULTS["pipe_w"]))
    ground = finite(geometry.get("ground_y"), DEFAULTS["ground"])
    speed = abs(finite(geometry.get("pipe_speed_px_per_frame"), DEFAULTS["speed"]))
    speed = speed or DEFAULTS["speed"]
    gravity = abs(finite(geometry.get("gravity_px_per_frame2"), DEFAULTS["gravity"]))
    gravity = gravity or DEFAULTS["gravity"]
    flap_v = finite(geometry.get("flap_velocity_px_per_frame"), DEFAULTS["flap_v"])
    max_down = abs(finite(geometry.get("max_vertical_velocity"), DEFAULTS["max_down"]))
    max_down = max(DEFAULTS["max_down"], max_down)

    pipes: list[dict[str, Any]] = []
    raw_pipes = raw.get("pipes")
    if isinstance(raw_pipes, list):
        for i, item in enumerate(raw_pipes):
            parsed = parse_pipe(item, height, "pipe_%d" % i)
            if parsed is not None:
                pipes.append(parsed)
    target = parse_pipe(raw.get("target_pipe"), height, "target")
    if target is not None and not any(
        abs(target["x"] - p["x"]) < 1e-6
        and abs(target["top"] - p["top"]) < 1e-6
        and abs(target["bottom"] - p["bottom"]) < 1e-6
        for p in pipes
    ):
        pipes.append(target)

    # The feature vector is an allowed public fallback when a convenience view is
    # absent.  It also makes pre-flight stand-ins with only features harmless.
    if not pipes:
        for i in (0, 3, 6):
            x = feature(i, width)
            top = feature(i + 1, 0.0)
            bottom = feature(i + 2, height)
            if valid_gap(top, bottom, height):
                pipes.append({"name": "feature_%d" % i, "x": x,
                              "top": top, "bottom": bottom})
    pipes.sort(key=lambda p: (p["x"], p["top"], p["bottom"]))

    active = None
    for pipe in pipes:
        if pipe["x"] + pipe_w >= bird_x - 1.0:
            active = pipe
            break
    if active is None and pipes:
        active = pipes[-1]
    if active is None:
        active = {"name": "unknown", "x": width, "top": 0.0, "bottom": height}
    following = [p for p in pipes if p["x"] > active["x"] + 1.0]

    phase = (
        "PASSED" if active["x"] + pipe_w < bird_x
        else "APPROACH" if active["x"] > bird_x + bird_w
        else "CROSSING"
    )
    top_margin = bird_y - active["top"]
    bottom_margin = active["bottom"] - (bird_y + bird_h)
    frame = integer(raw.get("frame"), 0)
    score = finite(raw.get("score", raw.get("pipe_pairs_passed", 0.0)), 0.0)
    done = bool(raw.get("done") or raw.get("terminated") or raw.get("truncated"))
    last_action = raw.get("previous_action", raw.get("action", raw.get("held_action", "IDLE")))
    if not isinstance(last_action, str):
        last_action = "IDLE"
    return {
        "frame": frame,
        "score": score,
        "done": done,
        "last_action": last_action.upper(),
        "width": width,
        "height": height,
        "bird_x": bird_x,
        "bird_y": bird_y,
        "bird_v": bird_v,
        "bird_w": bird_w,
        "bird_h": bird_h,
        "pipe_w": pipe_w,
        "ground": ground,
        "speed": speed,
        "gravity": gravity,
        "flap_v": flap_v,
        "max_down": max_down,
        "pipes": pipes,
        "active": active,
        "following": following,
        "phase": phase,
        "top_margin": top_margin,
        "bottom_margin": bottom_margin,
    }


def target_token(s: dict[str, Any]) -> tuple[Any, ...]:
    p = s["active"]
    # The inferred world-space x keeps two recycled pipes with equal gaps apart,
    # while the rounded gap edges keep harmless subpixel noise out of freshness.
    return (
        round(p["x"] + s["speed"] * s["frame"], 1),
        round(p["top"], 1),
        round(p["bottom"], 1),
    )


def step_y(y: float, velocity: float, action: str, s: dict[str, Any]) -> tuple[float, float]:
    if action == "FLAP":
        velocity = s["flap_v"]
        return y + velocity, velocity
    velocity = min(velocity + s["gravity"], s["max_down"])
    return y + velocity, velocity


def overlap(x: float, s: dict[str, Any]) -> bool:
    return (
        x < s["bird_x"] + s["bird_w"]
        and x + s["pipe_w"] > s["bird_x"]
    )


def pipe_margin(y: float, pipe: dict[str, Any], s: dict[str, Any]) -> float:
    return min(y - pipe["top"], pipe["bottom"] - (y + s["bird_h"]))


def interval_bits(lo: float, hi: float, maximum: int, frac: float) -> int:
    # Bit k represents the integer part k+frac of the bird's top coordinate.
    low = max(0, int(math.ceil(lo - frac - 1e-7)))
    high = min(maximum, int(math.floor(hi - frac + 1e-7)))
    if low > high:
        return 0
    return ((1 << (high - low + 1)) - 1) << low


def shift_predecessor(mask: int, displacement: int, universe: int) -> int:
    """Positions which become a member of mask after adding displacement."""
    if displacement >= 0:
        return mask >> displacement
    return (mask << (-displacement)) & universe


class SafetyCertificate:
    """Backward sets over vertical integer positions and the observed corridors."""

    def __init__(self) -> None:
        self.order = "new_velocity"
        self.matches = 0
        self.mismatches = 0
        self.last: dict[str, Any] | None = None
        self.errors: deque[bool] = deque(maxlen=96)

    def verify(self, s: dict[str, Any]) -> str | None:
        old = self.last
        self.last = s
        if old is None or s["frame"] - old["frame"] != 1:
            return None
        action = text(s.get("last_action", "")).upper()
        if action not in ACTIONS:
            return None
        if old["bird_y"] < 1.0 or s["bird_y"] + s["bird_h"] >= s["ground"] - 1.0:
            return None
        expected_v = s["flap_v"] if action == "FLAP" else min(old["bird_v"] + old["gravity"], old["max_down"])
        expected_y = old["bird_y"] + expected_v
        error = abs(expected_y - s["bird_y"]) + abs(expected_v - s["bird_v"])
        if error < 0.08:
            self.matches += 1
            self.errors.append(False)
            return "verified" if self.matches == 12 else None
        self.mismatches += 1
        self.errors.append(True)
        return "mismatch"

    def _horizon(self, s: dict[str, Any]) -> int:
        # Include the visible following pipe and a post-passage coast.  Unknown
        # offscreen placeholders are intentionally absent from s["pipes"].
        distances = [
            (p["x"] + s["pipe_w"] - s["bird_x"]) / max(s["speed"], 1.0)
            for p in s["pipes"]
            if p["x"] + s["pipe_w"] >= s["bird_x"] - 4.0
        ]
        if not distances:
            distances = [42.0]
        return max(24, min(104, int(math.ceil(max(distances))) + 18))

    def allowed(self, s: dict[str, Any], margin: float) -> tuple[tuple[str, ...], int]:
        frac = s["bird_y"] - math.floor(s["bird_y"])
        yi = int(math.floor(s["bird_y"]))
        maximum = int(math.floor(s["ground"] - s["bird_h"] - 1.0 - frac))
        velocity = int(round(s["bird_v"]))
        if (
            maximum < 1 or maximum > 2048 or yi < 0 or yi > maximum
            or velocity < V_MIN or velocity > V_MAX
            or abs(s["bird_v"] - velocity) > 0.08
        ):
            return (), 0

        horizon = self._horizon(s)
        universe = (1 << (maximum + 1)) - 1
        safe: list[int] = []
        for t in range(horizon + 1):
            low = 2.0
            high = s["ground"] - s["bird_h"] - 4.0
            for pipe in s["pipes"]:
                x = pipe["x"] - s["speed"] * t
                if overlap(x, s):
                    low = max(low, pipe["top"] + margin)
                    high = min(high, pipe["bottom"] - s["bird_h"] - margin)
            safe.append(interval_bits(low, high, maximum, frac))

        # At the final boundary, permit only a position from which a rising bird
        # can coast away from the ceiling.  This avoids a legal-looking flap chain
        # that exits the known corridor through the top edge.
        future = [safe[horizon] for _ in range(V_MAX - V_MIN + 1)]
        for v in range(V_MIN, 0):
            rise = ((-v) * (-v - 1)) / 2.0
            future[v - V_MIN] &= interval_bits(2.0 + rise, maximum + frac, maximum, frac)

        for t in range(horizon - 1, 0, -1):
            band = safe[t]
            current: list[int] = []
            for v in range(V_MIN, V_MAX + 1):
                next_v = min(v + 1, V_MAX)
                idle_dy = next_v
                idle = shift_predecessor(future[next_v - V_MIN], idle_dy, universe)
                flap = shift_predecessor(future[-9 - V_MIN], -9, universe)
                current.append((idle | flap) & band)
            future = current

        bit = 1 << yi
        choices: list[str] = []
        next_v = min(velocity + 1, V_MAX)
        next_y_idle = yi + next_v
        if 0 <= next_y_idle <= maximum and (future[next_v - V_MIN] & (1 << next_y_idle)):
            choices.append("IDLE")
        next_y_flap = yi - 9
        if 0 <= next_y_flap <= maximum and (future[-9 - V_MIN] & (1 << next_y_flap)):
            choices.append("FLAP")
        return tuple(choices), horizon


def gap_direction(s: dict[str, Any]) -> str:
    if not s["following"]:
        return "UNKNOWN"
    difference = s["following"][0]["top"] - s["active"]["top"]
    return "HIGHER" if difference < -12.0 else "LOWER" if difference > 12.0 else "LEVEL"


def local_preference(s: dict[str, Any], posture: str, since_flap: int) -> str:
    p = s["active"]
    center = (p["top"] + p["bottom"] - s["bird_h"]) / 2.0
    # A slightly-low anchor leaves room for the next upward impulse.  The named
    # postures alter only this preference; the certificate remains authoritative.
    anchor = center + (12.0 if posture == "TRANSITION" else 20.0)
    if posture == "PATIENT":
        anchor += 7.0
    if s["bird_y"] < 4.0:
        return "IDLE"
    urgent = s["bird_y"] + s["bird_h"] > s["ground"] - 34.0
    if since_flap < 3 and not urgent:
        return "IDLE"
    if s["bird_v"] >= 0.0 and s["bird_y"] >= anchor:
        return "FLAP"
    if posture == "TRANSITION" and s["bird_y"] > anchor + 24.0 and s["bird_v"] >= -2.0:
        return "FLAP"
    if urgent:
        return "FLAP"
    return "IDLE"


def beam_horizon(s: dict[str, Any]) -> int:
    distances = [
        (p["x"] + s["pipe_w"] - s["bird_x"]) / max(s["speed"], 1.0)
        for p in s["pipes"] if p["x"] + s["pipe_w"] >= s["bird_x"] - 4.0
    ]
    return max(18, min(78, int(math.ceil(max(distances or [40.0]))) + 10))


def beam_plan(s: dict[str, Any], posture: str, since_flap: int) -> dict[str, Any] | None:
    """Forward route beam: use following geometry, but not as a safety oracle."""
    horizon = beam_horizon(s)
    initial = {
        "y": s["bird_y"], "v": s["bird_v"], "xs": tuple(p["x"] for p in s["pipes"]),
        "actions": [], "since": max(3, int(since_flap)), "score": 0.0,
        "worst": 1000.0, "dead": False,
    }
    beam = [initial]
    for t in range(1, horizon + 1):
        expanded: list[dict[str, Any]] = []
        for node in beam:
            for action in ACTIONS:
                since = int(node["since"])
                urgent = node["y"] + s["bird_h"] > s["ground"] - 30.0
                if action == "FLAP" and since < 3 and not urgent:
                    continue
                y, v = step_y(node["y"], node["v"], action, s)
                xs = tuple(x - s["speed"] for x in node["xs"])
                dead = y + s["bird_h"] >= s["ground"]
                worst = float(node["worst"])
                value = float(node["score"])
                for i, pipe in enumerate(s["pipes"]):
                    if i >= len(xs) or not overlap(xs[i], s):
                        continue
                    clearance = pipe_margin(y, pipe, s)
                    worst = min(worst, clearance)
                    if clearance < 0.0:
                        dead = True
                        value -= 20000.0
                        break
                    value += min(24.0, clearance) * (0.20 if i == 0 else 0.12)
                if dead:
                    continue
                # The next corridor matters before it overlaps.  Its centre is a
                # route objective, never a replacement for rectangle collision.
                if s["following"]:
                    next_pipe = s["following"][0]
                    next_i = next((i for i, p in enumerate(s["pipes"])
                                   if p is next_pipe), len(s["pipes"]) - 1)
                    if next_i < len(xs):
                        desired = (next_pipe["top"] + next_pipe["bottom"] - s["bird_h"]) / 2.0
                        horizontal = abs(xs[next_i] + s["pipe_w"] / 2.0
                                         - (s["bird_x"] + s["bird_w"] / 2.0))
                        if horizontal < 150.0:
                            value -= abs(y - desired) * 0.014
                if y < 0.0:
                    value -= min(18.0, -y) * 0.03
                if action == "FLAP":
                    value -= 0.045
                if posture == "TRANSITION" and s["following"]:
                    value += (s["bird_y"] - y) * 0.003
                expanded.append({
                    "y": y, "v": v, "xs": xs,
                    "actions": list(node["actions"]) + [action],
                    "since": 0 if action == "FLAP" else since + 1,
                    "score": value, "worst": worst, "dead": False,
                })
        if not expanded:
            break
        expanded.sort(key=lambda n: (n["score"], n["worst"]), reverse=True)
        # Keep both first-action families alive so a later tie-breaker has a real
        # alternative.  Diversity by vertical bin avoids a single oscillating arc.
        kept: list[dict[str, Any]] = []
        for first in ACTIONS:
            family = [n for n in expanded if n["actions"] and n["actions"][0] == first]
            seen_bins: set[tuple[int, int]] = set()
            for n in family:
                key = (int(round(n["y"] / 8.0)), int(round(n["v"])))
                if key in seen_bins:
                    continue
                seen_bins.add(key)
                kept.append(n)
                if len(seen_bins) >= 8:
                    break
        known = {id(n) for n in kept}
        for n in expanded:
            if id(n) not in known:
                kept.append(n)
            if len(kept) >= 32:
                break
        beam = kept[:32]

    if not beam:
        return None
    beam.sort(key=lambda n: (n["score"], n["worst"]), reverse=True)
    best = beam[0]
    return {
        "first": best["actions"][0] if best["actions"] else "IDLE",
        "actions": best["actions"],
        "score": finite(best["score"]),
        "worst": finite(best["worst"], -1.0),
        "horizon": horizon,
    }


def route_label(s: dict[str, Any], plan: dict[str, Any] | None, allowed: tuple[str, ...]) -> dict[str, Any]:
    current = gap_direction(s)
    if current == "HIGHER":
        transition = "PREPARE_UP"
    elif current == "LOWER":
        transition = "PRESERVE_HEIGHT"
    else:
        transition = "STAY_CENTERED"
    return {
        "phase": s["phase"],
        "next_gap": current,
        "transition_intent": transition,
        "route_first_action": (plan or {}).get("first", "IDLE"),
        "route_margin_class": (
            "EDGE" if (plan or {}).get("worst", -1.0) < 8.0
            else "TIGHT" if (plan or {}).get("worst", -1.0) < 20.0
            else "WIDE"
        ),
        "safe_actions": list(allowed),
    }


def answer_choice(answer: Any) -> tuple[str | None, float, float | None]:
    value = getattr(answer, "value", None)
    probs = getattr(answer, "probabilities", None)
    if not isinstance(value, str) or not isinstance(probs, dict):
        return None, 0.0, None
    probability = finite(probs.get(value), 0.0)
    confidence = getattr(answer, "confidence", None)
    confidence_value = None if confidence is None else finite(confidence, -1.0)
    return value, probability, confidence_value


def answer_noul(answer: Any) -> float:
    value = getattr(answer, "value", 0.0)
    return max(0.0, min(1.0, finite(value, 0.0)))


def bounded_json(text_value: Any) -> dict[str, Any] | None:
    if not isinstance(text_value, str):
        return None
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text_value[:10000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text_value[pos:])
        except (TypeError, ValueError):
            continue
        return value if isinstance(value, dict) else None
    return None


def parse_astra(text_value: Any) -> str | None:
    value = text(text_value).strip().upper().strip("`'\" ")
    match = re.fullmatch(r"(BALANCED|TRANSITION|CAUTIOUS)[.!]?", value)
    return match.group(1) if match else None


class Harness:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.cert = SafetyCertificate()
        self.posture = "BALANCED"
        self.pending_posture: str | None = None
        self.revision = 0
        self.astra_future = None
        self.jev_future = None
        self.jev_tag: dict[str, Any] | None = None
        self.astra_started = False
        self.last_jev_frame = -1000
        self.last_jev_request_frame = -1000
        self.jev_intent: str | None = None
        self.jev_intent_frame = -1000
        self.jev_intent_target: tuple[Any, ...] | None = None
        self.jev_intent_revision = -1
        self.last_replan_frame = -1000
        self.last_token: tuple[Any, ...] | None = None
        self.route: dict[str, Any] | None = None
        self.since_flap = 100
        self.safe_relaxations = 0
        self.route_decisions = 0
        self.accepted_jev = 0
        self.rejected_jev = 0
        self.astra_session: str | None = None
        self.last_posted = "IDLE"
        self.last_frame = -1

    def _mode_for(self) -> str:
        if self.posture == "CAUTIOUS":
            return "PATIENT"
        if self.posture == "TRANSITION":
            return "TRANSITION"
        return "BALANCED"

    def _start_astra(self, s: dict[str, Any]) -> None:
        if self.astra_started or getattr(self.ctx, "system2", True) is None:
            return
        self.astra_started = True
        opening = {
            "frame": s["frame"], "bird_y": finite(s["bird_y"]),
            "bird_motion": "RISING" if s["bird_v"] < 0 else "FALLING",
            "gap_top": finite(s["active"]["top"]),
            "gap_bottom": finite(s["active"]["bottom"]),
            "following_direction": gap_direction(s),
        }
        prompt = (
            "You are an episode-level supervisor for a real-time symbolic Flappy Bird clone. "
            "A code controller owns all arithmetic, collision checks, and one-frame actuation. "
            "Choose exactly one bounded posture token: BALANCED, TRANSITION, or CAUTIOUS. "
            "BALANCED follows the certified route beam. TRANSITION gives more weight to the "
            "following visible gap. CAUTIOUS prefers the local centre rule. Never propose an "
            "action, seed-specific opening, new physics, or code. Return only the token. The "
            "certificate is always authoritative and the world keeps ticking while you think. "
            "Opening facts: " + json.dumps(opening, separators=(",", ":"))
        )

        def worker() -> Any:
            try:
                return self.ctx.think(prompt, timeout_s=90.0)
            except Exception:
                return None

        try:
            self.astra_future = self.ctx.spawn(worker)
        except Exception:
            self.astra_future = None

    def _poll_astra(self, s: dict[str, Any]) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
        except Exception:
            return
        self.astra_session = getattr(result, "session_id", None)
        parsed = parse_astra(getattr(result, "text", ""))
        if parsed is None or parsed == self.posture:
            return
        self.pending_posture = parsed
        self.ctx.log("astra_posture_ready", posture=parsed, frame=s["frame"])

    def _commit_posture(self, s: dict[str, Any]) -> None:
        if self.pending_posture is None:
            return
        token = target_token(s)
        if self.last_token is None or token != self.last_token or s["frame"] - self.last_replan_frame >= 18:
            self.posture = self.pending_posture
            self.pending_posture = None
            self.revision += 1
            self.route = None
            self.ctx.log("astra_posture_committed", posture=self.posture,
                         revision=self.revision, frame=s["frame"])

    def _questions(self) -> dict[str, Any]:
        route = {
            "route": {
                "what": "Use the forward route beam's first action, but only if it is in safe_actions.",
                "not_for": "Do not use it when the route label is EDGE or safe_actions excludes it.",
            },
            "protect": {
                "what": "Use the local centre-and-velocity pulse rule, filtered by safe_actions.",
                "not_for": "Do not invent a flap or override the certificate.",
            },
        }
        return {
            "intent_forward": Choice(
                "Choose one control intent for the next frame. Read route.route_margin_class, "
                "route.transition_intent, and certificate.safe_actions. ROUTE is appropriate "
                "only when the route beam is not EDGE; PROTECT is the conservative local rule. "
                "Do not calculate numbers and do not output an action.",
                route,
            ),
            "intent_reverse": Choice(
                "Independently choose one control intent for the next frame. Read route.route_margin_class, "
                "route.transition_intent, and certificate.safe_actions. ROUTE is appropriate "
                "only when the route beam is not EDGE; PROTECT is the conservative local rule. "
                "Do not calculate numbers and do not output an action.",
                dict(reversed(list(route.items()))),
            ),
            "transition_check": Noul(
                "Does the supplied route state explicitly support preparing for the following gap "
                "rather than only holding the current corridor? This is advisory and cannot make "
                "an unsafe action legal.",
                {"true": "The following gap direction is HIGHER or LOWER and the route is not EDGE.",
                 "false": "The following gap is unknown, level, or the route is EDGE."},
            ),
        }

    def _jev_state(self, s: dict[str, Any], allowed: tuple[str, ...], plan: dict[str, Any] | None) -> dict[str, Any]:
        direction = gap_direction(s)
        motion = "RISING" if s["bird_v"] < 0 else "FALLING_OR_LEVEL"
        return {
            "game": "flappy_bird",
            "frame": int(s["frame"]),
            "plan": {"posture": self.posture, "revision": self.revision,
                     "owner": "route_beam_plus_certificate"},
            "bird": {"motion": motion,
                     "vertical_band": "HIGH" if s["bird_y"] < s["active"]["top"] + 20.0
                     else "LOW" if s["bird_y"] + s["bird_h"] > s["active"]["bottom"] - 20.0
                     else "INTERIOR",
                     "top_margin": finite(s["top_margin"]),
                     "bottom_margin": finite(s["bottom_margin"])},
            "route": {
                "phase": s["phase"], "following_direction": direction,
                "transition_intent": ("PREPARE_UP" if direction == "HIGHER"
                                       else "PRESERVE_HEIGHT" if direction == "LOWER"
                                       else "STAY_CENTERED"),
                "route_first_action": (plan or {}).get("first", "IDLE"),
                "route_margin_class": ("EDGE" if (plan or {}).get("worst", -1.0) < 8.0
                                        else "TIGHT" if (plan or {}).get("worst", -1.0) < 20.0
                                        else "WIDE"),
            },
            "certificate": {"safe_actions": list(allowed),
                            "safety_is_non_overrideable": True,
                            "relaxation_level": self.safe_relaxations},
            "score": finite(s["score"]),
        }

    def _poll_jev(self, s: dict[str, Any]) -> None:
        future = self.jev_future
        tag = self.jev_tag
        if future is None or tag is None or not future.done():
            return
        self.jev_future = None
        self.jev_tag = None
        try:
            answers = future.result()
        except Exception:
            self.rejected_jev += 1
            return
        if not isinstance(answers, dict):
            self.rejected_jev += 1
            return
        age = s["frame"] - int(tag["frame"])
        fresh = (
            0 <= age <= 12
            and self.revision == tag["revision"]
            and target_token(s) == tag["target"]
            and abs(s["active"]["x"] - tag["x"]) <= 48.0
        )
        a, ap, ac = answer_choice(answers.get("intent_forward"))
        b, bp, bc = answer_choice(answers.get("intent_reverse"))
        threshold_a = 0.82 if ac is None else 0.62
        threshold_b = 0.82 if bc is None else 0.62
        consensus = a in ("route", "protect") and a == b
        confident = ap >= threshold_a and bp >= threshold_b
        if not (fresh and consensus and confident):
            self.rejected_jev += 1
            self.ctx.log("jev_intent_discarded", fresh=fresh, age=age,
                         consensus=consensus, frame=s["frame"])
            return
        self.accepted_jev += 1
        self.last_jev_frame = s["frame"]
        self.jev_intent = "ROUTE" if a == "route" else "PROTECT"
        self.jev_intent_frame = s["frame"]
        self.jev_intent_target = target_token(s)
        self.jev_intent_revision = self.revision
        transition = answer_noul(answers.get("transition_check")) >= 0.65
        self.ctx.memory["last_jev_transition"] = transition
        self.ctx.log("jev_intent_accepted", intent=self.jev_intent,
                     transition=transition, age=age, frame=s["frame"])

    def _start_jev(self, s: dict[str, Any], allowed: tuple[str, ...], plan: dict[str, Any] | None) -> None:
        if self.jev_future is not None or getattr(self.ctx, "system1", True) is None:
            return
        remaining = self.ctx.budget.remaining()
        if remaining.get("system1_calls", 0) <= 20 or remaining.get("wall_s", 0.0) < 2.0:
            return
        frame = s["frame"]
        if frame - self.last_jev_request_frame < 16:
            return
        state = self._jev_state(s, allowed, plan)
        questions = self._questions()
        tag = {"frame": frame, "revision": self.revision,
               "target": target_token(s), "x": s["active"]["x"]}

        def worker() -> Any:
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception:
                return None

        try:
            self.jev_future = self.ctx.spawn(worker)
            self.jev_tag = tag
            self.last_jev_request_frame = frame
        except Exception:
            self.jev_future = None
            self.jev_tag = None

    def _choose(self, s: dict[str, Any], allowed: tuple[str, ...], plan: dict[str, Any] | None) -> str:
        intent = self.jev_intent
        intent_fresh = (
            intent in ("ROUTE", "PROTECT")
            and 0 <= s["frame"] - self.jev_intent_frame <= 12
            and self.jev_intent_revision == self.revision
            and self.jev_intent_target == target_token(s)
        )
        if intent_fresh and intent == "ROUTE" and plan is not None:
            desired = plan.get("first", "IDLE")
        elif intent_fresh and intent == "PROTECT":
            desired = local_preference(s, self._mode_for(), self.since_flap)
        else:
            desired = local_preference(s, self._mode_for(), self.since_flap)
        if desired in allowed:
            return desired
        if allowed:
            # Keep IDLE as the pulse-safe tie-breaker; use FLAP only when it is
            # the sole certified continuation.
            return "IDLE" if "IDLE" in allowed else allowed[0]
        # No certificate at all means the integer model is outside its assumptions.
        # This is bounded recovery, not a claim that the trajectory is safe.
        recovery = local_preference(s, "TRANSITION", self.since_flap)
        if s["bird_y"] < 8.0:
            recovery = "IDLE"
        if s["bird_y"] + s["bird_h"] > s["ground"] - 38.0:
            recovery = "FLAP"
        return recovery if recovery in ACTIONS else "IDLE"

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        try:
            if not getattr(game, "tick_s", 0.0):
                return
            first = view(ctx.observe())
            self._start_astra(first)
            self.last_token = target_token(first)
            # This initial post prevents the runner's implicit IDLE from consuming
            # a frame before the harness has installed its explicit held action.
            ctx.act("IDLE")
            self.last_posted = "IDLE"

            while not ctx.done and not getattr(game, "done", False):
                current = view(ctx.observe())
                if current["done"]:
                    break
                if current["frame"] <= self.last_frame:
                    ctx.wait_for_tick()
                    continue
                self.last_frame = current["frame"]
                verification = self.cert.verify(current)
                if verification is not None:
                    ctx.log("physics_" + verification, frame=current["frame"],
                            matches=self.cert.matches, mismatches=self.cert.mismatches,
                            order=self.cert.order)

                # A FLAP is a pulse.  Revoke it before polling models or running a
                # planner, so a slow answer can never turn it into a held impulse.
                if self.last_posted == "FLAP":
                    ctx.act("IDLE")
                    self.last_posted = "IDLE"

                self._poll_astra(current)
                self._commit_posture(current)
                self._poll_jev(current)

                token = target_token(current)
                if self.route is None or token != self.last_token or current["frame"] - self.last_replan_frame >= 8:
                    self.route = beam_plan(current, self._mode_for(), self.since_flap)
                    self.last_replan_frame = current["frame"]
                    self.last_token = token
                    self.route_decisions += 1
                allowed: tuple[str, ...] = ()
                horizon = 0
                used_margin = 10.0 if self.posture != "TRANSITION" else 9.0
                for margin in (used_margin, 6.0, 2.0, 0.0):
                    allowed, horizon = self.cert.allowed(current, margin)
                    if allowed:
                        used_margin = margin
                        break
                if used_margin < (10.0 if self.posture != "TRANSITION" else 9.0):
                    self.safe_relaxations += 1

                self._start_jev(current, allowed, self.route)
                action = self._choose(current, allowed, self.route)

                # If the planner consumed a frame while the world ticked, do not
                # publish its old action.  The already-posted IDLE is the fallback.
                latest = view(ctx.observe())
                if latest["frame"] != current["frame"]:
                    self.last_posted = "IDLE"
                    continue
                ctx.act(action)
                self.last_posted = action
                self.since_flap = 0 if action == "FLAP" else self.since_flap + 1
                if current["frame"] == 0 or current["score"] != finite(ctx.memory.get("last_score"), -1.0):
                    ctx.memory["last_score"] = current["score"]
                    ctx.log("route_controller", frame=current["frame"], score=current["score"],
                            action=action, allowed=list(allowed), margin=used_margin,
                            route_first=(self.route or {}).get("first", "IDLE"),
                            route_worst=(self.route or {}).get("worst", -1.0),
                            certificate_horizon=horizon)
                if not ctx.done:
                    ctx.wait_for_tick()

        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return
        finally:
            ctx.memory["candidate_0007"] = {
                "physics_matches": self.cert.matches,
                "physics_mismatches": self.cert.mismatches,
                "route_decisions": self.route_decisions,
                "safe_relaxations": self.safe_relaxations,
                "accepted_jev": self.accepted_jev,
                "rejected_jev": self.rejected_jev,
                "posture": self.posture,
            }


def build(ctx: Any) -> Harness:
    return Harness(ctx)
