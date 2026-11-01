"""Receding-horizon Asteroids controller with model-selected escape lanes.

The controller is deliberately split into three layers.  Code makes short
trajectory estimates and owns every native frame.  Jev chooses between those
already-scored manoeuvre packets, rather than naming an action.  Astra is a
slow author of the objective weights only; it is never on the clock-critical
path.
"""
from __future__ import annotations

import json
import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


WIDTH = 160.0
HEIGHT = 210.0
PI = math.pi


# ---------- defensive, JSON-safe helpers ----------


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def integer(value: Any, default: int = 0) -> int:
    number = finite(value)
    if number is None:
        return default
    try:
        return int(number)
    except (TypeError, ValueError):
        return default


def mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def observation_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def wrap_delta(value: float, period: float) -> float:
    # The public screen dimensions are the only wrap periods used here.
    half = period / 2.0
    while value > half:
        value -= period
    while value < -half:
        value += period
    return value


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def answer_value(answer: Any) -> Any:
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def answer_prob(answer: Any, option: str) -> float:
    if isinstance(answer, dict):
        probs = answer.get("probabilities", {})
    else:
        probs = getattr(answer, "probabilities", {})
    if not isinstance(probs, dict):
        return 0.0
    value = finite(probs.get(option), 0.0)
    return max(0.0, min(1.0, value or 0.0))


def extract_object(text: Any) -> dict | None:
    """Find the first JSON object in arbitrary Astra output."""
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def heading_vector(code: int) -> tuple[float, float]:
    """Convert the documented 16-code compass to a unit vector."""
    angle = (code % 16) * PI / 8.0
    return -math.sin(angle), -math.cos(angle)


def vector_heading(dx: float, dy: float) -> int:
    if abs(dx) + abs(dy) < 1e-6:
        return 0
    return int(round(math.atan2(-dx, -dy) / (PI / 8.0))) % 16


def shortest_turn(current: int, desired: int, default: int = 1) -> int:
    delta = (desired - current) % 16
    if delta == 0:
        return 0
    if delta < 8:
        return 1  # LEFT increments orientation_code.
    if delta > 8:
        return -1  # RIGHT decrements orientation_code.
    return default


def band(value: float | None, low: float, high: float) -> str:
    if value is None:
        return "unknown"
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "middle"


def bucket_clearance(value: float) -> str:
    if value < 0.0:
        return "overlap"
    if value < 8.0:
        return "tight"
    if value < 22.0:
        return "narrow"
    if value < 50.0:
        return "open"
    return "wide"


def bucket_time(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 8.0:
        return "imminent"
    if value <= 18.0:
        return "soon"
    if value <= 36.0:
        return "later"
    return "not_closing"


class LanePlanner:
    """Owns observation arithmetic, packet simulation, and frame execution."""

    ROUTES = (
        "hold",
        "scan_left",
        "scan_right",
        "escape_left",
        "escape_right",
        "counterdrift",
        "defend",
    )

    def __init__(self, game: Any, ctx: Any):
        self.game = game
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        advertised = getattr(spec, "names", ()) if spec is not None else ()
        self.names = set(advertised or ())
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else (next(iter(self.names), "NOOP"))

        self.frame = -1
        self.lives = 0
        self.previous_lives: int | None = None
        self.score = 0.0
        self.facts: dict = {}
        self.routes: dict[str, dict] = {}
        self.intent = "hold"
        self.intent_until = -1
        self.intent_life: int | None = None
        self.packet = ""
        self.packet_start = -1
        self.packet_until = -1
        self.fire_next = 0
        self.defense_used = False
        self.last_defense = -100000
        self.respawn_until = -1
        self.sweep_sign = 1
        self.policy = {
            "objective": "safety_first",
            "sweep": "alternate",
            "defense": "last_resort",
        }
        self.policy_version = 0
        self.deaths = 0
        self.last_action = self.noop
        self.last_log_frame = -100000

    # ---- observations and derived geometry ----

    def legal(self, name: str, fallback: str | None = None) -> str:
        if name in self.names:
            return name
        if fallback and fallback in self.names:
            return fallback
        if self.noop in self.names:
            return self.noop
        return next(iter(self.names), self.noop)

    def _player(self, obs: dict) -> dict:
        value = obs.get("player")
        return value if isinstance(value, dict) else {}

    def _rocks(self, obs: dict, player: dict) -> list[dict]:
        if not bool(player.get("present")):
            return []
        px = finite(player.get("x"), WIDTH / 2.0) or WIDTH / 2.0
        py = finite(player.get("y"), HEIGHT / 2.0) or HEIGHT / 2.0
        pw = finite(player.get("width"), 6.0) or 6.0
        ph = finite(player.get("height"), 10.0) or 10.0
        pvelocity = player.get("velocity_px_per_frame")
        pvx = finite(pvelocity[0]) if isinstance(pvelocity, list) and len(pvelocity) >= 2 else None
        pvy = finite(pvelocity[1]) if isinstance(pvelocity, list) and len(pvelocity) >= 2 else None
        raw_rocks = obs.get("asteroids")
        if not isinstance(raw_rocks, list):
            return []
        result = []
        for index, raw in enumerate(raw_rocks):
            rock = mapping(raw)
            rx = finite(rock.get("x"))
            ry = finite(rock.get("y"))
            if rx is None or ry is None:
                continue
            rw = finite(rock.get("width"), 8.0) or 8.0
            rh = finite(rock.get("height"), 15.0) or 15.0
            dx = wrap_delta((rx + rw / 2.0) - (px + pw / 2.0), WIDTH)
            dy = wrap_delta((ry + rh / 2.0) - (py + ph / 2.0), HEIGHT)
            velocity = rock.get("velocity_px_per_frame")
            if isinstance(velocity, list) and len(velocity) >= 2:
                avx = finite(velocity[0])
                avy = finite(velocity[1])
            else:
                avx = avy = None
            rvx = avx - pvx if avx is not None and pvx is not None else None
            rvy = avy - pvy if avy is not None and pvy is not None else None
            size = str(rock.get("size") or "unknown").lower()
            # Approximate boxes are intentionally inflated for a safety planner.
            rock_radius = max(rw, rh) / 2.0
            player_radius = max(pw, ph) / 2.0
            clearance_radius = rock_radius + player_radius + 4.0
            distance = math.hypot(dx, dy)
            if rvx is not None and rvy is not None:
                vv = rvx * rvx + rvy * rvy
                ttc = clamp(-(dx * rvx + dy * rvy) / vv, 0.0, 36.0) if vv > 1e-8 else None
                closest = math.hypot(dx + rvx * (ttc or 0.0), dy + rvy * (ttc or 0.0))
            else:
                ttc = None
                closest = distance
            result.append({
                "slot": integer(rock.get("source_slot"), index),
                "size": size,
                "dx": dx,
                "dy": dy,
                "vx": rvx,
                "vy": rvy,
                "known_velocity": rvx is not None and rvy is not None,
                "distance": distance,
                "ttc": ttc,
                "closest": closest,
                "radius": clearance_radius,
            })
        return result

    def update(self, obs_value: Any) -> None:
        obs = observation_dict(obs_value)
        player = self._player(obs)
        self.frame = integer(obs.get("frame"), self.frame + 1)
        self.score = finite(obs.get("score"), self.score) or 0.0
        self.lives = integer(obs.get("lives"), self.lives)
        if self.previous_lives is not None and self.lives != self.previous_lives:
            if self.lives < self.previous_lives:
                self.deaths += self.previous_lives - self.lives
            self.intent = "hold"
            self.intent_until = self.frame - 1
            self.packet = ""
            self.packet_start = -1
            self.packet_until = -1
            self.defense_used = False
            self.last_defense = self.frame
            self.respawn_until = self.frame + 36
            self.fire_next = self.frame + 10
        self.previous_lives = self.lives

        present = bool(player.get("present"))
        px = finite(player.get("x")) if present else None
        py = finite(player.get("y")) if present else None
        pv = player.get("velocity_px_per_frame")
        pvx = finite(pv[0]) if isinstance(pv, list) and len(pv) >= 2 else None
        pvy = finite(pv[1]) if isinstance(pv, list) and len(pv) >= 2 else None
        heading = integer(player.get("orientation_code"), 0) % 16 if present else None
        rocks = self._rocks(obs, player)

        immediate = []
        approaching = []
        for rock in rocks:
            now_clearance = rock["distance"] - rock["radius"]
            future_clearance = rock["closest"] - rock["radius"]
            if now_clearance <= 4.0 or (
                rock["ttc"] is not None and rock["ttc"] <= 14.0 and future_clearance <= 10.0
            ):
                immediate.append(rock)
            elif now_clearance <= 22.0 or (
                rock["ttc"] is not None and rock["ttc"] <= 32.0 and future_clearance <= 24.0
            ):
                approaching.append(rock)

        if not present:
            danger = "unknown"
        elif immediate:
            danger = "emergency"
        elif approaching:
            danger = "approaching"
        elif rocks:
            danger = "clear"
        else:
            danger = "unknown"

        nearest_pool = immediate or approaching or rocks
        nearest = min(
            nearest_pool,
            key=lambda item: (item["closest"], item["distance"]),
            default=None,
        )
        missiles = obs.get("player_missiles")
        missile_count = len(missiles) if isinstance(missiles, list) else 0
        speed = math.hypot(pvx or 0.0, pvy or 0.0) if present else None
        center_dx = wrap_delta(WIDTH / 2.0 - (px or WIDTH / 2.0), WIDTH) if present else 0.0
        center_dy = wrap_delta(HEIGHT / 2.0 - (py or HEIGHT / 2.0), HEIGHT) if present else 0.0
        self.facts = {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "present": present and px is not None and py is not None,
            "x": px,
            "y": py,
            "vx": pvx,
            "vy": pvy,
            "speed": speed,
            "heading": heading,
            "rocks": rocks,
            "danger": danger,
            "nearest": nearest,
            "immediate_count": len(immediate),
            "approaching_count": len(approaching),
            "rock_count": len(rocks),
            "missile_count": missile_count,
            "center_dx": center_dx,
            "center_dy": center_dy,
            "motion_break": bool(player.get("motion_discontinuity")) if present else True,
            "enemy_unknown": True,
        }
        self.routes = self._make_routes(self.facts)

    def _route_path(self, facts: dict, kind: str) -> tuple[float, float, float, int]:
        """Return minimum predicted clearance and time for a short packet."""
        if not facts.get("present"):
            return 999.0, 0.0, 0.0, 0
        heading = integer(facts.get("heading"), 0) % 16
        x = y = 0.0
        vx = finite(facts.get("vx"), 0.0) or 0.0
        vy = finite(facts.get("vy"), 0.0) or 0.0
        min_clearance = 999.0
        min_time = 0.0
        # Include the current geometry: a route must not look safe merely because
        # its first simulated frame happens after an already-overlapping snapshot.
        for rock in facts.get("rocks", ()):
            separation = math.hypot(
                wrap_delta(rock["dx"] - x, WIDTH),
                wrap_delta(rock["dy"] - y, HEIGHT),
            )
            clearance = separation - rock["radius"]
            if clearance < min_clearance:
                min_clearance = clearance
                min_time = 0.0
        horizon = 28
        if kind in {"escape_left", "escape_right", "counterdrift"}:
            horizon = 20
        turn_sign = {
            "scan_left": 1,
            "scan_right": -1,
            "escape_left": 1,
            "escape_right": -1,
        }.get(kind, 0)
        desired_counter = None
        if kind == "counterdrift" and abs(vx) + abs(vy) > 0.15:
            desired_counter = vector_heading(-vx, -vy)

        for step in range(1, horizon + 1):
            # The cartridge turns in coarse increments; this is a deliberately
            # conservative packet estimate, not a claim about intermediate angles.
            if turn_sign and step % 4 == 0:
                heading = (heading + turn_sign) % 16
            thrust = kind in {"escape_left", "escape_right"}
            if kind == "counterdrift":
                turn = shortest_turn(heading, desired_counter or heading)
                if turn and step % 4 == 0:
                    heading = (heading + turn) % 16
                thrust = abs((desired_counter or heading) - heading) <= 1 or step > 8
            if thrust:
                hx, hy = heading_vector(heading)
                vx = clamp(vx + hx * 0.045, -1.8, 1.8)
                vy = clamp(vy + hy * 0.045, -1.8, 1.8)
            x += vx
            y += vy
            for rock in facts.get("rocks", ()):
                rx = rock["dx"] + ((rock["vx"] or 0.0) * step if rock["known_velocity"] else 0.0)
                ry = rock["dy"] + ((rock["vy"] or 0.0) * step if rock["known_velocity"] else 0.0)
                separation = math.hypot(wrap_delta(rx - x, WIDTH), wrap_delta(ry - y, HEIGHT))
                clearance = separation - rock["radius"]
                if clearance < min_clearance:
                    min_clearance = clearance
                    min_time = float(step)
        # Unknown velocities are uncertainty, not evidence of a safe lane.
        unknown_penalty = 5.0 if any(not r["known_velocity"] for r in facts.get("rocks", ())) else 0.0
        return min_clearance - unknown_penalty, min_time, float(horizon), heading

    def _make_routes(self, facts: dict) -> dict[str, dict]:
        routes: dict[str, dict] = {}
        for route_id in self.ROUTES:
            if route_id == "defend":
                clearance, contact_time, horizon, final_heading = -999.0, 0.0, 1.0, integer(facts.get("heading"), 0)
            else:
                clearance, contact_time, horizon, final_heading = self._route_path(facts, route_id)
            valid = True
            if route_id == "hold" and facts.get("danger") == "emergency":
                valid = False
            if route_id == "counterdrift" and (facts.get("speed") is None or (facts.get("speed") or 0.0) < 0.18):
                valid = False
            if route_id == "defend":
                valid = bool(facts.get("present")) and not self.defense_used
            routes[route_id] = {
                "id": route_id,
                "clearance": clearance,
                "contact_time": contact_time,
                "horizon": horizon,
                "final_heading": final_heading,
                "valid": valid,
                "clearance_bucket": bucket_clearance(clearance),
                "time_bucket": bucket_time(contact_time if contact_time else None),
            }
        return routes

    # ---- state sent to Jev ----

    def _route_description(self, route_id: str, route: dict) -> str:
        purpose = {
            "hold": "Keep the present heading and use only one-frame FIRE pulses; no thrust.",
            "scan_left": "Make a bounded counterclockwise aiming sweep with optional FIRE pulses; no thrust.",
            "scan_right": "Make a bounded clockwise aiming sweep with optional FIRE pulses; no thrust.",
            "escape_left": "Use a bounded counterclockwise turn while thrusting to leave the predicted collision lane; no indefinite hold.",
            "escape_right": "Use a bounded clockwise turn while thrusting to leave the predicted collision lane; no indefinite hold.",
            "counterdrift": "Turn toward the computed opposite-drift heading, then use a short thrust correction.",
            "defend": "Use one mode-0 hyperspace pulse now; only for immediate danger and never as braking.",
        }[route_id]
        status = "currently admissible" if route.get("valid") else "currently rejected by the local safety gate"
        return (
            purpose + " It is " + status + "; the code projected an "
            + route.get("clearance_bucket", "unknown") + " clearance and "
            + route.get("time_bucket", "unknown") + " closest approach."
        )

    def decision_state(self) -> dict:
        facts = self.facts
        heading = facts.get("heading")
        speed = facts.get("speed")
        route_list = []
        for route_id in self.ROUTES:
            route = self.routes.get(route_id, {})
            route_list.append({
                "id": route_id,
                "purpose": self._route_description(route_id, route),
                "clearance": route.get("clearance_bucket", "unknown"),
                "contact": route.get("time_bucket", "unknown"),
                "valid": bool(route.get("valid")),
            })
        return {
            "rules": {
                "mode": 0,
                "fire_requires_release": True,
                "noop_does_not_brake": True,
                "down_is_hyperspace": True,
                "left_increments_orientation": True,
                "right_decrements_orientation": True,
                "enemy_objects_unknown_not_absent": True,
                "packets_are_bounded": True,
            },
            "snapshot": {
                "frame": int(facts.get("frame", 0)),
                "lives": int(facts.get("lives", 0)),
                "score": int(max(0.0, facts.get("score", 0.0))),
                "player_present": bool(facts.get("present")),
                "position": {
                    "x_band": band(facts.get("x"), 55.0, 105.0),
                    "y_band": band(facts.get("y"), 70.0, 140.0),
                    "centrality": "central" if math.hypot(facts.get("center_dx", 0.0), facts.get("center_dy", 0.0)) < 32.0 else "off_center",
                },
                "heading_code": heading,
                "motion": "unknown" if speed is None else "still" if speed < 0.18 else "moving",
                "danger": facts.get("danger", "unknown"),
                "rocks": {
                    "count": int(facts.get("rock_count", 0)),
                    "immediate": int(facts.get("immediate_count", 0)),
                    "approaching": int(facts.get("approaching_count", 0)),
                },
                "missiles_visible": int(facts.get("missile_count", 0)),
                "motion_discontinuity": bool(facts.get("motion_break")),
                "enemy_coverage": "unknown_not_absent",
            },
            "objective": dict(self.policy),
            "current_intent": self.intent,
            "routes": route_list,
        }

    def questions(self) -> dict:
        descriptions = {
            route_id: self._route_description(route_id, self.routes.get(route_id, {}))
            for route_id in self.ROUTES
        }
        order = list(self.ROUTES)
        if (self.frame // 18) % 2:
            order.reverse()
        reverse = list(reversed(order))
        return {
            "route_a": Choice(
                "Choose the safest useful bounded route from snapshot and routes. "
                "Use the local validity and clearance labels; do not calculate numbers, "
                "invent an action, or turn a packet into an open-ended hold.",
                {name: descriptions[name] for name in order},
            ),
            "route_b": Choice(
                "Independently choose the best bounded route for exactly this snapshot. "
                "Immediate danger overrides score; when danger is not immediate, retain "
                "the central firing/sweep objective. Do not infer that unknown enemies are absent.",
                {name: descriptions[name] for name in reverse},
            ),
        }

    def accept_route(self, answers: Any, judged_frame: int, judged_lives: int) -> bool:
        if not isinstance(answers, dict):
            return False
        if self.frame < judged_frame or self.frame - judged_frame > 24 or self.lives != judged_lives:
            return False
        first = answer_value(answers.get("route_a"))
        second = answer_value(answers.get("route_b"))
        if first != second or first not in self.ROUTES:
            return False
        probability = min(answer_prob(answers.get("route_a"), first), answer_prob(answers.get("route_b"), first))
        threshold = 0.48 if first in {"escape_left", "escape_right", "defend", "counterdrift"} else 0.18
        if probability < threshold:
            return False
        current = self.routes.get(first, {})
        if first != "defend" and not current.get("valid", False) and self.facts.get("danger") == "emergency":
            return False
        if first == "defend" and self.facts.get("danger") != "emergency":
            return False
        self.intent = first
        self.intent_until = self.frame + (10 if first.startswith("escape") else 18)
        self.intent_life = self.lives
        try:
            self.ctx.log(
                "jev_route",
                frame=self.frame,
                judged_frame=judged_frame,
                age=self.frame - judged_frame,
                route=first,
                probability=round(probability, 3),
                danger=self.facts.get("danger"),
            )
        except Exception:
            pass
        return True

    # ---- local route choice and native action bridge ----

    def _best_escape(self) -> str:
        candidates = [
            route for route in ("escape_left", "escape_right", "counterdrift", "scan_left", "scan_right")
            if self.routes.get(route, {}).get("valid", False)
        ]
        if not candidates:
            candidates = ["escape_left", "escape_right"]
        return max(
            candidates,
            key=lambda route: (
                self.routes.get(route, {}).get("clearance", -999.0),
                1 if route.startswith("escape") else 0,
                1 if route == ("escape_left" if self.sweep_sign > 0 else "escape_right") else 0,
            ),
        )

    def local_route(self) -> str:
        facts = self.facts
        danger = facts.get("danger", "unknown")
        if danger == "emergency":
            best = self._best_escape()
            current = self.packet if self.packet in self.routes else ""
            if current in {"escape_left", "escape_right", "counterdrift"} and self.frame < self.packet_until:
                current_clearance = self.routes.get(current, {}).get("clearance", -999.0)
                best_clearance = self.routes.get(best, {}).get("clearance", -999.0)
                if current_clearance >= best_clearance - 12.0:
                    return current
            self.packet = best
            self.packet_start = self.frame
            self.packet_until = self.frame + (10 if best.startswith("escape") else 8)
            return best

        if danger == "approaching":
            best = self._best_escape()
            hold_clearance = self.routes.get("hold", {}).get("clearance", -999.0)
            best_clearance = self.routes.get(best, {}).get("clearance", -999.0)
            threshold = {
                "safety_first": 4.0,
                "balanced": 8.0,
                "fire_first": 14.0,
            }.get(self.policy.get("objective"), 4.0)
            if best.startswith("escape") and best_clearance > hold_clearance + threshold:
                self.packet = best
                self.packet_start = self.frame
                self.packet_until = self.frame + 8
                return best

        if self.frame < self.intent_until and self.intent_life == self.lives:
            preferred = self.intent
            route = self.routes.get(preferred, {})
            if preferred in {"scan_left", "scan_right", "hold", "counterdrift"} and route.get("valid", False):
                return preferred
            if preferred.startswith("escape") and danger in {"approaching", "emergency"} and route.get("valid", False):
                return preferred
            if preferred == "defend" and danger == "emergency":
                return preferred

        # Default central-fire sweep.  The route is allowed to turn but never thrusts
        # merely because a model answer is late.
        if self.policy.get("sweep") == "left":
            return "scan_left"
        if self.policy.get("sweep") == "right":
            return "scan_right"
        if self.frame >= self.packet_until:
            self.sweep_sign *= -1
            self.packet_until = self.frame + 72
        return "scan_left" if self.sweep_sign > 0 else "scan_right"

    def _defense_allowed(self) -> bool:
        facts = self.facts
        route = self.routes.get("escape_left", {})
        other = self.routes.get("escape_right", {})
        best_clearance = max(route.get("clearance", -999.0), other.get("clearance", -999.0))
        nearest = facts.get("nearest")
        ttc = nearest.get("ttc") if isinstance(nearest, dict) else None
        return (
            bool(facts.get("present"))
            and facts.get("danger") == "emergency"
            and self.policy.get("defense") == "last_resort"
            and not self.defense_used
            and self.frame - self.last_defense >= 180
            and ttc is not None
            and ttc <= 10.0
            and best_clearance < 0.0
        )

    def _fire_now(self) -> bool:
        if self.frame < self.fire_next:
            return False
        if self.facts.get("missile_count", 0) >= 2:
            return False
        self.fire_next = self.frame + (7 if self.policy.get("objective") == "safety_first" else 5)
        return True

    def _packet_action(self, route: str) -> str:
        facts = self.facts
        if not facts.get("present"):
            return self.noop
        if route == "defend":
            if self._defense_allowed():
                self.defense_used = True
                self.last_defense = self.frame
                return self.legal("DOWN")
            route = self.local_route()

        if route in {"escape_left", "escape_right"}:
            if self.packet != route:
                self.packet = route
                self.packet_start = self.frame
                self.packet_until = self.frame + 10
            offset = max(0, self.frame - self.packet_start)
            turn_action = "UPLEFT" if route == "escape_left" else "UPRIGHT"
            if offset < 8:
                return self.legal(turn_action, "UP")
            if offset < 11:
                return self.legal("UP")
            return self.noop

        if route == "counterdrift":
            vx = facts.get("vx") or 0.0
            vy = facts.get("vy") or 0.0
            desired = vector_heading(-vx, -vy)
            turn = shortest_turn(integer(facts.get("heading"), 0), desired, self.sweep_sign)
            if turn > 0:
                return self.legal("LEFT")
            if turn < 0:
                return self.legal("RIGHT")
            return self.legal("UP")

        if route == "scan_left":
            fire = self._fire_now()
            return self.legal("LEFTFIRE" if fire else "LEFT", "LEFT")
        if route == "scan_right":
            fire = self._fire_now()
            return self.legal("RIGHTFIRE" if fire else "RIGHT", "RIGHT")
        # hold and any conservative fallback: stationary firing with release gaps.
        return self.legal("FIRE" if self._fire_now() else self.noop, self.noop)

    def choose_action(self) -> str:
        facts = self.facts
        if not facts.get("present") or self.frame < self.respawn_until:
            self.last_action = self.noop
            return self.noop

        route = self.local_route()
        if self._defense_allowed() and facts.get("danger") == "emergency":
            route = "defend"
        action = self._packet_action(route)
        self.last_action = action
        if self.frame - self.last_log_frame >= 600 or facts.get("danger") == "emergency":
            try:
                self.ctx.log(
                    "lane_control",
                    frame=self.frame,
                    action=action,
                    route=route,
                    danger=facts.get("danger"),
                    clearance=bucket_clearance(self.routes.get(route, {}).get("clearance", 999.0)),
                    lives=self.lives,
                    score=round(self.score, 1),
                )
            except Exception:
                pass
            self.last_log_frame = self.frame
        return action

    def apply_policy(self, value: Any) -> bool:
        data = extract_object(value)
        if not isinstance(data, dict):
            return False
        choices = {
            "objective": {"safety_first", "balanced", "fire_first"},
            "sweep": {"alternate", "left", "right"},
            "defense": {"last_resort", "conservative"},
        }
        changed = {}
        for key, allowed in choices.items():
            candidate = data.get(key)
            if isinstance(candidate, str) and candidate in allowed:
                changed[key] = candidate
        if not changed:
            return False
        self.policy.update(changed)
        self.policy_version += 1
        try:
            self.ctx.log("astra_objective", frame=self.frame, policy=dict(self.policy), version=self.policy_version)
        except Exception:
            pass
        return True


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.planner: LanePlanner | None = None
        self.jev_future = None
        self.jev_frame = -1
        self.jev_lives = -1
        self.next_jev = -1
        self.s1_enabled = getattr(ctx, "system1", True) is not None
        self.astra_future = None
        self.astra_started = False

    def _start_jev(self) -> None:
        planner = self.planner
        if planner is None or not self.s1_enabled or self.jev_future is not None:
            return
        if planner.frame < self.next_jev:
            return
        state = planner.decision_state()
        questions = planner.questions()
        self.jev_frame = planner.frame
        self.jev_lives = planner.lives
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=3.0)
            )
            self.next_jev = planner.frame + 18
        except (BudgetExceeded, EpisodeOver):
            self.s1_enabled = False
        except ModelUnavailable:
            raise
        except Exception:
            self.jev_future = None
            self.next_jev = planner.frame + 18

    def _collect_jev(self) -> None:
        planner = self.planner
        future = self.jev_future
        if planner is None or future is None or not future.done():
            return
        self.jev_future = None
        try:
            result = future.result()
            planner.accept_route(result, self.jev_frame, self.jev_lives)
        except (BudgetExceeded, EpisodeOver):
            self.s1_enabled = False
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError, KeyError, AttributeError):
            pass
        except Exception:
            # Stand-in answers and cancelled futures are deliberately non-fatal.
            pass

    def _astra_prompt(self, planner: LanePlanner) -> str:
        state = planner.decision_state()
        return (
            "You are the slow strategic author for a real-time Atari 2600 Asteroids "
            "mode-0/difficulty-0 controller. The clock continues while you think. "
            "Code already owns collision arithmetic, trajectory simulation, native-frame "
            "execution, FIRE release cadence, life bookkeeping, and emergency defense. "
            "Jev will choose among code-generated bounded routes. Do not emit actions, "
            "timestamps, route ids, or a sequence. Select only durable objective weights "
            "for the next part of this episode. Unknown enemy coverage is not absence. "
            "Return one compact JSON object only with exact fields: objective is one of "
            "safety_first, balanced, fire_first; sweep is one of alternate, left, right; "
            "defense is one of last_resort, conservative. Prefer safety_first and "
            "last_resort unless the supplied evidence supports another choice.\nSTATE:\n"
            + json.dumps(state, separators=(",", ":"), allow_nan=False)
        )

    def _start_astra(self) -> None:
        if self.astra_started or self.planner is None:
            return
        self.astra_started = True
        try:
            remaining = self.ctx.budget.remaining()
            if integer(remaining.get("system2_calls"), 0) <= 0:
                return
            prompt = self._astra_prompt(self.planner)
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=90.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None
        except ModelUnavailable:
            raise
        except Exception:
            self.astra_future = None

    def _collect_astra(self) -> None:
        planner = self.planner
        future = self.astra_future
        if planner is None or future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            planner.apply_policy(getattr(result, "text", result))
        except (BudgetExceeded, EpisodeOver):
            pass
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError, TypeError, ValueError, KeyError, AttributeError):
            pass
        except Exception:
            pass

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        if getattr(game, "tick_s", None) is None:
            return
        planner = LanePlanner(game, ctx)
        self.planner = planner
        try:
            planner.update(game.observe())
        except Exception:
            planner.facts = {
                "frame": 0,
                "score": 0.0,
                "lives": 0,
                "present": False,
                "danger": "unknown",
                "rocks": [],
                "missile_count": 0,
                "motion_break": True,
            }
            planner.routes = {}
        self.next_jev = planner.frame
        self._start_astra()
        self._start_jev()

        try:
            # Post a bounded first action before the first native tick. All later
            # iterations wait for exactly one tick and post a fresh action.
            if not ctx.done and not getattr(game, "done", False):
                ctx.act(planner.choose_action())
        except (BudgetExceeded, EpisodeOver):
            return
        except ModelUnavailable:
            raise
        except Exception:
            try:
                ctx.act(planner.noop)
            except (BudgetExceeded, EpisodeOver):
                return

        while not ctx.done and not getattr(game, "done", False):
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return
            if ctx.done or getattr(game, "done", False):
                break
            try:
                planner.update(game.observe())
                self._collect_jev()
                self._collect_astra()
                self._start_jev()
                if not ctx.done and not getattr(game, "done", False):
                    ctx.act(planner.choose_action())
            except (BudgetExceeded, EpisodeOver):
                return
            except ModelUnavailable:
                raise
            except Exception as exc:
                # A decoder glitch cannot turn into an episode error. The current
                # native action is deliberately released for one frame.
                try:
                    ctx.log("frame_recovered", frame=planner.frame, error=type(exc).__name__)
                    ctx.act(planner.noop)
                except (BudgetExceeded, EpisodeOver):
                    return


def build(ctx):
    return Harness(ctx)
