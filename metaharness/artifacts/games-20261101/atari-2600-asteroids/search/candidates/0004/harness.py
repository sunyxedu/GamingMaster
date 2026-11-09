"""A frame-local Asteroids controller with asynchronous semantic advice.

The controller deliberately does not let a slow model own the emulator clock.  Code
turns the RAM-derived observation into collision and aiming facts, then executes a
short, repeatable sweep/fire policy every native frame.  Jev only selects a tactical
bias from those facts; Astra reviews the longer-lived policy in the background.
"""
from __future__ import annotations

import math
from concurrent.futures import Future
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


WIDTH = 160.0
HEIGHT = 210.0
CENTER_X = WIDTH / 2.0
CENTER_Y = HEIGHT / 2.0


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.jev_future: Future | None = None
        self.jev_frame = -10**9
        self.astra_future: Future | None = None
        self.astra_started = False
        self.astra_revision_started = False
        self.plan = {
            "mode": "sweep",
            "sweep": "right",
            "risk": "conservative",
            "defense": "emergency_only",
            "fire_gap": 7,
        }
        self.plan_version = 0
        self.model_bias = "follow_target"
        self.model_risk = "unknown"
        self.model_until = -1
        self.last_frame = -1
        self.last_lives = None
        self.life_number = 0
        self.last_defense_frame = -10**9
        self.defense_used_life = False
        self.prev_player = None
        self.sweep_sign = 1
        self.sweep_until = -1
        self.escape_until = -1
        self.escape_heading = 0
        self.next_fire_frame = 0
        self.fire_release_until = -1
        self.last_action = "NOOP"
        self.last_score = 0.0
        self.last_log_frame = -10**9

    # ---------- small, defensive observation helpers ----------

    @staticmethod
    def _dict(value: Any) -> dict:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _finite(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default

    @classmethod
    def _int(cls, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _wrap_delta(a: float, b: float, period: float) -> float:
        d = a - b
        while d > period / 2.0:
            d -= period
        while d < -period / 2.0:
            d += period
        return d

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _frame(self, obs: Any) -> int:
        return self._int(self._dict(obs).get("frame"), self.last_frame + 1)

    def _legal(self, game: Any, name: str) -> str:
        """Return a legal name, with the adaptor's advertised noop as fallback."""
        try:
            spec = game.action_spec
            names = list(spec.names)
            if name in names:
                return name
            noop = getattr(spec, "noop", "NOOP")
            return noop if noop in names else (names[0] if names else "NOOP")
        except Exception:
            return name

    # ---------- derived game facts ----------

    def _rock_facts(self, obs: dict, player: dict) -> list[dict]:
        px = self._finite(player.get("x"), CENTER_X)
        py = self._finite(player.get("y"), CENTER_Y)
        pv = player.get("velocity_px_per_frame")
        pvx = self._finite(pv[0], 0.0) if isinstance(pv, list) and len(pv) >= 2 else 0.0
        pvy = self._finite(pv[1], 0.0) if isinstance(pv, list) and len(pv) >= 2 else 0.0
        result = []
        rocks = obs.get("asteroids")
        if not isinstance(rocks, list):
            return result
        for index, rock0 in enumerate(rocks):
            rock = self._dict(rock0)
            if "x" not in rock or "y" not in rock:
                continue
            x = self._finite(rock.get("x"), px)
            y = self._finite(rock.get("y"), py)
            dx = self._wrap_delta(x, px, WIDTH)
            dy = self._wrap_delta(y, py, HEIGHT)
            velocity = rock.get("velocity_px_per_frame")
            if isinstance(velocity, list) and len(velocity) >= 2:
                vx = self._finite(velocity[0], 0.0) - pvx
                vy = self._finite(velocity[1], 0.0) - pvy
                velocity_known = True
            else:
                vx = vy = 0.0
                velocity_known = False
            vv = vx * vx + vy * vy
            if vv > 1e-6 and velocity_known:
                closest_t = self._clamp(-(dx * vx + dy * vy) / vv, 0.0, 60.0)
            else:
                closest_t = 60.0
            closest_dx = dx + vx * closest_t
            closest_dy = dy + vy * closest_t
            closest_distance = math.hypot(closest_dx, closest_dy)
            now_distance = math.hypot(dx, dy)
            size = str(rock.get("size", "unknown"))
            radius = {"large": 17.0, "medium": 12.0, "small": 8.0}.get(size, 13.0)
            player_radius = 7.0
            clearance = closest_distance - radius - player_radius
            now_clearance = now_distance - radius - player_radius
            # A short, wrap-aware lead; the action still uses only documented cardinal
            # labels rather than assuming an unverified intermediate angle calibration.
            lead_t = min(10.0, max(2.0, now_distance / 5.0))
            lead_x = self._wrap_delta(dx + vx * lead_t, 0.0, WIDTH)
            lead_y = self._wrap_delta(dy + vy * lead_t, 0.0, HEIGHT)
            desired = self._desired_code(lead_x, lead_y)
            result.append({
                "slot": self._int(rock.get("source_slot"), index),
                "size": size,
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy,
                "distance": now_distance,
                "ttc": closest_t,
                "clearance": clearance,
                "now_clearance": now_clearance,
                "desired_code": desired,
                "velocity_known": velocity_known,
            })
        return result

    @staticmethod
    def _desired_code(dx: float, dy: float) -> int:
        """Choose a documented cardinal heading without calibrating intermediate codes."""
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return 0
        # Only the four cardinal labels are public guarantees.  A dominant-axis
        # quantizer is safer than inventing equally spaced intermediate angles.
        if abs(dx) >= abs(dy):
            return 12 if dx > 0 else 4
        return 8 if dy > 0 else 0

    def _facts(self, obs: Any) -> dict:
        data = self._dict(obs)
        player = self._dict(data.get("player"))
        present = bool(player.get("present", False))
        rocks = self._rock_facts(data, player) if present else []
        imminent = [
            r for r in rocks
            if r["now_clearance"] <= 4.0
            or (r["clearance"] <= 8.0 and r["ttc"] <= 18.0)
        ]
        near = [
            r for r in rocks
            if r["now_clearance"] <= 18.0
            or (r["clearance"] <= 18.0 and r["ttc"] <= 38.0)
        ]
        if imminent:
            danger = "emergency"
        elif near:
            danger = "approaching"
        elif not rocks:
            danger = "unknown"
        else:
            danger = "clear"
        # Small rocks are useful targets, but a nearby object wins over value.
        size_value = {"small": 3, "medium": 2, "large": 1}
        target = min(
            rocks,
            key=lambda r: (
                0 if r["clearance"] < 0 else 1,
                r["ttc"] if r["ttc"] < 60 else 60,
                -size_value.get(r["size"], 0),
                r["distance"],
            ),
            default=None,
        )
        if target is None:
            target = min(rocks, key=lambda r: r["distance"], default=None)
        orientation = self._int(player.get("orientation_code"), 0) % 16
        vx = vy = 0.0
        velocity = player.get("velocity_px_per_frame")
        if isinstance(velocity, list) and len(velocity) >= 2:
            vx = self._finite(velocity[0])
            vy = self._finite(velocity[1])
        dx_center = self._wrap_delta(CENTER_X - self._finite(player.get("x"), CENTER_X), 0.0, WIDTH)
        dy_center = self._wrap_delta(CENTER_Y - self._finite(player.get("y"), CENTER_Y), 0.0, HEIGHT)
        missiles = data.get("player_missiles")
        missile_count = len(missiles) if isinstance(missiles, list) else 0
        return {
            "frame": self._frame(data),
            "score": self._finite(data.get("score")),
            "lives": self._int(data.get("lives"), 0),
            "present": present,
            "player_x": self._finite(player.get("x"), CENTER_X),
            "player_y": self._finite(player.get("y"), CENTER_Y),
            "orientation": orientation,
            "player_vx": vx,
            "player_vy": vy,
            "center_dx": dx_center,
            "center_dy": dy_center,
            "rocks": rocks,
            "rock_count": len(rocks),
            "imminent_count": len(imminent),
            "near_count": len(near),
            "missile_count": missile_count,
            "danger": danger,
            "target": target,
            "enemy_coverage_unknown": self._dict(data.get("coverage")).get("enemy_craft") is not None,
            "last_action": str(data.get("last_action", self.last_action)),
        }

    def _compact_facts(self, facts: dict) -> dict:
        """Make a small, strictly JSON state for Jev; no RAM or raw geometry is sent."""
        target = facts.get("target")
        target_summary = None
        if isinstance(target, dict):
            target_summary = {
                "size": target.get("size", "unknown"),
                "distance_bucket": self._distance_bucket(target.get("distance", 999)),
                "approach_bucket": "imminent" if target.get("ttc", 60) <= 18 else (
                    "approaching" if target.get("ttc", 60) <= 38 else "not_imminent"
                ),
                "clearance_bucket": self._clearance_bucket(target.get("clearance", 999)),
                "heading_relation": self._heading_relation(
                    facts.get("orientation", 0), target.get("desired_code", 0)
                ),
            }
        return {
            "frame": int(facts.get("frame", 0)),
            "lives": int(facts.get("lives", 0)),
            "score": int(max(0, facts.get("score", 0))),
            "player": {
                "present": bool(facts.get("present")),
                "centrality": self._centrality_bucket(facts.get("center_dx", 0), facts.get("center_dy", 0)),
                "motion": self._motion_bucket(facts.get("player_vx", 0), facts.get("player_vy", 0)),
            },
            "field": {
                "rock_count": int(facts.get("rock_count", 0)),
                "danger": str(facts.get("danger", "unknown")),
                "imminent_count": int(facts.get("imminent_count", 0)),
                "near_count": int(facts.get("near_count", 0)),
                "target": target_summary,
                "enemy_coverage": "unknown_not_absent" if facts.get("enemy_coverage_unknown") else "unknown",
            },
            "plan": {
                "mode": self.plan.get("mode", "sweep"),
                "risk": self.plan.get("risk", "conservative"),
                "defense": self.plan.get("defense", "emergency_only"),
                "sweep": self.plan.get("sweep", "right"),
            },
            "controller": {
                "current_bias": self.model_bias,
                "action_is_frame_local": True,
                "defense_used_this_life": bool(self.defense_used_life),
            },
        }

    @staticmethod
    def _distance_bucket(value: Any) -> str:
        d = Harness._finite(value, 999)
        if d < 20:
            return "contact_range"
        if d < 45:
            return "near"
        if d < 90:
            return "middle"
        return "far"

    @staticmethod
    def _clearance_bucket(value: Any) -> str:
        d = Harness._finite(value, 999)
        if d < 0:
            return "overlapping"
        if d < 10:
            return "tight"
        if d < 25:
            return "limited"
        return "clear"

    @staticmethod
    def _centrality_bucket(dx: Any, dy: Any) -> str:
        d = math.hypot(Harness._finite(dx), Harness._finite(dy))
        if d < 25:
            return "central"
        if d < 55:
            return "off_center"
        return "edgeward"

    @staticmethod
    def _motion_bucket(vx: Any, vy: Any) -> str:
        speed = math.hypot(Harness._finite(vx), Harness._finite(vy))
        if speed < 0.2:
            return "still_or_unknown"
        if speed < 0.8:
            return "drifting"
        return "fast_drift"

    @staticmethod
    def _heading_relation(current: Any, desired: Any) -> str:
        c = Harness._int(current) % 16
        d = Harness._int(desired) % 16
        delta = (d - c) % 16
        if delta == 0:
            return "aligned"
        if delta <= 2 or delta >= 14:
            return "almost_aligned"
        if delta < 8:
            return "turn_left"
        if delta > 8:
            return "turn_right"
        return "half_turn"

    # ---------- asynchronous model division ----------

    def _start_jev(self, facts: dict) -> None:
        if getattr(self.ctx, "system1", None) is None or self.jev_future is not None:
            return
        state = self._compact_facts(facts)
        # The duplicate question reverses choices to blunt the documented first-option bias.
        options = {
            "follow_target": "Turn only in the computed shortest direction toward the named target, fire in pulses, and do not thrust unless code detects a hazard.",
            "sweep_left": "Use a short counterclockwise aiming sweep while firing in pulses; no thrust unless code detects a hazard.",
            "sweep_right": "Use a short clockwise aiming sweep while firing in pulses; no thrust unless code detects a hazard.",
            "evade_left": "Use a bounded left-turn escape; do not select this merely because a rock is far away.",
            "evade_right": "Use a bounded right-turn escape; do not select this merely because a rock is far away.",
            "defend": "Request one emergency hyperspace pulse only when the computed danger is emergency; never repeat it while the danger facts are unchanged.",
            "hold": "Keep the current heading and fire only when the local pulse schedule permits.",
        }
        order = ["follow_target", "sweep_left", "sweep_right", "evade_left", "evade_right", "defend", "hold"]
        if state["frame"] // 24 % 2:
            order = list(reversed(order))
        criteria_a = {key: options[key] for key in order}
        criteria_b = {key: options[key] for key in reversed(order)}
        questions = {
            "risk": Choice(
                "Classify the immediate asteroid risk from the precomputed buckets. Do not recalculate distances. Choose emergency only when field.danger is emergency or the player is in overlapping/tight clearance; choose approaching for a near predicted crossing; otherwise clear or unknown.",
                {
                    "clear": "No computed near crossing: continue a bounded firing sweep.",
                    "approaching": "A computed crossing is near but not immediate: favor a short steering correction, not hyperspace.",
                    "emergency": "The computed clearance is imminent or overlapping: survival overrides scoring and defense may be considered once.",
                    "unknown": "The facts are insufficient: use the conservative no-thrust firing fallback.",
                },
            ),
            "bias_a": Choice(
                "Choose one short-lived tactical bias. Code already computes heading, target, collision timing, and legal actions; judge the named alternatives rather than doing arithmetic. The answer must be compatible with the plan and the latest field facts.",
                criteria_a,
            ),
            "bias_b": Choice(
                "Independently choose the safest useful tactical bias for exactly this state. Do not invent a new action or a long sequence; use the alternative descriptions.",
                criteria_b,
            ),
        }
        origin = state["frame"]
        self.jev_frame = origin

        def request():
            return self.ctx.decide(state, questions, timeout_s=3.0)

        try:
            self.jev_future = self.ctx.spawn(request)
        except Exception as exc:
            self.jev_future = None
            self.ctx.log("jev_spawn_failed", frame=origin, error=type(exc).__name__)

    def _collect_jev(self, current_frame: int) -> None:
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        try:
            # A Jev result is advice about its origin snapshot, not a delayed command.
            # Drop it if it spent too long in flight or if the episode advanced through
            # a discontinuity while it was pending.
            if current_frame - self.jev_frame > 42:
                self.ctx.log("jev_stale", frame=current_frame, judged_frame=self.jev_frame)
                return
            answers = future.result()
            if not isinstance(answers, dict):
                return
            risk_answer = answers.get("risk")
            a = answers.get("bias_a")
            b = answers.get("bias_b")
            risk = getattr(risk_answer, "value", None)
            choice_a = getattr(a, "value", None)
            choice_b = getattr(b, "value", None)
            legal_biases = {
                "follow_target", "sweep_left", "sweep_right", "evade_left",
                "evade_right", "defend", "hold",
            }
            if risk in {"clear", "approaching", "emergency", "unknown"}:
                risk_probability = self._answer_probability(risk_answer, risk)
                risk_floor = 0.55 if risk == "emergency" else 0.20
                self.model_risk = risk if risk_probability >= risk_floor else "unknown"
            # A disagreement is deliberately not turned into a strong maneuver.
            if choice_a == choice_b and choice_a in legal_biases:
                confidence_a = self._answer_probability(a, choice_a)
                confidence_b = self._answer_probability(b, choice_b)
                if choice_a in {"defend", "evade_left", "evade_right"}:
                    accepted = min(confidence_a, confidence_b) >= 0.42
                else:
                    accepted = min(confidence_a, confidence_b) >= 0.18
                if accepted:
                    self.model_bias = choice_a
            self.model_until = current_frame + 42
            self.ctx.log(
                "jev_bias",
                frame=current_frame,
                judged_frame=self.jev_frame,
                risk=self.model_risk,
                bias=self.model_bias,
                agreement=choice_a == choice_b,
            )
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable) as exc:
            self.ctx.log("jev_failed", frame=current_frame, error=type(exc).__name__)
        except Exception as exc:
            # Garbage stand-in answers and cancelled futures must never kill preflight.
            self.ctx.log("jev_bad_answer", frame=current_frame, error=type(exc).__name__)

    def _start_astra(self, facts: dict, reason: str) -> None:
        if self.astra_future is not None or getattr(self.ctx, "system2", True) is None:
            return
        state = self._compact_facts(facts)
        prompt = f"""You are the slow strategic reviewer for a real-time Atari 2600 Asteroids harness.
The game clock continues while you think, so do not give frame-by-frame actions. Code executes
one native frame at a time and already handles collision arithmetic, heading conversion, firing
pulses, life bookkeeping, and emergency fallback. Jev receives compact derived facts and chooses
only a short tactical bias.

Current review reason: {reason}
Current facts (possibly stale by the time you answer): {state}

Choose a durable policy for the next several hundred frames. Prefer central controlled firing,
short aim sweeps, and survival over blind thrust. Hyperspace is mode-0 defense, not braking, and
must remain emergency-only unless observations prove otherwise. Enemy fields are unknown, not
absent. Answer with one small JSON object, with only these string values:
{{"mode":"sweep" or "target", "sweep":"left" or "right", "risk":"conservative" or "balanced", "defense":"emergency_only"}}.
Do not include actions, timestamps, or a long explanation."""

        def request():
            return self.ctx.think(prompt, timeout_s=120.0)

        try:
            self.astra_future = self.ctx.spawn(request)
            if reason == "launch":
                self.astra_started = True
            else:
                self.astra_revision_started = True
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("astra_spawn_failed", frame=facts.get("frame", -1), error=type(exc).__name__)

    def _collect_astra(self, current_frame: int) -> None:
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            text = str(getattr(result, "text", result))
            new_plan = self._parse_plan(text)
            if new_plan:
                self.plan.update(new_plan)
                self.plan_version += 1
                self.sweep_sign = 1 if self.plan.get("sweep") == "right" else -1
                self.ctx.log("astra_plan", frame=current_frame, plan=dict(self.plan), version=self.plan_version)
            else:
                self.ctx.log("astra_unparsed", frame=current_frame)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable) as exc:
            self.ctx.log("astra_failed", frame=current_frame, error=type(exc).__name__)
        except Exception as exc:
            self.ctx.log("astra_bad_answer", frame=current_frame, error=type(exc).__name__)

    @staticmethod
    def _answer_probability(answer: Any, value: str) -> float:
        try:
            probs = getattr(answer, "probabilities", {})
            return float(probs.get(value, 0.0)) if isinstance(probs, dict) else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _parse_plan(text: str) -> dict:
        lower = text.lower()
        # Accept only explicit policy words.  If Astra emits prose or malformed JSON,
        # retaining the tested default is safer than guessing from incidental wording.
        result = {}
        if '"mode"' in lower:
            if '"mode":"target"' in lower or '"mode": "target"' in lower:
                result["mode"] = "target"
            elif '"mode":"sweep"' in lower or '"mode": "sweep"' in lower:
                result["mode"] = "sweep"
        if '"sweep"' in lower:
            if '"sweep":"left"' in lower or '"sweep": "left"' in lower:
                result["sweep"] = "left"
            elif '"sweep":"right"' in lower or '"sweep": "right"' in lower:
                result["sweep"] = "right"
        if '"risk"' in lower:
            if '"risk":"balanced"' in lower or '"risk": "balanced"' in lower:
                result["risk"] = "balanced"
            elif '"risk":"conservative"' in lower or '"risk": "conservative"' in lower:
                result["risk"] = "conservative"
        if '"defense"' in lower and ('"defense":"emergency_only"' in lower or '"defense": "emergency_only"' in lower):
            result["defense"] = "emergency_only"
        return result

    # ---------- frame-local controller ----------

    def _shortest_turn(self, current: int, desired: int) -> int:
        delta = (desired - current) % 16
        if delta == 0:
            return 0
        if delta < 8:
            return 1       # LEFT increments the orientation code
        if delta > 8:
            return -1      # RIGHT decrements it
        return self.sweep_sign

    def _escape_code(self, facts: dict) -> int:
        rocks = facts.get("rocks", [])
        if not rocks:
            return (facts.get("orientation", 0) + (4 if self.sweep_sign > 0 else -4)) % 16
        # Escape from the weighted nearest threat.  This is a heading, not a claim that
        # thrust instantly stops inertia.
        threat = min(rocks, key=lambda r: (r.get("clearance", 999), r.get("ttc", 60)))
        return self._desired_code(-threat.get("dx", 0.0), -threat.get("dy", 0.0))

    def _should_defend(self, facts: dict) -> bool:
        frame = facts["frame"]
        if not facts["present"] or facts["danger"] != "emergency":
            return False
        if self.defense_used_life:
            return False
        if frame - self.last_defense_frame < 120:
            return False
        if frame > self.model_until:
            return False
        # Never let a weak semantic answer cause repeated hyperspace. Code must see a
        # real imminent crossing; Jev can make this more conservative, never less safe.
        return self.model_risk == "emergency" or self.model_bias == "defend"

    def _action(self, game: Any, facts: dict) -> str:
        frame = facts["frame"]
        if not facts["present"]:
            return self._legal(game, "NOOP")

        if self._should_defend(facts):
            self.defense_used_life = True
            self.last_defense_frame = frame
            self.model_bias = "hold"
            return self._legal(game, "DOWN")

        danger = facts["danger"]
        target = facts.get("target")
        orientation = facts["orientation"]

        # Emergency steering is fully local and therefore does not wait for Jev.  It
        # uses a bounded thrust pulse; the heading is frozen for that pulse so a noisy
        # estimate cannot turn an escape into an unbounded thrust macro.
        if danger == "emergency":
            if frame >= self.escape_until:
                self.escape_heading = self._escape_code(facts)
                self.escape_until = frame + 6
            turn = self._shortest_turn(orientation, self.escape_heading)
            if turn:
                return self._legal(game, "UPLEFT" if turn > 0 else "UPRIGHT")
            if frame < self.escape_until:
                return self._legal(game, "UP")
        elif danger == "approaching":
            # Buy time with steering only; thrust is reserved for the bounded emergency
            # pulse because NOOP is not a brake and an early thrust can create drift.
            desired = self._escape_code(facts)
            turn = self._shortest_turn(orientation, desired)
            if turn:
                return self._legal(game, "LEFT" if turn > 0 else "RIGHT")
        else:
            self.escape_until = -1

        # Keep the launch drift and later steering near the central firing region.  This
        # correction is short and only happens well away from a collision.
        if abs(facts["center_dx"]) > 58 or abs(facts["center_dy"]) > 68:
            desired_center = self._desired_code(facts["center_dx"], facts["center_dy"])
            turn = self._shortest_turn(orientation, desired_center)
            if turn:
                return self._legal(game, "UPLEFT" if turn > 0 else "UPRIGHT")
            if frame % 5 != 0:
                return self._legal(game, "UP")

        # Decide which heading to hold for this frame. The model can bias the sweep,
        # but target geometry and shortest-turn arithmetic remain code-owned. After a
        # Jev bias expires, Astra's durable sweep/target mode selects the fallback.
        if frame <= self.model_until:
            bias = self.model_bias
        elif self.plan.get("mode") == "target":
            bias = "follow_target"
        else:
            bias = "sweep_right" if self.plan.get("sweep") == "right" else "sweep_left"
        if bias == "sweep_left":
            self.sweep_sign = -1
        elif bias == "sweep_right":
            self.sweep_sign = 1
        elif frame >= self.sweep_until:
            self.sweep_sign = 1 if self.plan.get("sweep") == "right" else -1

        if bias in {"evade_left", "evade_right"}:
            turn = 1 if bias == "evade_left" else -1
        elif bias == "hold":
            turn = 0
        elif bias == "follow_target" and isinstance(target, dict):
            turn = self._shortest_turn(orientation, self._int(target.get("desired_code"), orientation))
        else:
            turn = self.sweep_sign

        # Fire in a real press/release cadence.  The next_fire frame creates a one-frame
        # press and fire_release_until leaves a release gap even during a rotation.
        fire_gap = 5 if self.plan.get("risk") == "balanced" else 7
        if frame >= self.next_fire_frame and facts.get("missile_count", 0) < 2:
            fire = True
            self.next_fire_frame = frame + fire_gap
            self.fire_release_until = frame + 1
        else:
            fire = False

        if turn > 0:
            action = "LEFTFIRE" if fire else "LEFT"
        elif turn < 0:
            action = "RIGHTFIRE" if fire else "RIGHT"
        else:
            action = "FIRE" if fire else "NOOP"
        return self._legal(game, action)

    def _life_update(self, facts: dict) -> None:
        lives = facts.get("lives")
        if self.last_lives is None:
            self.last_lives = lives
            self.life_number = 0
            return
        if lives < self.last_lives:
            self.life_number += 1
            self.defense_used_life = False
            self.last_defense_frame = -10**9
            self.sweep_until = facts["frame"] + 30
            self.escape_until = -1
            self.model_bias = "follow_target"
            self.model_risk = "unknown"
            self.model_until = facts["frame"] - 1
            self.next_fire_frame = facts["frame"] + 12
            if self.astra_future is None and not self.astra_revision_started:
                self._start_astra(facts, "a ship was lost; review the aggregate policy")
            self.ctx.log("life_lost", frame=facts["frame"], lives=lives, score=facts["score"])
        self.last_lives = lives

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        if not getattr(game, "tick_s", None):
            return
        try:
            initial = game.observe()
            initial_facts = self._facts(initial)
            self.last_frame = initial_facts["frame"]
            self.last_score = initial_facts["score"]
            self.last_lives = initial_facts["lives"]
            # Astra is a policy reviewer only. It is never on the clock-critical path.
            self._start_astra(initial_facts, "launch")
        except Exception:
            # A malformed initial observation should still leave the local noop loop alive.
            initial_facts = None

        while not ctx.done and not getattr(game, "done", False):
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                break
            if ctx.done or getattr(game, "done", False):
                break
            try:
                obs = game.observe()
                facts = self._facts(obs)
                frame = facts["frame"]
                self.last_frame = frame
                self._life_update(facts)
                self._collect_jev(frame)
                self._collect_astra(frame)
                if self.jev_future is None and frame - self.jev_frame >= 24:
                    self._start_jev(facts)
                action = self._action(game, facts)
                ctx.act(action)
                self.last_action = action
                if frame - self.last_log_frame >= 600 or facts["score"] != self.last_score:
                    ctx.log(
                        "local_control",
                        frame=frame,
                        action=action,
                        score=facts["score"],
                        lives=facts["lives"],
                        danger=facts["danger"],
                        bias=self.model_bias,
                    )
                    self.last_log_frame = frame
                self.last_score = facts["score"]
            except (BudgetExceeded, EpisodeOver):
                break
            except ModelUnavailable:
                # The context marks the episode done; leave via the safe loop condition.
                break
            except Exception as exc:
                # Observation/data glitches are handled as a single local noop frame, not
                # allowed to make a preflight garbage answer fatal.
                try:
                    ctx.log("local_frame_recovered", frame=self.last_frame, error=type(exc).__name__)
                    ctx.act(self._legal(game, "NOOP"))
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    break


def build(ctx: Any) -> Harness:
    return Harness(ctx)
