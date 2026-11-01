"""Asynchronous Jev plus a deterministic central-fire Asteroids controller.

The emulator clock must keep moving while inference is in flight.  This module
therefore posts one bounded action on every tick from local observations; Jev
only selects a semantic posture and target, and Astra only revises aggregate
policy.  Neither model can leave a thrust or defense action latched.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


SCREEN_W = 160.0
SCREEN_H = 210.0
DEFAULT_ACTIONS = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
    "UPLEFTFIRE",
)


def _number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value, default=0):
    result = _number(value)
    return int(result) if result is not None else default


def _wrap_delta(delta, period):
    while delta > period / 2.0:
        delta -= period
    while delta < -period / 2.0:
        delta += period
    return delta


def _value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def _probabilities(answer):
    if isinstance(answer, dict):
        value = answer.get("probabilities", {})
    else:
        value = getattr(answer, "probabilities", {})
    return value if isinstance(value, dict) else {}


def _probability(answer, option):
    result = _number(_probabilities(answer).get(option), 0.0)
    return result if result is not None else 0.0


def _yes_probability(answer):
    """Noul.value is a probability locally; tolerate stand-in shapes too."""
    value = _value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    result = _number(value)
    if result is not None:
        return max(0.0, min(1.0, result))
    return max(0.0, min(1.0, _number(_probabilities(answer).get("true"), 0.0) or 0.0))


def _heading_for_vector(dx, dy):
    if abs(dx) + abs(dy) < 1e-6:
        return 0
    # Code 0 is up, 4 is left, 8 is down, and 12 is right.
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _band(value, low, high):
    if value is None:
        return "unknown"
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "middle"


def _bearing(dx, dy):
    horizontal = "right" if dx > 8 else "left" if dx < -8 else "center"
    vertical = "below" if dy > 8 else "above" if dy < -8 else "level"
    return horizontal + "_" + vertical


def _observation(game):
    try:
        value = game.observe()
    except Exception:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _extract_json(text):
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


class LocalPolicy:
    """Frame-level policy: stationary firing, short aim pulses, and escape."""

    def __init__(self, game, ctx):
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        names = getattr(spec, "names", DEFAULT_ACTIONS) if spec is not None else DEFAULT_ACTIONS
        self.names = set(names or DEFAULT_ACTIONS)
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else next(iter(self.names), "NOOP")

        self.frame = 0
        self.score = 0.0
        self.lives = 0
        self.previous_lives = None
        self.features = {}
        self.last_action = self.noop
        self.fire_cooldown = 0
        self.last_defense_frame = -100000
        self.sweep_direction = 1
        self.sweep_until = 0

        self.plan = {
            "posture": "auto",
            "risk": "conservative",
            "target_priority": "threat_then_small",
            "defense": "jev_emergency_only",
            "sweep_direction": "auto",
        }
        self.jev_posture = None
        self.jev_target_slot = None
        self.jev_target_size = None
        self.jev_defense = False
        self.jev_answer_frame = -100000
        self.jev_answer_lives = None

    def _player(self, obs):
        value = obs.get("player")
        return value if isinstance(value, dict) else {}

    def _rocks(self, obs):
        value = obs.get("asteroids")
        return value if isinstance(value, list) else []

    def update(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        self.frame = _int(obs.get("frame"), self.frame + 1)
        self.score = _number(obs.get("score"), self.score) or 0.0
        self.lives = _int(obs.get("lives"), self.lives)
        if self.previous_lives is not None and self.lives != self.previous_lives:
            # A death or respawn invalidates an old target and defense request.
            self.jev_target_slot = None
            self.jev_target_size = None
            self.jev_defense = False
            self.fire_cooldown = max(self.fire_cooldown, 3)
            self.last_defense_frame = self.frame
        self.previous_lives = self.lives
        self.features = self._derive(obs)

    def _derive(self, obs):
        player = self._player(obs)
        present = bool(player.get("present"))
        px = _number(player.get("x"))
        py = _number(player.get("y"))
        pw = _number(player.get("width"), 6.0) or 6.0
        ph = _number(player.get("height"), 10.0) or 10.0
        pvel = player.get("velocity_px_per_frame")
        pvx = _number(pvel[0]) if isinstance(pvel, list) and len(pvel) > 1 else None
        pvy = _number(pvel[1]) if isinstance(pvel, list) and len(pvel) > 1 else None
        if px is None or py is None:
            present = False

        objects = []
        if present:
            for raw in self._rocks(obs):
                if not isinstance(raw, dict):
                    continue
                ax = _number(raw.get("x"))
                ay = _number(raw.get("y"))
                if ax is None or ay is None:
                    continue
                aw = _number(raw.get("width"), 8.0) or 8.0
                ah = _number(raw.get("height"), 15.0) or 15.0
                velocity = raw.get("velocity_px_per_frame")
                avx = _number(velocity[0]) if isinstance(velocity, list) and len(velocity) > 1 else None
                avy = _number(velocity[1]) if isinstance(velocity, list) and len(velocity) > 1 else None
                dx = _wrap_delta(ax + aw / 2.0 - (px + pw / 2.0), SCREEN_W)
                dy = _wrap_delta(ay + ah / 2.0 - (py + ph / 2.0), SCREEN_H)
                distance = math.hypot(dx, dy)
                rvx = avx - pvx if avx is not None and pvx is not None else None
                rvy = avy - pvy if avy is not None and pvy is not None else None
                ttc = None
                if rvx is not None and rvy is not None and distance > 0.1:
                    closing = -(dx * rvx + dy * rvy) / distance
                    if closing > 0.05:
                        ttc = distance / closing
                clearance = max(aw, ah) / 2.0 + max(pw, ph) / 2.0 + 5.0
                if distance <= clearance + 8.0 or (ttc is not None and ttc <= 12 and distance < 70):
                    danger = "emergency"
                elif distance <= clearance + 28.0 or (ttc is not None and ttc <= 34 and distance < 100):
                    danger = "caution"
                else:
                    danger = "clear"
                size = str(raw.get("size") or "unknown").lower()
                value = {"small": 3, "medium": 2, "large": 1}.get(size, 0)
                urgency = {"clear": 0, "caution": 4, "emergency": 12}.get(danger, 0)
                measured = 1 if avx is not None and avy is not None else 0
                priority = urgency + value * 2 + measured - distance / 100.0
                if self.plan.get("target_priority") == "small_then_threat":
                    priority = urgency + value * 3 + measured - distance / 100.0
                elif self.plan.get("target_priority") == "nearest":
                    priority = urgency + (100.0 - min(distance, 100.0)) / 20.0
                projected_dx = dx + (avx or 0.0) * 8.0
                projected_dy = dy + (avy or 0.0) * 8.0
                objects.append({
                    "slot": _int(raw.get("source_slot"), -1),
                    "size": size,
                    "dx": round(dx, 2),
                    "dy": round(dy, 2),
                    "distance": round(distance, 2),
                    "bearing": _bearing(dx, dy),
                    "danger": danger,
                    "ttc": round(ttc, 2) if ttc is not None else None,
                    "velocity_known": measured == 1,
                    "priority": round(priority, 3),
                    "aim_code": _heading_for_vector(projected_dx, projected_dy),
                })

        objects.sort(key=lambda item: (-item["priority"], item["distance"], item["slot"]))
        danger_rank = {"clear": 0, "caution": 1, "emergency": 2}
        nearest = None
        if objects:
            # A farther rock with an imminent closing window outranks a nearer
            # receding rock for safety.  This is also the fact sent to Jev.
            emergency = [item for item in objects if item["danger"] == "emergency"]
            caution = [item for item in objects if item["danger"] == "caution"]
            danger_pool = emergency or caution or objects
            nearest = min(
                danger_pool,
                key=lambda item: (item["distance"], -danger_rank[item["danger"]]),
            )
        danger = nearest["danger"] if nearest is not None else "clear"
        speed = math.hypot(pvx or 0.0, pvy or 0.0) if present else None
        heading = _int(player.get("orientation_code"), 0) % 16 if present else None
        return {
            "frame": self.frame,
            "score": round(self.score, 2),
            "lives": self.lives,
            "player_present": present,
            "player_x_band": _band(px, 55, 105),
            "player_y_band": _band(py, 70, 140),
            "heading_code": heading,
            "speed_band": "unknown" if speed is None else "still" if speed < 0.25 else "moving",
            "danger": danger,
            "nearest_distance_band": (
                "none" if nearest is None else
                "near" if nearest["distance"] < 45 else
                "medium" if nearest["distance"] < 90 else "far"
            ),
            "nearest_ttc_band": (
                "unknown" if nearest is None or nearest["ttc"] is None else
                "imminent" if nearest["ttc"] < 15 else
                "soon" if nearest["ttc"] < 35 else "not_closing"
            ),
            "asteroid_count": len(objects),
            "small_count": sum(item["size"] == "small" for item in objects),
            "medium_count": sum(item["size"] == "medium" for item in objects),
            "large_count": sum(item["size"] == "large" for item in objects),
            "motion_discontinuity": bool(player.get("motion_discontinuity")) if present else True,
            "objects": objects,
        }

    def question_state(self):
        facts = self.features
        candidates = []
        for index, item in enumerate(facts.get("objects", [])[:4]):
            candidates.append({
                "id": "t" + str(index),
                "size": item["size"],
                "bearing": item["bearing"],
                "danger": item["danger"],
                "distance_band": (
                    "near" if item["distance"] < 45 else
                    "medium" if item["distance"] < 90 else "far"
                ),
                "velocity": "known" if item["velocity_known"] else "unknown",
            })
        return {
            "rules": {
                "mode": 0,
                "fire_is_button_level": True,
                "release_between_shots": True,
                "noop_does_not_brake": True,
                "left_increments_orientation": True,
                "right_decrements_orientation": True,
                "down_is_hyperspace": True,
                "unknown_enemy_coverage_is_not_absence": True,
            },
            "observation": {
                "frame": facts.get("frame", 0),
                "score": facts.get("score", 0.0),
                "lives": facts.get("lives", 0),
                "player_present": facts.get("player_present", False),
                "player_x_band": facts.get("player_x_band", "unknown"),
                "player_y_band": facts.get("player_y_band", "unknown"),
                "heading_code": facts.get("heading_code"),
                "speed_band": facts.get("speed_band", "unknown"),
                "danger": facts.get("danger", "unknown"),
                "nearest_distance_band": facts.get("nearest_distance_band", "none"),
                "nearest_ttc_band": facts.get("nearest_ttc_band", "unknown"),
                "asteroid_count": facts.get("asteroid_count", 0),
                "small_count": facts.get("small_count", 0),
                "medium_count": facts.get("medium_count", 0),
                "large_count": facts.get("large_count", 0),
                "motion_discontinuity": facts.get("motion_discontinuity", True),
            },
            "candidate_targets": candidates,
            "current_policy": self.plan,
            "last_action": self.last_action,
            "fire_release_frames_remaining": self.fire_cooldown,
            "defense": {
                "available": True,
                "cooldown": "ready" if self.frame - self.last_defense_frame >= 240 else "recently_used",
            },
        }

    def questions(self):
        objects = self.features.get("objects", [])[:4]
        risk_descriptions = {
            "clear": "Derived facts show no nearby or soon-closing asteroid requiring escape.",
            "caution": "Derived facts show a possible near-term clearance conflict; avoid thrust commitment.",
            "emergency": "Derived facts show a near or imminently closing asteroid; survival overrides scoring.",
            "unknown": "The supplied derived geometry or velocity cannot establish a danger class.",
        }
        risk_order = ["clear", "caution", "emergency", "unknown"]
        if self.frame // 24 % 2:
            risk_order.reverse()
        target_options = {}
        target_ids = ["t" + str(index) for index in range(len(objects))]
        ordered_targets = target_ids + ["none"]
        if self.frame // 24 % 2:
            ordered_targets = ["none"] + list(reversed(target_ids))
        for option in ordered_targets:
            if option == "none":
                target_options[option] = "Do not commit to one rock; use the safe sweep or steady-fire fallback."
            else:
                item = objects[int(option[1:])]
                target_options[option] = (
                    "Select this derived candidate: " + item["size"] + " rock, "
                    + item["bearing"] + ", " + item["danger"] + " danger, "
                    + ("velocity known" if item["velocity_known"] else "velocity unknown") + "."
                )
        posture_descriptions = {
            "focus": "Use short rotation pulses toward a selected target and pulse fire when released.",
            "sweep_left": "Rotate counterclockwise in bounded pulses and pulse fire through the sweep.",
            "sweep_right": "Rotate clockwise in bounded pulses and pulse fire through the sweep.",
            "steady_fire": "Do not thrust or rotate; pulse FIRE along the current heading.",
            "evade": "Use a short steering escape; do not score until danger is lower.",
            "recover": "A respawn, split, or discontinuity occurred; use conservative short actions.",
        }
        posture_order = list(posture_descriptions)
        if self.frame // 24 % 2:
            posture_order.reverse()
        return {
            "risk_a": Choice(
                "Classify danger from observation.danger, observation.nearest_distance_band, "
                "and observation.nearest_ttc_band. Do not perform arithmetic or infer absent "
                "enemy objects.",
                {key: risk_descriptions[key] for key in risk_order},
            ),
            "risk_b": Choice(
                "Independently repeat the same danger classification using the reordered options "
                "and the derived observation fields.",
                {key: risk_descriptions[key] for key in reversed(risk_order)},
            ),
            "target": Choice(
                "Choose the safest useful candidate_targets entry, or none. The local controller "
                "will perform all aiming arithmetic.",
                target_options,
            ),
            "posture": Choice(
                "Choose one bounded controller posture for the current observation. This is not "
                "permission to hold an action or emit a sequence.",
                {key: posture_descriptions[key] for key in posture_order},
            ),
            "hyperspace": Noul(
                "Should one mode-0 hyperspace pulse be used now? Answer yes only for explicitly "
                "imminent derived danger, with a present player and no recent defense; otherwise no.",
                {
                    "true": "One bounded defensive pulse is justified by the current emergency facts.",
                    "false": "Use steering, firing, or no-op; do not teleport now.",
                },
            ),
        }

    def accept_jev(self, answers, answer_frame, answer_lives):
        if not isinstance(answers, dict):
            return False
        age = self.frame - answer_frame
        if age < 0 or age > 48 or answer_lives != self.lives:
            return False

        risk_a = _value(answers.get("risk_a"))
        risk_b = _value(answers.get("risk_b"))
        model_risk = risk_a if risk_a == risk_b and risk_a in {"clear", "caution", "emergency", "unknown"} else None

        posture_answer = answers.get("posture")
        posture = _value(posture_answer)
        if posture not in {"focus", "sweep_left", "sweep_right", "steady_fire", "evade", "recover"}:
            posture = None
        elif _probability(posture_answer, posture) < 0.18:
            posture = None

        target_answer = answers.get("target")
        target = _value(target_answer)
        valid_targets = {"none"} | {"t" + str(index) for index in range(4)}
        if target not in valid_targets or _probability(target_answer, target) < 0.14:
            target = None
        if target is not None and target != "none":
            index = int(target[1:])
            objects = self.features.get("objects", [])
            if index < len(objects):
                self.jev_target_slot = objects[index]["slot"]
                self.jev_target_size = objects[index]["size"]
            else:
                self.jev_target_slot = None
                self.jev_target_size = None
        elif target == "none":
            self.jev_target_slot = None
            self.jev_target_size = None

        defense_answer = answers.get("hyperspace")
        self.jev_defense = _yes_probability(defense_answer) >= 0.78 and model_risk == "emergency"
        self.jev_posture = posture or ("evade" if model_risk == "emergency" else None)
        self.jev_answer_frame = answer_frame
        self.jev_answer_lives = answer_lives
        try:
            self.ctx.log(
                "jev_policy",
                frame=self.frame,
                judged_frame=answer_frame,
                age=age,
                risk=model_risk,
                posture=self.jev_posture,
                target=self.jev_target_slot,
                defense=self.jev_defense,
            )
        except Exception:
            pass
        return True

    def accept_astra(self, text):
        data = _extract_json(text)
        if not isinstance(data, dict):
            return False
        choices = {
            "posture": {"auto", "steady_fire", "sweep_left", "sweep_right"},
            "risk": {"conservative", "balanced", "aggressive"},
            "target_priority": {"threat_then_small", "small_then_threat", "nearest"},
            "defense": {"never", "jev_emergency_only"},
            "sweep_direction": {"auto", "left", "right"},
        }
        changed = {}
        for key, allowed in choices.items():
            value = data.get(key)
            if isinstance(value, str) and value in allowed:
                changed[key] = value
        if not changed:
            return False
        self.plan.update(changed)
        if self.plan["sweep_direction"] == "left":
            self.sweep_direction = 1
        elif self.plan["sweep_direction"] == "right":
            self.sweep_direction = -1
        try:
            self.ctx.log("astra_policy", frame=self.frame, plan=self.plan)
        except Exception:
            pass
        return True

    def _action(self, name):
        return name if name in self.names else self.noop

    def _fire(self):
        return self._action("FIRE")

    def _turn(self, direction, fire=False):
        # LEFT increments the code; positive means counterclockwise.
        if direction > 0:
            return self._action("LEFTFIRE" if fire else "LEFT")
        return self._action("RIGHTFIRE" if fire else "RIGHT")

    def _sweep(self):
        if self.plan.get("sweep_direction") == "left":
            self.sweep_direction = 1
        elif self.plan.get("sweep_direction") == "right":
            self.sweep_direction = -1
        elif self.frame >= self.sweep_until:
            self.sweep_direction *= -1
            self.sweep_until = self.frame + 48
        return self.sweep_direction

    def _target(self, objects):
        if self.jev_target_slot is not None and self.frame - self.jev_answer_frame <= 36:
            for item in objects:
                # Slots are reusable; size continuity plus a short freshness
                # window prevents a post-split object inheriting an old target.
                if item["slot"] == self.jev_target_slot and item["size"] == self.jev_target_size:
                    return item
        return objects[0] if objects else None

    def choose(self):
        facts = self.features
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        if self.frame - self.jev_answer_frame > 72:
            self.jev_posture = None
            self.jev_target_slot = None
            self.jev_target_size = None
            self.jev_defense = False

        if not facts.get("player_present"):
            self.fire_cooldown = max(self.fire_cooldown, 2)
            self.last_action = self.noop
            return self.noop

        danger = facts.get("danger", "unknown")
        if (
            danger == "emergency"
            and self.jev_defense
            and self.plan.get("defense") != "never"
            and self.frame - self.last_defense_frame >= 240
        ):
            self.last_defense_frame = self.frame
            self.jev_defense = False
            self.fire_cooldown = max(self.fire_cooldown, 3)
            self.last_action = self._action("DOWN")
            return self.last_action

        posture = self.jev_posture or self.plan.get("posture", "auto")
        if facts.get("motion_discontinuity") and posture == "focus":
            posture = "recover"
        if danger == "emergency":
            # The deterministic escape does not require Jev and never thrusts.
            nearest = min(facts.get("objects", []), key=lambda item: item["distance"], default=None)
            if nearest is not None:
                away_code = _heading_for_vector(-nearest["dx"], -nearest["dy"])
                heading = facts.get("heading_code")
                if heading is not None:
                    delta = (away_code - heading) % 16
                    if delta > 8:
                        delta -= 16
                    if delta:
                        action = self._turn(1 if delta > 0 else -1, False)
                        self.last_action = action
                        return action
            action = self._turn(self._sweep(), False)
            self.last_action = action
            return action

        if posture == "recover":
            action = self._turn(self._sweep(), False) if facts.get("objects") else self.noop
            self.last_action = action
            return action

        if posture in {"sweep_left", "sweep_right"}:
            direction = 1 if posture == "sweep_left" else -1
            fire = self.fire_cooldown == 0
            if fire:
                self.fire_cooldown = 4
            action = self._turn(direction, fire)
            self.last_action = action
            return action

        objects = facts.get("objects", [])
        target = self._target(objects)
        if posture == "steady_fire":
            if self.fire_cooldown == 0:
                self.fire_cooldown = 4
                action = self._fire()
            else:
                action = self.noop
            self.last_action = action
            return action

        if target is None:
            fire = self.fire_cooldown == 0
            if fire:
                self.fire_cooldown = 4
            action = self._turn(self._sweep(), fire)
            self.last_action = action
            return action

        heading = facts.get("heading_code")
        if heading is None:
            action = self.noop
        else:
            delta = (target["aim_code"] - heading) % 16
            if delta > 8:
                delta -= 16
            if abs(delta) > 1:
                fire = self.fire_cooldown == 0
                if fire:
                    self.fire_cooldown = 4
                action = self._turn(1 if delta > 0 else -1, fire)
            elif self.fire_cooldown == 0:
                self.fire_cooldown = 4
                action = self._fire()
            else:
                action = self.noop
        self.last_action = action
        return action


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _jev_future(self, state, questions):
        return self.ctx.spawn(lambda: self.ctx.decide(state, questions, timeout_s=2.0))

    def _astra_future(self, controller):
        summary = {
            "frame": controller.frame,
            "score": controller.score,
            "lives": controller.lives,
            "local_facts": {
                key: value for key, value in controller.features.items() if key != "objects"
            },
            "recent_policy": controller.plan,
            "jev_posture": controller.jev_posture,
            "last_action": controller.last_action,
        }
        prompt = (
            "You are the slow strategic supervisor for real-time Atari 2600 Asteroids "
            "mode 0/difficulty 0. The local controller owns every native frame and "
            "will ignore prose or action sequences. Review this aggregate state:\n"
            + json.dumps(summary, separators=(",", ":"))
            + "\nReturn JSON only, with any subset of these exact fields: "
            "posture (auto, steady_fire, sweep_left, sweep_right), "
            "risk (conservative, balanced, aggressive), "
            "target_priority (threat_then_small, small_then_threat, nearest), "
            "defense (never, jev_emergency_only), "
            "sweep_direction (auto, left, right). Choose a stable policy for the "
            "next several hundred frames. Do not invent enemy coverage."
        )
        return self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=90.0))

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(game, "tick_s", None) is None:
            return

        controller = LocalPolicy(game, ctx)
        controller.update(_observation(game))
        try:
            ctx.act(controller.choose())
        except (BudgetExceeded, EpisodeOver):
            return

        s1_enabled = getattr(ctx, "system1", True) is not None
        jev_future = None
        jev_frame = None
        jev_lives = None
        next_jev_frame = controller.frame

        astra_future = None
        astra_started = False
        astra_due = controller.frame + 480

        while not ctx.done and not getattr(game, "done", False):
            if s1_enabled and jev_future is None and controller.frame >= next_jev_frame:
                try:
                    judged_frame = controller.frame
                    judged_lives = controller.lives
                    jev_future = self._jev_future(controller.question_state(), controller.questions())
                    jev_frame = judged_frame
                    jev_lives = judged_lives
                    next_jev_frame = judged_frame + 24
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except ModelUnavailable:
                    raise
                except Exception:
                    s1_enabled = False

            if not astra_started and astra_future is None and controller.frame >= astra_due:
                try:
                    remaining = ctx.budget.remaining()
                    if _int(remaining.get("system2_calls"), 0) > 0:
                        astra_future = self._astra_future(controller)
                    astra_started = True
                except (BudgetExceeded, EpisodeOver):
                    astra_started = True
                except ModelUnavailable:
                    raise
                except Exception:
                    astra_started = True

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return
            if ctx.done or getattr(game, "done", False):
                break

            controller.update(_observation(game))

            if jev_future is not None and jev_future.done():
                finished = jev_future
                jev_future = None
                try:
                    controller.accept_jev(finished.result(), jev_frame, jev_lives)
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except ModelUnavailable:
                    raise
                except (TimeoutError, RuntimeError, TypeError, ValueError, KeyError, AttributeError):
                    pass
                except Exception:
                    # Garbage stand-in responses and malformed provider payloads
                    # must leave the local policy running.
                    pass

            if astra_future is not None and astra_future.done():
                finished = astra_future
                astra_future = None
                try:
                    result = finished.result()
                    controller.accept_astra(getattr(result, "text", result))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    raise
                except (TimeoutError, RuntimeError, TypeError, ValueError, KeyError, AttributeError):
                    pass
                except Exception:
                    pass

            try:
                ctx.act(controller.choose())
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx):
    return Harness(ctx)
