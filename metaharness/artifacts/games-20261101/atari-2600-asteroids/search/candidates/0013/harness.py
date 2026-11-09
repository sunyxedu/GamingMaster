"""Adaptive-lane Asteroids controller with asynchronous semantic supervision.

The controller keeps the high-yield part of the previous sweep policy (short fire
presses and code-owned aiming), but treats collision avoidance as a position
problem rather than a last-second turn.  It predicts which horizontal lanes will
be crossed by the observed rocks, makes a small, explicitly throttled thrust
pulse toward a safer lane, and returns to firing when the lane is clear.  Models
can select a tactical posture and a durable policy; neither model owns a native
frame or emits a raw action sequence.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


WIDTH = 160.0
HEIGHT = 210.0
DEFAULT_ACTIONS = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
    "UPLEFTFIRE",
)


def _finite(value, fallback=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _maybe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value, fallback=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _wrap_delta(value, period):
    value = _finite(value)
    while value > period / 2.0:
        value -= period
    while value < -period / 2.0:
        value += period
    return value


def _bucket(value, cuts, names):
    if value is None:
        return names[0]
    for cut, name in zip(cuts, names[1:]):
        if value < cut:
            return name
    return names[-1]


def _observation(game):
    try:
        value = game.observe()
    except Exception:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def _answer_probability(answer, key):
    probabilities = answer.get("probabilities", {}) if isinstance(answer, dict) else getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    return max(0.0, min(1.0, _finite(probabilities.get(key), 0.0)))


def _yes_probability(answer):
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, _finite(value, 0.0)))
    return _answer_probability(answer, "true")


def _heading_to(dx, dy):
    """Convert a vector to the documented 16-code nose convention."""
    if abs(dx) + abs(dy) < 1e-7:
        return 0
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _extract_object(text):
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


class LanePilot:
    """Owns every frame and separates lane safety from aiming."""

    def __init__(self, game, ctx):
        self.game = game
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        names = getattr(spec, "names", DEFAULT_ACTIONS) if spec is not None else DEFAULT_ACTIONS
        try:
            self.names = set(names or DEFAULT_ACTIONS)
        except Exception:
            self.names = set(DEFAULT_ACTIONS)
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else next(iter(self.names), "NOOP")

        self.frame = 0
        self.lives = 0
        self.previous_lives = None
        self.life_epoch = 0
        self.score = 0.0
        self.obs = {}
        self.facts = {}
        self.plan = {
            "aim": "target",
            "patrol": "adaptive_lane",
            "risk": "cautious",
            "defense": "last_resort",
            "fire": "normal",
        }

        self.tactical = "focus"
        self.tactical_until = -1
        self.tactical_target = None
        self.model_defense_hint = False
        self.judgment_frame = -100000
        self.judgment_epoch = -1

        self.lane_goal = None
        self.lane_goal_until = -1
        self.next_lane_review = 0
        self.thrust_until = -1
        self.next_thrust = 0
        # Emergency relocation may bypass the normal patrol cooldown, but this
        # separate gate prevents one forecast from becoming held thrust.
        self.next_urgent_thrust = 0
        self.next_fire = 0
        # Hyperspace is an escape attempt, not a once-per-life resource.  Keep a
        # small bounded retry budget and a real inter-pulse cooldown.
        self.defense_count = 0
        self.defense_used = False
        self.last_defense = -100000
        self.last_action = self.noop
        self.last_log = -100000

    # ---------- observation and prediction ----------

    def _player(self, obs):
        return _dict(obs.get("player"))

    def _rock_records(self, obs, player):
        if not bool(player.get("present")):
            return []
        pw = _finite(player.get("width"), 6.0) or 6.0
        ph = _finite(player.get("height"), 10.0) or 10.0
        px = _finite(player.get("x"), WIDTH / 2.0) + pw / 2.0
        py = _finite(player.get("y"), HEIGHT / 2.0) + ph / 2.0
        velocity = player.get("velocity_px_per_frame")
        pvx = _maybe_float(velocity[0]) if isinstance(velocity, list) and len(velocity) >= 2 else None
        pvy = _maybe_float(velocity[1]) if isinstance(velocity, list) and len(velocity) >= 2 else None
        records = []
        for index, raw0 in enumerate(_list(obs.get("asteroids"))):
            raw = _dict(raw0)
            if raw.get("x") is None or raw.get("y") is None:
                continue
            aw = _finite(raw.get("width"), 8.0) or 8.0
            ah = _finite(raw.get("height"), 15.0) or 15.0
            ax = _finite(raw.get("x")) + aw / 2.0
            ay = _finite(raw.get("y")) + ah / 2.0
            rv = raw.get("velocity_px_per_frame")
            avx = _maybe_float(rv[0]) if isinstance(rv, list) and len(rv) >= 2 else None
            avy = _maybe_float(rv[1]) if isinstance(rv, list) and len(rv) >= 2 else None
            # A null velocity is uncertainty, not zero motion.  Zero is used for
            # a bounded conservative forecast, while the confidence bit remains false.
            vx = (avx - pvx) if avx is not None and pvx is not None else 0.0
            vy = (avy - pvy) if avy is not None and pvy is not None else 0.0
            size = str(raw.get("size") or "unknown").lower()
            value = {"small": 3, "medium": 2, "large": 1}.get(size, 1)
            records.append({
                "slot": _integer(raw.get("source_slot"), index),
                "size": size,
                "x": ax,
                "y": ay,
                "dx": _wrap_delta(ax - px, WIDTH),
                "dy": _wrap_delta(ay - py, HEIGHT),
                "vx": vx,
                "vy": vy,
                "motion_known": avx is not None and avy is not None and pvx is not None and pvy is not None,
                "width": aw,
                "height": ah,
                "value": value,
            })
        return records

    @staticmethod
    def _impact_time(rock, lane_x, player_y, player_w=6.0, player_h=10.0, horizon=72):
        """Return the first sampled swept-box contact time in native frames."""
        x_clear = rock["width"] / 2.0 + player_w / 2.0 + 3.0
        y_clear = rock["height"] / 2.0 + player_h / 2.0 + 3.0
        # Two-frame sampling is enough for the deliberately coarse geometry and
        # keeps the controller comfortably below one native frame on the host.
        for t in range(0, horizon + 1, 2):
            dx = _wrap_delta(rock["x"] + rock["vx"] * t - lane_x, WIDTH)
            dy = _wrap_delta(rock["y"] + rock["vy"] * t - player_y, HEIGHT)
            if abs(dx) <= x_clear and abs(dy) <= y_clear:
                return t
        return None

    def _lane_cost(self, lane_x, records, player_y):
        total = 0.0
        first = None
        for rock in records:
            impact = self._impact_time(rock, lane_x, player_y)
            if impact is None:
                continue
            if first is None or impact < first:
                first = impact
            # Small rocks are worth more, but every contact is primarily a
            # survival cost.  Earlier crossings dominate later ones.
            urgency = 1.0 + max(0.0, (48.0 - impact) / 16.0)
            uncertainty = 1.30 if not rock["motion_known"] else 1.0
            total += (2.0 + rock["value"]) * urgency * uncertainty
        return total, first

    def _derive(self, obs):
        player = self._player(obs)
        present = bool(player.get("present")) and player.get("x") is not None and player.get("y") is not None
        if not present:
            return {
                "frame": self.frame,
                "score": self.score,
                "lives": self.lives,
                "present": False,
                "x": None,
                "y": None,
                "heading": None,
                "vx": None,
                "vy": None,
                "speed": None,
                "rocks": [],
                "threat": None,
                "danger": "unknown",
                "lanes": [],
                "lane_cost": 0.0,
                "missiles": 0,
                "discontinuity": True,
            }

        pw = _finite(player.get("width"), 6.0) or 6.0
        ph = _finite(player.get("height"), 10.0) or 10.0
        px = _finite(player.get("x"), WIDTH / 2.0) + pw / 2.0
        py = _finite(player.get("y"), HEIGHT / 2.0) + ph / 2.0
        velocity = player.get("velocity_px_per_frame")
        vx = _maybe_float(velocity[0]) if isinstance(velocity, list) and len(velocity) >= 2 else None
        vy = _maybe_float(velocity[1]) if isinstance(velocity, list) and len(velocity) >= 2 else None
        records = self._rock_records(obs, player)
        threat_data = []
        for rock in records:
            time = self._impact_time(rock, px, py, pw, ph, 90)
            if time is not None:
                threat_data.append((time, rock))
        threat_data.sort(key=lambda pair: (pair[0], -pair[1]["value"], pair[1]["slot"]))
        threat = threat_data[0][1] if threat_data else None
        threat_time = threat_data[0][0] if threat_data else None
        if threat_time is None:
            danger = "clear"
        elif threat_time <= 8:
            danger = "immediate"
        elif threat_time <= 30:
            danger = "watch"
        else:
            danger = "distant"
        lane_values = []
        for lane in (45.0, 65.0, 85.0, 105.0, 125.0):
            cost, first = self._lane_cost(lane, records, py)
            lane_values.append({"x": lane, "cost": cost, "first": first})
        current_cost, current_first = self._lane_cost(px, records, py)
        missiles = sum(1 for item in _list(obs.get("player_missiles")) if isinstance(item, dict))
        return {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "present": True,
            "x": px,
            "y": py,
            "heading": _integer(player.get("orientation_code"), 0) % 16,
            "vx": vx,
            "vy": vy,
            "speed": math.hypot(vx, vy) if vx is not None and vy is not None else None,
            "rocks": records,
            "threat": threat,
            "threat_time": threat_time,
            "danger": danger,
            "lanes": lane_values,
            "lane_cost": current_cost,
            "lane_first": current_first,
            "missiles": missiles,
            "discontinuity": bool(player.get("motion_discontinuity")),
        }

    def update(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        self.obs = obs
        old_lives = self.lives
        self.frame = _integer(obs.get("frame"), self.frame + 1)
        self.score = _finite(obs.get("score"), self.score)
        self.lives = _integer(obs.get("lives"), self.lives)
        changed = self.previous_lives is not None and self.lives != self.previous_lives
        if changed:
            self.life_epoch += 1
            self.lane_goal = None
            self.lane_goal_until = -1
            self.thrust_until = -1
            self.next_thrust = self.frame + 12
            self.next_urgent_thrust = self.frame
            self.next_fire = self.frame + 8
            self.defense_count = 0
            self.defense_used = False
            self.last_defense = -100000
            self.model_defense_hint = False
            self.tactical = "focus"
            self.tactical_until = self.frame + 36
            self.judgment_epoch = -1
            self._log("ship_change", frame=self.frame, old_lives=old_lives, lives=self.lives, score=self.score)
        self.previous_lives = self.lives
        self.facts = self._derive(obs)
        self._refresh_lane_goal(force=changed)

    def _refresh_lane_goal(self, force=False):
        facts = self.facts
        if not facts.get("present"):
            self.lane_goal = None
            return
        if not force and self.frame < self.next_lane_review and self.lane_goal_until >= self.frame:
            return
        lanes = facts.get("lanes", [])
        if not lanes:
            return
        current_x = facts["x"]
        current_cost = _finite(facts.get("lane_cost"), 0.0)
        # Hysteresis prevents the safety planner from chasing one flickering box.
        scored = []
        for lane in lanes:
            distance_penalty = abs(_wrap_delta(lane["x"] - current_x, WIDTH)) * 0.018
            scored.append((lane["cost"] + distance_penalty, lane["x"], lane["first"]))
        scored.sort(key=lambda item: (item[0], abs(item[1] - current_x)))
        best_score, best_x, best_first = scored[0]
        risk = self.plan.get("risk")
        trigger = 5.0 if risk == "cautious" else 7.0
        if self.plan.get("patrol") == "minimal":
            trigger += 2.0
        if current_cost >= trigger or (facts.get("danger") == "immediate"):
            # If the current lane is already safer than every alternative, keep
            # it.  Otherwise create a bounded relocation goal.
            if best_score + 1.0 < current_cost or facts.get("danger") == "immediate":
                self.lane_goal = best_x
                self.lane_goal_until = self.frame + 100
            elif self.lane_goal is None or self.lane_goal_until < self.frame:
                self.lane_goal = current_x
                self.lane_goal_until = self.frame + 45
        elif self.lane_goal is None or self.lane_goal_until < self.frame:
            self.lane_goal = current_x
            self.lane_goal_until = self.frame + 70
        self.next_lane_review = self.frame + (12 if facts.get("danger") in {"immediate", "watch"} else 45)
        if self.lane_goal is not None:
            self._log("lane_plan", frame=self.frame, goal=round(self.lane_goal, 1), current_cost=round(current_cost, 2), best_cost=round(best_score, 2), first=best_first)

    # ---------- compact model interface ----------

    def _lane_label(self, lane):
        if self.facts.get("x") is None:
            return "unknown"
        d = _wrap_delta(lane["x"] - self.facts["x"], WIDTH)
        side = "stay" if abs(d) < 12 else "left" if d < 0 else "right"
        if lane["cost"] < 4:
            risk = "safe"
        elif lane["cost"] < 10:
            risk = "watch"
        else:
            risk = "hot"
        return side + "_" + risk

    def state_for_jev(self):
        f = self.facts
        target = self._selected_target(f.get("rocks", []))
        target_summary = None
        if target is not None:
            target_summary = {
                "slot": target["slot"],
                "size": target["size"],
                "bearing": self._bearing(target["dx"], target["dy"]),
                "motion": "known" if target["motion_known"] else "uncertain",
                "distance": _bucket(math.hypot(target["dx"], target["dy"]), (25, 70, 120), ("unknown", "near", "middle", "far")),
            }
        lanes = []
        for lane in f.get("lanes", []):
            lanes.append({"id": self._lane_id(lane["x"]), "relation": self._lane_label(lane)})
        return {
            "rules": {
                "mode": 0,
                "fire": "level_press_then_release",
                "noop": "does_not_brake",
                "down": "mode_0_hyperspace",
                "heading": "left_increments_right_decrements",
                "enemy_objects": "unknown_not_absent",
            },
            "snapshot": {
                "frame": int(f.get("frame", 0)),
                "lives": int(f.get("lives", 0)),
                "score": int(max(0, f.get("score", 0.0))),
                "player_present": bool(f.get("present")),
                "position": "unknown" if f.get("x") is None else _bucket(f.get("x"), (55, 105), ("unknown", "left", "middle", "right")),
                "motion": "unknown" if f.get("speed") is None else ("still" if f["speed"] < 0.2 else "moving"),
                "danger": f.get("danger", "unknown"),
                "impact_window": _bucket(f.get("threat_time"), (9, 31, 61), ("none", "immediate", "soon", "later")),
                "rocks": min(15, len(f.get("rocks", []))),
                "missiles": min(2, int(f.get("missiles", 0))),
            },
            "lanes": lanes,
            "target": target_summary,
            "controller": {
                "posture": self.tactical,
                "lane_goal": self._lane_id(self.lane_goal) if self.lane_goal is not None else "none",
                "defense_count": int(self.defense_count),
                "defense_retry_ready": bool(self.frame - self.last_defense >= 16),
                "frame_local": True,
            },
            "plan": dict(self.plan),
        }

    @staticmethod
    def _lane_id(x):
        if x is None:
            return "none"
        value = int(round(float(x)))
        return "L" + str(value)

    @staticmethod
    def _bearing(dx, dy):
        horizontal = "right" if dx > 10 else "left" if dx < -10 else "center"
        vertical = "below" if dy > 10 else "above" if dy < -10 else "level"
        return horizontal + "_" + vertical

    def _selected_target(self, records):
        if self.tactical_target is not None and self.frame - self.judgment_frame <= 48:
            slot, size = self.tactical_target
            for rock in records:
                if rock["slot"] == slot and rock["size"] == size:
                    return rock
        if not records:
            return None
        def key(rock):
            distance = math.hypot(rock["dx"], rock["dy"])
            threat_bonus = 5.0 if self._impact_time(rock, self.facts.get("x", WIDTH / 2), self.facts.get("y", HEIGHT / 2), 6, 10, 42) is not None else 0.0
            return -(threat_bonus + 2.5 * rock["value"] + (1.0 if rock["motion_known"] else 0.0) - distance / 120.0)
        return min(records, key=key)

    def questions(self):
        danger_text = {
            "clear": "No observed swept-box crossing is near the player; preserve the safe lane and fire in pulses.",
            "distant": "A crossing is predicted but not soon; keep firing and allow only a bounded lane correction.",
            "watch": "A computed crossing is within the next half-second; prefer a short lane correction over thrust commitment.",
            "immediate": "A computed swept-box contact is imminent; survival overrides target choice and one bounded defense retry may be justified.",
            "unknown": "The player or geometry is unavailable; use the no-thrust conservative fallback.",
        }
        mode_text = {
            "focus": "Follow the code-selected target with short turns and fire pulses; never add thrust by this answer.",
            "sweep_left": "Make a bounded counterclockwise aiming sweep while firing in pulses.",
            "sweep_right": "Make a bounded clockwise aiming sweep while firing in pulses.",
            "hold": "Hold the current heading and use only the local fire cadence.",
            "avoid": "Favor the code-selected safe lane and suspend scoring only while danger is watch or immediate.",
        }
        order = ["focus", "sweep_left", "sweep_right", "hold", "avoid"]
        if (self.frame // 24) % 2:
            order = list(reversed(order))
        reversed_order = list(reversed(order))
        risk_order = ["clear", "distant", "watch", "immediate", "unknown"]
        if (self.frame // 48) % 2:
            risk_order = list(reversed(risk_order))
        lane_options = {"stay": "Keep the current lane; code will reject this if a swept collision is predicted."}
        for lane in self.facts.get("lanes", []):
            lane_options[self._lane_id(lane["x"])] = "Use this precomputed lane only as a bounded relocation suggestion: " + self._lane_label(lane) + "."
        lane_order = list(lane_options)
        if (self.frame // 24) % 2:
            lane_order.reverse()
        lane_options = {key: lane_options[key] for key in lane_order}
        return {
            "danger_a": Choice(
                "Classify the immediate risk using only snapshot.danger and snapshot.impact_window. Do not calculate distances or treat unknown enemy coverage as absence.",
                {key: danger_text[key] for key in risk_order},
            ),
            "danger_b": Choice(
                "Independently classify the same snapshot risk. Immediate means the code reports an imminent swept-box contact, not merely a nearby rock.",
                {key: danger_text[key] for key in reversed(risk_order)},
            ),
            "posture_a": Choice(
                "Choose one short-lived tactical posture. It is semantic advice to a frame-local controller, not a raw action or a sequence.",
                {key: mode_text[key] for key in order},
            ),
            "posture_b": Choice(
                "Choose the safest useful bounded posture for exactly this state. The controller owns arithmetic, timing, and legality.",
                {key: mode_text[key] for key in reversed_order},
            ),
            "lane": Choice(
                "Choose a precomputed lane suggestion or stay. Do not compare numeric distances; code performs the safety check again.",
                lane_options,
            ),
            "defense": Noul(
                "Should one mode-0 hyperspace pulse be considered if the local controller reports immediate impact, the retry budget is not exhausted, and the inter-pulse gap is satisfied? Otherwise answer false.",
                {"true": "One bounded last-resort defense retry is warranted for an imminent collision.", "false": "Continue with steering or firing; do not invoke defense now."},
            ),
        }

    def accept_jev(self, answers, judged_frame, judged_epoch):
        if not isinstance(answers, dict):
            return
        if judged_epoch != self.life_epoch or self.frame - judged_frame < 0 or self.frame - judged_frame > 48:
            self._log("jev_discard", frame=self.frame, judged_frame=judged_frame, reason="stale_or_life")
            return
        da = _answer_value(answers.get("danger_a"))
        db = _answer_value(answers.get("danger_b"))
        danger_ok = da == db and da in {"clear", "distant", "watch", "immediate", "unknown"}
        if danger_ok:
            probability = min(_answer_probability(answers.get("danger_a"), da), _answer_probability(answers.get("danger_b"), db))
            if probability >= (0.45 if da == "immediate" else 0.16):
                self.model_defense_hint = da == "immediate"
        pa = _answer_value(answers.get("posture_a"))
        pb = _answer_value(answers.get("posture_b"))
        allowed = {"focus", "sweep_left", "sweep_right", "hold", "avoid"}
        if pa == pb and pa in allowed:
            probability = min(_answer_probability(answers.get("posture_a"), pa), _answer_probability(answers.get("posture_b"), pb))
            if probability >= 0.16:
                self.tactical = pa
                self.tactical_until = self.frame + 54
        lane = _answer_value(answers.get("lane"))
        if isinstance(lane, str) and lane.startswith("L") and lane[1:].isdigit():
            if _answer_probability(answers.get("lane"), lane) >= 0.16:
                candidate = _finite(lane[1:], None)
                if candidate is not None:
                    # Jev can suggest a relocation only when the numeric forecast
                    # agrees that it is no worse than the current lane.  In
                    # particular, a first-option "stay" answer cannot cancel a
                    # safety goal that code has already chosen.
                    lane_facts = [item for item in self.facts.get("lanes", []) if abs(candidate - item["x"]) < 1.0]
                    if lane_facts:
                        chosen_cost = lane_facts[0]["cost"]
                        current_cost = _finite(self.facts.get("lane_cost"), 0.0)
                        if chosen_cost <= current_cost + 1.5 and (current_cost >= 5.0 or chosen_cost + 2.0 < current_cost):
                            self.lane_goal = candidate
                            self.lane_goal_until = max(self.lane_goal_until, self.frame + 30)
        self.model_defense_hint = self.model_defense_hint or _yes_probability(answers.get("defense")) >= 0.78
        self.judgment_frame = judged_frame
        self.judgment_epoch = judged_epoch
        self._log("jev_advice", frame=self.frame, judged_frame=judged_frame, posture=self.tactical, defense=self.model_defense_hint, danger=da if danger_ok else "disagree")

    # ---------- action bridge ----------

    def legal(self, action):
        return action if action in self.names else self.noop

    def _fire_ready(self):
        if self.frame < self.next_fire:
            return False
        if self.facts.get("missiles", 0) >= 2:
            return False
        self.next_fire = self.frame + (6 if self.plan.get("fire") == "spaced" else 5)
        return True

    def _turn_sign(self, current, desired):
        delta = (desired - current) % 16
        if delta == 0:
            return 0
        if delta < 8:
            return 1  # LEFT increments the orientation code.
        if delta > 8:
            return -1  # RIGHT decrements it.
        return 1 if ((self.frame // 16) % 2 == 0) else -1

    def _compose(self, turn, thrust=False, fire=False):
        if thrust:
            if turn > 0:
                name = "UPLEFTFIRE" if fire else "UPLEFT"
            elif turn < 0:
                name = "UPRIGHTFIRE" if fire else "UPRIGHT"
            else:
                name = "UPFIRE" if fire else "UP"
        elif turn > 0:
            name = "LEFTFIRE" if fire else "LEFT"
        elif turn < 0:
            name = "RIGHTFIRE" if fire else "RIGHT"
        else:
            name = "FIRE" if fire else "NOOP"
        return self.legal(name)

    def _lane_motion(self):
        f = self.facts
        if not f.get("present") or self.lane_goal is None:
            return None
        x = f.get("x")
        if x is None:
            return None
        delta = _wrap_delta(self.lane_goal - x, WIDTH)
        vx = f.get("vx")
        # Brake an overshoot with one short pulse in the opposite horizontal
        # direction.  This is deliberately rarer than relocation thrust.
        braking = abs(delta) < 14 and vx is not None and abs(vx) > 0.18
        if abs(delta) <= 8 and not braking:
            return None
        direction = 1 if (vx < -0.18 if braking else delta < 0) else -1
        desired = 4 if direction > 0 else 12
        current = f.get("heading")
        if current is None:
            return None
        turn = self._turn_sign(current, desired)
        urgent = f.get("danger") == "immediate"
        # Use a combined turn/thrust only in the urgent case, when waiting for a
        # full cardinal rotation would spend the collision window.  Normal lane
        # travel first turns, then uses at most two-frame thrust bursts.  An
        # urgent lane relocation has its own short gate: it may bypass the
        # ordinary patrol cooldown, but never repeats on adjacent frames.
        if turn != 0:
            if urgent and self.frame >= self.next_urgent_thrust:
                self.thrust_until = self.frame + 1
                self.next_urgent_thrust = self.frame + 6
                self.next_thrust = max(self.next_thrust, self.frame + 22)
                return turn, True
            return turn, False
        if self.frame < self.thrust_until:
            return 0, True
        if urgent and self.frame >= self.next_urgent_thrust:
            self.thrust_until = self.frame + 1
            self.next_urgent_thrust = self.frame + 6
            self.next_thrust = max(self.next_thrust, self.frame + 28)
            return 0, True
        if self.frame >= self.next_thrust:
            self.thrust_until = self.frame + 2
            self.next_thrust = self.frame + 28
            self.next_urgent_thrust = max(self.next_urgent_thrust, self.frame + 2)
            return 0, True
        return None

    def _should_defend(self):
        f = self.facts
        # Test retries, but keep them rare: a fresh local contact, a three-pulse
        # per-life cap, and a 16-frame gap are all required.  No stale model
        # answer can manufacture a DOWN action.
        if (
            not f.get("present")
            or self.defense_count >= 3
            or self.frame - self.last_defense < 16
            or self.plan.get("defense") == "never"
        ):
            return False
        t = f.get("threat_time")
        if t is None or t > 4:
            return False
        # Local geometry is the hard gate; Jev can support a slightly earlier
        # pulse but cannot create one without an imminent computed contact.
        supported = self.model_defense_hint and t <= 14
        return t <= 2 or supported

    def _aim_turn(self):
        f = self.facts
        heading = f.get("heading")
        if heading is None:
            return 0
        posture = self.tactical if self.frame <= self.tactical_until else None
        if posture == "sweep_left":
            return 1
        if posture == "sweep_right":
            return -1
        if posture in {"hold", "avoid"}:
            return 0
        target = self._selected_target(f.get("rocks", []))
        if posture == "focus" and target is not None:
            projected_x = target["dx"] + target["vx"] * 8.0
            projected_y = target["dy"] + target["vy"] * 8.0
            return self._turn_sign(heading, _heading_to(projected_x, projected_y))
        if target is not None and self.plan.get("aim") == "target":
            return self._turn_sign(heading, _heading_to(target["dx"], target["dy"]))
        # The default is a slow alternating sweep: unlike an open-loop Jev
        # action, this sign is re-evaluated every frame and changes are bounded.
        return 1 if ((self.frame // 52) % 2 == 0) else -1

    def choose(self):
        f = self.facts
        if not f.get("present"):
            self.last_action = self.noop
            return self.noop
        if self.frame > self.tactical_until:
            if self.plan.get("aim") == "target":
                self.tactical = "focus"
            else:
                self.tactical = "sweep_right" if ((self.frame // 52) % 2 == 0) else "sweep_left"
            self.tactical_until = self.frame + 54

        if self._should_defend():
            self.defense_count += 1
            self.defense_used = True
            self.last_defense = self.frame
            self.model_defense_hint = False
            self.thrust_until = -1
            self._log(
                "defense",
                frame=self.frame,
                threat_time=f.get("threat_time"),
                danger=f.get("danger"),
                defense_count=self.defense_count,
            )
            self.last_action = self.legal("DOWN")
            return self.last_action

        lane_action = self._lane_motion()
        fire = self._fire_ready()
        if lane_action is not None:
            turn, thrust = lane_action
            action = self._compose(turn, thrust=thrust, fire=fire)
        else:
            turn = self._aim_turn()
            action = self._compose(turn, thrust=False, fire=fire)
        self.last_action = action
        if self.frame - self.last_log >= 600 or fire:
            self._log("control", frame=self.frame, action=action, danger=f.get("danger"), lane=self.lane_goal, score=self.score, lives=self.lives)
            self.last_log = self.frame
        return action

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _start_jev(self, pilot):
        if getattr(self.ctx, "system1", True) is None:
            return None
        state = pilot.state_for_jev()
        questions = pilot.questions()
        origin = pilot.frame
        epoch = pilot.life_epoch

        def request():
            return self.ctx.decide(state, questions, timeout_s=2.4)

        try:
            return self.ctx.spawn(request), origin, epoch
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    def _start_astra(self, pilot, reason):
        remaining = {}
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            pass
        if _integer(remaining.get("system2_calls"), 0) <= 0:
            return None
        f = pilot.facts
        summary = {
            "reason": reason,
            "frame": pilot.frame,
            "score": int(max(0, pilot.score)),
            "lives": pilot.lives,
            "danger": f.get("danger", "unknown"),
            "impact_window": f.get("threat_time"),
            "lane_cost": round(_finite(f.get("lane_cost"), 0.0), 2),
            "lane_goal": pilot.lane_goal,
            "rocks": len(f.get("rocks", [])),
            "last_action": pilot.last_action,
            "plan": dict(pilot.plan),
        }
        prompt = (
            "You are the slow strategic reviewer for a real-time Atari 2600 Asteroids "
            "mode-0/difficulty-0 controller. The local code owns the clock, collision "
            "forecast, lane relocation, aiming, FIRE release cadence, life bookkeeping, "
            "and emergency fallback. It will ignore actions, sequences, timestamps, and "
            "prose. Review this aggregate evidence:\n" + json.dumps(summary, separators=(",", ":")) +
            "\nReturn one small JSON object only, using any subset of these exact fields: "
            "aim ('sweep' or 'target'), patrol ('adaptive_lane' or 'minimal'), "
            "risk ('cautious' or 'balanced'), defense ('last_resort' or 'never'), "
            "fire ('normal' or 'spaced'). Keep movement bounded; do not assume unknown "
            "enemy coverage is empty."
        )

        def request():
            return self.ctx.think(prompt, timeout_s=105.0)

        try:
            return self.ctx.spawn(request)
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    @staticmethod
    def _parse_plan(text):
        data = _extract_object(text)
        if not isinstance(data, dict):
            return {}
        allowed = {
            "aim": {"sweep", "target"},
            "patrol": {"adaptive_lane", "minimal"},
            "risk": {"cautious", "balanced"},
            "defense": {"last_resort", "never"},
            "fire": {"normal", "spaced"},
        }
        result = {}
        for key, choices in allowed.items():
            value = data.get(key)
            if isinstance(value, str) and value in choices:
                result[key] = value
        return result

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(game, "tick_s", None) is None:
            return
        pilot = LanePilot(game, ctx)
        pilot.update(_observation(game))

        # Start with code-owned control immediately.  Astra is advisory and is
        # deliberately delayed long enough to see the launch family in action.
        try:
            ctx.act(pilot.choose())
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return
        except Exception:
            try:
                ctx.act(pilot.noop)
            except Exception:
                return

        jev_future = None
        jev_origin = -100000
        jev_epoch = -1
        next_jev = pilot.frame
        astra_future = None
        astra_started = False
        astra_revisions = 0
        astra_due = pilot.frame + 300
        model_enabled = getattr(ctx, "system1", True) is not None

        while not ctx.done and not getattr(game, "done", False):
            if model_enabled and jev_future is None and pilot.frame >= next_jev:
                try:
                    result = self._start_jev(pilot)
                    if result is not None:
                        jev_future, jev_origin, jev_epoch = result
                        next_jev = pilot.frame + 24
                    else:
                        next_jev = pilot.frame + 36
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    model_enabled = False
                except Exception:
                    next_jev = pilot.frame + 36

            if astra_future is None and astra_revisions < 2 and pilot.frame >= astra_due:
                try:
                    astra_future = self._start_astra(pilot, "scheduled review" if not astra_started else "ship loss review")
                    astra_started = True
                    astra_revisions += 1
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    astra_revisions = 2
                except Exception:
                    astra_revisions = 2

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            if ctx.done or getattr(game, "done", False):
                break

            pilot.update(_observation(game))

            if jev_future is not None and jev_future.done():
                finished = jev_future
                jev_future = None
                try:
                    answers = finished.result()
                    pilot.accept_jev(answers, jev_origin, jev_epoch)
                except (BudgetExceeded, EpisodeOver):
                    model_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    # Provider garbage, a cancelled future, or a stand-in answer
                    # is a tactical miss, never a harness failure.
                    pass

            if astra_future is not None and astra_future.done():
                finished = astra_future
                astra_future = None
                try:
                    result = finished.result()
                    plan = self._parse_plan(getattr(result, "text", result))
                    if plan:
                        pilot.plan.update(plan)
                        pilot._log("astra_plan", frame=pilot.frame, plan=dict(pilot.plan))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    return
                except Exception:
                    pass

            try:
                ctx.act(pilot.choose())
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                # A malformed snapshot cannot justify a dangerous action.  The
                # game contract still gets a fresh release attempt on this frame.
                try:
                    ctx.act(pilot.noop)
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except Exception:
                    return

            # A first Astra review can finish after a death.  If it did not, the
            # scheduled second slot is still bounded by the two-call cap.
            if pilot.previous_lives is not None and pilot.frame >= astra_due and astra_started and astra_future is None and astra_revisions < 2:
                astra_due = pilot.frame + 180


def build(ctx):
    return Harness(ctx)
