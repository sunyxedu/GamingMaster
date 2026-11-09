"""Verified lane-pulse crossover for the tier-2 real-time adaptor.

The controller has two deliberately separate loops.  A small geometric radar owns
all frame timing, FIRE release, collision arithmetic, and the only actions that can
be posted.  Jev supplies short semantic labels and Astra supplies a slow preference;
neither is allowed to hold an action.  When the radar cannot justify a maneuver it
falls back to a stationary, intermittent firing posture (the useful part of the
idle baseline), rather than turning uncertainty into thrust.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


WIDTH = 160.0
HEIGHT = 210.0
LANES = (44.0, 66.0, 88.0, 110.0, 132.0)
ACTIONS = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
    "UPLEFTFIRE",
)


def _number(value, default=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _maybe_number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _sequence(value):
    return value if isinstance(value, list) else []


def _wrapped_delta(value, period):
    x = _number(value)
    while x > period / 2.0:
        x -= period
    while x < -period / 2.0:
        x += period
    return x


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


def _extract_json_object(text):
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def _answer_probs(answer):
    if isinstance(answer, dict):
        value = answer.get("probabilities", {})
    else:
        value = getattr(answer, "probabilities", {})
    return value if isinstance(value, dict) else {}


def _probability(answer, key):
    return max(0.0, min(1.0, _number(_answer_probs(answer).get(key), 0.0)))


def _yes(answer):
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, _number(value)))
    return _probability(answer, "true")


def _heading_vector(code):
    """Return the documented cardinal convention, with interpolated codes."""
    angle = (_integer(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def _heading_for_vector(dx, dy):
    if abs(dx) + abs(dy) < 1e-7:
        return 0
    # code 0 is up, 4 left, 8 down, 12 right.
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _bucket(value, cuts, names):
    if value is None:
        return names[0]
    for cut, name in zip(cuts, names[1:]):
        if value < cut:
            return name
    return names[-1]


class Radar:
    """Frame-local state, swept forecasts, and a conservative action arbiter."""

    def __init__(self, game, ctx):
        self.game = game
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        try:
            names = list(getattr(spec, "names", ACTIONS)) if spec is not None else list(ACTIONS)
        except Exception:
            names = list(ACTIONS)
        self.names = set(names or ACTIONS)
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else (next(iter(self.names), "NOOP"))

        self.frame = 0
        self.score = 0.0
        self.lives = 0
        self.old_lives = None
        self.life_epoch = 0
        self.obs = {}
        self.facts = {}
        self.previous_position = None
        self.previous_present = False

        # These timers are intentional expiry boundaries.  They prevent a single
        # bad geometry frame, late answer, or defense teleport from becoming a
        # long held thrust.
        self.coast_until = 0
        self.escape_again = 0
        self.escape_pulses = 0
        self.last_escape = None
        self.defense_frames = []
        self.next_fire = 0
        self.next_aim_review = 0
        self.aim_mode = "target"
        self.aim_until = 0
        self.model_caution_until = 0
        self.model_defense_hint = False
        self.last_jev_frame = -100000
        self.last_jev_epoch = -1
        self.last_action = self.noop
        self.last_log_frame = -100000
        self.plan = {
            "risk": "strict",
            "fire": "normal",
            "aim": "target",
            "escape": "reactive",
        }

    # ---------- geometry ----------

    def _player(self):
        return _mapping(self.obs.get("player"))

    @staticmethod
    def _velocity(value):
        if isinstance(value, list) and len(value) >= 2:
            x = _maybe_number(value[0])
            y = _maybe_number(value[1])
            if x is not None and y is not None:
                return x, y
        return None, None

    def _rocks(self, player):
        if not player.get("present"):
            return []
        pw = _number(player.get("width"), 6.0) or 6.0
        ph = _number(player.get("height"), 10.0) or 10.0
        px = _number(player.get("x"), WIDTH / 2.0) + pw / 2.0
        py = _number(player.get("y"), HEIGHT / 2.0) + ph / 2.0
        pvx, pvy = self._velocity(player.get("velocity_px_per_frame"))
        result = []
        for index, item in enumerate(_sequence(self.obs.get("asteroids"))):
            rock = _mapping(item)
            if rock.get("x") is None or rock.get("y") is None:
                continue
            rw = _number(rock.get("width"), 8.0) or 8.0
            rh = _number(rock.get("height"), 15.0) or 15.0
            vx, vy = self._velocity(rock.get("velocity_px_per_frame"))
            size = str(rock.get("size") or "unknown").lower()
            value = {"small": 3, "medium": 2, "large": 1}.get(size, 1)
            result.append({
                "slot": _integer(rock.get("source_slot"), index),
                "size": size,
                "x": _number(rock.get("x")) + rw / 2.0,
                "y": _number(rock.get("y")) + rh / 2.0,
                "w": rw,
                "h": rh,
                "vx": vx if vx is not None else 0.0,
                "vy": vy if vy is not None else 0.0,
                "known": vx is not None and vy is not None,
                "dx": _wrapped_delta(_number(rock.get("x")) + rw / 2.0 - px, WIDTH),
                "dy": _wrapped_delta(_number(rock.get("y")) + rh / 2.0 - py, HEIGHT),
                "value": value,
            })
        return result

    @staticmethod
    def _touch_time(rock, player_x, player_y, player_vx, player_vy,
                    ax=0.0, ay=0.0, horizon=48, margin=4.5):
        """First conservative swept-box contact for one initial thrust pulse."""
        xgap = rock["w"] / 2.0 + 3.0 + margin
        ygap = rock["h"] / 2.0 + 5.0 + margin
        best_clearance = 1000.0
        first = None
        for t in range(horizon + 1):
            # The pulse changes velocity once, then inertia continues.  This is
            # only a ranking model; the real state is re-read on every frame.
            px = player_x + player_vx * t + ax * max(0.0, t - 0.5)
            py = player_y + player_vy * t + ay * max(0.0, t - 0.5)
            rx = rock["x"] + rock["vx"] * t
            ry = rock["y"] + rock["vy"] * t
            dx = abs(_wrapped_delta(rx - px, WIDTH))
            dy = abs(_wrapped_delta(ry - py, HEIGHT))
            clearance = max(dx - xgap, dy - ygap)
            if clearance < best_clearance:
                best_clearance = clearance
            if dx <= xgap and dy <= ygap:
                first = t
                break
        return first, best_clearance

    def _candidate_forecast(self, action_key, base=None, horizon=48):
        f = base if isinstance(base, dict) else self.facts
        if not f.get("present"):
            return {"contact": 0, "clearance": -100.0, "action": action_key}
        h = f.get("heading")
        if h is None:
            h = 0
        if action_key == "thrust":
            code = h
        elif action_key == "left_thrust":
            code = (h + 1) % 16
        elif action_key == "right_thrust":
            code = (h - 1) % 16
        else:
            code = h
        if action_key == "coast":
            acceleration = (0.0, 0.0)
        else:
            ux, uy = _heading_vector(code)
            # A conservative mid-range Atari thrust estimate.  It is used only
            # to choose between pulses; no numerical trajectory is trusted after
            # the next observation.
            acceleration = (0.055 * ux, 0.055 * uy)
        first = None
        clearance = 1000.0
        for rock in f.get("rocks", []):
            impact, local = self._touch_time(
                rock, f["x"], f["y"], f.get("vx") or 0.0, f.get("vy") or 0.0,
                acceleration[0], acceleration[1], horizon=horizon,
                margin=5.0 if not rock["known"] else 3.5,
            )
            if impact is not None and (first is None or impact < first):
                first = impact
            clearance = min(clearance, local)
        if not f.get("rocks"):
            clearance = 1000.0
        return {"contact": first, "clearance": clearance, "action": action_key}

    def _derive(self, obs):
        player = self._player()
        present = bool(player.get("present")) and player.get("x") is not None and player.get("y") is not None
        if not present:
            return {
                "present": False, "frame": self.frame, "score": self.score,
                "lives": self.lives, "x": None, "y": None, "heading": None,
                "vx": None, "vy": None, "speed": None, "rocks": [],
                "ttc": None, "danger": "unknown", "uncertain": True,
                "candidates": {}, "best_escape": "coast", "missiles": 0,
                "discontinuity": True, "target": None,
            }
        pw = _number(player.get("width"), 6.0) or 6.0
        ph = _number(player.get("height"), 10.0) or 10.0
        px = _number(player.get("x"), WIDTH / 2.0) + pw / 2.0
        py = _number(player.get("y"), HEIGHT / 2.0) + ph / 2.0
        vx, vy = self._velocity(player.get("velocity_px_per_frame"))
        rocks = self._rocks(player)
        current = {
            "present": True,
            "x": px,
            "y": py,
            "heading": _integer(player.get("orientation_code"), 0) % 16,
            "vx": vx,
            "vy": vy,
            "rocks": rocks,
        }
        candidates = {
            key: self._candidate_forecast(key, current)
            for key in ("coast",)
        }
        coast = candidates["coast"]
        ttc = coast.get("contact")
        if ttc is None:
            danger = "clear" if coast.get("clearance", 0.0) > 8.0 else "watch"
        elif ttc <= 5:
            danger = "immediate"
        elif ttc <= 20:
            danger = "watch"
        else:
            danger = "distant"
        uncertain = vx is None or vy is None or any(not r["known"] for r in rocks)
        # An unknown close object is not absence.  It suppresses proactive
        # thrust, but an actual overlap remains an emergency.
        near_unknown = any(
            not r["known"] and math.hypot(r["dx"], r["dy"]) < 42.0 for r in rocks
        )
        if near_unknown and danger == "clear":
            danger = "watch"
        target = self._target(rocks, px, py)
        safe = [v for v in candidates.values() if v["contact"] is None]
        if safe:
            best = max(safe, key=lambda v: (v["clearance"], -self._action_rank(v["action"])))
        else:
            best = max(candidates.values(), key=lambda v: (v["contact"] if v["contact"] is not None else 999, v["clearance"]))
        missiles = sum(1 for item in _sequence(self.obs.get("player_missiles")) if isinstance(item, dict))
        return {
            "present": True, "frame": self.frame, "score": self.score,
            "lives": self.lives, "x": px, "y": py, "heading": _integer(player.get("orientation_code"), 0) % 16,
            "vx": vx, "vy": vy, "speed": math.hypot(vx, vy) if vx is not None and vy is not None else None,
            "rocks": rocks, "ttc": ttc, "danger": danger, "uncertain": uncertain or near_unknown,
            "near_unknown": near_unknown, "candidates": candidates, "best_escape": best["action"],
            "best_clearance": best["clearance"], "missiles": missiles,
            "discontinuity": bool(player.get("motion_discontinuity")), "target": target,
        }

    @staticmethod
    def _action_rank(action):
        return {"coast": 0, "thrust": 1, "left_thrust": 2, "right_thrust": 3}.get(action, 4)

    @staticmethod
    def _target(rocks, px, py):
        if not rocks:
            return None
        def rank(rock):
            distance = math.hypot(rock["dx"], rock["dy"])
            close = 5.0 if distance < 80 else 0.0
            return -(close + 3.0 * rock["value"] + (1.0 if rock["known"] else 0.0) - distance / 150.0)
        return min(rocks, key=rank)

    def update(self, obs):
        if not isinstance(obs, dict):
            obs = {}
        self.obs = obs
        old_frame = self.frame
        self.frame = _integer(obs.get("frame"), old_frame + 1)
        self.score = _number(obs.get("score"), self.score)
        self.lives = _integer(obs.get("lives"), self.lives)
        life_changed = self.old_lives is not None and self.lives != self.old_lives
        player = _mapping(obs.get("player"))
        now_present = bool(player.get("present")) and player.get("x") is not None and player.get("y") is not None
        px = _number(player.get("x"), 0.0) if now_present else None
        py = _number(player.get("y"), 0.0) if now_present else None
        jump = False
        if now_present and self.previous_present and self.previous_position is not None:
            dx = abs(_wrapped_delta(px - self.previous_position[0], WIDTH))
            dy = abs(_wrapped_delta(py - self.previous_position[1], HEIGHT))
            jump = dx > 24.0 or dy > 30.0
        explicit_discontinuity = bool(player.get("motion_discontinuity"))
        discontinuity = (not now_present) or explicit_discontinuity or jump
        if life_changed:
            self.life_epoch += 1
            self.coast_until = self.frame + 34
            self.escape_again = self.frame + 16
            self.escape_pulses = 0
            self.defense_frames = []
            self.model_defense_hint = False
            self.aim_mode = "target"
            self.aim_until = self.frame + 36
            self._log("life_epoch", frame=self.frame, old_lives=self.old_lives, lives=self.lives)
        if discontinuity:
            self.coast_until = max(self.coast_until, self.frame + (24 if now_present else 12))
            self.escape_again = max(self.escape_again, self.frame + 8)
            self.last_escape = None
            if now_present:
                self._log("radar_reset", frame=self.frame, jump=jump, explicit=explicit_discontinuity)
        self.old_lives = self.lives
        self.previous_present = now_present
        self.previous_position = (px, py) if now_present else None
        self.facts = self._derive(obs)
        if self.frame >= self.next_aim_review:
            self.next_aim_review = self.frame + (18 if self.facts.get("danger") in {"watch", "immediate"} else 48)

    # ---------- compact Jev interface ----------

    @staticmethod
    def _bearing(dx, dy):
        return ("right" if dx > 10 else "left" if dx < -10 else "center") + "_" + (
            "below" if dy > 10 else "above" if dy < -10 else "level"
        )

    def state_for_jev(self):
        f = self.facts
        target = f.get("target")
        target_summary = None
        if target is not None:
            target_summary = {
                "slot": target["slot"], "size": target["size"],
                "bearing": self._bearing(target["dx"], target["dy"]),
                "motion": "known" if target["known"] else "uncertain",
                "distance": _bucket(math.hypot(target["dx"], target["dy"]), (28, 72, 120), ("unknown", "near", "middle", "far")),
            }
        candidate_summary = {}
        for key, value in f.get("candidates", {}).items():
            candidate_summary[key] = {
                "contact": value.get("contact"),
                "clearance": _bucket(value.get("clearance"), (-2, 4, 12), ("unknown", "overlap", "tight", "open")),
            }
        return {
            "rules": {
                "mode": 0, "difficulty": 0,
                "fire": "press_then_release; holding is not rapid fire",
                "noop": "coasts; it is not a brake",
                "down": "mode-0 hyperspace; bounded defense only",
                "heading": "LEFT increments and RIGHT decrements orientation code",
                "enemy_coverage": "unknown; null is not absence",
            },
            "snapshot": {
                "frame": int(f.get("frame", 0)), "lives": int(f.get("lives", 0)),
                "score": int(max(0, f.get("score", 0.0))),
                "player": "present" if f.get("present") else "absent",
                "position": "unknown" if f.get("x") is None else _bucket(f.get("x"), (54, 106), ("unknown", "left", "middle", "right")),
                "motion": "unknown" if f.get("speed") is None else ("still" if f["speed"] < 0.2 else "moving"),
                "danger": f.get("danger", "unknown"),
                "impact_window": _bucket(f.get("ttc"), (6, 21, 49), ("none", "immediate", "soon", "later")),
                "rocks": min(15, len(f.get("rocks", []))),
                "missiles": min(2, int(f.get("missiles", 0))),
                "uncertain": bool(f.get("uncertain")),
            },
            "forecast": {
                "best_escape": f.get("best_escape", "coast"),
                "candidates": candidate_summary,
                "proactive_thrust": "forbidden unless a local forecast supplies a bounded pulse",
            },
            "target": target_summary,
            "controller": {
                "aim": self.aim_mode,
                "coast_guard": max(0, self.coast_until - self.frame),
                "defenses_this_life": len(self.defense_frames),
                "frame_local": True,
            },
            "plan": dict(self.plan),
        }

    def questions(self):
        danger_text = {
            "clear": "No conservative swept contact is near; keep the ship stationary and fire only in short pulses.",
            "distant": "A contact is predicted later; do not thrust yet, preserve a firing posture.",
            "watch": "A contact may arrive soon; prefer a bounded escape pulse if the local forecast says it improves clearance.",
            "immediate": "Contact is imminent; survival overrides aiming and only a locally justified pulse or defense is allowed.",
            "unknown": "The player or geometry is unavailable; coast and release controls.",
        }
        posture_text = {
            "focus": "Turn in short fresh steps toward the selected rock and fire on the release cadence; never add thrust from this answer.",
            "sweep_left": "Make a short counterclockwise aiming sweep while preserving the no-thrust safety rule.",
            "sweep_right": "Make a short clockwise aiming sweep while preserving the no-thrust safety rule.",
            "hold": "Do not rotate or thrust; use only the bounded FIRE pulse.",
            "coast": "Use the idle-like fallback: no rotation or thrust, and fire only when the local release timer permits.",
        }
        risk_order = ["clear", "distant", "watch", "immediate", "unknown"]
        posture_order = ["focus", "sweep_left", "sweep_right", "hold", "coast"]
        if (self.frame // 36) % 2:
            risk_order.reverse()
            posture_order.reverse()
        return {
            "danger_one": Choice(
                "Classify the supplied snapshot danger. Use snapshot.danger and snapshot.impact_window as labels; do not calculate numbers and do not infer that unknown enemy coverage is empty.",
                {key: danger_text[key] for key in risk_order},
            ),
            "danger_two": Choice(
                "Independently classify the same state. `immediate` means the local snapshot reports an imminent swept-box contact, not merely a nearby asteroid.",
                {key: danger_text[key] for key in reversed(risk_order)},
            ),
            "posture": Choice(
                "Choose one short-lived semantic posture. The controller owns timing, geometry, action legality, FIRE release, and emergency overrides; do not describe a sequence.",
                {key: posture_text[key] for key in posture_order},
            ),
            "defense": Noul(
                "Should the local controller consider one mode-0 hyperspace pulse now? Answer true only for an imminent local contact when no forecasted thrust pulse is safe; otherwise false.",
                {"true": "A bounded last-resort defense is warranted by an imminent collision.", "false": "Do not use defense now; use the local coast or escape rule."},
            ),
        }

    def accept_jev(self, answers, judged_frame, judged_epoch):
        if not isinstance(answers, dict):
            return
        if judged_epoch != self.life_epoch or self.frame < judged_frame or self.frame - judged_frame > 42:
            self._log("jev_drop", frame=self.frame, judged_frame=judged_frame, reason="expired")
            return
        first = _answer_value(answers.get("danger_one"))
        second = _answer_value(answers.get("danger_two"))
        valid_danger = {"clear", "distant", "watch", "immediate", "unknown"}
        if first == second and first in valid_danger:
            threshold = 0.45 if first == "immediate" else 0.18
            if min(_probability(answers.get("danger_one"), first), _probability(answers.get("danger_two"), second)) >= threshold:
                if first in {"watch", "immediate", "unknown"}:
                    self.model_caution_until = self.frame + 45
                if first == "immediate":
                    self.model_defense_hint = True
        posture = _answer_value(answers.get("posture"))
        if posture in {"focus", "sweep_left", "sweep_right", "hold", "coast"} and _probability(answers.get("posture"), posture) >= 0.18:
            self.aim_mode = posture
            self.aim_until = self.frame + 54
        if _yes(answers.get("defense")) >= 0.82:
            self.model_defense_hint = True
        self.last_jev_frame = judged_frame
        self.last_jev_epoch = judged_epoch
        self._log("jev_label", frame=self.frame, judged_frame=judged_frame, danger=first if first == second else "disagree", posture=self.aim_mode)

    # ---------- action bridge ----------

    def legal(self, action):
        return action if action in self.names else self.noop

    def _fire_ready(self):
        if self.frame < self.next_fire or self.facts.get("missiles", 0) >= 2:
            return False
        gap = 6 if self.plan.get("fire") == "spaced" else 5
        self.next_fire = self.frame + gap
        return True

    @staticmethod
    def _turn_sign(current, desired):
        delta = (desired - current) % 16
        if delta == 0:
            return 0
        if delta < 8:
            return 1
        if delta > 8:
            return -1
        return 1

    def _compose(self, turn=0, thrust=False, fire=False):
        if thrust:
            if turn > 0:
                key = "UPLEFTFIRE" if fire else "UPLEFT"
            elif turn < 0:
                key = "UPRIGHTFIRE" if fire else "UPRIGHT"
            else:
                key = "UPFIRE" if fire else "UP"
        elif turn > 0:
            key = "LEFTFIRE" if fire else "LEFT"
        elif turn < 0:
            key = "RIGHTFIRE" if fire else "RIGHT"
        else:
            key = "FIRE" if fire else "NOOP"
        return self.legal(key)

    def _should_defend(self):
        f = self.facts
        if not f.get("present") or f.get("ttc") is None or f["ttc"] > 3:
            return False
        if self.plan.get("risk") == "strict" and f["ttc"] > 2 and not self.model_defense_hint:
            return False
        # A second defense is a reserve, not a rapid retry.  The long gap is
        # important: 0013's adjacent hyperspace retries were not survival.
        if self.defense_frames and self.frame - self.defense_frames[-1] < 120:
            return False
        if len(self.defense_frames) >= 2:
            return False
        # At overlap the one-frame thrust forecast is too late to be trusted;
        # spend a still-available defense rather than pretending a 5-pixel
        # clearance estimate is a collision proof.
        if f["ttc"] <= 0:
            return True
        best = f.get("best_escape", "coast")
        forecast = f.get("candidates", {}).get(best, {})
        coast = f.get("candidates", {}).get("coast", {})
        escape_better = forecast.get("contact") is None and forecast.get("clearance", -100) > coast.get("clearance", -100) + 2.0
        return not escape_better and (f["ttc"] <= 1 or self.model_defense_hint)

    def _escape_action(self):
        f = self.facts
        if not f.get("present") or self.frame < self.escape_again:
            return None
        ttc = f.get("ttc")
        if ttc is None or ttc > 30:
            return None
        best_key = f.get("best_escape", "coast")
        best = f.get("candidates", {}).get(best_key, {})
        coast = f.get("candidates", {}).get("coast", {})
        # No proactive movement on an uncertain forecast unless coast itself is
        # predicted to contact.  This is the idle-derived safety firewall.
        if (f.get("near_unknown") or f.get("speed") is None) and (ttc is None or ttc > 5):
            return None
        improvement = best.get("clearance", -100.0) - coast.get("clearance", -100.0)
        coast_contact = coast.get("contact")
        best_contact = best.get("contact")
        if best_key == "coast" or (best_contact is not None and ttc > 5):
            return None
        # A forecast that says "contact" for every candidate is not a reason to
        # alternate thrust directions every few frames.  Require either a real
        # clearance margin or a materially later contact; otherwise the defense
        # gate (for ttc <= 1) or the idle-like fallback owns the frame.
        later_contact = (
            coast_contact is not None and best_contact is not None and
            best_contact >= coast_contact + 3
        )
        if improvement < 1.5 and not later_contact:
            return None
        self.last_escape = best_key
        self.escape_pulses += 1
        self.escape_again = self.frame + (4 if ttc <= 8 else 9)
        self._log("escape_pulse", frame=self.frame, choice=best_key, ttc=ttc, improvement=round(improvement, 2), uncertain=f.get("uncertain"))
        return best_key

    def _aim_turn(self):
        f = self.facts
        if not f.get("present") or self.frame > self.aim_until:
            self.aim_mode = "target" if self.plan.get("aim") == "target" else "sweep_left"
        mode = self.aim_mode
        if mode in {"hold", "coast"}:
            return 0
        if mode == "sweep_left":
            return 1
        if mode == "sweep_right":
            return -1
        target = f.get("target")
        heading = f.get("heading")
        if target is not None and heading is not None:
            # Lead a little only when the object velocity is known.  The target
            # is otherwise a bearing, never an invented intercept calculation.
            dx = target["dx"] + (target["vx"] * 6.0 if target["known"] else 0.0)
            dy = target["dy"] + (target["vy"] * 6.0 if target["known"] else 0.0)
            return self._turn_sign(heading, _heading_for_vector(dx, dy))
        return 1 if ((self.frame // 56) % 2 == 0) else -1

    def choose(self):
        f = self.facts
        if not f.get("present"):
            self.last_action = self.noop
            return self.noop
        fire = self._fire_ready()
        if self._should_defend():
            self.defense_frames.append(self.frame)
            self.model_defense_hint = False
            self.coast_until = max(self.coast_until, self.frame + 34)
            self.escape_again = self.frame + 10
            self._log("defense", frame=self.frame, ttc=f.get("ttc"), count=len(self.defense_frames))
            self.last_action = self.legal("DOWN")
            return self.last_action

        guarded = self.frame < self.coast_until or f.get("near_unknown")
        if not guarded:
            escape = self._escape_action()
            if escape is not None:
                if escape == "thrust":
                    action = self._compose(0, thrust=True, fire=fire)
                elif escape == "left_thrust":
                    action = self._compose(1, thrust=True, fire=fire)
                else:
                    action = self._compose(-1, thrust=True, fire=fire)
                self.last_action = action
                return action

        # The coast guard is intentionally stronger after a discontinuity,
        # hyperspace, or a new ship.  FIRE remains allowed, because it is a
        # button pulse and does not commit the ship to a trajectory.
        if guarded:
            action = self._compose(0, thrust=False, fire=fire)
            self.last_action = action
            return action

        # Watch contacts with no safe one-pulse improvement are not permission
        # for open-loop thrust.  Keep the high-yield turret posture instead.
        turn = self._aim_turn()
        if f.get("danger") in {"watch", "immediate"} and f.get("uncertain"):
            turn = 0
        action = self._compose(turn, thrust=False, fire=fire)
        self.last_action = action
        if self.frame - self.last_log_frame >= 480 or fire or f.get("danger") == "immediate":
            self._log(
                "control", frame=self.frame, action=action, danger=f.get("danger"),
                ttc=f.get("ttc"), escape=f.get("best_escape"), score=self.score,
                lives=self.lives, coast=max(0, self.coast_until - self.frame),
            )
            self.last_log_frame = self.frame
        return action

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass


class VerifiedLaneRadar(Radar):
    """Reserve a reachable short pulse, execute by observation, then verify.

    0015 supplies early, horizontal lane intent. 0017 supplies the no-thrust
    default and quarantine. Neither an attractive endpoint nor a Jev vote can
    bypass this trajectory gate. Acceleration is a ranking heuristic, not known
    cartridge physics: both a weak and strong response must improve clearance.
    """

    def __init__(self, game, ctx):
        super().__init__(game, ctx)
        self.reservation = None
        self.review_after = 0
        self.watch_since = None
        self.watch_slot = None
        self.last_watch_frame = -100
        self.plan_version = 0
        self.maneuver_stats = {"started": 0, "pulsed": 0, "completed": 0,
                               "cancelled": 0, "improved": 0, "failed": 0}
        self.outcomes = []
        self.thrust_times = []
        self.local_routes = []
        self.last_route_review = -100
        self.last_route_block = -100
        self.response_scale = 1.0

    @staticmethod
    def _turn_distance(current, desired):
        d = (desired - current) % 16
        return min(d, 16-d)

    def _path(self, heading, delay, pulses, accel, horizon=48):
        f = self.facts
        ux, uy = _heading_vector(heading)
        first, clearance, culprit = None, 1000.0, None
        path = []
        for t in range(horizon + 1):
            active = min(float(pulses), max(0.0, t-delay))
            drift = max(0.0, t-delay-pulses)
            displacement = accel * (0.5*active*active + pulses*drift)
            path.append((f["x"] + (f.get("vx") or 0.0)*t + ux*displacement,
                         f["y"] + (f.get("vy") or 0.0)*t + uy*displacement))
        for rock in f.get("rocks", []):
            # Far uncertain rocks cannot veto a well-observed local crossing;
            # close uncertain objects still trigger the inherited coast guard.
            if not rock["known"] and math.hypot(rock["dx"], rock["dy"]) > 60:
                continue
            margin = 3.5 if rock["known"] else 5.0
            gx, gy = rock["w"]/2+3+margin, rock["h"]/2+5+margin
            for t, (px, py) in enumerate(path):
                rx, ry = rock["x"]+rock["vx"]*t, rock["y"]+rock["vy"]*t
                dx = abs((rx-px+80.0) % 160.0-80.0)
                dy = abs((ry-py+105.0) % 210.0-105.0)
                # Signed separation of two AABBs. min(dx-gx,dy-gy) is NOT
                # clearance: a rock sharing one axis can be far away on the other.
                sep = max(dx-gx, dy-gy)
                clearance = min(clearance, sep)
                if sep <= 0 and (first is None or t < first):
                    first, culprit = t, rock["slot"]
        return {"contact": first, "clearance": clearance, "slot": culprit}

    def _routes(self):
        f = self.facts
        h = f["heading"]
        coast = self._path(h, 0, 0, 0)
        # Current-nose response is the fast alternative; cardinal lanes offer
        # actual left/right relocation and use the verified cardinal convention.
        headings = list(dict.fromkeys([h, 4, 12, 0, 8]))
        candidates = []
        for heading in headings:
            delay = 4*self._turn_distance(h, heading)
            if delay > 28:
                continue
            pulses = 8
            low = self._path(heading, delay, pulses, 0.035*self.response_scale)
            high = self._path(heading, delay, pulses, 0.075*self.response_scale)
            contacts = [x["contact"] for x in (low, high) if x["contact"] is not None]
            contact = min(contacts) if contacts else None
            clearance = min(low["clearance"], high["clearance"])
            # Reject a route that would still collide, not merely one that has
            # a slightly more flattering endpoint. Models never relax this.
            clears = contact is None and clearance >= 1.0
            gain = clearance-coast["clearance"]
            ready = coast["contact"] is not None and delay+pulses+1 < coast["contact"]
            candidates.append({"heading": heading, "delay": delay, "pulses": pulses,
                "contact": contact, "clearance": clearance, "gain": gain,
                "admissible": clears and ready and gain >= 2.0,
                "coast_ttc": coast["contact"], "slot": coast["slot"]})
        return coast, sorted(candidates, key=lambda x: (not x["admissible"],
                                -x["clearance"], x["delay"]))

    def _finish(self, reason):
        r = self.reservation
        if r is None:
            return
        completed = reason == "verified"
        self.maneuver_stats["completed" if completed else "cancelled"] += 1
        now = self.facts.get("ttc")
        expected_remaining = max(0, r["before_ttc"]-(self.frame-r["start"]))
        improved = completed and (now is None or now >= expected_remaining+8)
        # A forecast that merely counts down normally is not an improvement.
        # Failed verification shrinks the assumed response for later routes;
        # it never buys a longer thrust burst or a weaker safety threshold.
        if completed and not improved:
            self.response_scale = max(0.20, self.response_scale*0.5)
        if completed:
            self.maneuver_stats["improved" if improved else "failed"] += 1
        outcome = {"reason": reason, "age": self.frame-r["start"],
                   "pulses": r["sent"], "before_ttc": r["before_ttc"],
                   "after_ttc": now, "coast_expected_ttc": expected_remaining,
                   "improved": improved, "response_scale": self.response_scale}
        self.outcomes.append(outcome)
        self.outcomes = self.outcomes[-12:]
        self._log("route_finish", frame=self.frame, **outcome)
        self.reservation = None
        self.review_after = self.frame + (16 if completed else 6)

    def update(self, obs):
        epoch = self.life_epoch
        super().update(obs)
        if self.life_epoch != epoch:
            self._finish("life_change")
            self.watch_since = None
        if self.reservation is not None and (not self.facts.get("present") or
                                            self.frame < self.coast_until):
            self._finish("discontinuity_or_quarantine")
        # Gains of an extra ship do not license unbounded propulsion.
        self.thrust_times = [x for x in self.thrust_times if self.frame-x < 180]

    def _lane_action(self):
        f = self.facts
        if not f.get("present") or self.frame < self.coast_until or f.get("near_unknown") or f.get("speed") is None:
            self._finish("uncertain")
            return None
        r = self.reservation
        if r is not None:
            if self.frame > r["expiry"]:
                self._finish("expired")
                return None
            if r["stage"] == "verify":
                if self.frame >= r["verify_at"]:
                    self._finish("verified")
                # Coasting is deliberate: observe the real impulse before
                # planning another one. Rotation/firing can continue safely.
                return None
            if f.get("ttc") is None:
                self._finish("hazard_cleared")
                return None
            turn = self._turn_sign(f["heading"], r["heading"])
            if turn:
                # Never combine thrust and a not-yet-observed heading change.
                delay = 4*self._turn_distance(f["heading"], r["heading"])
                if f["ttc"] <= delay+3:
                    self._finish("turn_no_longer_reachable")
                    return None
                return turn, False
            remaining = r["limit"]-r["sent"]
            forecast = self._path(r["heading"], 0, remaining, 0.035*self.response_scale)
            if forecast["contact"] is not None and forecast["contact"] <= 8:
                self._finish("new_close_contact")
                return None
            r["stage"] = "pulse"
            r["sent"] += 1
            self.thrust_times.append(self.frame)
            self.escape_pulses += 1
            self.maneuver_stats["pulsed"] += 1
            self._log("route_pulse", frame=self.frame, heading=r["heading"],
                      number=r["sent"], ttc=f.get("ttc"),
                      player_velocity=[f.get("vx"), f.get("vy")])
            if r["sent"] >= r["limit"]:
                r["stage"] = "verify"
                r["verify_at"] = self.frame + 8
            return 0, True

        if self.frame < self.review_after or self.frame-self.last_route_review < 3:
            return None
        self.last_route_review = self.frame
        ttc = f.get("ttc")
        if ttc is None or ttc > 40 or ttc < 7 or len(self.thrust_times) > 8:
            self.watch_since = None
            return None
        coast, routes = self._routes()
        self.local_routes = routes
        slot = coast["slot"]
        if self.watch_slot != slot or self.frame-self.last_watch_frame > 6:
            self.watch_since = self.frame
            self.watch_slot = slot
        self.last_watch_frame = self.frame
        if self.watch_since is None or self.frame-self.watch_since < 3:
            return None
        allowed = [x for x in routes if x["admissible"]]
        if self.plan.get("escape") == "rare":
            allowed = [x for x in allowed if x["gain"] >= 4]
        if not allowed:
            if self.frame-self.last_route_block >= 18:
                self._log("route_blocked", frame=self.frame, ttc=ttc,
                    candidates=[{k: round(v,2) if isinstance(v,float) else v
                                 for k,v in x.items()} for x in routes])
                self.last_route_block = self.frame
            return None
        best = allowed[0]
        self.reservation = {"heading": best["heading"], "start": self.frame,
            "expiry": self.frame+best["delay"]+28, "stage": "align", "sent": 0, "limit": best["pulses"],
            "before_ttc": ttc, "slot": slot}
        self.maneuver_stats["started"] += 1
        self._log("route_start", frame=self.frame, **best)
        return self._turn_sign(f["heading"], best["heading"]), False

    def _escape_action(self):
        # All thrust now belongs to the reachable, verified lane reservation.
        return None

    def choose(self):
        if self._should_defend():
            self._finish("defense")
            return super().choose()
        lane = self._lane_action()
        if lane is not None:
            turn, thrust = lane
            self.last_action = self._compose(turn, thrust, self._fire_ready())
            return self.last_action
        return super().choose()

    def state_for_jev(self):
        state = super().state_for_jev()
        state["controller"]["response_scale"] = self.response_scale
        state["controller"]["route_stage"] = self.reservation["stage"] if self.reservation else "none"
        state["controller"]["route_outcome"] = self.outcomes[-1] if self.outcomes else None
        state["plan"]["version"] = self.plan_version
        state["forecast"]["proactive_thrust"] = "only a fresh code-validated turn, at most eight-frame pulse, and verification; no model can authorize thrust"
        return state


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _start_jev(self, radar):
        if getattr(self.ctx, "system1", True) is None:
            return None
        state = radar.state_for_jev()
        questions = radar.questions()
        origin, epoch = radar.frame, radar.life_epoch

        def call():
            return self.ctx.decide(state, questions, timeout_s=2.2)

        try:
            return self.ctx.spawn(call), origin, epoch
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    def _start_astra(self, radar, reason):
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if _integer(remaining.get("system2_calls"), 0) <= 0:
            return None
        f = radar.facts
        summary = {
            "reason": reason,
            "frame": radar.frame,
            "score": int(max(0, radar.score)),
            "lives": radar.lives,
            "danger": f.get("danger", "unknown"),
            "ttc": f.get("ttc"),
            "uncertain": bool(f.get("uncertain")),
            "escape_pulses": radar.escape_pulses,
            "maneuvers": dict(radar.maneuver_stats),
            "latest_outcomes": list(radar.outcomes[-6:]),
            "defenses": len(radar.defense_frames),
            "plan": dict(radar.plan),
        }
        prompt = (
            "You are the strategic reviewer for a real-time Atari 2600 Asteroids "
            "mode-0/difficulty-0 controller. The code owns every native frame, "
            "RAM-derived geometry, swept collision ranking, FIRE press/release, "
            "hyperspace gates, and all actions. Review the aggregate evidence below. "
            "Return one short JSON object only; never return actions or a sequence. "
            "Allowed values: risk is 'strict' or 'balanced'; fire is 'normal' or "
            "'spaced'; aim is 'target' or 'sweep'; escape is 'reactive' or 'rare'. Reactive admits a verified short lane pulse; rare requires stronger clearance. "
            "Unknown enemy coverage is not absence, and no-thrust idle fallback is "
            "preferred whenever geometry is uncertain.\n" + json.dumps(summary, separators=(",", ":"))
        )

        def call():
            return self.ctx.think(prompt, timeout_s=140.0)

        try:
            return self.ctx.spawn(call)
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    @staticmethod
    def _parse_plan(value):
        text = getattr(value, "text", value)
        data = _extract_json_object(text)
        if not isinstance(data, dict):
            return {}
        allowed = {
            "risk": {"strict", "balanced"},
            "fire": {"normal", "spaced"},
            "aim": {"target", "sweep"},
            "escape": {"reactive", "rare"},
        }
        result = {}
        for key, choices in allowed.items():
            item = data.get(key)
            if isinstance(item, str) and item in choices:
                result[key] = item
        return result

    def run_episode(self, game) -> None:
        if getattr(game, "tick_s", None) is None:
            return
        radar = VerifiedLaneRadar(game, self.ctx)
        radar.update(_observation(game))
        ctx = self.ctx
        try:
            ctx.act(radar.choose())
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return
        except Exception:
            try:
                ctx.act(radar.noop)
            except Exception:
                return

        jev_future = None
        jev_frame = -100000
        jev_epoch = -1
        next_jev = radar.frame
        astra_future = None
        astra_started = False
        astra_done = False
        astra_due = radar.frame + 1200
        model_enabled = getattr(ctx, "system1", True) is not None

        while not ctx.done and not getattr(game, "done", False):
            if model_enabled and jev_future is None and radar.frame >= next_jev:
                try:
                    result = self._start_jev(radar)
                    if result is not None:
                        jev_future, jev_frame, jev_epoch = result
                        next_jev = radar.frame + 30
                    else:
                        next_jev = radar.frame + 42
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    model_enabled = False
                except Exception:
                    next_jev = radar.frame + 42

            if not astra_started and not astra_done and radar.frame >= astra_due:
                astra_started = True
                try:
                    astra_future = self._start_astra(radar, "aggregate review of trajectory verification and life losses")
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    astra_done = True
                except Exception:
                    astra_done = True

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            if ctx.done or getattr(game, "done", False):
                break

            radar.update(_observation(game))
            if jev_future is not None and jev_future.done():
                future = jev_future
                jev_future = None
                try:
                    radar.accept_jev(future.result(), jev_frame, jev_epoch)
                except (BudgetExceeded, EpisodeOver):
                    model_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    # A malformed stand-in/provider answer is just a missed label.
                    pass

            if astra_future is not None and astra_future.done():
                future = astra_future
                astra_future = None
                try:
                    plan = self._parse_plan(future.result())
                    if plan:
                        radar.plan_version += 1
                        radar.plan.update(plan)
                        if plan.get("aim") == "sweep":
                            radar.aim_mode = "sweep_left" if ((radar.frame // 56) % 2 == 0) else "sweep_right"
                        radar._log("astra_plan", frame=radar.frame, plan=dict(radar.plan))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    return
                except Exception:
                    pass
                astra_done = True

            try:
                ctx.act(radar.choose())
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                try:
                    ctx.act(radar.noop)
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
