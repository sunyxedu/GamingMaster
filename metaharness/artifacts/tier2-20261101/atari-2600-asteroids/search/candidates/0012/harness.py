"""Atari Asteroids tier-2 controller: aligned threat shadows.

The clock and all native-frame actions stay in this module.  Models only provide
short-lived semantic preferences; collision geometry, firing cadence, defense
verification, and legal action conversion are deterministic.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


WIDTH = 160.0
HEIGHT = 210.0
ACTIONS = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
    "UPLEFTFIRE",
)


def num(value, default=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def maybe_num(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def obj(value):
    return value if isinstance(value, dict) else {}


def seq(value):
    return value if isinstance(value, list) else []


def pair(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = maybe_num(value[0])
        y = maybe_num(value[1])
        if x is not None and y is not None:
            return x, y
    return None, None


def wrap_delta(value, period):
    x = num(value)
    while x > period / 2.0:
        x -= period
    while x < -period / 2.0:
        x += period
    return x


def bucket(value, cuts, labels):
    if value is None:
        return labels[0]
    for cut, label in zip(cuts, labels[1:]):
        if value < cut:
            return label
    return labels[-1]


def read_observation(game):
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


def answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def answer_probability(answer, key):
    if isinstance(answer, dict):
        probabilities = answer.get("probabilities", {})
    else:
        probabilities = getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    return max(0.0, min(1.0, num(probabilities.get(key), 0.0)))


def yes_probability(answer):
    value = answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        if value.lower() in {"true", "yes"}:
            return 1.0
        if value.lower() in {"false", "no"}:
            return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, num(value, 0.0)))
    return answer_probability(answer, "true")


def json_object(text):
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def heading_for(dx, dy):
    """Public code: 0 up, 4 left, 8 down, 12 right."""
    if abs(dx) + abs(dy) < 1e-8:
        return 0
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def heading_vector(code):
    angle = (int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


class ShadowController:
    """Frame-local Asteroids policy with an explicit align-before-thrust phase."""

    LANES = (45.0, 65.0, 85.0, 105.0, 125.0)
    # A pulse is deliberately evaluated in every observed frame.  These values
    # bound scheduling; they are not claims about exact cartridge acceleration.
    NORMAL_THRUST_GAP = 28
    URGENT_THRUST_GAP = 5
    SHADOW_HORIZON = 38
    FIRE_GAP = 5

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
        self.epoch = 0
        self.obs = {}
        self.facts = {}

        self.plan = {
            "aim": "target",
            "risk": "cautious",
            "fire": "normal",
            "defense": "retry",
        }
        self.posture = "focus"
        self.posture_until = -1
        self.selected_target = None
        self.model_escape_hint = False
        self.model_defense_hint = False
        self.judged_frame = -100000
        self.judged_epoch = -1

        # Lane control is the calm-field policy inherited in spirit from 0011.
        self.lane_goal = None
        self.lane_until = -1
        self.next_lane_review = 0
        self.next_normal_thrust = 0
        self.normal_thrust_until = -1

        # Threat-shadow state is new: desired heading, phase, and one-frame pulse.
        self.shadow = None
        self.shadow_epoch = -1
        self.shadow_replan_at = 0
        self.shadow_pulses = 0
        self.shadow_last_pulse = -100000

        # Defense is a measured experiment rather than a once-per-life latch.
        self.defense_count = 0
        self.last_defense = -100000
        self.defense_record = None
        self.defense_verify = "none"
        self.verify_until = -1

        self.next_fire = 0
        self.last_action = self.noop
        self.last_log = -100000

    # ---------------------------- observations ----------------------------

    @staticmethod
    def _player(obs):
        return obj(obs.get("player"))

    def _make_rocks(self, obs, player):
        if not player.get("present"):
            return []
        pw = num(player.get("width"), 6.0) or 6.0
        ph = num(player.get("height"), 10.0) or 10.0
        px0 = maybe_num(player.get("x"))
        py0 = maybe_num(player.get("y"))
        if px0 is None or py0 is None:
            return []
        px = px0 + pw / 2.0
        py = py0 + ph / 2.0
        pvx, pvy = pair(player.get("velocity_px_per_frame"))
        player_velocity_known = pvx is not None and pvy is not None
        if pvx is None:
            pvx = 0.0
        if pvy is None:
            pvy = 0.0
        rocks = []
        for index, raw0 in enumerate(seq(obs.get("asteroids"))):
            raw = obj(raw0)
            ax0 = maybe_num(raw.get("x"))
            ay0 = maybe_num(raw.get("y"))
            if ax0 is None or ay0 is None:
                continue
            aw = num(raw.get("width"), 8.0) or 8.0
            ah = num(raw.get("height"), 15.0) or 15.0
            avx, avy = pair(raw.get("velocity_px_per_frame"))
            known = player_velocity_known and avx is not None and avy is not None
            if avx is None:
                avx = 0.0
            if avy is None:
                avy = 0.0
            size = str(raw.get("size") or "unknown").lower()
            # Unknown tracks are conservative in clearance; they are never
            # silently discarded just because the decoder lacks history.
            uncertainty = 4.0 if not known else 0.0
            rocks.append({
                "slot": integer(raw.get("source_slot"), index),
                "size": size,
                "x": ax0 + aw / 2.0,
                "y": ay0 + ah / 2.0,
                "dx": wrap_delta(ax0 + aw / 2.0 - px, WIDTH),
                "dy": wrap_delta(ay0 + ah / 2.0 - py, HEIGHT),
                "vx": avx - pvx,
                "vy": avy - pvy,
                "known": bool(known),
                "width": aw,
                "height": ah,
                "uncertainty": uncertainty,
                "value": {"small": 3, "medium": 2, "large": 1}.get(size, 1),
            })
        return rocks

    @staticmethod
    def _contact_time(rock, x, y, horizon=72, pw=6.0, ph=10.0):
        xgap = rock["width"] / 2.0 + pw / 2.0 + 3.0 + rock.get("uncertainty", 0.0)
        ygap = rock["height"] / 2.0 + ph / 2.0 + 3.0 + rock.get("uncertainty", 0.0)
        first = None
        for t in range(horizon + 1):
            dx = wrap_delta(rock["dx"] + rock["vx"] * t - x, WIDTH)
            dy = wrap_delta(rock["dy"] + rock["vy"] * t - y, HEIGHT)
            if abs(dx) <= xgap and abs(dy) <= ygap:
                first = t
                break
        return first

    @staticmethod
    def _clearance(rock, x, y, t):
        dx = wrap_delta(rock["dx"] + rock["vx"] * t - x, WIDTH)
        dy = wrap_delta(rock["dy"] + rock["vy"] * t - y, HEIGHT)
        xgap = rock["width"] / 2.0 + 3.0 + rock.get("uncertainty", 0.0)
        ygap = rock["height"] / 2.0 + 3.0 + rock.get("uncertainty", 0.0)
        # This is an axis-aligned conservative distance, not a collision mask.
        return math.hypot(max(0.0, abs(dx) - xgap), max(0.0, abs(dy) - ygap))

    def _risk_at(self, x, y, horizon=72):
        first = None
        minimum = float("inf")
        nearest = None
        for rock in self.facts.get("rocks", []):
            t = self._contact_time(rock, x, y, horizon=horizon)
            if t is not None and (first is None or t < first):
                first = t
                nearest = rock
            for sample in range(0, min(horizon, 36) + 1, 2):
                minimum = min(minimum, self._clearance(rock, x, y, sample))
        if not self.facts.get("rocks"):
            minimum = 80.0
        return first, minimum, nearest

    def _derive(self, obs):
        player = self._player(obs)
        present = bool(player.get("present"))
        px0 = maybe_num(player.get("x"))
        py0 = maybe_num(player.get("y"))
        pw = num(player.get("width"), 6.0) or 6.0
        ph = num(player.get("height"), 10.0) or 10.0
        if px0 is None or py0 is None:
            present = False
        px = px0 + pw / 2.0 if present else None
        py = py0 + ph / 2.0 if present else None
        pvx, pvy = pair(player.get("velocity_px_per_frame"))
        velocity_known = present and pvx is not None and pvy is not None
        speed = math.hypot(pvx, pvy) if velocity_known else None
        rocks = self._make_rocks(obs, player) if present else []
        threat = None
        threat_time = None
        if present:
            for rock in rocks:
                # Rock coordinates are relative to the current player center;
                # zero is the current ship position for this forecast.
                t = self._contact_time(rock, 0.0, 0.0, horizon=90, pw=pw, ph=ph)
                if t is not None and (threat_time is None or (t, -rock["value"]) < (threat_time, -threat["value"])):
                    threat_time = t
                    threat = rock
        current_clear = float("inf")
        if present:
            # Keep this as *present-frame* clearance.  A future crossing is
            # represented by threat_time; using a 36-frame minimum here would
            # label every eventual vertical crossing as an immediate collision.
            for rock in rocks:
                current_clear = min(current_clear, self._clearance(rock, 0.0, 0.0, 0))
        if not rocks:
            current_clear = 80.0
        if threat_time is None:
            danger = "clear"
        elif threat_time <= 2 or current_clear <= 0.0:
            danger = "critical"
        elif threat_time <= 8 or current_clear <= 3.0:
            danger = "immediate"
        elif threat_time <= 30 or current_clear <= 12.0:
            danger = "watch"
        else:
            danger = "distant"
        missiles = sum(1 for item in seq(obs.get("player_missiles")) if isinstance(item, dict))
        return {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "present": present,
            "x": px,
            "y": py,
            "pw": pw,
            "ph": ph,
            "vx": pvx,
            "vy": pvy,
            "velocity_known": velocity_known,
            "speed": speed,
            "heading": integer(player.get("orientation_code"), 0) % 16 if present else None,
            "rocks": rocks,
            "rock_count": len(rocks),
            "threat": threat,
            "threat_time": threat_time,
            "clearance": current_clear,
            "danger": danger,
            "missiles": missiles,
            "discontinuity": bool(player.get("motion_discontinuity")) if present else True,
        }

    def update(self, observation):
        obs = observation if isinstance(observation, dict) else {}
        old_lives = self.lives
        self.obs = obs
        self.frame = integer(obs.get("frame"), self.frame + 1)
        self.score = num(obs.get("score"), self.score)
        new_lives = integer(obs.get("lives"), self.lives)
        changed = self.old_lives is not None and new_lives != self.old_lives
        self.lives = new_lives
        if self.old_lives is None:
            self.old_lives = new_lives
        if changed:
            self.epoch += 1
            self.lane_goal = None
            self.lane_until = -1
            self.normal_thrust_until = -1
            self.next_normal_thrust = self.frame + 14
            self.shadow = None
            self.shadow_epoch = -1
            self.shadow_replan_at = self.frame
            self.shadow_pulses = 0
            self.defense_count = 0
            self.last_defense = -100000
            self.defense_record = None
            self.defense_verify = "none"
            self.verify_until = -1
            self.model_escape_hint = False
            self.model_defense_hint = False
            self.judged_epoch = -1
            self.posture = "focus"
            self.posture_until = self.frame + 36
            self.next_fire = max(self.next_fire, self.frame + 8)
            self.log("life_change", frame=self.frame, old_lives=old_lives, lives=new_lives, score=self.score)
        self.old_lives = new_lives
        self.facts = self._derive(obs)
        self._verify_defense_landing()
        self._refresh_lane(force=changed)
        if self.facts.get("discontinuity") and self.shadow is not None:
            # A teleport invalidates a vector computed before it.  It does not
            # automatically mean the new landing is safe.
            self.shadow = None
            self.shadow_epoch = self.epoch
            if self.defense_record is not None and self.frame > self.defense_record["frame"]:
                if self.facts.get("danger") in {"critical", "immediate"}:
                    self.defense_verify = "still_critical"
                else:
                    self.defense_verify = "improved"

    def _verify_defense_landing(self):
        record = self.defense_record
        if not isinstance(record, dict) or self.frame <= record.get("frame", -1):
            return
        p = self.facts
        old_x = record.get("x")
        old_y = record.get("y")
        moved = False
        if p.get("present") and old_x is not None and old_y is not None:
            moved = math.hypot(wrap_delta(p.get("x", old_x) - old_x, WIDTH), wrap_delta(p.get("y", old_y) - old_y, HEIGHT)) > 12.0
        landing = bool(p.get("discontinuity")) or moved or not p.get("present")
        if self.frame < self.verify_until:
            if landing:
                if p.get("danger") in {"critical", "immediate"}:
                    self.defense_verify = "still_critical"
                else:
                    self.defense_verify = "improved"
            return
        if p.get("danger") in {"critical", "immediate"}:
            self.defense_verify = "still_critical"
        elif self.defense_verify == "none":
            self.defense_verify = "improved"

    # -------------------------- numerical shadow --------------------------

    def _risk_with_heading(self, heading, pulse=1, horizon=None):
        """Forecast relative clearance after a finite thrust pulse."""
        if horizon is None:
            horizon = self.SHADOW_HORIZON
        hx, hy = heading_vector(heading)
        # Rock velocities in the track records are already relative to the
        # current ship velocity.  Only the prospective thrust displacement is
        # subtracted here; subtracting measured velocity again would double it.
        minimum = float("inf")
        first = None
        total = 0.0
        for rock in self.facts.get("rocks", []):
            rock_min = float("inf")
            hit = None
            for t in range(horizon + 1):
                used = min(t, pulse)
                # Small acceleration is only a ranking signal.  Current measured
                # velocity remains the dominant term.
                disp_x = pvx * t + 0.035 * hx * used * (used + 1.0) / 2.0
                disp_y = pvy * t + 0.035 * hy * used * (used + 1.0) / 2.0
                dx = wrap_delta(rock["dx"] + rock["vx"] * t - disp_x, WIDTH)
                dy = wrap_delta(rock["dy"] + rock["vy"] * t - disp_y, HEIGHT)
                xgap = rock["width"] / 2.0 + self.facts.get("pw", 6.0) / 2.0 + 3.0 + rock.get("uncertainty", 0.0)
                ygap = rock["height"] / 2.0 + self.facts.get("ph", 10.0) / 2.0 + 3.0 + rock.get("uncertainty", 0.0)
                clear = math.hypot(max(0.0, abs(dx) - xgap), max(0.0, abs(dy) - ygap))
                rock_min = min(rock_min, clear)
                if hit is None and abs(dx) <= xgap and abs(dy) <= ygap:
                    hit = t
            minimum = min(minimum, rock_min)
            total += rock_min
            if hit is not None and (first is None or hit < first):
                first = hit
        if not self.facts.get("rocks"):
            minimum = 80.0
            total = 80.0
        # Penalize an edge landing, but not enough to override a real collision.
        pvx = num(self.facts.get("vx"), 0.0) if self.facts.get("velocity_known") else 0.0
        pvy = num(self.facts.get("vy"), 0.0) if self.facts.get("velocity_known") else 0.0
        x = num(self.facts.get("x"), 80.0) + pvx * 20.0 + 0.035 * hx * pulse * (pulse + 1.0) / 2.0
        y = num(self.facts.get("y"), 105.0) + pvy * 20.0 + 0.035 * hy * pulse * (pulse + 1.0) / 2.0
        edge = min(x % WIDTH, WIDTH - (x % WIDTH), y % HEIGHT, HEIGHT - (y % HEIGHT))
        edge_penalty = max(0.0, 12.0 - edge) * 0.75
        return {"heading": int(heading) % 16, "clearance": minimum, "contact": first, "total": total, "score": minimum + 0.06 * total - edge_penalty}

    def _current_shadow_score(self):
        current = self._risk_with_heading(self.facts.get("heading", 0), pulse=0)
        return current

    def _choose_shadow(self):
        if not self.facts.get("present"):
            return None
        current = self._current_shadow_score()
        candidates = [self._risk_with_heading(h, pulse=1) for h in range(16)]
        candidates.sort(key=lambda c: (c["score"], c["clearance"], -abs(self._turn_distance(self.facts.get("heading", 0), c["heading"]))), reverse=True)
        best = candidates[0] if candidates else None
        if best is None:
            return None
        # In a clear field do not create an escape merely for a numerical tie.
        if self.facts.get("danger") not in {"critical", "immediate"} and best["score"] < current["score"] + 3.0:
            return None
        return best

    @staticmethod
    def _turn_distance(current, desired):
        delta = (int(desired) - int(current)) % 16
        return min(delta, 16 - delta)

    @staticmethod
    def _turn_sign(current, desired):
        delta = (int(desired) - int(current)) % 16
        if delta == 0:
            return 0
        if delta < 8:
            return 1  # LEFT increments orientation code.
        if delta > 8:
            return -1  # RIGHT decrements orientation code.
        return 1

    def _ensure_shadow(self, force=False):
        f = self.facts
        if not f.get("present") or f.get("danger") not in {"critical", "immediate"}:
            if f.get("danger") not in {"critical", "immediate"}:
                self.shadow = None
            return
        if not force and self.shadow is not None and self.frame < self.shadow_replan_at and self.shadow_epoch == self.epoch:
            return
        choice = self._choose_shadow()
        if choice is None:
            self.shadow = None
            return
        self.shadow = {
            "heading": choice["heading"],
            "phase": "align",
            "armed": False,
            "created": self.frame,
            "contact": choice.get("contact"),
            "clearance": choice.get("clearance"),
            "score": choice.get("score"),
        }
        self.shadow_epoch = self.epoch
        self.shadow_replan_at = self.frame + 5
        self.log("shadow_plan", frame=self.frame, heading=choice["heading"], clearance=round(choice["clearance"], 2), contact=choice.get("contact"), score=round(choice["score"], 2), current_clear=round(current_clearance(self.facts), 2))

    def _shadow_action(self, fire=False):
        shadow = self.shadow
        if not isinstance(shadow, dict) or not self.facts.get("present"):
            return None
        if self.facts.get("danger") not in {"critical", "immediate"}:
            self.shadow = None
            return None
        current = self.facts.get("heading")
        desired = integer(shadow.get("heading"), current if current is not None else 0) % 16
        if current is None:
            return None
        delta = self._turn_distance(current, desired)
        # A fresh observation can invalidate a plan whose vector no longer
        # improves the measured scene.  Replan before committing a pulse.
        if self.frame >= self.shadow_replan_at and shadow.get("phase") == "align":
            test = self._risk_with_heading(desired, pulse=1)
            if test["clearance"] + 1.0 < self._current_shadow_score()["clearance"]:
                self.shadow = None
                self._ensure_shadow(force=True)
                shadow = self.shadow
                if shadow is None:
                    return None
                desired = integer(shadow.get("heading"), current) % 16
                delta = self._turn_distance(current, desired)
            self.shadow_replan_at = self.frame + 5
        # Never thrust before alignment.  This is the structural change from
        # 0011's urgent combined turn/thrust bridge.
        if delta > 1:
            sign = self._turn_sign(current, desired)
            shadow["phase"] = "align"
            return self._compose(sign, thrust=False, fire=fire)
        if self.frame - self.shadow_last_pulse < self.URGENT_THRUST_GAP:
            shadow["phase"] = "settle"
            return self._compose(0, thrust=False, fire=fire)
        shadow["phase"] = "pulse"
        shadow["armed"] = True
        self.shadow_last_pulse = self.frame
        self.shadow_pulses += 1
        self.log("shadow_pulse", frame=self.frame, heading=desired, alignment=delta, danger=self.facts.get("danger"), clearance=round(num(shadow.get("clearance"), 0.0), 2))
        return self._compose(0, thrust=True, fire=fire)

    # ----------------------------- lane policy -----------------------------

    def _lane_cost(self, x):
        total = 0.0
        first = None
        for rock in self.facts.get("rocks", []):
            # A lane is an absolute x coordinate, so convert it to a
            # displacement from the current ship before using relative tracks.
            candidate_dx = wrap_delta(x - num(self.facts.get("x"), 80.0), WIDTH)
            t = self._contact_time(rock, candidate_dx, 0.0, horizon=72,
                                   pw=self.facts.get("pw", 6.0),
                                   ph=self.facts.get("ph", 10.0))
            if t is None:
                continue
            if first is None or t < first:
                first = t
            total += (2.0 + rock["value"]) * (1.0 + max(0.0, (45.0 - t) / 15.0)) * (1.3 if not rock["known"] else 1.0)
        return total, first

    def _refresh_lane(self, force=False):
        f = self.facts
        if not f.get("present"):
            self.lane_goal = None
            return
        if not force and self.frame < self.next_lane_review and self.lane_goal is not None and self.frame <= self.lane_until:
            return
        current_x = num(f.get("x"), 80.0)
        current_cost, current_first = self._lane_cost(current_x)
        values = []
        for x in self.LANES:
            cost, first = self._lane_cost(x)
            distance = abs(wrap_delta(x - current_x, WIDTH))
            values.append({"x": x, "cost": cost, "first": first, "rank": cost + distance * 0.018})
        values.sort(key=lambda q: (q["rank"], abs(q["x"] - current_x)))
        best = values[0]
        trigger = 5.0 if self.plan.get("risk") == "cautious" else 7.0
        if self.plan.get("patrol") == "minimal":
            trigger += 2.0
        if current_cost >= trigger or f.get("danger") in {"watch", "immediate", "critical"}:
            if best["rank"] + 1.0 < current_cost or f.get("danger") in {"immediate", "critical"}:
                goal = best["x"]
            else:
                goal = current_x
        else:
            goal = current_x if self.lane_goal is None or self.lane_until < self.frame else self.lane_goal
        self.lane_goal = goal
        self.lane_until = self.frame + (85 if f.get("danger") in {"watch", "immediate", "critical"} else 70)
        self.next_lane_review = self.frame + (10 if f.get("danger") in {"watch", "immediate", "critical"} else 45)
        self.log("lane_plan", frame=self.frame, goal=round(goal, 1), current_cost=round(current_cost, 2), best_cost=round(best["cost"], 2), first=best["first"])

    def _normal_lane_action(self, fire=False):
        f = self.facts
        if not f.get("present") or self.lane_goal is None:
            return None
        x = f.get("x")
        heading = f.get("heading")
        if x is None or heading is None:
            return None
        delta = wrap_delta(self.lane_goal - x, WIDTH)
        vx = f.get("vx") if f.get("velocity_known") else None
        braking = abs(delta) < 14.0 and vx is not None and abs(vx) > 0.18
        if abs(delta) <= 8.0 and not braking:
            return None
        direction = 1 if (vx < -0.18 if braking else delta < 0.0) else -1
        desired = 4 if direction > 0 else 12
        turn = self._turn_sign(heading, desired)
        if turn:
            return self._compose(turn, thrust=False, fire=fire)
        if self.frame < self.normal_thrust_until:
            return self._compose(0, thrust=True, fire=fire)
        if self.frame < self.next_normal_thrust:
            return None
        self.normal_thrust_until = self.frame + 2
        self.next_normal_thrust = self.frame + self.NORMAL_THRUST_GAP
        self.log("lane_pulse", frame=self.frame, goal=round(self.lane_goal, 1), delta=round(delta, 1))
        return self._compose(0, thrust=True, fire=fire)

    # ----------------------------- aiming/fire -----------------------------

    def _selected_rock(self):
        rocks = self.facts.get("rocks", [])
        if self.selected_target is not None and self.frame - self.judged_frame <= 48:
            slot, size = self.selected_target
            for rock in rocks:
                if rock["slot"] == slot and rock["size"] == size:
                    return rock
        if not rocks:
            return None
        def key(rock):
            t = self._contact_time(rock, 0.0, 0.0, horizon=42,
                                   pw=self.facts.get("pw", 6.0),
                                   ph=self.facts.get("ph", 10.0))
            threat = 5.0 if t is not None else 0.0
            distance = math.hypot(rock["dx"], rock["dy"])
            return -(threat + 2.5 * rock["value"] + (1.0 if rock["known"] else 0.0) - distance / 120.0)
        return min(rocks, key=key)

    def _fire_ready(self):
        if self.frame < self.next_fire or self.facts.get("missiles", 0) >= 2:
            return False
        self.next_fire = self.frame + (6 if self.plan.get("fire") == "spaced" else self.FIRE_GAP)
        return True

    def _aim_turn(self):
        heading = self.facts.get("heading")
        if heading is None:
            return 0
        if self.frame > self.posture_until:
            self.posture = "focus" if self.plan.get("aim") == "target" else ("sweep_left" if (self.frame // 52) % 2 else "sweep_right")
            self.posture_until = self.frame + 54
        if self.posture == "sweep_left":
            return 1
        if self.posture == "sweep_right":
            return -1
        if self.posture == "hold":
            return 0
        rock = self._selected_rock()
        if rock is None:
            return 1 if (self.frame // 52) % 2 == 0 else -1
        lead = 8.0 if rock["known"] else 3.0
        desired = heading_for(rock["dx"] + rock["vx"] * lead, rock["dy"] + rock["vy"] * lead)
        delta = (desired - heading) % 16
        if delta == 0:
            return 0
        return 1 if delta < 8 else -1

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
        return name if name in self.names else self.noop

    # ---------------------------- defense/action ---------------------------

    def _defense_allowed(self):
        f = self.facts
        if not f.get("present") or self.plan.get("defense") == "never":
            return False
        if self.defense_count >= 2 or self.frame - self.last_defense < 14:
            return False
        critical = f.get("danger") == "critical" or (f.get("threat_time") is not None and f.get("threat_time") <= 2)
        if not critical:
            return False
        if self.defense_count == 0:
            return True
        # A second pulse is an experiment only when the first landing did not
        # improve the measured contact.  A successful teleport is not spammed.
        return self.defense_verify == "still_critical" and self.frame >= self.verify_until

    def _use_defense(self):
        p = self.facts
        self.defense_count += 1
        self.last_defense = self.frame
        self.defense_record = {
            "frame": self.frame,
            "x": p.get("x"),
            "y": p.get("y"),
            "clearance": p.get("clearance"),
            "threat_time": p.get("threat_time"),
        }
        self.defense_verify = "pending"
        self.verify_until = self.frame + 14
        self.shadow = None
        self.model_defense_hint = False
        self.log("defense", frame=self.frame, count=self.defense_count, threat_time=p.get("threat_time"), clearance=round(num(p.get("clearance"), 0.0), 2))
        return self.noop if "DOWN" not in self.names else "DOWN"

    def choose(self):
        f = self.facts
        if not f.get("present"):
            self.shadow = None
            self.last_action = self.noop
            return self.noop
        fire = self._fire_ready()

        if self._defense_allowed():
            self.last_action = self._use_defense()
            return self.last_action

        if f.get("danger") in {"critical", "immediate"}:
            self._ensure_shadow()
            action = self._shadow_action(fire=fire)
            if action is not None:
                self.last_action = action
                return action

        # Once the urgent window has passed, a stale shadow cannot keep thrusting.
        self.shadow = None
        lane_action = self._normal_lane_action(fire=fire)
        if lane_action is not None and f.get("danger") not in {"critical", "immediate"}:
            self.last_action = lane_action
            return lane_action
        turn = self._aim_turn()
        action = self._compose(turn, thrust=False, fire=fire)
        self.last_action = action
        if self.frame - self.last_log >= 600 or fire or action in {"UP", "UPLEFT", "UPRIGHT", "DOWN"}:
            self.log("control", frame=self.frame, action=action, danger=f.get("danger"), lane=self.lane_goal, shadow=self.shadow.get("phase") if isinstance(self.shadow, dict) else "none", score=self.score, lives=self.lives)
            self.last_log = self.frame
        return action

    # ----------------------------- model state -----------------------------

    @staticmethod
    def lane_id(x):
        return "L" + str(int(round(num(x))))

    @staticmethod
    def bearing(dx, dy):
        return ("right" if dx > 10 else "left" if dx < -10 else "center") + "_" + ("below" if dy > 10 else "above" if dy < -10 else "level")

    def _shadow_state(self):
        shadow = self.shadow if isinstance(self.shadow, dict) else {}
        h = shadow.get("heading")
        heading_name = "none"
        if h is not None:
            heading_name = self.bearing(*heading_vector(integer(h)))
        phase = shadow.get("phase", "none")
        return {
            "phase": phase,
            "kind": "vector" if phase != "none" else "none",
            "heading": heading_name,
            "clearance": bucket(shadow.get("clearance"), (2, 10, 25), ("unknown", "critical", "tight", "open")),
            "contact": bucket(shadow.get("contact"), (3, 9, 24), ("none", "immediate", "soon", "later")),
        }

    def state_for_jev(self):
        f = self.facts
        target = self._selected_rock()
        target_state = None
        if target is not None:
            target_state = {
                "id": "T" + str(target["slot"]) + "_" + target["size"],
                "size": target["size"],
                "bearing": self.bearing(target["dx"], target["dy"]),
                "motion": "known" if target["known"] else "uncertain",
                "distance": bucket(math.hypot(target["dx"], target["dy"]), (25, 70, 120), ("unknown", "near", "middle", "far")),
            }
        candidates = ["target_fire", "sweep_left", "sweep_right", "hold"]
        if self.lane_goal is not None and f.get("danger") in {"watch", "distant"}:
            candidates.append("lane_escape")
        if self.shadow is not None and f.get("danger") in {"critical", "immediate"}:
            candidates.append("align_escape")
        return {
            "rules": {
                "mode": 0,
                "fire": "button_level_release_between_presses",
                "noop": "inertia_continues",
                "down": "mode_0_hyperspace_not_brake",
                "heading": "left_increments_right_decrements",
                "enemy_coverage": "unknown_not_absent",
            },
            "snapshot": {
                "frame": int(f.get("frame", 0)),
                "lives": int(f.get("lives", 0)),
                "score": int(max(0.0, self.score)),
                "player_present": bool(f.get("present")),
                "position": bucket(f.get("x"), (55, 105), ("unknown", "left", "middle", "right")),
                "motion": "unknown" if f.get("speed") is None else ("still" if f.get("speed") < 0.2 else "moving"),
                "danger": f.get("danger", "unknown"),
                "impact": bucket(f.get("threat_time"), (3, 9, 31), ("none", "immediate", "soon", "later")),
                "rocks": min(15, int(f.get("rock_count", 0))),
                "missiles": min(2, int(f.get("missiles", 0))),
            },
            "threat": target_state,
            "shadow": self._shadow_state(),
            "candidates": candidates,
            "defense": {
                "pulses": int(self.defense_count),
                "verify": self.defense_verify,
                "retry_available": bool(self.defense_count < 2),
            },
            "controller": {
                "posture": self.posture,
                "lane_goal": self.lane_id(self.lane_goal) if self.lane_goal is not None else "none",
                "frame_local": True,
            },
            "plan": dict(self.plan),
        }

    def questions(self):
        f = self.facts
        labels = {
            "target_fire": "Follow the code-selected target with short FIRE press/release pulses; do not thrust from this answer.",
            "sweep_left": "Make a bounded counterclockwise aiming sweep and release between FIRE presses; no thrust.",
            "sweep_right": "Make a bounded clockwise aiming sweep and release between FIRE presses; no thrust.",
            "hold": "Keep the current nose and use only the local firing cadence; no thrust.",
            "lane_escape": "Permit the code-generated short lane relocation; it is not an open-ended action hold.",
            "align_escape": "Prioritize the code-generated threat shadow: rotate to its heading and use only its bounded one-frame pulse.",
        }
        options = ["target_fire", "sweep_left", "sweep_right", "hold"]
        if self.lane_goal is not None and f.get("danger") in {"watch", "distant"}:
            options.append("lane_escape")
        if self.shadow is not None and f.get("danger") in {"critical", "immediate"}:
            options.append("align_escape")
        order = list(options)
        if (self.frame // 24) % 2:
            order.reverse()
        reverse = list(reversed(order))
        criteria_a = {key: labels[key] for key in order}
        criteria_b = {key: labels[key] for key in reverse}
        risks = {
            "clear": "No code-derived crossing is near; aim and use bounded FIRE.",
            "watch": "A later crossing is possible; allow only a short, rechecked relocation.",
            "immediate": "A near crossing needs rotate-only alignment or a bounded escape pulse.",
            "critical": "Current contact is imminent; survival and verified defense outrank score.",
            "unknown": "The player or geometry is unavailable; release controls.",
        }
        risk_order = ["clear", "watch", "immediate", "critical", "unknown"]
        if (self.frame // 48) % 2:
            risk_order.reverse()
        return {
            "mode_a": Choice(
                "Choose one id from candidates for this exact derived state. Do not invent actions, durations, or arithmetic.",
                criteria_a,
            ),
            "mode_b": Choice(
                "Independently choose the safest useful candidates id. align_escape means rotate before thrust; it never authorizes a held thrust.",
                criteria_b,
            ),
            "risk_a": Choice(
                "Classify snapshot.danger using only the supplied buckets. Unknown enemy coverage is not absence.",
                {key: risks[key] for key in risk_order},
            ),
            "risk_b": Choice(
                "Independently classify the supplied danger and impact buckets; do not calculate distances.",
                {key: risks[key] for key in reversed(risk_order)},
            ),
            "defense": Noul(
                "Should a bounded second mode-0 hyperspace pulse be considered only if defense.verify is still_critical and the current local gate remains critical? Otherwise false.",
                {
                    "true": "A failed landing may justify one verified retry under the local cap.",
                    "false": "Do not invoke hyperspace; use the current local escape or firing fallback.",
                },
            ),
        }

    def accept_jev(self, answers, origin, epoch):
        if not isinstance(answers, dict):
            return
        age = self.frame - origin
        if epoch != self.epoch or age < 0 or age > 48:
            self.log("jev_discard", frame=self.frame, judged_frame=origin, reason="stale_or_life")
            return
        mode_a = answer_value(answers.get("mode_a"))
        mode_b = answer_value(answers.get("mode_b"))
        allowed = set(self.questions()["mode_a"].criteria) if hasattr(self.questions()["mode_a"], "criteria") else {"target_fire", "sweep_left", "sweep_right", "hold", "lane_escape", "align_escape"}
        # Some local primitive versions do not expose criteria; use state-derived
        # allow-list rather than trusting arbitrary model strings.
        allowed = {"target_fire", "sweep_left", "sweep_right", "hold", "lane_escape", "align_escape"}
        if mode_a == mode_b and mode_a in allowed:
            threshold = 0.40 if mode_a == "align_escape" else 0.17
            if min(answer_probability(answers.get("mode_a"), mode_a), answer_probability(answers.get("mode_b"), mode_b)) >= threshold:
                if mode_a == "sweep_left":
                    self.posture = "sweep_left"
                elif mode_a == "sweep_right":
                    self.posture = "sweep_right"
                elif mode_a == "hold":
                    self.posture = "hold"
                elif mode_a == "align_escape":
                    self.posture = "avoid"
                    self.model_escape_hint = True
                else:
                    self.posture = "focus"
                self.posture_until = self.frame + 54
        risk_a = answer_value(answers.get("risk_a"))
        risk_b = answer_value(answers.get("risk_b"))
        risk_ok = risk_a == risk_b and risk_a in {"clear", "watch", "immediate", "critical", "unknown"}
        if risk_ok:
            p = min(answer_probability(answers.get("risk_a"), risk_a), answer_probability(answers.get("risk_b"), risk_b))
            if p >= (0.40 if risk_a == "critical" else 0.16):
                self.model_escape_hint = self.model_escape_hint or risk_a in {"critical", "immediate"}
        target_answer = answer_value(answers.get("target"))
        if isinstance(target_answer, str) and target_answer.startswith("T") and "_" in target_answer and answer_probability(answers.get("target"), target_answer) >= 0.15:
            body = target_answer[1:]
            slot, size = body.split("_", 1)
            if slot.lstrip("-").isdigit():
                self.selected_target = (int(slot), size)
        self.model_defense_hint = yes_probability(answers.get("defense")) >= 0.78
        self.judged_frame = origin
        self.judged_epoch = epoch
        self.log("jev_advice", frame=self.frame, judged_frame=origin, posture=self.posture, danger=risk_a if risk_ok else "disagree", defense=self.model_defense_hint)

    # ------------------------------ logging ------------------------------

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass


def current_clearance(facts):
    value = facts.get("clearance") if isinstance(facts, dict) else None
    return num(value, 0.0)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _jev(self, controller):
        if getattr(self.ctx, "system1", True) is None:
            return None
        state = controller.state_for_jev()
        questions = controller.questions()
        origin = controller.frame
        epoch = controller.epoch

        def call():
            return self.ctx.decide(state, questions, timeout_s=2.4)

        try:
            return self.ctx.spawn(call), origin, epoch
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    @staticmethod
    def _parse_plan(text):
        data = json_object(text)
        if not isinstance(data, dict):
            return {}
        allowed = {
            "aim": {"target", "sweep"},
            "risk": {"cautious", "balanced"},
            "fire": {"normal", "spaced"},
            "defense": {"retry", "never"},
            "patrol": {"adaptive_lane", "minimal"},
        }
        result = {}
        for key, choices in allowed.items():
            value = data.get(key)
            if isinstance(value, str) and value in choices:
                result[key] = value
        return result

    def _astra(self, controller, reason):
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if integer(obj(remaining).get("system2_calls"), 0) <= 0:
            return None
        f = controller.facts
        summary = {
            "reason": reason,
            "frame": controller.frame,
            "score": int(max(0.0, controller.score)),
            "lives": controller.lives,
            "danger": f.get("danger", "unknown"),
            "threat_time": f.get("threat_time"),
            "clearance": round(current_clearance(f), 2),
            "shadow_phase": controller.shadow.get("phase") if isinstance(controller.shadow, dict) else "none",
            "shadow_pulses": controller.shadow_pulses,
            "defense_count": controller.defense_count,
            "defense_verify": controller.defense_verify,
            "rocks": f.get("rock_count", 0),
            "plan": dict(controller.plan),
        }
        prompt = (
            "You are the slow strategic reviewer for a real-time Atari 2600 Asteroids "
            "mode-0/difficulty-0 controller. The local code owns every native frame, "
            "trajectory arithmetic, aligned threat-shadow pulses, FIRE release cadence, "
            "and verified hyperspace gates. Review aggregate evidence only. Return one "
            "small JSON object and no actions, sequence, timestamps, or explanation. "
            "Allowed fields are aim='target'|'sweep', risk='cautious'|'balanced', "
            "fire='normal'|'spaced', defense='retry'|'never', patrol='adaptive_lane'|'minimal'. "
            "Unknown enemy coverage is not absence. Evidence=" + json.dumps(summary, separators=(",", ":"))
        )

        def call():
            return self.ctx.think(prompt, timeout_s=105.0)

        try:
            return self.ctx.spawn(call)
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(game, "tick_s", None) is None:
            return
        controller = ShadowController(game, ctx)
        controller.update(read_observation(game))
        try:
            ctx.act(controller.choose())
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return
        except Exception:
            try:
                ctx.act(controller.noop)
            except Exception:
                return

        model_enabled = getattr(ctx, "system1", True) is not None
        jev_future = None
        jev_frame = -100000
        jev_epoch = -1
        next_jev = controller.frame
        astra_future = None
        astra_started = False
        astra_due = controller.frame + 320

        while not ctx.done and not getattr(game, "done", False):
            if model_enabled and jev_future is None and controller.frame >= next_jev:
                try:
                    started = self._jev(controller)
                    if started is not None:
                        jev_future, jev_frame, jev_epoch = started
                        next_jev = controller.frame + 24
                    else:
                        next_jev = controller.frame + 36
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    model_enabled = False
                except Exception:
                    next_jev = controller.frame + 36

            if not astra_started and astra_future is None and controller.frame >= astra_due:
                try:
                    astra_future = self._astra(controller, "scheduled aggregate review")
                    astra_started = True
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    astra_started = True
                except Exception:
                    astra_started = True

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            if ctx.done or getattr(game, "done", False):
                break

            controller.update(read_observation(game))

            if jev_future is not None and jev_future.done():
                finished = jev_future
                jev_future = None
                try:
                    controller.accept_jev(finished.result(), jev_frame, jev_epoch)
                except (BudgetExceeded, EpisodeOver):
                    model_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    pass

            if astra_future is not None and astra_future.done():
                finished = astra_future
                astra_future = None
                try:
                    result = finished.result()
                    new_plan = self._parse_plan(getattr(result, "text", result))
                    if new_plan:
                        controller.plan.update(new_plan)
                        controller.log("astra_plan", frame=controller.frame, plan=dict(controller.plan))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    return
                except Exception:
                    pass

            try:
                ctx.act(controller.choose())
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                try:
                    ctx.act(controller.noop)
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
