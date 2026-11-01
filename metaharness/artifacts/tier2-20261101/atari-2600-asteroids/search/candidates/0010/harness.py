"""Event-driven Asteroids controller with verified hyperspace retries.

The controller keeps the emulator clock local and treats defense as an experiment
rather than as a once-per-life magic button.  A DOWN pulse is followed by a short
verification window: if the ship did not actually relocate and the same collision
still exists, a second bounded pulse is allowed.  Ordinary movement is a small
receding-horizon escape plan, while aiming and FIRE cadence remain code-owned.
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


def _num(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value, default=0):
    result = _num(value)
    return int(result) if result is not None else default


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _pair(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _num(value[0])
        y = _num(value[1])
        if x is not None and y is not None:
            return x, y
    return None, None


def _wrap(value, period):
    value = _num(value, 0.0) or 0.0
    while value > period / 2.0:
        value -= period
    while value < -period / 2.0:
        value += period
    return value


def _heading_vector(code):
    angle = (int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def _vector_heading(dx, dy):
    if abs(dx) + abs(dy) < 1e-8:
        return 0
    # The public code convention is 0 up, 4 left, 8 down, 12 right.
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _turn_delta(current, desired):
    delta = (int(desired) - int(current)) % 16
    return delta - 16 if delta > 8 else delta


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


def _answer_prob(answer, key):
    if isinstance(answer, dict):
        probabilities = answer.get("probabilities", {})
    else:
        probabilities = getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    value = _num(probabilities.get(key), 0.0) or 0.0
    return max(0.0, min(1.0, value))


def _yes(answer):
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    number = _num(value)
    if number is not None:
        return max(0.0, min(1.0, number))
    return _answer_prob(answer, "true")


def _object_from_text(text):
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


class RecedingAsteroids:
    """Local physics approximation, firing policy, and defense state machine."""

    # These are deliberately small engineering horizons.  They bound action
    # persistence; they are not imported claims about the cartridge physics.
    ACCEL = 0.035
    FIRE_GAP = 5
    ESCAPE_THRUST = 4
    ESCAPE_THRUST_URGENT = 6
    ESCAPE_DEADLINE = 30
    MOVE_COOLDOWN = 46
    DEFENSE_RETRY_GAP = 16
    MAX_DEFENSES_PER_LIFE = 3
    RESPAWN_GRACE = 34

    def __init__(self, game, ctx):
        self.game = game
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        try:
            names = list(getattr(spec, "names", DEFAULT_ACTIONS)) if spec is not None else list(DEFAULT_ACTIONS)
        except Exception:
            names = list(DEFAULT_ACTIONS)
        self.names = set(names or DEFAULT_ACTIONS)
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else (next(iter(self.names), "NOOP"))

        self.frame = 0
        self.score = 0.0
        self.lives = 0
        self.previous_lives = None
        self.life_epoch = 0
        self.life_started = 0
        self.features = {}

        self.fire_next = 0
        self.sweep_sign = 1
        self.posture = "target"
        self.posture_until = -1
        self.model_defense_hint = False
        self.model_frame = -100000
        self.model_epoch = -1

        # Escape is a short plan, never an action latch.  It contains a desired
        # nose heading and a finite thrust allowance, not a future action list.
        self.escape = None
        self.next_move = 0
        self.last_action = self.noop
        self.last_log_frame = -100000

        # The central hypothesis of this candidate: a failed/ineffective DOWN
        # pulse should be verified and may be retried, subject to hard bounds.
        self.defenses_used = 0
        self.last_defense = -100000
        self.recovery_until = 0
        self.policy = {
            "aim": "target",
            "risk": "cautious",
            "fire": "normal",
            "defense": "retry",
        }

    # ------------------------------ observations ------------------------------

    def update(self, observation):
        obs = _dict(observation)
        old_lives = self.lives
        self.frame = _integer(obs.get("frame"), self.frame + 1)
        self.score = _num(obs.get("score"), self.score) or 0.0
        new_lives = _integer(obs.get("lives"), self.lives)

        if self.previous_lives is None:
            self.life_started = self.frame
            self.defenses_used = 0
        elif new_lives != self.previous_lives:
            self.life_epoch += 1
            self.life_started = self.frame
            self.defenses_used = 0
            self.last_defense = -100000
            self.escape = None
            self.model_defense_hint = False
            self.model_frame = -100000
            self.recovery_until = self.frame + self.RESPAWN_GRACE
            self.next_move = self.frame + self.RESPAWN_GRACE
            self.fire_next = max(self.fire_next, self.frame + 8)
            self._log("life_change", frame=self.frame, old_lives=old_lives, lives=new_lives, score=self.score)

        self.lives = new_lives
        self.previous_lives = new_lives
        self.features = self._derive(obs)

        if self.features.get("present") and self.features.get("discontinuity"):
            # A true discontinuity may be a hyperspace landing or a respawn.
            # Never continue an old escape vector across it.
            self.escape = None
            self.model_defense_hint = False
            self.recovery_until = max(self.recovery_until, self.frame + 10)

    def _derive(self, obs):
        player = _dict(obs.get("player"))
        present = bool(player.get("present"))
        px0 = _num(player.get("x"))
        py0 = _num(player.get("y"))
        pw = _num(player.get("width"), 6.0) or 6.0
        ph = _num(player.get("height"), 10.0) or 10.0
        if px0 is None or py0 is None:
            present = False
        px = (px0 + pw / 2.0) if present else None
        py = (py0 + ph / 2.0) if present else None
        pvx, pvy = _pair(player.get("velocity_px_per_frame"))
        velocity_known = present and pvx is not None and pvy is not None
        if pvx is None or pvy is None:
            pvx = pvy = 0.0

        rocks = []
        if present:
            for raw0 in _list(obs.get("asteroids")):
                raw = _dict(raw0)
                ax0 = _num(raw.get("x"))
                ay0 = _num(raw.get("y"))
                if ax0 is None or ay0 is None:
                    continue
                aw = _num(raw.get("width"), 8.0) or 8.0
                ah = _num(raw.get("height"), 15.0) or 15.0
                ax = ax0 + aw / 2.0
                ay = ay0 + ah / 2.0
                avx, avy = _pair(raw.get("velocity_px_per_frame"))
                known = velocity_known and avx is not None and avy is not None
                if avx is None or avy is None:
                    avx = avy = 0.0
                relative_x = avx - pvx
                relative_y = avy - pvy
                radius = max(aw, ah) / 2.0 + max(pw, ph) / 2.0 + 3.0
                # An unknown track is inflated rather than silently treated as a
                # safe, stationary object.
                if not known:
                    radius += 3.0
                rock = {
                    "slot": _integer(raw.get("source_slot"), -1),
                    "size": str(raw.get("size") or "unknown").lower(),
                    "dx": _wrap(ax - px, WIDTH),
                    "dy": _wrap(ay - py, HEIGHT),
                    "rvx": relative_x,
                    "rvy": relative_y,
                    "radius": radius,
                    "known": bool(known),
                    "width": aw,
                    "height": ah,
                }
                clear, near_t, hit_t = self._track_risk(rock, 0.0, 0.0, 0)
                rock["clearance"] = clear
                rock["near_t"] = near_t
                rock["hit_t"] = hit_t
                rocks.append(rock)

        nearest = min(rocks, key=lambda item: (item["clearance"], item["near_t"] if item["near_t"] is not None else 999)) if rocks else None
        if nearest is None:
            danger = "unknown" if not present else "clear"
            min_clear = None
            near_t = None
            hit_t = None
        else:
            min_clear = nearest["clearance"]
            near_t = nearest["near_t"]
            hit_t = nearest["hit_t"]
            if (hit_t is not None and hit_t <= 6) or min_clear <= 1.0:
                danger = "critical"
            elif (near_t is not None and near_t <= 17) or min_clear <= 8.0:
                danger = "urgent"
            elif (near_t is not None and near_t <= 35) or min_clear <= 22.0:
                danger = "watch"
            else:
                danger = "clear"

        # Value is a tie-breaker only.  A dangerous object always wins over a
        # distant high-value object.
        value = {"small": 3, "medium": 2, "large": 1}
        target = None
        if rocks:
            def target_key(item):
                imminent = item["hit_t"] if item["hit_t"] is not None else 999
                threat = 0 if imminent <= 45 else 1
                return (threat, imminent, -value.get(item["size"], 0), math.hypot(item["dx"], item["dy"]))
            target = min(rocks, key=target_key)

        missiles = sum(1 for item in _list(obs.get("player_missiles")) if isinstance(item, dict))
        heading = _integer(player.get("orientation_code"), 0) % 16 if present else None
        central_dx = _wrap((px if px is not None else 80.0) - 80.0, WIDTH)
        central_dy = _wrap((py if py is not None else 105.0) - 105.0, HEIGHT)
        return {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "present": present,
            "x": px,
            "y": py,
            "vx": pvx,
            "vy": pvy,
            "velocity_known": velocity_known,
            "speed": math.hypot(pvx, pvy) if present else 0.0,
            "heading": heading,
            "central_dx": central_dx,
            "central_dy": central_dy,
            "rocks": rocks,
            "rock_count": len(rocks),
            "missile_count": missiles,
            "nearest": nearest,
            "target": target,
            "min_clearance": min_clear,
            "near_t": near_t,
            "hit_t": hit_t,
            "danger": danger,
            "discontinuity": bool(player.get("motion_discontinuity")) if present else True,
        }

    @staticmethod
    def _track_risk(rock, ax, ay, pulse):
        """Return minimum clearance and first near/contact sample.

        ``ax, ay`` are a candidate's player acceleration and ``pulse`` is the
        number of frames for which it is applied.  All wrap arithmetic is local;
        no object slot is treated as a permanent identity.
        """
        best = float("inf")
        near_t = None
        hit_t = None
        rvx = rock.get("rvx", 0.0)
        rvy = rock.get("rvy", 0.0)
        radius = rock.get("radius", 15.0)
        for t in range(0, 61):
            used = min(t, max(0, int(pulse)))
            # Relative position = rock motion - player motion.
            player_x = 0.0
            player_y = 0.0
            if t:
                player_x = (0.5 * ax * used * used) + (ax * used * (t - used))
                player_y = (0.5 * ay * used * used) + (ay * used * (t - used))
            dx = _wrap(rock.get("dx", 0.0) + rvx * t - player_x, WIDTH)
            dy = _wrap(rock.get("dy", 0.0) + rvy * t - player_y, HEIGHT)
            clearance = math.hypot(dx, dy) - radius
            if clearance < best:
                best = clearance
            if near_t is None and clearance <= 10.0:
                near_t = t
            if hit_t is None and clearance <= 0.0:
                hit_t = t
        return best, near_t, hit_t

    # ---------------------------- receding-horizon ----------------------------

    def _project_heading(self, heading, pulse):
        hx, hy = _heading_vector(heading)
        total = 0.0
        minimum = float("inf")
        first_hit = None
        first_near = None
        for rock in self.features.get("rocks", []):
            clear, near_t, hit_t = self._track_risk(rock, self.ACCEL * hx, self.ACCEL * hy, pulse)
            minimum = min(minimum, clear)
            total += clear
            if near_t is not None and (first_near is None or near_t < first_near):
                first_near = near_t
            if hit_t is not None and (first_hit is None or hit_t < first_hit):
                first_hit = hit_t
        if not self.features.get("rocks"):
            minimum = 80.0
            total = 80.0
        # Keep the finite thrust pulse away from a boundary unless the threat
        # geometry strongly favors that side.  This is just a tie-breaker.
        f = self.features
        x = (f.get("x") or 80.0) + (f.get("vx") or 0.0) * 24.0 + 0.5 * self.ACCEL * hx * pulse * pulse
        y = (f.get("y") or 105.0) + (f.get("vy") or 0.0) * 24.0 + 0.5 * self.ACCEL * hy * pulse * pulse
        x %= WIDTH
        y %= HEIGHT
        edge = min(x, WIDTH - x, y, HEIGHT - y)
        edge_penalty = max(0.0, 16.0 - edge) * 1.0
        return {
            "heading": int(heading) % 16,
            "clearance": minimum,
            "total": total,
            "near_t": first_near,
            "hit_t": first_hit,
            "score": minimum + 0.08 * total - edge_penalty,
        }

    def _best_escape(self):
        if not self.features.get("present"):
            return None
        f = self.features
        current = self._project_heading(f.get("heading", 0), 0)
        candidates = [self._project_heading(heading, self.ESCAPE_THRUST_URGENT) for heading in range(0, 16, 2)]
        candidates.sort(key=lambda item: (item["score"], item["clearance"], -abs(_turn_delta(f.get("heading", 0), item["heading"]))), reverse=True)
        best = candidates[0] if candidates else None
        if best is None:
            return None
        # For a benign field do not turn merely to gain a numerical tie-breaker.
        if f.get("danger") in {"clear", "watch"} and best["score"] < current["score"] + 3.0:
            return None
        return best

    def _escape_needed(self):
        f = self.features
        if not f.get("present") or self.policy.get("risk") == "stationary":
            return False
        if f.get("danger") in {"critical", "urgent"}:
            return True
        if f.get("danger") == "watch" and self.frame >= self.next_move:
            best = self._best_escape()
            return best is not None and best["score"] > self._project_heading(f.get("heading", 0), 0)["score"] + 3.0
        return False

    def _arm_escape(self, force=False):
        if not self.features.get("present"):
            self.escape = None
            return
        if not force and self.frame < self.next_move:
            return
        best = self._best_escape()
        if best is None:
            return
        urgent = self.features.get("danger") in {"critical", "urgent"}
        self.escape = {
            "heading": best["heading"],
            "remaining": self.ESCAPE_THRUST_URGENT if urgent else self.ESCAPE_THRUST,
            "deadline": self.frame + (18 if urgent else self.ESCAPE_DEADLINE),
            "started": self.frame,
            "urgent": urgent,
        }
        self.next_move = self.frame + self.MOVE_COOLDOWN
        self._log("escape_plan", frame=self.frame, heading=best["heading"], score=round(best["score"], 2), danger=self.features.get("danger"))

    # ------------------------------- action map -------------------------------

    def legal(self, name):
        return name if name in self.names else self.noop

    def _turn_action(self, sign, fire=False):
        if sign > 0:
            return self.legal("LEFTFIRE" if fire else "LEFT")
        return self.legal("RIGHTFIRE" if fire else "RIGHT")

    def _thrust_action(self, sign=0, fire=False):
        if sign > 0:
            return self.legal("UPLEFTFIRE" if fire else "UPLEFT")
        if sign < 0:
            return self.legal("UPRIGHTFIRE" if fire else "UPRIGHT")
        return self.legal("UPFIRE" if fire else "UP")

    def _can_fire(self):
        if self.frame < self.fire_next:
            return False
        if self.features.get("missile_count", 0) >= 2:
            return False
        if self.policy.get("fire") == "spaced" and self.frame < self.fire_next + 2:
            return False
        return True

    def _fire(self):
        self.fire_next = self.frame + (7 if self.policy.get("fire") == "spaced" else self.FIRE_GAP)

    def _aim_action(self):
        f = self.features
        heading = f.get("heading")
        if heading is None:
            return self.noop
        fire = self._can_fire()
        target = f.get("target")
        if self.posture == "hold" or (self.posture == "sweep" and target is None):
            turn = self.sweep_sign
        elif self.posture == "sweep":
            turn = self.sweep_sign
        elif target is not None:
            # Lead a little in the measured relative direction; the model is
            # never asked to calculate this.
            lead = min(10.0, max(3.0, math.hypot(target["dx"], target["dy"]) / 7.0))
            desired = _vector_heading(target["dx"] + target["rvx"] * lead, target["dy"] + target["rvy"] * lead)
            delta = _turn_delta(heading, desired)
            if abs(delta) > 1:
                turn = 1 if delta > 0 else -1
            else:
                turn = 0
        else:
            if self.frame >= self.posture_until:
                self.sweep_sign *= -1
                self.posture_until = self.frame + 42
            turn = self.sweep_sign

        if turn:
            action = self._turn_action(turn, fire)
        else:
            action = self.legal("FIRE") if fire else self.noop
        if fire:
            self._fire()
        return action

    def _escape_action(self):
        plan = self.escape
        if not isinstance(plan, dict):
            return None
        if self.frame > _integer(plan.get("deadline"), self.frame):
            self.escape = None
            return None
        if self.features.get("danger") == "clear" and self.frame - _integer(plan.get("started"), self.frame) > 4:
            self.escape = None
            return None
        current = self.features.get("heading")
        if current is None:
            self.escape = None
            return None
        desired = _integer(plan.get("heading"), current) % 16
        delta = _turn_delta(current, desired)
        urgent = bool(plan.get("urgent")) or self.features.get("danger") in {"critical", "urgent"}
        if abs(delta) > 1:
            # Combined turn/thrust is reserved for an urgent window.  Normal
            # escape rotates first so inertia is not amplified accidentally.
            if urgent and _integer(plan.get("remaining"), 0) > 0:
                plan["remaining"] = _integer(plan.get("remaining"), 0) - 1
                return self._thrust_action(1 if delta > 0 else -1, False)
            return self._turn_action(1 if delta > 0 else -1, False)
        remaining = _integer(plan.get("remaining"), 0)
        if remaining <= 0:
            self.escape = None
            return None
        plan["remaining"] = remaining - 1
        return self._thrust_action(0, False)

    def _defense_allowed(self):
        f = self.features
        if not f.get("present"):
            return False
        if self.policy.get("defense") == "never":
            return False
        if self.defenses_used >= self.MAX_DEFENSES_PER_LIFE:
            return False
        if self.frame - self.last_defense < self.DEFENSE_RETRY_GAP:
            return False
        if self.frame - self.life_started < 8:
            return False
        if f.get("danger") != "critical":
            return False
        # Model advice can move the pulse slightly earlier only inside this hard
        # geometric gate.  The default local criterion is intentionally strict.
        return (
            (f.get("hit_t") is not None and f.get("hit_t") <= 5)
            or (f.get("min_clearance") is not None and f.get("min_clearance") <= 1.5)
            or (self.model_defense_hint and f.get("near_t") is not None and f.get("near_t") <= 10)
        )

    def _use_defense(self):
        self.defenses_used += 1
        self.last_defense = self.frame
        self.model_defense_hint = False
        self.escape = None
        self.recovery_until = self.frame + 10
        self._log("defense_pulse", frame=self.frame, count=self.defenses_used, danger=self.features.get("danger"), hit_t=self.features.get("hit_t"))
        return self.legal("DOWN")

    def choose(self):
        f = self.features
        if not f.get("present"):
            self.escape = None
            self.last_action = self.noop
            return self.noop

        # Current geometry always outranks a model answer.  In particular, this
        # gate permits a fresh retry when a previous DOWN did not relocate safely.
        if self._defense_allowed():
            self.last_action = self._use_defense()
            return self.last_action

        if self._escape_needed() and self.escape is None:
            self._arm_escape(force=True)
        if self.escape is not None:
            action = self._escape_action()
            if action is not None:
                self.last_action = action
                return action

        # Do not thrust during the short post-defense verification/recovery
        # interval.  FIRE/turning is harmless compared with an old escape plan.
        if self.frame < self.recovery_until:
            action = self._aim_action()
            self.last_action = action
            return action

        if self.posture_until < self.frame and self.posture == "sweep":
            self.posture = "target"
        action = self._aim_action()
        self.last_action = action
        if self.frame - self.last_log_frame >= 600 or action in {"FIRE", "UP", "UPLEFT", "UPRIGHT", "DOWN"}:
            self._log("control", frame=self.frame, action=action, danger=f.get("danger"), score=self.score, lives=self.lives, defenses=self.defenses_used)
            self.last_log_frame = self.frame
        return action

    # ---------------------------- model-facing data ----------------------------

    @staticmethod
    def _band(value, cuts, labels):
        if value is None:
            return labels[0]
        for cut, label in zip(cuts, labels[1:]):
            if value < cut:
                return label
        return labels[-1]

    @staticmethod
    def _bearing(dx, dy):
        h = "right" if dx > 10 else "left" if dx < -10 else "center"
        v = "below" if dy > 10 else "above" if dy < -10 else "level"
        return h + "_" + v

    def _candidate_labels(self):
        options = [
            ("target", "Follow the code-lead target and use bounded FIRE press/release pulses; never thrust from this answer."),
            ("sweep_left", "Make a short counterclockwise aiming sweep with bounded firing; no thrust."),
            ("sweep_right", "Make a short clockwise aiming sweep with bounded firing; no thrust."),
            ("hold", "Keep the current nose and use only the local firing cadence."),
        ]
        if self.escape is not None or self._escape_needed():
            options.append(("escape", "Permit the code-generated finite escape plan; it is not an instruction to hold thrust."))
        return options

    def state_for_jev(self):
        f = self.features
        target = f.get("target")
        target_summary = None
        if isinstance(target, dict):
            target_summary = {
                "size": target.get("size", "unknown"),
                "bearing": self._bearing(target.get("dx", 0.0), target.get("dy", 0.0)),
                "distance": self._band(math.hypot(target.get("dx", 0.0), target.get("dy", 0.0)), (30, 75, 125), ("unknown", "near", "middle", "far")),
                "motion": "known" if target.get("known") else "uncertain",
            }
        return {
            "rules": {
                "mode": 0,
                "fire": "button level; release before the next press",
                "noop": "does not brake; inertia continues",
                "down": "mode-0 hyperspace, not reverse thrust",
                "heading": "left increments and right decrements orientation code",
                "enemy_coverage": "unknown is not absence",
            },
            "snapshot": {
                "frame": int(f.get("frame", 0)),
                "lives": int(f.get("lives", 0)),
                "score": int(max(0.0, f.get("score", 0.0))),
                "player_present": bool(f.get("present")),
                "danger": f.get("danger", "unknown"),
                "impact_window": self._band(f.get("hit_t"), (6, 18, 40), ("none", "immediate", "soon", "later")),
                "clearance": self._band(f.get("min_clearance"), (2, 10, 25), ("unknown", "critical", "tight", "open")),
                "motion": "still" if f.get("speed", 0.0) < 0.2 else "moving",
                "rock_count": min(15, int(f.get("rock_count", 0))),
                "missiles": min(2, int(f.get("missile_count", 0))),
            },
            "threat": target_summary,
            "defense": {
                "used_this_life": int(self.defenses_used),
                "retry_available": self.defenses_used < self.MAX_DEFENSES_PER_LIFE,
                "last_pulse_age": max(0, self.frame - self.last_defense) if self.last_defense > -1000 else "none",
            },
            "candidate_modes": [{"id": key, "description": desc} for key, desc in self._candidate_labels()],
            "controller": {
                "posture": self.posture,
                "escape_active": self.escape is not None,
                "frame_local": True,
                "policy": dict(self.policy),
            },
        }

    def questions(self):
        labels = dict(self._candidate_labels())
        order = list(labels)
        if (self.frame // 24) % 2:
            order.reverse()
        reverse = list(reversed(order))
        criteria_a = {key: labels[key] for key in order}
        criteria_b = {key: labels[key] for key in reverse}
        danger_labels = {
            "clear": "No code-derived near crossing is present; aim and pulse FIRE.",
            "watch": "A crossing is possible later; keep thrust bounded and prefer an escape only if code generated one.",
            "urgent": "A near crossing needs a short escape or release policy; do not hold thrust.",
            "critical": "A current contact is imminent; local safety and a bounded defense retry outrank score.",
            "unknown": "Geometry is unavailable; release controls and wait for a fresh player observation.",
        }
        danger_order = ["clear", "watch", "urgent", "critical", "unknown"]
        if (self.frame // 48) % 2:
            danger_order.reverse()
        danger_criteria = {key: danger_labels[key] for key in danger_order}
        return {
            "mode_a": Choice(
                "Choose one candidate_modes id for this exact snapshot. The controller already computed geometry and finite durations. Do not invent a raw action or calculate numbers.",
                criteria_a,
            ),
            "mode_b": Choice(
                "Independently choose the safest useful candidate_modes id. An escape is a bounded code plan, never permission for an open-ended thrust hold.",
                criteria_b,
            ),
            "risk": Choice(
                "Classify snapshot.danger using the supplied buckets only. Unknown enemy coverage is not safety and arithmetic is already done.",
                danger_criteria,
            ),
            "defense": Noul(
                "Should the local controller be willing to retry one mode-0 hyperspace pulse if its current geometric gate remains critical and a prior pulse did not relocate safely? Otherwise answer false.",
                {
                    "true": "A bounded retry can be considered for a persistent current collision, subject to the local cooldown and per-life cap.",
                    "false": "Do not request hyperspace; let local steering or firing continue.",
                },
            ),
        }

    def accept_jev(self, answers, origin_frame, origin_epoch):
        if not isinstance(answers, dict):
            return False
        age = self.frame - origin_frame
        if age < 0 or age > 42 or origin_epoch != self.life_epoch:
            self._log("jev_discard", frame=self.frame, judged_frame=origin_frame, reason="stale_or_life")
            return False
        left = _answer_value(answers.get("mode_a"))
        right = _answer_value(answers.get("mode_b"))
        allowed = {key for key, _ in self._candidate_labels()}
        if left != right or left not in allowed:
            return False
        confidence = min(_answer_prob(answers.get("mode_a"), left), _answer_prob(answers.get("mode_b"), right))
        threshold = 0.34 if left == "escape" else 0.18
        if confidence < threshold:
            return False
        if left == "sweep_left":
            self.sweep_sign = 1
            self.posture = "sweep"
        elif left == "sweep_right":
            self.sweep_sign = -1
            self.posture = "sweep"
        elif left == "hold":
            self.posture = "hold"
        else:
            self.posture = "target"
        self.posture_until = self.frame + 50
        defense = answers.get("defense")
        self.model_defense_hint = _yes(defense) >= 0.76
        self.model_frame = origin_frame
        self.model_epoch = origin_epoch
        self._log("jev_policy", frame=self.frame, judged_frame=origin_frame, mode=left, risk=_answer_value(answers.get("risk")), defense=self.model_defense_hint)
        return True

    def accept_astra(self, text):
        data = _object_from_text(text)
        if not isinstance(data, dict):
            return False
        allowed = {
            "aim": {"target", "sweep"},
            "risk": {"cautious", "balanced"},
            "fire": {"normal", "spaced"},
            "defense": {"retry", "single", "never"},
        }
        changed = {}
        for key, values in allowed.items():
            value = data.get(key)
            if isinstance(value, str) and value in values:
                changed[key] = value
        if not changed:
            return False
        self.policy.update(changed)
        if self.policy.get("aim") == "sweep" and self.posture == "target":
            self.posture = "sweep"
            self.posture_until = self.frame + 60
        self._log("astra_policy", frame=self.frame, policy=dict(self.policy))
        return True

    def astra_summary(self, reason):
        f = self.features
        return {
            "reason": reason,
            "frame": self.frame,
            "score": int(max(0.0, self.score)),
            "lives": self.lives,
            "danger": f.get("danger", "unknown"),
            "min_clearance": round(_num(f.get("min_clearance"), 999.0) or 999.0, 2),
            "impact_frame": f.get("hit_t"),
            "rocks": int(f.get("rock_count", 0)),
            "defenses_used": self.defenses_used,
            "last_defense_age": max(0, self.frame - self.last_defense) if self.last_defense > -1000 else None,
            "policy": dict(self.policy),
        }

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _start_jev(self, controller):
        if getattr(self.ctx, "system1", True) is None:
            return None
        state = controller.state_for_jev()
        questions = controller.questions()
        origin_frame = controller.frame
        origin_epoch = controller.life_epoch

        def request():
            return self.ctx.decide(state, questions, timeout_s=2.5)

        try:
            future = self.ctx.spawn(request)
            return future, origin_frame, origin_epoch
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    def _start_astra(self, controller, reason):
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if _integer(_dict(remaining).get("system2_calls"), 0) <= 0:
            return None
        summary = controller.astra_summary(reason)
        prompt = (
            "You are a slow strategic reviewer for a real-time Atari 2600 Asteroids "
            "mode-0 difficulty-0 harness. The local code owns every native frame, "
            "wrap-aware geometry, finite thrust, FIRE release cadence, and a hard "
            "hyperspace retry gate. Do not emit actions, sequences, timestamps, or "
            "prose. Return one JSON object only.\n"
            + json.dumps(summary, separators=(",", ":"))
            + "\nAllowed fields and exact values: "
            '{"aim":"target"|"sweep", "risk":"cautious"|"balanced", '
            '"fire":"normal"|"spaced", "defense":"retry"|"single"|"never"}. '
            "Use aggregate evidence; keep movement bounded. The purpose of defense "
            "is to test a persistent current collision after an ineffective prior "
            "pulse, not to spam DOWN."
        )

        def request():
            return self.ctx.think(prompt, timeout_s=95.0)

        try:
            return self.ctx.spawn(request)
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
        controller = RecedingAsteroids(game, ctx)
        controller.update(_observation(game))
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
        astra_due = controller.frame + 360

        while not ctx.done and not getattr(game, "done", False):
            if model_enabled and jev_future is None and controller.frame >= next_jev:
                try:
                    started = self._start_jev(controller)
                    if started is not None:
                        jev_future, jev_frame, jev_epoch = started
                        next_jev = controller.frame + 24
                    else:
                        next_jev = controller.frame + 36
                except (BudgetExceeded, EpisodeOver):
                    model_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    next_jev = controller.frame + 36

            if not astra_started and astra_future is None and controller.frame >= astra_due:
                try:
                    astra_future = self._start_astra(controller, "scheduled review")
                    astra_started = True
                except (BudgetExceeded, EpisodeOver):
                    astra_started = True
                except ModelUnavailable:
                    return
                except Exception:
                    astra_started = True

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            if ctx.done or getattr(game, "done", False):
                break

            controller.update(_observation(game))

            if jev_future is not None and jev_future.done():
                finished = jev_future
                jev_future = None
                try:
                    answers = finished.result()
                    controller.accept_jev(answers, jev_frame, jev_epoch)
                except (BudgetExceeded, EpisodeOver):
                    model_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    # Timeout, malformed stand-in answers, and cancelled futures
                    # must not interrupt the local controller.
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
