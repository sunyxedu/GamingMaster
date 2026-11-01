"""Reachable-trajectory Asteroids controller.

The controller treats avoidance as a short-horizon trajectory choice rather than a
static lane choice.  It reserves a bounded cardinal thrust path while a contact is
still in the caution window, then rechecks that path every native frame.  Jev
chooses among code-generated, reachable maneuver packets; Astra reviews the
observed success of those packets and can change only durable policy.  The local
controller owns timing, geometry, FIRE release, and all emergency legality.
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
CARDINAL_CODES = (0, 4, 8, 12)
CARDINAL_NAMES = {0: "up", 4: "left", 8: "down", 12: "right"}


def _number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _integer(value, default=0):
    result = _number(value)
    return int(result) if result is not None else default


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _pair(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _number(value[0])
    y = _number(value[1])
    if x is None or y is None:
        return None
    return x, y


def _wrap(value, period):
    value = _number(value, 0.0) or 0.0
    return (value + period / 2.0) % period - period / 2.0


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
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _json_object(text):
    """Find the first JSON object in arbitrary Astra text."""
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


def _answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def _answer_probability(answer, key):
    if isinstance(answer, dict):
        probabilities = answer.get("probabilities", {})
    else:
        probabilities = getattr(answer, "probabilities", {})
    if not isinstance(probabilities, dict):
        return 0.0
    value = _number(probabilities.get(key), 0.0) or 0.0
    return max(0.0, min(1.0, value))


def _yes_probability(answer):
    value = _answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    value = _number(value)
    if value is not None:
        return max(0.0, min(1.0, value))
    return _answer_probability(answer, "true")


def _heading_vector(code):
    """The four calibrated cardinal codes, in screen coordinates."""
    angle = (int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def _desired_heading(dx, dy):
    if abs(dx) + abs(dy) < 1e-8:
        return 0
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def _turn_sign(current, desired):
    if current is None or desired is None:
        return 0
    delta = (int(desired) - int(current)) % 16
    if delta == 0:
        return 0
    if delta < 8:
        return 1  # LEFT increments the orientation code.
    if delta > 8:
        return -1  # RIGHT decrements the orientation code.
    return 1


class TrajectoryPilot:
    """Code-owned state machine with a finite, rechecked escape reservation."""

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
        self.obs = {}
        self.facts = {}
        self.previous_facts = None
        self.last_action = self.noop
        self.last_action_frame = -1

        # Astra may revise these durable knobs.  None of them can bypass the
        # local reachable-path and action-expiry gates.
        self.policy = {
            "escape": "reserve_early",
            "aim": "target",
            "fire": "normal",
            "defense": "overlap_only",
        }

        # A thrust estimate is learned only from observed finite differences.
        # It is deliberately bounded so a bad decoder cannot create a dangerous
        # high-speed imagined path.
        self.accel = 0.035
        self.accel_samples = 0
        self.reservations_started = 0
        self.reservations_completed = 0
        self.reservations_cancelled = 0
        self.thrust_frames = 0
        self.defenses = 0
        self.warning_streak = 0
        self.losses = 0

        self.move = None
        self.move_cooldown_until = 0
        self.respawn_until = 0
        self.defense_used = False
        self.last_defense_frame = -100000
        self.post_defense_until = 0

        self.fire_ready_frame = 0
        self.fire_release = 0
        self.sweep_sign = 1
        self.sweep_until = 0

        self.model_choice = None
        self.model_choice_until = -1
        self.model_defense_hint = False
        self.model_early_hint = False
        self.model_frame = -100000
        self.model_epoch = -1
        self.last_jev_options = []
        self.tactical = "focus"
        self.tactical_until = -1
        self.last_log = -100000

    # ---------- safe access and learned motion ----------

    def _log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def legal(self, action):
        return action if action in self.names else self.noop

    @staticmethod
    def _player(obs):
        return _dict(_dict(obs).get("player"))

    def _learn_motion(self, new_player):
        old = self.previous_facts or {}
        old_player = _dict(old.get("player"))
        old_v = old_player.get("velocity")
        new_v = _pair(new_player.get("velocity_px_per_frame"))
        old_v = _pair(old_v)
        if old_v is None or new_v is None:
            return
        if bool(new_player.get("motion_discontinuity")) or bool(old_player.get("discontinuity")):
            return
        if "UP" not in str(self.last_action):
            return
        heading = old_player.get("heading")
        if heading is None:
            return
        hx, hy = _heading_vector(heading)
        gain = (new_v[0] - old_v[0]) * hx + (new_v[1] - old_v[1]) * hy
        # The ROM has inertia and the RAM finite difference is quantized.  Only
        # accept a plausible positive component, never a guessed zero velocity.
        if 0.006 <= gain <= 0.12:
            self.accel = max(0.018, min(0.065, 0.88 * self.accel + 0.12 * gain))
            self.accel_samples += 1

    # ---------- geometry and trajectory scoring ----------

    def _rock_records(self, obs, player, px, py, pw, ph, pv, player_known):
        records = []
        for index, raw0 in enumerate(_list(obs.get("asteroids"))):
            raw = _dict(raw0)
            rx0 = _number(raw.get("x"))
            ry0 = _number(raw.get("y"))
            if rx0 is None or ry0 is None:
                continue
            rw = _number(raw.get("width"), 8.0) or 8.0
            rh = _number(raw.get("height"), 15.0) or 15.0
            av = _pair(raw.get("velocity_px_per_frame"))
            known = (
                av is not None and pv is not None and player_known
                and not bool(raw.get("motion_discontinuity"))
            )
            if known:
                rvx, rvy = av[0] - pv[0], av[1] - pv[1]
            else:
                rvx = rvy = 0.0
            size = str(raw.get("size") or "unknown").lower()
            value = {"small": 3, "medium": 2, "large": 1}.get(size, 1)
            cx, cy = rx0 + rw / 2.0, ry0 + rh / 2.0
            records.append({
                "slot": _integer(raw.get("source_slot"), index),
                "size": size,
                "value": value,
                "x": cx,
                "y": cy,
                "dx": _wrap(cx - px, WIDTH),
                "dy": _wrap(cy - py, HEIGHT),
                "rvx": rvx,
                "rvy": rvy,
                "velocity_known": bool(known),
                # Reference boxes are not masks.  Unknown motion receives a
                # larger conservative footprint rather than fabricated velocity.
                "half_x": rw / 2.0 + pw / 2.0 + 3.0 + (4.0 if not known else 0.0),
                "half_y": rh / 2.0 + ph / 2.0 + 3.0 + (4.0 if not known else 0.0),
                "distance": math.hypot(_wrap(cx - px, WIDTH), _wrap(cy - py, HEIGHT)),
            })
        return records

    def _project(self, rock, desired=None, thrust_frames=0, horizon=66, delay=0):
        """Return minimum clearance and first contact for a finite path."""
        if desired is None or thrust_frames <= 0:
            ax = ay = 0.0
        else:
            ax, ay = _heading_vector(desired)
        best = float("inf")
        best_t = None
        contact = None
        times = list(range(0, horizon + 1, 3))
        if times[-1] != horizon:
            times.append(horizon)
        relative_speed = math.hypot(rock["rvx"], rock["rvy"])
        uncertainty = 2.5 if not rock["velocity_known"] else 0.0
        for t in times:
            if desired is None or thrust_frames <= 0:
                extra_x = extra_y = 0.0
            else:
                active = min(max(0.0, float(t) - float(delay)), float(thrust_frames))
                after = max(0.0, float(t) - float(delay) - active)
                distance = self.accel * (0.5 * active * active + active * after)
                extra_x, extra_y = ax * distance, ay * distance
            rx = _wrap(rock["dx"] + rock["rvx"] * t - extra_x, WIDTH)
            ry = _wrap(rock["dy"] + rock["rvy"] * t - extra_y, HEIGHT)
            clearance = max(abs(rx) - rock["half_x"], abs(ry) - rock["half_y"])
            # Sampling every three frames can skip a crossing.  Inflate by a
            # small relative-motion bound; this is not a claimed collision mask.
            adjusted = clearance - min(8.0, 1.5 * relative_speed) - uncertainty
            if adjusted < best:
                best, best_t = adjusted, t
            if contact is None and adjusted <= 0.0:
                contact = max(0, t - 3)
        return best, best_t, contact

    @staticmethod
    def _risk_cost(clearance, contact, unknown=False):
        cost = max(0.0, 15.0 - clearance) * 0.62
        if contact is not None:
            cost += 22.0 + max(0.0, 38.0 - contact) * 1.55
        if unknown:
            cost *= 1.25
        return cost

    def _path(self, path_id, label, code, thrust, records, px, py, heading=None):
        total = 0.0
        minimum = 90.0
        now_minimum = 90.0
        first = None
        turn_delay = 0
        if code is not None and heading is not None:
            turn_delta = (int(code) - int(heading)) % 16
            turn_delta = min(turn_delta, 16 - turn_delta)
            turn_delay = min(8, turn_delta)
        for rock in records:
            uncertainty = 2.5 if not rock["velocity_known"] else 0.0
            now_clear = max(abs(rock["dx"]) - rock["half_x"], abs(rock["dy"]) - rock["half_y"]) - uncertainty
            now_minimum = min(now_minimum, now_clear)
            clearance, _, contact = self._project(rock, code, thrust, delay=turn_delay)
            minimum = min(minimum, clearance)
            total += self._risk_cost(clearance, contact, not rock["velocity_known"])
            if contact is not None and (first is None or contact < first):
                first = contact
        # A soft edge cost discourages a rescue that immediately commits to a
        # screen edge, but never defeats a real clearance gain.
        if code is not None and thrust > 0:
            vx, vy = _heading_vector(code)
            active = min(30.0, float(thrust))
            after = max(0.0, 30.0 - active)
            shift = self.accel * (0.5 * active * active + active * after)
            future_x = (px + vx * shift) % WIDTH
            future_y = (py + vy * shift) % HEIGHT
            edge = min(future_x, WIDTH - future_x, future_y, HEIGHT - future_y)
            total += max(0.0, 18.0 - edge) * 0.16
        return {
            "id": path_id,
            "label": label,
            "code": code,
            "thrust": thrust,
            "cost": total,
            "min_clear": minimum,
            "now_clear": now_minimum,
            "turn_delay": turn_delay,
            "first": first,
        }

    def _build_paths(self, records, px, py, heading):
        paths = [self._path("stay", "coast with no thrust", None, 0, records, px, py, heading)]
        if heading is not None:
            paths.append(self._path("nose", "two-frame thrust along the current nose", heading, 2, records, px, py, heading))
        for code in CARDINAL_CODES:
            path_id = CARDINAL_NAMES[code]
            paths.append(self._path(
                path_id,
                "bounded thrust toward the calibrated " + CARDINAL_NAMES[code] + " sector",
                code,
                3,
                records,
                px,
                py,
                heading,
            ))
        return paths

    def _derive(self, obs):
        player = self._player(obs)
        px0 = _number(player.get("x"))
        py0 = _number(player.get("y"))
        pw = _number(player.get("width"), 6.0) or 6.0
        ph = _number(player.get("height"), 10.0) or 10.0
        present = bool(player.get("present")) and px0 is not None and py0 is not None
        pv = _pair(player.get("velocity_px_per_frame")) if present else None
        player_known = pv is not None and not bool(player.get("motion_discontinuity"))
        px = px0 + pw / 2.0 if present else WIDTH / 2.0
        py = py0 + ph / 2.0 if present else HEIGHT / 2.0
        heading = _integer(player.get("orientation_code"), 0) % 16 if present else None
        records = self._rock_records(obs, player, px, py, pw, ph, pv, player_known) if present else []
        paths = self._build_paths(records, px, py, heading) if present else []
        stay = next((item for item in paths if item["id"] == "stay"), None)
        baseline_cost = stay["cost"] if stay is not None else 0.0
        candidates = [item for item in paths if item["id"] != "stay"]
        best = min(candidates, key=lambda item: (item["cost"], -item["min_clear"], item["thrust"]), default=stay)
        ttc = stay["first"] if stay is not None else None
        current_clear = stay["now_clear"] if stay is not None else None
        forecast_clear = stay["min_clear"] if stay is not None else None
        if not present:
            danger = "unknown"
        elif ttc is not None and ttc <= 6 or (forecast_clear is not None and forecast_clear <= 0):
            danger = "critical"
        elif ttc is not None and ttc <= 34 or (forecast_clear is not None and forecast_clear <= 8):
            danger = "caution"
        elif ttc is not None and ttc <= 66 or (forecast_clear is not None and forecast_clear <= 18):
            danger = "watch"
        else:
            danger = "clear"
        best_gain = baseline_cost - (best["cost"] if best is not None else baseline_cost)
        # A path is considered materially useful only when it beats coasting.
        # In critical contact the threshold is deliberately lower; any reachable
        # improvement is preferable to a lane label that may be the current lane.
        useful = bool(best is not None and best["id"] != "stay" and (
            best_gain >= (1.5 if danger == "critical" else 3.0)
            or (ttc is not None and ttc <= 34 and best["first"] is None)
        ))
        missiles = sum(1 for item in _list(obs.get("player_missiles")) if isinstance(item, dict))
        return {
            "frame": self.frame,
            "present": present,
            "x": px if present else None,
            "y": py if present else None,
            "heading": heading,
            "player": {
                "velocity": list(pv) if pv is not None else None,
                "heading": heading,
                "discontinuity": bool(player.get("motion_discontinuity")),
            },
            "rocks": records,
            "rock_count": len(records),
            "small_count": sum(r["size"] == "small" for r in records),
            "medium_count": sum(r["size"] == "medium" for r in records),
            "large_count": sum(r["size"] == "large" for r in records),
            "velocity_known": player_known,
            "speed": math.hypot(*pv) if pv is not None else None,
            "paths": paths,
            "stay": stay,
            "best": best,
            "best_gain": best_gain,
            "useful": useful,
            "ttc": ttc,
            "current_clear": current_clear,
            "forecast_clear": forecast_clear,
            "danger": danger,
            "missiles": missiles,
            "motion_discontinuity": bool(player.get("motion_discontinuity")) if present else True,
        }

    def update(self, obs):
        data = _dict(obs)
        self._learn_motion(_dict(data.get("player")))
        old_lives = self.lives
        self.previous_facts = self.facts if self.facts else None
        self.obs = data
        self.frame = _integer(data.get("frame"), self.frame + 1)
        self.score = _number(data.get("score"), self.score) or 0.0
        new_lives = _integer(data.get("lives"), self.lives)
        changed = self.previous_lives is not None and new_lives != self.previous_lives
        if self.previous_lives is None:
            self.respawn_until = self.frame + 24
        elif changed:
            self.life_epoch += 1
            if new_lives < old_lives:
                self.losses += old_lives - new_lives
            self.move = None
            self.reservations_cancelled += 1
            self.defense_used = False
            self.model_choice = None
            self.model_defense_hint = False
            self.model_early_hint = False
            self.model_epoch = -1
            self.respawn_until = self.frame + 30
            self.move_cooldown_until = self.frame + 34
            self.post_defense_until = self.frame + 20
            self.fire_release = max(self.fire_release, 2)
            self._log("life_change", frame=self.frame, old_lives=old_lives, lives=new_lives, score=self.score)
        self.lives = new_lives
        self.previous_lives = new_lives
        self.facts = self._derive(data)
        danger = self.facts.get("danger", "unknown")
        if danger in {"watch", "caution", "critical"}:
            self.warning_streak += 1
        else:
            self.warning_streak = max(0, self.warning_streak - 2)
        if self.facts.get("motion_discontinuity"):
            if self.move is not None:
                self.reservations_cancelled += 1
            self.move = None
            self.move_cooldown_until = max(self.move_cooldown_until, self.frame + 12)
            self.model_choice = None

        # A reservation is a hypothesis, not a command latch.  Re-score it on
        # every observation and cancel it if its path is no longer useful.
        if self.move is not None:
            active = next((p for p in self.facts.get("paths", []) if p["id"] == self.move.get("id")), None)
            if active is None or self.frame > _integer(self.move.get("expires"), self.frame):
                self.move = None
                self.reservations_cancelled += 1
                self.move_cooldown_until = max(self.move_cooldown_until, self.frame + 8)
            elif danger == "clear" and self.frame > _integer(self.move.get("started"), self.frame) + 3:
                self.move = None
                self.reservations_cancelled += 1
                self.move_cooldown_until = max(self.move_cooldown_until, self.frame + 8)
            elif not self.move.get("urgent") and active["cost"] > self.facts.get("stay", active)["cost"] + 2.0:
                self.move = None
                self.reservations_cancelled += 1
                self.move_cooldown_until = max(self.move_cooldown_until, self.frame + 8)

    # ---------- targets and local path reservation ----------

    def _target(self):
        records = self.facts.get("rocks", [])
        if not records:
            return None

        def key(rock):
            imminent = 8.0 if rock.get("distance", 999) < 60 else 0.0
            if rock.get("velocity_known"):
                imminent += 0.7
            # Threat first, then small-rock value, then a short shot.
            return (-imminent - 2.4 * rock.get("value", 1) + rock.get("distance", 100) / 150.0)

        return min(records, key=key)

    def _start_reservation(self, path, urgent):
        if not isinstance(path, dict) or path.get("code") is None:
            return False
        if self.frame < self.move_cooldown_until or self.frame < self.respawn_until:
            return False
        if self.move is not None:
            return True
        thrust = 2 if urgent else 3
        self.move = {
            "id": path.get("id"),
            "code": _integer(path.get("code"), 0) % 16,
            "thrust_left": thrust,
            "urgent": bool(urgent),
            "started": self.frame,
            "expires": self.frame + (16 if urgent else 26),
        }
        self.reservations_started += 1
        self._log(
            "reservation_start",
            frame=self.frame,
            path=path.get("id"),
            danger=self.facts.get("danger"),
            gain=round(_number(self.facts.get("best_gain"), 0.0) or 0.0, 2),
            urgent=bool(urgent),
        )
        return True

    def _reservation_action(self):
        plan = self.move
        if not isinstance(plan, dict):
            return None
        if self.frame > _integer(plan.get("expires"), self.frame):
            self.move = None
            self.reservations_cancelled += 1
            return None
        f = self.facts
        if not f.get("present") or f.get("heading") is None:
            self.move = None
            return self.noop
        desired = _integer(plan.get("code"), f.get("heading")) % 16
        current = _integer(f.get("heading"), desired) % 16
        delta = (desired - current) % 16
        if delta > 8:
            delta -= 16
        urgent = bool(plan.get("urgent")) or f.get("danger") == "critical"
        if delta != 0:
            # In critical contact, a combined sample is useful because waiting
            # for eight pure rotation frames can spend the contact window.  In
            # caution, rotate first so the reservation is genuinely reachable.
            turn = 1 if delta > 0 else -1
            if urgent and _integer(plan.get("thrust_left"), 0) > 0:
                plan["thrust_left"] = _integer(plan.get("thrust_left"), 0) - 1
                self.thrust_frames += 1
                return self._compose(turn, thrust=True, fire=False)
            return self._compose(turn, thrust=False, fire=False)
        left = _integer(plan.get("thrust_left"), 0)
        if left <= 0:
            self.reservations_completed += 1
            self.move = None
            self.move_cooldown_until = self.frame + (18 if urgent else 52)
            return None
        plan["thrust_left"] = left - 1
        self.thrust_frames += 1
        return self._compose(0, thrust=True, fire=False)

    def _best_path(self, urgent=False):
        f = self.facts
        paths = f.get("paths", [])
        if not paths:
            return None
        baseline = f.get("stay")
        best = f.get("best")
        if best is None or baseline is None:
            return None
        if best.get("id") == "stay":
            return None
        gain = _number(f.get("best_gain"), 0.0) or 0.0
        threshold = 1.0 if urgent else (2.0 if self.policy.get("escape") == "reserve_early" else 3.5)
        # Jev can pick between genuinely comparable reachable packets during a
        # noncritical window.  Code still owns the critical choice and rejects a
        # packet that is materially worse than both coasting and the best forecast.
        selected = self.model_choice if self.frame <= self.model_choice_until else None
        if selected and not urgent and f.get("danger") in {"watch", "caution"}:
            candidate = next((p for p in paths if p.get("id") == selected), None)
            if candidate is not None and candidate.get("id") != "stay":
                if (
                    candidate.get("cost", 999.0) <= baseline.get("cost", 0.0) + 0.5
                    and candidate.get("cost", 999.0) <= best.get("cost", 999.0) + 2.0
                ):
                    return candidate
        if gain >= threshold or (urgent and best.get("first") is None):
            return best
        # A model-selected path is accepted only if the latest local geometry
        # still says it is not materially worse than coasting.
        if selected:
            candidate = next((p for p in paths if p.get("id") == selected), None)
            if candidate is not None and candidate.get("id") != "stay":
                if candidate.get("cost", 999.0) <= baseline.get("cost", 0.0) + (1.5 if urgent else 0.5):
                    return candidate
        return None

    # ---------- defense, aiming, and FIRE ----------

    def _should_defend(self):
        f = self.facts
        if not f.get("present") or self.defense_used:
            return False
        if self.policy.get("defense") == "never":
            return False
        if f.get("danger") != "critical":
            return False
        stay = f.get("stay") or {}
        best = f.get("best") or {}
        ttc = f.get("ttc")
        clearance = _number(f.get("current_clear"), 999.0)
        forecast = _number(f.get("forecast_clear"), 999.0)
        # Hyperspace is deliberately not the normal response.  It is used only
        # when the current approximate boxes already overlap and no short thrust
        # trajectory improves the forecast.  Jev may support this gate but cannot
        # lower it or create a second pulse.
        no_escape = best.get("id") in {None, "stay"} or best.get("min_clear", -999.0) <= forecast + 1.0
        overlap = clearance <= -2.0 or (ttc is not None and ttc <= 1 and no_escape)
        supported = self.model_defense_hint and clearance <= -1.0 and no_escape
        return bool(overlap or supported)

    def _fire_ready(self):
        if self.frame < self.fire_ready_frame or self.fire_release > 0:
            return False
        if self.facts.get("missiles", 0) >= 2:
            return False
        gap = 6 if self.policy.get("fire") == "spaced" else 5
        self.fire_ready_frame = self.frame + gap
        self.fire_release = 1
        return True

    def _aim_turn(self):
        f = self.facts
        heading = f.get("heading")
        if heading is None:
            return 0
        if self.frame > self.tactical_until:
            self.tactical = "focus" if self.policy.get("aim") == "target" else (
                "sweep_left" if self.sweep_sign > 0 else "sweep_right"
            )
            self.tactical_until = self.frame + 48
        if self.tactical == "hold":
            return 0
        if self.tactical == "sweep_left":
            return 1
        if self.tactical == "sweep_right":
            return -1
        target = self._target()
        if target is not None and self.tactical == "focus":
            lead_x = target["dx"] + target["rvx"] * 7.0
            lead_y = target["dy"] + target["rvy"] * 7.0
            return _turn_sign(heading, _desired_heading(lead_x, lead_y))
        if target is not None and self.policy.get("aim") == "target":
            return _turn_sign(heading, _desired_heading(target["dx"], target["dy"]))
        if self.frame >= self.sweep_until:
            self.sweep_sign *= -1
            self.sweep_until = self.frame + 50
        return 1 if self.sweep_sign > 0 else -1

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

    def choose(self):
        f = self.facts
        if not f.get("present"):
            self.move = None
            self.fire_release = max(self.fire_release, 2)
            self.last_action = self.noop
            return self.noop
        if self.fire_release > 0:
            self.fire_release -= 1

        if self._should_defend():
            self.defense_used = True
            self.defenses += 1
            self.last_defense_frame = self.frame
            self.post_defense_until = self.frame + 20
            self.move = None
            self.model_defense_hint = False
            self._log(
                "defense",
                frame=self.frame,
                clearance=round(_number(f.get("current_clear"), 0.0) or 0.0, 2),
                ttc=f.get("ttc"),
            )
            self.last_action = self.legal("DOWN")
            return self.last_action

        urgent = f.get("danger") == "critical"
        if self.move is None and self.frame >= self.post_defense_until:
            path = self._best_path(urgent=urgent)
            # Reserve during caution under the early policy.  A model commit
            # hint may open the same local gate during a watch, never a clear
            # field and never a path that fails the latest forecast.
            reserve_now = urgent or f.get("danger") == "caution"
            if f.get("danger") == "watch":
                reserve_now = self.policy.get("escape") == "reserve_early" and self.model_early_hint
            if reserve_now and path is not None:
                self._start_reservation(path, urgent=urgent)

        action = self._reservation_action()
        if action is not None:
            self.last_action = action
            return action

        # Scoring posture: no thrust unless the finite-path arbiter requested it.
        turn = self._aim_turn()
        fire = False
        if f.get("danger") not in {"critical", "caution"}:
            fire = self._fire_ready()
        action = self._compose(turn, thrust=False, fire=fire)
        self.last_action = action
        if fire or self.frame - self.last_log >= 600:
            self._log(
                "control",
                frame=self.frame,
                action=action,
                danger=f.get("danger"),
                clearance=round(_number(f.get("current_clear"), 0.0) or 0.0, 2),
                ttc=f.get("ttc"),
                score=self.score,
                lives=self.lives,
            )
            self.last_log = self.frame
        return action

    # ---------- compact Jev interface ----------

    @staticmethod
    def _clearance_bucket(value):
        if value is None:
            return "unknown"
        if value < 0:
            return "overlap"
        if value < 7:
            return "tight"
        if value < 18:
            return "limited"
        return "clear"

    def _path_view(self):
        f = self.facts
        baseline = f.get("stay") or {}
        result = []
        for path in f.get("paths", []):
            if path.get("id") == "stay":
                continue
            cost = _number(path.get("cost"), 999.0) or 999.0
            baseline_cost = _number(baseline.get("cost"), 0.0) or 0.0
            # Only expose a maneuver packet when code has a real reason to
            # consider it.  This keeps Jev's option set small and meaningful.
            if path.get("id") == "nose" or cost <= baseline_cost + 5.0 or f.get("danger") in {"caution", "critical"}:
                result.append({
                    "id": path.get("id"),
                    "direction": CARDINAL_NAMES.get(path.get("code"), "current_nose"),
                    "clearance": self._clearance_bucket(path.get("min_clear")),
                    "contact": _bucket(path.get("first"), (4, 15, 35), ("none", "critical", "soon", "later")),
                    "bounded": True,
                })
        return result

    def jev_state(self):
        f = self.facts
        target = self._target()
        target_summary = None
        if target is not None:
            target_summary = {
                "size": target.get("size", "unknown"),
                "bearing": "right" if target.get("dx", 0.0) > 10 else "left" if target.get("dx", 0.0) < -10 else "center",
                "vertical": "below" if target.get("dy", 0.0) > 10 else "above" if target.get("dy", 0.0) < -10 else "level",
                "distance": _bucket(target.get("distance"), (30, 80, 125), ("unknown", "near", "middle", "far")),
                "motion": "known" if target.get("velocity_known") else "uncertain",
            }
        return {
            "rules": {
                "mode": 0,
                "fire": "press_then_release; holding does not rapid-fire",
                "noop": "preserves inertia; not a brake",
                "down": "mode-0 hyperspace; last resort only",
                "orientation": "LEFT increments and RIGHT decrements code",
                "enemy_coverage": "null means unknown, not absent",
            },
            "snapshot": {
                "frame": int(self.frame),
                "lives": int(self.lives),
                "score": int(max(0, self.score)),
                "player_present": bool(f.get("present")),
                "position": "unknown" if f.get("x") is None else _bucket(f.get("x"), (55, 105), ("unknown", "left", "middle", "right")),
                "speed": "unknown" if f.get("speed") is None else ("still" if f["speed"] < 0.25 else "moving"),
                "danger": f.get("danger", "unknown"),
                "clearance": self._clearance_bucket(f.get("current_clear")),
                "impact": _bucket(f.get("ttc"), (4, 15, 35, 67), ("none", "critical", "soon", "watch", "later")),
                "asteroids": int(f.get("rock_count", 0)),
                "small": int(f.get("small_count", 0)),
                "missiles": int(f.get("missiles", 0)),
                "warning_streak": min(30, int(self.warning_streak)),
                "motion_discontinuity": bool(f.get("motion_discontinuity")),
            },
            "target": target_summary,
            "paths": self._path_view(),
            "reservation": {
                "active": bool(self.move),
                "path": self.move.get("id") if isinstance(self.move, dict) else "none",
                "defenses_used_this_life": bool(self.defense_used),
                "early_policy": self.policy.get("escape"),
            },
            "controller": {
                "tactical": self.tactical,
                "acceleration_estimate": _bucket(self.accel, (0.026, 0.045, 0.058), ("low", "normal", "high", "very_high")),
                "frame_local": True,
            },
        }

    def jev_questions(self):
        path_view = self._path_view()
        options = {"turret": "Keep a no-thrust target/sweep posture; FIRE remains code-gated."}
        for item in path_view:
            options[item["id"]] = (
                "Reserve the named calibrated " + str(item.get("direction")) +
                " trajectory for only a few thrust frames; code rechecks it each frame."
            )
        if self.facts.get("danger") == "critical" and not self.defense_used:
            options["defend"] = "Consider one mode-0 hyperspace pulse only if the local overlap gate still passes."
        self.last_jev_options = list(options)
        order = list(options)
        if (self.frame // 30) % 2:
            order.reverse()
        reverse = list(reversed(order))
        risk_order = ["clear", "watch", "caution", "critical", "unknown"]
        if (self.frame // 48) % 2:
            risk_order.reverse()
        risk_text = {
            "clear": "No code-derived near-term contact; preserve the bounded firing posture.",
            "watch": "A later crossing is possible; do not thrust unless a named safe reservation is justified.",
            "caution": "A contact is approaching; prefer a named reachable reservation over scoring greed.",
            "critical": "Contact is current or imminent; survival uses the latest code path or overlap gate.",
            "unknown": "The player or motion facts are insufficient; release controls and do not invent safety.",
        }
        return {
            "mode_a": Choice(
                "Choose one option for this exact snapshot. Options are semantic names for code-generated bounded maneuvers, never raw actions or sequences. Do not invent a path.",
                {key: options[key] for key in order},
            ),
            "mode_b": Choice(
                "Independently choose the safest useful option. A reservation is legal only if the current code forecast still validates it; turret has no thrust.",
                {key: options[key] for key in reverse},
            ),
            "risk": Choice(
                "Classify risk from snapshot.danger, snapshot.clearance, and snapshot.impact. Do not calculate numbers and do not treat unknown enemy coverage as empty.",
                {key: risk_text[key] for key in risk_order},
            ),
            "commit": Noul(
                "Should the local controller open a cautious named trajectory reservation now? Say true only for watch or caution when a path in paths is materially safer; false for clear, unknown, or an unsafe path.",
                {"true": "A validated bounded reservation is useful before the contact becomes critical.", "false": "Keep the turret or local fallback; do not add thrust from this judgment."},
            ),
        }

    def accept_jev(self, answers, origin_frame, origin_epoch, origin_options):
        if not isinstance(answers, dict):
            return
        if origin_epoch != self.life_epoch or self.frame < origin_frame or self.frame - origin_frame > 44:
            self._log("jev_discard", frame=self.frame, judged_frame=origin_frame, reason="stale_or_life")
            return
        a, b = answers.get("mode_a"), answers.get("mode_b")
        va, vb = _answer_value(a), _answer_value(b)
        allowed = set(origin_options)
        if va != vb or va not in allowed:
            return
        threshold = 0.36 if va in {"defend", "up", "left", "down", "right", "nose"} else 0.18
        if min(_answer_probability(a, va), _answer_probability(b, vb)) < threshold:
            return
        if va == "turret":
            self.model_choice = "turret"
            self.model_choice_until = self.frame + 40
            self.tactical_until = min(self.tactical_until, self.frame + 30)
        elif va == "defend":
            self.model_defense_hint = True
        else:
            self.model_choice = va
            self.model_choice_until = self.frame + 36
        commit = answers.get("commit")
        if _yes_probability(commit) >= 0.72:
            self.model_early_hint = True
        else:
            self.model_early_hint = False
        risk = _answer_value(answers.get("risk"))
        if risk not in {"clear", "watch", "caution", "critical", "unknown"}:
            risk = "invalid"
        self.model_frame = origin_frame
        self.model_epoch = origin_epoch
        self._log("jev_advice", frame=self.frame, judged_frame=origin_frame, choice=va, risk=risk, commit=_yes_probability(commit))

    # ---------- Astra durable review ----------

    def astra_summary(self, reason):
        f = self.facts
        best = f.get("best") or {}
        return {
            "reason": reason,
            "frame": self.frame,
            "score": int(max(0, self.score)),
            "lives": self.lives,
            "losses": self.losses,
            "danger": f.get("danger"),
            "warning_streak": min(40, self.warning_streak),
            "clearance": round(_number(f.get("current_clear"), 0.0) or 0.0, 2),
            "impact": f.get("ttc"),
            "best_path": best.get("id"),
            "best_gain": round(_number(f.get("best_gain"), 0.0) or 0.0, 2),
            "reservations_started": self.reservations_started,
            "reservations_completed": self.reservations_completed,
            "reservations_cancelled": self.reservations_cancelled,
            "thrust_frames": self.thrust_frames,
            "defenses": self.defenses,
            "acceleration_estimate": round(self.accel, 4),
            "policy": dict(self.policy),
        }

    def apply_astra(self, text):
        data = _json_object(text)
        if not isinstance(data, dict):
            return False
        allowed = {
            "escape": {"reserve_early", "caution_only"},
            "aim": {"target", "sweep"},
            "fire": {"normal", "spaced"},
            "defense": {"overlap_only", "never"},
        }
        changed = {}
        for key, values in allowed.items():
            value = data.get(key)
            if isinstance(value, str) and value in values:
                changed[key] = value
        if not changed:
            return False
        self.policy.update(changed)
        self._log("astra_policy", frame=self.frame, policy=dict(self.policy))
        return True


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _start_jev(self, pilot):
        if getattr(self.ctx, "system1", True) is None:
            return None
        state = pilot.jev_state()
        questions = pilot.jev_questions()
        origin_frame = pilot.frame
        origin_epoch = pilot.life_epoch
        origin_options = list(pilot.last_jev_options)

        def request():
            return self.ctx.decide(state, questions, timeout_s=2.3)

        try:
            future = self.ctx.spawn(request)
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None
        return future, origin_frame, origin_epoch, origin_options

    def _start_astra(self, pilot, reason):
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            remaining = {}
        if _integer(_dict(remaining).get("system2_calls"), 0) <= 0:
            return None
        summary = pilot.astra_summary(reason)
        prompt = (
            "You are the slow strategic reviewer for a real-time Atari 2600 "
            "Asteroids mode-0/difficulty-0 controller. The code owns every native "
            "frame, object geometry, wrap-aware trajectory arithmetic, FIRE release, "
            "life bookkeeping, and emergency legality. Jev only chooses among named "
            "code-generated reachable packets. Review aggregate evidence, diagnose "
            "whether reservations are too late or too frequent, and return one small "
            "JSON object only. Never return actions, sequences, timestamps, or prose.\n"
            + json.dumps(summary, separators=(",", ":"))
            + "\nAllowed exact fields and values: "
            '{"escape":"reserve_early"|"caution_only",'
            '"aim":"target"|"sweep",'
            '"fire":"normal"|"spaced",'
            '"defense":"overlap_only"|"never"}. '
            "Unknown enemy coverage is not absence. Reserve early means only a "
            "validated finite path may begin during caution; it does not authorize "
            "open-loop thrust."
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
        pilot = TrajectoryPilot(game, ctx)
        pilot.update(_observation(game))

        def post(action):
            try:
                ctx.act(action)
                return True
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return False
            except Exception:
                try:
                    ctx.act(pilot.noop)
                    return True
                except Exception:
                    return False

        if not post(pilot.choose()):
            return

        s1_enabled = getattr(ctx, "system1", True) is not None
        jev_future = None
        jev_frame = -100000
        jev_epoch = -1
        jev_options = []
        next_jev = pilot.frame

        astra_future = None
        astra_started = False
        astra_due = pilot.frame + 360

        while not ctx.done and not getattr(game, "done", False):
            if s1_enabled and jev_future is None and pilot.frame >= next_jev:
                try:
                    result = self._start_jev(pilot)
                    if result is not None:
                        jev_future, jev_frame, jev_epoch, jev_options = result
                        next_jev = pilot.frame + 24
                    else:
                        next_jev = pilot.frame + 38
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except Exception:
                    next_jev = pilot.frame + 38

            if not astra_started and astra_future is None and pilot.frame >= astra_due:
                try:
                    astra_future = self._start_astra(pilot, "first aggregate review")
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

            pilot.update(_observation(game))

            if jev_future is not None and jev_future.done():
                future = jev_future
                jev_future = None
                try:
                    pilot.accept_jev(future.result(), jev_frame, jev_epoch, jev_options)
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    # A timeout, cancellation, arbitrary stand-in answer, or
                    # malformed typed response is an advisory miss.
                    pass

            if astra_future is not None and astra_future.done():
                future = astra_future
                astra_future = None
                try:
                    result = future.result()
                    pilot.apply_astra(getattr(result, "text", result))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    return
                except Exception:
                    pass

            if not post(pilot.choose()):
                return


def build(ctx):
    return Harness(ctx)
