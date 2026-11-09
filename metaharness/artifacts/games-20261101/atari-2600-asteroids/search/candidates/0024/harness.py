"""Candidate 0024: a reward-closed-loop gun controller.

The controller keeps a quiet, frame-local turret and learns two things from the
public observation stream: projectile velocity by heading, and which headings
actually preceded score.  A recent score-producing bearing gets a short,
revalidated dwell rather than immediately being replaced by a noisy slot
forecast.  Asteroid geometry, collision reactions, input expiry, and model
freshness remain deterministic.
"""
from __future__ import annotations

import json
import math

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


W = 160.0
H = 210.0
DEFAULT_ACTIONS = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
    "UPLEFTFIRE",
)


def number(value, default=None):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def integer(value, default=0):
    x = number(value)
    return int(x) if x is not None else default


def mapping(value):
    return value if isinstance(value, dict) else {}


def sequence(value):
    return value if isinstance(value, list) else []


def point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = number(value[0]), number(value[1])
    return (x, y) if x is not None and y is not None else None


def wrapped(value, period):
    x = number(value, 0.0) or 0.0
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


def observation(game):
    try:
        value = game.observe()
    except Exception:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def json_object(text):
    """Find one object in an arbitrary Astra response without trusting prose."""
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


def answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def answer_probability(answer, key):
    if isinstance(answer, dict):
        probs = answer.get("probabilities", {})
    else:
        probs = getattr(answer, "probabilities", {})
    if not isinstance(probs, dict):
        return 0.0
    value = number(probs.get(key), 0.0) or 0.0
    return max(0.0, min(1.0, value))


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
        return max(0.0, min(1.0, number(value, 0.0) or 0.0))
    return answer_probability(answer, "true")


def heading_vector(code):
    angle = (int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def heading_for(dx, dy):
    if abs(dx) + abs(dy) < 1e-9:
        return 0
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def turn_delta(current, desired):
    if current is None or desired is None:
        return 0
    d = (int(desired) - int(current)) % 16
    return d - 16 if d > 8 else d


def turn_distance(current, desired):
    return abs(turn_delta(current, desired))


def turn_sign(current, desired):
    d = turn_delta(current, desired)
    return 1 if d > 0 else -1 if d < 0 else 0


class RewardTurret:
    """A low-thrust turret with score feedback and a small escape envelope."""

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
        self.last_score = 0.0
        self.lives = 0
        self.old_lives = None
        self.epoch = 0
        self.obs = {}
        self.player = {}
        self.facts = {}
        self.center = None
        self.previous_center = None
        self.player_velocity = None
        self.last_heading = None
        self.last_action = self.noop

        # The model can select a durable *priority*, never a raw action.
        self.focus = "geometry"
        self.focus_until = -1
        self.plan = {"focus": "geometry", "lock": "short", "fire": "normal"}

        # Actual missile observations are an online calibration table.  A table
        # entry is {vx, vy, n, error}; the heading is the heading at FIRE press.
        self.missiles = {}
        self.ballistics = {}
        self.recent_presses = []
        self.shot_history = []
        self.shots_posted = 0
        self.shots_seen = 0
        self.calibration_rejects = 0

        # Reward attribution is intentionally local and approximate.  It is not
        # used as a score substitute: it only chooses a short aim dwell.
        self.heading_reward = [0.0] * 16
        self.heading_uses = [1.0] * 16
        self.reward_heading = None
        self.reward_lock_until = -1
        self.reward_events = 0
        self.reward_lock_events = 0
        self.last_reward_frame = -100000
        self.last_decay = 0

        self.selected = None
        self.opportunities = []
        self.rank_frame = -100000
        self.next_fire = 0
        self.last_shot_frame = -100000
        self.sweep_sign = 1
        self.next_sweep = 0

        # Collision response is deliberately sparse.  A candidate must improve
        # a short swept forecast before one pulse is allowed.
        self.escape = None
        self.next_escape = 0
        self.escape_starts = 0
        self.escape_pulses = 0
        self.recovery_until = 0
        self.defenses = 0
        self.last_defense = -100000
        self.defense_epoch_count = 0
        self.last_log = -100000

    # ----------------------------- safe plumbing ----------------------------

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def legal(self, action):
        return action if action in self.names else self.noop

    @staticmethod
    def _player_center(player):
        if not bool(player.get("present")):
            return None
        x, y = number(player.get("x")), number(player.get("y"))
        if x is None or y is None:
            return None
        width = number(player.get("width"), 6.0) or 6.0
        height = number(player.get("height"), 10.0) or 10.0
        return x + width / 2.0, y + height / 2.0, width, height

    # ------------------------------ observations ----------------------------

    def _read_velocity(self, player, finite, discontinuity):
        reported = point(player.get("velocity_px_per_frame"))
        if discontinuity:
            return None
        if reported is None:
            return finite
        if finite is None:
            return reported
        if abs(reported[0]) + abs(reported[1]) < 0.01 and abs(finite[0]) + abs(finite[1]) > 0.08:
            return finite
        return 0.75 * reported[0] + 0.25 * finite[0], 0.75 * reported[1] + 0.25 * finite[1]

    def _make_rocks(self, obs, center, player_velocity):
        if center is None:
            return []
        px, py, pw, ph = center
        rocks = []
        for index, raw0 in enumerate(sequence(obs.get("asteroids"))):
            raw = mapping(raw0)
            x0, y0 = number(raw.get("x")), number(raw.get("y"))
            if x0 is None or y0 is None:
                continue
            rw = number(raw.get("width"), 8.0) or 8.0
            rh = number(raw.get("height"), 15.0) or 15.0
            av = point(raw.get("velocity_px_per_frame"))
            known = av is not None and player_velocity is not None and not bool(raw.get("motion_discontinuity"))
            if av is None:
                av = (0.0, 0.0)
            pv = player_velocity if player_velocity is not None else (0.0, 0.0)
            cx, cy = x0 + rw / 2.0, y0 + rh / 2.0
            size = str(raw.get("size") or "unknown").lower()
            slot = integer(raw.get("source_slot"), index)
            rocks.append({
                "key": "%s:%s:%s" % (slot, size, index),
                "slot": slot,
                "size": size,
                "value": {"small": 3, "medium": 2, "large": 1}.get(size, 1),
                "x": cx,
                "y": cy,
                "dx": wrapped(cx - px, W),
                "dy": wrapped(cy - py, H),
                "vx": av[0] - pv[0],
                "vy": av[1] - pv[1],
                "known": bool(known),
                "half_x": rw / 2.0 + pw / 2.0 + 3.0 + (4.5 if not known else 0.0),
                "half_y": rh / 2.0 + ph / 2.0 + 3.0 + (4.5 if not known else 0.0),
                "gun_x": max(3.0, rw / 2.0 + (2.0 if known else 3.0)),
                "gun_y": max(4.0, rh / 2.0 + (2.0 if known else 3.0)),
                "distance": math.hypot(wrapped(cx - px, W), wrapped(cy - py, H)),
            })
        return rocks

    @staticmethod
    def _contact(rock, extra_x=0.0, extra_y=0.0, horizon=90):
        first = None
        minimum = float("inf")
        for t in range(horizon + 1):
            rx = wrapped(rock["dx"] + rock["vx"] * t - extra_x, W)
            ry = wrapped(rock["dy"] + rock["vy"] * t - extra_y, H)
            clear = max(abs(rx) - rock["half_x"], abs(ry) - rock["half_y"])
            minimum = min(minimum, clear)
            if first is None and abs(rx) <= rock["half_x"] and abs(ry) <= rock["half_y"]:
                first = t
        return first, minimum

    def _derive(self, obs, player, center, velocity):
        rocks = self._make_rocks(obs, center, velocity)
        contacts = {}
        ttc = None
        near = None
        current_clear = float("inf")
        for rock in rocks:
            hit, clear = self._contact(rock)
            contacts[rock["key"]] = hit
            current_clear = min(current_clear, clear if hit is not None else max(-2.0, clear))
            if hit is not None and (ttc is None or (hit, -rock["value"]) < (ttc, -near["value"])):
                ttc, near = hit, rock
        if not rocks:
            current_clear = 80.0
        if ttc is None:
            danger = "clear"
        elif ttc <= 2 or current_clear <= 0.0:
            danger = "critical"
        elif ttc <= 7 or current_clear <= 3.0:
            danger = "immediate"
        elif ttc <= 24 or current_clear <= 10.0:
            danger = "watch"
        else:
            danger = "distant"
        missile_count = sum(1 for item in sequence(obs.get("player_missiles")) if isinstance(item, dict))
        speed = math.hypot(*velocity) if velocity is not None else None
        return {
            "present": center is not None,
            "x": center[0] if center is not None else None,
            "y": center[1] if center is not None else None,
            "pw": center[2] if center is not None else 6.0,
            "ph": center[3] if center is not None else 10.0,
            "velocity": velocity,
            "speed": speed,
            "heading": integer(player.get("orientation_code"), 0) % 16 if center is not None else None,
            "rocks": rocks,
            "contacts": contacts,
            "ttc": ttc,
            "near": near,
            "clearance": current_clear,
            "danger": danger,
            "missiles": missile_count,
            "discontinuity": bool(player.get("motion_discontinuity")) if center is not None else True,
        }

    def _observe_missiles(self):
        facts = self.facts
        if not facts.get("present"):
            self.missiles.clear()
            return
        self.recent_presses = [p for p in self.recent_presses if self.frame - p["frame"] <= 8]
        seen = set()
        for raw0 in sequence(self.obs.get("player_missiles")):
            raw = mapping(raw0)
            x, y = number(raw.get("x")), number(raw.get("y"))
            if x is None or y is None:
                continue
            slot = integer(raw.get("source_slot"), -1)
            seen.add(slot)
            old = self.missiles.get(slot)
            continuous = bool(old and 0 < self.frame - old["frame"] <= 3
                              and abs(x - old["x"]) <= 12 and abs(y - old["y"]) <= 15)
            if not continuous:
                press = None
                for candidate in reversed(self.recent_presses):
                    age = self.frame - candidate["frame"]
                    if 0 < age <= 5 and math.hypot(x - (facts.get("x") or x), y - (facts.get("y") or y)) < 35:
                        press = candidate
                        break
                if press is None:
                    self.missiles.pop(slot, None)
                    continue
                self.missiles[slot] = {
                    "frame": self.frame, "x": x, "y": y,
                    "start_frame": self.frame, "start_x": x, "start_y": y,
                    "heading": press["heading"], "learned": False,
                }
                self.shots_seen += 1
                continue
            old["frame"], old["x"], old["y"] = self.frame, x, y
            age = self.frame - old["start_frame"]
            if age < 5 or old.get("learned"):
                continue
            old["learned"] = True
            vx = (x - old["start_x"]) / float(age)
            vy = (y - old["start_y"]) / float(age)
            speed = math.hypot(vx, vy)
            hx, hy = heading_vector(old["heading"])
            dot = vx * hx + vy * hy
            cross = abs(vx * hy - vy * hx)
            if not (0.5 <= speed <= 6.5 and dot > 0.15 and cross < speed * 0.8):
                self.calibration_rejects += 1
                continue
            code = old["heading"] % 16
            entry = self.ballistics.get(code)
            if entry is None:
                self.ballistics[code] = {"vx": vx, "vy": vy, "n": 1.0, "error": 0.0}
            else:
                residual = math.hypot(vx - entry["vx"], vy - entry["vy"])
                if entry["n"] >= 4 and residual > 1.4:
                    self.calibration_rejects += 1
                    continue
                weight = 1.0 / min(10.0, entry["n"] + 1.0)
                entry["vx"] += weight * (vx - entry["vx"])
                entry["vy"] += weight * (vy - entry["vy"])
                entry["error"] = 0.8 * entry["error"] + 0.2 * residual
                entry["n"] += 1.0
            entry = self.ballistics[code]
            self.rank_frame = -100000
            if int(entry["n"]) in (1, 4, 12, 32):
                self.log("projectile_model", frame=self.frame, heading=code,
                         vx=round(entry["vx"], 3), vy=round(entry["vy"], 3),
                         samples=int(entry["n"]), error=round(entry["error"], 3))
        for slot in list(self.missiles):
            if slot not in seen:
                del self.missiles[slot]

    def _feedback(self):
        current = self.score
        delta = current - self.last_score
        if delta > 0.0:
            self.reward_events += 1
            candidates = []
            for shot in self.shot_history:
                age = self.frame - shot["frame"]
                if 0 <= age <= 52 and not shot.get("credited"):
                    weight = math.exp(-float(age) / 22.0)
                    candidates.append((weight, shot))
            if candidates:
                weight, chosen = max(candidates, key=lambda item: item[0])
                code = int(chosen["heading"]) % 16
                self.heading_reward[code] += delta * weight
                self.heading_uses[code] += weight
                chosen["credited"] = True
                self.reward_heading = code
                self.reward_lock_until = self.frame + (34 if delta >= 50 else 26)
                self.reward_lock_events += 1
                self.rank_frame = -100000
                self.last_reward_frame = self.frame
                self.log("reward_bearing", frame=self.frame, delta=round(delta, 1),
                         heading=code, age=self.frame - chosen["frame"],
                         lock_until=self.reward_lock_until)
        self.last_score = current
        self.shot_history = [s for s in self.shot_history if self.frame - s["frame"] <= 80]
        if self.frame - self.last_decay >= 180:
            for i in range(16):
                self.heading_reward[i] *= 0.94
            self.last_decay = self.frame
        if self.frame > self.reward_lock_until:
            self.reward_heading = None

    def update(self, raw_observation):
        obs = raw_observation if isinstance(raw_observation, dict) else {}
        old_frame = self.frame
        self.obs = obs
        self.frame = integer(obs.get("frame"), self.frame + 1)
        self.score = number(obs.get("score"), self.score) or 0.0
        player = mapping(obs.get("player"))
        center = self._player_center(player)
        finite = None
        if center is not None and self.previous_center is not None:
            dt = max(1, self.frame - old_frame)
            dx = wrapped(center[0] - self.previous_center[0], W) / dt
            dy = wrapped(center[1] - self.previous_center[1], H) / dt
            if abs(dx) <= 25.0 and abs(dy) <= 30.0:
                finite = (dx, dy)
        discontinuity = center is None or bool(player.get("motion_discontinuity"))
        if finite is None and center is not None and self.previous_center is not None:
            discontinuity = True
        velocity = self._read_velocity(player, finite, discontinuity)

        new_lives = integer(obs.get("lives"), self.lives)
        changed = self.old_lives is not None and new_lives != self.old_lives
        if changed:
            if new_lives < self.old_lives:
                self.log("ship_loss", frame=self.frame, old_lives=self.old_lives, lives=new_lives)
            else:
                self.log("ship_gain", frame=self.frame, old_lives=self.old_lives, lives=new_lives)
            self.epoch += 1
            self.missiles.clear()
            self.recent_presses.clear()
            self.escape = None
            self.reward_heading = None
            self.reward_lock_until = -1
            self.shot_history.clear()
            self.selected = None
            self.opportunities = []
            self.rank_frame = -100000
            self.defense_epoch_count = 0
            self.recovery_until = self.frame + 20
            self.next_fire = max(self.next_fire, self.frame + 8)
        self.lives = new_lives
        self.old_lives = new_lives
        self.player = player
        self.center = center
        self.player_velocity = velocity
        self.facts = self._derive(obs, player, center, velocity)
        if discontinuity:
            self.escape = None
            self.missiles.clear()
            self.reward_heading = None
            self.reward_lock_until = -1
            self.shot_history.clear()
            self.selected = None
            self.opportunities = []
            self.rank_frame = -100000
        self._feedback()
        self._observe_missiles()
        self.previous_center = (center[0], center[1]) if center is not None else None
        self._rank_shots()

    # --------------------------- projectile targeting -----------------------

    def _bullet_velocity(self, code):
        entry = self.ballistics.get(int(code) % 16)
        if entry is not None and entry.get("n", 0) >= 2:
            return entry["vx"], entry["vy"], True
        # A deliberately broad provisional prior; it is replaced by public
        # missile measurements before it can be called calibrated.
        hx, hy = heading_vector(code)
        return hx * 2.35, hy * 3.0, False

    def _shot_solution(self, rock, code):
        current = self.facts.get("heading")
        if current is None:
            return None
        delay = min(24, 3 * turn_distance(current, code))
        bvx, bvy, calibrated = self._bullet_velocity(code)
        pv = self.facts.get("velocity") or (0.0, 0.0)
        rel_bvx, rel_bvy = bvx - pv[0], bvy - pv[1]
        hx, hy = heading_vector(code)
        muzzle_x, muzzle_y = hx * 4.0, hy * 5.0
        # Solve closest approach in the rock's conservative normalized box,
        # then check a few neighboring integer frames.  This is both faster and
        # less quantization-sensitive than scanning every projectile frame.
        p0x = wrapped(rock["dx"] + rock["vx"] * delay - muzzle_x, W)
        p0y = wrapped(rock["dy"] + rock["vy"] * delay - muzzle_y, H)
        qx, qy = rock["vx"] - rel_bvx, rock["vy"] - rel_bvy
        gx, gy = rock["gun_x"], rock["gun_y"]
        aa = (qx / gx) ** 2 + (qy / gy) ** 2
        bb = (p0x * qx / (gx * gx)) + (p0y * qy / (gy * gy))
        estimate = 1.0 if aa < 1.0e-8 else max(1.0, min(45.0, -bb / aa))
        flights = {1, 45, int(math.floor(estimate)), int(math.ceil(estimate))}
        flights.update({int(math.floor(estimate)) - 1, int(math.ceil(estimate)) + 1})
        best_miss = float("inf")
        best_flight = None
        for flight in sorted(f for f in flights if 1 <= f <= 45):
            tx = wrapped(rock["dx"] + rock["vx"] * (delay + flight), W)
            ty = wrapped(rock["dy"] + rock["vy"] * (delay + flight), H)
            bx = muzzle_x + rel_bvx * flight
            by = muzzle_y + rel_bvy * flight
            # Missiles are not assumed to wrap; reject a candidate after its
            # ray has clearly left the known display rather than inventing a
            # missile wrap period.
            if abs(bx) > W * 0.72 or abs(by) > H * 0.72:
                continue
            dx = wrapped(tx - bx, W)
            dy = wrapped(ty - by, H)
            miss = max(abs(dx) / gx, abs(dy) / gy)
            if miss < best_miss:
                best_miss, best_flight = miss, flight
        if best_flight is None or best_miss > 3.0:
            return None
        arrival = delay + best_flight
        contact = self.facts.get("contacts", {}).get(rock["key"])
        clean = bool(contact is not None and arrival + 3 < contact and best_miss <= 1.15 and rock["known"])
        quality = best_miss * 13.0 + delay * 0.28 + best_flight * 0.08
        return {
            "rock": rock, "code": int(code) % 16, "delay": delay,
            "flight": best_flight, "arrival": arrival, "miss": best_miss,
            "calibrated": calibrated, "contact": contact, "clean": clean,
            "quality": quality,
        }

    def _heading_bonus(self, code):
        i = int(code) % 16
        return self.heading_reward[i] / max(1.0, self.heading_uses[i])

    def _rank_shots(self):
        # Target/intercept ranking is a short-lived cache.  Collision danger is
        # still derived from the newest observation every frame; recomputing the
        # expensive 16-heading weapon table every fourth frame leaves enough
        # native-time margin for the real-time scheduler.
        if self.frame - self.rank_frame < 4:
            return
        self.rank_frame = self.frame
        self.opportunities = []
        self.selected = None
        if not self.facts.get("present"):
            return
        focus = self.focus if self.frame <= self.focus_until else self.plan.get("focus", "geometry")
        focus = {"reward_lock": "reward", "small_first": "small",
                 "intercept": "intercept", "sweep": "geometry"}.get(focus, focus)
        all_solutions = []
        for rock in self.facts.get("rocks", []):
            best = None
            for code in range(16):
                candidate = self._shot_solution(rock, code)
                if candidate is None:
                    continue
                if best is None or candidate["quality"] < best["quality"]:
                    best = candidate
            if best is None:
                continue
            reward = self._heading_bonus(best["code"])
            value = float(rock["value"])
            urgency = 46.0 if best["clean"] else 16.0 if best["contact"] is not None and best["contact"] <= 35 else 0.0
            if focus == "small":
                urgency += value * 5.0
            elif focus == "reward":
                urgency += reward * 1.8
            elif focus == "intercept":
                urgency += (8.0 if best["clean"] else 0.0)
            priority = urgency + value * 3.0 + reward * 1.2 - best["quality"]
            all_solutions.append(dict(best, priority=priority))
        all_solutions.sort(key=lambda item: item["priority"], reverse=True)
        self.opportunities = all_solutions[:6]
        if all_solutions:
            clean_near = next((s for s in all_solutions if s["clean"] and (s["contact"] is None or s["contact"] <= 50)), None)
            self.selected = clean_near or all_solutions[0]

    # ------------------------------- safety ---------------------------------

    def _swept_heading(self, code, pulse=1, horizon=30):
        hx, hy = heading_vector(code)
        minimum = float("inf")
        first = None
        total = 0.0
        for rock in self.facts.get("rocks", []):
            rock_min = float("inf")
            hit = None
            for t in range(horizon + 1):
                active = min(t, pulse)
                # Relative rock velocities already subtract current ship drift.
                thrust_x = 0.035 * hx * active * (active + 1.0) / 2.0
                thrust_y = 0.035 * hy * active * (active + 1.0) / 2.0
                dx = wrapped(rock["dx"] + rock["vx"] * t - thrust_x, W)
                dy = wrapped(rock["dy"] + rock["vy"] * t - thrust_y, H)
                clear = math.hypot(max(0.0, abs(dx) - rock["half_x"]),
                                   max(0.0, abs(dy) - rock["half_y"]))
                rock_min = min(rock_min, clear)
                if hit is None and abs(dx) <= rock["half_x"] and abs(dy) <= rock["half_y"]:
                    hit = t
            minimum = min(minimum, rock_min)
            total += rock_min
            if hit is not None and (first is None or hit < first):
                first = hit
        if not self.facts.get("rocks"):
            minimum, total = 80.0, 80.0
        return {"code": int(code) % 16, "clearance": minimum, "contact": first,
                "score": minimum + 0.05 * total}

    def _escape_choice(self):
        if not self.facts.get("present"):
            return None
        current_code = self.facts.get("heading", 0)
        baseline = self._swept_heading(current_code, pulse=0)
        choices = [self._swept_heading(code, pulse=1) for code in range(16)]
        choices.sort(key=lambda item: (item["score"], item["clearance"],
                                       -turn_distance(current_code, item["code"])), reverse=True)
        best = choices[0] if choices else None
        if best is None:
            return None
        improvement = best["score"] - baseline["score"]
        danger = self.facts.get("danger")
        threshold = 2.5 if danger == "watch" else 1.0
        if danger in {"critical", "immediate"} and improvement >= threshold:
            return best
        if danger == "watch" and self.facts.get("ttc") is not None and self.facts["ttc"] <= 20 and improvement >= threshold:
            return best
        return None

    def _begin_escape(self):
        if self.escape is not None or self.frame < self.next_escape:
            return
        danger = self.facts.get("danger")
        if danger not in {"watch", "immediate", "critical"}:
            return
        choice = self._escape_choice()
        if choice is None:
            return
        self.escape = {"code": choice["code"], "remaining": 1,
                       "created": self.frame, "expires": self.frame + 18,
                       "score": choice["score"], "clearance": choice["clearance"]}
        self.next_escape = self.frame + (8 if danger == "critical" else 18)
        self.escape_starts += 1
        self.log("escape_begin", frame=self.frame, heading=choice["code"],
                 danger=danger, ttc=self.facts.get("ttc"),
                 improvement=round(choice["score"] - self._swept_heading(self.facts.get("heading", 0), 0)["score"], 2))

    def _escape_action(self):
        move = self.escape
        if not isinstance(move, dict) or self.frame > integer(move.get("expires"), self.frame):
            self.escape = None
            return None
        if self.facts.get("danger") not in {"watch", "immediate", "critical"}:
            self.escape = None
            return None
        current = self.facts.get("heading")
        desired = integer(move.get("code"), current or 0) % 16
        # Revalidate the actual current scene before each thrust pulse.
        check = self._swept_heading(desired, pulse=1)
        baseline = self._swept_heading(current, pulse=0) if current is not None else check
        if check["score"] + 0.6 < baseline["score"] and self.facts.get("danger") != "critical":
            self.escape = None
            return None
        delta = turn_delta(current, desired)
        if abs(delta) > 1:
            return self._compose(turn_sign(current, desired), False, False)
        if integer(move.get("remaining"), 0) > 0:
            move["remaining"] = 0
            self.escape_pulses += 1
            self.log("escape_pulse", frame=self.frame, heading=desired,
                     alignment=delta, clearance=round(check["clearance"], 2))
            return self._compose(0, True, False)
        self.escape = None
        return self.noop

    def _defense_needed(self):
        if not self.facts.get("present") or self.frame - self.last_defense < 18:
            return False
        if self.defense_epoch_count >= 2:
            return False
        ttc = self.facts.get("ttc")
        if ttc is None or ttc > 3:
            return False
        choice = self._escape_choice()
        if choice is not None and choice["score"] > self._swept_heading(self.facts.get("heading", 0), 0)["score"] + 1.0:
            return False
        return True

    # ------------------------------- turret ---------------------------------

    def _compose(self, turn=0, thrust=False, fire=False):
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

    def _aim_code(self):
        danger = self.facts.get("danger")
        selected = self.selected
        # A measured score-bearing dwell is only an offensive preference.  A
        # current clean threat and all safety states take precedence.
        if (self.reward_heading is not None and self.frame <= self.reward_lock_until
                and danger not in {"critical", "immediate"}
                and self.plan.get("lock") != "ignore"):
            if not (selected and selected.get("clean") and selected.get("contact") is not None and selected["contact"] <= 35):
                return int(self.reward_heading) % 16
        if selected is not None:
            return int(selected["code"]) % 16
        if self.frame >= self.next_sweep:
            self.sweep_sign *= -1
            self.next_sweep = self.frame + 48
        current = self.facts.get("heading", 0)
        return (int(current) + (1 if self.sweep_sign > 0 else -1)) % 16

    def _fire_ready(self):
        if self.frame < self.next_fire or self.frame < self.recovery_until:
            return False
        if self.facts.get("missiles", 0) >= 2:
            return False
        if "FIRE" in str(self.last_action):
            return False
        return True

    def _should_fire(self, desired):
        if not self._fire_ready():
            return False
        current = self.facts.get("heading")
        aligned = current is not None and turn_distance(current, desired) == 0
        # Find a current-nose intersection, which is safer than relying on the
        # selected record after source-slot reuse.
        current_hit = False
        if current is not None:
            for rock in self.facts.get("rocks", []):
                solution = self._shot_solution(rock, current)
                if solution is not None and solution["miss"] <= 1.45:
                    current_hit = True
                    break
        if aligned or current_hit:
            return True
        # Never wait forever for approximate geometry to become exact.
        return self.frame - self.last_shot_frame >= 11

    def choose(self):
        facts = self.facts
        if not facts.get("present"):
            self.escape = None
            self.last_action = self.noop
            return self.noop

        if self._defense_needed():
            self.defense_epoch_count += 1
            self.defenses += 1
            self.last_defense = self.frame
            self.recovery_until = self.frame + 20
            self.escape = None
            self.log("defense", frame=self.frame, count=self.defense_epoch_count,
                     ttc=facts.get("ttc"), clearance=round(number(facts.get("clearance"), 0.0) or 0.0, 2))
            self.last_action = self.legal("DOWN")
            return self.last_action

        if facts.get("danger") in {"watch", "immediate", "critical"}:
            self._begin_escape()
        if self.escape is not None:
            action = self._escape_action()
            if action is not None:
                self.last_action = action
                return action

        desired = self._aim_code()
        current = facts.get("heading")
        sign = turn_sign(current, desired) if current is not None else 0
        fire = False if self.frame < self.recovery_until else self._should_fire(desired)
        action = self._compose(sign, False, fire)
        if fire:
            self.next_fire = self.frame + (6 if self.plan.get("fire") == "spaced" else 5)
            code = integer(current, desired) % 16
            self.shots_posted += 1
            self.last_shot_frame = self.frame
            self.heading_uses[code] += 1.0
            self.recent_presses.append({"frame": self.frame, "heading": code})
            self.shot_history.append({"frame": self.frame, "heading": code, "credited": False})
            if self.frame - self.last_log >= 90:
                selected = self.selected
                self.log("shot", frame=self.frame, heading=code,
                         desired=desired, target=selected["rock"]["key"] if selected else None,
                         arrival=round(selected["arrival"], 1) if selected else None,
                         miss=round(selected["miss"], 2) if selected else None,
                         clean=bool(selected and selected["clean"]),
                         reward_lock=self.reward_heading if self.frame <= self.reward_lock_until else None,
                         calibrated=sum(1 for e in self.ballistics.values() if e.get("n", 0) >= 3))
                self.last_log = self.frame
        elif action in {"UP", "UPLEFT", "UPRIGHT"}:
            self.log("thrust", frame=self.frame, action=action)
        self.last_action = action
        return action

    # ----------------------------- model views -------------------------------

    @staticmethod
    def bearing(dx, dy):
        return ("right" if dx > 10 else "left" if dx < -10 else "center") + "_" + ("below" if dy > 10 else "above" if dy < -10 else "level")

    def jev_state(self):
        facts = self.facts
        opportunities = []
        for item in self.opportunities[:4]:
            rock = item["rock"]
            opportunities.append({
                "size": rock["size"],
                "bearing": self.bearing(rock["dx"], rock["dy"]),
                "motion": "known" if rock["known"] else "uncertain",
                "trajectory": "precontact" if item["clean"] else "crossing" if item["contact"] is not None else "noncrossing",
                "intercept": "clean" if item["miss"] <= 1.0 else "marginal",
                "turn": "aligned" if item["delay"] == 0 else "short" if item["delay"] <= 9 else "long",
            })
        best_rates = sorted((self._heading_bonus(i), i) for i in range(16))
        return {
            "rules": {
                "fire": "press then release; held FIRE is not rapid fire",
                "noop": "coasts and does not brake",
                "down": "mode-0 hyperspace, not reverse thrust",
                "heading": "LEFT increments and RIGHT decrements orientation code",
                "coverage": "enemy craft/projectiles unknown; null is not absence",
            },
            "snapshot": {
                "frame": int(self.frame), "lives": int(self.lives),
                "score": int(max(0.0, self.score)),
                "player": "present" if facts.get("present") else "absent",
                "danger": facts.get("danger", "unknown"),
                "impact": bucket(facts.get("ttc"), (3, 8, 25), ("none", "immediate", "soon", "later")),
                "motion": "unknown" if facts.get("speed") is None else "moving" if facts["speed"] > 0.2 else "still",
                "rocks": min(15, len(facts.get("rocks", []))),
                "missiles": min(2, int(facts.get("missiles", 0))),
            },
            "opportunities": opportunities,
            "learning": {
                "calibration": "established" if sum(e.get("n", 0) >= 3 for e in self.ballistics.values()) >= 8 else "learning",
                "recent_score": "recent" if self.frame - self.last_reward_frame < 45 else "quiet",
                "reward_bearing": "active" if self.reward_heading is not None and self.frame <= self.reward_lock_until else "none",
                "top_bearing_quality": bucket(best_rates[-1][0] if best_rates else None, (0.5, 2.0, 6.0), ("unknown", "weak", "useful", "strong")),
            },
            "candidates": ["reward_lock", "intercept", "small_first", "sweep"],
            "local_override": "code owns safety, durations, target arithmetic and legal actions",
            "plan": dict(self.plan),
        }

    def jev_questions(self):
        labels = {
            "reward_lock": "Keep a short dwell on the most recent score-producing bearing; code revalidates danger and expires it.",
            "intercept": "Follow the code-ranked moving-rock intercept, preferring a supplied clean pre-contact opportunity.",
            "small_first": "Prefer supplied small valuable rocks when no immediate crossing is present; do not calculate geometry.",
            "sweep": "Use a bounded left/right aiming sweep and release between shots; do not thrust.",
        }
        order = ["reward_lock", "intercept", "small_first", "sweep"]
        if (self.frame // 36) % 2:
            order.reverse()
        risk_labels = {
            "clear": "The supplied local map has no near contact; offense is allowed.",
            "watch": "A later crossing is supplied; only a bounded rechecked escape may move the ship.",
            "immediate": "A near crossing needs local escape or defense before offensive preference.",
            "critical": "Contact is imminent; survival override is mandatory.",
            "unknown": "The player/geometry is unavailable; release controls.",
        }
        risk_order = ["clear", "watch", "immediate", "critical", "unknown"]
        if (self.frame // 48) % 2:
            risk_order.reverse()
        return {
            "focus_a": Choice(
                "Choose one supplied offensive preference for this exact state. Do not invent an action, duration, target id, or arithmetic. Local safety always wins.",
                {key: labels[key] for key in order},
            ),
            "focus_b": Choice(
                "Independently choose the useful offensive preference. A recent score bearing is evidence, not a permanent target; a clean supplied crossing outranks it.",
                {key: labels[key] for key in reversed(order)},
            ),
            "risk": Choice(
                "Classify the already-derived snapshot.danger. Unknown enemy coverage is not absence and this answer cannot authorize a defense.",
                {key: risk_labels[key] for key in risk_order},
            ),
            "lock": Noul(
                "Is the supplied recent score-bearing evidence strong enough to retain a short aim dwell when the local map is clear? Do not override a near collision.",
                {"true": "Retain the bounded reward-bearing preference.", "false": "Use the geometric fallback."},
            ),
        }

    def accept_jev(self, answers, origin_frame, origin_epoch):
        if not isinstance(answers, dict):
            return
        age = self.frame - origin_frame
        if origin_epoch != self.epoch or age < 0 or age > 48:
            self.log("jev_discard", frame=self.frame, judged_frame=origin_frame, reason="stale_or_life")
            return
        a, b = answer_value(answers.get("focus_a")), answer_value(answers.get("focus_b"))
        allowed = {"reward_lock", "intercept", "small_first", "sweep"}
        if a == b and a in allowed:
            confidence = min(answer_probability(answers.get("focus_a"), a), answer_probability(answers.get("focus_b"), b))
            if confidence >= 0.40:
                self.focus = a
                self.focus_until = self.frame + 60
        risk = answer_value(answers.get("risk"))
        if risk not in {"clear", "watch", "immediate", "critical", "unknown"}:
            risk = "unknown"
        lock_prob = yes_probability(answers.get("lock"))
        if lock_prob >= 0.82 and self.reward_heading is not None:
            self.plan["lock"] = "short"
        elif lock_prob <= 0.25:
            self.plan["lock"] = "ignore"
        self.log("jev_preference", frame=self.frame, judged_frame=origin_frame,
                 focus=self.focus if a == b and a in allowed else "fallback",
                 risk=risk, lock=round(lock_prob, 2))

    def summary(self, reason):
        return {
            "reason": reason, "frame": self.frame, "score": int(max(0.0, self.score)),
            "lives": int(self.lives), "epoch": self.epoch,
            "shots_posted": self.shots_posted, "shots_seen": self.shots_seen,
            "calibration_rejects": self.calibration_rejects,
            "calibrated_headings": sum(1 for e in self.ballistics.values() if e.get("n", 0) >= 3),
            "reward_events": self.reward_events, "reward_locks": self.reward_lock_events,
            "escape_starts": self.escape_starts, "escape_pulses": self.escape_pulses,
            "defenses": self.defenses, "plan": dict(self.plan),
            "bearing_rates": [round(self._heading_bonus(i), 3) for i in range(16)],
        }


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _start_jev(self, controller):
        if getattr(self.ctx, "system1", True) is None:
            return None
        state = controller.jev_state()
        questions = controller.jev_questions()
        frame, epoch = controller.frame, controller.epoch

        def request():
            return self.ctx.decide(state, questions, timeout_s=2.45)

        try:
            return self.ctx.spawn(request), frame, epoch
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
        if integer(mapping(remaining).get("system2_calls"), 0) <= 0:
            return None
        evidence = controller.summary(reason)
        evidence["danger"] = controller.facts.get("danger", "unknown")
        evidence["ttc"] = controller.facts.get("ttc")
        evidence["calibration"] = {
            str(k): {"vx": round(v["vx"], 2), "vy": round(v["vy"], 2), "n": int(v["n"])}
            for k, v in controller.ballistics.items()
        }
        prompt = (
            "Review one real-time Atari 2600 Asteroids mode-0/difficulty-0 controller. "
            "Code owns every native frame, collision arithmetic, projectile calibration, "
            "score bookkeeping, FIRE release, and emergency defense. Its new experiment "
            "attributes actual score deltas to recent FIRE headings and briefly dwells on a "
            "successful bearing; this is only a preference and is invalidated by danger. "
            "Return exactly one small JSON object, no prose and no actions. Allowed fields: "
            "focus='geometry'|'reward'|'small'|'intercept', lock='short'|'ignore', "
            "fire='normal'|'spaced'. Choose from aggregate evidence, not a seed or a "
            "memorized schedule. Unknown enemy objects are not absent.\nEVIDENCE="
            + json.dumps(evidence, separators=(",", ":"), allow_nan=False)
        )

        def request():
            return self.ctx.think(prompt, timeout_s=125.0)

        try:
            return self.ctx.spawn(request)
        except (BudgetExceeded, EpisodeOver):
            return None
        except ModelUnavailable:
            raise
        except Exception:
            return None

    @staticmethod
    def _parse_astra(text):
        data = json_object(text)
        if not isinstance(data, dict):
            return {}
        allowed = {
            "focus": {"geometry", "reward", "small", "intercept"},
            "lock": {"short", "ignore"},
            "fire": {"normal", "spaced"},
        }
        return {key: data[key] for key, values in allowed.items()
                if isinstance(data.get(key), str) and data[key] in values}

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(game, "tick_s", None) is None:
            return
        controller = RewardTurret(game, ctx)
        controller.update(observation(game))

        def post(action):
            try:
                ctx.act(controller.legal(action))
                return True
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return False
            except Exception:
                try:
                    ctx.act(controller.noop)
                    return True
                except Exception:
                    return False

        if not post(controller.choose()):
            return

        s1_enabled = getattr(ctx, "system1", True) is not None
        jev_future = None
        jev_frame = -100000
        jev_epoch = -1
        next_jev = controller.frame

        astra_future = None
        astra_started = False
        astra_due = controller.frame + 360

        while not ctx.done and not getattr(game, "done", False):
            if s1_enabled and jev_future is None and controller.frame >= next_jev:
                try:
                    started = self._start_jev(controller)
                    if started is not None:
                        jev_future, jev_frame, jev_epoch = started
                        next_jev = controller.frame + 36
                    else:
                        next_jev = controller.frame + 48
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except Exception:
                    next_jev = controller.frame + 48

            if not astra_started and astra_future is None and controller.frame >= astra_due:
                astra_started = True
                try:
                    astra_future = self._start_astra(controller, "first stable score review")
                except ModelUnavailable:
                    return
                except (BudgetExceeded, EpisodeOver):
                    astra_future = None
                except Exception:
                    astra_future = None

            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            if ctx.done or getattr(game, "done", False):
                break

            controller.update(observation(game))

            if jev_future is not None and jev_future.done():
                finished = jev_future
                jev_future = None
                try:
                    controller.accept_jev(finished.result(), jev_frame, jev_epoch)
                except (BudgetExceeded, EpisodeOver):
                    s1_enabled = False
                except ModelUnavailable:
                    return
                except Exception:
                    pass

            if astra_future is not None and astra_future.done():
                finished = astra_future
                astra_future = None
                try:
                    result = finished.result()
                    plan = self._parse_astra(getattr(result, "text", result))
                    if plan:
                        controller.plan.update(plan)
                        controller.log("astra_plan", frame=controller.frame, plan=dict(controller.plan))
                except (BudgetExceeded, EpisodeOver):
                    pass
                except ModelUnavailable:
                    return
                except Exception:
                    pass

            if not post(controller.choose()):
                return

        controller.log("reward_turret_summary", **controller.summary("episode complete"))


def build(ctx):
    return Harness(ctx)
