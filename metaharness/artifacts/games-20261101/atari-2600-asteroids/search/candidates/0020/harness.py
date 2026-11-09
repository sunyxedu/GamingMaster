"""Phase-shifted turret controller for real-time Atari 2600 Asteroids.

The controller deliberately does not wait for an approximate collision box to
become ``immediate``.  Asteroids in this ROM are usually moving on near
vertical tracks, while the RAM object list can flicker and reuse slots.  A
stationary ship can therefore be killed by a rock which was absent from the
last useful snapshot.  This harness keeps a low-duty, feedback-controlled
horizontal patrol running underneath a pulsed firing/aiming policy.  It is a
phase shift, not an open-loop orbit: every thrust pulse is one native frame,
its result is re-observed, and the patrol is stopped or redirected by a new
risk map.

Astra and Jev are advisory.  The local controller owns geometry, the emulator
clock, fire release, defense limits, and every action posted to the game.
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


def finite(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def maybe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def mapping(value):
    return value if isinstance(value, dict) else {}


def sequence(value):
    return value if isinstance(value, list) else []


def safe_observation(game):
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


def wrap_delta(value, period):
    value = finite(value)
    while value > period / 2.0:
        value -= period
    while value < -period / 2.0:
        value += period
    return value


def clamp(value, low, high):
    return max(low, min(high, value))


def velocity(value):
    if isinstance(value, list) and len(value) >= 2:
        x = maybe_float(value[0])
        y = maybe_float(value[1])
        if x is not None and y is not None:
            return x, y
    return None, None


def answer_value(answer):
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def answer_probability(answer, key):
    if isinstance(answer, dict):
        values = answer.get("probabilities", {})
    else:
        values = getattr(answer, "probabilities", {})
    if not isinstance(values, dict):
        return 0.0
    return clamp(finite(values.get(key), 0.0), 0.0, 1.0)


def yes_probability(answer):
    value = answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    if isinstance(value, (int, float)):
        return clamp(finite(value), 0.0, 1.0)
    return answer_probability(answer, "true")


def extract_json_object(value):
    """Extract the first JSON object from arbitrary Astra/stand-in text."""
    if not isinstance(value, str):
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char != "{":
            continue
        try:
            result, _ = decoder.raw_decode(value[index:])
        except (TypeError, ValueError):
            continue
        if isinstance(result, dict):
            return result
    return None


def heading_vector(code):
    """The documented code convention: 0 up, 4 left, 8 down, 12 right."""
    angle = (integer(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def heading_for(dx, dy):
    if abs(dx) + abs(dy) < 1e-8:
        return 0
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def turn_sign(current, desired):
    delta = (integer(desired) - integer(current)) % 16
    if delta == 0:
        return 0
    if delta < 8:
        return 1  # LEFT increments orientation code.
    if delta > 8:
        return -1  # RIGHT decrements orientation code.
    return 1


def bucket(value, cuts, names):
    if value is None:
        return names[0]
    for cut, name in zip(cuts, names[1:]):
        if value < cut:
            return name
    return names[-1]


def time_label(value):
    if value is None:
        return "none"
    if value <= 4:
        return "immediate"
    if value <= 14:
        return "soon"
    if value <= 32:
        return "later"
    return "distant"


def clearance_label(value):
    if value is None:
        return "unknown"
    if value < 0:
        return "overlap"
    if value < 6:
        return "tight"
    if value < 18:
        return "narrow"
    if value < 40:
        return "open"
    return "wide"


class PhaseController:
    """A frame-local patrol, risk map, and pulsed turret."""

    def __init__(self, game, ctx):
        self.game = game
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        advertised = getattr(spec, "names", DEFAULT_ACTIONS) if spec is not None else DEFAULT_ACTIONS
        try:
            self.names = set(advertised or DEFAULT_ACTIONS)
        except Exception:
            self.names = set(DEFAULT_ACTIONS)
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else next(iter(self.names), "NOOP")

        self.frame = 0
        self.score = 0.0
        self.lives = 0
        self.previous_lives = None
        self.life_epoch = 0
        self.obs = {}
        self.facts = {}

        # Short-lived slot tracks are used only to bridge decoder flicker.  A
        # slot is accepted again only after a continuity and size check; it is
        # never treated as a permanent physical identity.
        self.tracks = {}
        self.previous_player = None
        self.previous_player_frame = None
        self.previous_rock_count = None
        self.transition_until = -1

        # Patrol state.  The target is a broad central waypoint, not a lane
        # schedule keyed by seed.  Pulses are budgeted per leg and revalidated.
        self.phase = 0
        self.goal_x = None
        self.goal_until = -1
        self.next_patrol = 0
        self.next_emergency_replan = 0
        self.patrol_pulses = 0
        self.patrol_leg_start = 0
        self.next_pulse = 0
        self.burst_left = 0
        self.last_pulse = -100000
        self.recover_until = -1
        self.initialized = False

        self.next_fire = 0
        self.defense_frames = []
        self.model_defense_hint = False
        self.model_mode = "patrol"
        self.model_side = "hold"
        self.model_aim = "target"
        self.model_until = -1

        self.policy = {
            "patrol": "steady",
            "aim": "target",
            "fire": "normal",
            "risk": "cautious",
            "defense": "rare",
        }
        self.last_action = self.noop
        self.last_log = -100000

    # ------------------------------ bookkeeping ---------------------------

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def legal(self, action):
        if action in self.names:
            return action
        return self.noop

    def _reset_after_life(self, old, new):
        self.life_epoch += 1
        self.tracks = {}
        self.previous_player = None
        self.previous_player_frame = None
        self.goal_x = None
        self.goal_until = -1
        self.patrol_pulses = 0
        self.patrol_leg_start = self.frame
        self.next_patrol = self.frame + 70
        self.next_emergency_replan = self.frame + 24
        self.next_pulse = self.frame + 18
        self.burst_left = 0
        self.recover_until = self.frame + 32
        self.defense_frames = []
        self.model_defense_hint = False
        self.next_fire = self.frame + 8
        self.log("life_epoch", frame=self.frame, old_lives=old, lives=new,
                 epoch=self.life_epoch, score=round(self.score, 1))

    def _player_record(self):
        p = mapping(self.obs.get("player"))
        present = bool(p.get("present")) and p.get("x") is not None and p.get("y") is not None
        if not present:
            return {
                "present": False, "x": None, "y": None, "w": 6.0, "h": 10.0,
                "heading": None, "vx": None, "vy": None, "discontinuity": True,
            }
        w = finite(p.get("width"), 6.0) or 6.0
        h = finite(p.get("height"), 10.0) or 10.0
        x = finite(p.get("x"), WIDTH / 2.0) + w / 2.0
        y = finite(p.get("y"), HEIGHT / 2.0) + h / 2.0
        vx, vy = velocity(p.get("velocity_px_per_frame"))
        # The adaptor's finite-difference value is preferred.  A one-frame
        # local estimate fills only a null value and rejects wraps/jumps.
        if (vx is None or vy is None) and self.previous_player is not None:
            gap = self.frame - integer(self.previous_player_frame, self.frame)
            dx = wrap_delta(x - self.previous_player[0], WIDTH)
            dy = wrap_delta(y - self.previous_player[1], HEIGHT)
            if 1 <= gap <= 2 and abs(dx) <= 8.0 and abs(dy) <= 8.0 and not p.get("motion_discontinuity"):
                vx = dx / gap
                vy = dy / gap
        return {
            "present": True, "x": x, "y": y, "w": w, "h": h,
            "heading": integer(p.get("orientation_code"), 0) % 16,
            "vx": vx, "vy": vy,
            "discontinuity": bool(p.get("motion_discontinuity")),
        }

    def _read_rocks(self, player):
        if not player.get("present"):
            return []
        records = []
        seen = set()
        for index, raw_value in enumerate(sequence(self.obs.get("asteroids"))):
            raw = mapping(raw_value)
            if raw.get("x") is None or raw.get("y") is None:
                continue
            w = finite(raw.get("width"), 8.0) or 8.0
            h = finite(raw.get("height"), 15.0) or 15.0
            x = finite(raw.get("x")) + w / 2.0
            y = finite(raw.get("y")) + h / 2.0
            slot = integer(raw.get("source_slot"), index)
            size = str(raw.get("size") or "unknown").lower()
            avx, avy = velocity(raw.get("velocity_px_per_frame"))
            old = self.tracks.get(slot)
            inferred = False
            if (avx is None or avy is None) and old is not None and old.get("size") == size:
                gap = self.frame - integer(old.get("frame"), self.frame)
                dx = wrap_delta(x - finite(old.get("x")), WIDTH)
                dy = wrap_delta(y - finite(old.get("y")), HEIGHT)
                # A same-size slot that jumped is probably a reused slot or a
                # split, not a continuation.  Keep it uncertain instead.
                if 1 <= gap <= 4 and abs(dx) <= 16.0 and abs(dy) <= 16.0:
                    avx, avy = dx / gap, dy / gap
                    inferred = True
            known = avx is not None and avy is not None
            if not known:
                avx, avy = 0.0, 0.0
            seen.add(slot)
            self.tracks[slot] = {
                "x": x, "y": y, "frame": self.frame, "size": size,
                "vx": avx, "vy": avy, "known": known,
            }
            records.append({
                "slot": slot, "x": x, "y": y, "w": w, "h": h,
                "size": size, "vx": avx, "vy": avy,
                "known": known, "ghost": False, "inferred": inferred,
            })

        # Keep a ghost for at most three frames.  This is deliberately shorter
        # than a full identity tracker: it bridges RGB/RAM flicker but does not
        # turn a destroyed object into a permanent threat.
        for slot, old in list(self.tracks.items()):
            if slot in seen:
                continue
            age = self.frame - integer(old.get("frame"), self.frame)
            if 1 <= age <= 3 and old.get("known"):
                size = old.get("size", "unknown")
                dims = {"small": (4.0, 8.0), "medium": (8.0, 15.0), "large": (16.0, 28.0)}
                w, h = dims.get(size, (8.0, 15.0))
                records.append({
                    "slot": slot, "x": finite(old.get("x")) + finite(old.get("vx")) * age,
                    "y": finite(old.get("y")) + finite(old.get("vy")) * age,
                    "w": w, "h": h, "size": size,
                    "vx": finite(old.get("vx")), "vy": finite(old.get("vy")),
                    "known": True, "ghost": True, "inferred": True,
                })
        return records

    # ------------------------------ risk map ------------------------------

    def _collision_metrics(self, rock, x, horizon=72):
        """Return (first contact, minimum clearance) for a stationary x.

        This is a conservative ranking device, not a collision-mask oracle.  It
        explicitly asks where a vertical track crosses the ship, which is a
        different question from choosing one of five fixed labels.
        """
        p = self.facts
        if not p.get("present"):
            return None, -999.0
        px = finite(x)
        py = finite(p.get("y"), HEIGHT / 2.0)
        pvx = finite(p.get("vx"), 0.0) if p.get("vx") is not None else 0.0
        pvy = finite(p.get("vy"), 0.0) if p.get("vy") is not None else 0.0
        xgap = finite(rock.get("w"), 8.0) / 2.0 + finite(p.get("w"), 6.0) / 2.0 + 3.5
        ygap = finite(rock.get("h"), 15.0) / 2.0 + finite(p.get("h"), 10.0) / 2.0 + 3.5
        first = None
        minimum = 999.0
        for t in range(0, horizon + 1, 2):
            rx = finite(rock.get("x")) + finite(rock.get("vx")) * t
            ry = finite(rock.get("y")) + finite(rock.get("vy")) * t
            sx = px + pvx * t
            sy = py + pvy * t
            dx = abs(wrap_delta(rx - sx, WIDTH))
            dy = abs(wrap_delta(ry - sy, HEIGHT))
            clearance = min(dx - xgap, dy - ygap)
            if clearance < minimum:
                minimum = clearance
            if dx <= xgap and dy <= ygap and first is None:
                first = t
        return first, minimum

    @staticmethod
    def _size_weight(size):
        return {"small": 3.0, "medium": 2.0, "large": 1.0}.get(size, 1.5)

    def _risk_for_x(self, x, horizon=72):
        if not self.facts.get("present"):
            return 999.0, None, -999.0, 0
        risk = 0.0
        first = None
        minimum = 999.0
        unknown_near = 0
        for rock in self.facts.get("rocks", []):
            contact, clearance = self._collision_metrics(rock, x, horizon)
            minimum = min(minimum, clearance)
            weight = self._size_weight(rock.get("size"))
            if not rock.get("known") or rock.get("ghost"):
                weight *= 1.35
                if clearance < 32.0:
                    unknown_near += 1
            if contact is not None:
                if first is None or contact < first:
                    first = contact
                # Early crossings cost more.  A ghost costs extra but only for
                # its short lifetime.
                risk += weight * (1.0 + max(0.0, (44.0 - contact) / 22.0))
            elif clearance < 18.0:
                risk += weight * (18.0 - clearance) / 18.0
        return risk, first, minimum, unknown_near

    def _candidate_xs(self):
        f = self.facts
        current = finite(f.get("x"), 80.0)
        candidates = [56.0, 76.0, 96.0, 112.0]
        # These are generated from the live field, not a seed or a memorized
        # schedule.  Only the nearest two boundaries are retained so the
        # frame-local risk map stays below the native tick deadline.
        for rock in f.get("rocks", [])[:2]:
            rx = finite(rock.get("x"), current)
            candidates.append(clamp(rx - 20.0, 42.0, 118.0))
            candidates.append(clamp(rx + 20.0, 42.0, 118.0))
        candidates.extend((clamp(current - 24.0, 42.0, 118.0),
                          clamp(current + 24.0, 42.0, 118.0)))
        unique = []
        for value in candidates:
            value = round(float(value), 1)
            if all(abs(value - other) > 2.0 for other in unique):
                unique.append(value)
        return unique

    def _choose_goal(self, force=False, emergency=False):
        if not self.facts.get("present"):
            self.goal_x = None
            return
        current = finite(self.facts.get("x"), 80.0)
        candidates = []
        for x in self._candidate_xs():
            risk, contact, clear, unknown = self._risk_for_x(x, 72 if emergency else 56)
            distance = abs(wrap_delta(x - current, WIDTH))
            # Movement is cheap only when it buys a real phase shift.  The small
            # distance term prevents chasing every flickering coordinate.
            cost = risk + distance * (0.018 if not emergency else 0.010)
            candidates.append({
                "x": x, "risk": risk, "contact": contact,
                "clearance": clear, "unknown": unknown, "cost": cost,
            })
        candidates.sort(key=lambda item: (item["cost"], abs(item["x"] - current)))
        best = candidates[0] if candidates else {"x": current, "risk": 0.0, "contact": None, "clearance": 999.0, "unknown": 0}

        # The phase waypoint is the structural intervention: when the current
        # corridor is not hot, alternate broad central sides rather than waiting
        # for a model to prove a collision.  Reject it only when the measured
        # field risk is materially worse than the best alternative.
        phase_x = (clamp(current - 24.0, 48.0, 112.0)
                   if self.phase == 0 and abs(current - 80.0) > 18.0
                   else (58.0 if self.phase % 2 == 0 else 102.0))
        phase_risk, phase_contact, phase_clear, phase_unknown = self._risk_for_x(phase_x, 72 if emergency else 56)
        phase_cost = phase_risk + abs(wrap_delta(phase_x - current, WIDTH)) * (0.018 if not emergency else 0.010)
        selected = best
        if force and not emergency and phase_cost <= best["cost"] + 2.5:
            selected = {
                "x": phase_x, "risk": phase_risk, "contact": phase_contact,
                "clearance": phase_clear, "unknown": phase_unknown, "cost": phase_cost,
            }
        self.phase = (self.phase + 1) % 2
        old = self.goal_x
        self.goal_x = float(selected["x"])
        self.goal_until = self.frame + (84 if emergency else 180)
        if emergency:
            self.burst_left = 0
        if not emergency:
            self.patrol_pulses = 0
            self.patrol_leg_start = self.frame
        self.next_emergency_replan = self.frame + (18 if emergency else 24)
        interval = {"rare": 300, "responsive": 120, "steady": 180}.get(self.policy.get("patrol"), 180)
        self.next_patrol = self.frame + interval
        self.log("phase_plan", frame=self.frame, old_goal=old, goal=self.goal_x,
                 force=force, emergency=emergency, risk=round(selected["risk"], 2),
                 contact=selected["contact"], clearance=round(selected["clearance"], 2),
                 candidates=len(candidates))

    def _update_facts(self, player, rocks):
        self.facts = {
            "frame": self.frame, "score": self.score, "lives": self.lives,
            "present": bool(player.get("present")),
            "x": player.get("x"), "y": player.get("y"),
            "w": player.get("w", 6.0), "h": player.get("h", 10.0),
            "heading": player.get("heading"), "vx": player.get("vx"), "vy": player.get("vy"),
            "speed": (math.hypot(player["vx"], player["vy"])
                      if player.get("vx") is not None and player.get("vy") is not None else None),
            "discontinuity": bool(player.get("discontinuity")),
            "rocks": rocks,
            "missiles": sum(1 for item in sequence(self.obs.get("player_missiles")) if isinstance(item, dict)),
        }
        if not player.get("present"):
            self.facts.update({"risk": 999.0, "contact": None, "clearance": -999.0,
                               "danger": "unknown", "unknown_near": True,
                               "waypoints": [], "best_x": None})
            return
        current_risk, current_contact, current_clear, unknown = self._risk_for_x(player["x"], 60)
        if current_contact is None:
            if current_clear < 5.0 or unknown:
                danger = "uncertain"
            elif current_risk >= 5.0 or current_clear < 18.0:
                danger = "watch"
            else:
                danger = "clear"
        elif current_contact <= 3:
            danger = "immediate"
        elif current_contact <= 10:
            danger = "hot"
        elif current_contact <= 32:
            danger = "watch"
        else:
            danger = "distant"
        waypoints = []
        for x in self._candidate_xs():
            risk, contact, clear, unknown_count = self._risk_for_x(x, 48)
            waypoints.append({
                "id": "X" + str(int(round(x))),
                "x": x, "risk": risk, "contact": contact,
                "clearance": clear, "unknown": unknown_count,
            })
        waypoints.sort(key=lambda item: (item["risk"] + abs(item["x"] - player["x"]) * 0.02, item["x"]))
        self.facts.update({
            "risk": current_risk, "contact": current_contact, "clearance": current_clear,
            "danger": danger, "unknown_near": unknown > 0,
            "waypoints": waypoints[:8],
            "best_x": waypoints[0]["x"] if waypoints else player["x"],
        })

    def update(self, value):
        self.obs = value if isinstance(value, dict) else {}
        old_frame = self.frame
        self.frame = integer(self.obs.get("frame"), old_frame + 1)
        self.score = finite(self.obs.get("score"), self.score)
        new_lives = integer(self.obs.get("lives"), self.lives)
        changed = self.previous_lives is not None and new_lives != self.previous_lives
        if changed:
            old = self.previous_lives
            self.lives = new_lives
            self._reset_after_life(old, new_lives)
        else:
            self.lives = new_lives
        self.previous_lives = new_lives

        player = self._player_record()
        # A decoder discontinuity or a large player jump invalidates the patrol
        # target.  Do not infer a wrap period for the ship.
        if player.get("discontinuity"):
            self.goal_x = None
            self.burst_left = 0
            self.recover_until = max(self.recover_until, self.frame + 24)
            self.tracks = {}
        if player.get("present"):
            self.previous_player = (player["x"], player["y"])
            self.previous_player_frame = self.frame
        else:
            self.previous_player = None
            self.previous_player_frame = None

        rocks = self._read_rocks(player)
        visible_count = sum(1 for rock in rocks if not rock.get("ghost"))
        if self.previous_rock_count is not None and visible_count + 3 < self.previous_rock_count:
            self.transition_until = max(self.transition_until, self.frame + 14)
        self.previous_rock_count = visible_count
        self._update_facts(player, rocks)

        if not self.initialized:
            # Preserve the verified central firing opening.  The phase patrol
            # starts only after roughly twelve seconds of live evidence; later
            # legs, including post-life legs, are still feedback controlled.
            self.initialized = True
            self.next_patrol = self.frame + 720
            self.next_emergency_replan = self.frame + 720

        if self.facts.get("present") and self.goal_x is None and self.frame >= self.next_patrol and self.frame >= self.recover_until:
            self._choose_goal(force=True)
        elif self.facts.get("present") and self.frame >= self.next_patrol and self.frame >= self.recover_until:
            self._choose_goal(force=True)
        elif self.facts.get("present") and self.goal_x is not None:
            # An emergency map can replace a leg, but ordinary patrol does not
            # chase a newly flickering rock every frame.
            current_risk = finite(self.facts.get("risk"), 0.0)
            if (self.frame >= self.next_emergency_replan
                    and self.facts.get("danger") in {"immediate", "hot"}
                    and self.frame + 10 < self.goal_until):
                best_x = self.facts.get("best_x")
                if best_x is not None and abs(wrap_delta(best_x - self.goal_x, WIDTH)) > 10:
                    self._choose_goal(force=False, emergency=True)

    # --------------------------- Jev interface ----------------------------

    def _target(self):
        if not self.facts.get("present"):
            return None
        candidates = [r for r in self.facts.get("rocks", []) if not r.get("ghost")]
        if not candidates:
            return None
        heading = integer(self.facts.get("heading"), 0)
        px, py = finite(self.facts.get("x"), 80.0), finite(self.facts.get("y"), 105.0)
        def quality(rock):
            dx = wrap_delta(rock["x"] - px, WIDTH)
            dy = wrap_delta(rock["y"] - py, HEIGHT)
            desired = heading_for(dx + rock.get("vx", 0.0) * 6.0,
                                 dy + rock.get("vy", 0.0) * 6.0)
            turn_cost = min((desired - heading) % 16, (heading - desired) % 16)
            imminent = 5.0 if rock.get("size") == "small" else 2.0
            if rock.get("known"):
                imminent += 0.6
            return -(3.0 * self._size_weight(rock.get("size")) + imminent
                     - turn_cost * 0.07 - math.hypot(dx, dy) / 180.0)
        return min(candidates, key=quality)

    @staticmethod
    def _bearing(dx, dy):
        horizontal = "right" if dx > 12 else "left" if dx < -12 else "center"
        vertical = "below" if dy > 12 else "above" if dy < -12 else "level"
        return horizontal + "_" + vertical

    def decision_state(self):
        f = self.facts
        target = self._target()
        target_summary = None
        if target is not None:
            dx = wrap_delta(target["x"] - finite(f.get("x"), 80.0), WIDTH)
            dy = wrap_delta(target["y"] - finite(f.get("y"), 105.0), HEIGHT)
            target_summary = {
                "size": target.get("size", "unknown"),
                "bearing": self._bearing(dx, dy),
                "motion": "known" if target.get("known") else "uncertain",
                "distance": bucket(math.hypot(dx, dy), (28, 75, 120), ("unknown", "near", "middle", "far")),
            }
        waypoint_summary = []
        for item in f.get("waypoints", [])[:6]:
            relation = "hold" if abs(item["x"] - finite(f.get("x"), item["x"])) < 9 else ("left" if item["x"] < finite(f.get("x"), item["x"]) else "right")
            waypoint_summary.append({
                "id": item["id"], "relation": relation,
                "risk": "safe" if item["risk"] < 2 else "watch" if item["risk"] < 7 else "hot",
                "impact": time_label(item.get("contact")),
                "clearance": clearance_label(item.get("clearance")),
            })
        x = f.get("x")
        return {
            "rules": {
                "mode": 0, "difficulty": 0,
                "fire": "press_then_release; holding is not rapid fire",
                "noop": "coasts; it is not a brake",
                "down": "mode-0 hyperspace; not reverse thrust",
                "heading": "LEFT increments and RIGHT decrements orientation code",
                "enemy_coverage": "unknown; null is not absence",
                "model_role": "judge semantic posture only; code owns arithmetic and actions",
            },
            "snapshot": {
                "frame": int(self.frame), "lives": int(self.lives),
                "score": int(max(0, self.score)),
                "player": "present" if f.get("present") else "absent",
                "position": "unknown" if x is None else bucket(x, (55, 105), ("unknown", "left", "middle", "right")),
                "motion": "unknown" if f.get("speed") is None else ("still" if f["speed"] < 0.18 else "moving"),
                "danger": f.get("danger", "unknown"),
                "impact_window": time_label(f.get("contact")),
                "rocks": min(15, len(f.get("rocks", []))),
                "unknown_or_ghost": sum(1 for r in f.get("rocks", []) if not r.get("known") or r.get("ghost")),
                "missiles": min(2, integer(f.get("missiles"), 0)),
                "transition_guard": max(0, self.transition_until - self.frame),
            },
            "patrol": {
                "goal": None if self.goal_x is None else "left" if self.goal_x < finite(x, self.goal_x) - 8 else "right" if self.goal_x > finite(x, self.goal_x) + 8 else "hold",
                "goal_distance": "unknown" if x is None or self.goal_x is None else bucket(abs(self.goal_x - x), (10, 28, 55), ("near", "middle", "far", "far")),
                "waypoints": waypoint_summary,
                "pulses_this_leg": int(self.patrol_pulses),
                "phase": int(self.phase),
            },
            "target": target_summary,
            "controller": {
                "mode": self.model_mode,
                "aim": self.model_aim,
                "defenses_this_life": len(self.defense_frames),
                "frame_local": True,
            },
            "plan": dict(self.policy),
        }

    def questions(self):
        regimes = {
            "turret": "The field is readable and the ship should keep a firing posture; code may still perform its low-duty phase shift.",
            "patrol": "A broad, bounded horizontal phase shift is useful; choose among code-generated waypoints, never an open-loop action sequence.",
            "escape": "A current corridor is dangerous; survival takes priority and code must re-evaluate a bounded relocation.",
            "recover": "The player or object field just changed; coast and fire only while the local controller rebuilds tracks.",
            "uncertain": "Flicker or unknown motion prevents a semantic commitment; do not infer that missing enemy objects are absent.",
        }
        order = ["turret", "patrol", "escape", "recover", "uncertain"]
        if (self.frame // 30) % 2:
            order.reverse()
        aim = {
            "target": "Use the code-described high-value rock and short turns; code owns bearing arithmetic.",
            "sweep": "Use a bounded alternating scan; never hold a model-issued turn indefinitely.",
            "hold": "Keep the current nose while the patrol/firing controller handles the frame.",
        }
        aim_order = ["target", "sweep", "hold"]
        if (self.frame // 60) % 2:
            aim_order.reverse()
        side = {
            "hold": "Do not change the current patrol goal unless the local risk map changes materially.",
            "left": "If code needs a patrol tie-break, prefer the described left waypoint; code rechecks it.",
            "right": "If code needs a patrol tie-break, prefer the described right waypoint; code rechecks it.",
        }
        side_order = ["hold", "left", "right"]
        if (self.frame // 18) % 2:
            side_order.reverse()
        return {
            "regime_a": Choice(
                "Classify the supplied situation using snapshot.danger, impact_window, transition_guard, and patrol waypoints. Do not calculate numbers or issue an action.",
                {key: regimes[key] for key in order},
            ),
            "regime_b": Choice(
                "Independently classify the same situation. Immediate safety facts and recover/uncertain states outrank target value; missing enemy objects are unknown.",
                {key: regimes[key] for key in reversed(order)},
            ),
            "aim": Choice(
                "Choose a short semantic aiming posture. The local controller owns FIRE release, rotation, and all timing; this answer cannot create thrust.",
                {key: aim[key] for key in aim_order},
            ),
            "side": Choice(
                "Choose a side only as a tie-break for the next code-generated waypoint. Do not emit a sequence or compare raw distances.",
                {key: side[key] for key in side_order},
            ),
            "defense": Noul(
                "Should the local code consider one mode-0 hyperspace pulse? Answer true only for an imminent local contact when no bounded patrol path is materially safer and the life defense reserve permits it.",
                {"true": "A last-resort defense is warranted under all stated local gates.", "false": "Do not invoke hyperspace; use the bounded local fallback."},
            ),
        }

    def accept_jev(self, answers, judged_frame, judged_epoch):
        if not isinstance(answers, dict):
            return
        age = self.frame - integer(judged_frame)
        if judged_epoch != self.life_epoch or age < 0 or age > 48:
            self.log("jev_discard", frame=self.frame, judged_frame=judged_frame, age=age, reason="stale")
            return
        ra = answer_value(answers.get("regime_a"))
        rb = answer_value(answers.get("regime_b"))
        if ra == rb and ra in {"turret", "patrol", "escape", "recover", "uncertain"}:
            p = min(answer_probability(answers.get("regime_a"), ra), answer_probability(answers.get("regime_b"), rb))
            if p >= 0.18:
                self.model_mode = ra
                self.model_until = self.frame + 60
        aim = answer_value(answers.get("aim"))
        if aim in {"target", "sweep", "hold"} and answer_probability(answers.get("aim"), aim) >= 0.18:
            self.model_aim = aim
            self.model_until = max(self.model_until, self.frame + 60)
        side = answer_value(answers.get("side"))
        if side in {"hold", "left", "right"} and answer_probability(answers.get("side"), side) >= 0.22:
            self.model_side = side
        if yes_probability(answers.get("defense")) >= 0.84:
            self.model_defense_hint = True
        self.log("jev_posture", frame=self.frame, judged_frame=judged_frame,
                 mode=self.model_mode, aim=self.model_aim, side=self.model_side,
                 defense=self.model_defense_hint)

    # --------------------------- action bridge ----------------------------

    def _fire_ready(self):
        if self.frame < self.next_fire or integer(self.facts.get("missiles"), 0) >= 2:
            return False
        gap = 6 if self.policy.get("fire") == "spaced" else 5
        self.next_fire = self.frame + gap
        return True

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

    def _current_target_heading(self):
        target = self._target()
        if target is None or self.facts.get("heading") is None:
            if ((self.frame // 64) % 2) == 0:
                return (integer(self.facts.get("heading"), 0) + 1) % 16
            return (integer(self.facts.get("heading"), 0) - 1) % 16
        px, py = finite(self.facts.get("x"), 80.0), finite(self.facts.get("y"), 105.0)
        dx = wrap_delta(target["x"] - px, WIDTH) + target.get("vx", 0.0) * 6.0
        dy = wrap_delta(target["y"] - py, HEIGHT) + target.get("vy", 0.0) * 6.0
        return heading_for(dx, dy)

    def _goal_direction(self):
        x = self.facts.get("x")
        if x is None or self.goal_x is None:
            return 0, False
        delta = wrap_delta(self.goal_x - x, WIDTH)
        vx = self.facts.get("vx")
        # Once close to the waypoint, counter-thrust only when measured drift is
        # carrying the ship through it.  Otherwise coast and let a new map pick
        # the next phase.
        if abs(delta) <= 8.0 and (vx is None or abs(vx) < 0.16):
            return 0, False
        moving_toward = vx is not None and abs(vx) >= 0.16 and ((vx > 0) == (delta > 0))
        braking = abs(delta) < 18.0 and moving_toward
        if braking:
            direction = -1 if vx > 0 else 1
        else:
            direction = -1 if delta < 0 else 1
        return direction, braking

    def _move_action(self, fire, urgent=False):
        if not self.facts.get("present") or self.goal_x is None:
            return None
        if self.frame < self.recover_until:
            return None
        if self.burst_left > 0:
            # The burst was approved on the previous fresh snapshot.  It is
            # only two more native frames, and the next update can cancel it
            # through an emergency replan or a life/discontinuity reset.
            self.burst_left -= 1
            return self._compose(0, thrust=True, fire=fire)
        direction, braking = self._goal_direction()
        if direction == 0:
            return None
        desired = 4 if direction < 0 else 12
        current = integer(self.facts.get("heading"), 0) % 16
        delta = (desired - current) % 16
        distance = abs(wrap_delta(self.goal_x - finite(self.facts.get("x"), self.goal_x), WIDTH))
        # A one-frame combined turn/thrust is reserved for a close threat.  In
        # normal patrol the nose is aligned before the pulse, so movement stays
        # horizontal and has a measurable effect on the risk map.
        if delta != 0:
            turn = turn_sign(current, desired)
            combined = urgent and delta <= 4 and self.frame >= self.next_pulse
            if combined:
                self.next_pulse = self.frame + 8
                self.patrol_pulses += 1
                self.last_pulse = self.frame
                self.log("phase_pulse", frame=self.frame, kind="turn_thrust", goal=self.goal_x,
                         urgent=True, distance=round(distance, 1), braking=braking)
                return self._compose(turn, thrust=True, fire=fire)
            return self._compose(turn, thrust=False, fire=fire)
        if self.frame < self.next_pulse:
            return None
        # Six pulses per leg is enough to shift phase, but much smaller than the
        # movement duty cycle rejected by the thrust-heavy prior.
        pulse_limit = 8 if urgent else 5
        if self.patrol_pulses >= pulse_limit:
            return None
        self.patrol_pulses += 1
        self.next_pulse = self.frame + (8 if urgent else 24)
        self.burst_left = 2
        self.last_pulse = self.frame
        self.log("phase_pulse", frame=self.frame, kind="horizontal", goal=self.goal_x,
                 urgent=urgent, distance=round(distance, 1), braking=braking,
                 pulse=self.patrol_pulses)
        return self._compose(0, thrust=True, fire=fire)

    def _should_defend(self):
        if not self.facts.get("present") or self.policy.get("defense") == "never":
            return False
        if len(self.defense_frames) >= 1:
            return False
        contact = self.facts.get("contact")
        if contact is None or contact > 2:
            return False
        if self.frame - (self.defense_frames[-1] if self.defense_frames else -100000) < 120:
            return False
        # A waypoint that is genuinely clear is preferred over hyperspace.  At
        # contact <= 1, do not pretend the approximate path is a guarantee.
        best = finite(self.facts.get("best_x"), finite(self.facts.get("x"), 80.0))
        best_risk, best_contact, _, _ = self._risk_for_x(best, 24)
        if best_contact is None and best_risk < 3.0 and contact > 0:
            return False
        return contact <= 1 or self.model_defense_hint

    def choose(self):
        f = self.facts
        if not f.get("present"):
            self.last_action = self.noop
            return self.noop
        fire = self._fire_ready()
        if self._should_defend():
            self.defense_frames.append(self.frame)
            self.model_defense_hint = False
            self.goal_x = None
            self.burst_left = 0
            self.recover_until = self.frame + 34
            self.tracks = {}
            self.log("defense", frame=self.frame, contact=f.get("contact"), count=len(self.defense_frames))
            self.last_action = self.legal("DOWN")
            return self.last_action

        danger = f.get("danger", "unknown")
        urgent = danger in {"immediate", "hot"} or (f.get("contact") is not None and f.get("contact") <= 14)
        move = self._move_action(fire, urgent=urgent)
        if move is not None:
            self.last_action = move
            return move

        if self.frame < self.recover_until:
            action = self._compose(0, thrust=False, fire=fire)
        else:
            current = integer(f.get("heading"), 0)
            if self.model_aim == "hold" and self.model_until >= self.frame:
                turn = 0
            elif self.model_aim == "sweep" and self.model_until >= self.frame:
                turn = 1 if ((self.frame // 56) % 2) == 0 else -1
            else:
                turn = turn_sign(current, self._current_target_heading())
            # During a hot contact, do not rotate away from a horizontal patrol
            # correction unless movement is unavailable; FIRE itself remains a
            # one-frame pulse.
            if urgent and self.goal_x is not None and self.frame < self.goal_until:
                turn = 0 if abs(wrap_delta(self.goal_x - finite(f.get("x"), self.goal_x), WIDTH)) > 8 else turn
            action = self._compose(turn=turn, thrust=False, fire=fire)
        self.last_action = action
        if self.frame - self.last_log >= 480 or fire or danger in {"immediate", "hot"}:
            self.log("control", frame=self.frame, action=action, danger=danger,
                     contact=f.get("contact"), risk=round(finite(f.get("risk"), 0.0), 2),
                     goal=self.goal_x, score=round(self.score, 1), lives=self.lives)
            self.last_log = self.frame
        return action

    # --------------------------- Astra application ------------------------

    def apply_policy(self, result):
        data = extract_json_object(result)
        if not isinstance(data, dict):
            return False
        allowed = {
            "patrol": {"steady", "rare", "responsive"},
            "aim": {"target", "sweep"},
            "fire": {"normal", "spaced"},
            "risk": {"cautious", "balanced"},
            "defense": {"rare", "never"},
        }
        changed = {}
        for key, choices in allowed.items():
            value = data.get(key)
            if isinstance(value, str) and value in choices:
                changed[key] = value
        if not changed:
            return False
        self.policy.update(changed)
        self.log("astra_policy", frame=self.frame, policy=dict(self.policy))
        return True


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.s1_enabled = getattr(ctx, "system1", True) is not None
        self.controller = None
        self.jev_future = None
        self.jev_frame = -1
        self.jev_epoch = -1
        self.next_jev = -1
        self.astra_future = None
        self.astra_calls = 0
        self.next_astra = -1

    def _start_jev(self):
        c = self.controller
        if c is None or not self.s1_enabled or self.jev_future is not None or c.frame < self.next_jev:
            return
        state = c.decision_state()
        questions = c.questions()
        origin, epoch = c.frame, c.life_epoch
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=2.2)
            )
            self.jev_frame = origin
            self.jev_epoch = epoch
            self.next_jev = c.frame + 30
        except (BudgetExceeded, EpisodeOver):
            self.s1_enabled = False
        except ModelUnavailable:
            raise
        except Exception:
            self.jev_future = None
            self.next_jev = c.frame + 42

    def _collect_jev(self):
        c = self.controller
        future = self.jev_future
        if c is None or future is None or not future.done():
            return
        self.jev_future = None
        try:
            c.accept_jev(future.result(), self.jev_frame, self.jev_epoch)
        except (BudgetExceeded, EpisodeOver):
            self.s1_enabled = False
        except ModelUnavailable:
            raise
        except Exception:
            # A timeout, cancellation, arbitrary stand-in text, or malformed
            # answer is a missed semantic label, not a harness error.
            pass

    def _astra_prompt(self, reason):
        c = self.controller
        if c is None:
            return ""
        f = c.facts
        state = {
            "reason": reason,
            "frame": c.frame,
            "score": int(max(0, c.score)),
            "lives": c.lives,
            "danger": f.get("danger", "unknown"),
            "contact": f.get("contact"),
            "risk": round(finite(f.get("risk"), 0.0), 2),
            "clearance": round(finite(f.get("clearance"), 0.0), 2),
            "rocks": len(f.get("rocks", [])),
            "unknown_or_ghost": sum(1 for r in f.get("rocks", []) if not r.get("known") or r.get("ghost")),
            "goal_x": c.goal_x,
            "phase_pulses": c.patrol_pulses,
            "defenses_this_life": len(c.defense_frames),
            "policy": dict(c.policy),
        }
        return (
            "You are a slow strategic reviewer for real-time Atari 2600 Asteroids "
            "mode 0 difficulty 0. The local code owns the clock, object tracking, "
            "vertical-crossing risk map, horizontal one-frame thrust pulses, aiming, "
            "FIRE release, defense gates, and all actions. Jev only labels posture. "
            "A stationary turret is vulnerable to flickering or newly split rocks, "
            "but movement must stay low-duty and central. Review this aggregate state. "
            "Return one compact JSON object only, with any subset of exact fields: "
            "patrol ('steady','rare','responsive'), aim ('target','sweep'), fire "
            "('normal','spaced'), risk ('cautious','balanced'), defense ('rare','never'). "
            "Never return actions, timestamps, sequences, or prose. Unknown enemy "
            "coverage is not absence.\nSTATE:\n" +
            json.dumps(state, separators=(",", ":"), allow_nan=False)
        )

    def _start_astra(self, reason):
        c = self.controller
        if c is None or self.astra_future is not None or self.astra_calls >= 2:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if integer(remaining.get("system2_calls"), 0) <= 0:
                self.astra_calls = 2
                return
            prompt = self._astra_prompt(reason)
            self.astra_calls += 1
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=105.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_calls = 2
        except ModelUnavailable:
            raise
        except Exception:
            self.astra_future = None

    def _collect_astra(self):
        c = self.controller
        future = self.astra_future
        if c is None or future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            c.apply_policy(getattr(result, "text", result))
        except (BudgetExceeded, EpisodeOver):
            pass
        except ModelUnavailable:
            raise
        except Exception:
            pass

    def run_episode(self, game) -> None:
        if getattr(game, "tick_s", None) is None:
            return
        c = PhaseController(game, self.ctx)
        self.controller = c
        try:
            c.update(safe_observation(game))
        except Exception:
            c.obs = {}
            c.facts = {"present": False, "frame": 0, "score": 0.0, "lives": 0,
                       "rocks": [], "danger": "unknown", "contact": None}

        self.next_jev = c.frame
        self.next_astra = c.frame + 300
        try:
            self._start_jev()
            if not self.ctx.done and not getattr(game, "done", False):
                self.ctx.act(c.choose())
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return
        except Exception:
            try:
                self.ctx.act(c.noop)
            except Exception:
                return

        while not self.ctx.done and not getattr(game, "done", False):
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                return
            if self.ctx.done or getattr(game, "done", False):
                break
            try:
                c.update(safe_observation(game))
                self._collect_jev()
                self._collect_astra()

                if c.frame >= self.next_jev and self.jev_future is None:
                    try:
                        self._start_jev()
                    except (BudgetExceeded, EpisodeOver):
                        self.s1_enabled = False
                    except ModelUnavailable:
                        return
                    except Exception:
                        self.next_jev = c.frame + 42

                # One review is late enough to observe the launch family; a
                # second is requested only after the first life transition or a
                # long surviving run.  Astra can change policy, never a frame.
                if self.astra_future is None and self.astra_calls < 2:
                    if c.frame >= self.next_astra:
                        reason = "scheduled aggregate review"
                        if c.life_epoch > 0:
                            reason = "post-life evidence review"
                        try:
                            self._start_astra(reason)
                        except ModelUnavailable:
                            return
                        except (BudgetExceeded, EpisodeOver):
                            self.astra_calls = 2
                        except Exception:
                            self.astra_calls = 2
                        self.next_astra = c.frame + (360 if c.life_epoch == 0 else 240)

                self.ctx.act(c.choose())
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception as exc:
                try:
                    c.log("frame_recovered", frame=c.frame, error=type(exc).__name__)
                    self.ctx.act(c.noop)
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
