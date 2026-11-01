"""Event-arbiter controller for Atari 2600 Asteroids.

The emulator clock is deliberately not a model interface.  A small deterministic
arbiter turns each observation into a handful of bounded firing, aiming, and
escape programs.  Jev supplies semantic labels (field regime, useful target,
and a side preference); it does not select an action or perform geometry.  Astra
is a slow reviewer of the measured risk/score tradeoff and can change only the
long-lived posture of the arbiter.
"""
from __future__ import annotations

import json
import math
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


W = 160.0
H = 210.0
ACTIONS = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE",
    "UPLEFTFIRE",
)


def number(value: Any, default: float | None = None) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def integer(value: Any, default: int = 0) -> int:
    x = number(value)
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def arr(value: Any) -> list:
    return value if isinstance(value, list) else []


def observation(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def wrap(value: float, period: float) -> float:
    value = float(value)
    while value > period / 2.0:
        value -= period
    while value < -period / 2.0:
        value += period
    return value


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def answer_value(answer: Any) -> Any:
    if isinstance(answer, dict):
        return answer.get("value", answer.get("choice", answer.get("answer")))
    return getattr(answer, "value", None)


def probability(answer: Any, key: str) -> float:
    if isinstance(answer, dict):
        values = answer.get("probabilities", {})
    else:
        values = getattr(answer, "probabilities", {})
    if not isinstance(values, dict):
        return 0.0
    return clamp(number(values.get(key), 0.0) or 0.0, 0.0, 1.0)


def yes_probability(answer: Any) -> float:
    value = answer_value(answer)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.lower() in {"true", "yes"} else 0.0
    if isinstance(value, (int, float)):
        return clamp(float(value), 0.0, 1.0)
    return probability(answer, "true")


def json_object(text: Any) -> dict | None:
    """Extract a JSON object from arbitrary Astra or stand-in text."""
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


def heading_for(dx: float, dy: float) -> int:
    if abs(dx) + abs(dy) < 1e-8:
        return 0
    # Code 0 is up, 4 left, 8 down, 12 right; y grows down.
    return int(round(math.atan2(-dx, -dy) / (math.pi / 8.0))) % 16


def turn_sign(current: int, desired: int, tie: int = 1) -> int:
    delta = (desired - current) % 16
    if delta == 0:
        return 0
    if delta < 8:
        return 1       # LEFT increments the code.
    if delta > 8:
        return -1      # RIGHT decrements the code.
    return tie if tie in (-1, 1) else 1


def heading_vector(code: int) -> tuple[float, float]:
    angle = (int(code) % 16) * math.pi / 8.0
    return -math.sin(angle), -math.cos(angle)


def clearance_label(value: float) -> str:
    if value < 0.0:
        return "overlap"
    if value < 7.0:
        return "tight"
    if value < 18.0:
        return "narrow"
    if value < 42.0:
        return "open"
    return "wide"


def time_label(value: float | None) -> str:
    if value is None:
        return "none"
    if value <= 4:
        return "immediate"
    if value <= 14:
        return "soon"
    if value <= 30:
        return "later"
    return "distant"


def distance_label(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 28:
        return "near"
    if value < 75:
        return "middle"
    return "far"


class Arbiter:
    """Read-only geometry plus a bounded action program."""

    SIDE_OPTIONS = ("hold", "left", "right")
    REGIMES = ("turret", "fragmenting", "escape", "transition", "uncertain")

    def __init__(self, game: Any, ctx: Any):
        self.game = game
        self.ctx = ctx
        spec = getattr(game, "action_spec", None)
        advertised = getattr(spec, "names", ()) if spec is not None else ()
        self.names = set(advertised or ACTIONS)
        self.noop = getattr(spec, "noop", "NOOP") if spec is not None else "NOOP"
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else next(iter(self.names), "NOOP")

        self.frame = 0
        self.score = 0.0
        self.lives = 0
        self.old_lives: int | None = None
        self.life_epoch = 0
        self.obs: dict = {}
        self.facts: dict = {}
        self.rocks: list[dict] = []
        self.targets: list[dict] = []
        self.routes: dict[str, dict] = {}
        self.danger = "uncertain"
        self.threat_time: float | None = None
        self.current_target: str | None = None
        self.target_lease_until = -1
        self.model_side = "hold"
        self.model_side_until = -1
        self.model_regime = "turret"
        self.model_regime_until = -1
        self.model_defense = False

        # Policy is the only mutable product Astra may author.
        self.policy = {
            "priority": "survival",
            "mobility": "rare",
            "fire": "normal",
            "defense": "last_resort",
            "aim": "target",
        }

        self.respawn_until = -1
        self.goal_side: str | None = None
        self.goal_until = -1
        self.last_thrust = -100000
        self.next_thrust = 0
        self.next_fire = 0
        self.defense_used = False
        self.last_defense = -100000
        self.last_action = self.noop
        self.last_log = -100000
        self.last_move_log = -100000
        self.score_at_life = 0.0

    # ---------- small IO and bookkeeping ----------

    def log(self, event: str, **fields: Any) -> None:
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def legal(self, action: str, fallback: str | None = None) -> str:
        if action in self.names:
            return action
        if fallback and fallback in self.names:
            return fallback
        if self.noop in self.names:
            return self.noop
        return next(iter(self.names), self.noop)

    def _player(self) -> dict:
        return obj(self.obs.get("player"))

    @staticmethod
    def _velocity(value: Any) -> tuple[float | None, float | None]:
        if isinstance(value, list) and len(value) >= 2:
            return number(value[0]), number(value[1])
        return None, None

    def _reset_life_state(self, old: int, new: int) -> None:
        self.life_epoch += 1
        self.goal_side = None
        self.goal_until = -1
        self.model_side = "hold"
        self.model_side_until = -1
        self.current_target = None
        self.target_lease_until = -1
        self.model_regime = "turret"
        self.model_regime_until = -1
        self.model_defense = False
        self.defense_used = False
        self.last_defense = self.frame
        self.respawn_until = self.frame + 34
        self.next_fire = self.frame + 9
        self.next_thrust = self.frame + 18
        self.last_thrust = self.frame
        self.log("life_epoch", frame=self.frame, old_lives=old, lives=new,
                 score=round(self.score, 1), epoch=self.life_epoch)

    # ---------- observations and numerical facts ----------

    def _read_rocks(self, player: dict) -> list[dict]:
        if not bool(player.get("present")):
            return []
        px = number(player.get("x"), W / 2.0) or W / 2.0
        py = number(player.get("y"), H / 2.0) or H / 2.0
        pw = number(player.get("width"), 6.0) or 6.0
        ph = number(player.get("height"), 10.0) or 10.0
        pvx, pvy = self._velocity(player.get("velocity_px_per_frame"))
        result = []
        for index, raw_value in enumerate(arr(self.obs.get("asteroids"))):
            raw = obj(raw_value)
            rx = number(raw.get("x"))
            ry = number(raw.get("y"))
            if rx is None or ry is None:
                continue
            rw = number(raw.get("width"), 8.0) or 8.0
            rh = number(raw.get("height"), 15.0) or 15.0
            dx = wrap((rx + rw / 2.0) - (px + pw / 2.0), W)
            dy = wrap((ry + rh / 2.0) - (py + ph / 2.0), H)
            avx, avy = self._velocity(raw.get("velocity_px_per_frame"))
            known = avx is not None and avy is not None and pvx is not None and pvy is not None
            rvx = (avx - pvx) if known else 0.0
            rvy = (avy - pvy) if known else 0.0
            size = str(raw.get("size") or "unknown").lower()
            weight = {"small": 3.0, "medium": 2.0, "large": 1.0}.get(size, 1.0)
            # Inflated reference boxes are intentionally used only as a safety
            # screen, never as claimed collision masks.
            radius = max(rw, rh) / 2.0 + max(pw, ph) / 2.0 + 3.0
            distance = math.hypot(dx, dy)
            best_clear = distance - radius
            best_t = 0.0
            best_x, best_y = dx, dy
            for t in range(2, 61, 2):
                sx = dx + rvx * t
                sy = dy + rvy * t
                d = math.hypot(wrap(sx, W), wrap(sy, H)) - radius
                if d < best_clear:
                    best_clear, best_t, best_x, best_y = d, float(t), sx, sy
            result.append({
                "slot": integer(raw.get("source_slot"), index),
                "size": size,
                "weight": weight,
                "dx": dx,
                "dy": dy,
                "rvx": rvx,
                "rvy": rvy,
                "known": known,
                "radius": radius,
                "distance": distance,
                "clearance": best_clear,
                "contact": best_t,
                "future_x": best_x,
                "future_y": best_y,
            })
        return result

    def _ship_path(self, side: str | None, horizon: int = 42) -> tuple[float, float]:
        """Return (minimum inflated clearance, first close time) for a policy.

        This is deliberately a policy comparison, not a physics oracle.  It
        treats unknown rock velocity as stationary and charges uncertainty to
        every path so a missing finite-difference estimate cannot be rewarded.
        """
        player = self._player()
        if not self.facts.get("present"):
            return -999.0, 0.0
        pvx = self.facts.get("vx") or 0.0
        pvy = self.facts.get("vy") or 0.0
        heading = integer(self.facts.get("heading"), 0) % 16
        sx = sy = 0.0
        min_clear = 999.0
        first_close = float(horizon + 1)
        # Movement packets are short.  The rest of the horizon is coast; this
        # is materially less aggressive than assuming indefinite thrust.
        thrust_frames = 0 if side in (None, "hold") else 8
        turn = 1 if side == "left" else -1 if side == "right" else 0
        for step in range(horizon + 1):
            if step and turn and step % 4 == 0:
                heading = (heading + turn) % 16
            if step and turn and step <= thrust_frames:
                hx, hy = heading_vector(heading)
                pvx = clamp(pvx + hx * 0.045, -1.6, 1.6)
                pvy = clamp(pvy + hy * 0.045, -1.6, 1.6)
            if step:
                sx += pvx
                sy += pvy
            for rock in self.rocks:
                rx = rock["dx"] + (rock["rvx"] * step if rock["known"] else 0.0)
                ry = rock["dy"] + (rock["rvy"] * step if rock["known"] else 0.0)
                separation = math.hypot(wrap(rx - sx, W), wrap(ry - sy, H))
                c = separation - rock["radius"]
                if not rock["known"]:
                    c -= 3.5
                if c < min_clear:
                    min_clear = c
                if c <= 6.0 and first_close > horizon:
                    first_close = float(step)
        return min_clear, (first_close if first_close <= horizon else float(horizon + 1))

    def _make_target(self, rock: dict) -> dict:
        projected_x = rock["dx"] + rock["rvx"] * 7.0
        projected_y = rock["dy"] + rock["rvy"] * 7.0
        desired = heading_for(projected_x, projected_y)
        heading = integer(self.facts.get("heading"), 0) % 16
        turn_cost = min((desired - heading) % 16, (heading - desired) % 16)
        # Value, threat, and a clean corridor are code facts.  Jev only sees
        # their semantic buckets and can resolve the final target preference.
        threat_bonus = 4.0 if rock["contact"] <= 24 else 0.0
        uncertainty = 0.7 if rock["known"] else 0.0
        quality = 2.0 * rock["weight"] + threat_bonus + uncertainty - turn_cost * 0.08 - rock["distance"] / 180.0
        key = "R" + str(rock["slot"]) + "_" + rock["size"]
        if turn_cost <= 1:
            alignment = "on_axis"
        elif turn_cost <= 3:
            alignment = "near_axis"
        else:
            alignment = "off_axis"
        return {
            "id": key,
            "slot": rock["slot"],
            "size": rock["size"],
            "bearing": self._bearing(rock["dx"], rock["dy"]),
            "alignment": alignment,
            "distance": distance_label(rock["distance"]),
            "motion": "known" if rock["known"] else "uncertain",
            "contact": time_label(rock["contact"]),
            "desired": desired,
            "quality": quality,
        }

    @staticmethod
    def _bearing(dx: float, dy: float) -> str:
        lr = "right" if dx > 12 else "left" if dx < -12 else "center"
        ud = "below" if dy > 12 else "above" if dy < -12 else "level"
        return lr + "_" + ud

    def update(self, value: Any) -> None:
        self.obs = observation(value)
        self.frame = integer(self.obs.get("frame"), self.frame + 1)
        self.score = number(self.obs.get("score"), self.score) or self.score
        current_lives = integer(self.obs.get("lives"), self.lives)
        if self.old_lives is not None and current_lives != self.old_lives:
            self._reset_life_state(self.old_lives, current_lives)
        self.lives = current_lives
        self.old_lives = current_lives

        player = self._player()
        present = bool(player.get("present")) and number(player.get("x")) is not None and number(player.get("y")) is not None
        px = number(player.get("x")) if present else None
        py = number(player.get("y")) if present else None
        vx, vy = self._velocity(player.get("velocity_px_per_frame")) if present else (None, None)
        self.facts = {
            "frame": self.frame,
            "score": self.score,
            "lives": self.lives,
            "present": present,
            "x": px,
            "y": py,
            "vx": vx,
            "vy": vy,
            "speed": math.hypot(vx, vy) if vx is not None and vy is not None else None,
            "heading": integer(player.get("orientation_code"), 0) % 16 if present else None,
            "break": bool(player.get("motion_discontinuity")) if present else True,
        }
        self.rocks = self._read_rocks(player)
        self.targets = sorted((self._make_target(r) for r in self.rocks),
                              key=lambda t: (-t["quality"], t["id"]))[:6]

        if not present:
            self.danger = "uncertain"
            self.threat_time = None
        else:
            near = min(self.rocks, key=lambda r: (r["contact"], r["clearance"]), default=None)
            self.threat_time = near["contact"] if near is not None and near["contact"] <= 60 else None
            # The current point is weighted most heavily; predicted contact is
            # only urgent when the approximate clearance is also tight.
            overlap = any(r["clearance"] <= 0.0 and r["contact"] <= 14 for r in self.rocks)
            immediate = any(r["clearance"] <= 6.0 and r["contact"] <= 8 for r in self.rocks)
            soon = any(r["clearance"] <= 14.0 and r["contact"] <= 24 for r in self.rocks)
            later = any(r["clearance"] <= 22.0 and r["contact"] <= 42 for r in self.rocks)
            if overlap:
                self.danger = "overlap"
            elif immediate:
                self.danger = "immediate"
            elif soon:
                self.danger = "soon"
            elif later:
                self.danger = "watch"
            elif self.rocks:
                self.danger = "clear"
            else:
                self.danger = "transition"

        self.routes = {}
        for side in ("hold", "left", "right"):
            c, t = self._ship_path(None if side == "hold" else side)
            self.routes[side] = {
                "clearance": c,
                "contact": t,
                "clearance_label": clearance_label(c),
                "time_label": time_label(t if t <= 42 else None),
            }
        self.facts.update({
            "danger": self.danger,
            "threat_time": self.threat_time,
            "rock_count": len(self.rocks),
            "small_count": sum(1 for r in self.rocks if r["size"] == "small"),
            "unknown_count": sum(1 for r in self.rocks if not r["known"]),
            "missiles": len(arr(self.obs.get("player_missiles"))),
            "routes": self.routes,
        })

    # ---------- compact System-1 interface ----------

    def _best_target(self) -> dict | None:
        if self.current_target:
            for target in self.targets:
                if target["id"] == self.current_target:
                    return target
        return self.targets[0] if self.targets else None

    def _route_state(self, side: str) -> dict:
        route = self.routes.get(side, {})
        return {
            "side": side,
            "clearance": route.get("clearance_label", "unknown"),
            "contact": route.get("time_label", "unknown"),
            "preferred_by_code": side == self._code_side(),
        }

    def _code_side(self) -> str:
        if not self.routes:
            return "hold"
        # A move must buy a meaningful margin.  If it does not, staying is safer
        # than chasing an approximate box.
        hold = self.routes["hold"]["clearance"]
        options = [(self.routes[s]["clearance"], s) for s in ("left", "right")]
        options.sort(key=lambda x: (-x[0], x[1]))
        if self.danger in {"overlap", "immediate", "soon"}:
            return options[0][1]
        if options[0][0] > hold + 7.0:
            return options[0][1]
        return "hold"

    def decision_state(self) -> dict:
        f = self.facts
        player_position = "unknown"
        if f.get("x") is not None:
            player_position = "left" if f["x"] < 55 else "right" if f["x"] > 105 else "middle"
        target_list = []
        for target in self.targets[:5]:
            target_list.append({
                "id": target["id"],
                "size": target["size"],
                "bearing": target["bearing"],
                "alignment": target["alignment"],
                "distance": target["distance"],
                "motion": target["motion"],
                "contact": target["contact"],
            })
        return {
            "rules": {
                "mode": 0,
                "fire_requires_release": True,
                "noop_does_not_brake": True,
                "down_is_hyperspace": True,
                "left_increments_orientation": True,
                "right_decrements_orientation": True,
                "enemy_coverage": "unknown_not_absent",
                "model_selects_intent_not_action": True,
            },
            "snapshot": {
                "frame": int(f.get("frame", self.frame)),
                "lives": int(f.get("lives", self.lives)),
                "score": int(max(0.0, f.get("score", 0.0))),
                "player_present": bool(f.get("present")),
                "position": player_position,
                "motion": "unknown" if f.get("speed") is None else ("still" if f["speed"] < 0.18 else "moving"),
                "danger": self.danger,
                "impact_window": time_label(self.threat_time),
                "rocks": {
                    "count": int(f.get("rock_count", 0)),
                    "small": int(f.get("small_count", 0)),
                    "unknown_motion": int(f.get("unknown_count", 0)),
                },
                "missiles_visible": int(f.get("missiles", 0)),
                "motion_discontinuity": bool(f.get("break")),
            },
            "paths": [self._route_state(s) for s in ("hold", "left", "right")],
            "targets": target_list,
            "controller": {
                "code_side": self._code_side(),
                "current_target": self.current_target or "auto",
                "model_regime": self.model_regime,
                "model_side": self.model_side,
                "defense_used": bool(self.defense_used),
            },
            "policy": dict(self.policy),
        }

    def questions(self) -> dict:
        state = self.decision_state()
        regimes = {
            "turret": "Keep the ship's current position and use short aim/fire pulses; do not add thrust merely for a distant rock.",
            "fragmenting": "Keep a valuable firing target in view and be ready for a new fragment; use only a bounded escape if the code path facts require it.",
            "escape": "The code has identified a bounded relocation as the relevant safety subgoal; survival outranks target value until the path is rechecked.",
            "transition": "The field or player state is changing; avoid a long maneuver, restore a readable central firing posture, and release controls on uncertainty.",
            "uncertain": "The supplied geometry cannot justify a semantic commitment; use the conservative code fallback and do not invent enemy absence.",
        }
        regime_order = list(self.REGIMES)
        if (self.frame // 30) % 2:
            regime_order.reverse()
        target_options = {"sweep": "No target has a sufficiently trustworthy clean shot; use a bounded alternating aim sweep."}
        for target in self.targets[:5]:
            target_options[target["id"]] = (
                "Prefer this code-described rock: " + target["size"] + ", "
                + target["bearing"] + ", " + target["alignment"] + " alignment, "
                + target["distance"] + " distance, " + target["motion"] + " motion."
            )
        target_order = list(target_options)
        if (self.frame // 30) % 2:
            target_order.reverse()
        target_options = {k: target_options[k] for k in target_order}
        side_options = {
            "hold": "Do not relocate; preserve the current firing position unless the code's immediate safety gate overrides it.",
            "left": "If a relocation is needed, use the code-generated short left path; never hold thrust open-loop.",
            "right": "If a relocation is needed, use the code-generated short right path; never hold thrust open-loop.",
        }
        side_order = list(side_options)
        if (self.frame // 18) % 2:
            side_order.reverse()
        side_options = {k: side_options[k] for k in side_order}
        return {
            "regime_a": Choice(
                "Classify the field regime from snapshot.danger, snapshot.impact_window, paths, and targets. This is a short-lived semantic label; code owns all arithmetic and actions.",
                {k: regimes[k] for k in regime_order},
            ),
            "regime_b": Choice(
                "Independently classify the same field regime. Immediate safety facts take priority over scoring; unknown geometry is not a clear field.",
                {k: regimes[k] for k in reversed(regime_order)},
            ),
            "target_a": Choice(
                "Choose the most useful currently described target or sweep. Do not calculate a bearing, treat source slots as identities, or emit an action.",
                target_options,
            ),
            "target_b": Choice(
                "Independently choose the best clean target for this snapshot. If no candidate is trustworthy choose sweep; code will revalidate the choice.",
                {k: target_options[k] for k in reversed(target_order)},
            ),
            "side": Choice(
                "If the safety arbiter needs a bounded relocation, choose a side preference from these descriptions. This answer cannot create thrust or bypass the local path gate.",
                side_options,
            ),
            "defense": Noul(
                "Should a single mode-0 hyperspace pulse be considered only if the local code reports overlap/immediate danger, no bounded path is materially safer, and defense has not been used in this life? Otherwise false.",
                {"true": "A last-resort pulse is worth considering under all stated local gates.", "false": "Do not invoke hyperspace; continue the bounded steering or firing fallback."},
            ),
        }

    def accept_jev(self, answers: Any, judged_frame: int, judged_epoch: int) -> None:
        if not isinstance(answers, dict):
            return
        age = self.frame - judged_frame
        if judged_epoch != self.life_epoch or age < 0 or age > 42:
            self.log("jev_stale", frame=self.frame, judged_frame=judged_frame, age=age)
            return

        ra = answer_value(answers.get("regime_a"))
        rb = answer_value(answers.get("regime_b"))
        if ra == rb and ra in self.REGIMES:
            p = min(probability(answers.get("regime_a"), ra), probability(answers.get("regime_b"), rb))
            if p >= 0.18:
                self.model_regime = ra
                self.model_regime_until = self.frame + 54

        ta = answer_value(answers.get("target_a"))
        tb = answer_value(answers.get("target_b"))
        if ta == tb and isinstance(ta, str):
            valid = {t["id"] for t in self.targets} | {"sweep"}
            p = min(probability(answers.get("target_a"), ta), probability(answers.get("target_b"), tb))
            if ta in valid and p >= 0.16:
                # A target is a short lease, not a frame-by-frame vote.  This
                # prevents two equally plausible Jev labels from making the
                # nose oscillate while the shot is in flight.  A disappeared
                # target is naturally replaced by _best_target's code fallback.
                if self.current_target is None or self.frame >= self.target_lease_until:
                    self.current_target = None if ta == "sweep" else ta
                    self.target_lease_until = self.frame + 90

        side = answer_value(answers.get("side"))
        if side in self.SIDE_OPTIONS and probability(answers.get("side"), side) >= 0.25:
            self.model_side = side
            self.model_side_until = self.frame + 42

        self.model_defense = yes_probability(answers.get("defense")) >= 0.80
        self.log("jev_intent", frame=self.frame, judged_frame=judged_frame,
                 age=age, regime=self.model_regime, target=self.current_target or "sweep",
                 side=self.model_side, defense=self.model_defense)

    # ---------- deterministic action policy ----------

    def _fire_ready(self) -> bool:
        if self.frame < self.next_fire or self.facts.get("missiles", 0) >= 2:
            return False
        gap = 7 if self.policy.get("fire") == "spaced" else 5
        self.next_fire = self.frame + gap
        return True

    def _compose(self, turn: int = 0, thrust: bool = False, fire: bool = False) -> str:
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

    def _aim_turn(self) -> int:
        heading = self.facts.get("heading")
        if heading is None:
            return 0
        target = self._best_target()
        if self.model_regime == "transition" or self.facts.get("break"):
            target = None
        if target is not None and self.policy.get("aim") == "target":
            desired = integer(target.get("desired"), heading)
            return turn_sign(int(heading), desired, 1)
        # Sweeps are slow and bounded; a fresh direction is selected from the
        # frame epoch rather than held as a model action.
        return 1 if ((self.frame // 58) % 2 == 0) else -1

    def _start_goal(self, side: str, why: str) -> None:
        if side not in {"left", "right"}:
            return
        self.goal_side = side
        self.goal_until = self.frame + (48 if self.danger in {"overlap", "immediate"} else 82)
        self.last_move_log = self.frame
        self.log("goal_start", frame=self.frame, side=side, reason=why,
                 danger=self.danger, hold=clearance_label(self.routes.get("hold", {}).get("clearance", -999.0)),
                 left=clearance_label(self.routes.get("left", {}).get("clearance", -999.0)),
                 right=clearance_label(self.routes.get("right", {}).get("clearance", -999.0)))

    def _best_escape(self) -> str:
        if not self.routes:
            return "hold"
        candidates = [(self.routes[s]["clearance"], s) for s in ("left", "right")]
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best = candidates[0][1]
        # Model side is only a tie breaker / semantic preference.  It cannot
        # overturn a materially safer code path.
        hold = self.routes["hold"]["clearance"]
        if self.model_side in {"left", "right"} and self.model_side_until >= self.frame:
            preferred = self.routes[self.model_side]["clearance"]
            if preferred >= candidates[0][0] - 5.0:
                best = self.model_side
        if self.danger not in {"overlap", "immediate", "soon"} and self.routes[best]["clearance"] <= hold + 7.0:
            return "hold"
        return best

    def _movement_action(self, side: str, urgent: bool) -> str | None:
        player_x = self.facts.get("x")
        heading = self.facts.get("heading")
        if player_x is None or heading is None:
            return None
        # A side goal is a bounded lateral relocation, not a lane label to hold.
        desired_x = 54.0 if side == "left" else 106.0
        delta = wrap(desired_x - float(player_x), W)
        if abs(delta) <= 7.0:
            # Once the ship has reached the requested lateral band, coasting is
            # safer than repeatedly thrusting because a stale close rock can
            # keep the danger label high for several observations.
            self.goal_side = None
            return None
        desired_heading = 4 if side == "left" else 12
        turn = turn_sign(int(heading), desired_heading, 1 if side == "left" else -1)
        if turn != 0:
            # A single combined pulse is allowed only in a close window.  It
            # gives a late escape some displacement without latched thrust.
            fire = self._fire_ready() if self.danger != "overlap" else False
            if urgent and self.frame - self.last_thrust >= 5:
                self.last_thrust = self.frame
                self.next_thrust = max(self.next_thrust, self.frame + 7)
                return self._compose(turn, thrust=True, fire=fire)
            return self._compose(turn, thrust=False, fire=fire)
        if self.frame < self.next_thrust:
            fire = self._fire_ready() if self.danger != "overlap" else False
            return self._compose(0, thrust=False, fire=fire)
        # One native UP pulse, followed by a long coast.  Re-observation can
        # cancel the goal before another pulse is legal.
        self.last_thrust = self.frame
        self.next_thrust = self.frame + (18 if urgent else 28)
        fire = self._fire_ready() if self.danger != "overlap" else False
        return self._compose(0, thrust=True, fire=fire)

    def _should_defend(self) -> bool:
        if not self.facts.get("present") or self.defense_used:
            return False
        if self.policy.get("defense") == "never":
            return False
        if self.danger not in {"overlap", "immediate"}:
            return False
        if self.threat_time is None or self.threat_time > 2:
            return False
        best = max(self.routes.get("left", {}).get("clearance", -999.0),
                   self.routes.get("right", {}).get("clearance", -999.0))
        # Hyperspace is deliberately later than steering and is not a brake.
        # A local overlap alone is not enough: approximate boxes often report
        # a zero-time touch before the cartridge actually resolves a collision.
        # Require the high-consequence Jev gate as well.
        return best < 0.0 and self.model_defense

    def choose(self) -> str:
        if not self.facts.get("present") or self.frame < self.respawn_until:
            self.last_action = self.noop
            return self.noop

        best = self._best_escape()
        hold_clear = self.routes.get("hold", {}).get("clearance", -999.0)
        best_clear = self.routes.get(best, {}).get("clearance", -999.0)
        # A "soon" crossing is an observation for target/aim selection, not
        # permission to start a thrust program.  This keeps the anchored firing
        # regime that produced the parent comparison's long runs.
        urgent = self.danger in {"overlap", "immediate"}

        if self._should_defend():
            self.defense_used = True
            self.model_defense = False
            self.last_defense = self.frame
            self.goal_side = None
            self.log("defense", frame=self.frame, danger=self.danger, threat_time=self.threat_time)
            self.last_action = self.legal("DOWN")
            return self.last_action

        # Start or continue only if the local forecast says moving is useful.
        if urgent and best in {"left", "right"}:
            # Do not flip a live emergency program merely because one flickering
            # reference box changed the winning side by a few pixels.  A large
            # margin is required before replacing an active goal.
            active_bad = (
                self.goal_side in {"left", "right"}
                and self.goal_until >= self.frame
                and self.routes.get(self.goal_side, {}).get("clearance", -999.0) < best_clear - 10.0
            )
            if self.goal_side not in {"left", "right"} or self.goal_until < self.frame or active_bad:
                self._start_goal(best, "urgent_code_gate")
        elif self.danger == "watch" and best in {"left", "right"} and best_clear > hold_clear + 8.0:
            # The slow policy can request rare, proactive relocation, but the
            # default remains an anchored turret (unlike the thrust-heavy 0008).
            rare_ok = self.policy.get("mobility") == "periodic"
            model_ok = self.model_regime in {"escape", "fragmenting"} and self.model_regime_until >= self.frame
            if rare_ok and model_ok and self.frame - self.last_move_log >= 180:
                self._start_goal(best, "watch_code_and_model")

        if self.goal_side is not None and self.goal_until >= self.frame:
            movement = self._movement_action(self.goal_side, urgent)
            if movement is not None:
                self.last_action = movement
                if self.frame - self.last_log >= 420 or urgent:
                    self.log("control", frame=self.frame, action=movement, danger=self.danger,
                             goal=self.goal_side, score=round(self.score, 1), lives=self.lives)
                    self.last_log = self.frame
                return movement
            if not urgent:
                self.goal_side = None

        # Offensive turret: aim and fire only, with a real release gap between
        # presses.  This is the parent policy's strongest observed mechanism.
        turn = self._aim_turn()
        # FIRE remains useful during an immediate but non-overlapping window;
        # only an actual overlap suppresses the shot so the escape owns the
        # button for that frame.
        fire = self._fire_ready() if self.danger != "overlap" else False
        action = self._compose(turn=turn, fire=fire)
        self.last_action = action
        if self.frame - self.last_log >= 420 or fire:
            self.log("control", frame=self.frame, action=action, danger=self.danger,
                     goal=self.goal_side or "none", target=self.current_target or "auto",
                     score=round(self.score, 1), lives=self.lives)
            self.last_log = self.frame
        return action

    # ---------- slow policy application ----------

    def apply_policy(self, value: Any) -> bool:
        data = json_object(value)
        if not isinstance(data, dict):
            return False
        choices = {
            "priority": {"survival", "score"},
            "mobility": {"rare", "periodic"},
            "fire": {"normal", "spaced"},
            "defense": {"last_resort", "never"},
            "aim": {"target", "sweep"},
        }
        changed = {}
        for key, allowed in choices.items():
            candidate = data.get(key)
            if isinstance(candidate, str) and candidate in allowed:
                changed[key] = candidate
        if not changed:
            return False
        self.policy.update(changed)
        self.log("astra_policy", frame=self.frame, policy=dict(self.policy))
        return True


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.s1_enabled = getattr(ctx, "system1", True) is not None
        self.arbiter: Arbiter | None = None
        self.jev_future = None
        self.jev_frame = -1
        self.jev_epoch = -1
        self.next_jev = -1
        self.astra_future = None
        self.astra_calls = 0
        self.next_astra = -1

    def _start_jev(self) -> None:
        a = self.arbiter
        if a is None or not self.s1_enabled or self.jev_future is not None or a.frame < self.next_jev:
            return
        state = a.decision_state()
        questions = a.questions()
        self.jev_frame = a.frame
        self.jev_epoch = a.life_epoch
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=2.4)
            )
            self.next_jev = a.frame + 24
        except (BudgetExceeded, EpisodeOver):
            self.s1_enabled = False
        except ModelUnavailable:
            raise
        except Exception:
            self.jev_future = None
            self.next_jev = a.frame + 36

    def _collect_jev(self) -> None:
        a = self.arbiter
        future = self.jev_future
        if a is None or future is None or not future.done():
            return
        self.jev_future = None
        try:
            a.accept_jev(future.result(), self.jev_frame, self.jev_epoch)
        except (BudgetExceeded, EpisodeOver):
            self.s1_enabled = False
        except ModelUnavailable:
            raise
        except Exception:
            # A timeout, cancelled future, malformed provider object, or
            # arbitrary pre-flight stand-in is simply missing advice.
            pass

    def _astra_prompt(self, reason: str) -> str:
        a = self.arbiter
        if a is None:
            return ""
        f = a.facts
        state = {
            "reason": reason,
            "frame": a.frame,
            "score": int(max(0.0, a.score)),
            "lives": a.lives,
            "danger": a.danger,
            "threat_time": a.threat_time,
            "rocks": len(a.rocks),
            "unknown_motion": sum(1 for r in a.rocks if not r["known"]),
            "routes": {k: {"clearance": clearance_label(v.get("clearance", -999.0)),
                           "contact": time_label(v.get("contact"))}
                       for k, v in a.routes.items()},
            "current_policy": dict(a.policy),
            "last_action": a.last_action,
        }
        return (
            "You are the slow strategic reviewer for real-time Atari 2600 Asteroids "
            "mode 0 difficulty 0. The local arbiter owns the emulator clock, object "
            "arithmetic, collision gates, bounded thrust, FIRE release cadence, and "
            "all actions. Jev classifies the field and target. Do not output actions, "
            "routes, timestamps, or arithmetic. Review the aggregate state and return "
            "one compact JSON object with only these optional exact fields: priority "
            "('survival' or 'score'), mobility ('rare' or 'periodic'), fire ('normal' "
            "or 'spaced'), defense ('last_resort' or 'never'), aim ('target' or 'sweep'). "
            "Unknown enemy coverage is not absence. Prefer survival/rare movement unless "
            "the evidence justifies another posture.\nSTATE:\n"
            + json.dumps(state, separators=(",", ":"), allow_nan=False)
        )

    def _start_astra(self, reason: str) -> None:
        if self.arbiter is None or self.astra_future is not None or self.astra_calls >= 2:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if integer(remaining.get("system2_calls"), 0) <= 0:
                self.astra_calls = 2
                return
            prompt = self._astra_prompt(reason)
            self.astra_calls += 1
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=100.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_calls = 2
        except ModelUnavailable:
            raise
        except Exception:
            self.astra_future = None

    def _collect_astra(self) -> None:
        a = self.arbiter
        future = self.astra_future
        if a is None or future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            a.apply_policy(getattr(result, "text", result))
        except (BudgetExceeded, EpisodeOver):
            pass
        except ModelUnavailable:
            raise
        except Exception:
            pass

    def run_episode(self, game: Any) -> None:
        if getattr(game, "tick_s", None) is None:
            return
        a = Arbiter(game, self.ctx)
        self.arbiter = a
        try:
            a.update(game.observe())
        except Exception:
            a.obs = {}
            a.facts = {"frame": 0, "score": 0.0, "lives": 0, "present": False,
                       "break": True, "danger": "uncertain", "routes": {}}
            a.danger = "uncertain"

        self.next_jev = a.frame
        self.next_astra = a.frame + 180
        try:
            self._start_jev()
            self._start_astra("initial posture")
        except (BudgetExceeded, EpisodeOver):
            pass
        except ModelUnavailable:
            return
        try:
            if not self.ctx.done and not getattr(game, "done", False):
                self.ctx.act(a.choose())
        except (BudgetExceeded, EpisodeOver):
            return
        except ModelUnavailable:
            return
        except Exception:
            try:
                self.ctx.act(a.noop)
            except Exception:
                return

        while not self.ctx.done and not getattr(game, "done", False):
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            if self.ctx.done or getattr(game, "done", False):
                break
            try:
                a.update(game.observe())
                self._collect_jev()
                self._collect_astra()
                self._start_jev()
                # A second review is triggered by the first observed life
                # transition, or later if the episode remains unusually long.
                if self.astra_calls < 2 and a.frame >= self.next_astra:
                    self._start_astra("scheduled evidence review")
                    self.next_astra = a.frame + 300
                self.ctx.act(a.choose())
            except (BudgetExceeded, EpisodeOver):
                return
            except ModelUnavailable:
                return
            except Exception as exc:
                try:
                    a.log("frame_recovered", frame=a.frame, error=type(exc).__name__)
                    self.ctx.act(a.noop)
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
