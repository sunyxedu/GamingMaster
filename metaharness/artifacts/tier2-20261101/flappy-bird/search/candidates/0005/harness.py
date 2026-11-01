"""Crossover: idle-default pulse execution + asynchronous semantic judgments.

Only public observations are used.  A small exact, integer-lattice reachability
calculation keeps model/network latency out of the per-frame actuator.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


V_MIN = -9
V_MAX = 10
PROFILES = ("CRUISE", "LIFT", "DROP", "UNKNOWN")


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def observation(raw):
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    features = raw.get("features")
    if not isinstance(features, (list, tuple)) or len(features) < 12:
        return None
    f = [number(x, -10000) for x in features[:12]]
    if any(x == -10000 for x in f):
        return None
    bird = raw.get("bird") or {}
    geom = raw.get("geometry") or {}
    if not isinstance(bird, dict) or not isinstance(geom, dict):
        return None
    frame = int(number(raw.get("frame")))
    speed = number(geom.get("pipe_speed_px_per_frame"), 4)
    width = number(geom.get("screen_width"), 288)
    height = number(geom.get("screen_height"), 512)
    bx = number(bird.get("x"), 57)
    bw = number(bird.get("width"), 34)
    bh = number(bird.get("height"), 24)
    pw = number(geom.get("pipe_width"), 52)
    pipes = []
    seen = set()
    for i in (0, 3, 6):
        x, top, bottom = f[i:i + 3]
        # The package's offscreen placeholder is NOT an enormous real gap.
        if top <= 0 and bottom >= height - 1:
            continue
        if bottom <= top + bh or x + pw < bx - 3:
            continue
        key = (round(x, 3), round(top, 3), round(bottom, 3))
        if key in seen:
            continue
        seen.add(key)
        pipes.append({"x": x, "top": top, "bottom": bottom,
                      "id": [round(x + speed * frame, 2), top, bottom]})
    pipes.sort(key=lambda p: p["x"])
    return {"frame": frame, "y": f[9], "v": f[10], "bx": bx,
            "bw": bw, "bh": bh, "pw": pw, "speed": speed,
            "ground": number(geom.get("ground_y"), 404.48),
            "width": width, "height": height, "pipes": pipes,
            "action": raw.get("previous_action", "IDLE"),
            "score": number(raw.get("score")),
            "done": bool(raw.get("done") or raw.get("terminated")
                         or raw.get("truncated"))}


def interval_mask(lo, hi, maximum):
    lo = max(0, int(math.ceil(lo - 1e-7)))
    hi = min(maximum, int(math.floor(hi + 1e-7)))
    if lo > hi:
        return 0
    return ((1 << (hi - lo + 1)) - 1) << lo


def inverse_shift(mask, displacement, all_bits):
    if displacement >= 0:
        return mask >> displacement
    return (mask << (-displacement)) & all_bits


class Reachability:
    """Backward reachable sets; one Python integer is a whole y-position set."""

    def __init__(self):
        self.order = "new_velocity"
        self.matches = 0
        self.mismatches = 0
        self.last = None
        self.recent_errors = deque(maxlen=90)

    def update(self, s):
        old = self.last
        self.last = s
        if not old or s["frame"] - old["frame"] != 1:
            return None
        action = str(s["action"]).upper()
        if action not in ("FLAP", "IDLE"):
            return None
        if old["y"] < 1 or s["y"] + s["bh"] >= s["ground"] - 1:
            return None
        nv = -9.0 if action == "FLAP" else min(old["v"] + 1, 10)
        new_y = old["y"] + nv
        old_y = old["y"] + (-9 if action == "FLAP" else old["v"])
        en = abs(new_y - s["y"]) + abs(nv - s["v"])
        eo = abs(old_y - s["y"]) + abs(nv - s["v"])
        if min(en, eo) < 0.05:
            if action == "IDLE" and abs(en - eo) > 0.5:
                self.order = "new_velocity" if en < eo else "old_velocity"
            self.matches += 1
            self.recent_errors.append(False)
            return "verified" if self.matches == 12 else None
        self.mismatches += 1
        self.recent_errors.append(True)
        return "mismatch"

    def allowed(self, s, margin):
        frac = s["y"] - math.floor(s["y"])
        yi = int(math.floor(s["y"]))
        maximum = int(math.floor(s["ground"] - s["bh"] - 1 - frac))
        if maximum < 1 or maximum > 2048 or yi < 0 or yi > maximum:
            return (), 0
        velocity = int(round(s["v"]))
        if not V_MIN <= velocity <= V_MAX:
            return (), 0
        if abs(velocity - s["v"]) > 0.05 or s["speed"] <= 0:
            return (), 0
        pipes = s["pipes"]
        last_exit = max(((p["x"] + s["pw"] - s["bx"] + 3)
                         / s["speed"] for p in pipes), default=40)
        horizon = min(100, max(56, int(math.ceil(last_exit)) + 20))
        all_bits = (1 << (maximum + 1)) - 1
        safe = []
        for t in range(horizon + 1):
            lo, hi = 2.0, s["ground"] - s["bh"] - 4.0
            for p in pipes:
                x = p["x"] - s["speed"] * t
                # Two horizontal pixels also cover rounding at the rectangle edge.
                if x < s["bx"] + s["bw"] + 2 and x + s["pw"] > s["bx"] - 2:
                    lo = max(lo, p["top"] + margin)
                    hi = min(hi, p["bottom"] - s["bh"] - margin)
            safe.append(interval_mask(lo - frac, hi - frac, maximum))
        reachable = [safe[-1]] * 20
        # At the terminal boundary, a rising bird must still be able to coast
        # without touching the ceiling.  No hidden pipe or future RNG is used.
        for v in range(V_MIN, 0):
            rise = ((-v) * (-v - 1)) / 2
            if self.order == "old_velocity":
                rise += -v
            reachable[v - V_MIN] &= interval_mask(2 + rise - frac, maximum, maximum)
        for t in range(horizon - 1, 0, -1):
            flap = inverse_shift(reachable[0], -9, all_bits)
            current = []
            band = safe[t]
            for v in range(V_MIN, V_MAX + 1):
                nv = min(v + 1, V_MAX)
                dy = nv if self.order == "new_velocity" else v
                idle = inverse_shift(reachable[nv - V_MIN], dy, all_bits)
                current.append((idle | flap) & band)
            reachable = current
        nv = min(velocity + 1, V_MAX)
        dy = nv if self.order == "new_velocity" else velocity
        allowed = []
        for name, ny, next_v in (("IDLE", yi + dy, nv), ("FLAP", yi - 9, -9)):
            if 0 <= ny <= maximum and reachable[next_v - V_MIN] & (1 << ny):
                allowed.append(name)
        return tuple(allowed), horizon


def target(s):
    return next((p for p in s["pipes"] if p["x"] + s["pw"] > s["bx"]), None)


def route(s):
    p = target(s)
    if p is None:
        return (None, "UNKNOWN", "OPEN")
    following = next((q for q in s["pipes"] if q["x"] > p["x"] + 4), None)
    if following:
        delta = following["top"] - p["top"]
        direction = "HIGHER" if delta < -12 else "LOWER" if delta > 12 else "LEVEL"
    else:
        direction = "UNKNOWN"
    phase = "INSIDE" if p["x"] < s["bx"] + s["bw"] else "APPROACH"
    return (tuple(p["id"]), direction, phase)


def preference(s, profile, margin):
    p = target(s)
    if p is None:
        top, bottom = 125.0, min(275.0, s["ground"] - 30)
    else:
        top, bottom = p["top"], p["bottom"]
    center = (top + bottom - s["bh"]) / 2
    anchor = min(bottom - s["bh"] - margin - 2, center + 22)
    if profile == "LIFT":
        anchor -= 6
    elif profile == "DROP":
        anchor += 5
    if s["y"] < 3:
        return "IDLE"
    if s["v"] >= 0 and s["y"] >= anchor:
        return "FLAP"
    # A bounded early-climb preference, still filtered by exact reachability.
    threshold = -4 if profile == "LIFT" else -1
    if profile != "DROP" and s["y"] > anchor + 32 and s["v"] >= threshold:
        return "FLAP"
    return "IDLE"


def choice_value(answer):
    value = getattr(answer, "value", None)
    probs = getattr(answer, "probabilities", None)
    if value not in PROFILES or not isinstance(probs, dict):
        return "UNKNOWN", 0.0, None
    probability = number(probs.get(value), 0)
    confidence = getattr(answer, "confidence", None)
    if confidence is not None:
        confidence = number(confidence, -1)
    return value, probability, confidence


def small_json(text):
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text[:12000]):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[pos:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.model = Reachability()
        self.margin = 10
        self.focus = "CURRENT_GAP"
        self.plan_version = 0
        self.profile = "CRUISE"
        self.profile_key = None
        self.profile_until = -1
        self.s1 = None
        self.s1_tag = None
        self.s2 = None
        self.s2_count = 0
        self.s2_session = None
        self.last_request = -100
        self.last_s2 = -10000
        self.bad_judgments = 0
        self.judgments = 0
        self.relaxed = 0
        self.overrides = 0
        self.need_review = False
        self.latencies = deque(maxlen=40)

    def questions(self):
        criteria = {
            "CRUISE": {"what": "Maintain a regular single-flap arc in the current gap.",
                       "when": "The named priority direction is LEVEL or UNKNOWN."},
            "LIFT": {"what": "Prefer earlier upward preparation, only inside the safe action set.",
                     "when": "The named priority direction is HIGHER."},
            "DROP": {"what": "Prefer patient descent and avoid unnecessary upward impulses.",
                     "when": "The named priority direction is LOWER."},
            "UNKNOWN": "Facts are missing or contradictory; retain the code controller.",
        }
        instructions = (
            "Choose a control PROFILE, not a frame action. Read route.priority_direction "
            "and plan.goal. HIGHER calls for LIFT, LOWER for DROP, LEVEL for CRUISE; "
            "UNKNOWN direction with an otherwise valid gap calls for CRUISE. Use UNKNOWN "
            "only when facts are missing or contradictory. All profiles are filtered by "
            "the same exact safety controller. Do not compare or calculate numbers."
        )
        return {
            "profile": Choice(instructions, criteria),
            "profile_reverse": Choice(instructions, dict(reversed(list(criteria.items())))),
            "review": Noul(
                "Do diagnostics explicitly report repeated physics mismatch OR persistent "
                "safety fallback? These are evidence for a slow interface review, not "
                "permission to execute an unsafe action.",
                {"true": "At least one of those diagnostic flags is true.",
                 "false": "Both diagnostic flags are false."}),
        }

    def state_for_jev(self, s, allowed, margin):
        key, following, phase = route(s)
        p = target(s)
        if p is None:
            current = "UNKNOWN"
        else:
            desired = (p["top"] + p["bottom"] - s["bh"]) / 2
            coast_rise = max(0, -s["v"]) * max(0, -s["v"] - 1) / 2
            low_point = s["y"] - coast_rise
            current = ("HIGHER" if low_point > desired + 22 else
                       "LOWER" if s["y"] < desired - 18 else "LEVEL")
        priority = following if self.focus == "NEXT_GAP" and following != "UNKNOWN" else current
        return {
            "rules": {"goal": "Pass pipe pairs without collision.",
                      "y_axis": "Higher on screen means smaller y.",
                      "execution": "Flap is a one-frame pulse, followed by IDLE. Code handles every tick."},
            "plan": {"version": self.plan_version, "focus": self.focus,
                     "goal": "A safe arc in the current corridor, then a prepared transition."},
            "route": {"target_id": list(key) if key else None, "phase": phase,
                      "current_gap_direction": current, "following_gap_direction": following,
                      "priority_direction": priority},
            "motion": "RISING" if s["v"] < 0 else "FALLING_OR_LEVEL",
            "safety": {"allowed_actions": list(allowed), "clearance_margin_px": margin,
                       "models_never_override_reachability": True},
            "diagnostics": {"repeated_physics_mismatch": sum(self.model.recent_errors) >= 3,
                            "persistent_safety_fallback": self.relaxed >= 12},
        }

    def collect(self, s):
        ctx = self.ctx
        if self.s1 is not None and self.s1.done():
            future, tag = self.s1, self.s1_tag
            self.s1 = None
            try:
                answers = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                ctx.log("jev_discard", reason=type(exc).__name__)
                answers = None
            if isinstance(answers, dict):
                age = s["frame"] - tag["frame"]
                latency = max(0.0, ctx.clock.now() - tag["time"])
                self.latencies.append(latency)
                self.judgments += 1
                a, pa, ca = choice_value(answers.get("profile"))
                b, pb, cb = choice_value(answers.get("profile_reverse"))
                confident = (pa >= (0.85 if ca is None else 0.72)
                             and pb >= (0.85 if cb is None else 0.72)
                             and (ca is None or ca >= 0.35)
                             and (cb is None or cb >= 0.35))
                fresh = (0 <= age <= 24 and tag["key"] == route(s)
                         and tag["version"] == self.plan_version)
                if fresh and confident and a == b and a != "UNKNOWN":
                    self.profile = a
                    self.profile_key = tag["key"]
                    self.profile_until = tag["frame"] + 30
                    ctx.log("jev_profile", profile=a, age_frames=age,
                            probability=min(pa, pb), plan_version=self.plan_version)
                else:
                    self.bad_judgments += 1
                    ctx.log("jev_discard", reason="stale_or_uncertain", age_frames=age,
                            agreed=a == b, fresh=fresh)
                review = number(getattr(answers.get("review"), "value", 0))
                if review >= 0.9 and (sum(self.model.recent_errors) >= 3 or self.relaxed >= 12):
                    self.need_review = True
        if self.s2 is not None and self.s2.done():
            future = self.s2
            self.s2 = None
            try:
                answer = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                ctx.log("astra_discard", reason=type(exc).__name__)
                return
            config = small_json(getattr(answer, "text", None))
            self.s2_session = getattr(answer, "session_id", None)
            if not config or config.get("focus") not in ("CURRENT_GAP", "NEXT_GAP"):
                ctx.log("astra_discard", reason="invalid_bounded_plan")
                return
            margin = number(config.get("margin_px"), self.margin)
            if margin not in (6, 8, 10, 12):
                ctx.log("astra_discard", reason="invalid_margin")
                return
            self.margin = int(margin)
            self.focus = config["focus"]
            self.plan_version += 1
            self.profile_until = -1
            self.bad_judgments = 0
            self.relaxed = 0
            self.need_review = False
            ctx.log("astra_plan", focus=self.focus, margin_px=self.margin,
                    plan_version=self.plan_version)

    def maybe_request(self, s, allowed, margin):
        ctx = self.ctx
        remaining = ctx.budget.remaining()
        if (self.s1 is None and s["frame"] - self.last_request >= 12
                and remaining.get("system1_calls", 0) > 16
                and remaining.get("wall_s", 0) > 3
                and getattr(ctx, "system1", True) is not None):
            state = self.state_for_jev(s, allowed, margin)
            questions = self.questions()
            self.s1_tag = {"frame": s["frame"], "time": ctx.clock.now(),
                           "key": route(s), "version": self.plan_version}
            self.last_request = s["frame"]
            self.s1 = ctx.spawn(lambda: ctx.decide(state, questions, timeout_s=2.0))
        evidence = (self.need_review or sum(self.model.recent_errors) >= 3
                    or (self.bad_judgments >= 12 and self.judgments >= 16))
        if (evidence and self.s2 is None and self.s2_count < 2
                and s["frame"] - self.last_s2 >= 1800
                and s["frame"] >= 150 and remaining.get("system2_calls", 0) > 1
                and remaining.get("wall_s", 0) > 150):
            summary = self.state_for_jev(s, allowed, margin)
            summary["evidence"] = {"score": s["score"], "frame": s["frame"],
                                   "verified_transitions": self.model.matches,
                                   "physics_mismatches": self.model.mismatches,
                                   "rejected_judgments": self.bad_judgments,
                                   "judgments": self.judgments,
                                   "safety_overrides": self.overrides,
                                   "model_latency_s": list(self.latencies)}
            prompt = (
                "Review this Flappy Bird harness interface asynchronously. The game keeps "
                "running under exact observation-based reachability and one-frame flap pulses. "
                "The fixed physics are +1 gravity (cap +10), flap -9, pipe speed 4; observations "
                "have confirmed the listed transitions. Code alone handles arithmetic, collision "
                "geometry, and per-frame actuation; do not replace it. Jev chooses CRUISE/LIFT/DROP "
                "from named direction labels, with duplicate reverse-order Choices. Your ONLY "
                "permitted revision is focus CURRENT_GAP or NEXT_GAP and margin_px 6,8,10,12. "
                "CURRENT_GAP emphasizes the immediate arc; NEXT_GAP emphasizes transitions. "
                "Keep CURRENT_GAP/10 unless evidence supports another choice. A larger margin "
                "is more conservative but can reduce maneuverability. Return JSON only with "
                "keys focus, margin_px, reason (reason at most 30 words). No code.\n"
                "Jev usage contract: each request carries its entire state and independent "
                "typed questions; it inherits no prior context. Choice(instructions, criteria) "
                "returns .value option key, .probabilities, .confidence. Noul returns probability "
                "of yes, not severity. Code computes numeric comparisons and supplies named "
                "labels. No generation or arithmetic is requested from Jev. Reject disagreeing "
                "Choices and low probabilities; missing confidence is not certainty. Answers "
                "expire on route/plan change or after 24 frames. A request's 2-second timeout "
                "never blocks the tick loop. Your focus selects which direction label Jev sees "
                "as its priority; safety constraints remain authoritative.\n"
                + json.dumps(summary, separators=(",", ":"))
            )
            self.last_s2 = s["frame"]
            self.s2_count += 1
            session = self.s2_session
            self.s2 = ctx.spawn(lambda: ctx.think(prompt, session_id=session, timeout_s=120.0))
            ctx.log("astra_review_requested", frame=s["frame"], rejected=self.bad_judgments)

    def run_episode(self, game):
        ctx = self.ctx
        last_frame = -1
        last_posted = "IDLE"
        last_score = -1
        ctx.act("IDLE")
        while not ctx.done:
            s = observation(ctx.observe())
            if s is None:
                ctx.act("IDLE")
                ctx.wait_for_tick()
                continue
            if s["done"]:
                break
            if s["frame"] <= last_frame:
                ctx.wait_for_tick()
                continue
            # Revoke a pulse as soon as a subsequent tick is observed, BEFORE
            # parsing model answers, planning, or launching any background work.
            if last_posted == "FLAP":
                ctx.act("IDLE")
            status = self.model.update(s)
            if status:
                ctx.log("physics_" + status, frame=s["frame"], order=self.model.order,
                        matches=self.model.matches, mismatches=self.model.mismatches)
            self.collect(s)
            allowed, horizon, used_margin = (), 0, self.margin
            for margin in dict.fromkeys((self.margin, 6, 2, 0)):
                allowed, horizon = self.model.allowed(s, margin)
                used_margin = margin
                if allowed:
                    break
            if used_margin < self.margin:
                self.relaxed += 1
            baseline = preference(s, "CRUISE", used_margin)
            active_profile = (self.profile if s["frame"] <= self.profile_until
                              and self.profile_key == route(s) else "CRUISE")
            preferred = preference(s, active_profile, used_margin)
            if allowed:
                action = preferred if preferred in allowed else allowed[0]
                base_action = baseline if baseline in allowed else allowed[0]
                if action != preferred:
                    self.overrides += 1
                if action != base_action:
                    ctx.log("jev_changes_action", frame=s["frame"], profile=active_profile,
                            baseline=base_action, action=action)
            else:
                # No trajectory exists under the current public-state model.
                # This is recovery, not a claim that an unsafe action is safe.
                action = preference(s, "CRUISE", 0)
                if s["y"] + s["bh"] > s["ground"] - 35:
                    action = "FLAP"
                if s["y"] < 8:
                    action = "IDLE"
                self.need_review = True
            # If computation straddled a tick, do not post a stale impulse.
            latest = observation(ctx.observe())
            if latest is not None and latest["frame"] != s["frame"]:
                last_frame = s["frame"]
                last_posted = "IDLE"
                continue
            ctx.act(action)
            last_frame = s["frame"]
            last_posted = action
            if s["score"] != last_score or s["frame"] % 300 == 0:
                last_score = s["score"]
                ctx.log("controller_summary", frame=s["frame"], score=s["score"],
                        horizon=horizon, margin_px=used_margin, allowed=list(allowed),
                        profile=active_profile, relaxed_frames=self.relaxed,
                        verified_transitions=self.model.matches)
            if not ctx.done:
                self.maybe_request(s, allowed, used_margin)
                ctx.wait_for_tick()
        ctx.memory["candidate_0005"] = {
            "verified_transitions": self.model.matches,
            "physics_mismatches": self.model.mismatches,
            "judgments": self.judgments, "rejected_judgments": self.bad_judgments,
            "safety_overrides": self.overrides, "astra_reviews": self.s2_count,
        }


def build(ctx):
    return Harness(ctx)
