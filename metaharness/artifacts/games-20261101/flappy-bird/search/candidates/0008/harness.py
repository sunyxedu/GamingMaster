"""Event-triggered feedback certificates, with inference only on exceptions.

Candidate 0008: preserve the parent's successful public-physics controller while
removing routine model authority and per-frame backward-search reconstruction.
Only standard-library code and the documented metaharness context are used.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


VMIN, VMAX = -9, 10
ACTIONS = ("IDLE", "FLAP")
PROFILES = ("CRUISE", "LIFT", "DROP", "UNKNOWN")


def finite(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def decode(raw):
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    f = raw.get("features")
    if not isinstance(f, (list, tuple)) or len(f) < 12:
        return None
    f = [finite(v, -10000) for v in f[:12]]
    if -10000 in f:
        return None
    bird, geom = raw.get("bird") or {}, raw.get("geometry") or {}
    if not isinstance(bird, dict) or not isinstance(geom, dict):
        return None
    frame = int(finite(raw.get("frame")))
    speed = finite(geom.get("pipe_speed_px_per_frame"), 4)
    width = finite(geom.get("screen_width"), 288)
    height = finite(geom.get("screen_height"), 512)
    bx, bw, bh = (finite(bird.get(k), d) for k, d in
                  (("x", 57), ("width", 34), ("height", 24)))
    pw = finite(geom.get("pipe_width"), 52)
    pipes, seen = [], set()
    for i in (0, 3, 6):
        x, top, bottom = f[i:i + 3]
        if top <= 0 and bottom >= height - 1:
            continue  # Public offscreen sentinel: no future-gap information.
        if bottom <= top + bh or x + pw < bx - 3:
            continue
        identity = (round(x + speed * frame, 6), top, bottom)
        if identity in seen:
            continue
        seen.add(identity)
        pipes.append({"x": x, "top": top, "bottom": bottom, "id": identity})
    pipes.sort(key=lambda p: p["x"])
    return {"frame": frame, "y": f[9], "v": f[10], "bx": bx, "bw": bw,
            "bh": bh, "pw": pw, "speed": speed, "width": width, "height": height,
            "ground": finite(geom.get("ground_y"), 404.48), "pipes": pipes,
            "action": raw.get("previous_action", "IDLE"),
            "score": finite(raw.get("score")),
            "done": bool(raw.get("done") or raw.get("terminated") or raw.get("truncated"))}


def band_bits(lo, hi, maximum):
    lo, hi = max(0, int(math.ceil(lo - 1e-7))), min(maximum, int(math.floor(hi + 1e-7)))
    return 0 if lo > hi else ((1 << (hi - lo + 1)) - 1) << lo


def preimage(bits, dy, universe):
    return bits >> dy if dy >= 0 else (bits << -dy) & universe


def active_pipe(s):
    return next((p for p in s["pipes"] if p["x"] + s["pw"] > s["bx"]), None)


def cruise_preference(s, margin, profile="CRUISE"):
    p = active_pipe(s)
    top, bottom = ((p["top"], p["bottom"]) if p else
                   (125.0, min(275.0, s["ground"] - 30)))
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
    if profile != "DROP" and s["y"] > anchor + 32 and s["v"] >= (-4 if profile == "LIFT" else -1):
        return "FLAP"
    return "IDLE"


class DynamicsMonitor:
    """Check consecutive public observations, including integration order."""

    def __init__(self):
        self.order = "new_velocity"
        self.previous = None
        self.matches = 0
        self.mismatches = 0
        self.errors = deque(maxlen=90)

    def update(self, s):
        old, self.previous = self.previous, s
        if old is None or s["frame"] - old["frame"] != 1:
            return None
        action = s["action"]
        if not isinstance(action, str) or action.upper() not in ACTIONS:
            return None
        action = action.upper()
        if old["y"] < 1 or s["y"] + s["bh"] >= s["ground"] - 1:
            return None
        nv = -9 if action == "FLAP" else min(old["v"] + 1, 10)
        en = abs(old["y"] + nv - s["y"]) + abs(nv - s["v"])
        eo = abs(old["y"] + (-9 if action == "FLAP" else old["v"]) - s["y"]) + abs(nv - s["v"])
        prior = self.order
        if min(en, eo) < 0.05:
            if action == "IDLE" and abs(en - eo) > 0.5:
                self.order = "new_velocity" if en < eo else "old_velocity"
            self.matches += 1
            self.errors.append(False)
            return "order_change" if prior != self.order else "verified" if self.matches == 12 else None
        self.errors.append(True)
        self.mismatches += 1
        return "mismatch"


class Certificate:
    """A feedback tube: exact feasible y bitsets for every time and velocity.

    Unlike an open-loop action script, a tube admits feedback at ANY observed
    y/v on its lattice. It expires when public geometry/physics changes. Its
    terminal set is a fixed-point controlled invariant for empty sky, not an
    unconstrained finite-horizon state. Unknown future pipes are not certified.
    """

    def __init__(self, s, margin, order, terminal_cache):
        self.epoch = s["frame"]
        self.frac = s["y"] - math.floor(s["y"])
        self.maximum = int(math.floor(s["ground"] - s["bh"] - 1 - self.frac))
        self.order = order
        self.margin = margin
        self.layers = []
        self.horizon = 0
        if not 1 <= self.maximum <= 2048 or s["speed"] <= 0:
            return
        self.universe = (1 << (self.maximum + 1)) - 1
        last_exit = max(((p["x"] + s["pw"] - s["bx"] + 3) / s["speed"]
                         for p in s["pipes"]), default=40)
        self.horizon = min(100, max(56, int(math.ceil(last_exit)) + 20))
        open_band = band_bits(2 - self.frac, s["ground"] - s["bh"] - 4 - self.frac, self.maximum)
        terminal_key = (self.maximum, self.frac, open_band, order)
        terminal = terminal_cache.get(terminal_key)
        if terminal is None:
            terminal = tuple([open_band] * 20)
            # Monotone finite-state fixed point. At most 20*(maximum+1)
            # strict decreases are possible; no game/private state is read.
            for _ in range(20 * (self.maximum + 1) + 1):
                previous = terminal
                terminal = self.backward(previous, open_band)
                if terminal == previous:
                    break
            terminal_cache[terminal_key] = terminal
        safe = []
        for t in range(self.horizon + 1):
            lo, hi = 2.0, s["ground"] - s["bh"] - 4.0
            for p in s["pipes"]:
                x = p["x"] - s["speed"] * t
                if x < s["bx"] + s["bw"] + 2 and x + s["pw"] > s["bx"] - 2:
                    lo = max(lo, p["top"] + margin)
                    hi = min(hi, p["bottom"] - s["bh"] - margin)
            safe.append(band_bits(lo - self.frac, hi - self.frac, self.maximum))
        self.layers = [None] * (self.horizon + 1)
        self.layers[-1] = tuple(bits & safe[-1] for bits in terminal)
        for t in range(self.horizon - 1, -1, -1):
            self.layers[t] = self.backward(self.layers[t + 1], safe[t])

    def backward(self, following, band):
        flap = preimage(following[0], -9, self.universe)
        result = []
        for v in range(VMIN, VMAX + 1):
            nv = min(v + 1, VMAX)
            dy = nv if self.order == "new_velocity" else v
            idle = preimage(following[nv - VMIN], dy, self.universe)
            result.append((idle | flap) & band)
        return tuple(result)

    def allowed(self, s):
        t = s["frame"] - self.epoch
        if not self.layers or not 0 <= t < self.horizon:
            return ()
        yi = int(round(s["y"] - self.frac))
        v = int(round(s["v"]))
        if abs(yi + self.frac - s["y"]) > 1e-5 or abs(v - s["v"]) > 0.05 or not VMIN <= v <= VMAX:
            return ()
        nv = min(v + 1, VMAX)
        dy = nv if self.order == "new_velocity" else v
        result = []
        for action, ny, next_v in (("IDLE", yi + dy, nv), ("FLAP", yi - 9, -9)):
            if 0 <= ny <= self.maximum and self.layers[t + 1][next_v - VMIN] & (1 << ny):
                result.append(action)
        return tuple(result)


class CertificateBank:
    def __init__(self):
        self.key = None
        self.tables = {}
        self.terminal_cache = {}
        self.stats = Counter()

    def invalidate(self):
        self.key = None
        self.tables.clear()

    def actions(self, s, order):
        key = (order, round(s["y"] - math.floor(s["y"]), 6),
               s["bx"], s["bw"], s["bh"], s["pw"], s["speed"], s["ground"],
               tuple(p["id"] for p in s["pipes"]))
        expired = any(s["frame"] >= c.epoch + c.horizon - 2 for c in self.tables.values())
        if key != self.key or expired:
            self.key = key
            self.tables.clear()
            self.stats["epochs"] += 1
        for margin in (10, 6, 2, 0):
            certificate = self.tables.get(margin)
            if certificate is None:
                certificate = Certificate(s, margin, order, self.terminal_cache)
                self.tables[margin] = certificate
                self.stats["builds"] += 1
            else:
                self.stats["hits"] += 1
            allowed = certificate.allowed(s)
            if allowed:
                self.stats["margin_" + str(margin)] += 1
                return allowed, margin, certificate.epoch + certificate.horizon - s["frame"]
        self.stats["uncertified"] += 1
        return (), 0, 0


def direction(s, focus):
    p = active_pipe(s)
    if p is None:
        return "UNKNOWN"
    if focus == "FOLLOWING_GAP":
        following = next((q for q in s["pipes"] if q["x"] > p["x"] + 4), None)
        if following:
            delta = following["top"] - p["top"]
            return "HIGHER" if delta < -12 else "LOWER" if delta > 12 else "LEVEL"
    center = (p["top"] + p["bottom"] - s["bh"]) / 2
    coast = max(0, -s["v"]) * max(0, -s["v"] - 1) / 2
    return "HIGHER" if s["y"] - coast > center + 22 else "LOWER" if s["y"] < center - 18 else "LEVEL"


def answer_choice(answer):
    value = getattr(answer, "value", None)
    probs = getattr(answer, "probabilities", None)
    if not isinstance(value, str) or value not in PROFILES or not isinstance(probs, dict):
        return None
    confidence = getattr(answer, "confidence", None)
    threshold = 0.90 if confidence is None else 0.80
    if finite(probs.get(value), -1) < threshold:
        return None
    if confidence is not None and finite(confidence, -1) < 0.40:
        return None
    return value if value != "UNKNOWN" else None


def bounded_json(text):
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    attempts = 0
    for pos, c in enumerate(text[:8000]):
        if c == "{":
            attempts += 1
            if attempts > 16:
                break
            try:
                obj, _ = decoder.raw_decode(text[pos:8000])
                if isinstance(obj, dict):
                    return obj
            except RecursionError:
                return None
            except ValueError:
                pass
    return None


class ExceptionSupervisor:
    """No routine polling: inference is reserved for falsified certificates.

    Normal margin relaxation is NOT a modeling failure. Rejected/late model
    answers are NOT evidence for another model call. Model output never changes
    physics or bypasses the newly observed state's certificate.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.s1 = self.s2 = None
        self.tag = None
        self.focus = "CURRENT_GAP"
        self.version = 0
        self.profile = "CRUISE"
        self.profile_key = None
        self.profile_until = -1
        self.last_profile = "CRUISE"
        self.last_profile_source = "compiled"
        self.last_request = -10000
        self.incident_start = None
        self.incident = 0
        self.review_supported = False
        self.reviews = 0
        self.review_incident = None
        self.calls = 0
        self.changes = 0
        self.latencies = deque(maxlen=16)

    def state(self, s, allowed, monitor):
        p = active_pipe(s)
        return {
            "rules": {"goal": "Pass pipes; code owns arithmetic and one-frame flap release.",
                      "authority": "Only change a preference among CURRENT certified actions."},
            "plan": {"version": self.version, "focus": self.focus,
                     "goal": "Recover a stable arc without weakening collision constraints."},
            "route": {"target": list(p["id"]) if p else None,
                      "phase": "INSIDE" if p and p["x"] < s["bx"] + s["bw"] else "APPROACH",
                      "priority_direction": direction(s, self.focus)},
            "motion": "RISING" if s["v"] < 0 else "FALLING_OR_LEVEL",
            "evidence": {"repeated_transition_mismatch": sum(monitor.errors) >= 3,
                         "no_certified_action_even_at_zero_margin": not bool(allowed)},
            "safety": {"allowed_actions": list(allowed), "cannot_override_certificate": True},
        }

    @staticmethod
    def semantic_key(state):
        r = state["route"]
        return (tuple(r["target"]) if r["target"] else None,
                r["phase"], r["priority_direction"], state["motion"], state["plan"]["version"])

    @staticmethod
    def questions():
        options = {
            "CRUISE": "Retain the normal arc when priority_direction is LEVEL or UNKNOWN.",
            "LIFT": "Prefer earlier safe upward preparation when priority_direction is HIGHER.",
            "DROP": "Prefer patient safe descent when priority_direction is LOWER.",
            "UNKNOWN": "Missing or contradictory facts: keep the code controller.",
        }
        instruction = ("Choose a recovery PROFILE, not an old frame action. Read route.priority_direction: "
                       "HIGHER supports LIFT, LOWER supports DROP, LEVEL or UNKNOWN supports CRUISE. "
                       "Use UNKNOWN for contradictory facts. Do no arithmetic. Code rechecks the latest "
                       "certificate and cannot execute an unsafe profile preference.")
        return {"profile": Choice(instruction, options),
                "reverse": Choice(instruction, dict(reversed(list(options.items())))),
                "review": Noul("Does evidence explicitly establish a repeated transition mismatch OR "
                               "no certified action even at zero margin? Ordinary margin relaxation, "
                               "network delay, and rejected model answers are NOT such evidence.",
                               {"true": "At least one named evidence flag is true.",
                                "false": "Both named evidence flags are false."})}

    def poll(self, s, state):
        if self.s1 is not None and self.s1.done():
            future, self.s1 = self.s1, None
            tag = self.tag
            try:
                answers = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.ctx.log("exception_jev_discard", reason=type(exc).__name__)
                answers = None
            if isinstance(answers, dict):
                age = s["frame"] - tag["frame"]
                self.latencies.append(max(0.0, self.ctx.clock.now() - tag["time"]))
                a, b = answer_choice(answers.get("profile")), answer_choice(answers.get("reverse"))
                fresh = (0 <= age <= 24 and tag["key"] == self.semantic_key(state)
                         and tag["incident"] == self.incident and self.incident_start is not None)
                if fresh and a is not None and a == b:
                    self.profile, self.profile_key = a, tag["key"]
                    self.profile_until = tag["frame"] + 30
                    self.ctx.log("exception_profile", profile=a, age_frames=age)
                else:
                    self.ctx.log("exception_jev_discard", reason="stale_uncertain_or_inactive", age_frames=age)
                review = finite(getattr(answers.get("review"), "value", 0))
                if fresh and review >= 0.95:
                    self.review_supported = True
        if self.s2 is not None and self.s2.done():
            future, self.s2 = self.s2, None
            try:
                response = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.ctx.log("exception_astra_discard", reason=type(exc).__name__)
                return
            if self.incident_start is None or self.review_incident != self.incident:
                self.ctx.log("exception_astra_discard", reason="inactive_or_stale_incident")
                return
            config = bounded_json(getattr(response, "text", None))
            if not config or config.get("focus") not in ("CURRENT_GAP", "FOLLOWING_GAP"):
                self.ctx.log("exception_astra_discard", reason="invalid_bounded_interface")
                return
            self.focus = config["focus"]
            self.version += 1
            self.profile_until = -1
            self.ctx.log("exception_interface_revision", focus=self.focus, version=self.version)

    def preference(self, s, allowed, margin, monitor):
        anomalous = not allowed or sum(monitor.errors) >= 3
        if anomalous and self.incident_start is None:
            self.incident += 1
            self.incident_start = s["frame"]
            self.review_supported = False
            self.ctx.log("certificate_exception", frame=s["frame"], incident=self.incident,
                         no_certificate=not bool(allowed), physics_errors=sum(monitor.errors))
        elif not anomalous and self.incident_start is not None:
            self.ctx.log("certificate_exception_resolved", frame=s["frame"], incident=self.incident)
            self.incident_start = None
            self.profile_until = -1
            self.review_supported = False
        # Poll a completed exception call even if the certificate recovered;
        # otherwise a finished Future could remain uncollected until the deadline.
        state = None
        if anomalous or self.s1 is not None or self.s2 is not None:
            state = self.state(s, allowed, monitor)
            self.poll(s, state)
        # The archive verifies that Jev's normal profile judgment is a pure
        # recognition of this named direction (all six runs agreed). Compile
        # that verified mapping into the fast fallback; exception Jev may still
        # propose a bounded alternative, but never supplies raw actions.
        code_profile = {"HIGHER": "LIFT", "LOWER": "DROP"}.get(
            direction(s, self.focus), "CRUISE")
        if not anomalous:
            self.last_profile, self.last_profile_source = code_profile, "compiled"
            return cruise_preference(s, margin, code_profile)
        if state is None:
            state = self.state(s, allowed, monitor)
        model_profile = (self.profile if s["frame"] <= self.profile_until
                          and self.profile_key == self.semantic_key(state) else None)
        profile = model_profile or code_profile
        self.last_profile = profile
        self.last_profile_source = "jev" if model_profile else "compiled"
        return cruise_preference(s, margin, profile)

    def launch_after_post(self, s, allowed, monitor):
        if self.incident_start is None or self.ctx.done:
            return
        remaining = self.ctx.budget.remaining()
        state = self.state(s, allowed, monitor)
        if (self.s1 is None and s["frame"] - self.last_request >= 30 and self.calls < 24
                and remaining.get("system1_calls", 0) > 2 and remaining.get("wall_s", 0) > 4):
            questions = self.questions()
            self.tag = {"frame": s["frame"], "key": self.semantic_key(state), "incident": self.incident,
                        "time": self.ctx.clock.now()}
            self.last_request = s["frame"]
            self.calls += 1
            try:
                self.s1 = self.ctx.spawn(lambda: self.ctx.decide(state, questions, timeout_s=3.0))
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.s1 = None
                self.ctx.log("exception_jev_discard", reason=type(exc).__name__)
        if (self.review_supported and s["frame"] - self.incident_start >= 30
                and self.s2 is None and self.reviews < 1
                and remaining.get("system2_calls", 0) > 1 and remaining.get("wall_s", 0) > 150):
            prompt = (
                "Review a persistent Flappy Bird certificate exception, not normal networking staleness. "
                "Code performs exact public-state reachability, fixed gravity +1 capped at +10, flap -9, "
                "and per-tick pulse release. Do not change physics, geometry, or clearance. Choose only "
                "the Jev interface focus CURRENT_GAP or FOLLOWING_GAP; default CURRENT_GAP. Return JSON "
                "with focus and reason (under 25 words). CURRENT_GAP asks Jev about recovery of the "
                "immediate arc; FOLLOWING_GAP asks about the next observed corridor direction. "
                "Jev has no inherited context. Each call supplies named facts plus independent typed "
                "questions: two reverse-order Choice(instructions, criteria) questions returning .value, "
                ".probabilities, .confidence and a Noul returning probability of evidence. Jev must not "
                "calculate or generate code. HIGHER means LIFT, LOWER means DROP, LEVEL means CRUISE. "
                "Unknown/conflicting facts retain the code policy. Agreement and winning probability "
                ">=0.80 (>=0.90 if confidence absent) and supplied confidence >=0.40 are required. "
                "A profile is valid only for matching target/phase/direction/motion/version, age <=24 "
                "frames, and an active exception; no answer may bypass fresh code safety checks. "
                "A returned focus revises future questions only; it does not post an action.\n"
                + json.dumps({"state": state, "diagnostics": {"verified": monitor.matches,
                    "mismatches": monitor.mismatches, "latency_s": list(self.latencies)}}, separators=(",", ":"))
            )
            self.reviews += 1
            self.review_incident = self.incident
            try:
                self.s2 = self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=120.0))
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.s2 = None
                self.ctx.log("exception_astra_discard", reason=type(exc).__name__)
            self.ctx.log("exception_astra_requested", frame=s["frame"], incident=self.incident)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.monitor = DynamicsMonitor()
        self.bank = CertificateBank()
        self.supervisor = ExceptionSupervisor(ctx)
        self.overrides = 0
        self.skipped = 0

    def run_episode(self, game):
        ctx = self.ctx
        last_frame, last_score = -1, -1
        held = "IDLE"
        ctx.act("IDLE")
        try:
            while not ctx.done:
                s = decode(ctx.observe())
                if s is None:
                    ctx.act("IDLE")
                    held = "IDLE"
                    ctx.wait_for_tick()
                    continue
                if s["done"]:
                    break
                if s["frame"] <= last_frame:
                    ctx.wait_for_tick()
                    continue
                # Revoke the previously delivered impulse before any allocation,
                # backward search, log serialization, or model-result handling.
                if held == "FLAP":
                    ctx.act("IDLE")
                    held = "IDLE"
                status = self.monitor.update(s)
                if status in ("mismatch", "order_change"):
                    self.bank.invalidate()
                allowed, margin, horizon = self.bank.actions(s, self.monitor.order)
                baseline = cruise_preference(s, margin)
                preferred = self.supervisor.preference(s, allowed, margin, self.monitor)
                if allowed:
                    action = preferred if preferred in allowed else allowed[0]
                    base_action = baseline if baseline in allowed else allowed[0]
                    self.overrides += action != preferred
                else:
                    action = baseline
                    if s["y"] + s["bh"] > s["ground"] - 35:
                        action = "FLAP"
                    if s["y"] < 8:
                        action = "IDLE"
                    base_action = action
                latest = decode(ctx.observe())
                if latest is not None and latest["frame"] != s["frame"]:
                    self.skipped += max(1, latest["frame"] - s["frame"])
                    last_frame = s["frame"]
                    continue  # Never execute an old computed impulse.
                ctx.act(action)
                held, last_frame = action, s["frame"]
                # All optional work follows the posted action.
                if status:
                    ctx.log("physics_" + status, frame=s["frame"], order=self.monitor.order,
                            verified=self.monitor.matches, mismatches=self.monitor.mismatches)
                if action != base_action:
                    if self.supervisor.last_profile_source == "jev":
                        self.supervisor.changes += 1
                    ctx.log("profile_changes_action", frame=s["frame"],
                            source=self.supervisor.last_profile_source,
                            profile=self.supervisor.last_profile,
                            baseline=base_action, action=action)
                if s["score"] != last_score or s["frame"] % 300 == 0:
                    last_score = s["score"]
                    ctx.log("certificate_summary", frame=s["frame"], score=s["score"],
                            allowed=list(allowed), margin=margin, horizon_remaining=horizon,
                            cache=dict(self.bank.stats), verified=self.monitor.matches,
                            mismatches=self.monitor.mismatches, skipped=self.skipped,
                            profile=self.supervisor.last_profile,
                            profile_source=self.supervisor.last_profile_source,
                            jev_calls=self.supervisor.calls, astra_calls=self.supervisor.reviews)
                self.supervisor.launch_after_post(s, allowed, self.monitor)
                ctx.wait_for_tick()
        finally:
            ctx.memory["candidate_0008"] = {
                "verified": self.monitor.matches, "mismatches": self.monitor.mismatches,
                "cache": dict(self.bank.stats), "safety_overrides": self.overrides,
                "jev_calls": self.supervisor.calls, "astra_calls": self.supervisor.reviews,
                "model_action_changes": self.supervisor.changes, "skipped": self.skipped,
            }


def build(ctx):
    return Harness(ctx)
