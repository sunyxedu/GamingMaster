"""Candidate 0006: cached viability certificates and incident-scoped supervision.

The actuator owns the clock; models own diagnostic interpretation, not impulses.
All geometry comes from public observations. Nothing is keyed by seed.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import dataclass

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


ACTIONS = ("IDLE", "FLAP")
VELOCITIES = tuple(range(-9, 11))


def finite(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except (ValueError, TypeError, OverflowError):
        return float(default)


@dataclass(frozen=True)
class Pipe:
    x: float
    top: float
    bottom: float
    origin: float


@dataclass(frozen=True)
class State:
    frame: int
    y: float
    v: float
    bx: float
    bw: float
    bh: float
    pw: float
    ground: float
    speed: float
    pipes: tuple
    applied: str
    score: float
    done: bool

    @property
    def geometry(self):
        return (self.bx, self.bw, self.bh, self.pw, self.ground, self.speed)

    @property
    def layout(self):
        return tuple((p.origin, p.top, p.bottom) for p in self.pipes)


def decode(raw):
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError, RecursionError):
            return None
    if not isinstance(raw, dict):
        return None
    features = raw.get("features")
    if not isinstance(features, (list, tuple)) or len(features) < 12:
        return None
    f = tuple(finite(x, -100000) for x in features[:12])
    if any(x == -100000 for x in f):
        return None
    bird, geometry = raw.get("bird", {}), raw.get("geometry", {})
    if not isinstance(bird, dict) or not isinstance(geometry, dict):
        return None
    frame = int(finite(raw.get("frame")))
    bx = finite(bird.get("x"), 57)
    bw = finite(bird.get("width"), 34)
    bh = finite(bird.get("height"), 24)
    pw = finite(geometry.get("pipe_width"), 52)
    ground = finite(geometry.get("ground_y"), 404.48)
    speed = finite(geometry.get("pipe_speed_px_per_frame"), 4)
    height = finite(geometry.get("screen_height"), 512)
    if min(bw, bh, pw, speed) <= 0 or not 40 < ground < 2048:
        return None
    pipes, seen = [], set()
    for index in (0, 3, 6):
        x, top, bottom = f[index:index + 3]
        if top <= 0 and bottom >= height - 1:
            continue  # Offscreen placeholder, not a real giant opening.
        if bottom <= top + bh or x + pw < bx - 3:
            continue
        identity = (round(x + speed * frame, 5), top, bottom)
        if identity in seen:
            continue
        seen.add(identity)
        pipes.append(Pipe(x, top, bottom, identity[0]))
    pipes.sort(key=lambda p: p.x)
    action = raw.get("previous_action", "IDLE")
    action = action.upper() if isinstance(action, str) else "UNKNOWN"
    return State(frame, f[9], f[10], bx, bw, bh, pw, ground, speed,
                 tuple(pipes), action, finite(raw.get("score")),
                 bool(raw.get("done") or raw.get("terminated") or raw.get("truncated")))


def band_bits(low, high, limit):
    low = max(0, int(math.ceil(low - 1e-7)))
    high = min(limit, int(math.floor(high + 1e-7)))
    return 0 if high < low else ((1 << (high - low + 1)) - 1) << low


def preimage(bits, dy, mask):
    return bits >> dy if dy >= 0 else (bits << -dy) & mask


class PhysicsEvidence:
    """Check actually applied transitions, not requested or predicted actions."""

    def __init__(self):
        self.order = "gravity_first"
        self.previous = None
        self.matches = 0
        self.mismatches = 0
        self.other_votes = 0
        self.history = deque(maxlen=180)
        self.examples = deque(maxlen=6)

    def step(self, y, v, action):
        nv = -9 if action == "FLAP" else min(v + 1, 10)
        dy = nv if action == "FLAP" or self.order == "gravity_first" else v
        return y + dy, nv

    def observe(self, s):
        old, self.previous = self.previous, s
        if old is None or s.frame - old.frame != 1 or s.applied not in ACTIONS:
            return False
        if old.y < 1 or s.y + s.bh >= s.ground - 1:
            return False
        y, v = self.step(old.y, old.v, s.applied)
        error = abs(s.y - y) + abs(s.v - v)
        mismatch = error > 0.05
        self.history.append((s.frame, mismatch))
        if not mismatch:
            self.matches += 1
            self.other_votes = 0
            return False
        self.mismatches += 1
        self.examples.append({"frame": s.frame, "applied": s.applied,
                              "predicted_y": y, "observed_y": s.y,
                              "predicted_v": v, "observed_v": s.v})
        alternate_dy = old.v if self.order == "gravity_first" else v
        alternate_matches = (s.applied == "IDLE" and
                             abs(old.y + alternate_dy - s.y) + abs(v - s.v) < 0.05)
        self.other_votes = self.other_votes + 1 if alternate_matches else 0
        if self.other_votes >= 3:
            self.order = ("velocity_first" if self.order == "gravity_first"
                          else "gravity_first")
            self.other_votes = 0
            return True
        return False

    def recent_mismatches(self, frame):
        return sum(bad for f, bad in self.history if f >= frame - 120)


class Certificate:
    """All reachable y/v states at each future frame, not an open-loop path.

    A layer contains twenty integer bitsets. Its contract is existential:
    each bit has at least one safe continuation through every observed pipe.
    Changing the bird's velocity does not invalidate a layer; changing public
    geometry, the integration convention, or its vertical lattice does.
    """

    def __init__(self, s, order, margin):
        self.frame = s.frame
        self.layout = s.layout
        self.geometry = s.geometry
        self.order = order
        self.margin = margin
        self.fraction = s.y - math.floor(s.y)
        self.maximum = int(math.floor(s.ground - s.bh - 1 - self.fraction))
        self.layers = []
        self.horizon = 0
        if not 1 <= self.maximum <= 2048:
            return
        last_exit = max(((p.x + s.pw - s.bx + 3) / s.speed for p in s.pipes), default=40)
        self.horizon = min(100, max(56, int(math.ceil(last_exit)) + 20))
        full = (1 << (self.maximum + 1)) - 1
        bands = []
        for t in range(self.horizon + 1):
            lower, upper = 2.0, s.ground - s.bh - 4.0
            for p in s.pipes:
                x = p.x - s.speed * t
                if x < s.bx + s.bw + 2 and x + s.pw > s.bx - 2:
                    lower = max(lower, p.top + margin)
                    upper = min(upper, p.bottom - s.bh - margin)
            bands.append(band_bits(lower - self.fraction, upper - self.fraction, self.maximum))
        terminal = []
        for v in VELOCITIES:
            rise = (-v) * (-v - 1) // 2 if v < 0 else 0
            if v < 0 and order == "velocity_first":
                rise += -v
            terminal.append(bands[-1] & band_bits(2 + rise - self.fraction,
                                                  self.maximum, self.maximum))
        layers = [None] * (self.horizon + 1)
        layers[-1] = tuple(terminal)
        for t in range(self.horizon - 1, -1, -1):
            nxt = layers[t + 1]
            flap = (nxt[0] << 9) & full
            layer = []
            for v in VELOCITIES:
                nv = min(v + 1, 10)
                dy = nv if order == "gravity_first" else v
                idle = preimage(nxt[nv + 9], dy, full)
                layer.append((flap | idle) & bands[t])
            layers[t] = tuple(layer)
        self.layers = layers

    def reusable(self, s, order):
        age = s.frame - self.frame
        return (bool(self.layers) and 0 <= age <= self.horizon - 56
                and self.layout == s.layout and self.geometry == s.geometry
                and self.order == order
                and abs(self.fraction - (s.y - math.floor(s.y))) < 1e-6)

    def contains(self, frame, y, v):
        index = frame - self.frame
        yi = int(round(y - self.fraction))
        vi = int(round(v))
        if (not self.layers or not 0 <= index <= self.horizon
                or not 0 <= yi <= self.maximum or not -9 <= vi <= 10
                or abs(yi + self.fraction - y) > 0.05 or abs(vi - v) > 0.05):
            return False
        return bool(self.layers[index][vi + 9] & (1 << yi))

    def actions(self, s, physics, repeat=1):
        available = []
        for action in ACTIONS:
            y, v = s.y, s.v
            for delay in range(1, repeat + 1):
                y, v = physics.step(y, v, action)
                if not self.contains(s.frame + delay, y, v):
                    break
            else:
                available.append(action)
        return tuple(available)


class Kernel:
    """Geometry revisions compile; ordinary ticks only query an existing proof."""

    def __init__(self, physics):
        self.physics = physics
        self.certificate = None
        self.compiles = 0
        self.hits = 0
        self.relaxed = 0
        self.empty = 0

    def choose(self, s, jitter_guard=False, audit=False):
        c = self.certificate
        if not audit and c is not None and c.reusable(s, self.physics.order):
            self.hits += 1
        else:
            c = Certificate(s, self.physics.order, 10)
            self.compiles += 1
        allowed = c.actions(s, self.physics)
        # A relaxed certificate is for recovery only, never a persistent new policy.
        if not allowed:
            for margin in (6, 2, 0):
                c = Certificate(s, self.physics.order, margin)
                self.compiles += 1
                allowed = c.actions(s, self.physics)
                if allowed:
                    break
        self.certificate = c if c.margin == 10 and allowed else None
        if c.margin < 10:
            self.relaxed += 1
        if not allowed:
            self.empty += 1
        preferred = cruise(s, c.margin)
        nominal = preferred if preferred in allowed else allowed[0] if allowed else recovery(s)
        robust = c.actions(s, self.physics, repeat=2) if jitter_guard and allowed else ()
        chosen = (preferred if preferred in robust else robust[0]) if robust else nominal
        return chosen, {"allowed": allowed, "repeat_safe": robust, "margin": c.margin,
                        "horizon": c.horizon - (s.frame - c.frame),
                        "nominal": nominal, "guard_changed_action": chosen != nominal}


def cruise(s, margin):
    p = next((p for p in s.pipes if p.x + s.pw > s.bx), None)
    top, bottom = (p.top, p.bottom) if p else (125.0, min(275.0, s.ground - 30))
    anchor = min(bottom - s.bh - margin - 2, (top + bottom - s.bh) / 2 + 22)
    if s.y < 3:
        return "IDLE"
    if (s.v >= 0 and s.y >= anchor) or (s.y > anchor + 32 and s.v >= -1):
        return "FLAP"
    return "IDLE"


def recovery(s):
    if s.y < 8:
        return "IDLE"
    if s.y + s.bh > s.ground - 35:
        return "FLAP"
    return cruise(s, 0)


def answer_choice(answer, options):
    value = getattr(answer, "value", None)
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(value, str) or value not in options or not isinstance(probabilities, dict):
        return None
    probability = finite(probabilities.get(value), -1)
    confidence = getattr(answer, "confidence", None)
    threshold = 0.92 if confidence is None else 0.85
    if probability < threshold or probability > 1:
        return None
    if confidence is not None and finite(confidence, -1) < 0.40:
        return None
    return value


def bounded_json(text):
    if not isinstance(text, str) or len(text) > 12000:
        return None
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except (ValueError, RecursionError):
            continue
        return value if isinstance(value, dict) else None
    return None


class Supervisor:
    """Slow judgments attach to diagnostic incidents, never a bird coordinate."""

    def __init__(self, ctx, physics):
        self.ctx = ctx
        self.physics = physics
        self.recent = deque(maxlen=180)
        self.s1 = None
        self.s1_tag = None
        self.s2 = None
        self.s2_used = False
        self.last_request = -10000
        self.last_heartbeat = -10000
        self.guard_until = -1
        self.audit_until = -1
        self.version = 0
        self.focus = "CONTROL_INTEGRITY"
        self.stats = Counter()
        self.latencies = deque(maxlen=12)

    def record(self, frame, skipped, empty, relaxed):
        self.recent.append((frame, skipped, empty, relaxed))

    def flags(self, frame):
        recent = [r for r in self.recent if r[0] >= frame - 120]
        return {"physics_disagreement": self.physics.recent_mismatches(frame) >= 3,
                "clock_disruption": sum(r[1] for r in recent) >= 2,
                "continuation_missing": sum(r[2] for r in recent) >= 3,
                "clearance_pressure": sum(r[3] for r in recent) >= 12,
                "physics_verified": self.physics.matches >= 12}

    @staticmethod
    def incident_key(flags):
        return tuple(sorted(flags.items()))

    def guard_active(self, s):
        return s.frame <= self.guard_until and self.flags(s.frame)["clock_disruption"]

    def questions(self):
        criteria = {
            "KEEP": {"when": "Validated transitions and no active incident.",
                     "effect": "Keep certified per-tick execution; do not call Astra."},
            "CLOCK_GUARD": {"when": "Clock disruptions, but no physics disagreement or missing continuation.",
                            "effect": "Among viable actions prefer ones also safe if repeated once."},
            "AUDIT": {"when": "Repeated transition disagreement or missing continuation, or persistent clearance pressure.",
                      "effect": "Recompile from every observation; seek a slow review only for corroborated control faults."},
            "UNKNOWN": "The supplied evidence cannot establish any of these diagnoses.",
        }
        instructions = (
            "Diagnose the CONTROL INCIDENT, not a FLAP/IDLE action. Use evidence flags and "
            "their definitions in state. Prefer AUDIT for physical disagreement, missing "
            "continuation or persistent clearance pressure; otherwise CLOCK_GUARD for actual "
            "missed game ticks; otherwise KEEP when physics_verified. Use UNKNOWN if unverified. "
            "Late inference is not a missed actuator tick or a physics failure. No arithmetic. "
            "plan.focus identifies the review emphasis but does not override these criteria."
        )
        return {
            "incident": Choice(instructions, criteria),
            "incident_reversed": Choice(instructions, dict(reversed(list(criteria.items())))),
            "structural_fault": Noul(
                "Does evidence explicitly establish physics_disagreement OR continuation_missing? "
                "Inference latency, missing model confidence, and clearance pressure alone do NOT "
                "establish a structural fault. Read the already-computed flags; do not count.",
                {"true": "physics_disagreement or continuation_missing is true.",
                 "false": "Both are false; including clock-only incidents and healthy operation."}),
        }

    def state(self, s, report, flags):
        return {"task": "Diagnose controller integrity while code continues safe one-frame execution.",
                "rules": {"objective": "Pipe pairs passed, not shaped reward.",
                          "execution": "Each tick repeats the held command; FLAP must be explicitly released.",
                          "inference": "Models run off the actuator thread. Slow answers do not imply stale game ticks.",
                          "safety": "Only code-certified actions may be selected; a finite horizon is not an infinite proof."},
                "plan": {"revision": self.version, "focus": self.focus,
                         "goal": "Keep the verified controller; diagnose genuine integrity failures."},
                "evidence": dict(flags),
                "definitions": {"physics_disagreement": "At least three one-frame applied-action prediction mismatches recently.",
                                "clock_disruption": "At least two actual missed observation frames recently.",
                                "continuation_missing": "No feasible continuation even at zero extra margin on at least three recent frames.",
                                "clearance_pressure": "At least twelve recent frames needed less than ten pixels extra clearance.",
                                "physics_verified": "At least twelve observed one-frame transitions matched the public dynamics."},
                "controller": {"continuation": "AVAILABLE" if report["allowed"] else "MISSING",
                               "safe_actions": list(report["allowed"]),
                               "observation_reanchored_each_tick": True},
                "recent_mismatch_examples": list(self.physics.examples) if flags["physics_disagreement"] else []}

    def collect(self, s):
        ctx = self.ctx
        if self.s1 is not None and self.s1.done():
            future, tag = self.s1, self.s1_tag
            self.s1 = self.s1_tag = None
            try:
                answers = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                self.stats["model_failures"] += 1
                ctx.log("incident_answer_rejected", reason=type(exc).__name__)
                answers = None
            if isinstance(answers, dict):
                flags = self.flags(s.frame)
                age = s.frame - tag["frame"]
                fresh = (0 <= age <= 120 and tag["version"] == self.version
                         and tag["incident"] == self.incident_key(flags))
                self.latencies.append(max(0.0, ctx.clock.now() - tag["time"]))
                options = ("KEEP", "CLOCK_GUARD", "AUDIT", "UNKNOWN")
                a = answer_choice(answers.get("incident"), options)
                b = answer_choice(answers.get("incident_reversed"), options)
                if fresh and a == b and a not in (None, "UNKNOWN"):
                    self.stats["accepted"] += 1
                    if a == "CLOCK_GUARD" and flags["clock_disruption"]:
                        self.guard_until = s.frame + 180
                    elif a == "AUDIT" and any(flags[k] for k in
                            ("physics_disagreement", "continuation_missing", "clearance_pressure")):
                        self.audit_until = s.frame + 120
                    ctx.log("incident_advice", diagnosis=a, age_frames=age,
                            guard_until=self.guard_until, audit_until=self.audit_until)
                    structural = finite(getattr(answers.get("structural_fault"), "value", -1), -1)
                    if (a == "AUDIT" and 0.95 <= structural <= 1
                            and (flags["physics_disagreement"] or flags["continuation_missing"])):
                        self.request_astra(s, flags)
                else:
                    self.stats["rejected"] += 1
                    ctx.log("incident_answer_rejected", reason="expired_or_no_consensus", age_frames=age)
        if self.s2 is not None and self.s2.done():
            future, self.s2 = self.s2, None
            try:
                result = future.result()
            except (BudgetExceeded, ModelUnavailable):
                raise
            except Exception as exc:
                ctx.log("review_rejected", reason=type(exc).__name__)
                return
            config = bounded_json(getattr(result, "text", None))
            if (not config or not isinstance(config.get("focus"), str)
                    or config["focus"] not in ("CONTROL_INTEGRITY", "DYNAMICS", "CLOCK")
                    or config.get("cache_policy") not in ("VERIFIED_REUSE", "REBUILD_ON_OBSERVATION")):
                ctx.log("review_rejected", reason="invalid_configuration")
                return
            self.focus = config["focus"]
            self.version += 1
            flags = self.flags(s.frame)
            if (config["cache_policy"] == "REBUILD_ON_OBSERVATION"
                    and (flags["physics_disagreement"] or flags["continuation_missing"])):
                self.audit_until = max(self.audit_until, s.frame + 180)
            ctx.log("review_applied", revision=self.version, focus=self.focus,
                    cache_policy=config["cache_policy"], audit_until=self.audit_until)

    def request_astra(self, s, flags):
        remaining = self.ctx.budget.remaining()
        if (self.s2_used or remaining.get("system2_calls", 0) < 2
                or remaining.get("wall_s", 0) < 135):
            return
        evidence = {"flags": flags, "frame": s.frame, "score": s.score,
                    "dynamics_order": self.physics.order, "matched_transitions": self.physics.matches,
                    "mismatch_examples": list(self.physics.examples),
                    "model_latency_s": list(self.latencies)}
        prompt = (
            "Review a Flappy Bird controller integrity incident, not its live action. A code-owned "
            "actuator posts every tick and revokes FLAP after the applied tick. Published physics: "
            "IDLE adds gravity +1 capped at +10, then integrates; FLAP sets -9. Observed rectangles "
            "and both visible pipe corridors constrain exact finite-horizon reachability. "
            "Only public observations are available. Unknown offscreen gaps remain unknown. "
            "The code caches all viable states by future frame, not an open-loop action sequence. "
            "Geometry, fractional y lattice, integration order and horizon are checked before reuse. "
            "An error in an inference answer is NOT evidence of a dynamics failure. "
            "Your only permitted interface revision is JSON with focus CONTROL_INTEGRITY, DYNAMICS "
            "or CLOCK; cache_policy VERIFIED_REUSE or REBUILD_ON_OBSERVATION; and reason <=30 words. "
            "Prefer the current verified rules unless the evidence contradicts them. Do not change "
            "physics, safety margins, time, observations, game rules or executable code. "
            "Jev gets a self-contained state with computed evidence flags, their definitions and "
            "your focus, plus independent typed questions. Choice(instructions, criteria) returns "
            ".value, .probabilities, .confidence; Noul returns yes probability. It cannot reason "
            "over missing context, generate a plan, count events or reliably compare numbers. "
            "The incident Choices use reversed options and both must agree with probability >=.85 "
            "and confidence >=.40 (probability >=.92 if confidence is missing). AUDIT means "
            "recompile, CLOCK_GUARD means prefer actions safe if repeated once, KEEP leaves the "
            "verified controller. Noul structural_fault must be >=.95 AND code must corroborate "
            "actual repeated prediction errors or missing continuation to request you. "
            "A response expires on incident or interface revision changes or after 120 frames; "
            "no model answer can bypass the live safety certificate. Replies are bounded and "
            "late replies are not retroactively executed. Return only the small JSON object.\n"
            + json.dumps(evidence, separators=(",", ":"))
        )
        self.s2_used = True
        try:
            self.s2 = self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=120.0))
        except (BudgetExceeded, ModelUnavailable, RuntimeError, TimeoutError) as exc:
            self.s2 = None
            self.ctx.log("integrity_review_rejected", reason=type(exc).__name__)
            return
        self.ctx.log("integrity_review_requested", frame=s.frame, evidence=flags)

    def schedule(self, s, report):
        if (self.s1 is not None or s.frame < 30
                or getattr(self.ctx, "system1", True) is None):
            return
        flags = self.flags(s.frame)
        incident = any(flags[k] for k in ("physics_disagreement", "clock_disruption",
                                         "continuation_missing", "clearance_pressure"))
        # One commissioning audit, then at most once per minute when healthy.
        due = s.frame - self.last_heartbeat >= 1800
        if not due and not (incident and s.frame - self.last_request >= 180):
            return
        remaining = self.ctx.budget.remaining()
        if remaining.get("system1_calls", 0) < 8 or remaining.get("wall_s", 0) < 5:
            return
        state, questions = self.state(s, report, flags), self.questions()
        self.s1_tag = {"frame": s.frame, "time": self.ctx.clock.now(),
                       "version": self.version, "incident": self.incident_key(flags)}
        self.last_request = s.frame
        if due:
            self.last_heartbeat = s.frame
        self.stats["requests"] += 1
        try:
            self.s1 = self.ctx.spawn(lambda: self.ctx.decide(state, questions, timeout_s=3.0))
        except (BudgetExceeded, ModelUnavailable, RuntimeError, TimeoutError) as exc:
            self.s1 = self.s1_tag = None
            self.stats["model_failures"] += 1
            self.ctx.log("incident_request_rejected", reason=type(exc).__name__)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.physics = PhysicsEvidence()
        self.kernel = Kernel(self.physics)
        self.supervisor = Supervisor(ctx, self.physics)

    def run_episode(self, game):
        ctx, supervisor = self.ctx, self.supervisor
        last_frame, last_posted, last_score = -1, "IDLE", -1
        skipped_total = 0
        try:
            ctx.act("IDLE")
            while not ctx.done:
                s = decode(ctx.observe())
                if s is None:
                    ctx.act("IDLE")
                    last_posted = "IDLE"
                    ctx.wait_for_tick()
                    continue
                if s.done:
                    break
                if s.frame <= last_frame:
                    ctx.wait_for_tick()
                    continue
                # Pulse revocation precedes inference parsing, compiling and logging.
                if last_posted == "FLAP":
                    ctx.act("IDLE")
                    last_posted = "IDLE"
                skipped = max(0, s.frame - last_frame - 1) if last_frame >= 0 else 0
                skipped_total += skipped
                if self.physics.observe(s):
                    self.kernel.certificate = None
                guard = supervisor.guard_active(s)
                audit = s.frame <= supervisor.audit_until
                action, report = self.kernel.choose(s, jitter_guard=guard, audit=audit)
                supervisor.record(s.frame, skipped, not report["allowed"], report["margin"] < 10)
                latest = decode(ctx.observe())
                if latest is not None and latest.frame != s.frame:
                    # Never commit a pulse computed across a tick boundary.
                    last_frame = s.frame
                    continue
                ctx.act(action)
                last_posted, last_frame = action, s.frame
                # All discretionary work happens AFTER the fresh action is posted.
                if report["guard_changed_action"]:
                    ctx.log("jev_guard_changes_action", frame=s.frame,
                            baseline=report["nominal"], action=action)
                if s.score != last_score or s.frame % 900 == 0:
                    ctx.log("certificate_summary", frame=s.frame, score=s.score,
                            horizon=report["horizon"], margin=report["margin"],
                            allowed=list(report["allowed"]), compiles=self.kernel.compiles,
                            cache_hits=self.kernel.hits, physics_matches=self.physics.matches,
                            physics_mismatches=self.physics.mismatches, skipped_frames=skipped_total)
                    last_score = s.score
                if not ctx.done:
                    supervisor.collect(s)
                    supervisor.schedule(s, report)
                    ctx.wait_for_tick()
        except BudgetExceeded:
            return  # Budget/deadline/episode closure is a normal end.
        finally:
            ctx.memory["candidate_0006"] = {
                "cache_hits": self.kernel.hits, "compiles": self.kernel.compiles,
                "relaxed_frames": self.kernel.relaxed, "empty_frames": self.kernel.empty,
                "verified_transitions": self.physics.matches,
                "physics_mismatches": self.physics.mismatches,
                "missed_frames": skipped_total, "supervision": dict(supervisor.stats),
                "astra_requested": supervisor.s2_used,
            }


def build(ctx):
    return Harness(ctx)
