"""0008: copy-aware Frogger geometry and carrier-connected route control.

Uses only the public observation. NUSIZ is a sprite-copy layout, not a solid
width. A frog is an eight-pixel sprite rather than a point at its left edge.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul

ACTIONS = ("UP", "RIGHT", "LEFT", "DOWN", "NOOP")
# TIA copy offsets and scale, applied to both publicly decoded object tracks.
LAYOUT = {
    0: ((0,), 8), 1: ((0, 16), 8), 2: ((0, 32), 8),
    3: ((0, 16, 32), 8), 4: ((0, 64), 8), 5: ((0,), 16),
    6: ((0, 32, 64), 8), 7: ((0,), 32),
}


def num(x, default=0.0):
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except (TypeError, ValueError):
        return default


def unpack(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except (ValueError, TypeError):
            return {}
    return x if isinstance(x, dict) else {}


def parse_output(text):
    if not isinstance(text, str):
        return {}
    try:
        return unpack(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, TypeError):
        return {}


def row(o):
    r = int(num(o.get("frog", {}).get("row"), 255))
    return -1 if r == 255 else r


def epoch(o):
    return (o.get("deaths", 0), o.get("frogs_home", 0), o.get("level", 1))


class Geometry:
    """An immutable public-state prediction used for one bounded search."""
    def __init__(self, o, phases, extra_margin=0):
        self.o = o
        self.frame = int(num(o.get("frame", o.get("frame_number"))))
        self.lanes = {int(l["row"]): l for l in o.get("road", []) + o.get("river", [])}
        self.phases = dict(phases)
        self.extra = extra_margin
        self.cache = {}
        self.homes = [num(h.get("x_center")) for h in o.get("homes", [])
                      if not h.get("occupied") and h.get("available", True)
                      and h.get("crocodile") not in (True, "present", "dangerous")]
        self.wrap = int(num(o.get("difficulty"))) == 1

    def drift(self, r, dt):
        l = self.lanes.get(r)
        if not l or r < 6:
            return 0.0
        p = max(1, int(num(l.get("activity_mask"), 3)) + 1)
        s = -1 if l.get("direction") == "left" else 1
        phase = self.phases.get(r)
        if phase is None:
            return s * dt / p
        return s * ((self.frame + dt - phase) // p - (self.frame - phase) // p)

    def intervals(self, r, dt):
        key = (r, dt)
        if key in self.cache:
            return self.cache[key]
        l = self.lanes.get(r, {})
        p = max(1, int(num(l.get("activity_mask"), 3)) + 1)
        s = -1 if l.get("direction") == "left" else 1
        phase = self.phases.get(r)
        dx = s * dt / p if phase is None else s * (
            (self.frame + dt - phase) // p - (self.frame - phase) // p)
        intervals = []
        for obj in l.get("objects", []):
            if r >= 6 and (obj.get("hazard") or obj.get("kind") not in ("log", "turtle")):
                continue
            offsets, width = LAYOUT.get(int(num(obj.get("nusiz"))) & 7, LAYOUT[0])
            ox = (num(obj.get("x")) + dx) % 160
            for offset in offsets:
                a = (ox + offset) % 160
                for shift in (-160, 0, 160):
                    intervals.append((a + shift, a + width + shift))
        # Adjacent copies from distinct tracks form a real continuous carrier.
        intervals.sort()
        merged = []
        for a, b in intervals:
            if merged and a <= merged[-1][1] + 0.01:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.cache[key] = tuple(merged)
        return self.cache[key]

    def margin(self, r, x, dt, existing_support=False):
        if not (-1 <= r <= 11) or x < 1 or x > 151:
            return -100.0
        if r in (-1, 5):
            return 25.0
        if r == 11:
            return max((4.0 - abs(x + 4 - c) for c in self.homes), default=-100.0)
        ints = self.intervals(r, dt)
        if r < 5:
            # x is sprite origin; the old parent mistakenly treated x as its centre.
            a, b = x - 1 - self.extra, x + 9 + self.extra
            clear = 100.0
            for lo, hi in ints:
                if b >= lo and a <= hi:
                    return -10.0
                clear = min(clear, lo - b if b < lo else a - hi)
            return clear
        # Archive seed107 tick593 survives with frog x147 on the end of a
        # copy spanning [140,148]: the sprite need only overlap the carrier,
        # not put its centre inside it. Require two pixels on new landings,
        # but retain one-pixel already-observed support for escape/ride checks.
        need = 1 if existing_support else 2
        return max((min(x + 8 - a, b - x) - need for a, b in ints), default=-100.0)

    def transition(self, r, x, t, action, duration=10):
        nr, nx = r, x
        if action == "UP":
            nr += 1
        elif action == "DOWN":
            nr -= 1
        elif action == "LEFT":
            nx -= 8
        elif action == "RIGHT":
            nx += 8
        if nr < -1 or nr > 11:
            return None
        # Input is sampled after one to three frames. Protect the source until
        # then; protect the destination through the collision/release window.
        if action != "NOOP":
            src = min(self.margin(r, x + self.drift(r, t + s) - self.drift(r, t), t + s,
                                  existing_support=True)
                      for s in (1, 3))
            if src < 0:
                return None
        else:
            src = 100.0
        samples = (1, 3, 6, duration)
        margin = src
        for s in samples:
            xx = nx + self.drift(nr, t + s) - self.drift(nr, t if action == "NOOP" else t + 1)
            margin = min(margin, self.margin(nr, xx, t + s, existing_support=action == "NOOP"))
        if margin < 0:
            return None
        endx = nx + self.drift(nr, t + duration) - self.drift(nr, t if action == "NOOP" else t + 1)
        return nr, endx, t + duration, margin


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.phases = {}
        self.last_epoch = None
        self.press = None
        self.press_origin = None
        self.press_tick = -100
        self.release_until = 0
        self.next_hop = 0
        self.hop_frames = 10
        self.last_action = "NOOP"
        self.last_candidates = []
        self.extra_margin = 0
        self.last_progress = 0
        self.history = deque(maxlen=10)
        self.phase_version = 0
        self.policy = {"home_order": "nearest", "style": "connected", "uncertainty": "strict"}
        self.jev = None
        self.jev_meta = None
        self.advice = None
        self.next_jev = 0
        self.astra = None
        self.astra_meta = None
        self.next_astra = 0
        self.session = None
        self.last_evidence = -1
        self.target = None
        self.certificate = None

    def log(self, name, **fields):
        self.ctx.log(name, **fields)

    def update(self, o):
        tick = int(num(o.get("tick")))
        old = self.previous
        ep = epoch(o)
        if ep != self.last_epoch:
            if self.last_epoch is not None:
                death = ep[0] > self.last_epoch[0]
                event = {"tick": tick, "before": self.last_epoch, "after": ep,
                         "frog_before": old.get("frog") if old else None,
                         "last_action": self.last_action, "certificate": self.certificate}
                self.history.append(event)
                self.log("outcome", **event)
                # A contradicted collision certificate gets a deterministic
                # geometric margin increase, not a stale model direction.
                if death and old and row(old) < 5 and row(old) >= 0:
                    self.extra_margin = min(4, self.extra_margin + 1)
            self.last_epoch = ep
            self.press = None
            self.advice = None
            self.target = None
            self.phase_version += 1
            self.release_until = tick + 4
            self.next_hop = tick + 4
            self.last_progress = tick
        if old:
            if row(o) != row(old) or o.get("score") != old.get("score"):
                self.last_progress = tick
                self.log("progress", tick=tick, frog=o.get("frog"), score=o.get("score"),
                         deaths=o.get("deaths"), homes=o.get("frogs_home"))
            old_lanes = {int(l["row"]): l for l in old.get("road", []) + old.get("river", [])}
            if tick - int(num(old.get("tick"))) == 1:
                for l in o.get("road", []) + o.get("river", []):
                    r = int(l["row"])
                    prev = old_lanes.get(r, {}).get("objects", [])
                    cur = l.get("objects", [])
                    p = max(1, int(num(l.get("activity_mask"), 3)) + 1)
                    if prev and cur and prev[0].get("x") != cur[0].get("x"):
                        self.phases[r] = int(num(o.get("frame", o.get("frame_number")))) % p
        self.previous = o

    def score_node(self, r, x, margin, cost, target, reached):
        if r >= 11:
            return 3000 - cost + min(margin, 8)
        # Reaching the next safe bank is a route-level objective. On the river,
        # keep forward routes but penalize traversing fragmented carriers.
        value = r * 45 + min(margin, 12) * 0.6 - cost
        if r == 5:
            value += 8
        if r >= 6:
            value -= abs(x + 4 - target) * (0.25 if r < 9 else 0.6)
        return value

    def rank(self, o):
        g = Geometry(o, self.phases, self.extra_margin)
        r, x = row(o), num(o.get("frog", {}).get("x"), 80)
        homes = g.homes
        if not homes:
            return [], g
        if self.target not in homes:
            if self.policy["home_order"] == "left":
                self.target = min(homes)
            elif self.policy["home_order"] == "right":
                self.target = max(homes)
            else:
                self.target = min(homes, key=lambda c: abs(x + 4 - c))
        target = self.target
        roots = {}
        frontier = []
        for action in ACTIONS:
            z = g.transition(r, x, 0, action, self.hop_frames)
            if z:
                rr, xx, tt, mm = z
                root = {"action": action, "margin": round(mm, 2), "landing_row": rr,
                        "landing_x": round(xx, 2), "advances": rr > r,
                        "toward_home": abs(xx + 4 - target) < abs(x + 4 - target),
                        "value": -1e9, "reachable_row": rr, "path": [action]}
                roots[action] = root
                frontier.append((rr, xx, tt, mm, 0.0, action, [action]))
        # A shared beam, unlike separate shallow beams for each first action.
        # Keep first-action diversity; all search is over public geometry only.
        deadline = time.monotonic() + 0.008
        for depth in range(10):
            nxt = []
            for rr, xx, tt, mm, cost, first, path in frontier:
                value = self.score_node(rr, xx, mm, cost, target, r)
                if value > roots[first]["value"]:
                    roots[first].update(value=value, reachable_row=rr, path=path)
                if depth == 9 or rr >= 11:
                    continue
                for action in ACTIONS:
                    z = g.transition(rr, xx, tt, action, self.hop_frames)
                    if z is None:
                        continue
                    r2, x2, t2, m2 = z
                    penalty = {"UP": 0.5, "LEFT": 2.0, "RIGHT": 2.0, "DOWN": 5.0, "NOOP": 1.2}[action]
                    if rr >= 6 and action in ("LEFT", "RIGHT"):
                        penalty += 1.5
                    nxt.append((r2, x2, t2, min(mm, m2), cost + penalty, first, path + [action]))
            if time.monotonic() > deadline:
                break
            nxt.sort(key=lambda n: self.score_node(n[0], n[1], n[3], n[4], target, r), reverse=True)
            frontier = []
            seen = set()
            per_first = {}
            for n in nxt:
                key = (n[0], round(n[1] / 2), n[5])
                if key in seen or per_first.get(n[5], 0) >= 8:
                    continue
                seen.add(key)
                per_first[n[5]] = per_first.get(n[5], 0) + 1
                frontier.append(n)
                if len(frontier) >= 28:
                    break
            if not frontier:
                break
        result = list(roots.values())
        for c in result:
            c["value"] += (0.5 if c["advances"] else 0) - (0.3 if c["action"] == "NOOP" else 0)
            c["surface"] = "interior" if c["margin"] >= 3 else "edge"
            c["route"] = "home_reached" if c["reachable_row"] >= 11 else "forward_connected" if c["reachable_row"] > r else "holding_or_repositioning"
        result.sort(key=lambda c: c["value"], reverse=True)
        self.last_candidates = result
        return result, g

    def ask_jev(self, o, candidates):
        ctx = self.ctx
        tick = int(num(o.get("tick")))
        if self.jev is not None or not candidates or tick < self.next_jev:
            return
        if getattr(ctx, "system1", True) is None or ctx.budget.remaining().get("system1_calls", 0) < 2:
            return
        # Semantic leases outlive a particular action/coordinate. Jev assesses
        # the route class, not the time-critical next joystick sample.
        facts = [{"action": c["action"], "surface": c["surface"], "route": c["route"],
                  "advances": c["advances"], "toward_home": c["toward_home"]} for c in candidates[:3]]
        wait = next((c for c in candidates if c["action"] == "NOOP"), None)
        state = {"rules": "Frogger: road cars kill; water between repeated turtle sprites is not a carrier. Code has expanded copy layout and validated candidate transitions. Preserve connected forward routes, avoid sideways turtle-gap jumps, and fill distinct open homes. Do not calculate geometry.",
                 "policy": dict(self.policy), "zone": o.get("frog", {}).get("zone"),
                 "candidates": facts,
                 "wait_supported": wait is not None,
                 "forward_route_exists": any(c["reachable_row"] > row(o) for c in candidates),
                 "home_route_exists": any(c["reachable_row"] >= 11 for c in candidates),
                 "timer": "urgent" if num(o.get("timer_s"), 30) < 7 else "normal",
                 "recent_certificate_failure": bool(self.history and self.history[-1]["after"][0] > self.history[-1]["before"][0]),
                 "unknown": "Home crocodile is not separately exposed; diving and unfamiliar carriers are excluded."}
        opts = {
            "connected": "Favor a geometrically certified route that makes forward progress; avoid unsupported sideways movement.",
            "interior": "Among near-equivalent routes prefer a wider interior landing rather than an edge.",
            "align": "At the home approach prefer alignment with an open home over further lateral wandering.",
            "unknown": "The supplied facts do not support a route preference.",
        }
        instructions = "Choose the most appropriate durable route preference using policy, zone and candidates. Prefer connected when a forward route exists; interior when holding/repositioning near an edge; align only at home approach. No arithmetic or exact action timing is requested."
        questions = {"route": Choice(instructions, opts),
                     "verify": Choice(instructions, dict(reversed(list(opts.items())))),
                     "wait_veto": Noul("Do the facts explicitly say wait_supported is false? True only for an unsupported wait, not simply because this is a river.")}
        self.jev_meta = {"epoch": epoch(o), "version": self.phase_version, "tick": tick,
                         "zone": o.get("frog", {}).get("zone")}
        def request():
            try:
                start = ctx.clock.now()
                return ctx.decide(state, questions, timeout_s=3.0), ctx.clock.now() - start
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError, TypeError):
                return None, 0
        self.jev = ctx.spawn(request)
        self.next_jev = tick + 60

    def ask_astra(self, o):
        ctx = self.ctx
        tick = int(num(o.get("tick")))
        if self.astra is not None or tick < self.next_astra or getattr(ctx, "system2", True) is None:
            return
        if ctx.budget.remaining().get("system2_calls", 0) < 2:
            return
        evidence = len(self.history)
        if tick > 0 and evidence == self.last_evidence and tick - self.last_progress < 600:
            return
        self.last_evidence = evidence
        summary = {"frog": o.get("frog"), "homes": o.get("homes"), "timer": o.get("timer_s"),
                   "score": o.get("score"), "policy": self.policy,
                   "evidence": list(self.history), "extra_road_margin": self.extra_margin}
        prompt = (
            "You are Frogger's slow route adviser, concurrent with an active frame controller. "
            "Do not give an immediate action. The parent mistook NUSIZ repeated sprites for solid rectangles: "
            "in seed109 it landed x96 on road row2 with object base64,NUSIZ2, then died; "
            "in seed101 it jumped left on river row7 with predicted 10.5px clearance and died. "
            "New code expands all sprite copies, models frog width and checks source/landing/release windows. "
            "Choose a durable home order and route bias. Return ONLY tiny JSON: "
            '{"home_order":"nearest|left|right","style":"connected|interior|align","uncertainty":"strict"}. '
            "The pipes show enum alternatives; choose one word each. Survival and five homes beat bonuses. "
            "The fixed Jev interface carries rules, your policy, zone, precomputed candidate surface/route/"
            "advance/alignment facts, safe-wait predicate, timer bucket and certificate-failure evidence. "
            "Jev independently selects connected/interior/align/unknown twice with reversed option order, "
            "plus an unsupported-wait Noul. Agreeing answers with both top probabilities >=.65 set a "
            "same-zone/life/plan semantic lease for at most 90 ticks. Geometry is always rechecked and "
            "cannot be overridden. You may revise route preference after certificate failures, stalls or "
            "home completion; do not invent new mechanics from a stale state. Snapshot=" + json.dumps(summary, separators=(",", ":")))
        self.astra_meta = {"level": o.get("level", 1)}
        session = self.session
        def request():
            try:
                return ctx.think(prompt, session_id=session, timeout_s=120.0)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                raise
            except (TimeoutError, RuntimeError):
                return None
        self.astra = ctx.spawn(request)
        self.next_astra = tick + 3600

    def poll(self, o):
        tick = int(num(o.get("tick")))
        if self.astra is not None and self.astra.done():
            f, self.astra = self.astra, None
            result = f.result()
            if result:
                self.session = getattr(result, "session_id", None)
                parsed = parse_output(getattr(result, "text", ""))
                allowed = {"home_order": ("nearest", "left", "right"),
                           "style": ("connected", "interior", "align"), "uncertainty": ("strict",)}
                changes = {k: v for k, v in parsed.items() if k in allowed and isinstance(v, str) and v in allowed[k]}
                if changes and o.get("level", 1) == self.astra_meta["level"]:
                    self.policy.update(changes)
                    self.phase_version += 1
                    self.target = None
                    self.advice = None
                    self.log("astra_policy_adopted", tick=tick, policy=dict(self.policy))
                else:
                    self.log("astra_output_rejected", tick=tick)
        if self.jev is not None and self.jev.done():
            f, self.jev = self.jev, None
            answers, latency = f.result()
            if not isinstance(answers, dict):
                return
            a, b = answers.get("route"), answers.get("verify")
            val = getattr(a, "value", None)
            p1 = getattr(a, "probabilities", {}) or {}
            p2 = getattr(b, "probabilities", {}) or {}
            prob = min(num(p1.get(val)), num(p2.get(val))) if isinstance(val, str) else 0
            meta = self.jev_meta
            fresh = (epoch(o) == meta["epoch"] and self.phase_version == meta["version"]
                     and o.get("frog", {}).get("zone") == meta["zone"] and tick - meta["tick"] <= 90)
            if fresh and val == getattr(b, "value", None) and val in ("connected", "interior", "align") and prob >= 0.65:
                self.advice = {"style": val, "until": meta["tick"] + 90, "epoch": meta["epoch"],
                               "version": meta["version"], "zone": meta["zone"]}
                self.log("jev_lease_accepted", tick=tick, style=val, probability=prob, latency_s=latency)
            else:
                self.log("jev_lease_discarded", tick=tick, fresh=fresh, probability=prob, latency_s=latency)

    def motor(self, o, candidates, g):
        tick = int(num(o.get("tick")))
        r, x = row(o), num(o.get("frog", {}).get("x"), 80)
        # state 8 was repeatedly witnessed in death/home animations, where
        # inputs do not move the active frog. Do not churn pulses there.
        state = int(num(o.get("frog", {}).get("state"), 128))
        if state != 128 or r not in range(-1, 11):
            self.press = None
            self.release_until = tick + 4
            return "NOOP"
        if self.press is not None:
            rr, xx = self.press_origin
            moved = r != rr or (self.press in ("LEFT", "RIGHT") and abs(x - xx) >= 5)
            if moved or tick - self.press_tick >= 4:
                self.log("pulse_released", tick=tick, action=self.press, moved=moved,
                         frog=o.get("frog"), origin=self.press_origin)
                self.press = None
                self.release_until = tick + 3
                return "NOOP"
            return self.press
        if tick < self.release_until or tick < self.next_hop:
            return "NOOP"
        if not candidates:
            return "NOOP"
        selected = candidates[0]
        close = [c for c in candidates if c["value"] >= selected["value"] - 7
                 and c["reachable_row"] >= selected["reachable_row"]]
        style = self.policy["style"]
        adv = self.advice
        if adv and (tick <= adv["until"] and adv["epoch"] == epoch(o) and
                    adv["version"] == self.phase_version and adv["zone"] == o.get("frog", {}).get("zone")):
            style = adv["style"]
        if style == "interior" and r >= 5:
            selected = max(close, key=lambda c: (c["surface"] == "interior", c["value"]))
        elif style == "align" and r >= 8:
            selected = max(close, key=lambda c: (c["toward_home"], c["value"]))
        if selected is not candidates[0]:
            self.log("semantic_preference_changes_action", tick=tick, source="jev" if adv else "astra",
                     before=candidates[0]["action"], after=selected["action"], style=style)
        action = selected["action"]
        if action != "NOOP":
            self.press = action
            self.press_origin = (r, x)
            self.press_tick = tick
            self.next_hop = tick + self.hop_frames
            self.certificate = {"tick": tick, "origin": (r, x), "action": action,
                                "destination": (selected["landing_row"], selected["landing_x"]),
                                "margin": selected["margin"], "path": selected["path"]}
            self.log("hop_issued", tick=tick, frog=o.get("frog"), **{"certificate": self.certificate},
                     candidates=candidates[:3])
        return action

    def run_episode(self, game):
        ctx = self.ctx
        while not ctx.done:
            o = unpack(ctx.observe())
            if not o:
                ctx.act("NOOP")
                ctx.wait_for_tick()
                continue
            self.update(o)
            self.poll(o)
            tick = int(num(o.get("tick")))
            active = int(num(o.get("frog", {}).get("state"), 128)) == 128 and -1 <= row(o) <= 10
            ready = active and self.press is None and tick >= self.next_hop and tick >= self.release_until
            candidates, geometry = self.rank(o) if ready else ([], None)
            if ready:
                self.ask_jev(o, candidates)
            self.ask_astra(o)
            action = self.motor(o, candidates, geometry)
            self.last_action = action
            ctx.act(action)
            ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
