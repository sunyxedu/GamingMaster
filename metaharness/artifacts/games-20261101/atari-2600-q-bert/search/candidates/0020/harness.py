"""Candidate 0020: defer one-exit terminal cubes and advise the next hop in-flight.

The policy experiment is deliberately narrow: bottom tips (degree-one vertices)
are not coverage targets while any interior cube remains.  A predicted next-state
choice is sent to Jev during the current hop so its answer can be used *after*
landing, rather than becoming stale because Q*bert moved while it was thinking.
"""
from __future__ import annotations

import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
TIE = ("DOWN", "RIGHT", "UP", "LEFT")


def make_board():
    board = {i: {} for i in range(21)}
    row_of = {}
    for row in range(6):
        for col in range(row + 1):
            node = row * (row + 1) // 2 + col
            row_of[node] = row
            if row > 0:
                if col < row:
                    board[node]["UP"] = node - row
                if col > 0:
                    board[node]["LEFT"] = node - row - 1
            if row < 5:
                child = (row + 1) * (row + 2) // 2
                board[node]["DOWN"] = child + col
                board[node]["RIGHT"] = child + col + 1
    return board, row_of


BOARD, ROW = make_board()
TIPS = {n for n, exits in BOARD.items() if len(exits) == 1}
# For this 21-cube port, only the two bottom corners have a single graph exit.
assert TIPS == {15, 20}


def distances_from(start):
    if start not in BOARD:
        return [99] * 21
    d = [99] * 21
    d[start] = 0
    q = deque([start])
    while q:
        here = q.popleft()
        for nxt in BOARD[here].values():
            if d[nxt] == 99:
                d[nxt] = d[here] + 1
                q.append(nxt)
    return d


DIST = [distances_from(n) for n in range(21)]
COVER = (1, 4, 7, 11, 6, 10, 16, 12, 17, 18, 13, 9, 14,
         19, 20, 8, 2, 0, 3, 5, 15)
RANK = {n: i for i, n in enumerate(COVER)}


def num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    return int(num(value, default))


def node_id(value):
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if 0 <= n < 21 else None


def now_tick(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)))


def phase_info(obs):
    x = obs.get("qbert_motion")
    return x if isinstance(x, dict) else {"phase": str(x or "unknown")}


def is_standing(obs):
    n = node_id(obs.get("qbert_node"))
    m = phase_info(obs)
    p = str(m.get("phase", "unknown")).lower()
    return (n is not None and not bool(obs.get("qbert_in_hop"))
            and p not in ("hopping", "falling", "not_visible")
            and not bool(m.get("falling")))


def palette(obs):
    raw = obs.get("cube_colors")
    if not isinstance(raw, list) or len(raw) != 21:
        return None
    if any(isinstance(x, bool) or not isinstance(x, int) for x in raw):
        return None
    return tuple(raw)


def names(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [x for x in value if isinstance(x, str)]


def prob(answer, option):
    table = getattr(answer, "probabilities", None)
    return num(table.get(option), 0.0) if isinstance(table, dict) else 0.0


def answer_value(answer):
    x = getattr(answer, "value", None) if answer is not None else None
    return x if isinstance(x, str) else None


class BoardLedger:
    """Infer a useful destination palette from rewarded landings and flash tallies."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.last_palette = None
        self.last_score = None
        self.destination = None
        self.need = [1] * 21
        self.flashing = False
        self.flash_score = 0.0
        self.flash_changed_at = -999
        self.flash_origin = None
        self.epoch = 0
        self.clears = 0
        self.uniform_since = None
        self.last_progress = 0
        self.recent = deque(maxlen=8)

    def update(self, obs):
        t = now_tick(obs)
        current = palette(obs)
        score = num(obs.get("score"), self.last_score or 0.0)
        delta = score - self.last_score if self.last_score is not None else 0.0
        if current is None:
            self.last_score = score
            return False
        changed = ([i for i in range(21)
                    if self.last_palette is not None and current[i] != self.last_palette[i]])
        previous_score = score - delta

        if self.last_palette is not None and len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_score = previous_score
                self.flash_origin = node_id(obs.get("qbert_node"))
                self.ctx.log("endcap_flash_start", tick=t, score=score)
            self.flash_changed_at = t

        if self.flashing:
            quiet = t - self.flash_changed_at
            at_top = node_id(obs.get("qbert_node")) == 0
            if quiet >= 4 and (at_top or quiet >= 8):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.destination = None
                self.need = [1] * 21
                self.uniform_since = None
                self.last_progress = t
                self.recent.clear()
                self.flashing = False
                self.ctx.log("endcap_new_board", tick=t, epoch=self.epoch,
                             inferred_clears=self.clears, flash_tally=tally)
                self.last_palette, self.last_score = current, score
                return True
            self.last_palette, self.last_score = current, score
            self.need = [1] * 21
            return False

        # A +25 cube event (possibly combined with a +300 catch) calibrates D.
        if delta > 0 and int(round(delta)) % 100 == 25 and 0 < len(changed) <= 3:
            qn = node_id(obs.get("qbert_node"))
            landed = qn if qn in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                color = current[landed]
                self.destination = color
                self.last_progress = t
                self.recent.append({"tick": t, "node": landed,
                                    "color": color, "reward": delta})
                self.ctx.log("endcap_destination_learned", tick=t,
                             node=landed, palette=color, reward=delta)

        if self.destination is not None:
            self.need = [0 if c == self.destination else 1 for c in current]
        else:
            decoded = obs.get("needed_landings")
            if isinstance(decoded, list) and len(decoded) == 21:
                vals = [max(0, integer(x, 1)) for x in decoded]
                self.need = vals if any(vals) else [1] * 21
            else:
                self.need = [1] * 21

        if not any(self.need):
            if self.uniform_since is None:
                self.uniform_since = t
            elif t - self.uniform_since >= 24:
                # Some cartridge cycles have an intermediate uniform palette.
                # Probe another pass; only the actual global flash counts as a clear.
                self.destination = None
                self.need = [1] * 21
                self.uniform_since = None
                self.last_progress = t
                self.ctx.log("endcap_cycle_probe", tick=t, epoch=self.epoch)
        else:
            self.uniform_since = None

        self.last_palette, self.last_score = current, score
        return False


class ThreatMap:
    """Short-lived semantic memory; never infer a precise object from a timer byte."""

    def __init__(self):
        self.last = {}

    def update(self, obs):
        t = now_tick(obs)
        seen = set()
        for item in obs.get("hazards", []) if isinstance(obs.get("hazards"), list) else []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "hazard"))
            if kind not in ("purple_ball", "red_ball", "coily"):
                continue
            if item.get("lethal") is False:
                continue
            n = node_id(item.get("estimated_node", item.get("node")))
            if n is None:
                continue
            self.last[kind] = {"node": n, "tick": t, "visible": True,
                               "motion": str(item.get("motion", "unknown"))}
            seen.add(kind)
        for kind in list(self.last):
            age = t - integer(self.last[kind].get("tick"), t)
            if age > (8 if kind in ("purple_ball", "red_ball") else 6):
                del self.last[kind]
            else:
                self.last[kind]["visible"] = kind in seen

    def signature(self, t):
        return tuple(sorted((k, v["node"], min(3, t - v["tick"]))
                            for k, v in self.last.items()))

    def risk(self, source, destination, t):
        total = 0.0
        labels = []
        for kind, item in self.last.items():
            hn = item["node"]
            age = max(0, t - item["tick"])
            gap = DIST[destination][hn]
            source_gap = DIST[source][hn] if source is not None else 99
            if gap == 0:
                total += 7.0 if age <= 2 else 5.0
                labels.append(kind + "_on_landing")
            elif gap == 1:
                total += 1.7 if kind == "coily" else 1.2
                labels.append(kind + "_adjacent")
            if kind in ("purple_ball", "red_ball"):
                children = {v for a, v in BOARD[hn].items() if a in ("DOWN", "RIGHT")}
                if destination in children:
                    total += 1.4
                    labels.append(kind + "_down_branch")
            elif source_gap <= 1:
                total += 1.0
                labels.append("coily_takeoff_pressure")
        return total, labels

    def descriptions(self, source, t):
        out = []
        for kind, item in self.last.items():
            gap = DIST[source][item["node"]] if source is not None else 99
            out.append({"kind": kind,
                        "visibility": "rendered" if item["visible"] else "recent_memory",
                        "proximity": "same_cube" if gap == 0 else
                                     "adjacent" if gap == 1 else
                                     "near" if gap <= 2 else "far",
                        "motion": ("rolling_down" if kind in ("purple_ball", "red_ball")
                                   else "pursuing")})
        return out


class Coverage:
    """Receding-horizon coverage with conditional terminal-cube scheduling."""

    def __init__(self, ledger, threats):
        self.ledger = ledger
        self.threats = threats
        self.ctx = ledger.ctx
        self.goal = None
        self.visited = deque(maxlen=18)
        self.deferred = 0
        self.last_interior_count = None
        self.last_defer_log = -999

    def reset(self):
        self.goal = None
        self.visited.clear()
        self.deferred = 0
        self.last_interior_count = None

    def targets(self, current):
        needs = [n for n, demand in enumerate(self.ledger.need) if demand > 0]
        interior = [n for n in needs if n not in TIPS]
        if interior:
            self.deferred = sum(self.ledger.need[n] for n in needs if n in TIPS)
            return interior
        self.deferred = 0
        return needs

    def choose_goal(self, current):
        active = self.targets(current)
        if not active:
            self.goal = None
            return None
        if self.goal in active:
            return self.goal
        self.goal = min(active, key=lambda n: (DIST[current][n], RANK.get(n, 99)))
        return self.goal

    def options(self, obs, source=None, safe_only=True):
        here = node_id(obs.get("qbert_node")) if source is None else source
        if here is None:
            return []
        t = now_tick(obs)
        goal = self.choose_goal(here)
        active = self.targets(here)
        if not active:
            return []
        interior_pending = any(n not in TIPS for n in active)

        if source is None:
            if isinstance(obs.get("safe_moves"), list):
                allowed = set(names(obs.get("safe_moves")))
            else:
                allowed = set(names(obs.get("legal_moves")))
        else:
            # For the expected landing, geometry is known but a future safe_moves
            # report is not; action legality is checked again after the landing.
            allowed = set(BOARD[here])

        options = []
        visits = Counter(self.visited)
        for action in TIE:
            if action not in allowed or action not in BOARD[here]:
                continue
            dest = BOARD[here][action]
            endpoint_blocked = dest in TIPS and interior_pending
            if safe_only and endpoint_blocked:
                if t - getattr(self, "last_defer_log", -999) >= 30:
                    self.last_defer_log = t
                    self.ctx.log("endcap_deferred", tick=t, from_node=here,
                                 tip=dest, target=goal,
                                 interior_pending=sum(1 for n, d in enumerate(self.ledger.need)
                                                      if d > 0 and n not in TIPS))
                continue
            risk, reasons = self.threats.risk(here, dest, t)
            exits = 0
            for out_action, nxt in BOARD[dest].items():
                if nxt == here:
                    continue
                rr, _ = self.threats.risk(dest, nxt, t + 2)
                if rr < 3.0:
                    exits += 1
            dist_goal = DIST[dest][goal] if goal is not None else 0
            useful = self.ledger.need[dest] > 0
            # Endpoint work is deferred only while other useful cubes remain.
            # Do not add a blanket corner penalty to an otherwise safe route.
            cost = 2.0 * dist_goal + 10.0 * risk
            cost += 0.55 if not useful else 0.0
            cost += 0.12 * visits[dest] - 0.20 * min(exits, 3)
            if dest == here:
                cost += 2.0
            options.append({"key": action, "action": action, "landing": dest,
                            "cost": cost, "risk": risk, "risk_facts": reasons,
                            "useful": useful, "distance": dist_goal,
                            "exits": exits,
                            "escape": "open" if exits >= 2 else "narrow" if exits == 1 else "closed",
                            "terminal_deferred": endpoint_blocked})

        if not options and safe_only:
            # An emergency exception preserves a legal escape when the adapter's
            # conservative blockers leave only a deferred endpoint edge.
            return self.options(obs, source=None, safe_only=False)
        options.sort(key=lambda x: (x["cost"], RANK.get(x["landing"], 99),
                                    TIE.index(x["action"])))
        return options

    def arrived(self, n):
        if n is not None:
            self.visited.append(n)
        if self.goal == n and self.ledger.need[n] <= 0:
            self.goal = None


class RealtimePilot:
    def __init__(self, ctx, game):
        self.ctx, self.game = ctx, game
        self.ledger = BoardLedger(ctx)
        self.threats = ThreatMap()
        self.cover = Coverage(self.ledger, self.threats)
        self.last_lives = None
        self.loss_count = None
        self.life_version = 0
        self.motor = None
        self.neutral_to = 0
        self.recover_to = 0
        self.release_to = 0
        self.s1_future = None
        self.s1_meta = None
        self.advice = None
        self.last_s1_start = -999
        try:
            self.valid_actions = set(str(x) for x in game.action_spec.names)
        except Exception:
            self.valid_actions = {"NOOP", "FIRE", *DIRECTIONS}
        self.noop = "NOOP" if "NOOP" in self.valid_actions else next(iter(self.valid_actions), "NOOP")
        self.s2_future = None
        self.s2_session = None
        self.s2_asked = False

    def ingest(self, obs):
        t = now_tick(obs)
        board_changed = self.ledger.update(obs)
        self.threats.update(obs)
        lives = integer(obs.get("lives"), self.last_lives if self.last_lives is not None else 4)
        loss_counter = integer(obs.get("life_lost_count"), self.loss_count if self.loss_count is not None else 0)
        lost = ((self.last_lives is not None and lives < self.last_lives) or
                (self.loss_count is not None and loss_counter > self.loss_count))
        if board_changed:
            self.cover.reset()
            self.motor = None
            self.advice = None
            self.neutral_to = max(self.neutral_to, t + 2)
        if lost:
            self.life_version += 1
            self.cover.goal = None
            self.motor = None
            self.advice = None
            self.recover_to = max(self.recover_to, t + 5)
            self.neutral_to = max(self.neutral_to, t + 9)
            self.ctx.log("endcap_life_reset", tick=t, lives=lives,
                         life_version=self.life_version, clears=self.ledger.clears)
        self.last_lives, self.loss_count = lives, loss_counter

    def state_for_jev(self, obs, anticipated, choices):
        here = node_id(obs.get("qbert_node"))
        t = now_tick(obs)
        return {
            "game": "Atari 2600 Q*bert, six-row 21-cube board; directions are joystick edges.",
            "decision_stage": "These alternatives are for the expected landing cube; the current hop is already underway.",
            "rule": "Never choose FIRE or a missing graph edge. Treat survival before optional bonuses.",
            "position": {"observed_node": here, "expected_landing": anticipated,
                         "motion": phase_info(obs).get("phase", "unknown")},
            "coverage": {"board_epoch": self.ledger.epoch,
                         "inferred_scoring_clears": self.ledger.clears,
                         "unfinished_cubes": sum(1 for x in self.ledger.need if x > 0),
                         "interior_unfinished": sum(1 for n, x in enumerate(self.ledger.need)
                                                     if x > 0 and n not in TIPS),
                         "deferred_one_exit_tips": [n for n in sorted(TIPS)
                                                     if self.ledger.need[n] > 0],
                         "target": self.cover.goal},
            "hazards": self.threats.descriptions(here, t),
            "choices": [{"action": c["action"],
                         "landing": "unfinished_cube" if c["useful"] else "transit_cube",
                         "lethal_risk": "concrete" if c["risk"] >= 4 else
                                        "possible" if c["risk"] >= 1 else "no_supported_contact",
                         "escape": c["escape"],
                         "route_to_target": "at_target" if c["distance"] == 0 else
                                            "one_hop" if c["distance"] == 1 else "farther",
                         "code_preferred": i == 0}
                        for i, c in enumerate(choices)],
            "fallback": choices[0]["action"] if choices else "NOOP",
        }

    def request_jev_for_landing(self, obs, landing, hypothetical):
        if (self.s1_future is not None or not hypothetical or self.ctx.done
                or getattr(self.ctx, "system1", True) is None):
            return
        t = now_tick(obs)
        if t - self.last_s1_start < 5:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 1:
                return
        except Exception:
            pass

        shown = hypothetical[:4]
        opts = [x["action"] for x in shown]
        descriptions = {}
        for c in shown:
            descriptions[c["action"]] = {
                "what": "Take this graph-legal edge after landing; code will recheck safe_moves.",
                "landing": "unfinished" if c["useful"] else "transit",
                "risk": "concrete" if c["risk"] >= 4 else "possible" if c["risk"] >= 1 else "no supported contact",
                "escape": c["escape"],
                "route": "short" if c["distance"] <= 1 else "long",
                "not_for": "inventing an edge, going off-board, or using FIRE"}
        descriptions["unknown"] = "The evidence does not distinguish the offered alternatives."
        forward = {k: descriptions[k] for k in opts + ["unknown"]}
        reverse = {k: descriptions[k] for k in reversed(opts + ["unknown"])}
        fallback = shown[0]["action"]
        state = self.state_for_jev(obs, landing, shown)
        safety_instructions = (
            "Choose the safest useful continuation from choices for expected_landing. "
            "Prefer no_supported_contact to possible or concrete risk; if comparable, prefer "
            "more escape space and then code_preferred. Do not choose an absent action. "
            "Return unknown only when the facts do not distinguish them.")
        questions = {
            "route_a": Choice(safety_instructions, forward),
            "route_b": Choice(safety_instructions, reverse),
            "fallback_danger": Noul(
                "Do the stated facts support concrete lethal contact on the fallback choice? "
                "A distant or uncertain hazard is not enough.",
                {"true": "The fallback is marked concrete risk or an enemy occupies its landing.",
                 "false": "The fallback has no supported contact, or only possible/distant risk."}),
        }
        meta = {"tick": t, "expected": landing, "epoch": self.ledger.epoch,
                "life": self.life_version, "keys": opts, "fallback": fallback,
                "target": self.cover.goal, "hazards": self.threats.signature(t)}
        self.last_s1_start = t

        def work():
            try:
                return self.ctx.decide(state, questions, timeout_s=0.9)
            except Exception as exc:
                return {"_error": type(exc).__name__}

        try:
            self.s1_future = self.ctx.spawn(work)
            self.s1_meta = meta
        except Exception as exc:
            self.s1_future = None
            self.s1_meta = None
            self.ctx.log("endcap_jev_spawn_error", error=type(exc).__name__)

    def collect_jev(self, obs):
        if self.s1_future is None or not self.s1_future.done():
            return
        future, meta = self.s1_future, self.s1_meta
        self.s1_future, self.s1_meta = None, None
        try:
            answers = future.result()
            if not isinstance(answers, dict) or not isinstance(meta, dict):
                return
            a, b = answer_value(answers.get("route_a")), answer_value(answers.get("route_b"))
            agree = a == b and a in meta.get("keys", [])
            confidence = min(prob(answers.get("route_a"), a),
                             prob(answers.get("route_b"), b)) if agree else 0.0
            danger = num(getattr(answers.get("fallback_danger"), "value", 0.0), 0.0)
            self.advice = {**meta, "action": a if agree and confidence >= 0.62 else None,
                           "probability": confidence, "danger": danger}
        except Exception as exc:
            self.ctx.log("endcap_jev_result_error", error=type(exc).__name__)

    def apply_jev(self, obs, options):
        if not options:
            return None
        fallback = options[0]
        advice = self.advice
        self.advice = None
        if not isinstance(advice, dict):
            return fallback
        t = now_tick(obs)
        current = node_id(obs.get("qbert_node"))
        if (advice.get("expected") != current
                or advice.get("epoch") != self.ledger.epoch
                or advice.get("life") != self.life_version
                or t - integer(advice.get("tick"), t) < 0
                or t - integer(advice.get("tick"), t) > 14):
            self.ctx.log("endcap_jev_rejected", tick=t, reason="state_moved_or_old",
                         expected=advice.get("expected"), actual=current)
            return fallback
        if advice.get("target") != self.cover.goal:
            # The requested landing may itself have completed the previous
            # target. The actions below are rescored against today's target.
            self.ctx.log("endcap_jev_target_replanned", tick=t,
                         requested=advice.get("target"), current=self.cover.goal)
        choice = next((x for x in options if x["action"] == advice.get("action")), None)
        if choice is not None:
            if (choice["risk"] <= fallback["risk"] + 0.35
                    and choice["cost"] <= fallback["cost"] + 2.0
                    and not (choice["landing"] in TIPS
                             and any(d > 0 and n not in TIPS
                                     for n, d in enumerate(self.ledger.need))
                             and not choice.get("stall_release"))):
                if choice["action"] != fallback["action"]:
                    self.ctx.log("endcap_jev_changed_route", tick=t,
                                 old=fallback["action"], new=choice["action"],
                                 p=advice.get("probability"))
                return choice
        if advice.get("danger", 0.0) >= 0.82 and len(options) > 1:
            alternative = next((x for x in options[1:]
                                if x["risk"] + 0.2 < fallback["risk"]), None)
            if alternative is not None:
                self.ctx.log("endcap_jev_veto", tick=t, old=fallback["action"],
                             new=alternative["action"])
                return alternative
        return fallback

    def maybe_ask_astra(self, obs):
        # S2 is intentionally an offline policy auditor, not a game-clock
        # dependency. One compact question is enough to test a possible revision.
        if self.s2_asked or self.s2_future is not None or self.ledger.clears < 2 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            self.s2_asked = True
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                self.s2_asked = True
                return
        except Exception:
            pass
        summary = {
            "game": "Atari 2600 Q*bert, 21 graph nodes; endpoint policy defers nodes 15 and 20 while any interior need remains.",
            "tick": now_tick(obs), "epoch": self.ledger.epoch,
            "inferred_clears": self.ledger.clears,
            "unfinished": sum(x > 0 for x in self.ledger.need),
            "deferred_tips": [n for n in sorted(TIPS) if self.ledger.need[n] > 0],
            "interior_unfinished": sum(x > 0 for n, x in enumerate(self.ledger.need) if n not in TIPS),
            "recent_landing_evidence": list(self.ledger.recent),
            "lives": self.last_lives,
            "constraints": "Do not invent routes, controls, disc exits, or RAM semantics. The code tests strict conditional deferral; keep any revision conditional and survival-first."}
        prompt = (
            "You are the slow strategic auditor for a real-time Atari 2600 Q*bert harness. "
            "Code already controls the joystick and enforces a test: defer the two degree-one "
            "bottom tips only while other unfinished cubes exist, then visit tips when they are "
            "the remaining work. Its motor replans after every hop. Decide whether to retain this "
            "policy or recommend relaxing it only after a real interior-progress stall. Choose "
            "exactly one: keep_conditional_deferral or relax_only_after_stall. Do not provide an "
            "action or route. Return ONLY JSON with field policy and one allowed value; invalid "
            "text is ignored. The result may change only the stalled-board fallback, never the "
            "normal opening. State: " + str(summary))
        self.s2_asked = True
        session = self.s2_session

        def work():
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=100.0)
                return {"text": getattr(result, "text", ""),
                        "session_id": getattr(result, "session_id", None)}
            except Exception as exc:
                return {"text": "", "error": type(exc).__name__}

        try:
            self.s2_future = self.ctx.spawn(work)
        except Exception as exc:
            self.s2_future = None
            self.ctx.log("endcap_astra_spawn_error", error=type(exc).__name__)

    def collect_astra(self, obs):
        if self.s2_future is None or not self.s2_future.done():
            return
        future, self.s2_future = self.s2_future, None
        try:
            result = future.result()
            if isinstance(result, dict) and result.get("session_id"):
                self.s2_session = result["session_id"]
            text = result.get("text", "") if isinstance(result, dict) else ""
            # Parse the first object safely; arbitrary preflight prose is inert.
            policy = None
            if isinstance(text, str):
                import json
                decoder = json.JSONDecoder()
                for i, char in enumerate(text[:10000]):
                    if char == "{":
                        try:
                            obj, _ = decoder.raw_decode(text[i:])
                        except (ValueError, TypeError):
                            continue
                        if isinstance(obj, dict):
                            policy = obj.get("policy")
                            break
            if policy in ("keep_conditional_deferral", "relax_only_after_stall"):
                self.astra_policy = policy
                self.ctx.log("endcap_astra_policy", tick=now_tick(obs), policy=policy)
        except Exception as exc:
            self.ctx.log("endcap_astra_result_error", error=type(exc).__name__)

    def maybe_release_tip_on_stall(self, options, obs):
        """Only a valid Astra revision can relax the rule, and only after a stall."""
        policy = getattr(self, "astra_policy", "keep_conditional_deferral")
        if policy != "relax_only_after_stall" or not options:
            return options
        t = now_tick(obs)
        interior = any(v > 0 and n not in TIPS for n, v in enumerate(self.ledger.need))
        if not interior or t - self.ledger.last_progress < 90:
            return options
        current = node_id(obs.get("qbert_node"))
        if current is None:
            return options
        safe = set(names(obs.get("safe_moves")))
        for action in TIE:
            dest = BOARD[current].get(action)
            if action in safe and dest in TIPS and self.ledger.need[dest] > 0:
                risk, labels = self.threats.risk(current, dest, t)
                # Never relax into a represented near lethal hazard. This is
                # explicitly a stall escape, not a blanket corner penalty reversal.
                if risk < 1.0:
                    revised = list(options)
                    alt = {"key": action, "action": action, "landing": dest,
                           "cost": options[0]["cost"] + 1.0, "risk": risk,
                           "risk_facts": labels, "useful": True, "distance": 0,
                           "exits": 0, "escape": "closed", "terminal_deferred": False,
                           "stall_release": True}
                    revised.insert(0, alt)
                    self.ctx.log("endcap_stall_release", tick=t, node=dest,
                                 interior_pending=True, stall=t-self.ledger.last_progress)
                    return revised
        return options

    def action(self, obs):
        t = now_tick(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)
        if self.ledger.flashing:
            self.motor = None
            return self.noop
        n = node_id(obs.get("qbert_node"))
        motion = phase_info(obs)
        phase = str(motion.get("phase", "unknown")).lower()
        if n is None or phase in ("falling", "not_visible") or bool(motion.get("falling")):
            self.motor = None
            return self.noop
        if t < self.recover_to or t < self.neutral_to or t < self.release_to:
            return self.noop

        if self.motor is not None:
            m = self.motor
            age = t - m["start"]
            if bool(obs.get("qbert_in_hop")) or n != m["origin"]:
                m["moving"] = True
            if m["moving"] and is_standing(obs):
                self.motor = None
                self.cover.arrived(n)
                if n == m["destination"]:
                    self.ctx.log("endcap_landing", tick=t, origin=m["origin"], node=n,
                                 duration=age)
                else:
                    self.advice = None
                    self.ctx.log("endcap_landing_mismatch", tick=t,
                                 expected=m["destination"], observed=n)
                self.release_to = t + 1
                return self.noop
            if not m["moving"] and is_standing(obs) and age <= 3 and m["posts"] < 2:
                m["posts"] += 1
                return m["action"]
            if age > 18 and is_standing(obs):
                self.motor = None
                self.release_to = t + 2
                self.ctx.log("endcap_hop_not_started", tick=t, origin=m["origin"])
                return self.noop
            return self.noop

        if not is_standing(obs):
            return self.noop
        options = self.cover.options(obs)
        if not options:
            return self.noop
        options = self.maybe_release_tip_on_stall(options, obs)
        selected = self.apply_jev(obs, options)
        if selected is None:
            return self.noop
        action = selected["action"]
        safe = names(obs.get("safe_moves"))
        if action not in DIRECTIONS or action not in safe or action not in BOARD[n]:
            return self.noop
        dest = BOARD[n][action]
        next_options = self.cover.options(obs, source=dest, safe_only=True)
        self.request_jev_for_landing(obs, dest, next_options)
        self.motor = {"origin": n, "destination": dest, "action": action,
                      "start": t, "posts": 1, "moving": False}
        if dest in TIPS and any(v > 0 and i not in TIPS for i, v in enumerate(self.ledger.need)):
            # This can occur only through the explicit no-options emergency path.
            self.ctx.log("endcap_emergency_tip_entry", tick=t, node=dest,
                         remaining_interior=sum(v > 0 for i, v in enumerate(self.ledger.need) if i not in TIPS))
        if t % 35 == 0:
            self.ctx.log("endcap_route", tick=t, node=n, action=action, landing=dest,
                         goal=self.cover.goal, tips_pending=sorted(x for x in TIPS if self.ledger.need[x] > 0),
                         interior_pending=sum(v > 0 for i, v in enumerate(self.ledger.need) if i not in TIPS),
                         clears=self.ledger.clears)
        return action

    def run(self):
        ctx = self.ctx
        obs = ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.ingest(obs)
        while not ctx.done and not getattr(self.game, "done", False):
            obs = ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.ingest(obs)
            self.maybe_ask_astra(obs)
            action = self.action(obs)
            if action not in self.valid_actions:
                action = self.noop if self.noop in self.valid_actions else "NOOP"
            ctx.act(action)
            ctx.wait_for_tick()
        self.ctx.log("endcap_summary", score=getattr(self.game, "score", None),
                     inferred_clears=self.ledger.clears, epoch=self.ledger.epoch,
                     lives=self.last_lives, pending_tips=sorted(x for x in TIPS if self.ledger.need[x] > 0))


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        RealtimePilot(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
