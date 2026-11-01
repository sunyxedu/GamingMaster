"""Phase-aware, round-local Q*bert controller.

The cartridge's visible palette is reused by later rounds.  This controller
therefore learns the *transition cycle* for each inferred board instead of
making a boolean "visited" map.  The motor and inference bridge remain
non-blocking: a model can advise the next hop, but code owns legality, phase
arithmetic, and the real-time clock.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
# This ordering is only a deterministic tie break.  It is not an arcade map.
ACTION_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")

GRAPH = {}
ROWS = {}
for row in range(6):
    for col in range(row + 1):
        node = row * (row + 1) // 2 + col
        ROWS[node] = row
        edges = {}
        if row and col < row:
            edges["UP"] = node - row
        if col:
            edges["LEFT"] = node - row - 1
        if row < 5:
            edges["DOWN"] = node + row + 1
            edges["RIGHT"] = node + row + 2
        GRAPH[node] = edges

DIST = []
for root in range(21):
    distance = [99] * 21
    distance[root] = 0
    queue = deque([root])
    while queue:
        here = queue.popleft()
        for there in GRAPH[here].values():
            if distance[there] == 99:
                distance[there] = distance[here] + 1
                queue.append(there)
    DIST.append(distance)


CYCLES = {
    1: ("S", "D"),
    2: ("S", "I", "D"),
    3: ("S", "D"),
    4: ("S", "I", "D", "I"),
    5: ("S", "I", "D", "S"),
}


def finite_number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    return int(finite_number(value, default))


def valid_node(value):
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 21:
        return value
    return None


def now_tick(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)))


def qmotion(obs):
    value = obs.get("qbert_motion")
    if isinstance(value, dict):
        return value
    return {"phase": str(value or "unknown")}


def is_standing(obs):
    motion = qmotion(obs)
    return (
        valid_node(obs.get("qbert_node")) is not None
        and not bool(obs.get("qbert_in_hop"))
        and motion.get("phase") not in ("hopping", "falling", "not_visible")
        and not bool(motion.get("falling"))
    )


def cube_colors(obs):
    raw = obs.get("cube_colors")
    if (
        isinstance(raw, list)
        and len(raw) == 21
        and all(isinstance(value, int) and not isinstance(value, bool) for value in raw)
    ):
        return tuple(raw)
    return None


def answer_value(answer):
    value = getattr(answer, "value", None)
    return value if isinstance(value, str) else None


def answer_probability(answer, option):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return 0.0
    return finite_number(probabilities.get(option), 0.0)


def parse_json_object(text):
    """Parse the first JSON object, tolerating arbitrary Astra output."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(text[:12000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
        except (TypeError, ValueError):
            continue
    return {}


class PhaseBook:
    """Infer a board-local S/I/D cycle from scored local landings.

    ``need[n]`` is the number of landings still required to make cube n D.
    It is deliberately derived from raw observed transitions rather than from
    the adaptor's fixed phase names, which remain stale after a palette epoch
    in the archived traces.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.score = None
        self.flashing = False
        self.flash_tick = -999
        self.flash_score = 0.0
        self.flash_origin = None
        self.clears = 0
        self.epoch = 0
        self.start_color = None
        self.intermediate_color = None
        self.destination_color = None
        self.transitions = []
        self.evidence = deque(maxlen=10)
        self.need = [1] * 21
        self.last_progress = 0

    @property
    def level(self):
        # Four boards per level; level 5 repeats thereafter.
        return min(5, self.clears // 4 + 1)

    @property
    def cycle(self):
        return CYCLES[self.level]

    def event(self, name, **fields):
        self.ctx.log(name, **fields)

    def _reset_board_roles(self, raw, anchor_start=True):
        # During the global award flash the stable-looking palette is often a
        # transient animation colour.  Anchor later boards from their first
        # scored local transition instead of trusting that frame.
        self.start_color = raw[0] if raw and anchor_start else None
        self.intermediate_color = None
        self.destination_color = None
        self.transitions = []
        self.need = [1] * 21

    def _learn_transition(self, old, new, node, reward, obs):
        if old == new:
            return
        if self.start_color is None:
            self.start_color = old
            self.event("phase_start_learned", tick=now_tick(obs),
                       level=self.level, color=old, node=node)
        item = {"old": old, "new": new, "node": node,
                "tick": now_tick(obs), "reward": reward}
        self.transitions.append(item)
        self.transitions = self.transitions[-10:]
        self.evidence.append({"old": old, "new": new, "node": node,
                              "reward": reward, "level": self.level})
        cycle_level = self.level
        if cycle_level in (1, 3):
            if self.destination_color is None and old == self.start_color:
                self.destination_color = new
                self.event("phase_destination_learned", tick=now_tick(obs),
                           level=cycle_level, color=new, node=node,
                           transition=[old, new])
        else:
            if self.intermediate_color is None and old == self.start_color:
                self.intermediate_color = new
                self.event("phase_intermediate_learned", tick=now_tick(obs),
                           level=cycle_level, color=new, node=node,
                           transition=[old, new])
            elif (
                self.destination_color is None
                and self.intermediate_color is not None
                and old == self.intermediate_color
                and new not in (self.start_color, self.intermediate_color)
            ):
                self.destination_color = new
                self.event("phase_destination_learned", tick=now_tick(obs),
                           level=cycle_level, color=new, node=node,
                           transition=[old, new])

    def _required(self, color):
        """Translate a raw color into exact remaining cycle distance."""
        destination = self.destination_color
        intermediate = self.intermediate_color
        start = self.start_color
        level = self.level
        if destination is None:
            # The first S->I observation is already enough to distinguish an
            # untouched S cube (two landings) from an I cube (one).  Do not
            # wait for I->D before exposing that exact demand to the router.
            if level in (2, 4, 5):
                if color == intermediate:
                    return 1
                if color == start:
                    return 2
            return 1
        if color == destination:
            return 0
        if level in (1, 3):
            # On level 3 a D landing cycles D back to S, so do not land on D
            # unless the route needs it; all non-D observations need one pass.
            return 1
        if color == intermediate:
            return 1
        if color == start:
            return 2
        # A transient/unknown palette value is safer to treat as unfinished,
        # but not as a three-hop commitment.
        return 1

    def _recompute_need(self, raw):
        self.need = [self._required(value) for value in raw]

    def summary(self):
        counts = Counter(self.need)
        return {
            "inferred_clears": self.clears,
            "level_by_cycle": self.level,
            "cycle": list(self.cycle),
            "roles": {
                "start_known": self.start_color is not None,
                "intermediate_known": self.intermediate_color is not None,
                "destination_known": self.destination_color is not None,
            },
            "demand_counts": {"zero": counts.get(0, 0),
                              "one": counts.get(1, 0),
                              "two": counts.get(2, 0)},
            "recent_transitions": list(self.evidence)[-4:],
        }

    def update(self, obs):
        current = cube_colors(obs)
        t = now_tick(obs)
        score = finite_number(obs.get("score"), self.score or 0.0)
        old_score = self.score
        delta = score - old_score if old_score is not None else 0.0
        if current is None:
            fallback = obs.get("needed_landings")
            if isinstance(fallback, list) and len(fallback) == 21:
                self.need = [max(0, integer(value, 1)) for value in fallback]
            self.score = score
            return False

        changed = []
        if self.previous is not None:
            changed = [i for i in range(21) if current[i] != self.previous[i]]
        if self.start_color is None and self.previous is None:
            self._reset_board_roles(current)

        # A true round handoff recolors most of the board.  Quarantine every
        # intermediate palette frame so it cannot be interpreted as Sam or a
        # 21-cube sequence of landings.
        if len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_tick = t
                self.flash_score = old_score if old_score is not None else score - delta
                self.flash_origin = valid_node(obs.get("qbert_node"))
                self.event("phase_flash_begin", tick=t, score=score,
                           level=self.level, reported_rounds=obs.get("completed_rounds"))
            else:
                self.flash_tick = t

        if self.flashing:
            quiet = t - self.flash_tick
            returned = valid_node(obs.get("qbert_node")) == 0
            if quiet >= 4 and (returned or quiet >= 7):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.flashing = False
                self._reset_board_roles(current, anchor_start=False)
                self.last_progress = t
                self.event("phase_epoch", tick=t, epoch=self.epoch,
                           inferred_clears=self.clears, tally_points=tally,
                           level=self.level, start_color=self.start_color,
                           reported_rounds=obs.get("completed_rounds"))
                self.previous = current
                self.score = score
                return True
            self.need = [1] * 21
            self.previous = current
            self.score = score
            return False

        # A positive 25-point component plus a small local recoloring is the
        # strongest public evidence of a Q*bert landing.  Sam's bonus is not
        # allowed to author a forward phase transition.
        event_name = str(obs.get("last_event", ""))
        scored_landing = (
            delta > 0
            and int(round(delta)) % 100 == 25
            and event_name not in ("sam", "green_ball")
        )
        if scored_landing and 0 < len(changed) <= 3:
            qnode = valid_node(obs.get("qbert_node"))
            landed = qnode if qnode in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                old = self.previous[landed]
                new = current[landed]
                self._learn_transition(old, new, landed, delta, obs)
                self.last_progress = t
                self.event("phase_landing", tick=t, node=landed,
                           old_color=old, new_color=new, reward=delta,
                           level=self.level)

        if changed or delta > 0:
            self.last_progress = t
        self._recompute_need(current)
        self.previous = current
        self.score = score
        return False


class Controller:
    """Code-owned planner, motor transaction, and asynchronous advisers."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.progress = PhaseBook(ctx)
        self.life_epoch = 0
        self.lives = None
        self.pulse = None
        self.neutral_until = 0
        self.recent = deque(maxlen=24)
        self.centers = {}
        self.threats = {}
        self.last_request = -999
        self.s1 = None
        self.s1meta = None
        self.advice = None
        self.s2 = None
        self.s2_session = None
        self.s2_calls = 0
        self.s2_reason = "initial phase-cycle audit"
        self.last_s2_tick = -999
        self.policy_version = 0
        self.policy = {"order": "phase_first", "hazard": "predictive"}
        self.waiting_since = None
        self.last_log = -999

    def log(self, name, **fields):
        self.ctx.log(name, **fields)

    def update(self, obs):
        t = now_tick(obs)
        epoch_changed = self.progress.update(obs)
        lives = integer(obs.get("lives"), 4)
        lost = self.lives is not None and lives < self.lives
        if lost or epoch_changed:
            self.life_epoch += 1
            self.pulse = None
            self.advice = None
            self.recent.clear()
            self.neutral_until = t + 2
            self.waiting_since = None
            self.s2_reason = "life loss" if lost else "new inferred board cycle"
            self.log("control_epoch", tick=t, lives=lives,
                     life_loss=lost, phase_epoch=self.progress.epoch,
                     inferred_clears=self.progress.clears)
        self.lives = lives

        # Public cube coordinates are sufficient to translate rendered enemy
        # rectangles into conservative graph-neighbour facts.  No RGB is sent
        # to either model.
        if not self.centers:
            details = obs.get("cube_details", [])
            if isinstance(details, list):
                for item in details:
                    if isinstance(item, dict) and valid_node(item.get("node")) is not None:
                        self.centers[item["node"]] = (
                            finite_number(item.get("x")) + 10,
                            finite_number(item.get("y")) - 7,
                        )

        if self.progress.flashing:
            self.threats = {}
            self.pulse = None
            self.advice = None
            return

        raw_hazards = obs.get("hazards", [])
        if isinstance(raw_hazards, list):
            for item in raw_hazards:
                if not isinstance(item, dict) or not item.get("lethal"):
                    continue
                kind = str(item.get("kind", "hazard"))
                node = valid_node(item.get("estimated_node", item.get("node")))
                x = finite_number(item.get("x")) + finite_number(item.get("w"), 7) / 2
                y = finite_number(item.get("y")) + finite_number(item.get("h"), 8) / 2
                if self.centers:
                    # The rendered rectangle is the stronger public position
                    # signal during a hop; match the parent's calibrated cube
                    # anchor rather than trusting a lagging nearest-node byte.
                    node = min(self.centers, key=lambda candidate:
                               (self.centers[candidate][0] - x) ** 2
                               + (self.centers[candidate][1] - y) ** 2)
                dx = finite_number(item.get("dx"))
                dy = finite_number(item.get("dy"))
                old = self.threats.get(kind)
                if old and abs(dx) < 0.01 and abs(dy) < 0.01:
                    dx, dy = old.get("dx", 0.0), old.get("dy", 0.0)
                self.threats[kind] = {"kind": kind, "node": node,
                                     "x": x, "y": y, "dx": dx, "dy": dy,
                                     "tick": t}
        # A short shadow covers the interval between score-change observations,
        # but never becomes a permanent enemy.  The exact phase planner is not
        # allowed to infer a stale placeholder as a live sprite.
        self.threats = {
            kind: value for kind, value in self.threats.items()
            if t - value.get("tick", t) <= 5
        }

    def safe_edges(self, obs, node=None):
        origin = valid_node(obs.get("qbert_node")) if node is None else node
        if origin is None or origin not in GRAPH:
            return []
        safe = obs.get("safe_moves", obs.get("legal_moves", []))
        if not isinstance(safe, list):
            return []
        return [action for action in ACTION_ORDER
                if action in safe and action in GRAPH[origin]]

    def danger(self, destination, origin):
        # Keep the empirically stable parent hazard metric for the opening
        # boards.  The new phase scheduler must not confound palette learning
        # with a different estimate of immediate collision risk.
        result = 0.0
        for hazard in self.threats.values():
            node = hazard.get("node")
            if node is None or node not in GRAPH:
                continue
            if destination in self.centers:
                hx, hy = hazard.get("x", 0.0), hazard.get("y", 0.0)
                cx, cy = self.centers[destination]
                distance = math.hypot(cx - hx, cy - hy)
                result += 5.0 if distance < 17 else 1.2 if distance < 30 else 0.0
            if node == destination:
                result += 3.0
            kind = hazard.get("kind")
            if kind in ("purple_ball", "red_ball"):
                children = [there for action, there in GRAPH[node].items()
                            if action in ("DOWN", "RIGHT")]
                if destination in children:
                    result += 1.5
            elif kind == "coily":
                next_nodes = list(GRAPH[node].values())
                toward = min((DIST[there][origin] for there in next_nodes), default=99)
                if destination in next_nodes and DIST[destination][origin] == toward:
                    result += 2.0
                if DIST[node][destination] == 1:
                    result += 0.4
        return result

    def projected_hazard_nodes(self, steps):
        """A small uncertainty envelope for future hop landings."""
        possible = set()
        for hazard in self.threats.values():
            node = hazard.get("node")
            if node is None or node not in GRAPH:
                continue
            frontier = {node}
            for _ in range(max(0, steps)):
                nxt = set(frontier)
                for here in frontier:
                    if hazard.get("kind") in ("purple_ball", "red_ball"):
                        nxt.update(there for action, there in GRAPH[here].items()
                                   if action in ("DOWN", "RIGHT"))
                    else:
                        # Coily can choose a pursuit branch; retain one-step
                        # neighbours rather than pretending it is stationary.
                        nxt.update(GRAPH[here].values())
                frontier = nxt
            possible.update(frontier)
        return possible

    def best_continuation(self, start, previous, remaining, depth=2):
        """Choose a short phase-aware continuation for ranking one hop.

        This is not a queued macro: only its first edge is executed.  The
        continuation makes the tie-breaker see the next two landing decisions,
        which prevents a useful-looking cube from leading directly into a
        branch with no phase progress.
        """
        states = [(start, previous, dict(remaining), 0.0, 0, 0, [])]
        for step in range(1, depth + 1):
            expanded = []
            for here, before, demand, risk, gain, repeats, path in states:
                choices = list(GRAPH[here].items())
                if len(choices) > 1:
                    choices = [(a, n) for a, n in choices if n != before] or choices
                for action, there in choices:
                    local = dict(demand)
                    reduction = min(2, max(0, integer(local.get(there), 0)))
                    if reduction:
                        local[there] = max(0, local[there] - 1)
                    hazard_nodes = self.projected_hazard_nodes(step)
                    hit = 1 if there in hazard_nodes else 0
                    near = self.danger(there, here)
                    expanded.append((there, here, local,
                                     risk + 5.0 * hit + near,
                                     gain + reduction,
                                     repeats + (1 if there in path else 0),
                                     path + [action]))
            if not expanded:
                break
            # Keep a small beam; code arithmetic, not Jev, owns this search.
            expanded.sort(key=lambda state: (state[3], -state[4], state[5],
                                              min((DIST[state[0]][n] for n, v in state[2].items() if v > 0),
                                                  default=0)))
            states = expanded[:18]
        if not states:
            return {"risk": 99.0, "gain": 0, "path": [], "repeats": 0}
        best = min(states, key=lambda state: (state[3], -state[4], state[5],
                                               min((DIST[state[0]][n] for n, v in state[2].items() if v > 0),
                                                   default=0)))
        return {"risk": best[3], "gain": best[4], "path": best[6],
                "repeats": best[5]}

    def candidates(self, obs):
        origin = valid_node(obs.get("qbert_node"))
        if origin is None:
            return []
        targets = {node: demand for node, demand in enumerate(self.progress.need)
                   if demand > 0}
        if not targets:
            return []
        choices = []
        visits = Counter(self.recent)
        for action in self.safe_edges(obs):
            destination = GRAPH[origin][action]
            immediate = self.danger(destination, origin)
            continuation = self.best_continuation(destination, origin, targets, depth=2)
            demand = max(0, integer(targets.get(destination), 0))
            useful = demand > 0
            exits = sum(self.danger(there, destination) < 2.0
                        for there in GRAPH[destination].values())
            route = min((DIST[destination][target] for target in targets), default=0)
            # Preserve the proven parent ordering through the first inferred
            # level.  The new continuation scheduler is activated only once
            # the exact multi-stage/reversible schedules become relevant.
            cycle_planning = self.progress.level >= 2
            if cycle_planning:
                cost = 12.0 * immediate + 3.0 * continuation["risk"]
                cost += route * (1.25 if self.policy["order"] == "phase_first" else 1.8)
                cost += 0.25 if not useful else -0.45 * demand
                cost -= 0.16 * min(3, exits)
                cost += 0.10 * visits[destination]
                if continuation["gain"] == 0:
                    cost += 0.7
                if self.policy["hazard"] == "conservative":
                    cost += 0.5 * continuation["risk"]
            else:
                cost = 12.0 * immediate + route * 2.0
                cost += 0.0 if useful else 0.5
                cost += visits[destination] * 0.12 - exits * 0.12
                approaching_coily = any(
                    h.get("kind") == "coily" and h.get("node") is not None
                    and DIST[destination][h["node"]] <= 2
                    for h in self.threats.values()
                )
                if approaching_coily and exits == 0:
                    cost += 14.0
                if self.policy["order"] == "nearest":
                    pass
            choices.append({
                "action": action,
                "node": destination,
                "cost": cost,
                "danger_value": immediate,
                "danger": "occupied" if immediate >= 4 else
                          "intercept_possible" if immediate >= 1 else "clear",
                "landing": "two_landing_phase" if demand >= 2 else
                           "one_landing_phase" if useful else "transit",
                "useful": useful,
                "demand": demand,
                "distance": route,
                "escape": "multiple" if exits >= 2 else "narrow",
                "continuation_risk": continuation["risk"],
                "continuation_gain": continuation["gain"],
                "preview": list(continuation["path"]),
            })
        choices.sort(key=lambda item: (item["danger_value"] >= 4,
                                       item["cost"], ACTION_ORDER.index(item["action"])))

        # A descending ball can occupy both next branches.  Wait only while
        # the present node is materially safer and never while Coily is close.
        current_danger = self.danger(origin, origin)
        ball_present = any(h.get("kind") in ("purple_ball", "red_ball")
                           for h in self.threats.values())
        coily_close = any(h.get("kind") == "coily" and h.get("node") is not None
                          and DIST[origin][h["node"]] <= 2 for h in self.threats.values())
        wait_age = now_tick(obs) - self.waiting_since if self.waiting_since is not None else 0
        if (
            choices
            and ball_present
            and not coily_close
            and wait_age < 16
            and choices[0]["danger_value"] >= 1.0
            and current_danger + 0.35 < choices[0]["danger_value"]
        ):
            choices.insert(0, {
                "action": "NOOP", "node": origin, "cost": choices[0]["cost"] - 1.0,
                "danger_value": current_danger, "danger": "clear",
                "landing": "wait_for_branch_to_resolve", "useful": False,
                "demand": 0, "distance": min(DIST[origin][n] for n in targets),
                "escape": "multiple", "continuation_risk": current_danger,
                "continuation_gain": 0, "preview": [],
            })
        return choices

    def threat_signature(self):
        return tuple(sorted((str(kind), valid_node(item.get("node")))
                            for kind, item in self.threats.items()))

    def launch_jev(self, obs, candidates):
        if self.s1 is not None or not candidates or self.ctx.done:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        t = now_tick(obs)
        if t - self.last_request < 5:
            return
        if self.ctx.budget.remaining().get("system1_calls", 0) < 2:
            return
        origin = valid_node(obs.get("qbert_node"))
        options = {}
        for item in candidates:
            if item["danger_value"] >= 4:
                continue
            options[item["action"]] = {
                "landing": item["landing"],
                "danger": item["danger"],
                "escape": item["escape"],
                "phase_demand": item["demand"],
                "continuation": "risk_present" if item["continuation_risk"] >= 2 else "open",
                "preview": item["preview"][:2],
                "code_preferred": item is candidates[0],
            }
        if not options:
            return
        options["unknown"] = "No supported preference; retain code's legal fallback."
        state = {
            "game": "Atari 2600 Q*bert, six-row 21-cube graph; code owns legality and arithmetic",
            "rules": "Survival first; landing demand is an exact board-local S/I/D cycle; never choose FIRE or an edge absent from safe_moves.",
            "cycle": self.progress.summary(),
            "position": {"node": origin, "motion": qmotion(obs).get("phase", "unknown")},
            "alternatives": options,
            "fallback": candidates[0]["action"],
            "hazards": [{"kind": h["kind"],
                         "proximity": "adjacent" if h.get("node") is not None and origin is not None
                         and DIST[origin][h["node"]] <= 1 else "farther"}
                        for h in self.threats.values()],
            "timers": self._timer_buckets(obs),
        }
        instruction = (
            "Choose the safest supplied alternative. Never invent an action. "
            "An occupied/intercept_possible landing is worse than clear; among "
            "similarly safe options prefer a two_landing_phase or one_landing_phase "
            "that has an open continuation. Do not calculate distances. Unknown "
            "means let code continue."
        )
        criteria = {}
        for action, description in options.items():
            criteria[action] = description if isinstance(description, str) else (
                "Take this supplied legal edge; its structured phase demand, danger, "
                "escape and continuation facts are the evidence."
            )
        self.s1meta = {"tick": t, "node": origin, "life": self.life_epoch,
                       "epoch": self.progress.epoch, "policy": self.policy_version,
                       "threats": self.threat_signature()}
        self.last_request = t
        questions = {
            "choice_a": Choice(instruction, criteria),
            "choice_b": Choice(instruction, dict(reversed(list(criteria.items())))),
            "danger": Noul(
                "Do the supplied facts establish immediate lethal danger on the fallback? "
                "A distant enemy alone is not immediate danger.",
                {"true": "The fallback landing is occupied or a concrete collision is stated.",
                 "false": "The landing is clear, or only distant/uncertain threats are present."},
            ),
        }

        def work():
            return self.ctx.decide(state, questions, timeout_s=1.0)

        self.s1 = self.ctx.spawn(work)

    def _timer_buckets(self, obs):
        timers = obs.get("timers")
        if not isinstance(timers, dict):
            return {"scheduler": "unknown", "purple_state": "unknown"}
        scheduler = integer(timers.get("object_scheduler"), -1)
        purple = integer(timers.get("purple_ball_state"), -1)
        return {
            "scheduler": "unknown" if scheduler < 0 else
                         "recent_reset" if scheduler < 40 else
                         "mid_cycle" if scheduler < 160 else "late_cycle",
            "purple_state": "unknown" if purple < 0 else
                            "active_or_transition" if purple in (0, 4, 30, 36) else "other",
        }

    def collect_jev(self, obs):
        if self.s1 is None or not self.s1.done():
            return
        future, metadata = self.s1, self.s1meta
        self.s1 = None
        self.s1meta = None
        try:
            answers = future.result()
            if not isinstance(answers, dict):
                return
            first = answer_value(answers.get("choice_a"))
            second = answer_value(answers.get("choice_b"))
            agreed = first is not None and first == second
            confidence = min(answer_probability(answers.get("choice_a"), first),
                             answer_probability(answers.get("choice_b"), second)) if agreed else 0.0
            veto_value = finite_number(getattr(answers.get("danger"), "value", 0.0), 0.0)
            self.advice = dict(metadata, action=first if agreed and confidence >= 0.60 else None,
                               probability=confidence, veto=veto_value >= 0.84)
        except Exception as exc:
            self.log("jev_failure", error=type(exc).__name__)

    def launch_astra(self, obs, reason):
        if self.s2 is not None or self.s2_calls >= 3 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        if self.ctx.budget.remaining().get("system2_calls", 0) < 1:
            return
        t = now_tick(obs)
        if self.s2_calls and t - self.last_s2_tick < 300:
            return
        self.s2_calls += 1
        self.last_s2_tick = t
        self.s2_reason = None
        state = {
            "reason": reason,
            "tick": t,
            "score": finite_number(obs.get("score")),
            "lives": self.lives,
            "phase_book": self.progress.summary(),
            "policy": self.policy,
            "recent_threats": [{"kind": h["kind"], "node": h.get("node")}
                               for h in self.threats.values()],
        }
        prompt = (
            "You are the slow policy auditor for a real-time Atari 2600 Q*bert controller. "
            "The board has 21 nodes. Code has exact graph legality, a neutralized one-hop "
            "motor, and a round-local palette tracker. Unlike a boolean visited map, it uses "
            "the observed S/I/D transition cycle: levels 1 and 3 have two phases, levels 2 "
            "has S-I-D, level 4 S-I-D-I, and level 5 S-I-D-S; raw colors are learned per "
            "board. It also ranks a two-hop continuation for phase demand and collision risk. "
            "Choose only a durable policy: order=phase_first or nearest, hazard=predictive or "
            "visible_only. phase_first prioritizes cubes with two remaining landings; nearest "
            "prioritizes shortest route. predictive uses the code's conservative future branch "
            "envelope. Jev receives only generated alternatives and two reordered Choices; it "
            "cannot invent actions. Return ONLY JSON under 40 tokens, such as "
            "{\"order\":\"phase_first\",\"hazard\":\"predictive\"}. Evidence: "
            + json.dumps(state, separators=(",", ":"))
        )
        session = self.s2_session

        def work():
            return self.ctx.think(prompt, session_id=session, timeout_s=100.0)

        self.s2 = self.ctx.spawn(work)
        self.log("astra_requested", tick=t, reason=reason)

    def collect_astra(self, obs):
        if self.s2 is None or not self.s2.done():
            return
        future, self.s2 = self.s2, None
        try:
            result = future.result()
            self.s2_session = getattr(result, "session_id", None)
            parsed = parse_json_object(getattr(result, "text", ""))
            old = dict(self.policy)
            if parsed.get("order") in ("phase_first", "nearest"):
                self.policy["order"] = parsed["order"]
            if parsed.get("hazard") in ("predictive", "visible_only"):
                self.policy["hazard"] = parsed["hazard"]
            if old != self.policy:
                self.policy_version += 1
                self.advice = None
                self.log("astra_policy_changed", tick=now_tick(obs), old=old,
                         new=dict(self.policy))
            else:
                self.log("astra_policy_unchanged", tick=now_tick(obs), parsed=bool(parsed))
        except Exception as exc:
            self.log("astra_failure", error=type(exc).__name__)

    def advised(self, obs, candidates):
        fallback = candidates[0]["action"]
        advice = self.advice
        self.advice = None
        if not advice:
            return fallback
        fresh = (
            advice.get("node") == valid_node(obs.get("qbert_node"))
            and advice.get("life") == self.life_epoch
            and advice.get("epoch") == self.progress.epoch
            and advice.get("policy") == self.policy_version
            and advice.get("threats") == self.threat_signature()
            and 0 <= now_tick(obs) - advice.get("tick", now_tick(obs)) <= 8
        )
        if not fresh:
            return fallback
        by_action = {item["action"]: item for item in candidates}
        chosen = by_action.get(advice.get("action"))
        if chosen is not None and chosen["danger_value"] < 4:
            if chosen["cost"] <= candidates[0]["cost"] + 2.5 and chosen["danger_value"] <= candidates[0]["danger_value"] + 0.25:
                if chosen["action"] != fallback:
                    self.log("jev_changes_action", tick=now_tick(obs), old=fallback,
                             new=chosen["action"], probability=advice.get("probability", 0.0))
                return chosen["action"]
        if advice.get("veto") and len(candidates) > 1 and candidates[0]["danger_value"] >= 1:
            alternative = candidates[1]
            if alternative["danger_value"] <= candidates[0]["danger_value"] + 0.1:
                self.log("jev_veto_applied", tick=now_tick(obs), old=fallback,
                         new=alternative["action"])
                return alternative["action"]
        return fallback

    def motor_action(self, obs):
        t = now_tick(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)
        if self.progress.flashing:
            self.pulse = None
            return "NOOP"

        node = valid_node(obs.get("qbert_node"))
        phase = qmotion(obs).get("phase")
        if node is None or phase in ("falling", "not_visible") or qmotion(obs).get("falling"):
            self.pulse = None
            return "NOOP"

        # A pulse is a transaction.  Only the first two/three ticks can carry
        # its edge; every later tick is neutral, preventing a queued second hop.
        if self.pulse is not None:
            pulse = self.pulse
            age = t - pulse["tick"]
            if obs.get("qbert_in_hop") or node != pulse["origin"]:
                pulse["moving"] = True
            if pulse["moving"] and is_standing(obs) and age >= 4:
                self.recent.append(node)
                self.log("hop_landed", tick=t, origin=pulse["origin"], node=node,
                         expected=pulse["destination"], duration=age)
                self.pulse = None
                self.neutral_until = t + 1
            elif not pulse["moving"] and age <= 2 and pulse["action"] in self.safe_edges(obs):
                return pulse["action"]
            elif age > 20 and is_standing(obs):
                self.log("hop_resync", tick=t, expected=pulse["destination"], observed=node)
                self.pulse = None
                self.neutral_until = t + 1
            else:
                return "NOOP"

        if not is_standing(obs) or t < self.neutral_until:
            return "NOOP"
        candidates = self.candidates(obs)
        if not candidates:
            return "NOOP"
        selected_action = self.advised(obs, candidates)
        if selected_action == "NOOP":
            if self.waiting_since is None:
                self.waiting_since = t
                self.log("tactical_wait", tick=t, node=node,
                         threats=list(self.threat_signature()))
            return "NOOP"
        if selected_action not in self.safe_edges(obs):
            return "NOOP"
        self.waiting_since = None
        destination = GRAPH[node][selected_action]
        self.pulse = {"origin": node, "destination": destination,
                      "action": selected_action, "tick": t, "moving": False}

        # Pipeline advice for the state expected at the next landing.  The
        # response is never allowed to authorize a different node or stale risk.
        projected = dict(obs)
        projected["qbert_node"] = destination
        projected["qbert_in_hop"] = False
        projected["qbert_motion"] = {"phase": "standing", "falling": False}
        occupied = {h.get("node") for h in self.threats.values()}
        projected["legal_moves"] = list(GRAPH[destination])
        projected["safe_moves"] = [action for action, there in GRAPH[destination].items()
                                   if there not in occupied]
        self.launch_jev(projected, self.candidates(projected))
        if t - self.last_log >= 20:
            self.log("phase_coverage", tick=t, score=finite_number(obs.get("score")),
                     lives=self.lives, inferred_clears=self.progress.clears,
                     level=self.progress.level,
                     demand_counts=self.progress.summary()["demand_counts"],
                     node=node, action=selected_action,
                     threats=list(self.threat_signature()))
            self.last_log = t
        return selected_action


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        controller = Controller(ctx)
        initial = ctx.observe()
        if not isinstance(initial, dict):
            initial = {}
        controller.update(initial)
        controller.launch_astra(initial, "initial phase-cycle calibration")
        while not ctx.done and not game.done:
            obs = ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            controller.update(obs)
            action = controller.motor_action(obs)
            ctx.act(action if action in DIRECTIONS or action == "NOOP" else "NOOP")
            if controller.s2_reason:
                controller.launch_astra(obs, controller.s2_reason)
            elif (
                controller.s2_calls < 3
                and now_tick(obs) - controller.progress.last_progress > 180
            ):
                controller.launch_astra(obs, "phase progress stalled; audit cycle interpretation")
            ctx.wait_for_tick()
        ctx.log("candidate_summary", score=finite_number(getattr(game, "score", 0.0)),
                inferred_clears=controller.progress.clears,
                phase_level=controller.progress.level,
                roles=controller.progress.summary()["roles"],
                policy=dict(controller.policy))


def build(ctx):
    return HarnessProgram(ctx)
