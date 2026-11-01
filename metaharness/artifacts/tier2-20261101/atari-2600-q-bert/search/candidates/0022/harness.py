"""Candidate 0022: temporal dephasing with a phase-aware route ensemble.

The controller treats a Q*bert edge as a timed commitment.  It keeps the
parent's one-hop motor, but does not assume that a geometrically safe edge is
safe at every enemy-scheduler phase.  A small ledger records the phase of
losses and the route's recent edge timetable.  It changes between two
coverage orderings, and may pay a bounded neutral tick, when the next useful
edge would repeat a recently bad temporal phase.  Jev reviews supplied
(time-labelled) edges; it never invents joystick input.  Astra selects only a
coarse timing/coverage posture.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
TIE_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")


def make_graph():
    graph = {n: {} for n in range(21)}
    rows = {}
    for row in range(6):
        first = row * (row + 1) // 2
        for col in range(row + 1):
            n = first + col
            rows[n] = row
            if row:
                if col < row:
                    graph[n]["UP"] = n - row
                if col:
                    graph[n]["LEFT"] = n - row - 1
            if row < 5:
                child = (row + 1) * (row + 2) // 2
                graph[n]["DOWN"] = child + col
                graph[n]["RIGHT"] = child + col + 1
    return graph, rows


GRAPH, ROW = make_graph()


def make_distances():
    answer = []
    for root in range(21):
        distance = [99] * 21
        distance[root] = 0
        todo = deque([root])
        while todo:
            here = todo.popleft()
            for there in GRAPH[here].values():
                if distance[there] == 99:
                    distance[there] = distance[here] + 1
                    todo.append(there)
        answer.append(distance)
    return answer


DIST = make_distances()

# The first order preserves the useful early opening of the strongest parent.
# The second is a different sweep, not a reversed list: it changes the
# junctions at which the bottom and upper branches are visited.
ORDER_PRIMARY = (1, 4, 7, 11, 6, 10, 16, 12, 17, 18, 13, 9, 14,
                 19, 20, 8, 2, 0, 3, 5, 15)
ORDER_ALTERNATE = (2, 4, 8, 12, 7, 11, 6, 10, 16, 17, 18, 13, 9,
                   14, 19, 20, 5, 3, 1, 0, 15)
RANK_PRIMARY = {n: i for i, n in enumerate(ORDER_PRIMARY)}
RANK_ALTERNATE = {n: i for i, n in enumerate(ORDER_ALTERNATE)}


def finite(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def integer(value, default=0):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(finite(value, default))


def node_of(value):
    if isinstance(value, bool):
        return None
    if not isinstance(value, int):
        return None
    return value if 0 <= value < 21 else None


def tick_of(obs):
    if not isinstance(obs, dict):
        return 0
    return integer(obs.get("tick", obs.get("ticks", 0)), 0)


def qmotion(obs):
    value = obs.get("qbert_motion") if isinstance(obs, dict) else None
    return value if isinstance(value, dict) else {"phase": str(value or "unknown")}


def standing(obs):
    if not isinstance(obs, dict):
        return False
    n = node_of(obs.get("qbert_node"))
    motion = qmotion(obs)
    phase = str(motion.get("phase", "unknown")).lower()
    return (n is not None and not bool(obs.get("qbert_in_hop"))
            and phase not in ("hopping", "falling", "not_visible")
            and not bool(motion.get("falling")))


def names_list(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [x for x in value if isinstance(x, str)]


def palette_of(obs):
    value = obs.get("cube_colors") if isinstance(obs, dict) else None
    if not isinstance(value, list) or len(value) != 21:
        return None
    result = []
    for x in value:
        if isinstance(x, bool) or not isinstance(x, int):
            return None
        result.append(x)
    return tuple(result)


def safe_json(value):
    """Make a strict JSON value even when a stand-in gives odd observations."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [safe_json(x) for x in value]
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    return str(value)


def answer_string(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value if isinstance(value, str) else None


def probability(answer, key):
    values = getattr(answer, "probabilities", None) if answer is not None else None
    return finite(values.get(key), 0.0) if isinstance(values, dict) else 0.0


def parse_object(text):
    """Extract one JSON object; prose or arbitrary preflight output is harmless."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:12000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def segment_distance(px, py, ax, ay, bx, by):
    """Distance from a point to a board-edge segment, with no external deps."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    t = 0.0 if denom <= 0.0 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    dx, dy = px - (ax + t * vx), py - (ay + t * vy)
    return math.sqrt(dx * dx + dy * dy)


class PaletteBook:
    """Use adaptor phase demand when available and score flashes as board epochs."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.last_score = None
        self.goal = None
        self.need = [1] * 21
        self.flashing = False
        self.flash_last_change = -999
        self.flash_score = 0.0
        self.flash_origin = None
        self.epoch = 0
        self.clears = 0
        self.last_progress = 0
        self.handoff_until = -1
        self.evidence = deque(maxlen=8)

    def _decoded_need(self, obs):
        raw = obs.get("needed_landings") if isinstance(obs, dict) else None
        if not isinstance(raw, list) or len(raw) != 21:
            return None
        values = []
        for x in raw:
            if isinstance(x, bool):
                return None
            v = integer(x, -1)
            if v < 0:
                return None
            values.append(min(9, max(0, v)))
        return values

    def update(self, obs):
        if not isinstance(obs, dict):
            return False
        now = tick_of(obs)
        current = palette_of(obs)
        score = finite(obs.get("score"), self.last_score if self.last_score is not None else 0.0)
        delta = score - self.last_score if self.last_score is not None else 0.0
        changed = []
        if current is not None and self.previous is not None:
            changed = [i for i in range(21) if current[i] != self.previous[i]]

        if self.previous is not None and len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_score = self.last_score if self.last_score is not None else score
                self.flash_origin = node_of(obs.get("qbert_node"))
                self.ctx.log("temporal_flash_start", tick=now, score=score)
            self.flash_last_change = now

        if self.flashing:
            quiet = now - self.flash_last_change
            returned = node_of(obs.get("qbert_node")) == 0
            if quiet >= 4 and (returned or quiet >= 8):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.goal = None
                self.need = [1] * 21
                self.last_progress = now
                self.evidence.clear()
                self.handoff_until = now + 12
                self.flashing = False
                self.ctx.log("temporal_board_epoch", tick=now, epoch=self.epoch,
                             inferred_clears=self.clears, tally=tally)
                self.previous, self.last_score = current, score
                return True
            self.previous, self.last_score = current, score
            return False

        if now <= self.handoff_until:
            # On round transition all cubes may briefly render as destination,
            # followed by the new round's starting palette.  Do not let these
            # transient frames overwrite the settled decoder demand.
            self.previous, self.last_score = current, score
            raw = self._decoded_need(obs)
            if raw is not None and any(raw):
                self.need = raw
            return False

        decoded = self._decoded_need(obs)
        if decoded is not None and not any(decoded):
            phases = obs.get("cube_phase")
            # During the handoff to a new board, the decoder can briefly report
            # an all-zero demand vector before the reset S/I palette is settled.
            # Preserve unfinished visible phases rather than declaring victory.
            if (isinstance(phases, list) and len(phases) == 21 and
                    any(str(value).upper() in ("S", "I", "unknown") for value in phases)):
                decoded = [0 if str(value).upper() == "D" else 1 for value in phases]
        if decoded is not None:
            self.need = decoded
        elif current is not None:
            if delta > 0 and int(round(delta)) % 100 == 25 and 0 < len(changed) <= 3:
                qnode = node_of(obs.get("qbert_node"))
                landed = qnode if qnode in changed else changed[0] if len(changed) == 1 else None
                if landed is not None:
                    self.goal = current[landed]
                    self.evidence.append({"tick": now, "node": landed,
                                          "color": current[landed], "reward": delta})
                    self.last_progress = now
            if self.goal is not None:
                self.need = [0 if c == self.goal else 1 for c in current]
            else:
                phases = obs.get("cube_phase")
                if isinstance(phases, list) and len(phases) == 21:
                    self.need = [0 if str(x).upper() == "D" else 1 for x in phases]
                else:
                    self.need = [1] * 21

        if delta > 0:
            self.last_progress = now
        self.previous, self.last_score = current, score
        return False


class HazardField:
    """Rendered hazards plus a short, explicitly bounded rolling sighting cone."""

    def __init__(self):
        self.items = {}
        self.sightings = deque(maxlen=10)
        self.latest_roll = {}
        self.scheduler = -1
        self.purple_state = -1
        self.scheduler_band = -1
        self.scheduler_reset_until = -1
        self.centers = {}

    def learn_centers(self, obs):
        if self.centers:
            return
        details = obs.get("cube_details") if isinstance(obs, dict) else None
        if not isinstance(details, list):
            return
        for item in details:
            if not isinstance(item, dict):
                continue
            n = node_of(item.get("node"))
            if n is not None:
                self.centers[n] = (finite(item.get("x")) + 5.0,
                                   finite(item.get("y")) + 5.0)

    def clear_recent(self):
        self.items.clear()
        self.sightings.clear()
        self.latest_roll.clear()

    def update(self, obs, now):
        if not isinstance(obs, dict):
            return
        self.learn_centers(obs)
        raw = obs.get("hazards")
        visible = set()
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict) or not bool(item.get("lethal")):
                    continue
                kind = str(item.get("kind", "hazard"))
                if kind not in ("purple_ball", "red_ball", "coily"):
                    continue
                n = node_of(item.get("estimated_node", item.get("node")))
                if n is None:
                    continue
                entry = {
                    "kind": kind, "node": n, "last": now, "visible": True,
                    "x": finite(item.get("x")), "y": finite(item.get("y")),
                    "w": finite(item.get("w"), 7.0), "h": finite(item.get("h"), 8.0),
                }
                self.items[kind] = entry
                visible.add(kind)
                if kind in ("purple_ball", "red_ball"):
                    old = self.latest_roll.get(kind)
                    if old is None or old.get("node") != n:
                        self.sightings.append({"kind": kind, "node": n, "tick": now})
                    self.latest_roll[kind] = {"node": n, "tick": now}

        # RGB validation in the adaptor is the visibility authority.  Memory
        # can only be used for a few ticks and is never relabelled as visible.
        for kind in list(self.items):
            age = now - integer(self.items[kind].get("last"), now)
            limit = 8 if kind == "coily" else 5
            if age > limit:
                self.items.pop(kind, None)
            else:
                self.items[kind]["visible"] = kind in visible
        self.sightings = deque((x for x in self.sightings
                                if now - integer(x.get("tick"), now) <= 60), maxlen=10)

        timers = obs.get("timers")
        sched = integer(timers.get("object_scheduler"), -1) if isinstance(timers, dict) else -1
        if sched >= 0 and self.scheduler >= 0 and sched < self.scheduler - 40:
            self.scheduler_reset_until = max(self.scheduler_reset_until, now + 6)
        if sched >= 0:
            self.scheduler = sched
            self.scheduler_band = min(15, sched // 16)
        self.purple_state = (integer(timers.get("purple_ball_state"), -1)
                             if isinstance(timers, dict) else -1)

    @staticmethod
    def down_children(nodes):
        result = set()
        for n in nodes:
            for action, child in GRAPH.get(n, {}).items():
                if action in ("DOWN", "RIGHT"):
                    result.add(child)
        return result

    def possible(self, kind, now, horizon):
        item = self.items.get(kind)
        if not item:
            return set()
        source = item.get("node")
        if source not in GRAPH:
            return set()
        age = max(0, now - integer(item.get("last"), now))
        if kind in ("purple_ball", "red_ball"):
            steps = min(2, max(0, (age + horizon) // 9))
            possible = {source}
            frontier = {source}
            for _ in range(steps):
                frontier = self.down_children(frontier)
                possible.update(frontier)
            return possible
        radius = min(2, max(1, (age + horizon) // 10))
        possible = {source}
        frontier = {source}
        for _ in range(radius):
            frontier = {there for here in frontier for there in GRAPH[here].values()}
            possible.update(frontier)
        return possible

    def edge_risk(self, origin, destination, now, horizon=8):
        value = 0.0
        causes = []
        for kind, item in self.items.items():
            source = item.get("node")
            if source is None:
                continue
            distance = DIST[source][origin] if origin is not None else 99
            age = max(0, now - integer(item.get("last"), now))
            if kind == "coily" and distance <= 1:
                value += 3.0
                causes.append("coily_takeoff")
            if kind in ("purple_ball", "red_ball") and distance == 0:
                value += 3.0
                causes.append("rolling_takeoff")
            arrival = self.possible(kind, now, horizon)
            if destination in arrival:
                value += 5.5 if age <= 2 else 3.0
                causes.append(kind + "_arrival")
            if kind in ("purple_ball", "red_ball"):
                if destination in self.down_children(arrival) and destination not in arrival:
                    value += 1.8
                    causes.append(kind + "_branch")
            elif DIST[source][destination] <= 1:
                value += 0.7
                causes.append("coily_near_landing")

            # A hazard can collide during the diagonal animation even when its
            # node is not exactly the endpoint.  This is a soft swept-edge cue,
            # not a hard geometric block.
            if origin in self.centers and destination in self.centers:
                hx = item.get("x", 0.0) + item.get("w", 7.0) / 2.0
                hy = item.get("y", 0.0) + item.get("h", 8.0) / 2.0
                ax, ay = self.centers[origin]
                bx, by = self.centers[destination]
                if segment_distance(hx, hy, ax, ay, bx, by) < 13.0:
                    value += 1.1 if age > 2 else 2.0
                    causes.append(kind + "_swept_path")

        if now < self.scheduler_reset_until and not self.items:
            value += 0.18
            causes.append("spawn_uncertain")
        return value, causes

    def exit_count(self, landing, origin, now):
        count = 0
        for action, there in GRAPH.get(landing, {}).items():
            if there == origin:
                continue
            risk, _ = self.edge_risk(landing, there, now + 2, horizon=5)
            if risk < 3.4:
                count += 1
        return count

    def semantic(self, origin, now):
        answer = []
        for kind, item in self.items.items():
            n = item.get("node")
            d = DIST[origin][n] if origin is not None and n is not None else 99
            answer.append({"kind": kind,
                           "visibility": "rendered" if item.get("visible") else "recent_memory",
                           "proximity": "same" if d == 0 else
                                        "adjacent" if d == 1 else
                                        "near" if d <= 2 else "far",
                           "motion": "rolling_down" if kind in ("purple_ball", "red_ball")
                                     else "pursuit"})
        if now < self.scheduler_reset_until and not self.items:
            answer.append({"kind": "spawn_window", "visibility": "uncertain",
                           "proximity": "unknown", "motion": "not_localized"})
        return answer

    def signature(self, now):
        # Only discontinuous visual hazard changes invalidate an in-flight
        # semantic review.  Scheduler bytes advance during a normal hop; those
        # are recomputed into the timing cost at arrival rather than making
        # every asynchronous answer stale by construction.
        return (tuple(sorted((k, v.get("node"), bool(v.get("visible")))
                             for k, v in self.items.items())),
                now < self.scheduler_reset_until)


class TemporalLedger:
    """Remember *when* an edge failed, rather than banning that edge forever."""

    PERIOD = 64

    def __init__(self, ctx):
        self.ctx = ctx
        self.uses = deque(maxlen=260)
        self.failures = deque(maxlen=10)
        self.waits = deque(maxlen=20)
        self.last_loss_tick = -999
        self.loss_epoch = 0
        self.mode_until = -1

    @staticmethod
    def cyclic_distance(a, b, period):
        d = abs((a - b) % period)
        return min(d, period - d)

    def phase(self, now, field):
        return {"clock": now % self.PERIOD,
                 "scheduler_band": field.scheduler_band,
                 "purple_band": (field.purple_state // 8 if field.purple_state >= 0 else -1)}

    def note_use(self, origin, destination, now, field):
        self.uses.append({"origin": origin, "destination": destination, "tick": now,
                          "clock": now % self.PERIOD,
                          "scheduler_band": field.scheduler_band,
                          "purple_band": field.purple_state // 8 if field.purple_state >= 0 else -1})

    def note_wait(self, now):
        self.waits.append(now)

    def note_loss(self, edge, now, field):
        if edge is None:
            return
        origin, destination = edge
        phase = self.phase(now, field)
        self.failures.append({"origin": origin, "destination": destination,
                              "tick": now, **phase})
        self.last_loss_tick = now
        self.loss_epoch += 1
        self.mode_until = max(self.mode_until, now + 150)
        self.ctx.log("temporal_edge_failure", tick=now, origin=origin,
                     destination=destination, phase=phase, loss_epoch=self.loss_epoch)

    def edge_penalty(self, origin, destination, now, field, mode):
        penalty = 0.0
        same_phase = 0
        for failure in self.failures:
            failure_age = now - integer(failure.get("tick"), now)
            if failure_age < 0 or failure_age > 256:
                continue
            clock_gap = self.cyclic_distance(now % self.PERIOD,
                                              integer(failure.get("clock"), 0),
                                              self.PERIOD)
            direct = (origin == failure.get("origin") and
                      destination == failure.get("destination"))
            reverse = (origin == failure.get("destination") and
                       destination == failure.get("origin"))
            # Keep a failed edge's correlation for a short revisit horizon, not
            # indefinitely.  This respects the evidence without baking the
            # training run's time modulo into all later boards.
            if failure_age > 144:
                continue
            same_tip = destination in (15, 20) or failure.get("destination") in (15, 20)
            if direct:
                if clock_gap <= 7:
                    penalty += 6.0
                    same_phase += 1
                elif clock_gap <= 16:
                    penalty += 2.1
                elif clock_gap <= 28:
                    penalty += 0.65
                if (field.scheduler_band >= 0 and
                        field.scheduler_band == failure.get("scheduler_band")):
                    penalty += 1.1
            elif reverse and clock_gap <= 10:
                penalty += 1.2
            elif same_tip and clock_gap <= 7:
                penalty += 0.75

        # A route that keeps traversing the same edge in the same clock slot
        # can phase-lock even without a recorded death.  This is intentionally
        # a much smaller cost than the loss-phase memory.
        for use in self.uses:
            age = now - integer(use.get("tick"), now)
            if age < 0 or age > 180:
                continue
            if use.get("origin") == origin and use.get("destination") == destination:
                gap = self.cyclic_distance(now % self.PERIOD,
                                            integer(use.get("clock"), 0), self.PERIOD)
                if gap <= 4:
                    penalty += 0.35

        if mode in ("dephase", "diverse") and same_phase:
            penalty += 0.8
        return penalty

    def should_wait(self, now, top_risk, top_temporal, exits, mode):
        if mode not in ("dephase", "diverse") or exits < 2:
            return False
        if top_temporal < 2.3 or top_risk >= 4.0:
            return False
        recent = sum(1 for t in self.waits if now - t <= 80)
        return recent < 2

    def facts(self, now, field):
        recent_failures = []
        for item in list(self.failures)[-3:]:
            recent_failures.append({
                "edge": "same" if item.get("origin") is not None else "unknown",
                "phase_relation": "near_current" if self.cyclic_distance(
                    now % self.PERIOD, integer(item.get("clock"), 0), self.PERIOD) <= 10
                    else "different_phase",
            })
        return {"mode": "dephase" if now < self.mode_until else "steady",
                "recent_failures": recent_failures,
                "current_clock_band": "early" if now % self.PERIOD < 16 else
                                      "middle" if now % self.PERIOD < 48 else "late",
                "scheduler_band": field.scheduler_band}


class RoutePlanner:
    """A coverage planner with two route cohorts and timed edge costs."""

    def __init__(self, book, field, temporal):
        self.book = book
        self.field = field
        self.temporal = temporal
        self.target = None
        self.cursor = 0
        self.mode = "primary"
        self.safety = "ordinary"
        self.recent_nodes = deque(maxlen=24)
        self.wrong_edges = Counter()
        self.policy_version = 0
        self.last_epoch = 0
        self.wait_started = None

    def order(self):
        return ORDER_ALTERNATE if self.mode == "alternate" else ORDER_PRIMARY

    def reset_board(self):
        self.target = None
        self.cursor = 0
        self.recent_nodes.clear()
        self.wrong_edges.clear()
        self.wait_started = None
        self.last_epoch = self.book.epoch

    def set_mode(self, mode):
        if mode not in ("primary", "alternate"):
            return
        if mode != self.mode:
            self.mode = mode
            self.target = None
            self.cursor = 0
            self.policy_version += 1

    def note_landing(self, actual, expected, origin):
        if actual is None:
            return
        self.recent_nodes.append(actual)
        if expected is not None and origin is not None and actual != expected:
            self.wrong_edges[(origin, expected)] += 1
        if self.target == actual and self.book.need[actual] <= 0:
            self.target = None

    def unfinished(self):
        return [i for i, value in enumerate(self.book.need) if value > 0]

    def choose_target(self, here):
        remaining = set(self.unfinished())
        if self.target not in remaining or (self.target == here and self.book.need[here] <= 0):
            self.target = None
        order = self.order()
        if self.target is None:
            for offset in range(len(order)):
                i = (self.cursor + offset) % len(order)
                candidate = order[i]
                if candidate in remaining and candidate != here:
                    self.cursor = i
                    self.target = candidate
                    break
        if self.target is None and remaining and here is not None:
            self.target = min(remaining, key=lambda n: (DIST[here][n], ROW[n], n))
        return self.target

    def candidates_for(self, origin, allowed, now):
        if origin is None:
            return []
        allowed = [a for a in TIE_ORDER if a in allowed and a in GRAPH.get(origin, {})]
        if not allowed:
            return []
        target = self.choose_target(origin)
        remaining = self.unfinished()
        ranks = RANK_ALTERNATE if self.mode == "alternate" else RANK_PRIMARY
        visits = Counter(self.recent_nodes)
        choices = []
        for action in allowed:
            destination = GRAPH[origin][action]
            risk, causes = self.field.edge_risk(origin, destination, now, horizon=8)
            temporal = self.temporal.edge_penalty(origin, destination, now,
                                                  self.field, self.safety)
            exits = self.field.exit_count(destination, origin, now)
            useful = self.book.need[destination] > 0 if destination < len(self.book.need) else False
            distance = (DIST[destination][target] if target is not None else
                        min((DIST[destination][n] for n in remaining), default=0))
            repeated = visits[destination]
            dead_end = exits == 0 and len(allowed) > 1
            cost = 1.65 * distance
            cost += 0.52 if not useful else 0.0
            cost += 10.5 * min(8.0, risk)
            cost += temporal
            cost -= 0.18 * min(3, exits)
            cost += 0.10 * repeated
            cost += min(1.8, 0.65 * self.wrong_edges[(origin, destination)])
            if dead_end:
                cost += 0.50 if self.safety != "shelter" else 0.85
            if target is not None:
                cost += 0.11 * DIST[destination][target]
            choices.append({"key": action, "action": action,
                            "destination": destination, "risk": risk,
                            "risk_reasons": causes, "temporal": temporal,
                            "exits": exits, "useful": useful,
                            "distance": distance, "cost": cost,
                            "target": target, "dead_end": dead_end,
                            "rank": ranks.get(destination, 99),
                            "clock_band": "early" if now % 64 < 16 else
                                          "middle" if now % 64 < 48 else "late"})
        choices.sort(key=lambda x: (x["risk"] >= 7.0, x["cost"], x["rank"],
                                   TIE_ORDER.index(x["action"])))

        if choices:
            top = choices[0]
            if self.temporal.should_wait(now, top["risk"], top["temporal"],
                                         self.field.exit_count(origin, None, now),
                                         self.safety):
                if self.wait_started is None:
                    self.wait_started = now
                if now - self.wait_started < 3:
                    choices.insert(0, {"key": "wait", "action": "NOOP",
                                       "destination": origin, "risk": top["risk"] * 0.4,
                                       "risk_reasons": ["phase_dephase_wait"],
                                       "temporal": max(0.0, top["temporal"] - 0.7),
                                       "exits": self.field.exit_count(origin, None, now),
                                       "useful": False, "distance": top["distance"],
                                       "cost": top["cost"] - 0.15, "target": target,
                                       "dead_end": False, "rank": -1,
                                       "clock_band": top["clock_band"]})
                else:
                    self.wait_started = None
            else:
                self.wait_started = None
        return choices

    def available(self, obs):
        if not isinstance(obs, dict):
            return []
        here = node_of(obs.get("qbert_node"))
        safe = set(names_list(obs.get("safe_moves")))
        legal = set(names_list(obs.get("legal_moves")))
        allowed = safe if safe else legal
        return self.candidates_for(here, allowed, tick_of(obs))


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.book = PaletteBook(ctx)
        self.field = HazardField()
        self.temporal = TemporalLedger(ctx)
        self.planner = RoutePlanner(self.book, self.field, self.temporal)
        self.lives = None
        self.life_counter = None
        self.life_epoch = 0
        self.pending = None
        self.last_edge = None
        self.recover_until = 0
        self.neutral_until = 0
        self.wait_until = 0
        self.last_log = -999

        self.s1_future = None
        self.s1_meta = None
        self.s1_advice = None
        self.last_s1_tick = -999
        self.s2_future = None
        self.s2_session = None
        self.s2_calls = 0
        self.s2_reason = "initial timing posture"

        try:
            self.action_names = {str(x) for x in game.action_spec.names}
        except Exception:
            self.action_names = {"NOOP", *DIRECTIONS}
        self.noop = "NOOP" if "NOOP" in self.action_names else next(iter(self.action_names), "NOOP")

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def ingest(self, obs):
        if not isinstance(obs, dict):
            return
        now = tick_of(obs)
        old_epoch = self.book.epoch
        old_lives = self.lives
        old_counter = self.life_counter
        self.book.update(obs)
        self.field.update(obs, now)
        lives = integer(obs.get("lives"), self.lives if self.lives is not None else 4)
        loss_counter = integer(obs.get("life_lost_count"),
                               self.life_counter if self.life_counter is not None else 0)
        lost = ((old_lives is not None and lives < old_lives) or
                (old_counter is not None and loss_counter > old_counter))

        if self.book.epoch != old_epoch:
            self.planner.reset_board()
            self.pending = None
            self.s1_advice = None
            self.neutral_until = max(self.neutral_until, now + 2)
            self.s2_reason = "new scored board: choose a timing cohort"
            self.log("timing_board_reset", tick=now, epoch=self.book.epoch,
                     inferred_clears=self.book.clears)

        if lost:
            edge = None
            if isinstance(self.pending, dict):
                edge = (self.pending.get("origin"), self.pending.get("destination"))
            if edge is None or edge[0] is None or edge[1] is None:
                edge = self.last_edge
            self.temporal.note_loss(edge, now, self.field)
            self.life_epoch += 1
            self.pending = None
            self.s1_advice = None
            # Alternate cohorts after every loss, not just the first one.  This
            # prevents a second life from replaying the same post-loss opening;
            # the temporal ledger still keeps the exact failed edge evidence.
            next_mode = ("alternate" if self.planner.mode == "primary" else "primary")
            self.planner.set_mode(next_mode)
            self.planner.safety = "dephase"
            self.planner.policy_version += 1
            self.planner.target = None
            self.planner.cursor = 0
            self.planner.recent_nodes.clear()
            self.recover_until = max(self.recover_until, now + 6)
            self.neutral_until = max(self.neutral_until, now + 9)
            self.s2_reason = "life loss: switch route cohort and dephase"
            self.log("timing_recovery", tick=now, lives=lives,
                     life_epoch=self.life_epoch, edge=edge,
                     mode=self.planner.mode)

        self.lives = lives
        self.life_counter = loss_counter

    def compact_state(self, obs, projected, candidates, source_action):
        now = tick_of(obs)
        origin = node_of(obs.get("qbert_node")) if isinstance(obs, dict) else None
        option_data = []
        for c in candidates[:4]:
            option_data.append({
                "key": c.get("key"),
                "landing": "unfinished" if c.get("useful") else "transit",
                "immediate_danger": "concrete" if finite(c.get("risk")) >= 4.0 else "clear",
                "escape": "open" if integer(c.get("exits")) >= 2 else
                          "narrow" if integer(c.get("exits")) == 1 else "closed",
                "timing": "repeat_failure_phase" if finite(c.get("temporal")) >= 2.3 else
                          "ordinary_phase",
                "code_rank": "first" if c is candidates[0] else "alternate",
            })
        return safe_json({
            "game": "Atari 2600 Q*bert Atari port, six-row 21-cube pyramid",
            "decision": "The current hop is in flight; judge only the supplied next-junction options.",
            "current_hop": {"posted": source_action, "projected_landing": projected},
            "position": {"standing_after_hop": projected,
                         "safe_edges_now": [x for x in names_list(obs.get("safe_moves"))
                                             if x in DIRECTIONS]},
            "progress": {"inferred_board_clears": self.book.clears,
                          "unfinished_bucket": "many" if sum(v > 0 for v in self.book.need) > 8 else
                                               "several" if sum(v > 0 for v in self.book.need) > 2 else "few",
                          "priority": "survival, required phase progress, no bonus chase"},
            "timing": self.temporal.facts(now, self.field),
            "route_posture": {"cohort": self.planner.mode, "safety": self.planner.safety},
            "hazards": self.field.semantic(origin, now),
            "next_options": option_data,
            "rule": "Choose only a supplied option. Code rechecks node, current safe_moves, timing cost, and freshness.",
        })

    def start_jev_ahead(self, obs, source_action, landing):
        if self.s1_future is not None or self.ctx.done or landing is None:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        now = tick_of(obs)
        if now - self.last_s1_tick < 8:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 2:
                return
        except Exception:
            pass

        # Predict the next junction from the fixed board, but do not pretend to
        # know its future safe_moves.  The current motor rechecks those later.
        future = self.planner.candidates_for(landing, set(GRAPH.get(landing, {})), now + 8)
        if len(future) < 1:
            return
        shown = future[:4]
        keys = [c["key"] for c in shown]
        if len(keys) < 2:
            return
        descriptions = {}
        for c in shown:
            descriptions[c["key"]] = {
                "what": "Take this supplied edge for one hop after the current landing.",
                "landing": "unfinished cube" if c.get("useful") else "transit cube",
                "danger": "concrete" if finite(c.get("risk")) >= 4.0 else "clear",
                "escape": "open" if integer(c.get("exits")) >= 2 else
                          "narrow" if integer(c.get("exits")) == 1 else "closed",
                "timing": "near a recorded loss phase" if finite(c.get("temporal")) >= 2.3
                          else "ordinary timing",
                "not_for": "inventing a direction, FIRE, or bonus chase",
            }
        descriptions["keep_code"] = "Use the freshly recomputed deterministic temporal ranking."
        instruction = (
            "At the next junction, choose the safest supplied edge. Prefer clear immediate danger, "
            "then an open escape, then an unfinished landing. If a supplied edge repeats a recorded "
            "loss phase, prefer a different timing or keep_code. Do not change the in-flight hop and "
            "never invent an option."
        )
        first = list(keys) + ["keep_code"]
        second = list(reversed(keys)) + ["keep_code"]
        questions = {
            "timed_edge_a": Choice(instruction, {k: descriptions[k] for k in first}),
            "timed_edge_b": Choice(instruction, {k: descriptions[k] for k in second}),
            "dephase_needed": Noul(
                "Do the supplied facts support paying one short neutral tick before the next edge? "
                "Answer true only when a timing-phase conflict is explicit and the current cube has an escape.",
                {"true": "A bounded timing shift is supported by a repeat-phase fact and an open escape.",
                 "false": "No explicit repeat-phase conflict or the landing would be unsafe to delay."}),
        }
        state = self.compact_state(obs, landing, shown, source_action)
        meta = {"created": now, "expected_node": landing, "epoch": self.book.epoch,
                "life": self.life_epoch, "policy": self.planner.policy_version,
                "signature": self.field.signature(now), "keys": keys,
                "ranked": [dict(c) for c in shown]}
        self.last_s1_tick = now

        def work():
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception as exc:
                return {"_error": type(exc).__name__}

        try:
            self.s1_future = self.ctx.spawn(work)
            self.s1_meta = meta
            self.log("timed_review_started", tick=now, landing=landing, options=keys)
        except Exception as exc:
            self.s1_future = None
            self.s1_meta = None
            self.log("timed_review_spawn_failed", error=type(exc).__name__)

    def collect_jev(self, obs):
        if self.s1_future is None or not self.s1_future.done():
            return
        future, meta = self.s1_future, self.s1_meta
        self.s1_future = None
        self.s1_meta = None
        try:
            answers = future.result()
            if not isinstance(answers, dict) or "_error" in answers or not isinstance(meta, dict):
                self.log("timed_review_invalid")
                return
            now = tick_of(obs)
            if (node_of(obs.get("qbert_node")) != meta.get("expected_node") or
                    self.book.epoch != meta.get("epoch") or
                    self.life_epoch != meta.get("life") or
                    self.planner.policy_version != meta.get("policy") or
                    self.field.signature(now) != meta.get("signature") or
                    now - integer(meta.get("created"), now) < 0 or
                    now - integer(meta.get("created"), now) > 12):
                self.log("timed_review_stale", tick=now)
                return
            a = answer_string(answers.get("timed_edge_a"))
            b = answer_string(answers.get("timed_edge_b"))
            agree = a == b and a in list(meta.get("keys", [])) + ["keep_code"]
            p = min(probability(answers.get("timed_edge_a"), a),
                    probability(answers.get("timed_edge_b"), b)) if agree else 0.0
            dephase = finite(getattr(answers.get("dephase_needed"), "value", 0.0), 0.0)
            self.s1_advice = {**meta, "choice": a if agree else None,
                              "probability": p, "dephase": dephase}
            self.log("timed_review_result", tick=now, choice=a, probability=p,
                     dephase=dephase)
        except Exception as exc:
            self.log("timed_review_result_failed", error=type(exc).__name__)

    def apply_advice(self, obs, candidates):
        fallback = candidates[0] if candidates else None
        advice = self.s1_advice
        self.s1_advice = None
        if fallback is None or not advice:
            return fallback
        now = tick_of(obs)
        here = node_of(obs.get("qbert_node"))
        if (advice.get("expected_node") != here or
                advice.get("epoch") != self.book.epoch or
                advice.get("life") != self.life_epoch or
                advice.get("policy") != self.planner.policy_version or
                now - integer(advice.get("created"), now) > 3):
            self.log("timed_advice_expired", tick=now)
            return fallback
        key = advice.get("choice")
        if key == "keep_code" or finite(advice.get("probability")) < 0.60:
            return fallback
        chosen = next((c for c in candidates if c.get("key") == key), None)
        if chosen is None:
            return fallback
        if (chosen.get("risk", 99.0) <= fallback.get("risk", 99.0) + 0.45 and
                chosen.get("temporal", 99.0) <= fallback.get("temporal", 99.0) + 1.0 and
                chosen.get("cost", 99.0) <= fallback.get("cost", 99.0) + 1.8 and
                chosen.get("exits", 0) >= max(0, fallback.get("exits", 0) - 1) and
                (chosen.get("useful") or not fallback.get("useful"))):
            if chosen.get("key") != fallback.get("key"):
                self.log("timed_advice_changed_edge", tick=now,
                         old=fallback.get("key"), new=chosen.get("key"),
                         probability=advice.get("probability"))
            return chosen
        self.log("timed_advice_rejected", tick=now, option=key)
        return fallback

    def start_astra(self, obs, reason):
        if self.s2_future is not None or self.s2_calls >= 2 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            pass
        now = tick_of(obs)
        summary = safe_json({
            "reason": reason, "tick": now, "score": finite(obs.get("score")),
            "lives": self.lives, "inferred_clears": self.book.clears,
            "unfinished": sum(v > 0 for v in self.book.need),
            "timing": self.temporal.facts(now, self.field),
            "current": {"cohort": self.planner.mode, "safety": self.planner.safety},
            "recent_hazards": self.field.semantic(node_of(obs.get("qbert_node")), now),
        })
        prompt = (
            "You are the slow strategic adviser for a real-time Atari 2600 Q*bert harness. "
            "Code owns the 21-node graph, exact phase demand, rendered-hazard extraction, and a "
            "one-hop motor. The unresolved hypothesis is temporal phase-lock: a route can be "
            "geometrically legal yet repeatedly meet an enemy at the same scheduler phase. Jev "
            "reviews only supplied next-junction edges. Choose a durable posture and return ONLY "
            "compact JSON with timing in [steady,dephase,diverse] and coverage in [primary,alternate]. "
            "steady preserves throughput; dephase gives a small cost to repeat-phase edges and permits "
            "two bounded waits; diverse favors the alternate coverage cohort after a loss. Never return "
            "an action, route, node list, FIRE, or prose. Invalid output is ignored.\n" +
            json.dumps(summary, separators=(",", ":")))
        self.s2_calls += 1
        session = self.s2_session

        def work():
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=105.0)
                return {"text": getattr(result, "text", ""),
                        "session": getattr(result, "session_id", None)}
            except Exception as exc:
                return {"error": type(exc).__name__}

        try:
            self.s2_future = self.ctx.spawn(work)
            self.s2_reason = None
            self.log("timing_posture_started", tick=now, reason=reason)
        except Exception as exc:
            self.s2_future = None
            self.log("timing_posture_spawn_failed", error=type(exc).__name__)

    def collect_astra(self, obs):
        if self.s2_future is None or not self.s2_future.done():
            return
        future = self.s2_future
        self.s2_future = None
        try:
            result = future.result()
            if not isinstance(result, dict) or result.get("error"):
                self.log("timing_posture_invalid")
                return
            if result.get("session"):
                self.s2_session = result.get("session")
            data = parse_object(result.get("text", ""))
            old = (self.planner.mode, self.planner.safety)
            timing = data.get("timing")
            coverage = data.get("coverage")
            if timing in ("steady", "dephase", "diverse"):
                self.planner.safety = "ordinary" if timing == "steady" else timing
            if coverage == "primary":
                self.planner.set_mode("primary")
            elif coverage == "alternate":
                self.planner.set_mode("alternate")
            new = (self.planner.mode, self.planner.safety)
            if new != old:
                self.planner.policy_version += 1
                self.s1_advice = None
                self.log("timing_posture_changed", tick=tick_of(obs), old=list(old), new=list(new))
            else:
                self.log("timing_posture_unchanged", tick=tick_of(obs), parsed=bool(data))
        except Exception as exc:
            self.log("timing_posture_failed", error=type(exc).__name__)

    def motor(self, obs):
        now = tick_of(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)
        if (self.temporal.mode_until >= 0 and now >= self.temporal.mode_until
                and self.planner.safety == "dephase"):
            self.planner.safety = "ordinary"
            self.planner.policy_version += 1
            self.s1_advice = None
            self.log("timing_dephase_expired", tick=now)

        if self.book.flashing:
            self.pending = None
            return self.noop
        here = node_of(obs.get("qbert_node"))
        motion = qmotion(obs)
        phase = str(motion.get("phase", "unknown")).lower()
        if here is None or phase in ("falling", "not_visible") or bool(motion.get("falling")):
            self.pending = None
            return self.noop

        if now < self.recover_until:
            return self.noop
        if self.recover_until and now >= self.recover_until:
            self.recover_until = 0
        if now < self.wait_until:
            return self.noop

        if self.pending is not None:
            hop = self.pending
            age = now - integer(hop.get("start"), now)
            if bool(obs.get("qbert_in_hop")) or here != hop.get("origin"):
                hop["moving"] = True
            if hop.get("moving") and standing(obs):
                self.pending = None
                self.planner.note_landing(here, hop.get("destination"), hop.get("origin"))
                self.neutral_until = max(self.neutral_until, now + 1)
                if here == hop.get("destination"):
                    self.log("timed_hop_landed", tick=now, origin=hop.get("origin"),
                             destination=here, duration=age)
                else:
                    self.log("timed_hop_mismatch", tick=now, origin=hop.get("origin"),
                             expected=hop.get("destination"), observed=here, duration=age)
                    self.neutral_until = max(self.neutral_until, now + 4)
                return self.noop
            if (not hop.get("moving") and standing(obs) and age <= 3 and
                    integer(hop.get("posts"), 0) < 2):
                hop["posts"] += 1
                return hop.get("action", self.noop)
            if age > 18 and standing(obs):
                self.pending = None
                self.neutral_until = max(self.neutral_until, now + 3)
                self.log("timed_hop_not_started", tick=now,
                         origin=hop.get("origin"), destination=hop.get("destination"))
            else:
                return self.noop

        if now < self.neutral_until or not standing(obs):
            return self.noop
        candidates = self.planner.available(obs)
        if not candidates:
            return self.noop
        chosen = self.apply_advice(obs, candidates) or candidates[0]
        action = chosen.get("action")
        if action == "NOOP" or chosen.get("key") == "wait":
            self.temporal.note_wait(now)
            self.wait_until = now + 1
            self.log("phase_dephase_wait", tick=now, node=here,
                     temporal=chosen.get("temporal"))
            return self.noop
        safe_now = set(names_list(obs.get("safe_moves")))
        if action not in DIRECTIONS or action not in safe_now or action not in GRAPH.get(here, {}):
            return self.noop
        destination = GRAPH[here][action]
        self.pending = {"origin": here, "destination": destination, "action": action,
                        "start": now, "posts": 1, "moving": False}
        self.last_edge = (here, destination)
        self.temporal.note_use(here, destination, now, self.field)
        self.start_jev_ahead(obs, action, destination)
        if now - self.last_log >= 25:
            self.log("timed_route_decision", tick=now, score=finite(obs.get("score")),
                     lives=self.lives, epoch=self.book.epoch, clears=self.book.clears,
                     cohort=self.planner.mode, safety=self.planner.safety, node=here,
                     target=self.planner.target, action=action, destination=destination,
                     risk=chosen.get("risk"), temporal=chosen.get("temporal"),
                     exits=chosen.get("exits"), clock_band=chosen.get("clock_band"))
            self.last_log = now
        return action

    def run(self):
        ctx = self.ctx
        obs = ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.ingest(obs)
        self.start_astra(obs, "initial route cohort and timing policy")
        while not ctx.done and not getattr(self.game, "done", False):
            obs = ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.ingest(obs)
            action = self.motor(obs)
            if self.s2_reason:
                self.start_astra(obs, self.s2_reason)
            elif self.s2_calls < 2 and self.book.clears >= 4 and now_since(obs, self.book.last_progress) > 180:
                self.start_astra(obs, "late board: review temporal phase-lock")
            if action not in self.action_names:
                action = self.noop
            try:
                ctx.act(action)
                ctx.wait_for_tick()
            except Exception as exc:
                self.log("timed_loop_exit", error=type(exc).__name__)
                return
        self.log("candidate_summary", score=getattr(self.game, "score", None),
                 inferred_clears=self.book.clears, epochs=self.book.epoch,
                 lives=self.lives, failures=len(self.temporal.failures),
                 cohort=self.planner.mode, safety=self.planner.safety)


def now_since(obs, then):
    return max(0, tick_of(obs) - integer(then, tick_of(obs)))


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
