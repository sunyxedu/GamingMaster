"""0025: parent 0016 corridors with a soft late Coily reachability margin.

The controller separates three jobs which the earlier radar controllers mixed:

* ``ColourBook`` owns the board-local colour evidence and completion epochs.
* ``CorridorPlanner`` proposes a *corridor* (an edge plus the exits that remain
  after that edge), using public enemy positions and a short time envelope.
* ``HandshakeMotor`` owns the real-time joystick protocol.  It clears the
  previous sticky edge before arming a new one, posts an edge only while Q*bert
  is standing, and accepts only the observed landing.

Jev judges supplied corridor descriptions, rather than being allowed to invent
an action.  Astra chooses only a durable coverage/safety posture.  Both model
outputs are optional and are never on the tick clock.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
TIE_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")


def build_graph():
    graph = {n: {} for n in range(21)}
    rows = {}
    for row in range(6):
        first = row * (row + 1) // 2
        for col in range(row + 1):
            node = first + col
            rows[node] = row
            if row:
                if col < row:
                    graph[node]["UP"] = node - row
                if col:
                    graph[node]["LEFT"] = node - row - 1
            if row < 5:
                child = (row + 1) * (row + 2) // 2
                graph[node]["DOWN"] = child + col
                graph[node]["RIGHT"] = child + col + 1
    return graph, rows


GRAPH, ROW = build_graph()


def all_distances():
    result = []
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
        result.append(distance)
    return result


DIST = all_distances()

# A graph preference only.  It is not an action macro: every edge is freshly
# checked against safe_moves and the motor observes its result.
COVERAGE_ORDER = (1, 4, 7, 11, 6, 10, 16, 12, 17,
                  18, 13, 9, 14, 19, 20, 8, 2, 0, 3, 5, 15)
COVERAGE_RANK = {node: i for i, node in enumerate(COVERAGE_ORDER)}


def finite(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
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
    # Observations use native ints.  Being a little stricter here prevents a
    # malformed stand-in observation from turning a string into an edge.
    if isinstance(value, bool):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if 0 <= value < 21 else None


def tick_of(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)), 0)


def qmotion(obs):
    value = obs.get("qbert_motion")
    if isinstance(value, dict):
        return value
    return {"phase": str(value or "unknown")}


def standing(obs):
    n = node_of(obs.get("qbert_node"))
    motion = qmotion(obs)
    phase = str(motion.get("phase", "unknown")).lower()
    return (n is not None and not bool(obs.get("qbert_in_hop"))
            and phase not in ("hopping", "falling", "not_visible")
            and not bool(motion.get("falling")))


def palette(obs):
    value = obs.get("cube_colors")
    if not isinstance(value, list) or len(value) != 21:
        return None
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        result.append(item)
    return tuple(result)


def list_names(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str)]


def safe_json(value):
    """Return a strict JSON-shaped copy for model requests and logs."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    return str(value)


def answer_choice(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value if isinstance(value, str) else None


def answer_probability(answer, key):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return 0.0
    return finite(probabilities.get(key), 0.0)


def first_object(text):
    """Parse arbitrary Astra output without making prose a control command."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(text[:16000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


class ColourBook:
    """Track a learned destination palette without trusting stale phase labels."""

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
        self.complete_since = None
        self.last_progress = 0
        self.evidence = deque(maxlen=10)

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def update(self, obs):
        now = tick_of(obs)
        current = palette(obs)
        score = finite(obs.get("score"), self.last_score or 0.0)
        delta = score - self.last_score if self.last_score is not None else 0.0
        if current is None:
            self.last_score = score
            return False

        changed = []
        if self.previous is not None:
            changed = [i for i in range(21)
                       if current[i] != self.previous[i]]

        if self.previous is not None and len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_score = self.last_score if self.last_score is not None else score - delta
                self.flash_origin = node_of(obs.get("qbert_node"))
                self.log("corridor_flash", tick=now, origin=self.flash_origin,
                         score=score)
            self.flash_last_change = now

        if self.flashing:
            quiet = now - self.flash_last_change
            returned = node_of(obs.get("qbert_node")) == 0
            if quiet >= 4 and (returned or quiet >= 7):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.flashing = False
                self.epoch += 1
                self.goal = None
                self.need = [1] * 21
                self.complete_since = None
                self.last_progress = now
                self.evidence.clear()
                self.log("corridor_epoch", tick=now, epoch=self.epoch,
                         inferred_clears=self.clears, tally=tally)
                self.previous = current
                self.last_score = score
                return True
            self.need = [1] * 21
            self.previous = current
            self.last_score = score
            return False

        # A 25-point cube component is a stronger calibration signal than a
        # decoder phase after a palette handoff.  325 also qualifies because a
        # safe Sam/Green event can accompany the cube change.
        local_landing = (delta > 0 and int(round(delta)) % 100 == 25
                         and 0 < len(changed) <= 3)
        if local_landing:
            qnode = node_of(obs.get("qbert_node"))
            landed = qnode if qnode in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                new_goal = current[landed]
                if new_goal != self.goal:
                    self.goal = new_goal
                    self.log("corridor_goal", tick=now, epoch=self.epoch,
                             node=landed, color=new_goal, reward=delta)
                self.last_progress = now
                phases = obs.get("cube_phase")
                phase = phases[landed] if isinstance(phases, list) and landed < len(phases) else None
                self.evidence.append({"tick": now, "node": landed,
                                      "color": new_goal, "reward": delta,
                                      "decoder_phase": phase})

        if self.goal is None:
            # Bootstrap from needed_landings only while no scored colour has
            # established a local goal.  The post-flash all-zero vector is not
            # allowed to resurrect a one-cube target.
            raw_need = obs.get("needed_landings")
            if isinstance(raw_need, list) and len(raw_need) == 21 and any(integer(x, 0) > 0 for x in raw_need):
                self.need = [max(0, integer(x, 1)) for x in raw_need]
            else:
                self.need = [1] * 21
        else:
            self.need = [0 if color == self.goal else 1 for color in current]

        if not any(self.need):
            if self.complete_since is None:
                self.complete_since = now
            # Level 2/4/5 can have an all-equal intermediate palette without a
            # global tally.  Release a second coverage pass, but never claim a
            # clear until the actual flash score is observed.
            elif now - self.complete_since >= 24:
                self.goal = None
                self.need = [1] * 21
                self.complete_since = None
                self.last_progress = now
                self.log("corridor_cycle_probe", tick=now, epoch=self.epoch)
        else:
            self.complete_since = None

        self.previous = current
        self.last_score = score
        return False


class EnemyField:
    """Public-object memory plus a small time-expanded edge certificate."""

    def __init__(self):
        self.centers = {}
        self.items = {}
        self.last_scheduler = None
        self.scheduler_reset_until = -1
        self.last_purple_state = None

    def learn_centers(self, obs):
        if self.centers:
            return
        details = obs.get("cube_details")
        if not isinstance(details, list):
            return
        for item in details:
            if isinstance(item, dict):
                n = node_of(item.get("node"))
                if n is not None:
                    self.centers[n] = (finite(item.get("x")) + 10.0,
                                       finite(item.get("y")) - 7.0)

    def update(self, obs, late=False):
        now = tick_of(obs)
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
                x = finite(item.get("x")) + finite(item.get("w"), 7) / 2.0
                y = finite(item.get("y")) + finite(item.get("h"), 8) / 2.0
                if n is None and self.centers:
                    n = min(self.centers, key=lambda k:
                            (self.centers[k][0] - x) ** 2 +
                            (self.centers[k][1] - y) ** 2)
                old = self.items.get(kind)
                dx = finite(item.get("dx"))
                dy = finite(item.get("dy"))
                if old and abs(dx) < 0.01 and abs(dy) < 0.01:
                    dx, dy = old.get("dx", 0.0), old.get("dy", 0.0)
                if n is not None:
                    self.items[kind] = {"kind": kind, "node": n,
                                        "x": x, "y": y, "dx": dx, "dy": dy,
                                        "last": now, "visible": True}
                    visible.add(kind)

        # Keep a rolling object only long enough to cover one possible hop.
        # This is deliberately narrower than 0009's broad graph radar.
        for kind in list(self.items):
            age = now - integer(self.items[kind].get("last"), now)
            limit = 9 if kind in ("purple_ball", "red_ball") else 6
            if age > limit:
                del self.items[kind]
            else:
                self.items[kind]["visible"] = kind in visible

        timers = obs.get("timers")
        scheduler = integer(timers.get("object_scheduler"), -1) if isinstance(timers, dict) else -1
        if self.last_scheduler is not None and scheduler >= 0:
            # A reset is evidence of a new object scheduling opportunity, not
            # proof of a hazard at any particular node.
            if scheduler < self.last_scheduler - 40:
                self.scheduler_reset_until = max(self.scheduler_reset_until, now + (5 if late else 2))
        if scheduler >= 0:
            self.last_scheduler = scheduler
        self.last_purple_state = integer(timers.get("purple_ball_state"), -1) if isinstance(timers, dict) else -1

    @staticmethod
    def down_children(nodes):
        result = set()
        for n in nodes:
            if n in GRAPH:
                result.update(there for action, there in GRAPH[n].items()
                             if action in ("DOWN", "RIGHT"))
        return result

    def possible_at(self, kind, now, horizon):
        item = self.items.get(kind)
        if not item:
            return set()
        node = item.get("node")
        if node is None:
            return set()
        age = max(0, now - integer(item.get("last"), now))
        # One board edge takes roughly one hop; the rolling objects are slower
        # than Q*bert.  Use a possibility cone, not a claimed exact trajectory.
        steps = min(2, max(0, (age + horizon) // 9))
        possible = {node}
        frontier = {node}
        if kind in ("purple_ball", "red_ball"):
            for _ in range(steps):
                frontier = self.down_children(frontier)
                possible.update(frontier)
        else:
            radius = min(2, max(1, (age + horizon) // 10))
            frontier = {node}
            for _ in range(radius):
                frontier = {there for here in frontier for there in GRAPH[here].values()}
                possible.update(frontier)
        return possible

    def edge_risk(self, origin, destination, action, now, late=False):
        value = 0.0
        details = []
        for kind, item in self.items.items():
            n = item.get("node")
            if n is None:
                continue
            age = max(0, now - integer(item.get("last"), now))
            # A hazard next to the takeoff cube can collide before the intended
            # landing.  The old controllers mainly scored the destination.
            source_distance = DIST[n][origin] if origin is not None else 99
            if kind == "coily" and source_distance <= 1:
                value += 3.0
                details.append("coily_takeoff")
            if kind in ("purple_ball", "red_ball") and source_distance == 0:
                value += 3.0
                details.append("rolling_takeoff")

            arrival = self.possible_at(kind, now, 8)
            if destination in arrival:
                value += 5.5 if age <= 3 else 3.0
                details.append(kind + "_arrival")
            if kind in ("purple_ball", "red_ball"):
                next_branch = self.down_children(arrival)
                if destination in next_branch and destination not in arrival:
                    value += 2.0
                    details.append(kind + "_branch")
            elif kind == "coily":
                # Keep the parent's one-hop pursuit costs unchanged. A fresh
                # rendered Coily adds only a soft cost when the landing lies in
                # the second reachability ring; this is uncertainty, not a claim
                # that the snake will occupy that cube.
                if late and item.get("visible") and age <= 3 and source_distance <= 2:
                    extended = self.possible_at(kind, now, 20)
                    if destination in extended and destination not in arrival:
                        value += 0.25
                        details.append(kind + "_two_hop_window")
                if n in GRAPH and DIST[n][destination] <= 1:
                    value += 0.7
                    details.append("coily_near_landing")

        # Timer/scheduler uncertainty affects the certificate only slightly;
        # it cannot manufacture an enemy node.  This avoids 0013's blanket
        # UP/LEFT guard, which still lost despite many logged interventions.
        if late and now < self.scheduler_reset_until:
            value += 0.25
            details.append("spawn_window")
        return value, details

    def exit_count(self, node, origin, now, late=False):
        count = 0
        for action, there in GRAPH.get(node, {}).items():
            risk, _ = self.edge_risk(node, there, action, now + 2, late)
            if there != origin and risk < 3.2:
                count += 1
        return count

    def signature(self, now):
        return tuple(sorted((kind, item.get("node"),
                             "visible" if item.get("visible") else "memory",
                             min(2, now - integer(item.get("last"), now)))
                            for kind, item in self.items.items()))

    def facts(self, origin, now, late=False):
        result = []
        for kind, item in self.items.items():
            n = item.get("node")
            distance = DIST[origin][n] if origin is not None and n is not None else 99
            result.append({"kind": kind,
                           "visibility": "rendered" if item.get("visible") else "recent_memory",
                           "proximity": "same" if distance == 0 else
                                        "adjacent" if distance == 1 else
                                        "near" if distance <= 2 else "far",
                           "motion": "rolling_down" if kind in ("purple_ball", "red_ball")
                                     else "pursuit"})
        if late and now < self.scheduler_reset_until and not self.items:
            result.append({"kind": "spawn_window", "visibility": "uncertain",
                           "proximity": "unknown", "motion": "not_localized"})
        return result


class CorridorPlanner:
    """Select a useful edge whose *landing corridor* remains escapable."""

    def __init__(self, book, field):
        self.book = book
        self.field = field
        self.target = None
        self.cursor = 0
        self.previous_node = None
        self.recent_nodes = deque(maxlen=22)
        self.wrong_edges = Counter()
        self.posture = "spine"
        self.safety = "certified"
        self.wait_started = None

    def reset_board(self):
        self.target = None
        self.cursor = 0
        self.previous_node = None
        self.recent_nodes.clear()
        self.wrong_edges.clear()
        self.wait_started = None

    def note_landing(self, node, expected=None, source=None):
        if node is None:
            return
        if expected is not None and node != expected and source is not None:
            self.wrong_edges[(source, expected)] += 1
        if self.previous_node != node:
            self.recent_nodes.append(node)
        self.previous_node = node
        if self.target == node and self.book.need[node] <= 0:
            self.target = None

    def unfinished(self):
        return [n for n, value in enumerate(self.book.need) if value > 0]

    def choose_target(self, obs):
        current = node_of(obs.get("qbert_node"))
        remaining = set(self.unfinished())
        if self.target not in remaining:
            self.target = None
        if self.target is not None and self.target != current:
            return self.target

        if self.target == current and self.book.need[current] <= 0:
            self.target = None

        # A stable preference prevents the final cubes from being selected by
        # changing nearest-distance ties on every standing observation.
        if self.target is None:
            for offset in range(len(COVERAGE_ORDER)):
                index = (self.cursor + offset) % len(COVERAGE_ORDER)
                candidate = COVERAGE_ORDER[index]
                if candidate in remaining and candidate != current:
                    self.cursor = index
                    self.target = candidate
                    break

        if self.target is None and remaining:
            ranked = sorted(remaining, key=lambda n: (DIST[current][n] if current is not None else 99,
                                                       ROW[n], n))
            self.target = ranked[0]
        return self.target

    def available(self, obs):
        origin = node_of(obs.get("qbert_node"))
        if origin is None:
            return []
        safe = set(list_names(obs.get("safe_moves")))
        legal = set(list_names(obs.get("legal_moves")))
        allowed = safe if safe else legal
        allowed = [a for a in TIE_ORDER if a in allowed and a in GRAPH[origin]]
        if not allowed:
            return []
        target = self.choose_target(obs)
        remaining = self.unfinished()
        late = self.book.clears >= 4
        candidates = []
        visits = Counter(self.recent_nodes)
        for action in allowed:
            destination = GRAPH[origin][action]
            risk, reasons = self.field.edge_risk(origin, destination, action,
                                                   tick_of(obs), late)
            if any(reason.endswith("_two_hop_window") for reason in reasons):
                self.book.ctx.log("two_hop_risk_applied", tick=tick_of(obs),
                                  origin=origin, destination=destination,
                                  action=action, risk_reasons=reasons)
            exits = self.field.exit_count(destination, origin, tick_of(obs), late)
            distance = min((DIST[destination][n] for n in remaining), default=0)
            useful = self.book.need[destination] > 0
            # A corridor is a landing plus a next-exit certificate.  It is not
            # a hard block unless the adapter already removed the edge.
            corridor = ("open" if exits >= 2 else
                        "narrow" if exits == 1 else "closed")
            cost = 2.0 * distance + (0.0 if useful else 0.55)
            cost += 11.0 * min(risk, 10.0)
            cost -= 0.20 * min(exits, 3)
            cost += 0.11 * visits[destination]
            if destination == self.previous_node and len(allowed) > 1:
                cost += 0.45
            if self.wrong_edges[(origin, destination)] > 0:
                cost += min(2.0, 0.75 * self.wrong_edges[(origin, destination)])
            if target is not None:
                cost += 0.15 * DIST[destination][target]
            if self.posture == "upper" and useful:
                cost += 0.12 * ROW[destination]
            elif self.posture == "frontier" and useful:
                cost -= 0.14 * ROW[destination]
            if self.safety == "shelter":
                cost -= 0.20 * exits
            candidates.append({"key": "edge_" + action, "action": action,
                               "destination": destination, "cost": cost,
                               "risk": risk, "risk_reasons": reasons,
                               "useful": useful, "distance": distance,
                               "exits": exits, "corridor": corridor,
                               "target": target})
        candidates.sort(key=lambda item: (item["risk"] >= 7.0,
                                          item["cost"],
                                          COVERAGE_RANK.get(item["destination"], 99),
                                          TIE_ORDER.index(item["action"])))

        # Deliberate waiting is offered only when the current cube is safer and
        # a rolling hazard is actually represented by the public state/memory.
        # It is bounded by the motor, so it cannot recreate the 0008 goal stall.
        current_risk, current_reasons = self.field.edge_risk(
            origin, origin, "NOOP", tick_of(obs), late)
        has_rolling = any(k in self.field.items for k in ("purple_ball", "red_ball"))
        if (candidates and has_rolling and current_risk + 0.5 < candidates[0]["risk"]
                and candidates[0]["risk"] >= 2.0):
            if self.wait_started is None:
                self.wait_started = tick_of(obs)
            elif tick_of(obs) - self.wait_started >= 8:
                # The renderer may not reveal a branch promptly.  A bounded
                # wait must yield to a certified edge rather than becoming an
                # idle goal trap.
                self.wait_started = None
            else:
                candidates.insert(0, {"key": "wait", "action": "NOOP",
                                      "destination": origin, "cost": candidates[0]["cost"] - 0.35,
                                      "risk": current_risk, "risk_reasons": current_reasons,
                                      "useful": False, "distance": min((DIST[origin][n] for n in remaining), default=0),
                                      "exits": self.field.exit_count(origin, origin, tick_of(obs), late),
                                      "corridor": "wait", "target": target})
        else:
            self.wait_started = None
        return candidates


class Controller:
    """Nonblocking model bridge and sticky-action-safe state machine."""

    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.book = ColourBook(ctx)
        self.field = EnemyField()
        self.planner = CorridorPlanner(self.book, self.field)

        self.lives = None
        self.life_epoch = 0
        self.last_life_loss_count = None
        self.pending_hop = None
        self.neutral_until = 0
        self.recovery_until = 0
        self.recovery_top_seen = False
        self.last_unplanned_node = None
        self.last_log = -999

        self.s1_future = None
        self.s1_meta = None
        self.last_s1_tick = -999
        self.s1_advice = None

        self.s2_future = None
        self.s2_session = None
        self.s2_calls = 0
        self.s2_reason = "initial corridor posture"
        self.last_s2_tick = -999
        self.policy_version = 0

        try:
            self.names = {str(x) for x in game.action_spec.names}
        except Exception:
            self.names = {"NOOP", *DIRECTIONS}
        self.noop = "NOOP" if "NOOP" in self.names else next(iter(self.names), "NOOP")

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def ingest(self, obs):
        now = tick_of(obs)
        old_epoch = self.book.epoch
        self.book.update(obs)
        late = self.book.clears >= 4
        self.field.update(obs, late=late)

        lives = integer(obs.get("lives"), self.lives if self.lives is not None else 4)
        loss_count = integer(obs.get("life_lost_count"), self.last_life_loss_count if self.last_life_loss_count is not None else 0)
        lost = ((self.lives is not None and lives < self.lives) or
                (self.last_life_loss_count is not None and loss_count > self.last_life_loss_count))
        if self.book.epoch != old_epoch:
            self.planner.reset_board()
            self.pending_hop = None
            self.s1_advice = None
            self.neutral_until = max(self.neutral_until, now + 2)
            self.s2_reason = "new scored palette epoch"
            self.log("corridor_reset", tick=now, epoch=self.book.epoch,
                     inferred_clears=self.book.clears)
        if lost:
            self.life_epoch += 1
            self.pending_hop = None
            self.s1_advice = None
            self.planner.target = None
            self.planner.previous_node = None
            self.recovery_until = max(self.recovery_until, now + 4)
            self.neutral_until = max(self.neutral_until, now + 10)
            self.recovery_top_seen = False
            self.last_unplanned_node = None
            self.s2_reason = "life loss: audit corridor and recovery posture"
            self.log("handshake_recovery", tick=now, lives=lives,
                     life_epoch=self.life_epoch, wait_until=self.recovery_until)
        self.lives = lives
        self.last_life_loss_count = loss_count

    def current_signature(self, obs):
        return (self.field.signature(tick_of(obs)),
                self.field.last_purple_state,
                self.field.scheduler_reset_until > tick_of(obs))

    def compact_state(self, obs, candidates):
        origin = node_of(obs.get("qbert_node"))
        return safe_json({
            "game": "Atari 2600 Q*bert Atari port; 21-node six-row board",
            "rule": "Code owns graph legality, colour bookkeeping, and the two-tick neutral handshake. Never FIRE or invent an edge.",
            "position": {"node": origin, "standing": standing(obs),
                         "safe_moves": [a for a in list_names(obs.get("safe_moves")) if a in DIRECTIONS],
                         "last_posted": str(obs.get("last_action", "NOOP"))},
            "progress": {"epoch": self.book.epoch,
                          "inferred_clears": self.book.clears,
                          "unfinished": sum(value > 0 for value in self.book.need),
                          "target": self.planner.target,
                          "posture": self.planner.posture},
            "corridors": [{"key": c["key"], "action": c["action"],
                           "landing": "unfinished" if c["useful"] else "transit",
                           "danger": "occupied" if c["risk"] >= 7 else
                                     "concrete" if c["risk"] >= 2 else "clear",
                           "escape": c["corridor"],
                           "time_window": "spawn_uncertain" if "spawn_window" in c["risk_reasons"] else "ordinary",
                           "route": "short" if c["distance"] <= 1 else "long",
                           "code_preferred": c is candidates[0]}
                          for c in candidates],
            "hazards": self.field.facts(origin, tick_of(obs), self.book.clears >= 4),
            "fallback": candidates[0]["key"] if candidates else "wait",
        })

    def start_jev(self, obs, candidates):
        if self.s1_future is not None or not candidates or self.ctx.done:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        now = tick_of(obs)
        if now - self.last_s1_tick < 8 or not standing(obs):
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 1:
                return
        except Exception:
            pass

        # Avoid giving Jev a giant option set or opaque node arithmetic.  The
        # top four corridors are enough for the local judgement.
        shown = candidates[:4]
        keys = [item["key"] for item in shown]
        if "wait" not in keys:
            keys.append("wait")
        descriptions = {}
        for item in shown:
            descriptions[item["key"]] = {
                "what": "Take the supplied edge for one hop; code will release it and verify the landing.",
                "landing": "unfinished cube" if item["useful"] else "transit cube",
                "danger": "occupied/concrete" if item["risk"] >= 2 else "clear",
                "escape": item["corridor"],
                "not_for": "inventing an edge, FIRE, or chasing a bonus",
            }
        descriptions["wait"] = {"what": "Hold neutral for one deliberate tick while code waits for a represented rolling hazard.",
                                 "not_for": "a permanent substitute for coverage"}
        first = list(keys)
        second = list(reversed(first))
        instruction = (
            "Choose one supplied corridor for the next hop. Prefer a clear corridor over a concrete or occupied one; "
            "when danger is comparable prefer an open escape over a closed or narrow landing and then unfinished coverage. "
            "The neutral handshake is already enforced by code. Do not calculate distances and never invent a key."
        )
        criteria_a = {key: descriptions.get(key, "Use this supplied option only.") for key in first}
        criteria_b = {key: descriptions.get(key, "Use this supplied option only.") for key in second}
        questions = {
            "corridor_a": Choice(instruction, criteria_a),
            "corridor_b": Choice(instruction, criteria_b),
            "proposed_safe": Noul(
                "Do the supplied facts support the proposed corridor for the next hop? "
                "A concrete/occupied landing or a closed corridor is unsafe; distant or uncertain visibility alone is not proof.",
                {"true": "The proposed corridor is clear enough for this hop and preserves an escape.",
                 "false": "The proposed corridor has concrete lethal danger or no escape."}),
            "keep_escape": Noul(
                "Does the proposed corridor leave at least one meaningful legal exit after landing?",
                {"true": "The candidate explicitly has an open or narrow exit and code computed it.",
                 "false": "The candidate is closed or the exit fact is absent."}),
        }
        meta = {"tick": now, "node": node_of(obs.get("qbert_node")),
                "life": self.life_epoch, "epoch": self.book.epoch,
                "policy": self.policy_version, "keys": keys,
                "signature": self.current_signature(obs),
                "proposed": keys[0] if keys else "wait"}
        self.last_s1_tick = now

        def work():
            try:
                return {"answers": self.ctx.decide(self.compact_state(obs, shown), questions,
                                                     timeout_s=0.9), "error": None}
            except Exception as exc:
                return {"answers": None, "error": type(exc).__name__}

        try:
            self.s1_future = self.ctx.spawn(work)
            self.s1_meta = meta
        except Exception as exc:
            self.s1_future = None
            self.s1_meta = None
            self.log("jev_spawn_failure", error=type(exc).__name__)

    def collect_jev(self, obs):
        if self.s1_future is None or not self.s1_future.done():
            return
        future, meta = self.s1_future, self.s1_meta
        self.s1_future = None
        self.s1_meta = None
        try:
            result = future.result()
            answers = result.get("answers") if isinstance(result, dict) else None
            if not isinstance(answers, dict) or not isinstance(meta, dict):
                return
            now = tick_of(obs)
            if (meta.get("node") != node_of(obs.get("qbert_node")) or
                    meta.get("life") != self.life_epoch or
                    meta.get("epoch") != self.book.epoch or
                    meta.get("policy") != self.policy_version or
                    meta.get("signature") != self.current_signature(obs) or
                    now - integer(meta.get("tick"), now) < 0 or
                    now - integer(meta.get("tick"), now) > 8):
                self.log("jev_corridor_stale", origin_tick=meta.get("tick"), tick=now)
                return
            a = answer_choice(answers.get("corridor_a"))
            b = answer_choice(answers.get("corridor_b"))
            agree = a == b and a in meta.get("keys", [])
            confidence = min(answer_probability(answers.get("corridor_a"), a),
                             answer_probability(answers.get("corridor_b"), b)) if agree else 0.0
            safe_prob = finite(getattr(answers.get("proposed_safe"), "value", 0.0), 0.0)
            escape_prob = finite(getattr(answers.get("keep_escape"), "value", 0.0), 0.0)
            self.s1_advice = {"action_key": a if agree and confidence >= 0.60 else None,
                              "confidence": confidence, "safe": safe_prob,
                              "escape": escape_prob, **meta}
        except Exception as exc:
            self.log("jev_result_failure", error=type(exc).__name__)

    def advised(self, obs, candidates):
        fallback = candidates[0] if candidates else None
        advice = self.s1_advice
        self.s1_advice = None
        if fallback is None or not advice:
            return fallback
        now = tick_of(obs)
        if (advice.get("node") != node_of(obs.get("qbert_node")) or
                advice.get("life") != self.life_epoch or
                advice.get("epoch") != self.book.epoch or
                advice.get("policy") != self.policy_version or
                advice.get("signature") != self.current_signature(obs) or
                now - integer(advice.get("tick"), now) < 0 or
                now - integer(advice.get("tick"), now) > 3):
            return fallback
        chosen = next((c for c in candidates if c["key"] == advice.get("action_key")), None)
        if chosen is None:
            return fallback
        # Jev can select a corridor only when it agrees twice and its semantic
        # safety judgements do not contradict code's certificate.
        if advice.get("safe", 0.0) >= 0.62 and advice.get("escape", 0.0) >= 0.55:
            if (chosen["risk"] <= fallback["risk"] + 0.35 and
                    chosen["cost"] <= fallback["cost"] + 2.0 and
                    chosen["corridor"] != "closed"):
                if chosen["key"] != fallback["key"]:
                    self.log("jev_corridor_override", tick=now,
                             old=fallback["key"], new=chosen["key"],
                             confidence=advice.get("confidence"))
                return chosen
        return fallback

    def start_astra(self, obs, reason):
        if self.s2_future is not None or self.s2_calls >= 2 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        now = tick_of(obs)
        if self.s2_calls and now - self.last_s2_tick < 300:
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            pass
        summary = safe_json({
            "reason": reason, "tick": now, "score": finite(obs.get("score")),
            "lives": self.lives, "epoch": self.book.epoch,
            "inferred_clears": self.book.clears,
            "unfinished": sum(value > 0 for value in self.book.need),
            "recent_colour_evidence": list(self.book.evidence)[-4:],
            "hazards": self.field.facts(node_of(obs.get("qbert_node")), now,
                                        self.book.clears >= 4),
            "current_policy": {"posture": self.planner.posture,
                               "safety": self.planner.safety},
        })
        prompt = (
            "You are Astra, the slow strategic adviser for a real-time Atari 2600 Q*bert controller. "
            "Do not drive a joystick. Code has the exact 21-node graph, raw colour transition evidence, "
            "a one-hop corridor planner, and a sticky-action handshake that clears the previous edge before "
            "arming the next. Jev receives only supplied corridor facts and judges immediate safety. "
            "Choose a durable posture for future target ties: posture must be one of upper, spine, frontier; "
            "safety must be one of certified, shelter. upper favors the upper region early, spine favors the "
            "central coverage order, frontier favors deeper unfinished cubes. certified values throughput with "
            "an exit certificate; shelter gives a small preference to open exits after losses or invisible hazards. "
            "Never propose actions, FIRE, a macro, an arcade node, or raw RAM. Return ONLY JSON under 35 tokens, "
            "for example {\"posture\":\"spine\",\"safety\":\"certified\"}. Invalid or prose output is ignored.\n" +
            json.dumps(summary, separators=(",", ":")))
        self.s2_calls += 1
        self.last_s2_tick = now
        self.s2_reason = None
        session = self.s2_session

        def work():
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=100.0)
                return {"text": getattr(result, "text", ""),
                        "session_id": getattr(result, "session_id", None), "error": None}
            except Exception as exc:
                return {"text": "", "session_id": None, "error": type(exc).__name__}

        try:
            self.s2_future = self.ctx.spawn(work)
            self.log("astra_posture_requested", tick=now, reason=reason)
        except Exception as exc:
            self.s2_future = None
            self.log("astra_spawn_failure", error=type(exc).__name__)

    def collect_astra(self, obs):
        if self.s2_future is None or not self.s2_future.done():
            return
        future = self.s2_future
        self.s2_future = None
        try:
            result = future.result()
            if not isinstance(result, dict):
                return
            if result.get("session_id"):
                self.s2_session = result["session_id"]
            data = first_object(result.get("text", ""))
            old = (self.planner.posture, self.planner.safety)
            if data.get("posture") in ("upper", "spine", "frontier"):
                self.planner.posture = data["posture"]
            if data.get("safety") in ("certified", "shelter"):
                self.planner.safety = data["safety"]
            new = (self.planner.posture, self.planner.safety)
            if new != old:
                self.policy_version += 1
                self.s1_advice = None
                self.log("astra_posture_changed", tick=tick_of(obs), old=list(old),
                         posture=new[0], safety=new[1])
            else:
                self.log("astra_posture_unchanged", tick=tick_of(obs), parsed=bool(data))
        except Exception as exc:
            self.log("astra_result_failure", error=type(exc).__name__)

    def motor(self, obs):
        now = tick_of(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)

        if self.book.flashing:
            self.pending_hop = None
            return self.noop

        n = node_of(obs.get("qbert_node"))
        motion = qmotion(obs)
        phase = str(motion.get("phase", "unknown")).lower()
        if n is None or phase in ("falling", "not_visible") or bool(motion.get("falling")):
            self.pending_hop = None
            return self.noop

        # A life recovery is a real protocol state, not merely a two-tick
        # planner reset.  Prefer the top if visible; only release the gate after
        # the bounded recovery window when the decoder cannot show it.
        if now < self.recovery_until:
            if n == 0:
                self.recovery_top_seen = True
            return self.noop
        if self.recovery_until > 0 and not self.recovery_top_seen and now < self.recovery_until + 3:
            if n == 0:
                self.recovery_top_seen = True
            return self.noop
        if self.recovery_until > 0 and now >= self.recovery_until + 3:
            self.recovery_until = 0

        # Complete the currently armed hop before doing any planning.  A
        # direction never remains posted through animation.
        if self.pending_hop is not None:
            hop = self.pending_hop
            age = now - hop["start"]
            if bool(obs.get("qbert_in_hop")) or n != hop["origin"]:
                hop["moving"] = True
            if hop["moving"] and standing(obs):
                self.pending_hop = None
                self.planner.note_landing(n, hop["destination"], hop["origin"])
                if n == hop["destination"]:
                    self.log("corridor_landed", tick=now, origin=hop["origin"],
                             destination=n, duration=age)
                    self.neutral_until = max(self.neutral_until, now + 1)
                else:
                    self.log("corridor_mismatch", tick=now, origin=hop["origin"],
                             expected=hop["destination"], observed=n, duration=age)
                    self.neutral_until = max(self.neutral_until, now + 4)
                self.last_unplanned_node = n if n != hop["destination"] else None
                return self.noop
            # If the edge was ignored, one retry is allowed while standing.
            # The preceding neutral gate makes a stale previous direction much
            # less likely to be the action that starts this retry.
            if (not hop["moving"] and standing(obs) and age <= 3 and hop["posts"] < 2):
                hop["posts"] += 1
                return hop["action"]
            if age > 18 and standing(obs):
                self.pending_hop = None
                self.neutral_until = max(self.neutral_until, now + 3)
                self.log("corridor_no_start", tick=now, origin=hop["origin"],
                         expected=hop["destination"])
                return self.noop
            return self.noop

        # This is the sticky-action handshake.  The neutral interval is paid
        # after every landing and is longer after a loss or mismatch.
        if now < self.neutral_until:
            return self.noop
        if not standing(obs):
            return self.noop

        candidates = self.planner.available(obs)
        if not candidates:
            return self.noop
        self.start_jev(obs, candidates)
        chosen = self.advised(obs, candidates) or candidates[0]
        if chosen["action"] == "NOOP":
            self.neutral_until = max(self.neutral_until, now + 1)
            self.log("corridor_wait", tick=now, node=n,
                     reasons=chosen.get("risk_reasons", []))
            return self.noop
        action = chosen.get("action")
        if action not in DIRECTIONS or action not in list_names(obs.get("safe_moves")):
            return self.noop
        destination = GRAPH[n].get(action)
        if destination is None:
            return self.noop
        self.pending_hop = {"origin": n, "destination": destination,
                            "action": action, "start": now, "posts": 1,
                            "moving": False}
        if now - self.last_log >= 25:
            self.log("corridor_status", tick=now, score=finite(obs.get("score")),
                     lives=self.lives, epoch=self.book.epoch,
                     inferred_clears=self.book.clears,
                     unfinished=sum(value > 0 for value in self.book.need),
                     node=n, action=action, target=self.planner.target,
                     policy=[self.planner.posture, self.planner.safety],
                     hazards=self.field.facts(n, now, self.book.clears >= 4))
            self.last_log = now
        return action

    def run(self):
        ctx = self.ctx
        obs = ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.ingest(obs)
        self.start_astra(obs, "initial symbolic corridor plan")
        while not ctx.done and not getattr(self.game, "done", False):
            obs = ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.ingest(obs)
            action = self.motor(obs)
            if self.s2_reason:
                self.start_astra(obs, self.s2_reason)
            elif self.s2_calls < 2 and self.book.clears >= 4 and tick_of(obs) - self.book.last_progress > 180:
                self.start_astra(obs, "late-board corridor survival stall")
            if action not in self.names:
                action = self.noop if self.noop in self.names else "NOOP"
            try:
                ctx.act(action)
                ctx.wait_for_tick()
            except Exception as exc:
                # Deadline, game close, and model-budget closure are normal
                # episode exits.  A valid named action should not otherwise
                # raise; returning is safer than a contract error at shutdown.
                self.log("loop_exit", error=type(exc).__name__)
                return
        self.log("candidate_summary", score=getattr(self.game, "score", None),
                 inferred_clears=self.book.clears, epochs=self.book.epoch,
                 lives=self.lives, posture=self.planner.posture,
                 safety=self.planner.safety)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
