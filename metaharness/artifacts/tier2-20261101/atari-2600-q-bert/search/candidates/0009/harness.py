"""Predictive, transaction-checked Q*bert controller.

This candidate combines the useful mechanisms found in the two parents, but
makes the motor and the hazard interpretation independent of one another.  A
palette epoch is a planning epoch, not a claim about the adaptor's round
counter.  Every hop is a transaction: the controller records the intended
edge, releases the stick, and accepts only the actually observed landing.
Enemy observations are retained as a short time-expanded shadow, because a
missing RGB component is not evidence that a lethal ball has vanished.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
# The order is only a deterministic tie-breaker.  It is not a route macro.
TIE_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")


def make_graph():
    graph = {n: {} for n in range(21)}
    for row in range(6):
        first = row * (row + 1) // 2
        for col in range(row + 1):
            n = first + col
            if row and col < row:
                graph[n]["UP"] = n - row
            if row and col:
                graph[n]["LEFT"] = n - row - 1
            if row < 5:
                child = (row + 1) * (row + 2) // 2
                graph[n]["DOWN"] = child + col
                graph[n]["RIGHT"] = child + col + 1
    return graph


GRAPH = make_graph()
ROWS = {}
for r in range(6):
    for c in range(r + 1):
        ROWS[r * (r + 1) // 2 + c] = r


# All-pairs distances make the route planner small and deterministic.
DIST = []
for start in range(21):
    d = [99] * 21
    d[start] = 0
    queue = deque([start])
    while queue:
        here = queue.popleft()
        for there in GRAPH[here].values():
            if d[there] == 99:
                d[there] = d[here] + 1
                queue.append(there)
    DIST.append(d)


OPENING_RANK = {n: i for i, n in enumerate(
    (1, 2, 4, 3, 5, 8, 12, 7, 6, 10, 15, 16, 11, 17, 18, 13, 9, 14, 19, 20)
)}

# A graph-only frontier sweep.  It deliberately descends through the central
# spine before returning to the upper and outer cubes; this is a reusable board
# heuristic, not a training-seed action sequence.  BFS still chooses the actual
# edge and hazards can interrupt it at every node.
SWEEP_ORDER = (1, 4, 7, 11, 6, 10, 16, 12, 17, 18, 13, 9, 14, 19, 20,
               8, 2, 0, 3, 5, 15)
SWEEP_RANK = {n: i for i, n in enumerate(SWEEP_ORDER)}

# The order is a cyclic graph walk, not an action macro: every requested node
# is reached through a fresh BFS edge and can be skipped when already complete.
# It gives the planner a stable central-spine sweep while retaining target
# hysteresis and hazard detours.
SWEEP_PATH = (1, 4, 7, 11, 6, 10, 16, 11, 7, 12, 17, 12, 18, 13, 9,
              14, 19, 14, 20, 14, 20, 14, 19, 13, 8, 4, 2, 0, 1, 3,
              6, 10, 15, 10, 15, 10, 15, 10, 16, 11, 7, 12, 8)


def integer(value, default=0):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def finite(value, default=0.0):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def valid_node(value):
    n = integer(value, -1)
    return n if 0 <= n < 21 else None


def obs_tick(observation):
    return integer(observation.get("tick", observation.get("ticks", 0)), 0)


def motion(observation):
    value = observation.get("qbert_motion")
    return value if isinstance(value, dict) else {"phase": str(value or "unknown")}


def is_standing(observation):
    n = valid_node(observation.get("qbert_node"))
    m = motion(observation)
    phase = str(m.get("phase", "unknown")).lower()
    return (n is not None and not bool(observation.get("qbert_in_hop"))
            and phase not in ("hopping", "falling", "not_visible")
            and not bool(m.get("falling")))


def action_names(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [x for x in value if isinstance(x, str) and x in DIRECTIONS + ("NOOP",)]


def choice_value(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value if isinstance(value, str) else None


def answer_probability(answer, key):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return 0.0
    return finite(probabilities.get(key), 0.0)


def json_safe(value):
    """Strictly sanitize state sent to Jev or evidence sent to Astra."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def parse_first_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    # Astra is asked for a tiny object, but tolerate a preamble or markdown.
    for index, char in enumerate(text[:16000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def palette(observation):
    raw = observation.get("cube_colors")
    if isinstance(raw, list) and len(raw) == 21:
        values = [integer(x, -1) for x in raw]
        if all(x >= 0 for x in values):
            return tuple(values)
    details = observation.get("cube_details")
    if isinstance(details, list) and len(details) >= 21:
        values = [integer((x or {}).get("raw_color"), -1)
                  if isinstance(x, dict) else -1 for x in details[:21]]
        if all(x >= 0 for x in values):
            return tuple(values)
    return None


def observed_needed(observation):
    raw = observation.get("needed_landings")
    if not isinstance(raw, list) or len(raw) != 21:
        return None
    result = []
    for value in raw:
        x = integer(value, -1)
        result.append(x if x >= 0 else None)
    return result


class PaletteEpochs:
    """Infer useful coverage from transitions, while quarantining the flash.

    The adaptor's named phases are valuable but can be sampled during the
    cartridge's whole-board tally.  A locally learned destination colour is
    therefore the primary target; needed_landings is used as a bootstrap and
    a consistency check rather than as an authority during a flash.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.last_score = None
        self.goal = None
        self.need = [1] * 21
        self.flashing = False
        self.flash_origin = None
        self.flash_start_score = 0.0
        self.last_flash_change = -999
        self.barrier_until = 0
        self.epoch = 0
        self.inferred_clears = 0
        self.complete_since = None
        self.last_progress = 0
        self.evidence = deque(maxlen=10)

    def _fresh_observation_need(self, observation):
        raw = observed_needed(observation)
        # An all-zero vector is exactly the stale post-clear value this class
        # is designed not to trust.  Start a new planning epoch with targets.
        if raw is None or not any(x is not None and x > 0 for x in raw):
            return [1] * 21
        return [x if x is not None else 1 for x in raw]

    def _begin_flash(self, observation, now, score, delta):
        self.flashing = True
        self.flash_origin = valid_node(observation.get("qbert_node"))
        self.flash_start_score = score - delta
        self.last_flash_change = now
        self.goal = None
        self.complete_since = None
        self.ctx.log("epoch_flash_begin", tick=now, score=score,
                     origin=self.flash_origin,
                     reported_rounds=observation.get("completed_rounds"))

    def _finish_flash(self, observation, now, score):
        tally = score - self.flash_start_score
        if tally >= 1000:
            self.inferred_clears += 1
        self.flashing = False
        self.epoch += 1
        self.goal = None
        self.need = self._fresh_observation_need(observation)
        self.complete_since = None
        self.last_progress = now
        # A short neutral barrier separates the global tally from the first
        # hop of the new board.  It is deliberately shorter than 0008's fixed
        # 55-tick wait, but longer than one sampled stale palette frame.
        self.barrier_until = now + 8
        self.evidence.clear()
        self.ctx.log("epoch_rebased", tick=now, epoch=self.epoch,
                     inferred_clears=self.inferred_clears,
                     tally_points=tally, next_need=sum(self.need),
                     reported_rounds=observation.get("completed_rounds"))

    def update(self, observation):
        now = obs_tick(observation)
        score = finite(observation.get("score"), self.last_score or 0.0)
        delta = score - self.last_score if self.last_score is not None else 0.0
        current = palette(observation)
        if current is None:
            self.last_score = score
            return False
        changed = ([i for i in range(21)
                    if self.previous is not None and current[i] != self.previous[i]])

        if self.previous is not None and len(changed) >= 16:
            if not self.flashing:
                self._begin_flash(observation, now, score, delta)
            self.last_flash_change = now

        if self.flashing:
            quiet = now - self.last_flash_change
            at_top = valid_node(observation.get("qbert_node")) == 0
            # Rebase only after stable palette frames, or after a bounded
            # timeout.  This prevents all-S/all-D flash frames from creating a
            # one-cube target list.
            if quiet >= 4 and (at_top or quiet >= 10):
                self._finish_flash(observation, now, score)
                self.previous = current
                self.last_score = score
                return True
            self.need = [1] * 21
            self.previous = current
            self.last_score = score
            return False

        # A scored local colour change is strong evidence of the current
        # destination.  A 325-point event can include a safe bonus and still
        # has the cube's 25-point component.
        if delta > 0 and int(round(delta)) % 100 == 25 and 0 < len(changed) <= 3:
            qnode = valid_node(observation.get("qbert_node"))
            landed = qnode if qnode in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                new_goal = current[landed]
                if self.goal != new_goal:
                    self.goal = new_goal
                    self.ctx.log("destination_learned", tick=now,
                                 epoch=self.epoch, node=landed,
                                 color=new_goal, reward=delta)
                self.evidence.append({
                    "tick": now, "node": landed, "reward": delta,
                    "color": new_goal,
                    "decoded_phase": (observation.get("cube_phase") or [None] * 21)[landed],
                })
                self.last_progress = now

        if self.goal is None:
            boot = observed_needed(observation)
            if boot is not None and any(x is not None and x > 0 for x in boot):
                self.need = [x if x is not None else 1 for x in boot]
            else:
                self.need = [1] * 21
        else:
            self.need = [0 if color == self.goal else 1 for color in current]

        if not any(self.need):
            if self.complete_since is None:
                self.complete_since = now
        else:
            self.complete_since = None

        # Some later finite cycles can award an intermediate transition before
        # the global flash.  If the flash never arrives, probe a new phase after
        # a bounded quiet period instead of deadlocking forever.  This does not
        # call it a new round and does not erase a visible reversal.
        if (self.complete_since is not None and now - self.complete_since >= 24
                and not self.flashing):
            self.goal = None
            self.need = self._fresh_observation_need(observation)
            self.complete_since = None
            self.barrier_until = max(self.barrier_until, now + 2)
            self.ctx.log("epoch_probe", tick=now, epoch=self.epoch,
                         reason="complete_palette_without_flash")

        self.previous = current
        self.last_score = score
        return False

    def blocked(self, now):
        if self.flashing or now < self.barrier_until:
            return True
        # Wait briefly at a genuinely complete board for its flash to start.
        return self.complete_since is not None and now - self.complete_since < 24


class HazardRadar:
    """Keep a conservative, time-expanded shadow of lethal characters."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.records = {}
        self.last_tick = -1
        self.unknown_ball = False
        self.unknown_since = -1
        self.posture = "long"
        # The ordinary early-board controller trusts the adaptor's immediate
        # safe_moves and only visible occupancy.  The time-expanded shadow is
        # enabled once several palette epochs have demonstrated the late-round
        # failure regime; this prevents a conservative radar from slowing the
        # reliable opening sweep.
        self.late = False

    @staticmethod
    def _kind(item):
        return str(item.get("kind", "hazard")) if isinstance(item, dict) else "hazard"

    def update(self, observation):
        now = obs_tick(observation)
        raw = observation.get("hazards")
        seen = set()
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict) or not bool(item.get("lethal")):
                    continue
                kind = self._kind(item)
                n = valid_node(item.get("estimated_node", item.get("node")))
                if n is None:
                    continue
                old = self.records.get(kind)
                self.records[kind] = {
                    "kind": kind, "node": n, "last": now,
                    "previous": old.get("node") if old else n,
                    "risk": str(item.get("risk", item.get("risk_label", "unknown"))),
                    "visible": True,
                }
                seen.add(kind)

        # RGB/palette cross-checking intentionally suppresses stale sprite
        # placeholders.  Retain a shadow after suppression, but let it expire
        # rather than making a permanent obstacle.
        for kind in list(self.records):
            rec = self.records[kind]
            age = now - rec["last"]
            limit = 18 if kind in ("purple_ball", "red_ball") else 12
            if age > limit:
                del self.records[kind]
            else:
                rec["visible"] = kind in seen

        timers = observation.get("timers")
        state = integer((timers or {}).get("purple_ball_state"), 0) if isinstance(timers, dict) else 0
        has_ball = any(k in self.records for k in ("purple_ball", "red_ball"))
        # The observed late failures had purple_ball_state=36 while the
        # rendered object was absent.  Treat high nonzero states as uncertainty,
        # not as a fabricated node.  Initial bootstrap values are small.
        self.unknown_ball = (not has_ball and state >= 20)
        if self.unknown_ball and self.late:
            if self.unknown_since < 0:
                self.unknown_since = now
        else:
            self.unknown_since = -1
        self.last_tick = now

    def _descendants(self, node, depth):
        result = {node}
        frontier = {node}
        for _ in range(max(0, depth)):
            nxt = set()
            for n in frontier:
                for action, child in GRAPH[n].items():
                    if action in ("DOWN", "RIGHT"):
                        nxt.add(child)
            result.update(nxt)
            frontier = nxt
        return result

    def shadow_nodes(self, rec, now):
        age = max(0, now - rec["last"])
        n = rec["node"]
        kind = rec["kind"]
        if kind in ("purple_ball", "red_ball"):
            # A ball can be between decoded positions.  At a future landing
            # horizon, include one or two downward branches, not every cube.
            depth = min(3, age // 7)
            return self._descendants(n, depth)
        radius = min(2, age // 8 + 1)
        return {x for x in range(21) if DIST[n][x] <= radius}

    def risk(self, destination, origin, now, action=None):
        value = 0.0
        for rec in self.records.values():
            age = max(0, now - rec["last"])
            possible = self.shadow_nodes(rec, now) if self.late else {rec["node"]}
            if destination in possible:
                value += 8.0 if age <= 3 else 4.0
            if any(DIST[destination][x] == 1 for x in possible):
                # Contact can occur during the animation before the landing
                # node is occupied.  For a rolling lethal ball, an adjacent
                # branch is therefore materially worse than a merely distant
                # Coily shadow.
                value += 3.6 if rec["kind"] in ("purple_ball", "red_ball") else 1.4
            # A rolling ball can be visible one node above the next landing.
            # Occupancy-only blocking misses exactly the late branch collision
            # seen in the 0007 trace, so downward children get their own strong
            # but finite cost.
            if rec["kind"] in ("purple_ball", "red_ball"):
                next_branches = set()
                for x in possible:
                    next_branches.update(child for edge, child in GRAPH[x].items()
                                         if edge in ("DOWN", "RIGHT"))
                if destination in next_branches and destination not in possible:
                    value += 4.5
                if self.late:
                    # Also mark a second descending layer softly.  A Q*bert
                    # hop takes several ticks, so node 2 can make node 7 a
                    # bad next-next landing in the late-round regime.
                    future_branches = set()
                    for x in next_branches:
                        future_branches.update(child for edge, child in GRAPH[x].items()
                                               if edge in ("DOWN", "RIGHT"))
                    if destination in future_branches and destination not in next_branches:
                        value += 2.0
            if rec["kind"] == "coily" and origin is not None and DIST[origin][rec["node"]] <= 2:
                value += 1.0
        if self.unknown_ball and self.late:
            # Keep moving, but avoid blindly climbing into the ball's likely
            # upper branch.  This is a soft cost so a corner can still escape.
            value += 1.0
            if action in ("UP", "LEFT"):
                value += 1.3
        return value

    def current_risk(self, origin, now):
        if origin is None:
            return 99.0
        return self.risk(origin, origin, now)

    def facts(self, origin, now):
        result = []
        for rec in self.records.values():
            age = max(0, now - rec["last"])
            possible = self.shadow_nodes(rec, now) if self.late else {rec["node"]}
            nearest = min((DIST[origin][x] for x in possible), default=99) if origin is not None else 99
            result.append({
                "kind": rec["kind"],
                "visibility": "visible" if rec["visible"] else "recent_shadow",
                "proximity": "same" if nearest == 0 else "adjacent" if nearest == 1 else "near" if nearest <= 2 else "far",
                "motion": "descending_shadow" if rec["kind"] in ("purple_ball", "red_ball") else "pursuit_shadow",
            })
        if self.unknown_ball and self.late:
            result.append({"kind": "unseen_purple_state", "visibility": "occluded",
                           "proximity": "uncertain", "motion": "active_unknown"})
        return result

    def signature(self, now):
        # Age is used to recompute risk at execution, not as an identity tag:
        # otherwise a correct answer becomes stale merely because one tick
        # elapsed while the same enemy remained at the same node.
        return tuple(sorted((r["kind"], r["node"], bool(r["visible"]))
                            for r in self.records.values())) + (("unknown",) if self.unknown_ball else ())


class RoutePlanner:
    """Persistent target selection with escape-space and reversal penalties."""

    def __init__(self, progress, radar):
        self.progress = progress
        self.radar = radar
        self.goal = None
        self.waypoint = None
        self.previous_node = None
        self.recent_nodes = deque(maxlen=18)
        self.bad_edges = Counter()
        self.cursor = 0
        self.policy = "nearest"

    def reset_epoch(self):
        self.goal = None
        self.waypoint = None
        self.previous_node = None
        self.recent_nodes.clear()
        self.bad_edges.clear()
        self.cursor = 0

    def observe_landing(self, node, expected=None):
        if node is None:
            return
        if expected is not None and node != expected:
            self.bad_edges[(self.previous_node, expected)] += 1
        if self.previous_node != node:
            self.recent_nodes.append(node)
        self.previous_node = node
        if self.goal == node and self.progress.need[node] <= 0:
            self.goal = None
            self.waypoint = None

    def _unfinished(self):
        return [n for n, value in enumerate(self.progress.need) if value > 0]

    def _escape_count(self, node, now, origin):
        count = 0
        for nxt in GRAPH[node].values():
            if self.radar.risk(nxt, node, now) < 4.0 and nxt != origin:
                count += 1
        return count

    def _choose_waypoint(self, current, safe, now):
        targets = self._unfinished()
        ranked = []
        for action in action_names(safe):
            dest = GRAPH[current].get(action)
            if dest is None:
                continue
            risk = self.radar.risk(dest, current, now, action)
            escape = self._escape_count(dest, now, current)
            distance = min((DIST[dest][target] for target in targets), default=0)
            ranked.append((risk + (0 if escape >= 2 else 2), -escape,
                           distance, OPENING_RANK.get(dest, 99), dest))
        if not ranked:
            return None
        ranked.sort()
        return ranked[0][-1]

    def target(self, observation, now):
        current = valid_node(observation.get("qbert_node"))
        if current is None:
            return None
        unfinished = set(self._unfinished())
        if self.goal is not None and self.goal not in unfinished:
            self.goal = None
            self.waypoint = None

        if self.goal is not None and self.goal != current:
            return self.goal

        if self.goal == current:
            if self.progress.need[current] <= 0:
                self.goal = None
                self.waypoint = None
            else:
                if self.waypoint is None or self.waypoint == current:
                    self.waypoint = self._choose_waypoint(
                        current, observation.get("safe_moves", []), now)
                return current

        candidates = [n for n in unfinished if n != current]
        if not candidates:
            if current in unfinished:
                self.goal = current
                self.waypoint = self._choose_waypoint(
                    current, observation.get("safe_moves", []), now)
                return current
            return None

        # Advance through a reusable graph walk, skipping cubes already at the
        # learned destination.  Unlike a blind macro, the walk is only a target
        # preference: BFS, current hazards, and safe_moves choose each edge.
        for offset in range(len(SWEEP_PATH)):
            index = (self.cursor + offset) % len(SWEEP_PATH)
            candidate = SWEEP_PATH[index]
            if candidate in unfinished and candidate != current:
                self.cursor = index
                self.goal = candidate
                self.waypoint = None
                return self.goal
        self.goal = candidates[0]
        self.waypoint = None
        return self.goal

    def options(self, observation, now):
        current = valid_node(observation.get("qbert_node"))
        if current is None:
            return []
        safe = set(action_names(observation.get("safe_moves")))
        safe = [a for a in TIE_ORDER if a in safe and a in GRAPH[current]]
        if not safe:
            return []
        target = self.target(observation, now)
        if target is None:
            return []
        target = self.target(observation, now)
        if target is None:
            return []
        # Hysteresis is retained for a repeated landing on the current cube,
        # but ordinary coverage ranks the next edge against the full unfinished
        # frontier.  This lets later palette epochs choose a different return
        # path instead of replaying one fixed sweep.
        effective = self.waypoint if self.waypoint is not None and self.waypoint != current else None
        if self.waypoint == current:
            self.waypoint = None
        frontier = self._unfinished()
        if not frontier and effective is None:
            return []
        visits = Counter(self.recent_nodes)
        result = []
        for action in safe:
            dest = GRAPH[current][action]
            risk = self.radar.risk(dest, current, now, action)
            escape = self._escape_count(dest, now, current)
            useful = self.progress.need[dest] > 0
            distance = (DIST[dest][effective] if effective is not None
                        else min((DIST[dest][target] for target in frontier), default=0))
            reverse = dest == self.previous_node
            bad = self.bad_edges[(current, dest)]
            cost = 2.0 * distance + (0.0 if useful else 0.55)
            cost += 12.0 * min(risk, 10.0)
            cost += 4.0 if escape == 0 else 1.4 if escape == 1 else 0.0
            cost += 2.7 if reverse and len(safe) > 1 else 0.0
            cost += min(3.0, 0.8 * bad)
            cost += 0.15 * visits[dest]
            if self.policy == "frontier" and useful:
                cost -= 0.15 * ROWS[dest]
            elif self.policy == "nearest" and not useful:
                cost += 0.10 * ROWS[dest]
            # If the radar sees an active-but-occluded ball, climbing is the
            # least informed choice; keep it available only as an escape.
            if self.radar.unknown_ball and self.radar.late and action in ("UP", "LEFT"):
                cost += 1.1
            result.append({
                "action": action, "destination": dest, "risk": risk,
                "escape": escape, "useful": useful, "distance": distance,
                "reverse": reverse, "cost": cost,
            })
        result.sort(key=lambda item: (item["risk"] >= 8.0,
                                      item["cost"],
                                      OPENING_RANK.get(item["destination"], 99),
                                      TIE_ORDER.index(item["action"])))
        return result


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.progress = PaletteEpochs(ctx)
        self.radar = HazardRadar(ctx)
        self.planner = RoutePlanner(self.progress, self.radar)
        self.names = {"NOOP", "UP", "RIGHT", "LEFT", "DOWN"}
        try:
            self.names = {str(x) for x in game.action_spec.names}
        except Exception:
            pass
        self.noop = "NOOP" if "NOOP" in self.names else next(iter(self.names), "NOOP")

        self.lives = None
        self.life_epoch = 0
        self.pulse = None
        self.neutral_gate = 0
        self.wait_since = None
        self.last_posted = self.noop
        self.policy_version = 0
        self.advice = None
        self.s1 = None
        self.s1meta = None
        self.last_s1_tick = -999
        self.s2 = None
        self.s2session = None
        self.s2calls = 0
        self.s2reason = "initial interface audit"
        self.last_s2_tick = -999
        self.last_epoch = 0
        self.last_logged = -999

    def ingest(self, observation):
        now = obs_tick(observation)
        old_epoch = self.progress.epoch
        self.progress.update(observation)
        self.radar.late = self.progress.epoch >= 4
        self.radar.update(observation)
        if self.progress.epoch != old_epoch:
            self.planner.reset_epoch()
            self.pulse = None
            self.advice = None
            self.neutral_gate = max(self.neutral_gate, 2)
            self.s2reason = "new palette epoch; audit later-board posture"
            self.ctx.log("controller_epoch", tick=now,
                         epoch=self.progress.epoch,
                         inferred_clears=self.progress.inferred_clears)

        lives = integer(observation.get("lives"), self.lives if self.lives is not None else 4)
        if self.lives is not None and lives < self.lives:
            self.life_epoch += 1
            self.pulse = None
            self.advice = None
            self.planner.goal = None
            self.planner.waypoint = None
            self.planner.recent_nodes.clear()
            self.radar.records.clear()
            self.neutral_gate = max(self.neutral_gate, 5)
            self.wait_since = None
            self.s2reason = "life loss; increase hazard review"
            self.ctx.log("life_recovery", tick=now, lives=lives,
                         epoch=self.progress.epoch)
        self.lives = lives

    def _current_options(self, observation):
        return self.planner.options(observation, obs_tick(observation))

    def _compact_state(self, observation, options):
        now = obs_tick(observation)
        origin = valid_node(observation.get("qbert_node"))
        proposed = options[0]["action"] if options else self.noop
        candidates = []
        for item in options:
            candidates.append({
                "action": item["action"],
                "landing": "unfinished" if item["useful"] else "transit",
                "danger": "blocked" if item["risk"] >= 8 else
                           "shadow_near" if item["risk"] >= 2 else "clear",
                "escape_space": "none" if item["escape"] == 0 else
                                "narrow" if item["escape"] == 1 else "multiple",
                "route": "short" if item["distance"] <= 1 else "longer",
                "reverse": bool(item["reverse"]),
                "code_preferred": item is options[0],
            })
        return json_safe({
            "game": "Atari 2600 Q*bert Atari port; 21-node six-row graph",
            "rules": "Survive first; code has already checked legal/safe_moves. Never FIRE or invent an edge.",
            "position": {"node": origin, "standing": is_standing(observation),
                         "safe_moves": action_names(observation.get("safe_moves")),
                         "last_posted": self.last_posted},
            "progress": {"epoch": self.progress.epoch,
                          "inferred_clears": self.progress.inferred_clears,
                          "unfinished_count": sum(x > 0 for x in self.progress.need),
                          "goal": self.planner.goal,
                          "waypoint": self.planner.waypoint},
            "candidates": candidates,
            "hazards": self.radar.facts(origin, now),
            "proposed_action": proposed,
            "observation_event": str(observation.get("last_event", "none")),
        })

    def start_jev(self, observation, options):
        if self.s1 is not None or not options or self.ctx.done:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        now = obs_tick(observation)
        # A request is pipelined during the previous hop.  It must not be the
        # reason the motor waits at a standing node.
        if now - self.last_s1_tick < 7 or not is_standing(observation):
            return
        if self.ctx.budget.remaining().get("system1_calls", 1) <= 2:
            return
        actions = [x["action"] for x in options]
        actions.append(self.noop)
        actions = list(dict.fromkeys(actions))
        proposed = options[0]["action"]
        first = [proposed] + [x for x in actions if x != proposed]
        second = list(reversed(first))
        criteria_a = {
            x: {"what": "Take this supplied on-pyramid edge only if its code facts are safe.",
                "not_for": "FIRE, an absent safe_moves edge, or a speculative bonus."}
            for x in first
        }
        criteria_b = {
            x: {"what": "Choose this immediate action when it preserves survival and an escape.",
                "not_for": "A stale or invented edge."}
            for x in second
        }
        state = self._compact_state(observation, options)
        instruction = (
            "Choose one candidate for the next hop. Code-computed safe_moves and the candidate danger "
            "buckets are authoritative. Prefer clear over shadow_near, multiple escape space over narrow, "
            "and useful coverage when safety is tied. NOOP is a valid defer choice. Never invent an action."
        )
        questions = {
            "route_a": Choice(instruction, criteria_a),
            "route_b": Choice(instruction, criteria_b),
            "danger": Noul(
                "Is the proposed_action supported by an immediate lethal collision or edge-fall risk? "
                "A recent_shadow or occluded state is a reason for caution only when it names a concrete "
                "candidate threat; distant visibility alone is not immediate danger.",
                {"true": "The proposed landing is occupied or a concrete near collision is stated.",
                 "false": "The proposed landing is clear, or the threat is distant/uncertain."}),
        }
        meta = {
            "tick": now, "node": valid_node(observation.get("qbert_node")),
            "epoch": self.progress.epoch, "life": self.life_epoch,
            "policy": self.policy_version, "signature": self.radar.signature(now),
            "actions": actions, "proposed": proposed,
        }
        self.last_s1_tick = now

        def work():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.85),
                        "error": None}
            except Exception as exc:
                return {"answers": None, "error": type(exc).__name__}

        try:
            self.s1 = self.ctx.spawn(work)
            self.s1meta = meta
        except Exception as exc:
            self.s1 = None
            self.s1meta = None
            self.ctx.log("jev_spawn_failure", error=type(exc).__name__)

    def collect_jev(self, observation):
        if self.s1 is None or not self.s1.done():
            return
        future, meta = self.s1, self.s1meta
        self.s1 = None
        self.s1meta = None
        try:
            result = future.result()
            answers = result.get("answers") if isinstance(result, dict) else None
            if not isinstance(answers, dict) or not isinstance(meta, dict):
                return
            now = obs_tick(observation)
            if (now - integer(meta.get("tick"), now) < 0
                    or now - integer(meta.get("tick"), now) > 7
                    or not is_standing(observation)
                    or valid_node(observation.get("qbert_node")) != meta.get("node")
                    or self.progress.epoch != meta.get("epoch")
                    or self.life_epoch != meta.get("life")
                    or self.policy_version != meta.get("policy")
                    or self.radar.signature(now) != meta.get("signature")):
                return
            first = answers.get("route_a")
            second = answers.get("route_b")
            a = choice_value(first)
            b = choice_value(second)
            accepted = a == b and a in meta.get("actions", [])
            p = min(answer_probability(first, a), answer_probability(second, b)) if accepted else 0.0
            danger = finite(getattr(answers.get("danger"), "value", 0.0), 0.0)
            self.advice = {"action": a if accepted and p >= 0.58 else None,
                           "p": p, "danger": danger, **meta, "ready_tick": now}
        except Exception as exc:
            self.ctx.log("jev_result_failure", error=type(exc).__name__)

    def _advised(self, observation, options):
        fallback = options[0] if options else None
        advice = self.advice
        self.advice = None
        if fallback is None or not advice:
            return fallback
        now = obs_tick(observation)
        if (now - advice.get("tick", now) < 0 or now - advice.get("tick", now) > 3
                or valid_node(observation.get("qbert_node")) != advice.get("node")
                or self.progress.epoch != advice.get("epoch")
                or self.life_epoch != advice.get("life")
                or self.policy_version != advice.get("policy")
                or self.radar.signature(now) != advice.get("signature")):
            return fallback
        by_action = {item["action"]: item for item in options}
        candidate = by_action.get(advice.get("action"))
        if advice.get("danger", 0.0) >= 0.80:
            # A veto is useful only if code has an equally safe alternative.
            if fallback["action"] == advice.get("proposed"):
                for other in options[1:]:
                    if other["risk"] <= fallback["risk"] + 0.4:
                        self.ctx.log("jev_veto", tick=now,
                                     old=fallback["action"], new=other["action"])
                        return other
            return fallback
        if (candidate is not None and advice.get("p", 0.0) >= 0.58
                and candidate["risk"] <= fallback["risk"] + 0.65
                and candidate["escape"] + 1 >= fallback["escape"]):
            if candidate["action"] != fallback["action"]:
                self.ctx.log("jev_route_override", tick=now,
                             old=fallback["action"], new=candidate["action"],
                             probability=advice.get("p"))
            return candidate
        return fallback

    def start_astra(self, observation):
        if not self.s2reason or self.s2 is not None or self.s2calls >= 3 or self.ctx.done:
            return
        if self.ctx.budget.remaining().get("system2_calls", 1) <= 1:
            return
        now = obs_tick(observation)
        if self.s2calls and now - self.last_s2_tick < 360:
            return
        reason = self.s2reason
        self.s2reason = ""
        self.s2calls += 1
        self.last_s2_tick = now
        evidence = {
            "reason": reason, "tick": now, "score": finite(observation.get("score")),
            "lives": self.lives, "epoch": self.progress.epoch,
            "inferred_clears": self.progress.inferred_clears,
            "unfinished": sum(x > 0 for x in self.progress.need),
            "hazards": self.radar.facts(valid_node(observation.get("qbert_node")), now),
            "policy": {"route": self.planner.policy, "radar": self.radar.posture},
            "recent_evidence": list(self.progress.evidence)[-4:],
        }
        prompt = (
            "You are the slow strategic adviser for a real-time Atari 2600 Q*bert controller. "
            "The controller owns the verified 21-node graph, palette-epoch bookkeeping, legal edges, "
            "time-expanded lethal-hazard shadow, and a transaction motor that accepts only observed "
            "landings. A whole-board score flash is quarantined before a new epoch. Jev receives only "
            "code-generated safe candidates and two reordered typed Choices; it cannot press FIRE or "
            "invent an edge. Choose a durable posture, not a joystick sequence. Return ONLY a JSON object "
            "under 40 tokens with route one of nearest/frontier and radar one of long/balanced. "
            "Use long after life loss, invisible hazard evidence, or a late-round survival problem; use "
            "balanced only when the long shadow demonstrably stalls safe coverage. Evidence: "
            + json.dumps(json_safe(evidence), separators=(",", ":"))
        )
        session = self.s2session

        def work():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=105.0)
            except Exception as exc:
                return {"error": type(exc).__name__}

        try:
            self.s2 = self.ctx.spawn(work)
            self.ctx.log("astra_requested", tick=now, reason=reason)
        except Exception as exc:
            self.s2 = None
            self.ctx.log("astra_spawn_failure", error=type(exc).__name__)

    def collect_astra(self, observation):
        if self.s2 is None or not self.s2.done():
            return
        future = self.s2
        self.s2 = None
        try:
            result = future.result()
            text = getattr(result, "text", "") if result is not None else ""
            if isinstance(result, dict):
                text = ""
            data = parse_first_object(text)
            old = (self.planner.policy, self.radar.posture)
            if data.get("route") in ("nearest", "frontier"):
                self.planner.policy = data["route"]
            if data.get("radar") in ("long", "balanced"):
                self.radar.posture = data["radar"]
            if old != (self.planner.policy, self.radar.posture):
                self.policy_version += 1
                self.advice = None
                self.ctx.log("astra_policy_changed", tick=obs_tick(observation),
                             old=list(old), route=self.planner.policy,
                             radar=self.radar.posture)
            else:
                self.ctx.log("astra_policy_unchanged", tick=obs_tick(observation),
                             parsed=bool(data))
            self.s2session = getattr(result, "session_id", None)
        except Exception as exc:
            self.ctx.log("astra_result_failure", error=type(exc).__name__)

    def choose(self, observation):
        now = obs_tick(observation)
        self.collect_jev(observation)
        self.collect_astra(observation)

        if self.progress.blocked(now):
            self.pulse = None
            self.wait_since = None
            return self.noop

        m = motion(observation)
        phase = str(m.get("phase", "unknown")).lower()
        if (valid_node(observation.get("qbert_node")) is None
                or bool(observation.get("qbert_in_hop"))
                or phase in ("hopping", "falling", "not_visible")
                or bool(m.get("falling"))):
            self.pulse = None if phase in ("falling", "not_visible") else self.pulse
            return self.noop

        # Finish or verify an existing hop before asking for another edge.
        current = valid_node(observation.get("qbert_node"))
        if self.pulse is not None:
            p = self.pulse
            age = now - p["start"]
            if current != p["source"]:
                if current == p["expected"]:
                    self.planner.observe_landing(current, p["expected"])
                    self.ctx.log("verified_landing", tick=now,
                                 source=p["source"], destination=current,
                                 duration=age)
                else:
                    self.planner.observe_landing(current, p["expected"])
                    self.ctx.log("unexpected_landing", tick=now,
                                 source=p["source"], expected=p["expected"],
                                 observed=current, duration=age)
                self.pulse = None
                self.neutral_gate = 0
                self.wait_since = None
                return self.noop
            # At the source and still standing: a direction is posted at most
            # twice.  A neutral release separates retries and defeats a stale
            # sticky direction better than a long held macro.
            if age <= 2 and p["direction_posts"] < 2:
                p["direction_posts"] += 1
                return p["action"]
            self.pulse = None
            self.neutral_gate = max(self.neutral_gate, 1)
            return self.noop

        if self.neutral_gate:
            self.neutral_gate -= 1
            return self.noop

        options = self._current_options(observation)
        if not options:
            return self.noop

        # If every candidate is in the radar shadow and the present cube is
        # safer, wait briefly for the next rendered enemy state.  A hard bound
        # avoids the old idle/stall failure mode.
        best = options[0]
        current_risk = self.radar.current_risk(current, now)
        visible_ball = any(rec.get("kind") in ("purple_ball", "red_ball")
                           and rec.get("visible") for rec in self.radar.records.values())
        if (visible_ball and best["risk"] >= 3.0
                and current_risk + 0.5 < best["risk"]
                and self.wait_since is None):
            self.wait_since = now
            self.ctx.log("radar_wait", tick=now, node=current,
                         best_risk=best["risk"], current_risk=current_risk)
        if self.wait_since is not None:
            age = now - self.wait_since
            if age < 12 and best["risk"] >= 2.0:
                return self.noop
            self.wait_since = None
            options = self._current_options(observation)
            if not options:
                return self.noop
            best = options[0]

        self.start_jev(observation, options)
        selected = self._advised(observation, options)
        if selected is None or selected["action"] not in action_names(observation.get("safe_moves")):
            selected = best
        if selected["risk"] >= 8.0 and len(options) > 1:
            # Never let either model authorize a code-classified occupied/shadow
            # landing when another safe edge exists.
            selected = next((x for x in options if x["risk"] < 8.0), best)

        action = selected["action"]
        destination = GRAPH[current].get(action)
        if destination is None:
            return self.noop
        self.pulse = {"source": current, "expected": destination,
                      "action": action, "start": now,
                      "direction_posts": 1}
        if now - self.last_logged >= 25:
            self.ctx.log("coverage_status", tick=now,
                         score=finite(observation.get("score")), lives=self.lives,
                         epoch=self.progress.epoch,
                         inferred_clears=self.progress.inferred_clears,
                         unfinished=sum(x > 0 for x in self.progress.need),
                         node=current, goal=self.planner.goal,
                         action=action,
                         radar=self.radar.facts(current, now))
            self.last_logged = now
        return action

    def run(self):
        while not self.ctx.done and not getattr(self.game, "done", False):
            observation = self.ctx.observe()
            if not isinstance(observation, dict):
                observation = {}
            self.ingest(observation)
            self.start_astra(observation)
            action = self.choose(observation)
            if action not in self.names:
                action = self.noop if self.noop in self.names else "NOOP"
            self.ctx.act(action)
            self.last_posted = action
            self.ctx.wait_for_tick()
        self.ctx.log("candidate_summary", score=getattr(self.game, "score", None),
                     inferred_clears=self.progress.inferred_clears,
                     epoch=self.progress.epoch, lives=self.lives,
                     route=self.planner.policy, radar=self.radar.posture)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
