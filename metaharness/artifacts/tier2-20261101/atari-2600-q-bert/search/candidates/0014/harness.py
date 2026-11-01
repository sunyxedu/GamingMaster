"""Candidate 0014: temporal shelter control for Atari 2600 Q*bert.

The controller deliberately separates three jobs.  The board model owns palette
and graph bookkeeping, a short-horizon planner owns the next legal edge, and
Jev is a timing/hazard referee rather than an action generator.  Astra may
change only durable route and quarantine postures.  Every posted action is
still produced by the deterministic motor and is checked against the current
observation immediately before posting.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
ACTIONS = DIRECTIONS + ("NOOP",)
TIE_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")
# This is the six-row Atari board.  It is intentionally not the arcade map.
GRAPH = {n: {} for n in range(21)}
ROWS = {}
for row in range(6):
    base = row * (row + 1) // 2
    for col in range(row + 1):
        node = base + col
        ROWS[node] = row
        if row and col < row:
            GRAPH[node]["UP"] = node - row
        if row and col:
            GRAPH[node]["LEFT"] = node - row - 1
        if row < 5:
            child = (row + 1) * (row + 2) // 2
            GRAPH[node]["DOWN"] = child + col
            GRAPH[node]["RIGHT"] = child + col + 1

# All-pairs graph distance.  The planner uses this for candidate evaluation,
# never as permission to bypass the adaptor's current safe_moves.
DIST = []
for start in range(21):
    d = [99] * 21
    d[start] = 0
    q = deque([start])
    while q:
        here = q.popleft()
        for there in GRAPH[here].values():
            if d[there] == 99:
                d[there] = d[here] + 1
                q.append(there)
    DIST.append(d)

# A reusable coverage preference, not a sequence of joystick actions.  A fresh
# BFS edge is selected at every standing observation.
SWEEP = (1, 4, 7, 11, 6, 10, 16, 12, 17, 18, 13, 9, 14, 19, 20,
         8, 5, 3, 2, 0, 15)
SWEEP_RANK = {node: i for i, node in enumerate(SWEEP)}
CENTRAL = (4, 7, 8, 11, 12, 13, 6, 10, 16, 17, 3, 5, 9, 14)
CENTRAL_RANK = {node: i for i, node in enumerate(CENTRAL)}


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
        return default


def node_of(value):
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if 0 <= n < 21 else None


def now_of(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)), 0)


def motion_of(obs):
    value = obs.get("qbert_motion")
    if isinstance(value, dict):
        return value
    return {"phase": str(value or "unknown")}


def standing(obs):
    n = node_of(obs.get("qbert_node"))
    m = motion_of(obs)
    phase = str(m.get("phase", "unknown")).lower()
    return (n is not None and not bool(obs.get("qbert_in_hop"))
            and phase not in ("hopping", "falling", "not_visible")
            and not bool(m.get("falling")))


def legal_action_list(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [x for x in value if isinstance(x, str) and x in ACTIONS]


def colors_of(obs):
    raw = obs.get("cube_colors")
    if isinstance(raw, list) and len(raw) == 21:
        values = [integer(x, -1) for x in raw]
        if all(x >= 0 for x in values):
            return tuple(values)
    details = obs.get("cube_details")
    if isinstance(details, list) and len(details) >= 21:
        values = []
        for item in details[:21]:
            values.append(integer(item.get("raw_color"), -1)
                          if isinstance(item, dict) else -1)
        if all(x >= 0 for x in values):
            return tuple(values)
    return None


def needed_of(obs):
    raw = obs.get("needed_landings")
    if not isinstance(raw, list) or len(raw) != 21:
        return None
    result = [integer(x, -1) for x in raw]
    return result if all(x >= 0 for x in result) else None


def answer_value(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value if isinstance(value, str) else None


def answer_probability(answer, option):
    probabilities = getattr(answer, "probabilities", None) if answer is not None else None
    if not isinstance(probabilities, dict):
        return 0.0
    return finite(probabilities.get(option), 0.0)


def json_value(value):
    """Make a strict-JSON state even when an adaptor field is surprising."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    return str(value)


def first_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:12000]):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return {}


class PaletteBoard:
    """Observation-calibrated progress, separate from the adaptor's labels."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.last_score = None
        self.goal = None
        self.need = [1] * 21
        self.flashing = False
        self.flash_last_change = -999
        self.flash_start_score = 0.0
        self.flash_origin = None
        self.complete_since = None
        self.epoch = 0
        self.clears = 0
        self.last_progress = 0
        self.evidence = deque(maxlen=10)

    def update(self, obs):
        t = now_of(obs)
        score = finite(obs.get("score"), self.last_score or 0.0)
        delta = score - self.last_score if self.last_score is not None else 0.0
        current = colors_of(obs)
        if current is None:
            boot = needed_of(obs)
            if boot is not None and any(x > 0 for x in boot):
                self.need = list(boot)
            self.last_score = score
            return False

        changed = ([i for i in range(21)
                    if self.previous is not None and current[i] != self.previous[i]])
        if self.previous is not None and len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_start_score = score - delta
                self.flash_origin = node_of(obs.get("qbert_node"))
                self.ctx.log("temporal_flash_begin", tick=t,
                             score=score, origin=self.flash_origin)
            self.flash_last_change = t

        if self.flashing:
            quiet = t - self.flash_last_change
            at_top = node_of(obs.get("qbert_node")) == 0
            if quiet >= 4 and (at_top or quiet >= 10):
                tally = score - self.flash_start_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.flashing = False
                self.goal = None
                self.need = [1] * 21
                self.complete_since = None
                self.last_progress = t
                self.evidence.clear()
                self.ctx.log("temporal_epoch", tick=t, epoch=self.epoch,
                             inferred_clears=self.clears, tally=tally,
                             reported_rounds=obs.get("completed_rounds"))
                self.previous = current
                self.last_score = score
                return True
            self.need = [1] * 21
            self.previous = current
            self.last_score = score
            return False

        # A positive 25-point component is the strongest public evidence for
        # the current destination.  325 can include a safe bonus and remains a
        # valid calibration event.
        if delta > 0 and int(round(delta)) % 100 == 25 and 0 < len(changed) <= 3:
            qnode = node_of(obs.get("qbert_node"))
            landed = qnode if qnode in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                self.goal = current[landed]
                self.last_progress = t
                phases = obs.get("cube_phase")
                phase = phases[landed] if isinstance(phases, list) and landed < len(phases) else None
                self.evidence.append({"tick": t, "node": landed,
                                      "color": current[landed],
                                      "reward": delta, "phase": phase})
                self.ctx.log("destination_observed", tick=t, epoch=self.epoch,
                             node=landed, color=current[landed], reward=delta)

        if self.goal is None:
            boot = needed_of(obs)
            self.need = list(boot) if boot is not None and any(x > 0 for x in boot) else [1] * 21
        else:
            self.need = [0 if color == self.goal else 1 for color in current]

        if not any(self.need):
            if self.complete_since is None:
                self.complete_since = t
        else:
            self.complete_since = None

        # Do not immediately resurrect a stale one-cube target.  If a ROM
        # cycle gives no global flash, a slow probe is safer than the old rapid
        # false-handoff loop.
        if (self.complete_since is not None and t - self.complete_since >= 36
                and not self.flashing):
            boot = needed_of(obs)
            self.goal = None
            self.need = list(boot) if boot is not None and any(x > 0 for x in boot) else [1] * 21
            self.complete_since = None
            self.ctx.log("cycle_probe", tick=t, epoch=self.epoch)

        self.previous = current
        self.last_score = score
        return False

    def unfinished(self):
        return [n for n, value in enumerate(self.need) if value > 0]


class HazardField:
    """Spatial shadows plus a temporal quarantine for suppressed Purple Ball."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.records = {}
        self.purple_state = 0
        self.latent = False
        self.latent_since = -1
        self.last_tick = 0
        # The timer is a late-board signal in the archived traces.  Applying it
        # during the reliable opening sweep turns a harmless scheduler phase
        # into a self-inflicted route detour, so the controller enables the
        # time-expanded field only after the fourth inferred clear.
        self.late = False

    @staticmethod
    def _descendants(start, depth):
        result = {start}
        frontier = {start}
        for _ in range(max(0, depth)):
            nxt = set()
            for n in frontier:
                for action, child in GRAPH.get(n, {}).items():
                    if action in ("DOWN", "RIGHT"):
                        nxt.add(child)
            result.update(nxt)
            frontier = nxt
        return result

    def update(self, obs):
        t = now_of(obs)
        seen = set()
        hazards = obs.get("hazards")
        if isinstance(hazards, list):
            for item in hazards:
                if not isinstance(item, dict) or not bool(item.get("lethal")):
                    continue
                kind = str(item.get("kind", "hazard"))
                n = node_of(item.get("estimated_node", item.get("node")))
                if n is None:
                    continue
                old = self.records.get(kind)
                self.records[kind] = {
                    "kind": kind, "node": n, "previous": old.get("node", n) if old else n,
                    "last": t, "visible": True,
                }
                seen.add(kind)

        # Keep only a finite prediction window.  Missing RGB evidence is not
        # treated as a teleport to safety, but it also cannot create a wall
        # forever.
        for kind in list(self.records):
            rec = self.records[kind]
            age = t - rec["last"]
            limit = 22 if kind in ("purple_ball", "red_ball") else 15
            if age > limit:
                del self.records[kind]
            else:
                rec["visible"] = kind in seen

        timers = obs.get("timers")
        self.purple_state = (integer(timers.get("purple_ball_state"), 0)
                             if isinstance(timers, dict) else 0)
        visible_purple = any(k in self.records and self.records[k].get("visible")
                             for k in ("purple_ball", "red_ball"))
        new_latent = (self.late and not visible_purple and self.purple_state >= 20)
        if new_latent and not self.latent:
            self.latent_since = t
            self.ctx.log("latent_purple_begin", tick=t, state=self.purple_state)
        if not new_latent and self.latent:
            self.ctx.log("latent_purple_end", tick=t, state=self.purple_state)
        self.latent = new_latent
        self.last_tick = t

    def age(self, rec, t):
        return max(0, t - rec.get("last", t))

    def possible(self, rec, t):
        n = rec.get("node")
        if n is None:
            return set()
        age = self.age(rec, t)
        if not self.late:
            return {n}
        if rec.get("kind") in ("purple_ball", "red_ball"):
            return self._descendants(n, min(3, 1 + age // 8))
        radius = min(3, 1 + age // 9)
        return {x for x in range(21) if DIST[n][x] <= radius}

    def risk(self, destination, origin, action, t, posture="quarantine"):
        if destination is None:
            return 99.0
        # Before the fourth observed board use the parent's calibrated local
        # danger scale.  This keeps the opening comparison meaningful; the
        # expanded time field below is the experiment under test.
        if not self.late:
            value = 0.0
            for rec in self.records.values():
                n = rec.get("node")
                age = self.age(rec, t)
                if n is None or age > 4:
                    continue
                if n == destination:
                    value += 3.0
                if rec.get("kind") in ("purple_ball", "red_ball"):
                    children = [child for edge, child in GRAPH.get(n, {}).items()
                                if edge in ("DOWN", "RIGHT")]
                    if destination in children:
                        value += 1.5
                elif rec.get("kind") == "coily" and origin is not None:
                    nxt = list(GRAPH.get(n, {}).values())
                    toward = min((DIST[x][origin] for x in nxt), default=99)
                    if destination in nxt and DIST[destination][origin] == toward:
                        value += 2.0
                    if DIST[n][destination] == 1:
                        value += 0.4
            return value
        value = 0.0
        for rec in self.records.values():
            possible = self.possible(rec, t)
            age = self.age(rec, t)
            kind = rec.get("kind")
            if destination in possible:
                value += 9.0 if age <= 4 else 5.0
            if any(DIST[destination][p] == 1 for p in possible):
                value += 3.0 if kind in ("purple_ball", "red_ball") else 1.5
            if kind in ("purple_ball", "red_ball"):
                branches = set()
                for p in possible:
                    branches.update(child for edge, child in GRAPH[p].items()
                                    if edge in ("DOWN", "RIGHT"))
                if destination in branches and destination not in possible:
                    value += 4.0
            elif kind == "coily" and origin is not None:
                if DIST[origin][rec.get("node", origin)] <= 2:
                    value += 1.4
                if rec.get("node") in GRAPH.get(destination, {}).values():
                    value += 1.0

        # The named timer remains informative when the RGB cross-check
        # suppresses the sprite.  This is the key structural repair over the
        # former soft occlusion penalty: it changes route mode, not just cost.
        if self.latent and self.late:
            row = ROWS.get(destination, 5)
            if row >= 5:
                value += 6.0
            elif row >= 4:
                value += 1.8
            if len(GRAPH.get(destination, {})) <= 1:
                value += 6.0
            if action in ("DOWN", "RIGHT"):
                value += 1.4
            if posture == "normal":
                value *= 0.85
        return value

    def current_risk(self, node, t, posture="quarantine"):
        return self.risk(node, node, None, t, posture)

    def visible_facts(self, origin, t):
        result = []
        for rec in self.records.values():
            possible = self.possible(rec, t)
            nearest = min((DIST[origin][x] for x in possible), default=99) if origin is not None else 99
            result.append({
                "kind": rec.get("kind"),
                "visibility": "visible" if rec.get("visible") else "recent_shadow",
                "proximity": "same" if nearest == 0 else "adjacent" if nearest == 1 else
                             "near" if nearest <= 2 else "far",
                "motion": "descending" if rec.get("kind") in ("purple_ball", "red_ball")
                          else "pursuing",
            })
        if self.latent:
            result.append({"kind": "purple_timer", "visibility": "occluded",
                           "proximity": "uncertain", "motion": "active"})
        return result

    def signature(self, t):
        phase = (self.purple_state // 4) if self.purple_state >= 20 else 0
        return (tuple(sorted((k, v.get("node"), bool(v.get("visible")))
                             for k, v in self.records.items())),
                bool(self.latent), phase)


class RoutePlanner:
    """Short-horizon route generation with shelter and edge-memory modes."""

    def __init__(self, board, radar, ctx):
        self.board = board
        self.radar = radar
        self.ctx = ctx
        self.target = None
        self.last_node = None
        self.last_edge = None
        self.recent_nodes = deque(maxlen=20)
        self.recent_edges = deque(maxlen=12)
        self.bad_edges = {}  # directed edge -> expiry tick
        self.corner_cooldown = {}  # endpoint -> expiry tick
        self.wait_since = None
        self.route_posture = "nearest"
        self.temporal_posture = "quarantine"
        self.policy_version = 0

    def reset_epoch(self):
        self.target = None
        self.last_edge = None
        self.recent_nodes.clear()
        self.recent_edges.clear()
        self.wait_since = None

    def note_landing(self, source, destination, t, expected=None):
        if source is None or destination is None:
            return
        self.last_node = destination
        self.last_edge = (source, destination)
        self.recent_nodes.append(destination)
        self.recent_edges.append((source, destination, t))
        if self.radar.late and len(GRAPH.get(destination, {})) <= 1:
            self.corner_cooldown[destination] = t + 55
        if expected is not None and expected != destination:
            self.bad_edges[(source, expected)] = t + 180
            self.ctx.log("route_resync", tick=t, expected=expected, observed=destination)
        if self.target == destination and self.board.need[destination] <= 0:
            self.target = None

    def note_life_loss(self, t, transaction=None):
        # A death during an animation is evidence against the committed edge,
        # even if the decoder has already hidden Q*bert.  Penalize both travel
        # directions for a finite period, then allow a required edge again.
        edges = []
        if isinstance(transaction, dict):
            src, dst = transaction.get("source"), transaction.get("destination")
            if src is not None and dst is not None:
                edges.append((src, dst))
        if self.last_edge is not None:
            edges.append(self.last_edge)
        for src, dst in edges:
            self.bad_edges[(src, dst)] = max(self.bad_edges.get((src, dst), 0), t + 170)
            self.bad_edges[(dst, src)] = max(self.bad_edges.get((dst, src), 0), t + 170)
        self.target = None
        self.wait_since = None
        self.ctx.log("edge_memory_after_loss", tick=t, edges=edges)

    def _alive_bad(self, edge, t):
        return self.bad_edges.get(edge, 0) > t

    def choose_target(self, current, shelter=False):
        if shelter:
            candidates = [n for n in CENTRAL if n != current and self.board.need[n] >= 0]
            if not candidates:
                return None
            candidates.sort(key=lambda n: (DIST[current][n], CENTRAL_RANK.get(n, 99)))
            return candidates[0]

        unfinished = set(self.board.unfinished())
        if self.target in unfinished and self.target != current:
            return self.target
        if self.target == current and self.board.need[current] > 0:
            return current
        if not unfinished:
            return None
        ranks = SWEEP_RANK if self.route_posture == "sweep" else CENTRAL_RANK
        ordered = sorted(unfinished, key=lambda n: (ranks.get(n, 99), DIST[current][n]))
        # Prefer a target that is reachable without making a corner the first
        # necessary commitment, but do not discard a sole remaining cube.
        roomy = [n for n in ordered if len(GRAPH.get(n, {})) > 1]
        self.target = (roomy[0] if roomy else ordered[0])
        return self.target

    def _early_options(self, obs, t):
        """Keep the parent's proven opening motor until the late regime.

        The new experiment should not pay for a speculative temporal policy on
        boards where 0007 already supplied a strong, repeatable traversal.
        This branch still uses this candidate's observation-calibrated board
        and Jev referee, but restores the parent's nearest-useful objective.
        """
        current = node_of(obs.get("qbert_node"))
        if current is None:
            return []
        supplied = set(legal_action_list(obs.get("safe_moves")))
        edges = [(a, GRAPH[current].get(a)) for a in TIE_ORDER
                 if a in supplied and a in GRAPH[current]]
        targets = self.board.unfinished()
        if not edges or not targets:
            return []
        visits = Counter(self.recent_nodes)
        candidates = []
        for action, destination in edges:
            useful = self.board.need[destination] > 0
            distance = min(DIST[destination][target] for target in targets)
            risk = self.radar.risk(destination, current, action, t, "balanced")
            escape = sum(self.radar.risk(nxt, destination, None, t, "balanced") < 2.0
                         for nxt in GRAPH[destination].values())
            cost = 2.0 * distance + (0.0 if useful else 0.5)
            cost += 12.0 * risk + 0.12 * visits[destination] - 0.12 * escape
            coily_near = any(rec.get("kind") == "coily"
                             and rec.get("node") is not None
                             and DIST[destination][rec["node"]] <= 2
                             for rec in self.radar.records.values())
            if coily_near and escape == 0:
                cost += 14.0
            if self.route_posture == "central" and useful:
                cost += ROWS.get(destination, 5) * 0.15
            elif self.route_posture == "sweep" and useful:
                cost -= ROWS.get(destination, 5) * 0.12
            candidates.append({"action": action, "destination": destination,
                               "cost": cost, "risk": risk, "useful": useful,
                               "escape": escape, "reverse": False, "corner": False,
                               "bad": False, "distance": distance,
                               "retreat": False})
        candidates.sort(key=lambda item: (item["cost"], TIE_ORDER.index(item["action"])))

        current_risk = self.radar.current_risk(current, t, "balanced")
        visible_ball = any(rec.get("kind") in ("purple_ball", "red_ball")
                           and rec.get("visible") for rec in self.radar.records.values())
        best = candidates[0] if candidates else None
        if (best is not None and visible_ball and best["risk"] >= 1.2
                and current_risk + 0.4 < best["risk"]):
            wait_age = t - self.wait_since if self.wait_since is not None else 0
            if self.wait_since is None:
                self.wait_since = t
                wait_age = 0
                self.ctx.log("opening_hazard_wait", tick=t, node=current,
                             best_risk=best["risk"], current_risk=current_risk)
            if wait_age < 18:
                candidates.insert(0, {"action": "NOOP", "destination": current,
                                      "cost": best["cost"] - 1.0,
                                      "risk": current_risk, "useful": False,
                                      "escape": len(GRAPH[current]), "reverse": False,
                                      "corner": False, "bad": False, "distance": 0,
                                      "retreat": False, "wait": True})
        else:
            self.wait_since = None
        self.target = targets[0] if targets else None
        return candidates

    def options(self, obs, t, survival_until=0):
        # Route choice is intentionally asymmetric across the episode: the
        # proven opening controller is retained, while late epochs use the new
        # temporal shelter planner below.
        if not self.radar.late and t >= survival_until:
            return self._early_options(obs, t)
        current = node_of(obs.get("qbert_node"))
        if current is None:
            return []
        supplied = legal_action_list(obs.get("safe_moves"))
        edges = [(a, GRAPH[current].get(a)) for a in DIRECTIONS
                 if a in supplied and a in GRAPH[current]]
        if not edges:
            return []

        latent = self.radar.latent
        # Shelter only when Q*bert is actually exposed: at a bottom/near-
        # bottom node, during recovery, or on a current high-risk cube.  A
        # global shelter target on the apex made the late controller orbit
        # central nodes without covering cubes.
        current_risk = self.radar.current_risk(current, t, self.temporal_posture)
        # Hidden-timer evidence remains a cost/veto on exposed destinations,
        # but is not by itself a reason to abandon the coverage frontier:
        # doing so created an apex orbit in the first late trial.
        exposed = (ROWS.get(current, 5) >= 4 or len(GRAPH[current]) <= 2
                   or current_risk >= 2.5)
        retreat = t < survival_until
        target = self.choose_target(current, shelter=retreat)
        unfinished = set(self.board.unfinished())
        visits = Counter(self.recent_nodes)
        candidates = []
        for action, destination in edges:
            useful = destination in unfinished
            risk = self.radar.risk(destination, current, action, t,
                                    self.temporal_posture)
            escape = sum(self.radar.risk(nxt, destination, None, t,
                                         self.temporal_posture) < 3.0
                         for nxt in GRAPH[destination].values())
            reverse = (self.last_edge is not None
                       and self.last_edge[0] == destination
                       and self.last_edge[1] == current)
            bad = self._alive_bad((current, destination), t)
            corner = len(GRAPH[destination]) <= 1
            if target is None:
                distance = 0
            else:
                distance = DIST[destination][target]
            cost = 2.0 * distance + (0.0 if useful else 0.65)
            cost += 11.0 * min(risk, 10.0)
            cost += 0.35 * visits[destination]
            cost += 4.0 if escape == 0 else 1.5 if escape == 1 else 0.0
            if reverse:
                cost += 7.0
            if corner:
                cost += 5.0 if not retreat else 13.0
            if bad:
                cost += 28.0
            if self.corner_cooldown.get(destination, 0) > t:
                cost += 5.0
            if retreat:
                if ROWS.get(destination, 5) >= 5:
                    cost += 7.0
                if action in ("DOWN", "RIGHT"):
                    cost += 2.0
            elif latent and exposed:
                # Keep coverage running from safe central nodes, but do not
                # enter a bottom endpoint while the timer is unresolved.
                if ROWS.get(destination, 5) >= 5:
                    cost += 7.0
                if corner:
                    cost += 9.0
            candidates.append({
                "action": action, "destination": destination, "cost": cost,
                "risk": risk, "useful": useful, "escape": escape,
                "reverse": reverse, "corner": corner, "bad": bad,
                "distance": distance, "retreat": retreat,
            })

        candidates.sort(key=lambda x: (x["cost"], x["risk"], x["destination"]))

        # Structural filters.  They are deliberately conditional: an edge
        # leading to the only unfinished endpoint remains available when no
        # roomier alternative can reach the objective.
        nonreverse = [c for c in candidates if not c["reverse"]]
        if nonreverse:
            best_nonreverse = min(c["cost"] for c in nonreverse)
            candidates = [c for c in candidates
                          if not c["reverse"] or c["cost"] <= best_nonreverse + 2.0]
        noncorner = [c for c in candidates if not c["corner"]]
        if noncorner:
            best_open = min(c["cost"] for c in noncorner)
            candidates = [c for c in candidates
                          if not c["corner"] or c["cost"] <= best_open + (2.5 if retreat else 5.0)]
        unblocked = [c for c in candidates if not c["bad"]]
        if unblocked:
            candidates = unblocked

        # A short, explicit wait is a candidate only when the current cube is
        # safer than all exits.  It is never an unbounded idle fallback.
        current_risk = self.radar.current_risk(current, t, self.temporal_posture)
        if candidates and (latent or any(c["risk"] >= current_risk + 2.0 for c in candidates)):
            wait_age = t - self.wait_since if self.wait_since is not None else 0
            if self.wait_since is None and current_risk < 2.5:
                self.wait_since = t
                wait_age = 0
                self.ctx.log("shelter_wait_begin", tick=t, node=current,
                             latent=latent, current_risk=current_risk)
            if self.wait_since is not None and wait_age < (8 if latent else 5):
                candidates.insert(0, {
                    "action": "NOOP", "destination": current,
                    "cost": candidates[0]["cost"] - 1.0,
                    "risk": current_risk, "useful": False, "escape": len(GRAPH[current]),
                    "reverse": False, "corner": False, "bad": False,
                    "distance": 0, "retreat": True, "wait": True,
                })
        elif not latent:
            self.wait_since = None

        return candidates


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.board = PaletteBoard(ctx)
        self.radar = HazardField(ctx)
        self.planner = RoutePlanner(self.board, self.radar, ctx)
        self.names = set(ACTIONS)
        try:
            self.names = {str(x) for x in game.action_spec.names}
        except Exception:
            pass
        self.noop = "NOOP" if "NOOP" in self.names else next(iter(self.names), "NOOP")

        self.lives = None
        self.life_epoch = 0
        self.survival_until = 0
        self.transaction = None
        self.neutral_until = 0
        self.last_posted = self.noop
        self.last_log = -999

        # Jev is a referee for the code plan, not a source of joystick strings.
        self.s1 = None
        self.s1_meta = None
        self.s1_last = -999
        self.timing_advice = None

        # Astra authors a durable posture, off the game clock.
        self.s2 = None
        self.s2_session = None
        self.s2_calls = 0
        self.s2_last = -999
        self.s2_reason = "initial route and hazard contract"

    def ingest(self, obs):
        t = now_of(obs)
        old_epoch = self.board.epoch
        self.board.update(obs)
        # Archived losses that motivate this experiment occur after the fourth
        # palette epoch.  Preserve the parent's reliable early sweep and only
        # enable expanded shadows/timer quarantine in that late regime.
        self.radar.late = self.board.epoch >= 4
        self.radar.update(obs)
        if self.board.epoch != old_epoch:
            self.planner.reset_epoch()
            self.transaction = None
            self.timing_advice = None
            self.neutral_until = max(self.neutral_until, t + 2)
            self.s2_reason = "new palette epoch; revise coverage posture"
            self.ctx.log("controller_epoch", tick=t, epoch=self.board.epoch,
                         inferred_clears=self.board.clears)

        lives = integer(obs.get("lives"), self.lives if self.lives is not None else 4)
        if self.lives is not None and lives < self.lives:
            self.life_epoch += 1
            self.planner.note_life_loss(t, self.transaction)
            self.transaction = None
            self.timing_advice = None
            if self.radar.late:
                self.survival_until = t + 120
            else:
                self.survival_until = 0
            self.neutral_until = max(self.neutral_until, t + 6)
            self.s2_reason = "life loss; strengthen temporal shelter"
            self.ctx.log("life_recovery", tick=t, lives=lives,
                         life_epoch=self.life_epoch)
        self.lives = lives

    def _compact_state(self, obs, options):
        t = now_of(obs)
        current = node_of(obs.get("qbert_node"))
        proposed = options[0]["action"] if options else self.noop
        candidates = []
        for item in options:
            action = item["action"]
            candidates.append({
                "role": "hold" if action == "NOOP" else
                        "escape" if item.get("retreat") else "coverage",
                "action": action,
                "landing": "unfinished" if item.get("useful") else "transit",
                "danger": "blocked" if item["risk"] >= 8 else
                           "near" if item["risk"] >= 2 else "clear",
                "escape_space": "multiple" if item["escape"] >= 2 else
                                "narrow" if item["escape"] == 1 else "none",
                "temporal": "reverse" if item.get("reverse") else
                            "corner" if item.get("corner") else "ordinary",
                "code_rank": len(candidates),
            })
        return json_value({
            "game": "Atari 2600 Q*bert; six-row 21-node board",
            "rule": "Code has checked current safe_moves. Never FIRE or invent an edge.",
            "position": {"node": current, "standing": standing(obs),
                         "last_posted": self.last_posted},
            "progress": {"epoch": self.board.epoch,
                          "unfinished": len(self.board.unfinished()),
                          "inferred_clears": self.board.clears,
                          "target": self.planner.target},
            "temporal": {"purple_timer": self.radar.purple_state,
                          "latent_purple": self.radar.latent,
                          "quarantine_age": (t - self.radar.latent_since
                                              if self.radar.latent_since >= 0 else 0),
                          "survival_recovery": t < self.survival_until},
            "candidates": candidates,
            "proposed_action": proposed,
            "hazards": self.radar.visible_facts(current, t),
        })

    def start_jev(self, obs, options):
        if self.s1 is not None or not options or self.ctx.done:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        t = now_of(obs)
        if t - self.s1_last < 8 or not standing(obs):
            return
        if self.ctx.budget.remaining().get("system1_calls", 1) <= 2:
            return

        # The question's options are roles.  Only code-generated alternatives
        # are described; Jev cannot return a new direction.
        roles = ("commit", "hold", "retreat", "unknown")
        first = {
            "commit": "Permit the code-selected edge now: its supplied risk, escape space, and temporal posture support an immediate hop.",
            "hold": "Keep NOOP for a few ticks on the current cube: the current cube is safer than the exits or a hidden hazard needs to resolve.",
            "retreat": "Reject the code-selected coverage edge and use the code's next safest escape candidate; do not invent an edge.",
            "unknown": "The supplied facts do not support a timing preference.",
        }
        second = dict(reversed(list(first.items())))
        proposed = options[0]["action"]
        state = self._compact_state(obs, options)
        instruction = (
            "Judge timing, not graph arithmetic. Code-generated candidates and safe_moves are authoritative. "
            "Choose hold when the current cube is safer, choose retreat when the proposed landing is a concrete "
            "near hazard or a temporal corner commitment, and choose commit only when the proposed edge is "
            "clear enough. Unknown is acceptable. Never return FIRE or an action absent from candidates."
        )
        questions = {
            "timing_a": Choice(instruction, first),
            "timing_b": Choice(instruction, second),
            "immediate": Noul(
                "Do the supplied facts establish immediate lethal risk for proposed_action? "
                "An active hidden timer or a distant enemy alone is not proof unless it affects the proposed landing.",
                {"true": "The proposed landing or its hop path has concrete near lethal risk.",
                 "false": "The proposed landing is clear, or the threat is distant/uncertain."}),
        }
        self.s1_last = t
        self.s1_meta = {"tick": t, "node": node_of(obs.get("qbert_node")),
                        "epoch": self.board.epoch, "life": self.life_epoch,
                        "policy": self.planner.policy_version,
                        "signature": self.radar.signature(t),
                        "proposed": proposed, "options": [x["action"] for x in options]}

        def work():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.95),
                        "error": None}
            except Exception as exc:
                return {"answers": None, "error": type(exc).__name__}

        try:
            self.s1 = self.ctx.spawn(work)
        except Exception as exc:
            self.s1 = None
            self.s1_meta = None
            self.ctx.log("jev_spawn_failure", error=type(exc).__name__)

    def collect_jev(self, obs):
        if self.s1 is None or not self.s1.done():
            return
        future, meta = self.s1, self.s1_meta
        self.s1 = None
        self.s1_meta = None
        try:
            result = future.result()
            answers = result.get("answers") if isinstance(result, dict) else None
            if not isinstance(answers, dict) or not isinstance(meta, dict):
                return
            t = now_of(obs)
            fresh = (0 <= t - meta["tick"] <= 5
                     and node_of(obs.get("qbert_node")) == meta["node"]
                     and self.board.epoch == meta["epoch"]
                     and self.life_epoch == meta["life"]
                     and self.planner.policy_version == meta["policy"]
                     and self.radar.signature(t) == meta["signature"]
                     and standing(obs))
            if not fresh:
                return
            a = answer_value(answers.get("timing_a"))
            b = answer_value(answers.get("timing_b"))
            if a != b or a not in ("commit", "hold", "retreat", "unknown"):
                return
            p = min(answer_probability(answers.get("timing_a"), a),
                    answer_probability(answers.get("timing_b"), b))
            immediate = finite(getattr(answers.get("immediate"), "value", 0.0), 0.0)
            self.timing_advice = dict(meta, timing=a if p >= 0.60 else "unknown",
                                      probability=p, immediate=immediate)
        except Exception as exc:
            self.ctx.log("jev_result_failure", error=type(exc).__name__)

    def referee(self, obs, options):
        """Convert a fresh role judgment into a bounded control preference."""
        if not options:
            return None
        fallback = options[0]
        advice = self.timing_advice
        self.timing_advice = None
        if not advice:
            return fallback
        t = now_of(obs)
        fresh = (0 <= t - advice.get("tick", t) <= 5
                 and node_of(obs.get("qbert_node")) == advice.get("node")
                 and self.board.epoch == advice.get("epoch")
                 and self.life_epoch == advice.get("life")
                 and self.planner.policy_version == advice.get("policy")
                 and self.radar.signature(t) == advice.get("signature")
                 and standing(obs))
        if not fresh:
            return fallback
        if advice.get("immediate", 0.0) >= 0.82 and len(options) > 1:
            alt = next((x for x in options[1:] if x["risk"] <= fallback["risk"] + 1.0
                        and not x.get("bad")), None)
            if alt is not None:
                self.ctx.log("jev_immediate_veto", tick=t,
                             old=fallback["action"], new=alt["action"])
                return alt
            return fallback if fallback["action"] == "NOOP" else {
                **fallback, "action": "NOOP", "wait": True
            }
        timing = advice.get("timing")
        if timing == "hold":
            hold = next((x for x in options if x["action"] == "NOOP"), None)
            if hold is not None:
                self.ctx.log("jev_hold", tick=t)
                return hold
        if timing == "retreat":
            alt = next((x for x in options[1:]
                        if x["action"] != "NOOP" and not x.get("bad")
                        and x["risk"] <= fallback["risk"] + 1.5), None)
            if alt is not None:
                self.ctx.log("jev_retreat", tick=t,
                             old=fallback["action"], new=alt["action"])
                return alt
        return fallback

    def start_astra(self, obs):
        if not self.s2_reason or self.s2 is not None or self.s2_calls >= 3 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        if self.ctx.budget.remaining().get("system2_calls", 1) <= 1:
            return
        t = now_of(obs)
        if self.s2_calls and t - self.s2_last < 300:
            return
        reason = self.s2_reason
        self.s2_reason = ""
        self.s2_calls += 1
        self.s2_last = t
        evidence = {
            "reason": reason, "tick": t, "score": finite(obs.get("score")),
            "lives": self.lives, "epoch": self.board.epoch,
            "inferred_clears": self.board.clears,
            "unfinished": len(self.board.unfinished()),
            "latent_purple": self.radar.latent,
            "purple_timer": self.radar.purple_state,
            "hazards": self.radar.visible_facts(node_of(obs.get("qbert_node")), t),
            "recent_landings": list(self.planner.recent_edges)[-5:],
            "current_policy": {"route": self.planner.route_posture,
                               "temporal": self.planner.temporal_posture},
            "palette_evidence": list(self.board.evidence)[-4:],
        }
        prompt = (
            "You are the slow planner for a real-time Atari 2600 Q*bert controller. "
            "The board model owns the 21-node graph, finite colour bookkeeping, legal safe_moves, "
            "and a one-hop motor. A temporal hazard field retains suppressed Purple Ball evidence; "
            "a latent purple timer is treated as a quarantine, especially near bottom endpoints. "
            "Jev does not invent actions: it only judges commit/hold/retreat roles over code candidates. "
            "Choose a durable posture, not a joystick sequence. Return ONLY JSON under 55 tokens: "
            "route must be sweep, nearest, or central; temporal must be quarantine or balanced. "
            "Quarantine is preferred after hidden-hazard evidence, life loss, or late epochs. "
            "Balanced is allowed only if quarantine demonstrably stalls. Never recommend FIRE, a disc guess, "
            "an edge fall, or a seed-specific route. Evidence="
            + json.dumps(json_value(evidence), separators=(",", ":"))
        )
        session = self.s2_session

        def work():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=105.0)
            except Exception as exc:
                return {"error": type(exc).__name__}

        try:
            self.s2 = self.ctx.spawn(work)
            self.ctx.log("astra_requested", tick=t, reason=reason)
        except Exception as exc:
            self.s2 = None
            self.ctx.log("astra_spawn_failure", error=type(exc).__name__)

    def collect_astra(self, obs):
        if self.s2 is None or not self.s2.done():
            return
        future = self.s2
        self.s2 = None
        try:
            result = future.result()
            text = getattr(result, "text", "") if result is not None else ""
            data = first_object(text)
            old = (self.planner.route_posture, self.planner.temporal_posture)
            route = data.get("route")
            temporal = data.get("temporal")
            if route in ("sweep", "nearest", "central"):
                self.planner.route_posture = route
            if temporal in ("quarantine", "balanced"):
                self.planner.temporal_posture = temporal
            # The latent timer gate is never disabled by prose or an invalid
            # answer; Astra only changes the tie-breaking posture around it.
            if old != (self.planner.route_posture, self.planner.temporal_posture):
                self.planner.policy_version += 1
                self.timing_advice = None
                self.ctx.log("astra_posture_changed", tick=now_of(obs),
                             old=list(old), route=self.planner.route_posture,
                             temporal=self.planner.temporal_posture)
            else:
                self.ctx.log("astra_posture_unchanged", tick=now_of(obs),
                             parsed=bool(data))
            self.s2_session = getattr(result, "session_id", None)
        except Exception as exc:
            self.ctx.log("astra_result_failure", error=type(exc).__name__)

    def choose(self, obs):
        t = now_of(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)

        if self.board.flashing or self.board.complete_since is not None:
            self.transaction = None
            return self.noop

        phase = str(motion_of(obs).get("phase", "unknown")).lower()
        current = node_of(obs.get("qbert_node"))
        if current is None or phase in ("falling", "not_visible") or motion_of(obs).get("falling"):
            self.transaction = None
            return self.noop

        # Verify the previous hop before considering a new one.  A direction is
        # retried at most twice while Q*bert is still standing; all animation
        # ticks get neutral input, so a sticky action cannot become a long macro.
        if self.transaction is not None:
            tx = self.transaction
            age = t - tx["start"]
            if current != tx["source"] and standing(obs):
                self.planner.note_landing(tx["source"], current, t, tx["destination"])
                self.ctx.log("hop_observed", tick=t, source=tx["source"],
                             expected=tx["destination"], observed=current,
                             duration=age)
                self.transaction = None
                self.neutral_until = max(self.neutral_until, t + 1)
                return self.noop
            if not bool(obs.get("qbert_in_hop")) and phase == "standing" and age <= 2 and tx["posts"] < 2:
                tx["posts"] += 1
                return tx["action"]
            if age > 20 and standing(obs):
                self.planner.bad_edges[(tx["source"], tx["destination"])] = t + 80
                self.transaction = None
                self.neutral_until = max(self.neutral_until, t + 1)
            else:
                return self.noop

        if not standing(obs) or t < self.neutral_until:
            return self.noop

        options = self.planner.options(obs, t, self.survival_until)
        if not options:
            return self.noop
        self.start_jev(obs, options)
        selected = self.referee(obs, options)
        if selected is None:
            return self.noop
        action = selected.get("action")
        if action == "NOOP":
            return self.noop
        if action not in legal_action_list(obs.get("safe_moves")):
            return self.noop
        destination = GRAPH[current].get(action)
        if destination is None:
            return self.noop

        self.transaction = {"source": current, "destination": destination,
                            "action": action, "start": t, "posts": 1}
        if t - self.last_log >= 24:
            self.ctx.log("temporal_route", tick=t, score=finite(obs.get("score")),
                         lives=self.lives, epoch=self.board.epoch,
                         inferred_clears=self.board.clears,
                         unfinished=len(self.board.unfinished()), node=current,
                         target=self.planner.target, action=action,
                         latent=self.radar.latent,
                         hazards=self.radar.visible_facts(current, t))
            self.last_log = t
        return action

    def run(self):
        while not self.ctx.done and not getattr(self.game, "done", False):
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.ingest(obs)
            self.start_astra(obs)
            action = self.choose(obs)
            if action not in self.names:
                action = self.noop if self.noop in self.names else "NOOP"
            self.ctx.act(action)
            self.last_posted = action
            self.ctx.wait_for_tick()
        self.ctx.log("candidate_summary", score=getattr(self.game, "score", None),
                     inferred_clears=self.board.clears, epoch=self.board.epoch,
                     lives=self.lives, route=self.planner.route_posture,
                     temporal=self.planner.temporal_posture)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
