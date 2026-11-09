"""Adaptive transaction control for the Atari 2600 Q*bert port.

The controller deliberately separates three sorts of state:

* ``Board`` learns the palette cycle from the public colour transitions and
  supplies the landing demand.  The cartridge's phase labels are useful
  evidence, but are not treated as a permanent colour dictionary.
* ``FailureLedger`` records the edge and public scheduler state that preceded
  an actual life loss.  It is a short-lived, in-episode postmortem rather than
  a guessed enemy decoder or a seed-specific route.
* ``Motor`` executes one graph edge at a time.  It never waits for a model and
  rechecks the current legal/safe edges before using Jev's advice.

Only the public observation and the metaharness interfaces are used.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


# The Atari cartridge has six rows and 21 cubes.  Direction names are the ALE
# joystick edges; this is intentionally not the 28-cube arcade graph.
DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
ACTION_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")
ALL_ACTIONS = DIRECTIONS + ("NOOP",)
GRAPH = {}
ROW = {}
for row in range(6):
    base = row * (row + 1) // 2
    for col in range(row + 1):
        node = base + col
        ROW[node] = row
        edges = {}
        if row and col < row:
            edges["UP"] = node - row
        if row and col:
            edges["LEFT"] = node - row - 1
        if row < 5:
            edges["DOWN"] = node + row + 1
            edges["RIGHT"] = node + row + 2
        GRAPH[node] = edges

DIST = []
for root in range(21):
    dist = [99] * 21
    dist[root] = 0
    q = deque([root])
    while q:
        here = q.popleft()
        for there in GRAPH[here].values():
            if dist[there] == 99:
                dist[there] = dist[here] + 1
                q.append(there)
    DIST.append(dist)

# A loose central ordering is only a tie breaker during recovery.  It does not
# encode a joystick sequence.
CENTRAL = (4, 7, 8, 11, 12, 13, 6, 10, 16, 17, 3, 5, 9, 14)
CENTRAL_RANK = {n: i for i, n in enumerate(CENTRAL)}


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


def node_value(value):
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if 0 <= n < 21 else None


def current_tick(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)), 0)


def qmotion(obs):
    value = obs.get("qbert_motion")
    return value if isinstance(value, dict) else {"phase": str(value or "unknown")}


def is_standing(obs):
    n = node_value(obs.get("qbert_node"))
    m = qmotion(obs)
    phase = str(m.get("phase", "unknown")).lower()
    return (n is not None and not bool(obs.get("qbert_in_hop"))
            and phase not in ("hopping", "falling", "not_visible")
            and not bool(m.get("falling")))


def cube_colors(obs):
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


def safe_names(obs, field):
    value = obs.get(field, [])
    if not isinstance(value, (list, tuple)):
        return []
    return [x for x in value if isinstance(x, str) and x in ALL_ACTIONS]


def answer_choice(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value if isinstance(value, str) else None


def answer_probability(answer, option):
    probabilities = getattr(answer, "probabilities", None) if answer is not None else None
    if not isinstance(probabilities, dict):
        return 0.0
    return finite(probabilities.get(option), 0.0)


def object_from_text(text):
    """Extract one JSON object without trusting model formatting."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for offset, char in enumerate(text[:16000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def json_safe(value):
    """Keep every model request strict-JSON even with odd adaptor fields."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


class Board:
    """Observation-calibrated cube demand, including later colour cycles."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.last_score = None
        self.goal_roles = {}       # raw colour -> S/I/D
        self.goal_color = None     # simple destination model for the proven opening
        self.epoch = 0             # inferred scored whole-board transitions
        self.clears = 0
        self.flashing = False
        self.flash_last_change = -999
        self.flash_score = 0.0
        self.flash_origin = None
        self.need = [1] * 21
        self.last_progress = 0
        self.complete_since = None
        self.evidence = deque(maxlen=10)
        self.last_level = 1

    def level(self):
        # The public level field is helpful when available, but a cartridge
        # flash is the more reliable boundary in these traces.
        return min(5, self.epoch // 4 + 1)

    @staticmethod
    def transition(level):
        if level == 1:
            return {"S": "D", "D": "D"}
        if level == 2:
            return {"S": "I", "I": "D", "D": "D"}
        if level == 3:
            return {"S": "D", "D": "S"}
        if level == 4:
            return {"S": "I", "I": "D", "D": "I"}
        return {"S": "I", "I": "D", "D": "S"}

    def distance_to_destination(self, role, level):
        """Number of further landings until D under this level's cycle."""
        if role not in ("S", "I", "D"):
            return 1
        if level in (1, 2):
            return {"S": 1 if level == 1 else 2, "I": 1, "D": 0}[role]
        steps = 0
        here = role
        transition = self.transition(level)
        # Every documented cycle reaches D quickly.  The cap prevents a bad
        # observation from creating an infinite bookkeeping loop.
        while steps < 6:
            if here == "D" and steps > 0:
                return steps
            if here == "D" and steps == 0:
                # For reversible levels, landing on D changes it again.
                here = transition[here]
                steps += 1
                continue
            here = transition.get(here, "D")
            steps += 1
            if here == "D":
                return steps
        return 1

    def _assign_transition(self, old, new, level):
        old_role = self.goal_roles.get(old)
        if old_role is None:
            return
        nxt = self.transition(level).get(old_role)
        if nxt is None:
            return
        # A role can be observed again when a reversible cube returns to S/I.
        # Do not overwrite an already observed colour role merely because Sam
        # or a flash produced an out-of-order transition.
        if new not in self.goal_roles:
            self.goal_roles[new] = nxt

    def _recompute(self, colors):
        level = self.level()
        # Until a later-cycle transition has actually been observed, preserve
        # the parent's high-performing destination test.  The role table is
        # still collected in parallel; it becomes authoritative once a local
        # intermediate/reversal is evidenced rather than being guessed from a
        # raw palette byte alone.
        if self.goal_color is not None and not any(
                role == "I" for role in self.goal_roles.values()):
            self.need = [0 if color == self.goal_color else 1 for color in colors]
            return
        result = []
        for color in colors:
            role = self.goal_roles.get(color)
            result.append(self.distance_to_destination(role, level))
        self.need = result
        if not any(x > 0 for x in result):
            if self.complete_since is None:
                self.complete_since = self.last_progress
        else:
            self.complete_since = None

    def _new_epoch(self, colors, score, tick):
        tally = score - self.flash_score
        if tally >= 1000:
            self.clears += 1
        self.epoch += 1
        self.flashing = False
        self.flash_last_change = -999
        self.flash_origin = None
        self.goal_roles = {}
        if colors:
            self.goal_roles[colors[0]] = "S"
        self.goal_color = None
        self.complete_since = None
        self.last_progress = tick
        self.evidence.clear()
        self.ctx.log("board_epoch_rebased", tick=tick, epoch=self.epoch,
                     inferred_clears=self.clears, tally_points=tally,
                     level=self.level(), reported_rounds=None)

    def update(self, obs):
        t = current_tick(obs)
        score = finite(obs.get("score"), self.last_score if self.last_score is not None else 0.0)
        delta = score - self.last_score if self.last_score is not None else 0.0
        colors = cube_colors(obs)
        if colors is None:
            self.last_score = score
            return False
        changed = ([i for i in range(21)
                    if self.previous is not None and colors[i] != self.previous[i]])
        # The start colour is the most frequent colour at a stable opening.
        if not self.goal_roles and self.previous is None:
            self.goal_roles[colors[0]] = "S"
        if self.previous is not None and len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_score = score - delta
                self.flash_last_change = t
                self.flash_origin = node_value(obs.get("qbert_node"))
                self.ctx.log("board_flash_begin", tick=t, score=score,
                             epoch=self.epoch)
            else:
                self.flash_last_change = t
        if self.flashing:
            quiet = t - self.flash_last_change
            at_apex = node_value(obs.get("qbert_node")) == 0
            if quiet >= 4 and (at_apex or quiet >= 8):
                self._new_epoch(colors, score, t)
                self.previous = colors
                self.last_score = score
                return True
            self.previous = colors
            self.last_score = score
            self.need = [1] * 21
            return False

        level = self.level()
        if self.previous is not None and changed:
            # A local positive score change is the strongest publicly visible
            # evidence that this is Q*bert's landing, not Sam's reversal.
            local = [i for i in changed if i == node_value(obs.get("qbert_node"))]
            if not local and len(changed) == 1:
                local = changed
            if delta > 0 and local and len(changed) <= 3:
                for i in local:
                    self._assign_transition(self.previous[i], colors[i], level)
                    if self.goal_color is None and level == 1:
                        self.goal_color = colors[i]
                        self.goal_roles.setdefault(colors[i], "D")
                        self.ctx.log("destination_learned", tick=t, node=i,
                                     color=colors[i], delta=delta)
                    self.evidence.append({"tick": t, "node": i,
                                          "from": self.previous[i],
                                          "to": colors[i], "delta": delta})
                self.last_progress = t
            elif delta > 0:
                self.last_progress = t
            elif changed:
                # A visible change without reward may be Sam or a reversible
                # cartridge update.  It still invalidates a stale completion.
                self.last_progress = t
        self._recompute(colors)
        # Do not wait forever if a palette transition was missed by the RGB
        # cross-check.  This is a bounded recovery, not a zero-point handoff.
        if not any(x > 0 for x in self.need):
            if self.complete_since is None:
                self.complete_since = t
            elif t - self.complete_since > 28 and not self.flashing:
                self.ctx.log("board_reconciliation_probe", tick=t,
                             epoch=self.epoch)
                self.need = [1] * 21
                self.complete_since = t
        self.previous = colors
        self.last_score = score
        return False


class FailureLedger:
    """Short postmortems for real life losses, keyed by executed edges."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.records = {}       # edge -> {last, count, phase, epoch}
        self.recovery_until = 0
        self.losses = 0
        self.last_loss = None

    @staticmethod
    def phase_bucket(obs):
        timers = obs.get("timers")
        if not isinstance(timers, dict):
            return None
        # This is a signature only.  It is intentionally not interpreted as
        # an enemy position or used as a timer-only veto.
        value = integer(timers.get("purple_ball_state"), -1)
        return value if value >= 0 else None

    def note_loss(self, t, obs, tx, epoch):
        self.losses += 1
        phase = self.phase_bucket(obs)
        edges = []
        if isinstance(tx, dict):
            src, dst = node_value(tx.get("source")), node_value(tx.get("destination"))
            if src is not None and dst is not None:
                edges.append((src, dst))
        if not edges and isinstance(self.last_loss, dict):
            edge = self.last_loss.get("edge")
            if isinstance(edge, tuple):
                edges.append(edge)
        for src, dst in edges:
            for edge, strength in (((src, dst), 1), ((dst, src), 0.6)):
                old = self.records.get(edge, {})
                self.records[edge] = {
                    "last": t, "count": integer(old.get("count"), 0) + strength,
                    "phase": phase, "epoch": epoch,
                }
        if edges:
            self.last_loss = {"edge": edges[-1], "tick": t, "phase": phase}
        # A short central recovery gives the next spawn time to settle, but is
        # deliberately much shorter than the 120/170-tick shelter experiments.
        self.recovery_until = t + 42
        self.ctx.log("edge_postmortem", tick=t, edges=[list(e) for e in edges],
                     phase=phase, losses=self.losses,
                     recovery_until=self.recovery_until)

    def penalty(self, src, dst, t, phase, epoch):
        rec = self.records.get((src, dst))
        if not rec:
            return 0.0
        age = t - integer(rec.get("last"), t)
        if age < 0 or age > 260:
            return 0.0
        value = 10.0 if age < 140 else 4.0
        if integer(rec.get("count"), 0) >= 2:
            value += 8.0
        if rec.get("phase") is not None and phase is not None and rec.get("phase") == phase:
            value += 5.0
        if rec.get("epoch") == epoch:
            value += 2.0
        return value

    def signature(self, t):
        result = []
        for edge, rec in self.records.items():
            if t - integer(rec.get("last"), t) <= 260:
                result.append([edge[0], edge[1], integer(rec.get("count"), 0),
                               "same_phase" if rec.get("phase") is not None else "unknown_phase"])
        return result


class HazardMemory:
    """The parent spatial radar, kept unchanged before a postmortem exists."""

    def __init__(self):
        self.by_kind = {}
        self.centers = {}

    def update_centers(self, obs):
        details = obs.get("cube_details")
        if not isinstance(details, list):
            return
        for item in details:
            if isinstance(item, dict):
                n = node_value(item.get("node"))
                if n is not None:
                    self.centers[n] = (finite(item.get("x")) + 10.0,
                                       finite(item.get("y")) - 7.0)

    def update(self, obs, t):
        self.update_centers(obs)
        hazards = obs.get("hazards", [])
        if isinstance(hazards, list):
            for item in hazards:
                if not isinstance(item, dict) or not item.get("lethal"):
                    continue
                kind = str(item.get("kind", "hazard"))
                n = node_value(item.get("estimated_node", item.get("node")))
                if n is None:
                    continue
                old = self.by_kind.get(kind)
                dx = finite(item.get("dx"))
                dy = finite(item.get("dy"))
                # Preserve a reported motion vector when the decoder emits a
                # zero vector for an in-between animation frame.
                if old and abs(dx) < 0.01 and abs(dy) < 0.01:
                    dx, dy = old.get("dx", 0.0), old.get("dy", 0.0)
                self.by_kind[kind] = {
                    "kind": kind, "node": n, "tick": t,
                    "x": finite(item.get("x")) + finite(item.get("w"), 7.0) / 2.0,
                    "y": finite(item.get("y")) + finite(item.get("h"), 8.0) / 2.0,
                    "dx": dx, "dy": dy,
                    "visible": bool(item.get("visible", True)),
                }
        # Match the parent's short evidence horizon.  The new mechanism is
        # the postmortem ledger, not a broader generic radar shadow.
        for kind in list(self.by_kind):
            if t - integer(self.by_kind[kind].get("tick"), t) > 4:
                del self.by_kind[kind]

    def records(self, t):
        return [v for v in self.by_kind.values()
                if t - integer(v.get("tick"), t) <= 4]

    def risk(self, destination, origin, t):
        value = 0.0
        for h in self.records(t):
            n = h.get("node")
            if n is None:
                continue
            if destination in self.centers:
                x, y = self.centers[destination]
                d = math.hypot(x - h.get("x", x), y - h.get("y", y))
                value += 5.0 if d < 17 else 1.2 if d < 30 else 0.0
            if n == destination:
                value += 3.0
            kind = h.get("kind")
            if kind in ("purple_ball", "red_ball"):
                children = [v for a, v in GRAPH[n].items()
                            if a in ("DOWN", "RIGHT")]
                if destination in children:
                    value += 1.5
            elif kind == "coily":
                nxt = list(GRAPH[n].values())
                toward = min((DIST[x][origin] for x in nxt), default=99)
                if destination in nxt and DIST[destination][origin] == toward:
                    value += 2.0
                if DIST[n][destination] == 1:
                    value += 0.4
        return value

    def current_risk(self, node, t):
        return self.risk(node, node, t)

    def signature(self, t):
        return [[h.get("kind"), h.get("node"),
                 "recent" if t - integer(h.get("tick"), t) > 2 else "fresh"]
                for h in sorted(self.records(t), key=lambda x: str(x.get("kind")))]


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.board = Board(ctx)
        self.ledger = FailureLedger(ctx)
        self.hazards = HazardMemory()
        self.lives = None
        self.life_epoch = 0
        self.policy_version = 0
        self.policy = {"route": "nearest", "risk": "evidence_first"}
        self.recent_nodes = deque(maxlen=18)
        self.recent_edges = deque(maxlen=18)
        self.tx = None
        self.last_edge = None
        self.neutral_until = 0
        self.wait_since = None
        self.advice = None
        self.advice_meta = None
        self.jev_future = None
        self.last_jev_tick = -999
        self.astra_future = None
        self.astra_session = None
        self.astra_calls = 0
        self.last_astra_tick = -999
        self.astra_reason = "initial policy audit"
        self.last_log = -999

    def update(self, obs):
        t = current_tick(obs)
        old_lives = self.lives
        new_lives = integer(obs.get("lives"), old_lives if old_lives is not None else 4)
        epoch_changed = self.board.update(obs)
        self.hazards.update(obs, t)
        lost = old_lives is not None and new_lives < old_lives
        if lost:
            # Capture the active edge before clearing it.  This is the key
            # difference from the parent, which erased its pulse on a loss.
            self.ledger.note_loss(t, obs, self.tx or self.last_edge, self.board.epoch)
            self.life_epoch += 1
            self.tx = None
            self.advice = None
            self.advice_meta = None
            self.recent_nodes.clear()
            self.recent_edges.clear()
            self.neutral_until = t + 3
            self.wait_since = None
            self.astra_reason = "life loss: revise route after edge postmortem"
            self.ctx.log("life_epoch", tick=t, lives=new_lives,
                         board_epoch=self.board.epoch)
        if epoch_changed:
            self.tx = None
            self.advice = None
            self.advice_meta = None
            self.neutral_until = max(self.neutral_until, t + 2)
            self.astra_reason = "new palette epoch: review coverage demand"
        self.lives = new_lives

    def legal_edges(self, obs, at=None, allowed=None):
        n = node_value(obs.get("qbert_node")) if at is None else at
        if n is None or n not in GRAPH:
            return []
        supplied = set(safe_names(obs, "safe_moves") if allowed is None else allowed)
        geometry = GRAPH[n]
        return [(a, geometry[a]) for a in ACTION_ORDER
                if a in supplied and a in geometry]

    def current_phase(self, obs):
        return FailureLedger.phase_bucket(obs)

    def candidate_edges(self, obs, at=None, safe_override=None):
        t = current_tick(obs)
        origin = node_value(obs.get("qbert_node")) if at is None else at
        if origin is None:
            return []
        edges = self.legal_edges(obs, origin, safe_override)
        targets = [i for i, amount in enumerate(self.board.need) if amount > 0]
        if not targets:
            return []
        phase = self.current_phase(obs)
        visits = Counter(self.recent_nodes)
        edge_visits = Counter(self.recent_edges)
        postmortem_mode = bool(self.ledger.records)
        recovery = postmortem_mode and t < self.ledger.recovery_until
        candidates = []
        for action, destination in edges:
            useful = self.board.need[destination] > 0
            distance = min(DIST[destination][target] for target in targets)
            danger = self.hazards.risk(destination, origin, t)
            outgoing = GRAPH[destination]
            exits = 0
            for next_action, nxt in outgoing.items():
                if self.hazards.risk(nxt, destination, t) < 2.0:
                    exits += 1
            edge_penalty = self.ledger.penalty(origin, destination, t, phase,
                                                self.board.epoch)
            reverse = bool(self.last_edge and
                           self.last_edge.get("source") == destination and
                           self.last_edge.get("destination") == origin)
            revisit = visits[destination] + edge_visits[(origin, destination)]
            # This retains the parent's reliable nearest-useful opening when
            # there is no postmortem evidence, while exposing a real cost for
            # edges that actually consumed a life.
            cost = 2.0 * distance + (0.0 if useful else 0.5)
            cost += 12.0 * danger
            cost += visits[destination] * 0.12 - exits * 0.12
            approaching_coily = any(h.get("kind") == "coily"
                                    and h.get("node") is not None
                                    and DIST[destination][h["node"]] <= 2
                                    for h in self.hazards.records(t))
            if approaching_coily and exits == 0:
                cost += 14.0
            # Before the first actual postmortem, keep the parent exactly:
            # later route-policy and recovery adjustments are experimental and
            # must not perturb the reliable opening.
            if postmortem_mode:
                if reverse:
                    cost += 1.8
                if recovery:
                    if ROW[destination] >= 5:
                        cost += 5.0
                    if exits < 2:
                        cost += 2.0
                    if action in ("DOWN", "RIGHT"):
                        cost += 0.6
                if self.policy.get("route") == "frontier" and useful:
                    cost -= 0.18 * ROW[destination]
                elif self.policy.get("route") == "center" and useful:
                    cost += 0.16 * ROW[destination]
                if self.policy.get("risk") == "conservative":
                    cost += 0.4 * max(0.0, danger)
            cost += edge_penalty
            candidates.append({
                "action": action, "destination": destination,
                "cost": cost, "danger": danger, "useful": useful,
                "distance": distance, "exits": exits, "reverse": reverse,
                "postmortem": edge_penalty > 0.0,
                "recovery": recovery,
            })
        candidates.sort(key=lambda c: (c["cost"], ACTION_ORDER.index(c["action"])))

        # A one-tick deliberate wait is allowed only when a visible spatial
        # threat makes every exit worse and the current cube is less exposed.
        current_danger = self.hazards.current_risk(origin, t)
        visible_descender = any(h.get("kind") in ("purple_ball", "red_ball")
                                for h in self.hazards.records(t))
        if (candidates and visible_descender and self.wait_since is not None
                and t - self.wait_since < 14
                and current_danger + 0.35 < candidates[0]["danger"]):
            candidates.insert(0, {
                "action": "NOOP", "destination": origin,
                "cost": candidates[0]["cost"] - 1.0,
                "danger": current_danger, "useful": False, "distance": 0,
                "exits": len(GRAPH[origin]), "reverse": False,
                "postmortem": False, "recovery": recovery, "wait": True,
            })
        return candidates

    def _descriptions(self, candidates):
        result = []
        for i, c in enumerate(candidates):
            if c["action"] == "NOOP":
                role = "bounded_wait"
            elif c["useful"]:
                role = "useful_landing"
            else:
                role = "transit"
            danger = ("occupied" if c["danger"] >= 5 else
                      "near" if c["danger"] >= 1 else "clear")
            result.append({
                "option": c["action"], "role": role, "landing": danger,
                "escape_space": ("many" if c["exits"] >= 2 else
                                  "one" if c["exits"] == 1 else "none"),
                "route": ("adjacent_target" if c["distance"] == 0 else
                           "one_step_to_target" if c["distance"] == 1 else "farther"),
                "postmortem": "avoid_after_recent_loss" if c["postmortem"] else "no_loss_evidence",
                "recovery": bool(c.get("recovery")),
                "code_rank": i,
            })
        return result

    def launch_jev(self, obs, candidates, projected=False):
        if self.jev_future is not None or not candidates or self.ctx.done:
            return
        t = current_tick(obs)
        if t - self.last_jev_tick < 6:
            return
        try:
            remaining = self.ctx.budget.remaining()
        except Exception:
            return
        if remaining.get("system1_calls", 0) < 2:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        origin = node_value(obs.get("qbert_node"))
        descriptions = self._descriptions(candidates)
        options = {item["option"]: item for item in descriptions}
        options["unknown"] = {"role": "no_supported_choice",
                               "not_for": "do not invent an edge"}
        instruction = (
            "Judge the supplied Q*bert alternatives, not the graph arithmetic. "
            "Never select an edge absent from candidates. Prefer clear landings "
            "with room to escape, then useful progress. A recent postmortem is "
            "evidence against an option, but do not invent a hazard from a bare "
            "timer. Choose unknown when the facts do not distinguish options."
        )
        state = {
            "game": "Atari 2600 Q*bert, six-row 21-cube Atari port",
            "rules": "Survival and completion outrank 100/300-point bonuses; NOOP is only a deliberate bounded wait.",
            "position": {"node": origin, "motion": qmotion(obs).get("phase", "unknown")},
            "board": {"inferred_epoch": self.board.epoch,
                      "inferred_clears": self.board.clears,
                      "unfinished_cubes": sum(x > 0 for x in self.board.need),
                      "level_cycle": self.board.level()},
            "recovery": {"active": current_tick(obs) < self.ledger.recovery_until,
                         "loss_count": self.ledger.losses},
            "hazards": self.hazards.signature(t),
            "recent_failed_edges": self.ledger.signature(t),
            "candidates": descriptions,
            "fallback": candidates[0]["action"],
            "projected_decision": projected,
        }
        questions = {
            "route_a": Choice(instruction, options),
            "route_b": Choice(instruction, dict(reversed(list(options.items())))),
            "immediate": Noul(
                "Do the supplied facts establish immediate lethal risk on the fallback option? "
                "A concrete occupied or near collision is yes; a distant or timer-only fact is not.",
                {"true": "The fallback landing is occupied or a concrete near collision is stated.",
                 "false": "The fallback is clear, or the evidence is only distant/uncertain."}),
        }
        meta = {
            "tick": t, "node": origin, "life": self.life_epoch,
            "epoch": self.board.epoch, "policy": self.policy_version,
            "hazards": self.hazards.signature(t),
            "failed": self.ledger.signature(t),
            "projected": projected,
            "fallback": candidates[0]["action"],
        }
        self.last_jev_tick = t

        def work():
            return self.ctx.decide(state, questions, timeout_s=1.0)

        self.jev_future = self.ctx.spawn(work)
        self.advice_meta = meta

    def collect_jev(self, obs):
        if self.jev_future is None or not self.jev_future.done():
            return
        future = self.jev_future
        meta = self.advice_meta
        self.jev_future = None
        self.advice_meta = None
        try:
            answers = future.result()
            if not isinstance(answers, dict) or not isinstance(meta, dict):
                return
            a = answer_choice(answers.get("route_a"))
            b = answer_choice(answers.get("route_b"))
            valid = a == b and a in ALL_ACTIONS
            confidence = min(answer_probability(answers.get("route_a"), a),
                             answer_probability(answers.get("route_b"), b)) if valid else 0.0
            immediate = finite(getattr(answers.get("immediate"), "value", 0.0), 0.0)
            if valid and confidence >= 0.60:
                self.advice = dict(meta, action=a, confidence=confidence,
                                   veto=immediate >= 0.84)
        except Exception as exc:
            self.ctx.log("jev_fallback", error=type(exc).__name__)

    def consume_advice(self, obs, candidates):
        fallback = candidates[0]["action"]
        meta = self.advice
        self.advice = None
        if not meta:
            return fallback
        t = current_tick(obs)
        fresh = (meta.get("life") == self.life_epoch
                 and meta.get("epoch") == self.board.epoch
                 and meta.get("policy") == self.policy_version
                 and meta.get("node") == node_value(obs.get("qbert_node"))
                 and 0 <= t - integer(meta.get("tick"), t) <= 9
                 and meta.get("hazards") == self.hazards.signature(t)
                 and meta.get("failed") == self.ledger.signature(t))
        if not fresh:
            return fallback
        by_action = {c["action"]: c for c in candidates}
        chosen = by_action.get(meta.get("action"))
        best = candidates[0]
        if chosen is not None and chosen["danger"] < 5.0:
            # Jev can break a close tie, but not erase a code postmortem or
            # turn a clearly riskier edge into an action.
            allowed = (chosen["cost"] <= best["cost"] + 3.0
                       and chosen["danger"] <= best["danger"] + 0.45
                       and not (chosen["postmortem"] and not best["postmortem"]))
            if allowed:
                if chosen["action"] != fallback:
                    self.ctx.log("jev_changed_edge", tick=t, old=fallback,
                                 new=chosen["action"], confidence=meta["confidence"])
                return chosen["action"]
        if meta.get("veto") and len(candidates) > 1:
            alternative = next((c for c in candidates[1:]
                                if c["danger"] <= best["danger"] + 0.15
                                and not c["postmortem"]), None)
            if alternative is not None:
                self.ctx.log("jev_veto", tick=t, old=fallback,
                             new=alternative["action"])
                return alternative["action"]
        return fallback

    def request_astra(self, obs, reason):
        if self.astra_future is not None or self.astra_calls >= 3 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) < 1:
                return
        except Exception:
            return
        t = current_tick(obs)
        if self.astra_calls and t - self.last_astra_tick < 300:
            return
        self.astra_calls += 1
        self.last_astra_tick = t
        state = {
            "reason": reason, "score": finite(obs.get("score")),
            "lives": self.lives, "losses": self.ledger.losses,
            "inferred_clears": self.board.clears,
            "current_epoch": self.board.epoch,
            "recent_failed_edges": self.ledger.signature(t),
            "palette_evidence": list(self.board.evidence),
            "current_policy": self.policy,
        }
        prompt = (
            "You are the slow policy author for a real-time Atari 2600 Q*bert "
            "controller. Code owns the 21-node graph, colour-cycle arithmetic, "
            "hazard observations, one-hop motor, and a finite postmortem ledger. "
            "A postmortem records an actual life loss on an executed edge and "
            "temporarily outranks coverage; it is not a seed route. Jev receives "
            "only code-generated candidates and two reordered Choices plus a Noul. "
            "Choose a durable tie policy: route nearest, frontier, or center; "
            "choose risk evidence_first or conservative. These choices only alter "
            "tie costs and wording. Never emit joystick actions, FIRE, guessed disc "
            "exits, code, or a long explanation. Return ONLY JSON under 30 tokens: "
            '{"route":"nearest","risk":"evidence_first"}. Evidence=' +
            json.dumps(json_safe(state))
        )
        session = self.astra_session

        def work():
            return self.ctx.think(prompt, session_id=session, timeout_s=105.0)

        self.astra_future = self.ctx.spawn(work)
        self.astra_reason = ""
        self.ctx.log("astra_requested", tick=t, reason=reason)

    def collect_astra(self, obs):
        if self.astra_future is None or not self.astra_future.done():
            return
        future = self.astra_future
        self.astra_future = None
        try:
            result = future.result()
            self.astra_session = getattr(result, "session_id", None)
            data = object_from_text(getattr(result, "text", ""))
            old = dict(self.policy)
            if data.get("route") in ("nearest", "frontier", "center"):
                self.policy["route"] = data["route"]
            if data.get("risk") in ("evidence_first", "conservative"):
                self.policy["risk"] = data["risk"]
            if old != self.policy:
                self.policy_version += 1
                self.advice = None
                self.ctx.log("astra_policy_changed", tick=current_tick(obs),
                             old=old, new=dict(self.policy))
            else:
                self.ctx.log("astra_policy_unchanged", tick=current_tick(obs),
                             parsed=bool(data))
        except Exception as exc:
            self.ctx.log("astra_fallback", error=type(exc).__name__)

    def _projected_options(self, obs, destination):
        # During an animation, ask about the node expected at its end.  The
        # answer is only consumable if that node really becomes current.
        occupied = {h.get("node") for h in self.hazards.records(current_tick(obs))}
        legal = [a for a, n in GRAPH.get(destination, {}).items()
                 if n not in occupied]
        fake = dict(obs)
        fake["qbert_node"] = destination
        fake["safe_moves"] = legal
        fake["legal_moves"] = list(GRAPH.get(destination, {}))
        return fake

    def choose_action(self, obs):
        t = current_tick(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)

        if self.board.flashing:
            self.tx = None
            return "NOOP"
        m = qmotion(obs)
        if m.get("falling") or m.get("phase") == "not_visible":
            self.tx = None
            return "NOOP"

        n = node_value(obs.get("qbert_node"))
        if self.tx is not None:
            tx = self.tx
            age = t - integer(tx.get("tick"), t)
            if n is not None and n == tx.get("destination") and is_standing(obs) and age >= 2:
                self.recent_nodes.append(n)
                self.recent_edges.append((tx.get("source"), n))
                self.last_edge = dict(tx, landed=t)
                self.ctx.log("edge_landed", tick=t, source=tx.get("source"),
                             destination=n, expected=tx.get("destination"), age=age)
                self.tx = None
                self.neutral_until = max(self.neutral_until, t + 1)
            elif age > 20 and is_standing(obs):
                self.ctx.log("edge_resync", tick=t, expected=tx.get("destination"),
                             observed=n)
                self.tx = None
                self.neutral_until = max(self.neutral_until, t + 1)
            else:
                # A direction pulse is short; holding it through the whole hop
                # can launch an unintended second edge after landing.
                if not bool(obs.get("qbert_in_hop")) and age <= 2:
                    return tx.get("action") if tx.get("action") in safe_names(obs, "safe_moves") else "NOOP"
                return "NOOP"

        if n is None or not is_standing(obs) or t < self.neutral_until:
            return "NOOP"
        candidates = self.candidate_edges(obs)
        if not candidates:
            return "NOOP"
        if candidates[0]["action"] == "NOOP":
            if self.wait_since is None:
                self.wait_since = t
                self.ctx.log("bounded_spatial_wait", tick=t, node=n)
            return "NOOP"
        self.wait_since = None
        selected = self.consume_advice(obs, candidates)
        by_action = {c["action"]: c for c in candidates}
        chosen = by_action.get(selected, candidates[0])
        if chosen["action"] not in safe_names(obs, "safe_moves"):
            chosen = candidates[0]
        if chosen["action"] == "NOOP":
            return "NOOP"

        destination = chosen["destination"]
        self.tx = {"source": n, "destination": destination,
                   "action": chosen["action"], "tick": t}
        self.last_edge = dict(self.tx)
        self.ctx.log("edge_commit", tick=t, source=n, destination=destination,
                     action=chosen["action"], danger=chosen["danger"],
                     postmortem=chosen["postmortem"], recovery=chosen["recovery"])

        # Pipeline the next decision.  This makes Jev useful without putting it
        # on the game clock; all facts are recomputed at answer consumption.
        self.launch_jev(self._projected_options(obs, destination),
                        self.candidate_edges(self._projected_options(obs, destination),
                                             at=destination,
                                             safe_override=self._projected_options(obs, destination).get("safe_moves", [])),
                        projected=True)
        return chosen["action"]

    def run(self):
        obs = self.ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.update(obs)
        self.request_astra(obs, self.astra_reason)
        while not self.ctx.done and not self.game.done:
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.update(obs)
            # If no projected request exists, a standing state can seed one.
            action = self.choose_action(obs)
            if action not in ALL_ACTIONS:
                action = "NOOP"
            self.ctx.act(action)
            if self.astra_reason:
                self.request_astra(obs, self.astra_reason)
            elif self.board.last_progress and current_tick(obs) - self.board.last_progress > 180:
                self.request_astra(obs, "scoring stall: inspect palette and route policy")
            if current_tick(obs) - self.last_log >= 30:
                self.ctx.log("controller_state", tick=current_tick(obs),
                             score=finite(obs.get("score")), lives=self.lives,
                             epoch=self.board.epoch, inferred_clears=self.board.clears,
                             unfinished=sum(x > 0 for x in self.board.need),
                             node=node_value(obs.get("qbert_node")), action=action,
                             losses=self.ledger.losses,
                             failed_edges=self.ledger.signature(current_tick(obs)))
                self.last_log = current_tick(obs)
            ctx_wait = self.ctx.wait_for_tick
            ctx_wait()
        self.ctx.log("candidate_summary", score=getattr(self.game, "score", None),
                     inferred_clears=self.board.clears,
                     losses=self.ledger.losses,
                     failed_edges=self.ledger.signature(current_tick(obs)),
                     policy=dict(self.policy))


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        Controller(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
