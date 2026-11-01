"""0018: leaf-last coverage with landing-aligned Jev control.

The code owns exact board bookkeeping, legality, and a route guard that keeps
one-exit bottom corners out of the active route until the rest of the board is
cleared. Jev judges the *next* departure while Q*bert is in the current hop,
then the motor checks that advice against the observed landing. Astra chooses
only a durable coverage ordering; no model is on the tick clock.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
TIE_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")
GRAPH = {n: {} for n in range(21)}
ROW = {}
COL = {}
for r in range(6):
    for c in range(r + 1):
        n = r * (r + 1) // 2 + c
        ROW[n], COL[n] = r, c
        if r:
            if c < r:
                GRAPH[n]["UP"] = n - r
            if c:
                GRAPH[n]["LEFT"] = n - r - 1
        if r < 5:
            first = (r + 1) * (r + 2) // 2
            GRAPH[n]["DOWN"] = first + c
            GRAPH[n]["RIGHT"] = first + c + 1

LEAVES = frozenset((15, 20))


def _distances():
    all_dist = []
    for root in range(21):
        ds = [99] * 21
        ds[root] = 0
        todo = deque([root])
        while todo:
            n = todo.popleft()
            for nxt in GRAPH[n].values():
                if ds[nxt] == 99:
                    ds[nxt] = ds[n] + 1
                    todo.append(nxt)
        all_dist.append(ds)
    return all_dist


DIST = _distances()


def finite(value, default=0.0):
    try:
        n = float(value)
        return n if math.isfinite(n) else default
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
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return n if 0 <= n < 21 else None


def tick_of(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)))


def colors_of(obs):
    values = obs.get("cube_colors")
    if not isinstance(values, list) or len(values) != 21:
        return None
    if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
        return None
    return tuple(values)


def safe_names(value):
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]


def qmotion(obs):
    value = obs.get("qbert_motion")
    return value if isinstance(value, dict) else {"phase": str(value or "unknown")}


def is_standing(obs):
    n = node_of(obs.get("qbert_node"))
    phase = str(qmotion(obs).get("phase", "unknown")).lower()
    return (n is not None and not bool(obs.get("qbert_in_hop"))
            and phase not in ("hopping", "falling", "not_visible")
            and not bool(qmotion(obs).get("falling")))


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)):
        return [json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def parse_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, char in enumerate(text[:12000]):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[i:])
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def answer_key(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value if isinstance(value, str) else None


def answer_prob(answer, key):
    probs = getattr(answer, "probabilities", None) if answer is not None else None
    if isinstance(probs, dict):
        return max(0.0, min(1.0, finite(probs.get(key), 0.0)))
    return 0.0


class BoardClock:
    """Infer a destination from scored landings; rebase only on scored flashes.

    The adaptor's ``needed_landings`` is useful for diagnostics, but live traces
    show it alternating between all-zero and all-one while the cartridge
    globally flashes a completed board. It is therefore not used as a movement
    mask during that transition.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous_colors = None
        self.previous_score = None
        self.goal = None
        self.need = [1] * 21
        self.last_progress = 0
        self.epoch = 0
        self.clears = 0
        self.flashing = False
        self.flash_last_change = -1
        self.flash_score = 0.0
        self.complete_since = None
        self.evidence = deque(maxlen=8)

    def update(self, obs):
        now = tick_of(obs)
        current = colors_of(obs)
        score = finite(obs.get("score"), self.previous_score or 0.0)
        delta = score - self.previous_score if self.previous_score is not None else 0.0
        changed = []
        if current is not None and self.previous_colors is not None:
            changed = [i for i in range(21)
                       if current[i] != self.previous_colors[i]]

        if current is not None and self.previous_colors is not None and len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_score = self.previous_score if self.previous_score is not None else score - delta
                self.ctx.log("leaf_flash_started", tick=now, changed=len(changed), score=score)
            self.flash_last_change = now

        if self.flashing:
            quiet = now - self.flash_last_change
            returned_to_top = node_of(obs.get("qbert_node")) == 0
            if quiet >= 4 and (returned_to_top or quiet >= 7):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.flashing = False
                self.goal = None
                self.need = [1] * 21
                self.complete_since = None
                self.last_progress = now
                self.evidence.clear()
                self.ctx.log("leaf_board_rebased", tick=now, epoch=self.epoch,
                             clears=self.clears, tally=tally,
                             reported_round=obs.get("round"))
                self.previous_colors = current
                self.previous_score = score
                return True
            self.previous_colors = current
            self.previous_score = score
            return False

        # Learn the current destination from a real local cube score event. A
        # 325-point delta can be one cube plus a 300-point Sam capture.
        if (current is not None and self.previous_colors is not None and delta > 0
                and int(round(delta)) % 100 == 25 and 0 < len(changed) <= 3):
            qn = node_of(obs.get("qbert_node"))
            landed = qn if qn in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                if self.goal is None:
                    self.goal = current[landed]
                    self.ctx.log("leaf_goal_learned", tick=now, epoch=self.epoch,
                                 node=landed, color=self.goal)
                self.last_progress = now
                self.evidence.append({"node": landed, "color": current[landed],
                                      "reward": delta,
                                      "decoder_phase": (obs.get("cube_phase") or [None] * 21)[landed]})
                self.ctx.log("leaf_palette_evidence", tick=now, epoch=self.epoch,
                             node=landed, color=current[landed], need=self.need[landed])

        if self.goal is None:
            # Never adopt the decoder's global flash toggles as cube completion.
            self.need = [1] * 21
        elif current is not None:
            self.need = [0 if color == self.goal else 1 for color in current]

        if not any(self.need):
            if self.complete_since is None:
                self.complete_since = now
            elif now - self.complete_since >= 24:
                # A second pass is a probe of the cartridge's next phase, not a
                # claim of completion; only the next scored global flash rebases.
                self.goal = None
                self.need = [1] * 21
                self.complete_since = None
                self.last_progress = now
                self.ctx.log("leaf_cycle_probe", tick=now, epoch=self.epoch)
        else:
            self.complete_since = None

        if current is not None:
            self.previous_colors = current
        self.previous_score = score
        return False

    @property
    def unfinished(self):
        return [n for n, value in enumerate(self.need) if value > 0]


class ThreatMemory:
    """Short memory of rendered lethal objects; never guesses a hidden node."""

    def __init__(self):
        self.items = {}

    def update(self, obs):
        now = tick_of(obs)
        visible = set()
        hazards = obs.get("hazards")
        if isinstance(hazards, list):
            for item in hazards:
                if not isinstance(item, dict) or not item.get("lethal"):
                    continue
                kind = str(item.get("kind", ""))
                if kind not in ("red_ball", "purple_ball", "coily"):
                    continue
                node = node_of(item.get("estimated_node", item.get("node")))
                if node is None:
                    continue
                self.items[kind] = {"kind": kind, "node": node, "tick": now,
                                    "risk": str(item.get("risk", "unknown")),
                                    "visible": True,
                                    "motion": str(item.get("motion", "unknown"))}
                visible.add(kind)
        for kind in list(self.items):
            item = self.items[kind]
            age = now - item["tick"]
            limit = 9 if kind in ("purple_ball", "red_ball") else 6
            if age > limit:
                del self.items[kind]
            else:
                item["visible"] = kind in visible

    def view(self, origin):
        now_items = []
        for item in self.items.values():
            d = DIST[origin][item["node"]] if origin is not None else 99
            now_items.append({"kind": item["kind"],
                              "visibility": "rendered" if item["visible"] else "brief_memory",
                              "proximity": "same_cube" if d == 0 else
                                           "adjacent" if d == 1 else
                                           "near" if d <= 2 else "far",
                              "motion": "rolling_down" if item["kind"] in ("red_ball", "purple_ball")
                                        else "pursuit"})
        return now_items

    @staticmethod
    def _down_children(nodes):
        return {there for n in nodes for action, there in GRAPH[n].items()
                if action in ("DOWN", "RIGHT")}

    def risk(self, origin, destination, now, horizon=0):
        """A small public-object reachability envelope, not timer-based decoding."""
        total = 0.0
        reasons = []
        if origin is None or destination is None:
            return 10.0, ["unknown_node"]
        for kind, item in self.items.items():
            age = max(0, now - item["tick"])
            hnode = item["node"]
            d = DIST[hnode][destination]
            qd = DIST[hnode][origin]
            if item.get("risk") == "immediate":
                total += 5.0
                reasons.append(kind + "_immediate")
            if kind in ("red_ball", "purple_ball"):
                possible = {hnode}
                frontier = {hnode}
                # A rendered ball can move down either branch before the next
                # landing; brief memory adds at most one more branch layer.
                steps = 1 if age <= 2 else 2
                for _ in range(steps):
                    frontier = self._down_children(frontier)
                    possible.update(frontier)
                if destination in possible:
                    total += 3.8 if item["visible"] else 2.0
                    reasons.append(kind + "_branch_reachable")
                elif d == 1:
                    total += 1.1
                    reasons.append(kind + "_near")
                if qd == 0 and item["visible"]:
                    total += 1.2
                    reasons.append(kind + "_takeoff")
            else:
                # Pursuit movement is uncertain. Treat an adjacent landing as
                # meaningful risk, but do not mark every one-edge Coily
                # descendant occupied; that over-reroutes the coverage path.
                if destination == hnode:
                    total += 5.0
                    reasons.append("coily_on_landing")
                elif d == 1:
                    total += 1.4 if item["visible"] else 0.8
                    reasons.append("coily_adjacent")
                elif DIST[destination][hnode] < DIST[origin][hnode] and qd <= 3:
                    total += 0.6
                    reasons.append("moving_toward_near_coily")
        return total, reasons

    def facts_for(self, origin):
        return self.view(origin)

    def signature(self, now):
        return tuple(sorted((kind, item["node"], item["visible"],
                             min(3, max(0, now - item["tick"])))
                            for kind, item in self.items.items()))


class CoveragePlanner:
    """Receding one-hop coverage with a hard non-corner-before-frontier rule."""

    def __init__(self, book, threats):
        self.book = book
        self.threats = threats
        self.mode = "balanced"
        self.policy_version = 0
        self.recent_nodes = deque(maxlen=22)
        self.last_node = None
        self.target = None
        self.cursor = 0

    def reset(self):
        self.target = None
        self.cursor = 0
        self.recent_nodes.clear()
        self.last_node = None

    def _order_rank(self, n):
        base = (1, 4, 7, 11, 6, 10, 16, 12, 17,
                18, 13, 9, 14, 19, 20, 8, 2, 0, 3, 5, 15)
        if self.mode == "spine":
            return (ROW[n], abs(2 * COL[n] - ROW[n]), base.index(n))
        if self.mode == "zigzag":
            return (ROW[n], COL[n] if ROW[n] % 2 == 0 else ROW[n] - COL[n], base.index(n))
        return (base.index(n),)

    def _targets(self, current):
        remaining = set(self.book.unfinished)
        ordinary = remaining - LEAVES
        # Keep ordinary cubes as target tie winners, but retain the leaves as
        # reachable alternatives when detouring there is materially safer.
        return sorted(remaining, key=lambda n: (n in LEAVES and bool(ordinary),
                                               DIST[current][n], self._order_rank(n)))

    def _escape_count(self, destination, origin, now, horizon):
        exits = 0
        for nxt in GRAPH[destination].values():
            risk, _ = self.threats.risk(destination, nxt, now, horizon=horizon + 5)
            if nxt != origin and risk < 2.5:
                exits += 1
        return exits

    def candidates(self, obs, at_node=None, future=False):
        now = tick_of(obs)
        origin = node_of(obs.get("qbert_node")) if at_node is None else at_node
        if origin is None:
            return []
        safe = None if future else set(safe_names(obs.get("safe_moves")))
        legal = [a for a in TIE_ORDER
                 if a in GRAPH[origin] and (safe is None or a in safe)]
        if not legal:
            return []
        targets = self._targets(origin)
        target = targets[0] if targets else None
        self.target = target
        remaining = set(self.book.unfinished)
        ordinary_left = bool(remaining - LEAVES)
        visit_counts = Counter(self.recent_nodes)
        entries = []
        for action in legal:
            dest = GRAPH[origin][action]
            danger, reasons = self.threats.risk(origin, dest, now, horizon=7 if future else 0)
            escapes = self._escape_count(dest, origin, now, 7 if future else 0)
            distance = min((DIST[dest][candidate] for candidate in targets), default=0)
            useful = self.book.need[dest] > 0
            cost = 1.35 * distance + (-1.45 if useful else 0.55)
            cost += 2.0 * min(danger, 10.0) - 0.20 * min(escapes, 2)
            cost += 0.20 * min(visit_counts[dest], 5)
            if dest in LEAVES and ordinary_left:
                cost += 1.2
            if dest == self.last_node and len(legal) > 1:
                cost += 0.55
            if target is not None:
                cost += 0.12 * DIST[dest][target]
            entries.append({"action": action, "destination": dest,
                            "danger": danger, "reasons": reasons,
                            "escapes": escapes,
                            "escape_kind": "open" if escapes >= 2 else
                                           "one_way" if dest in LEAVES or escapes == 0 else "narrow",
                            "useful": useful, "distance": distance,
                            "cost": cost, "target": target})
        entries.sort(key=lambda item: (item["danger"] >= 7.0, item["cost"],
                                       TIE_ORDER.index(item["action"])))
        return entries

    def note_landing(self, node):
        if node is not None and node != self.last_node:
            self.recent_nodes.append(node)
        self.last_node = node


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.book = BoardClock(ctx)
        self.threats = ThreatMemory()
        self.plan = CoveragePlanner(self.book, self.threats)
        self.lives = None
        self.life_epoch = 0
        self.last_life_lost_count = None
        self.recovery_until = 0
        self.pending = None
        self.neutral_until = 0
        self.advice = None
        self.s1_future = None
        self.s1_meta = None
        self.last_s1_request = -999
        self.s2_future = None
        self.s2_calls = 0
        self.s2_session = None
        self.s2_waiting_after_loss = False
        self.last_astra_tick = -999
        self.last_log = -999
        try:
            self.action_names = {str(x) for x in game.action_spec.names}
        except Exception:
            self.action_names = {"NOOP", "FIRE", *DIRECTIONS}
        self.noop = "NOOP" if "NOOP" in self.action_names else "NOOP"

    def ingest(self, obs):
        now = tick_of(obs)
        old_epoch = self.book.epoch
        self.book.update(obs)
        self.threats.update(obs)
        if self.book.epoch != old_epoch:
            self.plan.reset()
            self.pending = None
            self.advice = None
            self.neutral_until = max(self.neutral_until, now + 2)
            self.ctx.log("leaf_plan_rebased", tick=now, epoch=self.book.epoch,
                         clears=self.book.clears)

        lives = integer(obs.get("lives"), self.lives if self.lives is not None else 4)
        resources = obs.get("resources") if isinstance(obs.get("resources"), dict) else {}
        loss_count = integer(resources.get("life_losses"),
                             self.last_life_lost_count if self.last_life_lost_count is not None else 0)
        lost = ((self.lives is not None and lives < self.lives) or
                (self.last_life_lost_count is not None and loss_count > self.last_life_lost_count))
        if lost:
            self.life_epoch += 1
            self.pending = None
            self.advice = None
            self.recovery_until = max(self.recovery_until, now + 3)
            self.neutral_until = max(self.neutral_until, now + 4)
            self.plan.target = None
            self.s2_waiting_after_loss = True
            self.ctx.log("leaf_life_reset", tick=now, lives=lives,
                         loss_count=loss_count, epoch=self.book.epoch)
        self.lives = lives
        self.last_life_lost_count = loss_count

    def _candidate_state(self, obs, expected, candidates):
        current = node_of(obs.get("qbert_node"))
        described = []
        for c in candidates:
            described.append({
                "key": c["action"],
                "destination": c["destination"],
                "coverage": "unfinished" if c["useful"] else "transit",
                "hazard": "immediate" if c["danger"] >= 7 else
                          "possible" if c["danger"] >= 2.5 else
                          "low" if c["danger"] >= 0.8 else "no_supported_contact",
                "escape": c["escape_kind"],
                "route": "closer_to_target" if self.plan.target is not None and
                         c["distance"] < DIST[expected][self.plan.target] else "not_closer",
                "code_rank": "first" if c is candidates[0] else "other",
            })
        return json_safe({
            "game": "Atari 2600 Q*bert, six-row 21-cube graph; one hop then reobserve",
            "question_scope": "Choose only among the listed joystick directions. Code owns exact graph geometry, score-bearing palette bookkeeping, legality, and the one-way-corner rule.",
            "position": {"observed_node": current,
                         "forecast_landing_node": expected,
                         "forecast_target_node": self.plan.target,
                         "current_motion": str(qmotion(obs).get("phase", "unknown"))},
            "progress": {"board_epoch": self.book.epoch,
                         "inferred_scored_clears": self.book.clears,
                         "unfinished_count": len(self.book.unfinished),
                         "corner_rule": "bottom leaves 15 and 20 have one graph exit; code adds a soft cost while ordinary demand remains"},
            "threats": self.threats.facts_for(expected),
            "next_hop_candidates": described,
            "proposed_next_action": candidates[0]["action"] if candidates else "none",
        })

    def request_jev(self, obs, expected):
        if self.s1_future is not None or self.ctx.done or getattr(self.ctx, "system1", True) is None:
            return
        now = tick_of(obs)
        if now - self.last_s1_request < 2:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 0:
                return
        except Exception:
            return
        future_candidates = self.plan.candidates(obs, at_node=expected, future=True)
        if not future_candidates:
            return
        keys = [c["action"] for c in future_candidates]
        descriptions = {}
        for c in future_candidates:
            descriptions[c["action"]] = {
                "what": "Take this supplied legal graph edge for one hop after the current landing.",
                "coverage": "unfinished cube" if c["useful"] else "transit cube",
                "hazard": "immediate risk" if c["danger"] >= 7 else
                          "possible contact" if c["danger"] >= 2.5 else
                          "low exposure" if c["danger"] >= 0.8 else "no supported contact",
                "escape": c["escape_kind"],
                "route_progress": "closer_to_forecast_target" if
                                  self.plan.target is not None and
                                  c["distance"] < DIST[expected][self.plan.target]
                                  else "not_closer_to_forecast_target",
                "not_for": "FIRE, edge-fall actions, or inventing a direction",
            }
        instruction = (
            "Choose the safest useful next hop from `next_hop_candidates`. The `hazard` and `escape` fields are code-computed facts. "
            "Avoid immediate risk first, then prefer low exposure and useful coverage. Prefer a candidate marked `closer_to_forecast_target`; "
            "do not select a `not_closer_to_forecast_target` route unless it is the listed code fallback. "
            "A one_way landing is a small disadvantage while ordinary demand remains, not a ban; prefer it when its safety or coverage is better. "
            "Do not calculate distances or invent actions."
        )
        questions = {
            "choice_a": Choice(instruction, descriptions),
            "choice_b": Choice(instruction, dict(reversed(list(descriptions.items())))),
            "veto_proposed": Noul(
                "Do the facts for `proposed_next_action` establish supported immediate lethal contact before or at its next landing? "
                "Possible or distant exposure alone is not immediate contact.",
                {"true": "A listed immediate-risk or occupied-contact fact supports likely lethal contact.",
                 "false": "No immediate lethal contact is supported, or only uncertain/distant exposure is listed."}),
        }
        state = self._candidate_state(obs, expected, future_candidates)
        self.s1_meta = {"tick": now, "expected": expected,
                        "epoch": self.book.epoch, "life": self.life_epoch,
                        "policy": self.plan.policy_version,
                        "keys": keys, "proposed": future_candidates[0]["action"]}
        self.last_s1_request = now

        def work():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.9),
                        "failure": None}
            except Exception as exc:
                return {"answers": None, "failure": type(exc).__name__}

        try:
            self.s1_future = self.ctx.spawn(work)
        except Exception as exc:
            self.s1_future = None
            self.s1_meta = None
            self.ctx.log("next_hop_jev_spawn_failed", error=type(exc).__name__)

    def collect_jev(self, obs):
        now = tick_of(obs)
        if self.s1_future is not None and self.s1_future.done():
            future, meta = self.s1_future, self.s1_meta
            self.s1_future, self.s1_meta = None, None
            try:
                result = future.result()
                answers = result.get("answers") if isinstance(result, dict) else None
                if not isinstance(answers, dict) or not isinstance(meta, dict):
                    self.ctx.log("next_hop_jev_empty", tick=now)
                else:
                    a, b = answer_key(answers.get("choice_a")), answer_key(answers.get("choice_b"))
                    agrees = a == b and a in meta.get("keys", [])
                    confidence = min(answer_prob(answers.get("choice_a"), a),
                                     answer_prob(answers.get("choice_b"), b)) if agrees else 0.0
                    veto = finite(getattr(answers.get("veto_proposed"), "value", 0.0), 0.0)
                    age = now - integer(meta.get("tick"), now)
                    if age <= 14 and age >= 0:
                        self.advice = {**meta,
                                       "action": a if agrees and confidence >= 0.58 else None,
                                       "confidence": confidence,
                                       "veto": veto}
                        self.ctx.log("next_hop_jev_ready", tick=now,
                                     expected=meta.get("expected"), action=self.advice["action"],
                                     confidence=confidence, veto=veto, age=age)
                    else:
                        self.ctx.log("next_hop_jev_late", tick=now, age=age)
            except Exception as exc:
                self.ctx.log("next_hop_jev_failed", tick=now, error=type(exc).__name__)
        if self.advice is not None:
            age = now - integer(self.advice.get("tick"), now)
            if age < 0 or age > 14:
                self.advice = None

    def choose(self, obs, candidates):
        if not candidates:
            return None
        fallback = candidates[0]
        # An answer is for exactly the next landing transaction. Never keep it
        # around to accidentally reuse it on a later visit to the same node.
        advice, self.advice = self.advice, None
        if not isinstance(advice, dict):
            return fallback
        if (advice.get("expected") != node_of(obs.get("qbert_node"))
                or advice.get("epoch") != self.book.epoch
                or advice.get("life") != self.life_epoch
                or advice.get("policy") != self.plan.policy_version
                or tick_of(obs) - integer(advice.get("tick"), tick_of(obs)) < 0
                or tick_of(obs) - integer(advice.get("tick"), tick_of(obs)) > 14):
            return fallback
        current = {c["action"]: c for c in candidates}
        action = advice.get("action")
        chosen = current.get(action)
        if chosen is not None:
            # Jev is a tactical tie-breaker, not a route author. It may only
            # choose a move that advances the existing target's shortest-path
            # potential; this blocks confident two-cube oscillations.
            target = self.plan.target
            current_node = node_of(obs.get("qbert_node"))
            advances = (target is not None and current_node is not None
                        and DIST[chosen["destination"]][target] < DIST[current_node][target])
            if (advances
                    and chosen["danger"] <= fallback["danger"] + 0.35
                    and chosen["cost"] <= fallback["cost"] + 1.0):
                if action != fallback["action"]:
                    self.ctx.log("jev_next_hop_changed_action", tick=tick_of(obs),
                                 old=fallback["action"], new=action,
                                 confidence=advice.get("confidence"), target=target)
                return chosen
        if (advice.get("proposed") == fallback["action"]
                and finite(advice.get("veto"), 0.0) >= 0.86):
            target = self.plan.target
            current_node = node_of(obs.get("qbert_node"))
            alternatives = [c for c in candidates[1:]
                            if c["danger"] < fallback["danger"]
                            and target is not None and current_node is not None
                            and DIST[c["destination"]][target] < DIST[current_node][target]]
            if alternatives:
                self.ctx.log("jev_next_hop_veto", tick=tick_of(obs),
                             old=fallback["action"], new=alternatives[0]["action"])
                return alternatives[0]
        return fallback

    def request_astra(self, obs, reason):
        if self.s2_future is not None or self.s2_calls >= 2 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            return
        now = tick_of(obs)
        node = node_of(obs.get("qbert_node"))
        state = json_safe({
            "reason": reason,
            "tick": now,
            "score": finite(obs.get("score")),
            "lives": self.lives,
            "inferred_clears": self.book.clears,
            "board_epoch": self.book.epoch,
            "position": node,
            "unfinished": self.book.unfinished[:21],
            "recent_landing_evidence": list(self.book.evidence),
            "public_threats": self.threats.facts_for(node),
            "fixed_rule": "21-node graph; destination phase inferred from local score-bearing palette changes; global clear tally rebases; nodes 15 and 20 have one graph exit and receive a soft cost while ordinary demand remains",
            "current_route_mode": self.plan.mode,
        })
        prompt = (
            "You are Astra, the slow strategic route adviser for a real-time Atari 2600 Q*bert controller. "
            "The world continues while you think. Code owns the complete 21-cube graph, score-bearing local palette bookkeeping, "
            "enemy legality, short-horizon hazard facts, and a soft preference to defer one-exit leaves 15/20 while ordinary cubes remain. "
            "Jev will judge a single next-hop choice while Q*bert is in flight. "
            "Choose one durable order for ties between equally safe reachable unfinished cubes: "
            "`balanced` = nearest-first with the public 21-node sweep as a tie-breaker; `spine` = favor the central band; "
            "`zigzag` = alternate the side swept on successive rows. This changes only target tie-breaking, "
            "never risk, legality, corner admission, or the immediate action. Use the current evidence to choose. "
            "Return only JSON with `route_mode` equal to one of those three strings, e.g. {\"route_mode\":\"balanced\"}. "
            "If evidence is insufficient, choose balanced. Do not return a route, node list, action, FIRE, or prose.\n"
            + json.dumps(state, separators=(",", ":")))
        meta = {"tick": now, "reason": reason, "session": self.s2_session}
        self.s2_calls += 1
        self.last_astra_tick = now

        def work():
            try:
                result = self.ctx.think(prompt, session_id=meta["session"], timeout_s=100.0)
                return {"text": getattr(result, "text", ""),
                        "session_id": getattr(result, "session_id", None), "failure": None}
            except Exception as exc:
                return {"text": "", "session_id": None,
                        "failure": type(exc).__name__}

        try:
            self.s2_future = self.ctx.spawn(work)
            self.ctx.log("astra_route_requested", tick=now, reason=reason,
                         mode=self.plan.mode)
        except Exception as exc:
            self.s2_future = None
            self.ctx.log("astra_route_spawn_failed", error=type(exc).__name__)

    def collect_astra(self, obs):
        if self.s2_future is None or not self.s2_future.done():
            return
        future, self.s2_future = self.s2_future, None
        try:
            result = future.result()
            if not isinstance(result, dict):
                return
            if result.get("session_id"):
                self.s2_session = result["session_id"]
            data = parse_object(result.get("text", ""))
            mode = data.get("route_mode")
            if mode in ("balanced", "spine", "zigzag") and mode != self.plan.mode:
                old = self.plan.mode
                self.plan.mode = mode
                self.plan.policy_version += 1
                self.advice = None
                self.ctx.log("astra_route_mode_changed", tick=tick_of(obs),
                             old=old, new=mode, age=tick_of(obs) - self.last_astra_tick)
            else:
                self.ctx.log("astra_route_mode_unchanged", tick=tick_of(obs),
                             parsed=bool(data), mode=self.plan.mode)
        except Exception as exc:
            self.ctx.log("astra_route_failed", error=type(exc).__name__)
        if self.s2_waiting_after_loss and self.s2_calls < 2:
            self.s2_waiting_after_loss = False
            self.request_astra(obs, "post-loss route review")

    def action(self, obs):
        now = tick_of(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)

        if self.book.flashing:
            self.pending = None
            self.advice = None
            return "NOOP", None
        if not self.book.unfinished:
            return "NOOP", None

        n = node_of(obs.get("qbert_node"))
        phase = str(qmotion(obs).get("phase", "unknown")).lower()
        if n is None or phase in ("falling", "not_visible") or bool(qmotion(obs).get("falling")):
            self.pending = None
            return "NOOP", None

        if now < self.recovery_until:
            return "NOOP", None
        if self.pending is not None:
            hop = self.pending
            age = now - hop["start"]
            if bool(obs.get("qbert_in_hop")) or n != hop["origin"]:
                hop["moving"] = True
            if hop["moving"] and is_standing(obs):
                self.pending = None
                self.plan.note_landing(n)
                self.neutral_until = max(self.neutral_until, now + 1)
                self.ctx.log("landing_aligned", tick=now,
                             origin=hop["origin"], expected=hop["destination"],
                             observed=n, duration=age,
                             mismatch=n != hop["destination"])
                return "NOOP", None
            if (not hop["moving"] and is_standing(obs)
                    and age <= 3 and hop["posts"] < 2):
                hop["posts"] += 1
                return hop["action"], None
            if age > 18 and is_standing(obs):
                self.pending = None
                self.neutral_until = max(self.neutral_until, now + 1)
                self.ctx.log("landing_not_started", tick=now,
                             origin=hop["origin"], action=hop["action"])
                return "NOOP", None
            return "NOOP", None

        if now < self.neutral_until or not is_standing(obs):
            return "NOOP", None
        candidates = self.plan.candidates(obs)
        if not candidates:
            return "NOOP", None
        chosen = self.choose(obs, candidates) or candidates[0]
        action = chosen["action"]
        if action not in DIRECTIONS or action not in safe_names(obs.get("safe_moves")):
            return "NOOP", None
        dest = GRAPH[n].get(action)
        if dest is None:
            return "NOOP", None
        self.pending = {"origin": n, "destination": dest, "action": action,
                        "start": now, "moving": False, "posts": 1}
        if now - self.last_log >= 30:
            self.ctx.log("leaf_coverage_decision", tick=now,
                         score=finite(obs.get("score")), lives=self.lives,
                         board_epoch=self.book.epoch, clears=self.book.clears,
                         unfinished=len(self.book.unfinished), node=n,
                         target=chosen.get("target"), action=action,
                         destination=dest, danger=chosen["danger"],
                         escapes=chosen["escapes"],
                         leaf_preference_active=bool(set(self.book.unfinished) - LEAVES),
                         hazards=self.threats.facts_for(n),
                         options=[{"action": c["action"], "node": c["destination"],
                                   "cost": c["cost"], "danger": c["danger"],
                                   "escape": c["escape_kind"], "useful": c["useful"]}
                                  for c in candidates])
            self.last_log = now
        return action, dest

    def run(self):
        obs = self.ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.ingest(obs)
        self.request_astra(obs, "initial reusable coverage ordering")
        while not self.ctx.done and not getattr(self.game, "done", False):
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.ingest(obs)
            action, forecast = self.action(obs)
            if action not in self.action_names:
                action = self.noop
                forecast = None
            try:
                self.ctx.act(action)
                if forecast is not None:
                    # Jev is asked about the next landing, not the node that is
                    # already being left. Its answer can finish during this hop.
                    self.request_jev(obs, forecast)
                if self.s2_waiting_after_loss and self.s2_future is None and self.s2_calls < 2:
                    self.s2_waiting_after_loss = False
                    self.request_astra(obs, "post-loss route review")
                self.ctx.wait_for_tick()
            except Exception as exc:
                # Shutdown, model-budget closure, and deadline races are normal
                # exits; no controller error should be raised at episode end.
                self.ctx.log("loop_post_exit", error=type(exc).__name__)
                return
        self.ctx.log("leaf_last_summary", score=getattr(self.game, "score", None),
                     board_epoch=self.book.epoch, inferred_clears=self.book.clears,
                     lives=self.lives, route_mode=self.plan.mode,
                     pending_hop=self.pending is not None)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
