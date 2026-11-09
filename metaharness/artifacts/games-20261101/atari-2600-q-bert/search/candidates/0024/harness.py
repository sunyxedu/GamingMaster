"""Candidate 0024: preserve the proven motor, change the later-board ledger.

For the first four scored boards this follows a destination-colour tracker.
Starting with the inferred Level-2 board, it switches to a board-local S/I/D
cycle learned from scored palette transitions.  The motor remains one-hop,
nonblocking, and revalidated against the latest observation.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver
from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
TIE_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")
COVERAGE_ORDER = (1, 4, 7, 11, 6, 10, 16, 12, 17,
                  18, 13, 9, 14, 19, 20, 8, 2, 0, 3, 5, 15)
COVERAGE_RANK = {node: index for index, node in enumerate(COVERAGE_ORDER)}
CYCLES = {1: ("S", "D"), 2: ("S", "I", "D"),
          3: ("S", "D", "S"), 4: ("S", "I", "D", "I"),
          5: ("S", "I", "D", "S")}


def make_graph():
    graph = {n: {} for n in range(21)}
    row_of = {}
    for row in range(6):
        first = row * (row + 1) // 2
        for col in range(row + 1):
            node = first + col
            row_of[node] = row
            if row:
                if col < row:
                    graph[node]["UP"] = node - row
                if col:
                    graph[node]["LEFT"] = node - row - 1
            if row < 5:
                child = (row + 1) * (row + 2) // 2
                graph[node]["DOWN"] = child + col
                graph[node]["RIGHT"] = child + col + 1
    return graph, row_of


GRAPH, ROW = make_graph()


def distances_from_all():
    result = []
    for root in range(21):
        dist = [99] * 21
        dist[root] = 0
        queue = deque([root])
        while queue:
            here = queue.popleft()
            for there in GRAPH[here].values():
                if dist[there] == 99:
                    dist[there] = dist[here] + 1
                    queue.append(there)
        result.append(dist)
    return result


DIST = distances_from_all()


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def integer(value, default=0):
    if isinstance(value, bool):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(number(value, default))


def node_id(value):
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if 0 <= result < 21 else None


def tick_id(obs):
    return integer(obs.get("tick", obs.get("ticks", 0)), 0)


def palette_of(obs):
    raw = obs.get("cube_colors")
    if (not isinstance(raw, list) or len(raw) != 21 or
            any(isinstance(x, bool) or not isinstance(x, int) for x in raw)):
        return None
    return tuple(raw)


def motion_of(obs):
    value = obs.get("qbert_motion")
    return value if isinstance(value, dict) else {"phase": str(value or "unknown")}


def standing(obs):
    n = node_id(obs.get("qbert_node"))
    motion = motion_of(obs)
    phase = str(motion.get("phase", "unknown")).lower()
    return (n is not None and not bool(obs.get("qbert_in_hop")) and
            phase not in ("hopping", "falling", "not_visible") and
            not bool(motion.get("falling")))


def clean_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    return str(value)


def parse_object(text):
    """Parse an optional JSON object without treating prose as a command."""
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


def answer_value(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value if isinstance(value, str) else None


def probability(answer, key):
    probs = getattr(answer, "probabilities", None) if answer is not None else None
    return number(probs.get(key), 0.0) if isinstance(probs, dict) else 0.0


class PaletteLedger:
    """Track palette epochs; use the exact phase cycle only from inferred L2."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.previous = None
        self.previous_score = None
        self.goal_color = None
        self.start_color = None
        self.intermediate_color = None
        self.destination_color = None
        self.needed = [1] * 21
        self.epoch = 0
        self.clears = 0
        self.flashing = False
        self.flash_score = 0.0
        self.flash_changed_at = -999
        self.complete_since = None
        self.last_progress = 0
        self.transitions = deque(maxlen=10)

    @property
    def level(self):
        return min(5, self.clears // 4 + 1)

    @property
    def exact_cycle_active(self):
        # Keep the first four boards behaviorally on the parent's simple rule.
        return self.clears >= 4

    def log(self, event, **fields):
        self.ctx.log(event, **fields)

    def _reset_board(self):
        self.goal_color = None
        self.start_color = None
        self.intermediate_color = None
        self.destination_color = None
        self.complete_since = None
        self.needed = [1] * 21

    def _learn_phase(self, old, new, landed, tick):
        level = self.level
        if old == new:
            return
        if self.start_color is None:
            self.start_color = old
        if level in (1, 3):
            if self.destination_color is None and old == self.start_color:
                self.destination_color = new
        elif self.intermediate_color is None and old == self.start_color:
            self.intermediate_color = new
        elif (self.destination_color is None and
              self.intermediate_color is not None and
              old == self.intermediate_color and
              new not in (self.start_color, self.intermediate_color)):
            self.destination_color = new
        self.transitions.append({"tick": tick, "node": landed, "old": old,
                                 "new": new, "level": level})
        self.log("phase_transition_seen", tick=tick, node=landed, old=old,
                 new=new, level=level, start=self.start_color,
                 intermediate=self.intermediate_color,
                 destination=self.destination_color)

    def _cycle_demand(self, color):
        level = self.level
        if self.destination_color is not None and color == self.destination_color:
            return 0
        if level in (1, 3):
            return 1
        if self.start_color is not None and color == self.start_color:
            return 2
        if self.intermediate_color is not None and color == self.intermediate_color:
            return 1
        return 1

    def _set_demand(self, colors, obs):
        if not self.exact_cycle_active:
            if self.goal_color is not None:
                self.needed = [0 if color == self.goal_color else 1 for color in colors]
            else:
                raw = obs.get("needed_landings")
                if isinstance(raw, list) and len(raw) == 21:
                    values = [integer(x, -1) for x in raw]
                    if any(x > 0 for x in values):
                        self.needed = [max(0, x) if x >= 0 else 1 for x in values]
                    else:
                        self.needed = [1] * 21
                else:
                    self.needed = [1] * 21
            return
        self.needed = [self._cycle_demand(color) for color in colors]

    def update(self, obs):
        tick = tick_id(obs)
        colors = palette_of(obs)
        score = number(obs.get("score"), self.previous_score or 0.0)
        old_score = self.previous_score
        delta = score - old_score if old_score is not None else 0.0
        if colors is None:
            self.previous_score = score
            return False

        changed = []
        if self.previous is not None:
            changed = [n for n in range(21) if colors[n] != self.previous[n]]
        if self.previous is not None and len(changed) >= 16:
            if not self.flashing:
                self.flashing = True
                self.flash_score = old_score if old_score is not None else score - delta
                self.log("board_flash_started", tick=tick, score=score,
                         changed=len(changed), level=self.level)
            self.flash_changed_at = tick

        if self.flashing:
            quiet = tick - self.flash_changed_at
            returned_to_top = node_id(obs.get("qbert_node")) == 0
            if quiet >= 4 and (returned_to_top or quiet >= 8):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.flashing = False
                self._reset_board()
                self.last_progress = tick
                self.transitions.clear()
                self.log("board_epoch", tick=tick, epoch=self.epoch,
                         inferred_clears=self.clears, tally=tally,
                         next_level=self.level, exact_cycle=self.exact_cycle_active)
                self.previous = colors
                self.previous_score = score
                return True
            self.needed = [1] * 21
            self.previous = colors
            self.previous_score = score
            return False

        # A scored one-cube transition is the calibration evidence.  Unscored
        # Sam reversals and Green Ball rewards must not define phase roles.
        event = str(obs.get("last_event", ""))
        if (self.previous is not None and delta > 0 and
                int(round(delta)) % 100 == 25 and
                event not in ("sam", "green_ball") and 0 < len(changed) <= 3):
            qnode = node_id(obs.get("qbert_node"))
            landed = qnode if qnode in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                old, new = self.previous[landed], colors[landed]
                if self.exact_cycle_active:
                    self._learn_phase(old, new, landed, tick)
                else:
                    self.goal_color = new
                    self.log("destination_color_seen", tick=tick, node=landed,
                             color=new, reward=delta)
                self.last_progress = tick

        self._set_demand(colors, obs)
        if not any(self.needed):
            if self.complete_since is None:
                self.complete_since = tick
            elif (not self.exact_cycle_active and
                  tick - self.complete_since >= 24):
                # Preserve the old controller's board-1-to-4 pass behavior as
                # a negative control; late cycles do not use this blind probe.
                self.goal_color = None
                self.needed = [1] * 21
                self.complete_since = None
                self.last_progress = tick
                self.log("destination_pass_probe", tick=tick, epoch=self.epoch)
        else:
            self.complete_since = None
        self.previous = colors
        self.previous_score = score
        return False

    def summary(self):
        counts = Counter(self.needed)
        return {"level": self.level, "inferred_clears": self.clears,
                "exact_cycle": self.exact_cycle_active,
                "cycle": list(CYCLES[self.level]),
                "roles": {"start": self.start_color,
                          "intermediate": self.intermediate_color,
                          "destination": self.destination_color},
                "demand_counts": {"zero": counts.get(0, 0),
                                  "one": counts.get(1, 0),
                                  "two": counts.get(2, 0)},
                "recent_transitions": list(self.transitions)[-4:]}


class HazardMemory:
    """Short-lived public-object evidence; timers never manufacture a node."""

    def __init__(self):
        self.objects = {}

    def update(self, obs):
        now = tick_id(obs)
        visible = set()
        hazards = obs.get("hazards")
        if isinstance(hazards, list):
            for item in hazards:
                if not isinstance(item, dict) or not item.get("lethal"):
                    continue
                kind = str(item.get("kind", "hazard"))
                if kind not in ("purple_ball", "red_ball", "coily"):
                    continue
                node = node_id(item.get("estimated_node", item.get("node")))
                if node is None:
                    continue
                self.objects[kind] = {"kind": kind, "node": node, "tick": now,
                                      "dx": number(item.get("dx")),
                                      "dy": number(item.get("dy")),
                                      "visible": True}
                visible.add(kind)
        for kind in list(self.objects):
            item = self.objects[kind]
            age = now - integer(item.get("tick"), now)
            limit = 9 if kind in ("purple_ball", "red_ball") else 6
            if age > limit:
                del self.objects[kind]
            else:
                item["visible"] = kind in visible

    @staticmethod
    def down_children(nodes):
        return {there for n in nodes for action, there in GRAPH.get(n, {}).items()
                if action in ("DOWN", "RIGHT")}

    def possible(self, item, now, horizon=8):
        node = item.get("node")
        if node is None:
            return set()
        age = max(0, now - integer(item.get("tick"), now))
        steps = min(2, max(0, (age + horizon) // 9))
        possible = {node}
        frontier = {node}
        if item.get("kind") in ("purple_ball", "red_ball"):
            for _ in range(steps):
                frontier = self.down_children(frontier)
                possible.update(frontier)
        else:
            radius = min(2, max(1, (age + horizon) // 10))
            for _ in range(radius):
                frontier = {there for here in frontier for there in GRAPH[here].values()}
                possible.update(frontier)
        return possible

    def risk(self, origin, destination, now, horizon=8):
        result = 0.0
        reasons = []
        for kind, item in self.objects.items():
            enemy = item.get("node")
            if enemy is None:
                continue
            source_distance = DIST[enemy][origin] if origin is not None else 99
            if kind == "coily" and source_distance <= 1:
                result += 3.0
                reasons.append("coily_near_takeoff")
            if kind in ("purple_ball", "red_ball") and source_distance == 0:
                result += 3.0
                reasons.append("rolling_ball_at_takeoff")
            arrivals = self.possible(item, now, horizon)
            age = max(0, now - integer(item.get("tick"), now))
            if destination in arrivals:
                result += 5.5 if age <= 3 else 3.0
                reasons.append(kind + "_arrival")
            if kind in ("purple_ball", "red_ball"):
                branch = self.down_children(arrivals)
                if destination in branch and destination not in arrivals:
                    result += 2.0
                    reasons.append(kind + "_down_branch")
            elif DIST[enemy][destination] <= 1:
                result += 0.7
                reasons.append("coily_near_landing")
        return result, reasons

    def facts(self, origin, now):
        result = []
        for kind, item in self.objects.items():
            n = item.get("node")
            d = DIST[origin][n] if origin is not None and n is not None else 99
            result.append({"kind": kind,
                           "visibility": "rendered" if item.get("visible") else "recent_memory",
                           "proximity": "same" if d == 0 else "adjacent" if d == 1
                                        else "near" if d <= 2 else "far",
                           "motion": "rolling_down" if kind in ("purple_ball", "red_ball")
                                     else "pursuit"})
        return result

    def signature(self, now):
        return tuple(sorted((kind, item.get("node"),
                             min(2, max(0, now - integer(item.get("tick"), now))))
                            for kind, item in self.objects.items()))

    def location_signature(self):
        # Ignore mere elapsed-age changes, but invalidate advice if a public
        # lethal object appears, disappears, or changes graph position.
        return tuple(sorted((kind, item.get("node"))
                            for kind, item in self.objects.items()))


class RoutePlanner:
    """Stable coverage order plus risk-ranked legal one-hop corridors."""

    def __init__(self, ledger, threats):
        self.ledger = ledger
        self.threats = threats
        self.target = None
        self.cursor = 0
        self.recent = deque(maxlen=22)
        self.posture = "spine"
        self.cycle_mode = "pair"
        self.policy_version = 0

    def reset_board(self):
        self.target = None
        self.cursor = 0
        self.recent.clear()

    def _choose_target(self, origin):
        remaining = {n for n, demand in enumerate(self.ledger.needed) if demand > 0}
        if self.target not in remaining:
            self.target = None
        if self.target is not None and self.target != origin:
            return self.target
        if self.target == origin and self.ledger.needed[origin] <= 0:
            self.target = None
        if self.target is None:
            # Retain the proven cyclic coverage order rather than introduce a
            # nearest-target policy before the later-cycle test activates.
            for offset in range(len(COVERAGE_ORDER)):
                index = (self.cursor + offset) % len(COVERAGE_ORDER)
                candidate = COVERAGE_ORDER[index]
                if candidate in remaining and candidate != origin:
                    self.cursor = index
                    self.target = candidate
                    break
        if self.target is None and remaining:
            ranked = sorted(remaining, key=lambda n: (DIST[origin][n], ROW[n], n))
            self.target = ranked[0]
        return self.target

    def candidates(self, obs, origin=None, safe_override=None, need_override=None):
        now = tick_id(obs)
        start = node_id(obs.get("qbert_node")) if origin is None else node_id(origin)
        if start is None:
            return []
        raw_safe = safe_override if safe_override is not None else obs.get("safe_moves", [])
        if not isinstance(raw_safe, (list, tuple)):
            return []
        safe = set(x for x in raw_safe if isinstance(x, str))
        allowed = [a for a in TIE_ORDER if a in safe and a in GRAPH[start]]
        if not allowed:
            return []
        needs = self.ledger.needed if need_override is None else need_override
        target = self._choose_target(start) if need_override is None else self.target
        remaining = [n for n, value in enumerate(needs) if value > 0]
        visits = Counter(self.recent)
        output = []
        for action in allowed:
            dest = GRAPH[start][action]
            risk, reasons = self.threats.risk(start, dest, now)
            exits = 0
            for next_action, there in GRAPH[dest].items():
                if there == start:
                    continue
                next_risk, _ = self.threats.risk(dest, there, now + 2)
                if next_risk < 3.2:
                    exits += 1
            distance = min((DIST[dest][n] for n in remaining), default=0)
            useful = needs[dest] > 0
            cost = 2.0 * distance + (0.0 if useful else 0.55)
            cost += 11.0 * min(risk, 10.0) - 0.20 * min(exits, 3)
            cost += 0.11 * visits[dest]
            if target is not None:
                cost += 0.15 * DIST[dest][target]
            if self.posture == "upper" and useful:
                cost += 0.12 * ROW[dest]
            elif self.posture == "frontier" and useful:
                cost -= 0.14 * ROW[dest]
            if self.ledger.exact_cycle_active and needs[dest] > 0:
                if self.cycle_mode == "pair" and needs[dest] >= 2:
                    cost += 0.18
                elif self.cycle_mode == "sweep" and needs[dest] == 1:
                    cost += 0.18
            output.append({"key": "edge_" + action, "action": action,
                           "destination": dest, "risk": risk, "reasons": reasons,
                           "exits": exits, "corridor": "open" if exits >= 2 else
                           "narrow" if exits == 1 else "closed", "useful": useful,
                           "demand": integer(needs[dest]), "distance": distance,
                           "cost": cost, "target": target})
        output.sort(key=lambda c: (c["risk"] >= 7.0, c["cost"],
                                   COVERAGE_RANK.get(c["destination"], 99),
                                   TIE_ORDER.index(c["action"])))
        return output

    def note_landing(self, node):
        if node is not None and (not self.recent or self.recent[-1] != node):
            self.recent.append(node)
        if self.target is not None and self.ledger.needed[self.target] <= 0:
            self.target = None

    def astra_summary(self, obs):
        return clean_json({"tick": tick_id(obs), "score": number(obs.get("score")),
                           "lives": integer(obs.get("lives"), -1),
                           "progress": self.ledger.summary(),
                           "route_policy": {"cycle_mode": self.cycle_mode,
                                            "posture": self.posture},
                           "hazards": self.threats.facts(
                               node_id(obs.get("qbert_node")), tick_id(obs))})


class QbertController:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.ledger = PaletteLedger(ctx)
        self.threats = HazardMemory()
        self.route = RoutePlanner(self.ledger, self.threats)
        self.lives = None
        self.life_epoch = 0
        self.last_loss_count = None
        self.pending = None
        self.neutral_until = 0
        self.last_log = -999
        self.last_s1_tick = -999
        self.s1_future = None
        self.s1_meta = None
        self.advice = None
        self.s2_future = None
        self.s2_session = None
        self.s2_calls = 0
        self.s2_reason = "initial later-cycle strategy"
        self.s2_requested_l2 = False
        try:
            self.names = {str(x) for x in game.action_spec.names}
        except Exception:
            self.names = {"NOOP", "FIRE", *DIRECTIONS}
        self.noop = "NOOP" if "NOOP" in self.names else "NOOP"

    def log(self, name, **fields):
        self.ctx.log(name, **fields)

    def ingest(self, obs):
        now = tick_id(obs)
        old_epoch = self.ledger.epoch
        self.ledger.update(obs)
        self.threats.update(obs)
        if self.ledger.epoch != old_epoch:
            self.route.reset_board()
            self.pending = None
            self.advice = None
            self.neutral_until = max(self.neutral_until, now + 2)
        lives = integer(obs.get("lives"), self.lives if self.lives is not None else 4)
        resources = obs.get("resources")
        loss_count = (integer(resources.get("life_losses"), 0)
                      if isinstance(resources, dict) else integer(obs.get("life_lost_count"), 0))
        lost = ((self.lives is not None and lives < self.lives) or
                (self.last_loss_count is not None and loss_count > self.last_loss_count))
        if lost:
            self.life_epoch += 1
            self.pending = None
            self.advice = None
            self.route.target = None
            self.neutral_until = max(self.neutral_until, now + 10)
            self.log("life_recovery", tick=now, lives=lives,
                     life_epoch=self.life_epoch, inferred_clears=self.ledger.clears)
        self.lives = lives
        self.last_loss_count = loss_count

    def _budget_left(self, field):
        try:
            return integer(self.ctx.budget.remaining().get(field), 0)
        except Exception:
            return 0

    def launch_astra(self, obs, reason):
        if (self.s2_future is not None or self.s2_calls >= 2 or self.ctx.done or
                getattr(self.ctx, "system2", True) is None or
                self._budget_left("system2_calls") <= 0):
            return
        state = self.route.astra_summary(obs)
        prompt = (
            "You are the slow strategic adviser for Atari 2600 Q*bert on the six-row, 21-cube Atari board. "
            "The first four rounds are already controlled by a stable destination-colour policy. From inferred "
            "clear 4 onward, code uses the documented exact cycles: level 2 S-I-D, level 3 S-D-S, level 4 "
            "S-I-D-I, level 5 S-I-D-S. Code owns the graph, palette arithmetic, safety, and one-hop execution. "
            "Choose only cycle_mode='pair' or 'sweep'. 'pair' favors finishing a partly progressed cube before "
            "spreading first landings; 'sweep' favors applying first landings broadly before second landings. "
            "Survival and round completion outrank bonuses. Return only a tiny JSON object, e.g. "
            "{\"cycle_mode\":\"pair\"}; no actions, FIRE, node lists, or prose commands. "
            "Reason: " + str(reason) + " State: " + json.dumps(state, separators=(",", ":"))
        )
        self.s2_calls += 1
        session = self.s2_session

        def work():
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=100.0)
                return {"text": getattr(result, "text", ""),
                        "session_id": getattr(result, "session_id", None)}
            except Exception as exc:
                return {"text": "", "session_id": None,
                        "error": type(exc).__name__}

        try:
            self.s2_future = self.ctx.spawn(work)
            self.log("astra_cycle_advice_requested", tick=tick_id(obs), reason=reason,
                     calls=self.s2_calls)
        except Exception as exc:
            self.s2_future = None
            self.log("astra_spawn_failure", error=type(exc).__name__)

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
            mode = data.get("cycle_mode")
            if mode in ("pair", "sweep") and mode != self.route.cycle_mode:
                old = self.route.cycle_mode
                self.route.cycle_mode = mode
                self.route.policy_version += 1
                self.advice = None
                self.log("astra_cycle_mode_changed", tick=tick_id(obs), old=old,
                         new=mode, inferred_clears=self.ledger.clears)
            else:
                self.log("astra_cycle_mode_unchanged", tick=tick_id(obs),
                         parsed=mode in ("pair", "sweep"))
        except Exception as exc:
            self.log("astra_result_failure", error=type(exc).__name__)

    def _jev_state_and_questions(self, obs, expected_node, candidates):
        shown = candidates[:4]
        if len(shown) < 2:
            return None, None
        options = {}
        for item in shown:
            options[item["action"]] = {
                "landing": "unfinished" if item["useful"] else "transit",
                "phase": "two_landing_cycle" if item["demand"] >= 2 else
                         "one_landing_cycle" if item["demand"] == 1 else "complete",
                "danger": "concrete" if item["risk"] >= 2.0 else "clear_or_uncertain",
                "escape": item["corridor"],
                "route": "short" if item["distance"] <= 1 else "long",
                "code_fallback": item is candidates[0],
            }
        options["unknown"] = "No supported preference; keep the code fallback."
        state = clean_json({
            "game": "Atari 2600 Q*bert; 21-node graph, not the arcade pyramid",
            "rules": "Code owns S/I/D arithmetic and graph legality. A candidate is speculative until rechecked at landing.",
            "position": {"expected_landing_node": expected_node,
                         "next_decision": "one legal hop after the current animation"},
            "cycle": self.ledger.summary(),
            "alternatives": options,
            "hazards": self.threats.facts(expected_node, tick_id(obs)),
            "fallback": candidates[0]["action"],
        })
        instructions = (
            "Choose the safest supplied next direction. Prefer clear danger facts, then an open escape, "
            "then useful cube progress. Do not calculate distances or invent an action. Choose unknown "
            "when the facts do not support a preference. The direction will be checked again against "
            "the actual safe_moves before use."
        )
        keys = list(options)
        first = {key: options[key] for key in keys}
        second = {key: options[key] for key in reversed(keys)}
        questions = {
            "choice_a": Choice(instructions, first),
            "choice_b": Choice(instructions, second),
            "fallback_safe": Noul(
                "Do the supplied facts support the code fallback as a reasonable next hop? "
                "A stated occupied/concrete hazard or closed exit is unsafe; uncertainty alone is not proof.",
                {"true": "The fallback is clear enough and has at least one usable exit.",
                 "false": "The fallback has supported immediate danger or no usable exit."}),
        }
        return state, questions

    def launch_jev_for_landing(self, obs, destination):
        if (self.s1_future is not None or self.ctx.done or
                getattr(self.ctx, "system1", True) is None or
                tick_id(obs) - self.last_s1_tick < 12 or
                self._budget_left("system1_calls") <= 1):
            return
        projected_need = list(self.ledger.needed)
        if projected_need[destination] > 0:
            projected_need[destination] -= 1
        projected = dict(obs)
        projected["qbert_node"] = destination
        projected["qbert_in_hop"] = False
        projected["qbert_motion"] = {"phase": "standing", "falling": False}
        candidates = self.route.candidates(
            projected, origin=destination, safe_override=list(GRAPH[destination]),
            need_override=projected_need)
        if len(candidates) < 2:
            return
        state, questions = self._jev_state_and_questions(obs, destination, candidates)
        if state is None:
            return
        now = tick_id(obs)
        meta = {"tick": now, "node": destination, "life": self.life_epoch,
                "epoch": self.ledger.epoch, "policy": self.route.policy_version,
                "options": [item["action"] for item in candidates[:4]],
                "hazards": self.threats.location_signature()}
        self.last_s1_tick = now

        def work():
            try:
                return self.ctx.decide(state, questions, timeout_s=0.8)
            except Exception as exc:
                return {"_error": type(exc).__name__}

        try:
            self.s1_future = self.ctx.spawn(work)
            self.s1_meta = meta
        except Exception as exc:
            self.log("jev_spawn_failure", error=type(exc).__name__)

    def collect_jev(self, obs):
        if self.s1_future is None or not self.s1_future.done():
            return
        future, meta = self.s1_future, self.s1_meta
        self.s1_future = None
        self.s1_meta = None
        try:
            answers = future.result()
            if not isinstance(answers, dict) or not isinstance(meta, dict):
                return
            a = answer_value(answers.get("choice_a"))
            b = answer_value(answers.get("choice_b"))
            agree = a is not None and a == b and a in meta.get("options", [])
            confidence = (min(probability(answers.get("choice_a"), a),
                              probability(answers.get("choice_b"), b))
                          if agree else 0.0)
            safety = number(getattr(answers.get("fallback_safe"), "value", 0.0), 0.0)
            now = tick_id(obs)
            if (now - integer(meta.get("tick"), now) > 16 or
                    meta.get("life") != self.life_epoch or
                    meta.get("epoch") != self.ledger.epoch or
                    meta.get("policy") != self.route.policy_version):
                self.log("jev_advice_stale", tick=now, origin_tick=meta.get("tick"))
                return
            self.advice = {**meta, "action": a if agree and confidence >= 0.60 else None,
                           "confidence": confidence, "fallback_safe": safety}
        except Exception as exc:
            self.log("jev_result_failure", error=type(exc).__name__)

    def apply_advice(self, obs, candidates):
        fallback = candidates[0] if candidates else None
        advice, self.advice = self.advice, None
        if fallback is None or not advice:
            return fallback
        now = tick_id(obs)
        if (advice.get("node") != node_id(obs.get("qbert_node")) or
                advice.get("life") != self.life_epoch or
                advice.get("epoch") != self.ledger.epoch or
                advice.get("policy") != self.route.policy_version or
                advice.get("hazards") != self.threats.location_signature() or
                now - integer(advice.get("tick"), now) < 0 or
                now - integer(advice.get("tick"), now) > 16):
            return fallback
        choice = next((item for item in candidates
                       if item["action"] == advice.get("action")), None)
        if choice is None or advice.get("fallback_safe", 0.0) < 0.60:
            return fallback
        # Fresh geometry and hazard facts outrank a delayed preference.  Jev may
        # only break a near tie; it cannot force a materially riskier corridor.
        if (choice["risk"] <= fallback["risk"] + 0.35 and
                choice["cost"] <= fallback["cost"] + 2.0 and
                choice["corridor"] != "closed" and
                choice["action"] in (obs.get("safe_moves") or [])):
            if choice["action"] != fallback["action"]:
                self.log("jev_next_hop_changed", tick=now, old=fallback["action"],
                         new=choice["action"], confidence=advice.get("confidence"))
            return choice
        return fallback

    def choose_action(self, obs):
        now = tick_id(obs)
        self.collect_astra(obs)
        self.collect_jev(obs)
        if self.ledger.flashing:
            self.pending = None
            return self.noop
        node = node_id(obs.get("qbert_node"))
        motion = motion_of(obs)
        phase = str(motion.get("phase", "unknown")).lower()
        if node is None or phase in ("falling", "not_visible") or motion.get("falling"):
            self.pending = None
            return self.noop
        if now < self.neutral_until:
            return self.noop
        if self.pending is not None:
            hop = self.pending
            age = now - hop["tick"]
            if bool(obs.get("qbert_in_hop")) or node != hop["origin"]:
                hop["moving"] = True
            if hop["moving"] and standing(obs):
                self.pending = None
                self.route.note_landing(node)
                if node == hop["destination"]:
                    self.log("hop_confirmed", tick=now, origin=hop["origin"],
                             destination=node, duration=age)
                else:
                    self.route.target = None
                    self.neutral_until = max(self.neutral_until, now + 3)
                    self.log("hop_resynchronized", tick=now,
                             expected=hop["destination"], observed=node)
                return self.noop
            if (not hop["moving"] and standing(obs) and age <= 3 and
                    hop["posts"] < 2 and hop["action"] in (obs.get("safe_moves") or [])):
                hop["posts"] += 1
                return hop["action"]
            if age > 18 and standing(obs):
                self.pending = None
                self.neutral_until = max(self.neutral_until, now + 3)
                self.log("hop_start_timeout", tick=now, origin=hop["origin"],
                         expected=hop["destination"])
            else:
                return self.noop
        if not standing(obs):
            return self.noop
        candidates = self.route.candidates(obs)
        if not candidates:
            return self.noop
        chosen = self.apply_advice(obs, candidates) or candidates[0]
        action = chosen.get("action")
        if action not in DIRECTIONS or action not in (obs.get("safe_moves") or []):
            return self.noop
        destination = GRAPH[node].get(action)
        if destination is None:
            return self.noop
        self.pending = {"origin": node, "destination": destination,
                        "action": action, "tick": now, "posts": 1,
                        "moving": False}
        if now - self.last_log >= 25:
            self.log("phase_route_status", tick=now,
                     score=number(obs.get("score")), lives=self.lives,
                     inferred_clears=self.ledger.clears,
                     level=self.ledger.level,
                     demands=self.ledger.summary()["demand_counts"],
                     node=node, action=action, target=self.route.target,
                     hazards=self.threats.facts(node, now))
            self.last_log = now
        self.launch_jev_for_landing(obs, destination)
        return action

    def run(self):
        obs = self.ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.ingest(obs)
        self.launch_astra(obs, "opening route and later colour-cycle plan")
        if self.s2_future is not None:
            self.s2_reason = None
        while not self.ctx.done and not getattr(self.game, "done", False):
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.ingest(obs)
            if self.ledger.clears >= 4 and not self.s2_requested_l2:
                self.s2_requested_l2 = True
                self.s2_reason = "adapt the Level-2 phase schedule after four inferred clears"
            action = self.choose_action(obs)
            if self.s2_reason and self.s2_future is None and self.s2_calls < 2:
                reason = self.s2_reason
                self.launch_astra(obs, reason)
                if self.s2_future is not None:
                    self.s2_reason = None
            if action not in self.names or action == "FIRE":
                action = self.noop
            try:
                self.ctx.act(action)
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                break
        self.log("candidate_summary", score=number(getattr(self.game, "score", 0.0)),
                 inferred_clears=self.ledger.clears, level=self.ledger.level,
                 exact_cycle=self.ledger.exact_cycle_active,
                 phase=self.ledger.summary(), lives=self.lives,
                 cycle_mode=self.route.cycle_mode)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        QbertController(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
