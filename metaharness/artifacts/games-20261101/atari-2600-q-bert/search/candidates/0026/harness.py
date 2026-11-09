"""0026: phase-demand routing with a target-level model interface.

The controller trusts the adaptor's observed cube phases/landing demand instead
of inferring a binary destination colour, and uses Jev only to choose among
near-equivalent *cube goals*. Code owns route legality, phase arithmetic,
hazard constraints, and the one-hop motor. Astra may set a low-impact global
tie preference, never an action or a safety override.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.models.jev import Choice


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
TIE_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")
COVERAGE_ORDER = (1, 4, 7, 11, 6, 10, 16, 12, 17,
                  18, 13, 9, 14, 19, 20, 8, 2, 0, 3, 5, 15)
COVERAGE_RANK = {node: index for index, node in enumerate(COVERAGE_ORDER)}


def make_graph():
    graph = {node: {} for node in range(21)}
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


GRAPH, ROW = make_graph()


def as_int(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def as_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def node_of(value):
    value = as_int(value)
    return value if value is not None and 0 <= value < 21 else None


def tick_of(obs):
    return max(0, as_int(obs.get("tick", obs.get("ticks", 0)), 0) or 0)


def names_in(value, allowed=DIRECTIONS):
    if not isinstance(value, (list, tuple)):
        return []
    allowed = set(allowed)
    return [name for name in value if isinstance(name, str) and name in allowed]


def standing(obs):
    node = node_of(obs.get("qbert_node"))
    motion = obs.get("qbert_motion")
    if not isinstance(motion, dict):
        motion = {"phase": str(motion or "unknown")}
    phase = str(motion.get("phase", "unknown")).lower()
    return (node is not None and not bool(obs.get("qbert_in_hop"))
            and phase not in ("hopping", "falling", "not_visible")
            and not bool(motion.get("falling")))


def phases_of(obs):
    raw = obs.get("cube_phase", obs.get("cube_state"))
    if not isinstance(raw, list) or len(raw) != 21:
        return None
    result = []
    for value in raw:
        label = str(value).upper()
        result.append(label if label in ("S", "I", "D") else "?")
    return result


def colors_of(obs):
    raw = obs.get("cube_colors")
    if not isinstance(raw, list) or len(raw) != 21:
        return None
    result = []
    for value in raw:
        number = as_int(value)
        if number is None:
            return None
        result.append(number)
    return tuple(result)


def demand_of(obs, phases, previous):
    """Read the observed landing demand; use the public phase labels as repair.

    needed_landings is the authoritative per-cube counter. The manual's phase
    cycle is a redundant consistency source and a fallback if a counter is
    missing. Unknown cells retain their prior value rather than being declared
    complete.
    """
    raw = obs.get("needed_landings")
    demand = list(previous) if isinstance(previous, list) and len(previous) == 21 else [1] * 21
    valid = False
    if isinstance(raw, list) and len(raw) == 21:
        parsed = []
        valid = True
        for value in raw:
            number = as_int(value)
            if number is None or number < 0 or number > 3:
                valid = False
                break
            parsed.append(number)
        if valid:
            demand = parsed

    level = as_int(obs.get("level"), 1) or 1
    level = min(5, max(1, level))  # Level 5 repeats after the initial five levels.
    for node, phase in enumerate(phases or []):
        if phase == "?":
            continue
        # Minimum useful landings to the next D phase. On Level 4, D advances
        # to I, so it takes two landings to return to D. On repeating Level 5,
        # D advances to S, so three landings return to D. Level 3's D->S is
        # still one landing back to D.
        if level == 1:
            expected = {"S": 1, "I": 1, "D": 0}.get(phase, 1)
        elif level == 2:
            expected = {"S": 2, "I": 1, "D": 0}.get(phase, 1)
        elif level == 3:
            expected = {"S": 1, "I": 1, "D": 0}.get(phase, 1)
        elif level == 4:
            expected = {"S": 2, "I": 1, "D": 2}.get(phase, 1)
        else:
            expected = {"S": 2, "I": 1, "D": 3}.get(phase, 1)
        if not valid:
            demand[node] = expected
    return [max(0, min(3, as_int(value, 1) or 0)) for value in demand]


def safe_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    return str(value)


def first_object(text):
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(text[:10000]):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def answer_value(answer):
    return getattr(answer, "value", None) if answer is not None else None


def answer_probability(answer, key):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return 0.0
    return max(0.0, min(1.0, as_float(probabilities.get(key), 0.0)))


def distances_from(start, phases, level):
    """Dijkstra cost to each cube, penalizing needless D transit on cycles."""
    distance = [float("inf")] * 21
    if start is None:
        return distance
    distance[start] = 0.0
    todo = set(range(21))
    while todo:
        here = min(todo, key=lambda node: distance[node])
        todo.remove(here)
        if not math.isfinite(distance[here]):
            break
        for there in GRAPH[here].values():
            step = 1.0
            # On Levels 3-5, D is reversible. Avoid changing a completed cube
            # merely as a transit square unless it is actually the goal.
            if level >= 3 and there != start and phases[there] == "D":
                step += 2.5
            candidate = distance[here] + step
            if candidate < distance[there]:
                distance[there] = candidate
    return distance


def hazard_cost(obs, origin, destination):
    """Conservative cost from currently rendered hazards only; no shadows."""
    raw = obs.get("hazards")
    if not isinstance(raw, list):
        return 0.0
    cost = 0.0
    for hazard in raw:
        if not isinstance(hazard, dict) or not hazard.get("lethal"):
            continue
        kind = str(hazard.get("kind", ""))
        enemy = node_of(hazard.get("estimated_node", hazard.get("node")))
        if enemy is None:
            continue
        risk = str(hazard.get("risk", "unknown")).lower()
        if destination == enemy:
            cost += 10.0 if risk in ("near", "imminent", "occupied") else 4.0
        elif kind == "coily" and destination in GRAPH.get(enemy, {}).values():
            cost += 1.7 if risk in ("near", "imminent") else 0.45
        elif kind in ("purple_ball", "red_ball"):
            # A rolling ball descends only along the two child directions.
            if any(action in ("DOWN", "RIGHT") and child == destination
                   for action, child in GRAPH.get(enemy, {}).items()):
                cost += 2.0 if risk in ("near", "imminent") else 0.6
        if origin == enemy and kind == "coily":
            cost += 4.0
    return cost


class PhaseBoard:
    """Observation-synchronized colour state and board handoff detector."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.phases = ["S"] * 21
        self.need = [1] * 21
        self.level = 1
        self.epoch = 0
        self.clears = 0
        self.previous_colors = None
        self.last_score = 0.0
        self.flashing = False
        self.flash_last = -999
        self.flash_score = 0.0
        self.handoff_until = 0
        self.complete_since = None
        self.last_progress = 0
        self.awaiting_epoch_calibration = False

    def update(self, obs):
        now = tick_of(obs)
        old_epoch = self.epoch
        colors = colors_of(obs)
        score = as_float(obs.get("score"), self.last_score)
        score_delta = score - self.last_score
        changed = []
        if colors is not None and self.previous_colors is not None:
            changed = [i for i in range(21)
                       if colors[i] != self.previous_colors[i]]

        if len(changed) >= 16:
            if not self.flashing:
                self.flash_score = self.last_score
                self.ctx.log("phase_flash_start", tick=now, score=score,
                             palette_changes=len(changed))
            self.flashing = True
            self.flash_last = now
            self.complete_since = None

        if self.flashing and now - self.flash_last >= 4:
            tally = score - self.flash_score
            if tally >= 1000:
                self.clears += 1
                self.awaiting_epoch_calibration = True
                self.need = [1] * 21
            self.epoch += 1
            self.flashing = False
            self.handoff_until = now + 2
            self.last_progress = now
            self.ctx.log("phase_board_epoch", tick=now, epoch=self.epoch,
                         inferred_clears=self.clears, tally=tally)

        phase_data = phases_of(obs)
        if phase_data is not None:
            self.phases = phase_data
        self.level = min(5, max(1, as_int(obs.get("level"), self.level) or self.level))
        self.need = demand_of(obs, self.phases, self.need)
        if self.awaiting_epoch_calibration:
            raw_need = obs.get("needed_landings")
            has_positive_counter = (isinstance(raw_need, list) and len(raw_need) == 21
                                    and any(as_int(value, 0) > 0 for value in raw_need))
            local_scored_transition = (0 < len(changed) <= 3 and score_delta > 0
                                       and int(round(score_delta)) % 100 == 25)
            if has_positive_counter or local_scored_transition:
                self.awaiting_epoch_calibration = False
            elif isinstance(raw_need, list) and len(raw_need) == 21 and all(
                    as_int(value, -1) == 0 for value in raw_need):
                # A few ROM frames can expose the previous board's all-zero
                # completion counter immediately after the global flash. Keep
                # coverage alive until the new board supplies positive demand
                # or one local scored landing confirms its phase palette.
                self.need = [1] * 21

        # Distinguish a real all-D board from a flashing/stale zero demand. The
        # global palette transition remains the completion authority.
        if not any(value > 0 for value in self.need) and not self.flashing:
            if self.complete_since is None:
                self.complete_since = now
            elif now - self.complete_since >= 20 and not self.flashing:
                # A palette decoder can suppress a flash. Re-arm from visible
                # phases only after a bounded wait; never claim a scored clear.
                if any(phase != "D" for phase in self.phases):
                    self.need = demand_of({"level": self.level}, self.phases, self.need)
                self.complete_since = None
        else:
            self.complete_since = None

        if changed or score != self.last_score:
            self.last_progress = now
        if colors is not None:
            self.previous_colors = colors
        self.last_score = score
        return self.epoch != old_epoch

    def unfinished(self):
        return [node for node, count in enumerate(self.need) if count > 0]


class TargetRouter:
    """Choose a persistent cube goal; route actions remain one-hop and local."""

    def __init__(self, ctx, board):
        self.ctx = ctx
        self.board = board
        self.target = None
        self.cursor = 0
        self.previous_node = None
        self.route_mode = "stable"
        self.pending_mode = None
        self.goal_advice = None
        self.goal_advice_meta = None
        self.last_goal_call = -999
        self.s1_future = None
        self.s1_meta = None
        self.s2_future = None
        self.s2_session = None
        self.s2_calls = 0
        self.last_s2_tick = -999

    def mode_rank(self, node):
        row = ROW[node]
        col = node - row * (row + 1) // 2
        center = abs(2 * col - row)
        if self.route_mode == "upper":
            return row * 0.2 + COVERAGE_RANK.get(node, 99) * 0.001
        if self.route_mode == "center":
            return center * 2.0 + row * 0.03
        if self.route_mode == "escape":
            return -len(GRAPH[node]) * 1.0 + COVERAGE_RANK.get(node, 99) * 0.001
        return COVERAGE_RANK.get(node, 99)

    def target_costs(self, origin):
        if origin is None:
            return {}
        distances = distances_from(origin, self.board.phases, self.board.level)
        result = {}
        for node in self.board.unfinished():
            # A two-step cube phase gets a small cost so a one-hop partial
            # landing is not mistaken for final completion; distance dominates.
            result[node] = distances[node] + 0.18 * max(0, self.board.need[node] - 1)
        return result

    def choose_target(self, obs, life_epoch):
        origin = node_of(obs.get("qbert_node"))
        unfinished = set(self.board.unfinished())
        if not unfinished or origin is None:
            self.target = None
            return None

        costs = self.target_costs(origin)
        if self.target in unfinished:
            advice = self.goal_advice
            meta = self.goal_advice_meta
            if advice is not None and isinstance(meta, dict):
                age = tick_of(obs) - as_int(meta.get("tick"), tick_of(obs))
                if (meta.get("epoch") == self.board.epoch
                        and meta.get("life") == life_epoch
                        and 0 <= age <= 12 and advice in unfinished
                        and costs.get(advice, float("inf")) <= costs.get(self.target, 99) + 1.0):
                    if advice != self.target:
                        self.ctx.log("jev_goal_adopted", tick=tick_of(obs), target=advice,
                                     cost=costs.get(advice), baseline=costs.get(self.target))
                    self.target = advice
                self.goal_advice = None
                self.goal_advice_meta = None
            return self.target
        else:
            # Keep the strongest parent's stable coverage sequence as the
            # default target scheduler; only the target-selection interface is
            # changed here. Route phase demand still recomputes each hop.
            baseline = None
            for offset in range(len(COVERAGE_ORDER)):
                index = (self.cursor + offset) % len(COVERAGE_ORDER)
                candidate = COVERAGE_ORDER[index]
                if candidate in unfinished and candidate != origin:
                    self.cursor = (index + 1) % len(COVERAGE_ORDER)
                    baseline = candidate
                    break
            if baseline is None:
                baseline = min(unfinished, key=lambda node: (costs.get(node, 99), node))

        advice = self.goal_advice
        meta = self.goal_advice_meta
        if advice is not None and isinstance(meta, dict):
            age = tick_of(obs) - as_int(meta.get("tick"), tick_of(obs))
            advised_cost = costs.get(advice, float("inf"))
            base_cost = costs.get(baseline, float("inf"))
            if (meta.get("epoch") == self.board.epoch
                    and meta.get("life") == life_epoch
                    and 0 <= age <= 12 and advice in unfinished
                    and advised_cost <= base_cost + 1.0):
                if advice != baseline:
                    self.ctx.log("jev_goal_adopted", tick=tick_of(obs), target=advice,
                                 cost=advised_cost, baseline=base_cost)
                baseline = advice
            self.goal_advice = None
            self.goal_advice_meta = None
        self.target = baseline
        return baseline

    def options(self, obs):
        origin = node_of(obs.get("qbert_node"))
        if origin is None:
            return []
        costs = self.target_costs(origin)
        baseline = self.target if self.target in costs else None
        if baseline is None:
            baseline = self.choose_target(obs, 0)
        if baseline is None:
            return []
        # Only offer comparable route goals; keep the deterministic coverage
        # target in the set so the advice can explicitly retain it.
        alternatives = [node for node in costs if node != origin
                        and costs[node] <= costs.get(baseline, 99) + 1.0]
        ranked = sorted((node for node in alternatives if node != baseline),
                        key=lambda node:
                        (abs(costs[node] - costs.get(baseline, 99)),
                         self.mode_rank(node), node))[:3]
        return [baseline] + ranked

    def state_for_jev(self, obs, options):
        origin = node_of(obs.get("qbert_node"))
        costs = self.target_costs(origin)
        descriptions = []
        for node in options:
            row = ROW[node]
            descriptions.append({
                "key": "cube_" + str(node),
                "location": "top" if row <= 1 else "upper" if row <= 2 else
                            "middle" if row <= 4 else "bottom",
                "phase_work": "two landings" if self.board.need[node] >= 2 else "one landing",
                "route": "here" if costs[node] < 0.5 else "near" if costs[node] <= 3 else "far",
                "destination_phase": self.board.phases[node],
                "code_rank": "best" if node == options[0] else "short alternative",
            })
        return safe_json({
            "game": "Atari 2600 Q*bert, six-row 21-cube board; never use FIRE",
            "board": {"level": self.board.level, "epoch": self.board.epoch,
                      "cycle": {"1": "S>D", "2": "S>I>D", "3": "S>D>S",
                                "4": "S>I>D>I", "5": "S>I>D>S"}.get(str(self.board.level), "level-5 repeat"),
                      "unfinished_count": len(self.board.unfinished()),
                      "route_mode": self.route_mode},
            "position": {"node": origin, "standing": standing(obs)},
            "target_options": descriptions,
            "constraint": "Choose a useful next cube; code will recheck demand, route and hazards before any hop.",
        })

    def start_jev(self, obs, options, life_epoch):
        ctx = self.ctx
        if (not options or self.s1_future is not None or ctx.done
                or getattr(ctx, "system1", True) is None):
            return
        now = tick_of(obs)
        if now - self.last_goal_call < 12:
            return
        try:
            if ctx.budget.remaining().get("system1_calls", 0) <= 1:
                return
        except Exception:
            pass
        keys = ["cube_" + str(node) for node in options]
        descriptions = {}
        for node in options:
            key = "cube_" + str(node)
            descriptions[key] = {
                "what": "Work toward this currently unfinished cube as the next coverage goal.",
                "phase_work": "two landings" if self.board.need[node] >= 2 else "one landing",
                "route_length": "short" if self.target_costs(node_of(obs.get("qbert_node"))).get(node, 99) <= 3 else "long",
                "not_for": "inventing a direction, ignoring a hazard, or collecting optional bonuses",
            }
        descriptions["defer"] = {"what": "Keep the code-selected goal; use when no alternative is clearly better."}
        options_a = list(keys) + ["defer"]
        options_b = list(reversed(options_a))
        instruction = (
            "Choose the next cube goal from target_options. Prefer a short route and a cube that still needs "
            "its observed phase work. A destination cube on a reversible later level should not be revisited "
            "without need. This is only a durable coverage preference: code recomputes every legal hop and "
            "handles visible hazards. Choose defer if the choices are not distinguishable."
        )
        criteria_a = {key: descriptions.get(key, "Only choose a listed code-generated goal.")
                      for key in options_a}
        criteria_b = {key: descriptions.get(key, "Only choose a listed code-generated goal.")
                      for key in options_b}
        questions = {
            "goal_a": Choice(instruction, criteria_a),
            "goal_b": Choice(instruction, criteria_b),
        }
        state = self.state_for_jev(obs, options)
        meta = {"tick": now, "epoch": self.board.epoch, "life": life_epoch,
                "nodes": list(options), "keys": keys}
        self.last_goal_call = now
        self.s1_meta = meta

        def work():
            try:
                return {"answers": ctx.decide(state, questions, timeout_s=1.0), "error": None}
            except Exception as exc:
                return {"answers": None, "error": type(exc).__name__}

        try:
            self.s1_future = ctx.spawn(work)
        except Exception as exc:
            self.s1_future = None
            self.s1_meta = None
            ctx.log("jev_spawn_failure", error=type(exc).__name__)

    def collect_jev(self, obs, life_epoch):
        future = self.s1_future
        if future is None or not future.done():
            return
        meta = self.s1_meta
        self.s1_future = None
        self.s1_meta = None
        try:
            result = future.result()
            answers = result.get("answers") if isinstance(result, dict) else None
            if not isinstance(answers, dict) or not isinstance(meta, dict):
                return
            a = answers.get("goal_a")
            b = answers.get("goal_b")
            key_a, key_b = answer_value(a), answer_value(b)
            if (key_a != key_b or key_a not in meta.get("keys", [])
                    or min(answer_probability(a, key_a), answer_probability(b, key_b)) < 0.60):
                return
            try:
                target = int(key_a.split("_", 1)[1])
            except (ValueError, IndexError, AttributeError):
                return
            age = tick_of(obs) - as_int(meta.get("tick"), tick_of(obs))
            if (meta.get("epoch") != self.board.epoch or meta.get("life") != life_epoch
                    or not 0 <= age <= 100 or target not in self.board.unfinished()):
                return
            self.goal_advice = target
            self.goal_advice_meta = meta
            self.ctx.log("jev_goal_ready", tick=tick_of(obs), target=target, age=age)
        except Exception as exc:
            self.ctx.log("jev_result_failure", error=type(exc).__name__)

    def start_astra(self, obs, reason, losses, session_id):
        ctx = self.ctx
        if getattr(self, "s2_future", None) is not None or self.s2_calls >= 2 or ctx.done:
            return
        if getattr(ctx, "system2", True) is None:
            return
        try:
            if ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            pass
        summary = safe_json({
            "reason": reason, "tick": tick_of(obs), "level": self.board.level,
            "epoch": self.board.epoch, "inferred_clears": self.board.clears,
            "unfinished": len(self.board.unfinished()), "lives": obs.get("lives"),
            "losses": losses, "phase_counts": {phase: self.board.phases.count(phase)
                                               for phase in ("S", "I", "D", "?")},
            "current_mode": self.route_mode,
        })
        prompt = (
            "You are Astra, a slow off-clock strategist for real-time Atari 2600 Q*bert. "
            "The six-row board has 21 nodes. Code reads observed S/I/D phases and needed_landings, "
            "uses the exact cycle by level (1 S>D; 2 S>I>D; 3 S>D>S; 4 S>I>D>I; "
            "5 S>I>D>S), validates safe_moves, and executes one hop at a time. Jev may choose "
            "only among code-listed unfinished cube goals; it cannot issue actions. Survival and "
            "round completion outrank bonuses. Recommend one durable tie preference for future "
            "coverage goals: route_mode must be stable, center, upper, or escape. These affect only "
            "near-equal goals; they never block a legal hop or authorize danger. Use the observed "
            "phase distribution/life-loss history, not a seed-specific route. Return ONLY JSON, "
            "for example {\"route_mode\":\"stable\"}; invalid output is ignored. State: "
            + json.dumps(summary, separators=(",", ":"))
        )
        self.s2_calls += 1
        self.s2_future = None
        if session_id is not None:
            self.s2_session = session_id

        def work():
            try:
                result = ctx.think(prompt, session_id=session_id, timeout_s=100.0)
                return {"text": getattr(result, "text", ""),
                        "session_id": getattr(result, "session_id", None), "error": None}
            except Exception as exc:
                return {"text": "", "session_id": None, "error": type(exc).__name__}

        try:
            self.s2_future = ctx.spawn(work)
            ctx.log("astra_route_review", tick=tick_of(obs), reason=reason)
        except Exception as exc:
            self.s2_future = None
            ctx.log("astra_spawn_failure", error=type(exc).__name__)

    def collect_astra(self, obs):
        future = getattr(self, "s2_future", None)
        if future is None or not future.done():
            return
        self.s2_future = None
        try:
            result = future.result()
            if not isinstance(result, dict):
                return
            if result.get("session_id"):
                self.s2_session = result["session_id"]
            value = first_object(result.get("text", "")).get("route_mode")
            if value in ("stable", "center", "upper", "escape"):
                old = self.route_mode
                self.route_mode = value
                if old != value:
                    self.ctx.log("astra_route_mode", tick=tick_of(obs), old=old, new=value)
        except Exception as exc:
            self.ctx.log("astra_result_failure", error=type(exc).__name__)


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.board = PhaseBoard(ctx)
        self.router = TargetRouter(ctx, self.board)
        try:
            self.names = {str(value) for value in game.action_spec.names}
        except Exception:
            self.names = {"NOOP", "FIRE", *DIRECTIONS}
        self.noop = "NOOP" if "NOOP" in self.names else "NOOP"
        self.pending = None
        self.neutral_until = 0
        self.recovery_until = 0
        self.life_epoch = 0
        self.losses = 0
        self.last_lives = None
        self.last_loss_count = None
        self.last_action_node = None
        self.last_log = -999
        self.s2_future = None
        self.s2_session = None
        self.s2_calls = 0
        self.request_second_review = False

    def observe_life(self, obs):
        now = tick_of(obs)
        lives = as_int(obs.get("lives"), self.last_lives)
        resources = obs.get("resources")
        loss_count = as_int(obs.get("life_lost_count"), None)
        if loss_count is None and isinstance(resources, dict):
            loss_count = as_int(resources.get("life_losses"), None)
        lost = ((self.last_lives is not None and lives is not None and lives < self.last_lives)
                or (self.last_loss_count is not None and loss_count is not None
                    and loss_count > self.last_loss_count))
        if lost:
            self.life_epoch += 1
            self.losses += 1
            self.router.target = None
            self.router.goal_advice = None
            self.router.goal_advice_meta = None
            self.pending = None
            self.recovery_until = now + 4
            self.neutral_until = max(self.neutral_until, now + 8)
            self.ctx.log("life_recovery", tick=now, lives=lives, losses=self.losses,
                         life_epoch=self.life_epoch, last_node=self.last_action_node)
            if self.losses == 1:
                self.request_second_review = True
        if lives is not None:
            self.last_lives = lives
        if loss_count is not None:
            self.last_loss_count = loss_count
        return lost

    def update(self, obs):
        epoch_changed = self.board.update(obs)
        if epoch_changed:
            self.router.target = None
            self.router.cursor = 0
            self.router.goal_advice = None
            self.router.goal_advice_meta = None
            self.pending = None
            self.neutral_until = max(self.neutral_until, tick_of(obs) + 2)
        self.observe_life(obs)
        self.router.collect_jev(obs, self.life_epoch)
        self.router.collect_astra(obs)

    def choose_edge(self, obs):
        now = tick_of(obs)
        origin = node_of(obs.get("qbert_node"))
        if origin is None:
            return None
        # The adaptor's conservative safe_moves are the hard boundary. If they
        # are genuinely empty, wait instead of guessing from the full graph.
        if "safe_moves" in obs:
            safe = set(names_in(obs.get("safe_moves")))
        else:
            safe = set(names_in(obs.get("legal_moves")))
        safe = [action for action in TIE_ORDER if action in safe and action in GRAPH[origin]]
        if not safe:
            return None

        target = self.router.choose_target(obs, self.life_epoch)
        if target is None:
            return None
        options = self.router.options(obs)
        self.router.start_jev(obs, options, self.life_epoch)

        distances = distances_from(target, self.board.phases, self.board.level)
        candidates = []
        for action in safe:
            dest = GRAPH[origin][action]
            route = distances[dest]
            value = route + 6.5 * hazard_cost(obs, origin, dest)
            if dest == target:
                value -= 2.0
            if self.board.need[dest] > 0 and dest != target:
                value -= 0.18
            # On reversible cycles, do not overwrite a completed D cube just
            # because it shortens the current route by one edge.
            if self.board.level >= 3 and self.board.phases[dest] == "D" and dest != target:
                value += 3.0
            if dest == self.router.previous_node and len(safe) > 1:
                value += 0.18
            # Keep access to a next hop when costs are otherwise comparable.
            value -= 0.08 * min(3, len(GRAPH[dest]))
            candidates.append((value, TIE_ORDER.index(action), action, dest))
        candidates.sort()
        return candidates[0][2], candidates[0][3]

    def motor(self, obs):
        now = tick_of(obs)
        if self.board.flashing or now < self.board.handoff_until:
            self.pending = None
            return self.noop
        motion = obs.get("qbert_motion")
        if not isinstance(motion, dict):
            motion = {"phase": str(motion or "unknown")}
        phase = str(motion.get("phase", "unknown")).lower()
        if phase in ("falling", "not_visible") or motion.get("falling"):
            self.pending = None
            return self.noop
        if now < self.recovery_until:
            return self.noop

        node = node_of(obs.get("qbert_node"))
        hop = self.pending
        if hop is not None:
            age = now - hop["tick"]
            if bool(obs.get("qbert_in_hop")) or node != hop["origin"]:
                hop["departed"] = True
            if hop["departed"] and standing(obs):
                self.pending = None
                self.router.previous_node = node
                self.neutral_until = max(self.neutral_until, now + 1)
                if node != hop["destination"]:
                    self.router.target = None
                    self.ctx.log("hop_resync", tick=now, expected=hop["destination"], observed=node)
                else:
                    self.ctx.log("hop_arrival", tick=now, origin=hop["origin"], destination=node)
                return self.noop
            if (not hop["departed"] and standing(obs) and age >= 3
                    and hop["retries"] < 1):
                actions = names_in(obs.get("safe_moves"))
                if hop["action"] in actions:
                    hop["retries"] += 1
                    hop["tick"] = now
                    return hop["action"]
                self.pending = None
                return self.noop
            if age > 18:
                self.pending = None
                self.neutral_until = max(self.neutral_until, now + 2)
                return self.noop
            return self.noop

        if now < self.neutral_until or not standing(obs) or node is None:
            return self.noop
        edge = self.choose_edge(obs)
        if edge is None:
            return self.noop
        action, destination = edge
        if action not in names_in(obs.get("safe_moves")):
            return self.noop
        self.pending = {"tick": now, "origin": node, "destination": destination,
                        "action": action, "departed": False, "retries": 0}
        self.last_action_node = node
        if now - self.last_log >= 30:
            self.ctx.log("phase_route", tick=now, score=as_float(obs.get("score")),
                         lives=self.last_lives, level=self.board.level,
                         epoch=self.board.epoch, clears=self.board.clears,
                         unfinished=len(self.board.unfinished()), node=node,
                         target=self.router.target, action=action,
                         target_demand=(self.board.need[self.router.target]
                                        if self.router.target is not None else None),
                         mode=self.router.route_mode)
            self.last_log = now
        return action

    def run(self):
        ctx = self.ctx
        obs = ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.update(obs)
        self.router.start_astra(obs, "initial observed phase plan", self.losses, self.s2_session)
        while not ctx.done and not getattr(self.game, "done", False):
            obs = ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.update(obs)
            if self.request_second_review and self.router.s2_future is None:
                calls_before = self.router.s2_calls
                self.router.start_astra(obs, "review after first observed life loss",
                                        self.losses, self.router.s2_session)
                if self.router.s2_calls > calls_before or self.router.s2_calls >= 2:
                    self.request_second_review = False
            action = self.motor(obs)
            if action not in self.names:
                action = self.noop
            ctx.act(action)
            ctx.wait_for_tick()
        ctx.log("phase_controller_summary", score=getattr(self.game, "score", None),
                ticks=getattr(self.game, "ticks", None), epoch=self.board.epoch,
                inferred_clears=self.board.clears, losses=self.losses,
                route_mode=self.router.route_mode)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
