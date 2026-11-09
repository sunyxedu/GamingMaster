"""0021: forced-retreat certificates with an all-round threat memory.

The controller keeps a one-hop joystick transaction, but evaluates what happens
when that hop leaves only one way out. Recent rendered rolling-ball sightings
and Coily positions are used as bounded possibility sets, never as asserted
hidden locations. Jev reviews a meaningful next-junction choice while the
current hop runs; code rechecks it against the actual landing state.
"""
from __future__ import annotations

import json
import math
from collections import Counter, deque

from metaharness.models.jev import Choice


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
TIE_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")
COVERAGE = (1, 4, 7, 11, 6, 10, 16, 12, 17, 18, 13, 9, 14, 19, 20,
            8, 2, 0, 3, 5, 15)


def make_board():
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


GRAPH, ROW = make_board()


def make_distances():
    matrix = []
    for start in range(21):
        d = [99] * 21
        d[start] = 0
        frontier = deque([start])
        while frontier:
            here = frontier.popleft()
            for there in GRAPH[here].values():
                if d[there] == 99:
                    d[there] = d[here] + 1
                    frontier.append(there)
        matrix.append(d)
    return matrix


DIST = make_distances()


def as_int(value, default=0):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return default


def as_num(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def as_node(value):
    n = as_int(value, -1)
    return n if 0 <= n < 21 else None


def tick_of(obs):
    return as_int(obs.get("tick", obs.get("ticks", 0)))


def motion(obs):
    value = obs.get("qbert_motion")
    return value if isinstance(value, dict) else {"phase": str(value or "unknown")}


def standing(obs):
    n = as_node(obs.get("qbert_node"))
    phase = str(motion(obs).get("phase", "unknown")).lower()
    return (n is not None and not bool(obs.get("qbert_in_hop"))
            and phase not in ("hopping", "falling", "not_visible")
            and not bool(motion(obs).get("falling")))


def json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def parse_object(text):
    """Find one JSON object in model text; prose/garbage is not a command."""
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text[:12000]):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def choice_value(answer):
    value = getattr(answer, "value", None) if answer is not None else None
    return value if isinstance(value, str) else None


def choice_probability(answer, key):
    probs = getattr(answer, "probabilities", None) if answer is not None else None
    if not isinstance(probs, dict):
        return 0.0
    return as_num(probs.get(key), 0.0)


class CubeLedger:
    """Track decoder landing demand and confirm new boards from palette flashes."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.colors = None
        self.score = None
        self.goal_color = None
        self.need = [1] * 21
        self.flashing = False
        self.flash_score = 0.0
        self.flash_last_tick = -999
        self.flash_started_at = -999
        self.epoch = 0
        self.clears = 0
        self.last_progress = 0
        self.color_events = deque(maxlen=8)

    def _demand(self, obs, colors):
        decoded = obs.get("needed_landings")
        if isinstance(decoded, list) and len(decoded) == 21:
            numbers = [as_int(v, -1) for v in decoded]
            # The adaptor can briefly expose an all-zero vector during the
            # palette/round handoff. Treating that transient as solved caused a
            # permanent NOOP controller state in the smoke trace. Nonzero exact
            # demand is authoritative; an empty vector is cross-checked below.
            if min(numbers, default=-1) >= 0 and any(numbers):
                self.need = numbers
                return
        if colors is not None and self.goal_color is not None:
            derived = [0 if color == self.goal_color else 1 for color in colors]
            # A globally equal palette can be an intermediate/completion handoff
            # rather than an actionable goal. The flash detector owns completion;
            # until a new board is visibly established, retain forward motion.
            self.need = derived if any(derived) else [1] * 21
        else:
            self.need = [1] * 21

    def update(self, obs):
        now = tick_of(obs)
        raw_colors = obs.get("cube_colors")
        colors = None
        if isinstance(raw_colors, list) and len(raw_colors) == 21:
            if all(isinstance(v, int) and not isinstance(v, bool) for v in raw_colors):
                colors = tuple(raw_colors)
        score = as_num(obs.get("score"), self.score if self.score is not None else 0.0)
        delta = score - self.score if self.score is not None else 0.0
        changed = []
        if self.colors is not None and colors is not None:
            changed = [i for i in range(21) if self.colors[i] != colors[i]]

        if len(changed) >= 16 and self.colors is not None:
            if not self.flashing:
                self.flashing = True
                self.flash_score = self.score if self.score is not None else score - delta
                self.flash_started_at = now
                self.ctx.log("retreat_board_flash", tick=now, score=score)
            self.flash_last_tick = now

        if self.flashing:
            quiet = now - self.flash_last_tick
            back_at_top = as_node(obs.get("qbert_node")) == 0
            if quiet >= 4 and (back_at_top or quiet >= 7):
                tally = score - self.flash_score
                if tally >= 1000:
                    self.clears += 1
                self.epoch += 1
                self.flashing = False
                self.goal_color = None
                self.need = [1] * 21
                self.color_events.clear()
                self.last_progress = now
                self.ctx.log("retreat_board_epoch", tick=now, epoch=self.epoch,
                             clears=self.clears, score_tally=tally)
                self.colors, self.score = colors, score
                return True
            self.need = [0] * 21
            self.colors, self.score = colors, score
            return False

        # A destination-colour score change is useful fallback evidence, but the
        # verified per-cube need array remains authoritative when available.
        if (delta > 0 and int(round(delta)) % 100 == 25 and
                0 < len(changed) <= 3 and colors is not None):
            q = as_node(obs.get("qbert_node"))
            landed = q if q in changed else changed[0] if len(changed) == 1 else None
            if landed is not None:
                self.goal_color = colors[landed]
                self.last_progress = now
                phase_list = obs.get("cube_phase")
                phase = phase_list[landed] if isinstance(phase_list, list) and len(phase_list) > landed else None
                self.color_events.append({"tick": now, "node": landed,
                                          "phase": phase, "score_delta": delta})
                self.ctx.log("retreat_cube_evidence", tick=now, node=landed,
                             color=colors[landed], delta=delta)

        self._demand(obs, colors)
        if any(self.need):
            self.last_progress = now if delta > 0 else self.last_progress
        self.colors, self.score = colors, score
        return False


class ThreatBelief:
    """Bounded reachable-node sets from visible enemies and recent ball sightings."""

    LETHAL = {"purple_ball", "red_ball", "coily"}

    def __init__(self):
        self.latest = {}
        self.ball_trail = deque(maxlen=10)
        self.last_signature = ()

    @staticmethod
    def _down(nodes):
        found = set()
        for n in nodes:
            for action in ("DOWN", "RIGHT"):
                there = GRAPH[n].get(action)
                if there is not None:
                    found.add(there)
        return found

    @staticmethod
    def _neighbors(nodes):
        return {there for n in nodes for there in GRAPH[n].values()}

    @staticmethod
    def _reach(source, steps, rolling):
        reached = {source}
        frontier = {source}
        for _ in range(max(0, min(4, steps))):
            frontier = (ThreatBelief._down(frontier) if rolling
                        else ThreatBelief._neighbors(frontier))
            reached.update(frontier)
        return reached

    def clear(self):
        self.latest.clear()
        self.ball_trail.clear()
        self.last_signature = ()

    def update(self, obs, now):
        current = {}
        hazards = obs.get("hazards")
        if isinstance(hazards, list):
            for item in hazards:
                if not isinstance(item, dict) or not item.get("lethal"):
                    continue
                kind = str(item.get("kind", ""))
                if kind not in self.LETHAL:
                    continue
                n = as_node(item.get("estimated_node", item.get("node")))
                if n is None:
                    continue
                previous = self.latest.get(kind)
                if kind in ("purple_ball", "red_ball"):
                    if previous is None or previous.get("node") != n:
                        self.ball_trail.append({"kind": kind, "node": n, "tick": now})
                current[kind] = {"kind": kind, "node": n, "tick": now,
                                 "visible": bool(item.get("visible", True))}
        # Keep the last rendered location as uncertainty, not as a live sprite.
        for kind, item in self.latest.items():
            if kind not in current:
                limit = 42 if kind in ("purple_ball", "red_ball") else 14
                if now - item["tick"] <= limit:
                    remembered = dict(item)
                    remembered["visible"] = False
                    current[kind] = remembered
        self.latest = current
        self.ball_trail = deque((v for v in self.ball_trail if now - v["tick"] <= 46), maxlen=10)
        self.last_signature = tuple(sorted((k, v["node"], v["visible"])
                                           for k, v in current.items()))

    def risk(self, origin, destination, now, horizon=8):
        total = 0.0
        causes = []
        for kind, item in self.latest.items():
            source = item["node"]
            age = max(0, now - item["tick"])
            rolling = kind in ("purple_ball", "red_ball")
            # A ball rolls only down the pyramid. Coily is represented by a
            # small undirected reachability envelope, not an exact pursuit path.
            if rolling:
                steps = min(3, max(0, (age + horizon + 4) // 11))
            else:
                steps = min(2, max(0, (age + horizon + 3) // 9))
            reachable = self._reach(source, steps, rolling)
            if destination in reachable:
                total += (5.0 if age <= 7 else 2.7) if item.get("visible") else 2.2
                causes.append(kind + "_reachable")
            elif rolling and destination in self._down(reachable):
                total += 1.0 if age <= 20 else 0.55
                causes.append(kind + "_next_branch")
            if origin is not None and DIST[source][origin] <= 1:
                total += 1.1 if rolling else 1.6
                causes.append(kind + "_near_takeoff")

        # The trail remains active from the first board. This is only a soft
        # descendant penalty; it never fabricates an enemy at a specific node.
        if destination is not None:
            for sighting in self.ball_trail:
                age = max(0, now - sighting["tick"])
                if age > 46:
                    continue
                depth = min(3, max(0, (age + horizon + 4) // 12))
                cone = self._reach(sighting["node"], depth, True)
                if destination in cone:
                    total += 0.85 if age <= 24 else 0.4
                    causes.append("recent_rolling_cone")
                    break
        return total, causes

    def facts(self, from_node, now):
        result = []
        for kind, item in self.latest.items():
            distance = DIST[item["node"]][from_node] if from_node is not None else 99
            result.append({"kind": kind,
                           "sighting": "visible" if item.get("visible") else "recent_memory",
                           "relation": "same_cube" if distance == 0 else
                                       "adjacent" if distance == 1 else
                                       "near" if distance <= 2 else "distant",
                           "motion": "rolling_down" if kind in ("purple_ball", "red_ball") else "possible_pursuit"})
        recent_balls = [x for x in self.ball_trail if now - x["tick"] <= 20]
        if recent_balls and not any(x["kind"] in ("purple_ball", "red_ball") for x in result):
            result.append({"kind": "recent_rolling_sighting", "sighting": "memory_only",
                           "relation": "location_uncertain", "motion": "downward_possibility"})
        return result


class CorridorRouter:
    """Coverage planner with an explicit certificate for mandatory retreats."""

    def __init__(self, ledger, belief):
        self.ledger = ledger
        self.belief = belief
        self.target = None
        self.cursor = 0
        self.posture = "spine"
        self.leaf_caution = "cautious"
        self.last_nodes = deque(maxlen=20)
        self.bad_landing = Counter()
        self.policy_epoch = 0

    def reset_board(self):
        self.target = None
        self.cursor = 0
        self.last_nodes.clear()
        self.bad_landing.clear()

    def choose_target(self, here):
        remaining = {i for i, count in enumerate(self.ledger.need) if count > 0}
        if self.target not in remaining:
            self.target = None
        # Once Q*bert reaches the active target, don't camp on it when it still
        # needs another phase. Rotate to another unfinished cube; return after
        # covering a different branch. Persistent targeting at the current node
        # produced avoidable short cycles on reversible/multi-landing states.
        if self.target == here and self.ledger.need[here] > 0:
            self.target = None
            self.cursor = (self.cursor + 1) % len(COVERAGE)
        elif self.target is not None:
            return self.target
        for offset in range(len(COVERAGE)):
            index = (self.cursor + offset) % len(COVERAGE)
            candidate = COVERAGE[index]
            if candidate in remaining and (candidate != here or len(remaining) == 1):
                self.target = candidate
                self.cursor = index
                break
        if self.target is None and remaining:
            self.target = min(remaining, key=lambda n: (DIST[here][n], ROW[n], n))
        return self.target

    def note_landing(self, actual, expected=None, origin=None):
        if actual is None:
            return
        self.last_nodes.append(actual)
        if expected is not None and actual != expected and origin is not None:
            self.bad_landing[(origin, expected)] += 1
        if self.target == actual and self.ledger.need[actual] <= 0:
            self.target = None

    def options(self, here, allowed, now):
        if here is None:
            return []
        remaining = [i for i, v in enumerate(self.ledger.need) if v > 0]
        if not remaining:
            return []
        target = self.choose_target(here)
        recent = Counter(self.last_nodes)
        previous = self.last_nodes[-2] if len(self.last_nodes) >= 2 else None
        actions = [a for a in TIE_ORDER if a in allowed and a in GRAPH[here]]
        result = []
        for action in actions:
            dest = GRAPH[here][action]
            exposure, reasons = self.belief.risk(here, dest, now, horizon=8)
            exits = []
            retreat_risk = 0.0
            for next_action, nxt in GRAPH[dest].items():
                edge_risk, _ = self.belief.risk(dest, nxt, now + 8, horizon=8)
                exits.append((next_action, nxt, edge_risk))
                if nxt == here:
                    retreat_risk = edge_risk
            nonreturn = [v for v in exits if v[1] != here]
            certified = sum(1 for _, _, r in nonreturn if r < 3.0)
            best_followup = min((r for _, _, r in exits), default=0.0)
            leaf = len(exits) <= 1
            urgent = len(remaining) <= 2
            # On a tip, the incoming edge is not a meaningful escape count, but
            # it is a forced return that must be safe. This term is deliberately
            # stronger than the ordinary open-exit preference.
            leaf_multiplier = (1.35 if self.leaf_caution == "cautious" else 1.0)
            forced_return_cost = (leaf_multiplier * max(0.0, retreat_risk - 0.4) *
                                  (1.55 if urgent else 2.05)) if leaf else 0.0
            if not leaf and certified == 0:
                forced_return_cost = max(0.0, best_followup - 1.0) * 0.85
            distance = min(DIST[dest][goal] for goal in remaining)
            useful = self.ledger.need[dest] > 0
            target_distance = DIST[dest][target] if target is not None else distance
            cost = 11.5 * min(9.0, exposure) + forced_return_cost
            cost += 1.85 * distance + 0.13 * target_distance
            cost += 0.0 if useful else 0.62
            cost -= 0.22 * min(2, certified)
            cost += min(2.0, 0.11 * recent[dest])
            if dest == previous and len(actions) > 1:
                cost += 0.45
            cost += min(1.6, 0.65 * self.bad_landing[(here, dest)])
            if self.posture == "upper" and useful:
                cost += 0.11 * ROW[dest]
            elif self.posture == "frontier" and useful:
                cost -= 0.10 * ROW[dest]
            # A soft tip penalty ensures a safer cube can be done first, while
            # urgency prevents permanently postponing the final required cubes.
            if leaf and not useful:
                cost += 0.35 if not urgent else 0.12
            result.append({"action": action, "destination": dest, "cost": cost,
                           "risk": exposure, "causes": reasons,
                           "exits": certified, "leaf": leaf,
                           "retreat_risk": retreat_risk,
                           "followup_risk": best_followup,
                           "useful": useful, "distance": distance,
                           "target": target})
        result.sort(key=lambda item: (item["risk"] >= 7.0,
                                      item["cost"],
                                      COVERAGE.index(item["destination"])
                                      if item["destination"] in COVERAGE else 99,
                                      TIE_ORDER.index(item["action"])))
        return result


class QbertRunner:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.ledger = CubeLedger(ctx)
        self.belief = ThreatBelief()
        self.router = CorridorRouter(self.ledger, self.belief)
        self.lives = None
        self.loss_count = None
        self.life_epoch = 0
        self.pending_hop = None
        self.recovery_until = 0
        self.neutral_until = 0
        self.advice_future = None
        self.advice_meta = None
        self.advice = None
        self.last_jev_tick = -999
        self.astra_future = None
        self.astra_session = None
        self.astra_count = 0
        self.astra_reason = "initial symbolic route review"
        self.last_status = -999
        try:
            self.action_names = {str(x) for x in game.action_spec.names}
        except Exception:
            self.action_names = {"NOOP", *DIRECTIONS}
        self.noop = "NOOP" if "NOOP" in self.action_names else next(iter(self.action_names), "NOOP")

    def ingest(self, obs):
        now = tick_of(obs)
        old_epoch = self.ledger.epoch
        self.ledger.update(obs)
        self.belief.update(obs, now)
        lives = as_int(obs.get("lives"), self.lives if self.lives is not None else 4)
        losses = as_int(obs.get("life_lost_count"), self.loss_count if self.loss_count is not None else 0)
        lost = ((self.lives is not None and lives < self.lives) or
                (self.loss_count is not None and losses > self.loss_count))
        if self.ledger.epoch != old_epoch:
            self.router.reset_board()
            self.pending_hop = None
            self.advice = None
            self.neutral_until = max(self.neutral_until, now + 2)
            self.belief.clear()
            self.astra_reason = "board transition: review sustained coverage posture"
            self.ctx.log("retreat_route_reset", tick=now, epoch=self.ledger.epoch,
                         clears=self.ledger.clears)
        if lost:
            self.life_epoch += 1
            self.pending_hop = None
            self.advice = None
            self.router.target = None
            self.recovery_until = max(self.recovery_until, now + 5)
            self.neutral_until = max(self.neutral_until, now + 9)
            self.belief.clear()
            self.astra_reason = "life loss: review forced-retreat safety and coverage"
            self.ctx.log("retreat_life_recovery", tick=now, lives=lives,
                         life_epoch=self.life_epoch, score=as_num(obs.get("score")))
        self.lives, self.loss_count = lives, losses

    def allowed_moves(self, obs, here):
        safe = obs.get("safe_moves")
        legal = obs.get("legal_moves")
        source = safe if isinstance(safe, list) and safe else legal
        if not isinstance(source, list):
            return []
        available = {x for x in source if isinstance(x, str)}
        return [a for a in TIE_ORDER if a in available and a in GRAPH[here]]

    def make_jev_snapshot(self, obs, landing, choices, current_action):
        origin = as_node(obs.get("qbert_node"))
        future = []
        for rank, item in enumerate(choices[:4]):
            future.append({
                "direction": item["action"],
                "landing": "cube_still_needs_a_phase" if item["useful"] else "transit_cube",
                "immediate_threat": "near_or_reachable" if item["risk"] >= 3.0 else
                                    "recent_ball_cone" if "recent_rolling_cone" in item["causes"] else "not_supported",
                "next_escape": "several" if item["exits"] >= 2 else
                               "one" if item["exits"] == 1 else "none_certified",
                "forced_retreat": "risky" if item["leaf"] and item["retreat_risk"] >= 2.0 else
                                  "not_a_tip" if not item["leaf"] else "no_current_signal",
                "code_rank": "first" if rank == 0 else "alternate",
            })
        return json_safe({
            "game": "Atari 2600 Q*bert, six-row 21-cube board",
            "decision_timing": "The current one-hop movement is underway; judge only the next junction.",
            "current_hop": {"action": current_action, "projected_landing": landing},
            "position": {"projected_node": landing, "options": future},
            "threats": self.belief.facts(origin, tick_of(obs)),
            "progress": {"scored_board_clears": self.ledger.clears,
                         "unfinished_bucket": "few" if sum(x > 0 for x in self.ledger.need) <= 2 else
                                              "several" if sum(x > 0 for x in self.ledger.need) <= 8 else "many",
                         "priority": "survive, complete required phases, avoid optional chases"},
            "instruction": "Use only supplied directions. A recent ball cone is uncertainty, not a confirmed occupant. Code rechecks safety and node on arrival."
        })

    def start_jev(self, obs, from_action, landing):
        if self.advice_future is not None or self.ctx.done or landing is None:
            return
        if getattr(self.ctx, "system1", True) is None:
            return
        now = tick_of(obs)
        # Only spend a call when the next junction has a real safety/escape
        # distinction. Equal, clear choices remain deterministic code's job.
        if now - self.last_jev_tick < 18:
            return
        try:
            if self.ctx.budget.remaining().get("system1_calls", 0) <= 2:
                return
        except Exception:
            pass
        saved_route_cursor = self.router.cursor
        saved_route_target = self.router.target
        try:
            future = self.router.options(landing, list(GRAPH[landing]), now + 8)
        finally:
            # Speculation must not consume the live coverage target. In particular,
            # target rotation at a projected arrival is not a real landing.
            self.router.cursor = saved_route_cursor
            self.router.target = saved_route_target
        if len(future) < 2:
            return
        best, second = future[0], future[1]
        meaningful = (abs(best["risk"] - second["risk"]) >= 0.75 or
                      best["exits"] != second["exits"] or
                      best["leaf"] != second["leaf"] or
                      bool(self.belief.facts(landing, now)))
        if not meaningful:
            return
        shown = future[:4]
        keys = [x["action"] for x in shown]
        options = {}
        for item in shown:
            options[item["action"]] = {
                "what": "One supplied legal hop at the next junction.",
                "landing": "unfinished phase" if item["useful"] else "transit",
                "threat": "near_or_reachable" if item["risk"] >= 3.0 else
                          "recent rolling possibility" if "recent_rolling_cone" in item["causes"] else "not supported",
                "escape": "several certified exits" if item["exits"] >= 2 else
                          "one certified exit" if item["exits"] == 1 else "no certified non-return exit",
                "tip_return": "forced return currently risky" if item["leaf"] and item["retreat_risk"] >= 2.0 else
                              "ordinary corridor" if not item["leaf"] else "tip with no current return signal",
            }
        options["keep_code_choice"] = "No clear improvement over the current deterministic corridor ranking."
        instruction = (
            "For the projected landing, choose the safest supplied next hop. Prefer no near/reachable threat, "
            "then a certified escape. Treat a tip's forced return as important; do not choose an uncertain ball "
            "cone as if it were occupied. Prefer unfinished coverage only after safety. Use keep_code_choice when tied."
        )
        criteria_a = {k: options[k] for k in keys + ["keep_code_choice"]}
        criteria_b = {k: options[k] for k in list(reversed(keys)) + ["keep_code_choice"]}
        questions = {
            "next_a": Choice(instruction, criteria_a),
            "next_b": Choice(instruction, criteria_b),
        }
        state = self.make_jev_snapshot(obs, landing, shown, from_action)
        meta = {"created": now, "landing": landing,
                "epoch": self.ledger.epoch, "life": self.life_epoch,
                "policy": self.router.policy_epoch, "keys": keys,
                "ranked": [{k: v for k, v in item.items()} for item in shown]}
        self.last_jev_tick = now

        def call():
            try:
                return self.ctx.decide(state, questions, timeout_s=1.0)
            except Exception as exc:
                return {"_call_error": type(exc).__name__}

        try:
            self.advice_future = self.ctx.spawn(call)
            self.advice_meta = meta
            self.ctx.log("retreat_jev_started", tick=now, landing=landing,
                         candidates=keys)
        except Exception as exc:
            self.advice_future = None
            self.advice_meta = None
            self.ctx.log("retreat_jev_spawn_failed", error=type(exc).__name__)

    def collect_jev(self, obs):
        future = self.advice_future
        if future is None or not future.done():
            return
        meta = self.advice_meta
        self.advice_future = None
        self.advice_meta = None
        try:
            answers = future.result()
            if not isinstance(answers, dict) or "_call_error" in answers or not isinstance(meta, dict):
                self.ctx.log("retreat_jev_invalid")
                return
            a = choice_value(answers.get("next_a"))
            b = choice_value(answers.get("next_b"))
            keys = meta.get("keys", [])
            if a != b or a not in keys + ["keep_code_choice"]:
                self.ctx.log("retreat_jev_disagreement", first=a, second=b)
                return
            probability = min(choice_probability(answers.get("next_a"), a),
                              choice_probability(answers.get("next_b"), b))
            self.advice = {**meta, "choice": a, "probability": probability}
            self.ctx.log("retreat_jev_ready", landing=meta.get("landing"),
                         choice=a, probability=probability)
        except Exception as exc:
            self.ctx.log("retreat_jev_result_failed", error=type(exc).__name__)

    def apply_advice(self, obs, candidates):
        fallback = candidates[0] if candidates else None
        advice = self.advice
        self.advice = None
        if fallback is None or not isinstance(advice, dict):
            return fallback
        now = tick_of(obs)
        here = as_node(obs.get("qbert_node"))
        age = now - as_int(advice.get("created"), now)
        if (here != advice.get("landing") or advice.get("epoch") != self.ledger.epoch or
                advice.get("life") != self.life_epoch or
                advice.get("policy") != self.router.policy_epoch or age < 0 or age > 12):
            self.ctx.log("retreat_jev_expired", tick=now, node=here,
                         expected=advice.get("landing"), age=age)
            return fallback
        key = advice.get("choice")
        if key == "keep_code_choice" or as_num(advice.get("probability")) < 0.66:
            return fallback
        chosen = next((c for c in candidates if c["action"] == key), None)
        if chosen is None:
            return fallback
        improves_risk = chosen["risk"] <= fallback["risk"] - 0.45
        improves_escape = (chosen["risk"] <= fallback["risk"] + 0.15 and
                           chosen["exits"] >= fallback["exits"] + 1 and
                           chosen["cost"] <= fallback["cost"] + 0.8 and
                           (chosen["useful"] or not fallback["useful"]))
        forced_safe = (not chosen["leaf"] or
                       chosen["retreat_risk"] <= fallback["retreat_risk"] + 0.2)
        if (forced_safe and chosen["cost"] <= fallback["cost"] + 1.4 and
                (improves_risk or improves_escape)):
            if chosen["action"] != fallback["action"]:
                self.ctx.log("retreat_jev_changed_route", tick=now,
                             old=fallback["action"], new=chosen["action"],
                             confidence=advice["probability"])
            return chosen
        self.ctx.log("retreat_jev_rejected", tick=now, option=key,
                     risk=chosen["risk"], fallback_risk=fallback["risk"],
                     forced_return=chosen["retreat_risk"])
        return fallback

    def start_astra(self, obs, reason):
        if self.astra_future is not None or self.astra_count >= 2 or self.ctx.done:
            return
        if getattr(self.ctx, "system2", True) is None:
            return
        try:
            if self.ctx.budget.remaining().get("system2_calls", 0) <= 0:
                return
        except Exception:
            pass
        now = tick_of(obs)
        summary = json_safe({
            "reason": reason, "tick": now, "score": as_num(obs.get("score")),
            "lives": self.lives, "inferred_clears": self.ledger.clears,
            "unfinished_cubes": sum(x > 0 for x in self.ledger.need),
            "recent_cube_events": list(self.ledger.color_events)[-4:],
            "public_threats": self.belief.facts(as_node(obs.get("qbert_node")), now),
            "current_policy": {"coverage_posture": self.router.posture,
                               "leaf_caution": self.router.leaf_caution}})
        prompt = (
            "You are the slow strategic adviser to a real-time Atari 2600 Q*bert harness. "
            "Code owns the 21-node graph, exact decoded cube demand, short corridor search and one-hop "
            "neutral/landing handshake. A special issue is entering a pyramid tip whose only exit is the "
            "forced return edge; the controller predicts that edge's risk before committing. Jev reviews "
            "only a next-junction choice while a hop runs, and code revalidates it. Choose a durable policy: "
            "coverage_posture is spine, upper, or frontier; leaf_caution is cautious or balanced. "
            "spine keeps the stable coverage tour; upper favors shallow unfinished cubes; frontier favors "
            "deeper cubes. cautious gives extra weight to an unsafe mandatory return; balanced is less deferring. "
            "Favor survival and completion over optional bonuses. Do not invent hazards, actions, routes, "
            "node numbers, or FIRE. Return only compact JSON, e.g. "
            '{"coverage_posture":"spine","leaf_caution":"cautious"}. Invalid text is ignored.\n' +
            json.dumps(summary, separators=(",", ":")))
        session = self.astra_session
        self.astra_count += 1
        self.astra_reason = None

        def call():
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=105.0)
                return {"text": getattr(result, "text", ""),
                        "session": getattr(result, "session_id", None)}
            except Exception as exc:
                return {"error": type(exc).__name__}

        try:
            self.astra_future = self.ctx.spawn(call)
            self.ctx.log("retreat_astra_started", tick=now, reason=reason)
        except Exception as exc:
            self.astra_future = None
            self.ctx.log("retreat_astra_spawn_failed", error=type(exc).__name__)

    def collect_astra(self, obs):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
            if not isinstance(result, dict) or "error" in result:
                self.ctx.log("retreat_astra_invalid")
                return
            if result.get("session"):
                self.astra_session = result["session"]
            policy = parse_object(result.get("text", ""))
            before = (self.router.posture, self.router.leaf_caution)
            if policy.get("coverage_posture") in ("spine", "upper", "frontier"):
                self.router.posture = policy["coverage_posture"]
            if policy.get("leaf_caution") in ("cautious", "balanced"):
                self.router.leaf_caution = policy["leaf_caution"]
            after = (self.router.posture, self.router.leaf_caution)
            if before != after:
                self.router.policy_epoch += 1
                self.advice = None
                self.ctx.log("retreat_astra_policy_changed", tick=tick_of(obs),
                             old=list(before), new=list(after))
            else:
                self.ctx.log("retreat_astra_policy_unchanged", tick=tick_of(obs),
                             parsed=bool(policy))
        except Exception as exc:
            self.ctx.log("retreat_astra_result_failed", error=type(exc).__name__)

    def decide_action(self, obs):
        now = tick_of(obs)
        self.collect_jev(obs)
        self.collect_astra(obs)
        if self.ledger.flashing:
            self.pending_hop = None
            return self.noop

        here = as_node(obs.get("qbert_node"))
        move = motion(obs)
        phase = str(move.get("phase", "unknown")).lower()
        if here is None or phase in ("falling", "not_visible") or move.get("falling"):
            self.pending_hop = None
            return self.noop
        if now < self.recovery_until:
            return self.noop
        if self.recovery_until and now >= self.recovery_until:
            self.recovery_until = 0

        # A direction is posted only to initiate a hop. Neutral is held until a
        # standing landing is seen, preventing sticky input from chaining edges.
        if self.pending_hop is not None:
            hop = self.pending_hop
            age = now - hop["started"]
            if bool(obs.get("qbert_in_hop")) or here != hop["origin"]:
                hop["moving"] = True
            if hop["moving"] and standing(obs):
                actual = here
                matched = actual == hop["destination"]
                self.router.note_landing(actual, hop["destination"], hop["origin"])
                self.pending_hop = None
                self.neutral_until = max(self.neutral_until, now + 1)
                self.ctx.log("retreat_hop_landed" if matched else "retreat_hop_mismatch",
                             tick=now, origin=hop["origin"], expected=hop["destination"],
                             observed=actual, duration=age)
                return self.noop
            if not hop["moving"] and standing(obs) and age <= 3 and hop["posts"] < 2:
                hop["posts"] += 1
                return hop["action"]
            if age > 18 and standing(obs):
                self.pending_hop = None
                self.neutral_until = max(self.neutral_until, now + 3)
                self.ctx.log("retreat_hop_not_started", tick=now,
                             origin=hop["origin"], expected=hop["destination"])
            else:
                return self.noop

        if now < self.neutral_until or not standing(obs):
            return self.noop
        allowed = self.allowed_moves(obs, here)
        options = self.router.options(here, allowed, now)
        if not options:
            return self.noop
        chosen = self.apply_advice(obs, options) or options[0]
        action = chosen["action"]
        if action not in DIRECTIONS or action not in allowed:
            return self.noop
        destination = GRAPH[here].get(action)
        if destination is None:
            return self.noop
        self.pending_hop = {"origin": here, "destination": destination,
                            "action": action, "started": now, "posts": 1,
                            "moving": False}
        self.start_jev(obs, action, destination)
        if now - self.last_status >= 25:
            self.ctx.log("retreat_route_choice", tick=now,
                         score=as_num(obs.get("score")), lives=self.lives,
                         clears=self.ledger.clears,
                         unfinished=sum(v > 0 for v in self.ledger.need),
                         node=here, target=self.router.target,
                         action=action, destination=destination,
                         immediate_risk=chosen["risk"], exits=chosen["exits"],
                         tip=chosen["leaf"], forced_return_risk=chosen["retreat_risk"],
                         causes=chosen["causes"])
            self.last_status = now
        return action

    def run(self):
        obs = self.ctx.observe()
        if not isinstance(obs, dict):
            obs = {}
        self.ingest(obs)
        self.start_astra(obs, self.astra_reason or "initial symbolic route review")
        while not self.ctx.done and not getattr(self.game, "done", False):
            action = self.decide_action(obs)
            if self.astra_reason:
                self.start_astra(obs, self.astra_reason)
            if action not in self.action_names:
                action = self.noop
            try:
                self.ctx.act(action)
                self.ctx.wait_for_tick()
            except Exception:
                if self.ctx.done or getattr(self.game, "done", False):
                    return
                raise
            if self.ctx.done or getattr(self.game, "done", False):
                break
            obs = self.ctx.observe()
            if not isinstance(obs, dict):
                obs = {}
            self.ingest(obs)
        self.ctx.log("retreat_episode_summary", score=getattr(self.game, "score", None),
                     clears=self.ledger.clears, epochs=self.ledger.epoch,
                     lives=self.lives, posture=self.router.posture,
                     leaf_caution=self.router.leaf_caution)


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        QbertRunner(self.ctx, game).run()


def build(ctx):
    return HarnessProgram(ctx)
