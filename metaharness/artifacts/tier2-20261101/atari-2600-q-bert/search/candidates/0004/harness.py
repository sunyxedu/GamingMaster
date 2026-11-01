"""A guarded route controller for the real-time Atari 2600 Q*bert port.

The controller deliberately does not let a model hold a joystick edge.  Code
owns the 21-node graph, target bookkeeping, hop timing, and fall prevention.
Jev judges a small local choice at a standing cube and Astra supplies an
optional long-horizon target order.  A model answer is advisory and can never
bypass the graph/safety checks.
"""
from __future__ import annotations

import json
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable
from metaharness.models.jev import Choice, Noul


# The Atari port is a triangular six-row board.  The action names below are
# the public ALE meanings, not the action layout of arcade Q*bert.
DIRECTION_ORDER = ("DOWN", "RIGHT", "UP", "LEFT")
DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")


def _make_graph():
    graph = {n: {} for n in range(21)}
    for row in range(6):
        for col in range(row + 1):
            node = row * (row + 1) // 2 + col
            if row:
                # UP is the upper-right edge; LEFT is the upper-left edge.
                if col <= row - 1:
                    graph[node]["UP"] = (row - 1) * row // 2 + col
                if col:
                    graph[node]["LEFT"] = (row - 1) * row // 2 + col - 1
            if row < 5:
                # DOWN is the lower-left edge; RIGHT is the lower-right edge.
                graph[node]["DOWN"] = (row + 1) * (row + 2) // 2 + col
                graph[node]["RIGHT"] = (row + 1) * (row + 2) // 2 + col + 1
    return graph


GRAPH = _make_graph()


def _as_int(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default=0.0):
    try:
        result = float(value)
        if result != result:  # NaN
            return default
        return result
    except (TypeError, ValueError):
        return default


def _node(value):
    value = _as_int(value)
    return value if value is not None and 0 <= value < 21 else None


def _list_of_names(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [str(x) for x in value if isinstance(x, (str, int))]


def _phase_list(obs):
    phases = obs.get("cube_phase")
    if not isinstance(phases, list) or len(phases) != 21:
        phases = obs.get("cube_state")
    if not isinstance(phases, list):
        return ["unknown"] * 21
    result = [str(x) for x in phases[:21]]
    return (result + ["unknown"] * 21)[:21]


def _needed_list(obs, phases):
    raw = obs.get("needed_landings")
    if isinstance(raw, list) and len(raw) >= 21:
        result = []
        for value in raw[:21]:
            number = _as_int(value, -1)
            result.append(number if number >= 0 else 1)
        return result
    return [0 if phase == "D" else 1 for phase in phases]


def _qbert_standing(obs):
    node = _node(obs.get("qbert_node"))
    if node is None or bool(obs.get("qbert_in_hop")):
        return False
    motion = obs.get("qbert_motion")
    if isinstance(motion, dict):
        phase = str(motion.get("phase", ""))
        if bool(motion.get("falling")) or phase in ("falling", "not_visible"):
            return False
    return True


def _distance_bucket(value):
    distance = _as_float(value, 999.0)
    if distance <= 0:
        return "same_cube"
    if distance <= 20:
        return "adjacent"
    if distance <= 55:
        return "near"
    return "far"


def _hazard_summary(obs):
    hazards = []
    raw = obs.get("hazards")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            hazards.append({
                "kind": str(item.get("kind", "unknown")),
                "node": _node(item.get("node", item.get("estimated_node"))),
                "risk": str(item.get("risk", item.get("risk_label", "unknown"))),
                "distance": _distance_bucket(item.get("distance", item.get("pixel_distance"))),
                "motion": str(item.get("motion", "unknown")),
            })
    # Keep a named object if the adaptor has not put it in hazards.  This is
    # intentionally semantic and small; raw RAM and stale sprite slots stay
    # out of the Jev request.
    for name in ("red_ball", "purple_ball", "coily", "sam", "green_ball"):
        item = obs.get(name)
        if not isinstance(item, dict) or item.get("visible") is False:
            continue
        entry = {"kind": name, "node": _node(item.get("node")),
                 "risk": "visible", "distance": _distance_bucket(item.get("distance")),
                 "motion": str(item.get("motion", "unknown"))}
        if entry not in hazards:
            hazards.append(entry)
    return hazards[:8]


def _json_safe(value):
    """Make a compact strict-JSON value even with odd stand-in observations."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _choice_value(answer):
    return getattr(answer, "value", None) if answer is not None else None


def _answer_probability(answer, key):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return 0.0
    return _as_float(probabilities.get(key), 0.0)


class Harness:
    """A route-following state machine with asynchronous model advisers."""

    # A Jev answer normally arrives in 2-3 ticks.  After this grace period
    # code takes one graph-legal hop rather than leaving Q*bert motionless.
    JEV_WAIT_TICKS = 6
    MAX_ASTRA_CALLS = 6

    def __init__(self, ctx):
        self.ctx = ctx
        self.target = None
        self.plan_targets = []
        self.plan_cursor = 0
        self.epoch = 0
        self.last_lives = None
        self.last_level = None
        self.last_round = None
        self.last_completed_rounds = None
        self.last_life_loss_count = None
        self.pending_jev = None
        self.pending_jev_meta = None
        self.pending_astra = None
        self.astra_session = None
        self.astra_calls_started = 0
        self.decision_action = None
        self.decision_target = None
        # A joystick edge can be lost to ALE sticky-action realization.  Retry
        # it once while the player is still standing, but never keep it armed
        # after a hop starts.
        self.armed_action = None
        self.armed_node = None
        self.armed_epoch = 0
        self.armed_posts = 0
        self.previous_hopping = False
        self.neutral_ticks = 0
        self.last_obs = {}
        self.game_description = ""
        self.names = set()
        self.noop = "NOOP"

    # ---------- observations and deterministic route computation ----------

    def _sync_progress(self, obs):
        """Invalidate model answers on resets, life loss, or round changes."""
        lives = _as_int(obs.get("lives"))
        level = _as_int(obs.get("level"))
        round_number = _as_int(obs.get("round"))
        completed = _as_int(obs.get("completed_rounds"))
        life_losses = _as_int(obs.get("life_lost_count"))
        hopping = bool(obs.get("qbert_in_hop"))
        if self.previous_hopping and not hopping and _node(obs.get("qbert_node")) is not None:
            # Neutralize the first standing tick after a landing.  This gives
            # a sticky previous edge one more chance to be replaced by NOOP.
            self.neutral_ticks = max(self.neutral_ticks, 1)
        self.previous_hopping = hopping
        major = False
        if self.last_lives is not None and lives is not None and lives < self.last_lives:
            major = True
        if self.last_level is not None and level is not None and level != self.last_level:
            major = True
        if self.last_round is not None and round_number is not None and round_number != self.last_round:
            major = True
        if (self.last_completed_rounds is not None and completed is not None
                and completed != self.last_completed_rounds):
            major = True
        if (self.last_life_loss_count is not None and life_losses is not None
                and life_losses > self.last_life_loss_count):
            major = True
        if major:
            self.epoch += 1
            self.target = None
            self.decision_action = None
            self.decision_target = None
            self.armed_action = None
            self.armed_node = None
            self.armed_posts = 0
            self.neutral_ticks = 1
        self.last_lives = lives if lives is not None else self.last_lives
        self.last_level = level if level is not None else self.last_level
        self.last_round = round_number if round_number is not None else self.last_round
        self.last_completed_rounds = completed if completed is not None else self.last_completed_rounds
        self.last_life_loss_count = life_losses if life_losses is not None else self.last_life_loss_count
        return major

    def _incomplete_nodes(self, obs):
        phases = _phase_list(obs)
        needed = _needed_list(obs, phases)
        return [n for n in range(21) if needed[n] > 0 or phases[n] not in ("D", "unknown")]

    def _path(self, start, goal, avoid_dest=False, phases=None):
        start = _node(start)
        goal = _node(goal)
        if start is None or goal is None:
            return []
        if start == goal:
            return []
        queue = deque([start])
        previous = {start: (None, None)}
        while queue:
            current = queue.popleft()
            actions = list(GRAPH.get(current, {}).keys())
            actions.sort(key=lambda a: DIRECTION_ORDER.index(a) if a in DIRECTION_ORDER else 99)
            for action in actions:
                nxt = GRAPH[current][action]
                if nxt in previous:
                    continue
                # In reversible levels, do not use a completed cube as a
                # transit landing unless the route has no alternative.
                if avoid_dest and phases and nxt != goal and phases[nxt] == "D":
                    continue
                previous[nxt] = (current, action)
                if nxt == goal:
                    queue.clear()
                    break
                queue.append(nxt)
        if goal not in previous:
            return self._path(start, goal, avoid_dest=False, phases=phases) if avoid_dest else []
        result = []
        cursor = goal
        while previous[cursor][0] is not None:
            parent, action = previous[cursor]
            result.append(action)
            cursor = parent
        result.reverse()
        return result

    def _candidate_targets(self, obs):
        incomplete = set(self._incomplete_nodes(obs))
        current = _node(obs.get("qbert_node"))
        if not incomplete:
            return []
        candidates = []

        def add(node):
            node = _node(node)
            if node is None or node not in incomplete or node in candidates:
                return
            # A standing Q*bert on a cube just reported as landed should leave
            # it before choosing it again (important for S->I and S->D->S).
            if node == current and str(obs.get("last_event", "")) == "cube":
                return
            candidates.append(node)

        # Keep the current strategic target if it remains unfinished.
        add(self.target)

        # Astra's order is a preference, never a command: only valid,
        # unfinished nodes survive this filter.
        while self.plan_cursor < len(self.plan_targets):
            planned = self.plan_targets[self.plan_cursor]
            if planned in incomplete:
                add(planned)
                break
            self.plan_cursor += 1

        # The port-specific opening clears the upper region before deep
        # coverage.  This is a code prior, not an assumption supplied by Jev.
        level = _as_int(obs.get("level"), 1)
        completed = _as_int(obs.get("completed_rounds"), 0)
        if level == 1 and completed == 0:
            for node in (1, 2, 3, 4, 0):
                add(node)

        phases = _phase_list(obs)
        use_avoid = level in (3, 4, 5)
        scored = []
        for node in incomplete:
            if node in candidates:
                continue
            route = self._path(current, node, avoid_dest=use_avoid, phases=phases)
            row = int(((__import__("math").sqrt(8 * node + 1) - 1) // 2))
            scored.append((len(route) if route else 99, row, node))
        for _, _, node in sorted(scored):
            add(node)
        # If the only unfinished cube is the current one, allow a route to
        # keep its place; the next landing will clear the target state.
        if not candidates and current in incomplete:
            candidates.append(current)
        return candidates[:10]

    def _select_target(self, obs):
        phases = _phase_list(obs)
        needed = _needed_list(obs, phases)
        if self.target is not None:
            if self.target < 0 or self.target >= 21 or needed[self.target] <= 0:
                self.target = None
            elif (_node(obs.get("qbert_node")) == self.target
                  and str(obs.get("last_event", "")) == "cube"):
                self.target = None
        if self.target is None:
            candidates = self._candidate_targets(obs)
            if candidates:
                self.target = candidates[0]
        return self.target

    def _route_action(self, obs, target=None):
        current = _node(obs.get("qbert_node"))
        target = self.target if target is None else _node(target)
        if current is None or target is None or current == target:
            return None
        phases = _phase_list(obs)
        level = _as_int(obs.get("level"), 1)
        route = self._path(current, target, avoid_dest=level in (3, 4, 5), phases=phases)
        preferred = route[0] if route else None
        legal = set(_list_of_names(obs.get("legal_moves")))
        safe = set(_list_of_names(obs.get("safe_moves")))
        available = safe or legal
        if preferred in available:
            return preferred
        # If a hazard blocks the planned edge, choose an available edge whose
        # neighbor has the shortest remaining graph route.
        ranked = []
        for action in available:
            if action not in GRAPH.get(current, {}):
                continue
            nxt = GRAPH[current][action]
            remainder = self._path(nxt, target, avoid_dest=level in (3, 4, 5), phases=phases)
            ranked.append((len(remainder) if remainder else 99, action))
        if ranked:
            ranked.sort(key=lambda pair: (pair[0], DIRECTION_ORDER.index(pair[1])
                                          if pair[1] in DIRECTION_ORDER else 99))
            return ranked[0][1]
        return None

    def _target_descriptions(self, obs):
        current = _node(obs.get("qbert_node"))
        descriptions = {}
        for target in self._candidate_targets(obs):
            route = self._path(current, target)
            route_text = ",".join(route[:6]) if route else "already here"
            descriptions["n" + str(target)] = (
                "Target cube node %d; it still needs a destination landing; "
                "shortest graph route begins %s." % (target, route_text)
            )
        if not descriptions:
            descriptions["wait"] = "No unfinished target is currently established; hold safely."
        return descriptions

    # ---------- compact model interfaces ----------

    def _jev_state(self, obs, proposed, candidates):
        phases = _phase_list(obs)
        needed = _needed_list(obs, phases)
        current = _node(obs.get("qbert_node"))
        target = self.target
        target_path = self._path(current, target)[:6] if target is not None else []
        objects = []
        for hazard in _hazard_summary(obs):
            objects.append(hazard)
        return _json_safe({
            "game": "Atari 2600 Q*bert, six-row 21-cube board",
            "position": {
                "node": current,
                "standing": _qbert_standing(obs),
                "legal_moves": [x for x in _list_of_names(obs.get("legal_moves")) if x in DIRECTIONS],
                "safe_moves": [x for x in _list_of_names(obs.get("safe_moves")) if x in DIRECTIONS],
                "proposed_edge": proposed or "NOOP",
            },
            "progress": {
                "level": _as_int(obs.get("level")),
                "round": _as_int(obs.get("round")),
                "lives": _as_int(obs.get("lives")),
                "unfinished_nodes": [n for n in range(21) if needed[n] > 0],
                "phases": phases,
            },
            "route": {
                "target": target,
                "next_edges": target_path,
                "candidate_targets": candidates,
            },
            "hazards": objects,
            "recent": {
                "last_action": str(obs.get("last_action", "NOOP")),
                "last_event": str(obs.get("last_event", "none")),
                "life_losses": _as_int(obs.get("life_lost_count")),
            },
        })

    def _build_jev_questions(self, obs, proposed, target_desc):
        legal = [x for x in _list_of_names(obs.get("legal_moves")) if x in DIRECTIONS]
        safe = [x for x in _list_of_names(obs.get("safe_moves")) if x in legal]
        allowed = safe or legal
        if proposed in allowed:
            ordered = [proposed] + [x for x in allowed if x != proposed]
        else:
            ordered = list(allowed)
        if not ordered:
            ordered = [self.noop]
        move_criteria_a = {}
        for action in ordered:
            move_criteria_a[action] = (
                "Take exactly this one legal joystick edge for the next hop. "
                "It is a candidate supplied by code; do not hold it after the hop."
            )
        move_criteria_a[self.noop] = "Wait one tick; use this only when every edge is unsafe or uncertain."
        move_criteria_b = dict(reversed(list(move_criteria_a.items())))

        target_criteria_a = dict(target_desc)
        target_criteria_b = dict(reversed(list(target_desc.items())))
        return {
            "move_a": Choice(
                "Choose the safest immediate action from the supplied candidates. "
                "Never choose FIRE. Prefer the proposed route edge only when the "
                "hazard facts do not make it dangerous.",
                move_criteria_a,
            ),
            "move_b": Choice(
                "Independently choose one candidate edge that keeps Q*bert on the "
                "pyramid for the next hop. The state already contains the geometry.",
                move_criteria_b,
            ),
            "danger": Noul(
                "Is the proposed edge likely to cause a lethal edge fall or contact "
                "with a lethal enemy before the next landing? Answer true only for "
                "an immediate lethal risk, not merely a distant visible object."
            ),
            "target_a": Choice(
                "Which supplied unfinished cube should the route pursue next? "
                "Choose wait only if no target is established.", target_criteria_a
            ),
            "target_b": Choice(
                "Independently select the safest supplied unfinished target cube for "
                "coverage. Do not invent a node or an action.", target_criteria_b
            ),
        }

    def _start_jev(self, obs):
        if self.pending_jev is not None:
            return
        if getattr(self.ctx, "system1", None) is None:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if _as_int(remaining.get("system1_calls"), 0) <= 0:
                return
        except Exception:
            pass
        if not _qbert_standing(obs):
            return
        self._select_target(obs)
        proposed = self._route_action(obs)
        candidates = self._candidate_targets(obs)
        state = self._jev_state(obs, proposed, candidates)
        questions = self._build_jev_questions(obs, proposed, self._target_descriptions(obs))
        meta = {
            "epoch": self.epoch,
            "tick": _as_int(obs.get("tick"), 0),
            "frame": _as_int(obs.get("frame"), 0),
            "node": _node(obs.get("qbert_node")),
            "lives": _as_int(obs.get("lives")),
            "level": _as_int(obs.get("level")),
            "round": _as_int(obs.get("round")),
            "proposed": proposed,
            "allowed": list(_list_of_names(obs.get("safe_moves")))
                       or list(_list_of_names(obs.get("legal_moves"))),
        }

        def job():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.80), "error": None}
            except Exception as exc:  # a late/garbled adviser is a normal fallback event
                return {"answers": None, "error": type(exc).__name__}

        try:
            self.pending_jev = self.ctx.spawn(job)
            self.pending_jev_meta = meta
        except Exception as exc:
            self.pending_jev = None
            self.pending_jev_meta = None
            self.ctx.log("jev_spawn_failed", error=type(exc).__name__)

    def _consume_jev(self, obs):
        future = self.pending_jev
        meta = self.pending_jev_meta
        if future is None or not future.done():
            return False
        self.pending_jev = None
        self.pending_jev_meta = None
        try:
            result = future.result()
        except Exception as exc:
            self.ctx.log("jev_result_failed", error=type(exc).__name__)
            return False
        answers = result.get("answers") if isinstance(result, dict) else None
        if not isinstance(answers, dict) or not isinstance(meta, dict):
            return False
        # A decision is a local judgement, not a reusable route command.
        fresh = (
            meta.get("epoch") == self.epoch
            and meta.get("node") == _node(obs.get("qbert_node"))
            and meta.get("lives") == _as_int(obs.get("lives"))
            and meta.get("level") == _as_int(obs.get("level"))
            and meta.get("round") == _as_int(obs.get("round"))
            and _qbert_standing(obs)
            and _as_int(obs.get("tick"), 0) - _as_int(meta.get("tick"), 0) <= self.JEV_WAIT_TICKS + 2
        )
        if not fresh:
            self.ctx.log("jev_stale", origin_tick=meta.get("tick"), current_tick=_as_int(obs.get("tick"), 0))
            return False

        move_a = answers.get("move_a")
        move_b = answers.get("move_b")
        chosen_a = _choice_value(move_a)
        chosen_b = _choice_value(move_b)
        allowed = set(meta.get("allowed", []))
        action = None
        if chosen_a == chosen_b and chosen_a in allowed.union({self.noop}):
            if chosen_a == self.noop or _answer_probability(move_a, chosen_a) >= 0.42:
                action = chosen_a

        danger = answers.get("danger")
        danger_probability = _as_float(getattr(danger, "value", 0.0), 0.0)
        if (meta.get("proposed") in allowed and danger_probability >= 0.64
                and action == meta.get("proposed")):
            action = None
            self.ctx.log("jev_veto", action=meta.get("proposed"), probability=danger_probability)

        target_a = answers.get("target_a")
        target_b = answers.get("target_b")
        target_key_a = _choice_value(target_a)
        target_key_b = _choice_value(target_b)
        if target_key_a == target_key_b and isinstance(target_key_a, str):
            if target_key_a.startswith("n"):
                candidate = _node(target_key_a[1:])
                if (candidate in self._candidate_targets(obs)
                        and _answer_probability(target_a, target_key_a) >= 0.50):
                    self.target = candidate
                    self.decision_target = candidate
        self.decision_action = action
        self.ctx.log("jev_local_decision", action=action or "code_route", target=self.target,
                     proposed=meta.get("proposed"), fresh=True)
        return True

    # ---------- Astra's asynchronous strategic layer ----------

    def _strategic_summary(self, obs):
        phases = _phase_list(obs)
        needed = _needed_list(obs, phases)
        return _json_safe({
            "level": _as_int(obs.get("level")),
            "round": _as_int(obs.get("round")),
            "completed_rounds": _as_int(obs.get("completed_rounds")),
            "lives": _as_int(obs.get("lives")),
            "qbert_node": _node(obs.get("qbert_node")),
            "unfinished_nodes": [n for n in range(21) if needed[n] > 0],
            "cube_phase": phases,
            "recent_event": str(obs.get("last_event", "none")),
            "hazards": _hazard_summary(obs),
        })

    def _astra_prompt(self, obs, reason):
        return (
            "You are the slow strategic planner for Atari 2600 Q*bert, not the "
            "per-tick controller. This is the six-row 21-cube Atari port. The "
            "code controller owns graph legality, exact cube bookkeeping, hop "
            "timing, and lethal fall prevention. Jev will judge local safety. "
            "Choose a conservative order of unfinished node numbers that keeps "
            "an escape route and values completing a round over optional bonuses. "
            "Respect the level cycles S->D, S->I->D, S->D->S, S->I->D->I, and "
            "S->I->D->S. Do not propose FIRE, raw joystick macros, arcade nodes, "
            "or invented board positions. Return ONLY a JSON object with a "
            "`targets` array of integers in 0..20 and an optional `risk` string. "
            "If uncertain, return an empty targets array; code has a safe fallback.\n\n"
            "Why this request was made: " + str(reason) + "\n"
            "Current symbolic observation:\n" + json.dumps(self._strategic_summary(obs), sort_keys=True)
        )

    def _start_astra(self, game, obs, reason):
        if self.pending_astra is not None:
            return
        if getattr(self.ctx, "system2", None) is None:
            return
        if self.astra_calls_started >= self.MAX_ASTRA_CALLS:
            return
        try:
            remaining = self.ctx.budget.remaining()
            if _as_int(remaining.get("system2_calls"), 0) <= 0:
                return
        except Exception:
            pass
        prompt = self._astra_prompt(obs, reason)
        session = self.astra_session
        self.astra_calls_started += 1

        def job():
            try:
                result = self.ctx.think(prompt, session_id=session, timeout_s=35.0)
                return {"text": getattr(result, "text", ""),
                        "session_id": getattr(result, "session_id", None), "error": None}
            except Exception as exc:
                return {"text": "", "session_id": None, "error": type(exc).__name__}

        try:
            self.pending_astra = self.ctx.spawn(job)
            self.ctx.log("astra_started", reason=reason, call=self.astra_calls_started)
        except Exception as exc:
            self.pending_astra = None
            self.ctx.log("astra_spawn_failed", error=type(exc).__name__)

    def _parse_astra_targets(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        candidates = []
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9_+-]*", "", stripped)
            stripped = stripped.replace("```", "").strip()
        candidates.append(stripped)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for blob in candidates:
            try:
                data = json.loads(blob)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            values = data.get("targets", data.get("priority_nodes", data.get("route", [])))
            if not isinstance(values, list):
                continue
            result = []
            for value in values:
                number = _node(value)
                if number is not None and number not in result:
                    result.append(number)
            if result:
                return result[:21]
        return []

    def _consume_astra(self, obs):
        future = self.pending_astra
        if future is None or not future.done():
            return False
        self.pending_astra = None
        try:
            result = future.result()
        except Exception as exc:
            self.ctx.log("astra_result_failed", error=type(exc).__name__)
            return False
        if not isinstance(result, dict):
            return False
        if result.get("session_id"):
            self.astra_session = result.get("session_id")
        targets = self._parse_astra_targets(result.get("text", ""))
        if targets:
            self.plan_targets = targets
            self.plan_cursor = 0
            self.ctx.log("astra_plan_accepted", targets=targets[:10])
            # Do not interrupt a hop or a local target; the next target choice
            # uses this order after the current landing is accounted for.
            return True
        self.ctx.log("astra_plan_ignored", error=result.get("error"))
        return False

    # ---------- episode loop ----------

    def run_episode(self, game) -> None:
        ctx = self.ctx
        spec = game.action_spec
        self.names = {str(x) for x in getattr(spec, "names", [])}
        try:
            candidate_noop = getattr(spec, "noop", "NOOP")
            self.noop = str(candidate_noop) if str(candidate_noop) in self.names else "NOOP"
        except Exception:
            self.noop = "NOOP"
        if self.noop not in self.names and self.names:
            self.noop = next(iter(self.names))
        try:
            self.game_description = str(game.describe())
        except Exception:
            self.game_description = "Atari 2600 Q*bert"

        try:
            obs = game.observe()
        except Exception:
            return
        if not isinstance(obs, dict):
            obs = {}
        self.last_obs = obs
        self._sync_progress(obs)
        self._start_astra(game, obs, "initial route and colour-cycle plan")

        while not ctx.done and not getattr(game, "done", False):
            try:
                obs = game.observe()
            except Exception:
                break
            if not isinstance(obs, dict):
                obs = {}
            major = self._sync_progress(obs)
            self._consume_astra(obs)
            self._consume_jev(obs)
            if major:
                self._start_astra(game, obs, "life loss or round/level transition")

            standing = _qbert_standing(obs)
            action = self.noop
            if standing:
                current_node = _node(obs.get("qbert_node"))
                armed_valid = (
                    self.armed_action in self.names
                    and self.armed_action in DIRECTIONS
                    and self.armed_node == current_node
                    and self.armed_epoch == self.epoch
                    and self.armed_posts < 2
                )
                if armed_valid:
                    # Retry exactly once if sticky realization made the first
                    # edge a NOOP.  A successful edge makes the next
                    # observation a hop, which clears this arm below.
                    action = self.armed_action
                    self.armed_posts += 1
                elif self.armed_action is not None:
                    self.armed_action = None
                    self.armed_node = None
                    self.armed_posts = 0
                    if self.neutral_ticks <= 0:
                        self.decision_action = None
                if self.neutral_ticks > 0 and not armed_valid:
                    action = self.noop
                    self.neutral_ticks -= 1
                elif not armed_valid and self.neutral_ticks <= 0:
                    self._select_target(obs)
                    if self.pending_jev is None:
                        self._start_jev(obs)
                    if self.decision_action in self.names:
                        action = self.decision_action
                        self.decision_action = None
                    elif self.pending_jev is not None and self.pending_jev_meta is not None:
                        age = _as_int(obs.get("tick"), 0) - _as_int(self.pending_jev_meta.get("tick"), 0)
                        if age > self.JEV_WAIT_TICKS:
                            # One graph-legal fallback hop; after it, the old
                            # Jev answer is rejected by its node/epoch freshness tag.
                            action = self._route_action(obs) or self.noop
                    else:
                        action = self._route_action(obs) or self.noop
            else:
                # Never carry a direction through an animation or fall.
                self.armed_action = None
                self.armed_node = None
                self.armed_posts = 0
            # Direction commands are armed for at most two standing ticks;
            # every hop tick itself receives a fresh NOOP.
            if action in DIRECTIONS and action in self.names and not self.armed_action:
                self.armed_action = action
                self.armed_node = _node(obs.get("qbert_node"))
                self.armed_epoch = self.epoch
                self.armed_posts = 1
            if action not in self.names:
                action = self.noop
            try:
                ctx.act(action)
            except (BudgetExceeded, ModelUnavailable):
                return
            except RuntimeError:
                # A valid named action should not fail; return rather than let
                # an infrastructure race turn a normal game end into an error.
                return
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, ModelUnavailable):
                return
            except RuntimeError:
                return


def build(ctx):
    return Harness(ctx)
