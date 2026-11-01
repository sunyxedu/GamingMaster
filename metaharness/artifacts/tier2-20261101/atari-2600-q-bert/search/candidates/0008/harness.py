"""Closed-loop Q*bert controller with an explicit completion handoff.

The previous controllers were good at the first sweep, but they trusted the
palette-derived cube ledger during the cartridge's post-clear flash.  This
controller treats that interval as a separate mode and keeps a small landing
ledger of its own.  It emits one joystick pulse per hop and neutral input while
an animation is in progress.  Jev is an optional, fresh local safety adviser;
its output can choose among code-supplied safe edges, but it cannot create an
edge or bypass the transition state machine.
"""

from collections import deque

from metaharness.models.jev import Choice, Noul


DIRECTIONS = ("UP", "RIGHT", "LEFT", "DOWN")


def _graph():
    """Return the row-major six-row graph and its public joystick labels."""
    result = {n: {} for n in range(21)}
    for row in range(6):
        first = row * (row + 1) // 2
        for col in range(row + 1):
            node = first + col
            if row:
                parent_first = (row - 1) * row // 2
                if col < row:
                    result[node]["UP"] = parent_first + col
                if col:
                    result[node]["LEFT"] = parent_first + col - 1
            if row < 5:
                child_first = (row + 1) * (row + 2) // 2
                result[node]["DOWN"] = child_first + col
                result[node]["RIGHT"] = child_first + col + 1
    return result


GRAPH = _graph()

# This is a useful deterministic tie-breaker, not a route command.  It clears
# the upper region early and then sweeps toward the lower rows.  The route is
# recomputed from observations, so a sticky action or an enemy never makes it a
# blind macro.
OPENING_ORDER = (
    1, 2, 4, 3, 5, 8, 12, 7, 6, 10, 15,
    16, 11, 17, 18, 13, 9, 14, 19, 20,
)
OPENING_RANK = {node: rank for rank, node in enumerate(OPENING_ORDER)}


def _int(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _number(value, default=0.0):
    try:
        answer = float(value)
        if answer != answer or answer in (float("inf"), float("-inf")):
            return default
        return answer
    except (TypeError, ValueError, OverflowError):
        return default


def _node(value):
    value = _int(value)
    return value if value is not None and 0 <= value < 21 else None


def _names(value, allowed=DIRECTIONS):
    if not isinstance(value, (list, tuple)):
        return []
    allowed = set(allowed)
    return [item for item in value if isinstance(item, str) and item in allowed]


def _phase_values(observation):
    raw = observation.get("cube_phase")
    if not isinstance(raw, list) or len(raw) != 21:
        raw = observation.get("cube_state")
    if not isinstance(raw, list):
        return ["unknown"] * 21
    values = []
    for item in raw[:21]:
        text = str(item).upper()
        values.append(text if text in ("S", "I", "D") else "unknown")
    return (values + ["unknown"] * 21)[:21]


def _needed_values(observation):
    raw = observation.get("needed_landings")
    if not isinstance(raw, list) or len(raw) != 21:
        return None
    answer = []
    for item in raw:
        value = _int(item, -1)
        answer.append(value if value >= 0 else None)
    return answer


def _json_value(value):
    """Make model state strict JSON even with hostile stand-in observations."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _answer_value(answer):
    return getattr(answer, "value", None) if answer is not None else None


def _probability(answer, key):
    probabilities = getattr(answer, "probabilities", None)
    if not isinstance(probabilities, dict):
        return 0.0
    return _number(probabilities.get(key), 0.0)


def _hazard_facts(observation):
    facts = []
    raw = observation.get("hazards")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            facts.append({
                "kind": str(item.get("kind", "unknown")),
                "lethal": bool(item.get("lethal", False)),
                "risk": str(item.get("risk", item.get("risk_label", "unknown"))),
                "distance": str(item.get("distance", item.get("pixel_distance", "unknown"))),
                "node": _node(item.get("estimated_node", item.get("node"))),
            })
    return facts[:8]


def _standing(observation):
    node = _node(observation.get("qbert_node"))
    if node is None:
        return False
    if bool(observation.get("qbert_in_hop")):
        return False
    motion = observation.get("qbert_motion")
    if isinstance(motion, dict):
        phase = str(motion.get("phase", "")).lower()
        if phase in ("hopping", "falling", "not_visible"):
            return False
        if bool(motion.get("falling")):
            return False
    return True


def _all_destination(observation):
    needed = _needed_values(observation)
    if needed is not None and all(value == 0 for value in needed):
        return True
    destination = _int(observation.get("destination_cubes"), -1)
    incomplete = _int(observation.get("incomplete_cubes"), -1)
    return destination >= 21 or incomplete == 0


def _all_start(observation):
    needed = _needed_values(observation)
    if needed is not None and all(value is not None and value > 0 for value in needed):
        return True
    return (
        _int(observation.get("destination_cubes"), -1) == 0
        and _int(observation.get("incomplete_cubes"), -1) == 21
    )


def _bfs_path(start, goal):
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
        # Stable direction order keeps the action choice reproducible without
        # making the controller a hard-coded action macro.
        edges = GRAPH.get(current, {})
        for action in ("UP", "RIGHT", "LEFT", "DOWN"):
            nxt = edges.get(action)
            if nxt is None or nxt in previous:
                continue
            previous[nxt] = (current, action)
            if nxt == goal:
                queue.clear()
                break
            queue.append(nxt)
    if goal not in previous:
        return []
    path = []
    current = goal
    while previous[current][0] is not None:
        parent, action = previous[current]
        path.append(action)
        current = parent
    path.reverse()
    return path


def _distance(start, goal):
    path = _bfs_path(start, goal)
    return len(path) if path or _node(start) == _node(goal) else 99


class RoundLedger:
    """Own the useful part of cube progress, while using the adaptor to audit it."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.mode = "play"
        self.generation = 0
        self.level = 1
        self.remaining = [1] * 21
        self.phase_index = [0] * 21
        self.goal = None
        self.waypoint = None
        self.transition_start = None
        self.transition_node = None
        self.transition_probes = 0
        self.last_tick = -1
        self.completion_armed = True
        self.last_lives = None
        self.last_round = None
        self.last_level = None
        self.last_completed_rounds = None
        self.last_cube_count = None
        self.last_cube_key = None
        self.ignore_observed_until = [0] * 21
        self.recent_event = "none"

    def _required(self, level):
        # Number of landings from the starting colour to the first D phase.
        return {1: 1, 2: 2, 3: 1, 4: 2, 5: 2}.get(level, 2)

    def _set_from_observation(self, observation, force=False):
        needed = _needed_values(observation)
        phases = _phase_values(observation)
        if needed is not None and (force or any(value is not None for value in needed)):
            if force:
                self.remaining = [value if value is not None else self._required(self.level)
                                  for value in needed]
            else:
                # Never erase a code-owned obligation on a transient palette
                # frame.  Positive observations can reveal Sam/reversion.
                for index, value in enumerate(needed):
                    if value is not None and value > self.remaining[index]:
                        self.remaining[index] = value
        for index, phase in enumerate(phases):
            if phase == "S":
                self.phase_index[index] = 0
            elif phase == "I":
                self.phase_index[index] = 1
            elif phase == "D":
                self.phase_index[index] = 2

    def _reset_for_board(self, observation):
        level = _int(observation.get("level"), self.level) or self.level
        self.level = max(1, min(5, level))
        required = self._required(self.level)
        self.remaining = [required] * 21
        self.phase_index = [0] * 21
        if not (_all_destination(observation) or _all_start(observation)):
            self._set_from_observation(observation, force=False)
        self.completion_armed = False
        # A post-clear RGB flash can report a nearly empty board.  The own
        # ledger deliberately keeps every cube as a candidate in that case.
        self.goal = None
        self.waypoint = None
        self.generation += 1
        self.ignore_observed_until = [0] * 21

    def _apply_landing(self, node, observation, tick):
        if node is None or self.mode != "play":
            return
        level = _int(observation.get("level"), self.level) or self.level
        self.level = max(1, min(5, level))
        current = self.phase_index[node]
        if self.level == 1:
            self.remaining[node] = 0
            self.phase_index[node] = 2
        elif self.level == 2:
            self.remaining[node] = max(0, self.remaining[node] - 1)
            self.phase_index[node] = min(2, current + 1)
        elif self.level == 3:
            # S,D,S: a D landing is useful, the following landing undoes it.
            new_phase = 1 if current != 1 else 0
            self.phase_index[node] = new_phase
            self.remaining[node] = 0 if new_phase == 1 else 1
        elif self.level == 4:
            # S,I,D,I: the first two landings reach D; later ones require a
            # return landing after the cycle has moved back to I.
            new_phase = (current + 1) % 3
            self.phase_index[node] = new_phase
            self.remaining[node] = 0 if new_phase == 2 else 1
        else:
            # S,I,D,S: use the repeating three-state phase directly.
            new_phase = (current + 1) % 3
            self.phase_index[node] = new_phase
            self.remaining[node] = 0 if new_phase == 2 else (1 if new_phase == 1 else 2)
        self.ignore_observed_until[node] = tick + 5
        if self.goal == node:
            self.goal = None
            self.waypoint = None

    def _observe_events(self, observation, tick):
        counters = observation.get("event_counts")
        cube_count = None
        if isinstance(counters, dict):
            cube_count = _int(counters.get("cube"))
        new_cube = (
            cube_count is not None
            and self.last_cube_count is not None
            and cube_count > self.last_cube_count
        )
        event = str(observation.get("last_event", "none"))
        node = _node(observation.get("qbert_node"))
        key = (tick, cube_count, node, event)
        if new_cube or (event == "cube" and key != self.last_cube_key):
            self._apply_landing(node, observation, tick)
            self.last_cube_key = key
        if cube_count is not None:
            self.last_cube_count = cube_count
        self.recent_event = event

    def _entered_new_round(self, observation):
        level = _int(observation.get("level"))
        round_number = _int(observation.get("round"))
        completed = _int(observation.get("completed_rounds"))
        changed = False
        if self.last_level is not None and level is not None and level != self.last_level:
            changed = True
        if self.last_round is not None and round_number is not None and round_number != self.last_round:
            changed = True
        if (self.last_completed_rounds is not None and completed is not None
                and completed != self.last_completed_rounds):
            changed = True
        if changed:
            self.mode = "play"
            self._reset_for_board(observation)
        self.last_level = level if level is not None else self.last_level
        self.last_round = round_number if round_number is not None else self.last_round
        self.last_completed_rounds = completed if completed is not None else self.last_completed_rounds
        lives = _int(observation.get("lives"))
        if self.last_lives is not None and lives is not None and lives < self.last_lives:
            self.goal = None
            self.waypoint = None
            # A life loss can restore Q*bert at a different node, but it does
            # not prove that the pyramid was reset.  Keep the landing ledger
            # and merge positive observations below.
        self.last_lives = lives if lives is not None else self.last_lives
        return changed

    def _flash_like(self, observation):
        event = str(observation.get("last_event", "none"))
        node = _node(observation.get("qbert_node"))
        return (
            event in ("green_ball", "round", "none")
            and (self.transition_node is None or node == self.transition_node)
            and (_all_destination(observation) or _all_start(observation))
            and not bool(observation.get("qbert_in_hop"))
        )

    def _begin_transition(self, observation, tick):
        if self.mode != "transition":
            self.mode = "transition"
            self.transition_start = tick
            self.transition_node = _node(observation.get("qbert_node"))
            self.transition_probes = 0
            self.goal = None
            self.waypoint = None
            self.ctx.log("round_handoff_started", tick=tick, node=self.transition_node)

    def _maybe_finish_transition(self, observation, tick):
        if self.mode != "transition":
            return False
        start = self.transition_start if self.transition_start is not None else tick
        age = max(0, tick - start)
        node = _node(observation.get("qbert_node"))
        moving = bool(observation.get("qbert_in_hop"))
        event = str(observation.get("last_event", "none"))
        # The ROM's post-clear animation is observed as alternating all-D and
        # all-S palette frames with repeated 100-point events.  Do not let one
        # such frame resurrect a one-cube route.
        if self._flash_like(observation) and age < 55:
            return False
        active = (
            age >= 5
            and (moving or event == "cube" or
                 (node is not None and node != self.transition_node))
        )
        # A safety valve is needed if the decoder suppresses the first active
        # frame.  It is reached only after the bounded flash interval.
        if not active and age >= 55:
            active = True
        if active:
            self._reset_for_board(observation)
            self.mode = "play"
            self.transition_start = None
            self.transition_node = None
            self.ctx.log("round_handoff_finished", tick=tick, generation=self.generation)
            return True
        return False

    def ingest(self, observation):
        if not isinstance(observation, dict):
            return
        tick = _int(observation.get("tick", observation.get("ticks")), self.last_tick + 1)
        if tick is None:
            tick = self.last_tick + 1
        self.last_tick = max(self.last_tick, tick)
        if self.last_cube_count is None:
            counters = observation.get("event_counts")
            if isinstance(counters, dict):
                self.last_cube_count = _int(counters.get("cube"))
            self.last_level = _int(observation.get("level"), 1)
            self.last_round = _int(observation.get("round"))
            self.last_completed_rounds = _int(observation.get("completed_rounds"))
            self.level = max(1, min(5, self.last_level or 1))
            initial = _needed_values(observation)
            if initial is not None and any(value is not None for value in initial):
                self.remaining = [value if value is not None else self._required(self.level)
                                  for value in initial]
            self._set_from_observation(observation, force=False)
        self._entered_new_round(observation)
        self._maybe_finish_transition(observation, tick)
        self._observe_events(observation, tick)

        # Merge only positive evidence.  The special all-D/all-S flash is not
        # allowed to erase or recreate the own ledger.
        if self.mode == "play" and not self._flash_like(observation):
            needed = _needed_values(observation)
            if needed is not None:
                event = str(observation.get("last_event", "none"))
                for index, value in enumerate(needed):
                    if value is None or tick < self.ignore_observed_until[index]:
                        continue
                    # A normal palette observation is not evidence that a
                    # just-landed cube needs work again; it is often sampled
                    # before the ROM's landing animation settles.  Only an
                    # explicit reversal/life-loss event can add obligations
                    # back to the code-owned ledger.
                    if event in ("sam", "life_lost") and value > self.remaining[index]:
                        self.remaining[index] = value

        # The own ledger is the completion signal.  The adaptor's all-D
        # palette can be a stale frame from the previous board, so it is never
        # sufficient by itself after a handoff.
        if (self.mode == "play"
                and not bool(observation.get("qbert_in_hop"))
                and not any(self.remaining)):
            self._begin_transition(observation, tick)

    def unfinished(self):
        return [index for index, value in enumerate(self.remaining) if value > 0]

    def _choose_detour(self, current, safe):
        options = []
        for action in safe:
            destination = GRAPH.get(current, {}).get(action)
            if destination is None:
                continue
            back = _bfs_path(destination, current)
            options.append((len(back) if back else 99, OPENING_RANK.get(destination, 99), action, destination))
        if not options:
            return None
        options.sort()
        return options[0][3]

    def target_for(self, observation):
        current = _node(observation.get("qbert_node"))
        if current is None:
            return None
        remaining = set(self.unfinished())
        if not remaining:
            return None

        if self.goal in remaining and self.goal != current:
            return self.goal
        if self.goal not in remaining:
            self.goal = None
            self.waypoint = None

        # A target that is the current just-landed cube needs a detour before
        # it can be landed on again (important for levels 2, 3, 4 and 5).
        if current in remaining and self.goal == current:
            if self.waypoint is None:
                safe = _names(observation.get("safe_moves"))
                self.waypoint = self._choose_detour(current, safe)
            return self.goal

        candidates = [node for node in remaining if node != current]
        if candidates:
            candidates.sort(key=lambda node: (
                _distance(current, node),
                OPENING_RANK.get(node, 99),
                node,
            ))
            self.goal = candidates[0]
            self.waypoint = None
            return self.goal

        # Only the current cube remains.  Leave it through a certified edge,
        # then come back, instead of oscillating between two newly selected
        # goals.
        self.goal = current
        if self.waypoint is None:
            self.waypoint = self._choose_detour(current, _names(observation.get("safe_moves")))
        return self.goal

    def route_action(self, observation):
        current = _node(observation.get("qbert_node"))
        if current is None:
            return None
        safe = _names(observation.get("safe_moves"))
        if not safe:
            return None
        target = self.target_for(observation)
        if target is None:
            return None

        effective = self.waypoint if self.waypoint is not None and current != self.waypoint else target
        if self.waypoint is not None and current == self.waypoint:
            self.waypoint = None
            effective = target
        if effective == current:
            return None
        path = _bfs_path(current, effective)
        if path and path[0] in safe:
            return path[0]

        # A hazard can remove the planned first edge between observations.  A
        # safe adjacent edge that shortens the same route is preferable to
        # abandoning the ledger or inventing a direction.
        alternatives = []
        for action in safe:
            destination = GRAPH.get(current, {}).get(action)
            if destination is None:
                continue
            tail = _bfs_path(destination, effective)
            alternatives.append((len(tail) if tail else 99,
                                OPENING_RANK.get(destination, 99), action))
        if alternatives:
            alternatives.sort()
            return alternatives[0][2]
        return safe[0]


class Controller:
    def __init__(self, ctx, game):
        self.ctx = ctx
        self.game = game
        self.ledger = RoundLedger(ctx)
        self.names = set()
        self.noop = "NOOP"
        self.pending_jev = None
        self.pending_meta = None
        self.advised_action = None
        self.advised_meta = None
        self.last_jev_tick = -99
        self.was_hopping = False
        self.release_ticks = 0
        self.pulse_action = None
        self.pulse_node = None
        self.pulse_attempts = 0
        self.last_obs = {}

        try:
            self.names = {str(item) for item in game.action_spec.names}
        except Exception:
            self.names = {"NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN"}
        try:
            candidate = str(getattr(game.action_spec, "noop", "NOOP"))
            if candidate in self.names:
                self.noop = candidate
        except Exception:
            pass
        if self.noop not in self.names:
            self.noop = "NOOP" if "NOOP" in self.names else (next(iter(self.names)) if self.names else "NOOP")

    def _compact_state(self, observation, proposed):
        current = _node(observation.get("qbert_node"))
        goal = self.ledger.goal
        target = self.ledger.waypoint if self.ledger.waypoint is not None else goal
        return _json_value({
            "game": "Atari 2600 Q*bert; six rows, 21 cubes; no FIRE during play",
            "mode": self.ledger.mode,
            "position": {
                "node": current,
                "standing": _standing(observation),
                "safe_moves": _names(observation.get("safe_moves")),
                "legal_moves": _names(observation.get("legal_moves")),
                "proposed": proposed or self.noop,
            },
            "progress": {
                "level": _int(observation.get("level"), self.ledger.level),
                "round": _int(observation.get("round")),
                "lives": _int(observation.get("lives")),
                "unfinished_nodes": self.ledger.unfinished(),
                "goal": goal,
                "next_waypoint": target,
            },
            "hazards": _hazard_facts(observation),
            "recent": {
                "event": str(observation.get("last_event", "none")),
                "last_action": str(observation.get("last_action", self.noop)),
                "qbert_motion": str((observation.get("qbert_motion") or {}).get("phase", "unknown"))
                    if isinstance(observation.get("qbert_motion"), dict) else "unknown",
            },
        })

    def _start_jev(self, observation, proposed):
        if self.pending_jev is not None or proposed is None:
            return
        if getattr(self.ctx, "system1", None) is None:
            return
        tick = _int(observation.get("tick", observation.get("ticks")), 0)
        if tick - self.last_jev_tick < 5:
            return
        if not _standing(observation) or self.ledger.mode != "play":
            return
        safe = _names(observation.get("safe_moves"))
        options = list(dict.fromkeys(safe + [self.noop]))
        if not options:
            return
        # Two independent choices with reversed criteria reduce the provider's
        # first-option bias.  Code accepts only agreement and validates again.
        first = [proposed] + [item for item in options if item != proposed]
        second = list(reversed(first))
        criteria_a = {
            item: {
                "what": "Post exactly this one certified joystick edge for the next hop.",
                "not_for": "An edge absent from position.safe_moves or FIRE.",
            }
            for item in first
        }
        criteria_b = {
            item: {
                "what": "Choose this candidate only if it keeps Q*bert on the pyramid immediately.",
                "not_for": "A speculative bonus chase or an edge not listed as safe.",
            }
            for item in second
        }
        state = self._compact_state(observation, proposed)
        questions = {
            "move_a": Choice(
                "Choose the safest useful immediate action from the supplied candidates. "
                "The code-computed safe_moves is authoritative; never invent an action.",
                criteria_a,
            ),
            "move_b": Choice(
                "Independently choose one safe immediate action. Prefer the proposed route "
                "unless a visible lethal hazard makes it unsafe. Do not choose FIRE.",
                criteria_b,
            ),
            "danger": Noul(
                "Is the proposed action likely to cause a lethal enemy contact or edge fall "
                "before the next landing? Answer true only for an immediate danger, not a "
                "distant visible enemy."
            ),
        }
        meta = {
            "tick": tick,
            "node": _node(observation.get("qbert_node")),
            "generation": self.ledger.generation,
            "proposed": proposed,
            "safe": list(safe),
        }
        self.last_jev_tick = tick

        def job():
            try:
                return {"answers": self.ctx.decide(state, questions, timeout_s=0.36), "error": None}
            except Exception as exc:
                return {"answers": None, "error": type(exc).__name__}

        try:
            self.pending_jev = self.ctx.spawn(job)
            self.pending_meta = meta
        except Exception as exc:
            self.pending_jev = None
            self.pending_meta = None
            try:
                self.ctx.log("jev_spawn_failed", error=type(exc).__name__)
            except Exception:
                pass

    def _consume_jev(self, observation):
        future = self.pending_jev
        meta = self.pending_meta
        if future is None or not future.done():
            return
        self.pending_jev = None
        self.pending_meta = None
        try:
            result = future.result()
        except Exception as exc:
            try:
                self.ctx.log("jev_result_failed", error=type(exc).__name__)
            except Exception:
                pass
            return
        if not isinstance(result, dict) or not isinstance(meta, dict):
            return
        tick = _int(observation.get("tick", observation.get("ticks")), 0)
        age = tick - _int(meta.get("tick"), tick)
        current = _node(observation.get("qbert_node"))
        safe = set(_names(observation.get("safe_moves")))
        answers = result.get("answers")
        if (not isinstance(answers, dict) or age < 0 or age > 4
                or not _standing(observation)
                or current != meta.get("node")
                or self.ledger.generation != meta.get("generation")):
            return
        first = answers.get("move_a")
        second = answers.get("move_b")
        choice_a = _answer_value(first)
        choice_b = _answer_value(second)
        chosen = None
        if choice_a == choice_b and (choice_a in safe or choice_a == self.noop):
            if choice_a == self.noop or _probability(first, choice_a) >= 0.48:
                chosen = choice_a
        danger = answers.get("danger")
        danger_probability = _number(_answer_value(danger), 0.0)
        proposed = meta.get("proposed")
        if proposed in safe and danger_probability >= 0.72:
            # The veto only changes a non-forced edge.  It never chooses an
            # illegal action; route_action supplies the replacement.
            chosen = None
            self.advised_meta = {"veto": proposed, "tick": tick}
        if chosen is not None:
            self.advised_action = chosen
            self.advised_meta = {"tick": tick, "node": current,
                                 "generation": self.ledger.generation}
            try:
                self.ctx.log("jev_advice_accepted", action=chosen, age=age)
            except Exception:
                pass

    def _action(self, observation):
        self._consume_jev(observation)
        tick = _int(observation.get("tick", observation.get("ticks")), self.ledger.last_tick)
        self.ledger.ingest(observation)

        if self.ledger.mode == "transition":
            self.pulse_action = None
            self.pulse_node = None
            self.pulse_attempts = 0
            self.advised_action = None
            return self.noop

        standing = _standing(observation)
        if not standing:
            self.was_hopping = True
            self.pulse_action = None
            self.pulse_node = None
            self.pulse_attempts = 0
            self.advised_action = None
            return self.noop

        if self.was_hopping:
            self.was_hopping = False
            self.release_ticks = max(self.release_ticks, 1)
            self.pulse_action = None
            self.pulse_node = None
            self.pulse_attempts = 0
        if self.release_ticks:
            self.release_ticks -= 1
            return self.noop

        current = _node(observation.get("qbert_node"))
        safe = set(_names(observation.get("safe_moves")))
        if current is None or not safe:
            return self.noop

        proposed = self.ledger.route_action(observation)
        self._start_jev(observation, proposed)

        action = None
        if self.advised_meta:
            meta = self.advised_meta
            if (meta.get("tick") is not None and tick - meta.get("tick", tick) <= 2
                    and meta.get("node") == current
                    and meta.get("generation") == self.ledger.generation):
                if self.advised_meta.get("veto") == proposed:
                    action = None
                elif self.advised_action in safe or self.advised_action == self.noop:
                    action = self.advised_action
            self.advised_action = None
            self.advised_meta = None
        if action not in safe:
            action = proposed if proposed in safe else None
        if action is None:
            # If there is no target route, a certified edge is still preferable
            # to pressing FIRE.  This is also the pre-flight garbage-model
            # fallback.
            action = next(iter(safe), self.noop)

        if self.pulse_action == action and self.pulse_node == current:
            if self.pulse_attempts < 2:
                self.pulse_attempts += 1
                return action
            self.pulse_action = None
            self.pulse_node = None
            self.pulse_attempts = 0
            return self.noop
        self.pulse_action = action
        self.pulse_node = current
        self.pulse_attempts = 1
        return action

    def run(self, game):
        ctx = self.ctx
        while not ctx.done and not getattr(game, "done", False):
            try:
                observation = game.observe()
            except Exception:
                return
            if not isinstance(observation, dict):
                observation = {}
            self.last_obs = observation
            action = self._action(observation)
            if action not in self.names:
                action = self.noop
            try:
                ctx.act(action)
            except Exception:
                return
            try:
                ctx.wait_for_tick()
            except Exception:
                return
        try:
            ctx.log(
                "episode_summary",
                score=getattr(game, "score", None),
                ticks=getattr(game, "ticks", None),
                mode=self.ledger.mode,
                generation=self.ledger.generation,
                unfinished=self.ledger.unfinished(),
            )
        except Exception:
            pass


class HarnessProgram:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        Controller(self.ctx, game).run(game)


def build(ctx):
    return HarnessProgram(ctx)
