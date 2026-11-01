# candidates/0005/harness.py

import json
import time
from collections import deque

from metaharness.models.jev import Choice


ACTIONS = ("UP", "RIGHT", "LEFT", "DOWN")
NOOP = "NOOP"

# Six-row Atari 2600 pyramid, row-major indexing.
def node_edges(node):
    """Return only in-board edges for the six-row pyramid.

    qbert_node is temporarily unknown during boot, hops, and falls.  Also,
    the two upper-parent formulas have boundary conditions: blindly adding
    the upper-left formula at the left boundary creates node -1.
    """
    if isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < 21:
        return {}

    row = 0
    while (row + 1) * (row + 2) // 2 <= node:
        row += 1

    col = node - row * (row + 1) // 2
    result = {}

    # Row-major nodes: UP is the upper-right parent and LEFT is the
    # upper-left parent.  Check the boundary before adding either one.
    if row > 0:
        if col < row:
            result["UP"] = node - row
        if col > 0:
            result["LEFT"] = node - row - 1

    if row < 5:
        result["DOWN"] = node + row + 1
        result["RIGHT"] = node + row + 2

    return {
        action: neighbor
        for action, neighbor in result.items()
        if isinstance(neighbor, int) and 0 <= neighbor < 21
    }


def normalize_actions(value):
    if not isinstance(value, list):
        return []

    return [a for a in value if a in ACTIONS]


def safe_int(value, default=-1):
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return default


def observation_summary(obs):
    if not isinstance(obs, dict):
        return {}

    fields = (
        "tick", "ticks", "frame", "lives", "level", "round",
        "completed_rounds", "completed_levels", "score",
        "decoded_score", "score_match", "qbert_node",
        "qbert_in_hop", "qbert_motion", "cube_state",
        "needed_landings", "destination_cubes",
        "intermediate_cubes", "incomplete_cubes",
        "safe_moves", "legal_moves", "blocked_moves",
        "hazards", "disc_exits", "red_ball", "purple_ball",
        "coily", "sam", "green_ball", "timers",
        "last_event", "last_action", "action_error",
        "terminal", "truncated", "done",
    )

    return {k: obs[k] for k in fields if k in obs}


def find_route(obs, preferred_order=None):
    node = safe_int(obs.get("qbert_node"))
    if node < 0:
        return None, None

    safe = normalize_actions(obs.get("safe_moves", []))
    legal = normalize_actions(obs.get("legal_moves", []))

    allowed = set(safe if safe else legal)

    if not allowed:
        return None, None

    needed = obs.get("needed_landings", [])
    if not isinstance(needed, list) or len(needed) != 21:
        return None, None

    targets = {
        i for i, n in enumerate(needed)
        if isinstance(n, (int, float)) and n > 0
    }

    if not targets:
        return None, None

    graph = {
        i: node_edges(i)
        for i in range(21)
    }

    # qbert_node is allowed to be unknown while a sprite is between nodes.
    # Refuse to search from an invalid value rather than indexing the graph.
    if node not in graph:
        return None, None

    # BFS from each reachable target, preserving the first action.
    queue = deque([(node, None, 0)])
    visited = {node}
    candidates = []

    while queue:
        current, first_action, distance = queue.popleft()

        if current not in graph:
            continue

        if current in targets:
            priority = 0
            if preferred_order:
                try:
                    priority = preferred_order.index(current)
                except (ValueError, AttributeError):
                    priority = 999

            candidates.append((
                distance,
                priority,
                current,
                first_action,
            ))

        for action, neighbor in graph.get(current, {}).items():
            if current == node and action not in allowed:
                continue

            if neighbor not in graph or neighbor in visited:
                continue

            visited.add(neighbor)
            queue.append((
                neighbor,
                action if first_action is None else first_action,
                distance + 1,
            ))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, target, action = candidates[0]

    if action not in allowed:
        return None, target

    return action, target


def hazard_present(obs):
    hazards = obs.get("hazards", [])

    if not isinstance(hazards, list):
        return False

    for hazard in hazards:
        if not isinstance(hazard, dict):
            continue

        if hazard.get("lethal") and hazard.get("risk") in (
            "high", "imminent", "critical"
        ):
            return True

    return False


def heuristic_action(obs, preferred_order=None):
    safe = normalize_actions(obs.get("safe_moves", []))

    if not safe:
        return NOOP

    route, target = find_route(obs, preferred_order)

    if route in safe:
        return route

    # Do not manufacture a movement action when the decoder
    # cannot establish a safe route.
    if len(safe) == 1:
        return safe[0]

    # Prefer actions that avoid revisiting completed cubes.
    node = safe_int(obs.get("qbert_node"))
    edges = node_edges(node)

    needed = obs.get("needed_landings", [])

    choices = []

    for action in safe:
        destination = edges.get(action)

        if destination is None:
            continue

        remaining = (
            needed[destination]
            if isinstance(needed, list)
            and len(needed) == 21
            else 1
        )

        if not isinstance(remaining, (int, float)):
            remaining = 1

        choices.append((remaining <= 0, action))

    unfinished = [a for complete, a in choices if not complete]

    if unfinished:
        return unfinished[0]

    return safe[0]


class QbertController:

    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = {
            "strategy": "survival_first",
            "target_order": [],
            "max_macro": 1,
            "avoid_optional_bonuses": True,
        }

        self.astra_future = None
        self.jev_future = None

        self.last_action = NOOP
        self.last_jev_tick = -1
        self.last_jev_action = None

        self.last_lives = None
        self.last_round = None
        self.last_cube_state = None

        self.failures = 0
        self.calls = 0
        self.last_observation = None

    def start_astra(self, obs):
        if self.astra_future is not None:
            return
        if getattr(self.ctx, "system2", True) is None:
            return

        prompt = """
You are planning an Atari 2600 Q*bert controller.

This is the Atari 2600 six-row, 21-cube board, not the arcade board.

Rules:
Level 1: S -> D
Level 2: S -> I -> D
Level 3: S -> D -> S
Level 4: S -> I -> D -> I
Level 5: S -> I -> D -> S

Four rounds per level. Level 5 repeats.

Priorities:
1. Preserve lives.
2. Complete the current pyramid.
3. Avoid lethal enemy collisions and edge falls.
4. Use validated disc exits only.
5. Ignore optional bonuses when they create danger.

Do not assume that every landing is permanent progress.
Use the actual needed_landings and cube_state observations.

Return a compact JSON object only:
{
  "strategy": "survival_first",
  "target_order": [node indices],
  "max_macro": 1,
  "avoid_optional_bonuses": true
}

The target_order should prioritize useful coverage of the pyramid.
Never invent node indices outside 0..20.

Initial observation:
""" + json.dumps(observation_summary(obs))

        def request():
            return self.ctx.think(
                prompt,
                timeout_s=90.0,
            )

        try:
            self.astra_future = self.ctx.spawn(request)
        except Exception as exc:
            self.astra_future = None
            try:
                self.ctx.log("astra_spawn_failed", error=str(exc))
            except Exception:
                pass

    def collect_astra(self):
        future = self.astra_future

        if future is None or not future.done():
            return

        self.astra_future = None

        try:
            result = future.result()
            raw = result.text

            start = raw.find("{")
            end = raw.rfind("}")

            if start < 0 or end <= start:
                return

            parsed = json.loads(raw[start:end + 1])

            if not isinstance(parsed, dict):
                return

            order = parsed.get("target_order", [])

            if (
                isinstance(order, list)
                and all(
                    isinstance(x, int) and 0 <= x < 21
                    for x in order
                )
            ):
                self.plan["target_order"] = order

            if parsed.get("strategy") in (
                "survival_first",
                "coverage_first",
            ):
                self.plan["strategy"] = parsed["strategy"]

            self.ctx.log(
                "astra_plan_updated",
                strategy=self.plan["strategy"],
                target_count=len(self.plan["target_order"]),
            )

        except Exception as exc:
            self.ctx.log("astra_plan_rejected", error=str(exc))

    def make_questions(self, obs, fallback):
        safe = normalize_actions(obs.get("safe_moves", []))

        if not safe:
            return None

        # Candidate descriptions are explicit; code still validates
        # the returned action against the current observation.
        criteria = {
            action: {
                "what": "Take the " + action + " joystick edge.",
                "not_for": "An edge absent from safe_moves.",
            }
            for action in safe
        }

        criteria["NOOP"] = {
            "what": "Wait deliberately without changing direction.",
            "not_for": "A movement opportunity that is safe.",
        }

        state = {
            "game": "Atari 2600 Q*bert",
            "rules": {
                "board_nodes": 21,
                "destination_cycle": "level-dependent S/I/D",
                "priority": "survival before optional points",
            },
            "plan": self.plan,
            "observation": observation_summary(obs),
            "fallback_action": fallback,
            "decision_age_ticks": 0,
        }

        questions = {
            "move": Choice(
                """
Choose the safest useful immediate joystick action.

Read observation.safe_moves as the authoritative conservative
movement set. Never select an action outside it.

Avoid lethal hazards and pyramid edge falls.
Prefer an action that advances toward a cube with positive
needed_landings.

Survival is more important than bonus points.
Select NOOP only when deliberate waiting is safer than movement.
""",
                criteria,
            )
        }

        return state, questions

    def request_jev(self, obs, fallback):
        if self.jev_future is not None:
            return
        if getattr(self.ctx, "system1", True) is None:
            return

        prepared = self.make_questions(obs, fallback)

        if prepared is None:
            return

        state, questions = prepared
        tick = safe_int(obs.get("tick", obs.get("ticks", -1)))

        self.last_jev_tick = tick

        def request():
            started = time.monotonic()

            answers = self.ctx.decide(
                state,
                questions,
                timeout_s=0.40,
            )

            elapsed = time.monotonic() - started

            answer = answers.get("move")

            return {
                "action": getattr(answer, "value", None),
                "probabilities": getattr(
                    answer, "probabilities", {}
                ),
                "confidence": getattr(
                    answer, "confidence", None
                ),
                "tick": tick,
                "elapsed": elapsed,
            }

        try:
            self.jev_future = self.ctx.spawn(request)
        except Exception as exc:
            self.jev_future = None
            self.failures += 1
            try:
                self.ctx.log("jev_spawn_failed", error=str(exc))
            except Exception:
                pass

    def collect_jev(self, obs):
        future = self.jev_future

        if future is None or not future.done():
            return None

        self.jev_future = None

        try:
            result = future.result()

            action = result.get("action")

            current_tick = safe_int(
                obs.get("tick", obs.get("ticks", -1))
            )

            age = current_tick - result.get("tick", -999)

            safe = normalize_actions(obs.get("safe_moves", []))

            # A delayed response is not an instruction for the
            # current world state.
            if age < 0 or age > 2:
                return None

            if action not in safe and action != NOOP:
                return None

            probabilities = result.get("probabilities", {})

            if not isinstance(probabilities, dict):
                probabilities = {}

            confidence = probabilities.get(action, 0.0)

            if not isinstance(confidence, (int, float)):
                confidence = 0.0

            if action != NOOP and confidence < 0.40:
                return None

            self.calls += 1

            self.ctx.log(
                "jev_decision_accepted",
                action=action,
                age=age,
                confidence=confidence,
                latency=result.get("elapsed"),
            )

            return action

        except Exception as exc:
            self.failures += 1
            self.ctx.log("jev_decision_failed", error=str(exc))
            return None

    def update_events(self, obs):
        lives = obs.get("lives")
        round_number = obs.get("completed_rounds")

        if self.last_lives is not None and lives is not None:
            if lives < self.last_lives:
                self.ctx.log(
                    "life_lost",
                    lives=lives,
                    round=obs.get("round"),
                )

        if self.last_round is not None and round_number is not None:
            if round_number > self.last_round:
                self.ctx.log(
                    "round_completed",
                    completed_rounds=round_number,
                )

        self.last_lives = lives
        self.last_round = round_number
        self.last_cube_state = obs.get("cube_state")

    def choose(self, obs):
        self.collect_astra()

        fallback = heuristic_action(
            obs,
            self.plan.get("target_order"),
        )

        # If the player is falling or a lethal hazard is imminent,
        # do not trust an old semantic decision.
        if obs.get("qbert_motion") == "falling":
            return fallback

        if hazard_present(obs):
            return fallback

        answer = self.collect_jev(obs)

        if answer is not None:
            return answer

        self.request_jev(obs, fallback)

        return fallback


class HarnessProgram:

    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        controller = QbertController(self.ctx)

        obs = game.observe()

        controller.start_astra(obs)

        while not self.ctx.done and not game.done:
            self.ctx.wait_for_tick()

            if self.ctx.done or game.done:
                break

            obs = game.observe()

            if not isinstance(obs, dict):
                continue

            controller.update_events(obs)

            if obs.get("terminal") or obs.get("done"):
                break

            action = controller.choose(obs)

            if action not in ACTIONS and action != NOOP:
                action = NOOP

            # Every posted action is immediate. The game never waits
            # for Astra or Jev.
            self.ctx.act(action)

            controller.last_action = action

            self.last_observation = obs

            self.ctx.log(
                "controller_tick",
                tick=obs.get("tick", obs.get("ticks")),
                action=action,
                lives=obs.get("lives"),
                score=obs.get("score"),
                fallback=action == heuristic_action(
                    obs,
                    controller.plan.get("target_order"),
                ),
            )

        self.ctx.log(
            "episode_summary",
            score=getattr(game, "score", None),
            ticks=getattr(game, "ticks", None),
            jev_calls=controller.calls,
            jev_failures=controller.failures,
            plan=controller.plan,
        )


def build(ctx):
    return HarnessProgram(ctx)
