"""Realtime adaptor for the pinned, symbolic Overcooked-AI environment.

The upstream environment is a synchronous transition oracle.  This module exposes one
six-Hz transition per ``tick`` so the Meta-Harness can run its normal tier-2 controller
against the real MDP without a display, game account, or network service.
"""
from __future__ import annotations

import collections
import copy
import os
import random
import threading
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

# Importing the visualisation module is part of the upstream environment import path.
# Keep pygame headless on the evaluator's Linux server.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from metaharness.games.base import ActionSpec
from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import (
    EVENT_TYPES,
    OvercookedGridworld,
    OvercookedState,
)
from overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    MotionPlanner,
    NO_COUNTERS_PARAMS,
)


TIER = 2
DETERMINISTIC = True

# The benchmark uses these fixed scenario identifiers.  The first three are the
# development/training scenarios; the rest are held out from search.
SEEDS = {
    "train": [0, 1, 2],
    "heldout": [100, 101, 102, 103, 104, 105, 106, 107],
}

REPOSITORY_COMMIT = "739950a079cdaed5a44fcc662efc40244c205d06"
HORIZON = 400
TICK_S = 1.0 / 6.0

_ACTIONS_BY_NAME = {
    "north": Direction.NORTH,
    "south": Direction.SOUTH,
    "east": Direction.EAST,
    "west": Direction.WEST,
    "stay": Action.STAY,
    "interact": Action.INTERACT,
}
_ACTION_NAMES = tuple(_ACTIONS_BY_NAME)
_ACTION_NAME_BY_NATIVE = {value: key for key, value in _ACTIONS_BY_NAME.items()}
_MOTION_ACTIONS = frozenset(Direction.ALL_DIRECTIONS)
_FEATURE_NAMES = ("O", "T", "D", "P", "S", "X")

ACTION_SPEC = ActionSpec(
    names=list(_ACTION_NAMES),
    descriptions={
        "north": "Move one walkable tile north; facing becomes north.",
        "south": "Move one walkable tile south; facing becomes south.",
        "east": "Move one walkable tile east; facing becomes east.",
        "west": "Move one walkable tile west; facing becomes west.",
        "stay": "Do not move and keep the current facing direction.",
        "interact": "Operate the feature immediately in front of this chef (pick up, put down, fill/start a pot, or serve).",
    },
    noop="stay",
    notes=(
        "One primitive is applied per 1/6-second tick. A missing decision repeats the last valid action; "
        "a malformed action is replaced by stay and is reported in the next observation. "
        "There are no action parameters and no commercial-game mechanics: this is the pinned "
        "Overcooked-AI grid MDP."
    ),
)


# A scenario is deliberately data, not a second game implementation.  Explicit held-out
# entries make the evaluation set stable and make controlled-agent reversal observable.
_SCENARIOS: dict[int, dict[str, Any]] = {
    0: {"layout": "cramped_room", "family": "cramped_room", "partner": "greedy", "start": "standard", "controlled": 0},
    1: {"layout": "asymmetric_advantages", "family": "asymmetric_advantages", "partner": "greedy", "start": "standard", "controlled": 0},
    2: {"layout": "coordination_ring", "family": "coordination_ring", "partner": "greedy", "start": "standard", "controlled": 0},
    100: {"layout": "counter_circuit_o_1order", "family": "counter_circuit_o_1order", "partner": "greedy", "start": "standard", "controlled": 0},
    101: {"layout": "forced_coordination", "family": "forced_coordination", "partner": "handoff", "start": "standard", "controlled": 0},
    102: {"layout": "forced_coordination", "family": "forced_coordination_agent1", "partner": "handoff", "start": "standard", "controlled": 1},
    103: {"layout": "counter_circuit", "family": "counter_circuit", "partner": "general", "start": "standard", "controlled": 0},
    104: {"layout": "cramped_room_tomato", "family": "tomato_mixed", "partner": "general", "start": "standard", "controlled": 0},
    105: {"layout": "asymmetric_advantages_tomato", "family": "tomato_mixed_reversed", "partner": "general", "start": "random_position_orientation", "controlled": 1},
    106: {"layout": "forced_coordination_tomato", "family": "forced_coordination_tomato", "partner": "handoff", "start": "random_position_orientation", "controlled": 0},
    107: {"layout": "counter_circuit_o_1order", "family": "counter_circuit_alt_start", "partner": "general", "start": "random_position_orientation", "controlled": 0},
}


def _scenario_for_seed(seed: int) -> dict[str, Any]:
    """Return a deterministic scenario for arbitrary evaluator seeds as well."""
    if seed in _SCENARIOS:
        return dict(_SCENARIOS[seed])
    # Unknown seeds remain useful for local experiments.  The hash is arithmetic
    # rather than Python's salted hash, so separate processes replay the same case.
    layouts = ("cramped_room", "asymmetric_advantages", "coordination_ring")
    layout = layouts[abs(int(seed)) % len(layouts)]
    return {
        "layout": layout,
        "family": f"development_{layout}",
        "partner": "greedy",
        "start": "random_position_orientation" if abs(int(seed)) % 5 == 0 else "standard",
        "controlled": 1 if abs(int(seed)) % 11 == 0 else 0,
    }


def _json_safe(value: Any) -> Any:
    """Convert upstream tuples/NumPy scalars into the strict observation subset."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # This is a defensive fallback; state.to_dict() should never reach it.
    return str(value)


def _native_action_name(action: Any) -> str:
    if action in _ACTION_NAME_BY_NATIVE:
        return _ACTION_NAME_BY_NATIVE[action]
    return "stay"


def _recipe_counts(ingredients: Iterable[str]) -> Counter[str]:
    return Counter(str(x) for x in ingredients)


def _action_for_goal(mdp: OvercookedGridworld, state: OvercookedState, player_index: int,
                     feature_positions: Iterable[tuple[int, int]],
                     blocked_position: tuple[int, int] | None = None) -> list[Any] | None:
    """Shortest primitive path to a feature, ending with INTERACT.

    MotionPlanner precomputes every possible pair and is excellent for the stock
    agent, but a generic scripted partner also needs to face counters.  This small
    BFS is the same four-neighbour movement relation as the official MDP and is
    computed only for the partner's current state.
    """
    player = state.players[player_index]
    start = (tuple(player.position), tuple(player.orientation))
    valid_positions = set(mdp.get_valid_player_positions())
    other_position = blocked_position

    goals: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    features = sorted(set(tuple(p) for p in feature_positions))
    for feature in features:
        for direction in Direction.ALL_DIRECTIONS:
            # To face a feature in ``direction``, stand one step opposite it.
            adjacent = Action.move_in_direction(feature, Direction.OPPOSITE_DIRECTIONS[direction])
            if adjacent in valid_positions and adjacent != other_position:
                goals.add((adjacent, direction))

    if not goals:
        return None
    if start in goals:
        return [Action.INTERACT]

    queue: deque[tuple[tuple[int, int], tuple[int, int]]] = deque([start])
    previous: dict[Any, tuple[Any, Any] | None] = {start: None}
    found: Any = None
    # This order is fixed, making partner tie-breaking part of the seed replay.
    for_state_actions = Direction.ALL_DIRECTIONS
    while queue:
        current = queue.popleft()
        for action in for_state_actions:
            nxt = mdp._move_if_direction(current[0], current[1], action)
            nxt = (tuple(nxt[0]), tuple(nxt[1]))
            if nxt in previous or nxt[0] == other_position:
                continue
            previous[nxt] = (current, action)
            if nxt in goals:
                found = nxt
                queue.clear()
                break
            queue.append(nxt)
        if found is not None:
            break
    if found is None:
        return None

    path: list[Any] = []
    cursor = found
    while previous[cursor] is not None:
        parent, action = previous[cursor]  # type: ignore[misc]
        path.append(action)
        cursor = parent
    path.reverse()
    path.append(Action.INTERACT)
    return path


class _ScriptedPartner:
    """Deterministic partner with native GreedyHumanModel and general fallback."""

    def __init__(self, mdp: OvercookedGridworld, env: OvercookedEnv,
                 partner_index: int, style: str, seed: int) -> None:
        self.mdp = mdp
        self.env = env
        self.partner_index = partner_index
        self.style = style
        self.seed = seed
        self.last_action = Action.STAY
        self.last_task = "wait"
        self._greedy: GreedyHumanModel | None = None
        self._rng = random.Random(seed ^ 0x5EED5EED)
        self._handoff_counter = self._find_handoff_counter()

        # GreedyHumanModel is the reference scripted baseline for the standard
        # one three-onion-order layouts.  Its own source warns against forced
        # coordination and it asserts on mixed recipes, so those use the general
        # partner below instead.
        if style == "greedy":
            try:
                mlam = MediumLevelActionManager(mdp, NO_COUNTERS_PARAMS)
                env._mlam = mlam
                env._mp = mlam.motion_planner
                self._greedy = GreedyHumanModel(mlam, auto_unstuck=True)
                self._greedy.set_agent_index(partner_index)
            except Exception:
                self._greedy = None

    def _find_handoff_counter(self) -> tuple[int, int] | None:
        if self.style != "handoff":
            return None
        # Both forced-coordination layouts intentionally share the central
        # counter at (2, 1).  Check it instead of assuming an asset is present.
        candidate = (2, 1)
        if candidate in self.mdp.terrain_pos_dict.get("X", []):
            return candidate
        counters = sorted(self.mdp.terrain_pos_dict.get("X", []))
        return counters[0] if counters else None

    def reset(self) -> None:
        self.last_action = Action.STAY
        self.last_task = "wait"
        if self._greedy is not None:
            self._greedy.reset()
            self._greedy.set_agent_index(self.partner_index)

    def _feature(self, kind: str) -> list[tuple[int, int]]:
        return list(self.mdp.terrain_pos_dict.get(kind, []))

    def _objects_at_features(self, state: OvercookedState, kind: str) -> list[tuple[int, int]]:
        out = []
        for pos, obj in state.objects.items():
            if self.mdp.get_terrain_type_at_pos(pos) == kind:
                out.append(tuple(pos))
        return sorted(out)

    def _recipe_for_pot(self, state: OvercookedState, pot_pos: tuple[int, int]) -> Any:
        recipes = list(state.all_orders)
        recipes.sort(key=lambda recipe: (-float(recipe.value), tuple(recipe.ingredients)))
        if state.has_object(pot_pos):
            current = _recipe_counts(state.get_object(pot_pos).ingredients)
            compatible = [
                r for r in recipes
                if all(current[k] <= _recipe_counts(r.ingredients)[k] for k in current)
            ]
            if compatible:
                return compatible[0]
        return recipes[0] if recipes else None

    def _pot_target_ingredient(self, state: OvercookedState, pot_pos: tuple[int, int]) -> str | None:
        recipe = self._recipe_for_pot(state, pot_pos)
        if recipe is None:
            return None
        wanted = _recipe_counts(recipe.ingredients)
        current = Counter()
        if state.has_object(pot_pos):
            current.update(state.get_object(pot_pos).ingredients)
        for ingredient in ("onion", "tomato"):
            if current[ingredient] < wanted[ingredient]:
                return ingredient
        return None

    def _fillable_pots(self, state: OvercookedState) -> list[tuple[int, int]]:
        result = []
        for pos in sorted(self.mdp.get_pot_locations()):
            if not state.has_object(pos):
                result.append(tuple(pos))
                continue
            soup = state.get_object(pos)
            if not getattr(soup, "is_ready", False) and not getattr(soup, "is_cooking", False):
                if self._pot_target_ingredient(state, tuple(pos)) is not None:
                    result.append(tuple(pos))
        return result

    def _cooking_or_ready(self, state: OvercookedState) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        ready: list[tuple[int, int]] = []
        cooking: list[tuple[int, int]] = []
        for pos in sorted(self.mdp.get_pot_locations()):
            if not state.has_object(pos):
                continue
            soup = state.get_object(pos)
            if getattr(soup, "is_ready", False):
                ready.append(tuple(pos))
            elif getattr(soup, "is_cooking", False):
                cooking.append(tuple(pos))
        return ready, cooking

    def _handoff_action(self, state: OvercookedState, player: Any,
                        other: Any) -> tuple[list[Any] | None, str] | None:
        """Reserve the shared counter for the explicit forced-coordination role."""
        counter = self._handoff_counter
        if counter is None:
            return None
        objects = [pos for pos in self._objects_at_features(state, "X") if pos == counter]
        if not player.has_object() and objects:
            return _action_for_goal(self.mdp, state, self.partner_index, [counter], other.position), "handoff pickup"
        if player.has_object() and player.get_object().name in {"onion", "tomato", "dish", "soup"}:
            # A left-side partner first hands off ingredients/dishes.  A right-
            # side partner uses the same counter only when it actually has an
            # object to pass or collect; the normal task planner handles pots.
            direct_pots = self._fillable_pots(state)
            if player.get_object().name in {"onion", "tomato"} and direct_pots:
                # Only use the handoff if no pot is reachable from this player.
                pot_plan = _action_for_goal(self.mdp, state, self.partner_index, direct_pots, other.position)
                if pot_plan is not None:
                    return None
            plan = _action_for_goal(self.mdp, state, self.partner_index, [counter], other.position)
            return plan, "handoff place"
        return None

    def _general_action(self, state: OvercookedState) -> tuple[Any, str]:
        player = state.players[self.partner_index]
        other = state.players[1 - self.partner_index]
        blocked = tuple(other.position)
        ready, cooking = self._cooking_or_ready(state)
        fillable = self._fillable_pots(state)
        feature: list[tuple[int, int]] = []
        task = "wait"

        # Forced coordination gets an actual shared-counter protocol rather than
        # the stock greedy assertion.  It is checked before the generic plan.
        if self.style == "handoff":
            handoff = self._handoff_action(state, player, other)
            if handoff is not None and handoff[0] is not None:
                return handoff[0][0], handoff[1]
            if not player.has_object() and self._handoff_counter is not None:
                if self._objects_at_features(state, "X"):
                    plan = _action_for_goal(self.mdp, state, self.partner_index, [self._handoff_counter], blocked)
                    if plan:
                        return plan[0], "handoff pickup"

        if player.has_object():
            obj_name = player.get_object().name
            if obj_name == "soup":
                feature, task = self._feature("S"), "deliver soup"
            elif obj_name in {"onion", "tomato"}:
                matching = [p for p in fillable if self._pot_target_ingredient(state, p) == obj_name]
                feature, task = matching or fillable, f"pot {obj_name}"
                if not feature and self.style == "handoff":
                    plan = _action_for_goal(self.mdp, state, self.partner_index, [self._handoff_counter] if self._handoff_counter else [], blocked)
                    if plan:
                        return plan[0], "handoff place"
            elif obj_name == "dish":
                feature, task = ready, "pick up ready soup"
                if not feature and cooking:
                    # Keep a dish staged on a counter while the soup cooks.
                    feature, task = self._feature("X"), "stage dish"
            else:
                feature, task = self._feature("X"), "stage object"
        else:
            # Soup on a counter is already a useful handoff; collect it first.
            soup_counters = [p for p in self._objects_at_features(state, "X")
                             if state.get_object(p).name == "soup"]
            if soup_counters:
                feature, task = soup_counters, "collect soup"
            elif ready:
                # A chef without a dish must fetch one before a ready pot can be
                # picked up.  If the other chef has one, wait rather than stealing.
                other_has_dish = other.has_object() and other.get_object().name == "dish"
                if not other_has_dish:
                    feature, task = self._feature("D"), "fetch dish"
            elif cooking:
                other_has_dish = other.has_object() and other.get_object().name == "dish"
                if not other_has_dish:
                    feature, task = self._feature("D"), "stage dish"
            else:
                # Start a completed-but-idle pot before gathering another order.
                idle_full = []
                for pos in sorted(self.mdp.get_pot_locations()):
                    if state.has_object(pos):
                        soup = state.get_object(pos)
                        if (not soup.is_ready and not soup.is_cooking
                                and self._pot_target_ingredient(state, tuple(pos)) is None):
                            idle_full.append(tuple(pos))
                if idle_full:
                    feature, task = idle_full, "start cooking"
                else:
                    # Prefer an ingredient needed by the nearest incomplete pot;
                    # otherwise begin the highest-value order from its first item.
                    ingredient = None
                    if fillable:
                        ingredient = self._pot_target_ingredient(state, fillable[0])
                    if ingredient is None:
                        recipe = self._recipe_for_pot(state, fillable[0]) if fillable else (list(state.all_orders)[0] if state.all_orders else None)
                        ingredient = recipe.ingredients[0] if recipe is not None else "onion"
                    feature, task = self._feature("O" if ingredient == "onion" else "T"), f"fetch {ingredient}"

            # Pick up a useful object staged on a counter if no dispenser task was
            # found (important for the forced-coordination side of the kitchen).
            if not feature:
                useful = []
                for pos in self._objects_at_features(state, "X"):
                    name = state.get_object(pos).name
                    if name in {"onion", "tomato", "dish"}:
                        useful.append(pos)
                if useful:
                    feature, task = useful, "collect staged object"

        plan = _action_for_goal(self.mdp, state, self.partner_index, feature, blocked)
        if plan:
            return plan[0], task
        return Action.STAY, "wait (no reachable task)"

    def action(self, state: OvercookedState) -> tuple[Any, str]:
        if self._greedy is not None:
            try:
                self._greedy.set_agent_index(self.partner_index)
                action, _info = self._greedy.action(state)
                if action in Action.ALL_ACTIONS:
                    self.last_action = action
                    self.last_task = "greedy human model"
                    return action, self.last_task
            except Exception:
                # Mixed/forced layouts and some randomized orientations are
                # intentionally outside GreedyHumanModel's documented domain.
                pass
        action, task = self._general_action(state)
        if action not in Action.ALL_ACTIONS:
            action = Action.STAY
            task = "wait (partner fallback)"
        self.last_action = action
        self.last_task = task
        return action, task


class OvercookedRealtimeGame:
    name = "Overcooked-AI"
    action_spec = ACTION_SPEC
    tick_s = TICK_S

    def __init__(self, controlled_agent_index: int | None = None,
                 partner_style: str | None = None, **_kwargs: Any) -> None:
        self._lock = threading.RLock()
        self._controlled_override = controlled_agent_index
        self._partner_style_override = partner_style
        self._closed = False
        self._episode_id = 0
        self._invalid_reason: str | None = None
        self._env: OvercookedEnv | None = None
        self._mdp: OvercookedGridworld | None = None
        self._partner: _ScriptedPartner | None = None
        self._scenario: dict[str, Any] = {}
        self._seed = 0
        self._controlled_index = 0
        self._partner_index = 1
        self._ticks = 0
        self._score = 0.0
        self._score_by_agent = [0.0, 0.0]
        self._shaped_score = 0.0
        self._shaped_by_agent = [0.0, 0.0]
        self._last_sparse_reward = 0.0
        self._last_sparse_by_agent = [0.0, 0.0]
        self._last_shaped_by_agent = [0.0, 0.0]
        self._last_events: list[str] = []
        self._event_counts: dict[str, list[int]] = {event: [0, 0] for event in EVENT_TYPES}
        self._collision_ticks = 0
        self._stuck_ticks = 0
        self._pot_idle_time = 0
        self._handoff_attempts = 0
        self._handoff_successes = 0
        self._last_action_error: str | None = None
        self._current_action = Action.STAY
        self._last_controlled_action = Action.STAY
        self._last_partner_action = Action.STAY
        self._done = False
        self._reset_snapshot: dict[str, Any] | None = None

    def _parse_action(self, action: Any) -> tuple[Any, str | None]:
        if action is None:
            return self._current_action, None
        if isinstance(action, str):
            key = action.strip().lower()
            if key in _ACTIONS_BY_NAME:
                return _ACTIONS_BY_NAME[key], None
            # Friendly aliases are accepted but not advertised as extra actions.
            aliases = {"up": "north", "down": "south", "left": "west", "right": "east", "noop": "stay", "space": "interact"}
            if key in aliases:
                return _ACTIONS_BY_NAME[aliases[key]], None
        if isinstance(action, (tuple, list)) and len(action) == 2:
            try:
                candidate = tuple(int(x) for x in action)
            except (TypeError, ValueError, OverflowError):
                candidate = None
            if candidate in _ACTION_NAME_BY_NATIVE:
                return candidate, None
        return Action.STAY, f"ignored malformed action {action!r}"

    def _make_start_state(self, mdp: OvercookedGridworld, scenario: dict[str, Any], seed: int) -> OvercookedState:
        if scenario.get("start") != "random_position_orientation":
            return mdp.get_standard_start_state().deepcopy()
        rng = np.random.default_rng((int(seed) * 1_000_003 + 17) & 0xFFFFFFFF)
        valid = mdp.get_valid_joint_player_positions()
        if not valid:
            return mdp.get_standard_start_state().deepcopy()
        positions = valid[int(rng.integers(0, len(valid)))]
        orientations = [Direction.ALL_DIRECTIONS[int(rng.integers(0, 4))] for _ in positions]
        players_pos_and_or = list(zip(positions, orientations))
        return OvercookedState.from_players_pos_and_or(
            players_pos_and_or,
            bonus_orders=mdp.start_bonus_orders,
            all_orders=mdp.start_all_orders,
        )

    def reset(self, seed: int) -> Any:
        with self._lock:
            self._closed = False
            self._episode_id += 1
            self._seed = int(seed)
            random.seed(self._seed ^ 0xA17E)
            np.random.seed(self._seed & 0xFFFFFFFF)
            self._scenario = _scenario_for_seed(self._seed)
            controlled = self._controlled_override
            self._controlled_index = int(self._scenario.get("controlled", 0) if controlled is None else controlled)
            if self._controlled_index not in (0, 1):
                self._controlled_index = 0
            self._partner_index = 1 - self._controlled_index
            style = self._partner_style_override or str(self._scenario.get("partner", "general"))
            if style not in {"greedy", "general", "handoff"}:
                style = "general"

            mdp = OvercookedGridworld.from_layout_name(
                str(self._scenario["layout"]), old_dynamics=False
            )
            start_state = self._make_start_state(mdp, self._scenario, self._seed)
            env = OvercookedEnv.from_mdp(
                mdp,
                start_state_fn=lambda s=start_state: s.deepcopy(),
                horizon=HORIZON,
                mlam_params=NO_COUNTERS_PARAMS,
                info_level=0,
            )
            # OvercookedEnv evaluates ``self.mp`` on every step even when
            # potential display is disabled.  Build the official movement
            # planner in memory; no package-directory cache or process is used.
            env._mp = MotionPlanner(mdp, [])
            self._mdp = mdp
            self._env = env
            self._partner = _ScriptedPartner(mdp, env, self._partner_index, style, self._seed)
            self._partner.reset()
            self._invalid_reason = None
            self._ticks = 0
            self._score = 0.0
            self._score_by_agent = [0.0, 0.0]
            self._shaped_score = 0.0
            self._shaped_by_agent = [0.0, 0.0]
            self._last_sparse_reward = 0.0
            self._last_sparse_by_agent = [0.0, 0.0]
            self._last_shaped_by_agent = [0.0, 0.0]
            self._last_events = []
            self._event_counts = {event: [0, 0] for event in EVENT_TYPES}
            self._collision_ticks = 0
            self._stuck_ticks = 0
            self._pot_idle_time = 0
            self._handoff_attempts = 0
            self._handoff_successes = 0
            self._last_action_error = None
            self._current_action = Action.STAY
            self._last_controlled_action = Action.STAY
            self._last_partner_action = Action.STAY
            self._done = False
            self._reset_snapshot = self.observe()
            return self._reset_snapshot

    def _event_delta(self, before: dict[str, list[int]]) -> list[str]:
        assert self._env is not None
        current_events: list[str] = []
        for event in EVENT_TYPES:
            rows = self._env.game_stats.get(event, [[], []])
            for index in range(min(2, len(rows))):
                count = len(rows[index])
                delta = max(0, count - before.get(event, [0, 0])[index])
                if delta:
                    self._event_counts[event][index] = count
                    current_events.extend([event] * delta)
        return current_events

    def _event_counts_before(self) -> dict[str, list[int]]:
        assert self._env is not None
        return {
            event: [len(row) for row in self._env.game_stats.get(event, [[], []])]
            for event in EVENT_TYPES
        }

    def tick(self, action: Any) -> None:
        """Apply one current action and one scripted-partner action immediately."""
        with self._lock:
            if self._done or self._env is None or self._mdp is None or self._partner is None:
                return
            native_action, error = self._parse_action(action)
            self._last_action_error = error
            # Invalid input is a nonfatal missed decision.  A valid input becomes
            # the current mailbox action and is applied on this transition.
            if error is None:
                self._current_action = native_action
            controlled_action = self._current_action if error is None or action is None else Action.STAY
            old_state = self._env.state
            old_positions = tuple(tuple(p.position) for p in old_state.players)
            before_events = self._event_counts_before()
            try:
                partner_action, _partner_task = self._partner.action(old_state)
                if partner_action not in Action.ALL_ACTIONS:
                    partner_action = Action.STAY
                joint = [Action.STAY, Action.STAY]
                joint[self._controlled_index] = controlled_action
                joint[self._partner_index] = partner_action
                intended_positions = tuple(
                    self._mdp._move_if_direction(player.position, player.orientation, a)[0]
                    for player, a in zip(old_state.players, joint)
                )
                if self._mdp.is_transition_collision(old_positions, intended_positions):
                    self._collision_ticks += 1
                if any(a in _MOTION_ACTIONS for a in joint) and intended_positions == old_positions:
                    self._stuck_ticks += 1
                _next_state, sparse_reward, done, env_info = self._env.step(tuple(joint))
                sparse_by_agent = [float(x) for x in env_info.get("sparse_r_by_agent", [0, 0])]
                shaped_by_agent = [float(x) for x in env_info.get("shaped_r_by_agent", [0, 0])]
                self._last_sparse_by_agent = sparse_by_agent[:2]
                self._last_shaped_by_agent = shaped_by_agent[:2]
                self._last_sparse_reward = float(sum(sparse_by_agent))
                self._score_by_agent = [a + b for a, b in zip(self._score_by_agent, sparse_by_agent)]
                self._shaped_by_agent = [a + b for a, b in zip(self._shaped_by_agent, shaped_by_agent)]
                self._score = float(sum(self._score_by_agent))
                self._shaped_score = float(sum(self._shaped_by_agent))
                self._last_events = self._event_delta(before_events)
                pot_states = self._mdp.get_pot_states(self._env.state)
                self._pot_idle_time += len(pot_states.get("empty", [])) + len(pot_states.get("ready", []))
                self._handoff_diagnostic(old_state, self._env.state, joint, self._scenario.get("partner"))
                self._last_controlled_action = controlled_action
                self._last_partner_action = partner_action
                self._ticks = int(self._env.state.timestep)
                self._done = bool(done)
            except Exception as exc:  # infrastructure failure: retain a useful report, never crash tick
                self._invalid_reason = f"Overcooked transition failed: {type(exc).__name__}: {exc}"[:500]
                self._done = True

    def _handoff_diagnostic(self, before: OvercookedState, after: OvercookedState,
                            joint: list[Any], style: Any) -> None:
        if style != "handoff" or self._partner is None or self._partner._handoff_counter is None:
            return
        counter = self._partner._handoff_counter
        for idx, act in enumerate(joint):
            player = before.players[idx]
            facing = Action.move_in_direction(player.position, player.orientation)
            if act == Action.INTERACT and facing == counter:
                self._handoff_attempts += 1
                changed = False
                before_obj = before.objects.get(counter)
                after_obj = after.objects.get(counter)
                if (before_obj is None) != (after_obj is None):
                    changed = True
                if before.players[idx].held_object is None and after.players[idx].held_object is not None:
                    changed = True
                if before.players[idx].held_object is not None and after.players[idx].held_object is None:
                    changed = True
                if changed:
                    self._handoff_successes += 1

    @property
    def done(self) -> bool:
        with self._lock:
            return bool(self._done or (self._env is not None and self._env.is_done()))

    @property
    def score(self) -> float:
        with self._lock:
            return float(self._score)

    @property
    def ticks(self) -> int:
        with self._lock:
            return int(self._ticks)

    def _observation_unlocked(self) -> dict[str, Any]:
        env = self._env
        mdp = self._mdp
        partner = self._partner
        if env is None or mdp is None:
            return {"name": self.name, "done": True, "tick": 0, "score": 0.0}
        state = _json_safe(env.state.to_dict())
        event_counts = {event: int(sum(rows)) for event, rows in self._event_counts.items() if sum(rows)}
        return {
            "name": self.name,
            "repository_commit": REPOSITORY_COMMIT,
            "episode_id": int(self._episode_id),
            "seed": int(self._seed),
            "scenario": _json_safe(self._scenario),
            "layout_name": str(mdp.layout_name),
            "terrain_mtx": _json_safe(mdp.terrain_mtx),
            "horizon": HORIZON,
            "tick": int(self._ticks),
            "deadline_tick": int(self._ticks + 1),
            "controlled_agent_index": int(self._controlled_index),
            "partner_agent_index": int(self._partner_index),
            "state": state,
            "sparse_score": float(self._score),
            "last_sparse_reward": float(self._last_sparse_reward),
            "sparse_r_by_agent": [float(x) for x in self._last_sparse_by_agent],
            "shaped_score_diagnostic": float(self._shaped_score),
            "last_shaped_r_by_agent": [float(x) for x in self._last_shaped_by_agent],
            "deliveries": int(event_counts.get("soup_delivery", 0)),
            "event_counts": event_counts,
            "last_events": list(self._last_events),
            "controlled_action_last": _native_action_name(self._last_controlled_action),
            "partner_action_last": _native_action_name(self._last_partner_action),
            "partner_task": partner.last_task if partner is not None else "wait",
            "action_error": self._last_action_error,
            "done": bool(self.done),
        }

    def observe(self) -> Any:
        with self._lock:
            return self._observation_unlocked()

    def score_breakdown(self) -> dict[str, float]:
        with self._lock:
            deliveries = float(sum(self._event_counts.get("soup_delivery", [0, 0])))
            return {
                "sparse_team_reward": float(self._score),
                "sparse_agent_0": float(self._score_by_agent[0]),
                "sparse_agent_1": float(self._score_by_agent[1]),
                "deliveries": deliveries,
                "episode_length": float(self._ticks),
                "shaped_reward_diagnostic": float(self._shaped_score),
                "collision_ticks": float(self._collision_ticks),
                "stuck_ticks": float(self._stuck_ticks),
                "pot_idle_time": float(self._pot_idle_time),
                "handoff_attempts": float(self._handoff_attempts),
                "handoff_successes": float(self._handoff_successes),
                "handoff_failures": float(max(0, self._handoff_attempts - self._handoff_successes)),
            }

    def info(self) -> dict[str, Any]:
        with self._lock:
            breakdown = self.score_breakdown()
            return {
                "valid_trial": self._invalid_reason is None,
                "invalid_reason": self._invalid_reason,
                "repository_commit": REPOSITORY_COMMIT,
                "seed": int(self._seed),
                "episode_id": int(self._episode_id),
                "layout_name": self._scenario.get("layout"),
                "scenario": _json_safe(self._scenario),
                "controlled_agent_index": int(self._controlled_index),
                "partner_agent_index": int(self._partner_index),
                "partner_style": self._partner.style if self._partner is not None else None,
                "ticks": int(self._ticks),
                "horizon": HORIZON,
                "score": float(self._score),
                "score_breakdown": breakdown,
                "event_counts_by_agent": {event: list(rows) for event, rows in self._event_counts.items()},
            }

    def describe(self) -> str:
        return (
            "Overcooked-AI symbolic gridworld, pinned to HumanCompatibleAI/overcooked_ai "
            f"commit {REPOSITORY_COMMIT}. This is a realtime two-chef cooperative kitchen: "
            "the world advances exactly one MDP transition every 0.166667 seconds, for at most "
            f"{HORIZON} ticks. You control one chef; a deterministic scripted partner controls the "
            "other, and the observation says which index is yours. Walkable cells are spaces in "
            "terrain_mtx; O/T/D/P/S/X mean onion/tomato/dish dispenser, pot, serving window, "
            "and counter. A chef faces the direction of its last movement. INTERACT operates the "
            "feature in the facing cell: collect an ingredient/dish, place an ingredient in a pot, "
            "start cooking under modern dynamics, take ready soup with a dish, place/pick up an "
            "object on a counter, or deliver soup. Pots cook automatically after explicit start; "
            "ingredients and recipes are defined by state.all_orders, and a delivery is worth the "
            "active recipe value (bonus_orders can multiply it). The score is cumulative sparse "
            "team delivery reward only; shaped subgoal rewards are diagnostics and never added to "
            "score. The complete JSON observation includes the terrain, serialized players and "
            "objects (including soup ingredients/cook state), orders, tick, sparse reward, recent "
            "events, action history, partner task, and diagnostics. Use exactly one of the six "
            "advertised primitive action names per tick; directional actions are not multi-step plans. "
            "A missed action repeats the previous valid action, while malformed input becomes stay. "
            "Seeds 0/1/2 select cramped_room, asymmetric_advantages, and coordination_ring; held-out "
            "seed 100 validates counter_circuit_o_1order, while later held-out seeds cover explicit "
            "handoff, mixed tomato, counter-circuit, reversed-agent, and randomized-start cases."
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # No subprocess, socket, display, or background thread is started by
            # this adaptor. Drop references so a completed episode is collectible.
            self._partner = None
            self._env = None
            self._mdp = None
            self._done = True


def make_game(**kwargs: Any) -> OvercookedRealtimeGame:
    return OvercookedRealtimeGame(**kwargs)
