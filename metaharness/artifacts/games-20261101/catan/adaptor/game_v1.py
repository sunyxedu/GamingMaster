"""Tier 1 Settlers of Catan adaptor backed by Catanatron 3.2.1.

The engine remains the authority for legality and rules.  The adaptor pauses before
each decision by the controlled player and runs the three built in opponents between
those decisions.
"""
from __future__ import annotations

import random
from enum import Enum
from typing import Any

from metaharness.games.base import ActionSpec

from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, RESOURCES
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_longest_road_color,
    get_longest_road_length,
    get_largest_army,
    get_visible_victory_points,
    player_key,
    player_num_resource_cards,
)

TIER = 1
DETERMINISTIC = True
SEEDS = {
    "train": [101, 102, 103, 104, 105, 106, 107, 108],
    "heldout": [201, 202, 203, 204, 205, 206, 207, 208],
    "dev": [301, 302],
    "validation": [401, 402, 403, 404, 405, 406, 407, 408],
}
EVAL_KWARGS: dict[str, Any] = {}

_ACTIONS = [
    "roll", "end_turn", "build_road", "build_settlement", "build_city",
    "buy_development_card", "move_robber", "discard", "play_knight_card",
    "play_year_of_plenty", "play_monopoly", "play_road_building", "maritime_trade",
]
_DESCRIPTIONS = {
    "roll": "Roll dice (once per normal turn).",
    "end_turn": "Finish the current turn.",
    "build_road": "Build on an edge; use the legal action id or `build_road U V`.",
    "build_settlement": "Build at a vertex; use the legal action id or `build_settlement N`.",
    "build_city": "Upgrade your settlement; use the legal action id or `build_city N`.",
    "buy_development_card": "Buy one development card.",
    "move_robber": "Move the robber; use the legal action id (victim is engine validated).",
    "discard": "Discard the engine-selected cards after a 7.",
    "play_knight_card": "Play a Knight and move the robber.",
    "play_year_of_plenty": "Take one or two resources; use the legal action id.",
    "play_monopoly": "Name a resource to monopolize; use the legal action id.",
    "play_road_building": "Play Road Building, then choose its roads.",
    "maritime_trade": "Trade with the bank; use the legal action id.",
}


class _Controlled(Player):
    def decide(self, game: Game, playable_actions):  # pragma: no cover - never called
        return playable_actions[0]


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(x) for x in value]
    if isinstance(value, list):
        return [_json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return value


class CatanGame:
    name = "Settlers of Catan (Catanatron)"
    action_spec = ActionSpec(names=_ACTIONS, descriptions=_DESCRIPTIONS,
                             notes="Use legal_actions[].id from the observation. IDs are valid only for the current observation. "
                                   "The engine uses standard base-game rules and randomizes the standard island per seed.")

    def __init__(self, max_turns: int = 1000, max_actions: int = 20000):
        self.max_turns = int(max_turns)
        self.max_actions = int(max_actions)
        self.game: Game | None = None
        self.controlled: Color = Color.RED
        self._ids: dict[str, Action] = {}
        self._last_error: str | None = None
        self._action_count = 0
        self._seed: int | None = None
        self._termination = "running"
        self._winner: str | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        # Game seeds Python's RNG; restore it after construction so unrelated harness
        # randomness is not consumed by planning code.
        self._seed = int(seed)
        players = [
            _Controlled(Color.RED, is_bot=False),
            RandomPlayer(Color.BLUE),
            WeightedRandomPlayer(Color.ORANGE),
            VictoryPointPlayer(Color.WHITE),
        ]
        self.game = Game(players, seed=self._seed)
        self.controlled = Color.RED
        self._action_count = 0
        self._last_error = None
        self._termination = "running"
        self._winner = None
        self._ids.clear()
        self._advance_to_controlled()
        return self.observe()

    @property
    def done(self) -> bool:
        if self.game is None:
            return False
        return self._termination != "running" or self.game.winning_color() is not None

    @property
    def score(self) -> float:
        if self.game is None:
            return 0.0
        return float(get_actual_victory_points(self.game.state, self.controlled))

    def _finish_if_needed(self) -> None:
        if self.game is None:
            return
        winner = self.game.winning_color()
        if winner is not None:
            self._winner = winner.value
            self._termination = "won" if winner == self.controlled else "lost"
        elif self._action_count >= self.max_actions or self.game.state.num_turns >= self.max_turns:
            self._termination = "truncated"

    def _advance_to_controlled(self) -> None:
        assert self.game is not None
        while not self.done:
            state = self.game.state
            if state.current_color() == self.controlled:
                self._refresh_ids()
                return
            actions = state.playable_actions
            if not actions:
                self._termination = "engine_error"
                self._last_error = "engine provided no legal action"
                return
            player = state.current_player()
            # Every non-controlled seat is a verified Catanatron built-in bot.
            action = player.decide(self.game, actions)
            self.game.execute(action)
            self._action_count += 1
            self._finish_if_needed()
        self._refresh_ids()

    def _refresh_ids(self) -> None:
        self._ids = {f"a{i}": action for i, action in enumerate(self.game.state.playable_actions)} if self.game else {}

    def _resolve(self, action: Any) -> Action | None:
        if isinstance(action, dict):
            action = action.get("id", action.get("action"))
        if isinstance(action, str) and action in self._ids:
            return self._ids[action]
        # Convenience grammar for simple parameterized actions. All resulting
        # actions still have to equal an engine-provided legal action.
        if not isinstance(action, str) or self.game is None:
            return None
        parts = action.split()
        typ = parts[0].lower()
        candidates = self.game.state.playable_actions
        wanted_type = {
            "roll": ActionType.ROLL, "end_turn": ActionType.END_TURN,
            "buy_development_card": ActionType.BUY_DEVELOPMENT_CARD,
            "discard": ActionType.DISCARD, "play_knight_card": ActionType.PLAY_KNIGHT_CARD,
            "play_road_building": ActionType.PLAY_ROAD_BUILDING,
            "build_road": ActionType.BUILD_ROAD, "build_settlement": ActionType.BUILD_SETTLEMENT,
            "build_city": ActionType.BUILD_CITY, "move_robber": ActionType.MOVE_ROBBER,
            "play_monopoly": ActionType.PLAY_MONOPOLY,
        }.get(typ)
        if wanted_type is None:
            return None
        matches = [a for a in candidates if a.action_type == wanted_type]
        if wanted_type in {ActionType.ROLL, ActionType.END_TURN, ActionType.BUY_DEVELOPMENT_CARD,
                           ActionType.DISCARD, ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_ROAD_BUILDING}:
            return matches[0] if len(matches) == 1 else None
        if not parts[1:]:
            return None
        try:
            n = int(parts[1])
            if wanted_type in {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}:
                return next((a for a in matches if a.value == n), None)
            if wanted_type == ActionType.BUILD_ROAD and len(parts) >= 3:
                edge = tuple(sorted((n, int(parts[2]))))
                return next((a for a in matches if tuple(sorted(a.value)) == edge), None)
        except (TypeError, ValueError):
            return None
        return None

    def step(self, action: Any) -> dict[str, Any]:
        if self.game is None:
            return {"error": "reset() must be called first"}
        if self.done:
            return self.observe()
        self._last_error = None
        resolved = self._resolve(action)
        if resolved is None:
            self._last_error = f"illegal or stale action: {action!r}"
            self._refresh_ids()
            return self.observe()
        try:
            self.game.execute(resolved)
            self._action_count += 1
            self._finish_if_needed()
            if not self.done:
                self._advance_to_controlled()
        except Exception as exc:  # malformed/engine illegal actions never end the trial
            self._last_error = f"{type(exc).__name__}: {exc}"[:300]
            self._refresh_ids()
        return self.observe()

    def _action_json(self, action: Action, aid: str) -> dict[str, Any]:
        value = action.value
        return {"id": aid, "type": action.action_type.value.lower(), "parameters": _json_value(value)}

    def _public_event(self, action: Action | None) -> Any:
        """Serialize the public part of the engine's fully specified action log.

        Catanatron annotates robber actions with the stolen resource and purchase
        actions with the drawn development card. Those details remain private
        when another player acted.
        """
        if action is None:
            return None
        value = action.value
        if action.color != self.controlled:
            if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                value = None
            elif action.action_type == ActionType.MOVE_ROBBER and isinstance(value, tuple):
                value = tuple(value[:2]) + (None,)
            elif action.action_type == ActionType.DISCARD:
                value = None
        return {"player": action.color.value, "type": action.action_type.value.lower(),
                "parameters": _json_value(value)}

    def observe(self) -> dict[str, Any]:
        if self.game is None:
            return {"phase": "uninitialized", "terminal": True, "legal_actions": []}
        state = self.game.state
        self._refresh_ids()
        key = player_key(state, self.controlled)
        self_color = self.controlled.value
        opponents = []
        for color in state.colors:
            if color == self.controlled:
                continue
            k = player_key(state, color)
            opponents.append({
                "id": color.value,
                "resource_count": player_num_resource_cards(state, color),
                "development_count": get_dev_cards_in_hand(state, color),
                "played_knights": state.player_state[f"{k}_PLAYED_KNIGHT"],
                "public_vp": get_visible_victory_points(state, color),
                "pieces": {"roads": state.player_state[f"{k}_ROADS_AVAILABLE"], "settlements": state.player_state[f"{k}_SETTLEMENTS_AVAILABLE"], "cities": state.player_state[f"{k}_CITIES_AVAILABLE"]},
            })
        hexes = []
        for coord, tile in sorted(state.board.map.land_tiles.items(), key=lambda x: x[1].id):
            hexes.append({"id": tile.id, "coordinate": list(coord), "terrain": tile.resource, "number": tile.number,
                          "robber": coord == state.board.robber_coordinate, "vertices": sorted(set(tile.nodes.values()))})
        vertices = []
        for node in sorted(state.board.map.land_nodes):
            building = state.board.buildings.get(node)
            vertices.append({"id": node, "building": None if building is None else building[1],
                             "owner": None if building is None else building[0].value,
                             "adjacent_hexes": [t.id for t in state.board.map.adjacent_tiles.get(node, [])],
                             "port": next((r for r, ns in state.board.map.port_nodes.items() if node in ns), None)})
        edges = {}
        for tile in state.board.map.land_tiles.values():
            for edge in tile.edges.values():
                e = tuple(sorted(edge)); edges[e] = state.board.roads.get(e)
        awards = get_largest_army(state)
        legal = [self._action_json(a, aid) for aid, a in self._ids.items()]
        return {
            "phase": state.current_prompt.value.lower(), "current_player": state.current_color().value,
            "seat_order": [c.value for c in state.colors], "turn_number": state.num_turns,
            "pending_decision": state.current_prompt.value.lower(),
            "board": {"hexes": hexes, "vertices": vertices,
                      "edges": [{"id": list(e), "owner": c.value if c else None} for e, c in sorted(edges.items())]},
            "self": {"id": self_color, "resources": {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES},
                     "development_cards": {d: state.player_state[f"{key}_{d}_IN_HAND"] for d in ("KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT")},
                     "public_vp": get_visible_victory_points(state, self.controlled), "true_vp": self.score,
                     "played_knights": state.player_state[f"{key}_PLAYED_KNIGHT"],
                     "pieces": {"roads": state.player_state[f"{key}_ROADS_AVAILABLE"], "settlements": state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"], "cities": state.player_state[f"{key}_CITIES_AVAILABLE"]},
                     "ports": list(state.board.get_player_port_resources(self.controlled))},
            "opponents": opponents,
            "awards": {"longest_road_owner": get_longest_road_color(state).value if get_longest_road_color(state) else None,
                       "longest_road_length": max((get_longest_road_length(state, c) for c in state.colors), default=0),
                       "largest_army_owner": awards[0].value if awards[0] else None, "largest_army_size": awards[1] or 0},
            "bank_counts": {r: state.resource_freqdeck[i] for i, r in enumerate(RESOURCES)},
            "legal_actions": legal, "last_event": self._public_event(state.actions[-1] if state.actions else None),
            "terminal": self.done, "termination": self._termination, "winner": self._winner,
            "action_count": self._action_count,
            **({"error": self._last_error} if self._last_error else {}),
        }

    def describe(self) -> str:
        return ("Standard four-player base-game Settlers of Catan, played to 10 true victory points. "
                "The controlled seat acts against Catanatron Random, WeightedRandom, and VictoryPoint built-in bots. "
                "The randomized standard island, tokens, ports, dice, development deck, and seat order are seeded by reset(seed). "
                "Each step applies exactly one legal action and then advances opponents until your next decision. "
                "Read legal_actions from the observation and submit its id (a0, a1, ...); ids expire after each step. "
                "The engine handles initial placement, rolls, production, seven/discards, robber theft, building, maritime trade, "
                "development cards, awards, and hidden information. Your resource and development identities are shown; "
                "opponent hands are represented only by counts. A hard cap reports termination='truncated' and is not a win.")

    def info(self) -> dict[str, Any]:
        return {"valid_trial": True, "seed": self._seed, "engine": "catanatron 3.2.1",
                "opponents": ["RandomPlayer", "WeightedRandomPlayer", "VictoryPointPlayer"],
                "controlled_color": self.controlled.value, "winner": self._winner,
                "true_vp": self.score, "public_vp": (get_visible_victory_points(self.game.state, self.controlled) if self.game else 0),
                "turns": self.game.state.num_turns if self.game else 0, "action_count": self._action_count,
                "termination": self._termination, "seat_order": [c.value for c in self.game.state.colors] if self.game else []}

    def score_breakdown(self) -> dict[str, float]:
        return {"true_victory_points": self.score, "win": float(self._termination == "won"),
                "public_victory_points": float(self.info()["public_vp"]), "turns": float(self.info()["turns"]),
                "actions": float(self._action_count)}

    def close(self) -> None:
        self.game = None
        self._ids.clear()


def make_game(**kwargs: Any) -> CatanGame:
    return CatanGame(**kwargs)
