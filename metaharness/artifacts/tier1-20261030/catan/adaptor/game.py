"""Tier 1 Catanatron adaptor; engine plumbing adapted from game_v1.py."""
from __future__ import annotations

import json
import random
from enum import Enum
from typing import Any

from metaharness.games.base import ActionSpec
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, RESOURCES, DEVELOPMENT_CARDS
from catanatron.models.player import Color, Player
from catanatron.players.minimax import AlphaBetaPlayer, DebugStateNode
from catanatron.state_functions import (
    get_actual_victory_points, get_dev_cards_in_hand, get_longest_road_color,
    get_longest_road_length, get_largest_army, get_visible_victory_points,
    player_key, player_num_resource_cards,
)

TIER = 1
DETERMINISTIC = True
# Equal seat mixtures: each main split visits all four seats once.
SEEDS = {
    "train": [101, 102, 103, 104],
    "heldout": [201, 202, 203, 204],
    "dev": [301, 302],
    "validation": [401, 402, 403, 404],
}
EVAL_KWARGS: dict[str, Any] = {}
ENGINE_COMMIT = "ecf931181b9a65bb4116a2153fb78c16f1438e00"
OPPONENT = {"class": "AlphaBetaPlayer", "depth": 2, "value_fn": "base",
            "candidate_pruning": False, "epsilon": None, "time_limit": None}
CAPABILITIES = {"standard_full_board": True, "domestic_trading": False,
                "maritime_trading": True, "chosen_discards": True,
                "pre_roll_development_cards": True, "robber_victim": "combined",
                "bank_counts_public": True, "initial_placement": "player_choices"}
RULES = {"target_vp": 10, "discard_limit": 7, "friendly_robber": False,
         "number_placement": "official_spiral", "expansions": False,
         "victory_check": "engine checks all players after every action"}
_ACTION_DESCRIPTIONS = {
    "roll": "Roll two dice once in your normal turn.",
    "end_turn": "Finish your normal turn.",
    "build_road": "Build a road on the offered edge [vertex, vertex].",
    "build_settlement": "Place a settlement on the offered vertex.",
    "build_city": "Upgrade the offered settlement to a city.",
    "buy_development_card": "Buy a private development card from the deck.",
    "move_robber": "Choose the offered [hex coordinate, victim or null].",
    "discard_resource": "Choose one resource to discard; repeat until required count is discarded.",
    "play_knight_card": "Play an old Knight, then choose a robber move.",
    "play_year_of_plenty": "Take the offered pair of resources from the bank.",
    "play_monopoly": "Take every opponent card of the offered resource.",
    "play_road_building": "Play an old Road Building card, then choose up to two roads.",
    "maritime_trade": "Trade the offered resources (null padding) for the final resource.",
    "noop": "Wait without changing state; no game clock runs.",
}


def _json_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return value


def _action_sort_key(action):
    return (action.action_type.value, json.dumps(_json_value(action.value), sort_keys=True))


class IsolatedGame(Game):
    """Copies own their RNG and legal lists, including nested search copies."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "playable_actions"):
            self.playable_actions.sort(key=_action_sort_key)

    def copy(self):
        result = super().copy()
        result.__class__ = IsolatedGame
        result.random = random.Random(0)
        result.random.setstate(self.state.random.getstate())
        result.state.random = result.random
        result.playable_actions = list(self.playable_actions)
        return result

    def execute(self, *args, **kwargs):
        result = super().execute(*args, **kwargs)
        self.playable_actions.sort(key=_action_sort_key)
        return result


class StrongPlayer(AlphaBetaPlayer):
    """Upstream AlphaBeta; finish depth 2 instead of using a timing-dependent cutoff."""
    def __init__(self, color):
        super().__init__(color, AlphaBetaPlayer.Params(
            depth=2, value_fn="base", prunning=False, epsilon=None))

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        node = DebugStateNode(str(len(game.state.action_records)), self.color)
        action, _ = self.alphabeta(game.copy(), self.params.depth,
                                   -float("inf"), float("inf"), float("inf"), node)
        if action not in playable_actions:
            raise RuntimeError("AlphaBeta failed to return a legal action")
        return action


class _Controlled(Player):
    IS_BOT = False

    def decide(self, game, playable_actions):
        raise RuntimeError("The adaptor must pause at controlled-player decisions")


class CatanGame:
    name = "Settlers of Catan (Catanatron AlphaBeta depth 2)"
    action_spec = ActionSpec(names=list(_ACTION_DESCRIPTIONS),
                            descriptions=_ACTION_DESCRIPTIONS, noop="noop",
                            notes="Submit a current legal_actions ID, or 'type JSON-parameters'. "
                                  "Names alone work only when exactly one action of that type is legal.")

    def __init__(self, max_turns=1000, max_actions=20000):
        self.max_turns = max(1, int(max_turns))
        self.max_actions = max(1, int(max_actions))
        self.game = None
        self.controlled = Color.RED
        self._ids = {}
        self._seed = None
        self._action_count = 0
        self._completed_turns = 0
        self._termination = "uninitialized"
        self._winner = None
        self._last_error = None
        self._invalid_reason = None
        self._rejections = 0
        self._events_start = 0
        self._initial_map = None

    def reset(self, seed):
        self.close()
        self._seed = int(seed)
        # Generate normal shuffled seats, then designate the requested realized seat.
        colors = list(Color)
        self.game = IsolatedGame([StrongPlayer(c) for c in colors], seed=self._seed,
                                 discard_limit=7, friendly_robber=False,
                                 vps_to_win=10, number_placement="official_spiral")
        seat = self._seed % 4
        self.controlled = self.game.state.colors[seat]
        controlled = _Controlled(self.controlled)
        self.game.state.players[seat] = controlled
        self.game.observers = list(self.game.state.players)
        self._action_count = 0
        self._completed_turns = 0
        self._termination = "running"
        self._winner = None
        self._last_error = None
        self._invalid_reason = None
        self._rejections = 0
        self._events_start = 0
        self._initial_map = self._board_json()
        try:
            self._advance_to_controlled()
        except Exception as exc:
            self._engine_failure(exc)
        return self.observe()

    @property
    def done(self):
        return self._termination != "running"

    @property
    def score(self):
        # Win is primary; true VP supplies a strictly smaller secondary term.
        vp = get_actual_victory_points(self.game.state, self.controlled) if self.game else 0
        return float(self._termination == "won") + min(float(vp), 10.0) / 100.0

    def _finish_if_needed(self):
        winner = self.game.winning_color()
        if winner is not None:
            self._winner = winner.value
            self._termination = "won" if winner == self.controlled else "lost"
        elif self._action_count >= self.max_actions or self._completed_turns >= self.max_turns:
            self._termination = "truncated"

    def _engine_failure(self, exc):
        self._invalid_reason = f"{type(exc).__name__}: {exc}"[:500]
        self._last_error = self._invalid_reason
        self._termination = "engine_error"
        self._ids = {}

    def _execute(self, action):
        self.game.execute(action)
        self._action_count += 1
        self._completed_turns += int(action.action_type == ActionType.END_TURN)
        self._finish_if_needed()

    def _advance_to_controlled(self):
        while not self.done and self.game.state.current_color() != self.controlled:
            actions = self.game.playable_actions
            if not actions:
                raise RuntimeError("Engine returned no legal actions")
            player = self.game.state.current_player()
            # Even built-in policies run on an independent game/RNG copy.
            action = player.decide(self.game.copy(), list(actions))
            self._execute(action)
        self._refresh_ids()

    def _refresh_ids(self):
        actions = self.game.playable_actions if self.game and not self.done else []
        self._ids = {f"r{self._action_count}:a{i}": action for i, action in enumerate(actions)}

    def _resolve(self, action):
        if isinstance(action, dict):
            action = action.get("id", action.get("action"))
        if not isinstance(action, str):
            return None
        if action in self._ids:
            return self._ids[action]
        parts = action.strip().split(maxsplit=1)
        if not parts:
            return None
        matches = [a for a in self._ids.values() if a.action_type.value.lower() == parts[0]]
        if len(parts) == 1:
            return matches[0] if len(matches) == 1 else None
        try:
            value = json.loads(parts[1])
        except (ValueError, RecursionError):
            return None
        return next((a for a in matches if _json_value(a.value) == value), None)

    def step(self, action):
        if self.game is None or self.done:
            return self.observe()
        self._last_error = None
        if isinstance(action, str) and action == "noop":
            return self.observe()
        resolved = self._resolve(action)
        if resolved is None:
            self._last_error = "Illegal, malformed, ambiguous, or stale action; choose a current legal_actions ID."
            self._rejections += 1
            return self.observe()
        self._events_start = self._action_count
        try:
            self._execute(resolved)
            self._advance_to_controlled()
        except Exception as exc:
            self._engine_failure(exc)
        return self.observe()

    def _public_event(self, record, index):
        action = record.action
        value, result = action.value, record.result
        if action.color != self.controlled:
            if action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                result = None
            elif action.action_type == ActionType.DISCARD_RESOURCE:
                value = result = None
            elif action.action_type == ActionType.MOVE_ROBBER and action.value[1] != self.controlled:
                result = None
        return {"index": index, "player": action.color.value,
                "type": action.action_type.value.lower(),
                "parameters": _json_value(value), "result": _json_value(result)}

    def _board_json(self):
        state = self.game.state
        hexes = [{"id": tile.id, "coordinate": list(coord), "terrain": tile.resource,
                  "number": tile.number, "robber": coord == state.board.robber_coordinate,
                  "vertices": sorted(set(tile.nodes.values()))}
                 for coord, tile in sorted(state.board.map.land_tiles.items(), key=lambda x: x[1].id)]
        edge_ids = sorted({tuple(sorted(e)) for t in state.board.map.land_tiles.values() for e in t.edges.values()})
        vertices = []
        for node in sorted(state.board.map.land_nodes):
            building = state.board.buildings.get(node)
            port = [(r, ns) for r, ns in state.board.map.port_nodes.items() if node in ns]
            vertices.append({"id": node, "building": building[1] if building else None,
                             "owner": building[0].value if building else None,
                             "adjacent_hexes": sorted(t.id for t in state.board.map.adjacent_tiles.get(node, [])),
                             "neighbors": sorted(e[1] if e[0] == node else e[0] for e in edge_ids if node in e),
                             "port": ("ANY" if port[0][0] is None else port[0][0]) if port else None})
        return {"hexes": hexes, "vertices": vertices,
                "edges": [{"id": list(e), "owner": state.board.roads[e].value if e in state.board.roads else None}
                          for e in edge_ids]}

    def _player_json(self, color):
        state = self.game.state
        key = player_key(state, color)
        result = {"id": color.value, "resource_count": player_num_resource_cards(state, color),
                  "development_count": get_dev_cards_in_hand(state, color),
                  "played_knights": state.player_state[f"{key}_PLAYED_KNIGHT"],
                  "public_vp": get_visible_victory_points(state, color),
                  "pieces": {p.lower(): state.player_state[f"{key}_{p}_AVAILABLE"]
                             for p in ("ROADS", "SETTLEMENTS", "CITIES")},
                  "ports": sorted("ANY" if r is None else r for r in state.board.get_player_port_resources(color))}
        if color == self.controlled:
            result.update({"resources": {r: state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES},
                           "development_cards": {d: state.player_state[f"{key}_{d}_IN_HAND"] for d in DEVELOPMENT_CARDS},
                           "old_development_cards": {d: state.player_state.get(f"{key}_{d}_OWNED_AT_START", 0)
                                                     for d in DEVELOPMENT_CARDS},
                           "has_rolled": state.player_state[f"{key}_HAS_ROLLED"],
                           "has_played_development_card": state.player_state[f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
                           "true_vp": get_actual_victory_points(state, color)})
        return result

    def observe(self):
        if self.game is None:
            return {"phase": "uninitialized", "terminal": True, "legal_actions": []}
        state = self.game.state
        self._refresh_ids()
        army_color, army_size = get_largest_army(state)
        road_color = get_longest_road_color(state)
        # Keep observations bounded even if a pathological action sequence is long.
        event_start = max(self._events_start, self._action_count - 128)
        events = [self._public_event(r, i) for i, r in enumerate(state.action_records[event_start:], event_start)]
        last_roll = next((r.result for r in reversed(state.action_records) if r.action.action_type == ActionType.ROLL), None)
        return {"phase": state.current_prompt.value.lower(), "current_player": state.current_color().value,
                "turn_owner": state.colors[state.current_turn_index].value,
                "seat_order": [c.value for c in state.colors], "controlled_seat": self._seed % 4,
                "turn_number": self._completed_turns, "pending_decision": state.current_prompt.value.lower(),
                "board": self._board_json(), "self": self._player_json(self.controlled),
                "opponents": [self._player_json(c) for c in state.colors if c != self.controlled],
                "awards": {"longest_road_owner": road_color.value if road_color else None,
                           "longest_road_length": max(get_longest_road_length(state, c) for c in state.colors),
                           "largest_army_owner": army_color.value if army_color else None, "largest_army_size": army_size or 0},
                "bank_counts": dict(zip(RESOURCES, state.resource_freqdeck)),
                "development_cards_remaining": len(state.development_listdeck),
                "dice_result": _json_value(last_roll), "events": events,
                "events_truncated": event_start > self._events_start,
                "legal_actions": [{"id": aid, "type": a.action_type.value.lower(), "parameters": _json_value(a.value)}
                                  for aid, a in self._ids.items()],
                "terminal": self.done, "termination": self._termination, "winner": self._winner,
                "action_count": self._action_count, "revision": self._action_count,
                **({"error": self._last_error} if self._last_error else {})}

    def describe(self):
        return (
            "Four-player base Catan on a seeded 19-hex standard island; three opponents are upstream AlphaBeta, "
            "full depth 2, base value function, no candidate pruning or exploration. Reach 10 true victory points. "
            "Settlements score 1, cities 2; hidden VP cards score 1; Longest Road (at least 5) and Largest Army "
            "(at least 3 played Knights) score 2 each, transferring only when exceeded. Engine checks victory after every action. "
            "Snake setup gives two settlements and two roads; second settlement gives adjacent resources. "
            "Settlement distance rule forbids adjacent buildings. Subsequent settlements need your road; roads connect "
            "to your network and cannot extend through enemy buildings. Roll two dice once each turn; matching hexes "
            "produce 1 per settlement or 2 per city unless blocked by robber or bank shortage. A 7 makes hands over 7 "
            "discard half (rounded down), one chosen card per action; then roller moves robber and picks a victim "
            "for one random stolen resource. Road costs WOOD+BRICK; settlement WOOD+BRICK+SHEEP+WHEAT; "
            "city 3 ORE+2 WHEAT; development 1 each ORE+WHEAT+SHEEP. Maritime trade is 4:1, ANY port 3:1, "
            "matching resource port 2:1. No domestic trading is exposed. Play at most one old non-VP development "
            "card per turn, before or after rolling: Knight moves robber, Road Building grants up to two roads, "
            "Year of Plenty takes two available resources, Monopoly takes all opponent cards of a chosen resource. "
            "Use current legal_actions as authoritative choices. Submit an ID such as 'r16:a2', {'id':'r16:a2'}, "
            "or 'type JSON-parameters', e.g. 'build_settlement 12', 'build_road [1,2]', 'play_monopoly \"ORE\"'. "
            "A type alone works if exactly one such action is legal. IDs expire after an applied action; noop and "
            "invalid actions do nothing. Each step applies your action and runs bots to your next decision. "
            "Observations include hex/vertex/edge topology, ports (ANY=3:1), pieces remaining, own private hand and "
            "true VP, old cards, opponent hand counts and public VP, awards, public bank, dice and up to 128 recent "
            "events. Events suppress opponents' purchases/discards and theft identities unless you are involved. "
            "Unknown hands and deck order are never shown. Turn number counts completed normal turns, excluding setup. "
            "Seats rotate by seed modulo 4. Hard caps truncate, never award a win. Score = win + min(true_VP,10)/100; "
            "report win rate as primary and mean terminal true VP separately. No in-game decision clock runs."
        )

    def info(self):
        state = self.game.state if self.game else None
        return {"valid_trial": self._invalid_reason is None, "invalid_reason": self._invalid_reason,
                "seed": self._seed, "engine": "catanatron 3.3.0", "engine_commit": ENGINE_COMMIT,
                "opponents": [dict(OPPONENT) for _ in range(3)],
                "controlled_color": self.controlled.value, "controlled_seat": self._seed % 4 if self._seed is not None else None,
                "seat_order": [c.value for c in state.colors] if state else [],
                "map": self._initial_map, "capabilities": dict(CAPABILITIES), "rules": dict(RULES),
                "caps": {"turns": self.max_turns, "actions": self.max_actions},
                "win": int(self._termination == "won"), "winner": self._winner,
                "true_vp": get_actual_victory_points(state, self.controlled) if state else 0,
                "public_vp": get_visible_victory_points(state, self.controlled) if state else 0,
                "turns": self._completed_turns, "engine_turn_counter": state.num_turns if state else 0, "action_count": self._action_count,
                "termination": self._termination, "truncated": self._termination == "truncated",
                "illegal_action_count": self._rejections}

    def score_breakdown(self):
        info = self.info()
        return {key: float(info[key]) for key in ("win", "true_vp", "public_vp", "turns", "action_count", "truncated")}

    def close(self):
        self.game = None
        self._ids.clear()
        self._termination = "closed"


def make_game(**kwargs):
    return CatanGame(**kwargs)
