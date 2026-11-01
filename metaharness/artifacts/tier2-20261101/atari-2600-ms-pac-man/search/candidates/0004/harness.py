"""Graph-and-guard controller for the real-time CX2675 Ms. Pac-Man task.

The controller deliberately does not ask a model to rediscover the maze on every
67 ms quantum.  It learns the corridor graph from the public dot coordinates,
keeps a route running while inference is pending, and uses Jev only for a local
junction or a measured ghost hazard.  Astra supplies a slow risk policy in the
background; all executable movement has a deterministic fallback.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


CARDINALS = ("up", "right", "left", "down")
DXDY = {
    "up": (0, -1),
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


# Actor coordinates in the adaptor are sprite upper-left coordinates.  A dot
# coordinate is approximately three pixels right and four pixels down from it.
# This offset is learned from the public traces (75,99 -> the 72,103 dot), but
# is only used for nearest-node association, never as a wall assertion.
ACTOR_TO_DOT = (3, 4)
DOT_EDGE_TOLERANCE = 12.5


def _finite_number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _xy(obj):
    """Return a finite integer-ish x/y pair from an adaptor object."""
    if not isinstance(obj, dict):
        return (0, 0)
    xy = obj.get("xy")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        return (_integer(xy[0]), _integer(xy[1]))
    return (_integer(obj.get("x")), _integer(obj.get("y")))


def _visible(obj, default=True):
    if not isinstance(obj, dict):
        return False
    value = obj.get("visible", default)
    return bool(value)


def _dedupe_points(objects, *, y_limit=220):
    points = set()
    if not isinstance(objects, list):
        return points
    for obj in objects:
        if not isinstance(obj, dict) or not _visible(obj, True):
            continue
        x, y = _xy(obj)
        if 0 <= x <= 180 and 0 <= y <= y_limit:
            points.add((x, y))
    return points


def _edge_action(a, b):
    """Joystick direction for an ordinary edge, including a learned wrap."""
    ax, ay = a
    bx, by = b
    if ay == by:
        dx = bx - ax
        if abs(dx) > 40:  # a side-tunnel edge: leave in the physical direction
            return "left" if ax > bx else "right"
        return "right" if dx > 0 else "left"
    if ax == bx:
        return "down" if by > ay else "up"
    # A malformed/inferred diagonal edge should never be sent as a diagonal.
    return "right" if bx > ax else "left"


def _bucket_distance(distance):
    if distance <= 16:
        return "adjacent"
    if distance <= 36:
        return "near"
    if distance <= 72:
        return "medium"
    return "far"


def _safe_text(value):
    return value if isinstance(value, str) else ""


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.spec = None
        self.valid_names = set()
        self.noop = "noop"

        self.maze_key = None
        self.nodes = set()
        self.adj = {}
        self.extra_edges = set()
        self.remaining = set()
        self.power_points = set()
        self.route = []
        self.route_target = None
        self.current_action = "left"
        self.last_node = None
        self.previous_player = None
        self.previous_deaths = 0
        self.previous_clears = 0
        self.last_score = 0.0

        self.risk_mode = "balanced"
        self.policy_version = 0
        self.astra_future = None
        self.astra_calls = 0

        self.jev_future = None
        self.jev_request = None
        self.last_query_key = None
        self.override = None

    # ---------- contract and observation helpers ----------

    def _prepare_actions(self, game):
        try:
            self.spec = game.action_spec
            names = getattr(self.spec, "names", ())
            self.valid_names = {str(name) for name in names}
            candidate_noop = getattr(self.spec, "noop", "noop")
            self.noop = str(candidate_noop) if str(candidate_noop) in self.valid_names else "noop"
        except Exception:
            self.valid_names = set()
            self.noop = "noop"

    def _valid(self, action):
        action = str(action) if isinstance(action, str) else self.noop
        if not self.valid_names or action in self.valid_names:
            return action
        if self.noop in self.valid_names:
            return self.noop
        return next(iter(self.valid_names), self.noop)

    @staticmethod
    def _player(obs):
        player = obs.get("player", {}) if isinstance(obs, dict) else {}
        x, y = _xy(player)
        return x, y, player

    @staticmethod
    def _maze_key(obs):
        maze = obs.get("maze", {}) if isinstance(obs, dict) else {}
        return (
            obs.get("maze_number"),
            obs.get("mazes_cleared"),
            maze.get("map_id") if isinstance(maze, dict) else None,
        )

    def _player_node(self, obs):
        if not self.nodes:
            return None
        x, y, _ = self._player(obs)
        return self._nearest_node((x + ACTOR_TO_DOT[0], y + ACTOR_TO_DOT[1]))

    def _nearest_node(self, point):
        if not self.nodes:
            return None
        px, py = point
        return min(self.nodes, key=lambda p: (abs(p[0] - px) + abs(p[1] - py), p[1], p[0]))

    def _object_node(self, point):
        if not self.nodes:
            return None
        return self._nearest_node((point[0] + ACTOR_TO_DOT[0], point[1] + ACTOR_TO_DOT[1]))

    def _player_node_distance(self, obs, node):
        if node is None:
            return 999
        x, y, _ = self._player(obs)
        return abs((x + 3) - node[0]) + abs((y + 4) - node[1])

    def _active_ghosts(self, obs):
        ghosts = []
        raw = obs.get("ghosts", []) if isinstance(obs, dict) else []
        if not isinstance(raw, list):
            return ghosts
        for ghost in raw:
            if not isinstance(ghost, dict) or not _visible(ghost, True):
                continue
            x, y = _xy(ghost)
            dx = _integer(ghost.get("dx"))
            dy = _integer(ghost.get("dy"))
            # The decoder reports all four not-yet-launched ghosts at (0,0).
            # A real ghost can be close to an edge, but never has this exact
            # stationary sentinel during the opening.
            if x == 0 and y == 0 and dx == 0 and dy == 0:
                continue
            ghosts.append({
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy,
                "node": self._object_node((x, y)),
                "colour": str(ghost.get("colour", "unknown")),
            })
        return ghosts

    # ---------- online map construction ----------

    def _rebuild_graph(self):
        self.adj = {node: set() for node in self.nodes}
        ordered = sorted(self.nodes)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                same_row = a[1] == b[1] and 0 < abs(a[0] - b[0]) <= DOT_EDGE_TOLERANCE
                same_col = a[0] == b[0] and 0 < abs(a[1] - b[1]) <= DOT_EDGE_TOLERANCE
                if same_row or same_col:
                    self.adj[a].add(b)
                    self.adj[b].add(a)
        for a, b in self.extra_edges:
            if a in self.adj and b in self.adj:
                self.adj[a].add(b)
                self.adj[b].add(a)

    def _reset_map(self, key, visible_points):
        self.maze_key = key
        self.nodes = set(visible_points)
        self.extra_edges = set()
        self.remaining = set(visible_points)
        self.route = []
        self.route_target = None
        self.last_node = None
        self.last_query_key = None
        self.override = None
        self._rebuild_graph()
        self.policy_version += 1

    def _learn_wrap(self, obs, visible_points):
        if self.previous_player is None or not visible_points:
            return
        x, y, player = self._player(obs)
        if not _visible(player, True):
            self.previous_player = (x, y)
            return
        ox, oy = self.previous_player
        if abs(x - ox) > 80 and abs(y - oy) <= 18:
            a = self._nearest_node((ox + 3, oy + 4))
            b = self._nearest_node((x + 3, y + 4))
            if a is not None and b is not None and a != b:
                edge = (a, b) if a <= b else (b, a)
                if edge not in self.extra_edges:
                    self.extra_edges.add(edge)
                    self._rebuild_graph()
                    self.policy_version += 1
        self.previous_player = (x, y)

    def _update_observation(self, obs):
        if not isinstance(obs, dict):
            return
        key = self._maze_key(obs)
        visible_points = _dedupe_points(obs.get("pills", []), y_limit=190)
        if self.maze_key is None:
            self._reset_map(key, visible_points)
        elif key != self.maze_key:
            self._reset_map(key, visible_points)
        else:
            if visible_points - self.nodes:
                self.nodes.update(visible_points)
                self._rebuild_graph()
            self.remaining = set(visible_points)

        self._learn_wrap(obs, visible_points)
        self.power_points = set()
        # The first decoder sample can contain an uninitialised [125,173]
        # duplicate.  Real corner pills in this adaptor are inside this box.
        for point in _dedupe_points(obs.get("power_pills", []), y_limit=165):
            if 0 <= point[0] <= 160 and 0 <= point[1] <= 165:
                self.power_points.add(point)

        deaths = _integer(obs.get("deaths"), self.previous_deaths)
        clears = _integer(obs.get("mazes_cleared"), self.previous_clears)
        if deaths != self.previous_deaths or clears != self.previous_clears:
            self.route = []
            self.route_target = None
            self.override = None
            self.last_query_key = None
        self.previous_deaths = deaths
        self.previous_clears = clears
        self.last_score = _finite_number(obs.get("score"), self.last_score)

    # ---------- graph search and route execution ----------

    def _direction_order(self, preferred):
        # Keep going if possible; at a dead end turn clockwise through the
        # maze rather than repeatedly selecting the first dictionary neighbor.
        if preferred == "left":
            return ("left", "down", "up", "right")
        if preferred == "right":
            return ("right", "up", "down", "left")
        if preferred == "down":
            return ("down", "right", "left", "up")
        return ("up", "left", "right", "down")

    def _ordered_neighbors(self, node, preferred=None):
        if node not in self.adj:
            return []
        order = self._direction_order(preferred or self.current_action)
        rank = {action: i for i, action in enumerate(order)}
        pairs = []
        for neighbor in self.adj[node]:
            action = _edge_action(node, neighbor)
            pairs.append((rank.get(action, 99), action, neighbor))
        pairs.sort(key=lambda item: (item[0], item[1], item[2][1], item[2][0]))
        return [(action, neighbor) for _, action, neighbor in pairs]

    def _path_to_goals(self, start, goals, preferred=None):
        if start is None or start not in self.adj:
            return []
        goals = set(goals)
        goals.discard(start)
        if not goals:
            return []
        parent = {start: None}
        queue = deque([start])
        found = None
        while queue:
            node = queue.popleft()
            for _, neighbor in self._ordered_neighbors(node, preferred if node == start else None):
                if neighbor in parent:
                    continue
                parent[neighbor] = node
                if neighbor in goals:
                    found = neighbor
                    queue.clear()
                    break
                queue.append(neighbor)
            if found is not None:
                break
        if found is None:
            return []
        path = []
        node = found
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def _ensure_route(self, obs, *, emergency=False):
        start = self._player_node(obs)
        if start is None or not self.adj:
            self.route = []
            self.route_target = None
            return start

        # Trim a route as the player reaches its nodes.  Replanning only at a
        # reached target keeps a turn decision stable while the player is in a
        # corridor and prevents model latency from causing oscillation.
        if self.route:
            try:
                index = self.route.index(start)
            except ValueError:
                self.route = []
            else:
                self.route = self.route[index:]
                if len(self.route) <= 1:
                    self.route = []

        if not self.route:
            if emergency and self.power_points:
                power_nodes = []
                for point in self.power_points:
                    node = self._nearest_node(point)
                    if node is not None:
                        power_nodes.append(node)
                path = self._path_to_goals(start, power_nodes, self.current_action)
            else:
                goals = set(self.remaining)
                goals.discard(start)
                if not goals:
                    goals = set(self.nodes)
                    goals.discard(start)
                path = self._path_to_goals(start, goals, self.current_action)
            self.route = path
            self.route_target = path[-1] if path else None
        return start

    def _route_action(self, obs):
        start = self._ensure_route(obs)
        if len(self.route) >= 2:
            return _edge_action(self.route[0], self.route[1])
        if start is not None and self.adj.get(start):
            neighbors = self._ordered_neighbors(start, self.current_action)
            if neighbors:
                return neighbors[0][0]
        _, _, player = self._player(obs)
        direction = player.get("direction") if isinstance(player, dict) else None
        if direction in CARDINALS:
            return direction
        return self.current_action if self.current_action in CARDINALS else "left"

    def _graph_distance(self, start, goal, cap=999):
        if start is None or goal is None or start not in self.adj or goal not in self.adj:
            return cap
        if start == goal:
            return 0
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            node, distance = queue.popleft()
            if distance >= cap:
                continue
            for neighbor in self.adj.get(node, ()):
                if neighbor == goal:
                    return distance + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return cap

    # ---------- deterministic ghost guard ----------

    def _ghost_relation(self, obs, ghost):
        px, py, _ = self._player(obs)
        gx, gy = ghost["x"], ghost["y"]
        dx, dy = ghost["dx"], ghost["dy"]
        distance = abs(px - gx) + abs(py - gy)
        predicted = abs(px - (gx + 3 * dx)) + abs(py - (gy + 3 * dy))
        relx, rely = px - gx, py - gy
        closing = False
        if abs(relx) >= abs(rely) and dx:
            closing = (relx > 0 and dx > 0) or (relx < 0 and dx < 0)
        elif dy:
            closing = (rely > 0 and dy > 0) or (rely < 0 and dy < 0)
        same_lane = abs(px - gx) <= 9 or abs(py - gy) <= 9
        graph_distance = self._graph_distance(self._player_node(obs), ghost.get("node"), cap=12)
        return {
            "distance": distance,
            "predicted": predicted,
            "closing": closing,
            "same_lane": same_lane,
            "graph_distance": graph_distance,
        }

    def _hazard_info(self, obs):
        relations = [self._ghost_relation(obs, ghost) for ghost in self._active_ghosts(obs)]
        level = 0
        for relation in relations:
            distance = relation["distance"]
            predicted = relation["predicted"]
            closing = relation["closing"]
            same_lane = relation["same_lane"]
            graph_distance = relation["graph_distance"]
            if (distance <= 14 or predicted <= 14 or
                    (same_lane and closing and distance <= 30) or
                    (graph_distance <= 1 and closing)):
                level = max(level, 2)
            elif ((same_lane and closing and distance <= 58) or
                  (graph_distance <= 3 and closing)):
                level = max(level, 1)
        if level == 2:
            label = "immediate collision risk"
        elif level == 1:
            label = "approaching ghost; a branch may be needed"
        else:
            label = "no measured closing ghost in the short horizon"
        nearest = min((r["distance"] for r in relations), default=999)
        closing_count = sum(1 for r in relations if r["closing"] and r["same_lane"])
        return {
            "level": level,
            "label": label,
            "nearest": nearest,
            "nearest_bucket": _bucket_distance(nearest),
            "closing_count": closing_count,
            "relations": relations,
        }

    def _candidate_risk(self, obs, node):
        if node is None:
            return "unknown"
        px, py, _ = self._player(obs)
        best = 999
        closing = False
        for ghost in self._active_ghosts(obs):
            gx, gy = ghost["x"], ghost["y"]
            candidate_x, candidate_y = node[0] - 3, node[1] - 4
            distance = abs(candidate_x - gx) + abs(candidate_y - gy)
            best = min(best, distance)
            relation = self._ghost_relation(obs, ghost)
            if relation["closing"] and relation["same_lane"]:
                closing = True
        if best <= 18 or (best <= 32 and closing):
            return "danger"
        if best <= 52:
            return "caution"
        return "open"

    def _escape_action(self, obs, base, hazard):
        node = self._player_node(obs)
        options = self._ordered_neighbors(node, self.current_action) if node is not None else []
        if not options:
            return OPPOSITE.get(base, base)

        # During an actual hazard, first try a graph path to a currently
        # decoded power pill.  This is an escape, not an invitation to chase a
        # ghost, and the normal route is rebuilt after the branch is taken.
        if self.power_points and hazard["level"] >= 1:
            power_nodes = [self._nearest_node(point) for point in self.power_points]
            power_nodes = [p for p in power_nodes if p is not None]
            best_path = None
            for action, neighbor in options:
                path = self._path_to_goals(neighbor, power_nodes, action)
                if path and (best_path is None or len(path) < len(best_path[1])):
                    best_path = (action, path)
            if best_path is not None:
                first = best_path[0]
                if self._candidate_risk(obs, dict(options)[first]) != "danger":
                    self.route = []
                    return first

        scored = []
        for action, neighbor in options:
            minimum = 99
            for ghost in self._active_ghosts(obs):
                distance = self._graph_distance(neighbor, ghost.get("node"), cap=12)
                minimum = min(minimum, distance)
            risk = self._candidate_risk(obs, neighbor)
            risk_penalty = {"danger": -100, "caution": -4, "open": 0, "unknown": -1}.get(risk, -1)
            # Keep a continuous route if risks are comparable, but never
            # choose a measured danger merely to preserve the dot target.
            continuity = 2 if action == base else 0
            scored.append((minimum * 10 + risk_penalty + continuity, action))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][1] if scored else base

    def _fallback_action(self, obs):
        base = self._route_action(obs)
        hazard = self._hazard_info(obs)
        if hazard["level"] >= 2:
            return self._escape_action(obs, base, hazard)
        if hazard["level"] == 1 and self.risk_mode == "survival":
            return self._escape_action(obs, base, hazard)
        return base

    # ---------- Astra: slow policy author, never on the control path ----------

    def _astra_summary(self, obs, reason):
        ghosts = []
        for ghost in self._active_ghosts(obs):
            ghosts.append({
                "xy": [ghost["x"], ghost["y"]],
                "velocity": [ghost["dx"], ghost["dy"]],
                "colour": ghost["colour"],
            })
        x, y, player = self._player(obs)
        return {
            "reason": reason,
            "frame": _integer(obs.get("frame")),
            "score": _finite_number(obs.get("score")),
            "lives": _integer(obs.get("lives")),
            "deaths": _integer(obs.get("deaths")),
            "mazes_cleared": _integer(obs.get("mazes_cleared")),
            "player": {"xy": [x, y], "direction": str(player.get("direction", "unknown"))},
            "ghosts": ghosts,
            "visible_pills": len(self.remaining),
            "graph_nodes": len(self.nodes),
            "graph_edges": sum(len(v) for v in self.adj.values()) // 2,
            "power_pill_positions": [[x0, y0] for x0, y0 in sorted(self.power_points)],
            "current_policy": self.risk_mode,
        }

    def _start_astra(self, obs, reason):
        if self.astra_future is not None or self.astra_calls >= 2:
            return
        try:
            budget = self.ctx.budget.remaining()
            if _integer(budget.get("system2_calls"), 0) <= 0:
                return
        except Exception:
            return
        summary = self._astra_summary(obs, reason)
        prompt = (
            "You are the slow strategic controller for Atari 2600 Ms. Pac-Man CX2675. "
            "The world keeps moving while you answer; do not give per-tick joystick "
            "commands and do not assume an arcade maze. Code has verified the current "
            "dot-coordinate corridor graph and a fast guard. Choose only a maze-level "
            "risk policy. Preserve lives and keep dot clearing primary; a power pill is "
            "an escape before it is a ghost-hunting opportunity. Side tunnels and the "
            "observed graph are safer evidence than imported arcade route tables. "
            "Return one explicit line MODE=survival, MODE=balanced, or MODE=score, "
            "then at most two short lines explaining when to take a power pill and when "
            "to abandon a route. The fast controller will ignore any malformed text.\n\n"
            + json.dumps(summary, separators=(",", ":"))
        )
        self.astra_calls += 1
        try:
            self.astra_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, timeout_s=90.0)
            )
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError):
            self.astra_future = None

    def _poll_astra(self):
        future = self.astra_future
        if future is None or not future.done():
            return
        self.astra_future = None
        try:
            result = future.result()
        except (BudgetExceeded, EpisodeOver):
            return
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError):
            return
        text = _safe_text(getattr(result, "text", result))
        lowered = text.lower()
        mode = None
        # Only parse an explicit assignment, avoiding accidental changes from
        # an Astra explanation that mentions all three modes.
        marker = lowered.find("mode")
        if marker >= 0:
            tail = lowered[marker : marker + 40]
            for candidate in ("survival", "balanced", "score"):
                if candidate in tail:
                    mode = candidate
                    break
        if mode in {"survival", "balanced", "score"} and mode != self.risk_mode:
            self.risk_mode = mode
            self.policy_version += 1
            try:
                self.ctx.log("astra_policy", mode=mode, policy_version=self.policy_version)
            except Exception:
                pass

    # ---------- Jev: local recognition at a branch ----------

    def _legal_options(self, obs, base):
        node = self._player_node(obs)
        options = [action for action, _ in self._ordered_neighbors(node, self.current_action)]
        if base in CARDINALS and base not in options:
            options.append(base)
        if not options:
            options = [base] if base in CARDINALS else ["left"]
        result = []
        for action in options:
            action = self._valid(action)
            if action in CARDINALS and action not in result:
                result.append(action)
        return result

    def _compact_jev_state(self, obs, base, options, hazard):
        node = self._player_node(obs)
        x, y, player = self._player(obs)
        option_facts = {}
        for action in options:
            neighbor = None
            for candidate_action, candidate_node in self._ordered_neighbors(node, self.current_action):
                if candidate_action == action:
                    neighbor = candidate_node
                    break
            option_facts[action] = {
                "continues_code_route": action == base,
                "corridor_risk": self._candidate_risk(obs, neighbor),
                "is_power_escape": False,
                "branch_degree": len(self.adj.get(neighbor, ())) if neighbor else 0,
            }
        escape_action = None
        if hazard["level"] >= 1 and self.power_points and node is not None:
            power_nodes = [self._nearest_node(point) for point in self.power_points]
            for action, neighbor in self._ordered_neighbors(node, self.current_action):
                if self._path_to_goals(neighbor, power_nodes, action):
                    escape_action = action
                    if action in option_facts:
                        option_facts[action]["is_power_escape"] = True
                    break
        return {
            "observation": {
                "frame": _integer(obs.get("frame")),
                "tick": _integer(obs.get("ticks")),
                "score": _finite_number(obs.get("score")),
                "lives": _integer(obs.get("lives")),
                "deaths": _integer(obs.get("deaths")),
                "player_xy": [x, y],
                "player_direction": str(player.get("direction", "unknown")),
                "visible_pills": len(self.remaining),
                "maze_clears": _integer(obs.get("mazes_cleared")),
            },
            "plan": {
                "risk_mode": self.risk_mode if _integer(obs.get("lives"), 3) > 1 else "survival",
                "fallback_action": base,
                "legal_actions": options,
                "at_graph_node": list(node) if node else [0, 0],
                "graph_degree": len(self.adj.get(node, ())) if node else 0,
                "policy_version": self.policy_version,
            },
            "threats": {
                "summary": hazard["label"],
                "nearest_distance_bucket": hazard["nearest_bucket"],
                "closing_same_lane_count": hazard["closing_count"],
                "immediate": hazard["level"] >= 2,
            },
            "options": option_facts,
            "escape_option": escape_action or "none",
        }

    def _choice_question(self, options, reversed_order=False):
        ordered = list(options)
        if reversed_order:
            ordered.reverse()
        criteria = {}
        for action in ordered:
            criteria[action] = (
                "Use the legal cardinal joystick direction that corresponds to this "
                "already-computed corridor option. Do not invent a route or a diagonal."
            )
        return Choice(
            "Choose exactly one value from `plan.legal_actions`. Prefer "
            "`plan.fallback_action` unless `threats.immediate` or the option facts "
            "show that it enters a danger corridor; do not calculate numeric distances.",
            criteria,
        )

    def _launch_jev(self, obs, base, options, hazard, node):
        if getattr(self.ctx, "system1", None) is None:
            return
        if self.jev_future is not None:
            return
        if self.ctx.done:
            return
        state = self._compact_jev_state(obs, base, options, hazard)
        # Rotating the input options and asking the same choice in reverse order
        # counters Jev's documented first-option bias without a second request.
        rotation = _integer(obs.get("ticks"), 0) % max(1, len(options))
        rotated = options[rotation:] + options[:rotation]
        questions = {
            "move_a": self._choice_question(rotated, reversed_order=False),
            "move_b": self._choice_question(rotated, reversed_order=True),
            "escape": Noul(
                "Given `threats` and `escape_option`, is taking the already-described "
                "power-pill escape safer now than continuing the fallback route?",
                {
                    "true": "The immediate or approaching threat makes the power escape the safer local subgoal.",
                    "false": "The fallback route remains safer; do not divert merely to seek a pill.",
                },
            ),
        }
        request_frame = _integer(obs.get("frame"), 0)
        request_deaths = _integer(obs.get("deaths"), 0)
        request_key = (node, base, hazard["level"], request_deaths, self.maze_key)
        self.last_query_key = request_key
        self.jev_request = {
            "frame": request_frame,
            "deaths": request_deaths,
            "node": node,
            "options": set(options),
            "escape_option": state.get("escape_option", "none"),
            "hazard": hazard["level"],
            "key": request_key,
        }
        try:
            self.jev_future = self.ctx.spawn(
                lambda: self.ctx.decide(state, questions, timeout_s=1.2)
            )
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
            self.jev_request = None
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError):
            self.jev_future = None
            self.jev_request = None

    @staticmethod
    def _answer_value(answer):
        if isinstance(answer, dict):
            return answer.get("value", answer.get("choice"))
        return getattr(answer, "value", None)

    @staticmethod
    def _answer_probability(answer, value):
        probabilities = answer.get("probabilities", {}) if isinstance(answer, dict) else getattr(answer, "probabilities", {})
        if not isinstance(probabilities, dict):
            return 0.0
        return _finite_number(probabilities.get(value), 0.0)

    def _poll_jev(self, obs):
        future = self.jev_future
        if future is None or not future.done():
            return
        self.jev_future = None
        request = self.jev_request
        self.jev_request = None
        try:
            answers = future.result()
        except (BudgetExceeded, EpisodeOver):
            return
        except ModelUnavailable:
            raise
        except (TimeoutError, RuntimeError):
            return
        if not isinstance(answers, dict) or not request or self.ctx.done:
            return

        now_frame = _integer(obs.get("frame"), request["frame"])
        now_deaths = _integer(obs.get("deaths"), request["deaths"])
        if now_deaths != request["deaths"] or now_frame - request["frame"] > 24:
            return
        current_node = self._player_node(obs)
        if current_node != request["node"] and request["hazard"] < 2:
            return
        legal_now = set(self._legal_options(obs, self._route_action(obs)))
        a = answers.get("move_a")
        b = answers.get("move_b")
        va = self._answer_value(a)
        vb = self._answer_value(b)
        if (not isinstance(va, str) or not isinstance(vb, str) or
                va != vb or va not in request["options"] or va not in legal_now):
            return
        choice_action = va
        pa = self._answer_probability(a, choice_action)
        pb = self._answer_probability(b, choice_action)
        threshold = 0.50 if request["hazard"] >= 1 else 0.34
        # Missing probabilities are treated as uncertainty, not confidence.
        if pa < threshold or pb < threshold:
            return
        # The Noul can select only the already-computed escape option, never an
        # arbitrary new direction. Its own probability is an additional gate.
        escape_answer = answers.get("escape")
        escape_value = self._answer_value(escape_answer)
        escape_probability = self._answer_probability(escape_answer, "true")
        escape_yes = (escape_value is True or escape_value == "true" or
                      (_finite_number(escape_value, 0.0) >= 0.50))
        selected = choice_action
        escape_option = request.get("escape_option")
        if (escape_yes and escape_probability >= 0.50 and
                escape_option in request["options"] and escape_option in legal_now):
            selected = escape_option
        if selected not in legal_now:
            return
        va = selected
        self.override = {
            "action": self._valid(va),
            "until": now_frame + 16,
            "node": current_node,
            "hazard": request["hazard"],
        }
        try:
            self.ctx.log(
                "jev_move",
                action=self.override["action"],
                frame=now_frame,
                hazard=request["hazard"],
                probability=min(pa, pb),
            )
        except Exception:
            pass

    def _maybe_query_jev(self, obs, base, hazard):
        if getattr(self.ctx, "system1", None) is None or self.jev_future is not None:
            return
        if self.ctx.done:
            return
        node = self._player_node(obs)
        degree = len(self.adj.get(node, ())) if node is not None else 0
        near_junction = degree >= 3 and self._player_node_distance(obs, node) <= 12
        turn_coming = base != self.current_action and node is not None
        should_query = near_junction or turn_coming or hazard["level"] >= 1 or len(self.nodes) < 4
        if not should_query:
            return
        key = (node, base, hazard["level"], _integer(obs.get("deaths")), self.maze_key)
        if key == self.last_query_key:
            return
        options = self._legal_options(obs, base)
        if len(options) <= 1:
            return
        self._launch_jev(obs, base, options, hazard, node)

    def _choose_action(self, obs, fallback):
        override = self.override
        if override is not None:
            frame = _integer(obs.get("frame"), 0)
            current_node = self._player_node(obs)
            hazard = self._hazard_info(obs)
            if frame <= override["until"] and override["action"] in self._legal_options(obs, fallback):
                if current_node == override["node"] or hazard["level"] >= 2:
                    return self._valid(override["action"])
            self.override = None
        return self._valid(fallback)

    # ---------- episode loop ----------

    def run_episode(self, game):
        if not getattr(game, "tick_s", None):
            return
        self._prepare_actions(game)
        try:
            obs = game.observe()
        except Exception:
            return
        self._update_observation(obs)
        self._start_astra(obs, "opening calibration and maze-level policy")

        while not self.ctx.done and not getattr(game, "done", False):
            try:
                obs = game.observe()
            except Exception:
                break
            self._update_observation(obs)
            self._poll_astra()
            self._poll_jev(obs)

            # A last life is not the time for a score-oriented Astra policy.
            if _integer(obs.get("lives"), 3) <= 1:
                self.risk_mode = "survival"
            fallback = self._fallback_action(obs)
            hazard = self._hazard_info(obs)
            self._maybe_query_jev(obs, fallback, hazard)
            action = self._choose_action(obs, fallback)
            self.current_action = action
            try:
                self.ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            self.previous_player = self._player(obs)[:2]
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx):
    return Harness(ctx)
