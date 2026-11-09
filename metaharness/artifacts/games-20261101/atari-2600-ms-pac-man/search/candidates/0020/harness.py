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
from collections import deque, Counter
import heapq
import re

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


class GraphExecutor:
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
        # Power pills are explicit, persistent checkpoint goals.  The ordinary
        # dot graph remains the corridor representation, but a pill's physical
        # coordinate is associated with its nearest observed corridor node.
        self.power_targets = {}
        self.visible_power_keys = set()
        self.checkpoint_target = None
        self.checkpoint_node = None
        self.goal_mode = "checkpoint"
        self.pill_active_until = -1
        self.power_eaten = 0
        self.life_phase = 0
        # Visit three checkpoints on the first life and retain one as a real
        # later-life/emergency resource.  A new life gets first claim on the
        # remaining resource rather than replaying the entire first route.
        self.checkpoint_count = 0
        self.checkpoint_quota = 3
        self.power_route_target = None

        self.route = []
        self.route_target = None
        self.route_kind = "dots"
        self.current_action = "left"
        self.last_node = None
        self.previous_player = None
        self.progress_previous_player = None
        self.previous_deaths = 0
        self.previous_clears = 0
        self.last_score = 0.0

        # Online evidence from 0007, but kept separate from the transient
        # recovery state so a bad observation cannot create a left/right loop.
        self.bad_edges = {}
        self.bad_actions = {}
        self.stalled_ticks = 0
        self.stall_node = None
        self.stall_tried = set()
        self.recovery_action = None
        self.recovery_node = None
        self.recovery_until = -1

        # The observation at which the last action was posted is the useful
        # pre-death context.  The sprite in the next observation may already be
        # in a death animation or respawn position.
        self.last_control_node = None
        self.last_control_next = None
        self.last_control_action = None
        self.last_control_deaths = 0

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
        self.power_targets = {}
        self.visible_power_keys = set()
        self.checkpoint_target = None
        self.checkpoint_node = None
        self.goal_mode = "checkpoint"
        self.pill_active_until = -1
        self.power_eaten = 0
        self.life_phase = 0
        self.checkpoint_count = 0
        self.checkpoint_quota = 3
        self.power_route_target = None
        self.route = []
        self.route_target = None
        self.route_kind = "dots"
        self.last_node = None
        self.progress_previous_player = None
        self.started_moving = False
        self.stalled_ticks = 0
        self.stall_node = None
        self.stall_tried = set()
        self.recovery_action = None
        self.recovery_node = None
        self.recovery_until = -1
        self.bad_edges = {}
        self.bad_actions = {}
        self.last_control_node = None
        self.last_control_next = None
        self.last_control_action = None
        self.last_control_deaths = self.previous_deaths
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

    def _power_grid_node(self, point):
        """Put a power-pill sprite on the observed pill lattice.

        The REM exposes the sprite's upper-left coordinate.  A corner power
        pill occupies the lattice cell where an ordinary dot would have been,
        but that ordinary dot is absent from ``pills``.  Snap only y to the
        already observed row lattice and retain the physical x coordinate.
        """
        x, y = point
        rows = sorted({node[1] for node in self.nodes})
        if not rows:
            return (x, y)
        row = min(rows, key=lambda value: (abs(value - y), value))
        return (x, row)

    def _register_power_targets(self):
        # Add synthetic checkpoint cells before rebuilding links.  They remain
        # in ``nodes`` after the object disappears, so a later route can still
        # use the learned corridor and a life reset cannot erase the geometry.
        changed = False
        for point in self.power_points:
            node = self._power_grid_node(point)
            if node not in self.nodes:
                self.nodes.add(node)
                changed = True
            old = self.power_targets.get(point)
            if old is None:
                self.power_targets[point] = {"point": point, "node": node, "consumed": False}
            else:
                old["node"] = node
        if changed:
            self._rebuild_graph()
            self.policy_version += 1

    def _checkpoint_candidates(self, obs):
        start = self._player_node(obs)
        candidates = []
        for point, entry in self.power_targets.items():
            if entry.get("consumed") or point not in self.visible_power_keys:
                continue
            node = entry.get("node")
            if node not in self.adj:
                continue
            distance = self._graph_distance(start, node, cap=999)
            candidates.append((distance, point, node))
        candidates.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        return candidates

    def _select_checkpoint(self, obs, *, force=False):
        if self.checkpoint_count >= self.checkpoint_quota:
            self.checkpoint_target = None
            self.checkpoint_node = None
            self.goal_mode = "dots"
            return
        if self.checkpoint_target is not None and not force:
            return
        candidates = self._checkpoint_candidates(obs)
        if not candidates:
            self.checkpoint_target = None
            self.checkpoint_node = None
            self.goal_mode = "dots"
            return
        # Rotate by life, but the per-life quota preserves at least one pill
        # after the first sweep.  Distances are code-computed; no seed-specific
        # target ordering is used.
        offset = self.life_phase % len(candidates)
        ordered = candidates[offset:] + candidates[:offset]
        _, point, node = ordered[0]
        self.checkpoint_target = point
        self.checkpoint_node = node
        self.goal_mode = "checkpoint"
        self.route = []
        self.route_target = node
        self.route_kind = "checkpoint"
        self.policy_version += 1

    def _detect_power_consumption(self, obs, previous_power, current_power):
        disappeared = sorted(previous_power - current_power)
        if not disappeared:
            return
        px, py, _ = self._player(obs)
        score_gain = _finite_number(obs.get("score"), self.last_score) - self.last_score
        # Score delta is the strongest confirmation; position proximity also
        # handles an adaptor that lags its reward/object update by a tick.
        consumed = []
        for point in disappeared:
            near = abs(px - point[0]) + abs(py - point[1]) <= 28
            if near or score_gain >= 45:
                consumed.append(point)
                if score_gain >= 45:
                    score_gain -= 50
        for point in consumed:
            entry = self.power_targets.get(point)
            if entry is not None:
                entry["consumed"] = True
            self.power_eaten += 1
            self.checkpoint_count += 1
            if point == self.checkpoint_target:
                self.checkpoint_target = None
                self.checkpoint_node = None
            self.goal_mode = "dots"
            self.route = []
            self.route_target = None
            self.route_kind = "dots"
            self.pill_active_until = _integer(obs.get("frame"), 0) + 180
            self.policy_version += 1
            try:
                self.ctx.log(
                    "power_checkpoint",
                    point=list(point),
                    power_eaten=self.power_eaten,
                    life_checkpoint_count=self.checkpoint_count,
                    quota=self.checkpoint_quota,
                )
            except Exception:
                pass

    def _update_observation(self, obs):
        if not isinstance(obs, dict):
            return
        key = self._maze_key(obs)
        visible_points = _dedupe_points(obs.get("pills", []), y_limit=190)
        old_maze = self.maze_key
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
        self._register_power_targets()

        deaths = _integer(obs.get("deaths"), self.previous_deaths)
        clears = _integer(obs.get("mazes_cleared"), self.previous_clears)
        maze_changed = old_maze is not None and key != old_maze
        death_changed = deaths != self.previous_deaths
        previous_power = set(self.visible_power_keys)
        current_power = set(self.power_points)
        if death_changed or clears != self.previous_clears or maze_changed:
            self.route = []
            self.route_target = None
            self.override = None
            self.last_query_key = None
            if death_changed:
                self.life_phase = deaths
                self.checkpoint_target = None
                self.checkpoint_node = None
                self.goal_mode = "checkpoint"
                self.pill_active_until = -1
        self._detect_power_consumption(obs, previous_power, current_power)
        self.visible_power_keys = current_power
        self.previous_deaths = deaths
        self.previous_clears = clears
        self.last_score = _finite_number(obs.get("score"), self.last_score)
        if not getattr(obs, "game_over", False) and not obs.get("game_over", False):
            self._select_checkpoint(obs)

    def _edge_bad(self, a, b):
        if a is None or b is None:
            return False
        return self.bad_edges.get((a, b), 0) > 0 or self.bad_actions.get(
            (a, _edge_action(a, b)), 0
        ) > 0

    def _edge_blocked(self, a, b):
        if self._edge_bad(a, b):
            return True
        return (
            a == self.stall_node and
            self.recovery_action in CARDINALS and
            _edge_action(a, b) == self.recovery_action
        )

    def _remember_control_edge(self, obs):
        """Penalize the edge actually being attempted before a death.

        Death/respawn observations are spatially ambiguous, so use the saved
        pre-tick node/next node, not the post-death player coordinate.
        """
        a = self.last_control_node
        b = self.last_control_next
        action = self.last_control_action
        if a is None or action not in CARDINALS:
            return
        if b is not None:
            key = (a, b)
            self.bad_edges[key] = min(3, self.bad_edges.get(key, 0) + 1)
        else:
            key = (a, action)
            self.bad_actions[key] = min(3, self.bad_actions.get(key, 0) + 1)
        self.route = []
        self.route_target = None
        self.override = None
        self.recovery_action = None
        self.recovery_node = None
        self.recovery_until = -1
        try:
            self.ctx.log(
                "predeath_edge",
                node=list(a),
                next_node=list(b) if b is not None else None,
                action=action,
                penalty=(self.bad_edges.get((a, b), 0) if b is not None
                         else self.bad_actions.get((a, action), 0)),
            )
            self.ctx.log("edge_taboo", node=list(a), action=action)
        except Exception:
            pass

    def _update_progress(self, obs, *, life_changed=False, maze_changed=False):
        x, y, _ = self._player(obs)
        current = (x, y)
        moved = self.progress_previous_player is not None and current != self.progress_previous_player
        if _finite_number(obs.get("score"), 0.0) > 0 or moved:
            self.started_moving = True
        if life_changed or maze_changed:
            self.stalled_ticks = 0
            self.stall_node = None
            self.stall_tried = set()
            self.recovery_action = None
            self.recovery_node = None
            self.recovery_until = -1
        elif not self.started_moving:
            self.stalled_ticks = 0
        elif moved:
            self.stalled_ticks = 0
            self.stall_node = None
            self.stall_tried = set()
            self.recovery_action = None
            self.recovery_node = None
            self.recovery_until = -1
        elif self.current_action in CARDINALS:
            self.stalled_ticks += 1

        node = self._player_node(obs)
        if node is not None and self.stalled_ticks >= 2:
            if self.stall_node != node:
                self.stall_node = node
                self.stall_tried = set()
                self.recovery_action = None
                self.recovery_node = node
                self.recovery_until = -1
            # Choose one new branch once, then commit to it for 6 ticks.  The
            # next branch is considered only after that hold has elapsed.
            if self.recovery_action is None or _integer(obs.get("frame"), 0) > self.recovery_until:
                options = self._ordered_neighbors(node, self.current_action, avoid=True)
                choices = [a for a, _ in options if a not in self.stall_tried]
                if not choices:
                    options = self._ordered_neighbors(node, self.current_action)
                    choices = [a for a, _ in options if a not in self.stall_tried]
                if choices:
                    action = choices[0]
                    self.stall_tried.add(action)
                    self.recovery_action = action
                    self.recovery_node = node
                    self.recovery_until = _integer(obs.get("frame"), 0) + 24
                    self.route = []
                    self.route_target = None
                    self.override = None
                    try:
                        self.ctx.log("progress_recovery", node=list(node), action=action,
                                     hold_until=self.recovery_until)
                    except Exception:
                        pass
        self.progress_previous_player = current

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

        # A checkpoint is a hard route goal until its object disappears.  This
        # is the structural difference from 0004: ordinary-dot BFS cannot
        # silently consume the whole first life without visiting an escape.
        if self.goal_mode == "checkpoint" and self.checkpoint_node is not None:
            target = self.checkpoint_node
            if self.route and self.route[-1] != target:
                self.route = []
            if self.route:
                try:
                    index = self.route.index(start)
                except ValueError:
                    self.route = []
                else:
                    self.route = self.route[index:]
                    if len(self.route) <= 1 and start != target:
                        self.route = []
            if not self.route and start != target:
                self.route = self._path_to_goals(start, [target], self.current_action)
                self.route_target = target if self.route else None
            if self.route:
                return start
            # If the synthetic checkpoint is not connected in the observed
            # graph, degrade to dots rather than holding or issuing a wall turn.
            self.goal_mode = "dots"
            self.checkpoint_target = None
            self.checkpoint_node = None

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
                    entry = self.power_targets.get(point)
                    node = entry.get("node") if entry else self._nearest_node(point)
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

        # During an actual hazard, first try the active checkpoint.  If the
        # checkpoint has already been consumed, use any remaining power pill.
        # This keeps an emergency turn aligned with the explicit goal route.
        power_nodes = []
        if self.checkpoint_node is not None and self.goal_mode == "checkpoint":
            power_nodes = [self.checkpoint_node]
        elif self.power_points and hazard["level"] >= 1:
            for point in self.power_points:
                entry = self.power_targets.get(point)
                candidate = entry.get("node") if entry else self._nearest_node(point)
                if candidate is not None:
                    power_nodes.append(candidate)
        if power_nodes and hazard["level"] >= 1:
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
                "continues_goal_route": action == base,
                "corridor_risk": self._candidate_risk(obs, neighbor),
                "reaches_checkpoint": bool(self.checkpoint_node is not None and neighbor == self.checkpoint_node),
                "branch_degree": len(self.adj.get(neighbor, ())) if neighbor else 0,
            }
        escape_action = None
        power_nodes = []
        if self.checkpoint_node is not None and self.goal_mode == "checkpoint":
            power_nodes = [self.checkpoint_node]
        else:
            for point in self.power_points:
                entry = self.power_targets.get(point)
                candidate = entry.get("node") if entry else self._nearest_node(point)
                if candidate is not None:
                    power_nodes.append(candidate)
        if hazard["level"] >= 1 and power_nodes and node is not None:
            for action, neighbor in self._ordered_neighbors(node, self.current_action):
                if self._path_to_goals(neighbor, power_nodes, action):
                    escape_action = action
                    if action in option_facts:
                        option_facts[action]["reaches_checkpoint"] = True
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
                "visible_power_pills": len(self.visible_power_keys),
                "pill_active": _integer(obs.get("frame"), 0) < self.pill_active_until,
                "maze_clears": _integer(obs.get("mazes_cleared")),
            },
            "plan": {
                "risk_mode": self.risk_mode if _integer(obs.get("lives"), 3) > 1 else "survival",
                "goal_kind": self.goal_mode,
                "goal_xy": list(self.checkpoint_target) if self.checkpoint_target else [0, 0],
                "life_phase": self.life_phase,
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
            "show that it enters a danger corridor; when `plan.goal_kind` is "
            "checkpoint, do not abandon the named checkpoint for a speculative "
            "route. Do not calculate numeric distances.",
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
        request_key = (node, base, hazard["level"], request_deaths, self.maze_key,
                       self.checkpoint_target, self.policy_version)
        self.last_query_key = request_key
        self.jev_request = {
            "frame": request_frame,
            "deaths": request_deaths,
            "node": node,
            "options": set(options),
            "escape_option": state.get("escape_option", "none"),
            "hazard": hazard["level"],
            "policy_version": self.policy_version,
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
        if (now_deaths != request["deaths"] or
                now_frame - request["frame"] > 18 or
                request.get("policy_version") != self.policy_version):
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
        threshold = 0.55 if request["hazard"] >= 1 else 0.40
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
        if (escape_yes and escape_probability >= 0.60
                and escape_option in request["options"] and escape_option in legal_now):
            selected = escape_option
        if selected not in legal_now:
            return
        # A checkpoint is a code-owned subgoal.  Jev may arbitrate a genuine
        # immediate hazard, but a normal junction answer cannot silently turn
        # the life back into the parent's dot-only sweep.
        if (self.goal_mode == "checkpoint" and request["hazard"] < 2 and
                selected != self._route_action(obs)):
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
class Harness(GraphExecutor):
    """Route council: no model is allowed to post or override a joystick."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.epoch = 0
        self.last_frame = -1
        self.previous_count = None
        self.refill_votes = 0
        self.epoch_min_count = 9999
        self.last_lives = None
        self.last_position = None
        self.last_motion_frame = -999
        self.motion = 'left'
        self.ghost_tracks = {}
        self.ghost_fields = []
        self.visit_counts = Counter()
        self.distance_cache = {}
        self.edge_failures = Counter()
        self.hold_action = None
        self.hold_until = -1
        self.hold_origin = None
        self.last_guard_frame = -999
        self.last_plan_frame = -999
        self.last_goal_progress = 0
        self.goal = None
        self.goal_kind = 'dots'
        self.goal_until = -1
        self.goal_version = 0
        self.last_goal_log = None
        self.packets = {}
        self.jev_future = None
        self.jev_request = None
        self.last_jev_frame = -999
        self.jev_latencies = deque(maxlen=24)
        self.s1_failures = 0
        self.s1_applied = 0
        self.s1_rejected = 0
        self.astra_future = None
        self.astra_request = None
        self.astra_calls = 0
        self.last_audit_frame = -9999
        self.audit_reason = 'opening'
        self.rubric = 'balanced'
        self.rubric_version = 0
        self.last_obs_deaths = 0
        self.last_epoch_frame = -9999
        self.freeze_until = -1
        self.graph_serial = 0
        self.tick_summary = {}

        # A route is a short-lived corridor commitment, not a direction cache.
        # It survives small changes in projected ghost risk and is revalidated
        # against the current node every tick.  Emergency paths get a stronger
        # lock so the controller does not alternate adjacent one-edge branches.
        self.lease_path = []
        self.lease_target = None
        self.lease_kind = None
        self.lease_risk = 0
        self.lease_until = -1
        self.lease_epoch = -1
        self.lease_goal_version = -1
        self.lease_started = -1
        self.escape_lock = False
        self.escape_until = -1
        self.escape_target = None
        self.escape_reason = None
        self.last_commit = None
        self.commit_count = 0
        self.escape_count = 0

    def _log(self, event, **fields):
        self.ctx.log(event, **fields)

    def _rebuild_graph(self):
        super()._rebuild_graph()
        self.distance_cache = {}
        self.graph_serial = getattr(self, 'graph_serial', 0) + 1

    def _distances(self, start):
        if start is None or start not in self.adj:
            return {}
        if start in self.distance_cache:
            return self.distance_cache[start]
        distances = {start: 0.0}
        heap = [(0.0, start)]
        while heap:
            dist, node = heapq.heappop(heap)
            if dist != distances[node]:
                continue
            for nxt in sorted(self.adj.get(node, ())):
                length = abs(nxt[0]-node[0]) + abs(nxt[1]-node[1])
                if length > 80:
                    length = 20
                cost = dist + length
                if cost < distances.get(nxt, 1e9):
                    distances[nxt] = cost
                    heapq.heappush(heap, (cost, nxt))
        self.distance_cache[start] = distances
        return distances

    def _reset_epoch(self, obs, dots, reason):
        self.epoch += 1
        self._reset_map(self._maze_key(obs), dots)
        self.goal = None
        self.goal_kind = 'dots'
        self.goal_version += 1
        self.goal_until = -1
        self.hold_action = None
        self.hold_until = -1
        self.ghost_tracks = {}
        self.visit_counts.clear()
        self.edge_failures.clear()
        self.epoch_min_count = len(dots)
        self.refill_votes = 0
        self.last_epoch_frame = _integer(obs.get('frame'))
        self.audit_reason = 'maze_resource_renewal'
        self._clear_corridor_lease('epoch')
        self._log('maze_epoch', epoch=self.epoch, reason=reason,
                  frame=self.last_epoch_frame, dots=len(dots),
                  reported_clears=obs.get('mazes_cleared'))

    def _clear_corridor_lease(self, reason):
        """Drop a path commitment without changing the graph or goal."""
        if self.lease_path or self.escape_lock:
            self._log('lease_clear', reason=reason, frame=self.last_frame,
                      escape=self.escape_lock)
        self.lease_path = []
        self.lease_target = None
        self.lease_kind = None
        self.lease_risk = 0
        self.lease_until = -1
        self.lease_epoch = -1
        self.lease_goal_version = -1
        self.lease_started = -1
        self.escape_lock = False
        self.escape_until = -1
        self.escape_target = None
        self.escape_reason = None

    def _update_observation(self, obs):
        if not isinstance(obs, dict):
            return
        frame = _integer(obs.get('frame'))
        if frame == self.last_frame:
            return
        dots = _dedupe_points(obs.get('pills', []), y_limit=190)
        # A sustained population jump is independent of the known adaptor bug
        # (zero -> full dot set does not increment its maze counter).
        refill = (self.previous_count is not None and self.epoch_min_count <= 8
                  and len(dots) >= 70 and frame-self.last_epoch_frame > 120)
        self.refill_votes = self.refill_votes + 1 if refill else 0
        if self.maze_key is None:
            self._reset_epoch(obs, dots, 'initial')
        elif self._maze_key(obs) != self.maze_key:
            self._reset_epoch(obs, dots, 'reported_layout_change')
        elif self.refill_votes >= 2:
            self._reset_epoch(obs, dots, 'sustained_dot_repopulation')
        deaths = _integer(obs.get('deaths'))
        life_changed = deaths != self.last_obs_deaths
        if life_changed:
            self.goal = None
            self.goal_version += 1
            self.hold_action = None
            self.hold_until = -1
            self.ghost_tracks = {}
            self.visit_counts.clear()
            self.audit_reason = 'life_loss'
            self.freeze_until = frame + 80
            # A respawn invalidates the old corridor's physical phase.  Keep
            # graph/ghost evidence, but never carry a pre-death escape turn
            # through the death animation.
            self._clear_corridor_lease('life_loss')
            self._log('life_reset', frame=frame, deaths=deaths, epoch=self.epoch)
        old_position = self.last_position
        old_frame = self.last_frame
        old_powers = set(self.visible_power_keys)
        old_score = self.last_score
        super()._update_observation(obs)
        position = self._player(obs)[:2]
        elapsed = max(1, frame-old_frame)
        if old_position is not None and not life_changed:
            dx, dy = position[0]-old_position[0], position[1]-old_position[1]
            if 0 < abs(dx)+abs(dy) <= 12 and elapsed <= 12:
                self.motion = ('right' if dx > 0 else 'left') if abs(dx) >= abs(dy) else ('down' if dy > 0 else 'up')
                self.last_motion_frame = frame
                self.freeze_until = -1
        # Confirm all pill events, including incidental pills not equal to the
        # selected checkpoint. RGB is never used as a frightened-state flag.
        gain = _finite_number(obs.get('score'))-old_score
        disappeared = old_powers-self.visible_power_keys
        for point in sorted(disappeared):
            if abs(position[0]-point[0])+abs(position[1]-point[1]) <= 24 and gain >= 40:
                self.pill_active_until = frame + 120
                self.goal = None
                self.goal_version += 1
                self.hold_action = None
                self.hold_until = -1
                self._log('power_window', frame=frame, epoch=self.epoch, point=list(point),
                          interpretation='recent consumption; not proven immunity')
        if self.last_score > old_score:
            self.last_goal_progress = frame
        node = self._player_node(obs)
        if node is not None and node != self.last_node:
            self.visit_counts[node] += 1
        self.last_node = node
        self.previous_count = len(dots)
        self.epoch_min_count = min(self.epoch_min_count, len(dots))
        self.last_obs_deaths = deaths
        self.last_lives = _integer(obs.get('lives'), 3)
        self.last_position = position
        self.last_frame = frame
        self._update_ghost_fields(obs)

    def _update_ghost_fields(self, obs):
        fields = []
        frame = _integer(obs.get('frame'))
        for index, ghost in enumerate(obs.get('ghosts', []) or []):
            if not _visible(ghost, False):
                continue
            x, y = _xy(ghost)
            if x == 0 and y == 0:
                continue
            slot = _integer(ghost.get('slot'), index)
            previous = self.ghost_tracks.get(slot)
            vx, vy = 0.0, 0.0
            if previous and 0 < frame-previous[2] <= 20:
                dt = frame-previous[2]
                dx, dy = x-previous[0], y-previous[1]
                if abs(dx)+abs(dy) <= dt*2:
                    vx, vy = dx/dt, dy/dt
            else:
                dx, dy = _integer(ghost.get('dx')), _integer(ghost.get('dy'))
                if abs(dx)+abs(dy) <= 8:
                    vx, vy = dx/4.0, dy/4.0
            self.ghost_tracks[slot] = (x,y,frame)
            node = self._object_node((x,y))
            centre = (x+3,y+4)
            residual = abs(centre[0]-node[0])+abs(centre[1]-node[1]) if node else 999
            # Ghosts in the central chamber are deliberately not projected
            # onto a distant pill corridor as if they were already there.
            aligned = residual <= 8
            speed = max(0.50, min(1.10, abs(vx)+abs(vy)))
            fields.append({'slot':slot,'xy':centre,'vx':vx,'vy':vy,'node':node,
                           'aligned':aligned,'distances':self._distances(node) if aligned else {},
                           'speed':speed})
        self.ghost_fields = fields

    def _risk_profile(self, obs, path):
        """Evaluate interception along the next 42 pixels, in native frames.

        Threats are approximate; never claim a path is verified safe. Predicted
        approach and graph reachability are separate from instantaneous proximity.
        """
        if not path:
            return {'risk':3,'margin':-99.0,'escape':False,'uncertain':True,'cost':999.0}
        px,py,_ = self._player(obs)
        centre = (px+3,py+4)
        minimum = 99.0
        risk = 0
        cost = 0.0
        travelled = 0.0
        previous = centre
        recent_power = self.last_frame < self.pill_active_until
        for i,node in enumerate(path[:8]):
            if i == 0:
                continue
            length = abs(node[0]-previous[0])+abs(node[1]-previous[1])
            if length > 80:
                length = 20
            travelled += length
            arrival = travelled / 0.65
            local = 99.0
            for ghost in self.ghost_fields:
                gx,gy = ghost['xy']
                direct = abs(node[0]-gx)+abs(node[1]-gy)
                graph_dist = ghost['distances'].get(node, 999.0)
                same_lane = abs(node[0]-gx) <= 5 or abs(node[1]-gy) <= 5
                projection = abs(node[0]-(gx+ghost['vx']*min(arrival,16)))+abs(node[1]-(gy+ghost['vy']*min(arrival,16)))
                # Conservative graph arrival within a bounded horizon; beyond
                # that horizon a ghost could turn and its policy is unknown.
                if ghost['aligned'] and graph_dist <= 58 and arrival <= 65:
                    toward = (node[0]-gx)*ghost['vx'] + (node[1]-gy)*ghost['vy']
                    retreat_credit = 10.0 if toward < -1 else 0.0
                    margin = (graph_dist-9)/ghost['speed'] - arrival + retreat_credit
                    local = min(local, margin)
                if same_lane and direct < 24 and projection < direct+4:
                    local = min(local, (direct-8)/ghost['speed']-min(arrival,16))
                if i == 1 and direct <= 10:
                    local = min(local,-18.0)
            minimum = min(minimum,local)
            if local < -10:
                risk = max(risk,3)
            elif local < 4:
                risk = max(risk,2)
            elif local < 18:
                risk = max(risk,1)
            cost += max(0,18-local)*0.55
            previous = node
            if travelled >= 42:
                break
        # A recent pill is uncertain protection, not permission to chase.
        # It may break ties but cannot remove the immediate collision veto.
        if recent_power:
            cost *= 0.65
        endpoint = path[min(len(path)-1,4)]
        escape = len(self.adj.get(endpoint,())) >= 3
        return {'risk':risk,'margin':round(minimum,2),'escape':escape,
                'uncertain':True,'cost':cost}

    def _weighted_path(self, obs, start, goals, first_action=None):
        goals = set(goals)-{start}
        if start not in self.adj or not goals:
            return []
        distances={start:0.0}; parents={start:None}; heap=[(0.0,start)]
        found=None
        while heap:
            cost,node=heapq.heappop(heap)
            if cost != distances[node]:
                continue
            if node in goals:
                found=node; break
            for action,nxt in self._ordered_neighbors(node,self.current_action):
                if node==start and first_action and action!=first_action:
                    continue
                length=abs(node[0]-nxt[0])+abs(node[1]-nxt[1])
                if length>80: length=20
                threat=0.0
                for g in self.ghost_fields:
                    d=g['distances'].get(nxt,999)
                    if d<40: threat=max(threat,(40-d)*0.8)
                visit=min(8,self.visit_counts[nxt])*0.3
                penalty=24*min(3,self.edge_failures[(node,action)])
                new=cost+length+threat+visit+penalty
                if new<distances.get(nxt,1e9):
                    distances[nxt]=new; parents[nxt]=node
                    heapq.heappush(heap,(new,nxt))
        if found is None: return []
        path=[]
        while found is not None:
            path.append(found); found=parents[found]
        return path[::-1]

    def _candidate_packets(self, obs):
        start=self._player_node(obs)
        if start is None: return {}
        packets={}
        base=self._route_action(obs)
        if len(self.route)>=2:
            packets['continue']={'path':list(self.route),'kind':self.goal_mode,
                                 'target':self.route[-1],'action':base}
        dots=set(self.remaining)-{start}
        if dots:
            path=self._weighted_path(obs,start,dots)
            if len(path)>=2:
                packets['clear_dots']={'path':path,'kind':'dots','target':path[-1],
                                       'action':_edge_action(path[0],path[1])}
        # Pills are neither hoarded until death nor indiscriminately hunted.
        # They are executable choices, ranked by interception, not distance alone.
        power_nodes=[e['node'] for p,e in self.power_targets.items()
                     if p in self.visible_power_keys and not e.get('consumed')]
        path=self._weighted_path(obs,start,power_nodes)
        if len(path)>=2:
            packets['power_escape']={'path':path,'kind':'escape','target':path[-1],
                                    'action':_edge_action(path[0],path[1])}
        # An alternative branch goes at least to a junction, not a one-tick
        # direction that can be immediately countermanded by an old model answer.
        for action,nxt in self._ordered_neighbors(start,self.current_action):
            path=[start,nxt]; prev=start; current=nxt
            for _ in range(6):
                forward=[n for n in sorted(self.adj.get(current,())) if n!=prev]
                if len(forward)!=1: break
                neighbor=forward[0]
                path.append(neighbor); prev,current=current,neighbor
            key='branch_'+action
            packets[key]={'path':path,'kind':'retreat','target':path[-1],'action':action}
        for key,packet in packets.items():
            packet['profile']=self._risk_profile(obs,packet['path'])
            packet['dots']=sum(n in self.remaining for n in packet['path'][1:])
            packet['reversal']=packet['action']==OPPOSITE.get(self.motion)
            packet['first_valid']=packet['action'] in self._executable_actions(obs)
        return packets

    def _executable_actions(self, obs):
        node=self._player_node(obs)
        options=[a for a,_ in self._ordered_neighbors(node,self.current_action)]
        x,y,_=self._player(obs)
        cx,cy=x+3,y+4
        result=[]
        for a in options:
            # Perpendicular turns require physical alignment. A nearby graph
            # node alone does not make its side corridor executable right now.
            perpendicular=(DXDY[a][0]==0)!=(DXDY[self.motion][0]==0)
            aligned=(abs(cx-node[0])<=4 if DXDY[a][0]==0 else abs(cy-node[1])<=4)
            if not perpendicular or aligned:
                result.append(a)
        return result

    def _packet_rank(self, key, p):
        profile=p['profile']
        utility=0.0
        if key=='continue': utility+=10
        if p['kind']=='checkpoint': utility+=9
        if p['kind']=='dots': utility+=4
        if p['kind']=='escape': utility+=5
        if p['kind']=='retreat': utility-=6
        utility+=min(6,p['dots'])*2
        if profile['escape']: utility+=3
        if p['reversal']: utility-=4
        utility-=min(8,self.visit_counts[p['target']])*1.5
        # Risk is lexicographically prior to reward. Inside the same class,
        # prefer progress and continuity rather than perpetual maximum distance.
        return (profile['risk'], -utility, profile['cost'],key)

    def _install_goal(self, obs, packet, source):
        target=packet['target']
        if self.goal != target or self.goal_kind != packet['kind']:
            self.goal_version+=1
        self.goal=target
        self.goal_kind=packet['kind']
        self.goal_until=self.last_frame+120
        self.route=list(packet['path'])
        self.route_target=target
        self.route_kind=packet['kind']
        # A route-goal lease supersedes an old checkpoint only when arbitration
        # actually chose a different resource/branch.
        if packet['kind'] not in ('checkpoint','escape'):
            self.goal_mode='dots'
            self.checkpoint_target=None
            self.checkpoint_node=None
        else:
            matches=[p for p,e in self.power_targets.items() if e['node']==target
                     and p in self.visible_power_keys]
            if matches:
                self.checkpoint_target=matches[0]
                self.checkpoint_node=target
                self.goal_mode=packet['kind']
        mark=(target,packet['kind'],source)
        if mark != self.last_goal_log:
            self._log('route_lease',source=source,kind=packet['kind'],target=list(target),
                      action=packet['action'],risk=packet['profile']['risk'],
                      margin=packet['profile']['margin'],epoch=self.epoch,
                      goal_version=self.goal_version,frame=self.last_frame)
            self.last_goal_log=mark

    def _lease_path_now(self, obs):
        """Return the current executable suffix of the committed corridor."""
        if not self.lease_path or self.lease_epoch != self.epoch:
            return None
        frame = _integer(obs.get('frame'))
        if frame > self.lease_until:
            return None
        start = self._player_node(obs)
        if start is None:
            return None
        target = self.lease_target
        if target is None:
            return None
        kind = self.lease_kind or 'dots'
        if kind == 'dots' and target not in self.remaining and start != target:
            return None
        if kind in ('checkpoint', 'escape') and start != target:
            live_power = any(
                entry.get('node') == target and point in self.visible_power_keys
                for point, entry in self.power_targets.items()
            )
            if not live_power:
                return None
        path = list(self.lease_path)
        if start in path:
            path = path[path.index(start):]
        elif start != target:
            # A tunnel or a skipped observation can move the nearest-node
            # association off the saved list.  Rejoin the same goal instead of
            # inventing a new one-edge retreat.
            path = self._weighted_path(obs, start, [target])
        if len(path) < 2:
            return None
        profile = self._risk_profile(obs, path)
        return {
            'path': path,
            'target': target,
            'kind': kind,
            'action': _edge_action(path[0], path[1]),
            'profile': profile,
            'first_valid': True,
        }

    def _commit_corridor(self, obs, packet, *, emergency=False, reason='replan'):
        """Install a path lease; route changes happen only at this boundary."""
        path = list(packet.get('path') or [])
        if len(path) < 2:
            return False
        frame = _integer(obs.get('frame'))
        target = packet.get('target') or path[-1]
        kind = str(packet.get('kind') or 'dots')
        profile = packet.get('profile') or self._risk_profile(obs, path)
        packet = dict(packet)
        packet['path'] = path
        packet['target'] = target
        packet['kind'] = kind
        packet['profile'] = profile
        # Keep the base executor's goal and route bookkeeping synchronized,
        # while the lease below is the authority for action selection.
        self._install_goal(obs, packet, 'escape_lock' if emergency else reason)
        self.lease_path = path
        self.lease_target = target
        self.lease_kind = kind
        self.lease_risk = _integer(profile.get('risk'), 3)
        self.lease_until = frame + (72 if emergency else 96)
        self.lease_epoch = self.epoch
        self.lease_goal_version = self.goal_version
        self.lease_started = frame
        self.escape_lock = bool(emergency)
        self.escape_until = frame + 72 if emergency else -1
        self.escape_target = target if emergency else None
        self.escape_reason = reason if emergency else None
        self.commit_count += 1
        self.last_commit = (target, kind, self.lease_risk, emergency)
        self._log(
            'corridor_commit',
            frame=frame,
            target=list(target),
            kind=kind,
            risk=self.lease_risk,
            margin=profile.get('margin'),
            emergency=bool(emergency),
            reason=reason,
            lease_until=self.lease_until,
            epoch=self.epoch,
        )
        if emergency:
            self.escape_count += 1
            self._log(
                'escape_commit',
                frame=frame,
                target=list(target),
                action=packet.get('action'),
                risk=self.lease_risk,
                margin=profile.get('margin'),
                reason=reason,
                hold_until=self.escape_until,
            )
        return True

    def _best_emergency_packet(self, packets):
        """Choose a whole escape corridor, not the locally nearest branch."""
        choices = [
            (key, packet) for key, packet in packets.items()
            if packet.get('first_valid') and len(packet.get('path') or []) >= 2
        ]
        if not choices:
            return None

        def rank(item):
            key, packet = item
            profile = packet.get('profile') or {}
            risk = _integer(profile.get('risk'), 3)
            # A reachable power pill is the preferred tie-breaker, then an
            # actual branch with an exit.  Margin is more useful than distance
            # when every candidate is already under pressure.
            if key == 'power_escape' or packet.get('kind') in ('escape', 'checkpoint'):
                resource = 0
            elif profile.get('escape'):
                resource = 1
            elif key == 'continue':
                resource = 2
            else:
                resource = 3
            return (
                risk,
                -_finite_number(profile.get('margin'), -999.0),
                resource,
                -_integer(packet.get('dots'), 0),
                _edge_action(packet['path'][0], packet['path'][1]),
            )

        return min(choices, key=rank)[1]

    def _best_normal_packet(self, packets, base):
        choices = [
            (key, packet) for key, packet in packets.items()
            if packet.get('first_valid') and len(packet.get('path') or []) >= 2
        ]
        if not choices:
            return None
        current = packets.get('continue')
        if current and current.get('first_valid') and len(current.get('path') or []) >= 2:
            # The observed route is the default.  A fresh route wins only when
            # it is materially safer; this prevents a harmless score tie from
            # rebuilding the path at every junction.
            best = min(choices, key=lambda item: self._packet_rank(*item))
            if (_integer(best[1].get('profile', {}).get('risk'), 3)
                    < _integer(current.get('profile', {}).get('risk'), 3)):
                return best[1]
            return current
        return min(choices, key=lambda item: self._packet_rank(*item))[1]

    def _lease_hold_action(self, obs, active, *, reason='hold'):
        path = active.get('path') or []
        action = active.get('action')
        if len(path) < 2 or action not in CARDINALS:
            return None
        frame = _integer(obs.get('frame'))
        # Trace a sparse hold event rather than filling the trace with one event
        # per 67 ms tick.  The action itself is still posted every tick.
        if frame == self.lease_started or frame % 48 == 0:
            self._log(
                'lease_hold', frame=frame, target=list(active['target']),
                kind=active['kind'], risk=active['profile'].get('risk'),
                margin=active['profile'].get('margin'), emergency=self.escape_lock,
                reason=reason,
            )
        self.tick_summary = {
            'risk': active['profile'].get('risk', 3),
            'fallback_action': action,
            'packet': 'active_escape' if self.escape_lock else 'active_lease',
            'lease_target': list(active['target']),
        }
        return self._valid(action)

    def _code_action(self, obs):
        """Run a path-level controller with hysteresis around ghost risk."""
        base = self._route_action(obs)
        packets = self._candidate_packets(obs)
        self.packets = packets
        frame = _integer(obs.get('frame'))
        hazard = self._hazard_info(obs)
        active = self._lease_path_now(obs)

        # An escape lease is intentionally sticky.  It may be superseded only
        # when its path is invalid or its fresh risk is strictly worse than a
        # new candidate, never because a one-edge packet happened to rank first.
        if self.escape_lock:
            if active is None or frame > self.escape_until:
                self._clear_corridor_lease('escape_expired')
                active = None
            else:
                if (hazard['level'] >= 2 and
                        _integer(active['profile'].get('risk'), 3) >= 3):
                    replacement = self._best_emergency_packet(packets)
                    if replacement is not None:
                        old_risk = _integer(active['profile'].get('risk'), 3)
                        new_risk = _integer(replacement.get('profile', {}).get('risk'), 3)
                        old_margin = _finite_number(active['profile'].get('margin'), -999)
                        new_margin = _finite_number(replacement.get('profile', {}).get('margin'), -999)
                        if new_risk < old_risk or new_margin > old_margin + 8:
                            self._commit_corridor(obs, replacement,
                                                  emergency=True,
                                                  reason='escape_invalidated')
                            active = self._lease_path_now(obs)
                # Once the player has held the escape for a turn window and is
                # at an open junction, allow normal planning to resume.  Before
                # that minimum hold, even a clear-looking frame cannot undo the
                # escape turn.
                node = self._player_node(obs)
                safe_junction = (
                    frame >= self.lease_started + 16 and
                    hazard['level'] == 0 and node is not None and
                    len(self.adj.get(node, ())) >= 3 and
                    _integer(active['profile'].get('risk'), 3) <= 1
                ) if active is not None else False
                if not safe_junction and active is not None:
                    action = self._lease_hold_action(obs, active, reason='escape_lock')
                    if action is not None:
                        return action
                if safe_junction:
                    self.escape_lock = False
                    self.escape_until = -1
                    self.escape_target = None
                    self.escape_reason = None
                    self._log('escape_release', frame=frame, reason='safe_junction')
                    active = self._lease_path_now(obs)

        # A fresh immediate/interception hazard is the only normal reason to
        # break a lease before its expiry.  The selected replacement is then
        # locked, so this branch cannot oscillate on successive ticks.
        threatened = hazard['level'] >= 2
        if active is not None:
            active_risk = _integer(active['profile'].get('risk'), 3)
            if hazard['level'] >= 1 and active_risk >= 2:
                threatened = True
            if active_risk >= 3:
                threatened = True
        if threatened and not self.escape_lock:
            emergency = self._best_emergency_packet(packets)
            if emergency is not None:
                self._commit_corridor(
                    obs, emergency, emergency=True,
                    reason='projected_ghost' if hazard['level'] else 'route_risk'
                )
                active = self._lease_path_now(obs)
                if active is not None:
                    action = self._lease_hold_action(obs, active, reason='new_escape')
                    if action is not None:
                        return action
            # If no graph packet is executable, retain the old guard's final
            # cardinal escape rather than issuing noop or waiting on Jev.
            return self._valid(self._escape_action(obs, base, hazard))

        # Continue an ordinary lease through small risk changes.  Only expiry,
        # arrival, invalidation, or a materially safer route opens a new lease.
        if active is not None:
            frame_expired = frame > self.lease_until
            at_target = self._player_node(obs) == self.lease_target
            if not frame_expired and not at_target:
                action = self._lease_hold_action(obs, active)
                if action is not None:
                    return action
            self._clear_corridor_lease('ordinary_boundary')

        packet = self._best_normal_packet(packets, base)
        if packet is not None:
            if self._commit_corridor(obs, packet, emergency=False, reason='boundary'):
                active = self._lease_path_now(obs)
                if active is not None:
                    action = self._lease_hold_action(obs, active, reason='new_corridor')
                    if action is not None:
                        return action

        self.tick_summary = {'risk': 0, 'fallback_action': base,
                             'packet': 'raw_route'}
        return self._valid(base)

    @staticmethod
    def _answer_value(answer):
        return answer.get('value') if isinstance(answer,dict) else getattr(answer,'value',None)

    @staticmethod
    def _answer_probability(answer, value):
        probs=answer.get('probabilities',{}) if isinstance(answer,dict) else getattr(answer,'probabilities',{})
        return _finite_number(probs.get(value),0.0) if isinstance(probs,dict) else 0.0

    def _jev_state(self, obs, short):
        labels=['clear','watch','interception_possible','interception_likely']
        descriptions={}
        for key,p in short.items():
            descriptions[key]={'purpose':p['kind'], 'risk':labels[p['profile']['risk']],
                               'exit_branch':p['profile']['escape'],
                               'collects_dots':'several' if p['dots']>=3 else ('some' if p['dots'] else 'none'),
                               'reverses_motion':p['reversal'],
                               'resource':'power_pill' if p['kind'] in ('escape','checkpoint') else 'none',
                               'continues_plan':key=='continue',
                               'geometry':'inferred dot corridors; code revalidates movement'}
        return {'rules':{'objective':'score through repeated dot clearing; avoid collisions',
                         'power':'recent pill is NOT proven immunity; do not chase ghosts',
                         'choices':'route goals, never raw joystick overrides',
                         'risk_order':['clear','watch','interception_possible','interception_likely']},
                'plan':{'rubric':self.rubric,'epoch':self.epoch,
                        'stage':'cleanup' if len(self.remaining)<=20 else 'sweep',
                        'last_life':self.last_lives==1,'goal_kind':self.goal_kind},
                'options':descriptions,
                'recent':{'score_progress':self.last_frame-self.last_goal_progress<120,
                          'power_event_recent':self.last_frame<self.pill_active_until,
                          'guard_intervened':self.last_frame-self.last_guard_frame<60}}

    def _maybe_query_jev(self, obs):
        if self.jev_future is not None or self.ctx.done:
            return
        # Jev can select the next route at a boundary, but an active escape
        # lease is an execution commitment, not a question for arbitration.
        if self.escape_lock:
            return
        if getattr(self.ctx,'system1',True) is None:
            return
        if self.last_frame-self.last_jev_frame<24 or self.last_frame<272:
            return
        if self.ctx.budget.remaining().get('system1_calls',0)<2:
            return
        usable=[(k,p) for k,p in self.packets.items() if p['first_valid']]
        if len(usable)<2: return
        usable.sort(key=lambda pair:self._packet_rank(*pair))
        best_risk=usable[0][1]['profile']['risk']
        # Arithmetic and comparisons belong to code. Jev sees only a shortlist
        # of non-dominated, currently executable semantic choices.
        usable=[(k,p) for k,p in usable if p['profile']['risk']<=best_risk][:4]
        unique=[]; seen=set()
        for k,p in usable:
            sig=(p['target'],p['action'])
            if sig not in seen: unique.append((k,p)); seen.add(sig)
        if len(unique)<2: return
        short=dict(unique)
        state=self._jev_state(obs,short)
        instruction=(
            'Choose a route from options or unknown. Code already removed worse risk classes. '
            'Prefer continued dot progress over empty retreat unless the current plan is trapped. '
            'A power route is useful under measured pressure, not simply because it exists. '
            'When plan.stage is cleanup, finish scattered dots rather than circling. '
            'Use plan.rubric: balanced prefers continuity; exits prefers an exit branch under '
            'pressure; cleanup prefers dot-collecting options. Do not calculate distances, '
            'invent geometry, or assume frightened immunity.')
        criteria={k:dict(v) for k,v in state['options'].items()}
        criteria['unknown']='The supplied evidence does not distinguish safe progress.'
        questions={
            'route_a':Choice(instruction,criteria),
            'route_b':Choice(instruction,dict(reversed(list(criteria.items())))),
            'renew':Noul('Does recent show a stalled score and repeated guard intervention that warrants replacing the current route?',
                         {'true':'No score progress and guard intervention support replanning.',
                          'false':'Score progresses or there is no evidence of a route trap.'})}
        self.last_jev_frame=self.last_frame
        self.jev_request={'frame':self.last_frame,'epoch':self.epoch,
                          'deaths':self.last_obs_deaths,'rubric':self.rubric_version,
                          'goals':{k:(p['target'],p['kind']) for k,p in short.items()},
                          'clock':self.ctx.clock.now()}
        self.jev_future=self.ctx.spawn(lambda:self.ctx.decide(state,questions,timeout_s=2.5))

    def _poll_jev(self, obs):
        if self.jev_future is None or not self.jev_future.done(): return
        future=self.jev_future; request=self.jev_request
        self.jev_future=None; self.jev_request=None
        try: answers=future.result()
        except ModelUnavailable: raise
        except (TimeoutError,RuntimeError,TypeError,ValueError):
            self.s1_failures+=1; return
        if not isinstance(answers,dict) or not request: return
        self.jev_latencies.append(max(0,self.ctx.clock.now()-request['clock']))
        if (request['epoch']!=self.epoch or request['deaths']!=self.last_obs_deaths
            or request['rubric']!=self.rubric_version or self.last_frame-request['frame']>48):
            self.s1_rejected+=1; return
        if self.escape_lock:
            self.s1_rejected += 1
            self._log('jev_held_by_escape', frame=self.last_frame,
                      request_frame=request['frame'])
            return
        a,b=answers.get('route_a'),answers.get('route_b')
        value=self._answer_value(a)
        if not isinstance(value,str) or value!=self._answer_value(b) or value not in request['goals']:
            self.s1_rejected+=1; return
        prob=min(self._answer_probability(a,value),self._answer_probability(b,value))
        if prob<0.70: self.s1_rejected+=1; return
        target,kind=request['goals'][value]
        start=self._player_node(obs)
        if kind=='dots' and target not in self.remaining: return
        if kind in ('escape','checkpoint') and not any(e['node']==target and p in self.visible_power_keys for p,e in self.power_targets.items()): return
        path=self._weighted_path(obs,start,[target])
        if len(path)<2: return
        action=_edge_action(path[0],path[1]); profile=self._risk_profile(obs,path)
        now=self._candidate_packets(obs)
        eligible=[p for p in now.values() if p['first_valid']]
        floor=min((p['profile']['risk'] for p in eligible),default=3)
        if action not in self._executable_actions(obs) or profile['risk']>floor:
            self.s1_rejected+=1; return
        # A model route is advisory until the path lease reaches a boundary.
        # Do not let a delayed answer recreate the one-edge churn that this
        # candidate is testing against.
        active = self._lease_path_now(obs)
        if (active is not None and self.last_frame <= self.lease_until
                and tuple(target) != tuple(active.get('target') or ())):
            self.s1_rejected += 1
            self._log('jev_deferred_by_lease', frame=self.last_frame,
                      target=list(target), active_target=list(active['target']))
            return
        renew=self._answer_value(answers.get('renew'))
        if _finite_number(renew,0)>=0.8 and self.last_frame-self.last_goal_progress>=120:
            self.audit_reason='jev_route_trap'
        # Keep routine checkpoints unless a better threat class justifies change.
        current=now.get('continue')
        if self.goal_mode=='checkpoint' and current and profile['risk']>=current['profile']['risk'] and kind!='checkpoint': return
        packet={'target':target,'kind':kind,'path':path,'action':action,'profile':profile}
        self._install_goal(obs,packet,'jev_council')
        self.s1_applied+=1
        self._log('jev_route_applied',frame=self.last_frame,route=value,probability=prob,
                  age_frames=self.last_frame-request['frame'],goal=list(target))

    def _maybe_audit(self, obs):
        if self.astra_future is not None or self.ctx.done or self.astra_calls>=3: return
        if self.audit_reason is None: return
        # Start with measured evidence, not a frame-zero empty ghost snapshot.
        if self.last_frame<600 or self.last_frame-self.last_audit_frame<1200: return
        budget=self.ctx.budget.remaining()
        if budget.get('system2_calls',0)<1 or budget.get('wall_s',0)<90: return
        evidence={'reason':self.audit_reason,'epoch':self.epoch,'lives':self.last_lives,
                  'remaining_dots':len(self.remaining),'score':self.last_score,
                  'power_resources':len(self.visible_power_keys),'rubric':self.rubric,
                  'jev_applied':self.s1_applied,'jev_rejected':self.s1_rejected,
                  'jev_failed':self.s1_failures,'guard_recent':self.tick_summary,
                  'frames_without_score':self.last_frame-self.last_goal_progress}
        prompt=(
            'Audit the decision INTERFACE for this real-time Atari 2600 Ms. Pac-Man controller. '
            'Code constructs an inferred pill-corridor graph, computes interception risk and '
            'short route candidates, rejects dominated risks, and posts each 67ms action. '
            'Jev asynchronously chooses a route GOAL, never a joystick. It receives semantic '
            'risk, escape branch, dot yield, current plan and recent progress. Two reordered '
            'Choices must agree at selected probability >=.70; stale or newly dangerous '
            'goals are discarded. A Noul probability >=.80 flags stalled progress plus '
            'guard interventions. Jev has no inherited context and should not do arithmetic. '
            'You select the rubric included in each subsequent Jev request, not immediate '
            'movement. Options: balanced=continuity and progress; exits=prefer branch exits '
            'under pressure; cleanup=prefer remaining dots over empty retreats. Geometry '
            'is inferred, normal RGB is not frightened evidence, and preserving pills '
            'until death failed prior tests. No arcade routes. Return EXACTLY one line '
            'RUBRIC=balanced or RUBRIC=exits or RUBRIC=cleanup. No explanation. '
            'Latest episode evidence: '+json.dumps(evidence,separators=(',',':')))
        self.astra_calls+=1
        self.last_audit_frame=self.last_frame
        self.astra_request={'epoch':self.epoch,'reason':self.audit_reason}
        self.audit_reason=None
        self.astra_future=self.ctx.spawn(lambda:self.ctx.think(prompt,timeout_s=100.0))

    def _poll_astra(self):
        if self.astra_future is None or not self.astra_future.done(): return
        future=self.astra_future; request=self.astra_request
        self.astra_future=None; self.astra_request=None
        try: result=future.result()
        except ModelUnavailable: raise
        except (TimeoutError,RuntimeError,TypeError,ValueError): return
        text=getattr(result,'text','')
        if not isinstance(text,str) or not request: return
        match=re.fullmatch(r'\s*RUBRIC=(balanced|exits|cleanup)\s*',text)
        if not match or request['epoch']!=self.epoch: return
        self.rubric=match.group(1); self.rubric_version+=1
        self._log('astra_interface',rubric=self.rubric,version=self.rubric_version,
                  reason=request['reason'],epoch=self.epoch,frame=self.last_frame)

    def run_episode(self, game):
        if not getattr(game,'tick_s',None): return
        self._prepare_actions(game)
        # First post precedes map construction and every inference request.
        self.ctx.act(self._valid('left'))
        while not self.ctx.done and not getattr(game,'done',False):
            obs=self.ctx.observe()
            if not isinstance(obs,dict):
                self.ctx.act(self._valid(self.current_action))
                self.ctx.wait_for_tick(); continue
            if obs.get('game_over'): return
            self._update_observation(obs)
            self._poll_astra()
            self._poll_jev(obs)
            action=self._code_action(obs)
            self.current_action=self._valid(action)
            self.ctx.act(self.current_action)
            self._maybe_query_jev(obs)
            self._maybe_audit(obs)
            self.ctx.wait_for_tick()


def build(ctx):
    return Harness(ctx)
