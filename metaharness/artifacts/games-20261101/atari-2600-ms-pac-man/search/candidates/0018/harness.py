"""0018: persistent collection missions with temporary, rejoining evasions.

Derived facts, corridor geometry and real-time safety are code-owned. Jev
selects durable collection missions, never a joystick or an empty retreat goal.
Astra revises the mission rubric after observed failure evidence. Inferred
geometry and uncertain power protection are never presented as verified facts.
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


class Navigation(GraphExecutor):
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
        self._log('maze_epoch', epoch=self.epoch, reason=reason,
                  frame=self.last_epoch_frame, dots=len(dots),
                  reported_clears=obs.get('mazes_cleared'))


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


    @staticmethod
    def _answer_value(answer):
        return answer.get('value') if isinstance(answer,dict) else getattr(answer,'value',None)


    @staticmethod
    def _answer_probability(answer, value):
        probs=answer.get('probabilities',{}) if isinstance(answer,dict) else getattr(answer,'probabilities',{})
        return _finite_number(probs.get(value),0.0) if isinstance(probs,dict) else 0.0



class Harness(Navigation):
    """Two controllers, two lifetimes: mission until consumption; maneuver to exit."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.mission = None
        self.mission_serial = 0
        self.mission_start = 0
        self.mission_progress = 0
        self.mission_best_distance = 1e9
        self.mission_history = deque(maxlen=16)
        self.maneuver = []
        self.maneuver_until = -1
        self.maneuver_kind = 'direct'
        self.maneuver_start = 0
        self.finish_started = None
        self.last_mission_query = -999
        self.query_signature = None
        self.mission_options = {}
        self.review_reason = 'opening_evidence'
        self.interface = 'completion'
        self.interface_version = 0
        self.s1_switches = 0
        self.guard_changes = 0
        self.retreat_history = deque(maxlen=40)
        self.last_choice = None
        self.mission_failures = Counter()
        self.frame_started = -1
        self.player_still_since = None
        self.ghost_motion = False
        self.recovery_until_frame = -1
        self.recovery_direction = None
        self.recovery_attempts = Counter()
        self.last_telemetry_frame = -999
        self.last_mission_epoch = 0
        self.last_mission_deaths = 0
        self.last_goal_observation = None

    def _update_observation(self, obs):
        old_frame = self.last_frame
        old_position = self.last_position
        old_deaths = self.last_obs_deaths
        old_epoch = self.epoch
        old_score = self.last_score
        super()._update_observation(obs)
        if self.last_frame == old_frame:
            return
        if self.epoch != old_epoch or self.last_obs_deaths != old_deaths:
            self.mission = None
            self.mission_serial += 1
            self.maneuver = []
            self.finish_started = None
            self.mission_best_distance = 1e9
            self.player_still_since = None
            self.recovery_until_frame = -1
            self.recovery_direction = None
            self.recovery_attempts.clear()
            self.query_signature = None
            self.review_reason = 'resource_epoch' if self.epoch != old_epoch else 'life_loss'
            if self.epoch != old_epoch:
                self.mission_failures.clear()
                self.retreat_history.clear()
        if self.last_score > old_score:
            self.mission_progress = self.last_frame
        if old_position == self.last_position and old_position is not None:
            if self.player_still_since is None:
                self.player_still_since = self.last_frame
        else:
            self.player_still_since = None
            self.recovery_direction = None
            self.recovery_until_frame = -1
        self.ghost_motion = any(abs(g['vx']) + abs(g['vy']) > 0.1 for g in self.ghost_fields)
        if self.mission and not self._target_exists(self.mission):
            self._log('mission_complete', frame=self.last_frame, epoch=self.epoch,
                      target=list(self.mission['target']), kind=self.mission['kind'],
                      duration_frames=self.last_frame-self.mission_start,
                      reason='collectible_disappeared', score=self.last_score)
            self.mission_history.append({'kind':self.mission['kind'], 'outcome':'consumed',
                                         'frames':self.last_frame-self.mission_start})
            self.mission = None
            self.mission_serial += 1
            self.maneuver = []
            self.finish_started = None
            self.mission_best_distance = 1e9
        self.last_mission_epoch = self.epoch
        self.last_mission_deaths = self.last_obs_deaths

    def _target_exists(self, mission):
        if not isinstance(mission, dict):
            return False
        target = mission.get('target')
        if mission.get('kind') == 'power':
            return any(p in self.visible_power_keys and e['node'] == target
                       for p,e in self.power_targets.items())
        return target in self.remaining

    def _length(self, a, b):
        distance = abs(a[0]-b[0])+abs(a[1]-b[1])
        return 20.0 if distance > 80 else float(distance)

    def _path_length(self, path):
        return sum(self._length(a,b) for a,b in zip(path,path[1:]))

    def _search(self, start, goals, banned=(), preferred=None):
        """Risk-weighted graph search; inferred edges remain provisional."""
        goals = set(goals)
        if start in goals:
            return [start]
        if start not in self.adj or not goals:
            return []
        blocked = set(banned)-{start}
        distances = {start:0.0}
        parents = {start:None}
        heap = [(0.0,start)]
        end = None
        while heap:
            cost,node = heapq.heappop(heap)
            if cost != distances[node]:
                continue
            if node in goals:
                end = node
                break
            for action,nxt in self._ordered_neighbors(node,preferred or self.motion):
                if nxt in blocked:
                    continue
                danger = 0.0
                for ghost in self.ghost_fields:
                    d = ghost['distances'].get(nxt,999.0)
                    danger = max(danger,max(0.0,36.0-d)*0.65)
                cost2 = cost+self._length(node,nxt)+danger
                cost2 += min(6,self.visit_counts[nxt])*0.25
                cost2 += min(3,self.edge_failures[(node,action)])*20
                if node == start and action == OPPOSITE.get(self.motion):
                    cost2 += 1.5
                if cost2 < distances.get(nxt,1e12):
                    distances[nxt] = cost2
                    parents[nxt] = node
                    heapq.heappush(heap,(cost2,nxt))
        if end is None:
            return []
        path=[]
        while end is not None:
            path.append(end)
            end=parents[end]
        return list(reversed(path))

    def _mission_packet(self, obs, target, kind, label):
        start=self._player_node(obs)
        path=self._search(start,[target])
        if not path:
            return None
        profile=self._risk_profile(obs,path) if len(path)>1 else {
            'risk':0,'margin':99.0,'escape':False,'uncertain':True,'cost':0.0}
        return {'target':target,'kind':kind,'label':label,'path':path,
                'profile':profile,'length':self._path_length(path),
                'dots':sum(n in self.remaining for n in path),
                'same':bool(self.mission and target==self.mission['target']),
                'failures':self.mission_failures[target]}

    def _collection_options(self, obs):
        """Persistent collectible goals, never one-dot empty escape waypoints."""
        options={}
        if self.mission and self._target_exists(self.mission):
            p=self._mission_packet(obs,self.mission['target'],self.mission['kind'],'finish current mission')
            if p: options['keep']=p
        start=self._player_node(obs)
        # Preserve the tested opening coverage mechanism. No life-loss resource gate.
        if self.checkpoint_node is not None and self.checkpoint_target in self.visible_power_keys:
            p=self._mission_packet(obs,self.checkpoint_node,'power','coverage power pill')
            if p: options['coverage']=p
        dots=set(self.remaining)
        if dots:
            nearest=self._search(start,dots)
            if nearest:
                p=self._mission_packet(obs,nearest[-1],'dots','nearest reachable dot')
                if p: options['near_dots']=p
            # A sweep traverses a connected run of uneaten dots. Intermediate
            # dots do not prematurely complete this mission or turn it around.
            if nearest:
                path=list(nearest)
                previous=path[-2] if len(path)>1 else None
                tip=path[-1]
                for _ in range(5):
                    forward=[n for n in sorted(self.adj.get(tip,()))
                             if n in dots and n not in path and n!=previous]
                    if len(forward)!=1:
                        break
                    previous,tip=tip,forward[0]
                    path.append(tip)
                if len(path)>len(nearest):
                    p=self._mission_packet(obs,path[-1],'dots','clear a connected dot run')
                    if p: options['sweep']=p
        # A reachable pill remains available before a death. It is a candidate,
        # not a promise that a distant pill grants current immunity.
        power_nodes=[e['node'] for p,e in self.power_targets.items()
                     if p in self.visible_power_keys and not e.get('consumed')]
        path=self._search(start,power_nodes)
        if path:
            p=self._mission_packet(obs,path[-1],'power','reachable power resource')
            if p: options['power']=p
        # De-duplicate while preferring explicit current mission and checkpoint.
        seen=set(); result={}
        for k,p in options.items():
            sig=(p['target'],p['kind'])
            if sig not in seen:
                result[k]=p
                seen.add(sig)
        return result

    def _start_mission(self, packet, source):
        if packet is None:
            return
        same=(self.mission and self.mission['target']==packet['target']
              and self.mission['kind']==packet['kind'])
        if same:
            return
        self.mission={'target':packet['target'],'kind':packet['kind']}
        self.mission_serial+=1
        self.mission_start=self.last_frame
        self.mission_progress=self.last_frame
        self.mission_best_distance=packet.get('length',1e9)
        self.maneuver=[]
        self.finish_started=None
        self._log('mission_started',frame=self.last_frame,epoch=self.epoch,
                  serial=self.mission_serial,source=source,kind=packet['kind'],
                  target=list(packet['target']),risk=packet['profile']['risk'],
                  dots=packet['dots'])

    def _default_mission(self, obs, options):
        if not options:
            return None
        if 'keep' in options:
            keep=options['keep']
            no_progress=self.last_frame-self.mission_progress
            if no_progress<600:
                return keep
            # This is a bounded failure timeout, not a label that invents a wall.
            alternatives=[p for k,p in options.items() if k!='keep']
            if not alternatives:
                return keep
            self.mission_failures[keep['target']]+=1
            self.review_reason='mission_no_progress'
            self._log('mission_timeout',frame=self.last_frame,target=list(keep['target']),
                      no_progress_frames=no_progress)
        coverage=options.get('coverage')
        if coverage and coverage['profile']['risk']<3:
            return coverage
        def rank(item):
            key,p=item
            risk=p['profile']['risk']
            # Coverage pills and actual dot runs break comparable-risk ties;
            # an empty refuge is deliberately not a strategic goal candidate.
            utility=min(6,p['dots'])*2.0
            if key=='coverage': utility+=8
            if p['kind']=='power' and risk<2: utility+=2
            return (risk,p['failures'],p['length']-utility,key)
        return min(options.items(),key=rank)[1]

    def _corridor_prefix(self, start, neighbor):
        path=[start,neighbor]
        previous=start; tip=neighbor
        for _ in range(8):
            onward=[n for n in sorted(self.adj.get(tip,())) if n!=previous and n not in path]
            if len(onward)!=1 or self._path_length(path)>=48:
                break
            previous,tip=tip,onward[0]
            path.append(tip)
        return path

    def _physical_finish(self, obs, target):
        """Nearest-node association is not proof of consuming a collectible."""
        px,py,_=self._player(obs)
        cx,cy=px+3,py+4
        dx,dy=target[0]-cx,target[1]-cy
        if abs(dx)>=abs(dy) and abs(dx)>1:
            return 'right' if dx>0 else 'left'
        if abs(dy)>1:
            return 'down' if dy>0 else 'up'
        if self.finish_started is None:
            self.finish_started=self.last_frame
        # Keep moving across the trigger rather than reverse at nearest-node
        # rounding. This short confirmation is interrupted by a measured threat.
        return self.motion if self.motion in CARDINALS else self.current_action

    def _maneuver_packets(self, obs):
        start=self._player_node(obs)
        if start is None or self.mission is None:
            return {}
        target=self.mission['target']
        direct=self._search(start,[target])
        if not direct:
            return {}
        routes={}
        if len(direct)>=2:
            routes['direct']={'path':direct,'action':_edge_action(direct[0],direct[1]),
                              'exit':direct[-1],'detour':False}
        elif target==start:
            action=self._physical_finish(obs,target)
            routes['finish']={'path':[start],'action':action,'exit':target,'detour':False}
        executable=set(self._executable_actions(obs))
        for action,nxt in self._ordered_neighbors(start,self.current_action):
            if action not in executable:
                continue
            prefix=self._corridor_prefix(start,nxt)
            # Rejoin the same mission after the next branch. An empty evasion
            # can change our route but cannot overwrite the collection objective.
            tail=self._search(prefix[-1],[target],banned=prefix[:-1])
            if not tail:
                tail=self._search(prefix[-1],[target])
            if not tail:
                continue
            path=prefix+tail[1:]
            if len(path)<2:
                continue
            routes['via_'+action]={'path':path,'action':action,'exit':prefix[-1],
                                  'detour':len(direct)>1 and action!=_edge_action(direct[0],direct[1])}
        if self.maneuver:
            try:
                index=self.maneuver.index(start)
                trimmed=self.maneuver[index:]
            except ValueError:
                trimmed=[]
            if len(trimmed)>=2 and self.last_frame<=self.maneuver_until:
                routes['held']={'path':trimmed,'action':_edge_action(trimmed[0],trimmed[1]),
                                'exit':trimmed[-1],'detour':self.maneuver_kind=='detour'}
        dist=self._distances(target)
        for k,p in routes.items():
            path=p['path']
            p['valid']=p['action'] in executable or k=='finish'
            p['profile']=self._risk_profile(obs,path) if len(path)>1 else {
                'risk':0,'margin':99.0,'escape':False,'uncertain':True,'cost':0.0}
            # The risk comparison includes the continuation after the refuge;
            # the old short-retreat vs long-dot horizon mismatch disappears.
            p['progress']=dist.get(start,999.0)-dist.get(p['exit'],999.0)
            p['dots']=sum(n in self.remaining for n in path[1:7])
            p['reversal']=p['action']==OPPOSITE.get(self.motion)
            p['loop']=sum(1 for f,n,a in self.retreat_history
                          if self.last_frame-f<=240 and n==start and a==p['action'])
            p['length']=self._path_length(path)
        return routes

    def _urgent_risk(self, obs, action):
        """Short current-motion collision check, separate from distant route risk."""
        px,py,_=self._player(obs)
        ux,uy=DXDY.get(action,(0,0))
        best=99.0
        for g in self.ghost_fields:
            gx,gy=g['xy'][0]-3,g['xy'][1]-4
            # Check actual lane contact only; a wall-separated ghost must not
            # veto a corridor simply because its Manhattan distance is small.
            aligned=abs(px-gx)<=5 if ux==0 else abs(py-gy)<=5
            if not aligned:
                continue
            forward=(gx-px)*ux+(gy-py)*uy
            if forward<0:
                continue
            approaching=-(g['vx']*ux+g['vy']*uy)
            closure=max(0.25,0.65+approaching)
            best=min(best,(forward-7.0)/closure)
        return best

    def _maneuver_rank(self, key, packet):
        profile=packet['profile']
        # Route risk remains the primary guard, but all routes lead to the same
        # mission. Longer empty detours no longer masquerade as new objectives.
        utility=0.0
        if key=='direct': utility+=8
        if key=='held': utility+=12
        utility+=min(6,packet['dots'])*1.5
        utility+=max(-10,min(10,packet['progress']*0.25))
        utility-=packet['loop']*5
        utility-=3 if packet['reversal'] else 0
        return (profile['risk'],-utility,packet['length']+profile['cost'],key)

    def _code_action(self, obs):
        options=self._collection_options(obs)
        self.mission_options=options
        desired=self._default_mission(obs,options)
        if desired is not None:
            self._start_mission(desired,'code')
        if self.mission is None:
            return self._route_action(obs)
        packets=self._maneuver_packets(obs)
        usable=[(k,p) for k,p in packets.items() if p['valid']]
        if not usable:
            return self._route_action(obs)
        usable.sort(key=lambda pair:self._maneuver_rank(*pair))
        key,selected=usable[0]
        held=packets.get('held')
        # A small risk-boundary fluctuation cannot reverse a chosen corridor.
        # A newly imminent contact or clearly worse risk can still interrupt it.
        if held and held['valid'] and self._urgent_risk(obs,held['action'])>10:
            if held['profile']['risk']<=selected['profile']['risk'] or (
                held['profile']['risk']==selected['profile']['risk']+1 and
                held['profile']['margin']>=selected['profile']['margin']-6):
                key,selected='held',held
        current_urgency=self._urgent_risk(obs,selected['action'])
        if current_urgency<=10:
            safer=[(k,p) for k,p in usable
                   if self._urgent_risk(obs,p['action'])>current_urgency+8]
            if safer:
                safer.sort(key=lambda pair:self._maneuver_rank(*pair))
                key,selected=safer[0]
        if key!='held' and len(selected['path'])>=2:
            # Store only the segment up to its next real branch, not an
            # irrevocable long journey through changing ghost positions.
            prefix=self._corridor_prefix(selected['path'][0],selected['path'][1])
            if self.mission['target'] in prefix:
                prefix=prefix[:prefix.index(self.mission['target'])+1]
            self.maneuver=prefix
            self.maneuver_until=self.last_frame+96
            self.maneuver_kind='detour' if selected['detour'] else 'direct'
            if selected['detour']:
                self.guard_changes+=1
                self.last_guard_frame=self.last_frame
                self.retreat_history.append((self.last_frame,self._player_node(obs),selected['action']))
            marker=(self.mission_serial,selected['action'],tuple(prefix),selected['detour'])
            if marker!=self.last_choice:
                self._log('maneuver',frame=self.last_frame,mission=self.mission_serial,
                          goal=list(self.mission['target']),action=selected['action'],
                          temporary_detour=selected['detour'],risk=selected['profile']['risk'],
                          rejoin_goal_unchanged=True)
                self.last_choice=marker
        start=self._player_node(obs)
        distance=self._distances(self.mission['target']).get(start,999)
        if distance<self.mission_best_distance-4:
            self.mission_best_distance=distance
            self.mission_progress=self.last_frame
        self.tick_summary={'risk':selected['profile']['risk'],'fallback_action':selected['action'],
                           'mission_serial':self.mission_serial,'detour':selected['detour']}
        if self.last_frame-self.last_telemetry_frame>=240:
            self.last_telemetry_frame=self.last_frame
            self._log('mission_status',frame=self.last_frame,epoch=self.epoch,
                      score=self.last_score,dots=len(self.remaining),
                      mission=self.mission_serial,goal=list(self.mission['target']),
                      no_progress_frames=self.last_frame-self.mission_progress,
                      guard_changes=self.guard_changes,jev_changes=self.s1_switches)
        return selected['action']

    def _state_for_jev(self, obs, options):
        labels=['clear','watch','interception_possible','interception_likely']
        data={}
        for key,p in options.items():
            data[key]={
                'purpose':p['label'], 'collectible':'power pill' if p['kind']=='power' else 'dots',
                'continues_mission':p['same'],'approach_risk':labels[p['profile']['risk']],
                'collects':'many' if p['dots']>=4 else ('some' if p['dots'] else 'none'),
                'travel':'short' if p['length']<=48 else ('medium' if p['length']<=112 else 'long'),
                'previously_unproductive':p['failures']>0,
            }
        return {'rules':{'game':'Atari 2600 Ms. Pac-Man',
                         'objective':'score through collecting dots and clearing successive mazes',
                         'decision':'Choose a collectible mission, NEVER a joystick or empty retreat.',
                         'execution':'Code navigates every tick and can temporarily evade without changing the mission.',
                         'uncertainty':'Dot graph is inferred. Ghost RGB is not a frightened-state signal.',
                         'power':'A reachable pill can be used before life loss. A distant pill is not present immunity.'},
                'policy':{'rubric':self.interface,
                          'completion':'Prefer current productive mission, then connected dot clearing.',
                          'mobility':'Prefer a short approach that is not repeatedly unproductive.',
                          'resources':'Prefer a reachable power route under measured pressure; no hoarding until death.'},
                'context':{'cleanup':len(self.remaining)<=20,'last_life':self.last_lives==1,
                           'mission_present':self.mission is not None,
                           'mission_stalled':self.last_frame-self.mission_progress>=240,
                           'score_stalled':self.last_frame-self.last_goal_progress>=180,
                           'evasion_repeating':sum(self.last_frame-f<=240 for f,_,_ in self.retreat_history)>=4,
                           'recent_power':self.last_frame<self.pill_active_until},
                'options':data}

    def _maybe_query_jev(self, obs):
        if self.ctx.done or self.jev_future is not None:
            return
        if getattr(self.ctx,'system1',True) is None:
            return
        if self.last_frame<272 or self.last_frame-self.last_mission_query<96:
            return
        if self.ctx.budget.remaining().get('system1_calls',0)<2:
            return
        options=self._collection_options(obs)
        if len(options)<2:
            return
        state=self._state_for_jev(obs,options)
        instructions=(
            'Choose a collectible mission from options, or unknown. Follow policy.rubric. '
            'Do not replace a productive current mission just because an empty retreat is safer; '
            'code already owns temporary evasions. Release a current mission only when context '
            'shows stalled progress or an alternative power approach is clearly safer. '
            'In cleanup prefer finishing remaining dots, not indefinitely preserving resources. '
            'Use the named risk/travel categories; do not calculate or invent geometry.')
        criteria={k:dict(v) for k,v in state['options'].items()}
        criteria['unknown']='The supplied evidence does not justify a mission change.'
        questions={
            'mission_a':Choice(instructions,criteria),
            'mission_b':Choice(instructions,dict(reversed(list(criteria.items())))),
            'release':Noul('Does context provide evidence to release an existing collection mission: stalled mission AND stalled score with repeated evasions?',
                           {'true':'All three conditions are present; the persistent objective may need replacement.',
                            'false':'The mission or score is progressing, or repeated evasions are not established.'}),
            'audit':Noul('Does context show repeated evasion without score progress, warranting an audit of the mission-selection rubric?',
                         {'true':'Repeated evasion and score stall both present.',
                          'false':'No joint evidence of repeated evasion and score stall.'})}
        self.last_mission_query=self.last_frame
        self.jev_request={'frame':self.last_frame,'epoch':self.epoch,'deaths':self.last_obs_deaths,
                          'mission':self.mission_serial,'interface':self.interface_version,
                          'options':{k:{'target':p['target'],'kind':p['kind'],'label':p['label']} for k,p in options.items()},
                          'clock':self.ctx.clock.now()}
        self.jev_future=self.ctx.spawn(lambda:self.ctx.decide(state,questions,timeout_s=2.5))

    def _poll_jev(self, obs):
        if self.jev_future is None or not self.jev_future.done():
            return
        future,request=self.jev_future,self.jev_request
        self.jev_future=None; self.jev_request=None
        try:
            answers=future.result()
        except ModelUnavailable:
            raise
        except (TimeoutError,RuntimeError,TypeError,ValueError):
            self.s1_failures+=1
            return
        if not isinstance(answers,dict) or not request:
            return
        self.jev_latencies.append(max(0,self.ctx.clock.now()-request['clock']))
        if (request['epoch']!=self.epoch or request['deaths']!=self.last_obs_deaths or
            request['interface']!=self.interface_version or request['mission']!=self.mission_serial or
            self.last_frame-request['frame']>90):
            self.s1_rejected+=1
            return
        if (_finite_number(self._answer_value(answers.get('audit')),0)>=0.80 and
            self.last_frame-self.last_goal_progress>=180):
            self.review_reason='jev_evasion_audit'
        a,b=answers.get('mission_a'),answers.get('mission_b')
        value=self._answer_value(a)
        if not isinstance(value,str) or value!=self._answer_value(b) or value not in request['options']:
            self.s1_rejected+=1
            return
        probability=min(self._answer_probability(a,value),self._answer_probability(b,value))
        if probability<0.70:
            self.s1_rejected+=1
            return
        spec=request['options'][value]
        if not self._target_exists(spec):
            return
        packet=self._mission_packet(obs,spec['target'],spec['kind'],spec['label'])
        if not packet:
            return
        if self.mission and self.mission['target']==spec['target']:
            self.s1_applied+=1
            return
        release=_finite_number(self._answer_value(answers.get('release')),0)
        old=self._mission_packet(obs,self.mission['target'],self.mission['kind'],'current') if self.mission else None
        resource_improves=(packet['kind']=='power' and old and old['profile']['risk']>=2
                           and packet['profile']['risk']<old['profile']['risk'])
        stalled=(self.last_frame-self.mission_progress>=240 and
                 self.last_frame-self.last_goal_progress>=180 and release>=0.80)
        if self.mission and not stalled and not resource_improves:
            self.s1_rejected+=1
            return
        if old and packet['profile']['risk']>old['profile']['risk']:
            self.s1_rejected+=1
            return
        before=self.mission_serial
        self._start_mission(packet,'jev')
        self.s1_applied+=1
        self.s1_switches+=1
        self._log('jev_mission_changed',frame=self.last_frame,old_serial=before,new_serial=self.mission_serial,
                  option=value,probability=probability,age_frames=self.last_frame-request['frame'],
                  reason='resource_improves' if resource_improves else 'stalled_mission')

    def _maybe_audit(self, obs):
        if self.ctx.done or self.astra_future is not None or self.astra_calls>=3:
            return
        if not self.review_reason or self.last_frame<600 or self.last_frame-self.last_audit_frame<1800:
            return
        budget=self.ctx.budget.remaining()
        if budget.get('system2_calls',0)<1 or budget.get('wall_s',0)<115:
            return
        evidence={'reason':self.review_reason,'epoch':self.epoch,'lives':self.last_lives,
                  'score':self.last_score,'dots':len(self.remaining),'powers':len(self.visible_power_keys),
                  'recent_completed_missions':list(self.mission_history)[-6:],
                  'jev_changed_missions':self.s1_switches,'jev_rejected':self.s1_rejected,
                  'temporary_evasions':self.guard_changes,
                  'frames_without_score':self.last_frame-self.last_goal_progress,
                  'current_rubric':self.interface}
        prompt=(
            'Select a durable MISSION-SELECTION rubric for Atari 2600 Ms. Pac-Man, not a joystick. '
            'Code infers corridors from observed dots, projects ghost interception, and controls every 67ms tick. '
            'The collection mission persists until an object disappears; temporary evasions rejoin it. '
            'Jev receives semantic options: dot run, nearest dot, power resource, or keep current mission. '
            'Jev has no inherited context: every request includes rules, rubric, named risk/travel categories, '
            'and recent progress. Two reordered Choices must agree with selected probabilities >=.70. '
            'Noul returns yes probability; >=.80 plus measured mission/score stall permits release. '
            'Code revalidates epoch/life/mission, target existence and current risk; it performs all math. '
            'Choose one rubric: completion=finish productive mission and dot runs; '
            'mobility=prefer shorter non-repeating approaches after repeated failed missions; '
            'resources=use safely reachable pills under pressure before death, not hoarding. '
            'Ghost RGB is not frightened evidence. No arcade route imports. '
            'Return only POLICY=completion or POLICY=mobility or POLICY=resources. '
            'The answer changes the instructions in subsequent Jev requests, never immediate steering.\n'
            +json.dumps(evidence,separators=(',',':')))
        self.astra_calls+=1
        self.last_audit_frame=self.last_frame
        self.astra_request={'epoch':self.epoch,'reason':self.review_reason}
        self.review_reason=None
        self.astra_future=self.ctx.spawn(lambda:self.ctx.think(prompt,timeout_s=110.0))

    def _poll_astra(self):
        if self.astra_future is None or not self.astra_future.done():
            return
        future,request=self.astra_future,self.astra_request
        self.astra_future=None; self.astra_request=None
        try:
            result=future.result()
        except ModelUnavailable:
            raise
        except (TimeoutError,RuntimeError,TypeError,ValueError):
            return
        text=getattr(result,'text','')
        if not isinstance(text,str) or not request or request['epoch']!=self.epoch:
            return
        # Archived Astra sometimes repeats the same assignment. Consistent
        # duplicates are harmless; conflicting assignments and prose are ignored.
        assignments=re.findall(r'POLICY\s*=\s*(completion|mobility|resources)',text,re.I)
        if not assignments or len(set(a.lower() for a in assignments))!=1:
            return
        rubric=assignments[0].lower()
        self.interface=rubric
        self.interface_version+=1
        self._log('astra_mission_interface',frame=self.last_frame,epoch=self.epoch,
                  rubric=rubric,version=self.interface_version,reason=request['reason'])

    def run_episode(self, game):
        if not getattr(game,'tick_s',None):
            return
        self._prepare_actions(game)
        self.ctx.act(self._valid('left'))
        while not self.ctx.done and not getattr(game,'done',False):
            obs=self.ctx.observe()
            if not isinstance(obs,dict):
                self.ctx.act(self._valid(self.current_action))
                self.ctx.wait_for_tick()
                continue
            if obs.get('game_over'):
                return
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
