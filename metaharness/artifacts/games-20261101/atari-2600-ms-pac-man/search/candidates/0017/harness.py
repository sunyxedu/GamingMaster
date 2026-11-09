"""Continuous graph controller with a code-owned emergency lock.

The controller is deliberately asymmetric: an observation-derived dot graph owns
ordinary motion, while Jev can arbitrate a safe junction.  Once a ghost is in the
short collision horizon, however, the answer from Jev is not allowed to replace
the deterministic escape choice.  The other important state machine is the maze
resource epoch: a jump from an almost empty dot set back to a full set creates a
new route/resource map even when the adaptor's maze counter is stale.
"""
from __future__ import annotations

import json
import math
from collections import deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice, Noul


CARDINALS = ("up", "right", "left", "down")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
ACTOR_TO_DOT = (3, 4)
EDGE_TOLERANCE = 12.5
ROUTINE_PILLS = 2


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _xy(obj):
    if not isinstance(obj, dict):
        return (0, 0)
    pair = obj.get("xy")
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return (_integer(pair[0]), _integer(pair[1]))
    return (_integer(obj.get("x")), _integer(obj.get("y")))


def _visible(obj, default=True):
    return isinstance(obj, dict) and bool(obj.get("visible", default))


def _points(objects, y_limit=220):
    result = set()
    if not isinstance(objects, list):
        return result
    for obj in objects:
        if not _visible(obj):
            continue
        x, y = _xy(obj)
        if 0 <= x <= 180 and 0 <= y <= y_limit:
            result.add((x, y))
    return result


def _action_between(a, b):
    ax, ay = a
    bx, by = b
    if ay == by:
        if abs(bx - ax) > 40:
            return "left" if ax > bx else "right"
        return "right" if bx > ax else "left"
    if ax == bx:
        return "down" if by > ay else "up"
    # A sparse or malformed dot set must never cause a diagonal command.
    return "right" if bx > ax else "left"


def _distance_bucket(distance):
    if distance <= 16:
        return "adjacent"
    if distance <= 36:
        return "near"
    if distance <= 72:
        return "medium"
    return "far"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.valid_names = set()
        self.noop = "noop"

        # Dynamic corridor model.  It is rebuilt on a real maze epoch, not on
        # every death, so deaths teach the controller which directed edge to try
        # later without destroying the learned geometry.
        self.base_maze_key = None
        self.epoch = 0
        self.nodes = set()
        self.adj = {}
        self.wrap_edges = set()
        self.remaining = set()
        self.route = []
        self.route_target = None
        self.current_action = "left"

        # Power pills are both goals and protected resources.  A target is never
        # an accidental intermediate node of ordinary dot BFS.
        self.power_points = set()
        self.visible_power = set()
        self.power_targets = {}
        self.power_used = 0
        self.active_power = None
        self.active_power_node = None
        self.goal_kind = "dots"
        self.force_escape = False
        self.failed_power = set()
        self.pill_effect_until = -1

        self.previous_xy = None
        self.previous_deaths = 0
        self.previous_clears = 0
        self.previous_visible_count = 0
        self.last_score = 0.0
        self.started_moving = False
        self.stall_ticks = 0
        self.stall_edge = None
        self.recovery = None
        self.edge_history = deque(maxlen=48)
        self.bad_edges = {}

        self.risk_mode = "balanced"
        self.policy_version = 0
        self.astra_future = None
        self.astra_calls = 0

        self.jev_future = None
        self.jev_request = None
        self.last_query_key = None
        self.override = None

    # ---------- defensive interface helpers ----------

    def _prepare_actions(self, game):
        try:
            names = getattr(game.action_spec, "names", ())
            self.valid_names = {str(x) for x in names}
            candidate = str(getattr(game.action_spec, "noop", "noop"))
            self.noop = candidate if candidate in self.valid_names else "noop"
        except Exception:
            self.valid_names = set()
            self.noop = "noop"

    def _valid(self, action):
        action = action if isinstance(action, str) else self.noop
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
    def _base_key(obs):
        maze = obs.get("maze", {}) if isinstance(obs, dict) else {}
        map_id = maze.get("map_id") if isinstance(maze, dict) else None
        if isinstance(map_id, (dict, list)):
            map_id = json.dumps(map_id, sort_keys=True, default=str)
        return (str(obs.get("maze_number")), str(obs.get("mazes_cleared")), str(map_id))

    @staticmethod
    def _is_done_obs(obs):
        return isinstance(obs, dict) and bool(obs.get("game_over", False))

    def _player_node(self, obs):
        if not self.nodes:
            return None
        x, y, _ = self._player(obs)
        return self._nearest((x + ACTOR_TO_DOT[0], y + ACTOR_TO_DOT[1]))

    def _nearest(self, point):
        if not self.nodes:
            return None
        px, py = point
        return min(self.nodes, key=lambda p: (abs(p[0] - px) + abs(p[1] - py), p[1], p[0]))

    def _object_node(self, point):
        if not self.nodes:
            return None
        return self._nearest((point[0] + ACTOR_TO_DOT[0], point[1] + ACTOR_TO_DOT[1]))

    def _node_distance(self, obs, node):
        if node is None:
            return 999
        x, y, _ = self._player(obs)
        return abs(x + 3 - node[0]) + abs(y + 4 - node[1])

    # ---------- map and resource epochs ----------

    def _rebuild_graph(self):
        self.adj = {node: set() for node in self.nodes}
        ordered = sorted(self.nodes)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                same_row = a[1] == b[1] and 0 < abs(a[0] - b[0]) <= EDGE_TOLERANCE
                same_col = a[0] == b[0] and 0 < abs(a[1] - b[1]) <= EDGE_TOLERANCE
                if same_row or same_col:
                    self.adj[a].add(b)
                    self.adj[b].add(a)
        for a, b in self.wrap_edges:
            if a in self.adj and b in self.adj:
                self.adj[a].add(b)
                self.adj[b].add(a)

    def _reset_epoch(self, key, points, reason):
        self.base_maze_key = key
        self.epoch += 1
        self.nodes = set(points)
        self.adj = {}
        self.wrap_edges = set()
        self.remaining = set(points)
        self.route = []
        self.route_target = None
        self.power_points = set()
        self.visible_power = set()
        self.power_targets = {}
        self.power_used = 0
        self.active_power = None
        self.active_power_node = None
        self.goal_kind = "dots"
        self.force_escape = False
        self.failed_power = set()
        self.pill_effect_until = -1
        self.bad_edges = {}
        self.edge_history.clear()
        self.recovery = None
        self.stall_edge = None
        self.stall_ticks = 0
        self.previous_xy = None
        self._rebuild_graph()
        self.policy_version += 1
        try:
            self.ctx.log("map_epoch", epoch=self.epoch, reason=reason, visible_dots=len(points))
        except Exception:
            pass

    def _learn_wrap(self, obs):
        if self.previous_xy is None or not self.nodes:
            return
        x, y, player = self._player(obs)
        if not _visible(player):
            self.previous_xy = (x, y)
            return
        ox, oy = self.previous_xy
        if abs(x - ox) > 80 and abs(y - oy) <= 18:
            a = self._nearest((ox + 3, oy + 4))
            b = self._nearest((x + 3, y + 4))
            if a is not None and b is not None and a != b:
                edge = (a, b) if a <= b else (b, a)
                if edge not in self.wrap_edges:
                    self.wrap_edges.add(edge)
                    self._rebuild_graph()
                    self.policy_version += 1
        self.previous_xy = (x, y)

    def _power_node(self, point):
        rows = sorted({n[1] for n in self.nodes})
        if not rows:
            return point
        row = min(rows, key=lambda r: (abs(r - point[1]), r))
        return (point[0], row)

    def _register_power(self):
        changed = False
        for point in self.power_points:
            node = self._power_node(point)
            if node not in self.nodes:
                self.nodes.add(node)
                changed = True
            entry = self.power_targets.get(point)
            if entry is None:
                self.power_targets[point] = {"node": node, "used": False}
            else:
                entry["node"] = node
        if changed:
            self._rebuild_graph()
            self.policy_version += 1

    def _reserve_nodes(self):
        result = set()
        for point, entry in self.power_targets.items():
            if point in self.power_points and not entry.get("used"):
                if point != self.active_power:
                    result.add(entry.get("node"))
        return {node for node in result if node in self.adj}

    def _power_candidates(self, obs, include_failed=False):
        start = self._player_node(obs)
        candidates = []
        for point, entry in self.power_targets.items():
            if point not in self.power_points or entry.get("used"):
                continue
            if not include_failed and point in self.failed_power:
                continue
            node = entry.get("node")
            distance = self._graph_distance(start, node, cap=999)
            if node in self.adj and distance < 999:
                candidates.append((distance, point, node))
        if not candidates and not include_failed:
            return self._power_candidates(obs, include_failed=True)
        candidates.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
        return candidates

    def _select_power_goal(self, obs, force=False):
        if self.active_power is not None and not force:
            return
        requested = force or self.force_escape
        if not requested and self.power_used >= ROUTINE_PILLS:
            self.goal_kind = "dots"
            self.active_power = None
            self.active_power_node = None
            return
        candidates = self._power_candidates(obs)
        if not candidates:
            self.goal_kind = "dots"
            self.active_power = None
            self.active_power_node = None
            return
        # A different corner is preferred after a death if the old approach was
        # the last edge recorded before that death.
        if requested and self.failed_power:
            fresh = [c for c in candidates if c[1] not in self.failed_power]
            candidates = fresh or candidates
        _, point, node = candidates[0]
        self.active_power = point
        self.active_power_node = node
        self.goal_kind = "escape" if requested else "checkpoint"
        self.route = []
        self.route_target = node
        self.force_escape = False
        self.policy_version += 1
        try:
            self.ctx.log("power_goal", point=list(point), kind=self.goal_kind,
                         life=_integer(obs.get("deaths")), epoch=self.epoch)
        except Exception:
            pass

    def _note_power_consumption(self, obs, old_power, current_power):
        disappeared = set(old_power) - set(current_power)
        if not disappeared:
            return
        px, py, _ = self._player(obs)
        score_gain = _number(obs.get("score"), self.last_score) - self.last_score
        near = {p for p in disappeared if abs(px - p[0]) + abs(py - p[1]) <= 30}
        if not near and self.active_power in disappeared and score_gain >= 45:
            near.add(self.active_power)
        if not near:
            return
        for point in near:
            if point in self.power_targets and not self.power_targets[point].get("used"):
                self.power_targets[point]["used"] = True
                self.power_used += 1
        if self.active_power in near:
            old_kind = self.goal_kind
            self.active_power = None
            self.active_power_node = None
            self.goal_kind = "dots"
            self.route = []
            self.route_target = None
            self.pill_effect_until = _integer(obs.get("frame")) + 180
            self.policy_version += 1
            try:
                self.ctx.log("power_used", kind=old_kind, used=self.power_used,
                             points=[list(x) for x in sorted(near)])
            except Exception:
                pass

    # ---------- death/progress memory ----------

    def _edge_bad(self, a, b):
        return a is not None and b is not None and self.bad_edges.get((a, b), 0) > 0

    def _edge_blocked(self, a, b):
        if self._edge_bad(a, b):
            return True
        if self.stall_edge is None or a is None or b is None:
            return False
        node, action = self.stall_edge
        return a == node and _action_between(a, b) == action

    def _neighbor_for_action(self, node, action, avoid=True):
        for candidate_action, neighbor in self._ordered_neighbors(node, self.current_action, avoid=avoid):
            if candidate_action == action:
                return neighbor
        return None

    def _remember_death(self, obs):
        records = list(self.edge_history)[-32:]
        if not records:
            return
        usable = [r for r in records if r.get("edge") is not None]
        if not usable:
            return
        # A contact-near record is better evidence than the respawn coordinate;
        # death observations themselves are deliberately never treated as the
        # collision location.
        near = [r for r in usable if r.get("ghost_distance", 999) <= 42]
        chosen = min(near or usable,
                     key=lambda r: (r.get("ghost_distance", 999), -r.get("frame", 0)))
        edge = tuple(chosen["edge"])
        self.bad_edges[edge] = min(3, self.bad_edges.get(edge, 0) + 1)
        self.route = []
        self.route_target = None
        self.override = None
        self.recovery = None
        try:
            self.ctx.log("death_edge_memory", edge=[list(edge[0]), list(edge[1])],
                         ghost_distance=chosen.get("ghost_distance", 999),
                         penalty=self.bad_edges[edge])
        except Exception:
            pass

    def _update_progress(self, obs, reset=False):
        x, y, _ = self._player(obs)
        current = (x, y)
        moved = self.previous_xy is not None and current != self.previous_xy
        score = _number(obs.get("score"), self.last_score)
        if moved or score > self.last_score:
            self.started_moving = True
        if reset or self.previous_xy is None or not self.started_moving:
            self.stall_ticks = 0
            self.stall_edge = None
            self.recovery = None
        elif moved:
            self.stall_ticks = 0
            self.stall_edge = None
            self.recovery = None
        elif self.current_action in CARDINALS:
            self.stall_ticks += 1
        node = self._player_node(obs)
        if self.stall_ticks >= 2 and node is not None and self.current_action in CARDINALS:
            edge = (node, self.current_action)
            if self.stall_edge != edge:
                self.route = []
                self.route_target = None
                self.override = None
                self.stall_edge = edge
                try:
                    self.ctx.log("no_progress", node=list(node), action=self.current_action)
                except Exception:
                    pass
        self.previous_xy = current
        self.last_score = score

    # ---------- observation update ----------

    def _update(self, obs):
        if not isinstance(obs, dict):
            return
        key = self._base_key(obs)
        points = _points(obs.get("pills", []), 190)
        old_count = len(self.remaining)
        clears = _integer(obs.get("mazes_cleared"), self.previous_clears)
        new_epoch = self.base_maze_key is None or key != self.base_maze_key
        # The adaptor can report maze_number=1 and mazes_cleared=0 after a
        # repopulation.  A near-empty -> full dot set is stronger evidence.
        repopulated = (
            self.base_maze_key is not None and old_count <= 12 and len(points) >= 40
        ) or (old_count <= 4 and len(points) > old_count + 40)
        reset = new_epoch or repopulated or clears != self.previous_clears
        if reset:
            self._reset_epoch(key, points, "initial" if self.base_maze_key is None else
                              ("counter" if new_epoch or clears != self.previous_clears else "repopulation"))
        else:
            if points - self.nodes:
                self.nodes.update(points)
                self._rebuild_graph()
            self.remaining = set(points)
        self._learn_wrap(obs)

        old_power = set() if reset else set(self.visible_power)
        current_power = _points(obs.get("power_pills", []), 165)
        current_power = {p for p in current_power if 0 <= p[0] <= 160 and 0 <= p[1] <= 165}
        self.power_points = current_power
        self._register_power()

        deaths = _integer(obs.get("deaths"), self.previous_deaths)
        death_changed = deaths > self.previous_deaths
        if death_changed:
            self._remember_death(obs)
            if self.active_power is not None and self.active_power in current_power:
                self.failed_power.add(self.active_power)
            self.active_power = None
            self.active_power_node = None
            self.goal_kind = "dots"
            self.force_escape = True
            self.route = []
            self.route_target = None
            self.stall_ticks = 0
            self.stall_edge = None
            self.recovery = None
        if death_changed or reset or clears != self.previous_clears:
            self.last_query_key = None
            self.override = None
        self._note_power_consumption(obs, old_power, current_power)
        self.visible_power = current_power
        self.previous_deaths = deaths
        self.previous_clears = clears
        self.previous_visible_count = len(points)
        if not reset and not self._is_done_obs(obs):
            self._select_power_goal(obs)
        self._update_progress(obs, reset=reset or death_changed)

    # ---------- graph search and route execution ----------

    def _direction_order(self, preferred):
        if preferred == "left":
            return ("left", "down", "up", "right")
        if preferred == "right":
            return ("right", "up", "down", "left")
        if preferred == "down":
            return ("down", "right", "left", "up")
        return ("up", "left", "right", "down")

    def _ordered_neighbors(self, node, preferred=None, avoid=True):
        if node not in self.adj:
            return []
        ranks = {a: i for i, a in enumerate(self._direction_order(preferred or self.current_action))}
        pairs = []
        for neighbor in self.adj[node]:
            if avoid and self._edge_blocked(node, neighbor):
                continue
            action = _action_between(node, neighbor)
            pairs.append((ranks.get(action, 99), action, neighbor))
        pairs.sort(key=lambda item: (item[0], item[1], item[2][1], item[2][0]))
        return [(action, neighbor) for _, action, neighbor in pairs]

    def _path(self, start, goals, preferred=None, avoid=True):
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
            for _, neighbor in self._ordered_neighbors(node, preferred if node == start else None, avoid=avoid):
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
        return list(reversed(path))

    def _ensure_route(self, obs):
        start = self._player_node(obs)
        if start is None or not self.adj:
            self.route = []
            return start

        if self.active_power is not None and self.active_power not in self.power_points:
            self.active_power = None
            self.active_power_node = None
            self.goal_kind = "dots"
            self.route = []

        if self.route:
            try:
                self.route = self.route[self.route.index(start):]
            except ValueError:
                self.route = []
            if len(self.route) <= 1:
                self.route = []
            elif self._edge_blocked(self.route[0], self.route[1]):
                self.route = []

        if self.active_power_node is not None:
            if not self.route:
                self.route = self._path(start, [self.active_power_node], self.current_action, avoid=True)
                if not self.route:
                    self.route = self._path(start, [self.active_power_node], self.current_action, avoid=False)
            self.route_target = self.active_power_node if self.route else None
            if self.route:
                return start
            self.active_power = None
            self.active_power_node = None
            self.goal_kind = "dots"

        if not self.route:
            goals = set(self.remaining) - self._reserve_nodes()
            goals.discard(start)
            if not goals:
                goals = set(self.nodes) - self._reserve_nodes()
                goals.discard(start)
            self.route = self._path(start, goals, self.current_action, avoid=True)
            if not self.route:
                self.route = self._path(start, goals, self.current_action, avoid=False)
            self.route_target = self.route[-1] if self.route else None
        return start

    def _route_action(self, obs):
        start = self._ensure_route(obs)
        if len(self.route) >= 2:
            return _action_between(self.route[0], self.route[1])
        if start is not None:
            options = self._ordered_neighbors(start, self.current_action, avoid=True)
            if not options:
                options = self._ordered_neighbors(start, self.current_action, avoid=False)
            if options:
                return options[0][0]
        _, _, player = self._player(obs)
        direction = player.get("direction") if isinstance(player, dict) else None
        return direction if direction in CARDINALS else (self.current_action if self.current_action in CARDINALS else "left")

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

    # ---------- ghosts and emergency escape ----------

    def _active_ghosts(self, obs):
        result = []
        frame = _integer(obs.get("frame"))
        raw = obs.get("ghosts", []) if isinstance(obs, dict) else []
        if not isinstance(raw, list):
            return result
        for ghost in raw:
            if not isinstance(ghost, dict) or not _visible(ghost):
                continue
            x, y = _xy(ghost)
            dx, dy = _integer(ghost.get("dx")), _integer(ghost.get("dy"))
            if x == 0 and y == 0 and dx == 0 and dy == 0:
                continue
            # During the automatic opening the decoder temporarily puts the
            # sprite coordinates into dx/dy.  Those are not velocities.
            if not self.started_moving and frame < 180:
                continue
            if dx == x and dy == y and abs(x) > 10 and abs(y) > 10:
                dx, dy = 0, 0
            result.append({"x": x, "y": y, "dx": max(-4, min(4, dx)),
                           "dy": max(-4, min(4, dy)),
                           "node": self._object_node((x, y))})
        return result

    def _relation(self, obs, ghost):
        px, py, _ = self._player(obs)
        gx, gy = ghost["x"], ghost["y"]
        vx, vy = ghost["dx"], ghost["dy"]
        physical = abs(px - gx) + abs(py - gy)
        predicted = abs(px - (gx + 3 * vx)) + abs(py - (gy + 3 * vy))
        relx, rely = px - gx, py - gy
        if abs(relx) >= abs(rely) and vx:
            closing = (relx > 0 and vx > 0) or (relx < 0 and vx < 0)
        elif vy:
            closing = (rely > 0 and vy > 0) or (rely < 0 and vy < 0)
        else:
            closing = False
        same_lane = abs(px - gx) <= 9 or abs(py - gy) <= 9
        return {"physical": physical, "predicted": predicted, "closing": closing,
                "same_lane": same_lane,
                "graph": self._graph_distance(self._player_node(obs), ghost.get("node"), 12)}

    def _hazard(self, obs):
        relations = [self._relation(obs, g) for g in self._active_ghosts(obs)]
        level = 0
        physical_immediate = False
        for r in relations:
            immediate = (
                r["physical"] <= 14 or r["predicted"] <= 16 or
                (r["same_lane"] and r["closing"] and r["physical"] <= 28) or
                (r["graph"] <= 1 and r["physical"] <= 34)
            )
            approaching = (
                (r["same_lane"] and r["closing"] and r["physical"] <= 55) or
                (r["graph"] <= 2 and r["physical"] <= 55) or r["physical"] <= 30
            )
            if immediate:
                level = 2
            elif approaching:
                level = max(level, 1)
            physical_immediate = physical_immediate or r["physical"] <= 18 or r["predicted"] <= 18
        nearest = min((r["physical"] for r in relations), default=999)
        if not self.started_moving and _integer(obs.get("deaths")) == 0:
            level = 0
            physical_immediate = False
        return {
            "level": level,
            "physical_immediate": physical_immediate,
            "nearest": nearest,
            "nearest_bucket": _distance_bucket(nearest),
            "closing": sum(1 for r in relations if r["closing"] and r["same_lane"]),
            "graph_near": sum(1 for r in relations if r["graph"] <= 2 and r["physical"] <= 55),
            "relations": relations,
            "label": "immediate" if level >= 2 else ("approaching" if level else "open"),
        }

    def _candidate_risk(self, obs, node):
        if node is None:
            return "unknown"
        best = 999
        best_graph = 999
        for ghost in self._active_ghosts(obs):
            candidate = (node[0] - 3, node[1] - 4)
            distance = abs(candidate[0] - ghost["x"]) + abs(candidate[1] - ghost["y"])
            future = abs(candidate[0] - (ghost["x"] + 2 * ghost["dx"])) + abs(candidate[1] - (ghost["y"] + 2 * ghost["dy"]))
            best = min(best, distance, future)
            best_graph = min(best_graph, self._graph_distance(node, ghost.get("node"), 12))
        if best <= 18 or (best_graph <= 1 and best <= 36):
            return "danger"
        if best <= 46 or (best_graph <= 2 and best <= 56):
            return "caution"
        return "open"

    def _start_escape(self, obs):
        if self.active_power is not None and self.goal_kind == "escape":
            return True
        start = self._player_node(obs)
        if start is None:
            return False
        options = self._ordered_neighbors(start, self.current_action, avoid=True)
        if not options:
            options = self._ordered_neighbors(start, self.current_action, avoid=False)
        choices = []
        for point, entry in self.power_targets.items():
            if point not in self.power_points or entry.get("used") or point in self.failed_power:
                continue
            node = entry.get("node")
            for action, neighbor in options:
                path = self._path(neighbor, [node], action, avoid=True)
                if not path:
                    path = self._path(neighbor, [node], action, avoid=False)
                if path:
                    risk = self._candidate_risk(obs, neighbor)
                    choices.append((risk == "danger", len(path), point, node))
        if not choices:
            return False
        choices.sort(key=lambda x: (x[0], x[1], x[2][1], x[2][0]))
        _, _, point, node = choices[0]
        self.active_power = point
        self.active_power_node = node
        self.goal_kind = "escape"
        self.route = []
        self.route_target = node
        self.policy_version += 1
        try:
            self.ctx.log("escape_commit", point=list(point), deaths=_integer(obs.get("deaths")))
        except Exception:
            pass
        return True

    def _escape_action(self, obs, base, hazard):
        start = self._player_node(obs)
        options = self._ordered_neighbors(start, self.current_action, avoid=True) if start is not None else []
        if not options and start is not None:
            options = self._ordered_neighbors(start, self.current_action, avoid=False)
        if not options:
            return OPPOSITE.get(base, base)

        lives = _integer(obs.get("lives"), 3)
        should_spend = hazard["physical_immediate"] or (lives <= 1 and hazard["level"] >= 1)
        if should_spend:
            self._start_escape(obs)
        if self.active_power_node is not None:
            power_action = self._route_action(obs)
            neighbors = {a: n for a, n in options}
            if power_action in neighbors and self._candidate_risk(obs, neighbors[power_action]) != "danger":
                return power_action

        scored = []
        neighbors = {a: n for a, n in options}
        for action, neighbor in options:
            risk = self._candidate_risk(obs, neighbor)
            distances = []
            graph_separation = []
            for ghost in self._active_ghosts(obs):
                candidate = (neighbor[0] - 3, neighbor[1] - 4)
                distances.append(abs(candidate[0] - ghost["x"]) + abs(candidate[1] - ghost["y"]))
                graph_separation.append(self._graph_distance(neighbor, ghost.get("node"), 12))
            physical = min(distances or [99])
            separation = min(graph_separation or [12])
            penalty = {"danger": -120, "caution": -10, "open": 0, "unknown": -3}[risk]
            continuity = 4 if action == base and not hazard["physical_immediate"] else 0
            retreat = 5 if hazard["physical_immediate"] and action == OPPOSITE.get(base) else 0
            scored.append((physical * 2 + separation * 14 + penalty + continuity + retreat, action))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if hazard["physical_immediate"]:
            self.route = []
        return scored[0][1] if scored else base

    def _fallback_action(self, obs):
        frame = _integer(obs.get("frame"))
        if self.recovery is not None:
            action, node, until = self.recovery
            if frame <= until and self._player_node(obs) == node and action in CARDINALS:
                return action
            self.recovery = None
        base = self._route_action(obs)
        hazard = self._hazard(obs)
        if self.stall_ticks >= 2 and self.stall_edge is not None:
            node, blocked_action = self.stall_edge
            options = self._ordered_neighbors(node, self.current_action, avoid=True)
            alternatives = [a for a, _ in options if a != blocked_action]
            if alternatives:
                action = alternatives[0]
                self.recovery = (action, node, frame + 36)
                self.route = []
                return action
        if hazard["level"] >= 2 or (hazard["level"] == 1 and self.risk_mode == "survival"):
            return self._escape_action(obs, base, hazard)
        return base

    def _record_edge(self, obs, action, hazard):
        node = self._player_node(obs)
        neighbor = self._neighbor_for_action(node, action, avoid=False) if node is not None else None
        if node is None or neighbor is None:
            return
        self.edge_history.append({
            "edge": (node, neighbor),
            "frame": _integer(obs.get("frame")),
            "ghost_distance": hazard.get("nearest", 999),
        })

    # ---------- Astra strategic policy ----------

    def _start_astra(self, obs):
        if self.astra_future is not None or self.astra_calls >= 1:
            return
        try:
            if _integer(self.ctx.budget.remaining().get("system2_calls"), 0) <= 0:
                return
        except Exception:
            return
        x, y, player = self._player(obs)
        summary = {
            "frame": _integer(obs.get("frame")),
            "score": _number(obs.get("score")),
            "lives": _integer(obs.get("lives")),
            "deaths": _integer(obs.get("deaths")),
            "maze_epoch": self.epoch,
            "visible_dots": len(self.remaining),
            "visible_power_pills": len(self.power_points),
            "player": {"xy": [x, y], "direction": str(player.get("direction", "unknown"))},
            "current_policy": self.risk_mode,
        }
        prompt = (
            "You are the slow strategic policy author for Atari 2600 Ms. Pac-Man CX2675. "
            "The realtime code owns every joystick tick and has a verified local dot graph. "
            "Do not output per-tick actions, an arcade map, or ghost arithmetic. Preserve "
            "lives, clear dots, and keep an unspent power pill as an escape; only use score "
            "mode when the current evidence is safe. Return exactly one line MODE=survival, "
            "MODE=balanced, or MODE=score, followed by at most two short policy notes. "
            "Malformed text is ignored.\n\n" + json.dumps(summary, separators=(",", ":"))
        )
        self.astra_calls += 1
        try:
            self.astra_future = self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=90.0))
        except (BudgetExceeded, EpisodeOver):
            self.astra_future = None
        except ModelUnavailable:
            raise
        except Exception:
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
        except Exception:
            return
        text = getattr(result, "text", result)
        if not isinstance(text, str):
            return
        low = text.lower()
        marker = low.find("mode=")
        if marker < 0:
            return
        tail = low[marker : marker + 28]
        for mode in ("survival", "balanced", "score"):
            if mode in tail and mode != self.risk_mode:
                self.risk_mode = mode
                self.policy_version += 1
                try:
                    self.ctx.log("astra_policy", mode=mode)
                except Exception:
                    pass
                break

    # ---------- Jev local arbitration ----------

    def _legal_options(self, obs, base):
        node = self._player_node(obs)
        pairs = self._ordered_neighbors(node, self.current_action, avoid=True)
        if not pairs:
            pairs = self._ordered_neighbors(node, self.current_action, avoid=False)
        options = [a for a, _ in pairs]
        if base in CARDINALS and base not in options:
            options.append(base)
        result = []
        for action in options:
            action = self._valid(action)
            if action in CARDINALS and action not in result:
                result.append(action)
        return result or ([base] if base in CARDINALS else ["left"])

    def _jev_state(self, obs, base, options, hazard):
        node = self._player_node(obs)
        facts = {}
        neighbors = dict(self._ordered_neighbors(node, self.current_action, avoid=False)) if node else {}
        for action in options:
            neighbor = neighbors.get(action)
            facts[action] = {
                "keeps_code_route": action == base,
                "risk": self._candidate_risk(obs, neighbor),
                "blocked_by_memory": bool(neighbor is not None and self._edge_blocked(node, neighbor)),
                "branch_degree": len(self.adj.get(neighbor, ())) if neighbor is not None else 0,
            }
        return {
            "observation": {
                "frame": _integer(obs.get("frame")),
                "score": _number(obs.get("score")),
                "lives": _integer(obs.get("lives")),
                "deaths": _integer(obs.get("deaths")),
                "player_xy": list(self._player(obs)[:2]),
                "visible_dots": len(self.remaining),
                "visible_power_pills": len(self.visible_power),
            },
            "plan": {
                "risk_mode": self.risk_mode if _integer(obs.get("lives"), 3) > 1 else "survival",
                "goal": self.goal_kind,
                "fallback": base,
                "legal_actions": options,
                "node": list(node) if node else [0, 0],
                "policy_version": self.policy_version,
            },
            "threat": {
                "level": hazard["level"],
                "label": hazard["label"],
                "nearest_bucket": hazard["nearest_bucket"],
                "physical_immediate": hazard["physical_immediate"],
                "closing_lanes": hazard["closing"],
            },
            "options": facts,
        }

    @staticmethod
    def _choice(options, reverse=False):
        ordered = list(options)
        if reverse:
            ordered.reverse()
        criteria = {
            action: "A legal cardinal branch already computed by code; preserve motion and avoid the option whose supplied risk is danger."
            for action in ordered
        }
        return Choice(
            "Choose exactly one value from plan.legal_actions. Prefer plan.fallback unless threat is immediate or an option is marked danger. Do not invent a direction or calculate distances.",
            criteria,
        )

    def _launch_jev(self, obs, base, options, hazard):
        if getattr(self.ctx, "system1", None) is None or self.jev_future is not None or self.ctx.done:
            return
        node = self._player_node(obs)
        state = self._jev_state(obs, base, options, hazard)
        rotated = options[_integer(obs.get("ticks")) % len(options):] + options[:_integer(obs.get("ticks")) % len(options)]
        questions = {
            "move_a": self._choice(rotated),
            "move_b": self._choice(rotated, reverse=True),
            "escape": Noul(
                "Is the already computed fallback power-pill escape safer than continuing the current route now?",
                {"true": "The supplied threat facts make the existing escape route safer.",
                 "false": "Do not divert to the existing escape route yet."},
            ),
        }
        request = {
            "frame": _integer(obs.get("frame")),
            "deaths": _integer(obs.get("deaths")),
            "node": node,
            "options": set(options),
            "hazard": hazard["level"],
            "physical": hazard["physical_immediate"],
            "epoch": self.epoch,
            "policy": self.policy_version,
        }
        self.jev_request = request
        self.last_query_key = (node, base, hazard["level"], request["deaths"], self.epoch, self.policy_version)
        try:
            self.jev_future = self.ctx.spawn(lambda: self.ctx.decide(state, questions, timeout_s=1.2))
        except (BudgetExceeded, EpisodeOver):
            self.jev_future = None
            self.jev_request = None
        except ModelUnavailable:
            raise
        except Exception:
            self.jev_future = None
            self.jev_request = None

    @staticmethod
    def _answer_value(answer):
        if isinstance(answer, dict):
            return answer.get("value", answer.get("choice", answer.get("noul")))
        return getattr(answer, "value", None)

    @staticmethod
    def _answer_probability(answer, value):
        probabilities = answer.get("probabilities", {}) if isinstance(answer, dict) else getattr(answer, "probabilities", {})
        if not isinstance(probabilities, dict):
            return 0.0
        return _number(probabilities.get(value), 0.0)

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
        except Exception:
            return
        if not isinstance(answers, dict) or request is None or self.ctx.done:
            return
        frame = _integer(obs.get("frame"))
        if (_integer(obs.get("deaths")) != request["deaths"] or
                frame - request["frame"] > 24 or request["epoch"] != self.epoch or
                request["policy"] != self.policy_version):
            return
        hazard = self._hazard(obs)
        # The emergency lock is the central change: Jev cannot turn a code
        # emergency into the repeated route seen in the parent trace.
        if request["physical"] or hazard["physical_immediate"] or request["hazard"] >= 2:
            return
        current_node = self._player_node(obs)
        if current_node != request["node"]:
            return
        legal = set(self._legal_options(obs, self._route_action(obs)))
        a, b = answers.get("move_a"), answers.get("move_b")
        va, vb = self._answer_value(a), self._answer_value(b)
        if not isinstance(va, str) or va != vb or va not in request["options"] or va not in legal:
            return
        pa = self._answer_probability(a, va)
        pb = self._answer_probability(b, va)
        threshold = 0.55 if request["hazard"] else 0.40
        if pa < threshold or pb < threshold:
            return
        if self._candidate_risk(obs, dict(self._ordered_neighbors(current_node, self.current_action, avoid=False)).get(va)) == "danger":
            return
        selected = va
        esc = answers.get("escape")
        ev = self._answer_value(esc)
        ep = self._answer_probability(esc, "true")
        escape_yes = ev is True or ev == "true" or _number(ev, 0.0) >= 0.5
        if not escape_yes or ep < 0.60:
            selected = va
        self.override = {"action": self._valid(selected), "until": frame + 16, "node": current_node}
        try:
            self.ctx.log("jev_branch", action=self.override["action"], probability=min(pa, pb))
        except Exception:
            pass

    def _maybe_jev(self, obs, base, hazard):
        if getattr(self.ctx, "system1", None) is None or self.jev_future is not None or self.ctx.done:
            return
        node = self._player_node(obs)
        degree = len(self.adj.get(node, ())) if node is not None else 0
        near_junction = degree >= 3 and self._node_distance(obs, node) <= 12
        turn_soon = node is not None and base != self.current_action
        if not (near_junction or turn_soon or hazard["level"] >= 1 or len(self.nodes) < 4):
            return
        key = (node, base, hazard["level"], _integer(obs.get("deaths")), self.epoch, self.policy_version)
        if key == self.last_query_key:
            return
        options = self._legal_options(obs, base)
        if len(options) > 1:
            self._launch_jev(obs, base, options, hazard)

    def _choose_action(self, obs, fallback):
        hazard = self._hazard(obs)
        if hazard["level"] >= 2 or hazard["physical_immediate"]:
            self.override = None
            return self._valid(fallback)
        if self.override is not None:
            frame = _integer(obs.get("frame"))
            node = self._player_node(obs)
            action = self.override.get("action")
            if (frame <= self.override.get("until", -1) and node == self.override.get("node") and
                    action in self._legal_options(obs, fallback)):
                return self._valid(action)
            self.override = None
        return self._valid(fallback)

    # ---------- realtime episode ----------

    def run_episode(self, game):
        if not getattr(game, "tick_s", None):
            return
        self._prepare_actions(game)
        try:
            obs = game.observe()
        except Exception:
            return
        self._update(obs)
        self._start_astra(obs)

        while not self.ctx.done and not getattr(game, "done", False):
            try:
                obs = game.observe()
            except Exception:
                break
            self._update(obs)
            self._poll_astra()
            self._poll_jev(obs)
            if _integer(obs.get("lives"), 3) <= 1:
                self.risk_mode = "survival"
            fallback = self._fallback_action(obs)
            hazard = self._hazard(obs)
            self._maybe_jev(obs, fallback, hazard)
            action = self._choose_action(obs, fallback)
            self.current_action = action
            self._record_edge(obs, action, hazard)
            try:
                self.ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            try:
                self.ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver):
                return


def build(ctx):
    return Harness(ctx)
