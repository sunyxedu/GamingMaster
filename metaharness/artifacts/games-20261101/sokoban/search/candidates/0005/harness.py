"""Certified bidirectional action search for observation-only Sokoban.

The inverse transitions exist only in the planner: the live game is never
asked to pull or undo.  A complete witness is replayed before any live push.
The two search frontiers retain exact player positions and use a balanced
matching potential, so walking cost is part of the objective, not a tie-break
on a player-component abstraction.  All data come from the current JSON.
"""
from __future__ import annotations

import heapq
import json
import re
import time
from collections import deque


NAMES = ("up", "down", "left", "right")
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPPOSITE = (1, 0, 3, 2)
INF = 10 ** 12
# Trace scores verify -0.50 per push and -0.01 per action.  Multiplying
# by 100 gives integer edge costs: a walk costs 1, a push costs 51.
PUSH_COST = 51


def as_observation(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    return value if isinstance(value, dict) else None


class Board:
    """Dense floor coordinates and exact, invertible primitive transitions."""

    def __init__(self, observation):
        obs = as_observation(observation)
        if obs is None:
            raise ValueError("Observation must be a JSON object")
        raw = obs.get("board", "")
        rows = raw.splitlines() if isinstance(raw, str) else list(raw)
        if not rows or len(rows) > 128 or max(map(len, rows)) > 128:
            raise ValueError("Missing or excessive board")
        self.cells = [(r, c) for r, row in enumerate(rows)
                      for c, char in enumerate(row) if char in " .@$*+-_"]
        self.ids = {p: i for i, p in enumerate(self.cells)}
        self.n = len(self.cells)
        self.shift = max(1, self.n.bit_length())
        self.player_mask = (1 << self.shift) - 1
        self.bits = [1 << i for i in range(self.n)]
        self.neighbors = [tuple(self.ids.get((r + dr, c + dc), -1)
                                for dr, dc in DELTAS) for r, c in self.cells]
        self.goals = self._coordinate_mask(obs.get("goals", [p for p in self.cells
                                                  if rows[p[0]][p[1]] in ".*+"]))
        self.goal_count = self.goals.bit_count()
        self.initial_boxes, self.initial_player = self.state(obs)
        if not 0 < self.goal_count == self.initial_boxes.bit_count() <= 10:
            raise ValueError("Unequal or unsupported box/goal counts")
        self.goal_maps = [self.static_distances(i, reverse=True)
                          for i in self.indices(self.goals)]
        self.start_maps = [self.static_distances(i, reverse=False)
                           for i in self.indices(self.initial_boxes)]
        self.useful = 0
        for i in range(self.n):
            if any(m[i] < INF for m in self.goal_maps):
                self.useful |= self.bits[i]
        self.heuristic_cache = {}

    def _coordinate_mask(self, coordinates):
        result = 0
        for value in coordinates:
            p = (int(value[0]), int(value[1]))
            if p not in self.ids:
                raise ValueError("Object outside floor")
            result |= self.bits[self.ids[p]]
        return result

    def state(self, observation):
        obs = as_observation(observation)
        if obs is None:
            raise ValueError("Missing observation")
        boxes = self._coordinate_mask(obs["boxes"])
        p = self.ids[tuple(obs["player"])]
        if boxes & self.bits[p]:
            raise ValueError("Player overlaps a box")
        return boxes, p

    @staticmethod
    def indices(mask):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    def pack(self, boxes, player):
        return (boxes << self.shift) | player

    def unpack(self, state):
        return state >> self.shift, state & self.player_mask

    def static_distances(self, source, reverse):
        """Single-box push distances ignoring other boxes and walking access."""
        distance = [INF] * self.n
        distance[source] = 0
        todo = deque([source])
        while todo:
            b = todo.popleft()
            for d in range(4):
                if reverse:
                    dest = self.neighbors[b][OPPOSITE[d]]
                    stand = self.neighbors[dest][OPPOSITE[d]] if dest >= 0 else -1
                else:
                    dest = self.neighbors[b][d]
                    stand = self.neighbors[b][OPPOSITE[d]]
                if dest >= 0 and stand >= 0 and distance[dest] == INF:
                    distance[dest] = distance[b] + 1
                    todo.append(dest)
        return distance

    @staticmethod
    def matching(positions, maps):
        n = len(maps)
        dp = [INF] * (1 << n)
        dp[0] = 0
        for mask in range((1 << n) - 1):
            i = mask.bit_count()
            if dp[mask] >= INF:
                continue
            for j in range(n):
                if not mask & (1 << j):
                    cost = dp[mask] + maps[j][positions[i]]
                    target = mask | (1 << j)
                    if cost < dp[target]:
                        dp[target] = cost
        return dp[-1]

    def heuristics(self, boxes):
        result = self.heuristic_cache.get(boxes)
        if result is not None:
            return result
        positions = tuple(self.indices(boxes))
        if boxes & ~self.useful:
            result = (INF, INF)
        else:
            result = (PUSH_COST * self.matching(positions, self.goal_maps),
                      PUSH_COST * self.matching(positions, self.start_maps))
        self.heuristic_cache[boxes] = result
        return result

    def advance(self, boxes, player, direction):
        target = self.neighbors[player][direction]
        if target < 0:
            return None
        if boxes & self.bits[target]:
            dest = self.neighbors[target][direction]
            if dest < 0 or boxes & self.bits[dest]:
                return None
            return boxes ^ self.bits[target] ^ self.bits[dest], target, PUSH_COST
        return boxes, target, 1

    def verify(self, actions, require_solved=True):
        boxes, player = self.initial_boxes, self.initial_player
        cost = 0
        for action in actions:
            if action not in NAMES or boxes == self.goals:
                return None
            nxt = self.advance(boxes, player, NAMES.index(action))
            if nxt is None:
                return None
            boxes, player, edge = nxt
            cost += edge
        if require_solved and boxes != self.goals:
            return None
        return cost, boxes, player


class BidirectionalSearch:
    """Two primitive graphs joined with a consistent balanced potential.

    The forward potential is h_goal - h_start (twice the usual potential).
    Backward search reverses every primitive edge, including pushes, and
    starts with boxes on goals and every possible free player square.  Its
    potential is the negative of the forward potential.  Both heuristics are
    consistent static matching bounds, making every reduced edge nonnegative.
    min(frontier_f) + min(frontier_b) >= 2 * incumbent proves optimal cost.
    """

    def __init__(self, board):
        self.board = board
        self.heap = [[], []]
        self.records = [{}, {}]
        self.expanded = [0, 0]
        self.best_cost = INF
        self.best_plan = None
        self.proved = False
        self.saturated = False
        self.partial = (board.initial_boxes & board.goals).bit_count(), 0, None
        self.start = board.pack(board.initial_boxes, board.initial_player)
        hg, hs = board.heuristics(board.initial_boxes)
        if hg >= INF or hs >= INF:
            self.proved = True
            return
        self.records[0][self.start] = (0, None, None)
        heapq.heappush(self.heap[0], (hg - hs, 0, self.start))
        hg, hs = board.heuristics(board.goals)
        if hg >= INF or hs >= INF:
            self.proved = True
            return
        for player in range(board.n):
            if not board.goals & board.bits[player]:
                state = board.pack(board.goals, player)
                self.records[1][state] = (0, None, None)
                heapq.heappush(self.heap[1], (hs - hg, 0, state))

    def _path(self, state, side):
        result = []
        while state is not None:
            _cost, parent, action = self.records[side][state]
            if parent is None:
                break
            result.append(NAMES[action])
            state = parent
        return result[::-1] if side == 0 else result

    def offer(self, actions):
        checked = self.board.verify(actions)
        if checked is not None and checked[0] < self.best_cost:
            self.best_cost = checked[0]
            self.best_plan = list(actions)
            return True
        return False

    def _join(self, state):
        a, b = self.records[0].get(state), self.records[1].get(state)
        if a is not None and b is not None and a[0] + b[0] < self.best_cost:
            # Snapshot the witness.  Later improvements can replace parent
            # labels, so every reconstructed witness is verified independently.
            self.offer(self._path(state, 0) + self._path(state, 1))

    def _clean(self, side):
        queue, records = self.heap[side], self.records[side]
        while queue and records.get(queue[0][2], (INF,))[0] != queue[0][1]:
            heapq.heappop(queue)

    def run(self, seconds, stop=lambda: False, state_limit=700000):
        deadline = time.monotonic() + max(0.0, seconds)
        b = self.board
        count = 0
        while not self.proved and not self.saturated:
            if count % 128 == 0:
                if time.monotonic() >= deadline or stop():
                    break
                if len(self.records[0]) + len(self.records[1]) >= state_limit:
                    self.saturated = True
                    break
            self._clean(0)
            self._clean(1)
            if not self.heap[0] or not self.heap[1]:
                self.proved = True
                break
            if self.heap[0][0][0] + self.heap[1][0][0] >= 2 * self.best_cost:
                self.proved = True
                break
            # Expand the lower reduced-distance frontier.  Alternate ties to
            # keep both directions alive on equal-cost plateaus.
            top0, top1 = self.heap[0][0][0], self.heap[1][0][0]
            side = 0 if top0 < top1 else 1 if top1 < top0 else count & 1
            _priority, cost, state = heapq.heappop(self.heap[side])
            boxes, player = b.unpack(state)
            self.expanded[side] += 1
            count += 1
            if side == 0:
                placed = (boxes & b.goals).bit_count()
                if (placed, -cost) > self.partial[:2]:
                    self.partial = placed, -cost, state
            for direction in range(4):
                edges = []
                if side == 0:
                    edge = b.advance(boxes, player, direction)
                    if edge is not None:
                        edges.append(edge)
                else:
                    prev = b.neighbors[player][OPPOSITE[direction]]
                    if prev < 0 or boxes & b.bits[prev]:
                        continue
                    # Inverse walk: previous player walks in direction to p.
                    edges.append((boxes, prev, 1))
                    box = b.neighbors[player][direction]
                    if box >= 0 and boxes & b.bits[box]:
                        # Inverse push: previous player at p-d, box at p;
                        # forward push places player at p and box at p+d.
                        edges.append((boxes ^ b.bits[box] ^ b.bits[player], prev, PUSH_COST))
                for nb, np, edge_cost in edges:
                    new_cost = cost + edge_cost
                    if new_cost >= self.best_cost:
                        continue
                    ns = b.pack(nb, np)
                    if new_cost >= self.records[side].get(ns, (INF,))[0]:
                        continue
                    hg, hs = b.heuristics(nb)
                    if hg >= INF or hs >= INF:
                        continue
                    if new_cost + (hg if side == 0 else hs) >= self.best_cost:
                        continue
                    self.records[side][ns] = (new_cost, state, direction)
                    priority = 2 * new_cost + (hg - hs if side == 0 else hs - hg)
                    heapq.heappush(self.heap[side], (priority, new_cost, ns))
                    self._join(ns)
        return self.best_plan

    def partial_plan(self):
        state = self.partial[2]
        if state is None:
            return None
        actions = self._path(state, 0)
        checked = self.board.verify(actions, require_solved=False)
        if checked is None:
            return None
        initial_goals = (self.board.initial_boxes & self.board.goals).bit_count()
        gain = 10000 * ((checked[1] & self.board.goals).bit_count() - initial_goals)
        return actions if gain > checked[0] else None


def extract_actions(text):
    """Accept a marked direction sequence or a strict JSON actions array."""
    if not isinstance(text, str) or len(text) > 100000:
        return []
    aliases = dict(zip("UDLR", NAMES))
    aliases.update({a.upper(): a for a in NAMES})
    source = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = parsed.get("actions")
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            source = " ".join(parsed)
    except (ValueError, TypeError):
        pass
    marked = re.findall(r"(?im)^\s*ACTIONS\s*:\s*([^\r\n]+)", text)
    if source is None and marked:
        source = marked[-1]
    if source is None:
        source = text.strip() if len(text.strip()) < 2000 else ""
    words = re.findall(r"[A-Za-z]+", source.upper())
    if len(words) == 1 and re.fullmatch(r"[UDLR]+", words[0]):
        words = list(words[0])
    if not words or any(w not in aliases for w in words):
        return []
    return [aliases[w] for w in words[:2000]]


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def remaining(self):
        try:
            return max(0.0, float(self.ctx.budget.remaining().get("wall_s", 0.0)))
        except Exception:
            return 0.0

    def log(self, event, **fields):
        try:
            self.ctx.log(event, **fields)
        except Exception:
            pass

    def execute(self, board, obs, actions):
        ctx = self.ctx
        boxes, player = board.state(obs)
        for action in actions:
            if ctx.done or obs.get("done"):
                return obs, True
            if self.remaining() < 0.05:
                return obs, False
            expected = board.advance(boxes, player, NAMES.index(action))
            valid = obs.get("valid_actions")
            if expected is None or (isinstance(valid, list) and action not in valid):
                self.log("witness_invalidated", action=action, reason="pre_action")
                return obs, False
            try:
                nxt = as_observation(ctx.act(action))
                if nxt is None:
                    nxt = as_observation(ctx.observe())
                if nxt is None:
                    return obs, False
                actual = board.state(nxt)
            except Exception as exc:
                self.log("action_stopped", error=type(exc).__name__)
                return obs, False
            if actual != expected[:2]:
                self.log("witness_invalidated", action=action, reason="transition_mismatch")
                return nxt, False
            obs, boxes, player = nxt, actual[0], actual[1]
        return obs, boxes == board.goals

    def run_episode(self, game):
        ctx = self.ctx
        try:
            obs = as_observation(game.observe())
        except Exception:
            return
        # At most two reconstructions after a real transition mismatch.  There
        # is no blind walking fallback and no requirement to exhaust the cap.
        for attempt in range(3):
            if obs is None or obs.get("done") or ctx.done:
                return
            try:
                board = Board(obs)
                if board.initial_boxes == board.goals:
                    return
                search = BidirectionalSearch(board)
                stop = lambda: ctx.done or self.remaining() < 5.0
                search.run(min(90.0, max(0.0, self.remaining() - 8.0)), stop)
                self.log("bidirectional_search", expanded=search.expanded,
                         labels=[len(x) for x in search.records],
                         best_cost=None if search.best_plan is None else search.best_cost,
                         optimal=search.proved, saturated=search.saturated)
                # A model is a witness proposer, never the live action source.
                # Its entire answer must replay legally and end solved.
                failures = []
                session = None
                for call in range(2):
                    if search.best_plan is not None or search.proved or ctx.done:
                        break
                    if self.remaining() < 380.0:
                        break
                    prompt = ("Solve this Sokoban position completely. Plan box-goal order and "
                              "player access; boxes cannot be pulled. Return one line ACTIONS: "
                              "followed by a sequence of UDLR letters (up/down/left/right). "
                              "No prose. The controller verifies the entire solution before "
                              "executing it. A partial or illegal sequence is rejected. "
                              "All coordinates are zero-based. Current JSON:\n" +
                              json.dumps(obs, separators=(",", ":")) +
                              ("\nPrevious proposal rejected: " + "; ".join(failures) if failures else ""))
                    try:
                        result = ctx.think(prompt, session_id=session, timeout_s=360.0)
                        session = getattr(result, "session_id", None)
                        actions = extract_actions(getattr(result, "text", ""))
                        if not search.offer(actions):
                            failures.append("not a complete legal solution")
                    except Exception as exc:
                        failures.append(type(exc).__name__)
                        if ctx.done:
                            return
                    if not search.saturated:
                        search.run(min(120.0, max(0.0, self.remaining() - 15.0)), stop)
                plan = search.best_plan or search.partial_plan()
                if not plan:
                    return
                if len(plan) > int(obs.get("action_cap", 2000)) - int(obs.get("actions", 0)):
                    return
                self.log("execute_witness", actions=len(plan), solved_witness=search.best_plan is not None,
                         cost=None if search.best_plan is None else search.best_cost)
                obs, solved = self.execute(board, obs, plan)
                if solved or search.best_plan is None:
                    return
            except Exception as exc:
                self.log("planner_stopped", error=type(exc).__name__, message=str(exc)[:200])
                return


def build(ctx):
    return Harness(ctx)
