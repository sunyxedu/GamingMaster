"""Candidate 0011: cycle controller with a bounded endgame food rescue.

The reference board is an even, obstacle-free rectangle.  The normal tick path
keeps candidate 0008's validated Hamiltonian-cycle invariant and does only a
bounded local check.  The one deliberate exception is a short, exact food
route in the final part of the fixed tick budget: this tests whether the
parent's conservative cycle-order filter leaves reachable food uncollected at
the time limit.  The rescue is bounded and never waits for a model.
"""
from __future__ import annotations

from collections import deque
from typing import Any


DIRECTIONS = ("UP", "RIGHT", "DOWN", "LEFT")
VEC = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _point(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _body(obs: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(obs, dict):
        return ()
    raw = obs.get("snake")
    if not isinstance(raw, (list, tuple)):
        return ()
    result = []
    for value in raw:
        cell = _point(value)
        if cell is None:
            return ()
        result.append(cell)
    return tuple(result)


def _food(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict) or obs.get("food") is None:
        return None
    return _point(obs.get("food"))


def _dims(obs: Any) -> tuple[int, int] | None:
    if not isinstance(obs, dict):
        return None
    try:
        rows, cols = int(obs.get("rows")), int(obs.get("cols"))
    except (TypeError, ValueError, OverflowError):
        return None
    if rows < 2 or cols < 2:
        return None
    return rows, cols


def _obs_int(obs: Any, key: str, default: int | None = None) -> int | None:
    if not isinstance(obs, dict):
        return default
    try:
        return int(obs.get(key))
    except (TypeError, ValueError, OverflowError):
        return default


def _heading(obs: Any, body: tuple[tuple[int, int], ...]) -> str:
    if isinstance(obs, dict) and obs.get("direction") in VEC:
        return obs["direction"]
    if len(body) >= 2:
        dr = body[0][0] - body[1][0]
        dc = body[0][1] - body[1][1]
        for name, vector in VEC.items():
            if vector == (dr, dc):
                return name
    # This is only for a malformed stand-in observation.  The real adaptor
    # always exposes direction.
    return "RIGHT"


def _inside(cell: tuple[int, int], rows: int, cols: int) -> bool:
    return 0 <= cell[0] < rows and 0 <= cell[1] < cols


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _valid_cycle(cycle: tuple[tuple[int, int], ...], rows: int, cols: int) -> bool:
    if len(cycle) != rows * cols or len(set(cycle)) != len(cycle):
        return False
    if any(not _inside(cell, rows, cols) for cell in cycle):
        return False
    return all(_adjacent(cycle[i], cycle[(i + 1) % len(cycle)])
               for i in range(len(cycle)))


def _even_row_cycle(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """Build a cycle when ``rows`` is even, for either parity of ``cols``.

    The top row is traversed first.  Interior rows snake through columns 1..,
    and the unused first-column cells close the cycle back to (0, 0).  This
    construction is useful for both 12x12 and even-by-odd rectangles.
    """
    if rows < 2 or cols < 2 or rows % 2:
        return None
    out = [(0, col) for col in range(cols)]
    for row in range(1, rows):
        columns = (range(cols - 1, 0, -1) if row % 2
                   else range(1, cols))
        out.extend((row, col) for col in columns)
    out.append((rows - 1, 0))
    out.extend((row, 0) for row in range(rows - 2, 0, -1))
    result = tuple(out)
    return result if _valid_cycle(result, rows, cols) else None


def _cycle_variants(rows: int, cols: int) -> list[tuple[tuple[int, int], ...]]:
    """Return validated cycles and simple reflections for the board."""
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _even_row_cycle(rows, cols)
    if direct is not None:
        bases.append(direct)
    if cols % 2 == 0:
        transposed = _even_row_cycle(cols, rows)
        if transposed is not None:
            bases.append(tuple((col, row) for row, col in transposed))

    result: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        # Reflections preserve adjacency and produce useful entry directions
        # for initial openings that are not aligned with the first base.
        for flip_rows in (False, True):
            for flip_cols in (False, True):
                transformed = tuple(
                    (rows - 1 - row if flip_rows else row,
                     cols - 1 - col if flip_cols else col)
                    for row, col in base
                )
                for oriented in (transformed, tuple(reversed(transformed))):
                    if oriented in seen:
                        continue
                    if _valid_cycle(oriented, rows, cols):
                        seen.add(oriented)
                        result.append(oriented)
    return result


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    delta = b[0] - a[0], b[1] - a[1]
    for name, vector in VEC.items():
        if vector == delta:
            return name
    return None


def _order_gaps(body: tuple[tuple[int, int], ...],
                cycle: tuple[tuple[int, int], ...]) -> list[int] | None:
    """Check head-first body order on a cycle.

    Gaps are allowed: a previous safe controller may have taken shortcuts.
    The only required invariant is that every older segment is behind the head
    in one traversal and that the occupied span does not wrap twice.
    """
    if not body or not cycle:
        return None
    index = {cell: number for number, cell in enumerate(cycle)}
    if any(cell not in index for cell in body) or len(set(body)) != len(body):
        return None
    size = len(cycle)
    gaps = []
    for i in range(1, len(body)):
        gap = (index[body[i - 1]] - index[body[i]]) % size
        if gap <= 0:
            return None
        gaps.append(gap)
    if sum(gaps) >= size:
        return None
    return gaps


def _next_cycle_action(
    body: tuple[tuple[int, int], ...],
    heading: str,
    cycle: tuple[tuple[int, int], ...],
) -> tuple[str, list[int]] | None:
    gaps = _order_gaps(body, cycle)
    if gaps is None:
        return None
    index = {cell: number for number, cell in enumerate(cycle)}
    next_cell = cycle[(index[body[0]] + 1) % len(cycle)]
    action = _direction_between(body[0], next_cell)
    if action is None or action == OPPOSITE.get(heading):
        return None
    return action, gaps


def _cycle_action(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    cycle: tuple[tuple[int, int], ...],
) -> str | None:
    """Choose a safe cycle-preserving shortcut toward the visible food.

    A grid neighbor can skip many cells in the cycle.  We accept that shortcut
    only when simulating the exact transition leaves the whole body in
    head-to-tail cycle order.  The next call repeats this proof from the fresh
    observation, so shortcuts never become an unchecked multi-tick plan.
    """
    if _order_gaps(body, cycle) is None:
        return None
    index = {cell: number for number, cell in enumerate(cycle)}
    size = len(cycle)
    current_index = index[body[0]]
    strict_next = cycle[(current_index + 1) % size]
    candidates = []
    for action in DIRECTIONS:
        if action == OPPOSITE.get(heading):
            continue
        result = _transition(body, heading, action, food, rows, cols)
        if result is None:
            continue
        new_body, new_heading, ate = result
        if _order_gaps(new_body, cycle) is None:
            continue
        if ate:
            food_distance = 0
        elif food is not None and food in index:
            food_distance = (index[food] - index[new_body[0]]) % size
        else:
            food_distance = 0
        # Eating wins; otherwise minimize forward cycle distance to food.
        # Staying on the exact cycle edge is the deterministic tie-breaker.
        rank = (
            int(ate),
            -food_distance,
            int(new_body[0] == strict_next),
            int(new_heading == heading),
        )
        candidates.append((rank, action))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _transition(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """One exact public-rule transition for the non-cycle fallback."""
    if not body or request not in VEC or heading not in VEC:
        return None
    accepted = heading if request == OPPOSITE.get(heading) else request
    dr, dc = VEC[accepted]
    new_head = body[0][0] + dr, body[0][1] + dc
    if not _inside(new_head, rows, cols):
        return None
    ate = food is not None and new_head == food
    occupied = set(body if ate else body[:-1])
    if new_head in occupied:
        return None
    return (new_head,) + (body if ate else body[:-1]), accepted, ate


def _area(body: tuple[tuple[int, int], ...], rows: int, cols: int) -> int:
    if not body:
        return 0
    blocked = set(body[1:])
    start = body[0]
    queue = deque([start])
    seen = {start}
    while queue:
        row, col = queue.popleft()
        for dr, dc in VEC.values():
            cell = row + dr, col + dc
            if (_inside(cell, rows, cols) and cell not in blocked
                    and cell not in seen):
                seen.add(cell)
                queue.append(cell)
    return len(seen)


def _short_food_route(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    max_depth: int,
    max_states: int,
) -> list[str] | None:
    """Find a short, exactly simulated route to food without cycle filtering.

    This is intentionally an endgame-only search.  Each queue state contains
    the complete moving body, so tail vacating and the special growth move are
    handled by ``_transition`` rather than by a static occupancy guess.  A
    reverse request is omitted because the real adaptor ignores it instead of
    changing heading.  The caller decides whether relaxing the cycle invariant
    is worth the finite remaining horizon.
    """
    if (not body or food is None or max_depth <= 0 or max_states <= 0
            or food in body):
        return None
    queue = deque([(body, heading, ())])
    seen = {(body, heading)}
    expanded = 0
    while queue and expanded < max_states:
        current_body, current_heading, path = queue.popleft()
        expanded += 1
        if len(path) >= max_depth:
            continue
        for action in DIRECTIONS:
            if action == OPPOSITE.get(current_heading):
                continue
            result = _transition(
                current_body, current_heading, action, food, rows, cols
            )
            if result is None:
                continue
            new_body, new_heading, ate = result
            new_path = path + (action,)
            if ate:
                return list(new_path)
            remaining = max_depth - len(new_path)
            distance = (abs(new_body[0][0] - food[0])
                        + abs(new_body[0][1] - food[1]))
            if distance > remaining:
                continue
            key = (new_body, new_heading)
            if key in seen:
                continue
            seen.add(key)
            queue.append((new_body, new_heading, new_path))
    return None


def _fallback_action(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    advertised: tuple[str, ...],
) -> str:
    """Cheap legal-area fallback for odd boards or malformed cycle entry."""
    candidates = []
    for action in DIRECTIONS:
        if action not in advertised or action == OPPOSITE.get(heading):
            continue
        result = _transition(body, heading, action, food, rows, cols)
        if result is None:
            continue
        new_body, new_heading, ate = result
        space = _area(new_body, rows, cols)
        distance = (0 if food is None else
                    abs(new_body[0][0] - food[0]) +
                    abs(new_body[0][1] - food[1]))
        # Food is valuable only after the exact transition check.  Space and
        # retaining the current heading break ties among safe moves.
        rank = (int(ate), space, -distance, int(new_heading == heading))
        candidates.append((rank, action))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    if heading in advertised:
        return heading
    return advertised[0] if advertised else "NOOP"


class _Controller:
    # The reference run has 4000 ticks.  Only the last 80 are allowed to
    # spend a bounded search and to trade the invariant for a guaranteed
    # attempt at a reachable food item.
    ENDGAME_WINDOW = 80
    ENDGAME_MAX_STATES = 1800
    ENDGAME_RETRY_TICKS = 8

    def __init__(self, ctx: Any, game: Any):
        self.ctx = ctx
        self.game = game
        self.cycles: list[tuple[tuple[int, int], ...]] = []
        self.cycle: tuple[tuple[int, int], ...] | None = None
        self.index: dict[tuple[int, int], int] = {}
        self.rows = 0
        self.cols = 0
        self.last_tick: int | None = None
        self.endgame_route: list[str] = []
        self.endgame_food: tuple[int, int] | None = None
        self.endgame_expected_body: tuple[tuple[int, int], ...] | None = None
        self.endgame_expected_heading: str | None = None
        self.endgame_last_probe_tick: int | None = None
        self.endgame_last_probe_food: tuple[int, int] | None = None

    def _set_dimensions(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        self.cycles = _cycle_variants(rows, cols)
        self.cycle = None
        self.index = {}

    def _clear_endgame_route(self) -> None:
        self.endgame_route = []
        self.endgame_food = None
        self.endgame_expected_body = None
        self.endgame_expected_heading = None
        self.endgame_last_probe_tick = None
        self.endgame_last_probe_food = None

    def _take_endgame_step(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int],
        rows: int,
        cols: int,
        advertised: tuple[str, ...],
    ) -> str | None:
        if not self.endgame_route:
            return None
        action = self.endgame_route[0]
        if action not in advertised or action == OPPOSITE.get(heading):
            self._clear_endgame_route()
            return None
        result = _transition(body, heading, action, food, rows, cols)
        if result is None:
            self._clear_endgame_route()
            return None
        new_body, new_heading, _ = result
        self.endgame_route.pop(0)
        self.endgame_expected_body = new_body
        self.endgame_expected_heading = new_heading
        return action

    def _endgame_action(
        self,
        obs: Any,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        rows: int,
        cols: int,
        advertised: tuple[str, ...],
    ) -> str | None:
        """Use one exact food route only in the finite endgame window.

        The route is accepted only when its first move would break the active
        cycle-order proof.  Thus the normal policy is unchanged; this branch
        specifically tests whether that proof is too conservative for a food
        item still reachable before ``max_ticks``.
        """
        tick = _obs_int(obs, "tick")
        max_ticks = _obs_int(obs, "max_ticks")
        if max_ticks is None:
            max_ticks = _obs_int(obs, "max_steps")
        if max_ticks is None:
            try:
                max_ticks = int(getattr(self.game, "max_ticks"))
            except (AttributeError, TypeError, ValueError, OverflowError):
                max_ticks = None
        if max_ticks is None:
            # The archive's reference observations omit this metadata despite
            # the public contract; retain the documented profile fallback.
            max_ticks = {
                (12, 12): 4000,
                (8, 8): 4000,
                (10, 14): 6000,
                (14, 10): 6000,
                (11, 11): 5000,
            }.get((rows, cols))
        if tick is None or max_ticks is None or food is None:
            return None
        remaining = max_ticks - tick
        if remaining <= 0 or remaining > self.ENDGAME_WINDOW:
            if self.endgame_route:
                self._clear_endgame_route()
            return None

        if self.endgame_food != food:
            self._clear_endgame_route()
            self.endgame_food = food

        if self.endgame_route:
            if (self.endgame_expected_body is not None
                    and (body != self.endgame_expected_body
                         or heading != self.endgame_expected_heading)):
                self._clear_endgame_route()
                self.endgame_food = food
            else:
                return self._take_endgame_step(
                    body, heading, food, rows, cols, advertised
                )

        should_probe = (
            self.endgame_last_probe_food != food
            or self.endgame_last_probe_tick is None
            or tick - self.endgame_last_probe_tick >= self.ENDGAME_RETRY_TICKS
        )
        if not should_probe:
            return None
        self.endgame_last_probe_tick = tick
        self.endgame_last_probe_food = food
        path = _short_food_route(
            body, heading, food, rows, cols,
            min(remaining, self.ENDGAME_WINDOW), self.ENDGAME_MAX_STATES,
        )
        if not path:
            return None

        first = _transition(body, heading, path[0], food, rows, cols)
        if first is None:
            return None
        first_body, _, _ = first
        # A route that still passes the existing proof is not the experiment;
        # candidate 0008 already handles it and remains the safer tie-breaker.
        if self.cycle is not None and _order_gaps(first_body, self.cycle) is not None:
            return None

        self.endgame_route = path
        self.endgame_food = food
        self.endgame_expected_body = None
        self.endgame_expected_heading = None
        try:
            self.ctx.log(
                "endgame_food_rescue",
                tick=tick,
                route_length=len(path),
                remaining=remaining,
                cycle_relaxed=True,
            )
        except Exception:
            pass
        return self._take_endgame_step(
            body, heading, food, rows, cols, advertised
        )

    def _select_cycle(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
    ) -> str | None:
        options = []
        for number, cycle in enumerate(self.cycles):
            action = _cycle_action(
                body, heading, food, self.rows, self.cols, cycle
            )
            if action is None:
                continue
            # Continuing straight is a small but useful tie-breaker.  It does
            # not weaken the cycle invariant or allow an unsafe shortcut.
            rank = (int(action == heading), -number)
            options.append((rank, action, cycle))
        if not options:
            return None
        options.sort(key=lambda item: item[0], reverse=True)
        _, action, cycle = options[0]
        self.cycle = cycle
        self.index = {cell: number for number, cell in enumerate(cycle)}
        try:
            tick = int(self.last_tick if self.last_tick is not None else 0)
        except (TypeError, ValueError):
            tick = 0
        try:
            self.ctx.log("cycle_selected", tick=tick, rows=self.rows,
                         cols=self.cols, action=action,
                         cycle_count=len(self.cycles), shortcuts=True)
        except Exception:
            # Logging is diagnostic; it must never be part of the control path.
            pass
        return action

    def choose(self, obs: Any, advertised: tuple[str, ...]) -> str:
        dimensions = _dims(obs)
        body = _body(obs)
        if dimensions is None:
            return advertised[0] if advertised else "NOOP"
        rows, cols = dimensions
        self._set_dimensions(rows, cols)
        heading = _heading(obs, body)
        self.last_tick = _obs_int(obs, "tick", 0)

        base_action: str | None = None
        if body:
            food = _food(obs)
            if self.cycle is not None:
                action = _cycle_action(
                    body, heading, food, rows, cols, self.cycle
                )
                if action is not None and action in advertised:
                    base_action = action
                else:
                    # An observation that no longer satisfies the invariant
                    # is treated as a new entry problem, not trusted blindly.
                    self.cycle = None
                    self.index = {}
            if base_action is None:
                selected = self._select_cycle(body, heading, food)
                if selected is not None and selected in advertised:
                    base_action = selected
            if base_action is None:
                action = _fallback_action(
                    body, heading, food, rows, cols, advertised
                )
                if action in advertised:
                    base_action = action

            rescue = self._endgame_action(
                obs, body, heading, food, rows, cols, advertised
            )
            if rescue in advertised:
                return rescue
            if base_action in advertised:
                return base_action

        if heading in advertised:
            return heading
        return advertised[0] if advertised else "NOOP"


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

    def run_episode(self, game: Any) -> None:
        # This candidate is intentionally model-free.  Its claim is that a
        # verified reference-board invariant beats waiting for either model.
        tick_s = getattr(game, "tick_s", None)
        if not tick_s:
            return
        spec = getattr(game, "action_spec", None)
        names = tuple(name for name in (getattr(spec, "names", ()) or ())
                      if isinstance(name, str))
        advertised = tuple(name for name in DIRECTIONS if name in names)
        if not advertised:
            noop = getattr(spec, "noop", None)
            advertised = (noop,) if isinstance(noop, str) and noop in names else names
        if not advertised:
            return

        controller = _Controller(self.ctx, game)
        while not self.ctx.done and not getattr(game, "done", False):
            obs = game.observe()
            action = controller.choose(obs, advertised)
            if action not in names:
                action = getattr(spec, "noop", None)
                if action not in names:
                    action = advertised[0]
            try:
                self.ctx.act(action)
            except Exception:
                # A close between the loop condition and the post is a normal
                # episode end.  Do not hide an actual game error while live.
                if self.ctx.done or getattr(game, "done", False):
                    return
                raise
            if self.ctx.done or getattr(game, "done", False):
                return
            self.ctx.wait_for_tick()


def build(ctx: Any) -> Harness:
    return Harness(ctx)
