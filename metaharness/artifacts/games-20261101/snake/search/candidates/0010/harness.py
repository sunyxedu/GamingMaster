"""Candidate 0010: a verified portfolio of Hamiltonian Snake cycles.

The previous cycle controller was safe, but one fixed cyclic ordering could put a
new food behind the head and force an almost complete lap.  This controller
keeps the same local proof while considering a small, independently validated
family of cycles.  It never asks a model to stand in the clock path.
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
    if rows < 2 or cols < 2 or rows * cols > 4096:
        return None
    return rows, cols


def _heading(obs: Any, body: tuple[tuple[int, int], ...]) -> str:
    if isinstance(obs, dict) and obs.get("direction") in VEC:
        return obs["direction"]
    if len(body) >= 2:
        delta = body[0][0] - body[1][0], body[0][1] - body[1][1]
        for name, vector in VEC.items():
            if vector == delta:
                return name
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



def _add_edge(
    adjacency: dict[tuple[int, int], set[tuple[int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> None:
    adjacency.setdefault(a, set()).add(b)
    adjacency.setdefault(b, set()).add(a)


def _remove_edge(
    adjacency: dict[tuple[int, int], set[tuple[int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> None:
    if a in adjacency:
        adjacency[a].discard(b)
    if b in adjacency:
        adjacency[b].discard(a)


def _walk_degree_two(
    adjacency: dict[tuple[int, int], set[tuple[int, int]]],
    rows: int,
    cols: int,
) -> tuple[tuple[int, int], ...] | None:
    """Turn a validated 2-regular graph into one ordered cycle."""
    count = rows * cols
    if len(adjacency) != count or any(len(v) != 2 for v in adjacency.values()):
        return None
    start = (0, 0)
    if start not in adjacency:
        return None
    result = [start]
    previous: tuple[int, int] | None = None
    current = start
    while True:
        choices = sorted(adjacency[current])
        if previous is not None:
            choices = [cell for cell in choices if cell != previous]
        if not choices:
            return None
        following = choices[0]
        if following == start:
            break
        if following in result:
            return None
        result.append(following)
        previous, current = current, following
        if len(result) > count:
            return None
    cycle = tuple(result)
    return cycle if _valid_cycle(cycle, rows, cols) else None


def _even_row_cycle(rows: int, cols: int) -> tuple[tuple[int, int], ...] | None:
    """The simple boundary-spine cycle used by candidate 0008."""
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


def _strip_cycle(
    rows: int,
    cols: int,
    seams: tuple[int, ...],
) -> tuple[tuple[int, int], ...] | None:
    """Build a cycle by joining two-row perimeter cycles at variable seams.

    A seam joins the two rows of adjacent two-row strips through columns c and
    c+1.  Every resulting graph is checked as a single Hamiltonian cycle; the
    seam pattern is only a generator, never an unchecked safety assumption.
    """
    if rows < 2 or cols < 2 or rows % 2:
        return None
    strip_count = rows // 2
    if len(seams) != max(0, strip_count - 1):
        return None
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for row in range(0, rows, 2):
        perimeter = (
            [(row, col) for col in range(cols)]
            + [(row + 1, col) for col in range(cols - 1, -1, -1)]
        )
        for i, cell in enumerate(perimeter):
            _add_edge(adjacency, cell, perimeter[(i + 1) % len(perimeter)])

    for strip in range(strip_count - 1):
        column = seams[strip]
        if column < 0 or column >= cols - 1:
            return None
        upper = 2 * strip + 1
        lower = upper + 1
        _remove_edge(adjacency, (upper, column), (upper, column + 1))
        _remove_edge(adjacency, (lower, column), (lower, column + 1))
        _add_edge(adjacency, (upper, column), (lower, column))
        _add_edge(adjacency, (upper, column + 1), (lower, column + 1))
    return _walk_degree_two(adjacency, rows, cols)


def _seam_patterns(rows: int, cols: int) -> list[tuple[int, ...]]:
    """Return a small deterministic, diverse set of valid seam parameters."""
    seam_count = max(0, rows // 2 - 1)
    width = cols - 1
    if seam_count == 0:
        return [()]
    patterns: list[tuple[int, ...]] = []

    def add(values: list[int] | tuple[int, ...]) -> None:
        pattern = tuple(int(x) % width for x in values)
        if pattern not in patterns:
            patterns.append(pattern)

    add([0] * seam_count)
    add([width - 1] * seam_count)
    add([width // 2] * seam_count)
    # Move one join at a time.  These cycles differ only in a small region,
    # which makes a later order-compatible switch possible.
    for seam in range(seam_count):
        values = [0] * seam_count
        values[seam] = (2 + 3 * seam) % width
        add(values)
    # Several coherent zig-zag patterns alter every strip join without making
    # the portfolio large enough to threaten the 100 ms control cadence.
    for phase in range(min(width, 6)):
        add([(phase + 2 * i) % width for i in range(seam_count)])
        add([(phase + width - 1 - 2 * i) % width
             for i in range(seam_count)])
    return patterns[:12]


def _cycle_variants(rows: int, cols: int) -> list[tuple[tuple[int, int], ...]]:
    """Generate a bounded portfolio of validated cycles and orientations."""
    bases: list[tuple[tuple[int, int], ...]] = []
    direct = _even_row_cycle(rows, cols)
    if direct is not None:
        bases.append(direct)
    # Keep the transposed 0008 family near the front of the portfolio.
    if cols % 2 == 0:
        transposed = _even_row_cycle(cols, rows)
        if transposed is not None:
            bases.append(tuple((col, row) for row, col in transposed))

    direct_patterns = _seam_patterns(rows, cols) if rows % 2 == 0 else []
    for pattern in direct_patterns:
        candidate = _strip_cycle(rows, cols, pattern)
        if candidate is not None:
            bases.append(candidate)
    if cols % 2 == 0:
        for pattern in _seam_patterns(cols, rows):
            candidate = _strip_cycle(cols, rows, pattern)
            if candidate is not None:
                bases.append(tuple((col, row) for row, col in candidate))

    result: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    # 64 oriented cycles is enough to provide structural diversity while
    # keeping the per-tick order checks bounded.
    for base in bases:
        for flip_rows in (False, True):
            for flip_cols in (False, True):
                transformed = tuple(
                    (rows - 1 - row if flip_rows else row,
                     cols - 1 - col if flip_cols else col)
                    for row, col in base
                )
                for oriented in (transformed, tuple(reversed(transformed))):
                    if oriented in seen or not _valid_cycle(oriented, rows, cols):
                        continue
                    seen.add(oriented)
                    result.append(oriented)
                    if len(result) >= 64:
                        return result
    return result


def _direction_between(
    a: tuple[int, int], b: tuple[int, int]
) -> str | None:
    delta = b[0] - a[0], b[1] - a[1]
    for name, vector in VEC.items():
        if vector == delta:
            return name
    return None


def _order_gaps(
    body: tuple[tuple[int, int], ...],
    cycle: tuple[tuple[int, int], ...],
    index: dict[tuple[int, int], int] | None = None,
) -> list[int] | None:
    """Check that head-first body cells occupy one forward cyclic interval.

    Occupied cycle indices may have gaps.  What is forbidden is a repeated
    index or a wrap that makes the occupied span cover the cycle twice.
    """
    if not body or not cycle:
        return None
    if index is None:
        index = {cell: number for number, cell in enumerate(cycle)}
    if any(cell not in index for cell in body) or len(set(body)) != len(body):
        return None
    size = len(cycle)
    gaps: list[int] = []
    for i in range(1, len(body)):
        gap = (index[body[i - 1]] - index[body[i]]) % size
        if gap <= 0:
            return None
        gaps.append(gap)
    if sum(gaps) >= size:
        return None
    return gaps


def _transition(
    body: tuple[tuple[int, int], ...],
    heading: str,
    request: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
) -> tuple[tuple[tuple[int, int], ...], str, bool] | None:
    """Apply one exact public-rule transition."""
    if not body or heading not in VEC or request not in VEC:
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
    queue = deque([body[0]])
    seen = {body[0]}
    while queue:
        row, col = queue.popleft()
        for dr, dc in VEC.values():
            cell = row + dr, col + dc
            if (_inside(cell, rows, cols) and cell not in blocked
                    and cell not in seen):
                seen.add(cell)
                queue.append(cell)
    return len(seen)


def _fallback_action(
    body: tuple[tuple[int, int], ...],
    heading: str,
    food: tuple[int, int] | None,
    rows: int,
    cols: int,
    advertised: tuple[str, ...],
) -> str:
    """Exact one-step recovery when no cycle is compatible."""
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
                    abs(new_body[0][0] - food[0])
                    + abs(new_body[0][1] - food[1]))
        candidates.append(((int(ate), space, -distance,
                           int(new_heading == heading)), action))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    if heading in advertised:
        return heading
    return advertised[0] if advertised else "NOOP"


class _Controller:
    def __init__(self, ctx: Any, game: Any):
        self.ctx = ctx
        self.game = game
        self.rows = 0
        self.cols = 0
        self.cycles: list[tuple[tuple[int, int], ...]] = []
        self.indexes: list[dict[tuple[int, int], int]] = []
        self.active: tuple[tuple[int, int], ...] | None = None
        self.last_tick: int | None = None
        self.logged_dimensions: tuple[int, int] | None = None

    def _set_dimensions(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self.rows, self.cols):
            return
        self.rows, self.cols = rows, cols
        self.cycles = _cycle_variants(rows, cols)
        self.indexes = [{cell: number for number, cell in enumerate(cycle)}
                        for cycle in self.cycles]
        self.active = None
        self.logged_dimensions = None

    def _cycle_options(
        self,
        body: tuple[tuple[int, int], ...],
        heading: str,
        food: tuple[int, int] | None,
        advertised: tuple[str, ...],
    ) -> list[tuple[tuple[int, ...], str, tuple[tuple[int, int], ...]]]:
        """Return all bounded, exact-safe one-step portfolio choices."""
        options = []
        for number, (cycle, index) in enumerate(zip(self.cycles, self.indexes)):
            if _order_gaps(body, cycle, index) is None:
                continue
            head_index = index[body[0]]
            strict_next = cycle[(head_index + 1) % len(cycle)]
            for action in DIRECTIONS:
                if action not in advertised or action == OPPOSITE.get(heading):
                    continue
                result = _transition(body, heading, action, food,
                                     self.rows, self.cols)
                if result is None:
                    continue
                new_body, new_heading, ate = result
                if _order_gaps(new_body, cycle, index) is None:
                    continue
                if ate:
                    food_distance = 0
                elif food is None:
                    food_distance = 0
                else:
                    food_distance = (index[food] - index[new_body[0]]) % len(cycle)
                rank = (
                    int(ate),
                    -food_distance,
                    int(cycle is self.active),
                    int(new_body[0] == strict_next),
                    int(new_heading == heading),
                    -number,
                )
                options.append((rank, action, cycle))
        options.sort(key=lambda item: item[0], reverse=True)
        return options

    def choose(self, obs: Any, advertised: tuple[str, ...]) -> str:
        if not isinstance(obs, dict):
            return advertised[0] if advertised else "NOOP"
        dimensions = _dims(obs)
        body = _body(obs)
        if dimensions is None or not body:
            heading = _heading(obs, body)
            return heading if heading in advertised else (advertised[0] if advertised else "NOOP")
        rows, cols = dimensions
        self._set_dimensions(rows, cols)
        heading = _heading(obs, body)
        food = _food(obs)
        try:
            self.last_tick = int(obs.get("tick", 0))
        except (TypeError, ValueError, OverflowError):
            self.last_tick = 0

        if self.logged_dimensions is None:
            self.logged_dimensions = (rows, cols)
            try:
                self.ctx.log("cycle_portfolio", tick=self.last_tick,
                             rows=rows, cols=cols,
                             cycle_count=len(self.cycles),
                             seam_family=True)
            except Exception:
                pass

        options = self._cycle_options(body, heading, food, advertised)
        if options:
            _, action, cycle = options[0]
            if cycle is not self.active:
                try:
                    old_number = (self.cycles.index(self.active)
                                  if self.active in self.cycles else None)
                    new_number = self.cycles.index(cycle)
                    self.ctx.log("cycle_switch", tick=self.last_tick,
                                 old_cycle=old_number, new_cycle=new_number,
                                 action=action,
                                 food_distance="eaten" if food == body[0] else "ordered")
                except Exception:
                    pass
                self.active = cycle
            return action

        # Odd-by-odd boards and malformed/irregular states use a mechanical
        # one-step rule.  It is not presented as a Hamiltonian proof.
        action = _fallback_action(body, heading, food, rows, cols, advertised)
        if action in advertised:
            self.active = None
            return action
        return heading if heading in advertised else (advertised[0] if advertised else "NOOP")


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx

    def run_episode(self, game: Any) -> None:
        if not getattr(game, "tick_s", None):
            return
        spec = getattr(game, "action_spec", None)
        names = tuple(name for name in (getattr(spec, "names", ()) or ())
                      if isinstance(name, str))
        advertised = tuple(name for name in DIRECTIONS if name in names)
        if not advertised:
            noop = getattr(spec, "noop", None)
            advertised = ((noop,) if isinstance(noop, str) and noop in names
                          else names)
        if not advertised:
            return

        controller = _Controller(self.ctx, game)
        while not self.ctx.done and not getattr(game, "done", False):
            obs = game.observe()
            action = controller.choose(obs, advertised)
            if action not in names:
                noop = getattr(spec, "noop", None)
                action = noop if isinstance(noop, str) and noop in names else advertised[0]
            try:
                self.ctx.act(action)
            except Exception:
                if self.ctx.done or getattr(game, "done", False):
                    return
                raise
            if self.ctx.done or getattr(game, "done", False):
                return
            self.ctx.wait_for_tick()


def build(ctx: Any) -> Harness:
    return Harness(ctx)
