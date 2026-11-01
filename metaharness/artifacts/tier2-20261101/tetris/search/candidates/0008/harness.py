"""Candidate 0008: deterministic queue-aware beam controller for realtime Tetris.

The model boundary is intentionally advisory.  Exact placement generation and a
short queue beam search run in the harness; Jev, when available, may only choose
between already legal top-ranked plans.  The clock is never held for inference.
"""
from __future__ import annotations

from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice

WIDTH = 10
HEIGHT = 20
PIECES = ("I", "J", "L", "O", "S", "T", "Z")

# SRS spawn shapes in the coordinate system exposed by the adaptor: x grows
# right and y grows upward; active.x/active.y is the shape origin.
_BASE = {
    "J": ((0, 2), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 2), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 2), (2, 2), (0, 1), (1, 1)),
    "Z": ((0, 2), (1, 2), (1, 1), (2, 1)),
    "T": ((1, 2), (0, 1), (1, 1), (2, 1)),
    "I": ((0, 2), (1, 2), (2, 2), (3, 2)),
    "O": ((1, 2), (2, 2), (1, 1), (2, 1)),
}


def _make_shapes():
    out = {}
    for piece, cells in _BASE.items():
        states = []
        cur = tuple(cells)
        for _ in range(4):
            states.append(tuple(sorted(cur)))
            if piece == "O":
                continue
            pivot = (1.5, 1.5) if piece == "I" else (1.0, 1.0)
            nxt = []
            for x, y in cur:
                dx, dy = x - pivot[0], y - pivot[1]
                # clockwise in bottom-origin coordinates
                nxt.append((int(round(pivot[0] + dy)),
                            int(round(pivot[1] - dx))))
            cur = tuple(nxt)
        out[piece] = tuple(states)
    return out


SHAPES = _make_shapes()


def _grid(obs: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    rows = [[0] * WIDTH for _ in range(HEIGHT)]
    raw = obs.get("board")
    if isinstance(raw, list):
        for top, row in enumerate(raw[:HEIGHT]):
            if not isinstance(row, list):
                continue
            y = HEIGHT - 1 - top
            for x, val in enumerate(row[:WIDTH]):
                try:
                    rows[y][x] = 1 if int(val) else 0
                except (TypeError, ValueError):
                    rows[y][x] = 0
    return tuple(tuple(row) for row in rows)


def _heights(grid):
    result = []
    for x in range(WIDTH):
        h = 0
        for y in range(HEIGHT - 1, -1, -1):
            if grid[y][x]:
                h = y + 1
                break
        result.append(h)
    return result


def _features(grid):
    heights = _heights(grid)
    holes = 0
    hole_depth = 0
    for x, h in enumerate(heights):
        blocks_above = 0
        for y in range(h - 1, -1, -1):
            if grid[y][x]:
                blocks_above += 1
            elif blocks_above:
                holes += 1
                hole_depth += blocks_above
    bump = sum(abs(heights[x] - heights[x + 1]) for x in range(WIDTH - 1))
    wells = 0
    for x, h in enumerate(heights):
        left = heights[x - 1] if x else HEIGHT
        right = heights[x + 1] if x + 1 < WIDTH else HEIGHT
        wells += max(0, min(left, right) - h)
    row_transitions = 0
    for y in range(HEIGHT):
        previous = 1
        for x in range(WIDTH):
            current = 1 if grid[y][x] else 0
            if current != previous:
                row_transitions += 1
            previous = current
        if previous == 0:
            row_transitions += 1
    return {
        "heights": heights,
        "aggregate": sum(heights),
        "max_height": max(heights) if heights else 0,
        "holes": holes,
        "hole_depth": hole_depth,
        "bump": bump,
        "wells": wells,
        "row_transitions": row_transitions,
    }


def _drop(grid, piece: str, orientation: int, x0: int):
    shapes = SHAPES.get(piece)
    if not shapes:
        return None
    shape = shapes[orientation % len(shapes)]
    if any(x0 + dx < 0 or x0 + dx >= WIDTH for dx, _ in shape):
        return None

    def legal(y0):
        for dx, dy in shape:
            x, y = x0 + dx, y0 + dy
            if x < 0 or x >= WIDTH or y < 0:
                return False
            if y < HEIGHT and grid[y][x]:
                return False
        return True

    y0 = HEIGHT + 4
    while legal(y0 - 1):
        y0 -= 1
    if not legal(y0):
        return None
    cells = tuple((x0 + dx, y0 + dy) for dx, dy in shape)
    if any(y >= HEIGHT for _, y in cells):
        return None
    placed = [list(row) for row in grid]
    for x, y in cells:
        placed[y][x] = 1
    remaining = [row for row in placed if not all(row)]
    lines = HEIGHT - len(remaining)
    remaining.extend([[0] * WIDTH for _ in range(lines)])
    result = tuple(tuple(row) for row in remaining)
    facts = _features(result)
    facts["lines"] = lines
    facts["landing_y"] = max(y for _, y in cells)
    facts["cells"] = cells
    return result, facts


# Guideline SRS kick tests, expressed in the same bottom-origin coordinates as
# the public active cells.  O has no kick table.
_KICKS = {
    "JLSTZ": {
        (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
        (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
        (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
        (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
        (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    },
    "I": {
        (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
        (1, 0): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
        (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
        (2, 1): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
        (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
        (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
        (3, 0): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
        (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    },
}


def _fits_at(grid, piece, orientation, x0, y0):
    shape = SHAPES.get(piece, ())
    if not shape:
        return False
    for dx, dy in shape[orientation % 4]:
        x, y = x0 + dx, y0 + dy
        if x < 0 or x >= WIDTH or y < 0:
            return False
        if y < HEIGHT and grid[y][x]:
            return False
    return True


def _rotate_state(grid, piece, orientation, x0, y0, clockwise):
    if piece == "O":
        return (orientation, x0, y0)
    new_orientation = (orientation + (1 if clockwise else -1)) % 4
    table = _KICKS["I" if piece == "I" else "JLSTZ"]
    tests = table.get((orientation % 4, new_orientation), ((0, 0),))
    for dx, dy in tests:
        nx, ny = x0 + dx, y0 + dy
        if _fits_at(grid, piece, new_orientation, nx, ny):
            return (new_orientation, nx, ny)
    return None


def _path_action(obs, target):
    """Return one SRS-legal steering action toward a searched destination."""
    active = obs.get("active") or {}
    piece = str(active.get("type", ""))
    if piece not in SHAPES:
        return None
    try:
        start = (int(active.get("orientation", 0)) % 4,
                 int(active.get("x", 3)), int(active.get("y", 17)))
    except (TypeError, ValueError):
        start = (0, 3, 17)
    try:
        desired = int(target.orientation) % 4
        desired_x = int(target.x)
    except (TypeError, ValueError):
        return None
    grid = _grid(obs)
    if start[0] == desired and start[1] == desired_x:
        return "hard_drop"

    # Search only steering states. Gravity is re-read after every action, so a
    # short path remains valid even as y changes between ticks.
    queue = [(start, [])]
    visited = {start}
    while queue and len(visited) < 180:
        state, path = queue.pop(0)
        ori, x0, y0 = state
        if ori == desired and x0 == desired_x:
            return path[0] if path else "hard_drop"
        neighbours = []
        if _fits_at(grid, piece, ori, x0 - 1, y0):
            neighbours.append(("left", (ori, x0 - 1, y0)))
        if _fits_at(grid, piece, ori, x0 + 1, y0):
            neighbours.append(("right", (ori, x0 + 1, y0)))
        cw = _rotate_state(grid, piece, ori, x0, y0, True)
        if cw is not None:
            neighbours.append(("rotate_cw", cw))
        ccw = _rotate_state(grid, piece, ori, x0, y0, False)
        if ccw is not None:
            neighbours.append(("rotate_ccw", ccw))
        for action, nxt in neighbours:
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [action]))
    # A target can become unreachable when gravity has moved against a high
    # stack. Hard drop is always a valid escape; do not repeatedly post a
    # blocked rotation or translation and inflate illegal-action bookkeeping.
    return "hard_drop"


def _board_value(facts, lines=0, depth=0):
    # Holes and the tallest column are lethal much sooner than a rough surface.
    # A clear receives approximately its level-1 line-clear value; this is also
    # useful at later levels because the same action improves survival.
    value = (
        108.0 * lines
        - 7.5 * facts["holes"]
        - 0.75 * facts["hole_depth"]
        - 0.55 * facts["aggregate"]
        - 1.35 * facts["max_height"]
        - 0.42 * facts["bump"]
        - 0.045 * facts["row_transitions"]
        - 0.06 * facts["wells"]
    )
    if lines == 4:
        value += 80.0
    # A high stack should dominate small drop-point opportunities.
    if facts["max_height"] >= 17:
        value -= 20.0 * (facts["max_height"] - 16)
    return value - 0.015 * depth


class _Plan:
    __slots__ = ("source", "piece", "orientation", "x", "grid", "facts",
                 "lines", "value", "instance")

    def __init__(self, source, piece, orientation, x, grid, facts, value,
                 instance=0):
        self.source = source
        self.piece = piece
        self.orientation = orientation
        self.x = x
        self.grid = grid
        self.facts = facts
        self.lines = int(facts.get("lines", 0))
        self.value = value
        self.instance = instance

    @property
    def use_hold(self):
        return self.source != "current"


def _placements(grid, piece, before, source="current", instance=0):
    result = []
    seen = set()
    for orientation, shape in enumerate(SHAPES.get(piece, ())):
        key_shape = tuple(shape)
        if key_shape in seen:
            continue
        seen.add(key_shape)
        min_x = min(dx for dx, _ in shape)
        max_x = max(dx for dx, _ in shape)
        for x0 in range(-min_x, WIDTH - max_x):
            dropped = _drop(grid, piece, orientation, x0)
            if dropped is None:
                continue
            after, facts = dropped
            lines = int(facts["lines"])
            # Penalize a hold only slightly: preserving a piece is often safer
            # than making a hole merely to avoid hold.
            hold_adjust = -0.8 if source != "current" else 0.0
            value = _board_value(facts, lines) + hold_adjust
            result.append(_Plan(source, piece, orientation, x0, after, facts,
                                value, instance))
    result.sort(key=lambda p: (p.value, p.lines, -p.facts["holes"],
                               -p.facts["max_height"], -p.facts["bump"]),
                reverse=True)
    return result


def _future_options(grid, current, queue, hold, allow_hold):
    """Return (plan-source, piece-to-place, next-current, next-queue, hold)."""
    q = tuple(queue)
    if current in PIECES:
        yield "current", current, (q[0] if q else None), q[1:], hold
    if not allow_hold:
        return
    if hold in PIECES and q:
        # Existing hold swaps with the current piece.
        yield "hold", hold, q[0], q[1:], current
    elif hold is None and q:
        # Empty hold stores current and spawns the first preview.
        yield "queue", q[0], (q[1] if len(q) > 1 else None), q[2:], current


def _beam(obs, lookahead=4, width=48):
    grid = _grid(obs)
    active = obs.get("active") or {}
    current = str(active.get("type", ""))
    if current not in PIECES:
        return []
    queue = tuple(p for p in (obs.get("queue") or []) if p in PIECES)
    hold = obs.get("hold") if obs.get("hold") in PIECES else None
    allow_hold = bool(obs.get("can_hold", False))
    before = _features(grid)

    # Each beam entry carries the first placement separately; later placements
    # only shape its value.  We intentionally keep all source choices in the
    # beam so an empty-hold opener is not treated as a special script.
    beam = []
    for source, piece, nxt, nq, nh in _future_options(
            grid, current, queue, hold, allow_hold):
        for p in _placements(grid, piece, before, source=source):
            beam.append({"grid": p.grid, "current": nxt, "queue": nq,
                         "hold": nh, "allow_hold": True,
                         "value": p.value, "first": p})
    if not beam:
        return []
    beam.sort(key=lambda n: n["value"], reverse=True)
    beam = beam[:width]

    for _depth in range(1, max(1, lookahead)):
        expanded = []
        for node in beam:
            nxt = node["current"]
            if nxt not in PIECES:
                expanded.append(node)
                continue
            facts_before = _features(node["grid"])
            options = list(_future_options(node["grid"], nxt, node["queue"],
                                           node["hold"], True))
            for source, piece, nn, nq, nh in options:
                for p in _placements(node["grid"], piece, facts_before,
                                     source=source):
                    expanded.append({
                        "grid": p.grid, "current": nn, "queue": nq,
                        "hold": nh, "allow_hold": True,
                        "value": node["value"] + 0.86 ** _depth * p.value,
                        "first": node["first"],
                    })
        if not expanded:
            break
        expanded.sort(key=lambda n: n["value"], reverse=True)
        # Keep a little source/side diversity.  A pure rank beam can lose the
        # only side-well route when all early placements are tied.
        beam = expanded[:width]

    beam.sort(key=lambda n: n["value"], reverse=True)
    plans = []
    seen = set()
    for node in beam:
        p = node["first"]
        key = (p.source, p.piece, p.orientation, p.x)
        if key in seen:
            continue
        seen.add(key)
        p.value = node["value"]
        plans.append(p)
        if len(plans) >= 9:
            break
    plans.sort(key=lambda p: (p.value, p.lines, -p.facts["holes"],
                              -p.facts["max_height"]), reverse=True)
    return plans


def _bucket(n, cuts):
    for label, limit in cuts:
        if n <= limit:
            return label
    return cuts[-1][0]


def _plan_description(p, index):
    clear = {0: "no line", 1: "single", 2: "double", 3: "triple",
             4: "Tetris"}.get(p.lines, "line clear")
    return (
        ("use hold then " if p.use_hold else "place current ") +
        "piece at orientation %d and target origin x=%d; %s; resulting "
        "stack is %s high with %s holes, %s surface."
        % (p.orientation, p.x, clear,
           _bucket(p.facts["max_height"], (("low", 6), ("medium", 12),
                                            ("high", 16), ("critical", 99))),
           _bucket(p.facts["holes"], (("hole-free", 0), ("few", 3),
                                      ("several", 8), ("many", 99))),
           _bucket(p.facts["bump"], (("flat", 8), ("uneven", 18),
                                      ("rough", 99)))))


def _state(obs, plans, style="BEAM"):
    active = obs.get("active") or {}
    return {
        "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
        "policy": style,
        "piece_instance": obs.get("pieces", 0),
        "board": {
            "height": _bucket(int(obs.get("top_out_height", 0)),
                              (("low", 6), ("medium", 12), ("high", 16),
                               ("critical", 99))),
            "holes": _bucket(int(obs.get("holes", 0)),
                             (("hole-free", 0), ("few", 3), ("several", 8),
                              ("many", 99))),
            "surface": _bucket(int(obs.get("bumpiness", 0)),
                                (("flat", 8), ("uneven", 18), ("rough", 99))),
            "lines": int(obs.get("lines", 0)),
            "level": int(obs.get("level", 1)),
        },
        "active": {
            "type": active.get("type"),
            "orientation": active.get("orientation"),
            "x": active.get("x"),
            "y": active.get("y"),
        },
        "hold": obs.get("hold"),
        "can_hold": bool(obs.get("can_hold", False)),
        "queue": list(obs.get("queue") or [])[:5],
        "plans": {"C%d" % i: _plan_description(p, i)
                  for i, p in enumerate(plans)},
        "fallback": "C0" if plans else None,
    }


def _prob(answer, key):
    try:
        return float((getattr(answer, "probabilities", {}) or {}).get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _jev_pick(answers, plans):
    if not isinstance(answers, dict) or not plans:
        return None
    valid = {"C%d" % i: p for i, p in enumerate(plans)}
    a = answers.get("pick_a")
    b = answers.get("pick_b")
    av = getattr(a, "value", None)
    bv = getattr(b, "value", None)
    if av not in valid or bv != av:
        return None
    # The model is only a tie-breaker between legal, already searched plans.
    # Agreement plus a moderate probability prevents a diffuse answer from
    # displacing the deterministic beam winner.
    if _prob(a, av) < 0.42 or _prob(b, av) < 0.42:
        return None
    return valid[av]


def _safe(spec, name):
    names = set(getattr(spec, "names", ()) or ())
    if name in names:
        return name
    noop = getattr(spec, "noop", None)
    if noop in names:
        return noop
    return next(iter(names), "noop")


def _piece_key(obs):
    """Fields that stay stable while a falling piece is being steered."""
    active = obs.get("active") or {}
    return (obs.get("pieces"), active.get("type"), obs.get("hold"),
            bool(obs.get("can_hold", False)))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _wait_action(self, game, action):
        if self.ctx.done or getattr(game, "done", False):
            return False
        try:
            self.ctx.act(action)
            self.ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver, ModelUnavailable):
            return False
        except Exception:
            return False
        return not self.ctx.done and not getattr(game, "done", False)

    def _ask(self, state, plans):
        try:
            common = (
                "Choose one plan label from the supplied `plans`. These are legal "
                "placements generated by code and already include their board result. "
                "Prefer a line clear when it does not create holes or critical height; "
                "otherwise prefer the lowest, flattest, hole-free result. Do not invent "
                "actions or calculate numbers. Recheck the same choice independently."
            )
            criteria_a = {"C%d" % i: _plan_description(p, i)
                          for i, p in enumerate(plans)}
            criteria_b = dict(reversed(list(criteria_a.items())))
            questions = {
                "pick_a": Choice(common, criteria_a),
                "pick_b": Choice(common, criteria_b),
            }
            return self.ctx.decide(state, questions, timeout_s=0.42)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver,
                ModelUnavailable):
            return None
        except Exception:
            return None

    def _plan(self, obs):
        # Standard play has enough fall time for a three-ply beam.  At faster
        # gravity, shorten the search so computation cannot consume the lock
        # window; the same executor remains valid.
        try:
            gravity = float(obs.get("gravity_s_per_cell", 1.0))
        except (TypeError, ValueError):
            gravity = 1.0
        if gravity < 0.12:
            return _beam(obs, lookahead=2, width=5)
        return _beam(obs, lookahead=3, width=6)

    def run_episode(self, game):
        ctx = self.ctx
        spec = game.action_spec
        target = None
        plans = []
        target_piece_count = None
        target_active_type = None
        hold_attempted = False
        future = None
        future_key = None

        while not ctx.done and not getattr(game, "done", False):
            try:
                obs = game.observe()
            except Exception:
                return
            if obs.get("game_over"):
                return
            active = obs.get("active") or {}
            piece_count = obs.get("pieces", 0)
            active_type = active.get("type")

            # Replan on every newly locked piece and after a hold replacement.
            if (target is None or piece_count != target_piece_count or
                    (target_active_type is not None and
                     active_type != target_active_type and not hold_attempted)):
                plans = self._plan(obs)
                target = plans[0] if plans else None
                target_piece_count = piece_count
                target_active_type = active_type
                hold_attempted = False
                # Never accumulate advisory workers.  If the previous request
                # is still running, it is simply discarded when it completes;
                # the deterministic plan remains independent of it.
                if future is not None and future.done():
                    future = None
                    future_key = None
                if (future is None and plans and
                        getattr(ctx, "system1", None) is not None):
                    state = _state(obs, plans)
                    # One outstanding advisory call at a time.  It is never on
                    # the action path; the deterministic target is executable now.
                    try:
                        future = ctx.spawn(
                            lambda s=state, ps=plans[:]: self._ask(s, ps))
                        future_key = _piece_key(obs)
                    except Exception:
                        future = None
                        future_key = None

            # Harvest only a response for the exact same falling piece.  x and
            # orientation are allowed to change while steering; hold and a
            # lock are not.
            if future is not None and future.done():
                old = future
                old_key = future_key
                future = None
                future_key = None
                try:
                    answers = old.result()
                except Exception:
                    answers = None
                if old_key == _piece_key(obs):
                    chosen = _jev_pick(answers, plans)
                    if chosen is not None:
                        target = chosen
                        try:
                            ctx.log("beam_jev_choice",
                                    piece=piece_count,
                                    candidate="C%d" % plans.index(chosen))
                        except Exception:
                            pass

            if target is None:
                if not self._wait_action(game, _safe(spec, "hard_drop")):
                    return
                continue

            # A hold plan is explicit.  Do not issue it after can_hold changed,
            # and do not let a stale Jev answer make a second hold.
            if target.use_hold and not hold_attempted:
                if bool(obs.get("can_hold", False)):
                    hold_attempted = True
                    target_active_type = target.piece
                    if not self._wait_action(game, _safe(spec, "hold")):
                        return
                    continue
                # Holding is no longer legal; choose the best non-hold plan
                # from a fresh board instead.
                fresh = [p for p in self._plan(obs) if not p.use_hold]
                target = fresh[0] if fresh else None
                plans = fresh
                hold_attempted = False
                if target is None:
                    if not self._wait_action(game, _safe(spec, "hard_drop")):
                        return
                    continue

            # If hold produced a different active piece than expected, discard
            # the old geometry and immediately compute a current-piece plan.
            if active_type != target.piece:
                plans = self._plan(obs)
                target = plans[0] if plans else None
                target_piece_count = piece_count
                target_active_type = active_type
                hold_attempted = False
                # A prior advisory may still finish; its key will fail the
                # freshness check and it will be reclaimed on the next loop.
                if target is None:
                    if not self._wait_action(game, _safe(spec, "hard_drop")):
                        return
                    continue

            action = _path_action(obs, target) or "hard_drop"
            if not self._wait_action(game, _safe(spec, action)):
                return


def build(ctx):
    return Harness(ctx)
