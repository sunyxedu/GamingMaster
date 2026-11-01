"""Candidate 0004: search placements in code, then let Jev choose among them.

The game is a real-time game, so neither model is allowed to be the input
stream.  The controller keeps a short, verified placement plan and posts one
valid low-level input per tick.  Code owns the SRS-shaped placement search and
finesse; Jev only ranks a small, named shortlist; Astra is an asynchronous
high-level policy reviewer.
"""
from __future__ import annotations

import json
import re
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


WIDTH = 10
HEIGHT = 20
PIECES = ("I", "J", "L", "O", "S", "T", "Z")

# These are the standard SRS spawn-orientation cell offsets in the adaptor's
# bottom-origin coordinates.  The origin agrees with active.x/active.y in the
# public observations (for example, the spawned T has offsets (1,2),(0,1),
# (1,1),(2,1)).  Rotating these around the SRS pivot gives the other states.
_BASE = {
    "J": ((0, 2), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 2), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 2), (2, 2), (0, 1), (1, 1)),
    "Z": ((0, 2), (1, 2), (1, 1), (2, 1)),
    "T": ((1, 2), (0, 1), (1, 1), (2, 1)),
    "I": ((0, 2), (1, 2), (2, 2), (3, 2)),
    "O": ((1, 2), (2, 2), (1, 1), (2, 1)),
}

# Keeping the kick tables here makes the reachability assumption explicit.  A
# low-level rotation is still sent to the engine, which is authoritative about
# the kick and legality.  The placement search only needs the fact that these
# are the Guideline SRS transitions, not an invented matrix rotation.
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


class _Plan:
    __slots__ = ("source", "piece", "orientation", "x", "rank", "lines", "features")

    def __init__(self, source: str, piece: str, orientation: int, x: int,
                 rank: float, lines: int, features: dict[str, Any]):
        self.source = source
        self.piece = piece
        self.orientation = orientation
        self.x = x
        self.rank = rank
        self.lines = lines
        self.features = features

    @property
    def use_hold(self) -> bool:
        return self.source == "hold"


# Shapes are derived once per process.  They contain only immutable tuples.
def _shapes() -> dict[str, tuple[tuple[tuple[int, int], ...], ...]]:
    result: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
    for piece, cells in _BASE.items():
        states: list[tuple[tuple[int, int], ...]] = []
        cur = tuple(cells)
        for _ in range(4):
            states.append(tuple(sorted(cur)))
            if piece == "O":
                continue
            pivot = (1.5, 1.5) if piece == "I" else (1.0, 1.0)
            nxt = []
            for x, y in cur:
                dx, dy = x - pivot[0], y - pivot[1]
                # Clockwise in bottom-origin coordinates.
                nx = pivot[0] + dy
                ny = pivot[1] - dx
                nxt.append((int(round(nx)), int(round(ny))))
            cur = tuple(nxt)
        result[piece] = tuple(states)
    return result


SHAPES = _shapes()


def _grid(obs: dict[str, Any]) -> list[list[int]]:
    raw = obs.get("board")
    out = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    if not isinstance(raw, list):
        return out
    # The observation is top-to-bottom; search uses y=0 at the bottom.
    for top_y in range(min(HEIGHT, len(raw))):
        row = raw[top_y]
        if not isinstance(row, list):
            continue
        y = HEIGHT - 1 - top_y
        for x in range(min(WIDTH, len(row))):
            try:
                out[y][x] = int(row[x])
            except (TypeError, ValueError):
                out[y][x] = 0
    return out


def _heights(grid: list[list[int]]) -> list[int]:
    hs = []
    for x in range(WIDTH):
        h = 0
        for y in range(HEIGHT - 1, -1, -1):
            if grid[y][x]:
                h = y + 1
                break
        hs.append(h)
    return hs


def _features(grid: list[list[int]]) -> dict[str, Any]:
    hs = _heights(grid)
    holes = 0
    hole_depth = 0
    for x, h in enumerate(hs):
        above = 0
        for y in range(h - 1, -1, -1):
            if grid[y][x]:
                above += 1
            elif above:
                holes += 1
                hole_depth += above
    bump = sum(abs(hs[i] - hs[i + 1]) for i in range(WIDTH - 1))
    max_h = max(hs) if hs else 0
    # A well is a lower column between taller neighbours.  It is reported as a
    # feature, but only weakly penalized: an I well can be strategically useful.
    wells = 0
    for x, h in enumerate(hs):
        left = hs[x - 1] if x else HEIGHT
        right = hs[x + 1] if x + 1 < WIDTH else HEIGHT
        wells += max(0, min(left, right) - h)
    transitions = 0
    for y in range(HEIGHT):
        prev = 1
        for x in range(WIDTH):
            cur = 1 if grid[y][x] else 0
            if cur != prev:
                transitions += 1
            prev = cur
        if prev == 0:
            transitions += 1
    return {
        "heights": hs,
        "aggregate_height": sum(hs),
        "max_height": max_h,
        "holes": holes,
        "hole_depth": hole_depth,
        "bumpiness": bump,
        "wells": wells,
        "transitions": transitions,
    }


def _drop(grid: list[list[int]], piece: str, orientation: int, ax: int) -> tuple[list[list[int]], dict[str, Any]] | None:
    """Return the exact visible result of a legal hard-drop placement."""
    shape = SHAPES.get(piece, SHAPES["O"])[orientation % 4]
    if any(ax + dx < 0 or ax + dx >= WIDTH for dx, _ in shape):
        return None

    def legal(y0: int) -> bool:
        for dx, dy in shape:
            x, y = ax + dx, y0 + dy
            if x < 0 or x >= WIDTH or y < 0:
                return False
            if y < HEIGHT and grid[y][x]:
                return False
        return True

    y = HEIGHT + 5
    while legal(y - 1):
        y -= 1
    if not legal(y):
        return None
    cells = [(ax + dx, y + dy) for dx, dy in shape]
    # A placement which still occupies the hidden buffer is a top-out risk and
    # is deliberately excluded from the normal shortlist.
    if any(cy >= HEIGHT for _, cy in cells):
        return None
    placed = [row[:] for row in grid]
    for x, yy in cells:
        placed[yy][x] = 1
    full = [row for row in placed if sum(1 for v in row if v) < WIDTH]
    lines = HEIGHT - len(full)
    full.extend([[0 for _ in range(WIDTH)] for _ in range(lines)])
    f = _features(full)
    f["lines"] = lines
    f["landing_height"] = max(y + dy for _, dy in shape)
    f["cells"] = cells
    return full, f


def _policy_score(f: dict[str, Any], before: dict[str, Any], style: str,
                  lines: int = 0, hold_penalty: float = 0.0) -> float:
    """A conservative placement value; arithmetic stays in code."""
    hole_w = 8.5
    height_w = 0.72
    max_w = 1.55
    bump_w = 0.34
    transition_w = 0.08
    well_w = 0.08
    line_w = 9.5
    landing_w = 0.10
    if style == "RECOVER":
        hole_w = 13.0
        height_w = 0.92
        max_w = 2.4
        line_w = 11.5
        bump_w = 0.45
    elif style == "WELL":
        # Preserve a controlled single-column well without worshipping it.
        well_w = -0.10
        line_w = 10.5

    delta_holes = f["holes"] - before.get("holes", 0)
    delta_height = f["aggregate_height"] - before.get("aggregate_height", 0)
    delta_max = f["max_height"] - before.get("max_height", 0)
    value = (
        line_w * (lines * lines + 0.30 * lines)
        - hole_w * f["holes"]
        - 2.1 * hole_w * max(0, delta_holes)
        - 0.34 * hole_w * f["hole_depth"]
        - height_w * f["aggregate_height"]
        - max_w * f["max_height"]
        - 0.55 * max_w * max(0, delta_max)
        - bump_w * f["bumpiness"]
        - transition_w * f["transitions"]
        - well_w * f["wells"]
        - landing_w * f["landing_height"]
        - 0.18 * max(0, delta_height)
        + hold_penalty
    )
    # A four-line clear is much more valuable than merely four cells of line
    # progress, while the primary objective remains survival when the board is
    # high.
    if lines == 4:
        value += 13.0
    return value


def _candidate_sources(obs: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    active = obs.get("active") or {}
    current = str(active.get("type", "O"))
    if current not in PIECES:
        current = "O"
    queue = [str(x) for x in (obs.get("queue") or []) if str(x) in PIECES]
    sources = [("current", current, queue)]
    if bool(obs.get("can_hold", False)):
        held = obs.get("hold")
        if held in PIECES:
            sources.append(("hold", str(held), queue))
        elif queue:
            # Empty hold consumes the first preview immediately.
            sources.append(("hold", queue[0], queue[1:]))
    return sources


def _best_one_piece(grid: list[list[int]], piece: str, before: dict[str, Any],
                    style: str) -> float:
    best = -10**12
    for ori in range(1 if piece == "O" else 4):
        for ax in range(-3, WIDTH + 1):
            result = _drop(grid, piece, ori, ax)
            if result is None:
                continue
            _, f = result
            v = _policy_score(f, before, style, int(f["lines"]))
            if v > best:
                best = v
    return best


def _placements(obs: dict[str, Any], style: str) -> list[_Plan]:
    grid = _grid(obs)
    before = _features(grid)
    queue = [str(x) for x in (obs.get("queue") or []) if str(x) in PIECES]
    all_plans: list[_Plan] = []
    for source, piece, post_hold_queue in _candidate_sources(obs):
        hold_penalty = -0.55 if source == "hold" else 0.0
        for ori in range(1 if piece == "O" else 4):
            for ax in range(-3, WIDTH + 1):
                result = _drop(grid, piece, ori, ax)
                if result is None:
                    continue
                after, f = result
                immediate = _policy_score(f, before, style, int(f["lines"]), hold_penalty)
                # One-ply lookahead makes the current decision preserve a
                # usable surface for the known next piece, without asking Jev
                # to do any arithmetic.
                next_piece = post_hold_queue[0] if post_hold_queue else None
                future = 0.0
                if next_piece in PIECES:
                    future = _best_one_piece(after, next_piece, f, style)
                    if future < -10**11:
                        future = -100.0
                rank = immediate + 0.28 * future
                all_plans.append(_Plan(source, piece, ori, ax, rank,
                                       int(f["lines"]), f))

    all_plans.sort(key=lambda p: (p.rank, p.lines, -p.features["holes"],
                                 -p.features["max_height"]), reverse=True)
    # Do not give Jev a wall of near-identical placements.  First retain the
    # best placement for every available source (current and hold), then fill
    # the remaining slots by rank while spreading orientation and side choices.
    # This keeps a tied empty-board O placement from hiding every current-piece
    # alternative.
    kept: list[_Plan] = []
    seen: set[tuple[str, int, int]] = set()

    def add(plan: _Plan) -> None:
        key = (plan.source, plan.orientation, plan.x)
        if key not in seen and len(kept) < 7:
            kept.append(plan)
            seen.add(key)

    if all_plans:
        add(all_plans[0])
    for source in ("current", "hold"):
        for plan in all_plans:
            if plan.source == source:
                add(plan)
                break
    for plan in all_plans:
        add(plan)
    return kept or [_Plan("current", "O", 0, 4, -10**12, 0, before)]


def _bucket_height(h: int) -> str:
    if h <= 5:
        return "low"
    if h <= 11:
        return "moderate"
    if h <= 16:
        return "high"
    return "critical"


def _bucket_holes(h: int) -> str:
    if h == 0:
        return "hole-free"
    if h <= 3:
        return "few shallow holes"
    if h <= 8:
        return "several holes"
    return "many or deep holes"


def _describe_plan(plan: _Plan, index: int) -> str:
    f = plan.features
    clear = {0: "no line", 1: "single", 2: "double", 3: "triple", 4: "Tetris"}.get(
        plan.lines, "line clear")
    source = "use hold, then place" if plan.use_hold else "place current piece"
    surface = "flat" if f["bumpiness"] <= 8 else ("uneven" if f["bumpiness"] <= 18 else "rough")
    return (
        f"{source}; orientation {plan.orientation}, target column {plan.x}; "
        f"{clear}; leaves {_bucket_height(f['max_height'])} stack, "
        f"{_bucket_holes(f['holes'])}, {surface} surface, "
        f"{f['wells']} well depth."
    )


def _choice_questions(options: list[tuple[str, str]], state: dict[str, Any]) -> dict[str, Choice]:
    criteria_a = {key: desc for key, desc in options}
    criteria_b = {key: desc for key, desc in reversed(options)}
    context = json.dumps(state, separators=(",", ":"), ensure_ascii=True)
    common = (
        "Choose only one candidate label. The candidate descriptions already contain "
        "the code-computed result of a legal SRS placement. Prefer a low, flat, "
        "hole-free board and a clear when it does not create top-out risk. Do not "
        "invent inputs, do arithmetic, or choose a label not present in the state. "
        "If two candidates are close, prefer the one with lower stack and fewer holes.\n"
        f"Decision state: {context}"
    )
    return {
        "pick_a": Choice(common + " This is the primary ranking.", criteria_a),
        "pick_b": Choice(common + " Recheck the same choice independently; option order is reversed.", criteria_b),
    }


def _probability(answer: Any, value: str) -> float:
    try:
        probs = getattr(answer, "probabilities", {}) or {}
        return float(probs.get(value, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _chosen_plan(answers: Any, plans: list[_Plan]) -> _Plan | None:
    if not isinstance(answers, dict):
        return None
    valid = {f"C{i}": p for i, p in enumerate(plans)}
    a = answers.get("pick_a")
    b = answers.get("pick_b")
    av = getattr(a, "value", None)
    bv = getattr(b, "value", None)
    if av not in valid or bv != av:
        return None
    # A response with a very diffuse distribution is not allowed to override
    # the verified code fallback.  The threshold is lower for a placement
    # choice than for an irreversible arbitrary action because all options are
    # already legal and hard-dropable.
    if _probability(a, av) < 0.36 or _probability(b, bv) < 0.36:
        return None
    return valid[av]


def _parse_style(text: str) -> str | None:
    upper = str(text or "").upper()
    # Avoid accepting a mention in a long explanation when possible; the
    # prompt asks for a single token, but garbage model output is expected.
    m = re.search(r"\b(RECOVER|BALANCED|WELL)\b", upper)
    if not m:
        return None
    return {"RECOVER": "RECOVER", "WELL": "WELL", "BALANCED": "BALANCED"}[m.group(1)]


def _safe_action(spec: Any, name: str) -> str:
    names = set(getattr(spec, "names", ()) or ())
    if name in names:
        return name
    noop = getattr(spec, "noop", None)
    if noop in names:
        return noop
    return next(iter(names), "noop")


class Harness:
    def __init__(self, ctx: Any):
        self.ctx = ctx
        self.style = "BALANCED"
        self._style_future = None
        self._style_requested = False
        self._style_revision = 0

    def _start_astra(self, game: Any, obs: dict[str, Any], reason: str) -> None:
        ctx = self.ctx
        if self._style_requested or getattr(ctx, "system2", None) is None:
            return
        self._style_requested = True
        compact = {
            "reason": reason,
            "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
            "board_metrics": {
                "holes": obs.get("holes", 0),
                "aggregate_height": obs.get("aggregate_height", 0),
                "top_out_height": obs.get("top_out_height", 0),
                "bumpiness": obs.get("bumpiness", 0),
            },
            "active": (obs.get("active") or {}).get("type"),
            "hold": obs.get("hold"),
            "queue": list(obs.get("queue") or [])[:5],
            "level": obs.get("level", 1),
        }
        prompt = (
            "You are the slow strategic reviewer for a real-time Guideline Tetris "
            "controller. Code enumerates legal placements and Jev ranks them. "
            "Choose the high-level risk policy from the observed facts below. "
            "Return exactly one token: BALANCED for a low, ordinary stack; "
            "RECOVER when holes or height make survival/downstacking dominant; "
            "WELL only when a controlled side well is safe and useful. Never "
            "specify keypresses or a seed-specific sequence.\n" + json.dumps(compact, separators=(",", ":"))
        )

        def call() -> str | None:
            try:
                result = ctx.think(prompt, timeout_s=70.0)
                return getattr(result, "text", "")
            except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
                return None
            except Exception:
                return None

        try:
            self._style_future = ctx.spawn(call)
        except Exception:
            self._style_future = None

    def _harvest_style(self) -> None:
        future = self._style_future
        if future is None or not future.done():
            return
        self._style_future = None
        try:
            style = _parse_style(future.result())
        except Exception:
            style = None
        if style and style != self.style:
            self.style = style
            self._style_revision += 1
            try:
                self.ctx.log("astra_policy_update", style=style, revision=self._style_revision)
            except Exception:
                pass

    def _ask_jev(self, state: dict[str, Any], options: list[tuple[str, str]]) -> Any:
        try:
            return self.ctx.decide(
                state,
                _choice_questions(options, state),
                timeout_s=0.45,
            )
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            return None
        except Exception:
            # Stand-in models and malformed model responses must never make the
            # game loop fail; the deterministic first candidate remains valid.
            return None

    def run_episode(self, game: Any) -> None:
        ctx = self.ctx
        spec = game.action_spec
        if getattr(ctx, "system2", None) is not None:
            try:
                initial = game.observe()
                self._start_astra(game, initial, "episode_start")
            except Exception:
                pass

        # A piece instance is identified by the adaptor's placed-piece count.
        # Hold changes active.type without creating a new instance.
        piece_number = None
        active_type = None
        plans: list[_Plan] = []
        fallback: _Plan | None = None
        selected: _Plan | None = None
        hold_sent = False
        decision_future = None
        decision_piece = None
        decision_plans: list[_Plan] = []
        warmup_ticks = 0

        while not ctx.done and not getattr(game, "done", False):
            self._harvest_style()
            try:
                obs = game.observe()
            except Exception:
                break
            if obs.get("game_over"):
                break

            number = obs.get("pieces", 0)
            if number != piece_number:
                piece_number = number
                active_type = (obs.get("active") or {}).get("type")
                plans = _placements(obs, self.style)
                fallback = plans[0] if plans else None
                selected = fallback
                hold_sent = False
                warmup_ticks = 0
                decision_piece = number
                decision_plans = plans[:]
                decision_future = None
                if getattr(ctx, "system1", None) is not None and plans:
                    options = [(f"C{i}", _describe_plan(p, i)) for i, p in enumerate(plans)]
                    state = {
                        "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
                        "policy": self.style,
                        "piece_instance": number,
                        "board": {
                            "height": _bucket_height(int(obs.get("top_out_height", 0))),
                            "holes": _bucket_holes(int(obs.get("holes", 0))),
                            "surface": "flat" if int(obs.get("bumpiness", 0)) <= 8 else "uneven",
                            "lines": int(obs.get("lines", 0)),
                            "level": int(obs.get("level", 1)),
                            "combo": "active" if int(obs.get("combo", -1)) >= 0 else "inactive",
                            "back_to_back": bool(obs.get("back_to_back", False)),
                        },
                        "active_piece": active_type,
                        "hold": obs.get("hold"),
                        "can_hold": bool(obs.get("can_hold", False)),
                        "next_queue": list(obs.get("queue") or [])[:5],
                        "fallback": "C0",
                        "candidates": {key: desc for key, desc in options},
                    }
                    try:
                        decision_future = ctx.spawn(lambda s=state, o=options: self._ask_jev(s, o))
                    except Exception:
                        decision_future = None

            # Harvest a Jev response only for the exact still-falling piece.
            if decision_future is not None and decision_piece == number and decision_future.done():
                try:
                    answer = decision_future.result()
                except Exception:
                    answer = None
                chosen = _chosen_plan(answer, decision_plans)
                if chosen is not None:
                    selected = chosen
                    try:
                        ctx.log("jev_candidate", piece=number,
                                candidate=f"C{decision_plans.index(chosen)}")
                    except Exception:
                        pass
                decision_future = None

            target = selected or fallback
            if target is None:
                action = _safe_action(spec, "noop")
            else:
                # A late Jev answer can name the original current piece after
                # a hold fallback has already changed active.type.  Never use
                # that stale geometry for the replacement: recompute the
                # current-piece fallback from the fresh observation.
                active = obs.get("active") or {}
                observed_type = active.get("type")
                if (observed_type in PIECES and
                        (not target.use_hold or hold_sent) and
                        observed_type != target.piece):
                    refreshed = _placements(obs, self.style)
                    if refreshed:
                        plans = refreshed
                        fallback = refreshed[0]
                        selected = fallback
                        target = fallback
                # The model may have chosen a hold candidate after the fallback
                # already started.  It is safe to use hold once, but never issue
                # it twice; after a hold the observed active piece is the target
                # piece and the same orientation/x plan remains valid.
                if target.use_hold and not hold_sent:
                    if bool(obs.get("can_hold", False)):
                        action = _safe_action(spec, "hold")
                        hold_sent = True
                    else:
                        selected = fallback
                        target = fallback
                        action = _safe_action(spec, "noop") if target is None else _safe_action(spec, "hard_drop")
                elif target.use_hold and hold_sent:
                    action = self._placement_action(obs, target, spec)
                else:
                    # Give a warm Jev request a couple of ticks at the top of a
                    # piece, but never wait through a grounded lock deadline.
                    if (decision_future is not None and warmup_ticks < 2 and
                            obs.get("lock_remaining_s") is None):
                        warmup_ticks += 1
                        action = _safe_action(spec, "noop")
                    else:
                        action = self._placement_action(obs, target, spec)

            try:
                ctx.act(action)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                # A valid named action should not fail, but returning keeps a
                # malformed stand-in adaptor from turning this candidate into
                # an evaluation error.
                return
            try:
                ctx.wait_for_tick()
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                return

    @staticmethod
    def _placement_action(obs: dict[str, Any], target: _Plan, spec: Any) -> str:
        active = obs.get("active") or {}
        try:
            ori = int(active.get("orientation", 0)) % 4
        except (TypeError, ValueError):
            ori = 0
        try:
            x = int(active.get("x", 3))
        except (TypeError, ValueError):
            x = 3
        delta = (target.orientation - ori) % 4
        if delta:
            name = "rotate_cw" if delta <= 2 else "rotate_ccw"
            return _safe_action(spec, name)
        if x < target.x:
            return _safe_action(spec, "right")
        if x > target.x:
            return _safe_action(spec, "left")
        return _safe_action(spec, "hard_drop")


def build(ctx: Any) -> Harness:
    return Harness(ctx)
