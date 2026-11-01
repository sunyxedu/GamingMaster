"""Jev-driven placement bridge for the realtime Tetris parent.

The parent asks Jev for a primitive key.  This variant keeps one Jev Choice,
but makes its options reachable horizontal placements.  Code evaluates the
resulting board exactly for the current orientation, and translates the chosen
x coordinate into verified one-tick moves followed by a hard drop.
"""
from __future__ import annotations

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
from metaharness.models.jev import Choice


WIDTH = 10


def _board(obs):
    rows = obs.get("board") or []
    height = len(rows) or 20
    occupied = set()
    for row_index, row in enumerate(rows):
        y = height - 1 - row_index
        for x, value in enumerate(row):
            if value:
                occupied.add((x, y))
    return occupied, height


def _active_info(obs):
    active = obs.get("active") or {}
    try:
        ax = int(active.get("x", 0))
        ay = int(active.get("y", 0))
    except (TypeError, ValueError):
        ax, ay = 0, 0
    offsets = []
    for cell in active.get("cells") or []:
        try:
            offsets.append((int(cell[0]) - ax, int(cell[1]) - ay))
        except (TypeError, ValueError, IndexError):
            pass
    if not offsets:
        offsets = [(0, 0)]
    try:
        orientation = int(active.get("orientation", 0))
    except (TypeError, ValueError):
        orientation = 0
    return active, ax, ay, orientation, tuple(offsets)


def _fits(occupied, offsets, x, y):
    for dx, dy in offsets:
        px, py = x + dx, y + dy
        if px < 0 or px >= WIDTH or py < 0 or (px, py) in occupied:
            return False
    return True


def _clear_and_metrics(occupied, cells, height):
    placed = set(occupied)
    placed.update(cells)
    full = {y for y in range(height) if all((x, y) in placed for x in range(WIDTH))}
    if full:
        shifted = set()
        for x, y in placed:
            if y in full:
                continue
            below = sum(1 for fy in full if fy < y)
            ny = y - below
            if 0 <= ny < height:
                shifted.add((x, ny))
        placed = shifted
    heights = []
    holes = 0
    for x in range(WIDTH):
        ys = [y for px, y in placed if px == x]
        top = max(ys) + 1 if ys else 0
        heights.append(top)
        holes += sum(1 for y in range(top) if (x, y) not in placed)
    aggregate = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i - 1]) for i in range(1, WIDTH))
    max_height = max(heights) if heights else 0
    return len(full), holes, aggregate, bumpiness, max_height


def _bucket(n, low, high):
    if n <= low:
        return "low"
    if n <= high:
        return "moderate"
    return "high"


def _placements(obs):
    occupied, height = _board(obs)
    active, ax, ay, orientation, offsets = _active_info(obs)
    min_dx = min(dx for dx, _ in offsets)
    max_dx = max(dx for dx, _ in offsets)
    candidates = []
    # Only horizontal motion is generated here.  This is deliberate: it is a
    # legal, short bridge from a Jev placement choice and avoids inventing SRS
    # coordinates that the observation does not expose.
    for target_x in range(-min_dx, WIDTH - max_dx):
        if not _fits(occupied, offsets, target_x, ay):
            continue
        landing_y = ay
        while _fits(occupied, offsets, target_x, landing_y - 1):
            landing_y -= 1
        cells = tuple((target_x + dx, landing_y + dy) for dx, dy in offsets)
        if any(y >= height for _, y in cells):
            continue
        lines, holes, aggregate, bumpiness, max_height = _clear_and_metrics(
            occupied, cells, height
        )
        # Strongly penalize holes and height; lines are valuable but never
        # justify a placement that immediately leaves a severe cavity.
        value = (
            lines * 80.0
            - holes * 16.0
            - aggregate * 1.0
            - bumpiness * 0.55
            - max_height * 2.5
        )
        candidates.append({
            "label": "x%d" % target_x,
            "x": target_x,
            "landing_y": landing_y,
            "lines": lines,
            "holes": holes,
            "aggregate": aggregate,
            "bumpiness": bumpiness,
            "max_height": max_height,
            "value": value,
        })
    candidates.sort(key=lambda item: (-item["value"], item["x"]))
    return candidates, (active, ax, ay, orientation, offsets)


def _description(candidate):
    lines = candidate["lines"]
    holes = candidate["holes"]
    return (
        "Place the current piece with its horizontal anchor at x=%d. %s; "
        "the resulting stack has %s holes, %s aggregate height, %s bumpiness, "
        "and %s maximum height. This is a legal landing for the current "
        "orientation; do not interpret the label as a primitive key."
        % (
            candidate["x"],
            "clears a line" if lines else "does not clear a line",
            "no" if holes == 0 else ("few" if holes <= 2 else "many"),
            _bucket(candidate["aggregate"], 8, 22),
            _bucket(candidate["bumpiness"], 5, 14),
            _bucket(candidate["max_height"], 4, 10),
        )
    )


def _state(obs, candidates):
    active = obs.get("active") or {}
    # Keep the state small enough that Jev sees the board and the alternatives,
    # while retaining the exact fields needed to disambiguate a placement.
    return {
        "ruleset": obs.get("ruleset", "guideline-marathon-v1"),
        "board": obs.get("board") or [],
        "active": {
            "type": active.get("type"),
            "orientation": active.get("orientation"),
            "x": active.get("x"),
            "y": active.get("y"),
            "cells": active.get("cells") or [],
        },
        "ghost_cells": obs.get("ghost_cells") or [],
        "hold": obs.get("hold"),
        "can_hold": bool(obs.get("can_hold", False)),
        "queue": obs.get("queue") or [],
        "stack": {
            "holes": obs.get("holes", 0),
            "aggregate_height": obs.get("aggregate_height", 0),
            "bumpiness": obs.get("bumpiness", 0),
            "top_out_height": obs.get("top_out_height", 0),
        },
        "lines": obs.get("lines", 0),
        "level": obs.get("level", 1),
        "pieces": obs.get("pieces", 0),
        "candidates": [
            {
                "label": c["label"],
                "x": c["x"],
                "landing_y": c["landing_y"],
                "lines": c["lines"],
                "holes": c["holes"],
                "aggregate": c["aggregate"],
                "bumpiness": c["bumpiness"],
                "max_height": c["max_height"],
            }
            for c in candidates
        ],
    }


def _piece_signature(obs):
    active = obs.get("active") or {}
    return (obs.get("pieces"), active.get("type"), active.get("orientation"))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _one_tick(self, game, action):
        if self.ctx.done or getattr(game, "done", False):
            return False
        self.ctx.act(action)
        try:
            self.ctx.wait_for_tick()
        except (BudgetExceeded, EpisodeOver):
            return False
        return not self.ctx.done and not getattr(game, "done", False)

    def _execute(self, game, target_x, signature):
        """Move only as far as verified observations permit, then drop."""
        ctx = self.ctx
        for _ in range(WIDTH + 2):
            if ctx.done or getattr(game, "done", False):
                return
            current = game.observe()
            if _piece_signature(current) != signature:
                return
            active = current.get("active") or {}
            try:
                current_x = int(active.get("x", 0))
            except (TypeError, ValueError):
                current_x = target_x
            if current_x == target_x:
                break
            action = "left" if current_x > target_x else "right"
            before_x = current_x
            if not self._one_tick(game, action):
                return
            after = game.observe()
            if _piece_signature(after) != signature:
                return
            try:
                after_x = int((after.get("active") or {}).get("x", before_x))
            except (TypeError, ValueError):
                after_x = before_x
            # A blocked move cannot reach this target without risking a loop.
            if after_x == before_x:
                break
        if not ctx.done and not getattr(game, "done", False):
            self._one_tick(game, "hard_drop")

    def run_episode(self, game):
        ctx = self.ctx
        if ctx.system1 is None:
            return
        while not ctx.done and not getattr(game, "done", False):
            observed = game.observe()
            candidates, _ = _placements(observed)
            if not candidates:
                # This is a last-resort valid action, not a model parse path.
                if not self._one_tick(game, "hard_drop"):
                    return
                continue
            by_label = {c["label"]: c for c in candidates}
            criteria = {c["label"]: _description(c) for c in candidates}
            question = Choice(
                "Choose the safest legal final placement for the current piece. "
                "Prefer clearing lines, then no holes, then a low and flat stack. "
                "Choose exactly one placement label from `candidates`. The code "
                "will execute the horizontal moves and hard drop; never answer "
                "with a primitive action name.",
                criteria=criteria,
            )
            state = _state(observed, candidates)
            signature = _piece_signature(observed)
            selected = None
            try:
                answers = ctx.decide(state, {"placement": question}, timeout_s=0.45)
                answer = answers.get("placement") if isinstance(answers, dict) else None
                selected = getattr(answer, "value", None)
            except (TimeoutError, RuntimeError):
                selected = None
            except (BudgetExceeded, EpisodeOver):
                return
            except ModelUnavailable:
                raise
            # A late answer may describe a different falling piece. Recompute
            # and use the current best candidate in that case.
            latest = game.observe()
            latest_candidates, _ = _placements(latest)
            if not latest_candidates:
                return
            latest_by_label = {c["label"]: c for c in latest_candidates}
            if _piece_signature(latest) != signature or selected not in latest_by_label:
                chosen = latest_candidates[0]
            else:
                chosen = latest_by_label[selected]
            ctx.log(
                "placement_choice",
                selected=chosen["label"],
                candidate_count=len(latest_candidates),
                fallback=selected not in latest_by_label,
            )
            self._execute(game, chosen["x"], _piece_signature(latest))


def build(ctx):
    return Harness(ctx)
