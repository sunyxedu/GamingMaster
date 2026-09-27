"""Offline breadth-first solver of the ARC-AGI-3 games — demonstrations for the tier-3
network's imitation data (the network itself never searches: this only makes targets,
like the engine labels of the board games).

Per game, level by level from the engine's seed-0 start: a breadth-first search over
action sequences from the level's first frame, the game state cloned per child
(``copy.deepcopy`` of the engine's game object, 0.4-6 ms), states deduplicated by
(frame, available actions), game-over states pruned.  Actions per state: every
available parameter-less action (ACTION1..5, ACTION7) and, when ACTION6 is available,
one click per connected single-colour object other than the background (the most
frequent colour) — the object-level abstraction of a click (7-65 per frame on the public
games).  The first sequence that completes the level is its (shortest, under that
abstraction) solution; the next level starts from the state it reaches.  A level the
budget (``--budget`` expansions, ``--max_depth``) cannot solve ends the game's search.
An action the engine raises on is pruned like a game over (the level's
``engine_errors``); a game whose search fails otherwise records ``"error"`` and the
other games' searches go on.

    OPERATION_MODE=offline ENVIRONMENTS_DIR=... uv run python scripts/arc3_solve.py \\
        --games cd82 lf52 ls20 sb26 --budget 20000 --out data/arc3/solutions.json

Output JSON: {game: {"win_levels": W, "levels": [{"level": k, "actions": [[type, x, y],
...], "expanded": n, "seconds": s}, ...], "unsolved_at": k | null}}; replay it into
IL shards with scripts/arc3_shards.py.  Each game's results so far are also written
to ``<out>.partial/<game>.json`` after every solved level, so a search cut short keeps
its solved levels (the shards script takes that directory's files as well).
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

Action = Tuple[int, int, int]               # (type, x, y); x, y = 0 unless ACTION6


def _sdk():
    os.environ.setdefault("OPERATION_MODE", "offline")
    logging.disable(logging.INFO)
    import arc_agi
    from arcengine import ActionInput, GameAction, GameState
    return arc_agi, ActionInput, GameAction, GameState


def components(grid: np.ndarray, background: int) -> List[Tuple[int, int]]:
    """One representative cell (x, y) per 4-connected single-colour component other
    than ``background``: the component's cell nearest its centroid."""
    H, W = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    reps = []
    for y in range(H):
        for x in range(W):
            if seen[y, x] or grid[y, x] == background:
                continue
            c = grid[y, x]
            stack, cells = [(y, x)], []
            seen[y, x] = True
            while stack:
                yy, xx = stack.pop()
                cells.append((yy, xx))
                for ny, nx in ((yy + 1, xx), (yy - 1, xx), (yy, xx + 1), (yy, xx - 1)):
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and grid[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            cy = sum(p[0] for p in cells) / len(cells)
            cx = sum(p[1] for p in cells) / len(cells)
            by, bx = min(cells, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)
            reps.append((bx, by))
    return reps


def candidate_actions(fd) -> List[Action]:
    grid = np.asarray(fd.frame[-1])
    out: List[Action] = []
    avail = [int(a) for a in fd.available_actions]
    for a in avail:
        if a != 6 and 1 <= a <= 7:
            out.append((a, 0, 0))
    if 6 in avail:
        vals, cnts = np.unique(grid, return_counts=True)
        bg = int(vals[cnts.argmax()])
        out.extend((6, x, y) for x, y in components(grid, bg))
    return out


def state_key(fd) -> bytes:
    h = hashlib.blake2b(np.asarray(fd.frame[-1], dtype=np.uint8).tobytes(), digest_size=16)
    h.update(bytes(sorted(int(a) for a in fd.available_actions)))
    return h.digest()


def solve_level(game_obj, fd0, level: int, budget: int, max_depth: int, sdk
                ) -> Tuple[Optional[List[Action]], Optional[object], Optional[object], int, Dict]:
    """BFS from (game_obj, fd0) until levels_completed > level.  Returns (actions,
    the solved game object, its frame data, expansions, engine errors) or (None, None,
    None, n, errors).  An action the engine raises on is a dead end (the games' own
    code fails on some states: ``RuntimeError: … not attached to …``); ``errors``
    counts them with the first message."""
    _, ActionInput, GameAction, GameState = sdk
    queue = collections.deque([(game_obj, fd0, [])])
    seen = {state_key(fd0)}
    expanded = 0
    errors: Dict = {"count": 0, "first": None}
    while queue and expanded < budget:
        g, fd, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for act in candidate_actions(fd):
            if expanded >= budget:
                break
            child = copy.deepcopy(g)
            t, x, y = act
            data = {"x": x, "y": y} if t == 6 else {}
            expanded += 1
            try:
                nfd = child.perform_action(ActionInput(id=GameAction.from_id(t), data=data),
                                           raw=True)
            except Exception as e:
                errors["count"] += 1
                if errors["first"] is None:
                    errors["first"] = f"{type(e).__name__}: {e} (after {path + [act]})"
                continue
            if nfd is None or nfd.state == GameState.GAME_OVER:
                continue
            if int(nfd.levels_completed) > level:
                return path + [act], child, nfd, expanded, errors
            k = state_key(nfd)
            if k in seen:
                continue
            seen.add(k)
            queue.append((child, nfd, path + [act]))
    return None, None, None, expanded, errors


def _save_partial(partial_dir: Optional[str], game: str, res: Dict) -> None:
    """Write a game's results so far (atomic): a long search that is cut short
    (time limit, preemption) keeps the levels it already solved."""
    if not partial_dir:
        return
    os.makedirs(partial_dir, exist_ok=True)
    tmp = os.path.join(partial_dir, f".{game}.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, os.path.join(partial_dir, f"{game}.json"))


def solve_game(args) -> Tuple[str, Dict]:
    """One game's levels in order; a failure outside the per-action dead ends (loading
    the game, cloning a state) ends this game's search with ``"error"`` in its results
    and leaves the other games' searches running."""
    game, budget, max_depth, env_dir, partial_dir = args
    res: Dict = {"win_levels": 0, "levels": [], "unsolved_at": None}
    try:
        _solve_levels(game, budget, max_depth, env_dir, partial_dir, res)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        _save_partial(partial_dir, game, res)
        print(f"[{game}] search failed: {res['error']}", flush=True)
    return game, res


def _solve_levels(game: str, budget: int, max_depth: int, env_dir, partial_dir,
                  res: Dict) -> None:
    if env_dir:
        os.environ["ENVIRONMENTS_DIR"] = env_dir
    sdk = _sdk()
    arc_agi = sdk[0]
    w = arc_agi.Arcade().make(game, seed=0)
    fd = w.reset()
    g = w._game
    res["win_levels"] = int(fd.win_levels)
    for level in range(int(fd.win_levels)):
        t0 = time.time()
        actions, g2, fd2, n, errors = solve_level(g, fd, level, budget, max_depth, sdk)
        row = {"level": level, "actions": None if actions is None else [list(a) for a in actions],
               "expanded": n, "seconds": round(time.time() - t0, 1)}
        if errors["count"]:
            row["engine_errors"] = errors["count"]
            row["first_engine_error"] = errors["first"]
            print(f"[{game}] level {level}: {errors['count']} engine errors, first "
                  f"{errors['first']}", flush=True)
        res["levels"].append(row)
        if actions is None:
            res["unsolved_at"] = level
            _save_partial(partial_dir, game, res)
            break
        _save_partial(partial_dir, game, res)
        g, fd = g2, fd2
        print(f"[{game}] level {level}: {len(actions)} actions ({n} expansions, "
              f"{time.time() - t0:.1f} s)", flush=True)


def main(argv=None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--games", nargs="+", default=["cd82", "lf52", "ls20", "sb26"])
    p.add_argument("--budget", type=int, default=20000, help="expansions per level")
    p.add_argument("--max_depth", type=int, default=80)
    p.add_argument("--environments_dir", default=os.environ.get("ENVIRONMENTS_DIR"))
    p.add_argument("--workers", type=int, default=0, help="processes (0 = one per game)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    partial = a.out + ".partial"
    jobs = [(g, a.budget, a.max_depth, a.environments_dir, partial) for g in a.games]
    n = a.workers or len(jobs)
    if n > 1:
        with mp.get_context("spawn").Pool(n) as pool:
            results = dict(pool.map(solve_game, jobs))
    else:
        results = dict(solve_game(j) for j in jobs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=1)
    for g, r in results.items():
        solved = sum(1 for lv in r["levels"] if lv["actions"] is not None)
        print(f"{g}: {solved}/{r['win_levels']} levels")
    return results


if __name__ == "__main__":
    main()
