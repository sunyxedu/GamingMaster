"""Candidate 0001 — "probe, model, autopilot; ask Astra only when stuck".

Hypothesis: the naive harness scores 0 because it spends one System-2 call
per action (budget 15 -> 15 of 80 actions used) and shows Astra nothing about
what earlier actions did, so it repeats blocked moves. Fix = move the cheap
parts of the reasoning into the harness:

  1. scripted PROBE (no model calls): try every parameter-less action once
     (again from a new position if it had no visible effect) and record the
     grid diff -> action->displacement hypotheses + colour passability stats;
  2. a game-agnostic WORLD MODEL: the agent is the unique grid value that
     moves; values are tagged passable / blocked / special (jump, vanish);
     hypotheses are refuted by contradiction (expected move, no move ->
     blocked; non-agent cells changed -> passability knowledge is stale);
  3. AUTOPILOT: if a goal value is known, Dijkstra over believed-passable
     cells and execute; re-plan on every surprise; no model call at all;
  4. ASTRA only when the autopilot has no path / no goal: it gets an ASCII
     map with coordinates + legend, the learned action table, colour stats,
     recent history and returns a JSON plan of up to 15 actions (+ goal value
     + a one-line lesson kept in ctx.memory for later seeds). Plans stop early
     when two consecutive actions change nothing or the score changes.

Everything is discovered from game.describe(), game.action_spec and the
observation dict (grid = first list-of-lists found; legend = dict of
str->str; available actions = list of action names) — no rulepuzzle keys.
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from typing import Any, Optional

MAX_PLAN = 15          # actions Astra may commit per call
MAX_ACTS = 3000        # hard cap on ctx.act per episode (safety)
PARAM_RE = re.compile(r"\bx\s*y\b|click|pass as|<x>|coordinates", re.I)


# ----------------------------------------------------------------- observation
def find_grid(obs: Any) -> tuple[Optional[str], Optional[list[list[Any]]]]:
    """Locate a 2-D grid in the observation (list of equal-length rows of scalars,
    or a multi-line string). Returns (key, rows). Largest grid wins."""
    cands: list[tuple[int, Optional[str], list[list[Any]]]] = []

    def _try(k, v):
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, (list, tuple)) for r in v):
            w = len(v[0])
            if w >= 2 and all(len(r) == w for r in v) and all(
                    isinstance(c, (int, float, str)) for r in v for c in r):
                cands.append((w * len(v), k, [list(r) for r in v]))
        elif isinstance(v, str) and "\n" in v:
            rows = [r for r in v.split("\n") if r.strip()]
            if len(rows) >= 2 and len({len(r) for r in rows}) == 1 and len(rows[0]) >= 2:
                cands.append((len(rows) * len(rows[0]), k, [list(r) for r in rows]))

    if isinstance(obs, dict):
        for k, v in obs.items():
            _try(k, v)
    else:
        _try(None, obs)
    if not cands:
        return None, None
    cands.sort(key=lambda t: -t[0])
    return cands[0][1], cands[0][2]


def find_legend(obs: Any) -> dict[str, str]:
    if not isinstance(obs, dict):
        return {}
    for k, v in obs.items():
        if isinstance(v, dict) and v and all(isinstance(x, str) for x in v.values()):
            return {str(a): b for a, b in v.items()}
    return {}


def find_available(obs: Any, names: list[str]) -> list[str]:
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                up = [x.upper() for x in v]
                if all(x in names for x in up):
                    return [n for n in names if n in up]
    return list(names)


def scalar_fields(obs: Any, grid_key: Optional[str]) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    out = {}
    for k, v in obs.items():
        if k == grid_key:
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list) and len(v) <= 12 and all(isinstance(x, (int, float, str)) for x in v):
            out[k] = v
    return out


def glyph(v: Any) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, int) or (isinstance(v, float) and v == int(v)):
        v = int(v)
        if 0 <= v <= 9:
            return str(v)
        if 10 <= v <= 35:
            return chr(ord("a") + v - 10)
        return "?"
    s = str(v)
    return s[0] if s else "."


def render(grid: list[list[Any]], agent: Optional[tuple[int, int]] = None) -> str:
    h, w = len(grid), len(grid[0])
    lines = ["    " + "".join(str(x % 10) for x in range(w))]
    for y in range(h):
        row = "".join("@" if agent == (x, y) else glyph(grid[y][x]) for x in range(w))
        lines.append(f"{y:3d} {row}")
    return "\n".join(lines)


def unique_positions(grid: list[list[Any]]) -> dict[Any, tuple[int, int]]:
    cnt: Counter = Counter()
    pos: dict[Any, tuple[int, int]] = {}
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            cnt[v] += 1
            pos[v] = (x, y)
    return {v: p for v, p in pos.items() if cnt[v] == 1}


def grid_diff(a: list[list[Any]], b: list[list[Any]]) -> list[tuple[int, int, Any, Any]]:
    if a is None or b is None or len(a) != len(b) or len(a[0]) != len(b[0]):
        return [(-1, -1, None, None)] * 9999  # incomparable => "everything changed"
    out = []
    for y in range(len(a)):
        ra, rb = a[y], b[y]
        if ra != rb:
            for x in range(len(ra)):
                if ra[x] != rb[x]:
                    out.append((x, y, ra[x], rb[x]))
    return out


def parse_plan(text: str, names: list[str]) -> tuple[dict[str, Any], list[str]]:
    """JSON object if present; else every action mention in order (with up to 2 ints)."""
    text = text or ""
    js: dict[str, Any] = {}
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.S):
        try:
            cand = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(cand, dict) and ("actions" in cand or "goal_value" in cand or "plan" in cand):
            js = cand
            break
    seq: list[str] = []
    raw = js.get("actions", js.get("plan")) if js else None
    if isinstance(raw, list):
        for a in raw:
            s = _canon(str(a), names)
            if s:
                seq.append(s)
    if not seq:
        pat = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        for m in re.finditer(rf"({pat})((?:[\s,]+-?\d+){{0,2}})", text, flags=re.I):
            s = _canon(m.group(0).replace(",", " "), names)
            if s:
                seq.append(s)
    return js, seq[:MAX_PLAN]


def _canon(tok: str, names: list[str]) -> Optional[str]:
    parts = tok.strip().replace("(", " ").replace(")", " ").split()
    if not parts:
        return None
    up = parts[0].upper()
    name = next((n for n in names if n.upper() == up), None)
    if name is None:
        return None
    args = [p for p in parts[1:] if re.fullmatch(r"-?\d+", p)]
    return " ".join([name] + args[:2])


GOAL_WORDS = re.compile(r"complete|goal|reach|exit|win|target|finish", re.I)


def goal_from_text(describe: str, legend: dict[str, str], grid: list[list[Any]], agent_value: Any) -> Any:
    """Generic prior: a legend name that appears in a sentence of the game
    description that talks about reaching/completing => candidate goal value
    (only if that value is on the grid and is not the agent)."""
    if not legend or not describe:
        return None
    present = {v for row in grid for v in row}
    for sent in re.split(r"[.;\n]", describe):
        if not GOAL_WORDS.search(sent):
            continue
        for k, name in legend.items():
            if len(name) < 3 or not re.search(rf"\b{re.escape(name)}\b", sent, re.I):
                continue
            for cand in (k, int(k) if k.lstrip('-').isdigit() else None):
                if cand is not None and cand in present and cand != agent_value:
                    return cand
    return None


# ----------------------------------------------------------------- world model
class World:
    """Everything learned within one episode (per-level maps are re-derived
    from the live grid; only action/colour knowledge persists across levels)."""

    def __init__(self, names: list[str], spec_desc: dict[str, str]):
        self.names = names
        self.param_actions = {n for n in names if PARAM_RE.search(spec_desc.get(n, "") or "")}
        self.deltas: dict[str, Counter] = defaultdict(Counter)     # action -> Counter[(dx,dy)]
        self.noeffect: Counter = Counter()                          # action -> count of "nothing changed"
        self.other_change: Counter = Counter()                      # action -> count of non-agent cell changes
        self.passed: Counter = Counter()                            # value -> times agent stood on it
        self.blocked: Counter = Counter()                           # value -> times a move into it failed
        self.hard_wall: set = set()
        self.special: dict[Any, Counter] = defaultdict(Counter)     # value -> {"jump":n,"vanish":n}
        self.portals: dict[tuple[int, int], tuple[int, int]] = {}   # cell stepped into -> where the agent landed
        self.agent_value: Any = None
        self.goal_value: Any = None
        self.history: list[str] = []
        self.grid: Optional[list[list[Any]]] = None
        self.grid_key: Optional[str] = None
        self.legend: dict[str, str] = {}
        self.score = 0.0
        self.level_changes = 0

    # -- action semantics ------------------------------------------------
    def move_delta(self, a: str) -> Optional[tuple[int, int]]:
        c = self.deltas.get(a)
        if not c:
            return None
        unit = {d: n for d, n in c.items() if max(abs(d[0]), abs(d[1])) == 1 and d != (0, 0)}
        if not unit:
            return None
        d, n = max(unit.items(), key=lambda kv: kv[1])
        return d if n >= 1 else None

    def move_map(self) -> dict[str, tuple[int, int]]:
        out = {}
        for a in self.names:
            d = self.move_delta(a)
            if d:
                out[a] = d
        return out

    def agent_pos(self, grid=None) -> Optional[tuple[int, int]]:
        grid = self.grid if grid is None else grid
        if grid is None or self.agent_value is None:
            return None
        return unique_positions(grid).get(self.agent_value)

    def is_passable(self, v: Any) -> Optional[bool]:
        if v == self.agent_value:
            return True
        if v in self.hard_wall:
            return False
        if self.passed[v] > 0 and self.blocked[v] == 0:
            return True
        if self.blocked[v] > 0 and self.passed[v] == 0:
            return False
        if self.passed[v] > 0 and self.blocked[v] > 0:
            return None if self.blocked[v] < 3 else False
        return None  # unknown

    def note(self, s: str) -> None:
        self.history.append(s)
        if len(self.history) > 60:
            del self.history[:-60]

    # -- observe a transition ---------------------------------------------
    def observe(self, action: str, before: list[list[Any]], after: list[list[Any]],
                score_before: float, score_after: float) -> dict[str, Any]:
        """Update hypotheses from one transition; returns an event summary."""
        base = action.split()[0]
        ev: dict[str, Any] = {"action": action, "moved": None, "changed": 0, "score": score_after - score_before}
        if score_after > score_before:
            self.level_changes += 1
            ev["level_up"] = True
            self.note(f"{action}: SCORE +{score_after - score_before:g} (level completed) -> new map")
            self.grid = after
            self.portals.clear()
            return ev
        diff = grid_diff(before, after)
        ev["changed"] = len(diff)
        ub, ua = unique_positions(before), unique_positions(after)
        # agent detection: unique value present in both grids that moved
        movers = [(v, ub[v], ua[v]) for v in ub if v in ua and ub[v] != ua[v]]
        if self.agent_value is None and len(movers) == 1:
            self.agent_value = movers[0][0]
        p0 = ub.get(self.agent_value) if self.agent_value is not None else None
        p1 = ua.get(self.agent_value) if self.agent_value is not None else None
        other = [d for d in diff if (d[0], d[1]) not in (p0, p1)]
        if p0 and p1 and p0 != p1:
            d = (p1[0] - p0[0], p1[1] - p0[1])
            ev["moved"] = d
            self.deltas[base][d] += 1
            landed = after[p0[1]][p0[0]]  # value revealed where the agent stood
            self.passed[landed] += 1
            if max(abs(d[0]), abs(d[1])) > 1:
                # a jump: the cell we stepped into (by the usual delta) is special
                md = self.move_delta(base)
                if md:
                    tx, ty = p0[0] + md[0], p0[1] + md[1]
                    if 0 <= ty < len(before) and 0 <= tx < len(before[0]):
                        v = before[ty][tx]
                        self.special[v]["jump"] += 1
                        self.passed[v] += 1
                        self.portals[(tx, ty)] = p1
                        # hypothesis: landing on a cell of the same value => paired, symmetric portal
                        if before[p1[1]][p1[0]] == v and p1 not in self.portals:
                            self.portals[p1] = (tx, ty)
                        self.note(f"{action}: stepping on {self.name(v)} at ({tx},{ty}) jumped agent to {p1}")
                else:
                    self.note(f"{action}: agent jumped {p0}->{p1}")
            else:
                self.note(f"{action}: moved {p0}->{p1}")
                if p1 in self.portals:          # refuted: entering this cell did not jump
                    del self.portals[p1]
                    self.note(f"  (cell {p1} did not teleport after all)")
        elif p0 and p1 and p0 == p1:
            md = self.move_delta(base)
            if md and not other:
                tx, ty = p0[0] + md[0], p0[1] + md[1]
                if 0 <= ty < len(before) and 0 <= tx < len(before[0]):
                    v = before[ty][tx]
                    self.blocked[v] += 1
                    if self.blocked[v] >= 3 and self.passed[v] == 0:
                        self.hard_wall.add(v)
                    self.note(f"{action}: did NOT move; blocked by {self.name(v)} at ({tx},{ty})")
                else:
                    self.note(f"{action}: did NOT move (grid edge)")
                ev["blocked"] = True
            elif not other:
                self.noeffect[base] += 1
                self.note(f"{action}: no visible effect at {p0}")
            if md:
                self.deltas[base][(0, 0)] += 1
        elif not diff:
            self.noeffect[base] += 1
            self.note(f"{action}: no visible effect")
        if other:
            self.other_change[base] += 1
            vanished = Counter(d[2] for d in other if d[3] != d[2])
            for v, n in vanished.items():
                if v != self.agent_value:
                    self.special[v]["vanish"] += n
            # the world changed: passability beliefs may be stale (keys, toggles)
            for v in list(self.blocked):
                if v not in self.hard_wall:
                    self.blocked[v] = 0
            ex = ", ".join(f"({d[0]},{d[1]}) {self.name(d[2])}->{self.name(d[3])}" for d in other[:6])
            self.note(f"{action}: {len(other)} other cell(s) changed: {ex}")
        self.grid = after
        return ev

    def name(self, v: Any) -> str:
        n = self.legend.get(str(v))
        return f"{glyph(v)}({n})" if n else glyph(v)

    # -- planning ---------------------------------------------------------
    def path_to(self, targets: set, grid=None) -> Optional[list[str]]:
        """Dijkstra from the agent to any target cell over believed-passable
        cells (unknown values cost more). Returns an action list or None."""
        grid = self.grid if grid is None else grid
        start = self.agent_pos(grid)
        mm = self.move_map()
        if not start or not mm or not targets:
            return None
        h, w = len(grid), len(grid[0])
        dist = {start: 0}
        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        pq = [(0, start)]
        while pq:
            d, p = heapq.heappop(pq)
            if p in targets:
                acts = []
                while p != start:
                    p, a = prev[p]
                    acts.append(a)
                return acts[::-1]
            if d > dist.get(p, 1e9):
                continue
            for a, (dx, dy) in mm.items():
                q = (p[0] + dx, p[1] + dy)
                if not (0 <= q[0] < w and 0 <= q[1] < h):
                    continue
                v = grid[q[1]][q[0]]
                ps = self.is_passable(v)
                if ps is False and q not in targets:
                    continue
                cost = 1 if ps else (3 if self.special[v]["jump"] == 0 else 2)
                if q in self.portals and q not in targets:
                    q = self.portals[q]          # stepping in lands somewhere else
                    cost = 1
                nd = d + cost
                if nd < dist.get(q, 1e9):
                    dist[q] = nd
                    prev[q] = (p, a)
                    heapq.heappush(pq, (nd, q))
        return None

    def open_cells(self, strict: bool = True, grid=None) -> set:
        """Cells whose 4 orthogonal neighbours are in-bounds and passable (strict:
        known passable; else merely not believed blocked) — good spots to test an
        action whose direction is still unknown."""
        grid = self.grid if grid is None else grid
        h, w = len(grid), len(grid[0])
        out = set()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self.is_passable(grid[y][x]) is False:
                    continue
                ok = True
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ps = self.is_passable(grid[y + dy][x + dx])
                    if ps is False or (strict and ps is not True):
                        ok = False
                        break
                if ok:
                    out.add((x, y))
        return out

    def cells_with(self, v: Any, grid=None) -> set:
        grid = self.grid if grid is None else grid
        return {(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c == v}

    def knowledge_text(self) -> str:
        lines = []
        for a in self.names:
            c = self.deltas.get(a)
            parts = []
            if c:
                for d, n in c.most_common():
                    parts.append(f"{'no move' if d == (0, 0) else f'delta {d}'} x{n}")
            if self.noeffect[a]:
                parts.append(f"no visible effect x{self.noeffect[a]}")
            if self.other_change[a]:
                parts.append(f"changed other cells x{self.other_change[a]}")
            if a in self.param_actions:
                parts.append("takes arguments")
            lines.append(f"  {a}: {', '.join(parts) if parts else 'never tried'}")
        vals = set(self.passed) | set(self.blocked) | set(self.special)
        vlines = []
        for v in sorted(vals, key=lambda z: str(z)):
            bits = []
            if self.passed[v]:
                bits.append(f"walked on x{self.passed[v]}")
            if self.blocked[v]:
                bits.append(f"blocked movement x{self.blocked[v]}")
            if v in self.hard_wall:
                bits.append("WALL")
            for k, n in self.special[v].items():
                bits.append(f"{k} x{n}")
            vlines.append(f"  {self.name(v)}: {', '.join(bits)}")
        mm = self.move_map()
        dirs = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}
        mm_txt = ", ".join(f"{a}={dirs.get(d, d)}" for a, d in mm.items()) or "none yet"
        return ("Action table (learned; (dx,dy) with y growing downwards):\n" + "\n".join(lines) +
                f"\nMovement map: {mm_txt}\n" +
                ("Cell values:\n" + "\n".join(vlines) if vlines else ""))


# ----------------------------------------------------------------- harness
class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.n_acts = 0
        self.bad_actions = 0

    # -- primitives ---------------------------------------------------------
    def _score(self, game) -> float:
        try:
            return float(game.score)
        except Exception:
            return 0.0

    def step(self, game, w: World, action: str) -> dict[str, Any]:
        ctx = self.ctx
        before = w.grid
        s0 = self._score(game)
        base = action.split()[0]
        if base in w.param_actions and len(action.split()) < 2:
            w.note(f"{action}: rejected by harness (this action needs arguments)")
            return {"action": action, "invalid": True, "moved": None, "changed": 0}
        # what cell would a known move step into? (to confirm the goal value on level-up)
        p0, md = w.agent_pos(), w.move_delta(base)
        tgt_val = None
        if p0 and md and before is not None:
            tx, ty = p0[0] + md[0], p0[1] + md[1]
            if 0 <= ty < len(before) and 0 <= tx < len(before[0]):
                tgt_val = before[ty][tx]
        try:
            obs = ctx.act(action)
        except (ValueError, TypeError, IndexError, KeyError) as e:   # game rejected the action string
            w.note(f"{action}: REJECTED by the game ({str(e)[:80]})")
            ctx.log(event="bad_action", action=action, error=str(e)[:200])
            self.bad_actions += 1
            return {"action": action, "invalid": True, "moved": None, "changed": 0}
        self.n_acts += 1
        s1 = self._score(game)
        _, grid = find_grid(obs)
        if grid is None:
            grid = before
        w.legend = find_legend(obs) or w.legend
        self.last_obs = obs
        ev = w.observe(action, before, grid, s0, s1)
        if ev.get("level_up") and tgt_val is not None:
            ev["target_value"] = tgt_val
            if w.goal_value is None:
                w.goal_value = tgt_val
            self._confirm_goal(w, tgt_val)
        return ev

    def can_think(self) -> bool:
        try:
            return self.ctx.budget.remaining()["system2_calls"] >= 1
        except Exception:
            return True

    def think(self, prompt: str) -> Optional[str]:
        if not self.can_think():
            return None
        try:
            res = self.ctx.think(prompt, session_id=self.session)
        except RuntimeError as e:
            self.ctx.log(event="think_failed", error=str(e))
            return None
        self.session = res.session_id
        return res.text or ""

    # -- probing ------------------------------------------------------------
    def probe(self, game, w: World, actions: list[str]) -> None:
        mem = self.ctx.memory
        prior = mem.get("action_stats", {})
        order = sorted(actions, key=lambda a: -(prior.get(a, {}).get("moved", 0) - prior.get(a, {}).get("none", 0)))
        fail_pos: dict[str, set] = defaultdict(set)
        for a in order:
            if self.ctx.done or a in w.param_actions:
                continue
            p = w.agent_pos()
            ev = self.step(game, w, a)
            if ev.get("level_up"):
                return
            if not ev.get("moved") and not ev.get("changed"):
                fail_pos[a].add(p)
        # second pass: actions with no visible effect the first time, retried from a
        # cell where every neighbour is passable (so "blocked" cannot masquerade as
        # "no effect"); one retry per action, detours capped at 4 moves
        for a in order:
            if self.ctx.done or a in w.param_actions:
                return
            if w.deltas.get(a) and any(d != (0, 0) for d in w.deltas[a]):
                continue
            if w.other_change[a]:
                continue
            if fail_pos[a] & w.open_cells(strict=True):
                continue          # failed where every neighbour is known passable: genuinely no effect there
            for _retry in range(1):
                p = w.agent_pos()
                if p is None:
                    break
                spots = None
                for strict in (True, False):
                    cand = w.open_cells(strict=strict) - fail_pos[a]
                    if cand:
                        spots = cand
                        break
                if spots is not None and p not in spots:
                    path = w.path_to(spots)
                    if path and len(path) <= 4:
                        why = self.execute(game, w, path, verify=True)
                        if why in ("level_up", "done"):
                            return
                    elif not path:
                        # nothing reachable: just step somewhere new
                        for m, d in w.move_map().items():
                            tx, ty = p[0] + d[0], p[1] + d[1]
                            if 0 <= ty < len(w.grid) and 0 <= tx < len(w.grid[0]) and \
                                    w.is_passable(w.grid[ty][tx]) and (tx, ty) not in fail_pos[a]:
                                ev = self.step(game, w, m)
                                if ev.get("level_up"):
                                    return
                                if ev.get("moved"):
                                    break
                p = w.agent_pos()
                if p in fail_pos[a]:
                    break
                ev = self.step(game, w, a)
                if ev.get("level_up"):
                    return
                if ev.get("moved") or ev.get("changed"):
                    break
                fail_pos[a].add(p)

    # -- astra --------------------------------------------------------------
    def prompt(self, game, w: World, reason: str) -> str:
        obs = self.last_obs
        head = scalar_fields(obs, w.grid_key)
        mem = self.ctx.memory
        lessons = mem.get("lessons", [])[-6:]
        agent = w.agent_pos()
        parts = [game.describe(), "Actions:", game.action_spec.describe(), ""]
        parts.append("You are the planner of a harness that already ran experiments. Facts learned THIS episode:")
        parts.append(w.knowledge_text())
        if w.agent_value is not None:
            parts.append(f"The controllable piece is value {w.name(w.agent_value)}, drawn as '@' on the map"
                         f"{f' at (x={agent[0]},y={agent[1]})' if agent else ''}.")
        else:
            parts.append("No single moving piece has been identified yet.")
        if w.goal_value is not None:
            parts.append(f"Assumed goal value: {w.name(w.goal_value)} (cells {sorted(w.cells_with(w.goal_value))[:6]}).")
        if lessons:
            parts.append("Lessons from earlier episodes (rules are re-drawn per episode, only methods transfer):")
            parts.extend(f"  - {l}" for l in lessons)
        parts.append("Recent events (oldest first):")
        parts.extend(f"  {h}" for h in w.history[-18:])
        parts.append(f"Other observation fields: {json.dumps(head, default=str)}")
        if w.legend:
            parts.append("Legend (glyph=value: meaning): " + ", ".join(f"{glyph(int(k)) if k.isdigit() else k}={v}" for k, v in w.legend.items()))
        parts.append("Map (columns x=0.. across the top, rows y=0.. down the side):")
        parts.append(render(w.grid, agent))
        parts.append("")
        parts.append(f"Why you are asked: {reason}")
        parts.append(
            "Reason about the hidden rules (walls, keys/doors, toggles, teleports, context-dependent actions), then "
            f"reply with ONE JSON object and nothing else:\n"
            '{"goal_value": <grid value that completes the level, or null>, '
            f'"actions": ["ACTIONk", ... up to {MAX_PLAN} actions, executed in order; the harness stops early when '
            'two consecutive actions change nothing or the level changes], '
            '"expect": "<one line: what these actions should do>", '
            '"lesson": "<one general sentence about how to discover rules in this game, for future episodes>"}\n'
            "Prefer experiments that resolve the biggest uncertainty (e.g. walk to an unexplained coloured cell, or use "
            "an untested action next to one) when no route to the goal is known.")
        return "\n".join(parts)

    def ask_astra(self, game, w: World, reason: str) -> tuple[dict[str, Any], list[str]]:
        text = self.think(self.prompt(game, w, reason))
        if text is None:
            return {}, []
        js, seq = parse_plan(text, w.names)
        gv = js.get("goal_value") if js else None
        if gv is not None and w.grid is not None:
            try:
                gvi = int(gv) if not isinstance(gv, str) or gv.strip().lstrip("-").isdigit() else gv
            except Exception:
                gvi = gv
            if w.cells_with(gvi):
                w.goal_value = gvi
        lesson = (js.get("lesson") or "").strip() if js else ""
        if lesson:
            ls = self.ctx.memory.setdefault("lessons", [])
            if lesson not in ls:
                ls.append(lesson)
                del ls[:-10]
        self.ctx.log(event="astra_plan", actions=seq, expect=(js.get("expect") if js else None), goal=w.goal_value)
        return js, seq

    def execute(self, game, w: World, seq: list[str], verify: bool = False) -> str:
        """Run a plan; returns why it stopped: done|level_up|refuted|finished|empty."""
        if not seq:
            return "empty"
        dud = 0
        for a in seq:
            if self.ctx.done or self.n_acts >= MAX_ACTS:
                return "done"
            exp = w.move_delta(a.split()[0]) if verify else None
            ev = self.step(game, w, a)
            if ev.get("level_up"):
                return "level_up"
            if ev.get("invalid"):
                dud += 1
                if dud >= 2 or self.bad_actions > 20:
                    return "refuted"
                continue
            if verify and exp and ev.get("moved") != exp:
                return "refuted"
            if ev.get("moved") is None and ev.get("changed", 0) == 0:
                dud += 1
                if dud >= 2:
                    return "refuted"
            else:
                dud = 0
        return "finished"

    def explore_step(self, game, w: World, actions: list[str]) -> None:
        """Model-free fallback: prefer moves toward unvisited cells, else any action."""
        p = w.agent_pos()
        mm = w.move_map()
        if p and mm and w.grid is not None:
            # move to the least-visited neighbour that is not believed blocked
            best = None
            for a, d in mm.items():
                q = (p[0] + d[0], p[1] + d[1])
                if not (0 <= q[1] < len(w.grid) and 0 <= q[0] < len(w.grid[0])):
                    continue
                v = w.grid[q[1]][q[0]]
                if w.is_passable(v) is False:
                    continue
                k = (self.visits[q], self.rng_i)
                if best is None or k < best[0]:
                    best = (k, a)
            self.rng_i += 1
            if best:
                self.step(game, w, best[1])
                return
        cand = [a for a in actions if a not in w.param_actions]
        if not cand:
            # only parameterised actions exist: sweep the grid with arguments
            h, wd = (len(w.grid), len(w.grid[0])) if w.grid else (1, 1)
            a = f"{actions[self.rng_i % len(actions)]} {(self.rng_i * 7) % wd} {(self.rng_i * 3) % h}"
        else:
            a = cand[self.rng_i % len(cand)]
        self.rng_i += 1
        self.step(game, w, a)

    # -- episode --------------------------------------------------------------
    def run_episode(self, game) -> None:
        ctx = self.ctx
        names = list(game.action_spec.names)
        w = World(names, dict(getattr(game.action_spec, "descriptions", {}) or {}))
        obs = game.observe()
        self.last_obs = obs
        w.grid_key, w.grid = find_grid(obs)
        w.legend = find_legend(obs)
        self.visits: Counter = Counter()
        self.rng_i = 0
        mem = ctx.memory
        actions = find_available(obs, names)
        if w.grid is None:
            # no grid at all: pure Astra loop on the raw observation
            self._no_grid_loop(game, names)
            return
        # priors from earlier seeds (validated below before use)
        gp = mem.get("goal_values", {})
        if gp:
            gv = max(gp.items(), key=lambda kv: kv[1])[0]
            try:
                gv = int(gv)
            except Exception:
                pass
            if w.cells_with(gv):
                w.goal_value = gv
        ap = mem.get("agent_values", {})
        if ap:
            av = max(ap.items(), key=lambda kv: kv[1])[0]
            try:
                av = int(av)
            except Exception:
                pass
            if av in unique_positions(w.grid):
                w.agent_value = av

        self.probe(game, w, actions)
        if w.goal_value is None and w.grid is not None:
            try:
                w.goal_value = goal_from_text(game.describe(), w.legend, w.grid, w.agent_value)
            except Exception:
                w.goal_value = None
            if w.goal_value is not None:
                w.note(f"goal hypothesis from the game description: {w.name(w.goal_value)}")
        ctx.log(event="probe_done", move_map={a: list(d) for a, d in w.move_map().items()},
                agent=w.agent_value, n_acts=self.n_acts)

        stuck_asks = 0
        last_ask_key = None
        reason = "initial plan after scripted probing"
        seen: Counter = Counter()
        while not ctx.done and self.n_acts < MAX_ACTS:
            p = w.agent_pos()
            self.visits[p] += 1
            skey = (p, json.dumps(w.grid, default=str))
            seen[skey] += 1
            looping = seen[skey] >= 3
            # 1) autopilot: known goal + movement map + path
            if looping:
                reason = ("the automatic path-follower keeps returning to this same state (loop), "
                          "so the map model is wrong somewhere; propose an experiment or a different route")
            elif w.goal_value is not None and p is not None:
                targets = w.cells_with(w.goal_value)
                path = w.path_to(targets)
                if path:
                    tgt_val = w.goal_value
                    why = self.execute(game, w, path, verify=True)
                    if why == "level_up":
                        reason = "new level after autopilot success"
                        stuck_asks = 0
                    elif why == "refuted":
                        continue  # replan with updated passability
                    elif why == "finished":
                        reason = "autopilot reached the goal cell but the level did not complete"
                    continue
                reason = ("no route to the goal through cells believed passable "
                          f"(blocked values: {[w.name(v) for v in w.blocked if w.blocked[v]]})")
            elif w.goal_value is None:
                reason = "goal not identified yet" if p is not None else "controllable piece and goal unknown"
            # 2) Astra plan (budget permitting); avoid re-asking on an identical stuck state
            key = (json.dumps(w.grid, default=str), w.goal_value, reason)
            if self.can_think() and not (key == last_ask_key and stuck_asks >= 2):
                last_ask_key = key
                js, seq = self.ask_astra(game, w, reason)
                if not seq:
                    stuck_asks += 1
                    self.explore_step(game, w, actions)
                    continue
                before_grid = w.grid
                why = self.execute(game, w, seq, verify=False)
                if why == "level_up":
                    stuck_asks = 0
                    reason = "new level after Astra's plan completed the level"
                elif why == "refuted":
                    stuck_asks += 1
                    reason = "your last plan was cut short: two consecutive actions changed nothing"
                elif w.grid == before_grid:
                    stuck_asks += 1
                    reason = "your last plan changed nothing on the map"
                else:
                    stuck_asks = 0
                    reason = "previous plan finished; decide the next steps"
                continue
            # 3) no budget: scripted exploration
            self.explore_step(game, w, actions)
        self._save_memory(w)

    def _confirm_goal(self, w: World, gv: Any) -> None:
        gvs = self.ctx.memory.setdefault("goal_values", {})
        gvs[str(gv)] = gvs.get(str(gv), 0) + 1

    def _save_memory(self, w: World) -> None:
        mem = self.ctx.memory
        st = mem.setdefault("action_stats", {})
        for a in w.names:
            d = st.setdefault(a, {"moved": 0, "none": 0})
            d["moved"] += sum(n for k, n in w.deltas.get(a, {}).items() if k != (0, 0))
            d["none"] += w.noeffect[a]
        if w.agent_value is not None:
            av = mem.setdefault("agent_values", {})
            av[str(w.agent_value)] = av.get(str(w.agent_value), 0) + 1
        mem["episodes"] = mem.get("episodes", 0) + 1
        self.ctx.log(event="memory", keys=list(mem.keys()), levels=w.level_changes)

    def _no_grid_loop(self, game, names: list[str]) -> None:
        ctx = self.ctx
        hist: list[str] = []
        obs = self.last_obs
        i = 0
        while not ctx.done and self.n_acts < MAX_ACTS:
            if self.can_think():
                prompt = (f"{game.describe()}\nActions:\n{game.action_spec.describe()}\n\nHistory:\n" +
                          "\n".join(hist[-20:]) + f"\n\nObservation:\n{json.dumps(obs, default=str)[:6000]}\n\n"
                          f'Reply with ONE JSON object: {{"actions": [up to {MAX_PLAN} action strings], "expect": "..."}}')
                text = self.think(prompt)
                _, seq = parse_plan(text or "", names)
            else:
                seq = [names[i % len(names)]]
                i += 1
            for a in seq or [names[0]]:
                if ctx.done:
                    break
                s0 = self._score(game)
                obs = ctx.act(a)
                self.n_acts += 1
                hist.append(f"{a}: score {s0}->{self._score(game)}")


def build(ctx):
    return Harness(ctx)
