"""Candidate 0002 — 0001's movement engine + a generic *object mode* for games
without a unique moving cell (click / ARC-style), plus two efficiency fixes.

Why (see proposal.md):
  * 0001 is perfect on the train game (2.0, 0 Astra calls) but scores 0 on the
    held-out click game: it never probes parameterised actions, asks Astra 15
    times with an empty knowledge table and then sweeps clicks blindly over the
    64x64 frame until the hidden per-level click budget kills the game.
  * Its level-2 routes are ~2x the oracle because a teleport is remembered as a
    cell pair (cleared on level-up) instead of as a *value* rule, and it spends
    two actions per episode probing actions after the movement map is complete.

What this harness does
  1. Movement mode (as 0001): scripted probe -> world model -> Dijkstra autopilot
     -> Astra only when stuck.  Changes: the probe stops as soon as four distinct
     unit displacements are known (the rest is probed lazily when stuck); a jump
     that lands on a cell of the same value becomes a value-level portal rule
     ("entering value P moves you to the other P cell") that survives level
     changes and is refuted the first time it fails.
  2. Object mode (new): used when no parameter-less action is available or when
     the probe finds no unique moving cell.  The frame is segmented into
     connected components (objects); cells that change on almost every action
     are tagged "indicator" (step meters etc.) and excluded from effect
     summaries; one cheap probe click per object *class* (colour, size) builds
     an object->effect table; Astra then sees an object table, the effect log,
     the scalar observation fields and a (downsampled) map and returns a JSON
     plan whose items may be raw actions, object ids or coordinates.  Plans stop
     early when two consecutive actions change nothing or the level changes.
  3. Memory across seeds keeps only discovery priors (probe order, responsive
     object classes, Astra "lessons").

Only the standard library is used; nothing is keyed on this game's observation
fields or rules — grid / legend / available actions / scalar fields are
discovered from the observation dict.
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from typing import Any, Optional

MAX_PLAN = 15          # actions Astra may commit per call (movement mode)
MAX_PLAN_OBJ = 8       # actions Astra may commit per call (object mode)
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


def detect_scale(grid: list[list[Any]]) -> int:
    """Largest s in (4,3,2) such that >=97% of s x s blocks are uniform (pixel-art
    frames drawn at an integer zoom); 1 otherwise."""
    h, w = len(grid), len(grid[0])
    if max(h, w) <= 24:
        return 1
    for s in (4, 3, 2):
        if h % s or w % s:
            continue
        bad = tot = 0
        for by in range(0, h, s):
            for bx in range(0, w, s):
                v = grid[by][bx]
                tot += 1
                if any(grid[y][x] != v for y in range(by, by + s) for x in range(bx, bx + s)):
                    bad += 1
        if tot and bad / tot <= 0.06:
            return s
    return 1


def render_scaled(grid: list[list[Any]], s: int) -> str:
    if s <= 1:
        return render(grid)
    h, w = len(grid), len(grid[0])
    small = []
    for by in range(0, h, s):
        row = []
        for bx in range(0, w, s):
            c = Counter(grid[y][x] for y in range(by, by + s) for x in range(bx, bx + s))
            row.append(c.most_common(1)[0][0])
        small.append(row)
    return render(small)


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


# ----------------------------------------------------------------- objects
class Obj:
    __slots__ = ("id", "colour", "x0", "y0", "x1", "y1", "size", "cells")

    def __init__(self, oid, colour, cells):
        self.id = oid
        self.colour = colour
        self.cells = cells
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        self.x0, self.x1, self.y0, self.y1 = min(xs), max(xs), min(ys), max(ys)
        self.size = len(cells)

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def cls(self) -> tuple:
        return (self.colour, self.w, self.h)

    def centre(self) -> tuple[int, int]:
        cx, cy = (self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2
        if (cx, cy) in self.cells:
            return cx, cy
        return min(self.cells, key=lambda c: abs(c[0] - cx) + abs(c[1] - cy))

    def describe(self, legend: dict[str, str]) -> str:
        n = legend.get(str(self.colour))
        col = f"{glyph(self.colour)}({n})" if n else glyph(self.colour)
        cx, cy = self.centre()
        box = f"x{self.x0}-{self.x1},y{self.y0}-{self.y1}" if self.size > 1 else f"x{self.x0},y{self.y0}"
        return f"obj{self.id}: colour {col}, {self.w}x{self.h} box [{box}], {self.size} cells, centre ({cx},{cy})"


def segment(grid: list[list[Any]], ignore: Optional[set] = None, max_objs: int = 400) -> tuple[list[Obj], Any]:
    """4-connected same-colour components, background (most common colour)
    excluded. Returns (objects sorted by size desc, background colour)."""
    h, w = len(grid), len(grid[0])
    cnt = Counter(v for row in grid for v in row)
    bg = cnt.most_common(1)[0][0]
    seen: set = set(ignore or ())
    objs: list[Obj] = []
    for y in range(h):
        for x in range(w):
            if (x, y) in seen or grid[y][x] == bg:
                continue
            col = grid[y][x]
            stack = [(x, y)]
            seen.add((x, y))
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and grid[ny][nx] == col:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            objs.append(Obj(0, col, cells))
            if len(objs) >= max_objs:
                break
        if len(objs) >= max_objs:
            break
    objs.sort(key=lambda o: (-o.size, o.y0, o.x0))
    for i, o in enumerate(objs):
        o.id = i
    return objs, bg


def summarize_diff(diff: list, indicator: set, legend: dict[str, str]) -> tuple[int, str]:
    """Human summary of a cell diff (indicator cells removed)."""
    d = [t for t in diff if (t[0], t[1]) not in indicator]
    if not d:
        return 0, "no change"
    if d[0][0] == -1:
        return len(d), "frame changed shape"
    xs = [t[0] for t in d]
    ys = [t[1] for t in d]
    trans = Counter((t[2], t[3]) for t in d)

    def nm(v):
        n = legend.get(str(v))
        return f"{glyph(v)}({n})" if n else glyph(v)
    tt = ", ".join(f"{nm(a)}->{nm(b)} x{n}" for (a, b), n in trans.most_common(4))
    return len(d), f"{len(d)} cells changed in x{min(xs)}-{max(xs)},y{min(ys)}-{max(ys)}: {tt}"


# ----------------------------------------------------------------- parsing
def _canon(tok: str, names: list[str]) -> Optional[str]:
    parts = tok.strip().replace("(", " ").replace(")", " ").replace(",", " ").split()
    if not parts:
        return None
    up = parts[0].upper()
    name = next((n for n in names if n.upper() == up), None)
    if name is None:
        return None
    args = [p for p in parts[1:] if re.fullmatch(r"-?\d+", p)]
    return " ".join([name] + args[:2])


def extract_json(text: str) -> dict[str, Any]:
    text = text or ""
    for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.S):
        try:
            cand = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(cand, dict) and any(k in cand for k in ("actions", "goal_value", "plan", "clicks")):
            return cand
    return {}


def parse_plan(text: str, names: list[str], max_plan: int = MAX_PLAN) -> tuple[dict[str, Any], list[str]]:
    """JSON object if present; else every action mention in order (with up to 2 ints)."""
    js = extract_json(text)
    seq: list[str] = []
    raw = js.get("actions", js.get("plan", js.get("clicks"))) if js else None
    if isinstance(raw, list):
        for a in raw:
            s = _canon(str(a), names)
            if s:
                seq.append(s)
    if not seq:
        pat = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        for m in re.finditer(rf"({pat})((?:[\s,]+-?\d+){{0,2}})", text or "", flags=re.I):
            s = _canon(m.group(0), names)
            if s:
                seq.append(s)
    return js, seq[:max_plan]


def parse_obj_plan(text: str, names: list[str], param_actions: set, objs: list[Obj],
                   max_plan: int = MAX_PLAN_OBJ) -> tuple[dict[str, Any], list[str]]:
    """Object-mode parser: items may be 'ACTION6 x y', 'ACTION1', 'obj3',
    'click obj 3', {'obj': 3}, {'x': 1, 'y': 2}, [x, y]."""
    js = extract_json(text)
    by_id = {o.id: o for o in objs}
    click = next((n for n in names if n in param_actions), None)
    seq: list[str] = []

    def add_click(x, y):
        if click:
            seq.append(f"{click} {int(x)} {int(y)}")

    def add_obj(i):
        o = by_id.get(int(i))
        if o is not None:
            add_click(*o.centre())

    raw = js.get("actions", js.get("plan", js.get("clicks"))) if js else None
    items = raw if isinstance(raw, list) else []
    for it in items:
        if isinstance(it, dict):
            if "obj" in it or "object" in it or "id" in it:
                try:
                    add_obj(it.get("obj", it.get("object", it.get("id"))))
                except Exception:
                    pass
            elif "x" in it and "y" in it:
                try:
                    add_click(it["x"], it["y"])
                except Exception:
                    pass
            elif "action" in it:
                s = _canon(str(it["action"]) + " " + " ".join(str(it.get(k, "")) for k in ("x", "y")), names)
                if s:
                    seq.append(s)
            continue
        if isinstance(it, (list, tuple)) and len(it) == 2:
            try:
                add_click(it[0], it[1])
            except Exception:
                pass
            continue
        s = str(it).strip()
        m = re.match(r"^(?:click\s*)?(?:obj(?:ect)?)\s*#?\s*(\d+)$", s, re.I)
        if m:
            add_obj(m.group(1))
            continue
        c = _canon(s, names)
        if c:
            seq.append(c)
            continue
        m = re.match(r"^(?:click\s*)?\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?$", s, re.I)
        if m:
            add_click(m.group(1), m.group(2))
    if not seq:
        for m in re.finditer(r"obj(?:ect)?\s*#?\s*(\d+)", text or "", re.I):
            add_obj(m.group(1))
        if not seq:
            _, seq = parse_plan(text, names, max_plan)
    return js, seq[:max_plan]


GOAL_WORDS = re.compile(r"complete|goal|reach|exit|win|target|finish", re.I)


def goal_from_text(describe: str, legend: dict[str, str], grid: list[list[Any]], agent_value: Any) -> Any:
    """Generic prior: a legend name that appears in a sentence of the game
    description that talks about reaching/completing => candidate goal value."""
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
    from the live grid; action/colour knowledge persists across levels)."""

    def __init__(self, names: list[str], spec_desc: dict[str, str]):
        self.names = names
        self.param_actions = {n for n in names if PARAM_RE.search(spec_desc.get(n, "") or "")}
        self.deltas: dict[str, Counter] = defaultdict(Counter)     # action -> Counter[(dx,dy)]
        self.noeffect: Counter = Counter()
        self.other_change: Counter = Counter()
        self.passed: Counter = Counter()
        self.blocked: Counter = Counter()
        self.hard_wall: set = set()
        self.special: dict[Any, Counter] = defaultdict(Counter)
        self.portals: dict[tuple[int, int], tuple[int, int]] = {}   # cell -> landing cell (this level)
        self.value_portal: Counter = Counter()                      # value -> confirmations of "P jumps to the other P"
        self.value_portal_refuted: set = set()
        self.agent_value: Any = None
        self.goal_value: Any = None
        self.history: list[str] = []
        self.grid: Optional[list[list[Any]]] = None
        self.grid_key: Optional[str] = None
        self.legend: dict[str, str] = {}
        self.score = 0.0
        self.level_changes = 0
        self.tried: set = set()

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
        return None

    def note(self, s: str) -> None:
        self.history.append(s)
        if len(self.history) > 60:
            del self.history[:-60]

    def portal_dest(self, q: tuple[int, int], grid=None) -> Optional[tuple[int, int]]:
        """Where stepping into q lands: cell-level edge, else value rule."""
        if q in self.portals:
            return self.portals[q]
        grid = self.grid if grid is None else grid
        v = grid[q[1]][q[0]]
        if self.value_portal[v] > 0 and v not in self.value_portal_refuted:
            cells = self.cells_with(v, grid)
            if len(cells) == 2:
                other = next(c for c in cells if c != q)
                return other
        return None

    # -- observe a transition ---------------------------------------------
    def observe(self, action: str, before: list[list[Any]], after: list[list[Any]],
                score_before: float, score_after: float) -> dict[str, Any]:
        base = action.split()[0]
        self.tried.add(base)
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
            landed = after[p0[1]][p0[0]]
            self.passed[landed] += 1
            if max(abs(d[0]), abs(d[1])) > 1:
                md = self.move_delta(base)
                if md:
                    tx, ty = p0[0] + md[0], p0[1] + md[1]
                    if 0 <= ty < len(before) and 0 <= tx < len(before[0]):
                        v = before[ty][tx]
                        self.special[v]["jump"] += 1
                        self.passed[v] += 1
                        self.portals[(tx, ty)] = p1
                        if before[p1[1]][p1[0]] == v:
                            if p1 not in self.portals:
                                self.portals[p1] = (tx, ty)
                            # value-level rule: entering any P cell moves you to the other P cell
                            if v not in self.value_portal_refuted:
                                self.value_portal[v] += 1
                        self.note(f"{action}: stepping on {self.name(v)} at ({tx},{ty}) jumped agent to {p1}")
                else:
                    self.note(f"{action}: agent jumped {p0}->{p1}")
            else:
                self.note(f"{action}: moved {p0}->{p1}")
                if p1 in self.portals:
                    del self.portals[p1]
                    self.note(f"  (cell {p1} did not teleport after all)")
                if self.value_portal[landed] > 0:
                    # entering a cell of a "portal value" without jumping refutes the value rule
                    self.value_portal_refuted.add(landed)
                    self.note(f"  (value {self.name(landed)} does not always teleport: rule dropped)")
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
                dest = self.portal_dest(q, grid) if q not in targets else None
                if dest is not None:
                    q = dest
                    cost = 1
                nd = d + cost
                if nd < dist.get(q, 1e9):
                    dist[q] = nd
                    prev[q] = (p, a)
                    heapq.heappush(pq, (nd, q))
        return None

    def open_cells(self, strict: bool = True, grid=None) -> set:
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
            if self.value_portal[v] and v not in self.value_portal_refuted:
                bits.append("hypothesis: entering it moves you to the other cell of this value")
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
        self.last_obs = None
        self.available: list[str] = []
        self.obj_mode = False
        self.level_start_grid = None
        self._ind_cache: tuple = (-1, set())
        # object-mode state
        self.cell_changes: Counter = Counter()      # cell -> number of actions that changed it
        self.n_obs_actions = 0                      # actions with a comparable before/after grid
        self.events: list[dict] = []                # object-mode event log (raw diffs kept short)
        self.prompt_objs: list[Obj] = []
        self.class_stats: dict = {}                 # (colour,w,h) -> {"clicks": n, "changed": n, "cells": total}

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
        if self.available and base not in self.available:
            w.note(f"{action}: rejected by harness (not in the available actions {self.available})")
            return {"action": action, "invalid": True, "moved": None, "changed": 0}
        p0, md = w.agent_pos(), w.move_delta(base)
        tgt_val = None
        if p0 and md and before is not None:
            tx, ty = p0[0] + md[0], p0[1] + md[1]
            if 0 <= ty < len(before) and 0 <= tx < len(before[0]):
                tgt_val = before[ty][tx]
        try:
            obs = ctx.act(action)
        except (ValueError, TypeError, IndexError, KeyError) as e:
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
        self.available = find_available(obs, w.names)
        n_hist = len(w.history)
        ev = w.observe(action, before, grid, s0, s1)
        if self.obj_mode:
            del w.history[n_hist:]          # object mode writes its own, indicator-free notes
        # object-mode bookkeeping (cheap; also useful evidence for the movement prompt)
        diff = grid_diff(before, grid) if not ev.get("level_up") else []
        if diff and diff[0][0] != -1:
            self.n_obs_actions += 1
            for t in diff:
                self.cell_changes[(t[0], t[1])] += 1
        elif not ev.get("level_up") and before is not None and grid is not None:
            self.n_obs_actions += 1
        self.events.append({"action": action, "diff": diff[:3000], "level_up": bool(ev.get("level_up")),
                            "score": s1, "fields": scalar_fields(obs, w.grid_key)})
        if len(self.events) > 80:
            del self.events[:-80]
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

    # -- indicator cells ------------------------------------------------------
    def indicator(self) -> set:
        """Cells that are noise for rule discovery: (a) cells that changed on
        (almost) every observed action, (b) whole rows/columns in which almost
        every action changed a *few* cells (a step meter / timer bar sweeping
        along a line changes different cells each time)."""
        n = self.n_obs_actions
        if n < 3:
            return set()
        if self._ind_cache[0] == len(self.events):
            return self._ind_cache[1]
        out = {c for c, k in self.cell_changes.items() if k >= max(3, 0.6 * n)}
        evs = [e for e in self.events if e["diff"] and e["diff"][0][0] != -1]
        grid = self.last_grid()
        if grid is not None and len(evs) >= 3:
            h, w = len(grid), len(grid[0])
            row_hits: Counter = Counter()
            col_hits: Counter = Counter()
            for e in evs:
                rows = Counter(t[1] for t in e["diff"])
                cols = Counter(t[0] for t in e["diff"])
                for r, k in rows.items():
                    if k <= max(4, 0.1 * w):
                        row_hits[r] += 1
                for c, k in cols.items():
                    if k <= max(4, 0.1 * h):
                        col_hits[c] += 1
            thr = max(3, 0.6 * len(evs))
            for r, k in row_hits.items():
                if k >= thr:
                    out |= {(x, r) for x in range(w)}
            for c, k in col_hits.items():
                if k >= thr:
                    out |= {(c, y) for y in range(h)}
        self._ind_cache = (len(self.events), out)
        return out

    def last_grid(self):
        return self._w.grid if getattr(self, "_w", None) is not None else None

    def indicator_text(self, w: World) -> str:
        ind = self.indicator()
        if not ind:
            return ""
        rows = Counter(c[1] for c in ind)
        cols = Counter(c[0] for c in ind)
        h, wd = len(w.grid), len(w.grid[0])
        full_rows = sorted(r for r, k in rows.items() if k == wd)
        full_cols = sorted(c for c, k in cols.items() if k == h)
        bits = []
        if full_rows:
            bits.append("row(s) " + ",".join(map(str, full_rows)))
        if full_cols:
            bits.append("column(s) " + ",".join(map(str, full_cols)))
        rest = len(ind) - len(full_rows) * wd - len(full_cols) * h
        if rest > 0:
            bits.append(f"{rest} other cells")
        txt = ("Indicator cells (change on almost every action, e.g. a step meter or timer; they are excluded from "
               "the effect summaries below): " + "; ".join(bits))
        comp_now = Counter(w.grid[y][x] for (x, y) in ind)
        line = ", ".join(f"{w.name(v)} x{k}" for v, k in comp_now.most_common(3))
        if self.level_start_grid is not None and len(self.level_start_grid) == h:
            comp0 = Counter(self.level_start_grid[y][x] for (x, y) in ind)
            line0 = ", ".join(f"{w.name(v)} x{k}" for v, k in comp0.most_common(3))
            txt += f"\n  indicator composition now: {line}; at level start: {line0} (if it is a budget meter, plan accordingly)"
        else:
            txt += f"\n  indicator composition now: {line}"
        return txt

    # -- movement probing --------------------------------------------------------
    def probe(self, game, w: World, actions: list[str]) -> None:
        """Try parameter-less actions once each until four distinct unit moves are
        known; retry no-effect actions once from an open cell. Actions that are
        still untried are probed lazily by lazy_probe() when the agent is stuck."""
        mem = self.ctx.memory
        prior = mem.get("action_stats", {})
        order = sorted(actions, key=lambda a: -(prior.get(a, {}).get("moved", 0) - prior.get(a, {}).get("none", 0)))
        fail_pos: dict[str, set] = defaultdict(set)
        for a in order:
            if self.ctx.done or a in w.param_actions:
                continue
            if len(set(w.move_map().values())) >= 4:
                break
            p = w.agent_pos()
            ev = self.step(game, w, a)
            if ev.get("level_up"):
                return
            if not ev.get("moved") and not ev.get("changed"):
                fail_pos[a].add(p)
        if w.agent_value is None:
            return
        for a in order:
            if self.ctx.done or a in w.param_actions or a not in w.tried:
                continue
            if w.deltas.get(a) and any(d != (0, 0) for d in w.deltas[a]):
                continue
            if w.other_change[a]:
                continue
            if fail_pos[a] & w.open_cells(strict=True):
                continue
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
                continue
            ev = self.step(game, w, a)
            if ev.get("level_up"):
                return
            if not (ev.get("moved") or ev.get("changed")):
                fail_pos[a].add(p)

    def lazy_probe(self, game, w: World, actions: list[str]) -> bool:
        """Try one still-untried parameter-less action (called when stuck)."""
        for a in actions:
            if a in w.param_actions or a in w.tried:
                continue
            ev = self.step(game, w, a)
            return True
        return False

    # -- astra (movement mode) ----------------------------------------------------
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
        if self.available and set(self.available) != set(w.names):
            parts.append(f"Currently available actions: {self.available}")
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
        self._store_lesson(js)
        self.ctx.log(event="astra_plan", actions=seq, expect=(js.get("expect") if js else None), goal=w.goal_value)
        return js, seq

    def _store_lesson(self, js: dict) -> None:
        lesson = (js.get("lesson") or "").strip() if js else ""
        if lesson:
            ls = self.ctx.memory.setdefault("lessons", [])
            if lesson not in ls:
                ls.append(lesson)
                del ls[:-10]

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
            self.object_explore_step(game, w, actions)
            return
        a = cand[self.rng_i % len(cand)]
        self.rng_i += 1
        self.step(game, w, a)

    # -- episode --------------------------------------------------------------
    def run_episode(self, game) -> None:
        ctx = self.ctx
        names = list(game.action_spec.names)
        w = World(names, dict(getattr(game.action_spec, "descriptions", {}) or {}))
        self._w = w
        obs = game.observe()
        self.last_obs = obs
        w.grid_key, w.grid = find_grid(obs)
        w.legend = find_legend(obs)
        self.visits: Counter = Counter()
        self.rng_i = 0
        mem = ctx.memory
        self.available = find_available(obs, names)
        actions = list(self.available)
        if w.grid is None:
            self._no_grid_loop(game, names)
            return
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

        nonparam = [a for a in actions if a not in w.param_actions]
        if nonparam:
            self.probe(game, w, actions)
        if w.goal_value is None and w.grid is not None:
            try:
                w.goal_value = goal_from_text(game.describe(), w.legend, w.grid, w.agent_value)
            except Exception:
                w.goal_value = None
            if w.goal_value is not None:
                w.note(f"goal hypothesis from the game description: {w.name(w.goal_value)}")
        ctx.log(event="probe_done", move_map={a: list(d) for a, d in w.move_map().items()},
                agent=w.agent_value, n_acts=self.n_acts, available=self.available)

        if w.agent_value is None or not w.move_map():
            # no unique moving cell: the world is made of objects, not of one agent
            ctx.log(event="mode", mode="object")
            self.object_mode(game, w)
            self._save_memory(w)
            return
        ctx.log(event="mode", mode="movement")
        self.movement_loop(game, w, actions)
        self._save_memory(w)

    def movement_loop(self, game, w: World, actions: list[str]) -> None:
        ctx = self.ctx
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
            if looping:
                reason = ("the automatic path-follower keeps returning to this same state (loop), "
                          "so the map model is wrong somewhere; propose an experiment or a different route")
            elif w.goal_value is not None and p is not None:
                targets = w.cells_with(w.goal_value)
                path = w.path_to(targets)
                if path:
                    why = self.execute(game, w, path, verify=True)
                    if why == "level_up":
                        reason = "new level after autopilot success"
                        stuck_asks = 0
                    elif why == "refuted":
                        continue
                    elif why == "finished":
                        reason = "autopilot reached the goal cell but the level did not complete"
                    continue
                reason = ("no route to the goal through cells believed passable "
                          f"(blocked values: {[w.name(v) for v in w.blocked if w.blocked[v]]})")
            elif w.goal_value is None:
                reason = "goal not identified yet" if p is not None else "controllable piece and goal unknown"
            # cheap first: an action we have never tried may be the key (interact/toggle)
            if not looping and self.lazy_probe(game, w, actions):
                continue
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
            self.explore_step(game, w, actions)

    # -- object mode --------------------------------------------------------------
    def _objects(self, w: World) -> tuple[list[Obj], Any]:
        ind = self.indicator()
        objs, bg = segment(w.grid, ignore=ind)
        return objs, bg

    def _obj_short(self, o: Obj, w: World) -> str:
        box = f"x{o.x0}-{o.x1},y{o.y0}-{o.y1}" if o.size > 1 else f"x{o.x0},y{o.y0}"
        return f"colour-{w.name(o.colour)} {o.w}x{o.h} object at [{box}]"

    def _class_key(self, o: Obj) -> str:
        return f"{o.colour}:{o.w}x{o.h}"

    def _record_click(self, o: Obj, changed: int) -> None:
        st = self.class_stats.setdefault(self._class_key(o), {"clicks": 0, "changed": 0, "cells": 0})
        st["clicks"] += 1
        st["changed"] += 1 if changed else 0
        st["cells"] += changed
        mem = self.ctx.memory.setdefault("class_stats", {})
        m = mem.setdefault(self._class_key(o), {"clicks": 0, "changed": 0})
        m["clicks"] += 1
        m["changed"] += 1 if changed else 0

    def _effect_of_last(self, w: World) -> tuple[int, str]:
        if not self.events:
            return 0, "no change"
        ev = self.events[-1]
        if ev["level_up"]:
            return 1, "LEVEL COMPLETED"
        return summarize_diff(ev["diff"], self.indicator(), w.legend)

    def object_probe(self, game, w: World, objs: list[Obj], budget: int) -> None:
        """One click per object class (colour,w,h) that has not been probed this
        episode; classes that responded before come first, huge regions last."""
        click = next((n for n in self.available if n in w.param_actions), None)
        if click is None:
            return
        total = len(w.grid) * len(w.grid[0])
        prior = self.ctx.memory.get("class_stats", {})
        done_classes: set = set()
        order = sorted(objs, key=lambda o: (o.size > 0.15 * total,
                                            -(prior.get(self._class_key(o), {}).get("changed", 0)),
                                            o.size))
        small_classes = {self._class_key(o) for o in objs if o.size <= 0.15 * total}
        n = 0
        for o in order:
            if self.ctx.done or n >= budget:
                break
            k = self._class_key(o)
            if k in done_classes or k in self.class_stats:
                continue
            if o.size > 0.15 * total and len(small_classes) >= 3:
                continue          # background-like region; only worth a click when little else exists
            done_classes.add(k)
            cx, cy = o.centre()
            ev = self.step(game, w, f"{click} {cx} {cy}")
            n += 1
            changed, summ = self._effect_of_last(w)
            self._record_click(o, changed)
            w.note(f"{click} {cx} {cy} on {self._obj_short(o, w)}: {summ}")
            if ev.get("level_up") or self.ctx.done:
                return

    def object_prompt(self, game, w: World, objs: list[Obj], bg: Any, reason: str) -> str:
        obs = self.last_obs
        head = scalar_fields(obs, w.grid_key)
        lessons = self.ctx.memory.get("lessons", [])[-6:]
        h, wd = len(w.grid), len(w.grid[0])
        parts = [game.describe(), "Actions:", game.action_spec.describe(), ""]
        parts.append("You are the planner of a harness that already ran experiments. The harness could not find a single "
                     "moving 'agent' cell, so it describes the frame as objects (connected same-colour components).")
        parts.append(f"Grid {wd} wide x {h} high (x to the right, y downwards). Background colour: {w.name(bg)}.")
        if self.available:
            parts.append(f"Currently available actions: {self.available}")
        itxt = self.indicator_text(w)
        if itxt:
            parts.append(itxt)
        if w.legend:
            parts.append("Legend: " + ", ".join(f"{glyph(int(k)) if k.isdigit() else k}={v}" for k, v in w.legend.items()))
        parts.append(f"Objects now ({len(objs)}; largest first; ids are valid for THIS reply only):")
        shown = objs[:45]
        for o in shown:
            st = self.class_stats.get(self._class_key(o))
            tag = ""
            if st:
                tag = f"  [class clicked x{st['clicks']}: {st['changed']} responded]"
            parts.append("  " + o.describe(w.legend) + tag)
        if len(objs) > len(shown):
            parts.append(f"  ... and {len(objs) - len(shown)} smaller objects")
        parts.append("Action table so far:")
        for a in w.names:
            bits = []
            if w.noeffect[a]:
                bits.append(f"no visible effect x{w.noeffect[a]}")
            if w.other_change[a]:
                bits.append(f"changed cells x{w.other_change[a]}")
            if w.deltas.get(a):
                bits.append("moved a unique cell")
            if a in w.param_actions:
                bits.append("takes x y")
            if a not in self.available:
                bits.append("NOT available now")
            parts.append(f"  {a}: {', '.join(bits) if bits else 'never tried'}")
        if lessons:
            parts.append("Lessons from earlier episodes (only methods transfer):")
            parts.extend(f"  - {l}" for l in lessons)
        parts.append("Experiment log (oldest first):")
        parts.extend(f"  {x}" for x in w.history[-24:])
        parts.append(f"Other observation fields: {json.dumps(head, default=str)}")
        s = detect_scale(w.grid)
        if s > 1:
            parts.append(f"Map, downsampled {s}x (each map cell = a {s}x{s} block; map cell (i,j) is grid x={s}*i..{s}*i+{s - 1}, "
                         f"y={s}*j..{s}*j+{s - 1}; click coordinates must be FULL-resolution grid coordinates or object ids):")
            parts.append(render_scaled(w.grid, s))
        else:
            parts.append("Map (columns x=0.. across the top, rows y=0.. down the side):")
            parts.append(render(w.grid))
        parts.append("")
        parts.append(f"Why you are asked: {reason}")
        parts.append(
            "Think about what kind of puzzle this is (what the objects are, which ones respond, what a completed level "
            "probably looks like), then reply with ONE JSON object and nothing else:\n"
            '{"hypothesis": "<one line: the rule you believe>", '
            '"goal": "<one line: what state completes the level>", '
            f'"actions": [up to {MAX_PLAN_OBJ} items, executed in order; each item is either "ACTIONk" for a parameter-less '
            'action, "ACTIONk x y" with full-resolution coordinates, or "obj<ID>" to click the centre of that object; '
            'the harness stops early when two consecutive actions change nothing or the level changes], '
            '"expect": "<one line: what these actions should change>", '
            '"lesson": "<one general sentence about how to discover rules in this game, for future episodes>"}\n'
            "If the rule is still unclear, choose actions that best discriminate between hypotheses (click an object whose "
            "class never responded only if nothing else is left). Remember the action budget is limited.")
        return "\n".join(parts)

    def object_explore_step(self, game, w: World, actions: list[str]) -> None:
        """No-budget fallback: click responsive object classes in rotation,
        otherwise unprobed objects; parameter-less actions in rotation."""
        click = next((n for n in self.available if n in w.param_actions), None)
        objs, _ = self._objects(w)
        if click and objs:
            def score(o):
                st = self.class_stats.get(self._class_key(o))
                if st is None:
                    return (1, o.size)
                if st["changed"] == 0:
                    return (2, o.size)
                return (0, -st["cells"] / max(1, st["clicks"]))
            cand = sorted(objs, key=score)
            top = [o for o in cand if score(o)[0] == 0] or cand
            o = top[self.rng_i % len(top)]
            self.rng_i += 1
            cx, cy = o.centre()
            self.step(game, w, f"{click} {cx} {cy}")
            changed, summ = self._effect_of_last(w)
            self._record_click(o, changed)
            w.note(f"{click} {cx} {cy} on {self._obj_short(o, w)} (scripted exploration): {summ}")
            return
        cand = [a for a in self.available if a not in w.param_actions] or [a for a in actions if a not in w.param_actions]
        if cand:
            a = cand[self.rng_i % len(cand)]
            self.rng_i += 1
            self.step(game, w, a)
        elif click:
            h, wd = len(w.grid), len(w.grid[0])
            self.step(game, w, f"{click} {(self.rng_i * 7) % wd} {(self.rng_i * 3) % h}")
            self.rng_i += 1

    def object_mode(self, game, w: World) -> None:
        ctx = self.ctx
        self.obj_mode = True
        self._w = w
        self.level_start_grid = w.grid
        level_id = 0
        probed_level = -1
        reason = "initial plan after scripted object probing"
        stuck = 0
        last_key = None
        seen: Counter = Counter()
        while not ctx.done and self.n_acts < MAX_ACTS:
            if w.level_changes != level_id:
                level_id = w.level_changes
                reason = "new level after the last plan completed the previous one"
                stuck = 0
                self.level_start_grid = w.grid
                self.class_stats = {}          # objects may mean something else now; re-probe cheaply
            objs, bg = self._objects(w)
            if probed_level != level_id:
                probed_level = level_id
                # parameter-less actions first (one each), then one click per object class
                for a in self.available:
                    if ctx.done:
                        break
                    if a not in w.param_actions and a not in w.tried:
                        self.step(game, w, a)
                        _, summ = self._effect_of_last(w)
                        w.note(f"{a}: {summ}")
                budget = 10 if level_id == 0 else 6
                self.object_probe(game, w, objs, budget)
                objs, bg = self._objects(w)
                if ctx.done or w.level_changes != level_id:
                    continue
            skey = json.dumps(w.grid, default=str)
            seen[skey] += 1
            if seen[skey] >= 4:
                reason = "the same frame keeps recurring (loop); the hypothesis must be wrong — try something different"
            key = (skey, reason)
            if self.can_think() and not (key == last_key and stuck >= 2):
                last_key = key
                self.prompt_objs = objs
                text = self.think(self.object_prompt(game, w, objs, bg, reason))
                if text is None:
                    self.object_explore_step(game, w, w.names)
                    continue
                js, seq = parse_obj_plan(text, w.names, w.param_actions, objs)
                self._store_lesson(js)
                ctx.log(event="astra_obj_plan", actions=seq, hypothesis=(js.get("hypothesis") if js else None),
                        goal=(js.get("goal") if js else None), expect=(js.get("expect") if js else None))
                if not seq:
                    stuck += 1
                    reason = "your last reply contained no executable action; reply with the JSON format requested"
                    continue
                before = w.grid
                why = self.execute_obj(game, w, seq, objs)
                if why == "level_up":
                    stuck = 0
                elif why == "refuted":
                    stuck += 1
                    reason = "your last plan was cut short: two consecutive actions changed nothing (indicator cells excluded)"
                elif w.grid == before:
                    stuck += 1
                    reason = "your last plan changed nothing on the frame"
                else:
                    stuck = 0
                    reason = "previous plan finished; decide the next steps"
                continue
            self.object_explore_step(game, w, w.names)

    def execute_obj(self, game, w: World, seq: list[str], objs: list[Obj]) -> str:
        dud = 0
        by_cell = {}
        for o in objs:
            for c in o.cells:
                by_cell[c] = o
        for a in seq:
            if self.ctx.done or self.n_acts >= MAX_ACTS:
                return "done"
            ev = self.step(game, w, a)
            if ev.get("level_up"):
                w.note(f"{a}: LEVEL COMPLETED")
                return "level_up"
            if ev.get("invalid"):
                dud += 1
                if dud >= 2 or self.bad_actions > 20:
                    return "refuted"
                continue
            changed, summ = self._effect_of_last(w)
            parts = a.split()
            if len(parts) == 3:
                o = by_cell.get((int(parts[1]), int(parts[2])))
                if o is not None:
                    self._record_click(o, changed)
                    w.note(f"{a} on {self._obj_short(o, w)}: {summ}")
                else:
                    w.note(f"{a} on background: {summ}")
            else:
                w.note(f"{a}: {summ}")
            if changed == 0:
                dud += 1
                if dud >= 2:
                    return "refuted"
            else:
                dud = 0
        return "finished"

    # -- memory ---------------------------------------------------------------
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
                try:
                    obs = ctx.act(a)
                except (ValueError, TypeError, IndexError, KeyError) as e:
                    hist.append(f"{a}: rejected ({str(e)[:60]})")
                    continue
                self.n_acts += 1
                hist.append(f"{a}: score {s0}->{self._score(game)}")


def build(ctx):
    return Harness(ctx)
