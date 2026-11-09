"""Exact-transition belief controller for partially observed MiniGrid.

The map is kept in the initial frame and searched over position and heading.
Forward transitions use the public MiniGrid rule directly (a visible
traversable front cell advances one cell); rendered-view similarity is only a
diagnostic. Astra is reserved for unsupported language or contradictory
evidence and every proposed action is checked before it is posted.
"""
from __future__ import annotations

import json
import re
from collections import Counter, deque

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
COLORS = ("red", "green", "blue", "purple", "yellow", "grey", "gray")
OBJECTS = ("goal", "ball", "box", "key", "door")
ACTIONS = {"left", "right", "forward", "pickup", "drop", "toggle", "done"}
ACT_RE = re.compile(r"\b(left|right|forward|pickup|drop|toggle|done)\b", re.I)


def decode(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def cells(obs):
    out = {}
    grid = obs.get("grid", []) if isinstance(obs, dict) else []
    for row in grid if isinstance(grid, list) else []:
        for c in row if isinstance(row, list) else []:
            if isinstance(c, dict) and "x" in c and "y" in c:
                try:
                    p = (int(c["x"]), int(c["y"]))
                except Exception:
                    continue
                out[p] = {k: c[k] for k in ("object", "color", "state") if k in c}
    return out


def sig(c):
    return (c.get("object"), c.get("color"), c.get("state"))


def add(p, d):
    return (p[0] + d[0], p[1] + d[1])


def walkable(c):
    return c.get("object") in ("empty", "floor", "goal", "agent") or (
        c.get("object") == "door" and c.get("state") == "open")


def json_object(text):
    raw = str(text or "")
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(raw[i:])
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = {}
        self.world = {}
        self.pos = (0, 0)
        self.heading = 0
        self.initial_direction = 0
        self.inventory = None
        self.contract = None
        self.manual = deque()
        self.history = deque(maxlen=10)
        self.revision = 0
        self.last_discovery = 0
        self.steps = 0
        self.illegal = 0
        self.repeat = Counter()
        self.seen = Counter()
        self.need_review = None
        self.allowed = set(ACTIONS)

    def transform(self, rel, pos=None, heading=None):
        pos = self.pos if pos is None else pos
        heading = self.heading if heading is None else heading
        fwd, right = DIRS[heading], DIRS[(heading + 1) % 4]
        return (pos[0] + rel[0] * right[0] + rel[1] * fwd[0],
                pos[1] + rel[0] * right[1] + rel[1] * fwd[1])

    def ingest(self, obs):
        self.obs = decode(obs)
        try:
            absolute = int(self.obs.get("direction", self.initial_direction))
        except Exception:
            absolute = self.initial_direction
        self.heading = (absolute - self.initial_direction) % 4
        self.inventory = self.obs.get("inventory")
        changed = 0
        for rel, cell in cells(self.obs).items():
            if cell.get("object") == "unknown":
                continue
            p = self.transform(rel)
            stored = {"object": "empty"} if rel == (0, 0) else cell
            if sig(stored) != sig(self.world.get(p, {})):
                changed += 1
            self.world[p] = stored
        if changed:
            self.revision += 1
            self.last_discovery = self.steps
        self.seen[((self.pos[0], self.pos[1], self.heading), self.revision)] += 1

    def parse_contract(self):
        mission = str(self.obs.get("mission", "")).lower().strip().rstrip(".")
        env = str(self.obs.get("environment_id", ""))
        if mission in ("get to the green goal square", "get to the goal",
                       "traverse the rooms to get to the goal",
                       "use the key to open the door and then get to the goal"):
            return {"kind": "reach", "object": "goal", "color": None, "location": None}
        if env.startswith("MiniGrid-") and "goal" in mission and not any(
                w in mission for w in ("pick", "open", "door", "key")):
            return {"kind": "reach", "object": "goal", "color": None, "location": None}
        m = re.fullmatch(r"(go to|pick up|open) (?:the|a) "
                         r"(?:(red|green|blue|purple|yellow|grey|gray) )?"
                         r"(ball|box|key|door)"
                         r"(?: (on your left|on your right|in front of you|behind you))?", mission)
        if m and (env.startswith("BabyAI-") or not env):
            verb, color, obj, location = m.groups()
            return {"kind": {"go to": "goto", "pick up": "pickup", "open": "open"}[verb],
                    "object": obj, "color": color, "location": location}
        return None

    def matches(self, p, c):
        spec = self.contract or {}
        if c.get("object") != spec.get("object"):
            return False
        want = spec.get("color")
        have = str(c.get("color", "")).replace("gray", "grey")
        if want and have != want.replace("gray", "grey"):
            return False
        loc = spec.get("location")
        return not ((loc == "on your left" and p[0] >= 0) or
                    (loc == "on your right" and p[0] <= 0) or
                    (loc == "in front of you" and p[1] <= 0) or
                    (loc == "behind you" and p[1] >= 0))

    def can_toggle(self, c):
        if c.get("object") == "box":
            return True
        if c.get("object") != "door":
            return False
        if c.get("state") == "closed":
            return True
        inv = self.inventory if isinstance(self.inventory, dict) else {}
        return c.get("state") == "locked" and inv.get("object") == "key" and inv.get("color") == c.get("color")

    def search(self):
        """BFS over observed pose, held key color, removed keys and opened doors."""
        spec = self.contract or {}
        keys = {p: c for p, c in self.world.items() if c.get("object") == "key"}
        doors = {p: c for p, c in self.world.items() if c.get("object") == "door"}
        needed_colors = {c.get("color") for c in doors.values() if c.get("state") == "locked"}
        if spec.get("kind") == "open" and spec.get("color"):
            needed_colors.add(spec.get("color"))
        ki = {p: i for i, p in enumerate(keys)}
        di = {p: i for i, p in enumerate(doors)}
        inv = self.inventory if isinstance(self.inventory, dict) else {}
        held = inv.get("color", "!") if inv.get("object") == "key" else ("!" if inv else "")
        start = (*self.pos, self.heading, held, 0, 0)
        q = deque([start]); parent = {start: None}; action_for = {}
        projected = {}

        def route(state):
            out = []
            while parent[state] is not None:
                out.append(action_for[state]); state = parent[state]
            out.reverse(); return out

        while q and len(parent) < 30000 and not self.ctx.done:
            state = q.popleft(); x, y, h, held, removed, opened = state
            projected[(x, y, h)] = state
            front = add((x, y), DIRS[h]); cell = dict(self.world.get(front, {}))
            k = ki.get(front); d = di.get(front)
            if k is not None and removed & (1 << k): cell = {"object": "empty"}
            if d is not None and opened & (1 << d): cell["state"] = "open"
            typ = cell.get("object")
            can_open = typ == "door" and (cell.get("state") == "closed" or
                       cell.get("state") == "locked" and held == cell.get("color"))
            if spec.get("kind") == "reach" and self.matches((x, y), self.world.get((x, y), {})):
                return route(state), projected
            if self.matches(front, cell):
                if spec.get("kind") == "goto":
                    return route(state), projected
                if spec.get("kind") == "pickup" and not held and typ in ("key", "ball", "box"):
                    return route(state) + ["pickup"], projected
                if spec.get("kind") == "open" and can_open:
                    return route(state) + ["toggle"], projected
            edges = []
            if walkable(cell): edges.append(((*front, h, held, removed, opened), "forward"))
            edges += [((x, y, (h - 1) % 4, held, removed, opened), "left"),
                      ((x, y, (h + 1) % 4, held, removed, opened), "right")]
            if can_open and d is not None:
                edges.append(((x, y, h, held, removed, opened | (1 << d)), "toggle"))
            if (k is not None and not held and not removed & (1 << k)
                    and (spec.get("kind") not in ("reach", "open") or
                         cell.get("color") in needed_colors)):
                edges.append(((x, y, h, cell.get("color"), removed | (1 << k), opened), "pickup"))
            for dest, act in edges:
                if dest not in parent and act in self.allowed:
                    parent[dest] = state; action_for[dest] = act; q.append(dest)
        paths = {p: route(s) for p, s in projected.items()}
        return None, paths

    def frontier_action(self, paths):
        choices = []
        for pose, path in paths.items():
            if not path or self.seen[(pose, self.revision)]:
                continue
            # Prefer a cheap pose that exposes unknown cells, with a small
            # bonus for a visible closed door (room transitions are informative).
            gain = 0
            for x in range(-3, 4):
                for y in range(0, 7):
                    p = self.transform((x, y), pose[:2], pose[2])
                    if p not in self.world: gain += 1
            front = self.world.get(add(pose[:2], DIRS[pose[2]]), {})
            if front.get("object") == "door" and front.get("state") in ("closed", "locked"):
                gain += 5
            if gain: choices.append((len(path) / (1 + gain), len(path), path))
        if choices:
            return min(choices, key=lambda z: (z[0], z[1]))[2][0]
        self.need_review = "No known route to target or informative frontier."
        return None

    def legal(self, action):
        if action not in self.allowed: return False
        front = cells(self.obs).get((0, 1), {})
        if action == "forward": return walkable(front)
        if action == "pickup": return not self.inventory and front.get("object") in ("key", "ball", "box")
        if action == "drop": return bool(self.inventory) and front.get("object") in ("empty", "floor")
        if action == "toggle": return self.can_toggle(front)
        return action in ("left", "right", "done")

    def ask(self, game, reason):
        try:
            rem = self.ctx.budget.remaining()
            if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 150: return False
            payload = {"mission": self.obs.get("mission"), "environment_id": self.obs.get("environment_id"),
                       "direction": self.obs.get("direction"), "inventory": self.inventory,
                       "grid": self.obs.get("grid"), "reason": reason, "recent": list(self.history),
                       "progress": self.ctx.progress(), "contract": self.contract,
                       "pose": [*self.pos, self.heading],
                       "known_cells": [[*p, c] for p, c in sorted(self.world.items())]}
            prompt = (game.describe() + "\nReturn JSON only for a MiniGrid recovery/semantic check: "
                      "{contract:{kind:reach|goto|pickup|open,object:goal|ball|box|key|door,color:null|red|green|blue|purple|yellow|grey,location:null|on your left|on your right|in front of you|behind you},actions:[...]}. "
                      "Preserve ordered missions; use manual:true with a short action list when no single contract is valid. "
                      "Unknown cells are unsafe. The executor checks every action. STATE=" + json.dumps(payload, separators=(",", ":"), default=str))
            result = self.ctx.think(prompt, timeout_s=min(240.0, float(rem.get("wall_s", 240.0)) - 5.0))
            data = json_object(getattr(result, "text", ""))
            c = data.get("contract")
            if isinstance(c, dict) and c.get("kind") in ("reach", "goto", "pickup", "open") and c.get("object") in OBJECTS and c.get("color") in (None,) + COLORS and c.get("location") in (None, "on your left", "on your right", "in front of you", "behind you"):
                self.contract = c
            if data.get("manual") is True: self.contract = None
            raw = data.get("actions", [])
            if isinstance(raw, str): raw = ACT_RE.findall(raw.lower())
            if isinstance(raw, list): self.manual.extend(a.lower() for a in raw[:12] if isinstance(a, str) and a.lower() in self.allowed)
            if data.get("reset_map") is True:
                self.world.clear(); self.ingest(self.obs)
            self.need_review = None; self.repeat.clear(); self.seen.clear()
            self.ctx.log("replan_result", parsed=data)
            return bool(self.contract or self.manual)
        except Exception as exc:
            self.ctx.log("replan_failed", error=type(exc).__name__)
            return False

    def step_checked(self, action, reason):
        before = self.obs; front = cells(before).get((0, 1), {}); old_inv = self.inventory
        old_pos, old_heading = self.pos, self.heading
        try:
            result = self.ctx.act(action)
        except Exception as exc:
            if self.ctx.done: return False
            self.need_review = "Action exception: " + type(exc).__name__; self.manual.clear(); return True
        self.steps += 1
        new = decode(result if result is not None else self.ctx.observe())
        # Exact action transition: MiniGrid's public rule says forward advances
        # iff the front cell was traversable. Do not infer displacement from a
        # symmetric rendered view; that was the source of false pose resets.
        if action == "forward" and walkable(front): self.pos = add(old_pos, DIRS[old_heading])
        if action in ("left", "right"):
            expected = (old_heading + (-1 if action == "left" else 1)) % 4
            try:
                actual = (int(new.get("direction", self.initial_direction + old_heading)) - self.initial_direction) % 4
                if actual != expected: self.need_review = "Turn heading contradicted by observation."
            except Exception: self.need_review = "Malformed direction after turn."
        self.ingest(new)
        prog = self.ctx.progress()
        if prog.get("illegal_actions", 0) > self.illegal: self.need_review = "ctx.progress reports rejected action."
        self.illegal = int(prog.get("illegal_actions", 0) or 0)
        after = cells(self.obs).get((0, 1), {})
        if action in ("pickup", "drop") and self.inventory == old_inv: self.need_review = "Inventory did not change after interaction."
        if action == "toggle" and sig(after) == sig(front): self.need_review = "Toggle did not change facing object."
        if action == "done" and not self.ctx.done: self.need_review = "Done did not terminate."
        fp = (self.pos, self.heading, str(self.inventory), self.revision); self.repeat[fp] += 1
        if self.repeat[fp] >= 3: self.need_review = "Repeated pose/map state indicates a cycle."
        if prog.get("since_obs_change", 0) >= 4: self.need_review = "ctx.progress reports stale observations."
        self.history.append({"action": action, "reason": reason, "pose": [*self.pos, self.heading], "progress": prog, "review": self.need_review})
        self.ctx.log("checked_step", action=action, reason=reason, pose=[*self.pos, self.heading], review=self.need_review, progress=prog)
        if self.need_review: self.manual.clear()
        return not (self.ctx.done or self.obs.get("terminated") or self.obs.get("truncated"))

    def run_episode(self, game):
        self.allowed = set(getattr(game.action_spec, "names", ()) or ACTIONS) & ACTIONS
        self.obs = decode(self.ctx.observe())
        try: self.initial_direction = int(self.obs.get("direction", 0))
        except Exception: self.initial_direction = 0
        self.ingest(self.obs); self.contract = self.parse_contract()
        if self.contract is None: self.need_review = "Mission outside verified grammar."
        while not self.ctx.done and not self.obs.get("terminated") and not self.obs.get("truncated"):
            if self.need_review:
                if not self.ask(game, self.need_review): return
            if self.manual: action, reason = self.manual.popleft(), "Astra recovery prefix"
            elif self.contract:
                route, paths = self.search()
                if route is not None:
                    if route:
                        action, reason = route[0], "exact-transition target route"
                    else:
                        self.need_review = "Compiled target pose is satisfied but episode is not terminal."
                        continue
                else: action, reason = self.frontier_action(paths), "exact-transition exploration"
            else:
                self.need_review = "Recovery prefix ended; need a new verified subgoal."; continue
            if action is None: continue
            if not self.legal(action):
                self.need_review = "Pre-action legality veto for " + str(action); self.manual.clear(); continue
            if not self.step_checked(action, reason): return


def build(ctx):
    return Harness(ctx)
