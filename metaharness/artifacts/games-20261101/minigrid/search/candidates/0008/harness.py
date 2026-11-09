"""Astra selects information objectives; verified joint search executes them."""
from __future__ import annotations

import json
import re
from collections import Counter, deque

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
COLORS = ("red", "green", "blue", "purple", "yellow", "grey", "gray")
OBJECTS = ("goal", "ball", "box", "key", "door")
ACTIONS = {"left", "right", "forward", "pickup", "drop", "toggle", "done"}
ACT_RE = re.compile(r"\b(left|right|forward|pickup|drop|toggle|done)\b", re.I)


def as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def visible(obs):
    out = {}
    grid = obs.get("grid", []) if isinstance(obs, dict) else []
    for row in grid if isinstance(grid, list) else []:
        for cell in row if isinstance(row, list) else []:
            if not isinstance(cell, dict) or "x" not in cell or "y" not in cell:
                continue
            try:
                p = (int(cell["x"]), int(cell["y"]))
            except (TypeError, ValueError):
                continue
            out[p] = {k: cell[k] for k in ("object", "color", "state") if k in cell}
    return out


def sig(cell):
    return (cell.get("object"), cell.get("color"), cell.get("state"))


def add(p, d):
    return (p[0] + d[0], p[1] + d[1])


def walkable(cell):
    return cell.get("object") in ("empty", "floor", "goal", "agent") or (
        cell.get("object") == "door" and cell.get("state") == "open")


def extract_json(text):
    raw = str(text or "")
    for i, char in enumerate(raw):
        if char != "{":
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
        self.seen = Counter()
        self.revision = 0
        self.steps = 0
        self.last_revision_step = 0
        self.last_illegal = 0
        self.model_calls = 0
        self.last_reason = "initial state"
        self.need_review = None
        self.repeat = Counter()
        self.last_review_step = 0
        self.search_limited = False
        self.explore_route = deque()
        self.explore_calls = 0
        self.explore_label = None
        self.explore_start_revision = 0
        self.explore_endpoint = None
        self.door_options = {}
        self.explore_results = deque(maxlen=5)
        self.pending_exploration = None

    def transform(self, rel, pos=None, heading=None):
        pos = self.pos if pos is None else pos
        heading = self.heading if heading is None else heading
        forward, right = DIRS[heading], DIRS[(heading + 1) % 4]
        return (pos[0] + rel[0] * right[0] + rel[1] * forward[0],
                pos[1] + rel[0] * right[1] + rel[1] * forward[1])

    def ingest(self, obs):
        self.obs = as_dict(obs)
        try:
            absolute = int(self.obs.get("direction", self.initial_direction))
        except (TypeError, ValueError):
            absolute = self.initial_direction
        self.heading = (absolute - self.initial_direction) % 4
        self.inventory = self.obs.get("inventory")
        changed = 0
        for rel, cell in visible(self.obs).items():
            if cell.get("object") == "unknown":
                continue
            absolute_pos = self.transform(rel)
            # The agent square can encode the carried object in some wrappers.
            stored = {"object": "empty"} if rel == (0, 0) else cell
            if sig(stored) != sig(self.world.get(absolute_pos, {})):
                changed += 1
            self.world[absolute_pos] = stored
        if changed:
            self.revision += 1
            self.last_revision_step = self.steps
        self.seen[(self.pos, self.heading, self.revision)] += 1

    def parse_local_contract(self):
        mission = str(self.obs.get("mission", "")).lower().strip().rstrip(".")
        env_id = str(self.obs.get("environment_id", ""))
        if mission in ("get to the green goal square", "get to the goal",
                       "traverse the rooms to get to the goal",
                       "use the key to open the door and then get to the goal"):
            return {"kind": "reach", "object": "goal", "color": None, "location": None}
        m = re.fullmatch(
            r"(go to|pick up|open) (?:the|a) "
            r"(?:(red|green|blue|purple|yellow|grey|gray) )?"
            r"(ball|box|key|door)"
            r"(?: (on your left|on your right|in front of you|behind you))?", mission)
        if m and (env_id.startswith("BabyAI-") or not env_id):
            verb, color, obj, location = m.groups()
            return {"kind": {"go to": "goto", "pick up": "pickup", "open": "open"}[verb],
                    "object": obj, "color": color, "location": location}
        return None

    def contract_valid(self, contract):
        if not isinstance(contract, dict) or contract.get("kind") not in ("reach", "goto", "pickup", "open"):
            return False
        if contract.get("object") not in OBJECTS:
            return False
        if contract.get("color") not in (None,) + COLORS:
            return False
        return contract.get("location") in (None, "on your left", "on your right", "in front of you", "behind you")

    def matches(self, position, cell):
        spec = self.contract or {}
        if cell.get("object") != spec.get("object"):
            return False
        if spec.get("color") and str(cell.get("color")).replace("gray", "grey") != spec["color"].replace("gray", "grey"):
            return False
        location = spec.get("location")
        # Qualifiers are frozen in the initial reference frame.
        if location == "on your left" and position[0] >= 0:
            return False
        if location == "on your right" and position[0] <= 0:
            return False
        if location == "in front of you" and position[1] <= 0:
            return False
        if location == "behind you" and position[1] >= 0:
            return False
        return True

    def can_toggle(self, cell):
        if cell.get("object") == "box":
            return True
        if cell.get("object") != "door":
            return False
        if cell.get("state") in ("closed", "open"):
            return cell.get("state") == "closed"
        if cell.get("state") == "locked":
            inv = self.inventory if isinstance(self.inventory, dict) else {}
            return inv.get("object") == "key" and inv.get("color") == cell.get("color")
        return False

    def search(self):
        """BFS over pose, held key, removed keys and opened doors.

        Every edge is ONE public action. A hypothetical pickup removes the key
        from its square; unlocking keeps it in hand. No unobserved cell is an
        edge. This jointly prices key selection, approach and subsequent travel.
        """
        self.door_options = {}
        spec = self.contract or {}
        kind = spec.get("kind")
        locked_colors = {c.get("color") for c in self.world.values()
                         if c.get("object") == "door" and c.get("state") == "locked"}
        keys = {p: c for p, c in self.world.items() if c.get("object") == "key"
                and c.get("color") in locked_colors and kind in ("reach", "open")}
        doors = {p: c for p, c in self.world.items() if c.get("object") == "door"}
        key_index = {p: i for i, p in enumerate(keys)}
        door_index = {p: i for i, p in enumerate(doors)}
        inv = self.inventory if isinstance(self.inventory, dict) else {}
        held = inv.get("color", "!") if inv.get("object") == "key" else ("!" if inv else "")
        start = (*self.pos, self.heading, held, 0, 0)
        queue = deque([start])
        parents = {start: None}
        projected = {}
        costs = {start: 0}
        self.search_limited = False

        def route(state, last=None):
            actions = [] if last is None else [last]
            while parents[state] is not None:
                state, action = parents[state]
                actions.append(action)
            actions.reverse()
            return actions

        while queue:
            if len(parents) >= 30000 or self.ctx.done:
                self.search_limited = True
                break
            state = queue.popleft()
            x, y, h, held, removed, opened = state
            cost = costs[state]
            pose = (x, y, h)
            if pose not in projected:
                projected[pose] = state
            front_pos = add((x, y), DIRS[h])
            cell = self.world.get(front_pos, {})
            ki = key_index.get(front_pos)
            di = door_index.get(front_pos)
            if ki is not None and removed & (1 << ki):
                cell = {"object": "empty"}
            if di is not None and opened & (1 << di):
                cell = dict(cell, state="open")
            typ = cell.get("object")
            can_open = typ == "door" and (cell.get("state") == "closed" or
                       (cell.get("state") == "locked" and held == cell.get("color")))
            if kind == "reach" and self.matches((x, y), self.world.get((x, y), {})):
                return route(state), {}, "target"
            if self.matches(front_pos, cell):
                if kind == "goto":
                    return route(state), {}, "target"
                if kind == "pickup" and not held and typ in ("key", "ball", "box"):
                    return route(state, "pickup"), {}, "target"
                if kind == "open" and can_open:
                    return route(state, "toggle"), {}, "target"
            if can_open and di is not None:
                candidate = route(state, "toggle")
                prev = self.door_options.get(front_pos)
                if prev is None or len(candidate) < len(prev["path"]):
                    self.door_options[front_pos] = {
                        "kind": "open door to observe beyond it", "path": candidate,
                        "pose": pose, "door": [*front_pos, cell], "gain": None}
            edges = []
            if walkable(cell):
                edges.append(((*front_pos, h, held, removed, opened), "forward"))
            # Orientation is part of the state and turns each cost one step.
            edges.extend([((x, y, (h - 1) % 4, held, removed, opened), "left"),
                          ((x, y, (h + 1) % 4, held, removed, opened), "right")])
            if can_open and di is not None:
                edges.append(((x, y, h, held, removed, opened | (1 << di)), "toggle"))
            if ki is not None and not held and not removed & (1 << ki):
                edges.append(((x, y, h, cell.get("color"), removed | (1 << ki), opened), "pickup"))
            for dest, action in edges:
                if dest not in parents and action in self.allowed:
                    parents[dest] = (state, action)
                    costs[dest] = cost + 1
                    queue.append(dest)
        return None, {pose: (costs[state], route(state)) for pose, state in projected.items()}, "frontier"

    def gain(self, pose):
        """Visibility estimate for ranking views only, never permission to move."""
        result = 0.0
        origin, heading = pose[:2], pose[2]
        spec = self.contract or {}
        for x in range(-3, 4):
            for y in range(0, 7):
                target = self.transform((x, y), origin, heading)
                if target in self.world:
                    continue
                dx, dy = target[0] - origin[0], target[1] - origin[1]
                n = max(abs(dx), abs(dy))
                blocked = False
                for i in range(1, n):
                    point = (round(origin[0] + dx * i / n), round(origin[1] + dy * i / n))
                    cell = self.world.get(point, {})
                    if cell.get("object") == "wall" or (cell.get("object") == "door" and cell.get("state") != "open"):
                        blocked = True
                        break
                if blocked:
                    continue
                location = spec.get("location")
                eligible_region = ((location == "behind you" and target[1] < 0) or
                                   (location == "in front of you" and target[1] > 0) or
                                   (location == "on your left" and target[0] < 0) or
                                   (location == "on your right" and target[0] > 0))
                result += (2 if eligible_region else 1) / (1 + .2 * (abs(x) + y))
        return result

    def map_text(self):
        # Every glyph comes from the observed belief map; ? means unobserved.
        xs = [p[0] for p in self.world] + [self.pos[0]]
        ys = [p[1] for p in self.world] + [self.pos[1]]
        xmin, xmax = max(min(xs), self.pos[0]-16), min(max(xs), self.pos[0]+16)
        ymin, ymax = max(min(ys), self.pos[1]-16), min(max(ys), self.pos[1]+16)
        glyph = {"wall": "#", "empty": ".", "floor": ".", "goal": "G",
                 "door": "D", "key": "K", "ball": "O", "box": "B", "lava": "!"}
        rows = ["x columns " + str(list(range(xmin, xmax+1)))]
        for y in range(ymax, ymin-1, -1):
            chars = []
            for x in range(xmin, xmax+1):
                chars.append("@" if (x,y) == self.pos else
                             glyph.get(self.world.get((x,y), {}).get("object"), "?"))
            rows.append(str(y) + " " + "".join(chars))
        return "\n".join(rows)

    def exploration_menu(self, paths):
        options = []
        views = []
        for pose, (cost, path) in paths.items():
            if not path or self.seen[(pose[:2], pose[2], self.revision)]:
                continue
            gain = self.gain(pose)
            if gain <= .1:
                continue
            opt = {"kind": "look from pose", "pose": pose, "path": path,
                   "gain": round(gain, 2), "value": (cost+1)/(1+min(gain,14))}
            views.append(opt)
        views.sort(key=lambda o: (o["value"], len(o["path"]), o["pose"]))
        # Include semantic transitions even when generic ray gain underrates
        # them; the model chooses between these and ordinary observation poses.
        for opt in sorted(self.door_options.values(), key=lambda o: len(o["path"]))[:5]:
            options.append(opt)
        # Offer all useful in-place headings. A 180-degree look is one objective,
        # so a low-yield intermediate turn cannot derail a useful rear scan.
        for opt in views:
            if opt["pose"][:2] == self.pos and len(opt["path"]) <= 2:
                options.append(opt)
        # Diversity: two similar endpoints do not crowd every alternative out.
        sectors = Counter()
        for opt in views:
            if any(o["path"] == opt["path"] for o in options):
                continue
            sector = (opt["path"][0], opt["pose"][2])
            if sectors[sector] >= 2:
                continue
            sectors[sector] += 1
            options.append(opt)
            if len(options) >= 12:
                break
        # Deterministic fallback retains the parent's frontier objective.
        fallback = 0
        if views:
            for i, opt in enumerate(options):
                if opt["path"] == views[0]["path"]:
                    fallback = i
                    break
            else:
                options.append(views[0])
                fallback = len(options)-1
        return options, fallback

    def choose(self):
        route, paths, why = self.search()
        if route is not None:
            self.explore_route.clear()
            self.explore_label = None
            if route:
                return route[0], "joint target route"
            self.need_review = "Compiled target pose is satisfied but environment is not terminal. Check verifier semantics."
            return None, "unconfirmed target"
        if self.explore_route:
            return self.explore_route.popleft(), "exploration objective: " + str(self.explore_label)
        if self.explore_label is not None:
            result = {"objective": self.explore_label,
                      "discovered": self.revision > self.explore_start_revision,
                      "reached": [*self.pos, self.heading], "expected": self.explore_endpoint}
            self.explore_results.append(result)
            self.ctx.log("exploration_completed", **result)
            self.explore_label = None
        options, fallback = self.exploration_menu(paths)
        if not options:
            self.need_review = ("No reachable informative view or target; inventory, locked door or blocker may require a new subgoal. "
                                + ("Search state limit reached." if self.search_limited else ""))
            return None, "no route"
        pick = self.choose_exploration(options, fallback)
        chosen = options[pick]
        self.explore_route = deque(chosen["path"])
        self.explore_label = {"kind": chosen["kind"], "pose": chosen["pose"], "door": chosen.get("door")}
        self.explore_endpoint = chosen["pose"]
        self.explore_start_revision = self.revision
        self.ctx.log("exploration_selected", choice=pick, objective=self.explore_label,
                     cost=len(chosen["path"]), calls=self.explore_calls)
        return self.explore_route.popleft(), "exploration objective: " + str(self.explore_label)

    def choose_exploration(self, options, fallback):
        """Astra selects an endpoint; paths remain entirely code-generated."""
        try:
            b = self.ctx.budget.remaining()
            # Reserve calls and time for actual contradictions or semantic repair.
            if (len(options) == 1 or self.explore_calls >= 6 or
                    b.get("system2_calls", 0) <= 6 or b.get("wall_s", 0) < 300):
                return fallback
            menu = []
            for i, opt in enumerate(options):
                menu.append({"id": i, "kind": opt["kind"], "cost": len(opt["path"]),
                             "end_pose": opt["pose"], "rough_gain": opt["gain"],
                             "door": opt.get("door"),
                             "requires_pickup": "pickup" in opt["path"]})
            objects = [[*p, c] for p,c in sorted(self.world.items())
                       if c.get("object") not in ("empty", "wall", "floor")]
            state = {"mission": self.obs.get("mission"), "contract": self.contract,
                     "pose": [*self.pos, self.heading], "inventory": self.inventory,
                     "map": self.map_text(), "objects": objects,
                     "recent_objectives": list(self.explore_results),
                     "steps_left": self.obs.get("remaining_steps"), "options": menu}
            prompt = (
                "Choose the best information-gathering objective for a partially observed MiniGrid/BabyAI task. "
                "Return only JSON {\"choice\":integer,\"reason\":\"short phrase\"}. "
                "The target is not yet reachable in the known map. Code has computed each safe route, "
                "including required key pickups and toggles, and will verify every step. You choose only an ID. "
                "Minimize expected TOTAL actions to mission success. Cheap scans are useful if they can reveal "
                "the target; opening an observed door can reveal a whole new room. Rough gain is a noisy "
                "geometric estimate, not a value function: it may count impossible cells beyond walls. "
                "Weigh the exact mission and room structure against route cost. Do not assume a target behind "
                "every door, or choose a long route just for more gain. Relative language refers to the INITIAL "
                "pose (0,0); +x initial-right, +y initial-forward. End heading 0=+y,1=+x,2=-y,3=-x. "
                "Map rows are labeled y decreasing upward-to-downward; x columns are listed. "
                "# wall, . floor, ? unknown, D door, K key, O ball, B box, G goal, ! lava, @ current agent. "
                "A useful intermediate discovery can preempt your objective. Never invent hidden cells. "
                "No route arithmetic or action list is needed. STATE=" + json.dumps(state, separators=(",", ":")))
            self.explore_calls += 1
            self.model_calls += 1
            # Backend queue time can outlive the requested model timeout. Bound
            # the optional wait at the harness boundary as well. ctx owns this
            # future and cancels an outstanding model request on episode end.
            self.pending_exploration = self.ctx.spawn(
                lambda: self.ctx.think(prompt, session_id=None, timeout_s=150.0))
            ans = self.pending_exploration.result(timeout=180.0)
            self.pending_exploration = None
            data = extract_json(getattr(ans, "text", ""))
            val = data.get("choice")
            if isinstance(val, str) and val.isdigit():
                val = int(val)
            valid = isinstance(val, int) and not isinstance(val, bool) and 0 <= val < len(options)
            self.ctx.log("exploration_answer", parsed=data, valid=valid, fallback=fallback, menu=menu)
            return val if valid else fallback
        except Exception as exc:
            # Do not pile another optional call behind one that is still queued.
            # The running request is inference only, with no action side effects.
            if self.pending_exploration is not None:
                self.explore_calls = 6
                try:
                    self.pending_exploration.cancel()
                except Exception:
                    pass
                self.pending_exploration = None
            self.ctx.log("exploration_choice_failed", error=type(exc).__name__, fallback=fallback)
            return fallback

    def ask_contract(self, game, reason):
        if self.ctx.done:
            return False
        try:
            budget = self.ctx.budget.remaining()
            if budget.get("system2_calls", 0) <= 0 or budget.get("wall_s", 0) < 150:
                return False
            payload = {
                "mission": self.obs.get("mission"),
                "environment_id": self.obs.get("environment_id"),
                "direction": self.obs.get("direction"),
                "inventory": self.inventory,
                "grid": self.obs.get("grid"),
                "reason": reason,
                "recent": list(self.history),
                "progress": self.ctx.progress(),
                "contract": self.contract,
                "frame": "INITIAL agent at (0,0), x initial-right, y initial-forward. Heading 0=+y,1=+x,2=-y,3=-x.",
                "pose": [*self.pos, self.heading],
                "known_cells": [[*p, c] for p, c in sorted(self.world.items())],
                "remaining_steps": self.obs.get("remaining_steps"),
            }
            prompt = (game.describe() + "\nYou are a semantic contract normalizer for a synchronous MiniGrid episode. "
                      "Use only this public state. Return JSON only: {contract:{kind:reach|goto|pickup|open, "
                      "object:goal|ball|box|key|door,color:null|red|green|blue|purple|yellow|grey,location:null|on your left|on your right|in front of you|behind you},actions:[...]}. "
                      "A reach contract enters the goal; goto faces the adjacent object; pickup requires inventory transition. "
                      "Location qualifiers always refer to the initial frame. Observation front=(0,1). "
                      "Preserve all mission constraints. For ordered/composite missions return manual:true and a short actions prefix, "
                      "not a simplified contract. You may return reset_map:true for inconsistent map evidence. "
                      "Only use observed cells. Unknown is not empty. Return at most 140 words. "
                      "The executor verifies every action. STATE=" + json.dumps(payload, separators=(",", ":"), default=str))
            result = self.ctx.think(prompt, session_id=None, timeout_s=min(240.0, float(budget.get("wall_s", 240.0)) - 5.0))
            self.model_calls += 1
            data = extract_json(getattr(result, "text", ""))
            usable = False
            previous_contract = self.contract
            if data.get("manual") is True:
                self.contract = None
            if self.contract_valid(data.get("contract")) and data.get("manual") is not True:
                self.contract = data["contract"]
                usable = previous_contract != self.contract
            raw = data.get("actions", [])
            if isinstance(raw, str):
                raw = ACT_RE.findall(raw.lower())
            if isinstance(raw, list):
                self.manual.extend(a.lower() for a in raw[:12] if isinstance(a, str) and a.lower() in ACTIONS)
            usable = usable or bool(self.manual)
            if data.get("reset_map") is True:
                self.world.clear()
                self.ingest(self.obs)
                usable = usable or bool(self.contract)
            if usable:
                self.need_review = None
                self.repeat.clear()
                self.seen.clear()
                self.last_review_step = self.steps
            self.ctx.log("contract_result", parsed=data, calls=self.model_calls)
            return usable
        except Exception as exc:
            self.model_calls += 1
            self.ctx.log("contract_failed", error=type(exc).__name__, reason=reason)
            return False

    def legal(self, action):
        if action not in self.allowed:
            return False
        front = visible(self.obs).get((0, 1), {})
        if action == "forward":
            return walkable(front)
        if action == "pickup":
            return not self.inventory and front.get("object") in ("key", "ball", "box")
        if action == "drop":
            return bool(self.inventory) and front.get("object") in ("empty", "floor")
        if action == "toggle":
            return self.can_toggle(front)
        return action in ("left", "right", "done")

    def alignment(self, obs, pos, heading):
        score = 0
        for rel, cell in visible(obs).items():
            if rel == (0, 0) or cell.get("object") == "unknown":
                continue
            old = self.world.get(self.transform(rel, pos, heading))
            if old is not None:
                weight = 4 if "wall" in (cell.get("object"), old.get("object")) else 1
                score += weight if sig(old) == sig(cell) else -2 * weight
        return score

    def act_checked(self, action, reason):
        before = self.obs
        before_front = visible(before).get((0, 1), {})
        before_inventory = self.inventory
        old_pos, old_heading = self.pos, self.heading
        try:
            result = self.ctx.act(action)
        except Exception as exc:
            if self.ctx.done:
                return False
            self.ctx.log("action_error", action=action, error=type(exc).__name__)
            self.need_review = "Action exception " + type(exc).__name__ + ": " + str(exc)[:160]
            self.manual.clear()
            return True
        self.steps += 1
        new = as_dict(result if result is not None else self.ctx.observe())
        new_heading = (int(new.get("direction", self.initial_direction + old_heading)) - self.initial_direction) % 4
        error = None
        if action == "forward":
            proposed = add(old_pos, DIRS[old_heading])
            move_score = self.alignment(new, proposed, new_heading)
            stay_score = self.alignment(new, old_pos, new_heading)
            if walkable(before_front) and move_score >= stay_score:
                self.pos = proposed
            else:
                error = "Forward observation better fits staying still than predicted motion; pose retained."
        elif action in ("left", "right"):
            expected = (old_heading + (-1 if action == "left" else 1)) % 4
            if new_heading != expected:
                error = "Reported heading contradicts the turn."
        self.ingest(new)
        progress = self.ctx.progress()
        if progress.get("illegal_actions", 0) > self.last_illegal:
            error = "ctx.progress reports a newly rejected action."
        self.last_illegal = int(progress.get("illegal_actions", 0) or 0)
        after_front = visible(self.obs).get((0, 1), {})
        if action == "pickup" and self.inventory == before_inventory:
            error = "Pickup did not change inventory."
        elif action == "drop" and self.inventory == before_inventory:
            error = "Drop did not change inventory."
        elif action == "toggle" and sig(after_front) == sig(before_front):
            error = "Toggle did not change the facing object."
        elif action == "done" and not self.ctx.done:
            error = "Done did not terminate the episode."
        fingerprint = (self.pos, self.heading, str(self.inventory), self.revision)
        self.repeat[fingerprint] += 1
        if self.repeat[fingerprint] >= 3:
            error = "Same pose/inventory/map visited three times: execution cycle."
        if progress.get("since_obs_change", 0) >= 4:
            error = "ctx.progress reports four actions without an observation change."
        if "exploration" in reason and self.steps - max(self.last_revision_step, self.last_review_step) >= 12:
            error = "Exploration has produced no map discovery in twelve steps."
        self.history.append({"action": action, "reason": reason, "error": error,
                             "pose": [*self.pos, self.heading], "progress": progress})
        self.ctx.log("checked_step", action=action, reason=reason, error=error,
                     pose=[*self.pos, self.heading], progress=progress)
        if error:
            self.manual.clear()
            self.need_review = error
        return not (self.ctx.done or self.obs.get("terminated") or self.obs.get("truncated"))

    def run_episode(self, game):
        self.allowed = set(getattr(game.action_spec, "names", ()) or ACTIONS) & ACTIONS
        self.obs = as_dict(self.ctx.observe())
        if not self.allowed:
            return
        self.initial_direction = int(self.obs.get("direction", 0))
        self.ingest(self.obs)
        self.contract = self.parse_local_contract()
        if self.contract is None:
            self.need_review = "Mission wording is outside the local grammar."
        while not self.ctx.done and not self.obs.get("terminated") and not self.obs.get("truncated"):
            if self.need_review:
                self.explore_route.clear()
                self.explore_label = None
                if not self.ask_contract(game, self.need_review):
                    # Malformed or unavailable inference is not permission to
                    # resume an already disproven script.
                    return
            if self.manual:
                action = self.manual.popleft()
                reason = "Astra prefix"
            elif self.contract:
                action, reason = self.choose()
            else:
                self.need_review = "Manual prefix ended; select the next subgoal from current evidence."
                continue
            if action is None:
                continue
            if not self.legal(action):
                self.need_review = "Pre-action legality veto for " + str(action) + "; front=" + str(visible(self.obs).get((0, 1)))
                self.manual.clear()
                continue
            if not self.act_checked(action, reason):
                return


def build(ctx):
    return Harness(ctx)
