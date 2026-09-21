"""Deadline-aware version of the evidence-led, lossless visual planner.

Only public descriptions/observations and the HarnessContext are used. There
are no environment IDs, puzzle mechanics, persistent seed answers, or evals.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import Counter, deque


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def grid_like(value):
    return (isinstance(value, list) and 2 <= len(value) <= 128
            and all(isinstance(r, list) for r in value)
            and 2 <= len(value[0]) <= 128
            and all(len(r) == len(value[0]) for r in value)
            and all(type(c) is int and 0 <= c <= 255 for r in value for c in r))


def inspect_observation(obs):
    """Discover matrices by structure, never by an observation field name."""
    grids, fields, images = {}, {}, []

    def walk(value, path):
        if grid_like(value):
            grids[path] = copy.deepcopy(value)
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list) and any(isinstance(x, (dict, list)) for x in value):
            for i, child in enumerate(value):
                walk(child, path + "/" + str(i))
        else:
            fields[path] = copy.deepcopy(value)
            if isinstance(value, str) and value.lower().endswith((".png", ".jpg", ".jpeg")):
                if os.path.isfile(value):
                    images.append(value)
    walk(obs, "")
    return grids, fields, images


def bands(values):
    result = []
    for i, value in enumerate(values):
        if result and value == values[result[-1][0]]:
            result[-1][1] = i
        else:
            result.append([i, i])
    return result


def encode_grid(grid):
    """Lossless run compression of identical adjacent rows and columns."""
    ys = bands(grid)
    xs = bands([list(col) for col in zip(*grid)])
    palette = sorted(set(c for row in grid for c in row))
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if len(palette) > len(alphabet):
        return {"rows": grid}
    symbols = {c: alphabet[i] for i, c in enumerate(palette)}
    return {"size": [len(grid[0]), len(grid)], "x_bands_inclusive": xs,
            "y_bands_inclusive": ys, "legend": {symbols[c]: c for c in palette},
            "rows": ["".join(symbols[grid[y][x]] for x, _ in xs) for y, _ in ys]}


def components(grid):
    height, width = len(grid), len(grid[0])
    seen, result = set(), []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen:
                continue
            color = grid[y][x]
            queue, points = [(x, y)], []
            seen.add((x, y))
            while queue:
                px, py = queue.pop()
                points.append((px, py))
                for nx, ny in ((px-1, py), (px+1, py), (px, py-1), (px, py+1)):
                    if (0 <= nx < width and 0 <= ny < height
                            and (nx, ny) not in seen and grid[ny][nx] == color):
                        seen.add((nx, ny))
                        queue.append((nx, ny))
            x0, x1 = min(p[0] for p in points), max(p[0] for p in points)
            y0, y1 = min(p[1] for p in points), max(p[1] for p in points)
            cx, cy = (x0+x1)/2, (y0+y1)/2
            rep = min(points, key=lambda p: ((p[0]-cx)**2+(p[1]-cy)**2, p))
            result.append({"color": color, "n": len(points), "box": [x0, y0, x1, y1],
                           "click": list(rep)})
    return sorted(result, key=lambda c: (c["n"] > width*height//3, -c["n"], c["box"]))


def difference(before, after):
    old_grids, old_fields, _ = before
    new_grids, new_fields, _ = after
    out = {"fields": {k: [old_fields.get(k), new_fields.get(k)]
                      for k in old_fields.keys() | new_fields.keys()
                      if old_fields.get(k) != new_fields.get(k)}, "grids": {}}
    for path in old_grids.keys() | new_grids.keys():
        a, b = old_grids.get(path), new_grids.get(path)
        if a == b:
            continue
        if a is None or b is None or len(a) != len(b) or len(a[0]) != len(b[0]):
            out["grids"][path] = {"reshaped": True}
            continue
        points = [(x, y, a[y][x], b[y][x]) for y in range(len(a))
                  for x in range(len(a[0])) if a[y][x] != b[y][x]]
        recolors = Counter((p[2], p[3]) for p in points)
        out["grids"][path] = {"pixels_changed": len(points),
            "box": [min(p[0] for p in points), min(p[1] for p in points),
                    max(p[0] for p in points), max(p[1] for p in points)],
            "recolors": [[a, b, n] for (a, b), n in recolors.most_common()],
            "samples_xy_old_new": points[:16]}
    return out


def strings(value):
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def read_reply(text):
    """Tolerate fences/prose; never execute model-generated code."""
    if not isinstance(text, str):
        return {}
    text = text[:100000]
    decoder = json.JSONDecoder()
    attempts = 0
    for i, char in enumerate(text):
        if char in "{[":
            attempts += 1
            if attempts > 64:
                break
            try:
                value, _ = decoder.raw_decode(text, i)
            except (ValueError, RecursionError):
                continue
            if isinstance(value, dict):
                return value
            if isinstance(value, list):
                return {"plan": value}
    return {"plan": [line.strip().strip("`\"'") for line in text.splitlines() if line.strip()]}


class Actions:
    def __init__(self, game, description):
        spec = game.action_spec
        self.names = list(spec.names)
        self.noop = getattr(spec, "noop", None)
        self.descriptions = getattr(spec, "descriptions", {})
        self.notes = getattr(spec, "notes", "")
        # This parameter grammar is enabled only when the public spec offers it.
        self.click = any(re.fullmatch(r"click\s+-?\d+\s+-?\d+", str(n)) for n in self.names)
        match = re.search(r"(?:integer|coordinates)[^\n]{0,80}?(-?\d+)\s*\.\.\s*(-?\d+)",
                          description + " " + str(self.descriptions) + " " + str(self.notes), re.I)
        self.bounds = tuple(map(int, match.groups())) if match else None

    def validate(self, action):
        if not isinstance(action, str):
            return None
        action = action.strip()
        if action in self.names:
            return action
        # An arbitrarily long digit string is malformed output, not an int to parse.
        if len(action) > 256:
            return None
        m = re.fullmatch(r"click\s+(-?\d+)\s+(-?\d+)", action)
        if m and self.click:
            try:
                x, y = map(int, m.groups())
            except ValueError:
                return None
            if self.bounds and not all(self.bounds[0] <= v <= self.bounds[1] for v in (x, y)):
                return None
            return f"click {x} {y}"
        return action if action in self.names else None

    def public(self):
        return {"names": self.names, "descriptions": self.descriptions,
                "notes": self.notes, "parameter_bounds": self.bounds}


class Deadline:
    """Admission control uses evaluator time, independently of the game's clock rules.

    No fixed episode duration is assumed. A finite budget is converted into a
    local ctx.clock deadline as well as polled again, so time spent preparing a
    prompt is charged even if a context reports a stale remaining-time snapshot.
    """
    RETURN_RESERVE = 10.0
    COLD_CALL_ALLOWANCE = 60.0

    def __init__(self, ctx):
        self.ctx = ctx
        self.end = math.inf
        self.peak_latency = 0.0
        self.remaining()

    def remaining(self):
        now = self.ctx.clock.now()
        value = self.ctx.budget.remaining().get("wall_s")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value):
                self.end = min(self.end, now + max(0.0, value))
        return max(0.0, self.end - now)

    def required(self):
        return self.RETURN_RESERVE + max(self.COLD_CALL_ALLOWANCE,
                                         1.5 * self.peak_latency)

    def admit(self):
        left = self.remaining()
        if left <= self.required():
            self.ctx.log(event="deadline_stop", remaining_s=left,
                         required_s=self.required(), peak_latency_s=self.peak_latency)
            return None
        return min(300.0, left - self.RETURN_RESERVE)

    def record(self, started):
        self.peak_latency = max(self.peak_latency, self.ctx.clock.now() - started)

    def can_act(self):
        return self.remaining() > self.RETURN_RESERVE


INSTRUCTIONS = """You control an unfamiliar turn-based game. Infer rules only from observed effects.
The harness executes your strings exactly and reports actual actions and differences. Thinking costs
no game actions. Optimize levels completed within remaining reasoning calls, not visual novelty.
Use the public description to interpret metadata. All observation fields are preserved; matrices
are encoded LOSSLESSLY: each character expands to its inclusive x/y bands using its numeric legend.
Coordinates are original pixels, not compressed indices. Components have no inferred semantic roles.
Screenshots, when attached, show the current observation. Unknown palettes/rules must be learned.

Maintain compact hypotheses: condition, effect, evidence event IDs, counterexample, confidence.
Choose discriminating probes when unsure; once mechanics are supported, plan a useful sequence.
Revisit assumptions when predicted effects fail. No change is evidence, not permission to repeat
forever. Different visible states may still hide different histories. Do not assume inverse controls.
Carry mechanics across levels, but re-read maps and legal actions. Reset may erase progress as
specified in the description; do not reset merely to observe. Avoid noop in turn-based games.

Return one JSON object with:
"memory": your revised compact rule table, goal, current plan, unresolved alternatives;
"schema": {"phase_paths": [JSON pointer paths for current level/phase],
 "state_paths": [paths for state/mode/selection needed to distinguish otherwise equal boards],
 "ignore_paths": [paths for telemetry counters/reward/artifact paths, NOT gameplay state],
 "recovery": {"path": "status pointer", "equals": "recoverable terminal value",
              "action": "legal retry action"}};
"mode": "probe" or "execute";
"plan": [{"action": "legal action string", "repeat": 1,
 "expect": {"changed": true, "pixels": [["grid pointer", x, y, expected_color]],
            "equals": {"metadata pointer": "expected value"}}}].
Omit schema entries and expectations you cannot justify. Schema uses observed field names, never
invented ones. Changed means any grid change (or nonignored metadata if there are no grids).
Repeat executes a step up to 16 times, checked after EACH action; expectations apply after EACH.
Use up to 32 actions in execute mode for a learned maneuver; up to 8 in probe mode for candidate
interactions. Probes stop at the FIRST changed observation. Any mode stops on score/phase/legal
change, recovery state, failed expectation, two unchanged steps, or repeated visible-state loops.
Plain action strings are also accepted. Every stop cancels the rest of the batch. State your rule
predictions in memory too: only supplied machine-checkable expectations can be checked by code.
"""


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        deadline = Deadline(ctx)
        description = str(game.describe())
        obs = ctx.observe()
        # Always start episode-local, including when a context is explicitly reused.
        evidence, salient = deque(maxlen=256), deque(maxlen=16)
        visits, tested, stats = Counter(), Counter(), {}
        recoveries = Counter()
        memory, schema = {}, {}
        ctx.memory["candidate_0002"] = {"evidence": evidence, "salient": salient}
        calls, step, invalid = 0, 0, 0
        notice = "Initial observation: learn controls with a discriminating probe."
        since_think = []

        def key(view):
            grids, fields, _ = view
            keep = strings(schema.get("state_paths")) + strings(schema.get("phase_paths"))
            selected = {k: fields.get(k) for k in keep}
            if not grids:
                selected = {k: v for k, v in fields.items() if k not in strings(schema.get("ignore_paths"))}
            return hashlib.sha256(dump([grids, selected]).encode()).hexdigest()[:16]

        def changed(delta, view):
            if view[0]:
                return bool(delta["grids"])
            return any(k not in strings(schema.get("ignore_paths")) for k in delta["fields"])

        def recovery(view, actions):
            rule = schema.get("recovery")
            if (isinstance(rule, dict) and isinstance(rule.get("path"), str)
                    and rule["path"] in view[1]):
                if view[1][rule["path"]] == rule.get("equals"):
                    return actions.validate(rule.get("action"))
            return None

        def fallback(view, actions):
            candidates = [n for n in actions.names if n != actions.noop and n != "reset"
                          and not str(n).startswith("click ")]
            if actions.click:
                for grid in view[0].values():
                    for comp in components(grid):
                        x, y = comp["click"]
                        candidates.append(f"click {x} {y}")
                    # Coverage fallback preserves access to regions segmentation may miss.
                    for y in range(len(grid)//16, len(grid), max(1, len(grid)//8)):
                        for x in range(len(grid[0])//16, len(grid[0]), max(1, len(grid[0])//8)):
                            candidates.append(f"click {x} {y}")
            candidates = list(dict.fromkeys(a for a in candidates if actions.validate(a)))
            if not candidates:
                candidates = [a for a in actions.names if a != "reset" and actions.validate(a)]
            if not candidates:
                return None
            return min(candidates, key=lambda a: (tested[key(view), a], stats.get(a, {}).get("n", 0)))

        def check(expect, view, did_change):
            if not isinstance(expect, dict):
                return []
            failed = []
            if type(expect.get("changed")) is bool and expect["changed"] != did_change:
                failed.append("changed prediction failed")
            equals = expect.get("equals")
            if isinstance(equals, dict):
                for p, value in equals.items():
                    if p not in view[1] or view[1][p] != value:
                        failed.append("metadata prediction failed: " + str(p))
            pixels = expect.get("pixels")
            if isinstance(pixels, list):
                for item in pixels[:128]:
                    if not isinstance(item, list) or len(item) != 4:
                        failed.append("malformed pixel prediction")
                        continue
                    path, x, y, value = item
                    grid = view[0].get(path) if isinstance(path, str) else None
                    if (grid is None or type(x) is not int or type(y) is not int
                            or not 0 <= y < len(grid) or not 0 <= x < len(grid[0])
                            or grid[y][x] != value):
                        failed.append("pixel prediction failed: " + dump(item))
            return failed

        while not ctx.done and not game.done and calls < 40 and step < 10000:
            budget = ctx.budget.remaining()
            if budget.get("system2_calls", 0) <= 0 or not deadline.can_act():
                break
            view = inspect_observation(obs)
            actions = Actions(game, description)
            current_key = key(view)
            visits[current_key] += 1
            auto = recovery(view, actions)
            if auto and recoveries[current_key, auto] < 2:
                recoveries[current_key, auto] += 1
                reply = {"mode": "probe", "plan": [auto]}
            else:
                # Check before expensive prompt preparation, then again immediately
                # before inference. A call with positive time left need not fit.
                if deadline.admit() is None:
                    break
                maps = {p: encode_grid(g) for p, g in view[0].items()}
                objects = {p: components(g)[:64] for p, g in view[0].items()}
                packet = {"budget": budget, "score": game.score, "actions": actions.public(),
                          "metadata": view[1], "grids": maps, "components_first_64": objects,
                          "state_key": current_key, "visits_at_planning": visits[current_key],
                          "current_state_tested_actions": {a: n for (k, a), n in tested.items() if k == current_key},
                          "memory": memory, "schema": schema, "stop_reason": notice,
                          "since_last_plan": since_think, "recent_evidence": list(evidence)[-12:],
                          "salient_evidence": list(salient), "action_stats": stats}
                # Fresh sessions prevent obsolete full-grid histories from accumulating.
                prompt = description + "\n" + INSTRUCTIONS + "\n" + dump(packet)
                timeout = deadline.admit()
                if timeout is None:
                    break
                started = ctx.clock.now()
                try:
                    result = ctx.think(prompt, session_id=None,
                                       images=view[2][-1:] or None, timeout_s=timeout)
                except TimeoutError as exc:
                    # Handle only a literal standard-library timeout. In particular,
                    # ModelUnavailable (or any specialized exception) must propagate.
                    if type(exc) is not TimeoutError:
                        raise
                    ctx.log(event="model_timeout_stop", timeout_s=timeout)
                    return
                deadline.record(started)
                calls += 1
                reply = read_reply(result.text)
                since_think = []
                if "memory" in reply:
                    memory = reply["memory"]
                if isinstance(reply.get("schema"), dict):
                    for name in ("phase_paths", "state_paths", "ignore_paths"):
                        if name in reply["schema"]:
                            schema[name] = [p for p in strings(reply["schema"][name]) if p in view[1]]
                    if isinstance(reply["schema"].get("recovery"), dict):
                        schema["recovery"] = reply["schema"]["recovery"]
                ctx.memory["candidate_0002"].update(memory=memory, schema=schema)
            mode = reply.get("mode", "probe")
            limit = 32 if mode == "execute" else 8
            plan = reply.get("plan", [])
            if not isinstance(plan, list):
                plan = []
            queue = []
            for item in plan[:32]:
                entry = {"action": item} if isinstance(item, str) else item
                if not isinstance(entry, dict):
                    continue
                action = actions.validate(entry.get("action"))
                if action is None:
                    continue
                repeat = entry.get("repeat", 1)
                repeat = max(1, min(16, repeat)) if type(repeat) is int else 1
                queue.extend([(action, entry.get("expect", {}))] * repeat)
                if len(queue) >= limit:
                    break
            queue = queue[:limit]
            if not queue or all(a == actions.noop for a, _ in queue):
                action = fallback(view, actions)
                ctx.log(event="parse_fallback", reply_excerpt=str(reply)[:500], selected=action)
                queue = [(action, {})] if action else []
                mode = "probe"
            if not queue:
                break
            batch_visits, unchanged = Counter(), 0
            notice = "Batch finished."
            for action, expect in queue:
                if ctx.done or game.done or step >= 10000 or not deadline.can_act():
                    break
                actions = Actions(game, description)
                if actions.validate(action) is None:
                    notice = "Action became unavailable; remaining plan cancelled."
                    break
                old_key, old_score = key(view), game.score
                old_names = actions.names
                try:
                    new_obs = ctx.act(action)
                except ValueError as exc:
                    # Public invalid-input failures occur before mutation; budget still charged.
                    invalid += 1
                    step += 1
                    notice = "Rejected action " + action + ": " + str(exc)
                    ctx.log(event="invalid_action", action=action, reason=str(exc))
                    obs = ctx.observe()
                    if invalid >= 3:
                        return
                    break
                step += 1
                obs = ctx.observe() if new_obs is None else new_obs
                after = inspect_observation(obs)
                delta = difference(view, after)
                did_change = changed(delta, view)
                failure = check(expect, after, did_change)
                new_key = key(after)
                tested[old_key, action] += 1
                summary = stats.setdefault(action, {"n": 0, "changed": 0, "score_increases": 0})
                summary["n"] += 1
                summary["changed"] += int(did_change)
                summary["score_increases"] += int(game.score > old_score)
                event = {"id": step, "action": action, "from": old_key, "to": new_key,
                         "score": [old_score, game.score], "delta": delta, "refuted": failure}
                evidence.append(event)
                since_think.append(event)
                ctx.log(event="measured_transition", **event)
                if failure or game.score != old_score or (did_change and summary["changed"] == 1):
                    salient.append(event)
                phase_change = any(view[1].get(p) != after[1].get(p)
                                   for p in strings(schema.get("phase_paths")))
                view = after
                batch_visits[new_key] += 1
                unchanged = 0 if did_change else unchanged + 1
                new_actions = Actions(game, description)
                if failure:
                    notice = "Prediction refuted: " + "; ".join(failure)
                elif game.score != old_score or phase_change:
                    notice = "Score or phase changed: inspect the new board before continuing."
                elif action == "reset":
                    notice = "Explicit reset completed; inspect its scope before continuing."
                elif new_actions.names != old_names:
                    notice = "Legal actions changed."
                elif recovery(view, new_actions):
                    notice = "Recoverable terminal state observed."
                elif mode != "execute" and did_change:
                    notice = "Probe produced an effect: interpret before continuing."
                elif unchanged >= 2 or batch_visits[new_key] >= 3:
                    notice = "Batch stalled or revisited a visible state; reconsider the rule."
                else:
                    continue
                break


def build(ctx):
    return Harness(ctx)
