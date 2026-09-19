"""Compile a reactive policy, then an audited model for bounded lookahead.

No game names, observation keys, movement rules, or reward functions are built in.
The small expression language below is populated only by a traced Astra call.
"""
from __future__ import annotations

import ast
import concurrent.futures
import copy
import hashlib
import json
import math
import subprocess
import time

from metaharness.models.jev import Choice


def bounded_range(*args):
    result = range(*args)
    if len(result) > 256:
        raise ValueError("range exceeds 256 elements")
    return result


FUNCTIONS = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len,
             "all": all, "any": any, "int": int, "float": float, "bool": bool,
             "list": list, "tuple": tuple, "set": set, "sorted": sorted,
             "enumerate": enumerate, "range": bounded_range}
NODES = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Store,
         ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.Dict, ast.Set,
         ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
         ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
         ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
         ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
         ast.Call, ast.keyword, ast.Attribute, ast.ListComp, ast.SetComp,
         ast.DictComp, ast.GeneratorExp, ast.comprehension)


def expression(source):
    if not isinstance(source, str) or len(source) > 4000:
        raise ValueError("expression must be a string of at most 4000 characters")
    tree = ast.parse(source, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 500:
        raise ValueError("expression too large")
    for node in nodes:
        if not isinstance(node, NODES):
            raise ValueError("unsupported expression node: " + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("private name")
        if isinstance(node, ast.Attribute) and node.attr != "get":
            raise ValueError("only dictionary .get is permitted")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS:
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                continue
            raise ValueError("unsupported call")
        if isinstance(node, ast.comprehension) and node.is_async:
            raise ValueError("async expression")
    return compile(tree, "<astra-expression>", "eval")


def evaluate(code, env):
    # One globals dictionary also makes f/o/x available inside comprehensions.
    return eval(code, {"__builtins__": {}, **FUNCTIONS, **env})


def feature_list(rows):
    if not isinstance(rows, list) or len(rows) > 8:
        raise ValueError("expected at most 8 ordered features")
    result = []
    for row in rows:
        name = row["name"]
        if not isinstance(name, str) or name in [n for n, _ in result]:
            raise ValueError("invalid or duplicate feature name")
        result.append((name, expression(row["expr"])))
    return result


class Policy:
    def __init__(self, raw):
        if not isinstance(raw, dict):
            raise ValueError("policy must be a JSON object")
        if len(json.dumps(raw)) > 8000:
            raise ValueError("policy exceeds 8000 characters; simplify it")
        self.raw = raw
        self.features = feature_list(raw.get("features", []))
        self.per_action = feature_list(raw.get("per_action", []))
        self.safe = expression(raw["safe"])
        modes = raw["modes"]
        if not isinstance(modes, list) or not 1 <= len(modes) <= 2:
            raise ValueError("expected one or two modes")
        self.modes = [("m" + str(i), str(m["when"])[:1000], expression(m["utility"]))
                      for i, m in enumerate(modes)]
        self.summary = str(raw.get("summary", ""))[:1500]

    def resolve(self, observation, names, last_action):
        env = {"o": observation, "actions": names, "last": last_action, "f": {}}
        for name, code in self.features:
            env["f"][name] = evaluate(code, env)
        rows = {}
        for action in names:
            local = {**env, "a": action, "x": {}}
            for name, code in self.per_action:
                local["x"][name] = evaluate(code, local)
            safe = evaluate(self.safe, local)
            if not isinstance(safe, bool):
                raise ValueError("safe expression must return bool")
            utilities = {mode: float(evaluate(code, local)) for mode, _, code in self.modes}
            if not all(math.isfinite(v) for v in utilities.values()):
                raise ValueError("utility is not finite")
            rows[action] = {"safe": safe, "facts": local["x"], "utility": utilities}
        return env["f"], rows


def parse_policy(text):
    # Accept a JSON object wrapped in prose or a markdown fence.
    decoder = json.JSONDecoder()
    errors = []
    for i, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and ("safe" in obj or "modes" in obj):
                try:
                    return Policy(obj)
                except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                    errors.append(type(exc).__name__ + ": " + str(exc)[:300])
    raise ValueError(errors[0] if errors else "no policy JSON object found")


def compiler_error(exc):
    # TimeoutExpired.__str__ contains the entire command (including prompt).
    # Sending that back as repair feedback obscures the actual cause.
    if isinstance(exc, (subprocess.TimeoutExpired, TimeoutError)):
        return "compiler deadline exceeded; return a smaller policy immediately"
    return type(exc).__name__ + ": " + str(exc)[:400]


def shape(value):
    if isinstance(value, dict):
        return {str(k): shape(v) for k, v in sorted(value.items())}
    # Dynamic entity counts and values must not change a rules cache key.
    return "list" if isinstance(value, (list, tuple)) else type(value).__name__


def mask_paths(value, names, path=()):
    result = []
    if isinstance(value, dict) and len(path) <= 4:
        overlap = set(value).intersection(names)
        if len(overlap) >= 2 and all(isinstance(value[k], bool) for k in overlap):
            result.append(path)
        for key, child in value.items():
            if isinstance(child, dict):
                result.extend(mask_paths(child, names, path + (key,)))
    return result[:4]


def lookup(value, path):
    for key in path:
        value = value[key]
    return value


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite model number")
    return value


class World:
    """Game semantics are supplied exclusively by the traced model response."""

    def __init__(self, raw):
        if not isinstance(raw, dict) or len(packed(raw)) > 12000:
            raise ValueError("world must be an object under 12000 characters")
        self.raw = raw
        self.features = feature_list(raw.get("features", []))
        self.per_action = feature_list(raw.get("per_action", []))
        for name in ("encode", "project", "safe", "next", "reward", "value"):
            setattr(self, name, expression(raw[name]))
        self.summary = str(raw.get("summary", ""))[:1000]

    def environment(self, o, s, names):
        env = {"o": o, "s": s, "actions": names, "f": {}}
        for name, code in self.features:
            env["f"][name] = evaluate(code, env)
        return env

    def initial(self, o, names):
        s = evaluate(self.encode, {"o": o, "actions": names})
        packed(s)
        return s

    def projection(self, o, s, names):
        value = evaluate(self.project, self.environment(o, s, names))
        if value is None or value == {} or value == []:
            raise ValueError("empty audit projection")
        return packed(value)

    def heuristic(self, o, s, names):
        return max(-0.5, min(0.5, finite(evaluate(
            self.value, self.environment(o, s, names)))))

    def edges(self, o, s, names, choices):
        env = self.environment(o, s, names)
        for action in choices:
            local = {**env, "a": action, "x": {}}
            for name, code in self.per_action:
                local["x"][name] = evaluate(code, local)
            safe = evaluate(self.safe, local)
            if not isinstance(safe, bool):
                raise ValueError("model safe must be Boolean")
            if safe:
                successor = evaluate(self.next, local)
                packed(successor)
                reward = finite(evaluate(self.reward, local))
                yield action, successor, reward

    def audit(self, before, action, after, names):
        start = self.initial(before, names)
        result = list(self.edges(before, start, names, [action]))
        if not result:
            raise ValueError("model rejects an observed surviving transition")
        predicted = self.projection(before, result[0][1], names)
        actual = self.projection(after, self.initial(after, names), names)
        if predicted != actual:
            raise ValueError("transition mismatch: predicted=" + predicted[:500]
                             + "; observed=" + actual[:500])


def parse_world(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == "{":
            try:
                raw, _ = decoder.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(raw, dict) and "unsupported" in raw:
                return None
            if isinstance(raw, dict) and "encode" in raw:
                return World(raw)
    raise ValueError("no world JSON object")


def rollout(world, observation, names, allowed, seconds, horizon=14, width=10):
    """Equal-depth beam search per first action; only commit complete layers.

    A missing continuation is absence of a witness in a pruned model search,
    never a proof of unavoidable loss. No game-specific search primitive.
    """
    started = time.perf_counter()
    deadline = started + seconds
    initial = world.initial(observation, names)
    beams, report = {}, {}
    for action, state, reward in world.edges(observation, initial, names, allowed):
        beams[action] = [(state, reward)]
        report[action] = {"depth": 1, "return": reward + world.heuristic(
            observation, state, names), "width": 1}
    # Root safety disagreements are left to the live reactive shield. A model
    # cannot veto actions until it has produced a complete, useful search layer.
    expanded, complete = 0, 1
    for depth in range(2, horizon + 1):
        next_beams, next_report = {}, copy.deepcopy(report)
        for root, beam in beams.items():
            unique = {}
            for state, total in beam:
                if time.perf_counter() >= deadline or expanded >= 6000:
                    return report, {"depth": complete, "expanded": expanded,
                                    "limited": True, "seconds": time.perf_counter() - started}
                for _, successor, reward in world.edges(observation, state, names, names):
                    expanded += 1
                    accumulated = total + (0.97 ** (depth - 1)) * reward
                    key = packed(successor)
                    rank = accumulated + world.heuristic(observation, successor, names)
                    if key not in unique or accumulated > unique[key][2]:
                        unique[key] = (rank, successor, accumulated)
            best = sorted(unique.values(), key=lambda row: row[0], reverse=True)[:width]
            if best:
                next_beams[root] = [(s, total) for _, s, total in best]
                next_report[root] = {"depth": depth, "return": best[0][0],
                                     "width": len(best)}
        beams, report, complete = next_beams, next_report, depth
        if not beams:
            break
    return report, {"depth": complete, "expanded": expanded, "limited": False,
                    "seconds": time.perf_counter() - started}


WORLD_COMPILER = """Extend the working reactive policy with a SMALL deterministic
transition model. Direct JSON only, no tools, research, tests or prose. Aim under
3500 characters. The harness implements generic beam search; do NOT write search.
Derive every game key, action effect and rule from the provided public material.
Return this schema (all expressions are strings):
{"summary":"model scope and uncertainty",
 "encode":"compact simulated state from current observation o",
 "features":[{"name":"...","expr":"..."}],
 "per_action":[{"name":"...","expr":"..."}],
 "safe":"bool: action's next transition survives",
 "next":"complete successor simulated state",
 "reward":"numeric genuine immediate reward for this transition",
 "value":"state progress potential between -0.5 and 0.5, higher is better",
 "project":"nonempty deterministic state components checked against reality"}
encode sees o, actions. Other expressions see o (the FROZEN root observation),
s (the current SIMULATED state), actions, and ordered features f. per_action,
safe, next and reward additionally see a and ordered per-action features x.
next and reward both see the PRE-transition s/x. value and project see s/f only.
Each feature list has at most 8 entries. Allowed syntax: literals, indexing,
slicing, arithmetic except **, comparisons, conditionals, comprehensions,
dictionary .get, and abs min max sum len all any int float bool list tuple set
sorted enumerate range (max 256 elements). No statements, imports, lambdas,
custom functions or other methods. JSON state only, no sets in state/project.

Simulate ALL changing components needed for future legality, including motion,
occupancy, resources, timers and ignored/default actions when the rules require
them. Never consult frozen immediate-action safety maps for simulated later
states. Use current o only for fixed parameters and the current known objective.
Do not invent future random events/targets. After a known objective is consumed,
remember that in s, grant its reward ONCE, keep simulating survival with no new
objective, and give zero target potential. Replanning will observe new events.
project selects predictable physical state, excluding random inputs and rollout
bookkeeping flags initialized by encode. We compare project(next(encode(before),
action)) with project(encode(after)) on actual one-tick transitions, including
reward events. Include meaningful dynamic components; do not use a constant.
Use supplied observed transitions to resolve ambiguities. Keep value simple,
bounded and increasing toward a currently known reward; do not duplicate reward
in value. If a useful predictive model is impossible from public information,
return {"unsupported":"brief reason"}; the reactive controller will continue.
"""


COMPILER = """Return a TINY reusable reactive policy, using only the rules and
observation below. This is a direct JSON answer: no tools, files, tests, external
research, or prose. Prefer a simple working heuristic to an elaborate solver.
Aim for under 2500 characters. Return:
{"summary":"one sentence",
 "features":[{"name":"...","expr":"Python expression"}],
 "per_action":[{"name":"...","expr":"Python expression"}],
 "safe":"Boolean expression",
 "modes":[{"when":"observable applicability criterion","utility":"numeric expression"}]}
Use at most 8 features in each list and 1-2 modes. Start with ONE adequate mode;
add a second only if it is short and clearly useful. Mode 0 must pursue reward
among safe actions. Avoid search, route planning, and elaborate future simulation.
Use a cheap danger/escape cue only if it is easy to compute from observed facts.

Each tick: evaluate features in order into f; for EACH action a evaluate
per_action in order into x; filter safe; maximize the selected mode's utility.
o=current observation, actions=action names, last=last posted action. Refer to
earlier features as f['name'] or x['name']. Jev classifies the mode from these
computed facts; mode choices can be two seconds old, but expressions always
receive fresh o. Derive every key and action effect from the supplied rules.
Do not freeze dynamic values from the example. Account for ignored/default
actions; use explicit observed safety information when present.

Expressions allow literals, indexing, arithmetic (no **), comparisons,
conditionals, comprehensions, dictionary .get, and these functions:
abs min max sum len all any int float bool list tuple set sorted enumerate range.
range has at most 256 elements. No statements, imports, lambdas, other methods,
or custom functions. safe returns bool; utilities return finite numbers.
"""


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        names = list(game.action_spec.names)
        if not names:
            return
        rules = game.describe() + "\n" + game.action_spec.describe()
        descriptions = game.action_spec.descriptions
        first = copy.deepcopy(game.observe())
        key_data = [rules, names, shape(first)]
        key = "audited_lookahead_v1:" + hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        cached = ctx.memory.get(key, {})
        masks = list(cached.get("masks", []))
        policy = None
        last = game.action_spec.noop or names[0]
        failure = ""
        failed_output = ""
        try:
            if "policy" in cached:
                policy = Policy(cached["policy"])
                policy.resolve(first, names, last)
        except Exception as exc:
            failure = str(exc)
            policy = None
        revision = 1 if policy else 0
        s1 = s2 = None
        s2_calls = s1_calls = 0
        policy_calls = world_calls = 0
        compile_kind = "policy"
        world = None
        world_failure = ""
        world_rejected = ""
        world_unsupported = bool(cached.get("world_unsupported", False))
        try:
            if cached.get("world"):
                world = World(cached["world"])
                world.initial(first, names)
        except Exception as exc:
            world_failure = compiler_error(exc)
            world = None
        transitions = []
        previous = None
        audited = 0
        search_frames = 0
        mode, mode_at = "m0", -100
        bootstrap_ranking = {name: 0.0 for name in names}
        bootstrap_at = -100
        history = []
        paths = mask_paths(first, names)
        classified = bool(masks) or not paths
        session = None
        next_compile = 0
        frame = 0
        compiled_frames = 0
        bootstrap_frames = 0
        decision_lifetime = 4 * (game.tick_s or 0.5)

        def remaining(kind):
            return ctx.budget.remaining().get(kind + "_calls", 0) > 0

        def remember():
            item = {"masks": masks, "world_unsupported": world_unsupported}
            if policy is not None:
                item["policy"] = policy.raw
            if world is not None:
                item["world"] = world.raw
            ctx.memory[key] = item

        def reject_world(exc):
            nonlocal world, world_failure, world_rejected, audited
            world_failure = compiler_error(exc)
            if world is not None:
                world_rejected = json.dumps(world.raw)
            world = None
            audited = 0
            remember()
            ctx.log(event="world_rejected", error=world_failure, frame=frame)

        def accept_compiler(future, observation):
            nonlocal policy, session, revision, failure, next_compile, mode_at, failed_output
            nonlocal world, world_failure, world_rejected, world_unsupported, audited
            output = ""
            try:
                result = future.result()
                session = result.session_id
                output = result.text
                if compile_kind == "world":
                    candidate = parse_world(output)
                    if candidate is None:
                        world_unsupported = True
                        ctx.log(event="world_unsupported", response=output[:1500])
                    else:
                        candidate.initial(observation, names)
                        for before, action, after in transitions:
                            candidate.audit(before, action, after, names)
                        # Verify multi-step execution, without requiring a safe
                        # continuation from an already terminal observation.
                        if not ctx.done:
                            rollout(candidate, observation, names, names, 0.025, horizon=3)
                        world = candidate
                        audited = len(transitions)
                        world_failure = world_rejected = ""
                        mode_at = -100
                        revision += 1
                        ctx.log(event="world_installed", world=world.raw, frame=frame,
                                audits=audited, episode_active=not ctx.done)
                    remember()
                    return
                candidate = parse_policy(output)
                for sample in [first, *history, observation]:
                    candidate.resolve(sample, names, last)
                policy = candidate
                revision += 1
                mode_at = -100
                failure = ""
                failed_output = ""
                remember()
                ctx.log(event="policy_installed", revision=revision, policy=policy.raw,
                        frame=frame, episode_active=not ctx.done,
                        output_chars=len(output), mode_count=len(policy.modes))
            except Exception as exc:
                if compile_kind == "world":
                    world_rejected = output[:12000]
                    reject_world(exc)
                    return
                failure = compiler_error(exc)
                failed_output = output[:8000]
                next_compile = frame + 4
                ctx.log(event="compiler_error", error=failure,
                        error_type=type(exc).__name__, attempt=s2_calls)

        while not ctx.done:
            if ctx.clock.mode == "virtual":
                pending = [f for f in (s1, s2) if f is not None]
                if pending:
                    # A virtual driver can otherwise finish all ticks before a
                    # Python worker gets CPU time. This yields the CPU, not game time.
                    concurrent.futures.wait(pending, timeout=0.001,
                                            return_when=concurrent.futures.FIRST_COMPLETED)
            observation = copy.deepcopy(game.observe())
            current_tick = getattr(game, "ticks", None)
            if previous is not None:
                before, action_taken, sent_tick = previous
                # Skip ambiguous intervals instead of diagnosing latency as a
                # bad transition model. The standard runner exposes ticks.
                if current_tick is not None and sent_tick is not None and current_tick == sent_tick + 1:
                    transitions.append((before, action_taken, observation))
                    transitions = transitions[-16:]
                    if world is not None:
                        try:
                            world.audit(before, action_taken, observation, names)
                            audited += 1
                            if audited <= 3 or audited % 25 == 0:
                                ctx.log(event="world_audit", matches=audited, frame=frame)
                        except Exception as exc:
                            reject_world(exc)
                previous = None
            history.append(observation)
            history = history[-3:]
            if s2 is not None and s2.done():
                accept_compiler(s2, observation)
                s2 = None

            if s1 is not None and s1.done():
                try:
                    answers, sent_revision, sent_frame, sent_time, options, sent_paths = s1.result()
                    for i, path in enumerate(sent_paths):
                        answer = answers.get("mask" + str(i))
                        if answer is not None and answer.value in ("unsafe_true", "safe_true"):
                            certainty = answer.probabilities.get(answer.value, answer.confidence or 0)
                            if certainty >= 0.7:
                                masks.append((path, answer.value == "unsafe_true"))
                                ctx.log(event="mask_learned", path=path, polarity=answer.value,
                                        probability=certainty)
                    if sent_paths:
                        classified = True
                        remember()
                    answer = answers.get("select")
                    answer_age = ctx.clock.now() - sent_time
                    if answer is not None and sent_revision == revision and answer_age <= decision_lifetime:
                        if policy is not None and answer.value in options:
                            mode, mode_at = answer.value, sent_time
                        elif policy is None:
                            bootstrap_ranking = {name: float(answer.probabilities.get(code, 0.0))
                                                 for code, name in options.items()}
                            if answer.value in options:
                                bootstrap_ranking[options[answer.value]] += 1.0
                            bootstrap_at = sent_time
                    ctx.log(event="selection_received", sent_revision=sent_revision,
                            age_frames=frame - sent_frame, age_s=answer_age,
                            selected=getattr(answer, "value", None))
                except Exception as exc:
                    ctx.log(event="classifier_error", error=str(exc)[:500])
                s1 = None

            if policy is None and s2 is None and frame >= next_compile and policy_calls < 2:
                if ctx.system2 is not None and remaining("system2"):
                    prompt = COMPILER + "\nRULES AND ACTIONS:\n" + rules
                    prompt += "\nCURRENT OBSERVATION:\n" + json.dumps(observation, default=str)
                    if failure:
                        prompt += "\nREPAIR: " + failure
                        prompt += ("\nReturn just ONE mode with immediate safety and simple reward progress. "
                                   "Omit optional future-viability features. Aim for under 1800 characters.")
                        if failed_output:
                            prompt += "\nREJECTED POLICY:\n" + failed_output
                    s2_calls += 1
                    policy_calls += 1
                    compile_kind = "policy"
                    ctx.log(event="compiler_requested", attempt=s2_calls,
                            frame=frame, timeout_s=90.0, prompt_chars=len(prompt),
                            repair=bool(failure))
                    # Preserve 0003's successful small-policy compilation path.
                    s2 = ctx.spawn(lambda p=prompt, sid=session: ctx.think(
                        p, session_id=sid, timeout_s=90.0))

            if (policy is not None and world is None and not world_unsupported
                    and s2 is None and world_calls < 2 and ctx.system2 is not None
                    and remaining("system2")):
                prompt = WORLD_COMPILER + "\nRULES AND ACTIONS:\n" + rules
                prompt += "\nCURRENT OBSERVATION:\n" + json.dumps(observation, default=str)
                prompt += "\nWORKING REACTIVE POLICY:\n" + json.dumps(policy.raw)
                prompt += "\nOBSERVED (before, action, after) TRANSITIONS:\n" + json.dumps(transitions[-3:], default=str)
                if world_failure:
                    prompt += "\nREPAIR THIS ERROR: " + world_failure
                    prompt += "\nREJECTED MODEL:\n" + world_rejected
                s2_calls += 1
                world_calls += 1
                compile_kind = "world"
                ctx.log(event="world_requested", attempt=world_calls, frame=frame,
                        repair=bool(world_failure), prompt_chars=len(prompt))
                s2 = ctx.spawn(lambda p=prompt, sid=session: ctx.think(
                    p, session_id=sid, timeout_s=90.0))

            features, rows = {}, None
            if policy is not None:
                try:
                    features, rows = policy.resolve(observation, names, last)
                except Exception as exc:
                    failure = compiler_error(exc)
                    failed_output = json.dumps(policy.raw)[:8000]
                    policy = None
                    revision += 1
                    remember()
                    ctx.log(event="policy_runtime_error", error=failure)

            # Masks are learned by a typed question, never inferred from key spelling.
            allowed = list(names)
            for path, unsafe_true in masks:
                try:
                    values = lookup(observation, path)
                    known_safe = [a for a in allowed if isinstance(values.get(a), bool)
                                  and values[a] != unsafe_true]
                    if known_safe:
                        allowed = known_safe
                except (KeyError, TypeError, AttributeError):
                    pass
            if rows is not None:
                safe = [a for a in allowed if rows[a]["safe"]]
                if safe:
                    allowed = safe
                else:
                    # A model-derived mask can disagree with a richer compiled rule.
                    allowed = [a for a in names if rows[a]["safe"]] or allowed
                    ctx.log(event="safety_disagreement_or_no_safe_action", allowed=allowed)
                valid_modes = {m for m, _, _ in policy.modes}
                active = mode if mode in valid_modes and ctx.clock.now() - mode_at <= decision_lifetime else "m0"
                action = max(allowed, key=lambda a: (rows[a]["utility"][active], a == last))
                state = {"rules_summary": policy.summary, "features": features,
                         "candidates": {a: rows[a] for a in allowed}, "current_mode": active}
                options = {m: m for m, _, _ in policy.modes}
                criteria = {m: when for m, when, _ in policy.modes}
                instructions = ("Classify the current situation into the most useful strategy. "
                                "Use the computed facts and the conditions in each option. "
                                "The strategy's highest-utility safe action is recomputed each tick. "
                                "Prefer reward progress with continued survival. Select m0 unless "
                                "the facts specifically support another strategy.")
                source = "compiled:" + active
                compiled_frames += 1
                if world is not None:
                    try:
                        reactive_choice = action
                        viable = []
                        search, stats = rollout(world, observation, names, allowed,
                                                min(0.06, (game.tick_s or 0.5) * 0.12))
                        # Require observations to support the model before it
                        # influences actions. Cached models are re-audited too.
                        if audited >= 2 and stats["depth"] >= 3 and search:
                            deepest = max(row["depth"] for row in search.values())
                            viable = [a for a in allowed if a in search and search[a]["depth"] == deepest]
                            choice_mode = mode if (mode in ("plan", "progress") and
                                ctx.clock.now() - mode_at <= decision_lifetime) else "plan"
                            if choice_mode == "progress":
                                action = max(viable, key=lambda a: (rows[a]["utility"]["m0"],
                                             search[a]["return"], a == last))
                            else:
                                action = max(viable, key=lambda a: (search[a]["return"],
                                             rows[a]["utility"]["m0"], a == last))
                            state = {"rules_summary": policy.summary,
                                     "model_scope": world.summary, "features": features,
                                     "search": search, "search_stats": stats,
                                     "verified_transitions": audited,
                                     "survival_candidates": viable,
                                     "reactive_utilities": {a: rows[a]["utility"]["m0"] for a in viable}}
                            options = {"plan": "plan", "progress": "progress"}
                            criteria = {
                                "plan": "Default: the audited model and completed search provide useful predicted reward or progress. Maximize predicted return among longest surviving continuations.",
                                "progress": "Predicted returns are tied or the model omits important unpredictable future rewards. Maximize immediate reactive utility among the SAME longest surviving continuations."}
                            instructions = ("Classify whether the supplied search has useful reward guidance. "
                                "Choose plan by default; progress only when the supplied facts support its criterion. "
                                "You select a reusable scoring rule, never an old action. Current safety and "
                                "continuations are recomputed each tick. Finite beam search is not a safety proof.")
                            source = "lookahead:" + choice_mode
                            search_frames += 1
                        ctx.log(event="lookahead", frame=frame, search=search,
                                stats=stats, audits=audited, selected=action,
                                reactive_choice=reactive_choice, survival_candidates=viable,
                                used=source.startswith("lookahead:"))
                    except Exception as exc:
                        reject_world(exc)
            else:
                # Cold start: semantic choices with opaque IDs avoid a default-action label prior.
                ranking = bootstrap_ranking if ctx.clock.now() - bootstrap_at <= decision_lifetime else {}
                action = max(allowed, key=lambda a: (ranking.get(a, 0), a == last))
                options = {"a" + str(i): a for i, a in enumerate(names)}
                state = {"observation": observation, "last_posted_action": last,
                         "currently_permitted": allowed}
                criteria = {code: "Select when executing " + repr(a) + " (" + descriptions.get(a, a)
                            + ") now avoids imminent failure and makes the best reward progress. "
                            "Do not select merely because it preserves the previous action."
                            for code, a in options.items()}
                instructions = rules + ("\nChoose the action that actively improves the situation "
                                        "while avoiding failure. Continuing automatically is a real "
                                        "action with consequences, not a safe default.")
                source = "bootstrap"
                bootstrap_frames += 1

            if ctx.done:
                break
            ctx.act(action)
            previous = (observation, action, current_tick)
            last = action
            ctx.log(event="reactive_action", frame=frame, action=action, source=source,
                    allowed=allowed, revision=revision, classifier_pending=s1 is not None)

            if s1 is None and ctx.system1 is not None and s1_calls < 4000 and remaining("system1"):
                questions = {"select": Choice(instructions=instructions, criteria=criteria)}
                pending_paths = [] if classified else paths
                if pending_paths:
                    state["observation"] = observation
                for i, path in enumerate(pending_paths):
                    questions["mask" + str(i)] = Choice(
                        instructions=rules + "\nWhat does the action-indexed Boolean map at path "
                        + json.dumps(path) + " in observation mean? Infer semantics from the rules.",
                        criteria={"unsafe_true": "True means this action is unsafe, fatal, or forbidden; false means permitted.",
                                  "safe_true": "True means this action is safe or legal; false means unsafe or forbidden.",
                                  "unrelated": "Neither polarity is explicitly a safety or legality mask, or unclear."})
                # Feature expressions may produce sets; normalize them for the API.
                state = json.loads(json.dumps(state, default=lambda v: list(v) if isinstance(v, set) else str(v)))
                def classify(st=state, qs=questions, rev=revision, fr=frame, sent=ctx.clock.now(),
                             opts=options, ps=pending_paths):
                    return ctx.decide(st, qs), rev, fr, sent, opts, ps
                s1_calls += 1
                s1 = ctx.spawn(classify)
            if game.tick_s:
                ctx.wait_for_tick()
            else:
                # This candidate targets real time, but do not spin in a turn-game adapter.
                if s1 is not None:
                    s1.result()
            frame += 1

        # Drain traced calls before runner closes the trace; retain a cold-start
        # compilation even if the first episode ended before Astra responded.
        if s2 is not None:
            accept_compiler(s2, copy.deepcopy(game.observe()))
        if s1 is not None:
            try:
                s1.result()
            except Exception:
                pass
        remember()
        ctx.log(event="policy_coverage", compiled_frames=compiled_frames,
                bootstrap_frames=bootstrap_frames, compiler_calls=s2_calls,
                search_frames=search_frames, world_audits=audited,
                world_cached=world is not None,
                cached_for_next_episode=policy is not None)


def build(ctx):
    return Harness(ctx)
