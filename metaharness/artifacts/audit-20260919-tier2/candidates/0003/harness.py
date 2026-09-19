"""Cold-start semantic bindings, then a compact Astra reactive policy.

All rule/field semantics come from traced model calls over public material.
No imports from another candidate and no policy cache across episodes.
"""
from __future__ import annotations

import ast
import concurrent.futures
import copy
import json
import math
import sys

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
        # A model-defined sufficient context for reusing a strategy judgment.
        # Without one, require equality of the full facts given to Jev.
        self.context = expression(raw["context"]) if "context" in raw else None
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
        context = evaluate(self.context, {**env, "rows": rows}) if self.context else {
            "features": env["f"], "candidates": rows}
        return env["f"], rows, context


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


def plain(value):
    return json.loads(json.dumps(value, default=lambda v: sorted(v, key=repr)
                                if isinstance(v, set) else str(v), allow_nan=False))


def fingerprint(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"))


def lookup(value, path):
    for part in path:
        value = value[part]
    return value


def numeric_vector(value):
    return (isinstance(value, (list, tuple)) and 1 <= len(value) <= 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


def discover(observation, names):
    """Discover shapes, never semantics or key-name substrings."""
    masks, vectors = [], []

    def walk(value, path):
        if len(path) > 4:
            return
        if numeric_vector(value) and len(vectors) < 12:
            vectors.append(path)
        if isinstance(value, dict):
            overlap = set(value).intersection(names)
            if (len(overlap) >= 2 and len(masks) < 4
                    and all(type(value[k]) is bool for k in overlap)):
                masks.append(path)
            for key, child in value.items():
                walk(child, path + (key,))
        # Deliberately do not enumerate dynamic entity arrays.

    walk(observation, ())
    return masks, vectors


def accepted(answer, probability=0.80, confidence=0.70):
    if answer is None:
        return False
    try:
        return (float(answer.confidence or 0) >= confidence
                and float(answer.probabilities.get(answer.value, 0)) >= probability)
    except (ValueError, TypeError, AttributeError):
        return False


def masked_actions(observation, names, masks):
    allowed = list(names)
    for path, unsafe_true in masks:
        try:
            values = lookup(observation, path)
            covered = [a for a in allowed if type(values.get(a)) is bool]
            safe = [a for a in covered if values[a] != unsafe_true]
            # An uncovered default may inherit unsafe motion. Prefer certified
            # entries whenever available, without assigning semantics to its name.
            if safe:
                allowed = safe
        except (KeyError, TypeError, AttributeError, IndexError):
            continue
    return allowed


COMPILER = """Compile a TINY reusable reactive policy from these public rules.
Direct JSON only; no tools/research/prose. Aim below 1400 characters. Runtime
already has provisional Jev semantic bindings; correct them if necessary.
Schema:
{"summary":"brief rule interpretation",
 "features":[{"name":"...","expr":"Python expression"}],
 "per_action":[{"name":"...","expr":"Python expression"}],
 "safe":"Boolean expression",
 "context":"small JSON regime descriptor for strategy classification",
 "modes":[{"when":"concrete criterion referencing context fields",
           "utility":"numeric expression"}]}
Expressions see o=current observation, actions=action names, last=last action,
f=ordered features. Per-action features, safe and utilities additionally see
a=action and x=ordered per-action features. context sees o, f, rows (computed
per-action safe/facts/utility). Jev chooses a mode using ONLY context and summary;
context must contain everything its mode criteria need. It is also a freshness
key: a mode answer is reusable only while context stays equal, for at most 2s.
Use 1-2 modes; mode 0 pursues reward with immediate safety. A second mode is
useful only for a real observable tradeoff such as uncertain risk vs progress.
Do not add a trivial second mode. Keep context small and independent of exact
position/time unless those values are necessary to classify the tradeoff.
At most 8 features per list. All safety/geometry/arithmetic belongs in expressions,
not in Jev's judgment. Account for ignored/default actions. Prefer explicit
observed safety masks. Derive EVERY field/action effect from public information.
No planning/simulation, no invented future events, no unnecessary exit analysis.
Syntax: literals, indexing/slicing, arithmetic except **, comparisons, if/else,
comprehensions, dict.get; abs min max sum len all any int float bool list tuple
set sorted enumerate range (range <=256). No statements/imports/lambdas/other
methods/custom functions. Safe must be bool, utilities finite. Produce the
smallest adequate policy now. Do not freeze dynamic sample values.
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
        tick_s = float(game.tick_s or 0.5)
        last = game.action_spec.noop or names[0]
        first = copy.deepcopy(game.observe())
        mask_paths, vector_paths = discover(first, names)
        masks, effects, evidence = [], {}, {}
        residual_path = None
        discovery_done = False
        effects_done = False
        policy = None
        session = None
        revision = 0
        s1 = s2 = None
        futures = []
        consumed = set()
        s1_calls = s2_calls = 0
        frame = 0
        failure = None
        rejected = ""
        samples = [first]
        mode_record = None
        direct_record = None
        previous = None
        mismatches = {}
        coverage = {"fallback": 0, "binding": 0, "compiled": 0}
        changes = {"jev_mask": 0, "jev_direct": 0, "jev_binding": 0, "jev_mode": 0, "astra": 0}

        def remaining(kind):
            return ctx.budget.remaining().get(kind + "_calls", 0) > 0

        def spawn(fn):
            future = ctx.spawn(fn)
            futures.append(future)
            return future

        def result(future):
            # Mark even a failing future consumed; its exception propagates.
            consumed.add(id(future))
            return future.result()

        def semantic_questions(observation):
            state = {"observation": observation, "vector_candidates": {
                "v" + str(i): {"path": list(path), "value": lookup(observation, path)}
                for i, path in enumerate(vector_paths)}}
            questions = {}
            for i, path in enumerate(mask_paths):
                questions["mask" + str(i)] = Choice(
                    instructions=rules + "\nClassify the RULE meaning of the Boolean map at "
                    + "`observation" + "".join("[" + repr(k) + "]" for k in path)
                    + "`. This is a reusable schema binding, not an action decision.",
                    criteria={"unsafe": "True explicitly means fatal, forbidden or unsafe; false permits the action.",
                              "safe": "True explicitly means safe or permitted; false rejects the action.",
                              "none": "Neither interpretation follows from the public rules."})
            if vector_paths:
                questions["residual"] = Choice(
                    instructions=rules + "\nWhich field in `vector_candidates` is ALREADY a signed "
                    "error/displacement to a rewarding objective, with reducing sum of absolute "
                    "components meaning progress? Classify the field's semantics, not its current "
                    "numbers. Require commensurate coordinates. Absolute locations, dimensions, "
                    "velocities and inventories do not qualify. Do not subtract other fields.",
                    criteria={**{"v" + str(i): "The field at " + repr(list(path))
                                + " is a signed objective-relative error vector whose zero is the objective."
                                for i, path in enumerate(vector_paths)},
                              "none": "No listed field explicitly meets this condition, or its units/rules are unclear."})
            if not questions:
                return None
            return state, questions, {"kind": "discover"}

        def effect_questions(observation):
            vector = lookup(observation, residual_path)
            codes = {}
            for axis in range(len(vector)):
                for delta in (-1, 1):
                    codes["e" + str(len(codes))] = [delta if j == axis else 0
                                                   for j in range(len(vector))]
            state = {"residual": {"path": list(residual_path), "value": vector},
                     "candidate_changes": codes}
            questions = {}
            for i, action in enumerate(names):
                questions["effect" + str(i)] = Choice(
                    instructions=rules + "\nFor action " + repr(action) + ", classify the "
                    "one-step ADDITIVE CHANGE TO `residual`, NOT to the controlled object's "
                    "absolute position. Assume the action is explicitly permitted by the live "
                    "safety mask and the same objective persists. Choose a constant unit change "
                    "only if entailed by the rules. For target-minus-position, residual change "
                    "has the opposite sign from position change. A state-dependent/default "
                    "action, nonunit change or unknown coordinate order requires none.",
                    criteria={**{code: "The residual changes by exactly " + repr(delta)
                                 + " for every permitted execution with the objective unchanged."
                                 for code, delta in codes.items()},
                              "none": "None of these constant unit-vector effects is established."})
            return state, questions, {"kind": "effects", "codes": codes}

        try:
            while not ctx.done:
                if ctx.clock.mode == "virtual":
                    pending = [f for f in (s1, s2) if f is not None]
                    if pending:
                        concurrent.futures.wait(pending, timeout=0.001,
                                                return_when=concurrent.futures.FIRST_COMPLETED)
                observation = copy.deepcopy(game.observe())
                now = ctx.clock.now()
                samples = (samples + [observation])[-4:]

                if s1 is not None and s1.done():
                    answers, meta = result(s1)  # ModelUnavailable is never swallowed.
                    s1 = None
                    kind = meta["kind"]
                    if kind == "discover":
                        discovery_done = True
                        for i, path in enumerate(mask_paths):
                            answer = answers.get("mask" + str(i))
                            if accepted(answer) and answer.value in ("unsafe", "safe"):
                                masks.append((path, answer.value == "unsafe"))
                                ctx.log(event="semantic_binding", role="safety", path=list(path),
                                        interpretation=answer.value, confidence=answer.confidence,
                                        probabilities=answer.probabilities, frame=frame)
                        answer = answers.get("residual")
                        opts = {"v" + str(i): p for i, p in enumerate(vector_paths)}
                        if accepted(answer) and answer.value in opts:
                            residual_path = opts[answer.value]
                            ctx.log(event="semantic_binding", role="objective_residual",
                                    path=list(residual_path), confidence=answer.confidence,
                                    probabilities=answer.probabilities, frame=frame)
                        else:
                            effects_done = True
                    elif kind == "effects":
                        effects_done = True
                        for i, action in enumerate(names):
                            answer = answers.get("effect" + str(i))
                            if accepted(answer) and answer.value in meta["codes"]:
                                effects[action] = meta["codes"][answer.value]
                                evidence[action] = {"confidence": answer.confidence,
                                                    "probability": answer.probabilities[answer.value]}
                        ctx.log(event="effects_bound", path=list(residual_path), effects=effects,
                                evidence=evidence, frame=frame)
                    elif kind == "mode":
                        answer = answers.get("select")
                        if accepted(answer, 0.70, 0.60) and answer.value in meta["options"]:
                            mode_record = (answer.value, meta["revision"], meta["key"], meta["sent"])
                    elif kind == "direct":
                        answer = answers.get("select")
                        if accepted(answer, 0.65, 0.55) and answer.value in meta["options"]:
                            direct_record = (meta["options"][answer.value], meta["key"], meta["sent"])
                    ctx.log(event="jev_received", kind=kind, frame=frame,
                            age_s=now - meta["sent"], accepted_effects=len(effects))

                if s2 is not None and s2.done():
                    response = result(s2)  # Inference failures propagate, including unavailable.
                    s2 = None
                    session = response.session_id
                    try:
                        candidate = parse_policy(response.text)
                        for sample in [first, *samples]:
                            candidate.resolve(sample, names, last)
                        policy = candidate
                        revision += 1
                        mode_record = None
                        failure = None
                        ctx.log(event="policy_installed", frame=frame, revision=revision,
                                policy=policy.raw, output_chars=len(response.text))
                    except (ValueError, TypeError, KeyError, IndexError, SyntaxError,
                            ArithmeticError, AttributeError, NameError) as exc:
                        failure = type(exc).__name__ + ": " + str(exc)[:400]
                        rejected = response.text[:8000]
                        ctx.log(event="compiler_rejected", frame=frame, error=failure)

                # Test reusable effects against actual public observations. A
                # zero predicted residual may cause objective replacement, so is
                # deliberately not a transition-validation case.
                if previous is not None and policy is None and residual_path is not None:
                    old_vector, taken, sent = previous
                    if taken in effects and 0.5 * tick_s <= now - sent <= 1.5 * tick_s:
                        try:
                            current = lookup(observation, residual_path)
                            expected = [v + d for v, d in zip(old_vector, effects[taken])]
                            if numeric_vector(current) and len(current) == len(expected) and any(expected):
                                match = all(abs(x - y) < 1e-8 for x, y in zip(expected, current))
                                mismatches[taken] = 0 if match else mismatches.get(taken, 0) + 1
                                ctx.log(event="binding_audit", action=taken, match=match,
                                        expected=expected, actual=current, frame=frame)
                                if mismatches[taken] >= 2:
                                    del effects[taken]
                                    ctx.log(event="binding_revoked", action=taken, frame=frame)
                        except (KeyError, IndexError, TypeError):
                            effects.clear()
                previous = None

                allowed = masked_actions(observation, names, masks)
                unshielded = last if last in names else names[0]
                fallback = last if last in allowed else allowed[0]
                shield_choice = fallback
                if direct_record is not None:
                    selected, key, sent = direct_record
                    if key == fingerprint(observation) and now - sent <= 2 * tick_s and selected in allowed:
                        fallback = selected
                bridge_action = fallback
                bridge_costs = {}
                if masks and residual_path is not None:
                    try:
                        vector = lookup(observation, residual_path)
                        if numeric_vector(vector):
                            # Use ONLY explicitly certified safe actions, not
                            # uncovered actions retained by emergency fallback.
                            for action in allowed:
                                certified = any(type(lookup(observation, path).get(action)) is bool
                                                and lookup(observation, path)[action] != polarity
                                                for path, polarity in masks)
                                if certified and action in effects and len(effects[action]) == len(vector):
                                    bridge_costs[action] = sum(abs(v + d) for v, d in zip(vector, effects[action]))
                            if bridge_costs:
                                bridge_action = min(bridge_costs, key=lambda a: (bridge_costs[a], a != last))
                    except (KeyError, TypeError, IndexError, AttributeError):
                        bridge_costs = {}
                action = bridge_action
                source = "binding" if bridge_costs else "fallback"
                question_job = None
                base_action = None
                selected_mode = "m0"

                if policy is not None:
                    try:
                        features, rows, context = policy.resolve(observation, names, last)
                        safe = [a for a in allowed if rows[a]["safe"]]
                        # Do not override an established shield on disagreement.
                        if safe:
                            allowed = safe
                            key = fingerprint([revision, context, allowed])
                            if mode_record is not None:
                                choice, rev, old_key, sent = mode_record
                                if rev == revision and old_key == key and now - sent <= 4 * tick_s:
                                    selected_mode = choice
                                else:
                                    ctx.log(event="mode_stale", frame=frame, age_s=now - sent)
                                    mode_record = None
                            base_action = max(allowed, key=lambda a: (rows[a]["utility"]["m0"], a == last))
                            action = max(allowed, key=lambda a: (rows[a]["utility"][selected_mode], a == last))
                            source = "compiled"
                            state = {"rules_summary": policy.summary, "context": plain(context)}
                            options = {m: m for m, _, _ in policy.modes}
                            criteria = {m: when for m, when, _ in policy.modes}
                            criteria["none"] = "No supplied criterion is supported by context; use the code default."
                            questions = {"select": Choice(
                                instructions="Using `context` and `rules_summary`, classify which strategy's "
                                "stated condition holds. This selects a scoring rule recomputed on fresh "
                                "observations, not an action. Choose m0 unless a different criterion has "
                                "concrete support; none if the context cannot establish any criterion.",
                                criteria=criteria)}
                            question_job = (state, questions, {"kind": "mode", "key": key,
                                                             "revision": revision, "options": options})
                        else:
                            ctx.log(event="policy_safety_disagreement", frame=frame, allowed=allowed)
                    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError,
                            AttributeError, NameError) as exc:
                        failure = type(exc).__name__ + ": " + str(exc)[:400]
                        rejected = json.dumps(policy.raw)
                        policy = None
                        revision += 1
                        ctx.log(event="policy_runtime_error", frame=frame, error=failure)

                if ctx.done:
                    break
                # Never wait for either model before posting a fresh action.
                ctx.act(action)
                coverage[source] += 1
                interventions = []
                if source == "fallback" and shield_choice != unshielded and fallback == shield_choice:
                    changes["jev_mask"] += 1
                    interventions.append("jev_mask")
                if source == "fallback" and fallback != shield_choice:
                    changes["jev_direct"] += 1
                    interventions.append("jev_direct")
                if source == "binding" and bridge_action != fallback:
                    changes["jev_binding"] += 1
                    interventions.append("jev_binding")
                if source == "compiled" and base_action != bridge_action:
                    changes["astra"] += 1
                    interventions.append("astra")
                if source == "compiled" and action != base_action:
                    changes["jev_mode"] += 1
                    interventions.append("jev_mode")
                ctx.log(event="control", frame=frame, action=action, source=source,
                        allowed=allowed, unshielded=unshielded, shield_choice=shield_choice,
                        direct_choice=fallback,
                        binding_choice=bridge_action, binding_costs=bridge_costs,
                        compiled_default=base_action, mode=selected_mode,
                        interventions=interventions, revision=revision)
                if residual_path is not None and policy is None:
                    try:
                        previous = (copy.deepcopy(lookup(observation, residual_path)), action, now)
                    except (KeyError, IndexError, TypeError):
                        pass
                last = action

                if policy is None and s2 is None and ctx.system2 is not None and s2_calls < 4 and remaining("system2"):
                    prompt = COMPILER + "\nRULES/ACTIONS:\n" + rules
                    prompt += "\nOBSERVATION:\n" + json.dumps(observation, default=str)
                    prompt += "\nPROVISIONAL BINDINGS:\n" + json.dumps({
                        "masks": masks, "residual_path": residual_path, "effects": effects})
                    if failure:
                        prompt += "\nREPAIR ERROR:\n" + failure + "\nREJECTED:\n" + rejected
                    s2_calls += 1
                    ctx.log(event="compiler_requested", frame=frame, attempt=s2_calls, repair=bool(failure))
                    s2 = spawn(lambda p=prompt, sid=session: ctx.think(p, session_id=sid))

                if s1 is None and ctx.system1 is not None and s1_calls < 240 and remaining("system1"):
                    if not discovery_done:
                        question_job = semantic_questions(observation)
                        if question_job is None:
                            discovery_done = effects_done = True
                    elif residual_path is not None and not effects_done:
                        question_job = effect_questions(observation)
                    elif policy is None and not bridge_costs:
                        options = {"a" + str(i): a for i, a in enumerate(allowed)}
                        state = {"observation": observation, "permitted_actions": allowed}
                        questions = {"select": Choice(
                            instructions=rules + "\nFrom `permitted_actions`, select immediate reward "
                            "progress that avoids imminent failure. none if the facts are insufficient.",
                            criteria={**{code: "Executing " + repr(a) + " now is safest rewarding progress: "
                                        + game.action_spec.descriptions.get(a, a) for code, a in options.items()},
                                      "none": "No listed action is supported by current facts."})}
                        question_job = (state, questions, {"kind": "direct", "options": options,
                                                         "key": fingerprint(observation)})
                    if question_job is not None:
                        state, questions, meta = question_job
                        meta["sent"] = ctx.clock.now()
                        s1_calls += 1
                        s1 = spawn(lambda st=plain(state), qs=questions, md=meta: (ctx.decide(st, qs), md))
                if game.tick_s:
                    ctx.wait_for_tick()
                elif s1 is not None:
                    # Tier-1 adapter: actions step synchronously, but give the
                    # outstanding inference time to finish before next action.
                    concurrent.futures.wait([s1])
                frame += 1
        finally:
            # Join EVERY spawned task on normal end AND exceptional exit. Never
            # cancel a running model and never turn ModelUnavailable into a score.
            active_error = sys.exc_info()[1]
            unavailable = lambda exc: exc is not None and any(
                cls.__name__ == "ModelUnavailable" for cls in type(exc).__mro__)
            drain_error = None
            for future in futures:
                if id(future) not in consumed:
                    try:
                        result(future)
                    except BaseException as exc:
                        if drain_error is None or unavailable(exc):
                            drain_error = exc
            ctx.log(event="coverage", frames=coverage, control_changes=changes,
                    system1_calls=s1_calls, system2_calls=s2_calls,
                    outstanding=sum(not f.done() for f in futures))
            if drain_error is not None and not unavailable(active_error) and (
                    active_error is None or unavailable(drain_error)):
                raise drain_error


def build(ctx):
    return Harness(ctx)
