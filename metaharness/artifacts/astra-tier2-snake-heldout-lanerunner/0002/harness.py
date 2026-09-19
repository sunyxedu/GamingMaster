"""Compile rules into reactive strategies; classify strategy, not stale action.

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
    if not isinstance(rows, list) or len(rows) > 24:
        raise ValueError("expected at most 24 ordered features")
    result = []
    for row in rows:
        name = row["name"]
        if not isinstance(name, str) or name in [n for n, _ in result]:
            raise ValueError("invalid or duplicate feature name")
        result.append((name, expression(row["expr"])))
    return result


class Policy:
    def __init__(self, raw):
        self.raw = raw
        self.features = feature_list(raw.get("features", []))
        self.per_action = feature_list(raw.get("per_action", []))
        self.safe = expression(raw["safe"])
        modes = raw["modes"]
        if not isinstance(modes, list) or not 1 <= len(modes) <= 3:
            raise ValueError("expected one to three modes")
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
    for i, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return Policy(obj)
            except (ValueError, KeyError, TypeError, SyntaxError):
                continue
    raise ValueError("no valid policy JSON object")


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


COMPILER = """Compile the supplied game's rules into a SMALL reactive controller.
Your output will be executed on fresh observations every tick; a fast zero-shot
classifier chooses a reusable MODE, and the controller selects the maximum-utility
safe action in that mode. Do not return a fixed move or a route for this snapshot.
Return ONLY JSON with this schema (all expressions are Python expression STRINGS):
{
  "summary": "short rule/goal explanation for a classifier",
  "features": [{"name": "feature_name", "expr": "expression"}],
  "per_action": [{"name": "feature_name", "expr": "expression"}],
  "safe": "boolean expression",
  "modes": [{"when": "observable conditions under which to select this mode",
             "utility": "numeric expression, larger means better"}]
}
Environment: o is the current observation; actions is the list of legal action
names; last is the last POSTED action (not necessarily executed). f is a dictionary
of global features, evaluated in list order. per_action additionally has a (the
candidate action name) and x (its ordered feature dictionary). safe and utility
can use o, actions, last, f, a, x. Use f['name'] / x['name'] for earlier features.
Only expressions: indexing/slicing, literals, arithmetic excluding exponentiation,
comparisons, conditionals, Boolean operators, comprehensions, dictionary .get,
abs/min/max/sum/len/all/any/int/float/bool/list/tuple/set/sorted/enumerate/range.
range is capped at 256. NO imports, assignments, lambdas, custom functions, other
attributes/methods, recursion, search loops, or external state. At most 24 global
and 24 per-action expressions, and 1-3 modes. Keep the entire output concise.
Derive ALL game semantics and observation keys from the supplied material. Use
current observable values, never freeze coordinates, entity counts or targets.
Handle empty collections and default values. Resolve actions that preserve state
or are ignored to their actual effect before scoring; an unchanged action is not
automatically safe. safe must reject immediate fatal/invalid effects. Use explicit
observed safety information where available. Build small computable facts about
reward progress and future danger/escape options from the rules. Include future
viability in utility so greedy reward pursuit does not force a trap. Do not invent
unobserved dynamics. Utilities need only be comparable WITHIN each mode.
Mode 0 is the general default: avoid danger while making reward progress. Other
modes, if needed, are for clearly distinct observable situations. Each 'when' is
an evaluative classification criterion, not merely a name or action description.
The mode can be held for two seconds but action and safety are recalculated every
tick. Design modes to be useful even if held that long. Keep features compact.
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
        key = "reactive_modes_v1:" + hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        cached = ctx.memory.get(key, {})
        masks = list(cached.get("masks", []))
        policy = None
        last = game.action_spec.noop or names[0]
        failure = ""
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
        mode, mode_at = "m0", -100
        bootstrap_ranking = {name: 0.0 for name in names}
        bootstrap_at = -100
        history = []
        paths = mask_paths(first, names)
        classified = bool(masks) or not paths
        session = None
        next_compile = 0
        frame = 0
        decision_lifetime = 4 * (game.tick_s or 0.5)

        def remaining(kind):
            return ctx.budget.remaining().get(kind + "_calls", 0) > 0

        def remember():
            item = {"masks": masks}
            if policy is not None:
                item["policy"] = policy.raw
            ctx.memory[key] = item

        def accept_compiler(future, observation):
            nonlocal policy, session, revision, failure, next_compile, mode_at
            try:
                result = future.result()
                session = result.session_id
                candidate = parse_policy(result.text)
                candidate.resolve(first, names, last)
                candidate.resolve(observation, names, last)
                policy = candidate
                revision += 1
                mode_at = -100
                failure = ""
                remember()
                ctx.log(event="policy_installed", revision=revision, policy=policy.raw)
            except Exception as exc:
                failure = type(exc).__name__ + ": " + str(exc)[:500]
                next_compile = frame + 4
                ctx.log(event="compiler_error", error=failure)

        while not ctx.done:
            if ctx.clock.mode == "virtual":
                pending = [f for f in (s1, s2) if f is not None]
                if pending:
                    # A virtual driver can otherwise finish all ticks before a
                    # Python worker gets CPU time. This yields the CPU, not game time.
                    concurrent.futures.wait(pending, timeout=0.001,
                                            return_when=concurrent.futures.FIRST_COMPLETED)
            observation = copy.deepcopy(game.observe())
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

            if policy is None and s2 is None and frame >= next_compile and s2_calls < 3:
                if ctx.system2 is not None and remaining("system2"):
                    prompt = COMPILER + "\nRULES AND ACTIONS:\n" + rules
                    prompt += "\nRECENT OBSERVATIONS:\n" + json.dumps(history, default=str)
                    if failure:
                        prompt += "\nPrevious compiler failure; correct it: " + failure
                    s2_calls += 1
                    s2 = ctx.spawn(lambda p=prompt, sid=session: ctx.think(
                        p, session_id=sid, timeout_s=40.0))

            features, rows = {}, None
            if policy is not None:
                try:
                    features, rows = policy.resolve(observation, names, last)
                except Exception as exc:
                    failure = type(exc).__name__ + ": " + str(exc)[:500]
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

            if ctx.done:
                break
            ctx.act(action)
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


def build(ctx):
    return Harness(ctx)
