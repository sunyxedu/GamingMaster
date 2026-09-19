"""Checked feedback policies synthesized in-episode, plus contingent-plan fallback."""
from __future__ import annotations

import copy
import ast
import json
import operator
import re
from collections import Counter


MISSING = {"$missing": True}


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def at(value, path):
    try:
        for key in path:
            value = value[key]
        return value
    except (KeyError, IndexError, TypeError):
        return MISSING


def matches(value, target, path=()):
    """Find all exact subvalues, returning paths relative to the selected value."""
    if value == target:
        return [list(path)]
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return []
    out = []
    for key, child in items:
        out.extend(matches(child, target, path + (key,)))
    return out


def features(obs, schema):
    out = {}
    for name, spec in schema.items():
        value = at(obs, spec["path"])
        if "find" in spec and value != MISSING:
            value = matches(value, spec["find"])
            if spec.get("count") is True:
                value = len(value)
        out[name] = value
    return out


def differences(before, after, path=(), limit=100):
    """Generic JSON delta, with explicit overflow rather than silent omission."""
    out = []

    def walk(a, b, p):
        if len(out) > limit or a == b:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys(), key=str):
                walk(a.get(key, MISSING), b.get(key, MISSING), p + (key,))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, p + (i,))
        else:
            out.append({"path": list(p), "was": a, "now": b})

    walk(before, after, path)
    if len(out) > limit:
        return out[:limit] + [{"overflow": "More changes; inspect current observation."}]
    return out


def render(value, depth=0):
    """Preserve field names; put rectangular scalar arrays on separate rows."""
    pad = "  " * depth
    if isinstance(value, dict):
        return "{\n" + ",\n".join(
            pad + "  " + dump(k) + ": " + render(v, depth + 1)
            for k, v in value.items()) + "\n" + pad + "}"
    if (isinstance(value, list) and value and all(isinstance(r, list) for r in value)
            and all(not isinstance(x, (list, dict)) for r in value for x in r)):
        return "[\n" + ",\n".join(pad + "  " + dump(r) for r in value) + "\n" + pad + "]"
    return dump(value)


def action_text(value, names):
    if isinstance(value, dict):
        name = value.get("name", value.get("action"))
        args = value.get("args", [])
        if not isinstance(name, str) or not isinstance(args, list):
            return None
        if not all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in args):
            return None
        value = " ".join([name] + [str(x) for x in args])
    if not isinstance(value, str):
        return None
    tokens = value.strip().split()
    if not tokens:
        return None
    canon = next((n for n in names if n.casefold() == tokens[0].casefold()), None)
    # Preserve scalar arguments, but never interpret prose as an action.
    if canon is None or any(not re.fullmatch(r"[-+\w.,:=/]+", t) for t in tokens[1:]):
        return None
    return " ".join([canon] + tokens[1:])


def parse_reply(text, names):
    decoder = json.JSONDecoder()
    # Accept fenced JSON and explanatory prefixes. Prefer an object with a plan.
    candidates = []
    for m in re.finditer(r"[\[{]", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and any(k in obj for k in ("plan", "actions", "policy")):
            return obj
        if isinstance(obj, list) and obj and all(isinstance(x, (str, dict)) for x in obj):
            candidates.append({"plan": obj})
    if candidates:
        return candidates[0]
    # A bare action is useful, but takes one step only; no arbitrary default.
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('`"')
        tokens = line.split()
        if tokens and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", x) for x in tokens[1:]):
            action = action_text(line, names)
            if action:
                return {"plan": [{"action": action}]}
    return {}


def validate_tree(plan, schema, names):
    """Validate every reachable branch before spending any game actions."""
    nodes = 0

    def check_map(value):
        return isinstance(value, dict) and all(k in schema for k in value)

    def walk(sequence, depth):
        nonlocal nodes
        if not isinstance(sequence, list) or len(sequence) > 24 or depth > 4:
            return "Plans require lists of at most 24 steps, with at most four case layers."
        for index, raw in enumerate(sequence):
            nodes += 1
            if nodes > 64:
                return "Tree exceeds 64 total action nodes."
            item = {"action": raw} if isinstance(raw, str) else raw
            if not isinstance(item, dict):
                return "Each step must be an action string or object."
            value = item.get("action", item)
            if isinstance(value, str) and "args" in item:
                value = {"name": value, "args": item["args"]}
            if action_text(value, names) is None:
                return "Illegal action in tree."
            if not check_map(item.get("before", {})) or not check_map(item.get("expect", {})):
                return "before/expect must map defined features to exact values."
            if "cases" not in item:
                continue
            cases = item["cases"]
            if index != len(sequence) - 1:
                return "A cases step must end its containing list; each then is the entire continuation."
            if item.get("expect"):
                return "Use either expect or cases on a step, not both."
            if not isinstance(cases, list) or not 1 <= len(cases) <= 4:
                return "cases requires one to four outcomes."
            for case in cases:
                if not isinstance(case, dict) or not check_map(case.get("when")):
                    return "Each case requires when mapping defined features to exact values."
                if not any(schema[k].get("role", "state") == "state" for k in case["when"]):
                    return "A case must test at least one state feature, not just a clock/boundary."
                error = walk(case.get("then"), depth + 1)
                if error:
                    return error
        return None

    return walk(plan, 0)


def matching_cases(cases, before, after, schema):
    """A case is an exact outcome plus an implicit unchanged-state frame."""
    changed = {k for k, spec in schema.items()
               if spec.get("role", "state") == "state" and before[k] != after[k]}
    # Missing schema paths must never accidentally authorize a continuation.
    if any(value == MISSING for value in after.values()):
        return []
    return [i for i, case in enumerate(cases)
            if changed.issubset(case["when"])
            and all(after[k] == v for k, v in case["when"].items())]


class PolicyError(Exception):
    pass


class Flow(Exception):
    def __init__(self, kind, value=None):
        self.kind, self.value = kind, value


class Policy:
    """Small Python-shaped data interpreter. Never exec/eval generated source.

    Inputs are plain copied data; the evaluator has no reference to ctx or game.
    Each invocation has a fresh instruction budget. Only enumerated operations exist.
    """
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
    compare = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
               ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
               ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y,
               ast.Is: operator.is_, ast.IsNot: operator.is_not}
    builtin = {"len": len, "range": range, "enumerate": enumerate, "zip": zip,
               "list": list, "tuple": tuple, "dict": dict, "set": set,
               "min": min, "max": max, "abs": abs, "sum": sum,
               "sorted": sorted, "reversed": reversed, "all": all, "any": any,
               "int": int, "str": str, "bool": bool}
    methods = {dict: {"get", "items", "keys", "values", "copy", "pop", "setdefault"},
               list: {"append", "pop", "copy", "count", "index", "reverse"},
               tuple: {"count", "index"}, set: {"add", "discard", "copy"}}

    def __init__(self, source):
        if not isinstance(source, str) or len(source) > 24000:
            raise PolicyError("policy must be a source string of at most 24000 characters")
        source = re.sub(r"^```(?:python)?\s*\n|\n```\s*$", "", source.strip())
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise PolicyError("provide only def policy(obs, features, memory, event)")
        fn = tree.body[0]
        args = fn.args
        if (fn.name != "policy" or [a.arg for a in args.args] != ["obs", "features", "memory", "event"]
                or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
                or args.defaults or fn.decorator_list or fn.returns
                or any(a.annotation for a in args.args)):
            raise PolicyError("signature must be policy(obs, features, memory, event), without annotations")
        allowed = (ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign,
                   ast.If, ast.For, ast.While, ast.Return, ast.Break, ast.Continue, ast.Pass,
                   ast.Expr, ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Dict, ast.Set,
                   ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
                   ast.IfExp, ast.Call, ast.Attribute, ast.keyword, ast.ListComp,
                   ast.SetComp, ast.DictComp, ast.comprehension, ast.Load, ast.Store,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                   ast.Not, ast.UAdd, ast.USub, ast.And, ast.Or,
                   ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                   ast.In, ast.NotIn, ast.Is, ast.IsNot)
        nodes = list(ast.walk(fn))
        if len(nodes) > 5000:
            raise PolicyError("policy AST too large")
        for node in nodes:
            if not isinstance(node, allowed) or isinstance(node, ast.FunctionDef) and node is not fn:
                raise PolicyError("unsupported syntax: " + type(node).__name__)
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise PolicyError("private names unavailable")
            if isinstance(node, ast.Attribute) and node.attr not in set().union(*self.methods.values()):
                raise PolicyError("unsupported method: " + node.attr)
            if isinstance(node, ast.comprehension) and node.is_async:
                raise PolicyError("async unavailable")
        self.body = fn.body
        self.source = source

    def spend(self):
        self.fuel -= 1
        if self.fuel < 0:
            raise PolicyError("instruction limit (120000 AST operations)")

    def bound(self, value):
        if isinstance(value, (list, tuple, dict, set, str, range)) and len(value) > 8192:
            raise PolicyError("container limit (8192 elements/characters)")
        if isinstance(value, int) and value.bit_length() > 256:
            raise PolicyError("integer size limit")
        return value

    def materialize(self, value):
        out = []
        for x in value:
            self.spend()
            out.append(x)
            self.bound(out)
        return out

    def operation(self, op, left, right):
        if isinstance(op, ast.Mult):
            if isinstance(left, (list, tuple, str)) and isinstance(right, int):
                if len(left) * max(0, right) > 8192:
                    raise PolicyError("sequence multiplication too large")
            if isinstance(right, (list, tuple, str)) and isinstance(left, int):
                if len(right) * max(0, left) > 8192:
                    raise PolicyError("sequence multiplication too large")
        if isinstance(op, ast.Mod) and isinstance(left, str):
            raise PolicyError("string formatting unavailable")
        return self.bound(self.binary[type(op)](left, right))

    def bind(self, target, value, env):
        self.spend()
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = self.materialize(value)
            if len(values) != len(target.elts):
                raise PolicyError("unpacking length mismatch")
            for t, v in zip(target.elts, values):
                self.bind(t, v, env)
        elif isinstance(target, ast.Subscript):
            obj, key = self.expr(target.value, env), self.expr(target.slice, env)
            if not isinstance(obj, (list, dict)) or isinstance(key, slice):
                raise PolicyError("only list/dict item assignment supported")
            obj[key] = value
            self.bound(obj)
        else:
            raise PolicyError("unsupported assignment target")

    def expr(self, node, env):
        self.spend()
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            if type(node.value) not in (str, int, float, bool, type(None)):
                raise PolicyError("unsupported constant")
            return self.bound(node.value)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            vals = [self.expr(x, env) for x in node.elts]
            return self.bound(tuple(vals) if isinstance(node, ast.Tuple) else
                              set(vals) if isinstance(node, ast.Set) else vals)
        if isinstance(node, ast.Dict):
            if None in node.keys:
                raise PolicyError("dictionary unpacking unavailable")
            return self.bound({self.expr(k, env): self.expr(v, env) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.Subscript):
            return self.bound(self.expr(node.value, env)[self.expr(node.slice, env)])
        if isinstance(node, ast.Slice):
            return slice(self.expr(node.lower, env), self.expr(node.upper, env), self.expr(node.step, env))
        if isinstance(node, ast.BinOp):
            return self.operation(node.op, self.expr(node.left, env), self.expr(node.right, env))
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            return {ast.Not: operator.not_, ast.UAdd: operator.pos, ast.USub: operator.neg}[type(node.op)](value)
        if isinstance(node, ast.BoolOp):
            value = self.expr(node.values[0], env)
            for child in node.values[1:]:
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    break
                value = self.expr(child, env)
            return value
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            for op, child in zip(node.ops, node.comparators):
                right = self.expr(child, env)
                if not self.compare[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            result = []
            local = dict(env)

            def generate(index):
                self.spend()
                if index == len(node.generators):
                    result.append((self.expr(node.key, local), self.expr(node.value, local))
                                  if isinstance(node, ast.DictComp) else self.expr(node.elt, local))
                    self.bound(result)
                    return
                gen = node.generators[index]
                for val in self.materialize(self.expr(gen.iter, local)):
                    self.bind(gen.target, val, local)
                    if all(self.expr(test, local) for test in gen.ifs):
                        generate(index + 1)

            generate(0)
            return dict(result) if isinstance(node, ast.DictComp) else set(result) if isinstance(node, ast.SetComp) else result
        if isinstance(node, ast.Call):
            args = [self.expr(x, env) for x in node.args]
            if any(k.arg is None for k in node.keywords):
                raise PolicyError("keyword unpacking unavailable")
            kw = {k.arg: self.expr(k.value, env) for k in node.keywords}
            if isinstance(node.func, ast.Name) and node.func.id in self.builtin:
                name = node.func.id
                if name == "range":
                    return self.bound(range(*args, **kw))
                if name == "str" and args and not isinstance(args[0], (str, int, float, bool, type(None))):
                    raise PolicyError("str accepts scalars only")
                value = self.builtin[name](*args, **kw)
                if name in ("zip", "enumerate", "reversed"):
                    value = self.materialize(value)
                return self.bound(value)
            if isinstance(node.func, ast.Attribute):
                obj = self.expr(node.func.value, env)
                name = node.func.attr
                if name not in self.methods.get(type(obj), set()):
                    raise PolicyError("method unavailable on this data type: " + name)
                value = getattr(obj, name)(*args, **kw)
                self.bound(obj)
                if name in ("items", "keys", "values"):
                    value = self.materialize(value)
                return self.bound(value)
            raise PolicyError("only listed builtins and data methods are callable")
        raise PolicyError("unsupported expression: " + type(node).__name__)

    def block(self, nodes, env):
        for node in nodes:
            self.spend()
            if isinstance(node, ast.Return):
                raise Flow("return", self.expr(node.value, env))
            if isinstance(node, ast.Assign):
                value = self.expr(node.value, env)
                for target in node.targets:
                    self.bind(target, value, env)
            elif isinstance(node, ast.AugAssign):
                value = self.operation(node.op, self.expr(node.target, env), self.expr(node.value, env))
                self.bind(node.target, value, env)
            elif isinstance(node, ast.Expr):
                self.expr(node.value, env)
            elif isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, (ast.For, ast.While)):
                # For iterables are snapshotted. Growing queues should use while + index.
                iterator = iter(self.materialize(self.expr(node.iter, env))) if isinstance(node, ast.For) else None
                broken = False
                while True:
                    self.spend()
                    if iterator is not None:
                        try:
                            value = next(iterator)
                        except StopIteration:
                            break
                        self.bind(node.target, value, env)
                    elif not self.expr(node.test, env):
                        break
                    try:
                        self.block(node.body, env)
                    except Flow as flow:
                        if flow.kind == "return":
                            raise
                        if flow.kind == "break":
                            broken = True
                            break
                if not broken:
                    self.block(node.orelse, env)
            elif isinstance(node, ast.Break):
                raise Flow("break")
            elif isinstance(node, ast.Continue):
                raise Flow("continue")
            elif not isinstance(node, ast.Pass):
                raise PolicyError("unsupported statement: " + type(node).__name__)

    def run(self, obs, values, memory, event):
        self.fuel = 120000
        env = copy.deepcopy({"obs": obs, "features": values, "memory": memory, "event": event})
        result = None
        try:
            self.block(self.body, env)
        except Flow as flow:
            if flow.kind != "return":
                raise PolicyError("break/continue outside loop")
            result = flow.value
        # Outputs and persistent local memory must remain acyclic, bounded JSON data.
        # Walk first to avoid serializing exponentially shared container structures.
        count = [0]

        def check(value, depth=0):
            count[0] += 1
            if count[0] > 24000 or depth > 40:
                raise PolicyError("output/memory size or depth limit")
            if isinstance(value, dict):
                if not all(isinstance(k, str) for k in value):
                    raise PolicyError("output/memory dictionary keys must be strings")
                for v in value.values():
                    check(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    check(v, depth + 1)
            elif type(value) not in (str, int, float, bool, type(None)):
                raise PolicyError("output/memory must be JSON data; convert tuples to lists")
            self.bound(value)

        check(result)
        check(env["memory"])
        if not isinstance(env["memory"], dict):
            raise PolicyError("memory must stay a dictionary")
        encoded = json.dumps([result, env["memory"]], allow_nan=False)
        if len(encoded) > 64000:
            raise PolicyError("output/memory exceeds 64000 characters")
        return result, env["memory"]


PROTOCOL = """Control this unfamiliar turn-based game using only its description and measured
transitions. Maximize objectives completed, then minimize reasoning calls. You can delegate
repeatable planning to a checked feedback POLICY that runs without another model call.
Initially gather a few informative experiments. Once basic effects are supported, prefer a
policy which reads the CURRENT observation and computes the next useful action, rather than
listing a route tied to the current scene. Reuse episode rules, never stale scene coordinates.
If a general policy cannot yet be justified, use checked batches or short probes and replan.
Never assume action meanings, colours, field names, dimensions or mechanics from other games.
Return one JSON object (fenced JSON is accepted):
{
 "schema":{"feature":{"path":["actual","key"],"role":"state"}},
 "hypotheses":[{"id":"h1","claim":"conditional effect","actions":["legal name"],"evidence":[1]}],
 "strategy":"objective, uncertainty, next discriminating test",
 "plan":[{"action":"legal name","before":{},"expect":{},"uses":["h1"]}]
}
SCHEMA optional after first call; if supplied, REPLACES the previous schema. Paths contain
actual keys and zero-based indices. Optional find:VALUE returns ALL relative paths matching
that value: finding one token in a matrix returns [[row,column]], not [row,column]. Optional
count:true with find gives the count. Interpret values using observations/description.
Select controllable state, goal/mechanisms/inventory and stage boundaries. role:state is the
default; unpredicted state changes stop execution. role:boundary detects any scene/stage change.
role:clock is for time/action counters, excluded from loop checks. Avoid tracking a whole
changing board if a few extracted features suffice. Complete raw observations remain available.

CHECKED STEPS: action is a legal name plus optional scalar arguments, or {name:...,args:[...]}.
before/expect are feature-name-to-exact-value dictionaries, checked before/after EACH action.
Unlisted state features implicitly stay unchanged unless probe:true. Mismatch discards the
suffix and requests reasoning. uses lists hypothesis ids; counterexamples challenge them.
A missing selected path prevents policy execution. Every known move should predict its effect.
plan has at most 24 actions. Explicit probe:true permits up to FOUR consecutive unchecked
experiments; other unchecked steps run once and replan. Discover only what is needed for progress.
Blocked in one context does not imply an action is useless. Avoid repeating failed state/actions.
Boundary changes always discard a static plan. Comparisons at a scene replacement are recorded
as out-of-scope, not automatically treated as a refutation of a within-scene action rule.

CASES: a step may have cases instead of expect. Each case is
{"when":{"feature":"exact post-action value"},"then":[...checked continuation...]}
The cases step MUST be last in its list. One to four cases, at most four nested layers and
64 total action nodes. Each when tests at least one state feature and covers ALL state features
changed by that outcome; unlisted state features must be unchanged. Exactly one case must match,
otherwise request reasoning. No wildcard/default. then:[] replans for static batches. Account
for occluded tokens and inventory changes. Cases represent possibilities, not proven rules.

REUSABLE FEEDBACK POLICY (preferred once effects are known): Instead of plan, return
"policy": "def policy(obs, features, memory, event):\n    ...",
"policy_across_boundaries": true
The source is a JSON string containing ONE function with exactly that signature. It runs on
copied data after every observed step, without another reasoning call, and returns ONE checked
step in the same format above, or {"replan":"why new reasoning is needed"}. It cannot call
the game or a model. obs is the current raw observation, features the current schema extraction,
memory a mutable JSON dict initially {}, and event the last numbered transition (initially {}).
Use memory for a search/cache or controlled experiment stage. Event includes before, after,
delta, action, and boundary, if any. Runtime memory is per policy installation and cleared on
EVERY scene boundary. Put supported episode-level action effects in source constants, learned
from this episode's evidence. Retain all still-supported hypotheses with evidence ids.
Return before with at least one state feature. Return expect for known actions; include all
changed selected state features. A policy may return cases with EMPTY then lists; a unique
match calls the policy again on the new observation. It may use probe:true to collect up to
four unchecked outcomes, then must return a checked action or request reasoning. Unexpected
checked outcomes always disable the policy; it cannot silently redefine a refuted prediction.
For an action that may complete a scene, set boundary_ok:true on THAT returned step. With
policy_across_boundaries:true, an observed boundary then clears memory and invokes the policy
on the replacement observation. Same-scene outcomes still undergo normal checks. Boundaries
without both authorizations replan. Terminal status always stops. A policy must handle new
layouts/state values by fresh computation or return replan; never assume future scenes repeat.
A supported transition model plus bounded graph search can compute a fresh route; a finite
state controller can manage nonspatial tasks. The harness supplies neither solver nor rules.
Prefer ordinary supported routes. If uncertainty prevents a route, select a specific useful
probe with plausible cases or request reasoning; don't invent passability or action effects.
No need to model every mechanic before acting. Keep strategies simple and code short.

POLICY LANGUAGE: a restricted, interpreted Python subset (NOT full Python): assignment,
augmented assignment, if/elif/else, for, while, break, continue, return; lists/tuples/sets/dicts,
indexing, slicing, unpacking, + - * / // %, comparisons/in/not in/is, and/or/not, conditional
expressions, list/set/dict comprehensions. Supported builtins: len, range, enumerate, zip,
list, tuple, dict, set, min, max, abs, sum, sorted, reversed, all, any, int, str, bool.
Methods: dict.get/items/keys/values/copy/pop/setdefault; list.append/pop/copy/count/index/reverse;
set.add/discard/copy; tuple.count/index. Calls can use keyword arguments. str only accepts scalars.
No imports, other functions, recursion, lambdas, generators, try/except, attributes except listed
method calls, f-strings, annotations, starred unpacking or external APIs. For-loops snapshot their
iterable; growing work queues need while with an index or pop. Tuple keys/sets are fine for local
search. Returned values and memory MUST be JSON data (lists, string-keyed dicts, scalars).
Budgets per invocation: 120000 interpreted operations, 8192 entries per container, 64000 chars
of output+memory. At most 160 actions per installation; repeated states also stop execution.
Source size cap 24000 chars. Errors consume NO game action and return to reasoning with details.
Do not call unsupported helpers. Use normal Python True/False/None INSIDE source strings.
A policy is optional; reliable checked batches remain available if it is not useful here.
"""


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def remaining(self, calls):
        budget = self.ctx.budget
        if callable(getattr(budget, "remaining", None)):
            rem = budget.remaining()
            if isinstance(rem, dict) and "system2_calls" in rem:
                return max(0, int(rem["system2_calls"]))
        if hasattr(budget, "max_system2_calls") and hasattr(budget, "system2_calls"):
            return max(0, budget.max_system2_calls - budget.system2_calls)
        return max(0, 15 - calls)

    def run_episode(self, game):
        ctx = self.ctx
        names = list(game.action_spec.names)
        if not names:
            return
        description = str(game.describe())
        action_spec = str(game.action_spec.describe())
        obs = copy.deepcopy(game.observe())
        schema, hypotheses, history, scenes = {}, [], [], []
        strategy, feedback = "Discover enough to act effectively.", "New episode; no rules known."
        visits = Counter()
        calls = steps = interrupts = failed_checks = branch_hits = branch_misses = 0
        memory_key = "feedback_policy_v3"
        last_source = ""
        policy_steps = policy_installs = policy_errors = boundary_resumes = 0
        previous = ctx.memory.get(memory_key, {})
        if not isinstance(previous, dict):
            previous = {}
        prior_stats = {k: previous.get(k, 0) for k in ("episodes", "steps", "calls", "interrupts")}
        scenes.append({"at_step": 0, "observation": copy.deepcopy(obs)})

        def signature(values):
            selected = {k: v for k, v in values.items() if schema[k].get("role") != "clock"}
            return dump(selected) if selected else dump(obs)

        def evidence():
            # Keep the first and latest experiment for every action, all recent steps,
            # and recent counterexamples. These are measured outcomes, not model summaries.
            chosen = set(range(max(0, len(history) - 16), len(history)))
            for name in names:
                found = [i for i, e in enumerate(history) if e["action"].split()[0] == name]
                if found:
                    chosen.update((found[0], found[-1]))
            chosen.update(i for i, e in enumerate(history) if e.get("failure") or "case_result" in e)
            selected = sorted(chosen)
            # Prioritize recent counterexamples if an unusually long game fills the ledger.
            return [history[i] for i in selected[-48:]]

        while not ctx.done and steps < 1000 and calls < 15:
            remaining = min(15 - calls, self.remaining(calls))
            if remaining <= 0:
                break
            current = features(obs, schema)
            # Fresh sessions keep prompts bounded. Everything needed is explicit.
            prompt = (PROTOCOL + "\nGAME DESCRIPTION:\n" + description +
                      "\nACTION SPECIFICATION:\n" + action_spec +
                      "\nCALLS LEFT INCLUDING THIS ONE: " + str(remaining) +
                      "\nCURRENT OBSERVATION (array coordinates are zero-based):\n" + render(obs) +
                      "\nCURRENT SCHEMA: " + dump(schema) +
                      "\nCURRENT FEATURES: " + dump(current) +
                      "\nSTRATEGY: " + strategy +
                      "\nPREVIOUS POLICY SOURCE (for repair, inactive): " + last_source +
                      "\nHYPOTHESES (untrusted unless supported): " + dump(hypotheses) +
                      "\nTRANSITION EVIDENCE: " + dump(evidence()) +
                      "\nRECENT SCENE STARTS: " + dump(scenes[-2:]) +
                      "\nCURRENT STATE VISITS: " + str(visits[signature(current)]) +
                      "\nEXECUTOR FEEDBACK: " + feedback +
                      "\nPRIOR EPISODE PROCEDURAL STATISTICS ONLY: " + dump(prior_stats) +
                      "\nOn your last call, prefer a reusable checked policy or the longest justified plan.")
            calls += 1
            try:
                result = ctx.think(prompt, session_id=None)
            except RuntimeError as exc:
                ctx.log(event="planner_unavailable", error=str(exc))
                break
            reply = parse_reply(getattr(result, "text", str(result)), names)
            proposed = reply.get("schema")
            if isinstance(proposed, dict) and proposed:
                clean = {k: v for k, v in proposed.items()
                         if isinstance(k, str) and isinstance(v, dict)
                         and isinstance(v.get("path"), list)
                         and all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in v["path"])
                         and v.get("role", "state") in ("state", "clock", "boundary")}
                if clean and clean != schema:
                    schema = dict(list(clean.items())[:32])
                    visits.clear()
                    visits[signature(features(obs, schema))] = 1
            if isinstance(reply.get("hypotheses"), list):
                hypotheses = [dict(x) for x in reply["hypotheses"][:24] if isinstance(x, dict)]
                for h in hypotheses:
                    actions = h.get("actions", [])
                    h["actions"] = [a for a in actions if isinstance(a, str) and a in names] if isinstance(actions, list) else []
            if isinstance(reply.get("strategy"), str):
                strategy = reply["strategy"][:2400]
            policy, policy_memory = None, {}
            across = reply.get("policy_across_boundaries") is True
            if "policy" in reply:
                try:
                    policy = Policy(reply["policy"])
                    last_source = policy.source
                    policy_installs += 1
                    ctx.log(event="policy_installed", call=calls, source=last_source,
                            across_boundaries=across)
                except Exception as exc:
                    policy_errors += 1
                    feedback = "Policy validation error; no action spent: " + type(exc).__name__ + ": " + str(exc)
                    last_source = str(reply["policy"])[:24000]
                    ctx.log(event="policy_error", call=calls, error=feedback)
                    continue
            plan = [] if policy else reply.get("plan", reply.get("actions", []))
            if not policy and (not isinstance(plan, list) or not plan):
                feedback = "Invalid/empty plan. Return protocol JSON with legal actions; no action was spent."
                ctx.log(event="plan_parse_error", call=calls)
                continue
            tree_error = validate_tree(plan, schema, names)
            if tree_error:
                feedback = "Invalid contingent plan; no action spent: " + tree_error
                ctx.log(event="plan_validation_error", call=calls, error=tree_error)
                continue
            feedback = "Batch finished normally."
            unchecked = 0
            executed = 0
            limit = 160 if policy else 24
            ctx.log(event="plan", call=calls, schema=schema, hypotheses=hypotheses, plan=plan[:24])
            pending = list(plan)
            while (pending or policy) and executed < limit:
                if ctx.done or steps >= 1000:
                    break
                if policy:
                    try:
                        current_values = features(obs, schema)
                        if not schema or any(v == MISSING for v in current_values.values()):
                            raise PolicyError("schema is empty or a selected path is missing")
                        raw, new_memory = policy.run(obs, current_values, policy_memory,
                                                     history[-1] if history else {})
                        if isinstance(raw, dict) and "replan" in raw:
                            feedback = "Policy requested reasoning: " + str(raw["replan"])[:2400]
                            ctx.log(event="policy_yield", call=calls, reason=feedback)
                            break
                        error = validate_tree([raw], schema, names)
                        if error:
                            raise PolicyError(error)
                        if not isinstance(raw, dict):
                            raise PolicyError("policy must return a checked step object")
                        if not any(schema[k].get("role", "state") == "state" for k in raw.get("before", {})):
                            raise PolicyError("policy before must check at least one state feature")
                        if any(case["then"] for case in raw.get("cases", [])):
                            raise PolicyError("policy cases use empty then; the policy itself is the continuation")
                        if "boundary_ok" in raw and not isinstance(raw["boundary_ok"], bool):
                            raise PolicyError("boundary_ok must be boolean")
                        policy_memory = new_memory
                    except Exception as exc:
                        policy_errors += 1
                        feedback = "Policy error; no action spent for this invocation: " + type(exc).__name__ + ": " + str(exc)
                        ctx.log(event="policy_error", call=calls, error=feedback)
                        break
                else:
                    raw = pending.pop(0)
                item = {"action": raw} if isinstance(raw, str) else raw
                if not isinstance(item, dict):
                    feedback = "Invalid plan step; suffix cancelled."
                    break
                action_value = item.get("action", item)
                if isinstance(action_value, str) and "args" in item:
                    action_value = {"name": action_value, "args": item["args"]}
                action = action_text(action_value, names)
                if action is None:
                    feedback = "Invalid action; suffix cancelled. Use the supplied vocabulary."
                    break
                before = features(obs, schema)
                guard, expect = item.get("before", {}), item.get("expect", {})
                if not isinstance(guard, dict) or not isinstance(expect, dict):
                    feedback = "before/expect must be objects mapping feature names to values."
                    break
                if any(k not in schema for k in set(guard) | set(expect)):
                    feedback = "Unknown feature in before/expect; define it in schema. No action spent."
                    break
                bad_guard = {k: {"expected": v, "actual": before[k]}
                             for k, v in guard.items() if before[k] != v}
                if bad_guard:
                    feedback = "Precondition failed; suffix cancelled: " + dump(bad_guard)
                    interrupts += 1
                    ctx.log(event="guard_failed", action=action, mismatch=bad_guard)
                    break
                probe = item.get("probe") is True
                cases = item.get("cases")
                unchecked = unchecked + 1 if not expect and cases is None else 0
                if unchecked > 4:
                    feedback = "Four unchecked probes completed; interpret evidence before more probes."
                    break
                old = copy.deepcopy(obs)
                nxt = ctx.act(action)
                obs = copy.deepcopy(game.observe() if nxt is None else nxt)
                steps += 1
                executed += 1
                if policy:
                    policy_steps += 1
                after = features(obs, schema)
                mismatch = {k: {"expected": v, "actual": after[k]}
                            for k, v in expect.items() if after[k] != v}
                boundary = [k for k, spec in schema.items()
                            if spec.get("role") == "boundary" and before[k] != after[k]]
                # A replacement observation cannot verify the old scene's final state.
                # Explicit predictions about boundary values remain falsifiable.
                scoped = {k: v for k, v in mismatch.items()
                          if boundary and schema[k].get("role") != "boundary"}
                mismatch = {k: v for k, v in mismatch.items() if k not in scoped}
                surprise = [k for k, spec in schema.items()
                            if spec.get("role", "state") == "state" and k not in expect
                            and before[k] != after[k]]
                selected_case = None
                if cases is not None:
                    matches_found = matching_cases(cases, before, after, schema)
                    if not boundary and len(matches_found) == 1:
                        selected_case = matches_found[0]
                record = {"id": steps, "action": action, "before": before, "after": after,
                          "delta": differences(old, obs), "probe": probe}
                if expect:
                    record["prediction"] = expect
                if cases is not None:
                    record["case_result"] = {
                        "selected": selected_case, "matching_indices": matches_found,
                        "outcomes": [c["when"] for c in cases],
                        "reason": "boundary" if boundary else (
                            "matched" if selected_case is not None else "unmatched_or_ambiguous"),
                    }
                if policy:
                    record["policy"] = True
                if scoped:
                    record["scope_ended"] = scoped
                if mismatch:
                    record["failure"] = mismatch
                    failed_checks += 1
                    uses = item.get("uses", [])
                    for h in hypotheses:
                        if (isinstance(uses, list) and h.get("id") in uses) or action.split()[0] in h.get("actions", []):
                            h["status"] = "challenged: prediction failed at transition " + str(steps)
                if boundary:
                    scenes.append({"at_step": steps, "features": after, "observation": copy.deepcopy(obs)})
                    scenes = scenes[-4:]
                    record["boundary"] = boundary
                history.append(record)
                key = signature(after)
                visits[key] += 1
                ctx.log(event="transition", **record)
                if ctx.done:
                    break
                if boundary and not mismatch and policy and across and item.get("boundary_ok") is True:
                    policy_memory = {}
                    visits.clear()
                    visits[key] = 1
                    unchecked = 0
                    boundary_resumes += 1
                    ctx.log(event="policy_boundary_resume", transition=steps, boundary=boundary)
                    continue
                if cases is not None and not boundary:
                    if selected_case is None:
                        branch_misses += 1
                        interrupts += 1
                        feedback = "Outcome unlisted, ambiguous, or changed an unlisted state feature; replan. " + dump(record["case_result"])
                        ctx.log(event="branch_fallback", transition=steps, **record["case_result"])
                        break
                    branch_hits += 1
                    pending = list(cases[selected_case]["then"])
                    ctx.log(event="branch_selected", transition=steps, case=selected_case,
                            continuation_length=len(pending))
                    if visits[key] >= 3:
                        interrupts += 1
                        feedback = "Case matched but repeated state reached three times; replan."
                        break
                    if not pending and not policy:
                        feedback = "Matched outcome explicitly requested replanning (empty then)."
                        break
                    unchecked = 0
                    continue
                if boundary or mismatch or (surprise and not probe):
                    interrupts += 1
                    feedback = "Suffix cancelled. " + dump({"boundary": boundary, "prediction_failure": mismatch,
                                                            "unpredicted_features": surprise, "transition": steps})
                    break
                if visits[key] >= 3 and (policy or not probe):
                    interrupts += 1
                    feedback = "Repeated state reached three times; suffix cancelled. Choose a new test or route."
                    break
                if not expect and not probe:
                    feedback = "One unchecked action executed. Predict features to enable batching, or explicitly request probes."
                    break
            if executed >= limit and (pending or policy) and not ctx.done:
                feedback = "Execution horizon reached; replan from the current observation."

            ctx.log(event="batch_end", call=calls, executed=executed, feedback=feedback)
        # No action identities, state schemas, maps, or inferred rules cross seed boundaries.
        ctx.memory[memory_key] = {
            "episodes": prior_stats["episodes"] + 1,
            "steps": prior_stats["steps"] + steps,
            "calls": prior_stats["calls"] + calls,
            "interrupts": prior_stats["interrupts"] + interrupts,
            "branch_hits": previous.get("branch_hits", 0) + branch_hits,
            "branch_misses": previous.get("branch_misses", 0) + branch_misses,
            "policy_steps": previous.get("policy_steps", 0) + policy_steps,
            "policy_errors": previous.get("policy_errors", 0) + policy_errors,
            "boundary_resumes": previous.get("boundary_resumes", 0) + boundary_resumes,
            "method": "Probe, synthesize a checked feedback policy, replan on counterexamples.",
        }
        ctx.log(event="episode_controller_end", calls=calls, steps=steps,
                prediction_failures=failed_checks, interrupts=interrupts,
                branch_hits=branch_hits, branch_misses=branch_misses,
                policy_installs=policy_installs, policy_steps=policy_steps,
                policy_errors=policy_errors, boundary_resumes=boundary_resumes)


def build(ctx):
    return Harness(ctx)
