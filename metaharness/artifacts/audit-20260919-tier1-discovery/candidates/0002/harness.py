"""Episode-local, model-compiled policies with checked execution and bounded interpretation.

The host contains no game rules, observation keys, spatial planner, or action meanings.
"""
from __future__ import annotations

import ast
import copy
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
        if isinstance(obj, dict) and ("plan" in obj or "actions" in obj or "policy" in obj):
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


class Interpreter:
    """Small Python-shaped data language, not exec/eval or access to host objects.

    Every AST visit consumes fuel. All loops, calls and collection growth are bounded.
    Runtime values are data only; calls dispatch through explicit whitelists.
    """
    LIMIT = 4096
    BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Div: operator.truediv}
    CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
           ast.Is: operator.is_, ast.IsNot: operator.is_not}
    BUILTINS = {"len": len, "abs": abs, "min": min, "max": max, "sum": sum,
                "int": int, "str": str, "bool": bool, "list": list, "tuple": tuple,
                "dict": dict, "set": set, "sorted": sorted}
    METHODS = {dict: {"get", "keys", "values", "items", "copy", "pop"},
               list: {"append", "extend", "pop", "count", "index", "copy"},
               set: {"add", "discard", "copy"}}
    ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign,
               ast.AugAssign, ast.If, ast.For, ast.While, ast.Break, ast.Continue,
               ast.Return, ast.Expr, ast.Pass, ast.Name, ast.Constant, ast.List,
               ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.Call,
               ast.Attribute, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
               ast.IfExp, ast.Load, ast.Store, ast.And, ast.Or, ast.Not, ast.USub,
               ast.UAdd) + tuple(BIN) + tuple(CMP)

    def __init__(self, source):
        if not isinstance(source, str) or len(source) > 24000:
            raise PolicyError("policy must be source text under 24000 characters")
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
        if len(nodes) > 6000 or any(not isinstance(n, self.ALLOWED) for n in nodes):
            raise PolicyError("unsupported syntax or policy too large")
        if any(isinstance(n, ast.Name) and n.id.startswith("_") for n in nodes):
            raise PolicyError("private names are unavailable")
        self.functions = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.decorator_list:
                raise PolicyError("top level must contain only undecorated function definitions")
            args = node.args
            if (args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs
                    or args.defaults or node.returns or any(a.annotation for a in args.args)):
                raise PolicyError("use plain positional function parameters without defaults")
            if node.name in self.functions or node.name.startswith("_"):
                raise PolicyError("duplicate/private function name")
            self.functions[node.name] = node
        if "policy" not in self.functions or len(self.functions["policy"].args.args) != 2:
            raise PolicyError("define policy(obs, memory)")
        self.fuel, self.depth = 0, 0

    def tick(self):
        self.fuel -= 1
        if self.fuel < 0:
            raise PolicyError("policy exhausted its 60000-node computation allowance")

    def bounded(self, value):
        if isinstance(value, (list, tuple, dict, set, str)) and len(value) > self.LIMIT:
            raise PolicyError("collection or string exceeds 4096 items")
        if type(value) in (int, float) and not -1e12 <= value <= 1e12:
            raise PolicyError("numeric value out of range")
        return value

    def binary(self, op, left, right):
        if isinstance(op, ast.Mult):
            for seq, count in ((left, right), (right, left)):
                if isinstance(seq, (list, tuple, str)) and isinstance(count, int):
                    if len(seq) * max(0, count) > self.LIMIT:
                        raise PolicyError("repetition exceeds collection limit")
        if isinstance(op, ast.Mod) and isinstance(left, str):
            raise PolicyError("string formatting is unavailable")
        return self.bounded(self.BIN[type(op)](left, right))

    def expr(self, n, env):
        self.tick()
        if isinstance(n, ast.Constant):
            if n.value is not None and type(n.value) not in (str, int, float, bool):
                raise PolicyError("only JSON-like constants are supported")
            return self.bounded(n.value)
        if isinstance(n, ast.Name):
            return env[n.id]
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            value = [self.expr(x, env) for x in n.elts]
            return self.bounded(tuple(value) if isinstance(n, ast.Tuple) else
                                set(value) if isinstance(n, ast.Set) else value)
        if isinstance(n, ast.Dict):
            if any(k is None for k in n.keys):
                raise PolicyError("dict unpacking is unavailable")
            return self.bounded({self.expr(k, env): self.expr(v, env)
                                 for k, v in zip(n.keys, n.values)})
        if isinstance(n, ast.Slice):
            return slice(*(self.expr(x, env) if x is not None else None
                           for x in (n.lower, n.upper, n.step)))
        if isinstance(n, ast.Subscript):
            return self.expr(n.value, env)[self.expr(n.slice, env)]
        if isinstance(n, ast.BinOp):
            return self.binary(n.op, self.expr(n.left, env), self.expr(n.right, env))
        if isinstance(n, ast.UnaryOp):
            value = self.expr(n.operand, env)
            return self.bounded(not value if isinstance(n.op, ast.Not) else
                                -value if isinstance(n.op, ast.USub) else +value)
        if isinstance(n, ast.BoolOp):
            for item in n.values:
                value = self.expr(item, env)
                if isinstance(n.op, ast.And) and not value:
                    break
                if isinstance(n.op, ast.Or) and value:
                    break
            return value
        if isinstance(n, ast.Compare):
            left = self.expr(n.left, env)
            for op, item in zip(n.ops, n.comparators):
                right = self.expr(item, env)
                if not self.CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return self.expr(n.body if self.expr(n.test, env) else n.orelse, env)
        if isinstance(n, ast.Call):
            args = [self.expr(x, env) for x in n.args]
            if n.keywords:
                raise PolicyError("keyword arguments are unavailable")
            if isinstance(n.func, ast.Name):
                name = n.func.id
                if name in self.functions:
                    return self.call(name, args)
                if name == "range":
                    value = range(*args)
                    if len(value) > self.LIMIT:
                        raise PolicyError("range exceeds 4096 items")
                    return list(value)
                if name in ("enumerate", "zip", "reversed"):
                    fn = {"enumerate": enumerate, "zip": zip, "reversed": reversed}[name]
                    return self.bounded(list(fn(*args)))
                if name in self.BUILTINS:
                    if name == "str" and args and isinstance(args[0], (list, tuple, dict, set)):
                        raise PolicyError("str accepts scalar data only")
                    return self.bounded(self.BUILTINS[name](*args))
            if isinstance(n.func, ast.Attribute):
                target = self.expr(n.func.value, env)
                method = n.func.attr
                if method not in self.METHODS.get(type(target), set()):
                    raise PolicyError("method unavailable: " + method)
                if method in ("append", "add") and len(target) >= self.LIMIT:
                    raise PolicyError("collection full")
                if method == "extend" and len(target) + len(args[0]) > self.LIMIT:
                    raise PolicyError("collection full")
                result = getattr(target, method)(*args)
                if method in ("keys", "values", "items"):
                    result = list(result)
                return self.bounded(result)
            raise PolicyError("call unavailable")
        raise PolicyError("expression unavailable: " + type(n).__name__)

    def assign(self, target, value, env):
        self.tick()
        self.bounded(value)
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            if len(target.elts) != len(value):
                raise PolicyError("unpacking length mismatch")
            for part, item in zip(target.elts, value):
                self.assign(part, item, env)
        elif isinstance(target, ast.Subscript):
            container = self.expr(target.value, env)
            index = self.expr(target.slice, env)
            if isinstance(index, slice) or type(container) not in (dict, list):
                raise PolicyError("only scalar list/dict assignments are supported")
            if isinstance(container, dict) and index not in container and len(container) >= self.LIMIT:
                raise PolicyError("collection full")
            container[index] = value
        else:
            raise PolicyError("assignment target unavailable")

    def block(self, statements, env):
        for n in statements:
            self.tick()
            if isinstance(n, ast.Assign):
                value = self.expr(n.value, env)
                for target in n.targets:
                    self.assign(target, value, env)
            elif isinstance(n, ast.AugAssign):
                self.assign(n.target, self.binary(n.op, self.expr(n.target, env),
                                                  self.expr(n.value, env)), env)
            elif isinstance(n, ast.Expr):
                self.expr(n.value, env)
            elif isinstance(n, ast.If):
                self.block(n.body if self.expr(n.test, env) else n.orelse, env)
            elif isinstance(n, (ast.For, ast.While)):
                iterator = iter(self.expr(n.iter, env)) if isinstance(n, ast.For) else None
                broke = False
                while True:
                    self.tick()
                    if iterator is not None:
                        try:
                            value = next(iterator)
                        except StopIteration:
                            break
                        self.assign(n.target, value, env)
                    elif not self.expr(n.test, env):
                        break
                    try:
                        self.block(n.body, env)
                    except Flow as flow:
                        if flow.kind == "break":
                            broke = True
                            break
                        if flow.kind != "continue":
                            raise
                if not broke:
                    self.block(n.orelse, env)
            elif isinstance(n, ast.Return):
                raise Flow("return", self.expr(n.value, env) if n.value else None)
            elif isinstance(n, (ast.Break, ast.Continue)):
                raise Flow("break" if isinstance(n, ast.Break) else "continue")
            elif not isinstance(n, ast.Pass):
                raise PolicyError("statement unavailable: " + type(n).__name__)

    def call(self, name, args):
        self.tick()
        node = self.functions[name]
        if len(args) != len(node.args.args) or self.depth >= 24:
            raise PolicyError("wrong argument count or recursion depth exceeded")
        self.depth += 1
        try:
            self.block(node.body, dict(zip((a.arg for a in node.args.args), args)))
        except Flow as flow:
            if flow.kind != "return":
                raise PolicyError("break/continue outside loop")
            return flow.value
        finally:
            self.depth -= 1
        return None

    def run(self, obs, memory):
        self.fuel, self.depth = 60000, 0
        # The language cannot see ctx/game/files/network or persistent module globals.
        output = self.call("policy", [copy.deepcopy(obs), memory])
        # Count expanded data BEFORE JSON encoding: aliased/cyclic structures must
        # not turn a tiny program into an unbounded serialization or scratch cache.
        pending = [(output, 0), (memory, 0)]
        seen_nodes = 0
        while pending:
            value, depth = pending.pop()
            seen_nodes += 1
            if seen_nodes > 16000 or depth > 48:
                raise PolicyError("output/scratch data too large or cyclic")
            self.bounded(value)
            if isinstance(value, dict):
                for k, v in value.items():
                    pending.extend(((k, depth + 1), (v, depth + 1)))
            elif isinstance(value, (list, tuple, set)):
                pending.extend((v, depth + 1) for v in value)
            elif value is not None and type(value) not in (str, int, float, bool):
                raise PolicyError("unsupported output/scratch value")
        encoded = json.dumps(output, allow_nan=False)
        if len(encoded) > 64000:
            raise PolicyError("policy output too large")
        return json.loads(encoded)


PROTOCOL = """Control an unfamiliar turn-based game using only its public description and
observations. Maximize objectives, then minimize reasoning calls. You are both experimental
scientist and policy compiler. No rules, field names, controls or spatial solver are built in.
Return ONE JSON object with schema, hypotheses, strategy, and either plan or policy.

SCHEMA: {"feature":{"path":["observed_key"],"role":"state"}}. Optional find:VALUE
returns all relative paths of exact matches (a matrix token gives [[row,column]]).
Optional count:true counts matches. role:boundary identifies scene/round changes;
role:clock excludes counters from state repetition checks; otherwise role:state.
Choose a sufficient small state: controlled entity, inventory/mechanisms and important flags.
The complete observation and raw changes remain available. Schema replaces the previous schema.
Do not guess meanings from raw numbers. Observe missing paths as errors, not matching values.

HYPOTHESES: [{"id":"h1","claim":"conditional rule","actions":["legal name"],
"evidence":[1]}]. Cite numbered real transitions; claims remain provisional. A failed prediction
challenges rules used by that action. A blocked action does not establish universal no-effect.

PLAN: [{"action":"legal name and optional arguments","before":{},"expect":{},
"uses":["h1"],"probe":true}]. before/expect map schema features to exact values.
Known steps need expect including a state feature. State features omitted from expect must stay
unchanged. A failed guard, prediction, missing feature or unexpected state change cancels the
suffix and invalidates the compiled policy. Boundaries cancel only the current suffix.
For discovery, use 2-4 low-risk probe:true steps; each result is observed and recorded. No more
than four unchecked probes per model call, including all compiled policy invocations. No random
default actions. Avoid repeated state/action pairs without a discriminating hypothesis.
Optional cases:[{"when":{"feature":"value"},"then":[...]}] on the LAST step of a list
supplies alternative exact outcomes; all changed state features must be listed in when.
Exactly one case must match. No wildcard. Use cases OR expect. Up to four cases, four nesting
layers, 64 nodes, 24 actions per batch. A bare unchecked action runs once then asks you again.

POLICY COMPILATION (preferred once useful rules are observed): supply "policy": a Python source
STRING defining policy(obs, memory), plus optional top-level helper functions. It returns the
same {"plan":[...]} object, or {"plan":[],"reason":"need experiment/reasoning because ..."}.
Its schema is the enclosing JSON schema; the policy cannot redefine schema or hypotheses.
The host calls this program on fresh complete observations after each completed batch, matched
empty branch, or scene boundary, without spending another reasoning call. memory is an empty
episode-local scratch dict, cleared at scene changes; hardcoded answers for future layouts are
wrong. Your SOURCE can encode action meanings learned IN THIS episode. Your algorithm must read
the actual current layout/state and compute decisions; the host supplies no solver. You may
implement bounded search, rules or arithmetic as warranted by the observed game. Never assume
rules from another game. Unknown situations should return an empty plan and explanation.
Prefer a compact policy with one checked action per invocation; it can replan cheaply each step.
For untested interactions, enumerate plausible checked cases, then recompute from the actual
outcome. If known actions already yield a route to the objective, do not investigate irrelevant
mechanisms. When a scene changes reuse episode rules but derive a new plan from its observation.
Do not stop merely because a previously anticipated branch occurred or a new scene appeared.
Schema boundary changes bypass predictions, then invoke the policy on the new scene.

SOURCE LANGUAGE: ordinary positional def helper(...), assignments (including list/dict index
and tuple unpacking), +=, if/elif/else, for, while, break, continue, return, pass. Expressions:
JSON-like constants, list/tuple/dict/set, indexing/slices, + - * / // %, comparisons, in/not in,
and/or/not, conditional expressions. No comprehensions/generators, imports, classes, lambdas,
exceptions, attributes except the methods below, keyword/default/annotated arguments or global
variables. Builtins: len, range, enumerate, zip, reversed, abs, min, max, sum, int, str, bool,
list, tuple, dict, set, sorted. Methods: dict.get/keys/values/items/copy/pop;
list.append/extend/pop/count/index/copy; set.add/discard/copy. Helpers can call one another.
No exec/eval or host access. Maximum 60000 evaluated AST nodes per invocation, 4096 items per
container/string, depth 24, source 24000 characters. Use a list plus a head index for queues.
Return JSON-compatible data; tuple/set scratch keys are allowed. Simple compact code is best.

First learn enough controls with short experiments; then compile immediately if possible.
An explicit checked plan remains available when compilation would be premature or difficult.
Later calls repair the policy from counterexamples. Calls are scarce; prefer a complete reusable
algorithm to repeated hand-written routes, but never continue after a refuted prediction.
"""


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def remaining(self, calls):
        # The task's hard cap also applies if the runtime has no budget accessor.
        left = max(0, 12 - calls)
        budget = self.ctx.budget
        if callable(getattr(budget, "remaining", None)):
            value = budget.remaining()
            if isinstance(value, dict) and "system2_calls" in value:
                left = min(left, max(0, int(value["system2_calls"])))
        elif hasattr(budget, "max_system2_calls") and hasattr(budget, "system2_calls"):
            left = min(left, max(0, budget.max_system2_calls - budget.system2_calls))
        return left

    def run_episode(self, game):
        ctx = self.ctx
        names = list(game.action_spec.names)
        if not names:
            return
        description, spec = str(game.describe()), str(game.action_spec.describe())
        obs = copy.deepcopy(game.observe())
        schema, hypotheses, history = {}, [], []
        scenes = [{"at_step": 0, "observation": copy.deepcopy(obs)}]
        visits, pairs = Counter(), Counter()
        policy, source, scratch = None, "", {}
        strategy, feedback = "Learn useful controls, then compile.", "Cold start."
        calls = steps = invocations = compiled_steps = refutations = unchecked = 0
        # Always overwrite: neither a cached policy nor seed-specific facts are loaded.
        record = {"calls": 0, "steps": 0, "policy_invocations": 0, "compiled_steps": 0}
        ctx.memory["compiled_policy_v3"] = record

        def signature(values):
            selected = {k: v for k, v in values.items() if schema[k].get("role") != "clock"}
            return dump(selected) if selected else dump(obs)

        def evidence():
            chosen = set(range(max(0, len(history) - 12), len(history)))
            for name in names:
                found = [i for i, e in enumerate(history) if e["action"].split()[0] == name]
                if found:
                    chosen.update((found[0], found[-1]))
            chosen.update(i for i, e in enumerate(history)
                          if e.get("failure") or e.get("case_result") or e.get("boundary"))
            # These episodes have at most 100 actions; retain all selected counterexamples.
            return [history[i] for i in sorted(chosen)]

        while not ctx.done and steps < 1000:
            from_policy = policy is not None
            if from_policy:
                invocations += 1
                try:
                    reply = policy.run(obs, scratch)
                    if not isinstance(reply, dict):
                        raise PolicyError("policy must return an object")
                except Exception as exc:
                    # This block interprets DATA only. No ctx/model call is caught here.
                    feedback = "Compiled policy failed: " + type(exc).__name__ + ": " + str(exc)
                    ctx.log(event="policy_error", call=calls, error=feedback)
                    policy, scratch = None, {}
                    continue
                if not reply.get("plan", reply.get("actions")):
                    feedback = "Policy yielded: " + str(reply.get("reason", "empty plan"))[:2400]
                    ctx.log(event="policy_yield", call=calls, reason=feedback)
                    policy = None
                    continue
            else:
                remaining = self.remaining(calls)
                if remaining <= 0:
                    break
                current = features(obs, schema)
                prompt = (PROTOCOL + "\nGAME DESCRIPTION:\n" + description +
                          "\nACTION SPECIFICATION:\n" + spec +
                          "\nCALLS LEFT INCLUDING THIS ONE: " + str(remaining) +
                          "\nCURRENT OBSERVATION (zero-based array coordinates):\n" + render(obs) +
                          "\nSCHEMA: " + dump(schema) + "\nFEATURES: " + dump(current) +
                          "\nSTRATEGY: " + strategy + "\nHYPOTHESES: " + dump(hypotheses) +
                          "\nTRANSITION EVIDENCE: " + dump(evidence()) +
                          "\nRECENT SCENE STARTS: " + dump(scenes[-2:]) +
                          "\nCURRENT STATE VISITS: " + str(visits[signature(current)]) +
                          "\nEXECUTOR FEEDBACK: " + feedback +
                          "\nPREVIOUS POLICY SOURCE (revise if supplied):\n" + source)
                calls += 1
                # Inference exceptions, including ModelUnavailable, MUST propagate.
                result = ctx.think(prompt, session_id=None)
                reply = parse_reply(getattr(result, "text", str(result)), names)
                proposed = reply.get("schema")
                if proposed is not None:
                    valid = isinstance(proposed, dict) and 0 < len(proposed) <= 32
                    if valid:
                        valid = all(isinstance(k, str) and isinstance(v, dict)
                                    and isinstance(v.get("path"), list)
                                    and all(type(x) in (str, int) for x in v["path"])
                                    and v.get("role", "state") in ("state", "clock", "boundary")
                                    for k, v in proposed.items())
                    if not valid or any(v == MISSING for v in features(obs, proposed).values()):
                        feedback = "Invalid schema or missing feature path; no action spent."
                        ctx.log(event="plan_validation_error", call=calls, error=feedback)
                        continue
                    if proposed != schema:
                        schema = proposed
                        visits.clear()
                        pairs.clear()
                        visits[signature(features(obs, schema))] = 1
                if isinstance(reply.get("hypotheses"), list):
                    hypotheses = [dict(x) for x in reply["hypotheses"][:24] if isinstance(x, dict)]
                if isinstance(reply.get("strategy"), str):
                    strategy = reply["strategy"][:2400]
                unchecked = 0
                if "policy" in reply:
                    candidate = reply["policy"]
                    try:
                        if not schema or not any(v.get("role", "state") == "state"
                                                 for v in schema.values()):
                            raise PolicyError("compiled policy requires a state schema")
                        policy = Interpreter(candidate)
                    except (PolicyError, SyntaxError, ValueError, TypeError, RecursionError) as exc:
                        feedback = "Policy rejected before execution: " + str(exc)
                        if isinstance(candidate, str):
                            source = candidate[:24000]
                        ctx.log(event="policy_compile_error", call=calls, error=feedback)
                        continue
                    source, scratch = candidate, {}
                    ctx.log(event="policy_installed", call=calls, source=source)
                    continue

            plan = reply.get("plan", reply.get("actions", []))
            error = validate_tree(plan, schema, names)
            if not plan or error:
                feedback = "Invalid/empty plan; no action spent: " + str(error)
                ctx.log(event="plan_validation_error", call=calls, error=feedback)
                policy = None
                continue
            pending, executed = list(plan), 0
            feedback = "Batch finished normally."
            ctx.log(event="plan", call=calls, origin="policy" if from_policy else "model",
                    schema=schema, plan=plan)
            while pending and executed < 24 and not ctx.done and steps < 1000:
                raw = pending.pop(0)
                item = {"action": raw} if isinstance(raw, str) else raw
                value = item.get("action", item)
                if isinstance(value, str) and "args" in item:
                    value = {"name": value, "args": item["args"]}
                action = action_text(value, names)
                before = features(obs, schema)
                guard, expect = item.get("before", {}), item.get("expect", {})
                cases, probe = item.get("cases"), item.get("probe") is True
                bad_guard = {k: {"expected": v, "actual": before[k]}
                             for k, v in guard.items() if before[k] != v}
                pair = (signature(before), action)
                if bad_guard or any(v == MISSING for v in before.values()):
                    feedback = "Precondition/missing feature failed: " + dump(bad_guard)
                    ctx.log(event="guard_failed", action=action, mismatch=bad_guard)
                    policy = None
                    break
                if pairs[pair] >= 2:
                    feedback = "State/action pair already tried twice; choose a new test or revise schema."
                    ctx.log(event="repetition_stop", action=action, signature=pair[0])
                    policy = None
                    break
                checked = cases is not None or any(schema[k].get("role", "state") == "state"
                                                   for k in expect)
                if not checked and unchecked >= 4:
                    feedback = "Four unchecked actions since last reasoning call; interpret evidence."
                    policy = None
                    break
                if not checked:
                    unchecked += 1
                old = copy.deepcopy(obs)
                # This is the point at which model-generated output changes actual control.
                ctx.log(event="control_action", action=action, call=calls,
                        origin="compiled_policy" if from_policy else "model_plan")
                nxt = ctx.act(action)
                obs = copy.deepcopy(game.observe() if nxt is None else nxt)
                steps += 1
                executed += 1
                compiled_steps += int(from_policy)
                pairs[pair] += 1
                after = features(obs, schema)
                boundary = [k for k, s in schema.items() if s.get("role") == "boundary"
                            and before[k] != after[k]]
                mismatch = {k: {"expected": v, "actual": after[k]}
                            for k, v in expect.items() if after[k] != v}
                surprise = [k for k, s in schema.items() if s.get("role", "state") == "state"
                            and k not in expect and before[k] != after[k]]
                entry = {"id": steps, "action": action, "before": before, "after": after,
                         "delta": differences(old, obs), "probe": probe,
                         "origin": "policy" if from_policy else "model"}
                if expect:
                    entry["prediction"] = expect
                selected = None
                if cases is not None:
                    found = matching_cases(cases, before, after, schema)
                    if len(found) == 1:
                        selected = found[0]
                    entry["case_result"] = {"selected": selected, "matching_indices": found,
                                            "outcomes": [c["when"] for c in cases]}
                missing = any(v == MISSING for v in after.values())
                failure = (missing or bool(mismatch) or (cases is not None and selected is None)
                           or (bool(surprise) and not probe and cases is None))
                # New scenes are new prediction domains, not counterexamples to old-scene rules.
                if boundary:
                    entry["boundary"] = boundary
                    scenes.append({"at_step": steps, "observation": copy.deepcopy(obs)})
                    scenes = scenes[-4:]
                    visits.clear()
                    pairs.clear()
                    scratch = {}
                elif failure and not ctx.done:
                    entry["failure"] = {"mismatch": mismatch, "unpredicted": surprise,
                                        "missing_feature": missing, "unmatched_case":
                                        cases is not None and selected is None}
                    refutations += 1
                    uses = item.get("uses", [])
                    for hypothesis in hypotheses:
                        actions = hypothesis.get("actions", [])
                        if ((isinstance(uses, list) and hypothesis.get("id") in uses)
                                or (isinstance(actions, list) and action.split()[0] in actions)):
                            hypothesis["status"] = "challenged at transition " + str(steps)
                    policy = None
                history.append(entry)
                visits[signature(after)] += 1
                ctx.log(event="transition", **entry)
                record.update(calls=calls, steps=steps, policy_invocations=invocations,
                              compiled_steps=compiled_steps, refutations=refutations)
                if ctx.done:
                    break
                if boundary:
                    feedback = "Scene changed; discard suffix and recompute from fresh observation."
                    ctx.log(event="scene_replan", transition=steps, using_policy=policy is not None)
                    break
                if failure:
                    feedback = "Policy/plan invalidated by transition " + str(steps) + ": " + dump(entry["failure"])
                    ctx.log(event="policy_invalidated", transition=steps, reason=entry["failure"])
                    break
                if cases is not None:
                    pending = list(cases[selected]["then"])
                    ctx.log(event="branch_selected", transition=steps, case=selected,
                            continuation_length=len(pending), recompute_policy=policy is not None)
                if visits[signature(after)] >= 3 and not probe:
                    feedback = "Repeated state three times; revise route or state abstraction."
                    policy = None
                    break
                if not checked and not probe:
                    feedback = "Unchecked action completed; provide state predictions to continue."
                    policy = None
                    break
            ctx.log(event="batch_end", call=calls, executed=executed, feedback=feedback)
        # No spawn is used: no futures can escape this synchronous tier-1 episode.
        record.update(calls=calls, steps=steps, policy_invocations=invocations,
                      compiled_steps=compiled_steps, refutations=refutations)
        ctx.log(event="episode_controller_end", **record)


def build(ctx):
    return Harness(ctx)
