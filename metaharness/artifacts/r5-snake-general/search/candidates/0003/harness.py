"""Staged runtime compilation: acquire a small controller, then improve it.

No game observation key or action name is embedded in this module. Geometry is
an optional, model-bound bootstrap; the durable policy is written from the
public interface by runtime System 2, separately in every episode.
"""
from __future__ import annotations

import ast
import copy
import heapq
import json
import math
import re
import sys
import time
from collections import deque

from metaharness.models.jev import Choice, usage_guide


UNKNOWN = "__unknown__"


def packed(value):
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def unavailable(exc):
    # The supplied interface does not prescribe the exception's import path.
    return any(c.__name__ == "ModelUnavailable" for c in type(exc).__mro__)


def accepted(answer, probability=0.65):
    if answer is None:
        return None
    value = getattr(answer, "value", None)
    probs = getattr(answer, "probabilities", {}) or {}
    confidence = getattr(answer, "confidence", None)
    if (isinstance(value, str) and confidence is not None
            and confidence >= 0.2 and probs.get(value, 0) >= probability):
        return value
    return None


def point(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(x) is int for x in value))


def bootstrap_questions(obs):
    """Bind structural candidates to meanings; S1 does no coordinate arithmetic."""
    numbers = {k: str(v) for k, v in obs.items()
               if type(v) is int and 2 <= v <= 512}
    trails = {k: packed(v) for k, v in obs.items()
              if isinstance(v, list) and v and all(point(p) for p in v)}
    targets = {k: packed(v) for k, v in obs.items() if point(v)}

    def fields(values, meaning):
        options = {UNKNOWN: "Not explicitly supported by the public rules."}
        options.update({k: "Observation field %r currently equals %s" % (k, v)
                        for k, v in list(values.items())[:100]})
        return Choice("Using rules and observation, identify " + meaning
                      + ". Select unknown if these semantics are not explicit.", options)

    return {
        "extent_x": fields(numbers, "the exclusive upper bound of zero-based x coordinates"),
        "extent_y": fields(numbers, "the exclusive upper bound of zero-based y coordinates"),
        "trail": fields(trails, "an ordered occupied trail whose FIRST point is the controlled moving head"),
        "target": fields(targets, "a desirable currently collectible target coordinate"),
        "geometry": Choice(
            "Do the rules explicitly support ALL these bootstrap assumptions: a zero-based rectangular "
            "integer [x,y] grid; action descriptions give one-cell absolute x/y moves; exterior walls "
            "are fatal or forbidden; an ordered trail is head first and its retained cells block movement; "
            "moving directly into the next trail point is forbidden or ignored?",
            {UNKNOWN: "At least one assumption is unsupported or contradicted.",
             "supported": "Every listed assumption is explicitly supported."}),
    }


class Bootstrap:
    """Conservative geometry reflex, enabled only by a semantic binding.

    This is not a completion strategy. It treats the entire old trail as
    blocked. Acquisition avoids the bound target; degraded control seeks it.
    """
    def __init__(self, descriptions):
        self.binding = None
        self.vectors = {}
        for action, description in descriptions.items():
            match = re.search(r"\b([xy])\s*([+-])\s*1\b", str(description), re.I)
            if match:
                amount = 1 if match[2] == "+" else -1
                self.vectors[action] = ((amount, 0) if match[1].lower() == "x"
                                        else (0, amount))

    def bind(self, answers):
        if not isinstance(answers, dict):
            return False
        values = {k: accepted(v) for k, v in answers.items()}
        keys = ("extent_x", "extent_y", "trail")
        if (values.get("geometry") == "supported" and len(self.vectors) == 4
                and all(values.get(k) not in (None, UNKNOWN) for k in keys)):
            self.binding = values
            return True
        return False

    def startup_pattern(self, obs, rules):
        # A bounded actuator pattern buys time for the very first semantic reply.
        # Discover ordering from prose and structure, never from a field name.
        # This is a temporary heuristic, not a claimed safety certificate.
        if len(self.vectors) != 4 or not re.search(r"head\s+(?:is\s+)?first", rules, re.I):
            return []
        chains = [v for v in obs.values() if isinstance(v, list) and len(v) >= 2
                  and all(point(p) for p in v)]
        if len(chains) != 1:
            return []
        chain = chains[0]
        dx, dy = chain[0][0] - chain[1][0], chain[0][1] - chain[1][1]
        inverse = {v: a for a, v in self.vectors.items()}
        vectors = [(dy, -dx), (dx, dy), (-dy, dx), (-dx, -dy)]
        if any(v not in inverse for v in vectors):
            return []
        return [inverse[v] for v in vectors]

    def choose(self, obs, fallback, seek=True):
        if not self.binding:
            return fallback
        b = self.binding
        try:
            w, h = obs[b["extent_x"]], obs[b["extent_y"]]
            body = obs[b["trail"]]
            if (type(w) is not int or type(h) is not int or not 2 <= w <= 512
                    or not 2 <= h <= 512 or w * h > 4096 or not body
                    or not all(point(p) for p in body)):
                return fallback
            head = tuple(body[0])
            occupied = {tuple(p) for p in body}
            if any(not (0 <= x < w and 0 <= y < h) for x, y in occupied):
                return fallback
            target = obs.get(b.get("target"))
            if not point(target):
                target = head
            ranked = []
            for action, (dx, dy) in self.vectors.items():
                dest = head[0] + dx, head[1] + dy
                if dest in occupied or not (0 <= dest[0] < w and 0 <= dest[1] < h):
                    continue
                reached, queue = {dest}, deque([dest])
                while queue:
                    x, y = queue.popleft()
                    for vx, vy in self.vectors.values():
                        p = x + vx, y + vy
                        if (0 <= p[0] < w and 0 <= p[1] < h
                                and p not in occupied and p not in reached):
                            reached.add(p)
                            queue.append(p)
                distance = abs(dest[0] - target[0]) + abs(dest[1] - target[1])
                if seek:
                    rank = (len(reached), -distance, action == fallback, action)
                else:
                    # During compilation preserve simple, low-growth configurations.
                    # This is only a temporary heuristic over model-bound meanings.
                    rank = (dest != tuple(target), len(reached), action == fallback, action)
                ranked.append(rank)
            return max(ranked)[-1] if ranked else fallback
        except (KeyError, TypeError, ValueError, IndexError):
            self.binding = None
            return fallback


class PolicyLimit(BaseException):
    pass


def limited(fn, *args):
    """Bound generated Python work without placing inference on the tick thread."""
    deadline = time.monotonic() + 0.025
    count = 0
    previous = sys.gettrace()

    def trace(frame, event, arg):
        nonlocal count
        count += 1
        if count > 60000 or (count % 128 == 0 and time.monotonic() > deadline):
            raise PolicyLimit("local policy work limit")
        return trace

    sys.settrace(trace)
    try:
        return fn(*args)
    finally:
        sys.settrace(previous)


def bounded_range(*args):
    r = range(*args)
    if len(r) > 20000:
        raise ValueError("range too large")
    return r


def decision(action, facts, goal, assumptions, regime, expect):
    """Compact generated-code ABI; one strategy needs no repeated schema boilerplate."""
    return {"candidates": [{"id": "local", "action": action, "safe": True,
                            "description": str(goal), "expect": expect}],
            "default": "local", "facts": facts, "goal": goal,
            "preconditions": assumptions, "context_key": regime,
            "question": "Are the supplied facts consistent with the strategy assumptions?"}


def compile_policy(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    source = blocks[0] if blocks else text.strip()
    if len(source) > 40000:
        raise ValueError("policy source too long")
    tree = ast.parse(source)
    # Definition-only modules: no imports, hidden object access, I/O or ctx.
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError("only top-level function definitions are accepted")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global,
                             ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try,
                             ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            raise ValueError("unsupported policy construct")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attribute access")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("private name access")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.args.defaults
                                                or any(node.args.kw_defaults)):
            raise ValueError("decorators/default arguments are not accepted")
        if isinstance(node, ast.FunctionDef) and (node.returns is not None
                or any(a.annotation is not None for a in node.args.args + node.args.kwonlyargs)):
            raise ValueError("annotations are not accepted")
    safe = {f.__name__: f for f in (abs, all, any, bool, dict, enumerate, float,
            int, isinstance, len, list, max, min, next, reversed, round, set,
            sorted, str, sum, tuple, zip)}
    safe["range"] = bounded_range
    scope = {"__builtins__": safe, "deque": deque, "heapq": heapq, "math": math, "decision": decision}
    limited(exec, compile(tree, "<episode-policy>", "exec"), scope)
    fn = scope.get("policy")
    if not callable(fn):
        raise ValueError("missing policy(observation, memory)")
    return fn


CODE_CONTRACT = """
Return only Python function definitions including policy(observation, memory).
Read keys/actions/rules exclusively from PUBLIC INPUT. Run on CURRENT observation;
the request snapshot will be old. No imports, classes, annotations, defaults,
decorators, try/except, private attributes, I/O, tools or model/context access.
Available: ordinary safe Python builtins, deque, heapq, math. Memory is a mutable
per-controller dict. Work per tick <=25ms / 60000 trace events; range <=20000.
To avoid boilerplate, return decision(action, facts, goal, assumptions, regime, expect).
action must be a currently legal spec name. facts is compact computed evidence;
goal/assumptions are English; regime identifies stable strategy meaning (not tick);
expect is a dict of deterministic NEXT-observation fields, or {} when none can be
predicted. Omit stochastic fields. Code checks safety; Jev does semantic review.
Do not mutate observation. Do not assume a previous suggested action was executed.
"""

ACQUIRE_REQUEST = """
FIRST DELIVERABLE: implement a small working local controller now. Target 40-70
lines, not an elaborate optimal solver. The game runs while you think. A temporary
reflex is withholding optional rewards to preserve maneuvering room until delivery.
Implement exact immediate rules, score-seeking choices and a cheap escape check
appropriate to THIS public game. Prefer a simple executable improvement over a
lengthy analysis. Do not construct a complex global plan in this first response.
Mark heuristic assumptions honestly in facts; legal does not mean long-term safe.
A second compilation will improve the strategy after this one is running.
Include exact next-state predictions for deterministic controlled fields where
possible, so code can detect misinterpreted rules without consulting a model.
""" + CODE_CONTRACT

IMPROVE_REQUEST = """
SECOND DELIVERABLE: improve the installed local controller into a durable strategy.
It keeps executing while you think. Return a COMPLETE replacement, preferably
under 140 lines. Use the observed failure/progress evidence and available wall
budget; maximize earned score, not survival alone. Compile arithmetic, bookkeeping,
legality and bounded search into code. Use a provable progress invariant when the
PUBLIC rules support one. For covering traversals verify coverage, adjacency,
closure and compatibility with the CURRENT configuration, including growth and
moving occupancy. Only use shortcuts that preserve the invariant and progress;
otherwise retain its certified successor. Explicitly recover if initialization
fails. These are conditional planning ideas, not assumptions about this game.
Avoid wasting wall budget on a traversal too slow to collect rewards. In other
rulesets choose an appropriate different strategy from their public rules.
The first controller may already be entangled; never assume the request's initial
state still holds when your replacement is installed.

For genuinely useful semantic alternatives you may instead return a dict with:
 candidates: 1..12 {id: stable strategy ID, action: current spec name, safe: bool,
   description: English strategy/evidence, expect: deterministic next-field dict};
 default: an ID with safe=True; facts; goal; preconditions; context_key; question.
Only offer locally verified alternatives; use facts to distinguish heuristic safety
from an actual invariant. question is focused English for Jev to compare supplied
evidence, not calculate coordinates. Candidate IDs mean reusable strategies, not
fixed directions. Jev's choice is leased for <=20 request ticks only when version,
regime and current safety match. If no meaningful alternative, use decision().
""" + CODE_CONTRACT



class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        spec = game.action_spec
        names = list(spec.names)
        fallback = spec.noop if spec.noop in names else names[0]
        descriptions = {n: spec.descriptions.get(n, str(n)) for n in names}
        public = {"rules": game.describe(), "actions": descriptions,
                  "noop": spec.noop, "notes": spec.notes}
        bootstrap = Bootstrap(descriptions)
        futures = []
        s1 = s2 = None
        policy = None
        memory = {}
        version = 0
        lease = None
        pending_audit = None
        expected = None
        bootstrap_prediction = None
        last = fallback
        session = None
        source = None
        failure = "initial acquisition"
        compiled_stage = None
        request_stage = None
        action_counts = {"bootstrap_hold": 0, "bootstrap_seek": 0, "compiled": 0}
        first_action_version = None
        attempts = 0
        last_compile = -1000.0
        last_review = -1000
        last_score = game.score
        last_progress = game.ticks
        boot_attempts = 0
        guide = usage_guide()
        history = deque(maxlen=5)
        initial_obs = ctx.observe()
        startup = bootstrap.startup_pattern(initial_obs, public["rules"]) if isinstance(initial_obs, dict) else []
        start_tick = game.ticks

        def remaining():
            return ctx.budget.remaining()

        def spawn_call(kind, state, questions=None, stage=None):
            wall = remaining().get("wall_s", 0)
            cap = (120.0 if stage == "acquire" else 180.0) if kind == "system2" else 3.0
            timeout = min(cap, max(0.01, wall - 0.1))
            current_session = session

            def work():
                start = time.monotonic()
                try:
                    if kind == "system2":
                        result = ctx.think(state, session_id=current_session, timeout_s=timeout)
                    else:
                        result = ctx.decide(state, questions, timeout_s=timeout)
                    return result
                finally:
                    ctx.log(event="worker_latency", kind=kind,
                            elapsed_s=time.monotonic() - start, stage=stage)

            future = ctx.spawn(work)
            futures.append(future)
            return future

        def result(future):
            try:
                return future.result()
            except Exception as exc:
                if unavailable(exc):
                    raise
                ctx.log(event="model_error", error=type(exc).__name__, detail=str(exc)[:250])
                return None

        try:
            while not ctx.done:
                tick = game.ticks
                obs = ctx.observe()
                if game.ticks != tick:
                    continue
                now = ctx.clock.now()
                if not isinstance(obs, dict):
                    obs = {"observation": obs}
                if game.score != last_score:
                    last_score, last_progress = game.score, tick

                if bootstrap_prediction is not None:
                    old_tick, field, destination = bootstrap_prediction
                    value = obs.get(field)
                    if tick == old_tick + 1 and (not isinstance(value, list) or not value
                            or not point(value[0]) or tuple(value[0]) != destination):
                        ctx.log(event="bootstrap_prediction_mismatch", field=field,
                                predicted=destination, observed=value)
                        bootstrap.binding = None
                        startup = []
                        failure = "observed motion contradicts bootstrap interpretation"
                    if tick != old_tick:
                        bootstrap_prediction = None

                if expected is not None:
                    previous_tick, fields = expected
                    if tick == previous_tick + 1 and any(packed(obs.get(k)) != packed(v)
                                                         for k, v in fields.items()):
                        failure = "deterministic prediction contradicted"
                        ctx.log(event="prediction_mismatch", predicted=fields,
                                actual={k: obs.get(k) for k in fields})
                        policy, lease, memory = None, None, {}
                        last = fallback
                    if tick != previous_tick:
                        expected = None

                if s2 is not None and s2.done():
                    response = result(s2)
                    s2 = None
                    if response is None:
                        # A timeout produced NO code. Do not report a compiler failure.
                        failure = "inference failed before a source artifact was returned"
                        ctx.log(event="compilation_no_artifact", stage=request_stage)
                    else:
                        session = getattr(response, "session_id", None)
                        try:
                            new_source = getattr(response, "text", "")
                            new_policy = compile_policy(new_source)
                            trial = limited(new_policy, copy.deepcopy(obs), {})
                            self.validate(trial, names)
                            policy, memory = new_policy, {}
                            version += 1
                            source = new_source
                            compiled_stage = request_stage
                            lease, expected, pending_audit = None, None, None
                            bootstrap_prediction = None
                            failure = ("acquired executable controller; now improve long-term progress"
                                       if compiled_stage == "acquire" else None)
                            ctx.log(event="policy_installed", version=version, stage=compiled_stage,
                                    tick=tick, score=game.score, source_chars=len(source))
                        except (Exception, PolicyLimit) as exc:
                            failure = "compile/validation: " + str(exc)[:300]
                            ctx.log(event="policy_rejected", reason=failure, stage=request_stage)

                # Consume semantic answers before computing this tick's action.
                if s1 is not None and s1[0].done():
                    future, request = s1
                    answers = result(future)
                    s1 = None
                    if request["kind"] == "bootstrap":
                        ok = bootstrap.bind(answers)
                        ctx.log(event="bootstrap_binding", accepted=ok, binding=bootstrap.binding)
                    elif isinstance(answers, dict) and request["version"] == version:
                        choice = accepted(answers.get("strategy"))
                        audit = accepted(answers.get("audit"), 0.75)
                        if choice and choice != UNKNOWN and tick - request["tick"] <= 20:
                            lease = (version, request["context"], choice, request["tick"] + 20)
                            ctx.log(event="jev_strategy_lease", strategy=choice, version=version)
                        if audit == "revise" and tick - request["tick"] <= 20:
                            pending_audit = (version, request["context"], request["tick"] + 20)

                output = None
                selected = None
                provisional = startup[((tick - start_tick) // 2) % len(startup)] if startup else last
                available = remaining()
                holding = policy is None and (s2 is not None or (
                    attempts < 4 and available.get("system2_calls", 0) > 0
                    and available.get("wall_s", 0) > 1))
                action = bootstrap.choose(obs, last if bootstrap.binding else provisional,
                                          seek=not holding)
                reflex_action = action
                if policy is not None:
                    try:
                        output = limited(policy, copy.deepcopy(obs), memory)
                        options = self.validate(output, names)
                        selected = options[output["default"]]
                        context = packed(output["context_key"])
                        if pending_audit:
                            v, key, expiry = pending_audit
                            if v == version and key == context and tick <= expiry:
                                failure = "Jev found unsupported preconditions or a missing progress strategy"
                                ctx.log(event="jev_requests_revision", version=version)
                            pending_audit = None
                        if lease:
                            v, key, choice, expiry = lease
                            if v == version and key == context and tick <= expiry and choice in options:
                                alternate = options[choice]
                                if alternate["action"] != selected["action"]:
                                    ctx.log(event="jev_changes_action", version=version, tick=tick,
                                            strategy=choice, default=selected["action"],
                                            action=alternate["action"])
                                selected = alternate
                            else:
                                ctx.log(event="jev_lease_invalidated", version=version, tick=tick)
                                lease = None
                        action = selected["action"]
                    except (Exception, PolicyLimit) as exc:
                        failure = "execution: " + str(exc)[:300]
                        ctx.log(event="policy_execution_rejected", reason=failure)
                        policy, lease, memory, output, selected = None, None, {}, None, None
                        last = fallback
                        action = bootstrap.choose(obs, fallback, seek=not holding)

                # Always post BEFORE building prompts or starting inference.
                if ctx.done:
                    break
                # If local work crossed a boundary, discard this action and recompute.
                if game.ticks != tick:
                    ctx.log(event="local_deadline_missed", observation_tick=tick, current_tick=game.ticks)
                    continue
                ctx.act(action)
                last = action
                action_counts["compiled" if selected else (
                    "bootstrap_hold" if holding else "bootstrap_seek")] += 1
                if selected and first_action_version != version:
                    first_action_version = version
                    ctx.log(event="astra_controller_first_action", version=version,
                            stage=compiled_stage, action=action, bootstrap_action=reflex_action,
                            changes_action=action != reflex_action, tick=tick)
                if policy is None and bootstrap.binding and action in bootstrap.vectors:
                    field = bootstrap.binding["trail"]
                    body = obs.get(field)
                    if isinstance(body, list) and body and point(body[0]):
                        dx, dy = bootstrap.vectors[action]
                        bootstrap_prediction = (tick, field, (body[0][0] + dx, body[0][1] + dy))
                if selected and selected.get("expect"):
                    expected = (tick, copy.deepcopy(selected["expect"]))
                history.append({"tick": tick, "score": game.score, "action": action,
                                "facts": output.get("facts") if output else None})

                budget = remaining()
                can_work = not ctx.done and budget.get("wall_s", 0) > 1.0
                if can_work and s1 is None and budget.get("system1_calls", 0) > 0:
                    if bootstrap.binding is None and boot_attempts < 2 and policy is None:
                        boot_attempts += 1
                        state = dict(public, observation=copy.deepcopy(obs))
                        s1 = (spawn_call("system1", state, bootstrap_questions(obs)),
                              {"kind": "bootstrap", "tick": tick})
                    elif output and tick - last_review >= 32:
                        last_review = tick
                        context = packed(output["context_key"])
                        state = dict(public, observation=copy.deepcopy(obs), plan_version=version,
                                     goal=output["goal"], preconditions=output["preconditions"],
                                     facts=output["facts"], candidates=output["candidates"],
                                     recent=list(history), default=output["default"])
                        criteria = {UNKNOWN: "Evidence insufficient; keep the local default."}
                        criteria.update({c["id"]: c["description"] for c in output["candidates"]
                                         if c.get("safe") is True})
                        questions = {
                            "strategy": Choice(
                                "Choose the best reusable strategy from candidates using rules, goal, "
                                "preconditions, facts and recent progress. Code has computed legality; "
                                "do not recalculate coordinates. " + output["question"], criteria),
                            "audit": Choice(
                                "Do rules, observation, facts and recent evidence support the plan's "
                                "preconditions and eventual score progress? Request revision for a "
                                "concrete contradiction or missing strategy; ordinary travel is not failure.",
                                {"keep": "Supported; continue this controller.",
                                 "revise": "Concrete evidence requires a different controller or interface.",
                                 UNKNOWN: "Cannot establish support or contradiction."}),
                        }
                        if len(criteria) <= 2:
                            del questions["strategy"]  # no choice when only one safe strategy exists
                        s1 = (spawn_call("system1", state, questions),
                              {"kind": "review", "version": version, "context": context, "tick": tick})

                if policy is not None and tick - last_progress >= 256 and now - last_compile >= 60:
                    failure = "256 ticks without score change; assess progress and revise only if needed"
                if (can_work and s2 is None and attempts < 4 and failure
                        and now - last_compile >= 10 and budget.get("system2_calls", 0) > 0):
                    attempts += 1
                    last_compile = now
                    payload = dict(public, observation=copy.deepcopy(obs),
                                   bootstrap_binding=bootstrap.binding, recent=list(history),
                                   revision_reason=failure, previous_source=source,
                                   remaining_budget=budget)
                    request_stage = "acquire" if policy is None else "improve"
                    instructions = ACQUIRE_REQUEST if request_stage == "acquire" else IMPROVE_REQUEST
                    prompt = instructions + "\nPUBLIC INPUT\n" + packed(payload) + "\nJEV CONTRACT\n" + guide
                    s2 = spawn_call("system2", prompt, stage=request_stage)
                    ctx.log(event="astra_compile_requested", attempt=attempts, reason=failure,
                            stage=request_stage, tick=tick)
                    failure = None
                if hasattr(game, "tick_s") and not ctx.done:
                    ctx.wait_for_tick()
        finally:
            # Drain every worker even if another worker raised ModelUnavailable.
            propagated = None
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    if unavailable(exc):
                        propagated = exc
            if propagated is not None:
                raise propagated
            ctx.log(event="controller_end", tick=game.ticks, score=game.score,
                    action_counts=action_counts, observation=ctx.observe(),
                    installed_version=version, stage=compiled_stage)

    @staticmethod
    def validate(output, names):
        if not isinstance(output, dict):
            raise ValueError("policy result is not a dict")
        for key in ("candidates", "default", "facts", "goal", "preconditions", "context_key", "question"):
            if key not in output:
                raise ValueError("missing policy result key " + key)
        candidates = output["candidates"]
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
            raise ValueError("invalid candidates")
        if not isinstance(output["question"], str):
            raise ValueError("question must be English text")
        safe = {}
        ids = set()
        for c in candidates:
            if (not isinstance(c, dict) or not isinstance(c.get("id"), str)
                    or c["id"] in ids or c["id"] == UNKNOWN or c.get("action") not in names
                    or not isinstance(c.get("description"), str)
                    or not isinstance(c.get("expect", {}), dict)):
                raise ValueError("invalid candidate schema")
            ids.add(c["id"])
            if c.get("safe") is True:
                safe[c["id"]] = c
        if output["default"] not in safe:
            raise ValueError("default lacks a current safety claim")
        if len(packed(output)) > 50000:
            raise ValueError("policy result too large")
        return safe


def build(ctx):
    return Harness(ctx)
