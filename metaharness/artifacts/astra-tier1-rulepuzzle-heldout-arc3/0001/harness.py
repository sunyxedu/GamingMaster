"""Evidence-carrying, guarded action batches; no game-specific observation keys."""
from __future__ import annotations

import copy
import json
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
        if isinstance(obj, dict) and ("plan" in obj or "actions" in obj):
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


PROTOCOL = """You are controlling an unfamiliar turn-based game. Maximize completed objectives,
then minimize reasoning calls. Infer rules ONLY from this game's description and observed
transitions. The harness has no built-in movement, terrain, colour, or field semantics.
You receive complete current observations, retained hypotheses and numbered experimental evidence.
Calls are scarce but actions are a separate budget. Spend calls on discovery and planning, not
individual predictable steps. Reason carefully, then return ONE JSON object (no code):
{
  "schema": {"feature_name": {"path": ["actual", "observation", "key"], "role": "state"}},
  "hypotheses": [{"id":"h1", "claim":"conditional action effect", "actions":["legal name"], "evidence":[1]}],
  "strategy": "short current objective, unresolved alternatives, next discriminating test",
  "plan": [{"action":"legal name and optional arguments", "before":{}, "expect":{}, "uses":["h1"], "probe":true}]
}
SCHEMA is optional on later calls; when supplied it REPLACES the previous schema.
Paths use exact observed dictionary keys and zero-based array indices. A feature reads that path.
Optional "find":VALUE returns ALL relative paths matching VALUE, in traversal order. For example,
selecting an observed matrix and finding its unique token yields [[row,column]], NOT [row,column].
Optional "count":true with find returns the match count instead. Never infer a token's meaning
from its numeric value alone. Select a small sufficient state: controllable entity, relevant
inventory/mechanisms, phase, and resource counters. "role":"boundary" means ANY change stops a
batch (use for stage/round/scene identifiers); "role":"clock" excludes action/time counters from
loop detection. Other features default to "state" and unexpected changes stop non-probe batches.
Use simple raw paths if no spatial interpretation is justified. Do not put the whole changing
board in a state feature when you can extract the entity and important mechanisms. Raw observations
and cell deltas are always supplied, even when they are not features.
PLAN: usually 6-16 actions for a known route, or 2-4 carefully selected low-risk experiments to
identify unknown controls. You may return up to 24 actions, particularly when few calls remain.
The executor runs these actions cheaply, observing after EACH one. "before" and "expect" map
feature names to exact values, checked before/after the step. Predict controllable state after
each known action. A failed check cancels the REST of the plan and challenges hypotheses in uses.
Any boundary change ends a batch so the next scene is replanned. A step without expect executes
only once then replans, UNLESS probe:true explicitly authorizes open-loop experimental execution;
no more than 4 consecutive such unchecked probes run. Probe results still enter the evidence log.
Do not label known routes as probes. A blocked attempt does not establish that an action has no
effect everywhere. Try an informative context to distinguish obstruction from wrong semantics.
After controls are identified, pursue the goal immediately; unnecessary interactions cost actions.
Interaction/unknown terrain: approach with checked steps, then end in a targeted probe. Never
reuse a route after an unexpected result. Avoid repeating failed state/action pairs without a new
hypothesis. Preserve episode rules across scenes only as hypotheses; never import seed-specific
semantics from another episode. Hypotheses are model claims, not automatically proven facts.
Retain useful earlier hypotheses in your replacement list; cite numbered evidence for each.
All schema, action meanings and plans must be inferred afresh for this episode.
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
        calls = steps = interrupts = failed_checks = 0
        memory_key = "guarded_program_v1"
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
            chosen.update(i for i, e in enumerate(history) if e.get("failure"))
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
                      "\nHYPOTHESES (untrusted unless supported): " + dump(hypotheses) +
                      "\nTRANSITION EVIDENCE: " + dump(evidence()) +
                      "\nRECENT SCENE STARTS: " + dump(scenes[-2:]) +
                      "\nCURRENT STATE VISITS: " + str(visits[signature(current)]) +
                      "\nEXECUTOR FEEDBACK: " + feedback +
                      "\nPRIOR EPISODE PROCEDURAL STATISTICS ONLY: " + dump(prior_stats) +
                      "\nOn your last call, supply the longest justified checked plan to the objective.")
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
                         and all(isinstance(x, (str, int)) for x in v["path"])}
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
            plan = reply.get("plan", reply.get("actions", []))
            if not isinstance(plan, list) or not plan:
                feedback = "Invalid/empty plan. Return protocol JSON with legal actions; no action was spent."
                ctx.log(event="plan_parse_error", call=calls)
                continue
            feedback = "Batch finished normally."
            unchecked = 0
            executed = 0
            ctx.log(event="plan", call=calls, schema=schema, hypotheses=hypotheses, plan=plan[:24])
            for raw in plan[:24]:
                if ctx.done or steps >= 1000:
                    break
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
                unchecked = unchecked + 1 if not expect else 0
                if unchecked > 4:
                    feedback = "Four unchecked probes completed; interpret evidence before more probes."
                    break
                old = copy.deepcopy(obs)
                nxt = ctx.act(action)
                obs = copy.deepcopy(game.observe() if nxt is None else nxt)
                steps += 1
                executed += 1
                after = features(obs, schema)
                mismatch = {k: {"expected": v, "actual": after[k]}
                            for k, v in expect.items() if after[k] != v}
                boundary = [k for k, spec in schema.items()
                            if spec.get("role") == "boundary" and before[k] != after[k]]
                surprise = [k for k, spec in schema.items()
                            if spec.get("role", "state") == "state" and k not in expect
                            and before[k] != after[k]]
                record = {"id": steps, "action": action, "before": before, "after": after,
                          "delta": differences(old, obs), "probe": probe}
                if expect:
                    record["prediction"] = expect
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
                if boundary or mismatch or (surprise and not probe):
                    interrupts += 1
                    feedback = "Suffix cancelled. " + dump({"boundary": boundary, "prediction_failure": mismatch,
                                                            "unpredicted_features": surprise, "transition": steps})
                    break
                if visits[key] >= 3 and not probe:
                    interrupts += 1
                    feedback = "Repeated state reached three times; suffix cancelled. Choose a new test or route."
                    break
                if not expect and not probe:
                    feedback = "One unchecked action executed. Predict features to enable batching, or explicitly request probes."
                    break
            ctx.log(event="batch_end", call=calls, executed=executed, feedback=feedback)
        # No action identities, state schemas, maps, or inferred rules cross seed boundaries.
        ctx.memory[memory_key] = {
            "episodes": prior_stats["episodes"] + 1,
            "steps": prior_stats["steps"] + steps,
            "calls": prior_stats["calls"] + calls,
            "interrupts": prior_stats["interrupts"] + interrupts,
            "method": "Use discriminating probes, verify each planned outcome, replan on counterexamples.",
        }
        ctx.log(event="episode_controller_end", calls=calls, steps=steps,
                prediction_failures=failed_checks, interrupts=interrupts)


def build(ctx):
    return Harness(ctx)
