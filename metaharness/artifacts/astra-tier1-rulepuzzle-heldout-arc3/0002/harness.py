"""Evidence-carrying contingent plans; no built-in game rules or field names."""
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
Any boundary change ends a batch so the next scene is replanned. A step without expect or cases
executes only once then replans, UNLESS probe:true authorizes open-loop experimental execution;
no more than 4 consecutive such unchecked probes run. Probe results still enter the evidence log.
Do not label known routes as probes. A blocked attempt does not establish that an action has no
effect everywhere. Try an informative context to distinguish obstruction from wrong semantics.
After controls are identified, pursue the goal immediately; unnecessary interactions cost actions.
Interaction/unknown terrain: approach with checked steps, then use a targeted probe with cases
when plausible outcomes have useful continuations (see CONTINGENT PLANS below). Never
reuse a route after an unexpected result. Avoid repeating failed state/action pairs without a new
hypothesis. Preserve episode rules across scenes only as hypotheses; never import seed-specific
semantics from another episode. Hypotheses are model claims, not automatically proven facts.
Retain useful earlier hypotheses in your replacement list; cite numbered evidence for each.
All schema, action meanings and plans must be inferred afresh for this episode.

CONTINGENT PLANS: Do not spend another call merely to recognize an outcome you can anticipate.
After a targeted experiment, optionally provide exact alternative outcomes and their continuations:
{"action":"legal name", "before":{}, "cases":[
  {"when":{"some_state_feature":"possible observed value"}, "then":[
    {"action":"legal name", "before":{}, "expect":{"some_state_feature":"next value"}}
  ]},
  {"when":{"some_state_feature":"another possible value"}, "then":[]}
]}
Feature names and values in this illustration are placeholders, not game facts. Construct cases
only from this episode's description and evidence. This is bounded JSON, not executable code.
A cases step MUST be LAST in its list. Each then is the entire continuation for that outcome;
there is no shared suffix or wildcard/default branch. Do not combine expect and cases on one step.
Every when must include a state feature and EXACT post-action values for ALL state features that
change on that outcome. All unlisted state features must remain unchanged. In particular, account
for token occlusion or inventory changes if your schema tracks them. Missing feature paths reject
continuations. Exactly one case must match; zero or multiple matches replan without further action.
A phase/boundary change ALWAYS replans even if a case matches. Terminal game status always stops.
then:[] explicitly requests replanning. Up to 4 cases per experiment, 4 nested case layers,
64 action nodes across the entire tree, and 24 actions actually executed per call.
Known-route steps still need ordinary expect checks. Cases do NOT license unchecked routes.
Use a small conditional tree when a test could enable immediate progress using already learned
controls. You need not enumerate all possible worlds: an unlisted outcome safely returns to you.
If useful continuations cannot yet be justified, use the ordinary short probe batch instead.
Cases are possible outcomes, NOT proven action rules; a match records evidence, not certainty.
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
        memory_key = "contingent_program_v2"
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
            plan = reply.get("plan", reply.get("actions", []))
            if not isinstance(plan, list) or not plan:
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
            ctx.log(event="plan", call=calls, schema=schema, hypotheses=hypotheses, plan=plan[:24])
            pending = list(plan)
            while pending and executed < 24:
                if ctx.done or steps >= 1000:
                    break
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
                after = features(obs, schema)
                mismatch = {k: {"expected": v, "actual": after[k]}
                            for k, v in expect.items() if after[k] != v}
                boundary = [k for k, spec in schema.items()
                            if spec.get("role") == "boundary" and before[k] != after[k]]
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
                    if not pending:
                        feedback = "Matched outcome explicitly requested replanning (empty then)."
                        break
                    unchecked = 0
                    continue
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
            if executed >= 24 and pending and not ctx.done:
                feedback = "Executed 24 actions; remaining tree discarded. Replan from current observation."
            ctx.log(event="batch_end", call=calls, executed=executed, feedback=feedback)
        # No action identities, state schemas, maps, or inferred rules cross seed boundaries.
        ctx.memory[memory_key] = {
            "episodes": prior_stats["episodes"] + 1,
            "steps": prior_stats["steps"] + steps,
            "calls": prior_stats["calls"] + calls,
            "interrupts": prior_stats["interrupts"] + interrupts,
            "branch_hits": previous.get("branch_hits", 0) + branch_hits,
            "branch_misses": previous.get("branch_misses", 0) + branch_misses,
            "method": "Preplan useful experiment outcomes; match exact state, otherwise replan.",
        }
        ctx.log(event="episode_controller_end", calls=calls, steps=steps,
                prediction_failures=failed_checks, interrupts=interrupts,
                branch_hits=branch_hits, branch_misses=branch_misses)


def build(ctx):
    return Harness(ctx)
