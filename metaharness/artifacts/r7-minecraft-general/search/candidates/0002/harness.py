"""Runtime-compiled guarded options. No game vocabulary or observation paths built in."""
from __future__ import annotations

import json
import math
import re
from collections import deque

from metaharness.models.jev import Choice, Noul, usage_guide


# A bounded data interpreter, never eval/exec or model-generated Python.
class Invalid(ValueError):
    pass


def expression(x, env, depth=0):
    if depth > 24:
        raise Invalid('expression depth')
    if not isinstance(x, list):
        if isinstance(x, dict):
            raise Invalid('expressions use arrays, not objects')
        return x
    if not x or not isinstance(x[0], str):
        raise Invalid('operator missing')
    op, args = x[0], x[1:]
    ev = lambda a: expression(a, env, depth + 1)
    if op == 'lit':
        return args[0]
    if op == 'get':
        value = env
        path = args[0]
        parts = path.split('.') if isinstance(path, str) else path
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and str(part).isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                if len(args) > 1:
                    return ev(args[1])
                raise Invalid('missing path: ' + str(path))
        return value
    if op == 'and':
        return all(ev(a) for a in args)
    if op == 'or':
        return any(ev(a) for a in args)
    if op == 'if':
        return ev(args[1]) if ev(args[0]) else ev(args[2])
    if op in ('filter', 'map'):
        seq = ev(args[0])
        if not isinstance(seq, list) or len(seq) > 2048:
            raise Invalid('collection limit')
        out = []
        for v in seq:
            value = expression(args[1], dict(env, v=v), depth + 1)
            if op == 'map':
                out.append(value)
            elif value:
                out.append(v)
        return out
    a = [ev(v) for v in args]
    if op == 'not': return not a[0]
    if op == 'eq': return a[0] == a[1]
    if op == 'ne': return a[0] != a[1]
    if op == 'lt': return a[0] < a[1]
    if op == 'le': return a[0] <= a[1]
    if op == 'gt': return a[0] > a[1]
    if op == 'ge': return a[0] >= a[1]
    if op == 'in': return a[0] in a[1]
    if op == 'contains': return a[1] in a[0]
    if op == 'endswith': return a[0].endswith(a[1])
    if op == 'startswith': return a[0].startswith(a[1])
    if op == 'replace': return a[0].replace(a[1], a[2])
    if op == 'concat': return ''.join(str(v) for v in a)
    if op == 'len': return len(a[0])
    if op == 'keys': return list(a[0])
    if op == 'values': return list(a[0].values())
    if op == 'items': return [{'key': k, 'value': v} for k, v in a[0].items()]
    if op == 'at': return a[0][a[1]]
    if op == 'sum': return sum(a[0])
    if op == 'add': return sum(a)
    if op == 'sub': return a[0] - a[1]
    if op == 'mul': return math.prod(a)
    if op == 'div': return a[0] / a[1]
    if op == 'min': return min(a)
    if op == 'max': return max(a)
    if op == 'abs': return abs(a[0])
    if op == 'ceil': return math.ceil(a[0])
    if op == 'floor': return math.floor(a[0])
    if op == 'dist': return math.dist(a[0], a[1])
    raise Invalid('unknown operator: ' + op)


DATA_ERRORS = (Invalid, ValueError, TypeError, KeyError, IndexError, AttributeError,
               ArithmeticError, RecursionError)


def compact(value, limit=48, depth=0):
    """Preserve every dictionary field; sample long lists without broken JSON."""
    if depth > 14:
        return '<depth limit>'
    if isinstance(value, dict):
        return {str(k): compact(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > limit:
            # Head plus evenly spaced tail, rather than losing all later types.
            half = limit // 2
            indices = list(range(half)) + [half + i * (len(value)-half)//half for i in range(half)]
            return {'sample_of': len(value), 'values': [compact(value[i], limit, depth+1) for i in indices]}
        return [compact(v, limit, depth + 1) for v in value]
    if isinstance(value, str):
        return value[:4000]
    return value


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def parse_json(text):
    if not isinstance(text, str) or len(text) > 100000:
        raise Invalid('missing/oversized planner response')
    decoder = json.JSONDecoder()
    for match in list(re.finditer(r'\{', text))[:40]:
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict) and isinstance(value.get('rules'), list):
                return value
        except ValueError:
            pass
    raise Invalid('expected policy JSON object with rules array')


COMPILER = r'''
Design a small executable policy for THIS game, using only its public description and
observations. The world continues while you think. Return JSON only, no Python.
Do not return bare action names when arguments are required. Your job is to compile
rules into reusable, fully parameterized action alternatives; Jev only recognizes
which of those alternatives is appropriate. Discover all field paths and game rules
from the supplied material. No other game is assumed. Prefer 6-12 short rules that
cover a productive dependency chain for the next 45 seconds, with recovery and
survival options. On the FIRST call, return only 4-7 rules for the immediate opening
and keep JSON under 1800 tokens so useful control starts early. Later expand it.

JSON schema (all expressions below use the DSL):
{
 "goal":"short strategic goal and dependency order",
 "semantics":"brief field meanings, exact relevant mechanics, risks and uncertainties for Jev",
 "select_hint":"how Jev should compare already instantiated alternatives",
 "danger_hint":"observable reasons to stop the current activity urgently",
 "busy":EXPR, "feedback":EXPR, "failed":EXPR,
 "remaining":EXPR, "stop":"a complete legal interruption command, or empty string",
 "pulse":true,
 "rules":[{
   "id":"unique name", "why":"subgoal, preconditions, benefit, risks",
   "priority":100, "when":EXPR,
   "each":EXPR, "where":EXPR, "order":EXPR,
   "action":[TOKEN_EXPR,...], "cooldown":3, "emergency":false
 }]
}
Higher priority first. 'each' yields an array to bind one element as c; omit each
for a single action. 'when' runs before binding c; 'where' filters each c; 'order'
is a numeric cost (lower first). Omitted when/where=true, order=0. All expressions
are JSON primitives (literal) or arrays [operator,arg,...]. Objects require ["lit",object].
Supported operators:
 ["get","o.some.path",DEFAULT] reads current observation; omit default to FAIL CLOSED
 on absent fields. Paths can also be arrays of literal keys/indices. o is observation,
 c is current bound candidate, m is harness memory. m.now is elapsed controller seconds,
 m.last_action is last dispatched command, m.last_at is dispatch time, m.sent is a
 dictionary command->last dispatch time. No other memory fields exist.
 ["lit",array_or_object] for literal containers (plain arrays are operators).
 eq ne lt le gt ge (2 args), and/or (short circuit, any count), not (1), if (3).
 in(value,container), contains(container,value), startswith/endswith(string,suffix),
 replace(string,old,new), concat(any scalars...), len(container), keys/values/items(dict).
 items returns an array of {key:...,value:...}; at(container,key) accesses a dynamic key.
 filter(array,predicate), map(array,expression): each element bound to v, preserving c/o/m.
 sum(array), add/sub/mul/div, min/max (scalar args), abs/ceil/floor, dist(two vectors).
 No wildcard paths, regex, assignment, Python, or other operations.
 Action is an array of evaluated scalar tokens, joined with spaces, preserving arguments.
 Example for an imaginary machine game (NOT facts about this game):
 {"id":"charge","priority":90,"when":["lt",["get","o.energy"],5],
  "each":["get","o.docks"],"where":["eq",["get","c.ready"],true],
  "order":["get","c.distance"],"action":["dock",["get","c.code"]],
  "why":"Use a ready charger while energy is low"}.

busy must detect the observed exclusive running job (false if no such concept).
feedback extracts the latest action outcome; failed is true ONLY on explicit failure.
remaining returns seconds left, or null if unavailable. stop must be the documented
interrupt action if one exists. pulse=true if actions are latched/bounded, so the
controller posts the spec's neutral action between jobs; false only for continuous
held controls. Emergency rules may run when idle. While busy the controller permits
only stop; it will wait for idle before dispatching any replacement.

Encode known arithmetic and resource prerequisites as guards. Use authoritative
observations to confirm completion. Avoid speculative exact inventory counts and
unobserved IDs. Bind moving/location targets from the CURRENT observation, never
literal snapshot IDs; all required resources/tools/recipes must be guarded. Use
feasible recipe data when supplied. Guard one-time benefits by observed milestone
or possession state, avoiding needless repeats. Preserve enough materials for the
next prerequisite. Acquisition should cover just-needed quantities before optional
reserves. Avoid trying to do pathfinding or long planning inside Jev. Offer a few
plausible reachable targets with explanations. Exploration rules must avoid cycling
and should have explicit cooldowns; use m.sent if useful. Never assume hidden terrain.
Delayed processes need loading and later retrieval rules, not blocking waits.
Use low-priority safe recovery if no progress option exists. Distinguish uncertain
facts from false facts. Every safety-critical guard must fail closed if absent.
An observation may contain {sample_of:N,values:[...]}: that is a PROMPT-ONLY summary;
actual o has the original full list. Don't compile a path through values/sample_of.
Revisions replace the whole policy. Keep useful rules; repair unsupported paths,
failures, missing options, and stale assumptions. Recent evidence and parser errors
are provided. No need to re-explain the DSL or emit the Jev API itself.
'''


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        spec = game.action_spec
        names = [str(n) for n in spec.names]
        neutral = spec.noop if spec.noop is not None else names[0]
        descriptions = dict(spec.descriptions)
        rules_text = game.describe()
        interface = {'names': names, 'descriptions': descriptions,
                     'noop': neutral, 'notes': spec.notes}
        # Infer minimum argument count only where the public text gives a signature.
        arity = {}
        for name in names:
            desc = str(descriptions.get(name, ''))
            match = re.match(r'^\s*' + re.escape(name) + r'\s+([^:]+):', desc)
            if match:
                required = re.sub(r'\[[^\]]*\]', '', match.group(1)).split()
                arity[name] = len(required)
        plan = None
        version = 0
        planner = judge = None
        planner_started = judge_started = 0.0
        judge_version = 0
        judge_menu = []
        history = deque(maxlen=18)
        errors = deque(maxlen=8)
        sent = {}
        blocked = {}
        last_action = neutral
        last_at = -1000.0
        last_feedback = None
        next_plan = 0.0
        last_plan_at = -1000.0
        next_judge = 0.0
        need_plan = True
        plan_attempts = 0
        origin = ctx.clock.now()
        guide = usage_guide()
        last_busy = False
        pulse_pending = False
        latency_s1 = []
        latency_s2 = []

        def now():
            return ctx.clock.now() - origin

        def environment(obs):
            return {'o': obs, 'm': {'now': now(), 'last_action': last_action,
                                   'last_at': last_at, 'sent': dict(sent)}}

        def value(key, obs, default):
            if plan is None or key not in plan:
                return default
            try:
                return expression(plan[key], environment(obs))
            except DATA_ERRORS:
                return default

        def command(tokens, env):
            if not isinstance(tokens, list) or not tokens or len(tokens) > 20:
                raise Invalid('action must be token array')
            pieces = [expression(t, env) for t in tokens]
            if any(not isinstance(v, (str, int, float)) or isinstance(v, bool) for v in pieces):
                raise Invalid('action tokens must be scalars')
            if any(not str(v) or re.search(r'\s', str(v)) for v in pieces):
                raise Invalid('empty/whitespace action token')
            pieces = [str(int(v)) if isinstance(v, float) and v.is_integer() else str(v) for v in pieces]
            if pieces[0] not in names or len(pieces)-1 < arity.get(pieces[0], 0):
                raise Invalid('unknown verb/missing required argument')
            result = ' '.join(pieces)
            if len(result) > 300:
                raise Invalid('command too long')
            return result

        def options(obs, busy):
            if not plan:
                return []
            env = environment(obs)
            found = []
            for index, rule in enumerate(plan['rules'][:32]):
                try:
                    if not isinstance(rule, dict) or not expression(rule.get('when', True), env):
                        continue
                    seq = expression(rule['each'], env) if 'each' in rule else [None]
                    if not isinstance(seq, list) or len(seq) > 2048:
                        raise Invalid('each must be bounded list')
                    local = []
                    for c in seq:
                        bound = dict(env, c=c)
                        try:
                            if not expression(rule.get('where', True), bound):
                                continue
                            action = command(rule.get('action'), bound)
                            if busy and action != plan.get('stop'):
                                continue
                            cooldown = max(1.0, min(60.0, float(rule.get('cooldown', 3))))
                            if now() < blocked.get(action, -1) or now()-sent.get(action, -1000) < cooldown:
                                continue
                            cost = float(expression(rule.get('order', 0), bound))
                            priority = float(rule.get('priority', 0))
                            if not math.isfinite(cost) or not math.isfinite(priority):
                                raise Invalid('nonfinite ranking')
                            local.append({'action': action, 'rule': str(rule.get('id', index)),
                                          'why': str(rule.get('why', ''))[:1600],
                                          'priority': priority, 'cost': cost,
                                          'target': compact(c, 12),
                                          'emergency': rule.get('emergency') is True})
                        except DATA_ERRORS as exc:
                            msg = str(exc)[:200]
                            if msg not in errors: errors.append(msg)
                    local.sort(key=lambda r: r['cost'])
                    found.extend(local[:3])
                except DATA_ERRORS as exc:
                    msg = str(exc)[:200]
                    if msg not in errors: errors.append(msg)
            found.sort(key=lambda r: (not r['emergency'], -r['priority'], r['cost']))
            unique = {}
            for item in found:
                unique.setdefault(item['action'], item)
            return list(unique.values())[:12]

        def timeout(call, seconds):
            # Never catch RuntimeError broadly: ModelUnavailable must escape.
            try:
                return call()
            except TimeoutError as exc:
                # Preserve an unavailable exception even if a provider subclasses timeout.
                if any(c.__name__ == 'ModelUnavailable' for c in type(exc).__mro__):
                    raise
                return {'_timeout': str(exc), 'limit': seconds}

        def is_timeout(result):
            return isinstance(result, dict) and '_timeout' in result

        try:
            while not ctx.done:
                obs = ctx.observe()
                t = now()
                busy = bool(value('busy', obs, False))
                if last_busy and not busy:
                    next_judge = min(next_judge, t)
                last_busy = busy
                feedback = value('feedback', obs, None)
                signature = dump(feedback)
                if signature != last_feedback:
                    last_feedback = signature
                    if feedback is not None:
                        history.append({'at': t, 'feedback': compact(feedback)})
                        if bool(value('failed', obs, False)):
                            blocked[last_action] = t + 25
                            need_plan = True
                            ctx.log(event='option_failed', action=last_action, feedback=feedback)

                if planner is not None and planner.done():
                    result = planner.result()
                    planner = None
                    latency_s2.append(t - planner_started)
                    try:
                        if is_timeout(result):
                            raise Invalid('planner timeout')
                        proposal = parse_json(getattr(result, 'text', None))
                        if not proposal['rules'] or len(proposal['rules']) > 32:
                            raise Invalid('policy requires 1..32 rules')
                        if not all(isinstance(r, dict) and isinstance(r.get('action'), list) for r in proposal['rules']):
                            raise Invalid('malformed rule')
                        stop = proposal.get('stop', '')
                        if not isinstance(stop, str):
                            raise Invalid('invalid stop command')
                        if stop:
                            command(stop.split(), {})
                        plan = proposal
                        version += 1
                        errors.clear()
                        need_plan = False
                        next_plan = t + 35
                        next_judge = t
                        ctx.log(event='policy_installed', version=version, policy=plan,
                                planner_latency_s=t-planner_started)
                    except DATA_ERRORS as exc:
                        errors.append(str(exc)[:240])
                        need_plan = True
                        next_plan = t + 8
                        ctx.log(event='policy_rejected', reason=str(exc))
                    busy = bool(value('busy', obs, False))

                menu = options(obs, busy)
                dispatch = None
                if judge is not None and judge.done():
                    answers = judge.result()
                    judge = None
                    age = t-judge_started
                    latency_s1.append(age)
                    if not is_timeout(answers) and isinstance(answers, dict) and judge_version == version and age <= 2.5:
                        choice = answers.get('option')
                        selected = getattr(choice, 'value', 'unknown')
                        probabilities = getattr(choice, 'probabilities', {}) or {}
                        probability = probabilities.get(selected, 0.0)
                        confidence = getattr(choice, 'confidence', None)
                        try:
                            acceptable = float(probability) >= 0.50 and (confidence is None or float(confidence) >= 0.15)
                        except (ValueError, TypeError):
                            acceptable = False
                        old = next((r for r in judge_menu if r['key'] == selected), None)
                        current = next((r for r in menu if old and r['action'] == old['action'] and r['rule'] == old['rule']), None)
                        danger = answers.get('danger')
                        danger_value = getattr(danger, 'value', 'unknown')
                        danger_p = (getattr(danger, 'probabilities', {}) or {}).get('stop', 0)
                        clear_p = (getattr(danger, 'probabilities', {}) or {}).get('clear', 0)
                        revise = getattr(answers.get('revise'), 'value', 0)
                        try:
                            if float(revise) >= 0.8: need_plan = True
                            urgent = danger_value == 'stop' and float(danger_p) >= 0.75
                            clear = danger_value == 'clear' and float(clear_p) >= 0.60
                        except (TypeError, ValueError):
                            urgent = False
                            clear = False
                        if urgent and busy and plan.get('stop'):
                            dispatch = plan['stop']
                            need_plan = True
                        elif current and acceptable and (clear or current['emergency']):
                            dispatch = current['action']
                        elif not (busy and selected == 'hold'):
                            need_plan = True
                        ctx.log(event='jev_control', version=version, selected=selected,
                                winning_probability=probability, confidence=confidence,
                                age_s=age, legal_now=current is not None, danger=danger_value,
                                dispatch=dispatch, replan=need_plan)
                    else:
                        ctx.log(event='jev_discarded', age_s=age, obsolete=judge_version != version,
                                timeout=is_timeout(answers))
                    next_judge = t + (0.5 if busy else 0.0)

                # Issue a neutral tick between bounded actions and while inference is late.
                # An accepted action is always left posted until a tick boundary.
                if pulse_pending:
                    ctx.act(neutral)
                    pulse_pending = False
                    dispatch = None
                elif dispatch is not None:
                    # Re-evaluate full guards against a newly acquired observation, not the request snapshot.
                    latest = ctx.observe()
                    busy_now = bool(value('busy', latest, False))
                    legal = options(latest, busy_now)
                    stopping = busy_now and dispatch == plan.get('stop')
                    if stopping or any(r['action'] == dispatch for r in legal):
                        ctx.act(dispatch)
                        last_action, last_at = dispatch, t
                        sent[dispatch] = t
                        history.append({'at': t, 'action': dispatch})
                        pulse_pending = plan.get('pulse', True) is not False
                        ctx.log(event='command_posted', action=dispatch, version=version)
                    else:
                        ctx.act(neutral)
                        ctx.log(event='dispatch_revalidation_failed', action=dispatch)
                else:
                    if plan is None or plan.get('pulse', True) is not False or busy:
                        ctx.act(neutral)
                    else:
                        # Continuous controls get at most a one-second lease without renewed judgment.
                        ctx.act(last_action if t-last_at <= 1 else neutral)

                budget = ctx.budget.remaining()
                remaining = value('remaining', obs, None)
                game_left = float(remaining) if isinstance(remaining, (int, float)) and not isinstance(remaining, bool) else math.inf
                wall_left = float(budget.get('wall_s', 0))
                if (planner is None and plan_attempts < 12 and budget.get('system2_calls', 0) > 1
                        and wall_left > 3 and game_left > 5
                        and (t >= next_plan or (need_plan and t-last_plan_at >= 10))):
                    prompt = (COMPILER + '\nJev local usage contract:\n' + guide
                              + '\nGAME:\n' + rules_text + '\nINTERFACE:\n' + dump(interface)
                              + '\nOBSERVATION:\n' + dump(compact(obs, 128))
                              + '\nPREVIOUS POLICY:\n' + dump(plan)
                              + '\nRECENT EVIDENCE:\n' + dump(list(history))
                              + '\nCOMPILER ERRORS:\n' + dump(list(errors)))
                    seconds = max(1.0, min(40.0, wall_left-1, game_left-1))
                    planner_started = last_plan_at = t
                    plan_attempts += 1
                    next_plan = t + 8
                    planner = ctx.spawn(lambda p=prompt, s=seconds: timeout(lambda: ctx.think(p, timeout_s=s), s))
                    ctx.log(event='policy_requested', attempt=plan_attempts, reason='revision' if plan else 'bootstrap')

                if (plan is not None and judge is None and t >= next_judge and not pulse_pending
                        and budget.get('system1_calls', 0) > 1 and wall_left > 1
                        and ctx.system1 is not None):
                    menu = options(ctx.observe(), busy)
                    judge_menu = [dict(r, key='option_' + str(i)) for i, r in enumerate(menu)]
                    criteria = {'unknown': 'No offered action has sufficiently supported preconditions and benefit; continue neutral safely.',
                                'hold': 'The running bounded task is progressing safely; let it continue without replacement.'}
                    criteria.update({r['key']: dump(r) for r in judge_menu})
                    state = {'rules': rules_text, 'interface': interface,
                             'plan': {k: plan.get(k) for k in ['goal', 'semantics', 'select_hint', 'danger_hint']},
                             'observation': compact(ctx.observe(), 32),
                             'summary_note': 'Long arrays appear as {sample_of:N,values:[...]}; omitted values are unknown. Options were generated from full observations.',
                             'options': judge_menu, 'busy': busy,
                             'recent': list(history)[-8:], 'version': version}
                    questions = {
                        'option': Choice('Choose the best fully instantiated option for plan.goal using observation, options and recent evidence. '
                                         'Prefer higher priority when safe and feasible; compare target risk and obstruction using plan.select_hint. '
                                         'Do not invent actions or plan an entire route. Return unknown if none applies.', criteria),
                        'danger': Choice('Using rules, observation, busy, and plan.danger_hint, is continuing the current activity unsafe right now? '
                                         'Judge independently of the option choice.', {
                                             'unknown': 'Evidence is insufficient to establish safety.',
                                             'clear': 'No observed reason requires an urgent interruption.',
                                             'stop': 'Observed immediate danger requires interrupting the current activity.'}),
                        'revise': Noul('Do observation and recent evidence show that plan.goal or the supplied options need strategic revision '
                                       '(repeated failure, missing prerequisite, no productive options while idle, or changed danger)? '
                                       'A safely progressing bounded task does not require revision merely because no replacement is offered.')}
                    seconds = max(0.2, min(2.0, wall_left-0.1, max(0.2, game_left)))
                    judge_started, judge_version = t, version
                    judge = ctx.spawn(lambda s=state, q=questions, d=seconds: timeout(lambda: ctx.decide(s, q, timeout_s=d), d))
                    next_judge = t + 0.5
                ctx.wait_for_tick()
        finally:
            # Join both even if one raises, and preserve ModelUnavailable (or any real error).
            failures = []
            for future in (planner, judge):
                if future is not None:
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(exc)
            ctx.log(event='harness_summary', plan_attempts=plan_attempts,
                    s1_observed_latency_s=latency_s1, s2_observed_latency_s=latency_s2,
                    commands=len(sent))
            if failures:
                unavailable = next((e for e in failures if any(c.__name__ == 'ModelUnavailable' for c in type(e).__mro__)), None)
                raise unavailable if unavailable is not None else failures[0]


def build(ctx):
    return Harness(ctx)
