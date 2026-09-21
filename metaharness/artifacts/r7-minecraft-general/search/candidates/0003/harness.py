"""Short grounded action menus, fresh batched recognition, and a tick executor.

Only the runtime models interpret game vocabulary. No generated code is executed.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import deque

from metaharness.models.jev import Choice, Noul


class Invalid(ValueError):
    pass


MISSING = object()
DATA_ERRORS = (ValueError, TypeError, KeyError, IndexError, AttributeError,
               ArithmeticError, RecursionError)


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def compact(value, limit=20, depth=0):
    """Keep all named fields, sample long arrays; never cut serialized JSON."""
    if depth > 16:
        return '<omitted deep data>'
    if isinstance(value, dict):
        return {str(k): compact(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= limit:
            return [compact(v, limit, depth + 1) for v in value]
        # Preserve head, tail and diverse records. The original indices are explicit.
        indices = sorted(set(list(range(limit // 2)) +
                             [int(i * (len(value) - 1) / (limit // 2 - 1))
                              for i in range(limit // 2)]))
        return {'omitted_count': len(value) - len(indices),
                'indexed_sample': {str(i): compact(value[i], limit, depth + 1)
                                   for i in indices}}
    return value[:3000] if isinstance(value, str) else value


def get_path(obs, path, default=MISSING):
    if not isinstance(path, list) or not path or len(path) > 16:
        return default
    try:
        for key in path:
            obs = obs[key]
        return obs
    except (TypeError, KeyError, IndexError):
        return default


def strings(value, depth=0):
    if depth > 24:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(set(str(k) for k in value),
                           *(strings(v, depth + 1) for v in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(strings(v, depth + 1) for v in value))
    return set()


def reference_evidence(obs, refs):
    """Retain records for selected targets even when the general array is sampled."""
    found = []
    def walk(value, depth=0):
        if depth > 24 or len(found) >= 20:
            return
        if isinstance(value, dict):
            if any(isinstance(v, str) and v in refs for v in value.values()):
                found.append(compact(value))
            else:
                for v in value.values():
                    walk(v, depth + 1)
        elif isinstance(value, (list, tuple)):
            if any(isinstance(v, str) and v in refs for v in value):
                found.append(compact(value))
            else:
                for v in value:
                    walk(v, depth + 1)
    walk(obs)
    return found


def guards_hold(guards, obs):
    """Exact conjunction of path comparisons; missing values fail closed."""
    try:
        for guard in guards:
            path, op, expected = guard[:3]
            actual = get_path(obs, path, guard[3] if len(guard) == 4 else MISSING)
            if actual is MISSING:
                return False
            if op == 'eq': ok = actual == expected
            elif op == 'ne': ok = actual != expected
            elif op == 'ge': ok = actual >= expected
            elif op == 'le': ok = actual <= expected
            elif op == 'gt': ok = actual > expected
            elif op == 'lt': ok = actual < expected
            elif op == 'contains': ok = expected in actual
            else: return False
            if not ok:
                return False
        return True
    except DATA_ERRORS:
        return False


def arities(names, descriptions):
    result = {}
    for name in names:
        match = re.match(r'^\s*' + re.escape(name) + r'\s+([^:]+):',
                         str(descriptions.get(name, '')))
        if match:
            result[name] = len(re.sub(r'\[[^\]]*\]', '', match.group(1)).split())
    return result


def command(value, names, minimum):
    if not isinstance(value, str) or len(value) > 300 or '\n' in value:
        raise Invalid('expected short complete command string')
    parts = value.split()
    if not parts or parts[0] not in names or len(parts) - 1 < minimum.get(parts[0], 0):
        raise Invalid('unknown verb or missing arguments')
    return ' '.join(parts)


def parse_menu(text, names, minimum):
    if not isinstance(text, str) or len(text) > 24000:
        raise Invalid('missing or oversized menu')
    decoder = json.JSONDecoder()
    obj = None
    for match in list(re.finditer(r'\{', text))[:20]:
        try:
            candidate, _ = decoder.raw_decode(text[match.start():])
            if isinstance(candidate, dict) and isinstance(candidate.get('options'), list):
                obj = candidate
                break
        except ValueError:
            pass
    if obj is None or not 1 <= len(obj['options']) <= 5:
        raise Invalid('expected JSON menu with 1..5 options')
    for key in ('goal', 'meaning'):
        if not isinstance(obj.get(key, ''), str):
            raise Invalid('goal and meaning must be text')
    for key in ('busy_path', 'remaining_path', 'feedback_path'):
        path = obj.get(key, [])
        if (not isinstance(path, list) or len(path) > 16 or
                any(not isinstance(k, (str, int)) or isinstance(k, bool) for k in path)):
            raise Invalid('paths must be lists of literal keys/indices')
        obj[key] = path
    obj['stop'] = command(obj['stop'], names, minimum) if obj.get('stop') else ''
    if obj.get('mode') not in ('bounded', 'continuous'):
        raise Invalid('mode must be bounded or continuous')
    for i, opt in enumerate(obj['options']):
        if not isinstance(opt, dict):
            raise Invalid('option must be an object')
        opt['command'] = command(opt.get('command'), names, minimum)
        if not isinstance(opt.get('ready'), str) or not opt['ready'].strip():
            raise Invalid('each option needs an observable readiness condition')
        opt['key'] = 'a' + str(i)
        opt.setdefault('refs', [])
        opt.setdefault('guards', [])
        if (not isinstance(opt['refs'], list) or len(opt['refs']) > 12 or
                any(not isinstance(r, str) or r not in opt['command'].split() for r in opt['refs'])):
            raise Invalid('refs must be exact object-ID command tokens')
        if not isinstance(opt['guards'], list) or len(opt['guards']) > 16:
            raise Invalid('invalid guards')
        for g in opt['guards']:
            if (not isinstance(g, list) or len(g) not in (3, 4) or
                    not isinstance(g[0], list) or not g[0] or
                    any(not isinstance(k, (str, int)) for k in g[0]) or
                    g[1] not in ('eq', 'ne', 'ge', 'le', 'gt', 'lt', 'contains')):
                raise Invalid('guard is [path, comparison, value, optional missing default]')
    return obj


PLANNER = '''You control an unfamiliar real-time game. Return a SMALL actionable menu,
not a policy program. Thinking consumes game time. Aim for under 450 output tokens.
Use the public rules, current observation, and recent results. Choose one short useful
subgoal and 2-4 complete commands covering its next steps or alternatives. A single
command is fine if uncertain. Preserve ALL arguments; never output only a verb when
parameters are required. No invented IDs or hidden information. Prefer prerequisite
progress and cheap score over optional reserves; use observed feasible actions/recipes.
Account exactly for quantities, consumed inputs, rounded output batches and delayed
processes. Later steps can await an earlier step's observed result. A command is used
at most once per menu. Do not repeat completed benefits or recently failed targets.
A safe ongoing job should finish; include an interruption only for observed danger.

JSON shape:
{"goal":"short subgoal", "meaning":"essential mechanics/field authority/risks",
 "mode":"bounded", "busy_path":["observed job field"],
 "remaining_path":["observed remaining seconds field"],
 "feedback_path":["observed result field"], "stop":"legal interruption or empty",
 "options":[{"command":"complete whitespace-separated action",
 "ready":"specific current facts that must hold; purpose and priority",
 "refs":["observed object ID tokens in this command"],
 "guards":[[["actual","field","path"],"ge",3]]}]}
Paths are literal key/index lists into the original observation. Use [] if there is
no such field. busy_path points to a nullable/falsy idle field (no expressions).
mode=bounded means neutral keeps a bounded job going and clears the command latch;
continuous means a chosen control needs renewal. stop must be the actual interrupt.
refs contains location/entity/object IDs requiring current presence, NOT item names,
output names, directions or quantities. guards are exact AND comparisons eq/ne/ge/le/
gt/lt/contains; optional fourth entry is a missing-value default ONLY when the rules
justify it (e.g. absent inventory count=0). Use guards for resource arithmetic and
other exact readiness facts. Never invent paths or substitute a missing fact with 0.
Jev also checks the natural-language ready conditions against LIVE observations.
Long arrays are sampled with original indices for display only; never use
indexed_sample/omitted_count as actual paths. You may omit guards if a condition is
not directly expressible, but keep explicit readiness text. No code, DSL, or essay.

Jev contract relevant to this interface: Jev sees only our explicit state/questions,
not your conversation. It returns typed Choice/Noul values and probabilities, not
plans or text. Independent per-option readiness checks and option selection are
batched. Put required context in goal/meaning/ready. Code checks guards/ID presence,
job exclusivity, freshness and one-shot bookkeeping before sending the whole command.
'''


def unavailable(exc):
    return any(c.__name__ == 'ModelUnavailable' for c in type(exc).__mro__)


def inference(call):
    """Recover ordinary deadline expiry without hiding provider unavailability."""
    start = time.monotonic()
    try:
        value = call()
        return value, time.monotonic() - start, None
    except TimeoutError as exc:
        if unavailable(exc):
            raise
        return None, time.monotonic() - start, str(exc)


def probability(answer, key):
    try:
        p = float((getattr(answer, 'probabilities', None) or {}).get(key, 0))
        confidence = getattr(answer, 'confidence', None)
        if confidence is not None and (not math.isfinite(float(confidence)) or float(confidence) < .1):
            return 0.0
        return p if math.isfinite(p) and 0 <= p <= 1 else 0.0
    except DATA_ERRORS:
        return 0.0


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game):
        ctx = self.ctx
        spec = game.action_spec
        names = list(spec.names)
        neutral = spec.noop if spec.noop is not None else names[0]
        interface = {'names': names, 'descriptions': dict(spec.descriptions),
                     'notes': spec.notes, 'noop': neutral}
        rules = game.describe()
        minimum = arities(names, interface['descriptions'])
        plan = None
        version = 0
        planner = judge = None
        planner_at = judge_at = 0.0
        judge_version = 0
        judge_options = []
        next_plan = next_judge = retry_after = 0.0
        last_request = -1000.0
        installed_at = 0.0
        used = set()
        sent = {}
        history = deque(maxlen=16)
        errors = deque(maxlen=4)
        last_feedback = None
        need_plan = True
        calls = commands = 0
        pulse = False
        held_action = neutral
        held_until = 0.0
        origin = ctx.clock.now()

        def now():
            return ctx.clock.now() - origin

        def busy(obs):
            if not plan or not plan['busy_path']:
                return False
            value = get_path(obs, plan['busy_path'])
            return value is MISSING or bool(value)

        def available(obs):
            if not plan or now() - installed_at > 40:
                return []
            present = strings(obs)
            return [o for o in plan['options'] if o['key'] not in used
                    and now() - sent.get(o['command'], -1000) >= 12
                    and all(r in present for r in o['refs'])
                    and guards_hold(o['guards'], obs)]

        try:
            while not ctx.done:
                obs = ctx.observe()
                t = now()
                dispatch = None
                chosen_key = None
                if plan and plan['feedback_path']:
                    feedback = get_path(obs, plan['feedback_path'], None)
                    signature = dump(feedback)
                    if signature != last_feedback:
                        last_feedback = signature
                        history.append({'at': t, 'feedback': compact(feedback)})

                if planner is not None and planner.done():
                    result, latency, error = planner.result()
                    planner = None
                    try:
                        if error:
                            raise Invalid('planner timeout: ' + error)
                        proposed = parse_menu(getattr(result, 'text', None), names, minimum)
                        # A command dispatched after this planner's snapshot must not
                        # be replayed by its delayed response.
                        plan = proposed
                        version += 1
                        installed_at = t
                        used = {o['key'] for o in plan['options']
                                if sent.get(o['command'], -1000) >= planner_at}
                        need_plan = False
                        next_plan = t + 18
                        next_judge = t
                        errors.clear()
                        ctx.log(event='menu_installed', version=version, menu=plan,
                                latency_s=latency, observation_age_s=t-planner_at)
                    except DATA_ERRORS as exc:
                        errors.append(str(exc)[:240])
                        need_plan = True
                        next_plan = retry_after = t + 6
                        ctx.log(event='menu_rejected', reason=str(exc), latency_s=latency)

                if judge is not None and judge.done():
                    answers, latency, error = judge.result()
                    judge = None
                    age = t - judge_at
                    if not error and isinstance(answers, dict) and judge_version == version and age <= 3:
                        choice = answers.get('choose')
                        selected = getattr(choice, 'value', None)
                        selected = selected if isinstance(selected, str) else 'unknown'
                        cp = probability(choice, selected)
                        ready_answer = answers.get('ready_' + selected)
                        rp = probability(ready_answer, 'ready')
                        danger = answers.get('danger')
                        urgent = (getattr(danger, 'value', None) == 'stop'
                                  and probability(danger, 'stop') >= .75)
                        revise = getattr(answers.get('revise'), 'value', 0)
                        if isinstance(revise, (float, int)) and revise >= .8:
                            need_plan = True
                        latest = ctx.observe()
                        current = available(latest)
                        picked = next((o for o in current if o['key'] == selected), None)
                        if urgent and plan['stop'] and busy(latest):
                            dispatch = plan['stop']
                            need_plan = True
                        elif picked and cp >= .45 and rp >= .7 and not busy(latest):
                            dispatch, chosen_key = picked['command'], picked['key']
                        ctx.log(event='jev_control', version=version, selected=selected,
                                choice_probability=cp, ready_probability=rp,
                                confidence=getattr(choice, 'confidence', None),
                                latency_s=latency, age_s=age, urgent=urgent,
                                replan=need_plan, proposed_action=dispatch)
                    else:
                        ctx.log(event='jev_discarded', age_s=age, latency_s=latency,
                                error=error, obsolete=judge_version != version)
                    next_judge = t + (1 if busy(obs) else .25)

                # Post once per tick. A neutral pulse clears latches between jobs.
                if pulse:
                    ctx.act(neutral)
                    pulse = False
                elif dispatch is not None:
                    latest = ctx.observe()
                    stopping = plan and dispatch == plan['stop'] and busy(latest)
                    legal = (stopping or (not busy(latest) and
                             any(o['key'] == chosen_key for o in available(latest))))
                    if legal:
                        ctx.act(dispatch)
                        sent[dispatch] = t
                        if chosen_key:
                            used.add(chosen_key)
                        commands += 1
                        held_action, held_until = dispatch, t + 1
                        pulse = plan['mode'] == 'bounded' or stopping
                        history.append({'at': t, 'action': dispatch, 'version': version})
                        ctx.log(event='command_posted', command=dispatch, version=version,
                                source='jev', key=chosen_key)
                    else:
                        ctx.act(neutral)
                else:
                    ctx.act(held_action if plan and plan['mode'] == 'continuous'
                            and t < held_until else neutral)

                budget = ctx.budget.remaining()
                wall = float(budget.get('wall_s', 0))
                remaining = get_path(obs, plan['remaining_path'], None) if plan else None
                left = float(remaining) if isinstance(remaining, (int, float)) else math.inf
                room = min(wall, left)
                unused = [] if not plan else [o for o in plan['options'] if o['key'] not in used]
                replenish = plan is not None and len(unused) <= 1
                if (planner is None and calls < 30 and budget.get('system2_calls', 0) > 1
                        and room > 5 and t >= next_plan
                        and t - last_request >= 8):
                    # Regular horizon expiry also refreshes an unproductive menu.
                    need_plan = True
                if (planner is None and calls < 30 and budget.get('system2_calls', 0) > 1
                        and room > 5 and t - last_request >= 8
                        and (need_plan or replenish) and t >= retry_after):
                    prompt = (PLANNER + '\nPUBLIC GAME:\n' + rules + '\nINTERFACE:\n' + dump(interface)
                              + '\nCURRENT OBSERVATION:\n' + dump(compact(obs, 24))
                              + '\nPREVIOUS MENU:\n' + dump(plan)
                              + '\nUSED OPTION KEYS:\n' + dump(sorted(used))
                              + '\nRECENT ACTIONS AND RESULTS:\n' + dump(list(history))
                              + '\nFORMAT ERRORS:\n' + dump(list(errors)))
                    seconds = max(.2, min(28, room - 1))
                    planner_at = last_request = t
                    calls += 1
                    need_plan = False
                    planner = ctx.spawn(lambda p=prompt, s=seconds:
                                        inference(lambda: ctx.think(p, timeout_s=s)))
                    ctx.log(event='menu_requested', attempt=calls, timeout_s=seconds)

                if (plan and judge is None and not pulse and t >= next_judge and room > .5
                        and budget.get('system1_calls', 0) > 1 and ctx.system1 is not None):
                    latest = ctx.observe()
                    judge_options = available(latest)
                    is_busy = busy(latest)
                    criteria = {'hold': 'Continue the safe running job, or idle because no offered action is ready.',
                                'unknown': 'Cannot establish a supported useful action; request new planning.'}
                    criteria.update({o['key']: dump(o) for o in judge_options})
                    state = {'rules': rules, 'interface': interface,
                             'plan': {'goal': plan.get('goal'), 'meaning': plan.get('meaning')},
                             'observation': compact(latest, 24), 'options': judge_options,
                             'busy': is_busy, 'recent': list(history)[-8:], 'version': version,
                             'target_evidence': reference_evidence(latest,
                                 {r for o in judge_options for r in o['refs']}),
                             'summary_note': 'Sampled arrays show original indices. Omitted facts are unknown. '
                                             'Commands are one-shot; do not assume previous output arrived.'}
                    questions = {
                        'choose': Choice('Using rules, plan, observation and recent outcomes, select the '
                                         'highest priority offered command whose readiness conditions hold NOW. '
                                         'No new plan or parameters are requested. If busy, hold unless unsafe.', criteria),
                        'danger': Choice('Does the current observation show immediate danger requiring '
                                         'interruption of the running job? Assess independently of the action choice.',
                                         {'clear': 'No observed immediate reason to interrupt.',
                                          'stop': 'Observed immediate danger calls for interruption.',
                                          'unknown': 'Insufficient evidence.'}),
                        'revise': Noul('Do current facts or recent failures contradict the plan or show '
                                       'a missing prerequisite/target that needs a new menu? Safe ongoing work '
                                       'and waiting for its confirmed result do not alone require revision.')}
                    for o in judge_options:
                        questions['ready_' + o['key']] = Choice(
                            'Check ONLY this command: ' + dump(o) + '. Using observation, rules and recent, '
                            'are ALL its prerequisites, target safety and usefulness supported NOW? '
                            'Do not predict a pending job has finished. Ignore other questions.',
                            {'unknown': 'A required fact cannot be established.',
                             'ready': 'All preconditions hold; useful, safe and ready to start now.',
                             'blocked': 'A prerequisite is false, job is busy, benefit already achieved, or unsafe.'})
                    judge_at, judge_version = t, version
                    seconds = max(.2, min(2.5, room - .1))
                    judge = ctx.spawn(lambda s=state, q=questions, d=seconds:
                                      inference(lambda: ctx.decide(s, q, timeout_s=d)))
                ctx.wait_for_tick()
        finally:
            failures = []
            for future in (planner, judge):
                if future is not None:
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(exc)
            ctx.log(event='harness_summary', planner_attempts=calls, commands=commands)
            if failures:
                raise next((e for e in failures if unavailable(e)), failures[0])


def build(ctx):
    return Harness(ctx)
