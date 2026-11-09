"""Text-only Jericho controller: lossless commands, short checked plans, evidence memory."""
from __future__ import annotations

import json
import re
from collections import Counter


SYSTEM = '''You play a turn-based parser adventure to maximize FINAL native score and solve the story.
Use only supplied observations and conversation. No tools or outside files. Commands cost game turns;
thinking does not. You control ordinary parser INPUT TEXT, not action names or API calls.
Use clues, exact observed nouns and learned grammar. Discover prerequisites and scoring opportunities,
keep light/food/unique tools safe, and avoid irreversible losses. Do not wander aimlessly or repeat failed
wording. Prioritize achievable points and prerequisite chains over exhaustive scenery examination.
Interpret a clarification question before issuing another normal command. Never save/restore/restart/undo/quit.

Each response is one JSON object:
{"memory":"Compact updated working notebook: current goal, inventory with uncertainty, learned rules,
important clues, directed routes, unresolved dependencies and failed hypotheses. Cite step numbers for facts.",
 "plan":[{"command":"one actual parser input","expect":["literal fragment"],"stop":false}]}

Return 1-8 commands, usually 3-6 when useful. A batch is executed sequentially, with EVERY response checked.
'expect' is optional: case-insensitive literal substrings, ANY of which must appear in that command's response
for execution to continue. Use short reliable fragments (e.g. 'Taken', a destination room name), never an
English prediction or regex. Empty expect [] accepts normal responses. 'stop':true forces a fresh decision
after this command. Parser errors, clarification questions and immediate hazards interrupt a batch anyway.
Group independent useful inspections, known routes, and clue-supported manipulations. End at a discovery
that determines the next choice; do not guess a long route through unknown rooms. Reading a clue need not
interrupt other independent inspections. New scores are recorded and do not alone interrupt a sound plan.
If you know a possible game solution, treat it as a hypothesis and verify prerequisites against observations.
Keep memory under 180 words, output concise JSON only. No rationale outside memory. No compound commands.
'''

RESERVED = re.compile(r'^(?:restart|restore|save|quit|q|undo|script|unscript|transcript|record|replay)(?:\s|$)', re.I)
FAILURE = re.compile(
    r"not a verb I recogni[sz]e|don't know the word|don't understand|didn't understand|"
    r"can't see any such|cannot see any such|not here|isn't here|aren't here|"
    r"can't go that way|cannot go that way|can't go in that direction|"
    r"you aren't holding|you are not holding|you don't have|you do not have|"
    r"you can't do that|you cannot do that|nothing happens|doesn't seem to work|"
    r"you can't use|you cannot use|that's not something|that is not something|"
    r"what do you want to|which .{0,80}do you mean|please .{0,30}(?:answer|specify)|"
    r"be more specific|not enough|too heavy|carrying too much|already (?:open|closed|have|carrying)", re.I)
HAZARD = re.compile(
    r"pitch (?:black|dark)|likely to be eaten|you are in (?:the )?darkness|"
    r"light .{0,30}(?:fading|dying|dimmer)|lamp .{0,30}(?:dim|flicker|out)|"
    r"you are (?:dead|dying)|you have died|\*\*\*|"
    r"(?:attacks|strikes|wounds|hits) you|you are (?:wounded|bleeding)|"
    r"(?:stolen|steals|disarmed)|(?:very|extremely) (?:hungry|thirsty)|"
    r"(?:yes or no|yes/no|y/n)", re.I)


def observation(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            pass
        return {'text': value}
    return {'text': str(value) if value is not None else ''}


def clean_command(value):
    if not isinstance(value, str):
        return None
    # Remove presentation wrappers only, never map arbitrary prose to an action name.
    value = value.strip().strip('`').strip()
    value = re.sub(r'^(?:command|action)\s*:\s*', '', value, flags=re.I)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    if not value or len(value) > 200 or any(c in value for c in '\n\r\x00'):
        return None
    if value.lower() in ('command', 'action', 'none', 'null') or RESERVED.match(value):
        return None
    if any(c in value for c in '{}[];') or re.search(r'\bthen\b|\.\s*\w', value, re.I):
        return None
    # A trailing punctuation mark is commonly presentation, rather than parser syntax.
    value = value.rstrip('.')
    return value or None


def decode(text):
    text = str(text or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```[^\n]*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text).strip()
    obj = None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        # Recover fenced/prefaced JSON without executing model-generated code.
        for m in re.finditer(r'[\[{]', text):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
                break
            except ValueError:
                continue
    memory = None
    if isinstance(obj, dict):
        memory = obj.get('memory')
        raw = obj.get('plan', obj.get('actions', [obj] if 'command' in obj else []))
    elif isinstance(obj, list):
        raw = obj
    elif isinstance(obj, str):
        raw = [obj]
    else:
        # Backwards-compatible plain parser inputs and numbered command lists.
        lines = [re.sub(r'^\s*(?:[-*>]|\d+[.)])\s*', '', x).strip()
                 for x in text.splitlines() if x.strip()]
        raw = lines if len(lines) <= 8 else []
    if not isinstance(raw, list):
        raw = [raw]
    plan = []
    for item in raw[:8]:
        command = clean_command(item.get('command') if isinstance(item, dict) else item)
        if not command:
            # Never skip an invalid prerequisite and execute its dependent suffix.
            break
        expect = item.get('expect', []) if isinstance(item, dict) else []
        if isinstance(expect, str):
            expect = [expect]
        if not isinstance(expect, list):
            expect = []
        expect = [s[:120].casefold() for s in expect[:6] if isinstance(s, str) and s.strip()]
        plan.append({'command': command, 'expect': expect,
                     'stop': bool(item.get('stop', False)) if isinstance(item, dict) else False})
    if isinstance(memory, (dict, list)):
        memory = json.dumps(memory, ensure_ascii=False)
    return plan, memory[:5000] if isinstance(memory, str) else None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.history = []
        self.memory = ''
        self.feedback = ''
        self.failures = Counter()
        self.call_count = 0

    def record(self, command, obs):
        row = {'step': obs.get('step_count', len(self.history)), 'command': command,
               'text': str(obs.get('text', '')), 'score': obs.get('score'),
               'delta': obs.get('score_delta'), 'moves': obs.get('native_moves')}
        self.history.append(row)
        self.ctx.memory['transcript'] = self.history

    def prompt(self, obs, remaining):
        # The notebook is a model interpretation; exact evidence takes precedence.
        recent = []
        used = 0
        for row in reversed(self.history):
            s = json.dumps(row, ensure_ascii=False)
            if used + len(s) > 42000 and recent:
                break
            recent.append(s)
            used += len(s)
        recent.reverse()
        older = self.history[:-len(recent)] if recent else self.history
        old_events = [r for r in older if r.get('delta')][:40]
        metadata = {k: obs.get(k) for k in ('game_id', 'score', 'max_score', 'score_delta',
                                          'native_moves', 'step_count', 'terminated', 'truncated')}
        return (SYSTEM + '\nCURRENT STATUS: ' + json.dumps(metadata) +
                '\nBUDGET: ' + json.dumps(remaining) +
                '\nWORKING NOTEBOOK (interpretation; verify against transcript):\n' + self.memory +
                '\nCONTROLLER FEEDBACK: ' + (self.feedback or 'New episode. Read the premise and pursue progress.') +
                '\nOLDER SCORE EVENTS: ' + json.dumps(old_events, ensure_ascii=False) +
                '\nEXACT RECENT TRANSCRIPT, oldest first:\n' + '\n'.join(recent) +
                '\nChoose the next useful short plan. Avoid spending a whole call on look when the room is already described.')

    def run_episode(self, game):
        ctx = self.ctx
        obs = observation(ctx.observe())
        self.record(None, obs)
        invalid_outputs = 0
        backend_failures = 0
        while not ctx.done and not obs.get('terminated') and not obs.get('truncated'):
            remaining = ctx.budget.remaining()
            if remaining.get('system2_calls', 0) <= 0 or remaining.get('wall_s', 0) < 6:
                return
            self.call_count += 1
            try:
                # Each call is self-contained, avoiding an ever-growing hidden conversation.
                result = ctx.think(self.prompt(obs, remaining), session_id=None,
                                   timeout_s=max(1.0, min(180.0, remaining['wall_s'] - 3.0)))
            except Exception as exc:
                ctx.log('think_failed', error=type(exc).__name__)
                backend_failures += 1
                if ctx.done or backend_failures >= 3:
                    return
                self.feedback = 'Previous reasoning call failed; continue from the exact unchanged state.'
                continue
            backend_failures = 0
            plan, memory = decode(getattr(result, 'text', ''))
            if memory:
                self.memory = memory
                ctx.memory['notebook'] = memory
            if not plan:
                invalid_outputs += 1
                self.feedback = 'Invalid output. Return JSON with plan:[{"command":"actual parser text"}]. Do not output the label command.'
                ctx.log('invalid_plan', attempt=invalid_outputs)
                if invalid_outputs >= 3:
                    return
                continue
            invalid_outputs = 0
            self.feedback = ''
            ctx.log('plan', commands=[p['command'] for p in plan], call=self.call_count)
            for item in plan:
                if ctx.done:
                    return
                command = item['command']
                before_text = str(obs.get('text', ''))
                key = (command.casefold(), before_text, obs.get('score'))
                if self.failures[key] >= 2:
                    self.feedback = 'Blocked repeated failed command in the same observed context: ' + command + '. Change the hypothesis or prerequisites.'
                    break
                try:
                    raw = ctx.act(command)
                    obs = observation(ctx.observe() if raw is None else raw)
                except Exception as exc:
                    ctx.log('action_failed', command=command, error=type(exc).__name__)
                    if ctx.done:
                        return
                    self.failures[key] += 1
                    self.feedback = 'Action rejected by interface: ' + command + '. Error: ' + type(exc).__name__
                    break
                self.record(command, obs)
                if ctx.done or obs.get('terminated') or obs.get('truncated'):
                    return
                response = str(obs.get('text', ''))
                fail = FAILURE.search(response)
                hazard = HAZARD.search(response)
                mismatch = bool(item['expect']) and not any(x in response.casefold() for x in item['expect'])
                if fail:
                    self.failures[key] += 1
                reason = ('parser/precondition response: ' + fail.group(0) if fail else
                          'hazard/prompt: ' + hazard.group(0) if hazard else
                          'expected response missing: ' + repr(item['expect']) if mismatch else
                          'requested observation checkpoint' if item['stop'] else '')
                if reason:
                    self.feedback = 'Plan interrupted after ' + repr(command) + ': ' + reason + '. Remaining commands were NOT executed. Reconcile the exact response.'
                    ctx.log('plan_interrupted', command=command, reason=reason)
                    break
            else:
                self.feedback = 'Plan completed. Use all new observations and update the notebook.'


def build(ctx):
    return Harness(ctx)
