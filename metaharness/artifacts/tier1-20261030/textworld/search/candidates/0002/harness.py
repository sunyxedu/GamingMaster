"""Observed-state interpreter for Astra's semantic quest program."""
from __future__ import annotations

import json
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


OPS = ('go', 'take', 'drop', 'open', 'close', 'unlock', 'lock', 'put', 'insert', 'eat', 'examine')


def norm(s):
    return ' '.join(str(s).strip().split()).casefold()


def commands(obs):
    return [x for x in obs.get('admissible_commands', []) if isinstance(x, str)]


def room(obs):
    m = re.search(r'-=\s*(.*?)\s*=-', str(obs.get('description', '')))
    return m.group(1) if m else 'unknown room'


def signature(obs):
    return (str(obs.get('description', '')), str(obs.get('inventory', '')),
            tuple(sorted(commands(obs))), obs.get('score'))


def ended(obs):
    return any(obs.get(k) for k in ('won', 'lost', 'terminated', 'truncated', 'infrastructure_error'))


def exact(wanted, offered):
    matches = [s for s in offered if norm(s) == norm(wanted)]
    return matches[0] if len(matches) == 1 else None


def parse_program(text, offered):
    """Accept JSON or exact command lines; never default to the first action."""
    decoder = json.JSONDecoder()
    candidates = []
    for i, char in enumerate(str(text)):
        if char not in '[{':
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            value = value.get('steps', value.get('plan'))
        if isinstance(value, list):
            candidates = value
            break
    if not candidates:
        for line in str(text).splitlines():
            line = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', line).strip(' `"')
            found = exact(line, offered)
            if found:
                candidates.append(found)
    steps = []
    for entry in candidates[:40]:
        if isinstance(entry, str):
            entry = {'command': entry}
        if not isinstance(entry, dict) or not isinstance(entry.get('command'), str):
            continue
        command = ' '.join(entry['command'].strip().split())
        if '\n' in entry['command'] or ';' in command:
            continue
        if command.split(' ', 1)[0].lower() not in OPS and command not in ('look', 'inventory'):
            continue
        steps.append({'command': command,
                      'room': entry.get('room') if isinstance(entry.get('room'), str) else '',
                      'consume_evidence': entry.get('consume_evidence') if isinstance(entry.get('consume_evidence'), str) else ''})
    return steps


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = []
        self.history = []
        self.rooms = {}
        self.edges = {}
        self.attempts = Counter()
        self.reason = 'Compile the objective into an ordered executable program.'
        self.malformed = 0
        self.auxiliary = 0

    def remember(self, obs):
        self.rooms[room(obs)] = {'description': obs.get('description', ''),
                                 'commands_when_last_seen': commands(obs)}
        self.ctx.memory['quest_program'] = self.plan
        self.ctx.memory['verified_history'] = self.history
        self.ctx.memory['observed_rooms'] = self.rooms

    def route(self, start, target):
        target = next((r for r in self.rooms if norm(r) == norm(target)), target)
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            here, path = queue.popleft()
            if here == target:
                return path
            for action, there in self.edges.get(here, {}).items():
                if there not in seen:
                    seen.add(there)
                    queue.append((there, path + [action]))
        return None

    def resolve(self, wanted, offered):
        found = exact(wanted, offered)
        if found:
            return found
        # Omitted key is a slot, not permission to guess: exactly one current
        # admissible command must establish the matching key.
        if wanted.startswith(('unlock ', 'lock ')) and ' with ' not in wanted:
            matches = [s for s in offered if norm(s).startswith(norm(wanted) + ' with ')]
            return matches[0] if len(matches) == 1 else None
        if wanted.startswith('take ') and ' from ' not in wanted:
            matches = [s for s in offered if norm(s).startswith(norm(wanted) + ' from ')]
            return matches[0] if len(matches) == 1 else None
        return None

    def achieved(self, wanted, obs):
        available = commands(obs)
        verb, _, obj = wanted.partition(' ')
        if verb == 'take':
            return bool(exact('drop ' + obj.split(' from ')[0], available))
        if verb == 'open':
            return bool(exact('close ' + obj, available))
        if verb == 'close':
            return bool(exact('open ' + obj, available) or self.resolve('unlock ' + obj, available))
        if verb == 'unlock':
            target = obj.split(' with ')[0]
            return bool(exact('open ' + target, available) or exact('close ' + target, available)
                        or self.resolve('lock ' + target, available))
        if verb == 'lock':
            return bool(self.resolve('unlock ' + obj.split(' with ')[0], available))
        return False

    def prerequisite(self, wanted, obs):
        offered = commands(obs)
        target = None
        if wanted.startswith('open '):
            target = wanted[5:]
            return self.resolve('unlock ' + target, offered)
        if wanted.startswith('take ') and ' from ' in wanted:
            target = wanted.split(' from ', 1)[1]
        if wanted.startswith('insert ') and ' into ' in wanted:
            target = wanted.split(' into ', 1)[1]
        if target:
            return exact('open ' + target, offered) or self.resolve('unlock ' + target, offered)
        return None

    def verified(self, action, before, after):
        if after.get('protocol_error') or after.get('infrastructure_error'):
            return False
        if after.get('won'):
            return True
        if signature(before) == signature(after):
            return action.startswith('examine ') or action in ('look', 'inventory')
        verb, _, obj = action.partition(' ')
        if verb == 'go':
            return room(before) != room(after)
        if verb in ('open', 'close', 'unlock', 'lock', 'take'):
            return self.achieved(action, after)
        if verb in ('drop', 'eat', 'put', 'insert'):
            item = obj
            if verb == 'put':
                item = obj.split(' on ', 1)[0]
            elif verb == 'insert':
                item = obj.split(' into ', 1)[0]
            removed = exact('drop ' + item, commands(before)) and not exact('drop ' + item, commands(after))
            if verb in ('put', 'insert'):
                sep = ' on ' if verb == 'put' else ' into '
                target = obj.split(sep, 1)[-1]
                return bool(removed and exact('take ' + item + ' from ' + target, commands(after)))
            return bool(removed)
        return True

    def prompt(self, obs):
        state = {k: obs.get(k) for k in ('objective', 'description', 'inventory', 'score', 'max_score',
                                       'commands_remaining', 'admissible_commands', 'protocol_error')}
        if self.history or obs.get('last_command'):
            state['feedback'] = obs.get('feedback')
        return '''You are the planner for a TextWorld quest-program interpreter.
Read the literal objective carefully; it often states an ordered route and actions.
Translate ALL remaining requirements into a short JSON program. Only observations
are facts. Future action templates are hypotheses and the controller checks them.
It will bind every template to the CURRENT admissible list, check the resulting
state after every command, and ask you again at an unknown precondition or failure.

Reply ONLY with {"steps":[{"command":"..."}, ...]}.
Commands: go north/south/east/west; take OBJECT [from HOLDER]; drop OBJECT;
open TARGET; close TARGET; unlock TARGET [with KEY]; lock TARGET [with KEY];
put OBJECT on SUPPORTER; insert OBJECT into CONTAINER; eat FOOD; examine OBJECT.
Use canonical full names and spelling from objective/current commands. Preserve
case of object names. You MAY omit "with KEY" if the key is not named: the controller
will choose only a unique admissible matching-key action. Do not invent a key.
The controller can open/unlock a named source or destination when this is a
necessary currently admissible prerequisite. It cannot guess routes or missing objects.
Use an optional "room" guard ONLY when the objective explicitly gives that action's
room; omit room if unknown. Movement steps' room guard means the SOURCE room.
Eating is irreversible: include "consume_evidence": "EXACT objective quote requiring
this food to be eaten" on every eat step, and eat only at the required point.
Never eat unrelated foods, undo, restart, or issue compound commands.
Do not repeat already VERIFIED steps. State predicates already satisfied may be
skipped; movement is never skipped merely because a reverse route was seen.
If all planned steps ran but the game did not end, investigate missing final
conditions. Partial score is not victory. Prefer goal actions and prerequisites;
use exploration only when information is missing. No need to look/inventory since
both are provided. If uncertain, give a short investigative plan and stop before
an unsupported guess. Avoid long prose or explanations.

REPLAN REASON: ''' + self.reason + '\nCURRENT:\n' + json.dumps(state, ensure_ascii=False) + \
            '\nPREVIOUS REMAINING PLAN:\n' + json.dumps(self.plan) + \
            '\nVERIFIED / FAILED ACTION HISTORY:\n' + json.dumps(self.history[-60:], ensure_ascii=False) + \
            '\nOBSERVED ROOMS (past snapshots may be stale):\n' + json.dumps(self.rooms, ensure_ascii=False) + \
            '\nOBSERVED DIRECTED EDGES:\n' + json.dumps(self.edges) + \
            '\nPROGRESS:\n' + json.dumps(self.ctx.progress())

    def replan(self, obs):
        remaining = self.ctx.budget.remaining()
        if remaining.get('system2_calls', 0) <= 0 or remaining.get('wall_s', 0) < 3:
            return False
        try:
            answer = self.ctx.think(self.prompt(obs), session_id=None,
                                    timeout_s=min(240.0, max(1.0, remaining['wall_s'] - 1)))
        except BudgetExceeded:
            return False
        except (TimeoutError, RuntimeError, ModelUnavailable) as exc:
            self.reason = 'Planner call failed: ' + type(exc).__name__ + '. Retry from current evidence.'
            self.malformed += 1
            return self.malformed < 3 and not self.ctx.done
        parsed = parse_program(answer.text, commands(obs))
        if not parsed:
            self.reason = 'Invalid response: return the requested JSON steps. No command was executed.'
            self.malformed += 1
            self.plan = []
            return self.malformed < 3
        self.plan = parsed
        self.malformed = 0
        self.auxiliary = 0
        self.reason = ''
        self.ctx.log('quest_program', steps=self.plan)
        return True

    def run_episode(self, game):
        ctx = self.ctx
        obs = ctx.observe()
        if not isinstance(obs, dict):
            return
        self.remember(obs)
        repair = True
        while not ctx.done and not ended(obs):
            if repair or not self.plan:
                if not self.replan(obs):
                    return
                repair = False
                if not self.plan or self.reason:
                    repair = True
                    continue
            step = self.plan[0]
            wanted = step['command']
            offered = commands(obs)
            action = None
            auxiliary = False
            guard = step.get('room')
            if guard and norm(guard) != norm(room(obs)):
                path = self.route(room(obs), guard)
                if path:
                    action = exact(path[0], offered)
                    auxiliary = True
                if not action:
                    self.reason = 'Room guard not satisfied, no observed traversable route: ' + str(step)
                    repair = True
                    continue
            if not action:
                if self.achieved(wanted, obs):
                    self.ctx.log('already_observed', step=step)
                    self.history.append({'already_satisfied': wanted, 'room': room(obs),
                                         'evidence': commands(obs)})
                    self.plan.pop(0)
                    continue
                action = self.resolve(wanted, offered)
                if not action:
                    action = self.prerequisite(wanted, obs)
                    auxiliary = True
            if not action:
                self.reason = 'Next template has no current admissible binding or verified prerequisite: ' + wanted
                repair = True
                continue
            if action.startswith('eat '):
                quote = step.get('consume_evidence', '')
                objective = str(obs.get('objective', ''))
                if (not quote or norm(quote) not in norm(objective)
                        or norm(action[4:]) not in norm(quote)
                        or not re.search(r'\b(eat|consume|devour|swallow|ingest)\b', quote, re.I)):
                    self.reason = 'Consumption guard requires an exact objective quote authorizing this food: ' + action
                    repair = True
                    continue
            key = (signature(obs), action)
            if self.attempts[key] >= (1 if action.startswith('examine ') or action in ('look', 'inventory') else 2):
                self.reason = 'Loop guard: repeated command in unchanged relevant state: ' + action
                repair = True
                continue
            self.attempts[key] += 1
            before = obs
            progress_before = ctx.progress()
            try:
                obs = ctx.act(action)
                if not isinstance(obs, dict):
                    obs = ctx.observe()
            except BudgetExceeded:
                return
            except Exception as exc:
                self.reason = 'Action raised ' + type(exc).__name__ + ': ' + str(exc)[:300]
                repair = True
                obs = ctx.observe()
                if not isinstance(obs, dict):
                    return
                continue
            if not isinstance(obs, dict):
                return
            progress = ctx.progress()
            valid = self.verified(action, before, obs)
            if progress.get('illegal_actions', 0) > progress_before.get('illegal_actions', 0):
                valid = False
            if valid and action.startswith('go ') and room(before) != room(obs):
                self.edges.setdefault(room(before), {})[action] = room(obs)
            if not valid and action.startswith('go '):
                self.edges.get(room(before), {}).pop(action, None)
            self.history.append({'command': action, 'from': room(before), 'to': room(obs),
                                 'verified': bool(valid), 'prerequisite_or_route': auxiliary,
                                 'feedback': obs.get('feedback', ''), 'protocol_error': obs.get('protocol_error'),
                                 'score': obs.get('score')})
            self.remember(obs)
            if ended(obs) or ctx.done:
                return
            if not valid:
                self.reason = 'Postcondition failed; do not assume the last action succeeded. See history and current state.'
                repair = True
                continue
            if not auxiliary:
                self.plan.pop(0)
                self.auxiliary = 0
            else:
                self.auxiliary += 1
            if progress.get('since_obs_change', 0) >= 2 or self.auxiliary >= 8:
                self.reason = 'Progress guard: no observation change or too many prerequisites. Diagnose and repair.'
                repair = True
            elif not self.plan:
                self.reason = 'Program completed but native game is not terminal. Find remaining conditions; do not stop at partial credit.'


def build(ctx):
    return Harness(ctx)
