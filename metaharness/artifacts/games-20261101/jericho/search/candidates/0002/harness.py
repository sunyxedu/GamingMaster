"""Astra compiles bounded, branching parser policies; code interprets evidence.

Text-only control: no VM access, simulator search, walkthrough, or seed logic.
Every node submits one ordinary parser string. Edges inspect its actual response.
"""
from __future__ import annotations

import json
import re
from collections import Counter


RESERVED = {
    'command', 'action', 'noop', 'none', 'null', 'restart', 'restore', 'save',
    'quit', 'q', 'undo', 'script', 'unscript', 'transcript', 'recording', 'replay',
}
QUERY = re.compile(r'^(?:look|l|inventory|i|examine|x|read|help|how\b|what\b)\b', re.I)
PARSER = re.compile(
    r"not a verb|don't know the word|do not know the word|don't understand|"
    r"do not understand|didn't understand|not understand|not a sentence|"
    r"beg your pardon|not recogni[sz]|only understood|don't know how|"
    r"can't see any such|cannot see any such|no such (?:thing|object)|"
    r"not something you need to refer", re.I)
BLOCKED = re.compile(
    r"can't go|cannot go|can't do|cannot do|not possible|can't reach|"
    r"cannot reach|not holding|not carrying|don't have|do not have|"
    r"seems to be closed|is closed|is locked|seems to be locked|"
    r"won't open|will not open|too heavy|carrying too|not strong enough|"
    r"can't take|cannot take|can't see|cannot see|you must first|you need to", re.I)
PROMPT = re.compile(
    r"which .{0,160}(?:do you mean|did you mean)|who do you mean|"
    r"what do you want to|what do you wish to|please (?:answer|type|enter).*"
    r"(?:yes|no)|yes or no|\(y/n\)|\[y/n\]|would you like to", re.I | re.S)
DANGER = re.compile(
    r"pitch[- ](?:black|dark)|grue|too dark to see|in (?:total|complete) darkness|"
    r"you are (?:dead|dying)|you have died|\*\*\*.*(?:died|dead)|"
    r"lamp.*(?:dimming|flicker|running out|going out)|"
    r"you are (?:hungry|thirsty|starving)|you feel (?:hungry|thirsty)|"
    r"attacks you|lunges at you|strikes you|wounds you|you are wounded|"
    r"rising water|water is rising|stolen|steals|disarmed", re.I)
MENU = re.compile(r'(?:q\s*=\s*resume game|return\s*=\s*read subject|help menu\s*:)', re.I)
MENU_EXIT = re.compile(r'\bq\s*=\s*resume game\b', re.I)
WORDS = re.compile(r'[a-z][a-z0-9-]{3,}')
STOPWORDS = set('that this with from have your there here been about into then when only they them were what where which would could should while after before command score inventory look examine'.split())


POLICY = '''You play a turn-based parser story to maximize FINAL native score. Use only
supplied observations and prior conversation. The action value is the raw parser
string ("east"), never the action-channel label "command". No tools or lifecycle
commands. This world waits while you think, but each command can consume a turn.

Your special capability is compiling a small CONDITIONAL PROGRAM for code to run
without another model call. Output JSON only, typically under 900 tokens:
{"goal":"reachable scoring prerequisite", "memory":"updated concise working notebook",
 "start":"a", "program":[
   {"id":"a","cmd":"open box","edges":[
      {"has":["is locked","seems to be locked"],"to":"u"}],"ok":"t"},
   {"id":"u","cmd":"unlock box with brass key","ok":"a"},
   {"id":"t","cmd":"take token","ok":"END"}]}
This example describes syntax, not your world. Use observed nouns and evidence.

Semantics:
- Execute cmd once, inspect its exact response, select FIRST matching edge.
- An edge's has list matches if ANY literal substring occurs (case-insensitive).
  Optional lacks list requires ALL those substrings to be absent. No regex.
- Explicit edges can handle anticipated parser errors, blocked actions, and
  clarification questions by jumping to a repair/answer node. Default is END.
- ok is a default successor ONLY when no parser/precondition error, clarification,
  or danger is detected. No implicit fallthrough. END means return to you.
- collect:true allows a query (examine/read/inventory/help) to follow ok even if
  that query fails. Use this to collect INDEPENDENT clues in one call. It never
  bypasses clarification or danger. Do not set collect for a puzzle operation.
- Danger and negative score always return to you. Positive score can continue.
- Interactive help menus and empty responses interrupt the program. If the
  response explicitly advertises Q = resume game, code sends that contextual
  menu answer once before returning to you; it is never sent at a story prompt.
- Maximum 16 nodes, 24 executed turns per call, 2 visits to each node. Identical
  command/result loops are interrupted. Never rely on a loop running forever.
- Raw string nodes are accepted but executed singly, without blind fallthrough.

Choose 4-10 useful commands where evidence supports them. Gather multiple
independent descriptions in a single program; branch to repair common obstacles
only when the repair is supported (e.g. a key actually carried). Guard known
routes with destination text. Unknown movement usually ends for inspection;
never invent a long route. A final novel action or clue naturally ends the program.
Prefer progress through puzzle dependencies over surveying every noun. Seek
reachable light/access/tools, learn exact spell or conversation syntax, and bank
score when observed rules reward it. Track scarce turns, time warnings and unique
items. Do not consume/give away/destroy unique tools without evidence. A room name
is not a unique room ID; maps are directed. Never repeat failed grammar without
new evidence. Repeatedly identical descriptions can still hide timers or RNG.

Memory is your working notebook, not verified truth. Preserve the mission, actual
inventory changes, directed routes, exact clue wording/spells, blocked puzzles,
failed hypotheses, and a next scoring objective. Cite observation IDs for facts;
label guesses. Revise it from actual observations; predicted node results are NOT
facts. You may include recall:[observation IDs] to pin old evidence for next call.
An old splash-screen instruction is superseded by later room text in the SAME
observation. Answer the latest live prompt. The opening text remains available.
'''


def text_of(obs):
    if isinstance(obs, dict):
        return str(obs.get('text', ''))
    return str(obs) if obs is not None else ''


def normalize_obs(obs):
    if isinstance(obs, dict):
        return obs
    if isinstance(obs, str):
        try:
            val = json.loads(obs)
            if isinstance(val, dict):
                return val
        except (ValueError, TypeError):
            pass
    return {'text': text_of(obs)}


def terminal(obs):
    return any(obs.get(k, False) for k in ('terminated', 'truncated', 'won', 'lost'))


def clean_command(value):
    """Unwrap presentation, reject lifecycle/compound input; never truncate."""
    if not isinstance(value, str) or len(value) > 1000:
        return None
    if any(ord(ch) < 32 for ch in value):
        return None
    s = value.strip()
    for _ in range(4):
        old = s
        s = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s+', '', s).strip()
        s = re.sub(r'^>\s*', '', s)
        m = re.fullmatch(r'<(?:command|action)>(.*?)</(?:command|action)>', s, re.I)
        if m:
            s = m.group(1).strip()
        m = re.fullmatch(r'(?:command|action)\s*\((.*?)\)', s, re.I)
        if m:
            s = m.group(1).strip()
        s = re.sub(r'^(?:command|action)\s*(?:[:=|]\s*|\s+)', '', s, flags=re.I).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in '\"\'`':
            s = s[1:-1].strip()
        if s == old:
            break
    if not s or len(s) > 200:
        return None
    if s.split()[0].lower() in RESERVED:
        return None
    if any(ch in s for ch in '\n\r\x00;{}[]') or re.search(r'\bthen\b|\.(?=\S|\s+\S)', s, re.I):
        return None
    return s


def decode_json(raw):
    if not isinstance(raw, str):
        return None
    s = raw.strip()[:50000]
    try:
        return json.loads(s)
    except (ValueError, TypeError, RecursionError):
        pass
    decoder = json.JSONDecoder()
    # Search structured starts only: don't accidentally treat a JSON key as a command.
    for m in list(re.finditer(r'[\[{]', s))[:40]:
        try:
            value, _ = decoder.raw_decode(s[m.start():])
            return value
        except (ValueError, TypeError, RecursionError):
            continue
    return None


def extract_commands(value, depth=0):
    if depth > 8:
        return []
    if isinstance(value, str):
        c = clean_command(value)
        return [c] if c else []
    if isinstance(value, list):
        out = []
        for item in value[:24]:
            out.extend(extract_commands(item, depth+1))
        return out[:24]
    if isinstance(value, dict):
        # Stop at the first effective payload; aliases must not duplicate actions.
        for k in ('cmd', 'commands', 'actions', 'command', 'arguments', 'args', 'input', 'value', 'action', 'plan'):
            if k in value:
                out = extract_commands(value[k], depth+1)
                if out:
                    return out
    return []


def strings(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [s.casefold() for s in value[:8] if isinstance(s, str) and 2 <= len(s) <= 160]


def target(value):
    return str(value)[:40] if isinstance(value, (str, int)) and not isinstance(value, bool) else 'END'


def parse_reply(raw):
    obj = decode_json(raw)
    meta = obj if isinstance(obj, dict) else {}
    source = meta.get('program')
    nodes = {}
    if isinstance(source, list):
        for i, item in enumerate(source[:16]):
            if isinstance(item, str):
                cmd = clean_command(item)
                item = {'id': str(i), 'cmd': cmd}
            if not isinstance(item, dict):
                continue
            commands = extract_commands(item.get('cmd', item.get('command')))
            if not commands:
                continue
            ident = target(item.get('id', str(i)))
            if ident in nodes or ident == 'END':
                continue
            edges = []
            if isinstance(item.get('edges'), list):
                for edge in item['edges'][:5]:
                    if not isinstance(edge, dict):
                        continue
                    has = strings(edge.get('has'))
                    if has:
                        edges.append({'has': has, 'lacks': strings(edge.get('lacks')), 'to': target(edge.get('to'))})
            nodes[ident] = {'cmd': commands[0], 'edges': edges,
                            'ok': target(item.get('ok')),
                            'collect': item.get('collect') is True}
    if not nodes:
        commands = extract_commands(obj)
        if not commands and isinstance(raw, str):
            for line in raw[:4000].splitlines():
                if line.strip().startswith('```'):
                    continue
                if re.search(r'\b(?:because|should|would|recommend|reasoning)\b', line, re.I):
                    continue
                c = clean_command(line)
                if c:
                    commands.append(c)
        # Without branches, execute only one node: no invented success contract.
        if commands:
            nodes['a'] = {'cmd': commands[0], 'edges': [], 'ok': 'END', 'collect': False}
    start = target(meta.get('start'))
    if start not in nodes:
        start = next(iter(nodes), 'END')
    return nodes, start, meta


def signals(command, obs):
    t = text_of(obs)
    return {
        'parser': bool(PARSER.search(t)),
        'blocked': bool(BLOCKED.search(t)) and not bool(QUERY.match(command)),
        'prompt': bool(PROMPT.search(t)),
        'danger': bool(DANGER.search(t)),
    }


def choose_edge(node, obs, delta):
    t = text_of(obs).casefold()
    flags = signals(node['cmd'], obs)
    if flags['danger'] or delta < 0:
        return 'END', 'danger or score loss', flags
    if MENU.search(text_of(obs)):
        return 'END', 'interactive menu', flags
    if not t.strip():
        return 'END', 'empty response; possible character-input mode', flags
    for edge in node['edges']:
        if any(s in t for s in edge['has']) and not any(s in t for s in edge['lacks']):
            return edge['to'], 'literal branch', flags
    if flags['prompt']:
        return 'END', 'unhandled clarification', flags
    if flags['parser'] or flags['blocked']:
        if not (node['collect'] and QUERY.match(node['cmd'])):
            return 'END', 'unhandled failure', flags
    return node['ok'], 'query collection' if node['collect'] else 'ok default', flags


def short(value, limit):
    s = str(value)
    return s if len(s) <= limit else s[:limit] + '\n[truncated; full observation retained]'


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.records = []
        self.notebook = ''
        self.goal = ''
        self.pins = []
        self.failures = []
        self.reason = 'Initial observation. Establish mission and prerequisites.'
        self.latest_inventory = None

    def record(self, command, obs):
        rec = {'id': len(self.records), 'cmd': command, 'text': text_of(obs),
               'score': obs.get('score'), 'moves': obs.get('native_moves'),
               'step': obs.get('step_count')}
        self.records.append(rec)
        if command and command.casefold() in {'inventory', 'i'} and not PARSER.search(rec['text']):
            self.latest_inventory = rec
        return rec

    def evidence(self):
        """Recent exact responses plus old pinned/relevant clues, all timestamped."""
        recent = []
        used = 0
        for rec in reversed(self.records[1:]):
            clipped = dict(rec, text=short(rec['text'], 5000))
            size = len(clipped['text']) + 150
            if recent and used + size > 21000:
                break
            recent.append(clipped)
            used += size
            if len(recent) >= 28:
                break
        recent.reverse()
        recent_ids = {r['id'] for r in recent}
        query = set(WORDS.findall((self.goal + ' ' + self.notebook + ' ' + self.records[-1]['text']).casefold())) - STOPWORDS
        old = []
        for rec in self.records[1:]:
            if rec['id'] in recent_ids:
                continue
            relevance = len(query & set(WORDS.findall((rec['cmd'] + ' ' + rec['text']).casefold())))
            if rec['id'] in self.pins:
                relevance += 1000
            if relevance:
                old.append((relevance, rec['id'], rec))
        old.sort(key=lambda x: (x[0], x[1]), reverse=True)
        recalled = [dict(r, text=short(r['text'], 3000)) for _, _, r in old[:4]]
        return {'recent': recent, 'older_relevant_or_pinned': recalled}

    def prompt(self, obs, remaining):
        packet = {
            'game_id': obs.get('game_id'),
            'score': obs.get('score'), 'max_score': obs.get('max_score'),
            'native_moves': obs.get('native_moves'), 'step_count': obs.get('step_count'),
            'remaining_calls': remaining.get('system2_calls'),
            'remaining_wall_s': round(number(remaining.get('wall_s')), 1),
            'why_control_returned': self.reason, 'prior_goal': self.goal,
            'working_notebook_unverified': self.notebook,
            'initial_observation': short(self.records[0]['text'], 9000),
            'latest_inventory_observation_may_be_stale': self.latest_inventory,
            'recent_failures': self.failures[-12:],
            'evidence': self.evidence(),
            'CURRENT': obs,
        }
        return POLICY + '\nEVIDENCE PACKET:\n' + json.dumps(packet, ensure_ascii=False, default=str)

    def run_episode(self, game):
        ctx = self.ctx
        obs = normalize_obs(ctx.observe())
        self.record('', obs)
        ctx.memory['conditional_policy_transcript'] = self.records
        errors = 0
        malformed = 0
        # Independent calls keep the policy self-contained; no unbounded hidden session.
        for _ in range(40):
            if ctx.done or terminal(obs):
                return
            rem = ctx.budget.remaining()
            wall = number(rem.get('wall_s', 1800))
            if rem.get('system2_calls', 0) <= 0 or wall <= 3:
                return
            try:
                result = ctx.think(self.prompt(obs, rem), session_id=None,
                                   timeout_s=min(240.0, wall - 2.0))
            except Exception as exc:
                ctx.log('policy_model_error', error=type(exc).__name__)
                errors += 1
                if ctx.done or errors >= 3:
                    return
                self.reason = 'Previous model call failed; continue from unchanged current evidence.'
                continue
            errors = 0
            nodes, current, meta = parse_reply(getattr(result, 'text', ''))
            if not nodes:
                malformed += 1
                self.reason = 'Malformed policy: output JSON with program nodes and raw parser cmd values.'
                if malformed >= 3:
                    return
                # A single recovery observation exercises action transport even
                # with arbitrary stand-in output. Never overwrite a live prompt.
                if malformed == 1 and not any(signals('', obs)[k] for k in ('prompt', 'danger')):
                    try:
                        recovered = ctx.act('look')
                        obs = normalize_obs(recovered if recovered is not None else ctx.observe())
                        self.record('look', obs)
                    except Exception as exc:
                        ctx.log('policy_recovery_error', error=type(exc).__name__)
                        if ctx.done:
                            return
                continue
            malformed = 0
            if isinstance(meta.get('memory'), str):
                self.notebook = meta['memory'][:5000]
                ctx.memory['conditional_policy_notebook'] = self.notebook
            if isinstance(meta.get('goal'), str):
                self.goal = meta['goal'][:600]
            if isinstance(meta.get('recall'), list):
                self.pins = [x for x in meta['recall'][:8] if isinstance(x, int) and 0 <= x < len(self.records)]
            ctx.log('policy_start', goal=self.goal, nodes=len(nodes), start=current)
            visits = Counter()
            outcomes = Counter()
            for _turn in range(24):
                if ctx.done or terminal(obs):
                    return
                if current not in nodes:
                    self.reason = 'Program ended; interpret new evidence and advance the scoring objective.'
                    break
                visits[current] += 1
                if visits[current] > 2:
                    self.reason = 'Program loop interrupted: node already executed twice.'
                    break
                node = nodes[current]
                cmd = node['cmd']
                old_score = number(obs.get('score'))
                try:
                    new_obs = ctx.act(cmd)
                    obs = normalize_obs(new_obs if new_obs is not None else ctx.observe())
                except Exception as exc:
                    ctx.log('policy_action_error', command=cmd, error=type(exc).__name__)
                    if ctx.done:
                        return
                    self.reason = 'Action rejected by interface: ' + cmd + ' (' + type(exc).__name__ + ')'
                    self.failures.append({'cmd': cmd, 'error': type(exc).__name__})
                    break
                rec = self.record(cmd, obs)
                if terminal(obs) or ctx.done:
                    return
                if MENU.search(rec['text']):
                    # This is a context-sensitive answer copied from the live
                    # menu instruction, not the story's lifecycle quit command.
                    self.reason = 'Interactive help menu interrupted the program.'
                    if MENU_EXIT.search(rec['text']):
                        try:
                            restored = ctx.act('q')
                            obs = normalize_obs(restored if restored is not None else ctx.observe())
                            self.record('q', obs)
                            self.reason += ' Sent the advertised Q = resume game answer once. Verify parser mode.'
                        except Exception as exc:
                            ctx.log('policy_menu_exit_error', error=type(exc).__name__)
                            self.reason += ' Menu exit was rejected; review the input mode.'
                            if ctx.done:
                                return
                    ctx.log('policy_menu', command=cmd, reason=self.reason)
                    break
                delta = number(obs.get('score')) - old_score
                nxt, why, flags = choose_edge(node, obs, delta)
                if flags['parser'] or flags['blocked'] or flags['prompt']:
                    self.failures.append({'id': rec['id'], 'cmd': cmd, 'text': short(rec['text'], 700)})
                key = (cmd.casefold(), rec['text'], obs.get('score'))
                outcomes[key] += 1
                if outcomes[key] >= 2:
                    nxt, why = 'END', 'identical command and response repeated in this program'
                ctx.log('policy_edge', node=current, command=cmd, next=nxt, reason=why,
                        observation_id=rec['id'], score_delta=delta, signals=flags)
                if nxt not in nodes:
                    self.reason = 'After ' + repr(cmd) + ': ' + why + '. Review latest result.'
                    break
                current = nxt
            else:
                self.reason = '24-turn policy execution cap reached. Review events before continuing.'
            ctx.log('policy_stop', reason=self.reason, actions=sum(visits.values()))


def build(ctx):
    return Harness(ctx)
