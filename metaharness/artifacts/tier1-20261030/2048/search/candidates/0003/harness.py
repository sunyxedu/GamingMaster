"""Verified seed-stream model-predictive control for 2048.

The seed is public observation data, never a scenario lookup. A replay model is
accepted only after matching the opening and is checked after every action.
"""
from __future__ import annotations

import heapq
import json
import random
import re
import time
from collections import deque
from functools import lru_cache

DIRECTIONS = ('up', 'down', 'left', 'right')
LINES = {
    'left': tuple(tuple(4*r+c for c in range(4)) for r in range(4)),
    'right': tuple(tuple(4*r+c for c in range(3,-1,-1)) for r in range(4)),
    'up': tuple(tuple(4*r+c for r in range(4)) for c in range(4)),
    'down': tuple(tuple(4*r+c for r in range(3,-1,-1)) for c in range(4)),
}


def observation(value):
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def flat(obs):
    return tuple(int(v) for row in obs['board'] for v in row)


@lru_cache(maxsize=32768)
def slide(row):
    src = [x for x in row if x]
    dst, gain, i = [], 0, 0
    while i < len(src):
        if i+1 < len(src) and src[i] == src[i+1]:
            dst.append(src[i]*2)
            gain += src[i]*2
            i += 2
        else:
            dst.append(src[i])
            i += 1
    return tuple(dst + [0]*(4-len(dst))), gain


def transition(board, action):
    out, gain = list(board), 0
    for line in LINES[action]:
        vals, g = slide(tuple(board[i] for i in line))
        gain += g
        for i, val in zip(line, vals):
            out[i] = val
    return tuple(out), gain


class SeedStream:
    """Compact branchable representation of random.Random's public operations.

    random() uses 27 + 26 leading bits; choice uses rejection-sampled
    getrandbits(n.bit_length()). Every branch stores only a word offset.
    """
    def __init__(self, seed):
        self.generator = random.Random(seed)
        self.words = []

    def word(self, pos):
        while pos >= len(self.words):
            self.words.extend(self.generator.getrandbits(32) for _ in range(1024))
        return self.words[pos]

    def spawn(self, board, pos):
        empty = [i for i, value in enumerate(board) if not value]
        if not empty:
            return board, pos
        n = len(empty)
        bits = n.bit_length()
        while True:
            index = self.word(pos) >> (32-bits)
            pos += 1
            if index < n:
                break
        u = ((self.word(pos) >> 5)*67108864 + (self.word(pos+1) >> 6))/9007199254740992
        pos += 2
        out = list(board)
        out[empty[index]] = 4 if u < .1 else 2
        return tuple(out), pos


def features(board):
    ranks = [v.bit_length()-1 if v else 0 for v in board]
    empty = board.count(0)
    mono = smooth = pairs = 0
    for direction in ('up', 'left'):
        for line in LINES[direction]:
            vals = [ranks[i] for i in line if ranks[i]]
            inc = dec = 0
            for a,b in zip(vals, vals[1:]):
                delta = a*a*a-b*b*b
                inc += max(0, delta)
                dec += max(0, -delta)
                smooth += abs(a-b)
                pairs += int(a == b)
            mono += min(inc, dec)
    return empty, mono, smooth, pairs


def potential(board, mode='balanced'):
    empty, mono, smooth, pairs = features(board)
    # Rotation-invariant layout; no permanent corner that can become a trap.
    w_empty = {'balanced': 65, 'survive': 115, 'cashout': 30}[mode]
    return w_empty*empty - 1.2*mono - 2*smooth + 22*pairs


def rule_error(before, after, action):
    """Verify a slide plus exactly one legal spawn and the merge-score delta."""
    b, a = flat(before), flat(after)
    moved, gain = transition(b, action)
    if moved == b:
        return 'selected action was locally a no-op'
    diffs = [i for i in range(16) if moved[i] != a[i]]
    if len(diffs) != 1 or moved[diffs[0]] != 0 or a[diffs[0]] not in (2,4):
        return 'observed board is not predicted slide plus one 2/4 spawn'
    if after.get('score',0)-before.get('score',0) != gain:
        return 'merge-score delta disagrees with exact slide'
    if after.get('moves',0) != before.get('moves',0)+1:
        return 'changed action did not advance move counter by one'
    spawn = after.get('spawn')
    if isinstance(spawn,dict):
        i = 4*spawn.get('row',-10)+spawn.get('col',-10)
        if i != diffs[0] or spawn.get('value') != a[diffs[0]]:
            return 'spawn metadata disagrees with board delta'
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.stream = None
        self.cursor = 0
        self.mode = 'balanced'
        self.depth = 18
        self.width = 64
        self.session = None
        self.calls = 0
        self.last_review = -100
        self.history = deque(maxlen=8)
        self.rules_ok = True
        self.last_illegal = 0
        self.rejected = set()
        self.failures = 0

    def review(self, obs, reason, candidates=None):
        ctx = self.ctx
        remaining = ctx.budget.remaining()
        if ctx.done or remaining.get('system2_calls',0) <= 1 or remaining.get('wall_s',0) < 220:
            return None
        if self.calls >= 8:
            return None
        self.calls += 1
        self.last_review = obs.get('moves',0)
        prompt = (
            'Supervise a 2048 receding-horizon search. Objective: merge score + max tile. '
            'The public default horizon is 200 changed moves. Exact local rules are checked '
            'after EVERY action. The documented public random.Random(seed) stream is used '
            'only while its predictions match. Search has no fixed corner and replans each turn. '
            'Choose mode balanced (space/order/score), survive (more space), or cashout '
            '(score emphasis); depth 12,18,24. If rules mismatch, supply one legal action '
            'for an observed-state probe. No arithmetic or long explanation needed. '
            'Reply one JSON object: {"mode":"balanced","depth":18,"action":"left",'
            '"reason":"short diagnosis"}. action is executed only for recovery; '
            'normal moves come from verified local search.\n' + json.dumps({
                'trigger':reason, 'observation':obs, 'progress':ctx.progress(),
                'seed_model_verified':self.stream is not None, 'rules_verified':self.rules_ok,
                'mode':self.mode, 'recent_verified_actions':list(self.history),
                'candidate_search':candidates,
            },separators=(',',':')))
        action = None
        try:
            result = ctx.think(prompt,session_id=self.session,timeout_s=180.0)
            self.session = getattr(result,'session_id',None)
            text = getattr(result,'text','') or ''
            match = re.search(r'\{[^{}]*\}',text)
            data = json.loads(match.group()) if match else {}
            if isinstance(data,dict):
                if data.get('mode') in ('balanced','survive','cashout'):
                    self.mode = data['mode']
                if data.get('depth') in (12,18,24):
                    self.depth = int(data['depth'])
                if data.get('action') in obs.get('legal_actions',[]):
                    action = data['action']
            ctx.log('supervisor',trigger=reason,mode=self.mode,depth=self.depth,reply=text[:400])
        except Exception as exc:
            # Includes slow calls, unavailable model, budget end, and malformed JSON.
            ctx.log('supervisor_unavailable',error=type(exc).__name__,trigger=reason)
        ctx.memory['search_policy'] = {'mode':self.mode,'depth':self.depth}
        return action

    def search(self, obs, legal):
        """Beam over exact future spawns, preserving all root actions separately."""
        board = flat(obs)
        # Public adaptor default; beyond it simply resume a rolling horizon.
        remaining = 200-obs.get('moves',0)
        horizon = min(self.depth,remaining) if remaining > 0 else self.depth
        time_left = self.ctx.budget.remaining().get('wall_s',0)
        stop = time.monotonic()+min(2.0,max(.02,time_left*.01))
        beams = {action:[(board,self.cursor,0,())] for action in legal}
        reports = {}
        cache = {}
        final = None
        for depth in range(horizon):
            if self.ctx.done or time.monotonic() >= stop:
                break
            current = {}
            # Full root rounds: avoid favoring an action because its search ran longer.
            for root, beam in beams.items():
                pool = {}
                for b,pos,gain,path in beam:
                    actions = (root,) if depth == 0 else DIRECTIONS
                    for action in actions:
                        moved,g = transition(b,action)
                        if moved == b:
                            continue
                        next_board,next_pos = self.stream.spawn(moved,pos)
                        key = (next_board,next_pos)
                        total = gain+g
                        if key in pool and pool[key][2] >= total:
                            continue
                        pool[key] = (next_board,next_pos,total,path+(action,))
                if not pool:
                    current[root] = []
                    continue
                scale = min(1.0,max(0.0,(remaining-depth-1)/12)) if remaining > 0 else 1.0
                def value(node):
                    b,_,gain,_ = node
                    if b not in cache:
                        cache[b] = potential(b,self.mode)
                    return gain + max(b) + scale*cache[b]
                keep = heapq.nlargest(self.width,pool.values(),key=value)
                current[root] = keep
                best = keep[0]
                reports[root] = {'value':round(value(best),2),'gain':best[2],
                                 'max_tile':max(best[0]),'depth':depth+1,
                                 'plan':list(best[3][:6])}
            alive = {a:b for a,b in current.items() if b}
            if not alive:
                break
            beams = alive
            # Terminal branches are excluded while another action survives deeper.
            final = max(alive,key=lambda a: reports[a]['value'])
        if final is None:
            final = max(legal,key=lambda a: potential(transition(board,a)[0],self.mode)+transition(board,a)[1])
        return final,reports

    def uncertain_search(self, board, legal):
        """Two-ply expectimax fallback: all possible 2/4 spawns, no seed reliance."""
        cache = {}
        def leaf(b):
            if b not in cache:
                cache[b] = potential(b,self.mode)+max(b)
            return cache[b]
        def next_best(b):
            vals = [g+leaf(n) for a in DIRECTIONS for n,g in [transition(b,a)] if n != b]
            return max(vals) if vals else leaf(b)-10000
        scores = {}
        for action in legal:
            b,g = transition(board,action)
            empty = [i for i,v in enumerate(b) if not v]
            total = 0.0
            for i in empty:
                for value,prob in ((2,.9),(4,.1)):
                    child = list(b)
                    child[i] = value
                    total += prob*next_best(tuple(child))
            scores[action] = g + (total/len(empty) if empty else next_best(b))
        return max(scores,key=scores.get),scores

    def run_episode(self, game):
        ctx = self.ctx
        obs = observation(ctx.observe())
        if not obs or 'board' not in obs:
            return
        if obs.get('moves',0) == 0 and obs.get('seed') is not None:
            try:
                stream = SeedStream(obs['seed'])
                board,pos = stream.spawn((0,)*16,0)
                board,pos = stream.spawn(board,pos)
                if board == flat(obs):
                    self.stream,self.cursor = stream,pos
            except (ValueError,TypeError):
                pass
        ctx.log('opening_model_check',seed_stream_match=self.stream is not None)
        self.review(obs,'opening: choose search objective, replay model opening check complete')
        while not ctx.done and not obs.get('game_over',False):
            board = flat(obs)
            legal = [a for a in obs.get('legal_actions',DIRECTIONS) if a in DIRECTIONS and a not in self.rejected]
            local = [a for a in legal if transition(board,a)[0] != board]
            if not legal:
                break
            if self.rules_ok and not local:
                self.rules_ok = False
                self.stream = None
            legal = local if self.rules_ok else legal
            if not self.rules_ok:
                action = self.review(obs,'rule model invalid: choose observed legal probe') or legal[0]
                reports = {}
            elif self.stream is not None:
                action,reports = self.search(obs,legal)
            else:
                action,reports = self.uncertain_search(board,legal)
            if ctx.done:
                break
            before = obs
            predicted = next_pos = None
            if self.stream is not None:
                moved,_ = transition(board,action)
                predicted,next_pos = self.stream.spawn(moved,self.cursor)
            try:
                nxt = ctx.act(action)
                obs = observation(nxt if nxt is not None else ctx.observe())
            except Exception as exc:
                if ctx.done:
                    break
                self.rejected.add(action)
                self.failures += 1
                self.review(obs,'action rejected: '+type(exc).__name__,reports)
                if self.failures >= 4:
                    break
                continue
            self.rejected.clear()
            progress = ctx.progress()
            error = rule_error(before,obs,action)
            seed_error = predicted is not None and predicted != flat(obs)
            if not error and self.rules_ok and next_pos is not None:
                self.cursor = next_pos
            if error:
                self.rules_ok = False
                self.stream = None
            elif seed_error:
                self.stream = None
            elif not self.rules_ok:
                # A successful probe re-enables only the independently checked rules.
                self.rules_ok = True
            gain = obs.get('score',0)-before.get('score',0)
            self.history.append({'move':obs.get('moves'),'action':action,'gain':gain,
                                 'empty':flat(obs).count(0),'rule_error':error,'seed_error':seed_error})
            ctx.log('verified_step',move=obs.get('moves'),action=action,gain=gain,
                    rule_error=error,seed_error=seed_error,seed_search=self.stream is not None,
                    search=reports)
            if ctx.done or obs.get('game_over',False):
                break
            reasons = []
            if error: reasons.append(error)
            if seed_error: reasons.append('predicted seeded spawn failed; using full spawn expectation')
            if progress.get('since_obs_change',0) >= 1: reasons.append('observation unchanged')
            if progress.get('illegal_actions',0) > self.last_illegal: reasons.append('new rejected action')
            self.last_illegal = progress.get('illegal_actions',0)
            if progress.get('since_score_change',0) >= 10: reasons.append('10 actions without score progress')
            if flat(obs).count(0) <= 2: reasons.append('space collapsed to at most two empty cells')
            urgent = bool(error or seed_error or progress.get('since_obs_change',0))
            if reasons and (urgent or obs.get('moves',0)-self.last_review >= 20):
                self.review(obs,'; '.join(reasons),reports)


def build(ctx):
    return Harness(ctx)
