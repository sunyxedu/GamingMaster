"""Full-horizon resource-acquisition search with a verified terminal incumbent.

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


@lru_cache(maxsize=65536)
def row_shape(row):
    ranks = [v.bit_length()-1 for v in row if v]
    up = down = pairs = 0
    for a, b in zip(ranks, ranks[1:]):
        d = a*a*a - b*b*b
        up += max(0, d)
        down += max(0, -d)
        pairs += a == b
    return min(up, down), pairs


@lru_cache(maxsize=65536)
def shape(board):
    mono = pairs = 0
    for direction in ('up', 'left'):
        for line in LINES[direction]:
            m, p = row_shape(tuple(board[i] for i in line))
            mono += m
            pairs += p
    return 65*board.count(0) - 1.2*mono + 22*pairs


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.stream = None
        self.cursor = 0
        self.rules_ok = True
        self.good_probes = 0
        self.plan = []
        self.commit = 0
        self.future = None
        self.session = None
        self.calls = 0
        self.last_review = -100
        self.last_illegal = 0
        self.mode = 'acquire'
        self.override_until = -1
        self.history = deque(maxlen=12)
        self.rejected = set()
        self.report = {}
        self.probe = None
        self.fours = 0

    def review(self, obs, reason):
        rem = self.ctx.budget.remaining()
        if self.future is not None or self.calls >= 5 or self.ctx.done:
            return
        if rem.get('system2_calls', 0) <= 1 or rem.get('wall_s', 0) < 210:
            return
        self.last_review = obs.get('moves', 0)
        self.calls += 1
        prompt = (
            'Supervise a verified 2048 resource-acquisition planner. Objective is final merge '
            'score + max tile at the documented default of 200 changed moves. A merge conserves '
            'tile mass (sum of cells); a new 4 adds twice the mass of a 2. Normal 200-move '
            'trajectories have only about 444 mass, so 512 is impossible even with perfect '
            'merging. The PUBLIC seed/RNG replay is verified every step: different legal '
            'moves change empty-cell counts, hence rejection-sampled RNG consumption and '
            'future tile values. Full-horizon beams acquire extra mass, then merge it; only a '
            'higher exact final score can replace the verified incumbent. Choose acquire (seek extra 4s while maintaining '
            'space), consolidate (prioritize merge score/order), or space (repair congestion) '
            'for the next 16 moves. Diagnose whether missing mass or stranded equal tiles is '
            'the bottleneck. A mode is guidance, not a stale action sequence. For a rule '
            'disagreement supply one observed-legal probe. Reply ONLY short JSON '
            '{"mode":"acquire","action":"left","reason":"brief"}.\n' + json.dumps({
                'reason': reason, 'observation': obs, 'progress': self.ctx.progress(),
                'rules_ok': self.rules_ok, 'replay_ok': self.stream is not None,
                'mass': sum(flat(obs)), 'fours_since_start': self.fours,
                'search': self.report, 'recent': list(self.history),
            }, separators=(',', ':')))
        session = self.session
        def ask():
            try:
                return self.ctx.think(prompt, session_id=session, timeout_s=180.0)
            except Exception:
                return None
        try:
            self.future = self.ctx.spawn(ask)
            self.ctx.log('resource_review_requested', move=obs.get('moves'), reason=reason)
        except Exception:
            self.future = None

    def poll(self, obs):
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        try:
            result = future.result()
            if result is None:
                return
            self.session = getattr(result, 'session_id', self.session)
            reply = getattr(result, 'text', '') or ''
            match = re.search(r'\{[^{}]*\}', reply)
            data = json.loads(match.group()) if match else {}
            if not isinstance(data, dict):
                return
            if data.get('mode') in ('acquire', 'consolidate', 'space'):
                self.mode = data['mode']
                self.override_until = obs.get('moves', 0)+16
                self.commit = 0
            action = data.get('action')
            if action in obs.get('legal_actions', []):
                self.probe = action
            self.ctx.memory['resource_policy'] = {'mode': self.mode, 'until': self.override_until}
            self.ctx.log('resource_review_applied', move=obs.get('moves'), mode=self.mode,
                         reply=reply[:500])
        except Exception:
            pass

    def stopped(self, stop):
        return time.monotonic() >= stop or self.ctx.done

    @staticmethod
    def objective(path):
        if not path:
            return (-1, -1)
        return (sum(step[3] for step in path)+max(path[-1][1]), len(path))

    def complete(self, board, cursor, horizon, prefix, stop):
        """A cheap feasible incumbent; never used as evidence of optimality."""
        path = list(prefix)
        if path:
            board, cursor = path[-1][1:3]
        while len(path) < horizon:
            if self.stopped(stop):
                return None
            options = []
            for action in DIRECTIONS:
                moved, gain = transition(board, action)
                if moved == board:
                    continue
                child, pos = self.stream.spawn(moved, cursor)
                options.append((shape(child)+.4*gain, action, child, pos, gain))
            if not options:
                break
            _, action, board, cursor, gain = max(options, key=lambda x: x[0])
            path.append((action, board, cursor, gain))
        return path

    def search(self, obs, legal):
        """Reach the terminal boundary before comparing resource proposals."""
        b0 = flat(obs)
        move = int(obs.get('moves', 0))
        remaining = 200-move if move < 200 else 32
        mass0 = sum(b0)
        wall = self.ctx.budget.remaining().get('wall_s', 0)
        # Reserve most remaining wall time for later replans and recovery.
        seconds = min(28., max(.03, (wall-30)*8/max(8, remaining)*.55))
        stop = time.monotonic()+seconds
        if move >= self.override_until:
            self.mode = 'acquire'
            if b0.count(0) <= 2:
                self.mode = 'space'
        incumbent = list(self.plan) if self.plan and self.plan[0][0] in legal else []
        if not incumbent:
            initial = self.complete(b0, self.cursor, remaining, (), stop)
            if initial and initial[0][0] in legal:
                incumbent = initial
        best = self.objective(incumbent)
        # A small complete beam is deliberately preferable to a very wide
        # search stopped at a heuristic leaf. Work limits keep it anytime.
        width = 192
        beam = [(b0, self.cursor, 0, (), mass0)]
        completed = 0
        injected_gain = 0
        fallback_prefix = []
        for depth in range(1, remaining+1):
            if self.stopped(stop):
                break
            pool = {}
            interrupted = False
            fade = min(1., (remaining-depth)/24.)
            weight = {'acquire': 85., 'consolidate': 10., 'space': 20.}[self.mode]*fade
            def value(node):
                board, _, gain, _, mass = node
                layout = shape(board)
                if self.mode == 'space':
                    layout += 70*board.count(0)
                return gain+max(board)+fade*layout+weight*(mass-mass0-2*depth)
            for index, node in enumerate(beam):
                if index % 16 == 0 and self.stopped(stop):
                    interrupted = True
                    break
                board, pos, gain, path, mass = node
                actions = legal if depth == 1 else DIRECTIONS
                for action in actions:
                    moved, g = transition(board, action)
                    if moved == board:
                        continue
                    child, cursor = self.stream.spawn(moved, pos)
                    key = (child, cursor)
                    if key in pool and pool[key][2] >= gain+g:
                        continue
                    step = (action, child, cursor, g)
                    pool[key] = (child, cursor, gain+g, path+(step,), sum(child))
            if interrupted or not pool:
                break
            # The incumbent is a protected feasible path even if its mass
            # is temporarily less than an acquisition branch's mass.
            protected = None
            if depth <= len(incumbent):
                action, board, pos, gain = incumbent[depth-1]
                injected_gain += gain
                protected = (board, pos, injected_gain, tuple(incumbent[:depth]), sum(board))
                pool[(board, pos)] = protected
            nodes = list(pool.values())
            ranked = sorted(nodes, key=value, reverse=True)
            buckets = {}
            for node in ranked:
                buckets.setdefault(node[0].count(0), []).append(node)
            reserve = max(1, width//max(1, 2*len(buckets)))
            kept = {}
            if protected:
                kept[(protected[0], protected[1])] = protected
            for bucket in buckets.values():
                signatures = {}
                accepted = 0
                for node in bucket:
                    signature = (node[1], node[4])
                    if signatures.get(signature, 0) >= 2:
                        continue
                    kept[(node[0], node[1])] = node
                    signatures[signature] = signatures.get(signature, 0)+1
                    accepted += 1
                    if accepted >= reserve:
                        break
            for node in ranked:
                if len(kept) >= width:
                    break
                kept[(node[0], node[1])] = node
            beam = list(kept.values())
            fallback_prefix = list(ranked[0][3])
            completed = depth
            if depth == remaining:
                for node in beam:
                    candidate = list(node[3])
                    utility = (node[2]+max(node[0]), len(candidate))
                    if utility > best:
                        incumbent, best = candidate, utility
        # Only a completed rollout may challenge the terminal guarantee.
        if completed < remaining and fallback_prefix and not self.stopped(stop):
            candidate = self.complete(b0, self.cursor, remaining, fallback_prefix, stop)
            if candidate and self.objective(candidate) > best:
                incumbent, best = candidate, self.objective(candidate)
        self.plan = incumbent
        self.commit = min(8, len(incumbent))
        if incumbent:
            leaf = incumbent[-1][1]
            self.report = {'depth': completed, 'horizon': remaining, 'mode': self.mode,
                           'terminal_complete': len(incumbent) == remaining,
                           'forecast': obs.get('score', 0)+best[0],
                           'mass_now': mass0, 'terminal_mass': sum(leaf),
                           'leaf_max': max(leaf), 'leaf_empty': leaf.count(0),
                           'plan': [step[0] for step in incumbent[:8]]}
            self.ctx.log('terminal_resource_search', move=move, **self.report)
            return incumbent[0][0]
        return self.fallback(b0, legal)

    def fallback(self, board, legal):
        """All-spawn expectation when the public seed hypothesis is falsified."""
        scores = {}
        for action in legal:
            moved, gain = transition(board, action)
            if moved == board:
                continue
            empty = [i for i, v in enumerate(moved) if not v]
            total = 0.
            for i in empty:
                for tile, prob in ((2, .9), (4, .1)):
                    child = list(moved)
                    child[i] = tile
                    child = tuple(child)
                    options = [g+shape(n)+max(n) for a in DIRECTIONS
                               for n, g in [transition(child, a)] if n != child]
                    total += prob*(max(options) if options else -10000)
            scores[action] = gain+total/max(1, len(empty))
        return max(scores, key=scores.get) if scores else legal[0]

    def run_episode(self, game):
        ctx = self.ctx
        obs = observation(ctx.observe())
        if 'board' not in obs:
            return
        try:
            if obs.get('moves', 0) == 0 and obs.get('seed') is not None:
                stream = SeedStream(obs['seed'])
                board, pos = stream.spawn((0,)*16, 0)
                board, pos = stream.spawn(board, pos)
                if board == flat(obs):
                    self.stream, self.cursor = stream, pos
        except (ValueError, TypeError):
            pass
        ctx.log('opening_model_check', seed_stream_match=self.stream is not None)
        while not ctx.done and not obs.get('game_over', False):
            self.poll(obs)
            board = flat(obs)
            legal = [a for a in obs.get('legal_actions', DIRECTIONS)
                     if a in DIRECTIONS and a not in self.rejected]
            if not legal:
                break
            local = [a for a in legal if transition(board, a)[0] != board]
            if self.rules_ok and not local:
                self.rules_ok = False
                self.stream = None
                self.plan = []
                self.review(obs, 'local simulator disagrees with observed legal actions')
            if not self.rules_ok:
                action = self.probe if self.probe in legal else legal[0]
                self.probe = None
            elif self.stream is None:
                action = self.fallback(board, local)
            elif self.commit > 0 and self.plan and self.plan[0][0] in local:
                action = self.plan[0][0]
            else:
                action = self.search(obs, local)
            if ctx.done:
                break
            predicted = next_pos = None
            if self.stream is not None:
                moved, _ = transition(board, action)
                predicted, next_pos = self.stream.spawn(moved, self.cursor)
            before = obs
            try:
                nxt = ctx.act(action)
                obs = observation(nxt if nxt is not None else ctx.observe())
            except Exception as exc:
                if ctx.done:
                    break
                self.rejected.add(action)
                self.plan = []
                self.commit = 0
                self.review(obs, 'rejected action: '+type(exc).__name__)
                continue
            self.rejected.clear()
            progress = ctx.progress()
            error = rule_error(before, obs, action)
            new_board = flat(obs)
            seed_error = predicted is not None and predicted != new_board
            mass_gain = sum(new_board)-sum(board)
            if not error and mass_gain not in (2, 4):
                error = 'mass conservation failed'
            if error:
                self.rules_ok = False
                self.good_probes = 0
                self.stream = None
            elif not self.rules_ok:
                self.good_probes += 1
                self.rules_ok = self.good_probes >= 2
            if seed_error:
                self.stream = None
            if self.stream is not None:
                self.cursor = next_pos
            if self.plan and not error and not seed_error and self.plan[0][1] == new_board:
                self.plan.pop(0)
                self.commit = max(0, self.commit-1)
            else:
                self.plan = []
                self.commit = 0
            if mass_gain == 4:
                self.fours += 1
            move = obs.get('moves', 0)
            self.history.append({'move': move, 'action': action,
                                 'gain': obs.get('score', 0)-before.get('score', 0),
                                 'mass_gain': mass_gain, 'empty': new_board.count(0),
                                 'rule_error': error, 'seed_error': seed_error})
            ctx.log('verified_resource_step', move=move, action=action, mass=sum(new_board),
                    fours=self.fours, mode=self.mode, rule_error=error,
                    seed_error=seed_error, search=self.report)
            if ctx.done or obs.get('game_over', False):
                break
            reasons = []
            if error: reasons.append(error)
            if seed_error: reasons.append('seed prediction failed; mass acquisition disabled')
            if progress.get('since_obs_change', 0) >= 1: reasons.append('observation unchanged')
            if progress.get('illegal_actions', 0) > self.last_illegal: reasons.append('new rejected action')
            self.last_illegal = progress.get('illegal_actions', 0)
            if progress.get('since_score_change', 0) >= 8: reasons.append('eight scoreless actions')
            if new_board.count(0) <= 2: reasons.append('space collapsed')
            if self.plan and len(self.plan) < max(1, 200-move):
                reasons.append('terminal plan predicts premature death')
            if move in (40, 100, 160):
                if sum(new_board)+2.6*max(0, 190-move) < 512:
                    reasons.append('mass trajectory misses 512 before consolidation window')
            urgent = bool(error or seed_error or progress.get('since_obs_change', 0))
            if reasons and (urgent or move-self.last_review >= 24):
                self.commit = 0
                self.review(obs, '; '.join(reasons))
            elif move == 32:
                self.review(obs, 'audit complete acquisition/consolidation forecast')


def build(ctx):
    return Harness(ctx)
