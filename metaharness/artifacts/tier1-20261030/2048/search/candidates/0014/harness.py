"""Verified full-horizon action beam for finite-horizon 2048.

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


# A trajectory stores (action, board after spawn, RNG cursor, cumulative gain).
# Unlike a leaf heuristic, its objective is the actual score at the move cap.
SNAKES = (
    (0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12),
    (3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15),
    (12,13,14,15,11,10,9,8,4,5,6,7,3,2,1,0),
    (15,14,13,12,8,9,10,11,7,6,5,4,0,1,2,3),
)


@lru_cache(maxsize=65536)
def row_features(row):
    ranks = [v.bit_length()-1 for v in row if v]
    up = down = pairs = 0
    for a,b in zip(ranks,ranks[1:]):
        d = a*a*a-b*b*b
        up += max(0,d)
        down += max(0,-d)
        pairs += a == b
    return row.count(0), min(up,down), pairs


@lru_cache(maxsize=65536)
def layout(board):
    empty = mono = pairs = 0
    for direction in ('up','left'):
        for line in LINES[direction]:
            e,m,p = row_features(tuple(board[i] for i in line))
            empty += e
            mono += m
            pairs += p
    return 140*empty - 2.5*mono + 45*pairs


def greedy_value(board, gain, policy):
    value = layout(board)
    if policy < 4:
        order = SNAKES[policy]
        # Fixed geometry is a rollout generator, never the selection objective.
        snake = sum(board[i]*(0.72**j) for j,i in enumerate(order))
        return value + 3*snake + .3*gain
    if policy == 4:
        return value + 2*gain
    return value + .1*gain + board.count(0)*100


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.stream = None
        self.cursor = 0
        self.plan = []
        self.plan_score = -1
        self.rng = random.Random(90210)  # proposal RNG, unrelated to episode seed
        self.preferred_policy = 4
        self.rules_ok = True
        self.future = None
        self.session = None
        self.calls = 0
        self.last_review = -100
        self.last_illegal = 0
        self.history = deque(maxlen=8)
        self.rejected = set()
        self.good_probes = 0
        self.search_trials = 0
        self.last_forecast = None
        self.next_search = 0
        self.plateau_since = 0
        self.best_forecast = -1
        self.beam_report = {}

    def review(self, obs, reason):
        """Queue a diagnosis without allowing model queue time to freeze the game."""
        self.poll_review(obs)
        rem = self.ctx.budget.remaining()
        if self.future is not None or self.ctx.done or self.calls >= 6:
            return
        if rem.get('system2_calls',0) <= 1 or rem.get('wall_s',0) < 210:
            return
        if getattr(self.ctx,'system2',True) is None:
            return
        self.calls += 1
        self.last_review = obs.get('moves',0)
        prompt = (
            'Supervise a 2048 planner. Exact slides and public seed replay are verified '
            'after every action. It improves complete candidate trajectories through the '
            'documented 200-move cap, ranks by actual predicted final merge score + max tile, '
            'and retains the verified suffix. Unlike greedy trajectory mutation, a beam expands every legal action at each depth to the move cap. Select the geometry used to preserve diverse beam nodes and finish interrupted searches: '
            'policy 0=top-left,1=top-right,2=bottom-left,3=bottom-right '
            'corner snake;4=flexible merging;5=space preservation. This is an asynchronous '
            'review; the board may advance while you respond. Choose a durable strategy, '
            'not a stale sequence. If rule verification fails, choose a single observed-legal '
            'probe action. Reply only JSON {"policy":4,"action":"left","reason":"brief"}.\n'
            + json.dumps({'reason':reason,'observation':obs,'progress':self.ctx.progress(),
                          'rules_ok':self.rules_ok,'seed_verified':self.stream is not None,
                          'forecast':self.last_forecast,'beam':self.beam_report,'recent':list(self.history),
                          'policy':self.preferred_policy},separators=(',',':')))
        session = self.session
        def ask():
            try:
                return self.ctx.think(prompt,session_id=session,timeout_s=180.0)
            except Exception:
                return None
        try:
            self.future = self.ctx.spawn(ask)
            self.ctx.log('review_requested',reason=reason,move=obs.get('moves'))
        except Exception:
            self.future = None

    def poll_review(self, obs):
        if self.future is None or not self.future.done():
            return None
        future,self.future = self.future,None
        try:
            result = future.result()
            if result is None:
                return None
            self.session = getattr(result,'session_id',self.session)
            reply = getattr(result,'text','') or ''
            match = re.search(r'\{[^{}]*\}',reply)
            data = json.loads(match.group()) if match else {}
            if not isinstance(data,dict):
                return None
            policy = data.get('policy')
            if type(policy) is int and 0 <= policy <= 5:
                self.preferred_policy = policy
                self.next_search = 0
            self.ctx.memory['trajectory_policy'] = self.preferred_policy
            self.ctx.log('review_applied',move=obs.get('moves'),policy=self.preferred_policy,
                         reply=reply[:400])
            action = data.get('action')
            return action if action in obs.get('legal_actions',[]) else None
        except Exception:
            return None

    def stopped(self, stop):
        return time.monotonic() >= stop or self.ctx.done

    def rollout(self, board, cursor, horizon, policy, prefix, stop, mutate=0):
        path = list(prefix)
        gain = path[-1][3] if path else 0
        if path:
            board,cursor = path[-1][1:3]
        while len(path) < horizon:
            if self.stopped(stop):
                return None
            options = []
            for action in DIRECTIONS:
                moved,g = transition(board,action)
                if moved != board:
                    child,pos = self.stream.spawn(moved,cursor)
                    options.append((greedy_value(child,g,policy),action,child,pos,g))
            if not options:
                break
            if mutate > 0:
                picked = self.rng.choice(options)
                mutate -= 1
            else:
                picked = max(options,key=lambda x:x[0])
            _,action,board,cursor,g = picked
            gain += g
            path.append((action,board,cursor,gain))
        return path

    @staticmethod
    def objective(path):
        if not path:
            return (-1,-1)
        # Exact terminal utility; geometry has no vote in accepting a plan.
        return (path[-1][3]+max(path[-1][1]),len(path))

    def optimize(self, obs, legal):
        """Expand actions through the actual cutoff; keep a complete safety suffix.

        Every layer is equal-depth. Geometry only prunes search; terminal plans
        are accepted by exact final utility. An interrupted layer is discarded.
        """
        move = int(obs.get('moves', 0))
        if self.plan and self.plan[0][0] in legal and move < self.next_search:
            return self.plan[0][0]
        board = flat(obs)
        horizon = 200-move if move < 200 else 32
        incumbent = self.plan if self.plan and self.plan[0][0] in legal else []
        old_objective = self.objective(incumbent)
        wall = self.ctx.budget.remaining().get('wall_s', 0)
        # Leave substantial room for execution, fallback and optional model work.
        allotment = min(35., max(.05, (wall-30)*.62/max(1, (horizon+7)//8)))
        stop = time.monotonic()+allotment
        beam_stop = stop-min(1.5, allotment*.15)
        # Always obtain a complete baseline before the expensive beam.
        if not incumbent:
            for policy in (self.preferred_policy, 0, 1, 2, 3, 5):
                path = self.rollout(board, self.cursor, horizon, policy, [], stop)
                if path and self.objective(path) > self.objective(incumbent):
                    incumbent = path
                if self.stopped(beam_stop):
                    break
        best = self.objective(incumbent)
        # node: board, stream cursor, cumulative merge gain, complete prefix.
        beam = [(board, self.cursor, 0, ())]
        width = 512
        completed = 0
        expanded = 0
        terminal_paths = []
        for depth in range(1, horizon+1):
            if self.stopped(beam_stop):
                break
            pool = {}
            interrupted = False
            for index, (b, pos, gain, path) in enumerate(beam):
                if index % 24 == 0 and self.stopped(beam_stop):
                    interrupted = True
                    break
                any_move = False
                for action in legal if depth == 1 else DIRECTIONS:
                    moved, g = transition(b, action)
                    if moved == b:
                        continue
                    any_move = True
                    child, cursor = self.stream.spawn(moved, pos)
                    score = gain+g
                    key = (child, cursor)
                    previous = pool.get(key)
                    if previous is None or previous[2] < score:
                        step = (action, child, cursor, score)
                        pool[key] = (child, cursor, score, path+(step,))
                    expanded += 1
                if not any_move and path:
                    terminal_paths.append(path)
            if interrupted or not pool:
                break
            # Preserve the already verified complete plan at every layer. A
            # shape-pruned alternative can never erase its terminal guarantee.
            if len(incumbent) >= depth:
                action, child, cursor, score = incumbent[depth-1]
                key = (child, cursor)
                if key not in pool or pool[key][2] < score:
                    pool[key] = (child, cursor, score, tuple(incumbent[:depth]))
            nodes = list(pool.values())
            completed = depth
            if depth == horizon:
                winner = max(nodes, key=lambda n: (n[2]+max(n[0]), len(n[3])))
                candidate = list(winner[3])
                if self.objective(candidate) > best:
                    incumbent, best = candidate, self.objective(candidate)
                beam = [winner]
                break
            # Multiple geometry lenses retain temporarily awkward setups. This
            # is action branching, not six greedy continuation policies.
            fade = min(1., (horizon-depth)/18.)
            qualities = {}
            for n in nodes:
                b, pos, gain, path = n
                base = gain+max(b)
                shape = layout(b)*.24
                qualities[(b, pos)] = (base, shape)
            kept = {}
            lenses = [self.preferred_policy, 0, 1, 2, 3, 4, 5]
            quota = max(1, width//len(lenses))
            for policy in lenses:
                def rank(n):
                    b, pos, gain, path = n
                    base, shape = qualities[(b, pos)]
                    if policy < 4:
                        snake = sum(b[i]*(.72**j) for j, i in enumerate(SNAKES[policy]))
                        guide = shape+1.5*snake
                    elif policy == 5:
                        guide = shape+50*b.count(0)
                    else:
                        guide = shape
                    return base+fade*guide
                for node in heapq.nlargest(quota, nodes, key=rank):
                    kept[(node[0], node[1])] = node
            # Fill overlap with the pure score/order lens, allowing board and
            # RNG cursor transpositions only when cumulative score dominates.
            def general(n):
                base, shape = qualities[(n[0], n[1])]
                return base+fade*shape
            for node in heapq.nlargest(width, nodes, key=general):
                if len(kept) >= width:
                    break
                kept[(node[0], node[1])] = node
            beam = list(kept.values())
        # If compute ran out, finish a few equal-depth survivors. Never act on
        # the numeric value of a partially explored layer or an unclosed prefix.
        if completed < horizon and beam:
            ordered = sorted(beam, key=lambda n: n[2]+max(n[0])+.24*layout(n[0]), reverse=True)
            for index, node in enumerate(ordered[:8]):
                policy = self.preferred_policy if index == 0 else index % 6
                path = self.rollout(board, self.cursor, horizon, policy, node[3], stop)
                if path and self.objective(path) > best:
                    incumbent, best = path, self.objective(path)
                if self.stopped(stop):
                    break
        for path in terminal_paths:
            if self.objective(path) > best:
                incumbent, best = list(path), self.objective(path)
        self.plan = incumbent
        self.next_search = move+8
        self.beam_report = {'depth':completed, 'horizon':horizon, 'expanded':expanded,
                            'old_utility':old_objective[0], 'new_utility':best[0],
                            'full_beam':completed == horizon, 'width':width}
        if incumbent:
            forecast = obs.get('score', 0)+best[0]
            if forecast > self.best_forecast:
                self.best_forecast, self.plateau_since = forecast, move
            self.last_forecast = {'final_combined':forecast, 'steps':len(incumbent),
                                  'terminal_mass':sum(incumbent[-1][1]),
                                  'terminal_max':max(incumbent[-1][1]),
                                  'beam':self.beam_report}
            self.ctx.log('full_horizon_search', move=move, **self.beam_report,
                         forecast=self.last_forecast)
            if move-self.plateau_since >= 48 and move-self.last_review >= 32:
                self.review(obs, 'complete terminal forecast unchanged for 48 moves; inspect pruning geometry')
            return incumbent[0][0]
        return self.fallback(board, legal)

    def fallback(self, board, legal):
        """Small all-spawn one-reply search when the RNG model is untrusted."""
        scores = {}
        for action in legal:
            moved,gain = transition(board,action)
            if moved == board:
                continue
            total = 0
            empty = [i for i,v in enumerate(moved) if not v]
            for i in empty:
                for tile,prob in ((2,.9),(4,.1)):
                    b = list(moved)
                    b[i] = tile
                    child = tuple(b)
                    opts = [greedy_value(n,g,self.preferred_policy) for a in DIRECTIONS
                            for n,g in [transition(child,a)] if n != child]
                    total += prob*(max(opts) if opts else layout(child)-10000)
            scores[action] = gain + total/max(1,len(empty))
        return max(scores,key=scores.get) if scores else legal[0]

    def run_episode(self, game):
        ctx = self.ctx
        obs = observation(ctx.observe())
        if 'board' not in obs:
            return
        try:
            if obs.get('moves',0) == 0 and obs.get('seed') is not None:
                stream = SeedStream(obs['seed'])
                board,cursor = stream.spawn((0,)*16,0)
                board,cursor = stream.spawn(board,cursor)
                if board == flat(obs):
                    self.stream,self.cursor = stream,cursor
        except (ValueError,TypeError):
            pass
        ctx.log('opening_model_check',seed_stream_match=self.stream is not None)
        while not ctx.done and not obs.get('game_over',False):
            board = flat(obs)
            probe = self.poll_review(obs)
            legal = [a for a in obs.get('legal_actions',DIRECTIONS)
                     if a in DIRECTIONS and a not in self.rejected]
            if not legal:
                break
            local = [a for a in legal if transition(board,a)[0] != board]
            if self.rules_ok and not local:
                self.rules_ok = False
                self.stream = None
                self.plan = []
                self.review(obs,'observed legal actions conflict with local rule model')
            if not self.rules_ok:
                action = probe if probe in legal else legal[0]
            elif self.stream is None:
                action = self.fallback(board,local)
            else:
                action = self.optimize(obs,local)
            if ctx.done:
                break
            expected = next_cursor = None
            if self.stream is not None:
                moved,_ = transition(board,action)
                expected,next_cursor = self.stream.spawn(moved,self.cursor)
            before = obs
            try:
                nxt = ctx.act(action)
                obs = observation(nxt if nxt is not None else ctx.observe())
            except Exception as exc:
                if ctx.done:
                    break
                self.rejected.add(action)
                self.plan = []
                self.review(obs,'action rejected: '+type(exc).__name__)
                continue
            self.rejected.clear()
            progress = ctx.progress()
            error = rule_error(before,obs,action)
            seed_error = expected is not None and expected != flat(obs)
            gain = obs.get('score',0)-before.get('score',0)
            if error:
                self.rules_ok = False
                self.stream = None
                self.plan = []
                self.good_probes = 0
            elif seed_error:
                self.stream = None
                self.plan = []
            else:
                if self.stream is not None:
                    self.cursor = next_cursor
                if self.plan:
                    self.plan = [(a,b,p,g-gain) for a,b,p,g in self.plan[1:]]
                if not self.rules_ok:
                    self.good_probes += 1
                    if self.good_probes >= 2:
                        self.rules_ok = True
            self.history.append({'move':obs.get('moves'),'action':action,'gain':gain,
                                 'empty':flat(obs).count(0),'rule_error':error,
                                 'seed_error':seed_error})
            ctx.log('verified_trajectory_step',move=obs.get('moves'),action=action,gain=gain,
                    rule_error=error,seed_error=seed_error,forecast=self.last_forecast,
                    retained=len(self.plan))
            if ctx.done or obs.get('game_over',False):
                break
            reasons = []
            if error: reasons.append(error)
            if seed_error: reasons.append('seed prediction failed; discarded complete trajectory')
            if progress.get('since_obs_change',0) >= 1: reasons.append('observation unchanged')
            if progress.get('illegal_actions',0) > self.last_illegal: reasons.append('action rejected')
            self.last_illegal = progress.get('illegal_actions',0)
            if progress.get('since_score_change',0) >= 8: reasons.append('eight scoreless actions')
            if flat(obs).count(0) <= 2: reasons.append('space collapsed')
            if self.plan and len(self.plan) < max(1,200-obs.get('moves',0)):
                reasons.append('forecast predicts premature game over')
            urgent = bool(error or seed_error or progress.get('since_obs_change',0))
            if reasons and (urgent or obs.get('moves',0)-self.last_review >= 24):
                self.review(obs,'; '.join(reasons))
            elif obs.get('moves',0) == 48:
                self.review(obs,'audit actual progress and complete-trajectory forecast')


def build(ctx):
    return Harness(ctx)
