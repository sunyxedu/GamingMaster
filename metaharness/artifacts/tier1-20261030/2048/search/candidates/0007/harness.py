"""Complete-layer anytime expectimax with a finite-horizon cashout objective.

Derived from 0002. Every root is compared at the same completed search depth;
partial deeper layers are discarded. No seed or RNG-state reconstruction.
"""
from __future__ import annotations

import json
import math
import re
import time
from functools import lru_cache

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


DIRS = ("up", "down", "left", "right")
MODES = ("balanced", "space", "merge")


@lru_cache(maxsize=32768)
def _line(values):
    vals = [int(v) for v in values if v]
    out, gain, i = [], 0, 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            out.append(vals[i] * 2)
            gain += vals[i] * 2
            i += 2
        else:
            out.append(vals[i])
            i += 1
    return tuple(out + [0] * (4 - len(out))), gain


@lru_cache(maxsize=65536)
def transition(board, action):
    """Exact slide result and merge gain, before the random spawn."""
    b = list(board)
    gain = 0
    if action in ("left", "right"):
        for r in range(4):
            ids = tuple(4 * r + c for c in range(4))
            if action == "right":
                ids = ids[::-1]
            vals, g = _line(tuple(board[i] for i in ids))
            gain += g
            for i, v in zip(ids, vals):
                b[i] = v
    else:
        for c in range(4):
            ids = tuple(4 * r + c for r in range(4))
            if action == "down":
                ids = ids[::-1]
            vals, g = _line(tuple(board[i] for i in ids))
            gain += g
            for i, v in zip(ids, vals):
                b[i] = v
    result = tuple(b)
    return result, gain, result != board


def _flat(obs):
    try:
        return tuple(int(x) for row in obs["board"] for x in row)
    except Exception:
        return (0,) * 16


def _snake_scores(board):
    """Return best monotone snake score and whether its anchor is occupied."""
    patterns = (
        (0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12),
        (3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15),
        (12, 13, 14, 15, 11, 10, 9, 8, 4, 5, 6, 7, 3, 2, 1, 0),
        (15, 14, 13, 12, 8, 9, 10, 11, 7, 6, 5, 4, 0, 1, 2, 3),
    )
    logs = [math.log2(v) if v else 0.0 for v in board]
    best, anchor = -1e9, 0
    for p in patterns:
        # Penalize descending violations heavily while rewarding a smooth rise.
        score = 0.0
        for a, b in zip(p, p[1:]):
            if board[a] and board[b]:
                d = logs[a] - logs[b]
                score += d if d >= 0 else 2.8 * d
        score += (logs[p[0]] * 3.0 if board[p[0]] else -3.0)
        if score > best:
            best, anchor = score, p[0]
    return best, anchor


def heuristic(board, mode):
    empty = board.count(0)
    mx = max(board) if board else 0
    snake, anchor = _snake_scores(board)
    logs = [math.log2(v) if v else 0.0 for v in board]
    smooth = 0.0
    pairs = 0.0
    for r in range(4):
        for c in range(4):
            i = 4 * r + c
            if not board[i]:
                continue
            if c < 3 and board[i + 1]:
                smooth -= abs(logs[i] - logs[i + 1])
                pairs += 1.0 if board[i] == board[i + 1] else 0.0
            if r < 3 and board[i + 4]:
                smooth -= abs(logs[i] - logs[i + 4])
                pairs += 1.0 if board[i] == board[i + 4] else 0.0
    # Space is most valuable early; merge potential matters more near a full board.
    if mode == "space":
        we, ws, wm, wp = 210.0, 30.0, 4.0, 26.0
    elif mode == "merge":
        we, ws, wm, wp = 125.0, 42.0, 8.0, 46.0
    else:
        we, ws, wm, wp = 175.0, 34.0, 6.0, 34.0
    return we * empty + ws * snake + wm * smooth + wp * pairs + 22.0 * math.log2(mx or 1) + 8.0 * math.log2(board[anchor] or 1)


class SearchStop(Exception):
    pass


class Search:
    """An incomplete iteration never contaminates root comparisons."""
    def __init__(self, ctx, mode, risk, remaining, seconds=.65):
        self.ctx, self.mode, self.risk = ctx, mode, risk
        self.remaining = max(1, remaining)
        self.stop = time.monotonic() + seconds
        self.nodes = 0
        self.cache, self.leaves = {}, {}
        self.max_depth = 3

    def check(self):
        self.nodes += 1
        if self.nodes % 64 == 0:
            if self.nodes >= 50000 or time.monotonic() >= self.stop or self.ctx.done:
                raise SearchStop()

    def leaf(self, board, turns):
        key = (board, turns)
        if key not in self.leaves:
            # At the documented horizon only the evaluator's score matters.
            # Smoothly remove shape bonuses over the final 16 changed moves.
            weight = min(1., max(0., turns / 16.))
            self.leaves[key] = (weight * heuristic(board, self.mode)
                                + (1. - weight) * max(board))
        return self.leaves[key]

    def value(self, board, depth, turns):
        self.check()
        if depth <= 0 or turns <= 0:
            return self.leaf(board, turns)
        key = (board, depth, turns)
        if key in self.cache:
            return self.cache[key]
        best = None
        for action in DIRS:
            moved, gain, changed = transition(board, action)
            if changed:
                value = gain + self.chance(moved, depth - 1, turns - 1)[0]
                best = value if best is None else max(best, value)
        if best is None:
            best = max(board) - (10000. if turns > 0 else 0.)
        self.cache[key] = best
        return best

    def chance(self, board, depth, turns):
        empties = [i for i, v in enumerate(board) if not v]
        if not empties:
            v = self.value(board, depth, turns)
            return v, v, v
        total, worst = 0., float('inf')
        for i in empties:
            for tile, probability in ((2, .9), (4, .1)):
                child = list(board)
                child[i] = tile
                value = self.value(tuple(child), depth, turns)
                total += probability * value
                worst = min(worst, value)
        mean = total / len(empties)
        # Risk applies only under actual space pressure, never at final cashout.
        risk = self.risk if len(empties) <= 5 and turns > 3 else 0.
        return (1. - risk) * mean + risk * worst, mean, worst

    def choose(self, board, legal):
        # Cheap legal fallback exists before the first deadline check.
        fallback = max(legal, key=lambda a: transition(board, a)[1] +
                       self.leaf(transition(board, a)[0], self.remaining - 1))
        completed, reports = 0, []
        for depth in range(1, min(self.max_depth, self.remaining) + 1):
            trial = []
            try:
                for action in legal:
                    moved, gain, changed = transition(board, action)
                    if not changed:
                        continue
                    value, mean, worst = self.chance(moved, depth - 1, self.remaining - 1)
                    trial.append({'action': action, 'value': gain + value,
                                  'expected': gain + mean, 'worst': gain + worst,
                                  'gain': gain, 'empty_after_slide': moved.count(0)})
            except SearchStop:
                break
            if trial:
                completed, reports = depth, trial
                fallback = max(trial, key=lambda row: row['value'])['action']
        return fallback, reports, completed, self.nodes


def parse(text):
    try:
        match = re.search(r'\{[^{}]*\}', text or '')
        data = json.loads(match.group()) if match else {}
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def verify(before, after, action):
    b, a = _flat(before), _flat(after)
    moved, gain, changed = transition(b, action)
    if not changed:
        return 'chosen move was locally unchanged'
    if after.get('score', 0) - before.get('score', 0) != gain:
        return 'merge score disagreed'
    if after.get('moves', 0) != before.get('moves', 0) + 1:
        return 'move counter disagreed'
    diff = [i for i in range(16) if moved[i] != a[i]]
    if len(diff) != 1 or moved[diff[0]] != 0 or a[diff[0]] not in (2, 4):
        return 'board was not slide plus one legal spawn'
    spawn = after.get('spawn')
    if isinstance(spawn, dict):
        if (spawn.get('row'), spawn.get('col'), spawn.get('value')) != (diff[0] // 4, diff[0] % 4, a[diff[0]]):
            return 'spawn metadata disagreed'
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.mode, self.risk = 'balanced', .1
        self.session = None
        self.calls, self.last_review, self.last_illegal = 0, -100, 0
        self.history, self.report = [], []
        self.rules_ok = True
        self.recovery = None

    def review(self, obs, reason):
        ctx = self.ctx
        rem = ctx.budget.remaining()
        if ctx.done or rem.get('system2_calls', 0) <= 1 or rem.get('wall_s', 0) < 240 or self.calls >= 5:
            return
        self.calls += 1
        self.last_review = int(obs.get('moves', 0))
        evidence = {'reason': reason, 'board': obs['board'], 'score': obs.get('score'),
                    'moves': obs.get('moves'), 'legal': obs.get('legal_actions'),
                    'progress': ctx.progress(), 'rules_verified': self.rules_ok,
                    'current': {'mode': self.mode, 'risk': self.risk},
                    'recent': self.history[-6:], 'root_search': self.report}
        prompt = ('Supervise 2048 local search. Objective merge score + maximum tile; '
                  'default episode horizon 200 changed moves. Every move is verified. '
                  'Root scores use only complete equal-depth expectimax layers, all 2/4 spawns. '
                  'Choose mode balanced/space/merge and risk neutral/cautious (worst-spawn '
                  'weight 0/.15, active only with <=5 empties). Shape bonuses fade in the last '
                  '16 moves to optimize terminal reward. Diagnose the supplied failure evidence. '
                  'If rules fail, choose one legal action as a probe; otherwise the search acts. '
                  'Reply compact JSON {"mode":"balanced","risk":"neutral",'
                  '"action":"left","reason":"brief"}.\n' + json.dumps(evidence, separators=(',', ':')))
        try:
            result = ctx.think(prompt, session_id=self.session, timeout_s=180.)
            self.session = getattr(result, 'session_id', self.session)
            data = parse(getattr(result, 'text', ''))
            if data.get('mode') in MODES:
                self.mode = data['mode']
            if data.get('risk') in ('neutral', 'cautious'):
                self.risk = {'neutral': 0., 'cautious': .15}[data['risk']]
            if data.get('action') in obs.get('legal_actions', []) and not self.rules_ok:
                self.recovery = data['action']
            ctx.log('policy_review', reason=reason, mode=self.mode, risk=self.risk, reply=data)
        except Exception as exc:
            ctx.log('review_unavailable', reason=reason, error=type(exc).__name__)

    def run_episode(self, game):
        ctx = self.ctx
        try:
            obs = ctx.observe()
        except Exception:
            try:
                obs = game.observe()
            except Exception:
                return
        if isinstance(obs, str):
            obs = json.loads(obs)
        if not isinstance(obs, dict) or 'board' not in obs:
            return
        rejected = set()
        # Start play immediately: opening arithmetic does not need a slow call.
        while not ctx.done and not obs.get('game_over', False):
            board = _flat(obs)
            legal = [a for a in obs.get('legal_actions', DIRS) if a in DIRS and a not in rejected]
            if not legal:
                return
            local = [a for a in legal if transition(board, a)[2]]
            if not local:
                self.rules_ok = False
                self.review(obs, 'all observed legal actions disagree with local rules')
                action = self.recovery if self.recovery in legal else legal[0]
            elif not self.rules_ok:
                action = self.recovery if self.recovery in legal else max(local, key=lambda a: transition(board, a)[1])
            else:
                move = int(obs.get('moves', 0))
                # Beyond the public default, continue with a rolling horizon.
                horizon = 200 - move if move < 200 else 200
                seconds = min(.65, max(.02, ctx.budget.remaining().get('wall_s', 0) * .002))
                action, self.report, depth, nodes = Search(ctx, self.mode, self.risk, horizon, seconds).choose(board, local)
                ctx.log('search', move=move, depth=depth, nodes=nodes, mode=self.mode,
                        risk=self.risk, ranking=self.report)
            self.recovery = None
            before = obs
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else ctx.observe()
                if isinstance(obs, str):
                    obs = json.loads(obs)
            except Exception as exc:
                if ctx.done:
                    return
                rejected.add(action)
                self.rules_ok = False
                self.review(before, 'action raised ' + type(exc).__name__)
                continue
            progress = ctx.progress()
            error = verify(before, obs, action)
            self.rules_ok = error is None
            if error:
                rejected.add(action)
            else:
                rejected.clear()
            self.history.append({'action': action, 'gain': obs.get('score', 0) - before.get('score', 0),
                                 'empty': _flat(obs).count(0), 'error': error})
            self.history = self.history[-8:]
            ctx.memory['policy'] = {'mode': self.mode, 'risk': self.risk, 'rules_ok': self.rules_ok}
            reasons = []
            if error:
                reasons.append(error)
            if progress.get('illegal_actions', 0) > self.last_illegal:
                reasons.append('rejected action in progress telemetry')
            self.last_illegal = progress.get('illegal_actions', 0)
            if progress.get('since_obs_change', 0) >= 1:
                reasons.append('observation stopped changing')
            if progress.get('since_score_change', 0) >= 10:
                reasons.append('ten actions without score progress')
            if _flat(obs).count(0) <= 3:
                reasons.append('space collapsed')
            move = int(obs.get('moves', 0))
            if reasons and (error or move - self.last_review >= 20):
                self.review(obs, '; '.join(reasons))
            elif move == 80 or move == 175:
                self.review(obs, 'midgame audit' if move == 80 else 'terminal cashout audit')


def build(ctx):
    return Harness(ctx)
