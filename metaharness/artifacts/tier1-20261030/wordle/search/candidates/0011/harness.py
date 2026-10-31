"""Wordle repeated policy improvement with answer-only simulated continuations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time


def feedback(answer, guess):
    """Duplicate-aware base-3 feedback: gray=0, yellow=1, green=2."""
    tiles = [0] * 5
    left = list(answer)
    for i in range(5):
        if answer[i] == guess[i]:
            tiles[i] = 2
            left[i] = None
    for i, ch in enumerate(guess):
        if tiles[i] == 0 and ch in left:
            tiles[i] = 1
            left[left.index(ch)] = None
    return sum(x * p for x, p in zip(tiles, (1, 3, 9, 27, 81)))


def history(obs):
    result = []
    for row in obs.get('history', []):
        if not isinstance(row, dict):
            raise ValueError('unrecognized history row')
        word = row.get('guess', row.get('word', ''))
        if isinstance(word, dict):
            word = word.get('guess', '')
        word = str(word).strip().lower().removeprefix('guess ').strip()
        if not re.fullmatch('[a-z]{5}', word):
            raise ValueError('unrecognized accepted guess')
        tiles = row.get('feedback', row.get('tiles', []))
        if isinstance(tiles, str):
            tiles = re.findall('green|yellow|gray|grey', tiles.lower())
        if len(tiles) != 5:
            raise ValueError('expected five feedback tiles')
        code = 0
        for tile, weight in zip(tiles, (1, 3, 9, 27, 81)):
            if isinstance(tile, dict):
                tile = tile.get('color', tile.get('state', tile.get('value', '')))
            color = str(tile).strip().lower()
            mapping = {'gray': 0, 'grey': 0, 'yellow': 1, 'green': 2,
                       '0': 0, '1': 1, '2': 2}
            if color not in mapping:
                raise ValueError('unknown tile: ' + color)
            code += mapping[color] * weight
        result.append((word, code))
    return result


def load_lists(rules):
    found = {}
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and os.path.isfile(item):
                    if any(s in key.lower() for s in ('answer', 'allowed', 'guess')):
                        found[key.lower()] = item
                elif isinstance(item, (dict, list)):
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(rules)
    ap = next((v for k, v in found.items() if 'answer' in k), None)
    gp = next((v for k, v in found.items() if 'allowed' in k or 'guess' in k), None)
    def read(path):
        if not path:
            raise ValueError('missing public word-list path')
        with open(path, 'rb') as stream:
            data = stream.read()
        expected = rules.get('wordlist_sha256', {}).get(os.path.basename(path))
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('public word-list hash mismatch')
        return sorted({w.strip().lower() for w in data.decode('utf-8').splitlines()
                       if re.fullmatch('[a-zA-Z]{5}', w.strip())})
    answers = read(ap)
    legal = sorted(set(read(gp)) | set(answers))
    if not answers or not legal:
        raise ValueError('empty public word lists')
    return answers, legal


class SearchStopped(Exception):
    pass


class RewardSearch:
    """One-step policy improvement over an attainable candidate-only tree.

    Probe every legal word at the root. Beyond the root, search only possible
    answers (exact through 24 candidates; minimax rollout for larger states).
    Recompute this improvement after observing the real next feedback.
    """
    def __init__(self, candidates, legal, ctx, seconds=150.0):
        self.words = tuple(candidates)
        self.legal = tuple(legal)
        self.ctx = ctx
        self.end = time.monotonic() + seconds
        self.rows = {}
        self.memo = {}
        self.greedy_memo = {}
        self.best_root = self.words[0]
        self.best_value = None
        self.nodes = 0
        self.root_tested = 0
        self.root_actions = 0
        self.complete = False
        self.large_rollouts = 0
        self.root_upper = None

    def check(self):
        if self.ctx.done or time.monotonic() >= self.end:
            raise SearchStopped()
        if self.ctx.budget.remaining().get('wall_s', 1800) < 8:
            raise SearchStopped()

    def groups(self, state, guess):
        if guess not in self.rows:
            self.rows[guess] = bytes(feedback(a, guess) for a in self.words)
        row = self.rows[guess]
        groups = {}
        for i in state:
            code = row[i]
            if code != 242:
                groups.setdefault(code, []).append(i)
        return tuple(sorted((tuple(v) for v in groups.values()),
                            key=lambda g: (-len(g), g)))

    @staticmethod
    def ceiling(n, r):
        if not n or r <= 0:
            return 0
        if r == 1:
            return 1
        return (r - 1) * n + 1

    def greedy(self, state, r):
        """An attainable candidate-only minimax tree, including horizon losses."""
        self.check()
        n = len(state)
        if not n or r <= 0:
            return 0, None
        first = self.words[state[0]]
        if n == 1 or r == 1:
            return r, first
        if n == 2:
            return 2*r-1, first
        key = state, r
        if key in self.greedy_memo:
            return self.greedy_memo[key]
        best = None
        for i in state:
            self.check()
            guess = self.words[i]
            groups = self.groups(state, guess)
            sizes = [len(g) for g in groups]
            rank = (max(sizes, default=0), sum(n*n for n in sizes), guess)
            if best is None or rank < best[0]:
                best = rank, guess, groups
        _, guess, groups = best
        value = r + sum(self.greedy(g, r-1)[0] for g in groups)
        self.greedy_memo[key] = value, guess
        return value, guess

    def tail(self, state, r):
        """Solve the restricted candidate-only continuation policy exactly."""
        self.check()
        n = len(state)
        if not n or r <= 0:
            return 0, None
        first = self.words[state[0]]
        if n == 1 or r == 1:
            return r, first
        if n == 2:
            return 2*r-1, first
        key = state, r
        if key in self.memo:
            return self.memo[key]
        best, best_guess = self.greedy(state, r)
        if n > 24:
            self.large_rollouts += 1
            self.memo[key] = best, best_guess
            return best, best_guess
        self.nodes += 1
        ceiling = self.ceiling(n, r)
        if best == ceiling:
            self.memo[key] = best, best_guess
            return best, best_guess
        records = []
        for i in state:
            guess = self.words[i]
            groups = self.groups(state, guess)
            upper = r + sum(self.ceiling(len(g), r-1) for g in groups)
            if upper > best:
                records.append((upper, sum(len(g)**2 for g in groups), guess, groups))
        records.sort(key=lambda rec: (-rec[0], rec[1], rec[2]))
        for upper, _, guess, groups in records:
            self.check()
            if upper <= best:
                continue
            val, unvisited = r, upper-r
            for g in groups:
                unvisited -= self.ceiling(len(g), r-1)
                val += self.tail(g, r-1)[0]
                if val + unvisited <= best:
                    break
            else:
                if val > best:
                    best, best_guess = val, guess
            if best == ceiling:
                break
        self.memo[key] = best, best_guess
        return best, best_guess

    def choose(self, remaining):
        state = tuple(range(len(self.words)))
        try:
            self.best_value, self.best_root = self.greedy(state, remaining)
            self.root_upper = self.ceiling(len(state), remaining)
            current = set(self.words)
            records, seen = [], set()
            # Rank all root probes before spending time on deeper branches.
            for j, guess in enumerate(self.legal):
                if j % 32 == 0:
                    self.check()
                groups = self.groups(state, guess)
                immediate = remaining if guess in current else 0
                upper = immediate + sum(self.ceiling(len(g), remaining-1) for g in groups)
                if upper <= self.best_value:
                    continue
                signature = immediate, tuple(sorted(groups))
                if signature in seen:
                    continue
                seen.add(signature)
                records.append((upper, sum(len(g)**2 for g in groups),
                                guess, groups, immediate))
            records.sort(key=lambda rec: (-rec[0], rec[1], -rec[4], rec[2]))
            self.root_actions = len(records)
            # This is an upper bound on the restricted rollout optimization,
            # not a claim about global all-action Wordle optimality.
            for index, (upper, _, guess, groups, immediate) in enumerate(records):
                self.check()
                self.root_upper = max(self.best_value, upper)
                if upper <= self.best_value:
                    break
                value, unvisited = immediate, upper-immediate
                for group in groups:
                    unvisited -= self.ceiling(len(group), remaining-1)
                    value += self.tail(group, remaining-1)[0]
                    if value + unvisited <= self.best_value:
                        break
                else:
                    if value > self.best_value:
                        self.best_value, self.best_root = value, guess
                self.root_tested += 1
            self.complete = True
            self.root_upper = self.best_value
        except SearchStopped:
            pass
        self.ctx.log('policy_improvement', candidates=len(self.words), guess=self.best_root,
                     attainable_reward=self.best_value,
                     rollout_upper=self.root_upper, root_scan_complete=self.complete,
                     candidate_tail_states=self.nodes, large_greedy_states=self.large_rollouts,
                     root_actions=self.root_actions, root_tested=self.root_tested,
                     feedback_rows=len(self.rows), globally_exact=False)
        return self.best_root


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.recoveries = 0
        self.fallbacks = 0

    def recover(self, obs, evidence, legal, used, candidates):
        ctx = self.ctx
        ctx.log('recovery_needed', evidence=str(evidence))
        if ctx.done or self.recoveries >= 3:
            return None
        budget = ctx.budget.remaining()
        if budget.get('system2_calls', 0) < 1 or budget.get('wall_s', 0) < 10:
            return None
        self.recoveries += 1
        prompt = ('A Wordle rollout plan failed verification. Diagnose from the '
                  'public observation and evidence, then return JSON {"guess":"xxxxx"}. '
                  'Choose one unused legal guess; no long explanation. The next action will '
                  'be checked again.\n' + json.dumps({'observation': obs, 'evidence': str(evidence),
                  'progress': ctx.progress(),
                  'consistent_candidates': candidates[:100],
                  'unused_examples': [w for w in legal if w not in used][:60]}, ensure_ascii=True))
        try:
            reply = ctx.think(prompt, session_id=self.session,
                              timeout_s=min(300.0, max(1, budget['wall_s']-5)))
            self.session = reply.session_id
            text = reply.text or ''
            try:
                data = json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
                words = [str(data.get('guess', '')).strip().lower()]
            except Exception:
                words = re.findall(r'(?<![a-z])[a-z]{5}(?![a-z])', text.lower())
            return next((w for w in words if re.fullmatch('[a-z]{5}', w)
                         and w not in used and (not legal or w in legal)), None)
        except Exception as exc:
            ctx.log('recovery_call_failed', error=repr(exc))
            return None

    def run_episode(self, game):
        ctx = self.ctx
        obs = ctx.observe()
        try:
            answers, legal = load_lists(obs.get('rules', {}))
            if obs.get('rules', {}).get('word_length', 5) != 5:
                raise ValueError('unexpected word length')
            issue = None
        except Exception as exc:
            answers, legal = [], []
            issue = repr(exc)
        banned = set()
        # One move per iteration, including recovery moves: all share verification.
        while not ctx.done and not obs.get('done'):
            used = set(banned)
            try:
                hist = history(obs)
                used.update(w for w, _ in hist)
                if len(hist) != int(obs.get('turn', len(hist))):
                    raise ValueError('history length disagrees with turn count')
                candidates = [a for a in answers if all(feedback(a, g) == f for g, f in hist)]
                if not candidates:
                    issue = issue or 'no public answer matches the complete accepted history'
            except Exception as exc:
                hist, candidates = [], []
                issue = repr(exc)
            remaining = int(obs.get('max_turns', 6)) - int(obs.get('turn', 0))
            if remaining <= 0:
                return
            if issue:
                guess = self.recover(obs, issue, legal, used, candidates)
                if not guess and not ctx.done and self.recoveries < 3 and self.fallbacks < 2:
                    # Astra saw the failure; garbage or a timeout need not
                    # discard a still-verified candidate set.
                    guess = next((w for w in candidates if w not in used), None)
                    self.fallbacks += 1
                if not guess:
                    return
            elif len(candidates) <= 2 or remaining == 1:
                guess = next((w for w in candidates if w not in used), None)
            elif not hist:
                # Fixed, seed-independent high-information opener; same puzzle for all.
                guess = next((w for w in ('soare', 'raise', 'slate') if w in legal), candidates[0])
            else:
                search = RewardSearch(candidates, [w for w in legal if w not in used], ctx,
                                      seconds=min(150.0, max(1, ctx.budget.remaining()['wall_s']-25)))
                guess = search.choose(remaining)
            if ctx.done:
                return
            if not guess or guess in used or (legal and guess not in legal):
                issue = 'selected guess is used, missing, or not in public legal list'
                continue
            before = ctx.progress()
            old_turn = int(obs.get('turn', 0))
            old_hist = obs.get('history', [])
            predicted = {feedback(a, guess) for a in candidates}
            try:
                nxt = ctx.act('guess ' + guess)
                if not isinstance(nxt, dict):
                    nxt = ctx.observe()
            except Exception as exc:
                if ctx.done:
                    return
                banned.add(guess)
                issue = 'ctx.act exception: ' + repr(exc)
                obs = ctx.observe()
                continue
            after = ctx.progress()
            failures = []
            if nxt.get('error'):
                failures.append('action error: ' + str(nxt['error']))
            if after.get('illegal_actions', 0) > before.get('illegal_actions', 0):
                failures.append('illegal_actions increased')
            if int(nxt.get('turn', old_turn)) != old_turn+1:
                failures.append('turn did not advance exactly once')
            if after.get('since_obs_change', 0) > 0 and not nxt.get('done'):
                failures.append('progress reports unchanged observation')
            try:
                new_hist = history(nxt)
                if not new_hist or new_hist[-1][0] != guess:
                    failures.append('accepted history does not end in posted guess')
                elif predicted and new_hist[-1][1] not in predicted:
                    failures.append('observed feedback absent from predicted branches')
                elif candidates and not nxt.get('done'):
                    survivors = [a for a in candidates
                                 if feedback(a, guess) == new_hist[-1][1]]
                    if len(survivors) >= len(candidates):
                        failures.append('accepted guess did not reduce the candidate set')
                if nxt.get('history', [])[:-1] != old_hist:
                    failures.append('previous accepted history changed')
            except Exception as exc:
                failures.append('history verification: ' + repr(exc))
            ctx.log('plan_verification', guess=guess, candidates=len(candidates),
                    progress=after, failures=failures)
            obs = nxt
            if nxt.get('done'):
                return
            if failures:
                banned.add(guess)
                issue = '; '.join(failures)
                # Issue is handled by Astra at the top, followed by the same verifier.
            else:
                issue = None


def build(ctx):
    return Harness(ctx)
