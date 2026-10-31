"""Wordle all-action branch-and-bound with certified reward gaps and recovery."""
from __future__ import annotations

from collections import Counter
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
    """All-action branch-and-bound, with attainable incumbents and proof gaps.

    No fixed beam removes actions. The upper bound counts the earliest that
    each answer could possibly solve; a complete policy provides a lower bound.
    Search terminates when those bounds meet, or returns its incumbent on time.
    """
    def __init__(self, candidates, legal, ctx, seconds=150.0):
        self.words = tuple(candidates)
        self.legal = tuple(legal)
        self.ctx = ctx
        self.end = time.monotonic() + seconds
        self.rows = {}
        self.memo = {}
        self.greedy_memo = {}
        self.index = {w: i for i, w in enumerate(self.words)}
        self.best_root = self.words[0]
        self.best_value = None
        self.root_upper = None
        self.nodes = 0
        self.root_tested = 0
        self.root_actions = 0
        self.complete = False

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
        return tuple(sorted((tuple(v) for v in groups.values()), key=lambda g: (-len(g), g)))

    @staticmethod
    def ceiling(n, r):
        if not n or r <= 0:
            return 0
        if r == 1:
            return 1
        return (r - 1) * n + 1

    def greedy(self, state, r):
        """An inexpensive *attainable* whole-tree incumbent, not a leaf estimate."""
        self.check()
        if not state or r <= 0:
            return 0, None
        first = self.words[state[0]]
        if len(state) == 1 or r == 1:
            return r, first
        if len(state) == 2:
            return 2*r-1, first
        key = state, r
        if key in self.greedy_memo:
            return self.greedy_memo[key]
        best = None
        for j, i in enumerate(state):
            if j % 32 == 0:
                self.check()
            guess = self.words[i]
            groups = self.groups(state, guess)
            # Candidate-only minimax is cheap, always progresses, and yields
            # fully executable plans even if optimal search has no time left.
            sizes = [len(g) for g in groups]
            rank = (max(sizes, default=0), sum(n*n for n in sizes), guess)
            if best is None or rank < best[0]:
                best = rank, guess, groups
        _, guess, groups = best
        value = r + sum(self.greedy(g, r-1)[0] for g in groups)
        self.greedy_memo[key] = value, guess
        return value, guess

    def solve(self, state, r, root=False):
        self.check()
        if not state or r <= 0:
            return 0, None
        n = len(state)
        first = self.words[state[0]]
        if n == 1 or r == 1:
            return r, first
        if n == 2:
            return 2*r-1, first
        key = state, r
        if key in self.memo:
            return self.memo[key]
        self.nodes += 1
        best, best_guess = self.greedy(state, r)
        ceiling = self.ceiling(n, r)
        if root:
            self.best_root, self.best_value = best_guess, best
            self.root_upper = ceiling
        if best == ceiling:
            self.memo[key] = best, best_guess
            return best, best_guess
        current = {self.words[i] for i in state}
        guesses = list(sorted(current)) + [w for w in self.legal if w not in current]
        records = []
        seen = set()
        for j, guess in enumerate(guesses):
            if j % 64 == 0:
                self.check()
            groups = self.groups(state, guess)
            immediate = r if guess in current else 0
            if not immediate and len(groups) == 1 and len(groups[0]) == n:
                continue
            upper = immediate + sum(self.ceiling(len(g), r-1) for g in groups)
            if upper <= best:
                continue
            # The upper bound is attainable for singleton and pair children.
            # Resolve such actions without constructing more subproblem options.
            if r == 2 or all(len(g) <= 2 for g in groups):
                value = immediate + sum((r-1 if len(g) == 1 else 2*(r-1)-1)
                                        if r > 2 else 1 for g in groups)
                if value > best:
                    best, best_guess = value, guess
                    if root:
                        self.best_root, self.best_value = guess, value
                if best == ceiling:
                    self.memo[key] = best, best_guess
                    if root:
                        self.root_upper = best
                    return best, best_guess
                continue
            signature = (immediate, tuple(sorted(groups)))
            if signature in seen:
                continue
            seen.add(signature)
            collision = sum(len(g)**2 for g in groups)
            records.append((upper, collision, guess, groups, immediate))
        # Every distinct promising action survives. Explore the highest possible
        # return first; branch-and-bound, rather than a fixed-width beam, prunes.
        records.sort(key=lambda rec: (-rec[0], rec[1], -rec[4], rec[2]))
        if root:
            self.root_actions = len(records)
            self.root_upper = max(best, max((rec[0] for rec in records), default=best))
        for idx, (upper, _, guess, groups, immediate) in enumerate(records):
            self.check()
            if upper <= best:
                continue
            if root:
                self.root_upper = max(best, upper)
            value = immediate
            unvisited = upper-immediate
            for group in groups:
                unvisited -= self.ceiling(len(group), r-1)
                child, _ = self.solve(group, r-1)
                value += child
                if value + unvisited <= best:
                    break
            else:
                if value > best:
                    best, best_guess = value, guess
                    if root:
                        self.best_root, self.best_value = guess, value
            if root:
                self.root_tested += 1
                self.root_upper = max(best, records[idx+1][0] if idx+1 < len(records) else best)
            if best == ceiling:
                break
        self.memo[key] = best, best_guess
        if root:
            self.root_upper = best
        return best, best_guess

    def choose(self, remaining):
        state = tuple(range(len(self.words)))
        try:
            value, guess = self.solve(state, remaining, root=True)
            self.best_root, self.best_value, self.root_upper = guess, value, value
            self.complete = True
        except SearchStopped:
            pass
        self.ctx.log('all_action_search', candidates=len(self.words), guess=self.best_root,
                     reward_lower=self.best_value, reward_upper=self.root_upper,
                     gap=(self.root_upper-self.best_value if self.root_upper is not None
                          and self.best_value is not None else None),
                     exact=self.complete, states=self.nodes, root_actions=self.root_actions,
                     root_tested=self.root_tested, feedback_rows=len(self.rows))
        return self.best_root

class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.recoveries = 0

    def recover(self, obs, evidence, legal, used):
        ctx = self.ctx
        ctx.log('recovery_needed', evidence=str(evidence))
        if ctx.done or self.recoveries >= 3:
            return None
        budget = ctx.budget.remaining()
        if budget.get('system2_calls', 0) < 1 or budget.get('wall_s', 0) < 10:
            return None
        self.recoveries += 1
        prompt = ('An exact Wordle decision-tree plan failed verification. Diagnose from the '
                  'public observation and evidence, then return JSON {"guess":"xxxxx"}. '
                  'Choose one unused legal guess; no long explanation. The next action will '
                  'be checked again.\n' + json.dumps({'observation': obs, 'evidence': str(evidence),
                  'progress': ctx.progress(),
                  'unused_examples': [w for w in legal if w not in used][:60]}, ensure_ascii=True))
        try:
            reply = ctx.think(prompt, session_id=self.session,
                              timeout_s=min(300.0, max(1, budget['wall_s']-5)))
            self.session = reply.session_id
            text = reply.text or ''
            try:
                data = json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
                words = [str(data.get('guess', '')).lower()]
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
                guess = self.recover(obs, issue, legal, used)
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
