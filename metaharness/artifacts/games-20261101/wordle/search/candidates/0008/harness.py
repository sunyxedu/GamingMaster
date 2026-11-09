"""Finite-horizon Wordle reward search, with independently verified execution."""
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


def choose_opener(answers, legal, ctx):
    """Pick a first probe from the public lists, scoring its first two turns.

    The score is an attainable finite-horizon bound: an answer guess can win
    immediately (6 reward units), while each distinct non-green feedback bin
    can contribute one win on turn two (5 units).  A lower sum of squared bin
    sizes breaks ties in favour of balanced partitions.  The pool is built
    from letter frequencies in the *current* public answer list, so the
    opener does not encode a seed or a particular answer snapshot.
    """
    if not answers:
        return next(iter(sorted(legal)), "crane")
    freq = Counter(ch for word in answers for ch in set(word))
    pool = set()
    for word in legal:
        if len(word) == 5 and len(set(word)) >= 4:
            vowels = sum(ch in "aeiou" for ch in set(word))
            if 1 <= vowels <= 4:
                pool.add(word)
    # Include familiar, high-coverage probes even if a list variant omits a
    # letter-frequency winner; all are still validated against legal words.
    pool.update(w for w in ("slate", "crane", "soare", "trace", "raise",
                            "stare", "irate", "arise", "roate", "audio")
                if w in legal)
    ranked = sorted(pool, key=lambda w: (-sum(freq.get(c, 0) for c in set(w)), w))[:128]
    answer_set = set(answers)
    best = None
    best_word = answers[0]
    for word in ranked:
        bins = {}
        for answer in answers:
            code = feedback(answer, word)
            if code != 242:
                bins[code] = bins.get(code, 0) + 1
        immediate = 6 if word in answer_set else 0
        key = (immediate + 5 * len(bins), -sum(n * n for n in bins.values()),
               immediate, -len(word), word)
        if best is None or key > best:
            best, best_word = key, word
    if ctx is not None:
        ctx.log("opener_selection", pool=len(ranked), guess=best_word,
                objective=list(best) if best else None)
    return best_word


class RewardSearch:
    """Full action partitions, exact small states, bounded lookahead for large ones.

    Reward units are score * max_turns. Solving with r guesses left gives r.
    V(S,r) = max_g [r * 1(g in S) + sum V(S_feedback,r-1)].
    """
    def __init__(self, candidates, legal, ctx, seconds=45.0):
        self.words = candidates
        self.legal = legal
        self.ctx = ctx
        self.end = time.monotonic() + seconds
        self.rows = []
        self.memo = {}
        self.choices = {}
        self.option_cache = {}
        self.exact = True
        self.best_root = candidates[0]

    def check(self):
        if self.ctx.done or time.monotonic() > self.end:
            raise SearchStopped()
        if self.ctx.budget.remaining().get('wall_s', 1800) < 5:
            raise SearchStopped()

    def prepare(self):
        self.index = {w: 1 << i for i, w in enumerate(self.words)}
        # No lexicographic truncation: every public legal word can be a probe.
        for j, guess in enumerate(self.legal):
            if j % 128 == 0:
                self.check()
            buckets = {}
            for i, answer in enumerate(self.words):
                code = feedback(answer, guess)
                if code != 242:
                    buckets[code] = buckets.get(code, 0) | (1 << i)
            self.rows.append((guess, self.index.get(guess, 0), tuple(buckets.values())))

    def options(self, state):
        if state in self.option_cache:
            return self.option_cache[state]
        n = state.bit_count()
        unique = {}
        for j, (guess, win, partitions) in enumerate(self.rows):
            if j % 256 == 0:
                self.check()
            solved = win & state
            groups = tuple(sorted(b & state for b in partitions if b & state))
            if not solved and len(groups) == 1:
                continue
            key = (solved, groups)
            if key not in unique:
                sizes = [g.bit_count() for g in groups]
                # Optimistic two-step reward deficit (smaller is better).
                # Immediate solve counts twice; each non-green branch counts once.
                merit = 2 * bool(solved) + len(groups)
                unique[key] = (guess, solved, groups, merit,
                               sum(k*k for k in sizes), max(sizes, default=0))
        options = sorted(unique.values(), key=lambda o: (-o[3], o[4], o[5], not o[1], o[0]))
        self.option_cache[state] = options
        return options

    def value(self, state, remaining, depth=0):
        n = state.bit_count()
        if n == 0 or remaining <= 0:
            return 0
        if n == 1 or remaining == 1:
            return remaining
        key = (state, remaining)
        if key in self.memo:
            return self.memo[key]
        self.check()
        options = self.options(state)
        # Two turns remaining has an exact closed form over all legal guesses.
        if remaining == 2:
            chosen = max(options, key=lambda o: (2*bool(o[1]) + len(o[2]), bool(o[1]), -o[4]))
            value = 2*bool(chosen[1]) + len(chosen[2])
            self.memo[key] = value
            self.choices[key] = chosen[0]
            return value
        best, best_guess = -1, None
        # For large states, bounded decision-tree search. All small states are exact
        # unless the explicit wall-clock guard interrupts the entire calculation.
        if n > 12:
            self.exact = False
            widths = 24 if depth == 0 else 6
            # Include low-collision choices as well as optimistic-reward choices.
            shortlisted = options[:widths]
            extra = sorted(options, key=lambda o: (o[4], o[5], not o[1], o[0]))[:6]
            known = {o[0] for o in shortlisted}
            options = shortlisted + [o for o in extra if o[0] not in known]
        for guess, solved, groups, _, _, _ in options:
            self.check()
            # Each branch can solve one answer next turn, all others no earlier
            # than the following turn. This is a rigorous upper reward bound.
            upper = remaining * bool(solved)
            upper += sum((remaining - 1) + (g.bit_count()-1)*(remaining-2) for g in groups)
            if upper <= best:
                continue
            val = remaining * bool(solved)
            # Large recursive branches use an attainable shallow policy after depth 2.
            for child in sorted(groups, key=int.bit_count, reverse=True):
                if depth >= 2 and child.bit_count() > 12:
                    self.exact = False
                    val += self.greedy_value(child, remaining-1)
                else:
                    val += self.value(child, remaining-1, depth+1)
            if val > best:
                best, best_guess = val, guess
                if depth == 0:
                    self.best_root = guess
            # Absolute optimum: solve one now, every other answer next turn.
            if best == remaining + (n-1)*(remaining-1):
                break
        if best < 0:
            best = sum(max(0, remaining-i) for i in range(min(n, remaining)))
            best_guess = self.words[(state & -state).bit_length()-1]
        self.memo[key] = best
        self.choices[key] = best_guess
        return best

    def greedy_value(self, state, remaining):
        if not state or remaining <= 0:
            return 0
        if state.bit_count() <= 12:
            return self.value(state, remaining, 3)
        options = self.options(state)
        chosen = min(options, key=lambda o: (o[4], o[5], not o[1], o[0]))
        return remaining*bool(chosen[1]) + sum(self.greedy_value(c, remaining-1) for c in chosen[2])

    def choose(self, remaining):
        try:
            self.prepare()
            state = (1 << len(self.words))-1
            opts = self.options(state)
            if opts:
                self.best_root = opts[0][0]
            value = self.value(state, remaining)
            self.best_root = self.choices.get((state, remaining), self.best_root)
            self.ctx.log('reward_search', candidates=len(self.words), guess=self.best_root,
                         reward_sum=value, remaining=remaining, exact=self.exact,
                         states=len(self.memo), stopped=False)
        except SearchStopped:
            self.ctx.log('reward_search', candidates=len(self.words), guess=self.best_root,
                         remaining=remaining, exact=False, stopped=True)
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
            return next((w for w in words if w not in used and (not legal or w in legal)), None)
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
                # Adapt the first probe to the announced public answer list.
                guess = choose_opener(candidates, legal, ctx)
            else:
                search = RewardSearch(candidates, [w for w in legal if w not in used], ctx,
                                      seconds=min(60.0, max(1, ctx.budget.remaining()['wall_s']-10)))
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
