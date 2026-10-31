"""Tail-sensitive Wordle policy trees with verified effects and Astra recovery."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict


COLORS = {"gray": 0, "grey": 0, "yellow": 1, "green": 2}
POWERS = (1, 3, 9, 27, 81)
SOLVED = 242


def feedback(answer, guess):
    """Ternary feedback: exact matches consume instances before yellows."""
    left = list(answer)
    code = 0
    green = 0
    for i in range(5):
        if answer[i] == guess[i]:
            code += 2 * POWERS[i]
            green |= 1 << i
            left[i] = ""
    for i in range(5):
        if not (green & (1 << i)) and guess[i] in left:
            code += POWERS[i]
            left[left.index(guess[i])] = ""
    return code


def parse_history(obs):
    """Unknown feedback is an error, never silently interpreted as gray."""
    result = []
    for row in obs.get("history", []):
        guess = str(row.get("guess", "")).strip().lower()
        tiles = row.get("feedback", [])
        if not re.fullmatch(r"[a-z]{5}", guess) or len(tiles) != 5:
            raise ValueError("unrecognized accepted-history row")
        code = 0
        for p, tile in zip(POWERS, tiles):
            if isinstance(tile, dict):
                tile = tile.get("color", tile.get("state"))
            if str(tile).lower() not in COLORS:
                raise ValueError("unknown feedback tile")
            code += p * COLORS[str(tile).lower()]
        result.append((guess, code))
    return result


def load_lists(rules):
    paths = {}
    def walk(x):
        if isinstance(x, dict):
            for key, value in x.items():
                k = str(key).lower()
                if isinstance(value, str) and os.path.isfile(value):
                    if "answer" in k:
                        paths["answers"] = value
                    elif "allowed" in k or "guess" in k:
                        paths["guesses"] = value
                elif isinstance(value, (list, dict)):
                    walk(value)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(rules)
    words = {}
    evidence = []
    hashes = rules.get("wordlist_sha256", {})
    for kind in ("answers", "guesses"):
        path = paths.get(kind)
        if not path:
            evidence.append("missing public " + kind + " path")
            words[kind] = []
            continue
        try:
            with open(path, "rb") as f:
                data = f.read()
            expected = hashes.get(os.path.basename(path))
            if expected and hashlib.sha256(data).hexdigest() != expected:
                evidence.append(kind + " file hash mismatch")
            words[kind] = sorted(set(re.findall(r"(?m)^[a-z]{5}$", data.decode("utf8").lower())))
        except (OSError, UnicodeError) as exc:
            evidence.append(kind + " file unreadable: " + str(exc))
            words[kind] = []
    return words["answers"], sorted(set(words["answers"] + words["guesses"])), evidence


class SearchStopped(Exception):
    pass


class TailSearch:
    """Minimize the entire bad-tail histogram of an attainable decision tree.

    H(S,r)[k] counts answers earning k remaining-turn reward units. Minimize
    H lexicographically: failures first, then late solves, then earlier solves.
    Child histograms add, making this objective time-consistent across feedback
    branches. A completed small state is exact; large states use a bounded beam.
    """
    def __init__(self, ctx, candidates, allowed, seconds=120.0):
        self.ctx = ctx
        self.answers = tuple(candidates)
        self.allowed = tuple(sorted(allowed))
        self.rows = {}
        self.memo = {}
        self.deadline = time.monotonic() + seconds
        self.nodes = 0
        self.best_root = candidates[0]
        self.best_hist = None
        self.complete = False
        self.beamed = False

    def check(self):
        if (self.ctx.done or time.monotonic() >= self.deadline or
                self.ctx.budget.remaining().get("wall_s", 1800) < 10):
            raise SearchStopped()

    def row(self, guess):
        if guess not in self.rows:
            self.rows[guess] = bytes(feedback(a, guess) for a in self.answers)
        return self.rows[guess]

    def groups(self, state, guess):
        row = self.row(guess)
        bins = defaultdict(list)
        for i in state:
            if row[i] != SOLVED:
                bins[row[i]].append(i)
        return tuple(sorted((tuple(v) for v in bins.values()), key=lambda v: (-len(v), v)))

    @staticmethod
    def optimistic(n, r):
        """Best possible histogram without considering letter constraints."""
        h = [0] * (r + 1)
        if not n:
            return tuple(h)
        if r == 0:
            return (n,)
        h[r] = 1
        h[r - 1] = n - 1
        return tuple(h)

    @staticmethod
    def sequential(n, r):
        """Histogram of guessing the candidates in a fixed order."""
        h = [0] * (r + 1)
        h[0] = max(0, n - r)
        for j in range(min(n, r)):
            h[r - j] = 1
        return tuple(h)

    @staticmethod
    def add(base, child):
        for k, count in enumerate(child):
            base[k] += count

    def solve(self, state, r, root=False):
        self.check()
        key = (state, r)
        if key in self.memo:
            return self.memo[key]
        n = len(state)
        first = self.answers[state[0]]
        if n <= 2 or r <= 1:
            return self.sequential(n, r), first
        self.nodes += 1
        current = {self.answers[i] for i in state}
        best, best_guess = self.sequential(n, r), first
        if root:
            self.best_root, self.best_hist = best_guess, best
        absolute_bound = self.optimistic(n, r)
        records = []
        seen = set()
        guesses = sorted(current) + [w for w in self.allowed if w not in current]
        for j, guess in enumerate(guesses):
            if j % 64 == 0:
                self.check()
            groups = self.groups(state, guess)
            win = int(guess in current)
            if not win and len(groups) == 1:
                continue
            bound = [0] * (r + 1)
            bound[r] = win
            for group in groups:
                self.add(bound, self.optimistic(len(group), r - 1))
            bound = tuple(bound)
            if bound >= best:
                continue
            # Singleton and pair children attain this optimistic histogram.
            if r == 2 or all(len(g) <= 2 for g in groups):
                best, best_guess = bound, guess
                if root:
                    self.best_root, self.best_hist = guess, best
                if best == absolute_bound:
                    self.memo[key] = best, best_guess
                    return best, best_guess
                continue
            signature = (win, tuple(sorted(groups)))
            if signature in seen:
                continue
            seen.add(signature)
            sizes = [len(g) for g in groups]
            records.append((bound, max(sizes), sum(k*k for k in sizes),
                            guess, groups, win))
        # Shape only orders work; the histogram decides every retained tree.
        records.sort(key=lambda x: (x[1], x[0], x[2], -x[5], x[3]))
        if n > 12 and len(records) > 64:
            self.beamed = True
            choices = records[:32]
            choices += sorted(records, key=lambda x: (x[0], x[2], x[1], x[3]))[:24]
            choices += sorted(records, key=lambda x: (x[2], x[1], x[0], x[3]))[:12]
            choices += [rec for rec in records if rec[5]][:12]
            included = set()
            records = [x for x in choices if not (x[3] in included or included.add(x[3]))]
        for bound, worst, collisions, guess, groups, win in records:
            self.check()
            if bound >= best:
                continue
            value = [0] * (r + 1)
            value[r] = win
            pending = [0] * (r + 1)
            for group in groups:
                self.add(pending, self.optimistic(len(group), r - 1))
            for group in groups:
                optimistic = self.optimistic(len(group), r - 1)
                for k, count in enumerate(optimistic):
                    pending[k] -= count
                child, _ = self.solve(group, r - 1)
                self.add(value, child)
                if tuple(v+p for v, p in zip(value, pending)) >= best:
                    break
            else:
                if tuple(value) < best:
                    best, best_guess = tuple(value), guess
                    if root:
                        self.best_root, self.best_hist = guess, best
                if best == absolute_bound:
                    break
        self.memo[key] = best, best_guess
        return best, best_guess

    def choose(self, turns):
        try:
            hist, guess = self.solve(tuple(range(len(self.answers))), turns, root=True)
            self.best_root, self.best_hist = guess, hist
            self.complete = True
        except SearchStopped:
            pass
        if self.best_hist is None:
            self.best_hist = self.sequential(len(self.answers), turns)
        self.ctx.log("tail_search", candidates=len(self.answers), guess=self.best_root,
                     reward_histogram=list(self.best_hist),
                     predicted_failures=self.best_hist[0],
                     reward_sum=sum(k*n for k, n in enumerate(self.best_hist)),
                     finished=self.complete, exact=self.complete and not self.beamed,
                     beamed=self.beamed, states=self.nodes, feedback_rows=len(self.rows))
        return self.best_root


def extract_word(text, legal, blocked):
    """Accept JSON, a bare word, or a short prose reply; never take substrings."""
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    candidates = None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            candidates = [str(obj.get("guess", "")).strip().lower()]
    except (ValueError, TypeError):
        pass
    if candidates is None:
        if legal:
            candidates = re.findall(r"\b[a-z]{5}\b", raw.lower())
        else:
            # Without a public dictionary, require an explicit complete action.
            match = re.fullmatch(r"(?:guess\s+)?([a-z]{5})", raw.lower())
            candidates = [match.group(1)] if match else []
    return next((w for w in candidates if re.fullmatch(r"[a-z]{5}", w)
                 and w not in blocked and (not legal or w in legal)), None)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.recoveries = 0

    def recover(self, obs, evidence, candidates, legal, blocked):
        ctx = self.ctx
        if ctx.done or self.recoveries >= 3:
            return None
        remaining = ctx.budget.remaining()
        if remaining.get("system2_calls", 0) <= 0 or remaining.get("wall_s", 0) < 230:
            return None
        self.recoveries += 1
        payload = {"observation": obs, "failure": evidence,
                   "progress": ctx.progress(), "consistent_candidates": candidates[:100],
                   "unused_legal_examples": sorted(legal - blocked)[:60],
                   "blocked": sorted(blocked)}
        prompt = (
            "Re-plan a failed Wordle solver from the observation and verification evidence. "
            "The target is unknown. Normal rules use green-first, duplicate-aware feedback. "
            "Choose one unused legal guess that can advance this state. If candidates are "
            "consistent, prefer them unless a probe separates a difficult family. "
            "Return only JSON {\"guess\":\"xxxxx\"}; do not output a sequence.\n" +
            json.dumps(payload, ensure_ascii=True)
        )
        try:
            result = ctx.think(prompt, session_id=self.session, timeout_s=220.0)
            self.session = result.session_id
            word = extract_word(result.text, legal, blocked)
            ctx.log("recovery", evidence=evidence, guess=word)
            return word
        except (TimeoutError, RuntimeError) as exc:
            ctx.log("recovery_call_failed", error=type(exc).__name__)
            return None

    def run_episode(self, game):
        ctx = self.ctx
        obs = ctx.observe()
        if not isinstance(obs, dict):
            return
        rules = obs.get("rules", {})
        answers, allowed, setup_errors = load_lists(rules)
        legal = set(allowed)
        blocked = set()
        if rules.get("word_length", 5) != 5 or rules.get("hard_mode", False):
            setup_errors.append("unsupported rule variant")
        evidence = setup_errors or None
        for _ in range(18):  # bounded even for inconsistent/rejected observations
            if ctx.done or obs.get("done") or obs.get("solved"):
                return
            try:
                hist = parse_history(obs)
            except (ValueError, TypeError, AttributeError) as exc:
                hist = []
                evidence = ["history parse failure: " + str(exc)]
            used = {w for w, _ in hist}
            blocked.update(used)
            candidates = [a for a in answers if a not in blocked and
                          all(feedback(a, g) == code for g, code in hist)]
            turn = int(obs.get("turn", len(hist)))
            turns = int(obs.get("max_turns", 6)) - turn
            if turns <= 0:
                return
            if len(hist) != turn:
                evidence = ["accepted turn count differs from parsed history"]
            if not candidates:
                evidence = evidence or ["no answer matches the full public history"]
            if evidence:
                guess = self.recover(obs, evidence, candidates, legal, blocked)
                # After giving Astra the failure, use only a verified fallback.
                if not guess:
                    guess = next(iter(candidates), None)
                if not guess:
                    return
            elif not hist:
                # Retain parent 0004 opener to isolate the new tree objective.
                guess = "slate" if "slate" in legal else candidates[0]
            elif len(candidates) <= 2 or turns == 1:
                guess = candidates[0]
            else:
                seconds = min(120.0, max(0.1, (ctx.budget.remaining()["wall_s"] - 250) / max(turns, 1)))
                search = TailSearch(ctx, candidates, legal - blocked, seconds)
                guess = search.choose(turns)
            if ctx.done:
                return
            if guess in blocked or (legal and guess not in legal):
                evidence = ["chosen action failed legality/repetition verification", guess]
                blocked.add(guess)
                continue
            before = ctx.progress()
            previous = obs
            blocked.add(guess)
            try:
                obs = ctx.act("guess " + guess)
                if obs is None:
                    obs = ctx.observe()
            except Exception as exc:
                if ctx.done:
                    return
                evidence = ["action raised", type(exc).__name__, str(exc), guess]
                obs = ctx.observe()
                continue
            after = ctx.progress()
            evidence = []
            if not isinstance(obs, dict):
                return
            if obs.get("error"):
                evidence.append("action error: " + str(obs["error"]))
            if after.get("illegal_actions", 0) > before.get("illegal_actions", 0):
                evidence.append("progress illegal_actions increased")
            if after.get("since_obs_change", 0) > 0:
                evidence.append("progress reports no observation change")
            if obs.get("turn") != turn + 1:
                evidence.append("accepted turn did not advance by exactly one")
            try:
                new_hist = parse_history(obs)
                if new_hist[:-1] != hist or not new_hist or new_hist[-1][0] != guess:
                    evidence.append("accepted history did not append exactly the action")
                if candidates and new_hist:
                    survivors = [a for a in candidates if feedback(a, guess) == new_hist[-1][1]]
                    if not survivors:
                        evidence.append("observed feedback was outside the predicted partition")
                    if len(candidates) > 1 and len(survivors) >= len(candidates) and not obs.get("done"):
                        evidence.append("the plan failed to reduce the answer set")
            except (ValueError, TypeError, AttributeError) as exc:
                evidence.append("new history unreadable: " + str(exc))
            ctx.memory["wordle_verification"] = {"turn": turn, "guess": guess,
                                                   "candidates_before": len(candidates),
                                                   "evidence": evidence, "progress": after}
            if evidence:
                ctx.log("verification_failure", evidence=evidence, before=before, after=after)


def build(ctx):
    return Harness(ctx)
