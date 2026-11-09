"""Wordle finite-horizon reward search with tail-aware tie breaking and recovery."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict

COLORS = {"gray": 0, "grey": 0, "yellow": 1, "green": 2}
POWERS = (1, 3, 9, 27, 81)
SOLVED = 242


def feedback(answer, guess):
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
            name = str(tile).lower()
            if name not in COLORS:
                raise ValueError("unknown feedback tile")
            code += p * COLORS[name]
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
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)
    walk(rules)
    words, evidence = {}, []
    hashes = rules.get("wordlist_sha256", {}) if isinstance(rules, dict) else {}
    for kind in ("answers", "guesses"):
        path = paths.get(kind)
        if not path:
            words[kind] = []
            evidence.append("missing public " + kind + " path")
            continue
        try:
            data = open(path, "rb").read()
            expected = hashes.get(os.path.basename(path))
            if expected and hashlib.sha256(data).hexdigest() != expected:
                evidence.append(kind + " file hash mismatch")
            words[kind] = sorted(set(re.findall(r"(?m)^[a-z]{5}$", data.decode("utf8").lower())))
        except (OSError, UnicodeError) as exc:
            words[kind] = []
            evidence.append(kind + " file unreadable: " + str(exc))
    return words["answers"], sorted(set(words["answers"] + words["guesses"])), evidence


class SearchStopped(Exception):
    pass


class RewardSearch:
    """Bellman reward search; equal reward trees prefer earlier solves."""
    def __init__(self, ctx, candidates, allowed, seconds=105.0):
        self.ctx = ctx
        self.answers = tuple(candidates)
        self.allowed = tuple(sorted(allowed))
        self.rows, self.memo = {}, {}
        self.deadline = time.monotonic() + seconds
        self.nodes = 0
        self.best_root = self.answers[0]
        self.best_value = None
        self.best_hist = None
        self.complete = False

    def check(self):
        if self.ctx.done or time.monotonic() >= self.deadline:
            raise SearchStopped()
        if self.ctx.budget.remaining().get("wall_s", 1800) < 8:
            raise SearchStopped()

    def row(self, guess):
        if guess not in self.rows:
            self.rows[guess] = bytes(feedback(a, guess) for a in self.answers)
        return self.rows[guess]

    def groups(self, state, guess):
        bins = defaultdict(list)
        row = self.row(guess)
        for i in state:
            if row[i] != SOLVED:
                bins[row[i]].append(i)
        return tuple(sorted((tuple(v) for v in bins.values()), key=lambda g: (-len(g), g)))

    @staticmethod
    def upper(n, r):
        if n <= 0 or r <= 0:
            return 0
        return 1 if r == 1 else (r - 1) * n + 1

    @staticmethod
    def hist_reward(hist):
        return sum(i * n for i, n in enumerate(hist))

    def solve(self, state, r, root=False):
        self.check()
        key = (state, r)
        if key in self.memo:
            return self.memo[key]
        n = len(state)
        first = self.answers[state[0]]
        if n == 1:
            return r, first, (0,) * r + (1,)
        if r <= 1:
            return 1, first, (0, 1)
        if n == 2:
            return 2 * r - 1, first, (0,) * (r - 1) + (1, 1)
        current = set(self.answers[i] for i in state)
        # A sequential candidate policy is always an attainable incumbent.
        best = sum(max(r - j, 0) for j in range(n))
        best_guess = first
        best_hist = [0] * (r + 1)
        for j in range(min(n, r)):
            best_hist[r - j] += 1
        if root:
            self.best_root, self.best_value, self.best_hist = best_guess, best, tuple(best_hist)
        absolute = self.upper(n, r)
        records, seen = [], set()
        guesses = sorted(current) + [w for w in self.allowed if w not in current]
        for j, guess in enumerate(guesses):
            if j % 128 == 0:
                self.check()
            groups = self.groups(state, guess)
            immediate = r if guess in current else 0
            if not immediate and len(groups) == 1:
                continue
            bound = immediate + sum(self.upper(len(g), r - 1) for g in groups)
            if bound < best:
                continue
            signature = (immediate, tuple(sorted(groups)))
            if signature in seen:
                continue
            seen.add(signature)
            sizes = [len(g) for g in groups]
            records.append((bound, sum(k * k for k in sizes), max(sizes, default=0), guess, groups, immediate))
        records.sort(key=lambda x: (-x[0], x[1], x[2], -x[5], x[3]))
        if n > 12:
            choices = records[:36]
            choices += sorted(records, key=lambda x: (x[1], -x[0], x[2], x[3]))[:18]
            choices += sorted(records, key=lambda x: (x[2], -x[0], x[1], x[3]))[:10]
            choices += [x for x in records if x[3] in current][:12]
            included = set()
            records = [x for x in choices if not (x[3] in included or included.add(x[3]))]
        for bound, _, _, guess, groups, immediate in records:
            self.check()
            if bound < best:
                continue
            value, hist = immediate, [0] * (r + 1)
            if immediate:
                hist[r] += 1
            remaining_bound = bound - immediate
            for group in groups:
                remaining_bound -= self.upper(len(group), r - 1)
                child, _, child_hist = self.solve(group, r - 1)
                value += child
                for k, count in enumerate(child_hist):
                    if k < len(hist):
                        hist[k] += count
                if value + remaining_bound < best:
                    break
            else:
                # Expected reward is primary. Histogram is a deterministic
                # tail tie-break only, avoiding the leader's mean-score loss.
                candidate_key = (value, tuple(hist[1:]))
                best_key = (best, tuple(best_hist[1:]))
                if candidate_key > best_key:
                    best, best_guess, best_hist = value, guess, hist
                    if root:
                        self.best_root, self.best_value, self.best_hist = guess, value, tuple(hist)
                if best == absolute:
                    break
        self.nodes += 1
        result = (best, best_guess, tuple(best_hist))
        self.memo[key] = result
        return result

    def choose(self, turns):
        try:
            value, guess, hist = self.solve(tuple(range(len(self.answers))), turns, root=True)
            self.best_root, self.best_value, self.best_hist = guess, value, hist
            self.complete = True
        except SearchStopped:
            pass
        self.ctx.log("reward_search", candidates=len(self.answers), guess=self.best_root,
                     total_reward=self.best_value, reward_histogram=list(self.best_hist or ()),
                     search_finished=self.complete, states=self.nodes, feedback_rows=len(self.rows))
        return self.best_root


def extract_word(text, legal, blocked):
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    candidates = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            candidates.append(str(obj.get("guess", "")).strip().lower())
    except (ValueError, TypeError):
        pass
    if legal:
        candidates += re.findall(r"\b[a-z]{5}\b", raw.lower())
    else:
        match = re.fullmatch(r"(?:guess\s+)?([a-z]{5})", raw.lower())
        if match:
            candidates.append(match.group(1))
    return next((w for w in candidates if re.fullmatch(r"[a-z]{5}", w)
                 and w not in blocked and (not legal or w in legal)), None)


class Harness:
    def __init__(self, ctx):
        self.ctx, self.session, self.recoveries = ctx, None, 0

    def recover(self, obs, evidence, candidates, legal, blocked):
        ctx = self.ctx
        if ctx.done or self.recoveries >= 3:
            return None
        rem = ctx.budget.remaining()
        if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 230:
            return None
        self.recoveries += 1
        payload = {"observation": obs, "failure": evidence, "progress": ctx.progress(),
                   "consistent_candidates": candidates[:100],
                   "unused_legal_examples": sorted(legal - blocked)[:60],
                   "blocked": sorted(blocked)}
        prompt = ("Re-plan a failed Wordle solver from this observation and verification evidence. "
                  "Choose one unused legal five-letter guess. Return only JSON "
                  "{\"guess\":\"xxxxx\"}; do not output a sequence.\n" +
                  json.dumps(payload, ensure_ascii=True))
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
        legal, blocked = set(allowed), set()
        if rules.get("word_length", 5) != 5 or rules.get("hard_mode", False):
            setup_errors.append("unsupported rule variant")
        for _ in range(18):
            if ctx.done or obs.get("done") or obs.get("solved"):
                return
            evidence = list(setup_errors)
            try:
                hist = parse_history(obs)
            except (ValueError, TypeError, AttributeError) as exc:
                hist, evidence = [], evidence + ["history parse failure: " + str(exc)]
            used = {w for w, _ in hist}
            blocked.update(used)
            candidates = [a for a in answers if a not in blocked and all(feedback(a, g) == code for g, code in hist)]
            turn = int(obs.get("turn", len(hist)))
            turns = int(obs.get("max_turns", 6)) - turn
            if turns <= 0:
                return
            if len(hist) != turn:
                evidence.append("accepted turn count differs from parsed history")
            if not candidates:
                evidence = evidence or ["no answer matches the full public history"]
            if evidence:
                guess = self.recover(obs, evidence, candidates, legal, blocked) or next(iter(candidates), None)
                if not guess:
                    return
            elif not hist:
                guess = "slate" if "slate" in legal else candidates[0]
            elif len(candidates) <= 2 or turns == 1:
                guess = candidates[0]
            else:
                seconds = min(105.0, max(0.1, (ctx.budget.remaining().get("wall_s", 1800) - 250) / max(turns, 1)))
                guess = RewardSearch(ctx, candidates, legal - blocked, seconds).choose(turns)
            if ctx.done:
                return
            if guess in blocked or (legal and guess not in legal):
                evidence = ["chosen action failed legality/repetition verification", guess]
                blocked.add(guess)
                continue
            before = ctx.progress()
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
