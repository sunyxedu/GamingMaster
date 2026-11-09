"""Adaptive Wordle harness: public-list opener lookahead and verified play."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter

POW = (1, 3, 9, 27, 81)
SOLVED = 242
COLORS = {"gray": 0, "grey": 0, "yellow": 1, "green": 2,
          "0": 0, "1": 1, "2": 2}


def feedback(answer: str, guess: str) -> int:
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
    return sum(v * p for v, p in zip(tiles, POW))


def parse_history(obs):
    out = []
    for row in obs.get("history", []):
        if not isinstance(row, dict):
            raise ValueError("history row is not an object")
        word = row.get("guess", row.get("word", ""))
        if isinstance(word, dict):
            word = word.get("guess", "")
        word = str(word).strip().lower().removeprefix("guess ").strip()
        if not re.fullmatch(r"[a-z]{5}", word):
            raise ValueError("bad accepted guess")
        tiles = row.get("feedback", row.get("tiles", []))
        if isinstance(tiles, str):
            tiles = re.findall(r"green|yellow|gray|grey", tiles.lower())
        if not isinstance(tiles, (list, tuple)) or len(tiles) != 5:
            raise ValueError("bad feedback length")
        code = 0
        for tile, power in zip(tiles, POW):
            if isinstance(tile, dict):
                tile = tile.get("color", tile.get("state", tile.get("value", "")))
            key = str(tile).strip().lower()
            if key not in COLORS:
                raise ValueError("unknown feedback tile")
            code += COLORS[key] * power
        out.append((word, code))
    return out


def load_lists(rules):
    paths = {}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                kl = str(k).lower()
                if isinstance(v, str) and os.path.isfile(v):
                    if "answer" in kl:
                        paths["answers"] = v
                    elif "allowed" in kl or "guess" in kl:
                        paths["guesses"] = v
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(rules)
    result = {}
    errors = []
    hashes = rules.get("wordlist_sha256", {}) if isinstance(rules, dict) else {}
    for kind in ("answers", "guesses"):
        path = paths.get(kind)
        if not path:
            errors.append("missing " + kind + " list")
            result[kind] = []
            continue
        try:
            data = open(path, "rb").read()
            expected = hashes.get(os.path.basename(path))
            if expected and hashlib.sha256(data).hexdigest() != expected:
                errors.append(kind + " list hash mismatch")
            result[kind] = sorted({w for w in data.decode("utf8").lower().splitlines()
                                   if re.fullmatch(r"[a-z]{5}", w)})
        except (OSError, UnicodeError) as exc:
            errors.append(kind + " list unreadable: " + str(exc))
            result[kind] = []
    legal = sorted(set(result["answers"]) | set(result["guesses"]))
    return result["answers"], legal, errors


class StopSearch(Exception):
    pass


class Policy:
    """Short-horizon expected-reward policy using only public words.

    The first action is selected by a shallow two-step continuation estimate;
    later actions use all surviving candidates and exact duplicate-aware bins.
    """
    def __init__(self, answers, legal, ctx, seconds=35.0):
        self.answers = tuple(answers)
        self.legal = tuple(legal)
        self.ctx = ctx
        self.end = time.monotonic() + seconds
        self.rows = {}

    def check(self):
        if self.ctx.done or time.monotonic() >= self.end:
            raise StopSearch()
        if self.ctx.budget.remaining().get("wall_s", 1800) < 8:
            raise StopSearch()

    def row(self, guess, words=None):
        words = tuple(self.answers if words is None else words)
        key = (guess, words)
        if key not in self.rows:
            self.rows[key] = tuple(feedback(a, guess) for a in words)
        return self.rows[key]

    def bins(self, words, guess):
        groups = {}
        for a, code in zip(words, self.row(guess, words)):
            if code != SOLVED:
                groups.setdefault(code, []).append(a)
        return tuple(groups.values())

    @staticmethod
    def one_step(words, guess, turns):
        groups = Policy._groups_static(words, guess)
        return turns * (guess in words) + (turns - 1) * len(groups)

    @staticmethod
    def sequential_value(n, turns):
        """Reward from guessing n candidates in a fixed order."""
        k = min(max(0, n), max(0, turns))
        return k * turns - k * (k - 1) // 2

    @staticmethod
    def _groups_static(words, guess):
        groups = {}
        for a in words:
            code = feedback(a, guess)
            if code != SOLVED:
                groups.setdefault(code, []).append(a)
        return tuple(groups.values())

    def continuation_score(self, words, turns):
        """Attainable two-turn estimate, with a candidate answer as probe."""
        if not words or turns <= 0:
            return 0
        if turns == 1:
            return 1
        best = -1
        # Candidate-only tails are deliberate: they are legal and cheap, while
        # the actual first move still scans a broad public probe pool.
        for i, guess in enumerate(words):
            if i % 32 == 0:
                self.check()
            groups = self.bins(words, guess)
            val = turns + sum(self.sequential_value(len(g), turns - 1)
                              for g in groups)
            if val > best:
                best = val
        return max(best, turns)

    def opener(self):
        answers = self.answers
        if not answers:
            return next(iter(self.legal), "crane")
        freq = Counter(ch for a in answers for ch in set(a))
        pool = [w for w in self.legal if len(set(w)) >= 4]
        pool.sort(key=lambda w: (-sum(freq[c] for c in set(w)), w))
        pool = pool[:48]
        # Keep familiar probes in the pool when legal, without assuming one is
        # correct for a particular seed.
        pool = list(dict.fromkeys(pool + [w for w in
                     ("slate", "crane", "soare", "trace", "raise", "stare")
                     if w in self.legal]))
        # First use a cheap partition screen.  Evaluating a deep tail for all
        # 50 probes can spend the entire episode before the familiar probes
        # are considered; only the strongest dozen receive lookahead.
        screened = []
        for guess in pool:
            groups = self.bins(answers, guess)
            sizes = [len(g) for g in groups]
            screened.append((6 * (guess in answers) + 5 * len(groups),
                             -sum(n * n for n in sizes), guess))
        pool = [x[2] for x in sorted(screened, reverse=True)[:12]]
        best_key = None
        best = answers[0]
        for idx, guess in enumerate(pool):
            if idx % 4 == 0:
                self.check()
            groups = self.bins(answers, guess)
            # Immediate win plus a best candidate-tail estimate in every bin.
            value = 6 * (guess in answers)
            for group in groups:
                value += self.continuation_score(tuple(group), 5)
            sizes = [len(g) for g in groups]
            key = (value, -sum(n * n for n in sizes), 6 * (guess in answers), -len(groups), guess)
            if best_key is None or key > best_key:
                best_key, best = key, guess
        self.ctx.log("adaptive_opener", pool=len(pool), guess=best,
                     estimate=list(best_key) if best_key else None)
        return best

    def choose(self, words, turns):
        if not words:
            return None
        if len(words) == 1 or turns <= 1:
            return words[0]
        # Probe candidates only in continuations. The opener is the broad
        # all-public scan; this keeps every later decision below the wall guard.
        if len(words) <= 30:
            memo = {}
            def exact(state, r):
                state = tuple(state)
                if not state or r <= 0:
                    return 0, state[0] if state else None
                if len(state) == 1 or r == 1:
                    return r, state[0]
                key = (state, r)
                if key in memo:
                    return memo[key]
                best = (-1, state[0])
                for guess in state:
                    groups = self.bins(state, guess)
                    val = r + sum(exact(tuple(g), r - 1)[0] for g in groups)
                    if val > best[0]:
                        best = (val, guess)
                memo[key] = best
                return best
            try:
                value, guess = exact(tuple(words), turns)
                self.ctx.log("exact_tail_choice", candidates=len(words), turns=turns,
                             guess=guess, reward=value)
                return guess
            except RecursionError:
                pass
        best = None
        best_key = None
        options = tuple(words)
        for i, guess in enumerate(options):
            if i % 24 == 0:
                self.check()
            groups = self.bins(words, guess)
            # One-step expected reward, with a balanced partition tie-break.
            value = turns * (guess in words) + sum(
                self.sequential_value(len(g), turns - 1) for g in groups)
            sizes = [len(g) for g in groups]
            key = (value, -sum(n*n for n in sizes), -max(sizes, default=0), guess)
            if best_key is None or key > best_key:
                best_key, best = key, guess
        self.ctx.log("continuation_choice", candidates=len(words), turns=turns,
                     guess=best, estimate=list(best_key) if best_key else None)
        return best


def extract_word(text, legal, blocked):
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    found = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            found = [str(obj.get("guess", "")).strip().lower()]
    except (ValueError, TypeError):
        found = re.findall(r"\b[a-z]{5}\b", raw.lower()) if legal else []
    return next((w for w in found if re.fullmatch(r"[a-z]{5}", w)
                 and w not in blocked and (not legal or w in legal)), None)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.recoveries = 0

    def recover(self, obs, evidence, candidates, legal, blocked):
        if self.ctx.done or self.recoveries >= 2:
            return None
        rem = self.ctx.budget.remaining()
        if rem.get("system2_calls", 0) <= 0 or rem.get("wall_s", 0) < 230:
            return None
        self.recoveries += 1
        payload = {"observation": obs, "evidence": evidence,
                   "progress": self.ctx.progress(), "candidates": candidates[:80],
                   "unused": sorted(legal - blocked)[:50]}
        prompt = ("Recover a Wordle solver after a verified contradiction. Choose one unused "
                  "legal five-letter guess. Return only JSON {\"guess\":\"xxxxx\"}.\n" +
                  json.dumps(payload, ensure_ascii=True))
        try:
            result = self.ctx.think(prompt, session_id=self.session, timeout_s=220.0)
            self.session = result.session_id
            word = extract_word(result.text, legal, blocked)
            self.ctx.log("recovery", evidence=evidence, guess=word)
            return word
        except (TimeoutError, RuntimeError) as exc:
            self.ctx.log("recovery_failed", error=type(exc).__name__)
            return None

    def run_episode(self, game):
        ctx = self.ctx
        obs = ctx.observe()
        if not isinstance(obs, dict):
            return
        answers, allowed, setup = load_lists(obs.get("rules", {}))
        legal = set(allowed)
        blocked = set()
        first = True
        evidence = setup[:]
        for _ in range(12):
            if ctx.done or obs.get("done") or obs.get("solved"):
                return
            try:
                hist = parse_history(obs)
            except (ValueError, TypeError, AttributeError) as exc:
                hist = []
                evidence = ["history parse failure: " + str(exc)]
            blocked.update(g for g, _ in hist)
            candidates = [a for a in answers if a not in blocked and
                          all(feedback(a, g) == code for g, code in hist)]
            turn = int(obs.get("turn", len(hist)))
            turns = int(obs.get("max_turns", 6)) - turn
            if turns <= 0:
                return
            if len(hist) != turn:
                evidence.append("turn/history mismatch")
            if not candidates:
                evidence.append("no candidates remain")
            try:
                if evidence:
                    guess = self.recover(obs, evidence, candidates, legal, blocked)
                    if not guess:
                        guess = next((a for a in candidates if a not in blocked), None)
                elif first:
                    guess = Policy(answers, legal - blocked, ctx, 30.0).opener()
                else:
                    policy = Policy(candidates, legal - blocked, ctx, 25.0)
                    guess = policy.choose(tuple(candidates), turns)
            except StopSearch:
                guess = next((a for a in candidates if a not in blocked), None)
            if not guess or guess in blocked or (legal and guess not in legal):
                evidence = ["no verified legal unused guess"]
                if not candidates:
                    return
                continue
            before = ctx.progress()
            old = obs
            blocked.add(guess)
            try:
                obs = ctx.act("guess " + guess)
                if obs is None:
                    obs = ctx.observe()
            except Exception as exc:
                if ctx.done:
                    return
                evidence = ["action raised", type(exc).__name__, str(exc)]
                obs = ctx.observe()
                continue
            after = ctx.progress()
            evidence = []
            if not isinstance(obs, dict):
                return
            if obs.get("error"):
                evidence.append("action error: " + str(obs["error"]))
            if after.get("illegal_actions", 0) > before.get("illegal_actions", 0):
                evidence.append("illegal action counter increased")
            if after.get("since_obs_change", 0) > 0:
                evidence.append("observation did not change")
            if obs.get("turn") != turn + 1:
                evidence.append("turn did not advance")
            try:
                nh = parse_history(obs)
                if nh[:-1] != hist or not nh or nh[-1][0] != guess:
                    evidence.append("history did not append guess")
                if candidates and nh:
                    survivors = [a for a in candidates if feedback(a, guess) == nh[-1][1]]
                    if not survivors:
                        evidence.append("feedback outside predicted partition")
            except (ValueError, TypeError, AttributeError) as exc:
                evidence.append("new history unreadable: " + str(exc))
            ctx.memory["wordle_verification"] = {"guess": guess, "turn": turn,
                "candidates": len(candidates), "evidence": evidence, "progress": after}
            if evidence:
                ctx.log("verification_failure", evidence=evidence, before=before, after=after)
            first = False


def build(ctx):
    return Harness(ctx)
