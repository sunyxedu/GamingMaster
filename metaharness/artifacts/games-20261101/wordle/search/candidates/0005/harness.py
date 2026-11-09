"""Expected-case Wordle solver with verified fallback recovery."""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict

try:
    from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
except Exception:  # preflight stubs may not expose the contract classes
    BudgetExceeded = EpisodeOver = ModelUnavailable = Exception


def feedback(answer: str, guess: str):
    """Duplicate-aware Wordle feedback, greens consumed before yellows."""
    out = ["gray"] * 5
    left = Counter(answer)
    for i in range(5):
        if answer[i] == guess[i]:
            out[i] = "green"
            left[guess[i]] -= 1
    for i, ch in enumerate(guess):
        if out[i] == "gray" and left[ch] > 0:
            out[i] = "yellow"
            left[ch] -= 1
    return tuple(out)


def _paths(value):
    found = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str) and os.path.isfile(v):
                    key = k.lower()
                    if "answer" in key or "allowed" in key or "guess" in key:
                        found.append((key, v))
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(value)
    ans = next((p for k, p in found if "answer" in k), None)
    allowed = next((p for k, p in found if "allowed" in k or "guess" in k), None)
    return ans, allowed


def _read(path):
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return sorted({s.strip().lower() for s in fh
                           if len(s.strip()) == 5 and s.strip().isalpha()})
    except Exception:
        return []


def _tile(value):
    if isinstance(value, dict):
        for key in ("color", "state", "status", "tile", "value"):
            if key in value:
                return _tile(value[key])
    text = str(value).lower()
    if "yellow" in text:
        return "yellow"
    if "green" in text:
        return "green"
    if "gray" in text or "grey" in text:
        return "gray"
    return None


def _history(obs):
    rows = obs.get("history", []) if isinstance(obs, dict) else []
    result = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            word = row.get("guess", row.get("word", row.get("action", "")))
            tiles = row.get("feedback", row.get("tiles", row.get("result", [])))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            word, tiles = row[0], row[1]
        else:
            continue
        if isinstance(word, dict):
            word = word.get("guess", word.get("value", ""))
        word = str(word).strip().lower()
        if word.startswith("guess "):
            word = word.split(None, 1)[1].strip()
        if isinstance(tiles, str):
            tiles = re.findall(r"green|yellow|gray|grey", tiles.lower())
        if isinstance(tiles, (list, tuple)):
            norm = tuple(_tile(x) for x in tiles)
            if re.fullmatch(r"[a-z]{5}", word) and len(norm) == 5 and None not in norm:
                result.append((word, norm))
    return result


def _word_from_model(text, legal):
    legal = set(legal)
    for word in re.findall(r"\b[a-z]{5}\b", str(text or "").lower()):
        if word in legal:
            return word
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _recover(self, obs, evidence, legal):
        ctx = self.ctx
        if ctx.done:
            return None
        prompt = (
            "Wordle solver recovery. The exact candidate computation could not be verified. "
            "Return one unused legal five-letter guess only, lowercase and without punctuation.\n"
            + json.dumps({"observation": obs, "evidence": str(evidence),
                          "legal_examples": sorted(legal)[:80]}, ensure_ascii=True)
        )
        try:
            answer = ctx.think(prompt, timeout_s=220.0)
            return _word_from_model(getattr(answer, "text", ""), legal)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            return None

    @staticmethod
    def _rank_probes(answers, allowed):
        """Static frequency rank keeps the early broad pool cheap and reproducible."""
        freq = Counter(ch for word in answers for ch in set(word))
        return sorted(allowed, key=lambda w: (-sum(freq[c] for c in set(w)), w))

    def _choose(self, candidates, pool, turns_left):
        if not candidates:
            return None
        if len(candidates) == 1 or turns_left <= 1:
            return candidates[0]
        probes = pool
        n = len(candidates)
        best = None
        best_key = None
        for i, guess in enumerate(probes):
            if i % 16 == 0 and self.ctx.done:
                return best or candidates[0]
            bins = defaultdict(int)
            for answer in candidates:
                bins[feedback(answer, guess)] += 1
            expected = sum(count * count for count in bins.values()) / float(n)
            worst = max(bins.values())
            # Prefer a possible answer after the information terms tie.
            key = (expected, worst, 0 if guess in candidates else 1, guess)
            if turns_left == 2:
                # Every non-green feedback bin can yield one final-turn win;
                # an immediate correct guess earns twice that reward.
                key = (-(len(bins) + int(guess in candidates)),) + key
            if best_key is None or key < best_key:
                best_key, best = key, guess
        return best

    def _endgame(self, candidates, legal, turns_left):
        """Exact finite-horizon reward search for <= 10 possible answers.

        Value is total score numerator across a uniform set of answers. A win
        now earns turns_left; a loss earns zero. Partition-equivalent probes
        are merged, keeping the green branch distinct because it terminates.
        """
        if len(candidates) <= 2 or turns_left <= 1:
            return candidates[0]
        deadline = time.monotonic() + min(45.0, max(0.0, self.ctx.budget.remaining()["wall_s"] - 250))
        green = ("green",) * 5
        actions = {}
        for i, guess in enumerate(legal):
            if i % 64 == 0 and (self.ctx.done or time.monotonic() >= deadline):
                return None
            bins = defaultdict(int)
            win = 0
            for j, answer in enumerate(candidates):
                code = feedback(answer, guess)
                if code == green:
                    win = 1 << j
                else:
                    bins[code] |= 1 << j
            parts = tuple(sorted(bins.values()))
            actions.setdefault((win, parts), guess)
        root_actions = [(win, parts, word) for (win, parts), word in actions.items()]
        memo = {}
        visits = 0

        def solve(mask, left):
            nonlocal visits
            visits += 1
            if visits % 128 == 0 and (self.ctx.done or time.monotonic() >= deadline):
                raise TimeoutError("bounded endgame search")
            n = mask.bit_count()
            if left <= 0:
                return 0, None
            first = candidates[(mask & -mask).bit_length() - 1]
            if n == 1 or left == 1:
                return left, first
            if n == 2:
                return 2 * left - 1, first
            key = mask, left
            if key in memo:
                return memo[key]
            options = {}
            for win, parts, word in root_actions:
                w = bool(win & mask)
                sub = tuple(sorted(p & mask for p in parts if p & mask))
                if not w and sub == (mask,):
                    continue
                options.setdefault((w, sub), word)
            ranked = sorted(options.items(), key=lambda item: (
                -(int(item[0][0]) * left + len(item[0][1]) * (left - 1)),
                sum(p.bit_count() ** 2 for p in item[0][1]), item[1]))
            best_value, best_word = -1, first
            for (w, parts), word in ranked:
                upper = int(w) * left + (n - int(w)) * (left - 1)
                if upper <= best_value:
                    continue
                value = int(w) * left
                if left == 2:
                    value += len(parts)
                else:
                    pending = n - int(w)
                    for p in sorted(parts, key=int.bit_count, reverse=True):
                        value += solve(p, left - 1)[0]
                        pending -= p.bit_count()
                        if value + pending * (left - 1) <= best_value:
                            break
                if value > best_value:
                    best_value, best_word = value, word
                if best_value == left + (n - 1) * (left - 1):
                    break
            memo[key] = best_value, best_word
            return memo[key]

        try:
            value, word = solve((1 << len(candidates)) - 1, turns_left)
            self.ctx.log("endgame", candidates=len(candidates), turns_left=turns_left,
                         total_reward_numerator=value, guess=word, states=len(memo))
            return word
        except TimeoutError:
            return None

    def run_episode(self, game) -> None:
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        rules = obs.get("rules", {}) if isinstance(obs, dict) else {}
        answer_path, allowed_path = _paths(rules)
        answers = _read(answer_path)
        allowed = set(_read(allowed_path)) | set(answers)
        list_problem = None
        if not answers:
            list_problem = "public answer file was unreadable or empty"
            answers = sorted(allowed)
        if not allowed:
            # Recovery gets the complete rules. These are merely legal-word
            # suggestions, never represented as an exhaustive answer list.
            allowed = {"crane", "slate", "trace", "arise", "cigar", "rebut"}
        ranked = self._rank_probes(answers, allowed)
        answer_set = set(answers)
        opener_pool = answers + [w for w in ranked if w not in answer_set][:500]
        opener = (self._choose(answers, opener_pool, 6) if answers else None) or ranked[0]
        used, rejected = set(), set()
        evidence = list_problem
        failures = 0

        while not ctx.done and not game.done:
            obs = game.observe()
            hist = _history(obs)
            used.update(word for word, _ in hist)
            if obs.get("done") or obs.get("solved") or obs.get("result", {}).get("solved"):
                break
            turn = int(obs.get("turn", len(hist)))
            limit = int(obs.get("max_turns", 6))
            if turn >= limit:
                break
            candidates = [a for a in answers
                          if all(feedback(a, g) == tile for g, tile in hist)]
            available = allowed - used - rejected
            if not available:
                break
            if len(hist) != turn:
                evidence = "history cannot be parsed exactly: accepted turn=" + str(turn)
            if not candidates:
                evidence = "no public answer reproduces the complete feedback history"
            if obs.get("error"):
                evidence = str(obs["error"])
            guess = None
            if evidence:
                guess = self._recover(obs, evidence, available)
                if ctx.done:
                    return
                ctx.log("recovery", evidence=evidence, proposed=guess)
                evidence = None
            if not guess and not hist and opener in available:
                guess = opener
            elif not guess and candidates:
                turns_left = limit - turn
                candidates = [w for w in candidates if w in available]
                candidate_set = set(candidates)
                pool = candidates + [w for w in ranked
                                     if w in available and w not in candidate_set][:1200]
                if 0 < len(candidates) <= 10:
                    guess = self._endgame(candidates, sorted(available), turns_left)
                if not guess:
                    guess = self._choose(candidates, pool, turns_left)
            if ctx.done:
                return
            if not guess or guess not in available:
                guess = next((w for w in candidates if w in available), None)
            if not guess:
                guess = next(w for w in ranked if w in available)
            before = ctx.progress()
            ctx.memory["wordle_plan"] = {"guess": guess, "turn": turn,
                                         "candidate_count": len(candidates)}
            try:
                new_obs = ctx.act("guess " + guess)
            except Exception as exc:
                if ctx.done:
                    return
                rejected.add(guess)
                evidence = "action raised: " + repr(exc)
                failures += 1
                if failures >= 3:
                    self._recover(obs, evidence, available - {guess})
                    return
                continue
            after = ctx.progress()
            new_obs = new_obs or game.observe()
            new_hist = _history(new_obs)
            reason = None
            if after.get("illegal_actions", 0) > before.get("illegal_actions", 0) or new_obs.get("error"):
                rejected.add(guess)
                reason = "action rejected: " + str(new_obs.get("error"))
            elif (int(new_obs.get("turn", turn)) != turn + 1
                  or len(new_hist) != len(hist) + 1
                  or not new_hist or new_hist[-1][0] != guess
                  or after.get("since_obs_change", 0) >= 2):
                reason = "action did not produce the predicted history/turn advance"
            elif candidates and not any(feedback(a, guess) == new_hist[-1][1] for a in candidates):
                reason = "observed feedback was outside every predicted branch"
            if reason:
                evidence = json.dumps({"reason": reason, "guess": guess,
                                       "before_progress": before, "after_progress": after})
                failures += 1
                if failures >= 3:
                    if not ctx.done:
                        self._recover(new_obs, evidence, available - {guess})
                    return
            else:
                failures = 0


def build(ctx):
    return Harness(ctx)
