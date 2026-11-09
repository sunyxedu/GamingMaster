"""Verified, duplicate-aware Wordle candidate elimination.

The model-led baseline spent its entire budget producing actions that were
parsed as the bare ``guess`` command.  This harness keeps arithmetic and action
construction deterministic; Astra is used only if the public observation is
inconsistent with the solver's belief.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

try:
    from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
except Exception:  # pragma: no cover - preflight has the package installed
    BudgetExceeded = EpisodeOver = ModelUnavailable = Exception


_WORDS = {}


def _read_words(path):
    path = os.path.abspath(str(path))
    if path not in _WORDS:
        try:
            with open(path, encoding="utf-8") as f:
                _WORDS[path] = [x.strip().lower() for x in f if x.strip()]
        except Exception:
            _WORDS[path] = []
    return _WORDS[path]


def _feedback(guess, answer):
    """Return the game's green/yellow/gray tuple, including duplicates."""
    out = ["gray"] * 5
    left = Counter(answer)
    for i, ch in enumerate(guess):
        if i < 5 and ch == answer[i]:
            out[i] = "green"
            left[ch] -= 1
    for i, ch in enumerate(guess):
        if out[i] == "gray" and left[ch] > 0:
            out[i] = "yellow"
            left[ch] -= 1
    return tuple(out)


def _status(x):
    if isinstance(x, dict):
        for k in ("color", "status", "tile", "value", "result"):
            if k in x:
                return _status(x[k])
    if isinstance(x, (list, tuple)) and x:
        return _status(x[0])
    s = str(x).strip().lower()
    if s.startswith("g"):
        return "green"
    if s.startswith("y"):
        return "yellow"
    return "gray"


def _history_pair(row):
    if isinstance(row, dict):
        g = row.get("guess", row.get("word", row.get("action", "")))
        fb = row.get("feedback", row.get("result", row.get("tiles", [])))
    elif isinstance(row, (list, tuple)) and len(row) >= 2:
        g, fb = row[0], row[1]
    else:
        return "", ()
    if isinstance(g, dict):
        g = g.get("guess", g.get("value", ""))
    g = str(g).strip().lower()
    if g.startswith("guess "):
        g = g.split(None, 1)[1]
    if isinstance(fb, str):
        fb = list(fb)
    try:
        fb = tuple(_status(v) for v in fb)
    except Exception:
        fb = ()
    return g, fb


def _word_from_model(text, allowed):
    # Accept bare words, `guess word`, JSON, or explanatory prose.
    s = str(text or "").lower()
    for w in re.findall(r"[a-z]{5}", s):
        if w in allowed:
            return w
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _recover(self, obs, allowed):
        """Ask Astra for a legal word only after deterministic evidence fails."""
        if getattr(self.ctx, "system2", None) is None:
            return None
        prompt = (
            "Wordle solver inconsistency. Return exactly one legal five-letter "
            "guess from the supplied public guess list. Do not explain.\n" +
            json.dumps({"observation": obs, "legal_examples": allowed[:80]}, separators=(",", ":"))
        )
        try:
            r = self.ctx.think(prompt, timeout_s=180.0)
            return _word_from_model(getattr(r, "text", ""), set(allowed))
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            return None

    def _choose(self, candidates, allowed, turn, max_turns):
        if not candidates:
            return None
        if len(candidates) <= 3 or turn >= max_turns - 2:
            return candidates[0]
        # Candidate-only entropy is exact, fast enough for 2,315 answers, and
        # avoids probes that cannot themselves be submitted as answers.
        pool = candidates
        best, best_key = pool[0], None
        n = float(len(candidates))
        for g in pool:
            counts = {}
            for a in candidates:
                key = _feedback(g, a)
                counts[key] = counts.get(key, 0) + 1
            # maximize entropy; prefer a guess that remains a candidate.
            ent = sum((c / n) * math.log(n / c) for c in counts.values())
            key = (ent, -max(counts.values()), -candidates.index(g))
            if best_key is None or key > best_key:
                best_key, best = key, g
        return best

    def run_episode(self, game):
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        rules = (obs or {}).get("rules", {}) if isinstance(obs, dict) else {}
        length = int(rules.get("word_length", 5) or 5)
        if length != 5:
            # Wordle's advertised variant is five letters; let Astra handle a
            # future variant rather than submitting malformed fixed-width words.
            w = self._recover(obs, [])
            if w:
                try:
                    ctx.act("guess " + w)
                except Exception:
                    pass
            return
        answers = _read_words(rules.get("answers_path", ""))
        allowed = _read_words(rules.get("allowed_guesses_path", ""))
        if not answers or not allowed:
            return
        allowed_set = set(allowed)
        max_turns = int((obs or {}).get("max_turns", rules.get("max_guesses", 6)) or 6)
        used = set()
        while not ctx.done and isinstance(obs, dict) and not obs.get("done"):
            hist = []
            for row in obs.get("history", []) or []:
                g, fb = _history_pair(row)
                if len(g) == 5 and len(fb) == 5:
                    hist.append((g, fb))
                    used.add(g)
            candidates = [a for a in answers if all(_feedback(g, a) == fb for g, fb in hist)]
            turn = int(obs.get("turn", len(hist)) or len(hist))
            word = self._choose(candidates, allowed, turn, max_turns)
            if word is None or word in used:
                word = next((w for w in candidates if w not in used), None)
            if word is None:
                word = self._recover(obs, allowed)
            if not word:
                return
            before = turn
            try:
                nxt = ctx.act("guess " + word)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except Exception:
                # A public-list mismatch or parser rejection is evidence that
                # the scripted plan failed; re-plan once through Astra.
                alt = self._recover(obs, allowed)
                if not alt or alt == word:
                    return
                try:
                    nxt = ctx.act("guess " + alt)
                except Exception:
                    return
            if nxt is None:
                return
            obs = nxt
            try:
                progress = ctx.progress()
            except Exception:
                progress = {}
            after = int(obs.get("turn", before) or before) if isinstance(obs, dict) else before
            stale = after == before or int((progress or {}).get("since_obs_change", 0) or 0) > 0
            if stale and not (isinstance(obs, dict) and obs.get("done")):
                # Detect a silently rejected/stale plan and hand off with the
                # evidence; the fallback itself is verified by the next loop.
                alt = self._recover(obs, allowed)
                if alt and alt != word:
                    try:
                        obs = ctx.act("guess " + alt)
                    except Exception:
                        return


def build(ctx):
    return Harness(ctx)
