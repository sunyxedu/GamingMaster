"""Exact Wordle candidate elimination with a guarded Astra recovery path."""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict

from metaharness.core.contract import BudgetExceeded, ModelUnavailable, EpisodeOver


def _feedback(answer: str, guess: str):
    """Return the evaluator's duplicate-aware green/yellow/gray feedback."""
    out = ["gray"] * 5
    left = Counter(answer)
    for i, (a, g) in enumerate(zip(answer, guess)):
        if a == g:
            out[i] = "green"
            left[g] -= 1
    for i, g in enumerate(guess):
        if out[i] == "gray" and left[g] > 0:
            out[i] = "yellow"
            left[g] -= 1
    return out


def _norm_feedback(value):
    if isinstance(value, dict):
        value = value.get("feedback", value.get("tiles", value.get("result", [])))
    if isinstance(value, str):
        value = re.findall(r"green|yellow|gray|grey", value.lower())
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for x in value:
        if isinstance(x, dict):
            x = x.get("color", x.get("state", x.get("tile", x.get("feedback", ""))))
        s = str(x).lower()
        if "green" in s:
            out.append("green")
        elif "yellow" in s:
            out.append("yellow")
        elif "gray" in s or "grey" in s:
            out.append("gray")
    return out


def _history(obs):
    h = obs.get("history", []) if isinstance(obs, dict) else []
    result = []
    for item in h if isinstance(h, list) else []:
        if not isinstance(item, dict):
            continue
        g = item.get("guess", item.get("word", item.get("action")))
        if isinstance(g, dict):
            g = g.get("guess")
        if isinstance(g, str):
            g = g.strip().lower().split()[-1]
        fb = _norm_feedback(item.get("feedback", item.get("tiles", item.get("result"))))
        if isinstance(g, str) and len(g) == 5 and len(fb) == 5:
            result.append((g, fb))
    return result


def _paths(rules):
    found = {}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str) and ("answer" in k.lower() or "allowed" in k.lower() or "guess" in k.lower()) and os.path.isfile(v):
                    found[k.lower()] = v
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(rules)
    ans = next((v for k, v in found.items() if "answer" in k), None)
    allowed = next((v for k, v in found.items() if "allowed" in k or "guess" in k), None)
    return ans, allowed


def _read_words(path):
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return sorted({x.strip().lower() for x in f if len(x.strip()) == 5 and x.strip().isalpha()})
    except Exception:
        return []


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _recovery(self, obs, evidence, session):
        """Ask Astra only after a verified solver/action failure."""
        ctx = self.ctx
        if ctx.system2 is None or ctx.done:
            return None, session
        prompt = (
            "You are a recovery planner for a normal Wordle game. The deterministic solver "
            "reported an inconsistency or rejected action. Inspect the JSON observation and "
            "evidence, then reply with exactly one legal five-letter guess, lowercase, and no "
            "punctuation. Do not claim the hidden answer.\nOBSERVATION=" +
            json.dumps(obs, ensure_ascii=True) + "\nEVIDENCE=" + str(evidence)
        )
        try:
            res = ctx.think(prompt, session_id=session, timeout_s=220.0)
            text = (res.text or "").lower()
            words = re.findall(r"[a-z]{5}", text)
            return (words[-1] if words else None), res.session_id
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            return None, session

    def run_episode(self, game):
        ctx = self.ctx
        obs = game.observe()
        rules = obs.get("rules", {}) if isinstance(obs, dict) else {}
        ap, gp = _paths(rules)
        answers = _read_words(ap)
        allowed = set(_read_words(gp)) | set(answers)
        # A standard opener remains useful if a nonstandard observation omits file paths.
        if not answers:
            answers = ["cigar", "rebut", "sissy", "humph", "awake"]
            allowed.update(answers)
        opener = next((w for w in ("slate", "crane", "trace", "arise") if w in allowed), answers[0])
        session = None
        used = set()
        while not ctx.done and not game.done:
            obs = game.observe()
            hist = _history(obs)
            used.update(g for g, _ in hist)
            if obs.get("done") or obs.get("solved") or obs.get("result", {}).get("solved"):
                break
            # Recompute the exact surviving answer set from the complete public history.
            candidates = [a for a in answers if all(_feedback(a, g) == fb for g, fb in hist)]
            if hist and not candidates:
                guess, session = self._recovery(obs, "candidate set became empty", session)
            elif not hist:
                guess = opener
            elif len(candidates) == 1:
                guess = candidates[0]
            else:
                remaining_turns = int(obs.get("max_turns", 6)) - int(obs.get("turn", len(hist)))
                # With little room, make a possible answer; otherwise maximize information.
                pool = candidates if remaining_turns <= 2 else (candidates + [w for w in allowed if w not in used])
                # Keep computation bounded while retaining all answer candidates.
                if len(pool) > 3500:
                    pool = candidates + sorted((w for w in allowed if w not in used and w not in candidates))[:1000]
                best, best_key = None, None
                n = len(candidates)
                for w in pool:
                    bins = defaultdict(int)
                    for a in candidates:
                        bins[tuple(_feedback(a, w))] += 1
                    worst = max(bins.values())
                    ent = sum(c * math.log2(c) for c in bins.values()) / max(1, n)
                    # Worst-case dominates; entropy and answer preference break ties deterministically.
                    key = (worst, ent, 0 if w in candidates else 1, w)
                    if best_key is None or key < best_key:
                        best, best_key = w, key
                guess = best or candidates[0]
            if not guess:
                guess = opener
            guess = str(guess).strip().lower()
            if len(guess) != 5 or (allowed and guess not in allowed) or guess in used:
                # A stale/invalid plan is checked before posting; recover with an unused legal word.
                fallback = next((w for w in (candidates if 'candidates' in locals() else []) if w not in used), None)
                guess = fallback or next((w for w in allowed if w not in used), opener)
            before = ctx.progress()
            try:
                new_obs = ctx.act("guess " + guess)
            except Exception as exc:
                # Action errors are evidence for a single replanning call, never a harness error.
                alt, session = self._recovery(obs, "action error: " + repr(exc), session)
                if alt and len(alt) == 5:
                    try:
                        new_obs = ctx.act("guess " + alt)
                    except Exception:
                        break
                else:
                    break
            after = ctx.progress()
            if after.get("illegal_actions", 0) > before.get("illegal_actions", 0):
                continue
            obs = new_obs or game.observe()


def build(ctx):
    return Harness(ctx)
