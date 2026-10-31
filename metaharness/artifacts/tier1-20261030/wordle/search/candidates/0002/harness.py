"""Deterministic duplicate-aware Wordle solver with guarded model recovery."""
from __future__ import annotations
import json, math, os, re
from collections import Counter, defaultdict
try:
    from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable
except Exception:
    BudgetExceeded = EpisodeOver = ModelUnavailable = Exception

_CACHE = {}

def _words(path):
    if not path: return []
    path = os.path.abspath(str(path))
    if path in _CACHE: return _CACHE[path]
    try:
        with open(path, encoding="utf-8") as f:
            out = sorted({s.strip().lower() for s in f if len(s.strip()) == 5 and s.strip().isalpha()})
    except Exception:
        out = []
    _CACHE[path] = out
    return out

def _paths(x):
    found = []
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, str) and os.path.isfile(v):
                lk = k.lower()
                if "answer" in lk or "allowed" in lk or "guess" in lk: found.append((lk, v))
            if isinstance(v, (dict, list, tuple)): found.extend(_path_pairs(v))
    elif isinstance(x, (list, tuple)):
        for v in x: 
            if isinstance(v, (dict, list, tuple)): found.extend(_path_pairs(v))
    return (next((v for k, v in found if "answer" in k), None),
            next((v for k, v in found if "allowed" in k or "guess" in k), None))

def _path_pairs(x):
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, str) and os.path.isfile(v):
                lk = k.lower()
                if "answer" in lk or "allowed" in lk or "guess" in lk: out.append((lk, v))
            elif isinstance(v, (dict, list, tuple)):
                out.extend(_path_pairs(v))
    elif isinstance(x, (list, tuple)):
        for v in x:
            if isinstance(v, (dict, list, tuple)): out.extend(_path_pairs(v))
    return out

def _feedback(guess, answer):
    out, left = ["gray"] * 5, Counter(answer)
    for i, ch in enumerate(guess):
        if i < 5 and ch == answer[i]: out[i], left[ch] = "green", left[ch] - 1
    for i, ch in enumerate(guess):
        if out[i] == "gray" and left[ch] > 0: out[i], left[ch] = "yellow", left[ch] - 1
    return tuple(out)

def _tile(v):
    if isinstance(v, dict):
        for k in ("color", "status", "tile", "value"):
            if k in v: return _tile(v[k])
    s = str(v).strip().lower()
    if s in ("gray", "grey"): return "gray"
    return "green" if s.startswith("g") else ("yellow" if s.startswith("y") else "gray")

def _history(obs):
    rows = obs.get("history", []) if isinstance(obs, dict) else []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict):
            g, fb = row.get("guess", row.get("word", row.get("action", ""))), row.get("feedback", row.get("tiles", row.get("result", [])))
        elif isinstance(row, (list, tuple)) and len(row) >= 2: g, fb = row[0], row[1]
        else: continue
        if isinstance(g, dict): g = g.get("guess", g.get("value", ""))
        g = str(g).strip().lower()
        if g.startswith("guess "): g = g.split(None, 1)[1]
        if isinstance(fb, str): fb = re.findall(r"green|yellow|gray|grey", fb.lower())
        if isinstance(fb, (list, tuple)):
            f = tuple(_tile(z) for z in fb)
            if len(g) == 5 and len(f) == 5: out.append((g, f))
    return out

def _model_word(text, legal):
    legal = set(legal)
    return next((w for w in re.findall(r"[a-z]{5}", str(text or "").lower()) if w in legal), None)

class Harness:
    def __init__(self, ctx): self.ctx = ctx

    def _recover(self, obs, evidence, legal):
        if getattr(self.ctx, "system2", None) is None or self.ctx.done: return None
        prompt = ("A Wordle action plan failed verification. Return exactly one unused legal "
                   "five-letter guess and no explanation.\n" + json.dumps({"observation": obs, "evidence": str(evidence), "examples": list(legal)[:100]}))
        try:
            r = self.ctx.think(prompt, timeout_s=220.0)
            return _model_word(getattr(r, "text", ""), legal)
        except (TimeoutError, RuntimeError, BudgetExceeded, EpisodeOver, ModelUnavailable):
            return None

    def _pick(self, candidates, turn, limit):
        if not candidates: return None
        if len(candidates) <= 2 or turn >= limit - 2: return candidates[0]
        n, best, best_key = len(candidates), candidates[0], None
        for guess in candidates:
            bins = defaultdict(int)
            for answer in candidates: bins[_feedback(guess, answer)] += 1
            worst = max(bins.values())
            entropy = sum(c * math.log(c) for c in bins.values()) / float(n)
            key = (worst, entropy, 0 if guess in candidates else 1, guess)
            if best_key is None or key < best_key: best_key, best = key, guess
        return best

    def run_episode(self, game):
        ctx = self.ctx
        try: obs = game.observe()
        except Exception: return
        rules = obs.get("rules", {}) if isinstance(obs, dict) else {}
        ap, gp = _paths(rules)
        answers, allowed = _words(ap), sorted(set(_words(gp)) | set(_words(ap)))
        if int(rules.get("word_length", 5) or 5) != 5 or not answers or not allowed: return
        legal, max_turns, recovery_used = set(allowed), int(obs.get("max_turns", rules.get("max_guesses", 6)) or 6), False
        while not ctx.done and not game.done:
            if not isinstance(obs, dict) or obs.get("done") or obs.get("solved"): return
            hist, used = _history(obs), set()
            for g, _ in hist: used.add(g)
            candidates = [a for a in answers if all(_feedback(g, a) == fb for g, fb in hist)]
            turn = int(obs.get("turn", len(hist)) or len(hist))
            guess = self._pick(candidates, turn, max_turns)
            if guess in used or guess not in legal: guess = next((a for a in candidates if a not in used), None)
            if not guess: guess = next((w for w in allowed if w not in used), None)
            if not guess: return
            before = ctx.progress()
            try: nxt = ctx.act("guess " + guess)
            except (BudgetExceeded, EpisodeOver, ModelUnavailable): return
            except Exception as exc:
                nxt = None
                if not recovery_used:
                    recovery_used = True
                    alt = self._recover(obs, "exception: " + repr(exc), allowed)
                    if alt and alt not in used:
                        try: nxt = ctx.act("guess " + alt)
                        except Exception: return
                if nxt is None: return
            if nxt is None: return
            after = ctx.progress()
            new_turn = int(nxt.get("turn", turn) or turn) if isinstance(nxt, dict) else turn
            rejected = after.get("illegal_actions", 0) > before.get("illegal_actions", 0) or (isinstance(nxt, dict) and nxt.get("error"))
            if new_turn <= turn and not (isinstance(nxt, dict) and nxt.get("done")): rejected = True
            if rejected and not recovery_used:
                recovery_used = True
                alt = self._recover(nxt, "no accepted turn after guess " + guess, allowed)
                if alt and alt not in used:
                    try: nxt = ctx.act("guess " + alt)
                    except Exception: return
            obs = nxt

def build(ctx): return Harness(ctx)
