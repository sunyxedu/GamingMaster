"""Fresh-context, one-step controller for generic TextWorld quests.

Each Astra answer is bound to a numbered snapshot of the commands available
right now. The answer is discarded after one action, so no plan can outlive
the state that grounded it.
"""
from __future__ import annotations

import json
import re

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


def _get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _sig(obs):
    return tuple(str(_get(obs, k, "")) for k in
                 ("description", "inventory", "objective", "score", "won", "lost"))


def _parse_index(text, n):
    """Accept only an unambiguous current-list index; arbitrary prose is invalid."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    try:
        value = json.loads(s)
        if isinstance(value, dict):
            value = value.get("index", value.get("action"))
        if isinstance(value, int) and not isinstance(value, bool):
            return value if 0 <= value < n else None
        if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
            value = int(value)
            return value if 0 <= value < n else None
    except (ValueError, TypeError):
        pass
    # Permit a bare integer, but do not mine a number out of explanatory text.
    if re.fullmatch(r"(?:action\s*)?#?\s*\d+\s*", s, re.I):
        m = re.search(r"\d+", s)
        value = int(m.group())
        return value if 0 <= value < n else None
    return None


def _destructive(command):
    return command.strip().lower().split(" ", 1)[0] in {
        "eat", "cook", "cut", "slice", "chop", "dice", "burn"
    }


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _fallback(self, names, obs, tried):
        """Make bounded, useful progress if Astra output is malformed/unavailable."""
        objective = str(_get(obs, "objective", "")).lower()
        candidates = [x for x in names if not _destructive(x)]
        unseen = [x for x in candidates if x not in tried]
        if unseen:
            # Immediate prerequisite/state-changing actions precede inspection;
            # movement is preferred to an irrelevant pickup when no local goal exists.
            rank = {"unlock": 0, "open": 1, "insert": 2, "put": 2,
                    "take": 3, "close": 4, "go": 5, "examine": 6,
                    "drop": 8, "inventory": 9, "look": 10}
            def key(action):
                verb = action.split(" ", 1)[0]
                score = rank.get(verb, 7)
                if any(w in objective for w in ("eat", "consume")) and verb == "eat":
                    score = -1
                return (score, len(action), action)
            unseen.sort(key=key)
            return unseen[0]
        # Stop instead of cycling when every safe candidate has already been
        # tried in this identical public state.
        return None

    def run_episode(self, game):
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        try:
            obs = ctx.observe()
        except Exception:
            return
        tried_by_state = {}
        failures = []
        malformed = 0
        while not ctx.done:
            try:
                names = list(getattr(game.action_spec, "names", ()) or ())
            except Exception:
                names = []
            if not names:
                return
            sig = _sig(obs)
            tried = tried_by_state.setdefault(sig, [])
            try:
                progress_now = ctx.progress()
            except Exception:
                progress_now = {}
            remaining = {}
            try:
                remaining = ctx.budget.remaining()
            except Exception:
                pass
            action = None
            if remaining.get("system2_calls", 1) > 0 and remaining.get("wall_s", 1) > 3:
                prompt = (
                    "Choose the single best next command for this TextWorld quest. "
                    "Use the literal objective and current observation; admissible means "
                    "executable, not safe. Preserve quest objects and never eat unless "
                    "the objective requires eating. Return ONLY a JSON object {\"index\": N}, "
                    "where N is a zero-based index in CURRENT COMMANDS. This is a fresh "
                    "decision: do not assume any earlier command or plan remains valid. "
                    "Avoid actions listed as tried in the identical public state."
                    f"\n\nOBJECTIVE: {str(_get(obs, 'objective', ''))[:1800]}"
                    f"\nOBSERVATION: {json.dumps({k: _get(obs, k, None) for k in ('description','inventory','feedback','score','max_score','commands_remaining')}, ensure_ascii=False, default=str)[:3600]}"
                    f"\nCURRENT COMMANDS (index: exact command): {json.dumps(list(enumerate(names)), ensure_ascii=False)}"
                    f"\nTRIED IN THIS UNCHANGED STATE: {json.dumps(tried[-8:], ensure_ascii=False)}"
                    f"\nHARNESS PROGRESS: {json.dumps(progress_now, ensure_ascii=False, default=str)}"
                    f"\nRECENT FAILURE EVIDENCE: {json.dumps(failures[-3:], ensure_ascii=False)}"
                )
                try:
                    result = ctx.think(prompt, session_id=None, timeout_s=min(145.0, max(1.0, remaining.get("wall_s", 146.0)-1.0)))
                    idx = _parse_index(getattr(result, "text", None), len(names))
                    if idx is not None:
                        action = names[idx]
                        malformed = 0
                    else:
                        malformed += 1
                        failures.append("unparseable/out-of-range action index")
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    break
                except (TimeoutError, RuntimeError) as exc:
                    malformed += 1
                    failures.append("model call: " + type(exc).__name__)
            if action is None:
                action = self._fallback(names, obs, tried)
            if action is None:
                return
            # Keep consumptive actions under explicit objective control even when
            # Astra returns a valid index or the fallback is running.
            if _destructive(action):
                objective = str(_get(obs, "objective", "")).lower()
                item = action.split(" ", 1)[-1].lower()
                if not any(w in objective for w in ("eat", "consume", "cook", "prepare", "slice", "chop", "dice")) or item not in objective:
                    failures.append("deferred unsupported destructive action: " + action)
                    tried.append(action)
                    continue
            before = obs
            try:
                obs = ctx.act(action)
            except (BudgetExceeded, EpisodeOver):
                return
            except Exception as exc:
                failures.append(action + " raised " + type(exc).__name__)
                tried.append(action)
                if ctx.done:
                    return
                try:
                    obs = ctx.observe()
                except Exception:
                    return
                continue
            tried.append(action)
            if len(tried) > 20:
                del tried[:-20]
            try:
                progress = ctx.progress()
            except Exception:
                progress = progress_now
            illegal = isinstance(progress, dict) and progress.get("illegal_actions", 0)
            changed = _sig(before) != _sig(obs)
            if illegal or not changed:
                failures.append(f"{action}: illegal={bool(illegal)}, public_state_changed={changed}")
                failures = failures[-8:]
            if _get(obs, "won", False) or _get(obs, "lost", False) or _get(obs, "terminated", False):
                return


def build(ctx):
    return Harness(ctx)
