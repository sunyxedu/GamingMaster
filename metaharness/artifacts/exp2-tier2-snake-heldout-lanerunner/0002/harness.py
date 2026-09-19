"""Candidate 0002 — "compiled bridge": Astra writes the Astra->Jev bridge as code.

Structure
---------
* Astra (System-2) is asked ONCE per evaluation (cached in ``ctx.memory``) to
  write a small stdlib-only Python function ``bridge(obs) -> dict`` from
  ``game.describe()``, ``game.action_spec.describe()`` and one sample
  observation. The function returns, for the observed state,
      options : {action: short consequence label}   (non-fatal actions only)
      utility : {action: float}                      (higher is better)
      state   : compact dict a zero-shot classifier can read
  It is validated on the live observation before it is adopted; tracebacks
  are sent back to Astra for repair; a death post-mortem between seeds lets
  Astra improve it (bounded number of revisions).
* Every tick the harness runs the bridge (microseconds) and posts the best
  option immediately -> a fresh, non-fatal action on every tick regardless of
  model latency. Only THEN is Jev (System-1) asked, synchronously, to
  arbitrate among the labelled options; the tick that elapses during the call
  already runs on that fresh action. When Jev returns, the loop re-observes,
  re-runs the bridge and blends Jev's calibrated probabilities (max-normalised,
  <= 2 ticks old, restricted to the options that are still live) with the
  bridge utilities before posting again. It only blocks on ``wait_for_tick``
  when no tick elapsed during the call, so it never spins and never races the
  clock (deterministic on the virtual clock, too).
* Before the bridge exists (seed 0, first seconds) a generic Jev-only path
  keeps the game alive: per-action "does this end the game?" (Noul) plus a
  Choice; actions with p(end) > 0.5 are excluded.

Nothing here references a specific game's rules or observation keys.
"""
from __future__ import annotations

import json
import re
import traceback
from typing import Any

from metaharness.models.jev import Choice, Noul

MAX_REPAIRS = 2          # Astra repair rounds for code that fails validation / raises
MAX_REVISIONS = 2        # post-mortem improvement rounds across seeds
JEV_MAX_AGE_TICKS = 2    # discard Jev answers older than this
JEV_WEIGHT = 0.5         # Jev prob (max-normalised) vs bridge utility ([0,1]): Jev settles near-ties only
RUNTIME_ERR_LIMIT = 3    # bridge exceptions before it is dropped


def _obs_text(obs: Any, limit: int = 4000) -> str:
    """Generic rendering: JSON for the dict, multi-line strings printed raw."""
    if isinstance(obs, dict):
        head = {k: v for k, v in obs.items() if not (isinstance(v, str) and "\n" in v)}
        tail = [f"{k}:\n{v}" for k, v in obs.items() if isinstance(v, str) and "\n" in v]
        s = json.dumps(head, default=str) + ("\n" + "\n".join(tail) if tail else "")
    else:
        s = json.dumps(obs, default=str)
    return s if len(s) <= limit else s[:limit] + " ...(truncated)"


def _extract_code(text: str) -> str | None:
    if not text:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    cands = [b for b in blocks if "def bridge" in b] or ([text] if "def bridge" in text else [])
    return cands[-1].strip() if cands else None


class Bridge:
    """A compiled Astra-written featurizer with validation and error accounting."""

    def __init__(self, code: str, names: list[str]):
        self.code = code
        self.names = list(names)
        ns: dict[str, Any] = {"__name__": "astra_bridge"}
        exec(compile(code, "<astra_bridge>", "exec"), ns)  # noqa: S102 — the whole point
        fn = ns.get("bridge")
        if not callable(fn):
            raise ValueError("module defines no callable bridge(obs)")
        self.fn = fn
        self.errors = 0

    def __call__(self, obs: Any) -> dict[str, Any]:
        out = self.fn(obs)
        if not isinstance(out, dict):
            raise TypeError(f"bridge returned {type(out).__name__}, expected dict")
        opts = out.get("options")
        if not isinstance(opts, dict) or not opts:
            raise ValueError("bridge must return a non-empty 'options' dict {action: label}")
        options = {str(k): str(v) for k, v in opts.items() if str(k) in self.names}
        if not options:
            raise ValueError(f"'options' keys {list(opts)} are not valid action names {self.names}")
        util_raw = out.get("utility") or {}
        utility = {}
        for k in options:
            try:
                utility[k] = float(util_raw.get(k, 0.0))
            except (TypeError, ValueError):
                utility[k] = 0.0
        state = out.get("state")
        if state is None:
            state = {}
        return {"options": options, "utility": utility, "state": state}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.mem = ctx.memory
        self.mem.setdefault("bridge_code", None)
        self.mem.setdefault("guidance", "")
        self.mem.setdefault("code_session", None)   # Astra thread that knows the bridge code
        self.mem.setdefault("last_code", "")
        self.mem.setdefault("repairs", 0)
        self.mem.setdefault("revisions", 0)
        self.mem.setdefault("episodes", [])

    # ------------------------------------------------------------ Astra side
    def _rules_block(self, game) -> str:
        return (f"GAME:\n{game.describe()}\n\nACTIONS (exact names to use):\n{game.action_spec.describe()}\n"
                f"action names = {list(game.action_spec.names)}")

    def _think(self, prompt: str, tag: str, *, code_thread: bool = True) -> str:
        sid = self.mem.get("code_session") if code_thread else None
        try:
            res = self.ctx.think(prompt, session_id=sid)
        except RuntimeError as e:  # budget / backend problem: degrade silently
            self.ctx.log(event="astra_failed", tag=tag, error=str(e)[:300])
            return ""
        if code_thread and res.session_id:
            self.mem["code_session"] = res.session_id
        self.ctx.log(event="astra_reply", tag=tag, chars=len(res.text or ""))
        return res.text or ""

    def _code_prompt(self, game, obs) -> str:
        return (
            f"{self._rules_block(game)}\n\nSAMPLE OBSERVATION (a Python dict, this exact shape every tick):\n"
            f"{_obs_text(obs)}\n\n"
            "You are the slow planner for a real-time game controller. The controller is a fast loop that, on "
            "every tick, calls a Python function you write, posts the best action immediately, and lets a "
            "zero-shot classifier arbitrate between the remaining options. The action posted is applied to "
            "the observed state on the NEXT tick.\n\n"
            "Write a complete Python module (standard library only, no I/O, no printing, must run in < 5 ms) "
            "defining\n\n    def bridge(obs: dict) -> dict\n\nreturning a dict with keys:\n"
            "  'options': {action_name: short label of the concrete consequence of that action from this "
            "state, e.g. 'safe, 2 cells closer to reward' or 'dodges obstacle'}. Include ONLY actions that do "
            "not end the game on the next tick as far as the observation lets you see; look ahead further "
            "when the observation allows it (dead ends, traps, things arriving in 2-3 ticks). Omit actions "
            "that are no-ops/ignored when a real alternative exists. If every action is fatal, return the "
            "least bad one. Never return an empty dict.\n"
            "  'utility': {action_name: float} for the same keys — higher is better; reward survival first "
            "(a few ticks of lookahead / open space), then score gain, then progress toward reward.\n"
            "  'state': a small dict (<= 8 short fields) summarising what matters for choosing between the "
            "options (distances, threats, recommended action and why).\n\n"
            "Use only the exact action names listed. Derive everything from the observation dict; do not "
            "assume fields that are not in the sample. The function must never raise: wrap the body in "
            "try/except and on any error return {'options': {a: 'unknown' for a in ALL_ACTIONS}, "
            "'utility': {}, 'state': {'error': str(e)}}.\n"
            "Reply with the module in one ```python fenced block and nothing else."
        )

    def _guidance_prompt(self, game, obs) -> str:
        return (
            f"{self._rules_block(game)}\n\nSAMPLE OBSERVATION:\n{_obs_text(obs, 2500)}\n\n"
            "A zero-shot classifier (no reasoning) will pick one action per tick from a short list of options, "
            "each labelled with its consequence. Write at most 5 short imperative rules it should follow, most "
            "important first, one per line, e.g. what to avoid at all costs, what to prefer, how to trade "
            "safety against score. No prose, no headings."
        )

    def _repair_prompt(self, err: str) -> str:
        return (
            "Your bridge module:\n```python\n" + (self.mem.get("last_code") or "(none)")[-6000:] +
            "\n```\nfailed:\n" + err[-2500:] +
            "\n\nReturn the full corrected module (same contract: bridge(obs) -> {'options','utility','state'}, "
            "standard library only, never raises) in one ```python fenced block and nothing else."
        )

    def _improve_prompt(self, summary: dict[str, Any]) -> str:
        return (
            "Post-mortem of the last episode using your bridge module:\n```python\n"
            + (self.mem.get("last_code") or "(none)")[-6000:] + "\n```\n"
            f"Ended at tick {summary.get('ticks')} with score {summary.get('score')}; game info: "
            f"{json.dumps(summary.get('info'), default=str)}.\nLast observations before the end (oldest first):\n"
            + "\n---\n".join(summary.get("last_obs", [])) +
            f"\nActions posted for them: {summary.get('last_actions')}\n\n"
            "Explain in one line what went wrong, then return the full improved module (same contract, standard "
            "library only, never raises) in one ```python fenced block. Keep everything that worked."
        )

    # ---------------------------------------------------------- code loading
    def _try_adopt(self, text: str, obs: Any, names: list[str], source: str) -> tuple[Bridge | None, str]:
        """Compile + validate Astra's reply on the live observation. Returns (bridge, error)."""
        code = _extract_code(text)
        if not code:
            self.ctx.log(event="bridge_reject", source=source, reason="no code block")
            return None, "no ```python block containing def bridge(obs) was found in your reply"
        self.mem["last_code"] = code
        try:
            b = Bridge(code, names)
            b(obs)  # validate on the live observation
        except Exception as e:  # noqa: BLE001
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.ctx.log(event="bridge_reject", source=source, reason=err[-800:])
            return None, err
        self.mem["bridge_code"] = code
        self.ctx.log(event="bridge_adopted", source=source, chars=len(code))
        return b, ""

    def _load_cached(self, obs, names) -> Bridge | None:
        code = self.mem.get("bridge_code")
        if not code:
            return None
        try:
            b = Bridge(code, names)
            b(obs)
            return b
        except Exception as e:  # noqa: BLE001
            self.ctx.log(event="bridge_cache_invalid", error=str(e)[:300])
            self.mem["bridge_code"] = None
            return None

    # -------------------------------------------------------------- Jev side
    def _generic_questions(self, spec) -> dict[str, Any]:
        qs: dict[str, Any] = {}
        for n in spec.names:
            desc = spec.descriptions.get(n, n)
            qs[f"end:{n}"] = Noul(
                instructions=f"According to the observation and rules, would taking action '{n}' ({desc}) right "
                             "now make the game end immediately (collision, crash, death, game over)?")
        qs["best"] = Choice(
            instructions="Which action gains the most score over the next few ticks while keeping the game "
                         "alive? Follow coach_rules if present. Never pick an action that ends the game.",
            criteria={n: spec.descriptions.get(n, n) for n in spec.names})
        return qs

    def _generic_pick(self, answers: dict[str, Any], names: list[str]) -> str | None:
        best = answers.get("best")
        if best is None:
            return None
        probs = dict(best.probabilities) or {best.value: 1.0}
        alive = {}
        for n in names:
            a = answers.get(f"end:{n}")
            p_end = float(a.value) if a is not None else 0.0
            if p_end <= 0.5:
                alive[n] = probs.get(n, 0.0) * (1.0 - p_end)
        if not alive:  # everything looks fatal: least fatal
            n = min(names, key=lambda k: float(answers[f"end:{k}"].value) if f"end:{k}" in answers else 1.0)
            return n
        return max(alive, key=alive.get)

    # ------------------------------------------------------------------ loop
    def run_episode(self, game):
        ctx = self.ctx
        spec = game.action_spec
        names = list(spec.names)
        rules_text = f"{game.describe()}\n{spec.describe()}"
        obs = game.observe()

        bridge: Bridge | None = self._load_cached(obs, names)
        astra_fut = None          # pending Astra call -> (kind, future)
        astra_kind = ""
        guidance_fut = None
        last_summary = self.mem["episodes"][-1] if self.mem["episodes"] else None

        # --- kick off Astra work in the background ---------------------------
        if bridge is None:
            if not self.mem.get("guidance"):
                guidance_fut = ctx.spawn(lambda o=obs: self._think(self._guidance_prompt(game, o), "guidance",
                                                                   code_thread=False))
            astra_kind = "code"
            astra_fut = ctx.spawn(lambda o=obs: self._think(self._code_prompt(game, o), "code"))
        elif (last_summary and not last_summary.get("clean_end") and
              self.mem["revisions"] < MAX_REVISIONS):
            self.mem["revisions"] += 1
            astra_kind = "improve"
            astra_fut = ctx.spawn(lambda s=last_summary: self._think(self._improve_prompt(s), "improve"))

        jev_res: dict[str, Any] | None = None   # {"tick", "probs"} from the last Jev arbitration
        last_action: str | None = None
        recent_obs: list[str] = []
        recent_actions: list[str] = []
        pending_err: str | None = None
        tick_i = 0
        posted_ticks = 0
        n_changed = 0

        def _ticks() -> int:
            try:
                return int(game.ticks)
            except Exception:  # noqa: BLE001
                return tick_i

        while not ctx.done:
            obs = game.observe()
            tick_i += 1
            ticks_before = _ticks()
            recent_obs.append(_obs_text(obs, 1500)); recent_obs = recent_obs[-4:]

            # ---- collect Astra results --------------------------------------
            if guidance_fut is not None and guidance_fut.done():
                try:
                    g = (guidance_fut.result() or "").strip()
                except Exception:  # noqa: BLE001
                    g = ""
                if g:
                    self.mem["guidance"] = g[:800]
                guidance_fut = None
            if astra_fut is not None and astra_fut.done():
                try:
                    text = astra_fut.result() or ""
                except Exception as e:  # noqa: BLE001
                    text = ""
                    ctx.log(event="astra_future_error", error=str(e)[:300])
                astra_fut = None
                if text:
                    got, err = self._try_adopt(text, obs, names, astra_kind)
                    if got is not None:
                        bridge = got
                        self.mem["repairs"] = 0
                    elif self.mem["repairs"] < MAX_REPAIRS:
                        self.mem["repairs"] += 1
                        astra_kind = "repair"
                        astra_fut = ctx.spawn(lambda e=err: self._think(self._repair_prompt(e), "repair"))
            if pending_err and astra_fut is None and self.mem["repairs"] < MAX_REPAIRS:
                self.mem["repairs"] += 1
                astra_kind = "repair"
                astra_fut = ctx.spawn(lambda err=pending_err: self._think(self._repair_prompt(err), "repair"))
                pending_err = None

            # ---- decide -------------------------------------------------------
            action: str | None = None
            consulted = False
            feat = None
            if bridge is not None:
                try:
                    feat = bridge(obs)
                except Exception as e:  # noqa: BLE001
                    bridge.errors += 1
                    err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    ctx.log(event="bridge_runtime_error", n=bridge.errors, error=err[-600:])
                    if bridge.errors >= RUNTIME_ERR_LIMIT:
                        pending_err = err
                        bridge = None
                        self.mem["bridge_code"] = None
            if feat is not None:
                opts, util = feat["options"], feat["utility"]
                lo, hi = min(util.values()), max(util.values())
                score = {k: ((util[k] - lo) / (hi - lo) if hi > lo else 0.0) for k in opts}
                best = max(score, key=score.get)
                used_jev = False
                if jev_res and tick_i - jev_res["tick"] <= JEV_MAX_AGE_TICKS:
                    p = {k: jev_res["probs"].get(k, 0.0) for k in opts}
                    pm = max(p.values()) if p else 0.0
                    if pm > 0:
                        for k in opts:
                            score[k] += JEV_WEIGHT * p[k] / pm
                        used_jev = True
                action = max(score, key=score.get)
                if used_jev and action != best:
                    n_changed += 1
                ctx.act(action)              # fresh, non-fatal action posted BEFORE the slow call
                last_action = action
                posted_ticks += 1
                ctx.log(event="decision", tick=tick_i, options=list(opts), best=best, action=action,
                        jev=used_jev, changed=(used_jev and action != best))
                jev_res = None
                if len(opts) > 1:
                    state = {"coach_rules": self.mem.get("guidance") or rules_text[:600],
                             "situation": feat["state"], "last_action": last_action}
                    q = {"move": Choice(
                        instructions="You are the reflex controller. Pick the option with the best outcome: "
                                     "stay alive first, then gain score. Each option is labelled with its "
                                     "predicted consequence; follow coach_rules.",
                        criteria=dict(opts))}
                    try:
                        ans = ctx.decide(state, q)
                        a = ans.get("move")
                        if a is not None:
                            jev_res = {"tick": tick_i, "probs": dict(a.probabilities) or {str(a.value): 1.0}}
                    except RuntimeError as e:
                        ctx.log(event="jev_error", error=str(e)[:300])
                    consulted = True
            else:
                # generic path (no bridge yet, or the bridge raised this tick): Jev alone, danger-filtered
                state = {"rules": rules_text, "coach_rules": self.mem.get("guidance") or "",
                         "observation": obs, "last_action": last_action}
                try:
                    ans = ctx.decide(state, self._generic_questions(spec))
                    action = self._generic_pick(ans, names)
                except RuntimeError as e:
                    ctx.log(event="jev_error", error=str(e)[:300])
                consulted = True
                if action is not None:
                    ctx.act(action)
                    last_action = action
                    posted_ticks += 1

            recent_actions.append(action or "(none)"); recent_actions = recent_actions[-4:]
            # A slow call already let a tick pass -> decide again right away on the new state;
            # otherwise block until the next tick boundary (never spin).
            if not consulted or _ticks() == ticks_before:
                ctx.wait_for_tick()

        # ---- post-mortem for the next seed ---------------------------------
        info = {}
        try:
            info = dict(game.info())
        except Exception:  # noqa: BLE001
            pass
        ticks = _ticks()
        max_ticks = getattr(game, "max_ticks", None)
        if not max_ticks:  # the runner's game view hides it; best effort from the prose description
            m = re.search(r"max(?:imum)?\s+(\d+)\s+ticks", game.describe(), flags=re.I)
            max_ticks = int(m.group(1)) if m else None
        clean_end = bool(max_ticks and ticks >= max_ticks)
        self.mem["episodes"].append({"ticks": ticks, "score": game.score, "info": info, "clean_end": clean_end,
                                     "last_obs": recent_obs, "last_actions": recent_actions,
                                     "had_bridge": bridge is not None})
        ctx.log(event="episode_summary", ticks=ticks, score=game.score, posted=posted_ticks,
                jev_changed=n_changed, had_bridge=bridge is not None, info=info)


def build(ctx):
    return Harness(ctx)
