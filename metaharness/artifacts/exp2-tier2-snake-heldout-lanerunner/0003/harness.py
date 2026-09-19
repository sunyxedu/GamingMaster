"""Candidate 0003 — "harvest the bridge": compiled Astra->Jev bridge that actually lands.

Parent: 0002 (compiled bridge). 0002's traces show the bridge never ran in any
episode: Astra's module was still being written when the snake died (3.5-6.6 s
wall), the pending future was thrown away at episode end, and every seed paid
the same Astra calls again (6 S2 calls, 0 harvested). Meanwhile the generic
Jev-only phase asked "does action X end the game *now*?" and Jev answered
~0.6 s later - one tick too late at a 0.5 s tick.

Three structural changes
------------------------
1. **Nothing Astra writes is ever dropped.** Every Astra future is kept in
   ``ctx.memory``; at the end of an episode the harness *blocks* (the game is
   over, the clock no longer matters) until the code arrives, validates it,
   runs the repair loop synchronously, and stores it. Seeds >= 1 therefore
   always start with a working bridge, and post-mortem improvements are also
   done in this free window rather than racing the next episode.
2. **One lean Astra call instead of two.** The module carries its own coach
   rules (``RULES`` constant) so guidance does not compete with the code for
   Astra time; the prompt asks for a short module.
3. **A lookahead-aware generic phase.** Before code exists Jev is asked
   "would action X - held for the next few ticks - end the game?" and the
   controller minimises that risk (a Choice only breaks ties among low-risk
   actions), so a one-tick-late answer is still useful. It also runs Jev in
   the background and posts every answer as soon as it arrives.

Once the bridge exists the loop is 0002's: run the bridge (microseconds), post
its argmax immediately, ask Jev to arbitrate among the labelled non-fatal
options, blend when the answer comes back. Nothing here names a game, an
observation key or an action.
"""
from __future__ import annotations

import json
import re
import time
import traceback
from typing import Any

from metaharness.models.jev import Choice, Noul

MAX_REPAIRS = 3           # repair rounds per broken module (validation / runtime errors)
MAX_REVISIONS = 2         # post-mortem improvement rounds across the evaluation
JEV_MAX_AGE_TICKS = 2     # discard Jev answers older than this
JEV_WEIGHT = 0.5          # Jev prob (max-normalised) vs bridge utility ([0,1]) -> settles near-ties
RUNTIME_ERR_LIMIT = 3     # bridge exceptions before it is dropped for repair
END_WAIT_S = 240.0        # max wall time spent harvesting/repairing Astra output after game over
GENERIC_RISK_OK = 0.35    # generic phase: actions with p(end) below this are "low risk"


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
        exec(compile(code, "<astra_bridge>", "exec"), ns)  # noqa: S102 - the whole point
        fn = ns.get("bridge")
        if not callable(fn):
            raise ValueError("module defines no callable bridge(obs)")
        self.fn = fn
        rules = ns.get("RULES")
        if isinstance(rules, (list, tuple)):
            rules = "\n".join(str(r) for r in rules)
        self.rules = str(rules)[:800] if rules else ""
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
        self.mem.setdefault("code_session", None)    # Astra thread that knows the bridge code
        self.mem.setdefault("last_code", "")
        self.mem.setdefault("repairs", 0)
        self.mem.setdefault("revisions", 0)
        self.mem.setdefault("episodes", [])
        self.mem.setdefault("pending", None)          # (kind, Future) surviving across episodes

    # ------------------------------------------------------------ Astra side
    def _rules_block(self, game) -> str:
        return (f"GAME:\n{game.describe()}\n\nACTIONS (exact names to use):\n{game.action_spec.describe()}\n"
                f"action names = {list(game.action_spec.names)}")

    def _think(self, prompt: str, tag: str) -> str:
        sid = self.mem.get("code_session")
        t0 = time.time()
        try:
            res = self.ctx.think(prompt, session_id=sid)
        except Exception as e:  # noqa: BLE001 - budget / backend problem: degrade silently
            self.ctx.log(event="astra_failed", tag=tag, error=str(e)[:300])
            return ""
        if res.session_id:
            self.mem["code_session"] = res.session_id
        self.ctx.log(event="astra_reply", tag=tag, chars=len(res.text or ""), wall_s=round(time.time() - t0, 2))
        return res.text or ""

    def _code_prompt(self, game, obs) -> str:
        return (
            f"{self._rules_block(game)}\n\nSAMPLE OBSERVATION (a Python dict, this exact shape every tick):\n"
            f"{_obs_text(obs)}\n\n"
            "You are the slow planner for a real-time game controller. The controller is a fast loop that, on "
            "every tick, calls a Python function you write, posts the best action immediately, and lets a "
            "zero-shot classifier arbitrate between the remaining options. The action posted is applied to "
            "the observed state on the NEXT tick.\n\n"
            "Write a complete, SHORT Python module (standard library only, no I/O, no printing, must run in "
            "< 5 ms, ideally under 120 lines) defining\n\n"
            "    RULES = [...]   # <= 5 short imperative coach rules for the classifier, most important first\n"
            "    def bridge(obs: dict) -> dict\n\nreturning a dict with keys:\n"
            "  'options': {action_name: short label of the concrete consequence of that action from this "
            "state, e.g. 'safe, 2 cells closer to reward' or 'dodges obstacle'}. Include ONLY actions that do "
            "not end the game on the next tick as far as the observation lets you see; look ahead further "
            "when the observation allows it (dead ends, traps, things arriving in 2-3 ticks). Omit actions "
            "that are no-ops/ignored when a real alternative exists. If every action is fatal, return the "
            "least bad one. Never return an empty dict.\n"
            "  'utility': {action_name: float} for the same keys - higher is better; reward survival first "
            "(a few ticks of lookahead / open space reachable afterwards), then score gain, then progress "
            "toward reward.\n"
            "  'state': a small dict (<= 8 short fields) summarising what matters for choosing between the "
            "options (distances, threats, recommended action and why).\n\n"
            "Use only the exact action names listed. Derive everything from the observation dict; do not "
            "assume fields that are not in the sample. The function must never raise: wrap the body in "
            "try/except and on any error return {'options': {a: 'unknown' for a in ALL_ACTIONS}, "
            "'utility': {}, 'state': {'error': str(e)}}.\n"
            "Reply with the module in one ```python fenced block and nothing else."
        )

    def _repair_prompt(self, err: str) -> str:
        return (
            "Your bridge module:\n```python\n" + (self.mem.get("last_code") or "(none)")[-6000:] +
            "\n```\nfailed:\n" + err[-2500:] +
            "\n\nReturn the full corrected module (same contract: RULES list + bridge(obs) -> "
            "{'options','utility','state'}, standard library only, never raises) in one ```python fenced "
            "block and nothing else."
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
        """Compile + validate Astra's reply on an observation. Returns (bridge, error)."""
        code = _extract_code(text)
        if not code:
            self.ctx.log(event="bridge_reject", source=source, reason="no code block")
            return None, "no ```python block containing def bridge(obs) was found in your reply"
        self.mem["last_code"] = code
        try:
            b = Bridge(code, names)
            b(obs)  # validate on a real observation
        except Exception as e:  # noqa: BLE001
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.ctx.log(event="bridge_reject", source=source, reason=err[-800:])
            return None, err
        self.mem["bridge_code"] = code
        self.ctx.log(event="bridge_adopted", source=source, chars=len(code), rules=bool(b.rules))
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

    def _spawn_astra(self, kind: str, prompt: str) -> None:
        self.mem["pending"] = (kind, self.ctx.spawn(lambda p=prompt, k=kind: self._think(p, k)))

    def _harvest(self, obs, names, *, block: bool, deadline: float | None = None) -> Bridge | None:
        """Collect a finished (or, if ``block``, awaited) Astra reply; adopt it or queue a repair.

        Returns a bridge when one was adopted this call, else None."""
        pend = self.mem.get("pending")
        if pend is None:
            return None
        kind, fut = pend
        if not fut.done():
            if not block:
                return None
            timeout = None if deadline is None else max(0.0, deadline - time.time())
            try:
                fut.result(timeout=timeout)
            except Exception:  # noqa: BLE001 - timeout or worker error; leave it pending
                return None
        self.mem["pending"] = None
        try:
            text = fut.result() or ""
        except Exception as e:  # noqa: BLE001
            self.ctx.log(event="astra_future_error", error=str(e)[:300])
            text = ""
        if not text:
            return None
        got, err = self._try_adopt(text, obs, names, kind)
        if got is not None:
            self.mem["repairs"] = 0
            return got
        if self.mem["repairs"] < MAX_REPAIRS:
            self.mem["repairs"] += 1
            self._spawn_astra("repair", self._repair_prompt(err))
        return None

    # -------------------------------------------------------------- Jev side
    def _generic_questions(self, spec) -> dict[str, Any]:
        qs: dict[str, Any] = {}
        for n in spec.names:
            desc = spec.descriptions.get(n, n)
            qs[f"end:{n}"] = Noul(
                instructions=f"If action '{n}' ({desc}) is taken now AND the resulting movement/heading is then "
                             "held for the next 2-3 ticks, would the game end (collision, crash, out of bounds, "
                             "death, game over) within those ticks, according to the observation and rules? "
                             "Answer yes if this direction runs into a wall, edge, obstacle or the player's own "
                             "body within 3 cells/ticks.")
        qs["best"] = Choice(
            instructions="Among the actions that stay safe for the next several ticks, which gains the most "
                         "score (moves toward reward/food/coins)? Never pick a direction that leads into a "
                         "wall, edge, obstacle or the player's own body within 3 ticks.",
            criteria={n: spec.descriptions.get(n, n) for n in spec.names})
        return qs

    def _generic_pick(self, answers: dict[str, Any], names: list[str]) -> str | None:
        best = answers.get("best")
        risk = {}
        for n in names:
            a = answers.get(f"end:{n}")
            risk[n] = float(a.value) if a is not None else 0.5
        probs = (dict(best.probabilities) or {best.value: 1.0}) if best is not None else {}
        low = [n for n in names if risk[n] <= GENERIC_RISK_OK]
        if low:  # among low-risk actions, Jev's preference breaks ties, risk still counts
            return max(low, key=lambda n: probs.get(n, 0.0) * (1.0 - risk[n]) ** 2 + 1e-3 * (1.0 - risk[n]))
        return min(names, key=lambda n: risk[n])  # everything risky: least risky

    # ------------------------------------------------------------------ loop
    def run_episode(self, game):
        ctx = self.ctx
        spec = game.action_spec
        names = list(spec.names)
        rules_text = f"{game.describe()}\n{spec.describe()}"
        obs = game.observe()
        last_summary = self.mem["episodes"][-1] if self.mem["episodes"] else None

        bridge: Bridge | None = self._load_cached(obs, names)
        if bridge is None:
            bridge = self._harvest(obs, names, block=False)   # a reply from a previous episode?
        if bridge is None and self.mem.get("pending") is None:
            self.mem["repairs"] = 0
            self._spawn_astra("code", self._code_prompt(game, obs))
        elif (bridge is not None and self.mem.get("pending") is None and last_summary
              and last_summary.get("had_bridge") and not last_summary.get("clean_end")
              and not last_summary.get("improved") and self.mem["revisions"] < MAX_REVISIONS):
            # improvement was not possible at the end of the last episode (e.g. timeout) -> background now
            self.mem["revisions"] += 1
            last_summary["improved"] = True
            self._spawn_astra("improve", self._improve_prompt(last_summary))

        jev_res: dict[str, Any] | None = None    # {"tick", "probs"} from the last Jev arbitration
        gen_fut = None                            # background generic Jev call
        gen_tick = 0
        last_action: str | None = None
        recent_obs: list[str] = []
        recent_actions: list[str] = []
        tick_i = 0
        posted_ticks = 0
        n_changed = 0

        def _ticks() -> int:
            try:
                return int(game.ticks)
            except Exception:  # noqa: BLE001
                return tick_i

        def _generic_call(o, la):
            state = {"rules": rules_text, "observation": o, "last_action": la}
            try:
                return ctx.decide(state, self._generic_questions(spec))
            except Exception as e:  # noqa: BLE001
                ctx.log(event="jev_error", error=str(e)[:300])
                return None

        while not ctx.done:
            obs = game.observe()
            tick_i += 1
            ticks_before = _ticks()
            recent_obs.append(_obs_text(obs, 1500)); recent_obs = recent_obs[-4:]

            # ---- collect Astra results (non-blocking while the game runs) ----
            if bridge is None or self.mem.get("pending") is not None:
                got = self._harvest(obs, names, block=False)
                if got is not None:
                    bridge = got
                    gen_fut = None

            # ---- decide ---------------------------------------------------
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
                        bridge = None
                        self.mem["bridge_code"] = None
                        if self.mem.get("pending") is None and self.mem["repairs"] < MAX_REPAIRS:
                            self.mem["repairs"] += 1
                            self._spawn_astra("repair", self._repair_prompt(err))
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
                    state = {"coach_rules": bridge.rules or rules_text[:600],
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
                    except Exception as e:  # noqa: BLE001
                        ctx.log(event="jev_error", error=str(e)[:300])
                    consulted = True
            else:
                # generic path (no bridge yet): Jev in the background, lookahead-risk minimiser
                if gen_fut is not None and gen_fut.done():
                    try:
                        ans = gen_fut.result()
                    except Exception:  # noqa: BLE001
                        ans = None
                    gen_fut = None
                    if ans:
                        action = self._generic_pick(ans, names)
                        if action is not None:
                            ctx.act(action)
                            last_action = action
                            posted_ticks += 1
                            ctx.log(event="generic_decision", tick=tick_i, for_tick=gen_tick, action=action,
                                    risk={n: round(float(ans[f"end:{n}"].value), 2)
                                          for n in names if f"end:{n}" in ans})
                if gen_fut is None:
                    gen_tick = tick_i
                    gen_fut = ctx.spawn(lambda o=obs, la=last_action: _generic_call(o, la))
                elif action is None and last_action is not None:
                    ctx.act(last_action)     # re-affirm the standing decision while Jev is still busy

            recent_actions.append(action or "(none)"); recent_actions = recent_actions[-4:]
            # A slow call already let a tick pass -> decide again right away on the new state;
            # otherwise block until the next tick boundary (never spin).
            if not consulted or _ticks() == ticks_before:
                ctx.wait_for_tick()

        # ---- game over: free window. Harvest / repair / improve synchronously -----
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
        summary = {"ticks": ticks, "score": game.score, "info": info, "clean_end": clean_end,
                   "last_obs": recent_obs, "last_actions": recent_actions,
                   "had_bridge": bridge is not None, "improved": False}
        self.mem["episodes"].append(summary)
        ctx.log(event="episode_summary", ticks=ticks, score=game.score, posted=posted_ticks,
                jev_changed=n_changed, had_bridge=bridge is not None, info=info)

        deadline = time.time() + END_WAIT_S
        try:
            last_obs = game.observe()
        except Exception:  # noqa: BLE001
            last_obs = obs
        # 1) wait for the code / repair chain to produce a working bridge
        while self.mem.get("pending") is not None and time.time() < deadline:
            got = self._harvest(last_obs, names, block=True, deadline=deadline)
            if got is not None:
                bridge = got
        # 2) if the bridge existed and the episode was not clean, improve it now (no clock pressure)
        if (bridge is not None and self.mem.get("pending") is None and summary["had_bridge"]
                and not clean_end and self.mem["revisions"] < MAX_REVISIONS and time.time() < deadline):
            self.mem["revisions"] += 1
            summary["improved"] = True
            prev_code = self.mem["bridge_code"]
            self._spawn_astra("improve", self._improve_prompt(summary))
            while self.mem.get("pending") is not None and time.time() < deadline:
                got = self._harvest(last_obs, names, block=True, deadline=deadline)
                if got is not None:
                    bridge = got
            if self.mem.get("bridge_code") is None:   # improvement chain failed: keep the old module
                self.mem["bridge_code"] = prev_code
                self.mem["last_code"] = prev_code or ""
        ctx.log(event="end_window", waited_s=round(time.time() - (deadline - END_WAIT_S), 1),
                have_bridge=self.mem.get("bridge_code") is not None, pending=self.mem.get("pending") is not None)


def build(ctx):
    return Harness(ctx)
