"""Candidate 0004 — "every tick is a bridge tick": close the stale-tick hole, break limit cycles.

Parent: 0003 (compiled Astra bridge, harvested across episodes; mean 9.33).

Diagnosis of 0003 (from its traces)
-----------------------------------
* Both deaths were on STALE ticks.  seed 0: ``TICK 203 up fresh (score 17) ->
  TICK 204 up STALE -> death: wall``; seed 2: ``TICK 74 down fresh ("eats food,
  must turn next tick") -> TICK 75 down STALE -> death: self``.  0003's loop
  ran the bridge, posted, then *blocked* on a synchronous Jev call (~0.58 s >
  0.5 s tick), so roughly every sixth tick the bridge did not run at all
  (13 % stale).  The bridge only guarantees "the posted action is non-fatal
  on the next tick"; a stale tick re-applies it one tick later, where it is
  fatal whenever the move ended facing a wall/body - i.e. exactly after
  eating food that sits against an obstacle.
* seed 1 lived 300 ticks but scored 4: a limit cycle.  The same situations
  (``[9,x] down "away from food"`` -> ``[7,2] right`` -> ...) recur every
  ~48 decisions for 250 ticks without any food.  And because the episode hit
  max ticks it counted as "clean", so no post-mortem was run.

Structural changes vs. 0003
---------------------------
1. **Tick-synchronous control loop.**  Every tick: observe -> bridge (µs) ->
   ``ctx.act`` -> ``wait_for_tick``.  Nothing slow ever sits between two
   ticks.  Jev runs in a background future and is used only as a
   *tie-breaker*: its (≤2 ticks old) preference may override the bridge
   argmax only among options whose utility is within ``TIE_MARGIN`` of the
   best.  Expected: stale ≈ 0 %, no more "died on the stale tick".
2. **Stall detection.**  If the score has not increased for ``STALL_TICKS``
   ticks the harness (a) enters *explore* mode - picks uniformly among
   near-best options for a few ticks to break the cycle, and (b) once per
   episode spawns an Astra "unstick" call carrying a sample of the situations
   from the stalled stretch; the revised module is validated and adopted
   mid-episode (safe now, because the loop never blocks).
3. **Module scoreboard.**  Every module version keeps the scores of the
   episodes it played.  An untested revision is always tried once; a tested
   revision that scored below the best tested module is reverted at the next
   episode start.  Post-mortems are now also triggered by a stalled (not just
   a fatal) episode.

Nothing here names a game, an observation key or an action; everything comes
from ``game.describe()``, ``game.action_spec`` and the observation dict.
"""
from __future__ import annotations

import json
import random
import re
import time
import traceback
from typing import Any

from metaharness.models.jev import Choice, Noul

MAX_REPAIRS = 3           # repair rounds per broken module
MAX_REVISIONS = 4         # improve/unstick rounds across the whole evaluation
JEV_MAX_AGE_TICKS = 2     # discard Jev answers older than this
TIE_MARGIN = 0.15         # Jev may override the bridge argmax only within this normalised-utility gap
RUNTIME_ERR_LIMIT = 3     # bridge exceptions before it is dropped for repair
END_WAIT_S = 200.0        # max wall time spent harvesting/repairing Astra output after game over
GENERIC_RISK_OK = 0.35    # generic phase: actions with p(end) below this are "low risk"
STALL_TICKS = 45          # no score gain for this many ticks => stalled
EXPLORE_TICKS = 6         # ticks of randomised near-best choices per stall burst
EXPLORE_MARGIN = 0.25     # options within this normalised gap of the best are explore candidates
STALL_SAMPLE = 12         # situations shown to Astra in the unstick prompt


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
        self.mem.setdefault("modules", [])            # [{"code", "source", "scores": [...]}]

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

    _CONTRACT = (
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
        "  'utility': {action_name: float} for the same keys - higher is better. Use LARGE negative terms "
        "for anything that risks survival (dead ends, little reachable space, forced collisions within a "
        "few ticks) and make PROGRESS TOWARD REWARD the main graded term among equally safe options, so "
        "the agent never orbits the reward or wanders in a cycle. Small tie-breaks (e.g. keep heading) "
        "must be smaller than one unit of progress.\n"
        "  'state': a small dict (<= 8 short fields) summarising what matters for choosing between the "
        "options (distances, threats, recommended action and why).\n\n"
        "Use only the exact action names listed. Derive everything from the observation dict; do not "
        "assume fields that are not in the sample. The function must never raise: wrap the body in "
        "try/except and on any error return {'options': {a: 'unknown' for a in ALL_ACTIONS}, "
        "'utility': {}, 'state': {'error': str(e)}}.\n"
    )

    def _code_prompt(self, game, obs) -> str:
        return (
            f"{self._rules_block(game)}\n\nSAMPLE OBSERVATION (a Python dict, this exact shape every tick):\n"
            f"{_obs_text(obs)}\n\n"
            "You are the slow planner for a real-time game controller. The controller is a fast loop that, on "
            "EVERY tick, calls a Python function you write and posts the highest-utility option immediately; "
            "when two options are nearly tied a zero-shot classifier breaks the tie using your labels. The "
            "action posted is applied to the observed state on the NEXT tick, and if the controller ever "
            "misses a tick the previous action is applied again - so prefer options that stay safe if "
            "repeated once.\n\n" + self._CONTRACT +
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
        stall = summary.get("stall_trace") or []
        stall_txt = ""
        if stall:
            stall_txt = ("\nThe score did not increase for " + str(summary.get("max_stall")) +
                         " ticks. Sampled situations from that stretch (your 'state' dict, the option labels "
                         "and the action posted), oldest first:\n" +
                         "\n".join(json.dumps(s, default=str)[:600] for s in stall) + "\n")
        return (
            "Post-mortem of the last episode using your bridge module:\n```python\n"
            + (self.mem.get("last_code") or "(none)")[-6000:] + "\n```\n"
            f"Ended at tick {summary.get('ticks')} with score {summary.get('score')}; game info: "
            f"{json.dumps(summary.get('info'), default=str)}.\nLast observations before the end (oldest first):\n"
            + "\n---\n".join(summary.get("last_obs", [])) +
            f"\nActions posted for them: {summary.get('last_actions')}\n" + stall_txt +
            "\nExplain in one line what went wrong (a death, or a cycle/wandering that never reached the "
            "reward), then return the full improved module (same contract, standard library only, never "
            "raises, < 5 ms) in one ```python fenced block. Keep everything that worked."
        )

    def _unstick_prompt(self, obs, stall_trace: list, stalled_for: int) -> str:
        return (
            "The game is STILL RUNNING with your bridge module:\n```python\n"
            + (self.mem.get("last_code") or "(none)")[-6000:] + "\n```\n"
            f"The score has not increased for {stalled_for} ticks - the agent is cycling or wandering instead "
            "of reaching the reward. Sampled situations from that stretch (your 'state' dict, option labels, "
            "action posted), oldest first:\n" +
            "\n".join(json.dumps(s, default=str)[:600] for s in stall_trace) +
            "\nCurrent observation:\n" + _obs_text(obs, 2500) +
            "\n\nExplain in one line why the utility never lets the agent commit to the reward, then return "
            "the full improved module (same contract, standard library only, never raises, < 5 ms) in one "
            "```python fenced block. Keep the safety logic that kept it alive."
        )

    # ---------------------------------------------------------- code loading
    def _register_module(self, code: str, source: str) -> None:
        for m in self.mem["modules"]:
            if m["code"] == code:
                return
        self.mem["modules"].append({"code": code, "source": source, "scores": []})

    def _try_adopt(self, text: str, obs: Any, names: list[str], source: str) -> tuple[Bridge | None, str]:
        """Compile + validate Astra's reply on an observation. Returns (bridge, error)."""
        code = _extract_code(text)
        if not code:
            self.ctx.log(event="bridge_reject", source=source, reason="no code block")
            return None, "no ```python block containing def bridge(obs) was found in your reply"
        self.mem["last_code"] = code
        try:
            b = Bridge(code, names)
            t0 = time.time()
            b(obs)  # validate on a real observation
            dt = time.time() - t0
        except Exception as e:  # noqa: BLE001
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.ctx.log(event="bridge_reject", source=source, reason=err[-800:])
            return None, err
        if dt > 0.1:
            self.ctx.log(event="bridge_reject", source=source, reason=f"too slow: {dt:.3f}s")
            return None, f"bridge(obs) took {dt:.3f} s on one observation; it must run in < 5 ms"
        self.mem["bridge_code"] = code
        self._register_module(code, source)
        self.ctx.log(event="bridge_adopted", source=source, chars=len(code), rules=bool(b.rules),
                     n_modules=len(self.mem["modules"]))
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

    def _pick_module_for_episode(self) -> None:
        """Scoreboard: try untested revisions once; otherwise play the best tested module."""
        mods = self.mem["modules"]
        cur = self.mem.get("bridge_code")
        if not mods or cur is None:
            return
        cur_m = next((m for m in mods if m["code"] == cur), None)
        if cur_m is None or not cur_m["scores"]:
            return  # untested (or unknown) module: give it its one trial
        tested = [m for m in mods if m["scores"]]
        best = max(tested, key=lambda m: sum(m["scores"]) / len(m["scores"]))
        if best is not cur_m and (sum(best["scores"]) / len(best["scores"])) > (sum(cur_m["scores"]) / len(cur_m["scores"])):
            self.mem["bridge_code"] = best["code"]
            self.mem["last_code"] = best["code"]
            self.ctx.log(event="module_revert", to_source=best["source"], best_scores=best["scores"],
                         cur_scores=cur_m["scores"])

    def _spawn_astra(self, kind: str, prompt: str) -> None:
        self.mem["pending"] = (kind, self.ctx.spawn(lambda p=prompt, k=kind: self._think(p, k)))

    def _harvest(self, obs, names, *, block: bool, deadline: float | None = None) -> Bridge | None:
        """Collect a finished (or, if ``block``, awaited) Astra reply; adopt it or queue a repair."""
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
        if low:
            return max(low, key=lambda n: probs.get(n, 0.0) * (1.0 - risk[n]) ** 2 + 1e-3 * (1.0 - risk[n]))
        return min(names, key=lambda n: risk[n])

    # ------------------------------------------------------------------ loop
    def run_episode(self, game):
        ctx = self.ctx
        spec = game.action_spec
        names = list(spec.names)
        rules_text = f"{game.describe()}\n{spec.describe()}"
        obs = game.observe()
        last_summary = self.mem["episodes"][-1] if self.mem["episodes"] else None
        ep_index = len(self.mem["episodes"])
        rng = random.Random(1000 + ep_index)

        self._pick_module_for_episode()
        bridge: Bridge | None = self._load_cached(obs, names)
        if bridge is None:
            bridge = self._harvest(obs, names, block=False)   # a reply from a previous episode?
        if bridge is None and self.mem.get("pending") is None:
            self.mem["repairs"] = 0
            self._spawn_astra("code", self._code_prompt(game, obs))
        elif (bridge is not None and self.mem.get("pending") is None and last_summary
              and last_summary.get("had_bridge") and last_summary.get("needs_improve")
              and not last_summary.get("improved") and self.mem["revisions"] < MAX_REVISIONS):
            self.mem["revisions"] += 1
            last_summary["improved"] = True
            self._spawn_astra("improve", self._improve_prompt(last_summary))

        jev_fut = None                 # background Jev tie-break call (bridge phase)
        jev_res: dict[str, Any] | None = None
        gen_fut = None                 # background generic Jev call (no bridge yet)
        gen_tick = 0
        last_action: str | None = None
        recent_obs: list[str] = []
        recent_actions: list[str] = []
        tick_i = 0
        posted_ticks = 0
        n_changed = 0
        n_explore = 0
        # stall bookkeeping
        last_score = float(game.score)
        last_gain_tick = 0
        max_stall = 0
        explore_until = 0
        unstick_done = False
        stall_trace: list[dict[str, Any]] = []     # sampled situations since the last score gain
        stall_trace_all: list[dict[str, Any]] = []

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

        def _jev_call(state, opts, t):
            q = {"move": Choice(
                instructions="You are the reflex controller. Pick the option with the best outcome: "
                             "stay alive first, then gain score. Each option is labelled with its "
                             "predicted consequence; follow coach_rules.",
                criteria=dict(opts))}
            try:
                ans = ctx.decide(state, q)
            except Exception as e:  # noqa: BLE001
                ctx.log(event="jev_error", error=str(e)[:300])
                return None
            a = ans.get("move") if ans else None
            if a is None:
                return None
            return {"tick": t, "probs": dict(a.probabilities) or {str(a.value): 1.0}, "pick": str(a.value)}

        while not ctx.done:
            obs = game.observe()
            tick_i += 1
            recent_obs.append(_obs_text(obs, 1500)); recent_obs = recent_obs[-4:]

            # ---- score / stall bookkeeping ---------------------------------
            sc = float(game.score)
            if sc > last_score:
                last_gain_tick = tick_i
                stall_trace = []
            last_score = sc
            stalled_for = tick_i - last_gain_tick
            max_stall = max(max_stall, stalled_for)

            # ---- collect Astra results (non-blocking while the game runs) ----
            if self.mem.get("pending") is not None:
                got = self._harvest(obs, names, block=False)
                if got is not None:
                    bridge = got
                    gen_fut = None
                    jev_res = None

            action: str | None = None
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
                # ---------------- bridge phase: decide in microseconds -------------
                opts, util = feat["options"], feat["utility"]
                lo, hi = min(util.values()), max(util.values())
                score = {k: ((util[k] - lo) / (hi - lo) if hi > lo else 1.0) for k in opts}
                best = max(score, key=score.get)
                action = best
                how = "bridge"
                # Jev tie-break (answer from a background call, <= JEV_MAX_AGE_TICKS old)
                if jev_fut is not None and jev_fut.done():
                    try:
                        jev_res = jev_fut.result()
                    except Exception:  # noqa: BLE001
                        jev_res = None
                    jev_fut = None
                if jev_res and tick_i - jev_res["tick"] <= JEV_MAX_AGE_TICKS:
                    near = [k for k in opts if score[best] - score[k] <= TIE_MARGIN]
                    if len(near) > 1:
                        p = {k: jev_res["probs"].get(k, 0.0) for k in near}
                        pick = max(p, key=p.get)
                        if p[pick] > 0 and pick != best:
                            action, how = pick, "jev"
                            n_changed += 1
                # stall exploration: randomise among near-best options for a few ticks
                if stalled_for >= STALL_TICKS and tick_i < explore_until:
                    cands = [k for k in opts if score[best] - score[k] <= EXPLORE_MARGIN]
                    if len(cands) > 1:
                        action, how = rng.choice(cands), "explore"
                        n_explore += 1
                ctx.act(action)
                last_action = action
                posted_ticks += 1
                ctx.log(event="decision", tick=tick_i, options=list(opts), best=best, action=action, how=how,
                        stalled_for=stalled_for)
                if jev_fut is None and len(opts) > 1:
                    second = sorted(score.values(), reverse=True)[1]
                    if score[best] - second <= TIE_MARGIN + 0.05:      # only ask when it can matter
                        state = {"coach_rules": bridge.rules or rules_text[:600],
                                 "situation": feat["state"], "last_action": last_action}
                        jev_fut = ctx.spawn(lambda s=state, o=dict(opts), t=tick_i: _jev_call(s, o, t))
                # sampled trace of the stalled stretch for Astra
                if stalled_for >= 1 and (tick_i % 4 == 0):
                    rec = {"tick": tick_i, "state": feat["state"], "options": opts, "action": action}
                    stall_trace.append(rec); stall_trace = stall_trace[-STALL_SAMPLE * 3:]
                    stall_trace_all.append(rec); stall_trace_all = stall_trace_all[-STALL_SAMPLE * 3:]
            else:
                # ---------------- generic phase (no bridge yet) --------------------
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
                if action is None and last_action is not None:
                    ctx.act(last_action)     # re-affirm the standing decision while Jev is still busy
                    posted_ticks += 1

            # ---- stall handling: explore bursts + one mid-episode Astra revision ----
            if bridge is not None and stalled_for >= STALL_TICKS and stalled_for % STALL_TICKS == 0:
                explore_until = tick_i + EXPLORE_TICKS
                ctx.log(event="stall", tick=tick_i, stalled_for=stalled_for, score=sc)
                if (not unstick_done and self.mem.get("pending") is None
                        and self.mem["revisions"] < MAX_REVISIONS):
                    unstick_done = True
                    self.mem["revisions"] += 1
                    sample = stall_trace[-STALL_SAMPLE:]
                    self._spawn_astra("unstick", self._unstick_prompt(obs, sample, stalled_for))

            recent_actions.append(action or "(none)"); recent_actions = recent_actions[-4:]
            ctx.wait_for_tick()

        # ---- game over: free window. Harvest / repair / improve synchronously -----
        info = {}
        try:
            info = dict(game.info())
        except Exception:  # noqa: BLE001
            pass
        ticks = _ticks()
        max_ticks = getattr(game, "max_ticks", None)
        if not max_ticks:
            m = re.search(r"max(?:imum)?\s+(\d+)\s+ticks", game.describe(), flags=re.I)
            max_ticks = int(m.group(1)) if m else None
        clean_end = bool(max_ticks and ticks >= max_ticks)
        stalled = max_stall >= STALL_TICKS
        summary = {"ticks": ticks, "score": game.score, "info": info, "clean_end": clean_end,
                   "last_obs": recent_obs, "last_actions": recent_actions,
                   "had_bridge": bridge is not None, "improved": False,
                   "max_stall": max_stall, "stall_trace": stall_trace_all[-STALL_SAMPLE:] if stalled else [],
                   "needs_improve": (not clean_end) or stalled}
        self.mem["episodes"].append(summary)
        code_now = self.mem.get("bridge_code")
        if bridge is not None and code_now:
            self._register_module(code_now, "unknown")
            for m_ in self.mem["modules"]:
                if m_["code"] == code_now:
                    m_["scores"].append(float(game.score))
        ctx.log(event="episode_summary", ticks=ticks, score=game.score, posted=posted_ticks,
                jev_changed=n_changed, explored=n_explore, max_stall=max_stall,
                had_bridge=bridge is not None, info=info,
                modules=[(m_["source"], m_["scores"]) for m_ in self.mem["modules"]])

        deadline = time.time() + END_WAIT_S
        try:
            last_obs = game.observe()
        except Exception:  # noqa: BLE001
            last_obs = obs
        # 1) wait for the code / repair / unstick chain to finish
        while self.mem.get("pending") is not None and time.time() < deadline:
            got = self._harvest(last_obs, names, block=True, deadline=deadline)
            if got is not None:
                bridge = got
        # 2) post-mortem when the episode died or stalled (no clock pressure now)
        if (bridge is not None and self.mem.get("pending") is None and summary["had_bridge"]
                and summary["needs_improve"] and self.mem["revisions"] < MAX_REVISIONS
                and time.time() < deadline):
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
