"""Priority-policy controller for the fixed-pool Yu-Gi-Oh. benchmark.

Astra chooses a short, turn-level priority list.  The executor only applies a
priority when the current native menu contains the named card/command, so a
plan never carries an assumed token or future zone index.
"""
from __future__ import annotations
import json, re
from metaharness.core.contract import BudgetExceeded, ModelUnavailable

FLIPS = {"Man-Eater Bug", "Penguin Soldier", "Magician of Faith"}
EFFECT_WORDS = ("activate", "chain", "target", "tribute", "cost", "select", "order")

def O(x):
    if isinstance(x, dict): return x
    try:
        y = json.loads(x)
        return y if isinstance(y, dict) else {}
    except Exception: return {}

def num(x, d=0):
    try: return int(x)
    except Exception: return d

def card_of(choice, obs):
    c = O(choice.get("card", {}))
    if c.get("name") and c.get("name") != "unknown card": return c
    for p in O(obs).get("players", []):
        for z in ("hand", "monsters", "spells_traps"):
            for q in p.get(z, []) or []:
                if c.get("slot") is not None and q.get("slot") == c.get("slot") and q.get("zone") == c.get("zone"):
                    return q
                if c.get("code") is not None and q.get("code") == c.get("code"): return q
    return c

def ids(p):
    return [num(O(c).get("index"), i) for i,c in enumerate(p.get("candidates") or [])]

def limits(p):
    c = dict(p); c.update(O(p.get("constraints")))
    lo = num(c.get("min", c.get("minimum", c.get("min_count", 0))))
    hi = num(c.get("max", c.get("maximum", c.get("max_count", len(p.get("candidates") or [])))) )
    return lo, hi, c

def valid_compound(a, p):
    m = re.fullmatch(r"(select|order)\s+(\S+)\s+(-|\d+(?:,\d+)*)", str(a).strip())
    if not m or str(m.group(2)) != str(p.get("token")): return False
    xs = [] if m.group(3) == "-" else [int(x) for x in m.group(3).split(",")]
    av = ids(p)
    if len(xs) != len(set(xs)) or any(x not in av for x in xs): return False
    order = "order" in str(p.get("kind", "")).lower() or p.get("mode") == "order" or str(p.get("grammar", "")).startswith("order ")
    if (m.group(1) == "order") != order: return False
    if order: return sorted(xs) == sorted(av)
    lo, hi, c = limits(p)
    # Tribute prompts use a weight minimum; ordinary subsets use a count.
    tribute = "tribute" in str(p).lower() or "weight" in str(c).lower()
    if tribute:
        ws = p.get("weights") or []
        weights = {i: num(ws[j], O(raw).get("weight", 1)) for j,(i,raw) in enumerate(zip(av, p.get("candidates") or []))}
        if len(xs) > hi or sum(weights.get(i, 1) for i in xs) < lo: return False
    elif not (lo <= len(xs) <= hi):
        return False
    return True

def parse_reply(text, obs):
    p = O(obs).get("prompt") or {}; choices = p.get("choices") or []
    s = str(text or "")
    # Prefer an exact displayed action, including actions embedded in JSON.
    for c in choices:
        a = c.get("action")
        if a and (a == s.strip() or any(line.strip() == a for line in s.splitlines()) or
                  re.search(r"(?<![A-Za-z0-9_])" + re.escape(a) + r"(?![0-9])", s)):
            return a
    tok = re.escape(str(p.get("token", "")))
    m = re.search(rf"\b(choose|select|order)\s+{tok}\s+(-|[0-9,\s]+)", s)
    if not m: return None
    a = f"{m.group(1)} {p.get('token')} {m.group(2).replace(' ', '')}"
    if m.group(1) == "choose": return a if any(c.get("action") == a for c in choices) else None
    return a if valid_compound(a, p) else None

def fallback(obs):
    p = O(obs).get("prompt") or {}; cs = p.get("choices") or []
    # A singleton cancel is not a valid answer to a nonempty candidate prompt.
    if p.get("candidates"):
        kind = str(p.get("kind", "")).lower()
        if "order" in kind or p.get("mode") == "order":
            a = f"order {p.get('token')} {','.join(map(str, ids(p)))}"
            if valid_compound(a,p): return a
        lo, hi, _ = limits(p)
        if lo > 0:
            a = f"select {p.get('token')} {','.join(map(str, ids(p)[:lo]))}"
            if valid_compound(a,p): return a
        # Optional selections are safest as an explicit empty set.
        a = f"select {p.get('token')} -"
        if valid_compound(a,p): return a
    for c in cs:
        if c.get("action"): return c["action"]
    return "noop"

def menu_action(obs, priorities):
    p = O(obs).get("prompt") or {}; cs = p.get("choices") or []
    if not cs or p.get("candidates"): return None
    # Never use a priority to answer effect/chain/target prompts.
    kind = str(p.get("kind", "")).lower()
    labels = [(str(c.get("label", "")).lower(), c) for c in cs]
    if any(w in kind for w in ("chain", "target", "position", "summon", "activation")): return None
    for want in priorities:
        want = str(want).lower()
        for lab,c in labels:
            card = card_of(c, obs)
            if want in lab or want == str(card.get("name", "")).lower(): return c.get("action")
    # Conservative transparent commands only.
    if "main_command" in kind:
        summons = [(num(card_of(c,obs).get("attack")), c) for l,c in labels if "normal summon" in l and card_of(c,obs).get("name") not in FLIPS]
        if summons: return max(summons, key=lambda x:x[0])[1].get("action")
        end = next((c for l,c in labels if "end turn" in l or "end phase" in l), None)
        return end.get("action") if end else None
    if "battle" in kind or any("attack" in l for l,_ in labels):
        ats = [(num(card_of(c,obs).get("attack")), c) for l,c in labels if "attack" in l]
        if ats: return max(ats, key=lambda x:x[0])[1].get("action")
    return None

class Harness:
    def __init__(self, ctx):
        self.ctx = ctx; self.session = None

    def ask(self, game, obs, reason):
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None: return fallback(obs), []
        prompt = (game.describe() + "\nCURRENT OBSERVATION JSON:\n" + json.dumps(obs, ensure_ascii=False, separators=(",", ":")) +
                  "\n\nYou are the tactical planner. Native choices are authoritative. Return JSON only: "
                  '{"action":"exact current action or empty","priorities":["card/command names in order"],"reason":"under 30 words"}. '
                  "Priorities are preferences for this turn, not future token guesses. Respect one Normal Summon/Set, first-turn battle restriction, and hidden backrow. "
                  "Use removal for a meaningful threat; preserve Flip monsters for value; do not treat MST as negation. " + str(reason))
        try:
            r = ctx.think(prompt, session_id=self.session, timeout_s=150.0)
            self.session = r.session_id
            d = O(r.text)
            a = parse_reply(r.text, obs) or parse_reply(d.get("action", ""), obs)
            pr = d.get("priorities", []) if isinstance(d.get("priorities", []), list) else []
            return a, [str(x) for x in pr[:8]]
        except (BudgetExceeded, ModelUnavailable):
            return None, []
        except (TimeoutError, RuntimeError, Exception) as e:
            try: ctx.log("planner_recovery", error=type(e).__name__)
            except Exception: pass
            return None, []

    def run_episode(self, game):
        ctx = self.ctx; obs = game.observe(); priorities=[]; last_token=None; last_bad=0; turn_key=None; asked=False; calls=0
        while not ctx.done and not game.done:
            if not isinstance(obs, dict): obs = game.observe()
            p = obs.get("prompt") or {}; token = p.get("token")
            prog = ctx.progress(); bad = num(prog.get("illegal_actions"))
            own = obs.get("turn_player") == obs.get("agent_seat")
            tk = (obs.get("round"), obs.get("turn"), obs.get("phase"))
            if tk != turn_key: turn_key=tk; asked=False; priorities=[]
            stalled = token == last_token or bad > last_bad or prog.get("since_obs_change", 0) >= 2
            complex_prompt = bool(p.get("candidates")) or any(w in str(p.get("kind", "")).lower() for w in EFFECT_WORDS)
            need = stalled or (own and not asked and str(p.get("kind", "")) == "main_command") or (complex_prompt and not asked)
            action = None
            if need and calls < 28 and ctx.budget.remaining().get("system2_calls", 0) > 0:
                asked = True; calls += 1
                action, priorities = self.ask(game, obs, "progress=" + json.dumps(prog, separators=(",", ":")))
            if action is None: action = menu_action(obs, priorities)
            if action is None and p.get("candidates") and not complex_prompt: action = fallback(obs)
            if action is None: action = fallback(obs)
            try:
                nxt = ctx.act(action)
            except (BudgetExceeded, ModelUnavailable): return
            except Exception as e:
                try: ctx.log("action_recovery", action=action, error=type(e).__name__)
                except Exception: pass
                nxt = ctx.observe()
            obs = nxt if isinstance(nxt, dict) else ctx.observe()
            last_token = token; last_bad = num(ctx.progress().get("illegal_actions"))

def build(ctx): return Harness(ctx)
