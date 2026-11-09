"""Novelty-aware text controller for turn-based Jericho stories.

Astra proposes short parser macros; code keeps exact evidence, blocks stale
command/response loops across model calls, and supplies conservative probes when
planning output is unusable.  No VM or game-specific state is inspected.
"""
from __future__ import annotations

import json
import re
from collections import Counter, deque

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


LIFE = re.compile(r"^(?:restart|restore|save|quit|q|undo|script|unscript|transcript|record|replay)(?:\s|$)", re.I)
COMPOUND = re.compile(r"[\n\r\x00;{}\[\]]|\bthen\b|\.(?=\s+\S)", re.I)
PARSER = re.compile(
    r"not a verb|don't know the word|do not know|don't understand|didn't understand|"
    r"not a sentence|not recogni[sz]|can't see any such|cannot see any such|"
    r"no such (?:thing|object)|not here|aren't here|isn't here|"
    r"you (?:can't|cannot) (?:do|go|take|open)|not holding|not carrying|"
    r"you don't have|you do not have|which .{0,120}(?:mean|refer)|"
    r"what do you want|be more specific|yes or no|\(y/n\)|\[y/n\]", re.I | re.S)
DANGER = re.compile(
    r"pitch[- ](?:black|dark)|grue|total darkness|you are (?:dead|dying)|you have died|"
    r"\*\*\*.*(?:dead|died)|lamp.{0,50}(?:dim|flicker|out|running)|"
    r"(?:attacks|strikes|wounds|hits) you|you are wounded|rising water|"
    r"(?:stolen|steals|disarmed)|(?:starving|very hungry|very thirsty)", re.I | re.S)
MENU = re.compile(r"q\s*=\s*resume game|return\s*=\s*read subject|help menu", re.I)

# Words used only to turn visible nouns into a small, generic probe set.  The
# probe is an emergency information action, never a guessed walkthrough.
STOP = set("the and you are was were this that with from have has here there into your about what which then only a an of to in on for is it as by can do does how not my our their will would could should game story room score inventory look examine command text latest current".split())
DIRECTIONS = ("north", "south", "east", "west", "up", "down", "in", "out")


def obs_dict(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            x = json.loads(value)
            if isinstance(x, dict):
                return x
        except Exception:
            pass
    return {"text": "" if value is None else str(value)}


def text_of(obs):
    return str(obs.get("text", "")) if isinstance(obs, dict) else str(obs or "")


def terminal(obs):
    return isinstance(obs, dict) and any(bool(obs.get(k)) for k in ("terminated", "truncated", "won", "lost"))


def clean(value):
    if not isinstance(value, str):
        return None
    s = value.strip()
    s = re.sub(r"^```(?:json|text|command)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()
    s = re.sub(r"^(?:[-*>]|\d+[.)])\s*", "", s).strip()
    s = re.sub(r"^(?:command|action|input|cmd)\s*[:=|]\s*", "", s, flags=re.I).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        s = s[1:-1].strip()
    if not s or len(s) > 200 or LIFE.match(s) or COMPOUND.search(s):
        return None
    if s.casefold() in {"command", "action", "none", "null", "noop"}:
        return None
    return s.rstrip(".").strip() or None


def commands_from(value, depth=0):
    if depth > 8:
        return []
    if isinstance(value, str):
        c = clean(value)
        return [c] if c else []
    if isinstance(value, list):
        out = []
        for item in value[:16]:
            out.extend(commands_from(item, depth + 1))
        return out[:16]
    if isinstance(value, dict):
        # Prefer explicit command payloads.  Never turn the channel label
        # "command" into the parser input.
        for key in ("commands", "program", "plan", "actions", "steps", "command", "cmd", "input", "value"):
            if key in value:
                v = value[key]
                if key == "command" and isinstance(v, str) and v.casefold() == "command":
                    continue
                out = commands_from(v, depth + 1)
                if out:
                    return out
    return []


def decode(raw):
    if not isinstance(raw, str):
        return [], ""
    s = raw.strip()[:50000]
    obj = None
    for candidate in (s, re.sub(r"^```(?:json|text)?\s*|\s*```$", "", s, flags=re.I)):
        try:
            obj = json.loads(candidate)
            break
        except Exception:
            continue
    if obj is None:
        # Recover a JSON object embedded in a brief preface.
        dec = json.JSONDecoder()
        for m in re.finditer(r"[\[{]", s):
            try:
                obj, _ = dec.raw_decode(s[m.start():])
                break
            except Exception:
                pass
    memory = ""
    if isinstance(obj, dict) and isinstance(obj.get("memory"), str):
        memory = obj["memory"][:3500]
    out = commands_from(obj)
    if not out:
        # Safe plain-text recovery: explicit command lines only, then one
        # short bare line.  Arbitrary prose cannot become a long action.
        for line in s.splitlines():
            if re.match(r"^\s*(?:command|action|input|cmd)\s*[:=]", line, re.I):
                c = clean(line)
                if c:
                    out.append(c)
        if not out:
            lines = [x.strip().strip("`\"'") for x in s.splitlines() if x.strip()]
            if len(lines) == 1:
                c = clean(lines[0])
                if c:
                    out = [c]
    return out[:16], memory


def visible_nouns(response):
    # Preserve hyphenated and apostrophe names, but avoid making every prose
    # word into an action.  First nouns are usually the salient objects.
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", response)
    result = []
    for w in words:
        x = w.casefold()
        if x in STOP or x in result or len(x) > 28:
            continue
        if x in {d for d in DIRECTIONS}:
            continue
        result.append(x)
        if len(result) >= 8:
            break
    return result


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.records = []
        self.notebook = ""
        self.reason = "Initial state: inspect the premise and pursue a reachable prerequisite."
        self.failed = deque(maxlen=18)
        self.seen_pairs = Counter()
        self.seen_commands = Counter()
        self.probes = deque()
        self.last_score = None
        self.invalid = 0
        self.model_errors = 0

    def record(self, cmd, obs):
        row = {
            "id": len(self.records),
            "cmd": cmd,
            "text": text_of(obs),
            "score": obs.get("score") if isinstance(obs, dict) else None,
            "delta": obs.get("score_delta") if isinstance(obs, dict) else None,
            "moves": obs.get("native_moves") if isinstance(obs, dict) else None,
            "step": obs.get("step_count") if isinstance(obs, dict) else None,
        }
        self.records.append(row)
        self.ctx.memory["transcript"] = self.records
        return row

    def compact_packet(self, obs, remaining):
        recent = []
        for r in self.records[-10:]:
            recent.append({k: (v if k != "text" else str(v)[:1000]) for k, v in r.items()})
        # Keep only the most useful stale-pair evidence.  This makes prompt
        # size stable even in long episodes.
        blocked = [{"cmd": c, "response": t[:240], "score": s}
                   for (c, t, s), n in self.seen_pairs.items() if n >= 1][-12:]
        return {
            "game": obs.get("game_id"), "score": obs.get("score"), "max": obs.get("max_score"),
            "moves": obs.get("native_moves"), "step": obs.get("step_count"),
            "calls_left": remaining.get("system2_calls"), "wall_left": round(float(remaining.get("wall_s", 0)), 1),
            "why": self.reason, "notebook_unverified": self.notebook[:3500],
            "opening": str(self.records[0].get("text", ""))[:4000] if self.records else "",
            "recent": recent, "blocked_exact_pairs": blocked,
            "failed_commands": list(self.failed)[-12:], "current": {**obs, "text": text_of(obs)[:5000]},
        }

    def prompt(self, obs, remaining):
        return ("You control a turn-based offline Jericho parser story. Maximize FINAL native score. "
                "Thinking costs wall time but no game turn; each submitted parser string costs one turn. "
                "Use only the evidence packet. Do not invent a walkthrough, use lifecycle commands, or "
                "repeat an exact command/response pair listed as blocked. Return JSON only: "
                "{\"goal\":\"...\",\"memory\":\"concise evidence notebook\","
                "\"commands\":[\"one parser command\",...]}. Give 4-10 short commands whose later "
                "steps are safe consequences of earlier ones; code stops on parser errors, clarification, "
                "hazards, score changes, or stale output. Use exact nouns and verbs from the story. Prefer "
                "reachable scoring prerequisites, light/access/tools, and clue-backed routes. Queries may "
                "be batched, but avoid repeated inspection with no new state. A clarification answer is "
                "a command only when the latest text asks for it. Commands must be <=200 chars, one line.\n\n"
                + json.dumps(self.compact_packet(obs, remaining), ensure_ascii=False, default=str))

    def make_probes(self, obs):
        if self.probes:
            return
        t = text_of(obs)
        for noun in visible_nouns(t):
            for verb in ("examine", "read", "open", "take"):
                c = clean(f"{verb} {noun}")
                if c and self.seen_commands[c.casefold()] == 0:
                    self.probes.append(c)
        # Movement is last: it can discover a route but is less conservative.
        for d in DIRECTIONS:
            if self.seen_commands[d] == 0:
                self.probes.append(d)

    def blocked(self, cmd, obs):
        key = (cmd.casefold(), re.sub(r"\s+", " ", text_of(obs)).strip(), obs.get("score") if isinstance(obs, dict) else None)
        return self.seen_pairs[key] >= 1

    def run_episode(self, game) -> None:
        ctx = self.ctx
        try:
            obs = obs_dict(ctx.observe())
        except Exception:
            obs = {"text": ""}
        self.record(None, obs)
        self.last_score = obs.get("score")
        queue = deque()
        while not ctx.done and not terminal(obs):
            rem = ctx.budget.remaining()
            wall = float(rem.get("wall_s", 0) or 0)
            if rem.get("system2_calls", 0) <= 0 or wall < 5:
                return
            if not queue:
                try:
                    result = ctx.think(self.prompt(obs, rem), session_id=None,
                                       timeout_s=max(1.0, min(180.0, wall - 2.0)))
                    queue.extend(decode(getattr(result, "text", ""))[0])
                    raw_mem = decode(getattr(result, "text", ""))[1]
                    if raw_mem:
                        self.notebook = raw_mem
                        ctx.memory["notebook"] = self.notebook
                    self.model_errors = 0
                except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                    return
                except (TimeoutError, RuntimeError) as exc:
                    ctx.log("think_failed", error=type(exc).__name__)
                    self.model_errors += 1
                    self.reason = "A reasoning call failed; continue from unchanged exact evidence."
                    if ctx.done or self.model_errors >= 3:
                        return
                except Exception as exc:
                    ctx.log("think_failed", error=type(exc).__name__)
                    self.model_errors += 1
                    if ctx.done or self.model_errors >= 3:
                        return
                if not queue:
                    self.invalid += 1
                    self.reason = "Planner output was malformed or empty; use strict JSON commands."
                    self.make_probes(obs)
                    if self.probes:
                        queue.append(self.probes.popleft())
                    elif self.invalid >= 3:
                        return
                else:
                    self.invalid = 0
                    self.reason = "Review the latest result and continue only with a non-stale command."
            if not queue:
                continue
            cmd = clean(queue.popleft())
            if not cmd:
                continue
            # Exact pair recurrence is a stronger guard than command recurrence:
            # the same route command is allowed after a changed observation.
            if self.blocked(cmd, obs):
                self.failed.append({"cmd": cmd, "why": "exact command/response pair already observed"})
                queue.clear()
                self.reason = "Planner proposed a stale exact command/response pair; choose a new test."
                continue
            before_score = obs.get("score")
            before_text = text_of(obs)
            try:
                raw = ctx.act(cmd)
                obs = obs_dict(raw if raw is not None else ctx.observe())
            except Exception as exc:
                ctx.log("act_failed", command=cmd, error=type(exc).__name__)
                self.failed.append({"cmd": cmd, "why": type(exc).__name__})
                queue.clear()
                self.reason = "The interface rejected the command; revise from exact evidence."
                if ctx.done:
                    return
                continue
            rec = self.record(cmd, obs)
            response = text_of(obs)
            key = (cmd.casefold(), re.sub(r"\s+", " ", response).strip(), obs.get("score"))
            self.seen_pairs[key] += 1
            self.seen_commands[cmd.casefold()] += 1
            if terminal(obs) or ctx.done:
                return
            # Some stories expose a character-input help menu.  The displayed
            # `Q = resume game` is a contextual answer, not a lifecycle quit;
            # honor it once so later planning sees the normal parser again.
            if MENU.search(response):
                queue.clear()
                if re.search(r"q\s*=\s*resume game", response, re.I):
                    try:
                        resumed = ctx.act("q")
                        obs = obs_dict(resumed if resumed is not None else ctx.observe())
                        r2 = self.record("q", obs)
                        response2 = text_of(obs)
                        key2 = ("q", re.sub(r"\s+", " ", response2).strip(), obs.get("score"))
                        self.seen_pairs[key2] += 1
                        self.seen_commands["q"] += 1
                        ctx.log("menu_resumed", observation_id=r2["id"])
                    except Exception as exc:
                        ctx.log("menu_resume_failed", error=type(exc).__name__)
                        self.reason = "The contextual menu answer was rejected; inspect the latest evidence."
                        if ctx.done:
                            return
                        continue
                self.reason = "An interactive menu interrupted the plan; inspect the resumed parser state."
                if terminal(obs) or ctx.done:
                    return
                continue
            delta = (obs.get("score") - before_score) if isinstance(obs.get("score"), (int, float)) and isinstance(before_score, (int, float)) else 0
            # Any state-changing score event, parser failure, clarification,
            # menu, hazard, or empty answer invalidates the unobserved suffix.
            if delta != 0 or PARSER.search(response) or DANGER.search(response) or MENU.search(response) or not response.strip():
                queue.clear()
                self.failed.append({"cmd": cmd, "response": response[:300], "delta": delta})
                self.reason = "The latest response changed the state or needs interpretation; replan from it."
            elif len(self.records) >= 2 and self.records[-1]["text"].strip() == self.records[-2]["text"].strip() and cmd.casefold() == str(self.records[-2]["cmd"]).casefold():
                queue.clear()
                self.reason = "The latest command repeated an unchanged result; find a different hypothesis."
            self.last_score = obs.get("score")
            ctx.log("step_recorded", command=cmd, observation_id=rec["id"], score=obs.get("score"), delta=delta)


def build(ctx):
    return Harness(ctx)
