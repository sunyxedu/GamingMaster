"""Evidence-gated conditional controller for turn-based Jericho stories.

The planner emits a small response-conditioned program.  The harness owns a
directed, transcript-derived room ledger and interrupts a program when it
would revisit an unchanged state or repeat a blocked edge.  This tests whether
the remaining loss is route cycling/speculative actions rather than more
context or larger macros.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, deque
from typing import Any

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


LIFE = re.compile(r"^(?:restart|restore|save|quit|undo|script|unscript|transcript|record|replay)(?:\s|$)", re.I)
COMPOUND = re.compile(r"[\x00\r\n;{}\[\]]|\b(?:then|and then)\b", re.I)
PARSER = re.compile(
    r"not a verb|don't know the word|do not know|don't understand|do not understand|"
    r"not a sentence|not recogni[sz]|can't see any such|cannot see any such|"
    r"no such (?:thing|object)|not here|can't go|cannot go|not holding|not carrying|"
    r"don't have|do not have|which .{0,120}(?:mean|refer)|be more specific|"
    r"yes or no|\by/n\b|nothing happens|doesn't seem to work|won't open", re.I | re.S)
DANGER = re.compile(
    r"pitch[- ]black|darkness|lamp.{0,70}(?:dim|flicker|out|dying|fading)|"
    r"attacks? you|strikes? you|wounded|bleeding|flood|stolen|disarmed|"
    r"you are dead|you have died|game over|congratulations|you win", re.I | re.S)
MENU = re.compile(r"q\s*=\s*resume game|press\s+space|character input|help menu", re.I)
MOVE = re.compile(r"^(?:n|s|e|w|north|south|east|west|up|down|in|out|enter|exit|climb)\b", re.I)
RISKY = re.compile(r"\b(?:attack|kill|fight|throw|drop|give|offer|eat|drink|burn|destroy|break|sacrifice|jump|dive|turn off)\b", re.I)


def obs_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"text": "" if value is None else str(value)}


def text_of(obs: Any) -> str:
    return str(obs.get("text", "")) if isinstance(obs, dict) else str(obs or "")


def terminal(obs: dict[str, Any]) -> bool:
    return any(bool(obs.get(k)) for k in ("terminated", "truncated", "won", "lost"))


def number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def clean_command(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.replace("\x00", " ").strip()
    for _ in range(3):
        old = s
        s = re.sub(r"^(?:[-*>]\s*|\d+[.)]\s*)", "", s).strip()
        s = re.sub(r"^(?:command|action|input|cmd)\s*[:=|]\s*", "", s, flags=re.I).strip()
        m = re.fullmatch(r"<(?:command|action)>(.*?)</(?:command|action)>", s, re.I | re.S)
        if m:
            s = m.group(1).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
            s = s[1:-1].strip()
        if s == old:
            break
    s = s.rstrip(".").strip()
    if not s or len(s) > 200 or LIFE.match(s) or COMPOUND.search(s):
        return None
    if s.casefold() in {"command", "action", "none", "null", "noop"}:
        return None
    return s


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [x.casefold()[:180] for x in value[:8] if isinstance(x, str) and x.strip()]


def decode(raw: Any) -> tuple[list[dict[str, Any]], str, str]:
    """Total decoder; arbitrary pre-flight model text is harmless."""
    if not isinstance(raw, str):
        return [], "", ""
    source = raw.strip()[:50000]
    obj: Any = None
    for candidate in (source, re.sub(r"^```[^\n]*\n?|\n?```$", "", source).strip()):
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(candidate)
                break
            except Exception:
                pass
        if obj is not None:
            break
    if not isinstance(obj, dict):
        return [], "", ""
    memory = obj.get("memory", obj.get("beliefs", ""))
    if isinstance(memory, (dict, list)):
        memory = json.dumps(memory, ensure_ascii=False)
    if not isinstance(memory, str):
        memory = ""
    goal = str(obj.get("goal", ""))[:700] if isinstance(obj.get("goal", ""), str) else ""
    source_nodes = obj.get("program", obj.get("commands", obj.get("plan", [])))
    if isinstance(source_nodes, str):
        source_nodes = [source_nodes]
    if not isinstance(source_nodes, list):
        return [], memory[:3200], goal
    nodes: list[dict[str, Any]] = []
    for i, item in enumerate(source_nodes[:16]):
        if isinstance(item, str):
            item = {"id": str(i), "cmd": item}
        if not isinstance(item, dict):
            continue
        cmd = clean_command(item.get("cmd", item.get("command", item.get("input"))))
        if not cmd:
            continue
        ident = str(item.get("id", i))[:40]
        node = {"id": ident, "cmd": cmd, "ok": str(item.get("ok", "END"))[:40],
                "edges": [], "evidence": _strings(item.get("evidence")),
                "proceed": item.get("proceed") is True}
        edges = item.get("edges", [])
        if isinstance(edges, list):
            for edge in edges[:6]:
                if not isinstance(edge, dict):
                    continue
                has = _strings(edge.get("has"))
                if has:
                    node["edges"].append({"has": has, "lacks": _strings(edge.get("lacks")),
                                           "to": str(edge.get("to", "END"))[:40]})
        nodes.append(node)
    return nodes, memory[:3200], goal


def state_sig(obs: dict[str, Any]) -> str:
    # Room titles are generally the first non-empty line.  Include a short
    # description prefix so repeated generic titles remain distinguishable.
    lines = [re.sub(r"\s+", " ", x).strip() for x in text_of(obs).splitlines() if x.strip()]
    room = " | ".join(lines[:3])[:420]
    # Native move count is intentionally excluded: returning to the same room
    # with the same score is the cycle signal, regardless of how many turns
    # were spent getting there.
    return f"{room}\u241f{obs.get('score')}"


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.records: list[dict[str, Any]] = []
        self.failed: deque[dict[str, Any]] = deque(maxlen=16)
        self.notebook = ""
        self.goal = ""
        self.reason = "Start from the opening text and identify a reachable scoring prerequisite."
        self.edge_results: Counter[tuple[str, str, str]] = Counter()
        self.move_attempts: Counter[tuple[str, str]] = Counter()
        self.state_visits: Counter[str] = Counter()
        self.inventory_text = ""
        self.model_errors = 0
        self.invalid = 0

    def record(self, command: str | None, obs: dict[str, Any]) -> None:
        row = {"step": obs.get("step_count", len(self.records)), "command": command,
               "text": text_of(obs)[:5000], "score": obs.get("score"),
               "delta": obs.get("score_delta"), "moves": obs.get("native_moves")}
        self.records.append(row)
        self.ctx.memory["transcript"] = self.records[-100:]
        if command and command.casefold() in {"inventory", "i"} and not PARSER.search(row["text"]):
            self.inventory_text = row["text"][:1200]

    def packet(self, game, obs: dict[str, Any], rem: dict[str, Any]) -> str:
        blocked = [{"room": r, "cmd": c, "response": t[:240], "count": n}
                   for (r, c, t), n in list(self.edge_results.items())[-18:] if n]
        data = {
            "status": {k: obs.get(k) for k in ("game_id", "score", "max_score", "score_delta", "native_moves", "step_count", "terminated", "truncated", "won", "lost")},
            "opening": self.records[0]["text"][:3500] if self.records else "",
            "current": {**obs, "text": text_of(obs)[:5000]},
            "recent": self.records[-10:], "inventory_last": self.inventory_text,
            "blocked_edges": blocked, "state_visit_counts": dict(list(self.state_visits.items())[-12:]),
            "failed": list(self.failed)[-12:], "memory_unverified": self.notebook,
            "why_replan": self.reason, "budget": {"calls": rem.get("system2_calls"), "wall_s": round(number(rem.get("wall_s")), 1)},
        }
        return (
            "You control a turn-based offline Jericho parser story. Maximize FINAL native score. "
            "Every command consumes one turn; thinking is free. The parser value is the command "
            "text, never the channel label `command`. Return ONLY JSON with this schema:\n"
            '{"goal":"...","memory":"brief evidence notes","program":[{"id":"a","cmd":"...","ok":"b","edges":[{"has":["literal response"],"to":"c"}],"evidence":["exact prior phrase"],"proceed":false}]}\n'
            "A program is response-conditioned: after each command code follows the first edge "
            "whose literal has text matches, otherwise ok. Use 3-9 commands and stop at an "
            "uncertain or irreversible decision. Use exact nouns from evidence. Do not invent "
            "walkthroughs, lifecycle/compound commands, or a route through a repeated state. "
            "For attack/give/drop/throw/eat/drink/burn/break/sacrifice/jump actions, set "
            "proceed:true only when the supplied evidence explicitly supports that operation; "
            "include a short exact evidence phrase. Code interrupts parser errors, clarification, "
            "hazards, score loss, menus, repeated blocked edges, and room-state cycles. Positive "
            "score alone does not prove every later suffix. Keep memory provisional and cite steps.\n\n"
            f"GAME:\n{str(game.describe())[:2800]}\nACTION SPEC:\n{str(game.action_spec.describe())[:1600]}\n"
            f"EVIDENCE PACKET:\n{json.dumps(data, ensure_ascii=False, default=str)}"
        )

    def run_episode(self, game) -> None:
        ctx = self.ctx
        if getattr(ctx, "system2", True) is None:
            return
        try:
            obs = obs_dict(ctx.observe())
        except Exception:
            obs = obs_dict(game.observe())
        self.record(None, obs)
        self.state_visits[state_sig(obs)] += 1
        while not ctx.done and not terminal(obs):
            rem = ctx.budget.remaining()
            if rem.get("system2_calls", 0) <= 0 or number(rem.get("wall_s")) < 8:
                return
            try:
                result = ctx.think(self.packet(game, obs, rem), session_id=None,
                                   timeout_s=max(1.0, min(120.0, number(rem.get("wall_s")) - 5.0)))
                nodes, memory, goal = decode(getattr(result, "text", ""))
                self.model_errors = 0
            except (BudgetExceeded, EpisodeOver, ModelUnavailable):
                return
            except (TimeoutError, RuntimeError) as exc:
                ctx.log("think_failed", error=type(exc).__name__)
                self.model_errors += 1
                if ctx.done or self.model_errors >= 3:
                    return
                self.reason = "Reasoning call failed; continue from unchanged evidence."
                continue
            except Exception as exc:
                ctx.log("think_parse_failed", error=type(exc).__name__)
                self.model_errors += 1
                if self.model_errors >= 3:
                    return
                continue
            if memory:
                self.notebook = memory
                ctx.memory["notebook"] = memory
            if goal:
                self.goal = goal
            if not nodes:
                self.invalid += 1
                self.reason = "Planner output was empty; return JSON program nodes with parser text."
                if self.invalid == 1 and not PARSER.search(text_of(obs)) and not MENU.search(text_of(obs)):
                    try:
                        obs = obs_dict(ctx.act("look"))
                        self.record("look", obs)
                    except Exception:
                        if ctx.done:
                            return
                if self.invalid >= 3:
                    return
                continue
            self.invalid = 0
            node_map = {n["id"]: n for n in nodes}
            current = nodes[0]["id"]
            visits: Counter[str] = Counter()
            local_outcomes: Counter[tuple[str, str, str]] = Counter()
            ctx.log("program_start", goal=self.goal, nodes=len(nodes))
            for _ in range(24):
                if ctx.done or terminal(obs) or current not in node_map:
                    break
                visits[current] += 1
                if visits[current] > 2:
                    self.reason = "Program loop limit reached; choose a different objective."
                    break
                node = node_map[current]
                cmd = node["cmd"]
                room = state_sig(obs)
                # A second visit to an unchanged state means a route cycle,
                # even when the alternating commands differ.
                if (self.state_visits[room] > 1 and number(obs.get("score_delta")) == 0
                        and MOVE.match(cmd) and self.move_attempts[(room, cmd.casefold())] >= 1):
                    self.reason = "Directed room ledger detected a no-progress state cycle; explore another edge."
                    break
                if RISKY.search(cmd):
                    evidence_blob = "\n".join(r["text"] for r in self.records).casefold()
                    cited = [p for p in node.get("evidence", []) if p and p in evidence_blob]
                    if not node.get("proceed") or not cited:
                        self.reason = "Risky command lacked an exact cited evidence phrase and proceed gate; replan safely."
                        self.failed.append({"command": cmd, "reason": self.reason})
                        break
                response_before = text_of(obs)
                try:
                    out = ctx.act(cmd)
                    obs = obs_dict(out if out is not None else ctx.observe())
                except Exception as exc:
                    ctx.log("act_failed", command=cmd, error=type(exc).__name__)
                    self.failed.append({"command": cmd, "error": type(exc).__name__})
                    self.reason = "Action was rejected; repair the parser command."
                    if ctx.done:
                        return
                    break
                self.record(cmd, obs)
                response = text_of(obs)
                delta = number(obs.get("score")) - number(self.records[-2].get("score"))
                key = (room, cmd.casefold(), re.sub(r"\s+", " ", response).strip()[:900])
                self.edge_results[key] += 1
                local_outcomes[key] += 1
                if MOVE.match(cmd):
                    self.move_attempts[(room, cmd.casefold())] += 1
                    self.state_visits[state_sig(obs)] += 1
                if terminal(obs) or ctx.done:
                    return
                interrupted = bool(PARSER.search(response) or DANGER.search(response) or MENU.search(response) or not response.strip())
                if delta < 0 or local_outcomes[key] >= 2 or self.edge_results[key] >= 2:
                    interrupted = True
                if interrupted:
                    self.failed.append({"command": cmd, "response": response[:350], "delta": delta})
                    self.reason = "Latest response invalidated the program; revise from exact evidence."
                    break
                nxt = "END"
                low = response.casefold()
                for edge in node["edges"]:
                    if any(x in low for x in edge["has"]) and not any(x in low for x in edge["lacks"]):
                        nxt = edge["to"]
                        break
                if nxt == "END":
                    nxt = node.get("ok", "END")
                if nxt not in node_map:
                    self.reason = "Program reached a novel result; replan from it."
                    break
                current = nxt
            ctx.log("program_stop", reason=self.reason)


def build(ctx):
    return Harness(ctx)
