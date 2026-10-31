"""Deterministic public-state search with checked one-step recovery."""
from __future__ import annotations

import itertools
import json
import re
from functools import lru_cache


CODES = tuple(itertools.permutations(range(1, 7), 4))
TEXT = tuple("[" + " ".join(map(str, c)) + "]" for c in CODES)
INDEX = {a: i for i, a in enumerate(TEXT)}
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
CLUE_RE = re.compile(
    r"Submitted\s*(\[[1-6\s,]+\])\s*\.\s*Feedback:\s*"
    r"(\d+)\s*black\s*peg\(s\),\s*(\d+)\s*white\s*peg\(s\)", re.I)


def parse_action(raw):
    if not isinstance(raw, str):
        return None
    values = re.findall(r"[1-6]", raw)
    if len(values) != 4:
        return None
    action = "[" + " ".join(values) + "]"
    return action if action in INDEX else None


def parse_clues(obs):
    clues = []
    text = obs.get("observation_text", "") if isinstance(obs, dict) else ""
    for raw, black, white in CLUE_RE.findall(text):
        action = parse_action(raw)
        item = (INDEX[action], int(black), int(white)) if action else None
        if item is None or item[1] + item[2] > 4:
            raise ValueError("Unrecognized Mastermind feedback")
        if item not in clues:
            clues.append(item)
    if len(clues) != obs.get("move_count", 0):
        raise ValueError("Feedback count disagrees with native move_count")
    if len({g for g, _, _ in clues}) != len(clues):
        raise ValueError("Repeated or conflicting feedback entry")
    return tuple(clues)


def feedback(guess, secret):
    black = sum(a == b for a, b in zip(guess, secret))
    return black, len(set(guess) & set(secret)) - black


def winner(board):
    for a, b, c in LINES:
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    return None


@lru_cache(maxsize=20000)
def minimax(board, turn, learner):
    won = winner(board)
    if won is not None:
        return 1 if won == learner else -1
    if None not in board:
        return 0
    values = []
    for i, cell in enumerate(board):
        if cell is None:
            child = board[:i] + (turn,) + board[i + 1:]
            values.append(minimax(child, 1 - turn, learner))
    return (max if turn == learner else min)(values)


class Search:
    def __init__(self):
        self.rows = {}

    def row(self, guess):
        if guess not in self.rows:
            code = CODES[guess]
            self.rows[guess] = tuple(feedback(code, secret) for secret in CODES)
        return self.rows[guess]

    def belief(self, clues):
        return tuple(s for s in range(360)
                     if all(self.row(g)[s] == (b, w) for g, b, w in clues))

    def final(self, belief, legal):
        present = [0] * 7
        position = [[0] * 7 for _ in range(4)]
        for secret in belief:
            for p, digit in enumerate(CODES[secret]):
                present[digit] += 1
                position[p][digit] += 1
        weights = [[present[d] + position[p][d] for d in range(7)] for p in range(4)]
        possible = set(belief)
        best = None
        for guess in legal:
            a, b, c, d = CODES[guess]
            value = weights[0][a] + weights[1][b] + weights[2][c] + weights[3][d]
            rank = (value, int(guess in possible), -guess)
            if best is None or rank > best[0]:
                best = (rank, guess)
        if best is None:
            raise ValueError("No legal final guess")
        return best[1], best[0][0], best[0][1]

    def choose(self, obs, ctx):
        clues = parse_clues(obs)
        legal_actions = obs.get("legal_actions", [])
        legal = tuple(INDEX[a] for a in legal_actions if a in INDEX)
        used = {g for g, _, _ in clues}
        if set(legal) != set(range(360)) - used:
            raise ValueError("Native legal guesses disagree with the public model")
        if len(clues) > 2:
            raise ValueError("Unexpected Mastermind horizon")
        belief = self.belief(clues)
        if not belief:
            raise ValueError("No secret satisfies public feedback")
        if not clues:
            action = "[1 2 3 4]"
            if INDEX[action] not in legal:
                raise ValueError("Canonical opening is not legal")
            return action, {"belief_size": 360, "method": "canonical opening"}
        if len(clues) == 2 or len(belief) == 1:
            guess, value, _ = self.final(belief, legal)
            return TEXT[guess], {"belief_size": len(belief),
                                 "expected_score": value / (8 * len(belief))}

        best = None
        checked = 0
        for guess in legal:
            if checked % 8 == 0 and (ctx.done or ctx.budget.remaining()["wall_s"] < 3):
                break
            groups = {}
            for secret in belief:
                groups.setdefault(self.row(guess)[secret], []).append(secret)
            remaining = tuple(g for g in legal if g != guess)
            total = solves = 0
            for response, subset in groups.items():
                if response == (4, 0):
                    total += 8 * len(subset)
                    solves += len(subset)
                else:
                    _, value, solved = self.final(tuple(subset), remaining)
                    total += value
                    solves += solved
            key = (total, solves, int(guess in belief), -guess)
            if best is None or key > best[0]:
                best = (key, guess)
            checked += 1
        if best is None:
            guess, _, _ = self.final(belief, legal)
            return TEXT[guess], {"method": "bounded legal fallback"}
        key, guess = best
        return TEXT[guess], {"belief_size": len(belief),
                             "expected_score": key[0] / (8 * len(belief)),
                             "probes_evaluated": checked}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.search = Search()
        self.rejected = set()
        self.session = None
        self.recovery = False

    def legal(self, obs):
        return [a for a in obs.get("legal_actions", [])
                if isinstance(a, str) and a != "noop" and a not in self.rejected]

    def recover(self, game, obs, issue):
        legal = self.legal(obs)
        if not legal:
            return None
        budget = self.ctx.budget.remaining()
        if budget["system2_calls"] > 0 and budget["wall_s"] > 5 and not self.ctx.done:
            prompt = ("The verified local policy failed. Use only public evidence and output one legal action.\n"
                      + game.describe() + "\nFailure: " + str(issue)
                      + "\nProgress: " + json.dumps(self.ctx.progress())
                      + "\nObservation: " + json.dumps(obs)
                      + "\nLegal actions: " + " ".join(legal))
            try:
                result = self.ctx.think(
                    prompt, session_id=self.session,
                    timeout_s=min(360.0, max(1.0, budget["wall_s"] - 2)))
                self.session = result.session_id
                for raw in reversed(re.findall(r"\[[^\]\n]+\]", result.text or "")):
                    action = parse_action(raw) or raw.strip()
                    if action in legal:
                        return action
            except Exception as exc:
                self.ctx.log("recovery_failed", error=type(exc).__name__)
        return legal[0]

    def choose(self, obs):
        if obs.get("game_id") == "Mastermind-v0":
            return self.search.choose(obs, self.ctx)
        if obs.get("game_id") != "TicTacToe-v0":
            raise ValueError("Unknown game ID")
        board = obs.get("public_state", {}).get("board")
        me = obs.get("player_id")
        if (not isinstance(board, list) or len(board) != 9 or me not in (0, 1)
                or any(v not in (None, 0, 1) for v in board)
                or obs.get("current_player") != me):
            raise ValueError("Unrecognized active TicTacToe state")
        expected = {f"[{i}]" for i, v in enumerate(board) if v is None}
        legal = set(self.legal(obs))
        if legal != expected:
            raise ValueError("Legal moves disagree with public board")
        ranked = []
        for i in (4, 0, 2, 6, 8, 1, 3, 5, 7):
            if board[i] is not None:
                continue
            child = tuple(board[:i] + [me] + board[i + 1:])
            ranked.append((minimax(child, 1 - me, me), f"[{i}]"))
        return max(ranked, key=lambda pair: pair[0])[1], {"minimax": max(v for v, _ in ranked)}

    def verify(self, before, action, after, prior):
        progress = self.ctx.progress()
        errors = []
        if progress.get("illegal_actions", 0) > prior.get("illegal_actions", 0):
            self.rejected.add(action)
            errors.append("ctx.progress reports a rejected action")
        if progress.get("since_obs_change", 0) > 0:
            errors.append("observation did not change after action")
        if after.get("move_count", 0) <= before.get("move_count", 0):
            errors.append("native move_count did not advance")
        if before.get("game_id") == "Mastermind-v0":
            prior_clues = parse_clues(before)
            if after.get("done") and after.get("score") == 1:
                if after.get("move_count") != before.get("move_count", 0) + 1:
                    errors.append("solving guess did not advance exactly one move")
                elif INDEX.get(action) not in self.search.belief(prior_clues):
                    errors.append("native solve contradicts the prior public belief")
                self.ctx.log("effect_check", action=action, errors=errors, progress=progress)
                return errors
            old, new = prior_clues, parse_clues(after)
            if len(new) != len(old) + 1 or new[:-1] != old:
                errors.append("feedback did not append exactly one clue")
            elif TEXT[new[-1][0]] != action:
                errors.append("feedback action differs from submitted guess")
            if not self.search.belief(new):
                errors.append("feedback contradicts all public secrets")
            if after.get("done") and new:
                _, b, w = new[-1]
                if abs(after.get("score", 0) - (2 * b + w) / 8) > 1e-8:
                    errors.append("terminal score disagrees with feedback")
        elif before.get("game_id") == "TicTacToe-v0":
            old = before.get("public_state", {}).get("board", [])
            new = after.get("public_state", {}).get("board", [])
            i = int(action[1:-1])
            if len(old) != 9 or len(new) != 9 or new[i] != before.get("player_id"):
                errors.append("submitted placement missing from board")
            elif any(old[j] is not None and old[j] != new[j] for j in range(9)):
                errors.append("existing mark changed")
        self.ctx.log("effect_check", action=action, errors=errors, progress=progress)
        return errors

    def run_episode(self, game):
        obs = self.ctx.observe()
        for _ in range(32):
            if self.ctx.done or not isinstance(obs, dict) or obs.get("done"):
                return
            try:
                if self.recovery:
                    raise ValueError("Previous local plan failed verification")
                action, details = self.choose(obs)
                self.ctx.log("deterministic_plan", action=action, **details)
            except Exception as exc:
                self.recovery = True
                action = self.recover(game, obs, exc)
            if action is None or self.ctx.done:
                return
            if action not in self.legal(obs):
                self.recovery = True
                obs = self.ctx.observe()
                continue
            prior = self.ctx.progress()
            before = obs
            try:
                obs = self.ctx.act(action)
                if obs is None:
                    obs = self.ctx.observe()
            except Exception as exc:
                if self.ctx.done:
                    return
                self.rejected.add(action)
                self.recovery = True
                self.ctx.log("action_failed", action=action, error=type(exc).__name__)
                obs = self.ctx.observe()
                continue
            if not isinstance(obs, dict):
                return
            try:
                issues = self.verify(before, action, obs, prior)
            except Exception as exc:
                issues = ["verification raised " + type(exc).__name__ + ": " + str(exc)]
            if issues:
                self.recovery = True
                self.ctx.log("plan_invalidated", evidence=issues)


def build(ctx):
    return Harness(ctx)
