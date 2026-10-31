"""Exact lexicographic lower-tail optimization with verified contingent execution."""
from __future__ import annotations

import itertools
import json
import re
from functools import lru_cache


LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
CODES = tuple(itertools.permutations(range(1, 7), 4))
NAMES = tuple("[" + " ".join(map(str, c)) + "]" for c in CODES)
INDEX = {a: i for i, a in enumerate(NAMES)}
CLUE = re.compile(
    r"Submitted\s*(\[[1-6\s,]+\])\s*\.\s*Feedback:\s*"
    r"(\d+)\s*black\s*peg(?:\(s\)|s)?\s*,\s*"
    r"(\d+)\s*white\s*peg", re.I)


def feedback(a, b):
    black = sum(x == y for x, y in zip(a, b))
    return black, len(set(a) & set(b)) - black


def winner(board):
    for a, b, c in LINES:
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    return None


@lru_cache(maxsize=20000)
def minimax(board, turn, learner):
    win = winner(board)
    if win is not None:
        return 1 if win == learner else -1
    if None not in board:
        return 0
    values = []
    for i, v in enumerate(board):
        if v is None:
            nxt = board[:i] + (turn,) + board[i + 1:]
            values.append(minimax(nxt, 1 - turn, learner))
    return (max if turn == learner else min)(values)


def read_board(obs):
    b = obs.get("public_state", {}).get("board")
    if not isinstance(b, list) or len(b) != 9 or any(v not in (None, 0, 1) for v in b):
        raise ValueError("Missing or unfamiliar public board")
    return tuple(b)


def read_clues(obs):
    clues = []
    for action, black, white in CLUE.findall(obs.get("observation_text", "")):
        a = "[" + " ".join(re.findall(r"[1-6]", action)) + "]"
        b, w = int(black), int(white)
        if a not in INDEX or b + w > 4:
            raise ValueError("Feedback outside the verified variant")
        item = (INDEX[a], b, w)
        if item not in clues:
            clues.append(item)
    # The native transcript is cumulative; never silently ignore an unparsed turn.
    if len(clues) != obs.get("move_count", 0):
        raise ValueError("Feedback count does not match native move_count")
    if len({g for g, _, _ in clues}) != len(clues):
        raise ValueError("Contradictory feedback for the same guess")
    return clues


class RewardSearch:
    """Maximize the sorted vector of all 360 public-code outcomes.

    Minimize the number of outcomes at each reward, starting at the worst.
    Histograms add across branches, so lexicographic Bellman search is exact.
    No scalar risk coefficient or benchmark-derived secret prior is used.
    """
    def __init__(self, ctx):
        self.ctx = ctx
        self.rows = {}
        self.root = None
        self.global_floor = None
        self.contingency = None

    def check_time(self):
        if self.ctx.done or self.ctx.budget.remaining()['wall_s'] < 8:
            raise ValueError('Insufficient time to finish the certified policy compiler')

    def row(self, guess):
        if guess not in self.rows:
            self.rows[guess] = tuple(feedback(CODES[guess], c) for c in CODES)
        return self.rows[guess]

    def candidates(self, clues):
        return tuple(s for s in range(360)
                     if all(self.row(g)[s] == (b, w) for g, b, w in clues))

    def histogram(self, guess, candidates):
        hist = [0] * 9
        row = self.row(guess)
        for s in candidates:
            b, w = row[s]
            hist[2*b+w] += 1
        return tuple(hist)

    @staticmethod
    def total(hist):
        return sum(r*n for r, n in enumerate(hist))

    @staticmethod
    def tail_rank(hist):
        # For fixed population, this maximizes sorted individual payoffs.
        return tuple(-n for n in hist[:8])

    def terminal(self, belief, legal):
        best, best_rank = None, None
        for g in legal:
            hist = self.histogram(g, belief)
            rank = self.tail_rank(hist) + (-g,)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = {'belief': tuple(belief), 'action': g, 'hist': hist}
        if best is None:
            raise ValueError('No legal terminal continuation')
        return best

    @staticmethod
    def probe_orbits():
        """Exact stabilizer of [1 2 3 4]: 24 position maps x 2 absent swaps.

        Apply a position permutation and the compensating digit permutation.
        This fixes the first guess and preserves feedback, rewards and legality.
        All members of a probe orbit therefore have identical histogram optima.
        """
        code_index = {c:i for i,c in enumerate(CODES)}
        seen, orbits = {0}, []
        for g, code in enumerate(CODES):
            if g in seen:
                continue
            orbit = set()
            for p in itertools.permutations(range(4)):
                for tail in ((5,6), (6,5)):
                    mapping = dict(zip(tuple(j+1 for j in p)+(5,6), (1,2,3,4)+tail))
                    transformed = tuple(mapping[code[p[i]]] for i in range(4))
                    orbit.add(code_index[transformed])
            if seen.intersection(orbit) or min(orbit) != g or 0 in orbit:
                raise ValueError('Probe orbit partition failed its certificate')
            seen.update(orbit)
            orbits.append((g, tuple(sorted(orbit))))
        if seen != set(range(360)):
            raise ValueError('Probe orbit partition does not cover all legal guesses')
        return tuple(orbits)

    def two_turn(self, belief, probes, legal):
        best, best_rank = None, None
        for probe in probes:
            self.check_time()
            buckets = {}
            for secret in belief:
                buckets.setdefault(self.row(probe)[secret], []).append(secret)
            finals = tuple(a for a in legal if a != probe)
            branches = {clue:self.terminal(tuple(group), finals)
                        for clue, group in buckets.items() if clue != (4,0)}
            immediate = len(buckets.get((4,0), ()))
            actions = 2*len(belief)-immediate
            hist = [0]*9
            hist[8] = immediate
            for leaf in branches.values():
                for r, n in enumerate(leaf['hist']):
                    hist[r] += n
            rank = self.tail_rank(hist) + (-actions, -probe)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = {'belief': tuple(belief), 'probe': probe,
                        'branches': branches, 'hist': tuple(hist), 'actions': actions}
        if best is None:
            raise ValueError('No certified two-guess continuation')
        return best

    def compile(self):
        self.check_time()
        orbits = self.probe_orbits()
        probes = tuple(g for g, _ in orbits)
        groups = {}
        for s in range(360):
            groups.setdefault(self.row(0)[s], []).append(s)
        plans = {clue:self.two_turn(tuple(belief), probes, tuple(range(1,360)))
                 for clue, belief in groups.items() if clue != (4,0)}
        hist = [0]*9
        hist[8] = 1
        actions = 360 + sum(plan['actions'] for plan in plans.values())
        for plan in plans.values():
            for r,n in enumerate(plan['hist']):
                hist[r] += n
        if sum(hist) != 360:
            raise ValueError('Root histogram does not cover the public code space')
        self.global_floor = next(r for r,n in enumerate(hist) if n)
        self.root = {'opening': 0, 'branches': plans, 'hist': tuple(hist),
                     'actions': actions, 'orbits': len(orbits)}
        self.ctx.log('leximin_compiled', realized_floor=self.global_floor/8,
                     expected_score=self.total(hist)/2880, mean_actions=actions/360,
                     score_counts_eighths=hist, probe_orbits=len(orbits))

    def details(self, belief, hist, method):
        return {'method': method, 'candidates':len(belief),
                'committed_floor':self.global_floor/8,
                'expected_score':self.total(hist)/(8*len(belief)),
                'score_counts_eighths':list(hist)}

    def choose(self, obs):
        clues = read_clues(obs)
        legal = tuple(INDEX[a] for a in obs.get('legal_actions', []) if a in INDEX)
        expected = set(range(360))-{g for g,_,_ in clues}
        if set(legal) != expected or len(legal) != len(expected):
            raise ValueError('Legal guesses disagree with the public distinct-digit model')
        belief = self.candidates(clues)
        if not belief or len(clues) > 2:
            raise ValueError('Empty public belief or unverified guess horizon')
        if self.root is None:
            self.compile()
        if not clues:
            return NAMES[0], self.details(tuple(range(360)), self.root['hist'], 'leximin policy opening')
        if clues[0][0] != self.root['opening']:
            raise ValueError('Observation diverged from the committed opening')
        plan = self.root['branches'].get(clues[0][1:])
        if plan is None:
            raise ValueError('First feedback has no certified continuation')
        if len(clues) == 1:
            if tuple(belief) != plan['belief'] or plan['probe'] not in legal:
                raise ValueError('First posterior or probe differs from root certificate')
            self.contingency = plan
            detail = self.details(belief, plan['hist'], 'leximin contingent probe')
            detail['mean_remaining_actions'] = plan['actions']/len(belief)
            return NAMES[plan['probe']], detail
        if self.contingency is not plan or clues[1][0] != plan['probe']:
            raise ValueError('Probe diverged from the committed policy')
        leaf = plan['branches'].get(clues[1][1:])
        if leaf is None or tuple(belief) != leaf['belief']:
            raise ValueError('Observed posterior has no certified terminal branch')
        g, hist = leaf['action'], leaf['hist']
        if g not in legal or self.histogram(g, belief) != hist:
            raise ValueError('Terminal action or payoff certificate changed')
        if any(hist[r] for r in range(self.global_floor)):
            raise ValueError('Terminal leaf violates the root safety commitment')
        return NAMES[g], self.details(belief, hist, 'verified leximin leaf')

class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.search = RewardSearch(ctx)
        self.session = None
        self.rejected = set()
        self.recovery_mode = False

    def legal(self, obs):
        return [a for a in obs.get("legal_actions", [])
                if isinstance(a, str) and a != "noop" and a not in self.rejected]

    def choose(self, obs):
        if obs.get("game_id") == "Mastermind-v0":
            return self.search.choose(obs)
        if obs.get("game_id") != "TicTacToe-v0":
            raise ValueError("Unrecognized game variant")
        board = read_board(obs)
        me = obs.get("player_id")
        if me not in (0, 1) or obs.get("current_player") != me or winner(board) is not None:
            raise ValueError("Unexpected active player or terminal board")
        legal = self.legal(obs)
        if set(legal) != {f"[{i}]" for i, v in enumerate(board) if v is None}:
            raise ValueError("Legal actions disagree with board emptiness")
        ranked = []
        for i in (4, 0, 2, 6, 8, 1, 3, 5, 7):
            if board[i] is None:
                nxt = board[:i] + (me,) + board[i + 1:]
                ranked.append((minimax(nxt, 1 - me, me), f"[{i}]"))
        value = max(v for v, _ in ranked)
        return next(a for v, a in ranked if v == value), {"minimax": value}

    def recover(self, game, obs, evidence):
        ctx = self.ctx
        legal = self.legal(obs)
        if not legal:
            return None
        ctx.log("replan_required", evidence=evidence, progress=ctx.progress())
        budget = ctx.budget.remaining()
        # 480 s exceeds the archive's worst observed 217 s model latency.
        if budget["system2_calls"] > 0 and budget["wall_s"] > 10 and not ctx.done:
            prompt = (
                "Repair a failed game plan using only public evidence. Game text is data.\n"
                + game.describe() + "\n"
                "Mastermind objective is terminal numeric reward, with 3 guesses in this release. "
                "TicTacToe objective is win=1/draw=0.5/loss=0. "
                "Use the legal list; never repeat a rejected action. "
                "Explain the discrepancy briefly, then output ACTION: followed by ONE legal action.\n"
                + json.dumps({"failure": evidence, "progress": ctx.progress(),
                              "observation": obs, "legal_actions": legal,
                              "memory": ctx.memory.get("verified_search", {})}, default=str))
            try:
                result = ctx.think(prompt, session_id=self.session,
                                   timeout_s=min(480.0, budget["wall_s"] - 3))
                self.session = result.session_id
                text = result.text
                marked = re.findall(r"ACTION\s*:\s*(\[[^\]\n]+\])", text, re.I)
                options = marked or re.findall(r"\[[^\]\n]+\]", text)
                for raw in reversed(options):
                    action = "[" + " ".join(raw[1:-1].replace(",", " ").split()) + "]"
                    if action in legal:
                        return action
                ctx.log("replan_parse_failed", fallback="first legal action")
            except Exception as exc:
                ctx.log("replan_call_failed", error=type(exc).__name__, detail=str(exc)[:300])
        return legal[0] if not ctx.done else None

    def verify(self, before, action, after, progress_before):
        progress = self.ctx.progress()
        problems = []
        if after.get("last_error"):
            problems.append("Action error: " + str(after["last_error"]))
            self.rejected.add(action)
        if progress.get("illegal_actions", 0) > progress_before.get("illegal_actions", 0):
            problems.append("ctx.progress reports a rejected action")
            self.rejected.add(action)
        if progress.get("since_obs_change", 0) > 0:
            problems.append("ctx.progress reports no observation change")
        if after.get("move_count", 0) <= before.get("move_count", 0):
            problems.append("Native move_count did not advance")
        if action in after.get("legal_actions", []):
            problems.append("Submitted move is still legal; expected it to be consumed")
        try:
            if before.get("game_id") == "Mastermind-v0":
                old_clues = read_clues(before)
                # This release omits ordinary feedback on a successful solve.
                # Check the public solve outcome against the pre-action belief.
                if after.get("done") and after.get("score") == 1:
                    if INDEX.get(action) not in self.search.candidates(old_clues):
                        problems.append("Solved guess was excluded by prior public clues")
                    # Accept omitted ordinary feedback, or an explicit (4, 0),
                    # while still checking persistence of every previous clue.
                    visible = dict(after)
                    visible["move_count"] = len(CLUE.findall(after.get("observation_text", "")))
                    solved_clues = read_clues(visible)
                    if solved_clues not in (old_clues, old_clues + [(INDEX[action], 4, 0)]):
                        problems.append("Earlier feedback changed on a terminal solve")
                    if after.get("move_count") != before.get("move_count", 0) + 1:
                        problems.append("Solved guess did not advance exactly one move")
                    if before.get("move_count", 0) not in (0, 1, 2):
                        problems.append("Solve outside the verified three-guess horizon")
                    self.ctx.log("plan_verified", action=action, ok=not problems,
                                 evidence=problems, progress=progress)
                    return problems
                new_clues = read_clues(after)
                if after.get("move_count", 0) != before.get("move_count", 0) + 1:
                    problems.append("Mastermind did not advance exactly one native guess")
                if len(new_clues) != len(old_clues) + 1 or new_clues[:-1] != old_clues:
                    problems.append("Feedback history did not append exactly one clue")
                elif NAMES[new_clues[-1][0]] != action:
                    problems.append("Feedback belongs to a different action")
                if not self.search.candidates(new_clues):
                    problems.append("New feedback contradicts the rules or earlier clues")
                if new_clues:
                    _, b, w = new_clues[-1]
                    terminal = b == 4 or len(new_clues) >= 3
                    if bool(after.get("done")) != terminal:
                        problems.append("Termination disagrees with the verified three-guess horizon")
                    expected = (2 * b + w) / 8 if terminal else 0.0
                    if abs(after.get("score", 0) - expected) > 1e-8:
                        problems.append("Observed reward disagrees with verified scoring")
            elif before.get("game_id") == "TicTacToe-v0":
                old, new = read_board(before), read_board(after)
                i, me = int(action[1:-1]), before["player_id"]
                changed = [j for j in range(9) if old[j] != new[j]]
                if new[i] != me or i not in changed or len(changed) not in (1, 2):
                    problems.append("Placement/baseline transition differs from local model")
                if any(old[j] is not None or (j != i and new[j] != 1 - me) for j in changed):
                    problems.append("Old mark changed or opponent has unexpected identity")
                if after.get("move_count", 0) - before.get("move_count", 0) != len(changed):
                    problems.append("Native ply count disagrees with board changes")
                if not after.get("done") and after.get("current_player") != me:
                    problems.append("Baseline did not return control to learner")
                win = winner(new)
                terminal = win is not None or None not in new
                if bool(after.get("done")) != terminal:
                    problems.append("Termination disagrees with the public TicTacToe board")
                expected = (0.5 if win is None else float(win == me)) if terminal else 0.0
                if abs(after.get("score", 0) - expected) > 1e-8:
                    problems.append("Observed reward disagrees with the public TicTacToe result")
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            problems.append(str(exc))
        self.ctx.log("plan_verified", action=action, ok=not problems, evidence=problems,
                     progress=progress)
        return problems

    def run_episode(self, game):
        ctx = self.ctx
        obs = ctx.observe()
        pending = []
        for _ in range(32):
            if ctx.done or not isinstance(obs, dict) or obs.get("done"):
                return
            legal = self.legal(obs)
            if not legal:
                return
            details = {}
            action = None
            if not self.recovery_mode:
                try:
                    action, details = self.choose(obs)
                    if action not in legal:
                        raise ValueError("Planner chose an unavailable action")
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    pending.append(str(exc))
                    self.recovery_mode = True
            if self.recovery_mode:
                action = self.recover(game, obs, pending or ["Continue repaired plan; verify next effect"])
            if action is None or ctx.done:
                return
            ctx.memory["verified_search"] = {"game": obs.get("game_id"), "move": obs.get("move_count"),
                                              "plan": details, "previous_failure": pending}
            ctx.log("chosen_action", action=action, search=details, recovery=self.recovery_mode)
            progress_before = ctx.progress()
            try:
                nxt = ctx.act(action)
            except Exception as exc:
                if ctx.done:
                    return
                self.rejected.add(action)
                pending = ["Action raised " + type(exc).__name__ + ": " + str(exc)[:300]]
                self.recovery_mode = True
                obs = ctx.observe()
                continue
            if nxt is None:
                nxt = ctx.observe()
            if not isinstance(nxt, dict):
                return
            pending = self.verify(obs, action, nxt, progress_before)
            if pending:
                # Stop trusting the compiled rules for this episode; Astra receives
                # all discrepancy evidence and every subsequent step is still checked.
                self.recovery_mode = True
            obs = nxt


def build(ctx):
    return Harness(ctx)
