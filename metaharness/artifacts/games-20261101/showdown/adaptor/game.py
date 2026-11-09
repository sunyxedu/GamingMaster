"""Offline, player-visible Gen 9 random battles using the real Showdown engine."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import struct
import subprocess
import threading
import sys
import signal

from metaharness.games.base import ActionSpec

TIER = 1
DETERMINISTIC = False  # Foul Play MCTS uses native randomness and a wall-clock search budget.
# All groups use exactly the same generation and opponent settings.
SEEDS = {
    "train": [104729, 130363, 155921, 181081, 206369, 232003, 257053, 282617, 308081, 333667],
    "heldout": [359357, 384673, 410117, 435907, 461273, 486719, 512009, 537593, 563051, 588599],
    "validation": [614041, 639599, 665069, 690719, 716161, 741809, 767249, 792881, 818327, 843899],
    "dev": [869431, 895003],
}
EVAL_KWARGS = {"opponent": "foulplay", "max_turns": 120, "max_decisions": 180}
HERE = Path(__file__).resolve().parent


def engine_seed(seed: int, domain: str) -> list[int]:
    """Domain-separated stable uint16 seed words, independent of Python hash()."""
    data = hashlib.sha256(f"showdown-adaptor-v1:{domain}:{seed}".encode()).digest()
    return list(struct.unpack(">4H", data[:8]))


class ShowdownGame:
    name = "Pokémon Showdown — Gen 9 Random Battle"

    def __init__(self, opponent="foulplay", max_turns=120, max_decisions=180, timeout_s=6):
        if opponent != "foulplay":
            raise ValueError("opponent must be foulplay; evaluation always uses the strongest installed opponent")
        if not 1 <= int(max_turns) <= 1000 or not 1 <= int(max_decisions) <= 10000:
            raise ValueError("max_turns must be 1..1000; max_decisions 1..10000")
        if not 0 < float(timeout_s) <= 6:
            raise ValueError("timeout_s must be >0 and <=6 (hard transport bound)")
        self.opponent, self.max_turns = opponent, int(max_turns)
        self.max_decisions, self.timeout_s = int(max_decisions), float(timeout_s)
        names = ["noop"] + [f"move {i}" for i in range(1, 5)] + [
            f"move {i} terastallize" for i in range(1, 5)
        ] + [f"switch {i}" for i in range(1, 7)]
        self.action_spec = ActionSpec(
            names=names,
            descriptions={a: ("Wait without advancing the battle." if a == "noop" else
                "Choose this move slot and spend the team's one Terastallization." if "terastallize" in a else
                "Choose this active move slot." if a.startswith("move") else
                "Choose this current request.side.pokemon slot (switch or revive).") for a in names},
            noop="noop",
            notes="Only observation.legal_actions are currently available; noop always waits. "
                  "Slots are 1-based and team order changes on switches. Illegal actions report an error without advancing.",
        )
        self._process = None
        self._reader = None
        self._stderr = None
        self._seed = None
        self._valid = True
        self._invalid_reason = None
        self._state = {"observation": {"done": False, "last_error": "Call reset(seed) first."},
                       "done": False, "referee": {}, "termination_reason": "not_started"}

    def _start(self):
        runtime = HERE / "runtime"
        runtime.mkdir(exist_ok=True)
        # No credentials, sockets, server, network, or runtime source generation.
        self._stderr = (runtime / f"bridge-{os.getpid()}-{id(self):x}.log").open("w")
        self._process = subprocess.Popen(
            [os.environ.get("SHOWDOWN_NODE", "node"), str(HERE / "bridge.cjs")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, encoding="utf-8", cwd=runtime, start_new_session=True,
            env={**os.environ, "SHOWDOWN_PYTHON": sys.executable},
        )
        self._responses = queue.Queue()
        proc, responses = self._process, self._responses

        def read():
            try:
                for line in proc.stdout:
                    responses.put(line)
            finally:
                responses.put(None)

        self._reader = threading.Thread(target=read, name="showdown-reader", daemon=True)
        self._reader.start()

    def _rpc(self, payload):
        try:
            self._process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            self._process.stdin.flush()
            line = self._responses.get(timeout=self.timeout_s)
            if line is None:
                raise RuntimeError("Showdown bridge exited; see runtime/bridge-*.log")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(response["error"])
            self._state = response
        except Exception as exc:
            self._invalidate(f"{type(exc).__name__}: {exc}")

    def _invalidate(self, message):
        self._valid = False
        self._invalid_reason = message[:2000]
        self._state["done"] = True
        self._state["termination_reason"] = "infrastructure_error"
        self._state["observation"].update(done=True, legal_actions=[],
            last_error="Simulator failure; evaluator should retry this trial.",
            termination_reason="infrastructure_error")
        self.close()

    def reset(self, seed: int):
        self.close()
        self._seed = int(seed)
        self._valid, self._invalid_reason = True, None
        self._state = {"observation": {"done": False}, "done": False,
                       "referee": {}, "termination_reason": "starting"}
        try:
            self._start()
        except Exception as exc:
            self._invalidate(f"Could not start Node: {exc}")
            return self.observe()
        self._rpc({"op": "reset", "battle_seed": engine_seed(self._seed, "battle"),
                   "team_seeds": [engine_seed(self._seed, "team-p1"), engine_seed(self._seed, "team-p2")],
                   "bot_seed": engine_seed(self._seed, "opponent"), "opponent": self.opponent,
                   "max_turns": self.max_turns, "max_decisions": self.max_decisions})
        return self.observe()

    def observe(self):
        return copy.deepcopy(self._state["observation"])

    def step(self, action):
        if self.done:
            return self.observe()
        obs = self._state["observation"]
        if isinstance(action, str) and action == "noop":
            obs["last_error"] = None
        elif not isinstance(action, str) or action not in obs.get("legal_actions", []):
            obs["last_error"] = "Illegal action. Choose a string from legal_actions, or noop to wait."
        elif self._process is None:
            self._invalidate("Simulator is closed; reset before playing.")
        else:
            self._rpc({"op": "step", "action": action})
        return self.observe()

    @property
    def done(self):
        return bool(self._state["done"])

    @property
    def score(self):
        return float(self._state["referee"].get("score", 0.0))

    def score_breakdown(self):
        return {k: float(v) for k, v in self._state["referee"].items()}

    def info(self):
        return {"valid_trial": self._valid, "invalid_reason": self._invalid_reason,
                "seed": self._seed, "battle_seed": engine_seed(self._seed, "battle"),
                "opponent_seed": engine_seed(self._seed, "opponent"),
                "team_seeds": [engine_seed(self._seed, "team-p1"), engine_seed(self._seed, "team-p2")],
                "opponent": self.opponent, "showdown_revision": self._state.get("revision"),
                "termination_reason": self._state["termination_reason"],
                "truncated": self._state["termination_reason"] in ("turn_limit", "decision_limit", "engine_time_limit"),
                **self.score_breakdown()}

    def describe(self):
        return """Play Pokémon Showdown Gen 9 Random Battle singles as p1 (Agent) against a
Foul Play MCTS opponent. It uses damage simulation, switch search and public-set inference. Both teams have six randomly generated Pokémon,
with format-assigned levels, moves, items, abilities and Tera types. There is no team preview.
Win by fainting all six opponents. Pokémon types, stats, speed, priority, move accuracy/PP,
abilities, items, status, boosts, hazards, weather and terrain follow the real Gen 9 engine.
Both sides choose before a turn resolves; forced replacements and pivots create extra decisions.
The world waits while you think. You may Terastallize once per battle, when the request allows.

Send exactly 'move 1' .. 'move 4', 'move 1 terastallize' .. 'move 4 terastallize', or
'switch 1' .. 'switch 6'. Only strings in legal_actions are currently available.
Move indices refer to request.active[0].moves; switch indices refer to the CURRENT
request.side.pokemon array, whose order changes on switches. A forced switch requires a switch;
Revival Blessing instead selects a fainted teammate. 'noop' waits without advancing.
Malformed/unavailable choices report last_error without consuming a decision. A hidden trapping
effect can reject an apparently available switch; the engine then updates the request.

request is your private Showdown request: exact own HP/maxHP, status, stats, items, abilities,
moves/PP, Tera options, disabled moves and known trapping. battle_log contains the most recent
700 p1-visible protocol lines (also capped at 65k characters); recent_events contains events
since the last submitted choice (up to 300 lines/30k characters). Protocol is authoritative.
Public switches reveal species/level; opponent HP is rounded by Showdown. opponent_revealed,
active and public_summary cache common public events; summaries are conveniences, not a complete
mechanics interpreter (Illusion, Transform and unusual boost effects require reading the log).
move_dex supplies public move descriptions and base data for your moves; species_dex contains
public types/base stats for your team and revealed species. Hidden opponent sets and exact HP
are never exposed.
No scenario seed or referee score is included in observations. result and termination_reason
are populated at termination. The configured turn/decision limits and 480 seconds of cumulative
engine work adjudicate a tie. Your thinking time is excluded from that engine-work budget.

Evaluator score is outcome (+1 win, 0 tie/unfinished, -1 loss) + 0.25 * HP margin.
Each team's HP fraction is the sum of its six individual remaining-HP/max-HP fractions divided
by six; HP margin is yours minus theirs. Winning is primary; preserving HP breaks ties in quality.
Exact score components, survivors, turns and termination reason are evaluator-only.
"""

    def close(self):
        proc, self._process = self._process, None
        if proc is not None:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except (OSError, ValueError):
                    pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=2)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if self._reader:
                self._reader.join(timeout=2)
            if proc.stdout:
                proc.stdout.close()
        self._reader = None
        if self._stderr:
            self._stderr.close()
            self._stderr = None


def make_game(**kwargs):
    return ShowdownGame(**kwargs)
