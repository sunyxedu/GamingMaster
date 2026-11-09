"""Offline, seeded Wordle and Mastermind; no solver or network dependency."""
from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
from typing import Any

from metaharness.games.base import ActionSpec
from scoring import scalar_reward

TIER = 1
DETERMINISTIC = True
SEEDS = {
    "train": [0, 1, 2, 3, 4, 5, 7, 8, 11, 13, 17, 43],
    "heldout": [6, 9, 10, 12, 16, 19, 20, 26, 31, 36, 47, 52],
    "dev": [1001, 1002],
    "validation": [14, 15, 18, 21, 22, 23, 38, 42, 54, 68, 70, 116],
}
EVAL_KWARGS = {"mode": "wordle"}

_ROOT = Path(__file__).resolve().parent
WORD_LIST_VERSION = "wordle-original-2022-01-2315-12972-v1"
WORDLIST_HASHES = {
    "answers.txt": "f40e47975cf360ff9febe7d49c8e285054d8fa3fb9644cea575922060951d6ee",
    "allowed_guesses.txt": "feacc549249b5dcb43e2799fbbcda14ebc33d55904fb15d043779104e60c1579",
}


def _load_words(filename: str, count: int) -> tuple[str, ...]:
    data = (_ROOT / filename).read_bytes()
    if hashlib.sha256(data).hexdigest() != WORDLIST_HASHES[filename]:
        raise ValueError(f"Bundled word data checksum mismatch: {filename}")
    words = tuple(data.decode("ascii").splitlines())
    if len(words) != count or len(set(words)) != count or any(
        len(w) != 5 or not all("a" <= c <= "z" for c in w) for w in words
    ):
        raise ValueError(f"Invalid bundled word list: {filename}")
    return words


ANSWERS = _load_words("answers.txt", 2315)
ALLOWED_GUESSES = frozenset(_load_words("allowed_guesses.txt", 12972))
if not set(ANSWERS) <= ALLOWED_GUESSES:
    raise ValueError("Every answer must be an allowed guess")


def wordle_feedback(answer: str, guess: str) -> list[str]:
    result = ["gray"] * 5
    remaining: dict[str, int] = {}
    for i, (a, g) in enumerate(zip(answer, guess)):
        if a == g:
            result[i] = "green"
        else:
            remaining[a] = remaining.get(a, 0) + 1
    for i, g in enumerate(guess):
        if result[i] != "green" and remaining.get(g, 0):
            result[i] = "yellow"
            remaining[g] -= 1
    return result


def mastermind_feedback(code: tuple[int, ...], guess: tuple[int, ...]) -> dict[str, int]:
    exact = sum(a == b for a, b in zip(code, guess))
    left: dict[int, int] = {}
    right: dict[int, int] = {}
    for a, b in zip(code, guess):
        if a != b:
            left[a] = left.get(a, 0) + 1
            right[b] = right.get(b, 0) + 1
    return {"exact": exact, "misplaced": sum(min(n, right.get(c, 0)) for c, n in left.items())}


def _integer(name: str, value: Any, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return value


class PuzzleGame:
    def __init__(self, *, mode: str = "wordle", max_guesses: int | None = None,
                 positions: int = 4, colors: int = 6, repeats: bool = True):
        if mode not in ("wordle", "mastermind"):
            raise ValueError("mode must be 'wordle' or 'mastermind'")
        self.mode = mode
        self.name = "Wordle" if mode == "wordle" else "Mastermind"
        self.max_guesses = _integer("max_guesses", max_guesses if max_guesses is not None else
                                    (6 if mode == "wordle" else 10), 1, 100)
        self.positions = _integer("positions", positions, 1, 8)
        self.colors = _integer("colors", colors, 2, 10)
        if type(repeats) is not bool:
            raise ValueError("repeats must be a boolean")
        self.repeats = repeats
        if mode == "wordle" and (self.max_guesses != 6 or positions != 4 or colors != 6 or not repeats):
            raise ValueError("Wordle has exactly six guesses; Mastermind variant kwargs do not apply")
        if mode == "mastermind" and not repeats and positions > colors:
            raise ValueError("Without repeats, positions cannot exceed colors")
        self.ruleset_version = ("wordle-normal-5x6-v1" if mode == "wordle" else
                                f"mastermind-{positions}x{colors}-repeats{int(repeats)}-limit{self.max_guesses}-v1")
        grammar = "guess crane" if mode == "wordle" else "guess " + "0" * positions
        self.action_spec = ActionSpec(
            names=["guess", "noop"],
            descriptions={"guess": f"Submit a complete guess, e.g. '{grammar}'; see describe() for grammar.",
                          "noop": "Leave the board and guess count unchanged; clear any action error."},
            noop="noop",
            notes="Parameterized actions use 'guess <value>'. A bare value or {'guess': value} is also accepted. Invalid actions do not cost guesses.",
        )
        self.reset(0)

    def reset(self, seed: int) -> dict[str, Any]:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._seed = seed
        rng = random.Random(seed)
        if self.mode == "wordle":
            self._target: str | tuple[int, ...] = ANSWERS[rng.randrange(len(ANSWERS))]
        elif self.repeats:
            self._target = tuple(rng.randrange(self.colors) for _ in range(self.positions))
        else:
            self._target = tuple(rng.sample(range(self.colors), self.positions))
        self._history: list[dict[str, Any]] = []
        self._done = False
        self._solved = False
        self._last_error: str | None = None
        return self.observe()

    def _rules(self) -> dict[str, Any]:
        rules: dict[str, Any] = {"version": self.ruleset_version, "max_guesses": self.max_guesses}
        if self.mode == "wordle":
            rules.update(word_length=5, hard_mode=False, word_list_version=WORD_LIST_VERSION,
                         answer_count=len(ANSWERS), allowed_guess_count=len(ALLOWED_GUESSES),
                         answers_path=str(_ROOT / "answers.txt"),
                         allowed_guesses_path=str(_ROOT / "allowed_guesses.txt"),
                         wordlist_sha256=WORDLIST_HASHES.copy())
        else:
            rules.update(positions=self.positions, colors=self.colors, repeats=self.repeats,
                         color_ids=list(range(self.colors)))
        return rules

    def _result(self) -> dict[str, Any]:
        return {"solved": self._solved, "guesses_used": len(self._history), "max_guesses": self.max_guesses}

    def observe(self) -> dict[str, Any]:
        return {
            "game": self.name, "mode": self.mode, "turn": len(self._history),
            "max_turns": self.max_guesses, "rules": self._rules(),
            "history": copy.deepcopy(self._history), "done": self._done,
            "solved": self._solved, "result": self._result(), "score": self.score,
            "error": self._last_error,
        }

    def _invalid(self, message: str) -> dict[str, Any]:
        self._last_error = message
        return self.observe()

    def step(self, action: Any) -> dict[str, Any]:
        # Terminal actions cannot mutate completed episodes.
        if self._done:
            return self.observe()
        self._last_error = None
        if isinstance(action, str) and action.strip() == "noop":
            return self.observe()
        if isinstance(action, str) and action.strip() == "guess":
            return self._invalid("The guess command needs a value, e.g. 'guess crane' or 'guess 0123'.")
        if isinstance(action, dict):
            if set(action) != {"guess"}:
                return self._invalid("Expected exactly {'guess': value}.")
            raw = action["guess"]
        else:
            raw = action
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("guess "):
                raw = raw[6:].strip()
        if self.mode == "wordle":
            if not isinstance(raw, str) or not raw.isascii():
                return self._invalid("Guess must be five ASCII letters.")
            guess = raw.lower()
            if len(guess) != 5 or not guess.isalpha():
                return self._invalid("Guess must be five ASCII letters.")
            if guess not in ALLOWED_GUESSES:
                return self._invalid("Word is absent from the pinned allowed-guess set.")
            feedback = wordle_feedback(self._target, guess)
            self._history.append({"guess": guess, "feedback": feedback})
            self._solved = guess == self._target
        else:
            if isinstance(raw, str):
                if len(raw) != self.positions or any(c not in "0123456789" for c in raw):
                    return self._invalid(f"Use {self.positions} ASCII digits or an integer list.")
                vals = [int(c) for c in raw]
            elif isinstance(raw, (list, tuple)):
                vals = list(raw)
            else:
                return self._invalid(f"Use {self.positions} ASCII digits or an integer list.")
            if len(vals) != self.positions or any(type(v) is not int or not 0 <= v < self.colors for v in vals):
                return self._invalid(f"Expected {self.positions} color IDs in 0..{self.colors - 1}.")
            if not self.repeats and len(set(vals)) != len(vals):
                return self._invalid("Repeated colors are disabled in this variant.")
            code = tuple(vals)
            feedback = mastermind_feedback(self._target, code)
            self._history.append({"guess": vals, "feedback": feedback})
            self._solved = feedback["exact"] == self.positions
        self._done = self._solved or len(self._history) >= self.max_guesses
        return self.observe()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def score(self) -> float:
        # Only this adapter/evaluator boundary converts the primary result to a scalar.
        return scalar_reward(**self._result())

    def score_breakdown(self) -> dict[str, float]:
        return {**{k: float(v) for k, v in self._result().items()}, "score": self.score}

    def info(self) -> dict[str, Any]:
        info = {"valid_trial": True, "seed": self._seed, "mode": self.mode,
                "ruleset_version": self.ruleset_version, **self._result(), "score": self.score}
        if self.mode == "wordle":
            info.update(word_list_version=WORD_LIST_VERSION, wordlist_sha256=WORDLIST_HASHES.copy())
        if self._done:
            info["answer"] = self._target if isinstance(self._target, str) else list(self._target)
        return info

    def describe(self) -> str:
        common = (
            " The secret is fixed for the round. 'noop' changes no guesses. Invalid actions return an error "
            "without using a guess; repeats of legal guesses consume a turn. Terminal actions do nothing. "
            "Observations include game/mode, rules, turn (accepted guesses so far), max_turns, history, "
            "done, solved, result={solved, guesses_used, max_guesses}, score, and error (null or text). "
            "The secret and seed are never in observations. Score is 0 for unfinished/lost games; "
            "a solve scores (max_guesses + 1 - guesses_used) / max_guesses. Fewer guesses is better."
        )
        if self.mode == "wordle":
            return (
                "Wordle normal mode: guess the five-letter hidden word in six guesses. "
                "Use 'guess crane', bare 'crane', or {'guess':'crane'}; ASCII case and outer whitespace "
                "are normalized. Every answer is legal, as are other words in the pinned guess file. "
                "Rules provide absolute paths to both public word files, their version and hashes. "
                "Feedback is five green/yellow/gray tiles: greens consume exact letter matches first; "
                "remaining copies can yield yellows left to right, gray means no unused copy remains. "
                "There is no hard-mode constraint. " + common
            )
        return (
            f"Mastermind: guess a code of {self.positions} color IDs in 0..{self.colors - 1} "
            f"within {self.max_guesses} guesses. Repeated colors are {'allowed' if self.repeats else 'forbidden'}. "
            "Use 'guess <digits>', a bare compact ASCII digit string, or {'guess':[integer IDs]}. "
            "The list/string must have one entry per position. Feedback is {exact, misplaced}: "
            "exact counts correct positions; misplaced counts remaining matching color multiplicities, "
            "with no location information. Rules are fixed across seeds for this configuration." + common
        )

    def close(self) -> None:
        """No external resources are owned; safe to call repeatedly."""


def make_game(**kwargs: Any) -> PuzzleGame:
    return PuzzleGame(**kwargs)
