"""Scalar compatibility mapping at the TurnGame/evaluator boundary."""

def scalar_reward(*, solved: bool, guesses_used: int, max_guesses: int) -> float:
    # Preserve the explicit result tuple separately; any win outranks any loss.
    return float((max_guesses + 1 - guesses_used) / max_guesses) if solved else 0.0
