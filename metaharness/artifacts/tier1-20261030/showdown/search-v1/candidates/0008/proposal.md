parent: 0006

## Hypothesis

The main failure is over-delegation: Astra is called after almost every new opponent identity, even when the legal move ranking has a clear high-value attack. Its slow, high-context answers then make premature switches and Tera choices, and consume the episode wall budget before a stable endgame. Candidate 0004's losing seed 232003 shows model choices on turns 1, 3, 5, 9, 10, 12, 15, 16 and 17, including an early Tera and repeated switches; seed 333667 similarly switched repeatedly before the deterministic fallback took over. Candidate 0006's line planner reduces calls but still lets Astra decide routine close-looking requests without a calibrated damage model.

## What changed vs. the parent

0008 keeps the parent's legal parser, type chart, forced-switch handling and stale-line invalidation, but makes the controller code-first. It computes the best damaging move and the best safe replacement from the current request. Astra is queried only when the top move is ambiguous, no move has meaningful power, a switch materially outranks the attack, or a Tera variant is plausibly decisive. A model line is accepted only for its current first token and only if it belongs to that ambiguity set; subsequent tokens are discarded after each action so reordered switch slots cannot leak through. Routine turns use the deterministic scorer, preserving calls and wall time for strategic forks. A short anti-oscillation window remains, but is overridden for forced replacement and critical HP.

## Evidence that would confirm or refute it

Confirmation would be fewer Astra calls, fewer adjacent voluntary switches and improved worst-seed scores while deterministic attacks still finish battles before the deadline. Refutation would be losses where the scorer repeatedly chooses a neutral attack despite a clearly superior defensive switch, or where a needed setup/Tera line is excluded by the ambiguity gate; traces would show large damage/typing gaps or forced switches missed by the gate. That would support adding a narrow threat detector rather than returning to per-turn model control.
