parent: 0000

## Hypothesis

The current leader is failing before its first action because its generic parser
assumes the model's first whitespace token is one of `action_spec.names`. The
trace shows Astra returning a valid-looking `move 3`, followed by
`StopIteration` in `parse_action`; all ten episodes therefore have zero steps
and no game score. Even after repairing that crash, a raw full-observation
prompt gives Astra no explicit Pokémon ledger or fallback when a response is
late or malformed.

## What changed

This harness matches action phrases anywhere in arbitrary text and only accepts
actions present in the *current* `legal_actions`, including the current Tera and
team-slot state. It catches model timeout, budget, backend, and stand-in-output
failures, then uses a deterministic fallback that handles forced replacements,
rough type effectiveness, STAB, base power, accuracy, recovery and hazards.
Each strategic call receives a bounded JSON ledger containing the private own
team, active request, public revealed opponents, recent protocol events, move
and species data, and the legal actions. The model is asked for one action per
decision, while the fallback keeps the episode progressing after the 36-call
strategic cap or a backend failure.

## Evidence that would confirm or refute it

Confirmation would be zero harness errors, nonzero accepted decisions, and
scores above the all-zero parent on the same seeds; traces should show the
model's action token being accepted even when surrounded by prose. If episodes
advance but remain weak, the parser diagnosis is confirmed while the strategic
ledger/fallback needs refinement. Refutation would be a preflight or training
episode error unrelated to parsing, or accepted actions consistently losing
because the local ranking chooses illegal/obviously losing switches despite a
valid model answer.
