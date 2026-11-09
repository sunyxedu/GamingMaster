parent: 0000

## Hypothesis

The apparent failure of the first Astra harness is an adaptor failure, not a battle-policy failure: candidate 0000 made one slow call and then passed arbitrary model text to `parse_action`, which raised `StopIteration` before posting its first move in all ten episodes. Even a strong answer is unusable if it is wrapped in prose, malformed by a stand-in model, or if a visible switch is rejected by a hidden trap. A second risk is spending the whole episode budget on one model call per request when most choices can be ranked from the private request and public typing.

## What changed vs. the parent

This harness is code-first. It has a bounded type-effectiveness/power/status scorer that always chooses a legal command, handles forced replacements, and reserves Tera for a clear matchup improvement. Astra is called only on the first decision, when the active opponent changes, and every twelfth accepted action, with an eight-call episode cap; its answer is accepted only after exact case-insensitive matching against the current legal list. Any timeout, budget/backend failure, arbitrary stand-in response, or rejected action falls back to a freshly observed scripted choice, so no model or parser exception can terminate an episode. The prompt is a compact ledger containing own HP/moves/items/abilities, revealed opponent data, recent events, and current legal actions.

## Evidence that would confirm or refute it

Confirmation would be zero harness errors and nonzero accepted decisions in preflight/dev, with scores above the zero-step baseline even when Astra returns garbage. A useful trace signature is that most actions are posted without a fresh model call and model outputs that contain explanations still resolve to legal commands. Refutation would be clean episodes that nonetheless lose at near-neutral or negative HP margin because the local ranking cannot recognize key defensive switches, setup lines, speed/priority, or Tera timing; that would motivate adding verified mechanics rather than more parser logic.
