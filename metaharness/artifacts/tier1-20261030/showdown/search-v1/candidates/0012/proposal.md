parent: 0007

## Hypothesis

The leader's search is useful, but Astra can bypass it at exactly the moments where a cheap local invariant is strongest. In the worst 0007 traces the model repeatedly volunteered switches after the previous action was already a switch, creating a momentum leak and exposing replacements before they could act. In seed 282617, Astra chose `switch 6` on turn 1, then `switch 3` on turn 2 after a revealed High Horsepower hit for 72.9% of Reshiram, then `switch 5` on turn 4 and `switch 4` on turn 6; the battle ended at score -1.0668. In seed 257053, Astra similarly chose `switch 6` then `switch 3`, while later fallback choices continued cycling among slots after revealed Gigadrain/Brave Bird damage. These are action-selection failures, not missing protocol data.

## What changed

I retain 0007's public protocol ledger, per-move evidence, approximate damage ranking, bounded policy continuation, timeout recovery, and legal-action fallback. I add a deterministic model-action firewall after parsing Astra's answer and before saving its policy:

- A voluntary switch is vetoed when the ranked active move is an estimated immediate KO, or when the proposed replacement is likely to faint while the active Pokémon can still make meaningful progress.
- Two voluntary switches in succession are vetoed unless the replacement has both tolerable incoming damage and a credible attack.
- Tera is vetoed after it is spent or when the estimated damage/survival change is small.
- Every veto returns the current ranked legal action, and the model's policy is saved only for that post-firewall action.

The firewall is deliberately narrow; it does not override forced replacements, a genuinely lethal matchup, or ordinary model attacks.

## Evidence that would confirm or refute

Confirm if losing traces show fewer consecutive voluntary switches, fewer replacement KOs, and more accepted attacks after a switch while preserving the leader's wins and zero-error contract. Refute if the guard traps necessary double switches, sacrifices a low-value active into a winning counter, or materially lowers scores on seeds where switching is the only safe line. Logs label each veto (`veto_switch_loop`, `veto_switch_ko`, `veto_lethal_switch`, `veto_small_tera`) so later iterations can separate a bad guard from a bad damage estimate.
