parent: 0012

## Hypothesis

0012 is strong overall, but its worst loss (seed 333667, score -1.128) shows a specific failure after information arrives. Noivern revealed Boomburst and removed Volcanion, then removed 54% of Raichu and 65% of Chimecho. The fallback policy still spent Tera on Chimecho even though Poison Tera did not reduce Normal Boomburst damage, and then continued entering exposed slots. The existing Tera veto only guarded Astra answers; code fallbacks could still make the same mistake when the next model call was unavailable near the wall-clock reserve.

## What changed vs. 0012

I retained 0012's public ledger, damage evidence, policy continuation, parser, and model guard. I added a deterministic `emergency_action` filter applied whenever code chooses a fallback (and as a final legality fallback):

- A Terastallize action is rejected unless the estimated defensive threat drops materially or it changes a non-KO into a KO.
- A fast estimated knockout takes precedence over the emergency switch, so the filter cannot throw away a winning finish.
- When a revealed attack threatens the active slot, a voluntary switch is selected by lowest estimated incoming damage, then by ability to retaliate; near-certain replacement KOs are excluded.
- The filter logs `veto_fallback_tera` and `emergency_switch` for trace inspection.

This is intentionally narrow: ordinary ranking and Astra choices remain intact, and forced replacements are untouched.

## Evidence that would confirm or refute it

Confirmation would show fewer fallback Tera uses with no defensive benefit, fewer switches into the same revealed high-damage move, and improved worst-seed scores without reducing 0012's wins. Refutation would be losses caused by refusing a necessary offensive Tera or by choosing a nominally safer but strategically passive replacement; traces can identify these via the safety log events.
