parent: 0014

## Hypothesis

The staged expansion executor can preserve a useful route yet still lose when an opponent reaches the finish race first: its normal project ranking may continue financing a settlement milestone while a public opponent is already at 8–9 VP. The archived leader traces show this concrete endgame failure pattern: seed 103 remains at 6 true VP while repeated legal rolls/end-turns continue after the rival race has become decisive, and seed 102 only converts after a late Year of Plenty settlement. I test whether a small, public-threat interrupt improves conversion without changing the opening or the installment compiler.

## Change vs. parent

Retain 0014's live construction contracts, exact maritime stage funding, postcondition checks, bounded Astra recovery, and robust legal-action fallback. Add a threat interrupt driven only by current public VP: when an opponent is at least 9 VP (or at least 8 while we are below 9), the executor temporarily releases an expansion contract and enters an endgame mode. In that mode it prioritizes an immediately legal city/settlement or development purchase, plays a useful old development card, and uses the existing robber scorer; it will not end a turn while a legal conversion exists. The interrupt expires after six own turns or after the threat falls, and every action remains matched to the fresh legal menu. A threat event is included in the next Astra request with `ctx.progress()` evidence, so a mistaken interrupt can recover to a new contract.

This is deliberately a narrow structural test: it changes only opportunity-cost arbitration under a visible race, leaving resource valuation, route selection, and model prompts otherwise inherited.

## Confirmation / refutation

Support would be fewer long stretches of end-turn/roll actions after an opponent reaches 8–9 public VP, more immediate point conversions, and higher win rate or weak-seed terminal VP with no increase in rejected actions. Refute if releasing a nearly complete settlement wastes a winning route, if public VP overstates hidden-card threat, or if the interrupt merely buys unfundable development cards and delays our own points. A useful trace should show the interrupt, chosen legal conversion, and post-action VP/contract evidence; unchanged behavior or lower conversion would reject the hypothesis.
