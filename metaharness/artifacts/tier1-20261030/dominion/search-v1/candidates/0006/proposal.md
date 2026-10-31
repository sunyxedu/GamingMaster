parent: 0003

## Hypothesis

The rollout controller's remaining losses come from race blindness. Its sampled hidden-deck search models the opponent policy, but it does not treat publicly visible Province ownership and pile pressure as a hard strategic change. In the completed seed 1101 trace (`base_intro`, BM-1, seat 1), the agent bought Market, Village, Merchant and several other terminals while the opponent reached four Provinces. The Province pile then emptied on turn 16 with agent 29 VP versus opponent 30 VP. The trace contains no illegal action or parser failure; the failure is timing against a known public race.

## What changed vs. 0003

I kept the public-only forward model, local prompt executor, bounded buy rollouts, and optional Astra archetype advice. Before rollout search at every buy prompt, candidate 0006 applies a narrow public race lock:

* take an affordable Province once five or fewer remain, once the opponent publicly owns at least two Provinces, or when the opponent leads by four VP near the pile;
* take an affordable Duchy when three or fewer Provinces remain and no Province can be bought;
* against WM-1, consider one early Moat after a Witch is publicly owned, only when no green card is affordable and the agent has no Moat.

This tests whether explicit opponent progress and end-race pressure improve margins without replacing the controller's engine construction elsewhere. The intervention is local and reversible: it never uses hidden hands or deck order, and it leaves exact legality and VP arithmetic to the adaptor.

## Evidence that would confirm or refute it

Confirmation would be fewer losses in BM/SM races, especially traces where `race_lock` fires before the last Provinces disappear, with completion and illegal-action rates unchanged. A useful confirmation is an earlier green purchase followed by a positive final margin on base_intro-like seeds. Refutation would be unchanged margins despite timely green purchases, or losses caused by buying green cards so early that the deck cannot reach $8; those traces would show that the trigger thresholds are too aggressive. WM games that lose after a Moat purchase would refute the defense branch and suggest removing it.

The two development seeds were mixed (`-5`, `+4`, no errors). On the `-5` Gardens game the lock fired three times with the opponent at three Provinces, but the deck still finished five VP behind; this is evidence that a public race trigger is necessary but not sufficient, and leaves the threshold and payload interaction open for training evaluation.
