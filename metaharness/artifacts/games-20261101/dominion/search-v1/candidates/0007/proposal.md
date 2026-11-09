parent: 0006

## Hypothesis

Candidate 0006 fixed one failure mode (missing a visible Province race), but its
race lock is too eager. In its own training trace for seed 1101 it bought a
Province with five Provinces remaining and only two opponent Provinces, then
locked again at three remaining and bought Province/Duchy while the public score
was still close. In seed 1300 it locked at five and four Provinces remaining,
including a Duchy while the agent led 18--8. The resulting mean margin (7.667)
fell below candidate 0003's 9.500 despite identical local rollout machinery.

The hypothesis is that public race information should change the policy only
when the opponent can plausibly end soon or the Province pile is itself nearly
empty. A premature green purchase loses payload and delays the next $8 turn;
the rollout already handles ordinary scoring once the race is not urgent.

## What changed vs. the parent

I kept 0006's public-belief forward model, common-random-number buy search,
prompt executor, bounded asynchronous Astra advice, and parser/legality
fallbacks. I replaced only `race_choice` with a lead-aware gate:

* Province is forced when the opponent owns at least three Provinces, when it
  owns two and only three Provinces remain, when the opponent leads by four VP
  with four Provinces remaining, or when the pile has two cards left.
* Duchy is forced only at one Province remaining. An agent that is ahead keeps
  buying payload until this late threshold, allowing rollout search to choose
  its own green timing.
* The WM-1 Moat branch is restricted to turns where no green card is affordable,
  so defense cannot displace an available Province or Duchy.

This is a structural test of when to switch from construction to scoring, rather
than another card value or Monte Carlo sample-count tweak.

## Evidence that would confirm or refute it

Confirmation would show fewer early `race_lock` events, preserved or improved
final margins on engine kingdoms, and wins in traces where 0006 bought Duchies
while leading. The useful event pattern is a later lock followed by a Province
purchase before the opponent empties the pile, with no illegal actions or
truncations.

Refutation would be a clean loss in which the opponent reaches three Provinces
while this controller continues buying terminals, or a systematic tie/loss from
waiting until two Provinces remain. Such traces would justify moving the threat
threshold earlier for the relevant opponent, while keeping the lead-aware
condition that prevents unnecessary Duchies.

The pre-flight dry run completed 64 legal steps (score 15, no model failures);
it used stand-in models and therefore only validates contract robustness, not
the strategic hypothesis.
