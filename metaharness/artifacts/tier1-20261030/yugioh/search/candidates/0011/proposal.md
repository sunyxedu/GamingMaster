parent: 0007

## Hypothesis

The next bottleneck is allocation of the once-per-turn Normal Summon, not another
chain-response parameter. Fixed command scores prefer a small new body over a
Tribute Summon even when all existing bodies are blocked, and select face-up
attackers without accounting for the exposed opponent counterattack. Compare
complete deployment alternatives (including which bodies pay the Tribute) by
their resulting battlefield rather than printed ATK or a fixed card priority.

The current leader 0008's worst training episode is seed 1111. Its trace at
step 106 chooses `choose r2p49 2`, normal summoning Mystical Elf, while Luster
Dragon (1900 ATK) and Battle Ox (1700 ATK) face a 2000-DEF Mystical Elf.
`choose r2p49 0` is an offered Blue-Eyes Tribute Summon. All three attackers
then fail to clear that defender. At step 185, `choose r3p58 1` summons Luster
Dragon against 2000-ATK Warwolf, although `choose r3p58 0` offers Blue-Eyes and
two 2000-DEF bodies are present. These are missed deployment alternatives,
not proof that either counterfactual necessarily wins through hidden backrow.

## Change against parent

Branch from 0007, retaining its validated public combat assignment, proactive
Rush expansion, legal selections, position semantics and guarded model transport.
Replace summon/set ranking with a small public-board deployment search. Enumerate
each offered summon/set and all relevant one/two-body Tribute combinations;
score resulting useful bodies, immediate battle opportunities, card expenditure
and a bounded opponent counterattack projection. Hidden defenders remain
uncertain; no identities, opponent hands, future draws or seed data are read.
The native menu remains the legality authority. Submit one command, bind its
Tributes to the subsequent exact weighted menu, and audit the observed body and
materials after intervening placements/chains. A mismatch invalidates the plan
and hands the evidence to Astra. Never queue hypothetical attacks.

Astra receives the competing deployment evaluations and can choose one current
legal action. There is no future main-command queue to override newly observed
deployment evidence. Every action retains progress, illegal-action, prompt and
semantic-cycle checks. Model requests use a 235-second timeout and separate
210-second wait guard, both above the observed 120-second archive maximum;
late answers are discarded. Failed inference falls back to fresh legal play.

## Confirmation / refutation

Confirm the mechanism if the two archived blocked-body positions now consider
and prefer an offered Tribute breakthrough, retain legal target/material
binding, and avoid spending productive attackers on unnecessary Tributes.
Strategic confirmation requires more native wins across deck/seat scenarios,
especially tribute_pressure, without errors, forfeits or truncations.
Refute if the resource model overvalues big monsters, overcommits into removal,
sets too passively, or improves arithmetic without improving native wins.

## Validation

Public replay covered 3,591 archived observations with zero invalid generated
actions and 242 deployment comparisons. The seed-1111 fixtures at steps 106 and
185 select the offered Blue-Eyes action and bind both current Tribute bodies.
Garbage model replies, timeouts, stale tokens, weighted Tributes, complete
orders, and interrupted deployments all recover in checks.py. A development
run on seed 3301 completed all three duels natively at 0.378958 (2 wins, 1
loss), with no errors, forfeits, truncations, or invalid actions; it used one
model request, which hit the independent wait guard, so this number mostly
measures the public deployment policy rather than Astra's contribution.
