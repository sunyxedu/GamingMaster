parent: 0000

## Hypothesis

The baseline's main failure is not insufficient card knowledge in a long plan: it spends one slow Astra call on every engine prompt and exposes no tactical structure or feedback. The evidence is the flat `-1.000` score on all 12 training seeds despite 112 successful calls, and the seed 1101 opening: the legal menu offered normal summons of Penguin Soldier, Gene-Warped Warwolf, and Giant Soldier of Stone, while the response chose index 0 (Penguin Soldier) despite the available 2,000 ATK attacker. The harness simply forwards a single action choice and has no memory or policy layer.

## What changed

This proposal uses a structural split controller. It ranks only current legal main-phase commands and displayed battle attacks using public card stats and labels: summon the strongest useful ordinary attacker, preserve known Flip monsters face-down, set useful interaction, and avoid attacking into a stronger attack-position monster. Astra handles effect chains, target/cost/subset prompts, and any prompt the cheap policy does not recognize. Replies must match the live prompt token and legal choice string; compound replies are bounds-checked. After each action, the harness checks `ctx.progress()`, the prompt token, and whether the request advanced; a rejection or stall routes the unchanged evidence to Astra for re-planning.

## Evidence that would confirm or refute it

The hypothesis is supported if training episodes improve over the uniform-loss baseline while using fewer System-2 calls, especially in games where strong normal summons, Flip sets, or safe battle arithmetic decide tempo. Trace review should show the cheap actions advancing to new prompts without illegal actions, with Astra concentrated on genuine interaction decisions. It is refuted if the script spends important interaction at poor times, misreads attack targets/positions, or loses the same matchups despite fewer calls. Check per-seat and per-deck outcomes and rejection counts, since a single lucky LP margin is not enough.
