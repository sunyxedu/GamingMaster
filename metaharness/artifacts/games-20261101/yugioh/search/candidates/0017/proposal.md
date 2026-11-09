parent: 0015

## Hypothesis

The two-sided public combat search still has a specific blind spot: it can prefer changing a known 2000 DEF wall into a weak face-up attacker when the immediate attack does no damage and the resulting board is exposed to a lethal reply. Candidate 0015's archive shows this failure in seed 1111: after two defenders were removed by a Tribute Summon, the remaining Giant Soldier of Stone stayed at 2000 DEF until the local policy selected `change position / flip summon` (steps 186–187), turning it into 1300 ATK. It then attacked into a four-monster board and was destroyed, ending the duel at 0 LP (step 191). The search's unknown-card stopping leaf and resource score were insufficient to protect that wall.

## Change vs. parent

Starting from 0015, add one narrow guard to the position ranker and to Astra's returned action. A face-up defense to face-up attack transition is rejected when the projected immediate damage is zero or when the visible reply rises materially beyond the prior board's reply exposure. The threshold is bounded by the current LP and uses only `position_options`' existing public two-sided horizon. This does not alter legal action encoding, tribute selection, effect handling, Rush planning, or attack choices. A rejected model recommendation falls back to the fresh local policy, and every action remains checked by the existing token/progress/consequence audits.

## Confirmation / refutation

Confirm if replay and evaluation show defense walls retained in blocked positions, while profitable attacks and lethal conversions still proceed, with no increase in invalid actions, truncations, or model failures. Strategic confirmation requires improved weak-seat/deck strata on training and unseen seeds. Refute if the guard causes passivity when a wall must attack to win, suppresses legitimate damage through a known low-defense board, or leaves the 0015 loss pattern unchanged.
