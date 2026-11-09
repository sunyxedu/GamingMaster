parent: 0005

## Hypothesis

The recent candidates lose momentum because Astra is asked to choose every request independently. The traces show long sequences of voluntary switches (for example candidate 0003 repeatedly switches at steps 4, 5, 7, 18, 23 and 30 in seed 104729) and roughly 50–80 seconds of model latency per decision. This creates oscillation and leaves too few accepted decisions before the 1800-second deadline. A short, validated action line should preserve tactical context while a two-step commitment window prevents immediate switch reversals.

## What changed vs. the parent

Candidate 0006 asks Astra for up to three commands in a single bounded prompt. It executes the first command and retains later commands only while each remains in the current legal-action list; a changed active or forced replacement clears stale plans. The local type/power/utility scorer remains the fallback. After a voluntary switch, it suppresses another voluntary switch for two accepted decisions while an attack is legal and the active is not critically low HP. Calls are capped at 12 and made at coarse checkpoints, leaving more time for engine decisions and reducing raw ledger repetition. Hidden-trap errors trigger a fresh observation and re-ranking.

## Evidence that would confirm or refute it

Confirmation would show more accepted actions per episode, fewer adjacent voluntary switches, and improved weak-seed HP margins or wins with fewer Astra calls. Refutation would be losses where a necessary immediate double-switch is blocked, or where model-proposed later tokens become systematically stale and the deterministic scorer chooses poor attacks; that would favor a more precise damage/speed ledger or a softer commitment rule.
