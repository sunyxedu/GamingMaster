parent: 0011

## Hypothesis

A material share of the leader's wasted turns comes from parser vocabulary mismatch rather than puzzle misunderstanding. The 0011 traces show Astra choosing `northeast` and then `north` in Enchanter: the story answers “I don't know the word” and “You can't go that way,” triggering another 40–60 second call. The prose can describe a diagonal direction while the parser accepts an abbreviation (`ne`), or can mention a route that is not an exit from the current room. A code-side repair after an explicit parser failure should recover common direction aliases without buying another model call, while retaining the frontier to stop route loops.

## What changed vs. the parent

0014 keeps 0011's fresh Astra sessions, compact evidence packet, directed frontier, exact transcript, short model timeout, and total output decoding. It adds a conservative parser-repair table for direction aliases (`northeast`→`ne`, `northwest`→`nw`, etc.) and a one-shot retry policy. The retry is attempted only when the immediately preceding response explicitly indicates an unknown verb/direction or an impossible exit, and only when the replacement is a distinct canonical alias. Every repair is logged and marked as an attempted edge, so it cannot loop. The prompt also asks Astra to prefer the exact short direction tokens observed in the current text and to avoid treating every mentioned direction as a confirmed exit. Non-direction parser errors still invalidate the route and trigger replanning.

This tests a structural parser-adaptation hypothesis rather than another batching or context-length tweak. It uses no title, seed, walkthrough, or VM state assumptions.

## Evidence that would confirm or refute it

Confirmation would show `repair` events immediately converting direction parser failures into successful room transitions, fewer repeated Astra replans after “I don't know the word”/“can't go” responses, and higher score on Enchanter-like seeds without more deaths. It would also show that the frontier remains useful after a repaired move. Refutation would be near-zero repair hits, or repairs that turn valid prose directions into wrong moves and lower scores; then the limiting problem is puzzle dependency planning or exit grounding rather than parser spelling.
