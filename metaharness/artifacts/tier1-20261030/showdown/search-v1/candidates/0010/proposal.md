parent: 0007

## Hypothesis

The leader's approximate search is often reasonable, but Astra can override it with a stale voluntary switch or a low-value Terastallization at the exact point where a cheap local invariant is decisive. This is a controller arbitration failure: the model sees a large prose state, while the code has current legal actions, speed, damage and HP estimates. The 0007 seed 282617 trace contains literal records `{"source":"astra","action":"switch 6","turn":1}` and then `{"source":"astra","action":"switch 3","turn":2}`, followed by switches 5 and 4 on turns 4 and 6. In seed 257053 it chose `switch 6`, `switch 3`, and a Tera move before fallback play settled into more switch/attack alternation. Those openings spend turns exposing replacements even when the active can make progress; the later battle scores were -1.067 and -1.064.

## What changed vs. the parent

I copied 0007's public-only protocol ledger, move-specific damage evidence, missing-move fact validation, bounded policy continuation, timeout recovery, and deterministic ranking. After parsing Astra's JSON, candidate 0010 applies a narrow action firewall before accepting the model action or saving its continuation policy:

- a voluntary switch is vetoed when the active is estimated to move first and secure a KO while it is not itself near the incoming-damage threshold;
- a second consecutive voluntary switch is vetoed when the proposed entrant is estimated to take at least 62% and cannot return meaningful damage;
- a Tera command is vetoed when its current ranking value is within seven points of the corresponding ordinary move, preserving the one-use resource for a decisive change.

The gate returns to the same request-local deterministic ranking, so forced replacement, hidden-trap retries, malformed prose and model timeouts retain the parent's safe path. It logs each veto separately and saves a policy only for the accepted command.

## Evidence that would confirm or refute it

Confirmation would show fewer adjacent voluntary switches, fewer replacement KOs, and more attacks immediately after model checkpoints without reducing the leader's strong wins. The logs should contain `veto_switch_ko`, `veto_switch_chain`, or `veto_small_tera` at the previously observed failure patterns. Refutation would be a seed where the gate blocks a necessary double switch or defensive Tera: traces would show a veto followed by a large incoming hit despite a viable alternative, or a lower HP margin on positions where the parent survived. The thresholds are intentionally local and approximate; they should be relaxed or removed if such false vetoes dominate.
