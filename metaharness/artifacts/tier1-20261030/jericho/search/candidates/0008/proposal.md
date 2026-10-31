parent: 0005

## Hypothesis

The main failure in 0005 is context and latency accumulation rather than lack of
puzzle reasoning. Its `session_id` continuation lets Astra's hidden transcript
grow across replans; candidate 0003 already showed inputs expanding from roughly
14k to 50k tokens while routes collapsed into repeated directions. In 0005's
seed 1 trace, the very first model request remained in flight until the 1800
second deadline, so the episode posted no action. In the successful traces,
long prompts and repeated calls still leave many turns unused. Candidate 0006
shows that simply shortening visible history is insufficient when its generic
noun probes turn prose fragments such as `can't` into commands.

## What changed vs. the parent

0008 starts a fresh Astra session for every planning call and sends a bounded,
explicit packet: the opening text, current text, eight recent exact
command/response records, score/move counters, and a short unverified-belief
notebook. It asks for only 2--5 commands, uses a 100-second per-call timeout,
and retains a suffix through positive score events while cutting it on parser
errors, clarification/menu text, hazards, negative score changes, or repeated
exact command/response/score states. Model output decoding is total over JSON,
fenced JSON, tagged lines, and garbage. After malformed or timed-out calls the
only scripted recovery is one `look`; the noun-probe generator from 0006 is
intentionally removed because it produced actions like `examine can't` and
`take can't` in the enchanter traces. Exact transcript and failed-action
bookkeeping remain in code, while beliefs remain explicitly provisional for
Astra.

## Evidence that would confirm or refute it

Confirmation would show smaller and more stable Astra requests, no episodes
lost inside one model call, more parser actions per completed call, and equal or
higher final native score than 0005 without stale-route repeats. Fresh-session
traces should replan from the compact evidence packet rather than replay a
hidden conversation. Refutation would be lower scores despite more completed
calls, or safe routes that need the longer 0005 commitment window; repeated
identical responses after fresh replanning would indicate that puzzle
interpretation, rather than context growth, is the limiting factor.
