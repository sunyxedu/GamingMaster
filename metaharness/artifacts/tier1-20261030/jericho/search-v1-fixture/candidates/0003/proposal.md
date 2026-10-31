parent: 0000-tier1_astra_naive

# Hypothesis

The zero score is primarily a harness/adaptor mismatch, rather than Astra being unable to read the task. The baseline prompt asks for a parser command, but its generic `parse_action` searches for the action *name* (`command`). In the seed-6 trace Astra first returned `{"command":"east"}`, yet the harness submitted the literal `command`; the game replied “That's not a verb I recognise.” The next replies `east`, `command east`, and `command("east")` were likewise reduced to `command`, so every later turn repeated the same parser failure. The same pattern appears in the seed-0/4 traces, and the episodes spent the 1,800-second deadline on 17–27 failed turns.

# What changed vs. the parent

Candidate 0003 treats the Jericho action as its actual string value. Astra is asked for a short JSON list of parser sentences, while the harness extracts commands from JSON, fenced text, wrapper forms, or bare lines and submits the extracted string directly to `ctx.act`. It commits one command at a time, retaining a bounded plan to avoid paying for a 50–80 second Astra call before every turn. A parser-error response or a score change clears the cached suffix and triggers replanning, so an incorrect route does not continue blindly. All model and action exceptions are caught, and malformed stand-in output falls back to `look`, preserving the pre-flight contract.

# Evidence that would confirm or refute it

Confirmation would be a nonzero native score on the same fixture with the first action reaching the described east room, followed by the carrot/chest operations, while using substantially fewer Astra calls than the parent. Trace evidence should show `ctx.act` receiving strings such as `east` and `put carrot in chest`, never the schema label `command`. Refutation would be a trace where direct parser strings are accepted but the task still makes no progress, or where the model's plans are consistently semantically wrong after the channel mismatch is fixed; that would motivate a stronger state ledger or puzzle-specific replanning in the next iteration.
