# NetHack adaptor

This adaptor runs the MIT-licensed NetHack Learning Environment (NLE) in process
through `nle.env.tasks.NetHackScore`. It starts no server, display, account, or
network connection. Install `nle==1.3.0` into the evaluator's Python environment
using `requirements.txt`.

The adaptor is Tier 1: `step(action)` submits one NLE action and waits for the
resulting turn. The game is paused between calls, so model deliberation does not
consume wall-clock turns. `action_spec` exposes NLE's 23 task actions as readable
strings (`move n`, `run se`, `up`, `eat`, `search`, and so on); `wait` is the
no-op. Integer indices 0 through 22 are also accepted. Prompt and menu text is
left in `message` and `tty` so the harness can answer it on a later turn.

Observations are strict JSON. They contain glyph, character, and colour arrays,
the textual map and terminal screen, cursor, message, inventory lines, raw
`blstats`, named player statistics, score, maximum depth, terminal flags, and the
last turn reward. The primary `score` is the terminal NetHack score from blstats.
`score_breakdown()` additionally reports maximum dungeon depth and an ascension
flag; `info()` reports engine status and marks a trial invalid after an engine
failure. No screenshots or runtime files are needed.

`reset(seed)` derives NetHack core, display, and level-generator RNG seeds from
the supplied integer and disables reseeding, making fresh instances reproducible
while still varying roles, races, alignments, dungeon layouts, monsters, items,
and branch choices across the declared train/dev/validation/held-out seeds.
The groups are disjoint and have the same broad mix. Seeds divisible by four run
the NLE staircase curriculum and succeed on reaching a downstairs; the others
are ordinary score episodes. Reaching a staircase is visible in the map and
depth fields, and `scenario_success` identifies staircase completion or
ascension. The final score remains the reported objective for every episode.

The bundled engine has no external credentials. `env.example` is included only
to document that fact and an optional caller-side default seed.
