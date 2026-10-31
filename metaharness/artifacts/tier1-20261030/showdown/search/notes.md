
### Candidate 0002 — counterfactual combat cards

- The worst baseline loss (333667, -1.1845) was not a deadline issue: Volcanion lost 239/256 HP chipping Calm Mind Giratina, and Clefable later selected Fire Blast but died to Gigaton Hammer before moving. Explicit survival and check-preservation audits are a separate hypothesis from latency optimization.
- Protocol damage and move-order observations should be retained as contextual evidence. Move order alone does not prove raw Speed: priority, Trick Room and activation effects can explain it.
- Public move_dex contains own-team moves, not necessarily revealed enemy moves. Hypothetical STAB probes must be labeled as probes and must not be presented as inferred enemy sets or damage bounds.
- A fresh stateless prompt with a short strategic memory avoids accumulating duplicate full observations. On 204 archived actionable states the new combat-card prompts stayed below 16.4k characters; no loss of the latest request or recent protocol to generic truncation.
- Preflight passed but its short budget caused zero Astra calls. Separate archive replay verified legal fallback selection and malformed/JSON/plain-action parsing; a passing fallback-only preflight does not establish model-path coverage.

### Candidate 0003 — adversarial response matrix

- A distinct intervention from supplying combat calculations is to ask Astra for counterfactual values over several opponent responses, then let code select a downside-aware action. This makes the opponent-response assumptions inspectable; it does not make them correct.
- The first dev matrix omitted the actual physical attack from an unrevealed opponent. After Body Slam was observed, the next matrix explicitly compared repeating it, switching and defensive Tera. Keep observations capable of overturning the model's initial set hypothesis.
- Public Tera is stored as `teraTypeRevealed` in the observed public summary; private requests use `terastallized`. Verify field names against traces before computing type matchups.
- Terminal observations can have `request: null`. The initial preflight caught this ledger edge case; the corrected preflight passed with one answered System-2 call and no warnings. A short preflight otherwise exercises only fallback if model admission requires a large remaining wall budget.
- Direct-damage samples need to exclude `[from]` residual/recoil events, flag critical hits and faint censoring, and distinguish pre-turn context from effects gained during the turn. Archive replay covered 193 available transitions; 108 serialized observations were truncated and were skipped rather than treated as complete states.

## Candidate 0001 — evidence-first reports (iteration 1)

- The leader's generic renderer truncates *inside* `move_dex`: seed 333667's first three prompts are 10,286 characters and omit `species_dex` and `recent_events`. Fix information delivery before diagnosing every bad action as model strategy failure.
- Recent protocol must precede bulky static data. Keep public damage/move-order evidence and a short strategic memo across fresh model sessions; a reused session cannot recover events never supplied.
- Observation schemas need direct checks: terminal `request` can be null; hazard keys include `move: `; opponent Tera appears as `teraTypeRevealed`. The first pre-flight caught the terminal-null bug, and the repair passed.
- Plans should bind Pokémon identities and move IDs, then remap slots and verify effects after each action. New opponent moves, mechanics, failed attacks, unsafe HP and no-progress signals invalidate continuations and trigger Astra when budget allows.
- A 150-second call reserve avoids beginning a late call near the deadline, but also means the short pre-flight may exercise only fallback. Separate archived-observation and malformed-answer checks are necessary to cover report construction and parsing.

- Candidate 0002 timing follow-up: account for elapsed episode-clock time around ctx.think, not just result.latency_s. The first dev call consumed about 177 wall seconds while the trace reported 55.8 seconds of model latency; admission now reserves the largest observed end-to-end call plus finishing time.
