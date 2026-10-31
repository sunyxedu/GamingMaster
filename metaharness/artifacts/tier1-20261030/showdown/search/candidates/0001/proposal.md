parent: 0000-tier1_astra_naive

# Hypothesis: the leader is playing without the evidence needed to correct tactical beliefs

The generic observation renderer truncates the observation inside the move dex. In seed 333667, the initial and next two model prompts are exactly 10,286 characters, contain the truncation marker, and omit both `species_dex` and `recent_events`. Thus the model does not actually receive the complete observation described by the task. Reusing its session does not restore information that was never sent.

Concrete consequences in the worst episode (-1.18451): Volcanion uses Steam Eruption into Calm Mind Giratina, doing only 10% before Leftovers. Next it uses resisted Flamethrower while Dragon Pulse takes 121/256 HP; Haze on the following turn leaves it at 17/256. Only then does it switch to healthy Unaware Clefable, which is immune to Dragon Pulse. Later, at step 11, Clefable tries Fire Blast into Tinkaton and faints to Gigaton Hammer before attacking. The relevant costs, speed evidence and defensive resources need to be visible together. In seed 308081, Suicune repeats Calm Mind after Vileplume switches in and dies to Giga Drain (a critical hit, so not proof that any setup loses); Spectrier later uses Nasty Plot and takes 237/274 from Earthquake. These traces motivate explicit survival and opponent-reply checks rather than a bare action response.

A second structural problem is paying roughly 47 seconds for every decision, including straightforward continuations: three episodes reached the 1800-second deadline and another stopped after 40 calls. A verified continuation can free reasoning budget for genuinely changed situations.

# Changes

* Replace generic truncated JSON and ever-growing sessions with a complete, self-contained battle report. Most recent protocol comes first; all six own sets, current slots, PP, status, boosts, field, opponent reveals, species data and move descriptions are retained. No generic character cutoff hides fields.
* Maintain explicit strategic memory and context-specific direct-damage examples from public protocol. Show recent decisions, observed move order (not mislabeled as speed proof), and their HP consequences.
* Supply transparent type/power ranking aids, current own speed before situational modifiers, and approximate entry damage. These are explicitly not exact damage ranges or guaranteed KOs. Astra must use observed damage and mechanics over approximations. No hidden team data or external simulator is used.
* Ask Astra for a compact tactical rationale, an opponent reply, preserved checks and a win condition. Encourage early defensive Tera when it saves a decisive piece; forbid blindly greedy setup.
* Let Astra authorize at most two extra damaging attacks, or one named follow-up after a switch/setup. Bind plans to identities and move IDs, remapping to current slots. Cancel on opponent changes/reveals, changed mechanics, unsafe HP, failed effects or insufficient damage progress. Every step checks `ctx.progress()` and hands failed assumptions back to Astra.
* Skip calls for a unique legal action. Use fresh sessions, 180-second timeouts (above the archived 104-second maximum), robust JSON/action parsing and legal fallback on timeout/garbage. Reserve the final call and stop starting calls below 150 seconds, allowing the fallback to finish the battle rather than knowingly starting a deadline-bound call.

# Confirmation / refutation

Confirm if prompts now include the recent protocol and all matchup resources, costly setup/survival mistakes decrease, and checked continuations reduce calls without repeating into changed matchups. Strongest evidence would be improved weak-seed outcomes and more completed wins, not merely converting losses to timeouts. Refute if complete evidence does not improve tactics, the approximate hints mislead Astra, or continuations reduce adaptation enough to lose otherwise favorable games. Dev results are diagnostic only and will be recorded below; no seed-specific rules are present.

# Validation

Pre-flight initially found `request: null` at termination; corrected it and re-ran successfully (20 actions, natural game completion). Its short wall allowance exercised fallback only, so separate mock-context runs covered model JSON, garbage, TimeoutError and RuntimeError paths: each completed seven legal actions without escaping errors. Archive replay checked 210 complete snapshots (truncated archived observations were skipped), and 47 public resistance/super-effective events agreed with the type chart. Targeted continuation checks passed for successful continuation and cancellation on new moves, boosts, Tera, immunity and unsafe HP.

Public-schema corrections: hazard keys include `move: ` and opponent Tera uses `teraTypeRevealed`. Persistent public item/ability/Tera facts supplement the convenience summary; damage examples flag critical and faint-censored hits. These reporting refinements were added after the second dev run loaded its harness and therefore require separate validation; the dev run tests the same core control policy but not those final report additions.

The first dev invocation was stopped after the initial pre-flight found the terminal-null bug (it had already loaded the older module). The second, final permitted dev invocation is running both dev seeds. Results pending.
