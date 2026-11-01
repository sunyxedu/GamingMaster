# Candidate 0012 — direct mining transactions escape the navigator basin

## Parent and isolated hypothesis

Parent: `candidates/0006/` (with the best valid comparator `0008/`, and the
postcondition/quarantine refinement `0011/`).  This is a structural rewrite of
the resource executor, not a copy of their frontier policy.

The hypothesis is that the missing stone/iron transition is primarily an
**execution-interface failure**, not a lack of a better milestone menu.  The
prior candidates ask `gather` to path, select a tool, mine, and pick up a block
in one bounded job.  They then count navigation displacement as progress.  In
the traces this produces a false-success loop: 0009 seed 0 made 130
`stone_pick`/122 `approach_table` posts without stone; 0008 seed 1 repeatedly
reported `navigate s:...` as `action_progress` while the inventory remained
empty; and 0008/0011 retried the same hidden log `b:-12,64,-21` after the
adaptor had already shown it could not acquire it.  The initial seed-1 block is
an invisible oak log in swamp water (`b:-3,64,-25`), so “a safe nearby stand was
visited” is not evidence that the block is exposed or that a drop was picked up.

0012 tests a different causal chain: do not use `gather` for milestone blocks.
Bind one observed block, navigate to its interaction position, explicitly
select the required tool, aim with `look_at`, issue one direct `dig`, and then
walk to the observed item entity.  A mining transaction completes only on an
inventory/drop/block postcondition; navigation alone never completes it.  If
all currently exposed candidates are absent or fail once, a nine-second manual
cardinal excursion changes the loaded terrain rather than cycling the same
stand candidates.  The excursion is bounded and has hazard overrides.

## What changed

* Added a direct resource transaction with `approach -> equip -> look_at ->
  dig -> pickup`; wood is mined by hand, stone by a wooden pick, and iron/coal
  by a stone pick.  Failed target IDs receive a cooldown and cannot monopolize
  the episode.
* Added explicit staged workstation interactions (`equip` before `place`) and
  real furnace start/take timing.  Crafting is still code-owned and uses only
  table IDs observed in the current snapshot.
* Added short `look_at` + `controls forward,jump` frontier excursions in four
  directions.  They stop as soon as a relevant block/entity is exposed.  This
  is deliberately not a permanent movement autopilot and is cancelled by
  lava/fire/drowning/hostile-threat handling.
* Kept a small deterministic survival progression: wood/table, wooden pick,
  exposed stone/stone pick, food/weapon, exposed iron/furnace/shield, and bed
  when wool or a bed is actually observed.  No coordinates, seeds, structures,
  or underground blocks are assumed.
* Jev is used only as an asynchronous immediate-threat selector.  Astra is not
  called: a 35–100 second macro answer cannot repair a bad `gather` primitive
  while the world continues, and this experiment isolates the direct-action
  hypothesis.

Evidence that would confirm the hypothesis is an improvement in verified
stone/iron inventory or the `iron_shield` score component, together with trace
records `direct_mine_confirmed`; manual segments should be followed by new
candidate IDs rather than repeated stand-only movement.  It is refuted if
`dig` transactions reach valid in-reach/visible blocks but still produce no
block change or pickup, or if runs remain in wood despite new exposed logs.
A successful opening with the same 3-point plateau would also refute the claim
that gather/postcondition coupling was the main bottleneck.

## Jev interface and real-time fallback

Each threat request sends a compact JSON state:

```json
{
  "purpose": "immediate survival response",
  "self": {"health_band": "stable|low|critical", "food_band": "fed|low",
            "dimension": "overworld|nether|the_end", "lava_fire": false},
  "threats": [{"kind": "creeper|ranged|lava_or_fire|...",
                "range": "touching|near"}],
  "legal_actions": ["shield on", "attack <observed id>", "controls back,jump", "noop"]
}
```

The batched questions are:

* `response: Choice` — choose one exact value from `legal_actions`; code
  rejects invented or stale values and requires probability at least `0.55`.
* `continue_safe: Noul` — an independent safety check retained in the request;
  it is not allowed to override the exact emergency rules, but is useful trace
  evidence about whether the tactical menu is being presented correctly.

Threat signatures contain only observed kind, entity ID, and a named distance
band.  A response is usable only for the same signature and a 2.5-second
lease.  While Jev is pending, code immediately uses a deterministic fallback:
leave lava/fire, surface from drowning, eat at low health/hunger, attack a
close visible target with an observed weapon, raise an observed shield, or
back-jump.  A current bounded job is cancelled before a movement/combat
replacement.  Stale observations cancel jobs; death sends one `respawn` and
waits.  Malformed model values, timeouts, and low-confidence choices fall back
without escaping `run_episode`; budget/model-unavailable exceptions retain the
runtime contract.

The direct mining state machine itself is model-free.  It never treats a
navigation move, nonempty feedback, or model choice as success: only a changed
block plus authoritative inventory/drop evidence advances the transaction.
Repeated one-shot actions are separated by `noop`, and controls are renewed
only during the bounded manual segment.
