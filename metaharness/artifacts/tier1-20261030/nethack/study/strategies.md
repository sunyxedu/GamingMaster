# Strong-play knowledge and where to implement it

NetHack has enormous item, monster, dungeon and event variation. The [NetHack Guidebook](https://nethack.org/v366/Guidebook.html),
[NetHack Wiki](https://nethackwiki.com/wiki/Main_Page), and [NetHack strategy pages](https://nethackwiki.com/wiki/Strategy)
are the main references. Treat the following as priors, with Astra allowed to override them from
the current map and messages.

## Opening and early dungeon

* **Explore conservatively.** Map rooms and corridors, search suspicious squares, pick up useful
  food/armor/weapons, and identify safe escape routes before fighting. Keep hunger under control
  and do not spend turns wandering when HP is low. Encode movement legality, map memory, hunger
  and HP thresholds in **code**; ask Astra to choose routes and risk tradeoffs.
* **Use the starting inventory and role.** Read the role/race/alignment in the initial status and
  equip the best weapon/armor only after checking encumbrance and cursed status. Role skills and
  starting pets strongly affect the opening. **Code** can track equipment and cursed/uncursed
  knowledge; **Astra** plans role-specific priorities.
* **Level and resource discipline.** Gain a few safe experience levels, keep food and healing,
  and avoid unnecessary melee with unknown monsters. The wiki's [Beginning players' guide](https://nethackwiki.com/wiki/Beginner%27s_guide)
  and [conduct](https://nethackwiki.com/wiki/Conduct) explain common early choices. Put exact
  inventory counts and HP/energy alarms in **code**; Jev can answer “is retreat/healing required?”

## Combat and survival

* **Exploit terrain and distance.** Fight from corridors/doorways, use ranged attacks and wands,
  close doors, and avoid being surrounded. Retreat when HP, armor class, or escape resources are
  poor. Monsters have resistances and special attacks; consult the [monster database](https://nethackwiki.com/wiki/Monster)
  and [resistance guide](https://nethackwiki.com/wiki/Resistance). **Code** should maintain a
  visible-monster threat table and legal escape actions; **Jev** should rapidly classify the
  immediate threat and best one-turn response; **Astra** handles multi-turn positioning.
* **Identify before trusting.** Test or identify unknown potions, scrolls, rings and wands safely;
  never quaff/read unknown dangerous items while polymorphed, blind, confused, or in combat.
  Maintain object identity, BUC (blessed/uncursed/cursed), charges and known effects in **code**;
  have **Astra** plan identification experiments. Jev can answer typed recognition questions from
  an item description and known history. See [identification](https://nethackwiki.com/wiki/Identification)
  and [BUC](https://nethackwiki.com/wiki/BUC).
* **Escape beats damage.** Keep teleportation, digging, emergency healing, cancellation and a
  safe staircase available. A guaranteed escape is usually better than one more attack. Encode
  hard rules such as “do not enter lethal adjacent combat” and inventory availability in **code**;
  Astra selects among escapes when several are possible.

## Mid/late game and ascension

* **Prepare resistances and key tools.** Seek fire/cold/sleep/shock/poison/disintegration
  resistance, reflection, magic resistance, intrinsic speed and reliable AC. Keep a blessed
  weapon, food, healing and a way to dig or teleport. The [ascension kit](https://nethackwiki.com/wiki/Ascension_kit)
  and [resistances](https://nethackwiki.com/wiki/Resistance) pages give checklists. **Code** tracks
  achieved intrinsics, equipment slots and missing kit items; **Astra** prioritizes routes and
  shops.
* **Use shops and Sokoban intelligently.** Price-identify objects, buy critical food/scrolls and
  avoid shop theft unless the plan supports it. Sokoban can provide an early amulet of reflection
  or bag of holding but has boulder traps and alignment choices ([Sokoban](https://nethackwiki.com/wiki/Sokoban)).
  **Astra** decides whether optional branches are worth risk; **code** handles gold and carrying
  capacity.
* **Endgame discipline.** Before entering Gehennom, verify resistances, escape items, blessed
  weapon, charging/holy water and inventory slots. Locate the vibrating square, retrieve the
  Amulet, then return by a known safe route; avoid fighting unless needed. Use **Astra prompts**
  for route planning and endgame contingencies, with **code** enforcing readiness gates.

## Common failure modes and counters

* Walking into a monster while confused/blind, forgetting a prompt, or repeatedly attacking a
  resistant monster wastes turns. **Jev questions:** “Is there an unresolved prompt?”, “Which
  adjacent square is safest?”, “Does this monster resist the proposed attack?”
* Eating corpses can grant intrinsics but can also poison, induce illness, polymorph, or violate
  conduct. Track corpse age/type and conduct in **code**; let **Astra** decide when the benefit is
  worth the risk ([corpse](https://nethackwiki.com/wiki/Corpse)).
* Cursed items, unblessed water, inventory overload and hunger silently kill runs. Make BUC,
  encumbrance, hunger and HP checks deterministic **code** gates before risky actions.
* The game is stochastic and partially observed. Astra should maintain a concise belief state
  (map, monster locations, item identities, resistances, escape routes), update it after every
  message/map change, and ask Jev many independent typed questions rather than asking Jev to
  plan a route. This division follows the environment's per-turn observations and preserves
  exact bookkeeping in the harness.

## NLE-specific policy choices

Use `NetHackScore` for the primary objective and retain maximum depth as a secondary metric.
For curriculum/debugging, `NetHackStaircase` rewards reaching a deeper staircase and
`NetHackChallenge` exposes a harder fixed task; compare reward definitions in the [NLE README](https://github.com/facebookresearch/nle#environments).
Seeded resets should vary role/race/alignment and dungeon RNG while holding the same observation
contract. Do not memorize a single map: held-out seeds test exploration, item identification and
safe recovery.
