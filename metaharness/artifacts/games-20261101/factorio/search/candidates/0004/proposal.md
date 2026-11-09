parent: 0001

Hypothesis: the previous winner's map-relative build compiler is sound, but its fixed service order spends the shared 100-item transfer quota on furnace and assembler outputs before satisfying the current science demand. This leaves labs missing the pack required by the active research while machines report healthy production, especially on seeds where copper/iron timing differs. A demand-ranked service loop should preserve the winner's early bootstrap while improving late science consumption.

What changed vs. the parent: construction, recipe compilation, placement transforms, and asynchronous repair advice are retained. The service policy now infers the active research ingredient names, computes lab deficits, prioritizes assembler inputs and required lab packs, and only extracts furnace results when a downstream deficit or output pressure justifies it. It also records progress and asks Astra for bounded repairs when the ranked policy stalls or an operation is rejected.

Evidence that would confirm/refute it: confirm if late observations show fewer `missing_science_packs` / `item_ingredient_shortage` statuses, higher consumed science and a later-turn score lift without more failed operations. Refute if output buffers fill and production stalls, or if science consumption is unchanged while the quota allocator delays bootstrap milestones.
