### Important pre-flight considerations

Before treating this as evaluation-ready, I would particularly check three implementation details:

1. **Rivet traversal:** the current two-rivet target selection is deliberately conservative, but its assumed row-to-screen geometry needs observation-based validation.
2. **Jump duration:** the four-tick local hold is an initial control heuristic, not a proven equivalent of the cartridge's complete 31-frame jump.
3. **Ladder selection:** the nearest upward ladder is a useful fallback, but the actual ladder topology and broken-ladder status should eventually determine route selection.

The most important architectural improvement is that a 141 ms Jev response or a 60-second Astra response no longer stops the actor from operating. This is the main hypothesis that candidate 0004 is designed to test.