Implemented the Malody adaptor at the requested path:

- Added deterministic Tier 2 native `.mc` Key-mode emulator in `game.py`
- Added `ADAPTOR.md`, `requirements.txt`, and `env.example`
- Supports seeded 4K–7K charts, BPM changes, offsets, patterns, holds, chords, Flip/Const mods, judgments, scoring, and RM proxy
- Handles malformed actions and missed holds safely

Validation passes:

```text
# Adaptor validation: PASS
```

Also verified perfect, noisy, random, noop, timing-window, and hold scenarios.