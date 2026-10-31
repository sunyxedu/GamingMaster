# Programmatic interfaces

## 1. Official NetHack Learning Environment (chosen)

**Maturity and exposure.** NLE is the maintained research wrapper used by the NetHack RL
community (MIT-licensed code; the bundled NetHack engine has its own permissive NetHack license).
It launches a headless NetHack process in-process and implements the Gym/Gymnasium `reset`,
`step`, `action_space`, and `observation_space` contract. The standard observation is a Python
dict containing a 21x79 glyph map (`glyphs`), terminal character/color arrays (`chars`, `colors`,
`special`, `tty_chars`, `tty_cursor`), the latest message, and numeric player statistics
(`blstats`, including x/y, HP, max HP, strength, experience, gold, dungeon depth, hunger,
alignment and time). This is richer and less error-prone than OCR. See the [NLE README](https://github.com/facebookresearch/nle)
and [observation definitions](https://github.com/facebookresearch/nle/blob/main/nle/nethack/nethack.py).

**Control and latency.** `env.step(action_integer)` performs exactly one NetHack command and
returns `(observation, reward, terminated, truncated, info)` (older releases return the four-item
Gym tuple). Actions are the integer key codes in `nle.nethack.ACTIONS`; use `nle.nethack.actions`
or the enum rather than hard-coding ASCII values. `env.reset(seed=seed)` starts a reproducible
episode when the engine/build supports seeded RNG. Headless local steps normally take
milliseconds, while model deliberation can happen between steps without game-time advancing.

**Install on the evaluator.** Use the evaluator interpreter, never system Python:

```bash
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -m pip install --upgrade pip
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -m pip install nle
```

If no cp312 wheel exists, install build tools (`cmake`, `bison`, `flex`, a C/C++ compiler and
Python development headers) with the host package manager, then install from source:

```bash
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -m pip install --no-binary nle nle
```

NLE's build instructions and supported versions are in [INSTALL.md](https://github.com/facebookresearch/nle/blob/main/INSTALL.md).
Verify with:

```bash
/home/hemlix/GamingMaster-mh-tier1/runs/venvs/nethack/bin/python -c \
  'import nle; from nle.env import NetHackScore; e=NetHackScore(); o,i=e.reset(seed=0); print(e.observation_space, e.action_space, o.keys()); e.close()'
```

No account, server, API key, GUI, or network credential is required. Environment variables are
empty for the normal local install. Set `NLE_SEED` only in an adaptor if a harness-level seed
configuration is desired; pass the actual seed to `reset(seed=...)`.

**Legal/cost/risk.** Free and offline after installation. It runs the bundled open-source engine,
so online-server terms and anti-cheat do not apply. Pin the NLE version and engine seed because
minor releases can alter observation details. Python 3.12 wheels may be unavailable; a source
build is the main setup risk.

## 2. Direct NetHack engine / tty bot

Building NetHack from the [upstream source](https://github.com/NetHack/NetHack) and driving its
tty protocol gives complete textual screens and keystrokes. Mature for human-facing bots, but
requires parsing menus, prompts, cursor position and terminal control sequences. It is slower and
more brittle than NLE, and exposes less structured state (no direct glyph/blstats dict). Offline
and free; no credentials. Use only if NLE cannot compile.

## 3. Existing bot frameworks

[TAEB](https://github.com/ekg/taeb) and [Nethack4j](https://github.com/NullDev/Nethack4j)
demonstrate autonomous play and parsers, but are old/community projects rather than a stable
Python 3.12 interface. They can supply heuristics or a tty parser, not a drop-in evaluator API.
Their use against public servers may violate server rules; local bundled NetHack avoids that risk.

## 4. Screen capture and input injection

`pyte`/`curses` plus `xdotool` or `python-xlib` can capture a terminal and inject keys. This is a
last-resort Linux GUI/TTY method: it needs a display, has OCR/cursor synchronization failures,
adds latency, and violates the headless constraint. Free software but no advantage over NLE.

## 5. Remote public NetHack server

Telnet/SSH to a public server provides authentic games, but needs an account/server address,
network access and a terminal bot. Latency, disconnects, server policy and anti-bot rules make it
unsuitable for this offline evaluator. Never put credentials in study files.

## Adaptor contract for NLE

1. Construct `nle.env.NetHackScore` (or `NetHackChallenge` if challenge scoring is explicitly
   wanted) with a fixed `observation_keys` set that includes `glyphs`, `chars`, `colors`,
   `blstats`, `message`, `tty_chars`, and `tty_cursor`.
2. Call `reset(seed=seed)` and serialize observations to JSON-friendly lists/base64 only at the
   Astra/Jev boundary; retain NumPy arrays internally.
3. Render `glyphs` using NLE's glyph helpers or use `tty_chars` for a 21x79 textual map. Decode
   `message` and inspect `blstats` every step. Track `info`, `terminated`, and `truncated`.
4. Map a planned command to the matching `ACTIONS` member and call `step`. If a message reports
   a prompt/menu, expose it and continue sending menu-selection keys until the prompt clears.
5. Report final `info`/`blstats` score and maximum dungeon depth; close the environment on exit.

The [NLE example agents](https://github.com/facebookresearch/nle/tree/main/examples) show action
selection and observation handling.
