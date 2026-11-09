# Repairs

- 2026-10-31 08:00: the search stopped at patience 2 after generation 3 (best 0005 = 288 of 864; generations 2-3 ran
  before the try command, FIFO slots and the tool-less / post-error session fixes). It was stopped during validation
  (one validation episode had finished, never logged); the three validation directories moved out of the archive to
  `validation-aborted-0758/` unread, and the search resumed with the new default patience 3 (operational change for
  every game, runs/launcher/restart_game.sh).
