#!/usr/bin/env python3
"""
Unified data pipeline for generals.io replays.

One command takes each game end-to-end:

    metadata fetch → filter → download → simulate → convert → training JSON

and keeps going until --target trainable games per player exist on disk (or
the player's history is exhausted). Every candidate game's fate lives in a
per-player manifest, so reruns resume exactly where they stopped and never
redo work. Run the SAME command on any machine:

    uv run python -m scripts.prepare_data \\
        --players Human.exe --target 3000 \\
        --output /scratch/$USER/gaming-master/data-v2

Machine notes:
  - fetch/filter/download/simulate need no numpy (old login nodes work);
    convert does. Without numpy, finished games park in the "simulated"
    state (they still count toward --target) and the next run on a
    numpy-capable machine converts them.
  - Only the fetch step needs internet; a run without new fetching (e.g.
    finishing conversions on a compute node) works offline.

Layout:
    output/
      manifest/<player>.json        single source of truth per game
      replays/<id>.json             raw replays (RL map pool, viewers)
      filtered/<player>.json        trainable-game list (train_rl --filtered_file)
      simulated/<player>/<id>.json  parking lot when numpy is unavailable
      training/<player>/<id>.json   final training format

Game statuses in the manifest:
    pending    filtered candidate, not yet processed
    converted  in training/ (terminal, counts toward --target)
    simulated  awaiting convert on a numpy machine (counts toward --target)
    skipped    simulator rejection — oversize map, modifiers, modern
               features (terminal, never retried; reason recorded)
    error      transient failure, retried on the next run
"""

import argparse
import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

FILTER_START_DATE = "2025-04-01T00:00:00Z"


# ----------------------------------------------------------------------
# Worker functions (top-level for pickling; imports inside so the parent
# process never needs numpy)
# ----------------------------------------------------------------------

def _process_game(args) -> str:
    """simulate (+ convert when possible) one game. Returns final status."""
    replay_path, player, training_path, sim_fallback_path, do_convert = args
    from data_processing import simulate_replay

    sim = simulate_replay(replay_path, player)  # ValueError = permanent reject
    if do_convert:
        from data_processing import convert_simulator_to_gio
        gio = convert_simulator_to_gio(sim)
        with open(training_path, "w") as f:
            json.dump(gio, f)
        return "converted"
    with open(sim_fallback_path, "w") as f:
        json.dump(sim, f)
    return "simulated"


def _convert_parked(args) -> str:
    """Finish a game parked in the simulated state; removes the parking file."""
    sim_path, training_path = args
    from data_processing import convert_simulator_to_gio

    with open(sim_path) as f:
        sim = json.load(f)
    gio = convert_simulator_to_gio(sim)
    with open(training_path, "w") as f:
        json.dump(gio, f)
    os.remove(sim_path)
    return "converted"


def _numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

class PlayerPipeline:
    def __init__(self, player: str, output_dir: Path, args):
        self.player = player
        self.args = args
        self.manifest_file = output_dir / "manifest" / f"{player}.json"
        self.replays_dir = output_dir / "replays"
        self.sim_dir = output_dir / "simulated" / player
        self.training_dir = output_dir / "training" / player
        self.filtered_file = output_dir / "filtered" / f"{player}.json"
        for d in (self.manifest_file.parent, self.replays_dir, self.sim_dir,
                  self.training_dir, self.filtered_file.parent):
            d.mkdir(parents=True, exist_ok=True)

        if self.manifest_file.exists():
            m = json.load(open(self.manifest_file))
        else:
            m = {"fetched_offset": 0, "games": {}}
        self.offset: int = m["fetched_offset"]
        self.games: Dict[str, dict] = m["games"]

    # -- bookkeeping ----------------------------------------------------

    def save(self) -> None:
        tmp = self.manifest_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"fetched_offset": self.offset, "games": self.games}, f, indent=1)
        tmp.replace(self.manifest_file)

    def trainable(self) -> int:
        return sum(1 for g in self.games.values()
                   if g["status"] in ("converted", "simulated"))

    def by_status(self, *statuses: str) -> List[str]:
        return [gid for gid, g in self.games.items() if g["status"] in statuses]

    def _passes(self, entries: List[dict]) -> List[dict]:
        from data_processing import filter_replays
        return filter_replays(
            entries,
            min_turns=self.args.min_turns,
            player_name=self.player,
            min_opponent_stars=self.args.min_opponent_stars,
            start_date=FILTER_START_DATE,
        )

    def _task(self, gid: str, do_convert: bool):
        return (str(self.replays_dir / f"{gid}.json"), self.player,
                str(self.training_dir / f"{gid}.json"),
                str(self.sim_dir / f"{gid}.json"), do_convert)

    # -- pipeline steps ---------------------------------------------------

    def finish_parked(self, pool: ProcessPoolExecutor) -> None:
        """Convert games parked in the simulated state (needs numpy)."""
        parked = self.by_status("simulated")
        if not parked:
            return
        print(f"  converting {len(parked)} parked games...")
        futures = {
            pool.submit(_convert_parked, (str(self.sim_dir / f"{gid}.json"),
                                          str(self.training_dir / f"{gid}.json"))): gid
            for gid in parked
        }
        for fut in as_completed(futures):
            gid = futures[fut]
            try:
                self.games[gid]["status"] = fut.result()
            except Exception as e:
                self.games[gid].update(status="error", reason=str(e))
                print(f"  error converting {gid}: {e}")
        self.save()

    async def run(self, pool: ProcessPoolExecutor, do_convert: bool) -> None:
        """Overlapped pipeline: metadata fetch and file downloads run as
        background asyncio tasks while the process pool simulates — the
        CPUs never wait for the network and vice versa."""
        from data_processing import download_replays
        from data_processing.download.api_client import fetch_replay_pages

        target = self.args.target
        workers = self.args.workers or os.cpu_count() or 4
        print(f"\n=== {self.player}: {self.trainable()}/{target} trainable ===")

        if do_convert:
            self.finish_parked(pool)

        # Retry transient errors from previous runs, then fresh candidates
        for gid in self.by_status("error"):
            self.games[gid]["status"] = "pending"
        pending = self.by_status("pending")           # metadata only
        ready: List[str] = []                         # replay file on disk
        for gid in list(pending):
            if (self.replays_dir / f"{gid}.json").exists():
                pending.remove(gid)
                ready.append(gid)

        pages = fetch_replay_pages(self.player, start_offset=self.offset)
        exhausted = False
        loop = asyncio.get_running_loop()
        in_flight: Dict[asyncio.Future, str] = {}
        dl_task: Optional[asyncio.Task] = None
        dirty = 0

        def need_more() -> bool:
            return self.trainable() + len(in_flight) + len(ready) + len(pending) < target

        async def fetch_page() -> None:
            nonlocal exhausted
            try:
                page = await anext(pages)
            except StopAsyncIteration:
                exhausted = True
                return
            except Exception as e:  # offline machine: finish what we have
                print(f"  fetch failed ({e}) — continuing without new games")
                exhausted = True
                return
            self.offset += len(page)
            for entry in self._passes(page):
                if entry["id"] not in self.games:
                    self.games[entry["id"]] = {"status": "pending", "meta": entry}
                    pending.append(entry["id"])
            print(f"  fetched {self.offset} games, candidates: {len(pending)}")

        async def download_batch() -> None:
            batch = pending[:2 * workers]
            del pending[:2 * workers]
            missing = [g for g in batch
                       if not (self.replays_dir / f"{g}.json").exists()]
            if missing:
                await download_replays(missing, str(self.replays_dir),
                                       max_concurrent=self.args.max_concurrent)
            for g in batch:
                if (self.replays_dir / f"{g}.json").exists():
                    ready.append(g)
                else:
                    self.games[g].update(status="error", reason="download failed")
                    print(f"  download failed: {g} (retried next run)")

        while True:
            # 1) candidate metadata: fetch ahead while more will be needed
            if not exhausted and need_more() and not pending and not ready:
                await fetch_page()
                continue

            # 2) downloads: run in the background, keep a 2×workers buffer
            if pending and len(ready) < 2 * workers and (dl_task is None or dl_task.done()):
                dl_task = asyncio.create_task(download_batch())

            # 3) CPUs: keep the pool full while the target is unmet
            while ready and len(in_flight) < workers \
                    and self.trainable() + len(in_flight) < target:
                gid = ready.pop()
                in_flight[loop.run_in_executor(
                    pool, _process_game, self._task(gid, do_convert))] = gid

            if not in_flight:
                if dl_task is not None and not dl_task.done():
                    await dl_task
                    continue
                if not exhausted and need_more():
                    continue  # go fetch another page
                break  # target met, or nothing left anywhere

            done, _ = await asyncio.wait(in_flight.keys(),
                                         return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                gid = in_flight.pop(fut)
                try:
                    self.games[gid]["status"] = fut.result()
                    self.games[gid].pop("reason", None)
                except ValueError as e:  # simulator rejection — permanent
                    self.games[gid].update(status="skipped", reason=str(e))
                    print(f"  skipped {gid}: {e}")
                except Exception as e:  # transient
                    self.games[gid].update(status="error", reason=str(e))
                    print(f"  error {gid}: {e}")
            dirty += len(done)
            if dirty >= 32:
                self.save()
                dirty = 0
                print(f"  trainable {self.trainable()}/{target}")

        if dl_task is not None and not dl_task.done():
            await dl_task
        self.save()
        if self.trainable() < target:
            print(f"  history exhausted: {self.trainable()}/{target} trainable")

        # filtered/<player>.json = exactly the trainable games (this is what
        # train_rl --filtered_file consumes for the RL map pool)
        trainable_meta = [g["meta"] for g in self.games.values()
                          if g["status"] in ("converted", "simulated")]
        with open(self.filtered_file, "w") as f:
            json.dump(trainable_meta, f, indent=1)

        n_parked = len(self.by_status("simulated"))
        counts = {}
        for g in self.games.values():
            counts[g["status"]] = counts.get(g["status"], 0) + 1
        print(f"  done: {counts}")
        if n_parked:
            print(f"  NOTE: {n_parked} games parked as 'simulated' — rerun the "
                  f"same command on a numpy-capable machine to convert them.")


async def async_main():
    p = argparse.ArgumentParser(
        description="Unified generals.io data pipeline (see module docstring)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--players", "-p", nargs="+", required=True,
                   help="Player usernames to collect")
    p.add_argument("--output", "-o", type=Path, required=True,
                   help="Output directory")
    p.add_argument("--target", "--max-replays", type=int, default=10000,
                   dest="target",
                   help="Trainable games per player (counts games that "
                        "survive filter AND simulation)")
    p.add_argument("--min-turns", type=int, default=80)
    p.add_argument("--min-opponent-stars", type=int, default=40)
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel simulate/convert workers (default: CPU count)")
    p.add_argument("--max-concurrent", type=int, default=None,
                   help="Concurrent replay downloads (default: CPU count)")
    args = p.parse_args()

    do_convert = _numpy_available()
    if not do_convert:
        print("numpy unavailable on this machine — games will park in the "
              "'simulated' state; rerun on a numpy machine to finish.")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for player in args.players:
            pipeline = PlayerPipeline(player, args.output, args)
            await pipeline.run(pool, do_convert)

    print("\nDone. Validate with: uv run python scripts/validate_dataset.py "
          f"{args.output}/training/<player>")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
