"""Fault-injection checks for native-search and worker deadline enforcement."""
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time

import game
import foulplay_worker as worker


def stalled_search(battle, connection, seed):
    if battle == 'partial':
        connection.send({'policy': {'tackle': 1.0}, 'visits': 1000})
    time.sleep(30)


def main():
    original = worker.search
    worker.search = stalled_search
    timings = {}
    try:
        for mode in ('partial', 'empty'):
            start = time.monotonic()
            try:
                result = worker.bounded_search(mode, start + 0.2)
            except RuntimeError as exc:
                assert mode == 'empty' and 'without a completed MCTS sample' in str(exc)
            else:
                assert mode == 'partial' and result == ({'tackle': 1.0}, 1000, True)
            timings[mode] = time.monotonic() - start
            assert timings[mode] < 1.0
            assert not multiprocessing.active_children()
    finally:
        worker.search = original

    g = game.make_game()
    try:
        obs = g.reset(game.SEEDS['dev'][0])
        assert g.info()['valid_trial'], g.info()
        node_pid = g._process.pid
        children = Path(f'/proc/{node_pid}/task/{node_pid}/children').read_text().split()
        assert len(children) == 1, children
        worker_pid = int(children[0])
        # Suspend the whole worker, including its parser, to exercise Node's
        # independent watchdog rather than a cooperative Python timer.
        os.kill(worker_pid, signal.SIGSTOP)
        start = time.monotonic()
        g.step(obs['legal_actions'][0])
        timings['worker_watchdog'] = time.monotonic() - start
        assert g.done and not g.info()['valid_trial'], g.info()
        assert '3-second decision watchdog' in g.info()['invalid_reason'], g.info()
        assert timings['worker_watchdog'] < 6.0
        assert g._process is None
        assert not Path(f'/proc/{node_pid}').exists()
        assert not Path(f'/proc/{worker_pid}').exists()
    finally:
        g.close()
        g.close()
    report = {'passed': True, 'seconds': timings,
              'partial_search_retained': True, 'empty_search_invalidates': True,
              'worker_stall_invalidates': True, 'children_reaped': True}
    (game.HERE / 'runtime/deadline-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
