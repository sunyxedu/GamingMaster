"""Tier-1 Dominion adaptor backed by a separate offline pyminion process."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile

from metaharness.games.base import ActionSpec
from scenarios import SEEDS, SCENARIOS, ENGINE_COMMIT, ADAPTOR_VERSION

TIER = 1
DETERMINISTIC = True
EVAL_KWARGS = {}
ROOT = Path(__file__).resolve().parent


class Dominion:
    name = 'Dominion (offline pyminion)'
    action_spec = ActionSpec(
        names=['choose', 'pass', 'wait'],
        descriptions={
            'choose': 'choose DECISION_ID OPTION_ID ... : select distinct prompt options in order.',
            'pass': 'Decline an optional prompt (min=0); otherwise report an illegal action.',
            'wait': 'Leave the pending decision and game state unchanged.',
        }, noop='wait',
        notes='One step answers one decision, including reactions during the opponent turn. '
              'Use prompt option IDs, not card names. choose ID with no options means pass. '
              'JSON {"decision_id": ID, "option_ids": ["0", ...]} is also accepted.')

    def __init__(self, max_turns=400, max_decisions=20000):
        if type(max_turns) is not int or not 1 <= max_turns <= 10000:
            raise ValueError('max_turns must be an integer in 1..10000')
        if type(max_decisions) is not int or not 1 <= max_decisions <= 100000:
            raise ValueError('max_decisions must be an integer in 1..100000')
        self.max_turns = max_turns
        self.max_decisions = max_decisions
        self._process = None
        self._trace = None
        self._stderr = None
        self._packet = {'observation': {}, 'info': {}, 'done': False}
        self.illegal_action_count = 0
        self.fallback_count = 0
        self._error = None
        self._started = False

    def reset(self, seed):
        self.close()
        if type(seed) is not int or seed not in SCENARIOS:
            raise ValueError('Unsupported seed: use the declared SEEDS manifest.')
        self.illegal_action_count = self.fallback_count = 0
        self._error = None
        self._packet = {'observation': {}, 'info': {}, 'done': False}
        config = dict(SCENARIOS[seed], seed=seed, max_turns=self.max_turns,
                      max_decisions=self.max_decisions)
        runtime = ROOT / 'runtime'
        runtime.mkdir(exist_ok=True)
        self._run_dir = Path(tempfile.mkdtemp(prefix='episode-', dir=runtime))
        self._trace = (self._run_dir / 'replay.jsonl').open('w')
        self._trace.write(json.dumps(dict(config=config, engine_commit=ENGINE_COMMIT,
                                          adaptor_version=ADAPTOR_VERSION)) + '\n')
        self._trace.flush()
        self._stderr = (self._run_dir / 'stderr.log').open('w')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
        self._process = subprocess.Popen(
            [sys.executable, '-u', str(ROOT / 'worker.py')], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._stderr, cwd=ROOT, env=env,
            text=True, bufsize=1)
        self._started = True
        self._process.stdin.write(json.dumps(config) + '\n')
        self._process.stdin.flush()
        self._receive()
        return self.observe()

    def _receive(self):
        try:
            if not select.select([self._process.stdout], [], [], 30)[0]:
                raise TimeoutError('engine did not reach a decision within 30 seconds')
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError('engine exited before returning a decision')
            packet = json.loads(line)
            if 'error' in packet:
                raise RuntimeError(packet['error'])
            json.dumps(packet['observation'], allow_nan=False)
            if len(line.encode()) >= 190000:
                raise RuntimeError('engine observation exceeded size budget')
            self._packet = packet
            if packet['observation']['decision_id'] > self.max_decisions and not packet['done']:
                self._packet['done'] = True
                self._packet['info'].update(truncated=True, outcome='truncated', win_score=0.0,
                                            end_reason='decision_limit')
                self._packet['observation']['prompt'] = None
                self._packet['observation']['terminal'] = self.score_report()
                self._stop_process()
        except Exception as exc:
            self._packet['done'] = True
            self._packet['info'].update(valid_trial=False, invalid_reason=str(exc)[:4000],
                                        outcome='invalid', win_score=0.0, end_reason='engine_failure')
            self._packet['observation']['prompt'] = None
            self._packet['observation']['terminal'] = {'outcome': 'invalid', 'end_reason': 'engine_failure'}
            self._stop_process()

    def step(self, action):
        if not self._started or self.done:
            return self.observe()
        if isinstance(action, str) and action.strip() == 'wait':
            return self.observe()
        prompt = self._packet['observation']['prompt']
        decision = self._packet['observation']['decision_id']
        try:
            if isinstance(action, str):
                parts = action.split()
                if parts == ['pass']:
                    reply = {'decision_id': decision, 'option_ids': []}
                elif len(parts) >= 2 and parts[0] == 'choose':
                    reply = {'decision_id': int(parts[1]), 'option_ids': parts[2:]}
                else:
                    raise ValueError('Expected choose DECISION_ID OPTION_ID ..., pass, or wait.')
            elif isinstance(action, dict):
                reply = action
            else:
                raise ValueError('Action must be a string or a decision object.')
            if type(reply.get('decision_id')) is not int or reply['decision_id'] != decision:
                raise ValueError('Stale or missing decision_id.')
            ids = reply.get('option_ids')
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                raise ValueError('option_ids must be a list of strings.')
            if not prompt['min'] <= len(ids) <= prompt['max']:
                raise ValueError(f"Choose {prompt['min']}..{prompt['max']} options.")
            if len(ids) != len(set(ids)):
                raise ValueError('An option instance cannot be selected twice.')
            allowed = {o['id'] for o in prompt['options']}
            if any(i not in allowed for i in ids):
                raise ValueError('Option not present in the pending prompt.')
        except (ValueError, TypeError, KeyError) as exc:
            self.illegal_action_count += 1
            self._error = str(exc)
            return self.observe()
        self._error = None
        reply = {'decision_id': decision, 'option_ids': ids}
        self._trace.write(json.dumps(reply) + '\n')
        self._trace.flush()
        try:
            self._process.stdin.write(json.dumps(reply) + '\n')
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._packet['done'] = True
            self._packet['info'].update(valid_trial=False, invalid_reason=str(exc), outcome='invalid',
                                        win_score=0.0, end_reason='engine_failure')
            self._packet['observation']['prompt'] = None
            self._packet['observation']['terminal'] = {'outcome': 'invalid', 'end_reason': 'engine_failure'}
            self._stop_process()
            return self.observe()
        self._receive()
        return self.observe()

    def observe(self):
        obs = copy.deepcopy(self._packet['observation'])
        obs['last_error'] = self._error
        obs['illegal_action_count'] = self.illegal_action_count
        obs['fallback_count'] = self.fallback_count
        return obs

    @property
    def done(self):
        return bool(self._packet['done'])

    @property
    def score(self):
        return float(self._packet['info'].get('vp_margin', 0))

    def score_report(self):
        keys = ('agent_vp', 'opponent_vp', 'vp_margin', 'win_score', 'outcome',
                'turns_taken', 'end_reason', 'truncated')
        return {k: self._packet['info'].get(k) for k in keys}

    def score_breakdown(self):
        info = self.info()
        return {k: float(info.get(k, 0)) for k in
                ('agent_vp', 'opponent_vp', 'vp_margin', 'win_score', 'truncated',
                 'illegal_action_count', 'fallback_count')}

    def info(self):
        result = copy.deepcopy(self._packet['info'])
        result.update(illegal_action_count=self.illegal_action_count, fallback_count=self.fallback_count)
        if self._started:
            result['replay_path'] = str(self._run_dir / 'replay.jsonl')
        return result

    def describe(self):
        return '''Play standard two-player Dominion, Base second-edition cards, against the disclosed
BM-1, SM-1 or WM-1 opponent. Each player starts with 7 Copper and 3 Estate, shuffled,
and draws five cards. Seat 0 acts first. Each turn starts with 1 Action, 1 Buy, $0.
Playing an Action spends one Action and resolves all its text; +Actions enable further plays.
Copper/Silver/Gold produce $1/$2/$3 and are automatically all played after Actions.
Buying costs coins and a Buy and normally gains to discard. Gains from Actions do not spend
coins or Buys. Cleanup discards hand and played cards and draws five; shuffle discard only
when the draw deck is exhausted. Attacks may ask you to discard or reveal Moat during an
enemy turn. Moat protects only when in hand and revealed for that attack.
Finish the entire turn before checking for an empty Province pile or any three empty piles.
Most VP wins; equal VP favors fewer turns taken, otherwise draw. All owned cards count,
including Curses (-1 each) and Gardens (floor(total cards/10) per copy). Objective: win and
maximize final VP margin; score is the unweighted VP margin, win_score is reported separately.
Supply has 8 of each Victory (including Gardens), 10 other Kingdom cards, 10 Curses,
46 Copper, 40 Silver, 30 Gold after starting decks. Exact card costs/types/text are in supply.

Each step answers one pending prompt, then advances to your next choice or game end.
Use `choose DECISION_ID OPTION_ID ...`, selecting distinct option IDs within prompt min/max;
`choose DECISION_ID` or `pass` declines only if min=0. `wait` is a true no-op.
Alternatively send {"decision_id": integer, "option_ids": ["0", ...]}.
IDs identify indistinguishable card occurrences locally to this prompt, not persistent cards.
False/True choices are explicit labeled options. Sentry inspected cards are listed top first;
its final True swaps the two remaining cards. Forced sole responses resolve automatically.
Invalid/stale actions leave the game unchanged and set last_error; there is no automatic fallback.

Observations: viewer and active/decision player, seat, phase, per-player turns STARTED,
active player's resources, kingdom/opponent IDs, exact supply with rules, own hand/discard
multisets, public played cards, hand/deck/discard counts, public ownership and exact known VP,
trash, ordered public_log_delta since the last decision, prompt, derived pile/VP arithmetic,
and terminal result. Repeated observe() is stable; log deltas are not consumed by observe().
Opponent hand, both unrevealed deck orders and RNG/seed are hidden. episode_id is a constant
non-secret label; decision_id increases within an episode. Do not reuse responses across resets.
BM-1 buys Province, then Duchy at <=4 Provinces, Estate at <=2, then Gold, Silver, else passes.
SM-1 inserts one affordable Smithy before Gold/Silver while >=5 Provinces remain. WM-1 instead
inserts one Witch while >=5 Provinces and some Curses remain. Both play and retain their one
terminal when possible. Opponents otherwise retain highest Treasure money then VP on discards.
Safety limits end as truncated, never as a Dominion victory. There is no decision time limit.'''

    def _stop_process(self):
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            else:
                process.wait()
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except (BrokenPipeError, OSError):
                        pass
            self._process = None

    def close(self):
        self._stop_process()
        for name in ('_trace', '_stderr'):
            stream = getattr(self, name, None)
            if stream:
                stream.close()
            setattr(self, name, None)
        self._started = False


def make_game(**kwargs):
    return Dominion(**kwargs)
