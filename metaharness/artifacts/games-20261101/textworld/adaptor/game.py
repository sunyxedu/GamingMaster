"""Tier 1, native TextWorld generic quests with Glulx and explicit assistance."""
from __future__ import annotations
import copy
import json
import math
from typing import Any
from metaharness.games.base import ActionSpec
from tw_runtime import Interpreter, PROFILES, resolve

TIER = 1
DETERMINISTIC = True
EVAL_KWARGS = {}
# Exact equal difficulty weights in every group. Three dev seeds are necessary
# to cover three profiles; the prompt's 1–2 dev suggestion cannot also do that.
SEEDS = {
    'train': list(range(100, 109)),
    'heldout': list(range(200, 209)),
    'dev': [300, 301, 302],
    'validation': list(range(400, 409)),
}

MEANINGS = {
    'look': 'Describe the current room; the noop, costing one submitted command.',
    'inventory': 'List carried objects, costing one submitted command.',
    'go': 'Move through the named exit.',
    'take': 'Pick up the named object, optionally from a container or supporter.',
    'drop': 'Put a carried object on the floor.',
    'open': 'Open a closed, unlocked container or door.',
    'close': 'Close an open container or door.',
    'lock': 'Lock a closed container or door using the named matching key.',
    'unlock': 'Unlock a container or door using the named matching key.',
    'put': 'Place a carried object on the named supporter.',
    'insert': 'Place a carried object into the named open container.',
    'eat': 'Consume a carried food item irreversibly; this can lose the quest.',
    'examine': 'Inspect the named visible object.',
}


class TextWorldGame:
    name = 'TextWorld generic Glulx quests (admissible commands)'

    def __init__(self):
        self._engine = None
        self._clear()

    def _clear(self):
        self._seed = None
        self._profile = PROFILES[0]
        self._manifest = None
        self._artifact = None
        self._state = {k: '' for k in ('feedback', 'description', 'inventory', 'objective')}
        self._state.update(admissible_commands=[], score=0.0, max_score=0.0,
                           won=False, lost=False, terminated=False, moves=0)
        self._commands = 0
        self._last_command = None
        self._last_delta = 0.0
        self._truncated = False
        self._invalid_reason = None
        self._protocol_error = None
        self._closed = False

    @property
    def action_spec(self):
        names = sorted(set(self._state['admissible_commands']) | {'look', 'inventory'})
        return ActionSpec(
            names=names,
            descriptions={n: MEANINGS.get(n.split()[0], 'Issue this exact native command.') for n in names},
            noop='look',
            notes='Names are the current concrete command strings; this list changes after each step. '
                  'Send one exact string from observation.admissible_commands, or look/inventory. '
                  'Admissibility is assistance, not a guarantee of safety. No command batching.')

    def _stop(self):
        if self._engine is not None:
            engine, self._engine = self._engine, None
            engine.close()

    def _fail(self, exc):
        self._invalid_reason = f'{type(exc).__name__}: {exc}'
        self._stop()

    def _accept(self, state):
        # Do not clamp broken scores or silently substitute missing engine fields.
        for key in ('score', 'max_score', 'returned_score'):
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid native {key}')
        if state['max_score'] <= 0 or not 0 <= state['score'] <= state['max_score']:
            raise ValueError('Native score outside [0, positive max_score]')
        if state['returned_score'] != state['score']:
            raise ValueError('Returned cumulative score disagrees with requested score')
        if state['max_score'] != self._manifest['acceptance']['max_score']:
            raise ValueError('Native max_score differs from accepted artifact')
        for key in ('won', 'lost', 'terminated'):
            if type(state[key]) is not bool:
                raise ValueError(f'Missing native boolean {key}')
        for key in ('feedback', 'description', 'inventory', 'objective'):
            if not isinstance(state[key], str):
                raise ValueError(f'Missing native text {key}')
        if not isinstance(state['admissible_commands'], list) or not all(
                isinstance(c, str) for c in state['admissible_commands']):
            raise ValueError('Missing native command list')
        if len(json.dumps(state, allow_nan=False).encode()) >= 190_000:
            raise ValueError('Observation too large')
        state = dict(state)
        state.pop('returned_score')
        self._last_delta = float(state['score'] - self._state['score'])
        self._state = state

    def reset(self, seed: int):
        self.close()
        self._clear()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2^32)')
        self._seed = seed
        self._profile = PROFILES[seed % 3]
        try:
            self._artifact, self._manifest = resolve(seed)
            self._engine = Interpreter(self._artifact)
            self._accept(self._engine.request('reset'))
            self._last_delta = 0.0
        except Exception as exc:
            self._fail(exc)
        return self.observe()

    def observe(self):
        observation = copy.deepcopy(self._state)
        observation.update(
            truncated=self._truncated, commands_used=self._commands,
            command_limit=self._profile['cap'],
            commands_remaining=max(0, self._profile['cap'] - self._commands),
            last_command=self._last_command, last_score_delta=self._last_delta,
            protocol_error=self._protocol_error,
            infrastructure_error=self._invalid_reason is not None,
        )
        return observation

    def step(self, action: Any):
        if self.done:
            return self.observe()
        self._protocol_error = None
        self._last_delta = 0.0
        if not isinstance(action, str):
            self._protocol_error = 'Action must be one command string; the game was not advanced.'
            return self.observe()
        self._commands += 1
        self._last_command = action[:512]
        # Exact allowlist blocks Inform debugging, undo/restart/save, multiple
        # commands, control characters, and parser disambiguation loops.
        if action not in self.action_spec.names:
            self._protocol_error = ('Command rejected. Use one exact current admissible command, '
                                    'look, or inventory. This attempt consumed one command.')
        else:
            try:
                self._accept(self._engine.request('step', command=action))
            except Exception as exc:
                self._fail(exc)
        if self._commands >= self._profile['cap'] and not self._state['terminated']:
            self._truncated = True
        if self.done:
            self._stop()
        return self.observe()

    @property
    def done(self):
        return bool(self._state['terminated'] or self._truncated or self._invalid_reason or self._closed)

    @property
    def score(self):
        maximum = self._state['max_score']
        return float(self._state['score'] / maximum) if maximum > 0 else 0.0

    def score_breakdown(self):
        return dict(normalized_score=self.score, native_score=float(self._state['score']),
                    native_max_score=float(self._state['max_score']))

    def info(self):
        return dict(seed=self._seed, profile=self._profile['name'],
                    native_score=self._state['score'], native_max_score=self._state['max_score'],
                    won=self._state['won'], lost=self._state['lost'],
                    terminated=self._state['terminated'], truncated=self._truncated,
                    commands_used=self._commands, valid_trial=self._invalid_reason is None,
                    invalid_reason=self._invalid_reason,
                    artifact=str(self._artifact) if self._artifact else None,
                    generation_seed=self._manifest['generation_seed'] if self._manifest else None)

    def describe(self):
        return (
            'Play a native TextWorld generated text adventure, one command per turn. '
            'Achieve the objective shown in the observation. Explore rooms, move objects and '
            'operate containers/doors with matching keys. Eating is irreversible and can make '
            'a goal impossible. There are no opponents and waiting in real time changes nothing. '
            'Send one exact string from the current admissible_commands list; look and inventory '
            'are also always accepted. Names in action_spec are these concrete strings and change '
            'with state. Grammar: look; inventory; go <north|south|east|west>; examine <object>; '
            'take <object> [from <container/supporter>]; drop <object>; open <target>; close <target>; '
            'unlock <target> with <key>; lock <target> with <key>; put <object> on <supporter>; '
            'insert <object> into <container>; eat <food>. Preserve exact names and spelling. '
            'No multi-command strings, undo, restart, debug or save/load. Every string attempt '
            'costs one command, including look/inventory, invalid strings and noop (look). '
            'Malformed non-string payloads are rejected without advancing or using the budget. '
            'Easy/medium/hard quests have 3/5/10 rooms, 5/10/20 objects, generation lengths '
            '2/5/10 and command caps 100/200/400. Seeds vary the world and text. '
            'Observation fields: feedback is the latest native response; description, inventory '
            'and objective are current native text. admissible_commands lists currently executable '
            'interactions, an explicitly allowed assistance level that does not imply safety. '
            'score and max_score are native cumulative values; won/lost/terminated are native '
            'outcomes, truncated means the command cap was reached. moves is the engine count; '
            'commands_used/command_limit/commands_remaining track the separate budget. '
            'last_command and last_score_delta describe the previous submission; protocol_error '
            'explains rejected input; infrastructure_error means the trial must be retried. '
            'The evaluator score is final cumulative native score / positive max_score, preserving '
            'any partial credit on loss or truncation. No score for speed or exploration. '
            'No hidden facts, walkthroughs, future policy commands or intermediate rewards are exposed. '
            'After termination further steps have no effect.'
        )

    def close(self):
        self._stop()
        self._closed = True


def make_game(**kwargs):
    if kwargs:
        raise TypeError(f'Unknown TextWorld configuration: {sorted(kwargs)}')
    return TextWorldGame()
