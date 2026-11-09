"""Tier 1 offline Yu-Gi-Oh! adaptor, backed by isolated EDOPro native duels."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
from metaharness.games.base import ActionSpec

ROOT=Path(__file__).resolve().parent
TIER=1
DETERMINISTIC=True
SEEDS={'train':[1101,1102,1103,1104,1105,1106],'heldout':[2201,2202,2203,2204,2205,2206],
       'dev':[3301,3302],'validation':[4401,4402,4403,4404,4405,4406]}
EVAL_KWARGS={}
_VERIFIED=None

def verify_assets():
    global _VERIFIED
    lock=ROOT/'assets.lock.json'
    manifest=json.loads(lock.read_text())
    # Recheck changed files on every construction, reuse hashes of unchanged files.
    signature=tuple((name,(ROOT/name).stat().st_size,(ROOT/name).stat().st_mtime_ns) for name in manifest['files'])
    marker=hashlib.sha256(lock.read_bytes()).hexdigest(),signature
    if marker!=_VERIFIED:
        for name,expected in manifest['files'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Pinned asset checksum mismatch: {name}')
        _VERIFIED=marker

class YugiohGame:
    name='Yu-Gi-Oh! EDOPro — offline three-duel benchmark'
    action_spec=ActionSpec(names=['choose','select','order','noop'],descriptions={
        'choose':'choose TOKEN INDEX: submit the exact action string from prompt.choices.',
        'select':'select TOKEN I,J,...: choose a legal subset of prompt.candidates; - means empty.',
        'order':'order TOKEN I,J,...: order all candidates, first to last.',
        'noop':'Leave the current decision unchanged; the response deadline still runs.'},noop='noop',
        notes='One step answers one engine prompt, including responses on the opponent turn. Tokens reject stale choices. Bare parameterized action names are harmless invalid actions.')
    def __init__(self,decision_timeout_s=None,max_turns=200,max_decisions=10000,worker_timeout_s=30):
        verify_assets()
        self.timeout=float(decision_timeout_s if decision_timeout_s is not None else os.environ.get('YGO_DECISION_TIMEOUT_S','300'))
        self.worker_timeout=float(worker_timeout_s)
        if not math.isfinite(self.timeout) or self.timeout<=0:raise ValueError('decision_timeout_s must be positive and finite')
        if not math.isfinite(self.worker_timeout) or self.worker_timeout<=0:raise ValueError('worker_timeout_s must be positive and finite')
        if type(max_turns)!=int or type(max_decisions)!=int or max_turns<1 or max_decisions<1:raise ValueError('caps must be positive integers')
        self.limits={'max_turns':max_turns,'max_decisions':max_decisions}
        self.proc=None;self.log=None;self.buffer=b'';self.deadline=None
        self.state={'observation':{'done':False,'status':'call reset(seed)'},'done':False,'score':0.0,'info':{'valid_trial':True}}
    def _stop(self):
        if self.proc:
            p=self.proc;self.proc=None
            if p.poll() is None:
                try:p.stdin.write(b'{"op":"close"}\n');p.stdin.flush();p.wait(timeout=2)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                    p.terminate()
                    try:p.wait(timeout=2)
                    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2)
            for stream in (p.stdin,p.stdout):
                if stream:stream.close()
        if self.log:self.log.close();self.log=None
    def _failure(self,error):
        self.state['done']=True;self.state['score']=0.0
        self.state['observation'].update(done=True,prompt=None,status='technical error; trial invalid')
        self.state['info'].update(valid_trial=False,invalid_reason=str(error),technical_error=True)
        self.deadline=None;self._stop()
    def _rpc(self,command):
        try:
            if not self.proc or self.proc.poll() is not None:raise RuntimeError('Native worker exited')
            self.proc.stdin.write(json.dumps(command,allow_nan=False,separators=(',',':')).encode()+b'\n');self.proc.stdin.flush()
            end=time.monotonic()+self.worker_timeout
            while b'\n' not in self.buffer:
                remain=end-time.monotonic()
                if remain<=0:raise TimeoutError('Native worker watchdog expired')
                ready,_,_=select.select([self.proc.stdout],[],[],remain)
                if not ready:raise TimeoutError('Native worker watchdog expired')
                chunk=os.read(self.proc.stdout.fileno(),65536)
                if not chunk:raise RuntimeError(f'Native worker disconnected (exit {self.proc.poll()})')
                self.buffer+=chunk
                if len(self.buffer)>2000000:raise RuntimeError('Worker response exceeds limit')
            line,self.buffer=self.buffer.split(b'\n',1);result=json.loads(line)
            if 'error' in result:raise RuntimeError(result['error'])
            if len(json.dumps(result['observation'],allow_nan=False).encode())>=200000:raise RuntimeError('Observation exceeds 200 kB')
            self.state=result
        except Exception as e:self._failure(e)
    def _arm(self):
        self.deadline=None if self.state['done'] else time.monotonic()+self.timeout
    def _check_timeout(self):
        if self.deadline and time.monotonic()>self.deadline and not self.state['done']:
            self._rpc({'op':'forfeit'});self._arm()
    def reset(self,seed):
        if type(seed)!=int:raise ValueError('seed must be an integer')
        self._stop();self.buffer=b''
        self.state={'observation':{'done':False},'done':False,'score':0.0,'info':{'valid_trial':True}}
        (ROOT/'runtime/logs').mkdir(parents=True,exist_ok=True)
        self.log=tempfile.TemporaryFile(dir=ROOT/'runtime/logs')
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env.pop('DISPLAY',None)
        self.proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py')],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env=env,bufsize=0)
        self._rpc({'op':'reset','seed':seed,'limits':self.limits});self._arm()
        return self.observe()
    def observe(self):
        self._check_timeout();return json.loads(json.dumps(self.state['observation'],allow_nan=False))
    def step(self,action):
        self._check_timeout()
        if self.state['done'] or action=='noop':return self.observe()
        if not isinstance(action,str) or len(action)>8192:
            self.state['observation']['last_action_error']='Action must be a string of at most 8192 characters';return self.observe()
        token=(self.state['observation'].get('prompt') or {}).get('token')
        self._rpc({'op':'step','action':action})
        new_token=(self.state['observation'].get('prompt') or {}).get('token')
        if token!=new_token:self._arm()
        return self.observe()
    @property
    def done(self):self._check_timeout();return bool(self.state['done'])
    @property
    def score(self):self._check_timeout();return float(self.state['score'])
    def info(self):
        self._check_timeout();out=json.loads(json.dumps(self.state['info']))
        out['decision_timeout_s']=self.timeout;out['limits']=self.limits.copy()
        return out
    def score_breakdown(self):
        results=self.info().get('results',[]);native=[r for r in results if r['kind']=='native'];scored=[r for r in results if r['kind'] in ('native','forfeit')]
        return {'score':self.score,'completed_duels':float(len(results)),'native_completed':float(len(native)),
                'win_rate':sum(r['win'] for r in scored)/len(scored) if scored else 0.0,
                'outcome_component':sum(1 if r['outcome']=='win' else -1 if r['outcome']=='loss' else 0 for r in scored)/3,
                'lp_component':sum(0.1*max(-1,min(1,(r['agent_lp']-r['opponent_lp'])/8000)) for r in scored)/3,
                'wins':float(sum(r['win'] for r in native)),'losses':float(sum(r['native_result']=='loss' for r in native)),
                'draws':float(sum(r['native_result']=='draw' for r in native)),
                'truncations':float(sum(r['truncated'] for r in results)),'forfeits':float(sum(r['forfeit'] for r in results)),
                'technical_error':float(not self.state['info'].get('valid_trial',True))}
    def describe(self):
        recipes=json.loads((ROOT/'decks.json').read_text());from collections import Counter
        pool='\nPublic fixed deck pool:\n'+'\n'.join(name+': '+', '.join(str(n)+'x '+recipes['cards'][str(c)]['name'] for c,n in Counter(deck).items()) for name,deck in recipes['decks'].items())
        return '''Play three independent Yu-Gi-Oh! duels using the real EDOPro core (Master Rule 5), against a seeded random-legal opponent. Each duel starts at 8000 LP, five cards, empty Extra/Side Decks. The first player skips the first draw and cannot battle on turn one. This is a custom three-copy format with no tournament banlist. Normal/Tribute Summons, setting, positions, battle, Flip effects, targeting, costs and chains follow the card text and native rules. Reduce opposing LP to zero or win by another native rule (including opponent deck-out). Chain effects normally resolve last-in-first-out; destroying a source does not by itself negate its effect.
Each step answers one decision, not one entire turn. Answer even on the opponent turn. Copy an action string from prompt.choices: "choose TOKEN INDEX". For a subset prompt use "select TOKEN I,J,..." with distinct candidate indices satisfying the displayed constraint; use "-" for empty. Tribute selections require summed weights >= minimum and count <= maximum. A choice may explicitly cancel/finish. For an order prompt use "order TOKEN I,J,..." containing all candidate indices in desired order, first chain link / first deck position; chain links resolve in reverse order. Mandatory singleton choices advance automatically. "noop" waits; invalid or stale actions leave the decision pending and report last_action_error. Tokens identify the current round and prompt. Never invent native responses.
Observations include your seat; turn/phase; both players' LP, zone counts and public field/graveyard; your hand and face-down cards; current stats, positions, targets and equips; current chain; the last 40 public/own events; and the prompt with legal choices or exact constraints. card_catalog gives local text and base statistics for all 24 pool cards (type/race/attribute are engine bitmasks). Unknown cards have no identity. Opponent hand contents, face-down identities, deck order, seed and RNG state are private. Slot indices identify current zones, not permanent card identities. Own monster fields include current attack/defense. Effect hints explain targets and decisions. The deck pool is normal_pressure, flip_control and tribute_pressure; each has 24 monsters and the identical 16-card Spell/Trap package. The full public recipes are available in decks.json; no modern/Extra Deck cards are used.
After each duel the next starts automatically; completed_duels preserves outcomes. Final score is the mean of three duel scores: win=1, loss=-1, true draw=0, plus 0.1 times LP difference/8000 clipped to [-1,1]. Native winner overrides LP. A 200-turn or 10000-engine-decision cap is a benchmark truncation scoring zero. Each pending agent decision has a 300-second default response deadline, including retries; exceeding it forfeits that duel. Engine/opponent processing is excluded. Technical faults invalidate the trial. Win rate and per-matchup/seat results are recorded separately for evaluation.'''+pool
    def close(self):self.deadline=None;self._stop()

def make_game(**kwargs):return YugiohGame(**kwargs)
