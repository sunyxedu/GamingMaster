"""Minecraft Java 1.21.1 real-time adaptor. Policy has no console access."""
from __future__ import annotations
import atexit, collections, copy, hashlib, json, math, os, pathlib, re, shutil, socket, subprocess, sys, threading, time, uuid, weakref
from metaharness.games.base import ActionSpec
from scoring import Score, parse_inventory

TIER=2
DETERMINISTIC=False
ROOT=pathlib.Path(__file__).resolve().parent
CONFIG='vanilla-1.21.1-natural-dragon-v1'
GAMERULES={'keepInventory':'false','doMobSpawning':'true','doDaylightCycle':'true','doWeatherCycle':'true','naturalRegeneration':'true','doImmediateRespawn':'false','doFireTick':'true','mobGriefing':'true','doTileDrops':'true','doEntityDrops':'true','doInsomnia':'true','doPatrolSpawning':'true','doTraderSpawning':'true','doWardenSpawning':'true','randomTickSpeed':'3','spawnRadius':'0','spawnChunkRadius':'2','announceAdvancements':'true','reducedDebugInfo':'false','sendCommandFeedback':'true'}
_INSTANCES=weakref.WeakSet()

class InfrastructureError(RuntimeError): pass

def seed_config(seed):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    # Stable domain-separated 64-bit mapping; JSON carries the long as decimal text.
    digest = hashlib.sha256(f'minecraft:1.21.1:world:{seed}'.encode('ascii')).digest()
    world_seed = int.from_bytes(digest[:8], 'big', signed=True)
    if world_seed == 0:
        world_seed = 1
    seed_property = str(world_seed)
    return {'seed': seed, 'world_seed': str(world_seed), 'seed_property': seed_property,
            'time_of_day': 1000, 'config': CONFIG, 'difficulty': 'normal',
            'gamemode': 'survival', 'gamerules': dict(GAMERULES)}

ACTIONS={
 'noop':'Continue an existing bounded job; otherwise stay still and release shield.',
 'cancel':'Immediately stop the current job, navigation, digging, item use and window.',
 'navigate':'navigate CANDIDATE_ID: travel to a listed stand or block interaction position.',
 'look_at':'look_at X Y Z: look at a nearby point, avoiding known Enderman gaze.',
 'gather':'gather BLOCK_ID [N=1]: mine and collect 1–4 exposed blocks of that type, choosing a held tool.',
 'equip':'equip ITEM [DEST=hand]: equip inventory item to hand/off-hand/head/torso/legs/feet.',
 'craft':'craft ITEM NEW_COUNT [TABLE_ID]: craft at least NEW_COUNT new outputs from inventory.',
 'place':'place ITEM REFERENCE_BLOCK_ID FACE: place on up/down/east/west/north/south face.',
 'furnace_start':'furnace_start FURNACE_ID INPUT COUNT FUEL COUNT: load an actual furnace, then close it.',
 'furnace_take':'furnace_take FURNACE_ID: retrieve finished output, if any.',
 'eat':'eat ITEM: consume held food when hungry.',
 'attack':'attack ENTITY_ID: one visible in-range melee attack, respecting cooldown.',
 'shield':'shield on|off: equip/use shield in the off-hand or release it.',
 'wait':'wait [SECONDS=0.5]: wait 0–5 seconds without pausing the world.',
 'controls':'controls forward,jump|none: hold listed movement keys; renew each tick; expires after 0.8s.',
 'goto':'goto X Y Z: path to loaded safe feet position within 48 blocks; no automatic mining/bridging.',
 'dig':'dig X Y Z: mine one visible in-reach block with the held tool, without pickup automation.',
 'activate_block':'activate_block BLOCK_ID [FACE=up]: right-click block with held item (doors, bucket, flint, eyes).',
 'activate_entity':'activate_entity ENTITY_ID: ordinary right-click a visible entity within 4 blocks.',
 'use_item':'use_item [SECONDS=0.2] [hand|off-hand]: hold then release use (food, bow, eye, bucket).',
 'hotbar':'hotbar SLOT: select a hotbar slot 0–8.',
 'open_container':'open_container BLOCK_ID: open a nearby chest/barrel/container and retain its window.',
 'close_window':'Close the current container window.',
 'transfer':'transfer take|put ITEM COUNT: move 1–64 real items between open container and inventory.',
 'move_slot':'move_slot SOURCE DEST: move a real stack between window slot numbers.',
 'toss':'toss ITEM COUNT: drop 1–64 carried items (including gold for piglin bartering).',
 'sleep':'sleep BED_ID: use an Overworld bed when vanilla permits; set ordinary respawn point.',
 'wake':'Leave the bed.',
 'respawn':'Respawn normally after death; inventory remains dropped, no restoration.',
 'inspect_trades':'inspect_trades VILLAGER_ID: open nearby villager; report its offered trades.',
 'trade':'trade VILLAGER_ID INDEX [TIMES=1]: perform a real offered trade with carried payment.',
 'enchant':'enchant TABLE_ID ITEM CHOICE: enchant carried item with lapis/XP, choice 0–2.',
 'chat':'chat TEXT: normal player chat only; slash commands/control characters are forbidden.'}


class MinecraftGame:
    name='Minecraft Java 1.21.1 — Survival to Free the End'
    tick_s=.5
    action_spec=ActionSpec(names=list(ACTIONS),descriptions=ACTIONS,noop='noop',notes='Actions are whitespace-separated strings. Bare parameterized names are rejected as feedback, not Python exceptions. Only one bounded job owns movement/hands. Repeated identical current actions do not restart; send noop between deliberate repeats. noop preserves an existing job. cancel before replacing a busy job.')
    def __init__(self, **kwargs):
        allowed={'run_dir','port','username','max_seconds','objective','target','target_count'}
        unknown=set(kwargs)-allowed
        if unknown: raise TypeError(f'Unknown options: {sorted(unknown)}')
        self.run=pathlib.Path(kwargs.get('run_dir',os.environ.get('MC_RUN_DIR',ROOT/'runtime'))).resolve()
        if not self.run.is_relative_to(ROOT/'runtime'): raise ValueError('run_dir must be beneath adaptor/runtime')
        if os.environ.get('MC_HOST','127.0.0.1')!='127.0.0.1': raise ValueError('MC_HOST must be 127.0.0.1')
        if os.environ.get('MC_VERSION','1.21.1')!='1.21.1': raise ValueError('Only Minecraft 1.21.1 is validated')
        self.username=str(kwargs.get('username',os.environ.get('MC_USERNAME','HarnessBot')))
        if not re.fullmatch(r'[A-Za-z0-9_]{1,16}',self.username): raise ValueError('Invalid offline username')
        self.fixed_port=kwargs.get('port',os.environ.get('MC_PORT'))
        if os.environ.get('MC_AUTH','offline')!='offline': raise ValueError('MC_AUTH must be offline')
        self.duration=float(kwargs.get('max_seconds',3600))
        if not math.isfinite(self.duration) or not 1<=self.duration<=86400: raise ValueError('max_seconds must be 1..86400')
        self.objective=kwargs.get('objective','dragon')
        if self.objective not in ('dragon','advancement','resource'): raise ValueError('objective must be dragon, advancement, or resource')
        self.target=kwargs.get('target')
        if self.objective!='dragon' and (not isinstance(self.target,str) or not re.fullmatch(r'[a-z0-9_:/.-]+',self.target)): raise ValueError('target required for specialized objective')
        self.target_count=int(kwargs.get('target_count',1))
        if self.target_count<1: raise ValueError('target_count must be positive')
        self._initial_server_age=None
        self._last_snapshot_seq=0
        self._lock=threading.RLock();self._condition=threading.Condition(self._lock);self._write_lock=threading.Lock();self._console_lock=threading.Lock()
        self._server=None;self._bridge=None;self._server_ready=threading.Event();self._bridge_ready=threading.Event()
        self._closing=True;self._active=False;self._terminal=None;self._invalid=None;self._ticks=0;self._start=None;self._end=None;self._snapshot={};self._score=Score();self._request_id=0;self._responses={};self._audit_queue=collections.deque();self._audit_serial=0;self._last_audit=0.;self._last_message=0.;self._snapshot_received=0.;self._last_action=None;self._action_feedback=None;self._failures=[];self._server_lines=collections.deque(maxlen=400);self._monitor_stop=threading.Event();self._journal=None;self._scenario=None;self._episode=None;self._logfiles=[];self._finish_thread=None;self._final_health=None;self._shutdown_lock=threading.Lock();self._clean_shutdown=True;self._lag_warnings=0
        _INSTANCES.add(self)
    def _artifacts(self):
        path=self.run/'artifacts.json'
        if not path.exists():
            path=ROOT/'runtime/artifacts.json'
        if not path.exists(): raise InfrastructureError('Run ./setup.sh first; runtime/artifacts.json is missing')
        artifacts=json.loads(path.read_text())
        self.java=os.environ.get('JAVA_BIN',artifacts['java']);self.node=os.environ.get('NODE_BIN','node')
        jar=pathlib.Path(os.environ.get('MC_SERVER_JAR',artifacts['server_jar']))
        if not jar.is_absolute():jar=ROOT/jar
        jar=jar.resolve()
        if not jar.is_relative_to(ROOT/'runtime'):raise InfrastructureError('Server JAR must be beneath runtime/')
        self.jar=str(jar)
        with jar.open('rb') as stream:
            checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        if checksum!='e3bc55693e93cda0188f2e60aea28113fc647c5e85a15fa3d1b347349231b4bb':raise InfrastructureError('Pinned Minecraft 1.21.1 server checksum mismatch')
    def _port(self):
        if self.fixed_port:return int(self.fixed_port)
        with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
    def _properties(self,directory,scenario,port):
        props={'server-ip':'127.0.0.1','server-port':port,'online-mode':'false','enforce-secure-profile':'false','gamemode':'survival','force-gamemode':'true','difficulty':'normal','hardcore':'false','max-players':1,'pvp':'false','spawn-protection':0,'level-name':'world','level-seed':scenario['seed_property'],'level-type':'minecraft:normal','generate-structures':'true','view-distance':6,'simulation-distance':5,'enable-rcon':'false','enable-query':'false','enable-command-block':'false','white-list':'false','player-idle-timeout':0,'allow-flight':'false','allow-nether':'true','sync-chunk-writes':'true','max-tick-time':60000,'broadcast-console-to-ops':'false','network-compression-threshold':256}
        (directory/'server.properties').write_text(''.join(f'{k}={v}\n' for k,v in props.items()))
        (directory/'eula.txt').write_text('eula=true\n')
    def _launch(self,directory,scenario):
        directory.mkdir(parents=True,exist_ok=True);self._process_dir=directory;self._closing=False;self._server_ready.clear();self._bridge_ready.clear();self._snapshot={};self._audit_queue.clear();self._responses.clear();self._server_lines.clear();self._last_audit=time.monotonic();self._last_message=time.monotonic()
        port=self._port();self._properties(directory,scenario,port)
        logdir=self.run/'logs';logdir.mkdir(parents=True,exist_ok=True)
        tag=directory.name
        self._server_log=logdir/f'{tag}-server.log';self._bridge_log=logdir/f'{tag}-bridge.log'
        (directory/'tmp').mkdir(exist_ok=True)
        proc=subprocess.Popen([sys.executable,'-I',str(ROOT/'server_owner.py'),self.java,f'-Djava.io.tmpdir={directory / "tmp"}','-XX:-UsePerfData','-XX:ActiveProcessorCount=4','-Xms512M','-Xmx2G','-jar',self.jar,'nogui'],cwd=directory,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        self._server=proc
        threading.Thread(target=self._read_server,args=(proc,self._server_log),daemon=True).start()
        if not self._server_ready.wait(120):raise InfrastructureError(f'Server startup timed out; see {self._server_log}')
        if proc.poll() is not None:raise InfrastructureError(f'Server exited; see {self._server_log}')
        self._commands([f'gamerule {key} {value}' for key,value in GAMERULES.items()]+['difficulty normal','weather clear','time set 1000'])
        err=open(self._bridge_log,'w');self._logfiles.append(err)
        bridge=subprocess.Popen([self.node,str(ROOT/'bridge/bridge.js'),json.dumps({'port':port,'username':self.username,'episode':self._episode})],cwd=ROOT/'bridge',stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
        self._bridge=bridge
        threading.Thread(target=self._read_bridge,args=(bridge,),daemon=True).start()
        if not self._bridge_ready.wait(45):raise InfrastructureError(f'Bot readiness timed out: {self._invalid}; see {self._bridge_log}')
        if self._invalid:raise InfrastructureError(self._invalid)
    def _read_server(self,proc,path):
        with open(path,'w') as log:
            for line in proc.stdout:
                log.write(line);log.flush()
                with self._condition:
                    self._server_lines.append(line.strip())
                    if "Can't keep up!" in line:self._lag_warnings+=1
                    if 'Done (' in line and self._server is proc:self._server_ready.set()
                    marker=f'{self.username} has the following entity data: '
                    payload=line.split('[Server thread/INFO]: ',1)[-1]
                    if payload.startswith(marker):
                        raw=payload[len(marker):].strip()
                        if raw.startswith('['):
                            counts=parse_inventory(raw);stamp=time.monotonic();self._last_audit=stamp;self._audit_serial+=1
                            if self._active and not self._terminal:
                                elapsed=min(self.duration,max(0,stamp-self._start));self._score.update(counts,elapsed)
                                if self._journal:self._journal.write(json.dumps({'t':elapsed,'inventory':counts})+'\n');self._journal.flush()
                            else:self._initial_inventory=counts
                            if self._audit_queue:
                                kind,audit_id=self._audit_queue.popleft()
                                if kind=='bridge':
                                    try:self._send({'cmd':'audit_ack','audit_id':audit_id})
                                    except (BrokenPipeError,InfrastructureError):pass
                    self._condition.notify_all()
        if not self._closing and self._server is proc:self._mark_invalid('server_exited')
        if self._server is proc:self._server_ready.set()
    def _read_bridge(self,proc):
        for line in proc.stdout:
            if self._bridge is not proc:break
            try:
                m=json.loads(line)
                if not isinstance(m,dict):continue
                with self._condition:
                    self._last_message=time.monotonic();kind=m.get('type')
                    if kind=='ready':self._bridge_ready.set()
                    elif kind=='observation' and not self._terminal:
                        self._snapshot=m;self._snapshot_received=time.monotonic()
                        if self._active:
                            self._score.observe(m,self._elapsed())
                            if m.get('self',{}).get('game_mode') not in (None,'survival'): self._mark_invalid('non_survival_mode')
                            if self._objective_complete(): self._finish('success')
                    elif kind=='response':
                        if isinstance(m.get('id'),int) and m['id']<0: self._action_feedback=m.get('result') or {'status':'rejected','error':m.get('error')}
                        else: self._responses[m['id']]=m
                    elif kind=='audit_request':self._query_inventory('bridge',m['id'])
                    elif kind=='death' and not self._terminal:
                        if m.get('observation'):self._snapshot=m['observation'];self._snapshot_received=time.monotonic()
                        if self._active:self._score.death(self._snapshot,self._elapsed())
                    elif kind=='action_result':
                        self._action_feedback=m
                        if m.get('status')!='succeeded':self._failures.append(m);self._failures=self._failures[-30:]
                    elif kind=='infrastructure_error' and not self._closing:self._mark_invalid(m.get('error','bridge_error'))
                    self._condition.notify_all()
            except Exception as e:
                if not self._closing:self._mark_invalid(f'bridge_protocol_error: {e}')
        if not self._closing and self._bridge is proc:self._mark_invalid('bridge_exited')
        if self._bridge is proc:self._bridge_ready.set()
    def _send(self,message):
        if not self._bridge or self._bridge.poll() is not None:raise InfrastructureError('Bridge is not running')
        with self._write_lock:self._bridge.stdin.write(json.dumps(message)+'\n');self._bridge.stdin.flush()
    def _rpc(self,cmd,timeout=10,**fields):
        with self._condition:
            self._request_id+=1;ident=self._request_id;self._send({'id':ident,'cmd':cmd,**fields});deadline=time.monotonic()+timeout
            while ident not in self._responses:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise InfrastructureError(f'Bridge {cmd} timed out')
                self._condition.wait(min(.5,remaining))
                if self._invalid:raise InfrastructureError(self._invalid)
            response=self._responses.pop(ident)
            if response.get('error'):raise InfrastructureError(response['error'])
            return response.get('result')
    def _console(self,command):
        if not self._server or self._server.poll() is not None:raise InfrastructureError('Server is not running')
        if self._active and command != f'data get entity {self.username} Inventory': raise InfrastructureError('Administrative mutation forbidden during scored play')
        with self._console_lock:self._server.stdin.write(command+'\n');self._server.stdin.flush()
    def _commands(self,commands,timeout=15):
        marker='HARNESS_BARRIER_'+uuid.uuid4().hex
        with self._condition:
            for cmd in commands:self._console(cmd)
            self._console('say '+marker);deadline=time.monotonic()+timeout
            while not any(marker in s for s in self._server_lines):
                if time.monotonic()>deadline:raise InfrastructureError('Server command barrier timed out')
                self._condition.wait(.1)
    def _query_inventory(self,kind='poll',ident=None):
        with self._condition:
            self._audit_queue.append((kind,ident))
            try:self._console(f'data get entity {self.username} Inventory')
            except Exception:self._audit_queue.pop();raise
    def _inventory_barrier(self):
        with self._condition:
            target=self._audit_serial+len(self._audit_queue)+1;self._query_inventory();deadline=time.monotonic()+5
            while self._audit_serial<target:
                if time.monotonic()>deadline:raise InfrastructureError('Inventory audit timed out')
                self._condition.wait(.1)
    def reset(self,seed:int):
        self.close();self._artifacts();setup_start=time.monotonic()
        self._invalid=None;self._terminal=None;self._start=None;self._end=None;self._ticks=0;self._score=Score();self._failures=[];self._lag_warnings=0;self._action_feedback=None;self._last_action=None;self._final_health=None;self._episode=uuid.uuid4().hex
        self._initial_server_age=None;self._initial_inventory={};self._audit_serial=0
        scenario=seed_config(seed)
        self._scenario={**scenario,'objective':self.objective,'target':self.target,'target_count':self.target_count,'max_seconds':self.duration}
        episode=self.run/'episodes'/self._episode;episode.mkdir(parents=True)
        self._directory=episode
        try:
            self._launch(episode,scenario)
            # The bot has never played or been teleported; it joins a newly generated world.
            snap=self._rpc('observe')
            if snap['self']['game_mode']!='survival':raise InfrastructureError('Expected ordinary survival player')
            self._inventory_barrier()
            if self._initial_inventory:raise InfrastructureError('Fresh player inventory is not empty')
            self._scenario['ordinary_spawn']=snap['self']['position']
            self._initial_server_age=snap.get('world',{}).get('server_age_ticks')
            self._journal=open(episode/'inventory-audit.jsonl','w')
            self._snapshot=self._rpc('activate');self._snapshot_received=time.monotonic()
            with self._lock:
                self._start=time.monotonic();self._active=True;self._last_audit=self._start
                self._score.observe(self._snapshot,0)
            self._setup_seconds=time.monotonic()-setup_start
            (episode/'scenario.json').write_text(json.dumps(self._scenario,indent=2)+'\n')
            self._monitor_stop=threading.Event();threading.Thread(target=self._monitor,args=(self._monitor_stop,),daemon=True).start()
            return self.observe()
        except Exception:
            self.close();raise
    def _objective_complete(self):
        if self.objective=='dragon':return self._score.success
        if self.objective=='advancement':return self.target in self._score.advancements
        return self._score.inventory.get(self.target,0)>=self.target_count
    def _monitor(self,stop):
        while not stop.wait(.25):
            with self._lock:
                if not self._active:return
                if time.monotonic()-self._start>=self.duration:self._finish('time_limit');return
                if self._snapshot.get('world',{}).get('server_time_age_s',0)>8:self._mark_invalid('server_time_stale');return
                if time.monotonic()-self._last_message>8:self._mark_invalid('bridge_stale');return
                if time.monotonic()-self._snapshot_received>8:self._mark_invalid('observation_stream_stale');return
                if not self._snapshot.get('self',{}).get('dead') and time.monotonic()-self._last_audit>8:self._mark_invalid('inventory_audit_stale');return
                if self._objective_complete():self._finish('success');return
                try:self._query_inventory()
                except Exception as e:self._mark_invalid(str(e));return
    def _mark_invalid(self,reason):
        with self._lock:
            if self._terminal:return
            self._invalid=str(reason)
            if self._active:self._finish('infrastructure_failure')
    def _finish(self,reason):
        if self._terminal:return
        self._final_health=self._snapshot.get('self',{}).get('health',20);self._terminal=reason;self._end=min(time.monotonic(),self._start+self.duration) if self._start else time.monotonic();self._active=False;self._monitor_stop.set()
        self._closing=True
        self._finish_thread=threading.Thread(target=self._finish_cleanup,daemon=True);self._finish_thread.start()
    def _finish_cleanup(self):
        try:
            if getattr(self,'_directory',None):(self._directory/'result.json').write_text(json.dumps(self.info(),indent=2)+'\n')
            self._shutdown_processes()
        except Exception:pass
    def tick(self,action):
        with self._lock:
            if not self._start:raise RuntimeError('Call reset(seed) before tick')
            if self._terminal:return
            self._ticks+=1
            # Optional freshness envelope; normal harnesses pass the compact string directly.
            if isinstance(action,dict):
                if action.get('episode')!=self._episode or not isinstance(action.get('seq'),int) or not self._snapshot.get('seq',0)-10<=action['seq']<=self._snapshot.get('seq',0):
                    self._action_feedback={'status':'rejected','error':'stale_episode_or_observation'};return
                action=action.get('action')
            if action is None:action='noop'
            if not isinstance(action,str):self._action_feedback={'status':'rejected','error':'expected_action_string'};return
            action=action.strip()
            if action==self._last_action and action.split(' ',1)[0] not in ('noop','cancel','controls','respawn','attack'):return
            self._last_action=action
        try:
            if len(action)>512:self._action_feedback={'status':'rejected','error':'action_too_long'};return
            self._request_id+=1
            self._send({'id':-self._request_id,'cmd':'act','action':action})
        except (InfrastructureError,OSError,ValueError) as e:self._mark_invalid(str(e))
    def _elapsed(self):return 0. if self._start is None else max(0.,min(self.duration,(self._end or time.monotonic())-self._start))
    def _progress(self):
        result=self._score.report(self._elapsed(),self.duration,self._final_health if self._final_health is not None else self._snapshot.get('self',{}).get('health',20),self._terminal)
        result['objective']=self.objective
        result['objective_complete']=self._objective_complete()
        if self.objective!='dragon':
            value=1. if self._objective_complete() else (min(1.,self._score.inventory.get(self.target,0)/self.target_count) if self.objective=='resource' else 0.)
            result['score']=100.*value
            result['components']={'objective_progress':100.*value}
            result['target']=self.target;result['target_count']=self.target_count
        result['elapsed_server_ticks']=max(0,int(self._snapshot.get('world',{}).get('server_age_ticks',0)-(self._initial_server_age or 0)))
        return result
    def observe(self):
        with self._lock:
            result=copy.deepcopy(self._snapshot);result.pop('type',None)
            result.update(episode=self._episode,elapsed_s=self._elapsed(),remaining_s=max(0.,self.duration-self._elapsed()),observation_age_s=max(0.,time.monotonic()-self._snapshot_received),inventory=dict(self._score.inventory),inventory_source='atomic server Inventory NBT; personal slots only',progress=self._progress(),terminal=self._terminal,valid_trial=not bool(self._invalid),invalid_reason=self._invalid,action_feedback=copy.deepcopy(self._action_feedback))
            return result
    @property
    def done(self):return bool(self._terminal)
    @property
    def score(self):
        with self._lock:return float(self._progress()['score'])
    @property
    def ticks(self):return self._ticks
    def score_breakdown(self):
        with self._lock:return dict(self._progress()['components'])
    def info(self):
        with self._lock:return {'name':self.name,'tier':2,'benchmark':CONFIG,'scenario':self._scenario,'episode':self._episode,'duration_s':self.duration,'tick_s':self.tick_s,'ticks':self._ticks,'elapsed_s':self._elapsed(),'setup_seconds':getattr(self,'_setup_seconds',None),'deterministic':False,'terminal':self._terminal,'valid_trial':not bool(self._invalid),'invalid_reason':self._invalid,'progress':self._progress(),'death_records':copy.deepcopy(self._score.death_records),'recent_action_failures':copy.deepcopy(self._failures),'server_log':str(getattr(self,'_server_log','')),'inventory_audit_age_s':time.monotonic()-self._last_audit,'health':self._snapshot.get('self',{}).get('health'),'final_health':self._final_health,'server_time_age_s':self._snapshot.get('world',{}).get('server_time_age_s'),'server_lag_warnings':self._lag_warnings}
    def describe(self):
        objective='Defeat the Ender Dragon: success is the server advancement minecraft:end/kill_dragon (Free the End).'
        if self.objective=='advancement':objective=f'Complete the server advancement {self.target}.'
        if self.objective=='resource':objective=f'Possess {self.target_count} {self.target} in personal inventory.'
        return f"""Play Minecraft Java 1.21.1 Survival, Normal difficulty, in a fresh naturally generated world. {objective}
Time limit {self.duration:g} wall-clock seconds, excluding setup. The actual server and bot physics run continuously at target 20Hz even when you do not call tick or observe. Input cadence is 0.5 seconds. Death drops inventory (keepInventory=false); use respawn to continue normally. No cheats, bonus resources, teleport, modified drops, peaceful mode or privileged map information. Initial day 1000 and clear weather are set before joining; ordinary time, weather, mobs, hunger and damage then run freely. Spawn is an ordinary vanilla world spawn with spawnRadius=0. Natural seeds are not filtered for easy wood/food; seed is evaluator-only. Coordinates: +x east, +y up, +z south.
Dragon score [0,100]: opening 10 (having held both logs and crafting table 3, food 3, any bed or sleep advancement 4); iron 5 and shield 5; Nether entry 8, fortress advancement 6, blaze rod 6; pearl 4, eye 6, stronghold advancement 5, seen reachable active End portal 5; End entry 10; dragon advancement 30. Flags are permanent, no points from idling or duplicate transfers. Successful dragon completion is always score 100; any unearned preparation weights become a separately reported completion bonus. Primary comparison is success, then milestone progress; raw deaths/damage/resources remain reported. Specialized resource scenarios score proportional current possession; advancement scenarios score 100 on completion only. Non-infrastructure death is not an invalid trial.
Observations: episode/seq/age; self position/velocity/yaw/pitch/ground, survival mode/dimension, health/food/saturation/oxygen/effects, water/lava/dead, equipment/XP/bed knowledge; authoritative audited personal inventory and labelled client inventory/slot previews; open container/cursor; world day/time/biome/weather and last server-time age; nearby entities (some occluded, visible flag); exposed block candidates and collision labels; limited block changes/events; server-completed advancements; feasible recipe hints; current bounded task, action results, score and raw progress. Candidates b:x,y,z and s:x,y,z come from loaded nearby terrain. Exposure is adjacent free space, NOT pixel-equivalent vision; visible is a raycast from your eyes. Unknown chunks, seed, underground solid interiors, structure coordinates and console are not exposed. Inventory score excludes cursor, recipe output preview and containers. Snapshot data can lag; inspect observation_age_s, task and feedback.
Actions are whitespace-separated strings, never code or commands. One bounded asynchronous job owns hands/navigation. noop continues a job, but when idle releases controls/item use; cancel interrupts. Busy commands are rejected: cancel then wait for task=null. Identical one-shot actions are latched, so send noop between deliberate repeats; controls renew each tick, attacks repeat with cooldown. controls KEYS is a comma-list of forward,back,left,right,jump,sprint,sneak or none, relative to current yaw; its 0.8s lease clears if ticker stalls. use_item holds 0–5s then releases (bow); equip first, look_at first. shield on stays active until another action/noop/cancel. Gathering chooses tools/path then mines 1–4 blocks, tries real pickup, and times out after25s. Navigate/goto have15s deadline; others10s. Navigation never mines or bridges, avoids known hazards and drops >2, and cancels on damage; manual controls can deliberately move into water/portals and are not a survival autopilot. dig is direct in-reach mining with held tool, no auto-pickup; no mining your support. place takes ITEM REFERENCE_ID FACE with up/down/east/west/north/south, crafting wants NEW_COUNT outputs (rounded to whole recipe executions). Furnaces consume real input/fuel and take real time. Container windows remain open for transfer/move_slot; cancel/close_window closes them. Trade and enchanting consume ordinary payment/XP. Rejected or malformed input reports feedback, not a Python exception. Optional {{episode,seq,action}} envelope rejects stale (>10 snapshots) observations.
"""+self.action_spec.describe()
    def _shutdown_processes(self):
        with self._shutdown_lock:self._shutdown_impl()
    def _shutdown_impl(self):
        self._closing=True
        bridge=self._bridge;server=self._server
        if bridge and bridge.poll() is None:
            try:self._send({'cmd':'close'})
            except Exception:pass
            try:bridge.wait(timeout=4)
            except subprocess.TimeoutExpired:
                bridge.terminate()
                try:bridge.wait(timeout=3)
                except subprocess.TimeoutExpired:bridge.kill();bridge.wait(timeout=3)
        self._bridge=None
        if server and server.poll() is None:
            try:self._console('save-all flush');self._console('stop')
            except Exception:pass
            try:server.wait(timeout=35)
            except subprocess.TimeoutExpired:
                server.terminate()
                try:server.wait(timeout=35)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        self._clean_shutdown=not server or server.returncode==0
        self._server=None
        process_dir=getattr(self,'_process_dir',None)
        if process_dir and process_dir.parent==self.run/'episodes' and process_dir.exists():
            (process_dir/'.stopped').touch()
            stopped=sorted((p for p in (self.run/'episodes').iterdir() if (p/'.stopped').exists()),key=lambda p:(p/'.stopped').stat().st_mtime,reverse=True)
            for old in stopped[3:]:
                for generated in ('world','libraries','versions'):shutil.rmtree(old/generated,ignore_errors=True)
        for proc in (bridge,server):
            if proc:
                for stream in (proc.stdin,proc.stdout):
                    if stream:
                        try:stream.close()
                        except Exception:pass
        for log in self._logfiles:log.close()
        self._logfiles=[]
        if self._journal:self._journal.close();self._journal=None
    def close(self):
        with self._lock:
            if self._active and not self._terminal:
                self._end=time.monotonic();self._terminal='closed';self._final_health=self._snapshot.get('self',{}).get('health',20)
            self._active=False;self._closing=True;self._monitor_stop.set()
        if self._finish_thread and self._finish_thread is not threading.current_thread():self._finish_thread.join(timeout=45)
        self._finish_thread=None;self._shutdown_processes()
        if getattr(self,'_directory',None) and self._start:
            (self._directory/'result.json').write_text(json.dumps(self.info(),indent=2)+'\n')
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def make_game(**kwargs):return MinecraftGame(**kwargs)

def _cleanup():
    for game in list(_INSTANCES):
        try:game.close()
        except Exception:pass
atexit.register(_cleanup)
