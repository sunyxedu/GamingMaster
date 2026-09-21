"""Minecraft Java 1.21.1 real-time adaptor. Policy has no console access."""
from __future__ import annotations
import atexit, collections, copy, hashlib, json, math, os, pathlib, re, shutil, socket, subprocess, sys, threading, time, uuid, weakref
from metaharness.games.base import ActionSpec
from scoring import Score, parse_inventory

TIER=2
DETERMINISTIC=False
ROOT=pathlib.Path(__file__).resolve().parent
CONFIG='vanilla-1.21.1-admission-v3-score-v1'
_INSTANCES=weakref.WeakSet()

class InfrastructureError(RuntimeError): pass
class InadmissibleSeed(ValueError): pass

def seed_config(seed):
    seed=int(seed)
    digest=hashlib.sha256(f'minecraft:1.21.1:world:{seed}'.encode()).digest()
    pair=hashlib.sha256(f'minecraft:1.21.1:time:{seed//2}'.encode()).digest()[0]&1
    return {'seed':seed,'world_seed':str(int.from_bytes(digest[:8], 'big', signed=True)), 'time_of_day':9000 if (seed%2)^pair else 1000,'config':CONFIG}

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
 'wait':'wait [SECONDS=0.5]: wait for 0–5 seconds without freezing the world.'}

class MinecraftGame:
    name='Minecraft Java 1.21.1 — natural survival'
    tick_s=.5
    action_spec=ActionSpec(names=list(ACTIONS),descriptions=ACTIONS,noop='noop',notes='Actions are whitespace-separated strings. Bare parameterized names are rejected as feedback, not Python exceptions. Only one bounded job owns movement/hands. Repeated identical current actions do not restart; send noop between deliberate repeats. noop preserves an existing job. cancel before replacing a busy job.')
    def __init__(self, **kwargs):
        allowed={'run_dir','port','username'}
        unknown=set(kwargs)-allowed
        if unknown: raise TypeError(f'Unknown options: {sorted(unknown)}')
        self.run=pathlib.Path(kwargs.get('run_dir',os.environ.get('MC_RUN_DIR',ROOT/'runtime'))).resolve()
        if os.environ.get('MC_HOST','127.0.0.1')!='127.0.0.1': raise ValueError('MC_HOST must be 127.0.0.1')
        if os.environ.get('MC_VERSION','1.21.1')!='1.21.1': raise ValueError('Only Minecraft 1.21.1 is validated')
        self.username=str(kwargs.get('username',os.environ.get('MC_USERNAME','HarnessBot')))
        if not re.fullmatch(r'[A-Za-z0-9_]{1,16}',self.username): raise ValueError('Invalid offline username')
        self.fixed_port=kwargs.get('port',os.environ.get('MC_PORT'))
        self.duration=300.0  # Fixed benchmark, never silently shortened.
        self._lock=threading.RLock();self._condition=threading.Condition(self._lock);self._write_lock=threading.Lock();self._console_lock=threading.Lock()
        self._server=None;self._bridge=None;self._server_ready=threading.Event();self._bridge_ready=threading.Event()
        self._closing=True;self._active=False;self._terminal=None;self._invalid=None;self._ticks=0;self._start=None;self._end=None;self._snapshot={};self._score=Score();self._request_id=0;self._responses={};self._audit_queue=collections.deque();self._audit_serial=0;self._last_audit=0.;self._last_message=0.;self._snapshot_received=0.;self._last_action=None;self._action_feedback=None;self._failures=[];self._server_lines=collections.deque(maxlen=400);self._monitor_stop=threading.Event();self._journal=None;self._scenario=None;self._episode=None;self._logfiles=[];self._finish_thread=None;self._final_health=None;self._shutdown_lock=threading.Lock();self._clean_shutdown=True;self._lag_warnings=0
        _INSTANCES.add(self)
    def _artifacts(self):
        path=self.run/'artifacts.json'
        if not path.exists(): raise InfrastructureError('Run ./setup.sh first; runtime/artifacts.json is missing')
        artifacts=json.loads(path.read_text())
        self.java=os.environ.get('JAVA_BIN',artifacts['java']);self.jar=str(pathlib.Path(os.environ.get('MC_SERVER_JAR',artifacts['server_jar'])).resolve());self.node=os.environ.get('NODE_BIN','node')
        if hashlib.sha1(pathlib.Path(self.jar).read_bytes()).hexdigest()!=artifacts['server_sha1']:raise InfrastructureError('Server jar checksum mismatch')
    def _port(self):
        if self.fixed_port:return int(self.fixed_port)
        with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
    def _properties(self,directory,scenario,port):
        props={'server-ip':'127.0.0.1','server-port':port,'online-mode':'false','enforce-secure-profile':'false','gamemode':'survival','force-gamemode':'true','difficulty':'normal','hardcore':'false','max-players':1,'pvp':'false','spawn-protection':0,'level-name':'world','level-seed':scenario['world_seed'],'level-type':'minecraft:normal','generate-structures':'true','view-distance':6,'simulation-distance':5,'enable-rcon':'false','enable-query':'false','enable-command-block':'false','white-list':'false','player-idle-timeout':0,'allow-flight':'false','sync-chunk-writes':'true','max-tick-time':60000,'broadcast-console-to-ops':'false','network-compression-threshold':256}
        (directory/'server.properties').write_text(''.join(f'{k}={v}\n' for k,v in props.items()))
        (directory/'eula.txt').write_text('eula=true\n')
    def _launch(self,directory,scenario,preparation=False):
        directory.mkdir(parents=True,exist_ok=True);self._process_dir=directory;self._closing=False;self._server_ready.clear();self._bridge_ready.clear();self._snapshot={};self._audit_queue.clear();self._responses.clear();self._server_lines.clear();self._last_audit=time.monotonic();self._last_message=time.monotonic()
        port=self._port();self._properties(directory,scenario,port)
        logdir=self.run/'logs';logdir.mkdir(parents=True,exist_ok=True)
        tag=directory.name
        self._server_log=logdir/f'{tag}-server.log';self._bridge_log=logdir/f'{tag}-bridge.log'
        proc=subprocess.Popen([sys.executable,str(ROOT/'server_owner.py'),self.java,'-Xms512M','-Xmx2G','-jar',self.jar,'nogui'],cwd=directory,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        self._server=proc
        threading.Thread(target=self._read_server,args=(proc,self._server_log),daemon=True).start()
        if not self._server_ready.wait(120):raise InfrastructureError(f'Server startup timed out; see {self._server_log}')
        if proc.poll() is not None:raise InfrastructureError(f'Server exited; see {self._server_log}')
        if preparation:self._commands(['gamerule spawnRadius 0','difficulty peaceful','gamerule doMobSpawning false','gamerule doDaylightCycle false','weather clear','time set 1000'])
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
                    if marker in line:
                        raw=line.split(marker,1)[1].strip()
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
                    elif kind=='observation' and not self._terminal:self._snapshot=m;self._snapshot_received=time.monotonic()
                    elif kind=='response':self._responses[m['id']]=m
                    elif kind=='audit_request':self._query_inventory('bridge',m['id'])
                    elif kind=='death':
                        if m.get('observation'):self._snapshot=m['observation']
                        if self._active:self._finish('death')
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
    def _prepare(self,scenario,baseline):
        import fcntl
        baseline.parent.mkdir(parents=True,exist_ok=True)
        with open(baseline.with_suffix('.lock'),'a') as lockfile:
            fcntl.flock(lockfile,fcntl.LOCK_EX)
            return self._prepare_locked(scenario,baseline)
    def _prepare_locked(self,scenario,baseline):
        if baseline.exists() and not (baseline/'manifest.json').exists():shutil.rmtree(baseline)
        if baseline.exists():
            record=json.loads((baseline/'manifest.json').read_text())
            if not record['admitted']:raise InadmissibleSeed(f'Seed {scenario["seed"]} inadmissible: {record["attempts"][-1]["reason"]}')
            return record
        build=self.run/'building'/uuid.uuid4().hex;build.mkdir(parents=True)
        record={**scenario,'admitted':False,'predicate':'dry surface; directed walking +1/-2; >=12 logs reachable from surface; >=24 stone beneath <=2 cover with solid support; radius48; max13 centers within96','attempts':[]}
        try:
            self._launch(build,scenario,True)
            self._commands([f'gamemode spectator {self.username}'])
            snap=self._rpc('observe');x,_,z=snap['self']['position'];x=math.floor(x);z=math.floor(z);record['natural_spawn']=[x,z]
            offsets=[(0,0),(-48,0),(0,-48),(0,48),(48,0),(-48,-48),(-48,48),(48,-48),(48,48),(-96,0),(0,-96),(0,96),(96,0)]
            for dx,dz in offsets:
                cx,cz=x+dx,z+dz
                self._commands([f'tp {self.username} {cx+.5} 160 {cz+.5}'])
                # Explicitly wait for each required column, not only initial login chunks.
                result=self._rpc('qualify',timeout=90,x=cx,z=cz)
                record['attempts'].append(result)
                if result['admitted']:
                    record['admitted']=True;record['spawn']=result['spawn'];break
            if record['admitted']:
                sx,sy,sz=record['spawn']
                self._commands([f'setworldspawn {math.floor(sx)} {int(sy)} {math.floor(sz)}','gamerule spawnRadius 0','difficulty normal','gamerule doMobSpawning true','gamerule doDaylightCycle true','gamerule doImmediateRespawn false','gamerule keepInventory false','gamerule naturalRegeneration true','weather clear',f'time set {scenario["time_of_day"]}'])
            self._shutdown_processes()
            if not self._clean_shutdown:raise InfrastructureError('Preparation server did not stop cleanly; baseline discarded')
            if record['admitted']:
                for folder in ('playerdata','stats','advancements'):shutil.rmtree(build/'world'/folder,ignore_errors=True)
                # Baselines only contain stopped natural worlds and provenance, never played players.
                baseline.mkdir(parents=True);shutil.move(str(build/'world'),str(baseline/'world'))
            else:baseline.mkdir(parents=True)
            (baseline/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
            shutil.rmtree(build,ignore_errors=True)
            if not record['admitted']:raise InadmissibleSeed(f'Seed {scenario["seed"]} failed fixed terrain admission; see {baseline}/manifest.json')
            return record
        finally:self._shutdown_processes()
    def reset(self,seed:int):
        self.close();self._artifacts();setup_start=time.monotonic();self._invalid=None;self._terminal=None;self._start=None;self._end=None;self._ticks=0;self._score=Score();self._failures=[];self._lag_warnings=0;self._action_feedback=None;self._last_action=None;self._final_health=None;self._episode=uuid.uuid4().hex
        scenario=seed_config(seed)
        key=hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest()[:24]
        baseline=self.run/'baselines'/key
        try:
            record=self._prepare(scenario,baseline);self._scenario={**scenario,'spawn':record['spawn'],'baseline_id':key}
            episode=self.run/'episodes'/self._episode;episode.mkdir(parents=True)
            # APFS clone is fast and copy-on-write; portable fallback is an ordinary full copy.
            if os.uname().sysname=='Darwin':subprocess.run(['/bin/cp','-cR',str(baseline/'world'),str(episode/'world')],check=True)
            else:shutil.copytree(baseline/'world',episode/'world')
            self._directory=episode;self._launch(episode,scenario)
            x,y,z=record['spawn']
            self._commands([f'tp {self.username} {x} {y} {z}',f'gamemode survival {self.username}',f'deop {self.username}',f'clear {self.username}',f'effect clear {self.username}',f'experience set {self.username} 0 points',f'experience set {self.username} 0 levels','weather clear',f'time set {scenario["time_of_day"]}'])
            # Readiness is a state barrier. A fresh player must have vanilla full health/food.
            deadline=time.monotonic()+10
            while True:
                snap=self._rpc('observe');p=snap['self']['position']
                if snap['self']['game_mode']=='survival' and snap['self']['health']==20 and snap['self']['food']==20 and abs(p[0]-x)<.1 and abs(p[1]-y)<.1 and abs(p[2]-z)<.1:break
                if time.monotonic()>deadline:raise InfrastructureError('Fresh player failed health/food/position readiness')
                time.sleep(.05)
            self._inventory_barrier()
            if self._initial_inventory:raise InfrastructureError('Fresh baseline player inventory was not empty')
            hostile={'zombie','skeleton','creeper','spider','enderman','witch','drowned','pillager','husk','stray','slime'}
            if any(e['name'] in hostile and e['distance']<16 for e in snap['entities']):raise InfrastructureError('Hostile near start; trial not started')
            self._commands(['weather clear',f'time set {scenario["time_of_day"]}'])
            self._journal=open(episode/'inventory-audit.jsonl','w')
            self._snapshot=self._rpc('activate');self._snapshot_received=time.monotonic()
            self._commands(['weather clear',f'time set {scenario["time_of_day"]}'])
            with self._lock:self._start=time.monotonic();self._active=True;self._last_audit=self._start;self._score=Score()
            self._setup_seconds=time.monotonic()-setup_start
            (episode/'scenario.json').write_text(json.dumps(self._scenario,indent=2)+'\n')
            self._monitor_stop=threading.Event();threading.Thread(target=self._monitor,args=(self._monitor_stop,),daemon=True).start()
            return self.observe()
        except Exception:
            self.close();raise
    def _monitor(self,stop):
        while not stop.wait(.1):
            with self._lock:
                if not self._active:return
                if time.monotonic()-self._start>=self.duration:self._finish('time_limit');return
                if self._snapshot.get('world',{}).get('server_time_age_s',0)>8:self._mark_invalid('server_time_stale');return
                if time.monotonic()-self._last_message>8:self._mark_invalid('bridge_stale');return
                if time.monotonic()-self._last_audit>5:self._mark_invalid('inventory_audit_stale');return
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
            if action==self._last_action and action not in ('noop','cancel'):return
            self._last_action=action
        try:self._action_feedback=self._rpc('act',action=action,timeout=2)
        except InfrastructureError as e:self._mark_invalid(str(e))
    def _elapsed(self):return 0. if self._start is None else max(0.,min(self.duration,(self._end or time.monotonic())-self._start))
    def _progress(self):return self._score.report(self._elapsed(),self.duration,self._final_health if self._final_health is not None else self._snapshot.get('self',{}).get('health',20),self._terminal)
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
    def info(self):
        with self._lock:return {'name':self.name,'tier':2,'benchmark':CONFIG,'scenario':self._scenario,'episode':self._episode,'duration_s':self.duration,'tick_s':self.tick_s,'ticks':self._ticks,'elapsed_s':self._elapsed(),'setup_seconds':getattr(self,'_setup_seconds',None),'deterministic':False,'terminal':self._terminal,'valid_trial':not bool(self._invalid),'invalid_reason':self._invalid,'progress':self._progress(),'recent_action_failures':copy.deepcopy(self._failures),'server_log':str(getattr(self,'_server_log','')),'inventory_audit_age_s':time.monotonic()-self._last_audit,'health':self._snapshot.get('self',{}).get('health'),'final_health':self._final_health,'server_time_age_s':self._snapshot.get('world',{}).get('server_time_age_s'),'server_lag_warnings':self._lag_warnings}
    def describe(self):
        return '''Play ordinary Minecraft Java 1.21.1 Survival on Normal difficulty for 300 wall-clock seconds. The vanilla server and bot physics keep running while you think. Death ends the episode without respawn. You start healthy, empty-handed on an admitted natural dry surface, with nearby wood and shallow stone. Initial clear weather and time 1000 or 9000; night, weather, monsters, hunger, fall/fire/drowning damage, tool requirements, crafting and smelting follow vanilla rules. Unknown terrain is not supplied. Coordinates are x east, y up, z south.
Objective: maximize absolute score [0,100]. Resource credit uses episode maxima of simultaneously held personal inventory: min(unstripped Overworld logs,10)+0.5*min(cobblestone,20)+min(coal+charcoal,5)+min(raw_iron,5), up to30. First possession gives crafting_table5, wooden_pickaxe5, stone_pickaxe10, stone_axe5, furnace5, torch5, cooked_food5, iron_ingot5, shield5, up to50. Cooked food is cooked_beef/porkchop/mutton/chicken/rabbit/cod/salmon or baked_potato. Loot counts. Spent/dropped items retain earned maxima/milestone credit; repeated transfers do not add credit. Survive for 10*seconds/300 points. At an alive time-limit finish only, add 10*health/20; death subtracts20; clamp total. Full healthy idling yields20. Infrastructure failures invalidate a trial.
Observations include episode/sequence/freshness, elapsed and remaining seconds, self position/velocity/health/food/oxygen/equipment, authoritative personal inventory, predictive inventory_client, day/weather, nearby entities, local collision labels relative to your floored position, exposed block and stand candidates, last-observed furnace contents (not live after closing), feasible recipes, current task, action results, and score components/milestone times. Candidate IDs are strings b:x,y,z or s:x,y,z supplied by observations; resource candidates are limited to exposed loaded blocks within24 blocks, never a hidden ore scan. Stand points are locally checked; routes can fail. Entities may be occluded. Unknown chunks are omitted. No screenshot is needed. Inventory scoring comes from atomic read-only server queries, not client crafting previews. The policy cannot use console commands, teleport, creative inventory, raw packets or arbitrary code.
Use whitespace-separated action strings below. Craft NEW_COUNT means requested new outputs, rounded up to whole recipe executions (craft oak_planks 4 uses one log). Table recipes require a nearby placed table; optionally give its candidate ID. Furnace loading returns immediately; smelting takes actual time; retrieve later. Navigation/gather automatically choose a conservative route but never dig/bridge to travel. Gathering checks tool, reach and exposure, mines and tries to pick up a real drop. One job runs at a time. Deadlines are25s gather,15s navigation,10s others. noop lets a bounded job continue; cancel really clears controls. Wait and invalid actions still consume real time. Identical current actions are latched and do not replay after finishing: send noop then the same command to repeat deliberately. A different command while busy is rejected; cancel then wait for task=null. Actions receive feedback rather than raising for bad arguments. Optional dictionary envelope {episode,seq,action} rejects earlier episodes and observations more than10 snapshots old.
'''+self.action_spec.describe()
    def _shutdown_processes(self):
        with self._shutdown_lock:self._shutdown_impl()
    def _shutdown_impl(self):
        self._closing=True
        bridge=self._bridge;server=self._server
        if bridge and bridge.poll() is None:
            try:self._send({'cmd':'close'})
            except Exception:pass
            try:bridge.wait(timeout=4)
            except subprocess.TimeoutExpired:bridge.terminate();bridge.wait(timeout=3)
        self._bridge=None
        if server and server.poll() is None:
            try:self._console('save-all flush');self._console('stop')
            except Exception:pass
            try:server.wait(timeout=25)
            except subprocess.TimeoutExpired:
                server.terminate()
                try:server.wait(timeout=5)
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
        if self._active and not self._terminal:
            self._end=time.monotonic();self._terminal='closed';self._invalid='episode_closed_before_terminal';self._final_health=self._snapshot.get('self',{}).get('health',20)
        self._active=False;self._monitor_stop.set()
        if self._finish_thread and self._finish_thread is not threading.current_thread():self._finish_thread.join(timeout=45)
        self._finish_thread=None;self._shutdown_processes()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def make_game(**kwargs):return MinecraftGame(**kwargs)

def _cleanup():
    for game in list(_INSTANCES):
        try:game.close()
        except Exception:pass
atexit.register(_cleanup)
