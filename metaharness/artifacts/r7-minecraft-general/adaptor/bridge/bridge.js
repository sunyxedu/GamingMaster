'use strict'
const mineflayer=require('mineflayer')
const {pathfinder,Movements,goals}=require('mineflayer-pathfinder')
const {Vec3}=require('vec3')
const readline=require('readline')
const {qualify,hazards,logs}=require('./terrain')
const cfg=JSON.parse(process.argv[2])
const now=()=>performance.now()/1000
const send=o=>process.stdout.write(JSON.stringify(o)+'\n')
const sleep=ms=>new Promise(r=>setTimeout(r,ms))
async function pause(j,ms){const end=now()+ms/1000;while(now()<end){check(j);await sleep(Math.min(50,Math.max(1,(end-now())*1000)))}check(j)}
const bot=mineflayer.createBot({host:'127.0.0.1',port:cfg.port,username:cfg.username,version:'1.21.1',auth:'offline',respawn:false,logErrors:false,viewDistance:'short'})
bot.loadPlugin(pathfinder)
let ready=false,active=false,closing=false,dead=false,seq=0,job=null,lastResult=null,jobCounter=0,lastTime=now(),physics=0,lastCandidateUpdate=0,candidates=[],candidateMap=new Map(),lastAttack=0,auditCounter=0
const audits=new Map(),workstations=new Map()
const pos=v=>v ? [Number(v.x.toFixed(3)),Number(v.y.toFixed(3)),Number(v.z.toFixed(3))] : null
const distance=p=>bot.entity.position.distanceTo(p)
const block=p=>bot.blockAt(p,false)
const empty=b=>b&&b.boundingBox==='empty'&&!hazards.has(b.name)
const safeStand=p=>{const floor=block(p.offset(0,-1,0));return floor&&floor.boundingBox==='block'&&!hazards.has(floor.name)&&empty(block(p))&&empty(block(p.offset(0,1,0)))}
const item=n=>bot.inventory.slots.slice(5,46).find(i=>i&&i.name===n)
const counts=()=>{const out={};for(const i of bot.inventory.slots.slice(5,46))if(i)out[i.name]=(out[i.name]||0)+i.count;return out}
const vec=a=>new Vec3(...a)
function stopControls(){bot.pathfinder.setGoal(null);bot.clearControlStates();bot.stopDigging();bot.deactivateItem();if(bot.currentWindow)bot.closeWindow(bot.currentWindow);else if(job?.cancelled&&bot.inventory&&(bot.inventory.selectedItem||bot.inventory.slots.slice(1,5).some(Boolean)))bot.closeWindow(bot.inventory)}
function cancel(reason='cancelled') {if(job&&!job.cancelled){job.cancelled=true;job.reason=reason;job.cancelledAt=now()}stopControls()}
function check(j){if(j.cancelled||dead||!active)throw Error(j.reason||'cancelled');if(now()>j.deadline){cancel('timeout');throw Error('timeout')}}
async function audit(j) {
  if(j)check(j)
  const id=++auditCounter
  const p=new Promise((resolve,reject)=>{const timer=setTimeout(()=>{audits.delete(id);reject(Error('authoritative_inventory_timeout'))},3000);audits.set(id,{resolve:()=>{clearTimeout(timer);resolve()}})})
  send({type:'audit_request',id});await p;if(j)check(j)
}
function exposed(b){return [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]].some(a=>empty(block(b.position.plus(vec(a)))))}
function updateCandidates(){
  const p=bot.entity.position.floored(), out=[]
  const groups=[[[...logs],24],[['crafting_table','furnace'],8],[['coal_ore','iron_ore'],12],[['stone','cobblestone'],40],[['dirt','grass_block'],12]]
  const found=groups.flatMap(([names,count])=>bot.findBlocks({matching:names.map(n=>bot.registry.blocksByName[n].id),maxDistance:24,count,useExtraInfo:exposed}))
  for(const p of found){const b=block(p);if(!b||!exposed(b))continue
    const id=`b:${p.x},${p.y},${p.z}`
    out.push({id,kind:'block',name:b.name,position:pos(p),distance:+distance(p).toFixed(2),in_reach:bot.canDigBlock(b),harvestable:b.canHarvest(bot.heldItem?.type??null),seen_at:now()})
    if(out.length>=96)break
  }
  for(const [dx,dz] of [[1,0],[-1,0],[0,1],[0,-1],[3,0],[-3,0],[0,3],[0,-3],[6,0],[-6,0],[0,6],[0,-6]])for(let dy=1;dy>=-2;dy--){const q=p.offset(dx,dy,dz);if(safeStand(q)){out.push({id:`s:${q.x},${q.y},${q.z}`,kind:'stand',position:pos(q),distance:+distance(q).toFixed(2),seen_at:now()});break}}
  candidates=out;candidateMap=new Map(out.map(c=>[c.id,c]));lastCandidateUpdate=now()
}
function candidate(id,kind){const c=candidateMap.get(id);if(!c||now()-c.seen_at>8)throw Error('unknown_or_stale_candidate');if(kind&&c.kind!==kind)throw Error('wrong_candidate_kind');const p=vec(c.position);if(distance(p)>32)throw Error('candidate_too_far');if(c.kind==='block'&&block(p)?.name!==c.name)throw Error('target_changed');return c}
function snapshot(){
  if(!ready||!bot.entity)return null
  if(active&&now()-lastCandidateUpdate>1)updateCandidates()
  const collision=[];const origin=bot.entity.position.floored();for(let dx=-2;dx<=2;dx++)for(let dz=-2;dz<=2;dz++)for(let dy=-1;dy<=2;dy++){const b=block(origin.offset(dx,dy,dz));collision.push([dx,dy,dz,b?(hazards.has(b.name)?'hazard':b.boundingBox==='empty'?'passable':'solid'):'unknown'])}
  const entities=Object.values(bot.entities).filter(e=>e!==bot.entity&&distance(e.position)<=32).sort((a,b)=>distance(a.position)-distance(b.position)).slice(0,32).map(e=>({id:e.id,name:e.name||e.displayName||e.type,position:pos(e.position),velocity:pos(e.velocity),distance:+distance(e.position).toFixed(2)}))
  const tableCandidate=candidates.find(c=>c.name==='crafting_table'&&distance(vec(c.position))<=4)
  const table=tableCandidate?block(vec(tableCandidate.position)):null
  const recipes=[]
  if(active)for(const n of ['oak_planks','spruce_planks','birch_planks','jungle_planks','acacia_planks','dark_oak_planks','mangrove_planks','cherry_planks','stick','crafting_table','wooden_pickaxe','stone_pickaxe','stone_axe','furnace','torch','shield']){const id=bot.registry.itemsByName[n]?.id;if(id&&bot.recipesFor(id,null,1,table).length)recipes.push(n)}
  return {type:'observation',protocol:1,episode:cfg.episode,seq:++seq,node_time:now(),self:{position:pos(bot.entity.position),velocity:pos(bot.entity.velocity),yaw:bot.entity.yaw,pitch:bot.entity.pitch,on_ground:bot.entity.onGround,health:bot.health,food:bot.food,saturation:bot.foodSaturation,oxygen:bot.oxygenLevel??null,game_mode:bot.game.gameMode,dimension:bot.game.dimension,held_item:bot.heldItem?.name??null,equipment:bot.entity.equipment?.map(i=>i?{name:i.name,count:i.count}:null)},inventory_client:counts(),inventory_empty_slots:bot.inventory.emptySlotCount(),world:{time_of_day:bot.time.timeOfDay,raining:bot.isRaining,server_time_age_s:now()-lastTime,physics_ticks:physics,unknown_chunks:'not traversable; omitted',local_collision:collision},candidates:active?candidates:[],entities,workstations:active?[...workstations.values()]:[],feasible_recipes:recipes,task:job?{id:job.id,action:job.action,phase:job.phase,cancelled:job.cancelled,remaining_s:Math.max(0,job.deadline-now())}:null,last_result:lastResult}
}
function visibleBlock(b,eye=bot.entity.position.offset(0,1.62,0)){
  if(!b)return false
  const delta=b.position.offset(.5,.5,.5).minus(eye),hit=bot.world.raycast(eye,delta.unit(),delta.norm()+.01)
  return !!hit&&hit.position.equals(b.position)
}
class InteractionGoal extends goals.Goal {
  constructor(p){super();this.p=p}
  heuristic(node){return Math.max(0,node.distanceTo(this.p)-3)}
  isEnd(node){const eye=new Vec3(node.x+.5,node.y+1.62,node.z+.5);return eye.distanceTo(this.p.offset(.5,.5,.5))<=4.3&&visibleBlock(block(this.p),eye)}
}
async function navigate(j,c){check(j);const p=vec(c.position);if(c.kind==='block'&&bot.entity.position.offset(0,1.62,0).distanceTo(p.offset(.5,.5,.5))<=4.3&&visibleBlock(block(p)))return;j.phase='navigate';await bot.pathfinder.goto(c.kind==='stand'?new goals.GoalBlock(p.x,p.y,p.z):new InteractionGoal(p));check(j)}
async function equip(j,name,dest='hand'){check(j);const i=item(name);if(!i)throw Error('missing_item');await bot.equip(i,dest);check(j);await audit(j)}
async function run(j,parts){
  const [verb,...a]=parts;check(j)
  const number=(s,min,max)=>{const v=Number(s);if(s===undefined||!Number.isFinite(v)||v<min||v>max)throw Error('invalid_number');return v}
  const arity=(lo,hi=lo)=>{if(a.length<lo||a.length>hi)throw Error('bad_arguments')}
  if(verb==='navigate'){arity(1);await navigate(j,candidate(a[0]));return}
  if(verb==='look_at'){arity(3);const p=vec(a.map(n=>number(n,-3e7,3e7)));if(distance(p)>32)throw Error('out_of_range');if(Object.values(bot.entities).some(e=>e.name==='enderman'&&e.position.distanceTo(p)<2))throw Error('unsafe_enderman_gaze');await bot.lookAt(p);return}
  if(verb==='equip'){arity(1,2);if(a[1]&&!['hand','off-hand','head','torso','legs','feet'].includes(a[1]))throw Error('invalid_destination');await equip(j,a[0],a[1]);return}
  if(verb==='gather'){
    arity(1,2);const c=candidate(a[0],'block'),limit=a[1]===undefined?1:number(a[1],1,4);if(!Number.isInteger(limit))throw Error('integer_required')
    let target=c
    for(let n=0;n<limit;n++){
      await navigate(j,target);check(j);const p=vec(target.position),b=block(p);if(!b||b.name!==c.name)throw Error('target_changed')
      if(hazards.has(b.name)||!exposed(b))throw Error('unsafe_or_hidden_target')
      if(p.y<Math.floor(bot.entity.position.y)&&Math.abs(p.x-Math.floor(bot.entity.position.x))<1&&Math.abs(p.z-Math.floor(bot.entity.position.z))<1)throw Error('cannot_mine_support')
      const tools=bot.inventory.items().filter(i=>b.canHarvest(i.type)).sort((x,y)=>b.digTime(x.type,false,false,false)-b.digTime(y.type,false,false,false))
      if(tools.length)await equip(j,tools[0].name)
      if(!b.canHarvest(bot.heldItem?.type??null))throw Error('wrong_tool')
      if(!bot.canDigBlock(b)||!visibleBlock(b))throw Error('out_of_reach_or_occluded')
      for(const d of [[0,1,0],[1,0,0],[-1,0,0],[0,0,1],[0,0,-1]]){const near=block(p.plus(vec(d)));if(near&&(hazards.has(near.name)||['sand','gravel'].includes(near.name)))throw Error('unsafe_dig')}
      const before=counts();j.phase='dig';await bot.lookAt(b.position.offset(.5,.5,.5));check(j);await bot.dig(b,'ignore');check(j);if(block(p)?.type===b.type)throw Error('dig_not_confirmed')
      j.phase='pickup'
      const hasGain=()=>Object.entries(counts()).some(([name,count])=>count>(before[name]||0))
      const pickupEnd=now()+4
      while(!hasGain()&&now()<pickupEnd){
        check(j)
        const drops=Object.values(bot.entities).filter(e=>e.name==='item'&&e.position.distanceTo(p.offset(.5,.5,.5))<5).sort((x,y)=>distance(x.position)-distance(y.position))
        if(drops[0]){
          const d=drops[0].position,center=d.floored(),pickupGoals=[]
          for(let dx=-1;dx<=1;dx++)for(let dz=-1;dz<=1;dz++)for(let dy=-1;dy<=0;dy++){
            const stand=center.offset(dx,dy,dz)
            if(Math.abs(stand.x+.5-d.x)<=1.2&&Math.abs(stand.z+.5-d.z)<=1.2&&safeStand(stand))pickupGoals.push(new goals.GoalBlock(stand.x,stand.y,stand.z))
          }
          if(pickupGoals.length){try{await bot.pathfinder.goto(new goals.GoalCompositeAny(pickupGoals))}catch(e){if(j.cancelled)throw e}}
        }
        if(!hasGain())await pause(j,100)
      }
      check(j);await audit(j);if(!hasGain())throw Error('no_drop_or_inventory_full')
      if(n+1<limit){updateCandidates();target=candidates.find(t=>t.kind==='block'&&t.name===c.name);if(!target)break}
    }return
  }
  if(verb==='craft'){
    arity(2,3);const requested=number(a[1],1,64);if(!Number.isInteger(requested))throw Error('integer_required')
    const id=bot.registry.itemsByName[a[0]]?.id;if(!id)throw Error('unknown_item')
    let table=null;if(a[2]){const c=candidate(a[2],'block');if(c.name!=='crafting_table')throw Error('not_a_table');await navigate(j,c);table=block(vec(c.position))}else table=bot.findBlock({matching:bot.registry.blocksByName.crafting_table.id,maxDistance:4})
    const recipe=bot.recipesFor(id,null,requested,table)[0];if(!recipe)throw Error('missing_materials_or_table')
    const ops=Math.ceil(requested/recipe.result.count);j.phase='craft'
    for(let i=0;i<ops;i++){check(j);await bot.craft(recipe,1,table);check(j);await audit(j)}return
  }
  if(verb==='place'){
    arity(3);const c=candidate(a[1],'block'),faces={up:[0,1,0],down:[0,-1,0],east:[1,0,0],west:[-1,0,0],south:[0,0,1],north:[0,0,-1]};if(!faces[a[2]])throw Error('invalid_face')
    if(!bot.registry.blocksByName[a[0]])throw Error('not_a_placeable_block')
    await navigate(j,c);const p=vec(c.position),dest=p.plus(vec(faces[a[2]])),b=block(p)
    if(!empty(block(dest)))throw Error('occupied_destination')
    const bp=bot.entity.position;if(bp.x+.3>dest.x&&bp.x-.3<dest.x+1&&bp.z+.3>dest.z&&bp.z-.3<dest.z+1&&bp.y+1.8>dest.y&&bp.y<dest.y+1)throw Error('self_collision')
    if(!visibleBlock(b))throw Error('occluded_reference')
    await equip(j,a[0]);j.phase='place';await bot.placeBlock(b,vec(faces[a[2]]));check(j);await audit(j);if(block(dest)?.name!==a[0])throw Error('placement_not_confirmed');return
  }
  if(verb==='furnace_start'||verb==='furnace_take'){
    arity(verb==='furnace_start'?5:1);const c=candidate(a[0],'block');if(c.name!=='furnace')throw Error('not_a_furnace');await navigate(j,c)
    const f=await bot.openFurnace(block(vec(c.position)));check(j)
    try{if(verb==='furnace_start'){const input=item(a[1]),fuel=item(a[3]),n=number(a[2],1,64),m=number(a[4],1,64);if(!Number.isInteger(n)||!Number.isInteger(m)||!input||!fuel||input.count<n||fuel.count<m)throw Error('missing_materials');await f.putInput(input.type,null,n);check(j);await audit(j);await f.putFuel(fuel.type,null,m);check(j);await audit(j)}else{if(!f.outputItem())throw Error('no_output_yet');await f.takeOutput();check(j);await audit(j)}}finally{const desc=i=>i?{name:i.name,count:i.count}:null;workstations.set(c.id,{id:c.id,position:c.position,last_observed_at:now(),contents:{input:desc(f.inputItem()),fuel:desc(f.fuelItem()),output:desc(f.outputItem())},progress_at_observation:f.progress,live:false});f.close()}return
  }
  if(verb==='eat'){arity(1);if(bot.food>=20)throw Error('not_hungry');if(!bot.registry.foodsByName[a[0]])throw Error('not_food');await equip(j,a[0]);await bot.consume();check(j);await audit(j);return}
  if(verb==='attack'){arity(1);const id=number(a[0],0,2**31),e=bot.entities[id];if(!e||distance(e.position)>3)throw Error('target_out_of_reach');const eye=bot.entity.position.offset(0,1.62,0),aim=e.position.offset(0,.8,0),delta=aim.minus(eye);const hit=bot.world.raycast(eye,delta.unit(),delta.norm());if(hit)throw Error('occluded_target');const delay=bot.heldItem?.name.endsWith('_axe')?1.25:.65;if(now()-lastAttack<delay)throw Error('attack_cooldown');await bot.lookAt(aim);check(j);bot.attack(e);lastAttack=now();return}
  if(verb==='shield'){arity(1);if(a[0]==='off'){bot.deactivateItem();return}if(a[0]!=='on')throw Error('expected_on_or_off');await equip(j,'shield','off-hand');bot.activateItem(true);return}
  if(verb==='wait'){arity(0,1);const seconds=a.length?number(a[0],0,5):.5;await pause(j,seconds*1000);check(j);return}
  throw Error('unknown_action')
}
async function action(text){
  if(!active||dead)return {accepted:false,error:'not_playing'}
  if(typeof text!=='string'||text.length>256)return {accepted:false,error:'action_must_be_short_string'}
  const parts=text.trim().split(/\s+/),verb=parts[0]
  if(verb==='noop'){if(!job)bot.deactivateItem();return {accepted:true}}
  if(verb==='cancel'){cancel();return {accepted:true}}
  if(job)return {accepted:false,error:'busy; cancel first'}
  bot.deactivateItem()
  const j={id:++jobCounter,action:text,phase:'starting',deadline:now()+(verb==='gather'?25:verb==='navigate'?15:10),cancelled:false};job=j
  run(j,parts).then(()=>{lastResult={id:j.id,action:text,status:'succeeded'}}).catch(e=>{lastResult={id:j.id,action:text,status:j.cancelled?'cancelled':'failed',error:String(e.message).slice(0,200)}}).finally(()=>{if(verb!=='shield'||j.cancelled)stopControls();job=null;lastCandidateUpdate=-Infinity;send({type:'action_result',...lastResult});send(snapshot())})
  return {accepted:true,job_id:j.id}
}
bot.once('spawn',async()=>{
  try{
    const rawWrite=bot._client.write.bind(bot._client)
    bot._client.write=(name,data)=>{if(job?.cancelled&&(name==='window_click'||name==='block_place'||name==='use_item'))throw Error('cancelled');return rawWrite(name,data)}
    const originalClick=bot.clickWindow.bind(bot)
    bot.clickWindow=async(...args)=>{if(job&&(job.cancelled||!active))throw Error('cancelled');const result=await originalClick(...args);if(active)await audit(job);return result}
    const movements=new Movements(bot);movements.canDig=false;movements.allow1by1towers=false;movements.allowParkour=false;movements.allowSprinting=false;movements.maxDropDown=2;movements.infiniteLiquidDropdownDistance=false;movements.scafoldingBlocks=[];movements.dontCreateFlow=true;movements.dontMineUnderFallingBlock=true
    for(const name of hazards){const b=bot.registry.blocksByName[name];if(b)movements.blocksToAvoid.add(b.id)}
    bot.pathfinder.setMovements(movements);bot.pathfinder.thinkTimeout=3000;bot.pathfinder.tickTimeout=8;bot.pathfinder.searchRadius=48
    await bot.waitForChunksToLoad();ready=true;send({type:'ready',position:pos(bot.entity.position)});send(snapshot())
  }catch(e){send({type:'infrastructure_error',error:String(e)})}
})
let inventoryAuditScheduled=false
for(const packet of ['set_slot','window_items','collect'])bot._client.on(packet,()=>{
  if(!active||inventoryAuditScheduled)return
  inventoryAuditScheduled=true
  setTimeout(()=>{inventoryAuditScheduled=false;if(active)audit(null).catch(e=>send({type:'infrastructure_error',error:String(e)}))},10)
})
bot.on('time',()=>{lastTime=now()});bot.on('physicsTick',()=>{physics++})
bot.on('death',()=>{dead=true;cancel('death');send({type:'death',observation:snapshot()})})
bot.on('error',e=>send({type:'infrastructure_error',error:String(e)}))
bot.on('kicked',r=>{if(!closing)send({type:'infrastructure_error',error:'kicked: '+JSON.stringify(r)})})
bot.on('end',r=>{if(!closing)send({type:'infrastructure_error',error:'disconnected: '+r});process.exit(closing?0:1)})
setInterval(()=>{if(ready&&active)try{send(snapshot())}catch(e){send({type:'infrastructure_error',error:'snapshot: '+e})}},200)
setInterval(()=>{
  if(!ready||!active||dead)return
  if(job&&now()>job.deadline&&!job.cancelled)cancel('timeout')
  if(job?.cancelled&&now()>job.cancelledAt+6){send({type:'infrastructure_error',error:'job_failed_to_cancel'});closing=true;bot.quit();process.exitCode=1}
  // Independent neutralization at immediate liquid/fire exposure. No autonomous mining/combat.
  const p=bot.entity.position.floored(),feet=block(p)
  if(feet&&hazards.has(feet.name)){cancel('hazard');if(feet.name==='water')bot.setControlState('jump',true)}
},50)
const rl=readline.createInterface({input:process.stdin})
rl.on('line',async line=>{
  let m
  try{
    if(line.length>65536)throw Error('oversized_message');m=JSON.parse(line);let result
    if(m.cmd==='act')result=await action(m.action)
    else if(m.cmd==='observe')result=snapshot()
    else if(m.cmd==='activate'){active=true;updateCandidates();result=snapshot()}
    else if(m.cmd==='deactivate'){active=false;cancel('episode_end');result=snapshot()}
    else if(m.cmd==='qualify'){
      if(active)throw Error('preparation_only')
      const deadline=now()+25
      while(true){
        let loaded=Math.abs(bot.entity.position.x-m.x-.5)<1&&Math.abs(bot.entity.position.z-m.z-.5)<1
        for(let x=(m.x-50)>>4;x<=(m.x+50)>>4;x++)for(let z=(m.z-50)>>4;z<=(m.z+50)>>4;z++)if(!bot.world.getColumn(x,z))loaded=false
        if(loaded)break
        if(now()>deadline)throw Error('qualification_chunks_timeout')
        await sleep(100)
      }
      result=qualify(bot,m.x,m.z)
    }
    else if(m.cmd==='audit_ack'){audits.get(m.audit_id)?.resolve();audits.delete(m.audit_id);return}
    else if(m.cmd==='close'){closing=true;active=false;cancel();bot.quit();setTimeout(()=>process.exit(0),500);result=true}
    else throw Error('unknown_command')
    send({type:'response',id:m.id,result})
  }catch(e){send({type:'response',id:m?.id,error:String(e.message)})}
})
rl.on('close',()=>{closing=true;active=false;cancel();bot.quit();setTimeout(()=>process.exit(),500)})
