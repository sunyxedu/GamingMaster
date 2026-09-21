// Privileged preparation only. This module's output never enters policy observations.
const {Vec3}=require('vec3')
const hazards=new Set(['water','lava','fire','soul_fire','magma_block','cactus','powder_snow','sweet_berry_bush','campfire','soul_campfire'])
const fullGround=new Set(['stone','granite','diorite','andesite','deepslate','tuff','calcite','dirt','grass_block','coarse_dirt','rooted_dirt','podzol','mycelium','moss_block','clay','sand','red_sand','gravel','terracotta','snow_block','sandstone','red_sandstone'])
const logs=new Set(['oak','spruce','birch','jungle','acacia','dark_oak','mangrove','cherry'].map(s=>s+'_log'))
function qualify(bot, cx, cz) {
  const registry=bot.registry, cells=new Map(), cols=new Map()
  const key=(x,z)=>x+','+z
  const state=(x,y,z)=> {
    const k=key(x>>4,z>>4)
    if(!cols.has(k)) cols.set(k,bot.world.getColumn(x>>4,z>>4))
    const col=cols.get(k)
    if(!col || y < -64 || y>319) return null
    return registry.blocksByStateId[col.getBlockStateId(new Vec3((x%16+16)%16,y,(z%16+16)%16))]
  }
  const clear=b=>b&&b.boundingBox==='empty'&&!hazards.has(b.name)
  const solid=b=>b&&(fullGround.has(b.name)||b.name.endsWith('_terracotta'))
  for(let x=cx-50;x<=cx+50;x++) for(let z=cz-50;z<=cz+50;z++) {
    // Highest non-foliage ground; do not admit tree canopies or unknown columns.
    for(let y=319;y>=-60;y--) {
      const b=state(x,y,z)
      if(!b) break
      if(b.name.endsWith('_leaves')||logs.has(b.name)||b.boundingBox==='empty'&&!hazards.has(b.name)) continue
      if(solid(b)&&clear(state(x,y+1,z))&&clear(state(x,y+2,z))) cells.set(key(x,z),y+1)
      break
    }
  }
  // Bounded deterministic nearest dry cell, maximum eight blocks from candidate center.
  const starts=[]
  for(let dx=-8;dx<=8;dx++) for(let dz=-8;dz<=8;dz++) {
    const y=cells.get(key(cx+dx,cz+dz))
    if(y!==undefined) starts.push({x:cx+dx,y,z:cz+dz,d:dx*dx+dz*dz})
  }
  starts.sort((a,b)=>a.d-b.d||a.x-b.x||a.z-b.z)
  if(!starts.length) return {admitted:false,reason:'no_dry_surface',center:[cx,cz]}
  const start=starts[0], seen=new Set([key(start.x,start.z)]), queue=[start], logSet=new Set(), stoneSet=new Set()
  const safe=(x,y,z)=>{
    for(const [dx,dz] of [[0,0],[1,0],[-1,0],[0,1],[0,-1]])
      for(let dy=-1;dy<=1;dy++) if(hazards.has(state(x+dx,y+dy,z+dz)?.name)) return false
    return true
  }
  if(!safe(start.x,start.y,start.z)) return {admitted:false,reason:'unsafe_start',center:[cx,cz]}
  for(let i=0;i<queue.length;i++) {
    const p=queue[i]
    for(const [dx,dz] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const x=p.x+dx,z=p.z+dz,y=cells.get(key(x,z)),k=key(x,z)
      if(y===undefined||seen.has(k)||(x-start.x)**2+(y-start.y)**2+(z-start.z)**2>48**2||y-p.y>1||p.y-y>2||!safe(x,y,z)) continue
      // Head clearance across a step, not just at its destination.
      if(y>p.y&&!clear(state(p.x,p.y+2,p.z))) continue
      seen.add(k); queue.push({x,y,z})
    }
    for(let dx=-1;dx<=1;dx++) for(let dz=-1;dz<=1;dz++) {
      for(let dy=0;dy<=4;dy++) {
        const x=p.x+dx,y=p.y+dy,z=p.z+dz
        if((x-start.x)**2+(y-start.y)**2+(z-start.z)**2<=48**2&&logs.has(state(x,y,z)?.name)) logSet.add(`${x},${y},${z}`)
      }
    }
    // Ordinary stone under <=2 cover blocks; solid support prevents a cave drop.
    for(let dy=1;dy<=3;dy++) {
      const y=p.y-dy, b=state(p.x,y,p.z)
      if(!solid(b)) break
      if((p.x-start.x)**2+(y-start.y)**2+(p.z-start.z)**2<=48**2&&b.name==='stone'&&solid(state(p.x,y-1,p.z))&&solid(state(p.x,y-2,p.z))&&safe(p.x,y,p.z)) stoneSet.add(`${p.x},${y},${p.z}`)
    }
  }
  return {admitted:logSet.size>=12&&stoneSet.size>=24,reason:logSet.size<12?'insufficient_logs':stoneSet.size<24?'insufficient_shallow_stone':null,center:[cx,cz],spawn:[start.x+.5,start.y,start.z+.5],reachable_surface_cells:queue.length,reachable_logs:logSet.size,shallow_stone:stoneSet.size,log_witnesses:[...logSet].sort().slice(0,12),stone_witnesses:[...stoneSet].sort().slice(0,24)}
}
module.exports={qualify,hazards,logs}
