-- MIT licensed bridge. All economic production is performed by vanilla machines.
local N=128
local ROLES={fuel=defines.inventory.fuel,source=defines.inventory.furnace_source,result=defines.inventory.furnace_result,input=defines.inventory.assembling_machine_input,output=defines.inventory.assembling_machine_output,lab=defines.inventory.lab_input,chest=defines.inventory.chest}
local SCIENCE={['automation-science-pack']=1,['logistic-science-pack']=2,['chemical-science-pack']=6,['production-science-pack']=12,['utility-science-pack']=12,['space-science-pack']=30}
local function stock() return storage.stock end
local function surface() return game.surfaces[1] end
local function force() return game.forces.player end
local function contents(inv)
 local t={} if inv and inv.valid then for _,v in pairs(inv.get_contents()) do t[v.name]=(t[v.name] or 0)+v.count end end return t
end
local function register(e)
 local id=storage.next_id; storage.next_id=id+1; storage.entities[id]=e; return id
end
local function entity(id)
 local e=storage.entities[id]; if not e or not e.valid then error('unknown or stale entity ID') end return e
end
local function inside(p) return p.x>=-N and p.x<N and p.y>=-N and p.y<N end
local function num(x,lo,hi)
 if type(x)~='number' or x~=x or x<lo or x>hi then error('number outside allowed range') end return x
end
local function integer(x,lo,hi) num(x,lo,hi); if x~=math.floor(x) then error('integer required') end return x end
local function addledger(t,name,n) t[name]=(t[name] or 0)+n end
local function stats()
 local st=force().get_item_production_statistics(surface())
 local a,b={},{}
 for name,_ in pairs(st.input_counts) do a[name]=st.get_input_count({name=name,quality='normal'}) end
 for name,_ in pairs(st.output_counts) do b[name]=st.get_output_count({name=name,quality='normal'}) end
 return a,b
end
local function coord(u,v,rot)
 if rot==0 then return u,v elseif rot==1 then return -v,u elseif rot==2 then return -u,-v else return v,-u end
end
script.on_init(function()
 assert(not script.active_mods['space-age'] and not script.active_mods.quality and not script.active_mods['elevated-rails'],'expansion mods must be disabled')
 local s=surface(); local seed=s.map_gen_settings.seed
 storage.seed=seed; storage.entities={}; storage.next_id=1; storage.jobs={}; storage.manual={}; storage.transfers=0; storage.turn=0; storage.used=0; storage.results={}; storage.research_events={}; storage.lab_consumed={}; storage.last_consumed={}; storage.page=0;storage.trigger_manual={}
 s.request_to_generate_chunks({0,0},4); s.force_generate_chunk_requests()
 for _,e in pairs(s.find_entities()) do e.destroy() end
 local rng=game.create_random_generator(seed); local rot=rng(0,3); local phase=rng()*6.28; local coast=rng(89,104)
 local tiles={}; storage.water={}
 for y=-N,N-1 do
  local runs={}; local start=nil
  for x=-N,N-1 do
   local u,v=coord(x,y,(4-rot)%4)
   local water=u>coast+7*math.sin(v/19+phase)+3*math.sin(v/7)
   tiles[#tiles+1]={name=water and 'water' or 'grass-1',position={x,y}}
   if water and not start then start=x end
   if start and (not water or x==N-1) then runs[#runs+1]={start,water and x or x-1}; start=nil end
  end
  if #runs>0 then storage.water[#storage.water+1]={y=y,runs=runs} end
 end
 s.set_tiles(tiles,true,false,false)
 local names={'iron-ore','copper-ore','coal','stone'}
 for i=4,2,-1 do local j=rng(1,i); names[i],names[j]=names[j],names[i] end
 storage.patches={}
 for i,name in ipairs(names) do
  local angle=(i-1)*math.pi/2+0.5+(rng()-0.5)*0.45
  local dist=rng(30,65); local cx=math.floor(math.cos(angle)*dist); local cy=math.floor(math.sin(angle)*dist)
  local rx=rng(7,11); local ry=rng(6,10); local total=0; local count=0
  for y=cy-ry-1,cy+ry+1 do for x=cx-rx-1,cx+rx+1 do
   if ((x-cx)/rx)^2+((y-cy)/ry)^2<0.8+rng()*0.4 then
    local amount=rng(1800,3200); local e=s.create_entity{name=name,position={x+0.5,y+0.5},amount=amount}
    if e then register(e); total=total+amount;count=count+1 end
   end
  end end
  assert(total>200000,'insufficient resources')
  storage.patches[#storage.patches+1]={name=name,center={x=cx,y=cy},radius={rx,ry},initial_amount=total,tiles=count}
 end
 for i=1,240 do
  local x,y=rng(-120,120),rng(-120,120)
  if math.abs(x)>12 or math.abs(y)>12 then
   if s.can_place_entity{name='tree-01',position={x+0.5,y+0.5}} and #s.find_entities_filtered{position={x+0.5,y+0.5},radius=2,type='resource'}==0 then
    register(s.create_entity{name='tree-01',position={x+0.5,y+0.5}})
   end
  end
 end
 -- Actual coastal placement is validated by the engine. Deterministic scan.
 storage.pump_site=nil
 for y=-120,120 do
  if storage.pump_site then break end
  for x=-120,120 do
   if storage.pump_site then break end
   for _,d in ipairs({0,4,8,12}) do
    if s.can_place_entity{name='offshore-pump',position={x+0.5,y+0.5},direction=d,force=force(),build_check_type=defines.build_check_type.manual} then storage.pump_site={x=x+0.5,y=y+0.5,direction=d};break end
   end
  end
 end
 assert(storage.pump_site,'no valid pump site')
 storage.stock=game.create_inventory(200)
 for name,n in pairs({['burner-mining-drill']=1,['stone-furnace']=1,['iron-plate']=8,wood=1}) do stock().insert{name=name,count=n} end
 game.map_settings.pollution.enabled=false; game.map_settings.enemy_expansion.enabled=false
 storage.baseline_p,storage.baseline_c=stats();storage.start_tick=game.tick;storage.last_work_tick=game.tick
 game.speed=20; game.tick_paused=true
end)
local function recipe_info(r)
 local ins,outs={},{}
 for _,v in pairs(r.ingredients) do ins[#ins+1]={name=v.name,type=v.type,amount=v.amount} end
 for _,v in pairs(r.products) do outs[#outs+1]={name=v.name,type=v.type,amount=v.amount,amount_min=v.amount_min,amount_max=v.amount_max,probability=v.probability} end
 return {name=r.name,enabled=r.enabled,category=r.category,seconds=r.energy,ingredients=ins,products=outs}
end
local function technology_info(t)
 local pre={} for name,p in pairs(t.prerequisites) do pre[#pre+1]={name=name,researched=p.researched} end table.sort(pre,function(a,b)return a.name<b.name end)
 return {name=t.name,researched=t.researched,enabled=t.enabled,prerequisites=pre,ingredients=t.research_unit_ingredients,units=t.research_unit_count,seconds=t.research_unit_energy/60,trigger=t.prototype.research_trigger}
end
local function role_inventory(e,role)
 local typ=e.type
 if role=='fuel' and not e.burner then return nil end
 if (role=='source' or role=='result') and typ~='furnace' then return nil end
 if (role=='input' or role=='output') and typ~='assembling-machine' then return nil end
 if role=='lab' and typ~='lab' then return nil end
 if role=='chest' and typ~='container' and typ~='logistic-container' then return nil end
 return e.get_inventory(ROLES[role])
end
local STATUS={} for name,id in pairs(defines.entity_status) do STATUS[id]=name end
local function view_entity(id,e)
 local r={id=id,name=e.name,type=e.type,position=e.position,direction=e.direction,box=e.bounding_box,health=e.health,status=e.status,energy_j=e.energy,inventories={}}
 for role,index in pairs(ROLES) do local inv=role_inventory(e,role); if inv then r.inventories[role]=contents(inv) end end
 if e.type=='assembling-machine' or e.type=='furnace' then local recipe=e.get_recipe(); r.recipe=recipe and recipe.name; r.crafting_progress=e.crafting_progress end
 if e.burner then r.burner={remaining_j=e.burner.remaining_burning_fuel,fuel=e.burner.currently_burning and e.burner.currently_burning.name} end
 r.fluids={};for i=1,#e.fluidbox do local f=e.fluidbox[i];if f then r.fluids[#r.fluids+1]={index=i,name=f.name,amount=f.amount,temperature=f.temperature} end end
 if e.type=='transport-belt' then r.belt_lines={e.get_transport_line(1).get_contents(),e.get_transport_line(2).get_contents()} end
 r.electric_network_id=e.electric_network_id;r.status_name=STATUS[e.status]
 r.prototype={collision_box=e.prototype.collision_box,}
 if e.type=='mining-drill' then r.prototype.mining_speed=e.prototype.mining_speed end
 if e.type=='assembling-machine' or e.type=='furnace' then r.prototype.crafting_speed=e.crafting_speed;r.prototype.crafting_categories=e.prototype.crafting_categories end
 return r
end
local function observe()
 local p,c=stats(); local recipes={};local techs={};local completed={}
 for _,r in pairs(force().recipes) do if r.enabled and not r.hidden and not r.prototype.is_parameter then recipes[#recipes+1]=recipe_info(r) end end
 table.sort(recipes,function(a,b)return a.name<b.name end)
 for name,t in pairs(force().technologies) do
  if t.researched then completed[#completed+1]=name elseif t.enabled then
   local ok=true;for _,pre in pairs(t.prerequisites) do if not pre.researched then ok=false end end
   if ok then techs[#techs+1]=technology_info(t) end
  end
 end
 table.sort(completed);table.sort(techs,function(a,b)return a.name<b.name end)
 local resources={};local trees={};local buildings={};local total=0
 for id=1,storage.next_id-1 do local e=storage.entities[id]
  if e and e.valid then
   if e.type=='resource' then resources[#resources+1]={id,e.name,e.position.x,e.position.y,e.amount}
   elseif e.type=='tree' then trees[#trees+1]={id,e.position.x,e.position.y}
   else total=total+1; if total>storage.page and total<=storage.page+100 then buildings[#buildings+1]=view_entity(id,e) end end
  end
 end
 return {schema=1,ruleset='cold-coast-v1',engine='2.0.72',seed=storage.seed,turn=storage.turn,remaining_turns=128-storage.turn,tick=game.tick,charged_ticks=game.tick-storage.start_tick,paused=game.tick_paused,stock=contents(stock()),stock_slots=200,stock_empty_slots=stock().count_empty_stacks(),queue=storage.jobs,manual_output=storage.manual,transfers=storage.transfers,transfer_remaining=100-storage.used,results=storage.results,bounds={-N,-N,N,N},water_rows=storage.water,patches=storage.patches,pump_site=storage.pump_site,resources=resources,resource_columns={'id','name','x','y','amount'},trees=trees,tree_columns={'id','x','y'},entities=buildings,entity_total=total,entity_offset=storage.page,enabled_recipes=recipes,available_research=techs,research={current=force().current_research and force().current_research.name,progress=force().research_progress,completed=completed,events=storage.research_events},production=p,consumption=c,baseline_production=storage.baseline_p,baseline_consumption=storage.baseline_c,lab_consumption=storage.lab_consumed}
end
local function enqueue(job)
 if #storage.jobs>=32 then error('manual queue: 32-job limit') end
 local ticks=job.ticks*job.count;for _,j in ipairs(storage.jobs) do ticks=ticks+(j.remaining or j.ticks)+(j.count-1)*j.ticks end
 if ticks>3600 then error('manual queue: 3600-tick limit') end
 job.remaining=job.ticks;storage.jobs[#storage.jobs+1]=job
end
local function apply(a)
 local op=a[1]
 if op=='wait' then return {ok=true} end
 if op=='handcraft' then
  local r=force().recipes[a[2]];local n=integer(a[3],1,1000)
  if not r or not r.enabled or r.hidden or r.prototype.is_parameter or r.prototype.hidden_from_player_crafting or not prototypes.entity.character.crafting_categories[r.category] then error('locked or non-handcraftable recipe') end
  local outs={};for _,v in pairs(r.products) do if v.type~='item' or not v.amount or v.probability and v.probability~=1 then error('unsupported manual product') end;outs[#outs+1]={name=v.name,count=v.amount} end
  for _,v in pairs(r.ingredients) do if v.type~='item' or stock().get_item_count(v.name)<v.amount*n then error('missing ingredients or fluid recipe') end end
  local job={kind=op,recipe=r.name,ticks=math.max(1,math.ceil(r.energy*60)),count=n,products=outs}
  enqueue(job)
  for _,v in pairs(r.ingredients) do stock().remove{name=v.name,count=v.amount*n} end
  return {ok=true,queued=n}
 elseif op=='mine_resource' or op=='mine_tree' then
  local e=entity(integer(a[2],1,storage.next_id));local n=op=='mine_tree' and 1 or integer(a[3],1,1000)
  if op=='mine_tree' and e.type~='tree' or op=='mine_resource' and (e.type~='resource' or not ({['iron-ore']=true,['copper-ore']=true,coal=true,stone=true})[e.name]) then error('wrong entity for mining') end
  local props=e.prototype.mineable_properties
  enqueue{kind=op,id=a[2],ticks=math.max(1,math.ceil(props.mining_time*60/prototypes.entity.character.mining_speed)),count=n}
  return {ok=true,queued=n}
 elseif op=='place' then
  local item=prototypes.item[a[2]];if not item or not item.place_result then error('item has no supported place result') end
  if stock().get_item_count(item.name)<1 then error('no building item in stock') end
  local p={x=num(a[3],-N,N-1),y=num(a[4],-N,N-1)};local d=integer(a[5] or 0,0,15)
  if d%4~=0 then error('direction must be 0,4,8,12') end
  local proto=item.place_result;local box=proto.collision_box
  local margin=math.max(math.abs(box.left_top.x),math.abs(box.left_top.y),math.abs(box.right_bottom.x),math.abs(box.right_bottom.y))
  if p.x-margin<-N or p.y-margin<-N or p.x+margin>N or p.y+margin>N then error('footprint outside bounds') end
  if not surface().can_place_entity{name=proto.name,position=p,direction=d,force=force(),build_check_type=defines.build_check_type.manual} then error('blocked placement or invalid terrain') end
  local e=surface().create_entity{name=proto.name,position=p,direction=d,force=force(),create_build_effect_smoke=false}
  if not e then error('placement failed') end
  local actual=e.bounding_box
  if actual.left_top.x<-N or actual.left_top.y<-N or actual.right_bottom.x>N or actual.right_bottom.y>N then e.destroy();error('snapped footprint outside bounds') end
  stock().remove{name=item.name,count=1};return {ok=true,id=register(e),position=e.position}
 elseif op=='insert' or op=='extract' then
  local e=entity(integer(a[2],1,storage.next_id));if e.force~=force() then error('not an owned building') end
  local role=ROLES[a[3]];if not role then error('unknown inventory role') end
  local inv=role_inventory(e,a[3]); if not inv then error('inventory role unavailable') end
  if not prototypes.item[a[4]] then error('unknown item') end
  local count=integer(a[5],1,100000);local src=op=='insert' and stock() or inv;local dest=op=='insert' and inv or stock()
  count=math.min(count,100-storage.used,src.get_item_count(a[4]));if count<=0 then error('empty source or transfer quota exhausted') end
  local moved=0;local temp=game.create_inventory(1)
  for i=1,#src do
   local stack=src[i]
   if moved<count and stack.valid_for_read and stack.name==a[4] and stack.quality.name=='normal' then
    temp[1].set_stack(stack);temp[1].count=math.min(stack.count,count-moved)
    local n=dest.insert(temp[1]);stack.count=stack.count-n;moved=moved+n;temp.clear()
   end
  end
  temp.destroy()
  storage.used=storage.used+moved;storage.transfers=storage.transfers+moved;return {ok=true,moved=moved}
 elseif op=='rotate' then
  local e=entity(integer(a[2],1,storage.next_id));if e.force~=force() then error('not owned') end
  return {ok=e.rotate(),direction=e.direction}
 elseif op=='set_recipe' then
  local e=entity(integer(a[2],1,storage.next_id));local r=force().recipes[a[3]]
  if e.type~='assembling-machine' or e.force~=force() or not r or not r.enabled or r.hidden or r.prototype.is_parameter or not e.prototype.crafting_categories[r.category] then error('locked or incompatible recipe') end
  -- Conservative empty-machine rule prevents cargo/partial-work duplication.
  for _,idx in pairs({defines.inventory.assembling_machine_input,defines.inventory.assembling_machine_output}) do local inv=e.get_inventory(idx);if inv and not inv.is_empty() then error('extract input/output before changing recipe') end end
  if e.crafting_progress~=0 then error('finish current recipe before changing') end
  for i=1,#e.fluidbox do if e.fluidbox[i] then error('fluid-filled machine cannot change recipe') end end
  e.set_recipe(r.name);return {ok=true}
 elseif op=='mine_building' then
  local e=entity(integer(a[2],1,storage.next_id));if e.force~=force() or e.type=='character' then error('not an owned building') end
  -- Empty buildings only: recover cargo through quota-limited extract first.
  for _,i in pairs(defines.inventory) do local inv=e.get_inventory(i);if inv and not inv.is_empty() then error('extract all cargo before mining building') end end
  if (e.type=='assembling-machine' or e.type=='furnace') and e.crafting_progress~=0 then error('finish current recipe before mining') end
  if e.type=='transport-belt' and (#e.get_transport_line(1)>0 or #e.get_transport_line(2)>0) then error('belt has cargo') end
  if e.type=='inserter' and e.held_stack.valid_for_read then error('inserter has cargo') end
  return {ok=e.mine{inventory=stock(),force=false},note='Remaining fluids and burning fuel are lost, as in engine mining.'}
 elseif op=='set_research' then
  local t=force().technologies[a[2]];if not t or not t.enabled or t.researched then error('unknown, locked or completed technology') end
  for _,p in pairs(t.prerequisites) do if not p.researched then error('prerequisite missing') end end
  if t.prototype.research_trigger then error('this technology unlocks automatically through its trigger') end
  force().research_queue={t.name}
  return {ok=force().current_research~=nil and force().current_research.name==t.name}
 else error('unknown action') end
end
local function manual_trigger(name,count)
 -- Virtual handcrafts do not touch production statistics; mirror only the native
 -- craft-item research trigger, after ingredients/time and prerequisites are paid.
 for _,t in pairs(force().technologies) do
  local trigger=t.prototype.research_trigger
  local trigger_item=trigger and trigger.item
  local trigger_name=type(trigger_item)=='table' and trigger_item.name or trigger_item
  if t.enabled and not t.researched and trigger and trigger.type=='craft-item' and trigger_name==name then
   local ready=true;for _,pre in pairs(t.prerequisites) do if not pre.researched then ready=false end end
   if ready then
    addledger(storage.trigger_manual,t.name,count)
    if storage.trigger_manual[t.name]>=(trigger.count or 1) then
     storage.manual_trigger_active=true;t.researched=true;storage.manual_trigger_active=nil
    end
   end
  end
 end
end
local function manual_tick()
 local j=storage.jobs[1];if not j then return end
 if j.remaining>0 then j.remaining=j.remaining-1 end
 if j.remaining>0 then return end
 if j.kind=='handcraft' then
  -- Test simultaneous capacity in a temporary engine inventory.
  local test=game.create_inventory(#stock()); for i=1,#stock() do if stock()[i].valid_for_read then test[i].set_stack(stock()[i]) end end
  local fits=true;for _,v in ipairs(j.products) do if test.insert(v)~=v.count then fits=false end end;test.destroy()
  if not fits then j.blocked='stock full';return end
  for _,v in ipairs(j.products) do stock().insert(v);addledger(storage.manual,v.name,v.count);manual_trigger(v.name,v.count) end
 else
  local e=storage.entities[j.id]
  if not e or not e.valid then table.remove(storage.jobs,1);return end
  if j.kind=='mine_resource' then
   local products=e.prototype.mineable_properties.products;local product=products[1]
   if stock().get_insertable_count(product.name)<product.amount then j.blocked='stock full';return end
   stock().insert{name=product.name,count=product.amount};addledger(storage.manual,product.name,product.amount)
   if e.amount<=1 then e.destroy() else e.amount=e.amount-1 end
  else
   local before=contents(stock());if not e.mine{inventory=stock(),force=false} then j.blocked='stock full';return end
   for name,n in pairs(contents(stock())) do if n>(before[name] or 0) then addledger(storage.manual,name,n-(before[name] or 0)) end end
  end
 end
 j.count=j.count-1;j.remaining=j.ticks;j.blocked=nil;if j.count==0 then table.remove(storage.jobs,1) end
end
script.on_event(defines.events.on_research_finished,function(ev)
 storage.research_events[#storage.research_events+1]={name=ev.research.name,tick=game.tick,source=storage.manual_trigger_active and 'paid-manual-craft-trigger' or 'engine'}
end)
script.on_event(defines.events.on_tick,function()
 if not storage.target then game.tick_paused=true;return end
 if game.tick>storage.last_work_tick then
  storage.last_work_tick=game.tick;manual_tick()
 end
 local st=force().get_item_production_statistics(surface())
 for name,_ in pairs(SCIENCE) do
  local c=st.get_output_count({name=name,quality='normal'});local old=storage.last_consumed[name] or 0
  if c>old then
   -- Base-only science has no recipe sink. Only owned, operating labs can consume it.
   assert(surface().count_entities_filtered{type='lab',force=force()}>0,'science consumed without labs')
   addledger(storage.lab_consumed,name,c-old)
  end
  storage.last_consumed[name]=c
 end
 if game.tick>=storage.target then storage.target=nil;game.tick_paused=true end
end)
commands.add_command('mh','Private typed bridge',function(event)
 local ok,result=pcall(function()
  local req=helpers.json_to_table(event.parameter or '');if not req then error('invalid JSON') end
  if req.method=='observe' then return observe()
  elseif req.method=='status' then return {turn=storage.turn,tick=game.tick,paused=game.tick_paused}
  elseif req.method=='page' then storage.page=integer(req.offset,0,100000);return observe()
  elseif req.method=='recipe' then local r=force().recipes[req.name];if not r then error('unknown recipe') end;return recipe_info(r)
  elseif req.method=='technology' then local t=force().technologies[req.name];if not t then error('unknown technology') end;return technology_info(t)
  elseif req.method=='step' then
   if req.turn==storage.turn then return {turn=storage.turn,duplicate=true} end
   if not game.tick_paused or req.turn~=storage.turn+1 or storage.turn>=128 then error('stale turn, running, or horizon reached') end
   if type(req.actions)~='table' or #req.actions>32 then error('invalid batch') end
   storage.used=0;storage.results={}
   for _,a in ipairs(req.actions) do local success,value=pcall(apply,a);storage.results[#storage.results+1]=success and value or {ok=false,error=tostring(value):gsub('^.-:%d+: ','')} end
   storage.turn=storage.turn+1;storage.target=game.tick+1800;game.tick_paused=false
   return {turn=storage.turn,target=storage.target}
  else error('unknown method') end
 end)
 local payload=helpers.table_to_json({ok=ok,result=ok and result or nil,error=not ok and tostring(result) or nil})
 -- Explicit bounded chunks, one RCON response per chunk; Python uses a sentinel command.
 for pos=1,#payload,3000 do rcon.print('MH:'..payload:sub(pos,pos+2999)) end
end)
