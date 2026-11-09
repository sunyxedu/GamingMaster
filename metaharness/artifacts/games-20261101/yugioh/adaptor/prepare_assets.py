"""Resolve the exact study recipes against the staged database. Setup only."""
import hashlib,json,sqlite3,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT/'runtime'
PACK={'Mystical Space Typhoon':2,'Book of Moon':2,'Fissure':2,'Dark Hole':1,'Monster Reborn':1,'Rush Recklessly':2,'Trap Hole':2,'Mirror Force':1,'Compulsory Evacuation Device':2,'Call of the Haunted':1}
MONSTERS={
'normal_pressure':['Alexandrite Dragon','Gene-Warped Warwolf','Gemini Elf','Luster Dragon','Vorse Raider','La Jinn the Mystical Genie of the Lamp','Battle Ox','Mystical Elf'],
'flip_control':['Alexandrite Dragon','Gene-Warped Warwolf','Gemini Elf','Mystical Elf','Giant Soldier of Stone','Man-Eater Bug','Penguin Soldier','Magician of Faith'],
'tribute_pressure':['Summoned Skull','Blue-Eyes White Dragon','Luster Dragon','Vorse Raider','La Jinn the Mystical Genie of the Lamp','Battle Ox','Mystical Elf','Giant Soldier of Stone']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=sqlite3.connect(R/'vendor/cdb/cards.cdb'); c.row_factory=sqlite3.Row
 cards={};decks={}
 for deck,monsters in MONSTERS.items():
  entries=[]
  for name,n in {**{s:3 for s in monsters},**PACK}.items():
   rows=c.execute('select d.*,t.* from datas d join texts t using(id) where name=? and alias=0 and d.id<100000000',(name,)).fetchall()
   assert len(rows)==1,(name,len(rows))
   row=dict(rows[0]);code=row['id'];entries += [code]*n
   if not row['type']&0x10:assert (R/f'vendor/scripts/official/c{code}.lua').exists(),name
   cards[str(code)]={'code':code,'name':name,'text':row['desc'],'type':row['type'],'level':row['level']&255,'attack':row['atk'],'defense':row['def'],'attribute':row['attribute'],'race':row['race'],'effects':[row[f'str{i}'] for i in range(1,17)]}
  assert len(entries)==40
  decks[deck]=entries
 with (R/'cards.tsv').open('w') as f:
  for row in c.execute('select id,alias,setcode,type,level,race,attribute,atk,def from datas order by id'):
   a=list(row);a[2]&=(1<<64)-1
   f.write(' '.join(map(str,a))+'\n')
 (R/'cards.json').write_text(json.dumps(cards,indent=2))
 (ROOT/'decks.json').write_text(json.dumps({'version':1,'format':'custom MR5, three copies, no banlist, empty Extra/Side','decks':decks,'cards':cards},indent=2))
 sources={}
 for name in ['core','scripts','cdb','edopro','windbot']:
  path=R/'vendor'/name
  sources[name]=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
 sources['lua']=subprocess.check_output(['git','-C',str(R/'vendor/core/lua/src'),'rev-parse','HEAD'],text=True).strip()
 sources['strings']='54a6e2395c532648ff762540e9615319fac4f51b'
 files=['bridge.cpp','runtime/build/libocgcore.so','runtime/libbridge.so','runtime/cards.tsv','runtime/cards.json','runtime/strings.conf','runtime/vendor/cdb/cards.cdb','decks.json']
 files += [str(p.relative_to(ROOT)) for p in sorted((R/'vendor/scripts').rglob('*.lua'))]
 files += ['opponent.v3.json','PoolExecutor.cs','WindbotBridge.cs','runtime/windbot/ExecutorBase.dll','runtime/windbot/WindbotBridge.exe','runtime/vendor/windbot/Dialogs/default.json','runtime/vendor/windbot/LICENSE','runtime/vendor/windbot/COPYING']
 files += [str(p.relative_to(ROOT)) for p in sorted((R/'vendor/windbot/ExecutorBase').rglob('*.cs')) if not {'obj','bin'}&set(p.parts)]
 manifest={'version':3,'api':[11,0],'sources':sources,'rules':'DUEL_MODE_MR5','opponent':'windbot-fixed-pool-v3','files':{p:sha(ROOT/p) for p in files}}
 (ROOT/'assets.lock.json').write_text(json.dumps(manifest,indent=2))
 print('Resolved',len(cards),'cards; pinned',len(files),'files',sources)
if __name__=='__main__': main()
