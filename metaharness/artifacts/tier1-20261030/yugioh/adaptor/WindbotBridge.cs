// SPDX-License-Identifier: AGPL-3.0-or-later
// A pipe transport for upstream WindBot GameAI, without a TCP host or GUI.
using System;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Web.Script.Serialization;
using WindBot.Game;
using WindBot.Game.AI;
using YGOSharp.OCGWrapper;
using YGOSharp.OCGWrapper.Enums;

public static class WindbotBridge
{
    static JavaScriptSerializer json=new JavaScriptSerializer();
    static Duel duel; static GameAI ai; static int seat; static int chainEpoch;
    static Dictionary<string,ClientCard> previous=new Dictionary<string,ClientCard>();
    static Dictionary<string,ClientCard> current;
    static Dictionary<string,object> D(object o) {return (Dictionary<string,object>)o;}
    static object[] A(object o) {return (object[])o;}
    static int N(Dictionary<string,object> d,string k,int def=0) {return d.ContainsKey(k)?Convert.ToInt32(d[k]):def;}
    static int Loc(string s) {switch(s) {case "deck":return 1;case "hand":return 2;case "monsters":return 4;case "spells_traps":return 8;case "graveyard":return 16;case "banished":return 32;case "extra":return 64;default:return 0;}}
    static int Pos(string s) {switch(s) {case "face-up attack":return 1;case "face-down attack":return 2;case "face-up defense":return 4;case "face-down defense":return 8;default:return 0;}}
    static void Field(BinaryWriter w,uint flag,int value) {w.Write((short)8);w.Write(flag);w.Write(value);}
    static ClientCard Card(object obj)
    {
        var d=D(obj); int player=N(d,"player",seat)^seat; int loc=d.ContainsKey("zone")?Loc((string)d["zone"]):0;
        int slot=N(d,"slot"); int code=N(d,"code"); string key=player+":"+loc+":"+slot;
        ClientCard c;
        if(current.TryGetValue(key,out c) && (code==0 || code==c.Id)) return c;
        if(!previous.TryGetValue(key,out c) || c.Id!=code) c=new ClientCard(code,(CardLocation)loc,slot,0,player);
        c.Controller=player;c.Location=(CardLocation)loc;c.Sequence=slot;
        var data=NamedCard.Get(code);
        using(var ms=new MemoryStream()) using(var w=new BinaryWriter(ms)) {
            Field(w,2,d.ContainsKey("position")?Pos((string)d["position"]):0);
            Field(w,8,N(d,"type",data==null?0:(int)data.Type));
            Field(w,16,N(d,"level",data==null?0:data.Level));
            Field(w,256,N(d,"attack",data==null?0:data.Attack));
            Field(w,512,N(d,"defense",data==null?0:data.Defense));
            Field(w,1024,data==null?0:data.Attack); Field(w,2048,data==null?0:data.Defense);
            Field(w,64,data==null?0:(int)data.Attribute);
            w.Write((short)12);w.Write((uint)128);w.Write((ulong)(data==null?0:data.Race));
            w.Write((short)4);w.Write((uint)0x80000000);w.Flush();ms.Position=0;c.Update(new BinaryReader(ms),duel);
        }
        current[key]=c;return c;
    }
    static void Board(Dictionary<string,object> msg)
    {
        current=new Dictionary<string,ClientCard>();
        var players=A(msg["players"]);
        for(int p=0;p<2;p++) {
            var d=D(players[p]);int local=p^seat;var f=duel.Fields[local];f.Clear();
            f.LifePoints=N(d,"lp"); f.BattlingMonster=null;f.UnderAttack=false;
            foreach(string zone in new[]{"hand","monsters","spells_traps","graveyard","banished"}) {
                if(!d.ContainsKey(zone)) continue;
                foreach(var obj in A(d[zone])) {
                    var cd=D(obj); var c=Card(cd);int slot=N(cd,"slot");
                    switch(zone) {
                        case "hand":f.Hand.Add(c);break;
                        case "monsters":f.MonsterZone[slot]=c;break;
                        case "spells_traps":f.SpellZone[slot]=c;break;
                        case "graveyard":f.Graveyard.Add(c);break;
                        case "banished":f.Banished.Add(c);break;
                    }
                }
            }
            if(!d.ContainsKey("hand")) for(int i=0;i<N(d,"hand_count");i++) f.Hand.Add(new ClientCard(0,CardLocation.Hand,i,0,local));
            for(int i=0;i<N(d,"deck_count");i++) f.Deck.Add(new ClientCard(0,CardLocation.Deck,i,0,local));
        }
        int turn=N(msg,"turn"),phase=N(msg,"phase");duel.Player=N(msg,"turn_player")^seat;
        if(duel.Turn!=turn) {duel.Turn=turn;ai.OnNewTurn();}
        if((int)duel.Phase!=phase) {duel.Phase=(DuelPhase)phase;ai.OnNewPhase();}
        var links=A(msg["chain"]);
        if(chainEpoch!=N(msg,"chain_epoch")) {ai.OnChainEnd();chainEpoch=N(msg,"chain_epoch");}
        duel.CurrentChain.Clear();duel.CurrentChainInfo.Clear();duel.ChainTargets.Clear();duel.LastChainPlayer=-1;
        foreach(var obj in links) {
            var link=D(obj);var c=Card(link["card"]);duel.CurrentChain.Add(c);duel.CurrentChainInfo.Add(new ChainInfo(c));duel.LastChainPlayer=c.Controller;
            if(link.ContainsKey("targets")) foreach(var target in A(link["targets"])) duel.ChainTargets.Add(Card(target));
        }
        if(msg.ContainsKey("battle") && msg["battle"]!=null) {
            var b=D(msg["battle"]); var attacker=Card(b["attacker"]);
            duel.Fields[attacker.Controller].BattlingMonster=attacker;
            duel.Fields[1-attacker.Controller].UnderAttack=true;
            if(b["target"]!=null) {var target=Card(b["target"]);duel.Fields[target.Controller].BattlingMonster=target;}
        }
        previous=current;
    }
    static object Choose(int n) {return new {choose=n};}
    static object Select(IEnumerable<int> indices) {return new {select=indices.ToArray()};}
    static int Wire(Dictionary<string,object> opt) {return N(opt,"wire");}
    static long Desc(Dictionary<string,object> opt) {return opt.ContainsKey("description_id")?Convert.ToInt64(opt["description_id"]):0;}
    static object Decide(Dictionary<string,object> msg)
    {
        Board(msg);var p=D(msg["prompt"]);int kind=N(msg,"kind");
        var opts=A(p["choices"]).Select(D).ToList();
        int hint=N(msg,"hint_id");
        if(kind==11 || kind==10) {
            var main=new MainPhase();var battle=new BattlePhase();duel.MainPhase=main;duel.BattlePhase=battle;
            foreach(var o in opts) {
                int wire=Wire(o),cmd=wire&65535,index=wire>>16;
                ClientCard c=o.ContainsKey("card")?Card(o["card"]):null;
                if(c!=null) c.ActionIndex[cmd]=index;
                if(kind==11) switch(cmd) {
                    case 0:main.SummonableCards.Add(c);break;
                    case 1:main.SpecialSummonableCards.Add(c);break;
                    case 2:main.ReposableCards.Add(c);break;
                    case 3:main.MonsterSetableCards.Add(c);break;
                    case 4:main.SpellSetableCards.Add(c);break;
                    case 5:main.ActivableCards.Add(c);main.ActivableDescs.Add(Desc(o));c.ActionActivateIndex[Desc(o)]=index;break;
                    case 6:main.CanBattlePhase=true;break;
                    case 7:main.CanEndPhase=true;break;
                } else switch(cmd) {
                    case 0:battle.ActivableCards.Add(c);battle.ActivableDescs.Add(Desc(o));break;
                    case 1:c.CanDirectAttack=Convert.ToBoolean(o["direct_attack_available"]);battle.AttackableCards.Add(c);break;
                    case 2:battle.CanMainPhaseTwo=true;break;
                    case 3:battle.CanEndPhase=true;break;
                }
            }
            int result=kind==11?ai.OnSelectIdleCmd(main).ToValue():ai.OnSelectBattleCmd(battle).ToValue();
            int selected=opts.FindIndex(o=>Wire(o)==result);
            if(selected<0) throw new Exception("WindBot selected an unavailable command: "+result);
            return Choose(selected);
        }
        if(kind==16) {
            var acts=opts.Where(o=>o.ContainsKey("card")).ToList();
            int n=ai.OnSelectChain(acts.Select(o=>Card(o["card"])).ToList(),acts.Select(Desc).ToList(),acts.Count==opts.Count);
            return Choose(n<0?opts.Count-1:n);
        }
        if(kind==12) return Choose(ai.OnSelectEffectYn(Card(opts[0]["card"]),Convert.ToInt64(msg["question_id"]))?0:1);
        if(kind==13) return Choose(N(msg,"question_id")==30?(ai.OnSelectBattleReplay()?0:1):(ai.OnSelectYesNo(Convert.ToInt64(msg["question_id"]))?0:1));
        if(kind==14) return Choose(ai.OnSelectOption(opts.Select(Desc).ToList()));
        if(kind==19) {
            var pos=opts.Select(o=>(CardPosition)Wire(o)).ToList();
            int val=(int)ai.OnSelectPosition(N(D(opts[0]["card"]),"code"),pos);
            return Choose(opts.FindIndex(o=>Wire(o)==val));
        }
        if(kind==18 || kind==24) return Choose(0); // No column/zone effects exist in the frozen pool.
        if(kind==15 || kind==20) {
            var cards=A(p["candidates"]).Select(Card).ToList();int min=N(p,"minimum"),max=N(p,"maximum");
            var chosen=kind==20?ai.OnSelectTribute(cards,min,max,hint,opts.Count>0):ai.OnSelectCard(cards,min,max,hint,opts.Count>0);
            if(chosen.Count==0 && min>0 && opts.Count>0) return Choose(0);
            return Select(chosen.Select(c=>cards.IndexOf(c)));
        }
        if(kind==21 || kind==25) {
            var cards=A(p["candidates"]).Select(Card).ToList();var sorted=ai.OnCardSorting(cards);
            return new {order=(sorted??cards).Select(c=>cards.IndexOf(c)).ToArray()};
        }
        if(kind==26) {
            var choices=opts.Where(o=>o.ContainsKey("card")).ToList();
            if(choices.Count==0) return Choose(opts.Count-1);
            var cards=choices.Select(o=>Card(o["card"])).ToList();
            var result=ai.OnSelectCard(cards,1,1,hint,false);return Choose(cards.IndexOf(result[0]));
        }
        throw new Exception("Unhandled WindBot prompt "+kind);
    }
    public static void Main(string[] args)
    {
        NamedCardsManager.LoadDatabase(args[0]);
        string line;
        while((line=Console.ReadLine())!=null) try {
            var msg=D(json.DeserializeObject(line));
            if(msg.ContainsKey("init")) {
                seat=N(msg,"seat");duel=new Duel {IsFirst=seat==0,IsNewRule=true,IsNewRule2020=true};
                foreach(var f in duel.Fields) f.Init(0,0);
                ai=new GameAI(duel,"default",(s,b)=>{},(s,n)=>{},()=>{},args[1]);
                ai.Executor=new PoolExecutor(ai,duel); ai.Executor.Rand=new Random(N(msg,"rng_seed"));
                previous.Clear();chainEpoch=0;Console.WriteLine("{}");
            } else Console.WriteLine(json.Serialize(Decide(msg)));
        } catch(Exception ex) {Console.WriteLine(json.Serialize(new {error=ex.ToString()}));}
    }
}
