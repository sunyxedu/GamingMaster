// SPDX-License-Identifier: AGPL-3.0-or-later
// The fixed benchmark pool registered with upstream WindBot's decision engine.
using System;
using System.Linq;
using System.Collections.Generic;
using WindBot.Game;
using WindBot.Game.AI;
using YGOSharp.OCGWrapper;
using YGOSharp.OCGWrapper.Enums;

public sealed class PoolExecutor : DefaultExecutor
{
    const int Bug=54652250, Penguin=93920745, Faith=31560081;
    bool Flip(ClientCard c) { return c.Id==Bug || c.Id==Penguin || c.Id==Faith; }
    public PoolExecutor(GameAI ai, Duel duel) : base(ai,duel)
    {
        AddExecutor(ExecutorType.Activate, 53129443, DefaultDarkHole);
        AddExecutor(ExecutorType.Activate, 66788016, () => Enemy.GetMonsters().Any(c=>c.IsFaceup()));
        AddExecutor(ExecutorType.Activate, 5318639, DefaultMysticalSpaceTyphoon);
        AddExecutor(ExecutorType.Activate, 83764718, Reborn);
        AddExecutor(ExecutorType.Activate, 97077563, Revive);
        AddExecutor(ExecutorType.Activate, 44095762, DefaultTrap);
        AddExecutor(ExecutorType.Activate, 4206964, DefaultTrap);
        AddExecutor(ExecutorType.Activate, 94192409, Bounce);
        AddExecutor(ExecutorType.Activate, 14087893, Moon);
        AddExecutor(ExecutorType.Activate, 70046172, Rush);
        AddExecutor(ExecutorType.Activate, Bug, () => true);
        AddExecutor(ExecutorType.Activate, Penguin, () => true);
        AddExecutor(ExecutorType.Activate, Faith, () => true);
        // Flip summons earn removal/recovery before committing another monster.
        AddExecutor(ExecutorType.Repos, () => Flip(Card) && Card.IsFacedown());
        AddExecutor(ExecutorType.SummonOrSet, StrongestSummon);
        AddExecutor(ExecutorType.MonsterSet, () => Flip(Card) || Card.Defense>Card.Attack);
        AddExecutor(ExecutorType.Repos, DefaultMonsterRepos);
        AddExecutor(ExecutorType.SpellSet, DefaultSpellSet);
    }
    bool StrongestSummon()
    {
        if (Flip(Card)) return false;
        var choices=Duel.MainPhase.SummonableCards.Where(c=>!Flip(c));
        if (choices.Any(c=>c.Attack>Card.Attack && c.Level<=4)) return false;
        return DefaultMonsterSummon();
    }
    bool Reborn()
    {
        var target=Bot.Graveyard.Concat(Enemy.Graveyard).Where(c=>c.IsMonster()).OrderByDescending(c=>c.Attack).FirstOrDefault();
        if(target==null) return false;
        AI.SelectCard(target); return true;
    }
    bool Revive()
    {
        var target=Bot.Graveyard.Where(c=>c.IsMonster()).OrderByDescending(c=>c.Attack).FirstOrDefault();
        if(target==null) return false;
        if(Duel.Player==0 || Bot.UnderAttack || Util.IsChainTarget(Card)) {AI.SelectCard(target);return true;}
        return DefaultCallOfTheHaunted();
    }
    bool Bounce()
    {
        if(DefaultCompulsoryEvacuationDevice()) return true;
        if(Enemy.BattlingMonster!=null && Bot.UnderAttack) {AI.SelectCard(Enemy.BattlingMonster);return true;}
        return false;
    }
    bool Moon()
    {
        if(DefaultBookOfMoon()) return true;
        if(Bot.UnderAttack && Enemy.BattlingMonster!=null) {AI.SelectCard(Enemy.BattlingMonster);return true;}
        if(Duel.Player==0 && Duel.CurrentChain.Count==0) {
            var target=Enemy.GetMonsters().Where(c=>c.IsFaceup() && c.Attack>c.Defense && c.Attack>=Util.GetBestPower(Bot)).OrderByDescending(c=>c.Attack).FirstOrDefault();
            if(target!=null && Bot.GetMonsterCount()>0) {AI.SelectCard(target);return true;}
        }
        return false;
    }
    bool Rush()
    {
        var mine=Bot.BattlingMonster; var enemy=Enemy.BattlingMonster;
        if(mine==null || !mine.IsAttack()) return false;
        int power=enemy==null?0:enemy.GetDefensePower();
        if((enemy!=null && mine.Attack<=power && mine.Attack+700>power) ||
           (Duel.Player==0 && mine.Attack-power<Enemy.LifePoints && mine.Attack+700-power>=Enemy.LifePoints)) {
            AI.SelectCard(mine);return true;
        }
        return false;
    }
    public override IList<ClientCard> OnSelectCard(IList<ClientCard> cards,int min,int max,long hint,bool cancelable)
    {
        // Preserve WindBot's queued attack/effect target selectors.
        if(AI.HaveSelectedCards()) return null;
        IEnumerable<ClientCard> ordered=cards;
        if(hint==HintMsg.Release || hint==HintMsg.Discard) ordered=cards.OrderBy(c=>c.Attack);
        else if(hint==HintMsg.Destroy || hint==HintMsg.ReturnToHand)
            ordered=cards.OrderByDescending(c=>(c.Controller==1?10000:0)+c.Attack+(c.Level>4?1000:0));
        else if(hint==HintMsg.SpSummon) ordered=cards.OrderByDescending(c=>c.Attack);
        else if(hint==HintMsg.AddToHand) ordered=cards.OrderByDescending(c=>c.Id==83764718?100:c.Id==53129443?90:c.Id==66788016?80:10);
        int count=min;
        if(hint==HintMsg.ReturnToHand) count=Math.Min(max,Math.Max(min,cards.Count(c=>c.Controller==1)));
        return ordered.Take(count).ToList();
    }
    public override CardPosition OnSelectPosition(int id,IList<CardPosition> positions)
    {
        var card=NamedCard.Get(id);
        if(card!=null && (Duel.Player==1 || Duel.Turn==1 || card.Attack<1000) && card.Defense>card.Attack)
            return CardPosition.FaceUpDefence;
        return base.OnSelectPosition(id,positions);
    }
}
