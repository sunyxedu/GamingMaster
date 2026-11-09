"""Private JSON-line worker. Native crashes cannot kill the evaluator."""
import json,sys,traceback
from scenarios import schedule
from ygo_engine import Duel

class Episode:
    def __init__(self,seed,limits):
        self.seed=seed;self.limits=limits;self.schedule=schedule(seed);self.results=[];self.coverage=[];self.duel=None
        self.start()
    def start(self):
        i=len(self.results);a,b,s=self.schedule[i]
        self.duel=Duel(self.seed,i,a,b,s,self.limits)
    def collect(self):
        while self.duel.result and len(self.results)<3:
            self.results.append(dict(self.duel.result));self.coverage.append(self.duel.diagnostics())
            if len(self.results)==3:break
            self.duel.close();self.start()
    def call(self,command,action=None):
        if len(self.results)<3:
            if command=='step':self.duel.step(action)
            elif command=='forfeit':self.duel.finish('forfeit',1-self.duel.agent,'agent response deadline exceeded')
            self.collect()
        obs=self.duel.observation();obs['done']=len(self.results)==3;obs['rounds_total']=3
        obs['completed_duels']=[{k:v for k,v in r.items() if k not in ('agent_deck','opponent_deck')} for r in self.results]
        info={'valid_trial':True,'seed':self.seed,'manifest_version':3,'opponent':'windbot-fixed-pool-v3',
              'results':self.results,'completed_duels':len(self.results),'strata':[{'agent_deck':a,'opponent_deck':b,'agent_first':s==0} for a,b,s in self.schedule],
              'coverage':self.coverage,'current':self.duel.diagnostics(),
              'truncated':any(r['truncated'] for r in self.results),'technical_error':False,
              'forfeit':any(r['forfeit'] for r in self.results)}
        return {'observation':obs,'info':info,'score':sum(r['score'] for r in self.results)/3,'done':obs['done']}
    def close(self):
        if self.duel:self.duel.close()

def main():
    episode=None
    try:
        for line in sys.stdin:
            try:
                cmd=json.loads(line);op=cmd['op']
                if op=='close':break
                if op=='reset':
                    if episode:episode.close()
                    episode=Episode(cmd['seed'],cmd['limits'])
                out=episode.call(op,cmd.get('action'))
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                out={'error':f'{type(e).__name__}: {e}'}
            print(json.dumps(out,separators=(',',':'),allow_nan=False),flush=True)
    finally:
        if episode:episode.close()
if __name__=='__main__':main()
