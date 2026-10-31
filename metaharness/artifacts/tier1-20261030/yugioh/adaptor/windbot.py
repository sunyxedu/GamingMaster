"""Private WindBot transport. Only the opponent's filtered view crosses the pipe."""
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import time

ROOT=Path(__file__).resolve().parent
POLICY='windbot-fixed-pool-v3'

class WindBot:
    def __init__(self,duel):
        from ygo_engine import derive
        self.proc=subprocess.Popen(['mono','--optimize=all',str(ROOT/'runtime/windbot/WindbotBridge.exe'),
            str(ROOT/'runtime/vendor/cdb/cards.cdb'),str(ROOT/'runtime/vendor/windbot')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,cwd=ROOT/'runtime/windbot')
        self.buffer=b''
        try:self.call({'init':True,'seat':1-duel.agent,'rng_seed':int.from_bytes(derive(duel.seed,f'round-{duel.round_index}:opponent')[:4],'little')&0x7fffffff})
        except BaseException:self.close();raise
    def call(self,msg):
        self.proc.stdin.write(json.dumps(msg,separators=(',',':'),allow_nan=False).encode()+b'\n');self.proc.stdin.flush()
        end=time.monotonic()+10
        while b'\n' not in self.buffer:
            delay=end-time.monotonic()
            if delay<=0 or not select.select([self.proc.stdout],[],[],delay)[0]:raise TimeoutError('WindBot response deadline')
            chunk=os.read(self.proc.stdout.fileno(),65536)
            if not chunk:raise RuntimeError('WindBot disconnected')
            self.buffer+=chunk
            if len(self.buffer)>1000000:raise RuntimeError('Oversized WindBot response')
        line,self.buffer=self.buffer.split(b'\n',1)
        answer=json.loads(line)
        if 'error' in answer:raise RuntimeError('WindBot: '+answer['error'])
        return answer
    def response(self,duel,prompt):
        p=prompt.public()
        desc=iter(prompt.description_ids)
        for o,response in zip(p['choices'],prompt.responses):
            if len(response)==4:o['wire']=struct.unpack('<i',response)[0]
            if 'effect' in o or prompt.kind==14:o['description_id']=next(desc)
        battle=None
        if duel.battle:
            a,b=duel.battle
            battle={'attacker':duel.reference(0,a,prompt.player),
                    'target':duel.reference(0,b,prompt.player) if b[1] else None}
        answer=self.call({'players':duel.snapshot(prompt.player),'turn':duel.turn,'phase':duel.phase,
            'turn_player':duel.turn_player,'chain_epoch':duel.chain_epoch,'chain':duel.chain,'battle':battle,'prompt':p,'kind':prompt.kind,
            'hint_id':prompt.hint_id,'question_id':prompt.question_id})
        if 'choose' in answer:action=f"choose {prompt.token} {answer['choose']}"
        else:
            verb='select' if 'select' in answer else 'order'
            action=f'{verb} {prompt.token} '+(','.join(map(str,answer[verb])) or '-')
        # All bot answers pass the same legality checks as agent answers.
        try:return prompt.response(action)
        except (ValueError,TypeError) as e:raise RuntimeError(f'Illegal WindBot response to {p["kind"]}: {answer}: {e}') from e
    def close(self):
        p=getattr(self,'proc',None)
        if not p:return
        self.proc=None
        if p.stdin:p.stdin.close()
        try:p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=1)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=1)
        if p.stdout:p.stdout.close()
