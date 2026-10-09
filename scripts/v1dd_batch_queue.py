"""Durable batch claims; expensive file probes stay outside the state lock."""
import fcntl
from v1dd_file_lock import acquire_lock
import json
import os
from pathlib import Path
from v1dd_storage_budget import StorageBudget

class BatchQueue:
    def __init__(self,base):
        self.base=Path(base);self.path=self.base/'batch_queue.json';self.lock=self.base/'batch_queue.lock'
        self.plans=self.base/'batch_plans';self.claims=self.base/'batch_claims'
        self.plans.mkdir(exist_ok=True);self.claims.mkdir(exist_ok=True)
        self.budget=StorageBudget(base)
        self.cursor=int(os.environ.get('SLURM_JOB_ID','0'))

    def initialize(self,counts,window=None):
        cells={}
        for rid,n in counts:
            folder=self.base/'inputs_v1'/str(rid);done=[];n_batches=(n+127)//128
            for i in range(n_batches):
                if (folder/f'batch_{i:06d}'/'metadata.json').exists():done.append(i)
            cells[str(rid)]=dict(n_nodes=n,n_batches=n_batches,done=done,planned=False,admitted=False,
                status='DONE' if (self.base/'embeddings'/f'{rid}.npz').exists() else 'OPEN',gc_prefix=0)
        with self.lock.open('a') as handle:
            acquire_lock(handle)
            self._save(dict(window=window,cells=cells))

    def _save(self,state):
        temp=self.path.with_suffix('.tmp.json');temp.write_text(json.dumps(state,separators=(',',':'))+'\n');temp.replace(self.path)

    def read(self):
        # Writers publish by atomic rename, so readers need no global lock.
        return json.loads(self.path.read_text())

    def update(self,callback):
        with self.lock.open('a') as handle:
            acquire_lock(handle)
            state=self.read();before=json.dumps(state,separators=(',',':'))
            result=callback(state)
            if json.dumps(state,separators=(',',':'))!=before:self._save(state)
            return result

    def try_lock(self,name):
        handle=(self.claims/f'{name}.lock').open('a')
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:handle.close();return None
        return handle

    @staticmethod
    def prefix(cell):
        done=set(cell['done']);i=0
        while i in done:i+=1
        return i

    def claim(self):
        state=self.read();cells=state['cells'];budget=None
        ordered=sorted(cells,key=lambda rid:(cells[rid]['n_batches']-len(cells[rid]['done']),-len(cells[rid]['done'])))
        for rid in ordered:
            cell=cells[rid]
            if cell['status']!='OPEN':continue
            if not cell['planned']:
                handle=self.try_lock(f'{rid}_setup')
                if handle:
                    fresh=self.snapshot(rid)
                    if fresh['status']=='OPEN' and not fresh['planned']:return 'setup',int(rid),None,handle
                    handle.close()
                continue
            if not cell['admitted']:
                # A read-only preview avoids hundreds of denied reservations,
                # each previously rewriting budget JSON under the queue lock.
                if budget is None:budget=json.loads(self.budget.path.read_text())
                used=budget['common_bytes']+sum(v['bytes'] for v in budget['cells'].values())
                old=budget['cells'].get(rid,dict(bytes=0,reserved=False))
                requested=int(cell['reservation']) if old['reserved'] else max(int(cell['reservation']),old['bytes'])
                if used-old['bytes']+requested>budget['limit_bytes']:continue
                admission=self.try_lock(f'{rid}_admit')
                if admission is None:continue
                try:
                    cell=self.snapshot(rid)
                    if cell['status']!='OPEN':continue
                    if not cell['admitted']:
                        if not self.budget.reserve(rid,cell['reservation']):continue
                        self.update(lambda s:s['cells'][rid].update(admitted=True))
                        cell['admitted']=True
                        budget=None
                finally:admission.close()
            if len(cell['done'])==cell['n_batches']:
                handle=self.try_lock(f'{rid}_gc')
                if handle:return 'finalize',int(rid),None,handle
                continue
            first=self.prefix(cell);end=cell['n_batches'] if state['window'] is None else min(first+state['window'],cell['n_batches'])
            done=set(cell['done']);pending=[i for i in range(first,end) if i not in done]
            # Spread workers across candidate batches; no per-cell worker cap.
            offset=self.cursor%len(pending) if pending else 0
            for i in pending[offset:]+pending[:offset]:
                handle=self.try_lock(f'{rid}_batch_{i:06d}')
                if not handle:continue
                fresh=self.snapshot(rid)
                if fresh['status']!='OPEN' or i in fresh['done'] or (self.base/'embedding_commits'/f'{rid}.json').exists():
                    handle.close();continue
                if (self.base/'inputs_v1'/rid/f'batch_{i:06d}'/'metadata.json').exists():
                    self.complete(rid,i);handle.close();continue
                self.cursor+=1
                return 'batch',int(rid),i,handle
        if all(c['status']!='OPEN' for c in cells.values()):return 'finished',None,None,None
        return None

    def planned(self,rid,reservation):
        def set_plan(s):s['cells'][str(rid)].update(planned=True,reservation=int(reservation))
        self.update(set_plan)

    def complete(self,rid,batch):
        def mark(s):
            cell=s['cells'][str(rid)]
            if batch not in cell['done']:cell['done'].append(batch)
            return len(cell['done'])==cell['n_batches']
        return self.update(mark)

    def snapshot(self,rid):
        return self.read()['cells'][str(rid)]

    def gc_complete(self,rid,prefix,ready=False):
        def mark(s):
            cell=s['cells'][str(rid)];cell['gc_prefix']=max(prefix,cell['gc_prefix'])
            if ready:cell['status']='READY'
        self.update(mark)
