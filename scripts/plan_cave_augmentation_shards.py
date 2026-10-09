"""Greedily balance confidence-0.7 CAVE augmentation work by target nodes."""
import argparse,json,multiprocessing as mp
from pathlib import Path
from scripts.generate_cave_embedding_augmentations import DATABASE,MANIFEST,CAVE_CACHE,target_nodes
import pickle
def count(rid):
 p=DATABASE/'cells'/f'{rid}.npz'
 if p.exists(): return rid,len(target_nodes(p)[0])
 with open(CAVE_CACHE/f'{rid}.pkl','rb') as f:return rid,len(pickle.load(f).coords)
def main():
 p=argparse.ArgumentParser();p.add_argument('--shards',type=int,default=18);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 m=json.loads(MANIFEST.read_text());roots=sorted(int(r) for r,x in m['cells'].items() if x['split']=='train')
 with mp.Pool(16) as pool: counts=dict(pool.map(count,roots))
 shards=[[] for _ in range(a.shards)];loads=[0]*a.shards
 for rid in sorted(roots,key=lambda r:counts[r],reverse=True):
  i=min(range(a.shards),key=loads.__getitem__);shards[i].append(rid);loads[i]+=counts[rid]
 payload=dict(shards=shards,node_loads=loads,total_nodes=sum(loads));a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(dict(loads=loads,total=sum(loads),ratio=max(loads)/min(loads)),indent=2))
if __name__=='__main__':main()
