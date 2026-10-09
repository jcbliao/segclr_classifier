"""Import validated pretrained synapse embeddings as named postsynaptic points.

Default is a read-only preflight. Pass --write to import into the shared store.
"""
from pathlib import Path
import argparse,json,sys
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from segclr_db import store as st
from segclr_db.experiment import SegCLRExperiment
from segclr_db.writer import SegCLRWriter

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,default=Path('/orcd/scratch/orcd/013/jcbliao/postsynaptic_point_embeddings'))
 p.add_argument('--db-root',default='/orcd/compute/sdorkenw/001/segclr-db')
 p.add_argument('--point-set-name',default='postsynaptic_sites')
 p.add_argument('--write',action='store_true')
 p.add_argument('--limit',type=int)
 p.add_argument('--report',type=Path,default=Path('analysis/neuron_incoming_points/import_report.json'))
 args=p.parse_args()
 plan=json.loads((args.input/'plan.json').read_text())
 manifest=json.loads((args.input/'validated_manifest.json').read_text())
 assert manifest['checkpoint']==plan['checkpoint']
 assert manifest['n_points']==plan['n_points']
 store=st.open_store(args.db_root,'microns')
 assert st.read_meta(store).mat_version==plan['materialization_version']
 experiment=SegCLRExperiment.from_registry('resnet_860b_reshuffled',store)
 resolved=experiment.resolve_checkpoint(Path(plan['checkpoint']).stem)
 registered = [ck for run in experiment.runs if run.run_id == resolved.run_id
               for ck in run.checkpoints if ck.checkpoint_id == resolved.checkpoint_id]
 assert len(registered) == 1
 assert Path(registered[0].path).resolve()==Path(plan['checkpoint']).resolve()
 grouped=defaultdict(list)
 for part in manifest['parts']:grouped[part['root_id']].append(part)
 assert len(grouped)==manifest['n_cells']
 assert sum(x['n_points'] for parts in grouped.values() for x in parts)==manifest['n_points']
 for parts in grouped.values():
  offset=0
  for part in sorted(parts,key=lambda x:x['start']):
   assert part['start']==offset and Path(part['path']).is_file()
   offset+=part['n_points']
 report=dict(point_set_name=args.point_set_name,materialization_version=plan['materialization_version'],source=str(args.input),db_root=args.db_root,checkpoint=plan['checkpoint'],n_points=manifest['n_points'],n_cells=len(grouped))
 print(json.dumps(report),flush=True)
 if not args.write:
  print('Read-only preflight passed; no database writes.',flush=True);return
 writer=SegCLRWriter(store=store)
 written=skipped=0
 for i,(root,parts) in enumerate(sorted(grouped.items())):
  if args.limit is not None and i>=args.limit:break
  coords=[];vectors=[];ids=[]
  for part in sorted(parts,key=lambda x:x['start']):
   with np.load(part['path']) as z:
    assert int(z['root_id'])==root
    assert z['embeddings'].shape==(part['n_points'],64)
    coords.append(z['positions_nm']);vectors.append(z['embeddings']);ids.append(z['synapse_ids'])
  result=writer.add_named_point_embeddings(experiment,root,args.point_set_name,np.concatenate(coords),np.concatenate(vectors),np.concatenate(ids),checkpoint_id=resolved.checkpoint_id)
  written+=result.n_written;skipped+=result.n_skipped
  print(json.dumps(dict(root_id=root,written=result.n_written,skipped=result.n_skipped,cell=i+1)),flush=True)
 report.update(written=written,skipped=skipped)
 args.report.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
