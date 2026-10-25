"""Cached, matched GT/MPNN interpretation and inference ablations; no retraining."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from data.dataset_lcpn import load_hierarchy
from data.dataset_windowed import WindowedGraphDatasetLCPN
from data.packed_windows import PackedWholeCellWindows
from data.packed_presynaptic import PackedPresynapticWindowDataset
from gnn.model import WindowClassifier
from gnn.training_performance import static_top_down
from gnn.structure_inspection import intervention,randomized_edges
from gnn.structure_statistics import StructureStatistics,parameter_statistics
from gnn.metrics import summarize,majority_vote_by_group
from scripts.eval_ablations import _recompute_pos_enc

BASE=ROOT/'analysis/all_windows/structure_ablation'
CACHE=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/whole_cell_structure_n10_20261011')
RUNS={'GT':'gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_fold0',
      'MPNN':'gnn_lcpn_scratch_mpnn_L2_position_lpe_resnet4x128_n10_fold0',
      'Mean control':'gnn_lcpn_scratch_mean_resnet4x128_n10_fold0'}
COMMON=['identity','recompute_lpe','rewire','rewire_lpe_stale','rewire_lpe0','drop_edges',
        'drop_edges_stale_lpe','zero_pos_enc','zero_rel_pos','zero_geometry','permute_x']
SPECIFIC={'GT':['no_qk','no_adjacency_bias','uniform_attention','nodewise_mean_readout','no_attention_update'],
          'MPNN':['no_neighbors','no_self'],'Mean control':[]}
RANDOM={'rewire','rewire_lpe_stale','rewire_lpe0','permute_x'}
MANIFEST_OVERRIDE=None

def configure_fold(fold):
    global BASE,CACHE,RUNS,MANIFEST_OVERRIDE
    if fold==0:return
    catalog=json.loads((ROOT/'analysis/report_folds_20261012/plan.json').read_text())
    job=next(j for j in catalog['jobs'] if j['category']=='Whole-cell linear versus ResNet' and j['fold']==fold
             and '/linear_vs_resnet/' not in j['report'])
    for name,run in RUNS.items():
        assert (ROOT/'results/all_windows'/f'{run.replace("fold0",f"fold{fold}")}.json').is_file(),f'Missing completed {name} fold {fold}'
    MANIFEST_OVERRIDE=Path(json.loads((ROOT/job['config']).read_text())['manifest'])
    BASE=BASE/f'fold{fold}'
    CACHE=CACHE.with_name(CACHE.name+f'_fold{fold}')
    RUNS={name:run.replace('fold0',f'fold{fold}') for name,run in RUNS.items()}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def plan():
    manifest=MANIFEST_OVERRIDE or ROOT/'analysis/all_windows/linear_vs_resnet/manifest.json'
    preflight=ROOT/'analysis/all_windows/linear_vs_resnet/preflight.json'
    return {'manifest':str(manifest),'manifest_sha256':sha(manifest),
            'preflight':str(preflight),'preflight_sha256':sha(preflight),
            'runs':{name:{'checkpoint':str(ROOT/'results/all_windows'/run/'checkpoint_best.pt'),
                          'checkpoint_sha256':sha(ROOT/'results/all_windows'/run/'checkpoint_best.pt'),
                          'report':str(ROOT/'results/all_windows'/f'{run}.json')}
                    for name,run in RUNS.items()},'nodes':10,'seeds':[0,1,2],
            'soma_policy':'original 5um Euclidean nucleus exclusion; primary metrics exclude unknown-nucleus cells; historical cohort reported separately',
            'interpretation':'class-stratified held-out sample, up to 20000 windows per supported class, seed 0',
            'conditions':COMMON,'model_conditions':SPECIFIC}

def prepare(workers):
    torch.set_num_threads(1)
    BASE.mkdir(parents=True,exist_ok=True)
    frozen=plan();atomic_json(BASE/'plan.json',frozen);digest=sha(BASE/'plan.json')
    if (CACHE/'complete.json').exists():
        assert json.loads((CACHE/'complete.json').read_text())['plan_sha256']==digest
        print('Reusing validated packed cache',flush=True);return
    source=WindowedGraphDatasetLCPN(json.loads(Path(frozen['manifest']).read_text()),'test',num_embeddings=10,pos_dim=8)
    ds=PackedWholeCellWindows(source);ds.prepare_packed(workers=workers)
    temporary=CACHE.with_name(CACHE.name+'.preparing');temporary.mkdir(parents=True,exist_ok=True)
    for name,array in ds._packed['fields'].items():np.save(temporary/f'field_{name}.npy',array)
    for name,array in ds._packed.items():
        if isinstance(array,np.ndarray):np.save(temporary/f'{name}.npy',array)
    for name in ['index_root_ids','index_labels','window_sizes']:np.save(temporary/f'{name}.npy',getattr(ds,name))
    preflight_data=json.loads(Path(frozen['preflight']).read_text())
    heldout=set(map(int,ds.index_root_ids))
    uncut=[r['root_id'] for r in preflight_data['roots'] if r['root_id'] in heldout and not r['cut_applied']]
    strict=~np.isin(ds.index_root_ids,uncut);np.save(temporary/'strict_soma_mask.npy',strict)
    # Actual original extraction is the packing oracle; validate distributed rows.
    from torch_geometric.data import Batch
    chosen=np.linspace(0,len(ds)-1,100,dtype=int).tolist()
    expected=Batch.from_data_list([source[i] for i in chosen]);actual=ds.get_batch(chosen)
    for name in ['x','edge_index','edge_attr','pos_enc','rel_pos','y_levels','root_id']:
        torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
    metadata={'plan_sha256':digest,'windows':len(ds),'strict_soma_windows':int(strict.sum()),
              'uncut_heldout_roots':uncut,'packing_seconds':ds.packing_seconds,'packing_bytes':ds.packing_bytes,
              'original_extraction_parity_windows':len(chosen)}
    atomic_json(temporary/'complete.json',metadata)
    temporary.rename(CACHE)
    print(json.dumps(metadata),flush=True)

class CachedWindows(PackedPresynapticWindowDataset):
    def __len__(self):return len(self.index_root_ids)

def load_cache():
    assert json.loads((CACHE/'complete.json').read_text())['plan_sha256']==sha(BASE/'plan.json')
    ds=CachedWindows.__new__(CachedWindows)
    for name in ['index_root_ids','index_labels','window_sizes']:
        setattr(ds,name,np.load(CACHE/f'{name}.npy',mmap_mode='r'))
    fields={p.stem[6:]:np.load(p,mmap_mode='r') for p in CACHE.glob('field_*.npy')}
    ds._packed={'fields':fields,'synapses':None,'post':None}
    for name in ['ptr','targets','centers','edges','edge_offsets']:
        ds._packed[name]=np.load(CACHE/f'{name}.npy',mmap_mode='r')
    ds.prepare_device(torch.device('cuda'))
    return ds

def metrics(y,p,roots,classes):
    cy,cp=majority_vote_by_group(roots,y,p)
    return {'window':summarize(y,p,len(classes),classes),'cell':summarize(cy,cp,len(classes),classes)}

@torch.inference_mode()
def validate():
    """Exercise actual checkpoints and each intervention before full inference."""
    from torch_geometric.data import Data,Batch
    from scripts.test_default_training_pipeline import whole_windows
    torch.set_num_threads(1)
    frozen=plan();hierarchy=load_hierarchy(json.loads(Path(frozen['manifest']).read_text()))
    source=whole_windows(False);batch=Batch.from_data_list([source[i] for i in range(len(source))]).to('cuda')
    for name,info in frozen['runs'].items():
        saved=torch.load(info['checkpoint'],map_location='cuda',weights_only=False)
        model=WindowClassifier(saved['config'],hierarchy).cuda().eval();model.load_state_dict(saved['model_state'])
        def forward():return model(batch.x,batch.edge_index,batch.batch,pos_enc=batch.pos_enc,rel_pos=batch.rel_pos,batch_shape=(6,3))
        with torch.autocast('cuda',dtype=torch.float16):
            original=forward()
            with intervention(model,'identity',StructureStatistics()):torch.testing.assert_close(forward(),original,rtol=.002,atol=.002)
            for condition in SPECIFIC[name]:
                with intervention(model,condition):assert torch.isfinite(forward()).all(),condition
            torch.testing.assert_close(forward(),original,rtol=0,atol=0)
        print(name,'checkpoint / FP16 / interventions passed',flush=True)

@torch.inference_mode()
def evaluate(seed,batch_size,inspect=False):
    torch.set_num_threads(1);torch.manual_seed(seed);device=torch.device('cuda')
    frozen=json.loads((BASE/'plan.json').read_text());ds=load_cache()
    hierarchy=load_hierarchy(json.loads(Path(frozen['manifest']).read_text()));classes=hierarchy.level_classes[-1]
    strict=np.load(CACHE/'strict_soma_mask.npy');ys=ds._packed['targets'][:,-1];roots=ds.index_root_ids
    selected=np.arange(len(ds))
    if inspect:
        rng=np.random.default_rng(0);selected=np.sort(np.concatenate([
            rng.choice(rows,min(20000,len(rows)),replace=False) for label in np.unique(ys[strict])
            for rows in [np.flatnonzero(strict&(ys==label))]]))
    models={};observers={}
    for name,info in frozen['runs'].items():
        assert sha(info['checkpoint'])==info['checkpoint_sha256']
        saved=torch.load(info['checkpoint'],map_location=device,weights_only=False)
        model=WindowClassifier(saved['config'],hierarchy).to(device).eval()
        model.load_state_dict(saved['model_state']);models[name]=model
        observers[name]=StructureStatistics()
    conditions={name:['identity'] if inspect else ([c for c in COMMON if seed==0 or c in RANDOM]+(SPECIFIC[name] if seed==0 else [])) for name in models}
    predictions={(name,c):np.empty(len(selected),np.int8) for name in models for c in conditions[name]}
    all_conditions=list(dict.fromkeys(c for cs in conditions.values() for c in cs))
    started=time.perf_counter()
    for start in range(0,len(selected),batch_size):
        ids=selected[start:start+batch_size];b=ds.get_batch(ids,device='device_fields' in ds._packed)
        b=b.to(device);b.edge_attr=torch.zeros((b.edge_index.shape[1],1),device=device)
        shape=(len(ids),int(ds.window_sizes[ids].max()))
        random_edges=randomized_edges(b) if not inspect else None
        recomputed=_recompute_pos_enc(b.edge_index,b,8,device) if 'recompute_lpe' in all_conditions else None
        random_lpe=_recompute_pos_enc(random_edges,b,8,device) if 'rewire' in all_conditions else None
        permutation=(b.batch.double()*2+torch.rand(len(b.x),device=device,dtype=torch.float64)).argsort() if 'permute_x' in all_conditions else None
        for condition in all_conditions:
            x,edges,pe,rel=b.x,b.edge_index,b.pos_enc,b.rel_pos
            if condition.startswith('rewire'):
                edges=random_edges
                if condition=='rewire':pe=random_lpe
                if condition=='rewire_lpe0':pe=torch.zeros_like(pe)
            elif condition.startswith('drop_edges'):
                edges=edges[:,:0]
                if condition=='drop_edges':pe=torch.zeros_like(pe)
            elif condition=='recompute_lpe':pe=recomputed
            elif condition=='zero_pos_enc':pe=torch.zeros_like(pe)
            elif condition=='zero_rel_pos':rel=torch.zeros_like(rel)
            elif condition=='zero_geometry':pe=torch.zeros_like(pe);rel=torch.zeros_like(rel)
            elif condition=='permute_x':x=x[permutation]
            for name,model in models.items():
                if condition not in conditions[name]:continue
                observer=observers[name] if inspect else None
                if observer is not None:
                    from gnn.dense_layout import dense_layout,pack_dense
                    layout=dense_layout(b.batch,shape[0],shape[1]);dense,valid=pack_dense(b.rel_pos,layout)
                    observer.radial_order=dense.norm(dim=-1).masked_fill(~valid,float('inf')).argsort(-1)
                with intervention(model,condition,observer),torch.autocast('cuda',dtype=torch.float16):
                    hidden=model(x,edges,b.batch,pos_enc=pe,rel_pos=rel,batch_shape=shape)
                    pred=static_top_down(model.cls_head,hidden)[:,-1]
                predictions[name,condition][start:start+len(ids)]=pred.cpu().numpy().astype(np.int8)
        if start//batch_size%100==0:print(f'{start+len(ids):,}/{len(selected):,} windows; {time.perf_counter()-started:.1f}s',flush=True)
    output=BASE/'cache';output.mkdir(exist_ok=True)
    if inspect:
        atomic_json(output/'weight_analysis.json',{'sample_windows':len(selected),'classes':classes,
            'sample_class_counts':np.bincount(ys[selected],minlength=len(classes)).tolist(),
            'activation_statistics':{name:o.export() for name,o in observers.items()},
            'parameter_statistics':{name:parameter_statistics(m) for name,m in models.items()},
            'plan_sha256':sha(BASE/'plan.json')})
    else:
        rows=[]
        for (name,condition),pred in predictions.items():
            row={'model':name,'condition':condition,'seed':seed,
                 'historical_cohort':metrics(ys,pred,roots,classes),
                 'strict_soma':metrics(ys[strict],pred[strict],roots[strict],classes)}
            if seed==0:
                baseline=predictions[name,'identity']
                row['prediction_change_fraction']=float(np.mean(pred!=baseline))
            rows.append(row)
        if seed==0:
            base=predictions['Mean control','identity']
            for condition in conditions['Mean control']:
                assert np.mean(predictions['Mean control',condition]!=base)<.001,condition
        np.savez_compressed(output/f'predictions_seed{seed}.npz',**{f'{name}__{c}':p for (name,c),p in predictions.items()})
        atomic_json(output/f'ablations_seed{seed}.json',{'rows':rows,'classes':classes,
                    'windows':len(ds),'strict_soma_windows':int(strict.sum()),'seconds':time.perf_counter()-started,
                    'plan_sha256':sha(BASE/'plan.json')})
    print('Complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','evaluate','inspect','validate'])
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--workers',type=int,default=15)
    parser.add_argument('--batch-size',type=int,default=2048)
    parser.add_argument('--fold',type=int,choices=range(5),default=0);args=parser.parse_args()
    configure_fold(args.fold)
    if args.mode=='prepare':prepare(args.workers)
    elif args.mode=='validate':validate()
    else:evaluate(args.seed,args.batch_size,args.mode=='inspect')
