"""Complete added thalamocortical density windows using cached inputs only."""
import json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_report_fold_sweep import DATA,read,save
OUT=Path('/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/report_folds_20261012/density')
FULL=DATA/'casey_k17_full_tc_source/scale16/k17'
SCALES={32:9,16:17,8:33,4:65}

def prepare_root(rid):
    import numpy as np
    import pandas as pd
    import torch
    from data.build_presynaptic_axon_database import _induced_cut,_atomic_savez
    from data.skeleton_subsampling import nested_skeletons
    from data.native_centered_windows import CenteredWindows,CENTER_POLICY
    torch.set_num_threads(1)
    if all((OUT/f'scale{s}/k{k}/cells/{rid}.npz').is_file() for s,k in SCALES.items()):return rid
    with np.load(FULL/f'cells/{rid}.npz') as z:
        embedding=Path(str(z['embedding_path']));old_geometry=Path(str(z['geometry_path']))
    with np.load(old_geometry) as z:center=z['root_xyz_nm']
    route=next(r for r in read(Path('/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911/routes.json')) if r['root_id']==rid)
    with np.load(Path(route['base'])/f'resampled_111nm/{rid}.npz') as z:pos,edges=z['vertices'],z['edges']
    with np.load(embedding) as z:
        np.testing.assert_array_equal(z['node_ids'],np.arange(len(pos)))
        assert z['embeddings'].shape==(len(pos),64) and np.isfinite(z['embeddings']).all()
    keep,_,pos,edges,lengths,_=_induced_cut(pos,edges,center)
    original=np.flatnonzero(keep);scales=nested_skeletons(pos,edges,lengths,original)
    sites=pd.read_parquet(FULL/f'sites/{rid}.parquet')
    xyz=sites[['cell_x_nm','cell_y_nm','cell_z_nm']].to_numpy(float)
    geometry=OUT/f'topology/full/{rid}.npz'
    _atomic_savez(geometry,root_id=rid,skeleton_name=route['skeleton_name'],**scales[1])
    for s,k in SCALES.items():
        dest=OUT/f'scale{s}/k{k}/cells/{rid}.npz'
        if dest.is_file():continue
        selector=CenteredWindows(pos,edges,lengths,original,scales[s],xyz)
        arrays=selector.arrays(sites.synapse_id.to_numpy(),k)
        valid=arrays['new_valid_k_window'];starts=arrays['new_window_offsets'][:-1][valid]
        np.testing.assert_array_equal(original[arrays['new_window_members'][starts]],original[selector.centers[valid]])
        _atomic_savez(dest,root_id=rid,k_observed=k,factor=s,center_policy=CENTER_POLICY,
            new_center_original_node_ids=original[selector.centers],skeleton_name=route['skeleton_name'],
            geometry_path=str(geometry),embedding_path=str(embedding),**arrays)
        print(rid,s,int(valid.sum()),flush=True)
    return rid

def main():
    manifest=read(DATA/'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0/manifest.json')
    missing=set()
    for s,k in SCALES.items():
        source=DATA/f'resampled_teasar_2209/training_local_center/scale{s}/k{k}'
        dest=OUT/f'scale{s}/k{k}';(dest/'cells').mkdir(parents=True,exist_ok=True)
        for rid in manifest['cells']:
            p=source/f'cells/{rid}.npz';target=dest/f'cells/{rid}.npz'
            if p.is_file():
                if not target.exists():target.symlink_to(p.resolve())
            else:missing.add(int(rid))
        metadata={**read(source/'metadata.json'),'source_database':str(source),'added_roots':sorted(missing),
                  'n_cells':len(manifest['cells']),'missing_root_ids':[],'morpho_confidence_min':.7,
                  'cohort':'six-class morpho; same root assignments at every density'}
        metadata.pop('valid_site_windows',None)  # historical full-cohort count is inapplicable
        save(dest/'metadata.json',metadata)
        save(dest/'manifest.json',manifest)
    with ProcessPoolExecutor(max_workers=min(15,len(missing))) as pool:roots=list(pool.map(prepare_root,sorted(missing)))
    for s,k in SCALES.items():
        assert all((OUT/f'scale{s}/k{k}/cells/{rid}.npz').is_file() for rid in manifest['cells'])
    save(OUT/'complete.json',{'added_roots':roots,'scales':SCALES,'cohort_cells':len(manifest['cells']),'cached_inputs_only':True})

if __name__=='__main__':main()
