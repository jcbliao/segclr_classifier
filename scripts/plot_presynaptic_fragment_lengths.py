"""Measure scale-32 fragments using the production window selector; no embeddings needed."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.build_registered_presynaptic import NATIVE
from data.skeleton_subsampling import nested_skeletons, NATIVE_SCALE_WINDOWS
from data.native_centered_windows import CenteredWindows, CENTER_POLICY

OUT = Path('analysis/presynaptic/new_skeletons_native/fragment_lengths_local_center') / f'k{NATIVE_SCALE_WINDOWS[32]}'


def measure(path, out):
    dest = out/'cells'/path.name
    signature = [path.stat().st_size, path.stat().st_mtime_ns, NATIVE_SCALE_WINDOWS[32]]
    if dest.exists():
        with np.load(dest) as z:
            if z['signature'].tolist() == signature:
                return
    with np.load(path) as z:
        pos,edges,lengths=z['new_pos_nm'],z['new_edges'],z['new_edge_length_nm']
        original=z['original_node_ids']
        scales=nested_skeletons(pos,edges,lengths,original)
        geo = scales[32]
        selector=CenteredWindows(pos,edges,lengths,original,geo,z['new_synapse_xyz_nm'])
        center_ids=original[selector.centers]
        points = pd.DataFrame(z['new_synapse_xyz_nm'], columns=['cell_x_nm','cell_y_nm','cell_z_nm'])
        points['synapse_id'] = z['new_synapse_id']
    values = {'signature': np.array(signature, np.int64), 'sites': len(points)}
    for k in (7, NATIVE_SCALE_WINDOWS[32]):
        arrays=selector.arrays(points.synapse_id.to_numpy(),k,compute_lpe=False)
        valid = np.flatnonzero(arrays['new_valid_k_window'])
        starts=arrays['new_window_offsets'][valid]
        np.testing.assert_array_equal(original[arrays['new_window_members'][starts]],center_ids[valid])
        seen, unique = set(), []
        for i in valid:
            lo, hi = arrays['new_window_offsets'][i:i+2]
            key = tuple(sorted(arrays['new_window_members'][lo:hi].tolist()))
            if key not in seen:
                seen.add(key)
                unique.append(arrays['new_cable_length_nm'][i])
        values[f'k{k}_unique_cable_nm'] = np.asarray(unique, np.float64)
        values[f'k{k}_valid_sites'] = len(valid)
    temp = dest.with_suffix('.partial.npz')
    np.savez_compressed(temp, **values)
    temp.replace(dest)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sources = sorted((NATIVE/'cells').glob('*.npz'))
    paths = sorted((out/'cells').glob('*.npz'))
    assert {p.name for p in paths} == {p.name for p in sources}, 'Incomplete cohort'
    ks = (7, NATIVE_SCALE_WINDOWS[32])
    chunks = {k: [] for k in ks}
    site_counts = {k: dict(valid=0) for k in ks}
    sites = 0
    for path in paths:
        with np.load(path) as z:
            sites += int(z['sites'])
            for k in ks:
                chunks[k].append(z[f'k{k}_unique_cable_nm'])
                site_counts[k]['valid'] += int(z[f'k{k}_valid_sites'])
    lengths = {k: np.concatenate(chunks[k])/1000 for k in ks}
    new = lengths[ks[1]]
    def stats(x):
        return dict(n=len(x), mean_um=float(x.mean()), median_um=float(np.median(x)),
                    p5_um=float(np.percentile(x,5)), p95_um=float(np.percentile(x,95)),
                    max_um=float(x.max()), fraction_above_25=float(np.mean(x>25)),
                    mode_1um_bin_left=float(np.argmax(np.histogram(x,bins=np.arange(0,np.ceil(x.max())+2))[0]))) if len(x) else dict(n=0)
    report = dict(cells=len(paths), presynaptic_sites=sites, factor=32, cable_length_cutoff=None,
                  center_policy=CENTER_POLICY,
                  weighting='one count per unique root-scoped node membership, matching training deduplication',
                  centered_k7=stats(lengths[7]), selected_k=ks[1], selected=stats(new), site_counts=site_counts)
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(out/'fragment_lengths_um.npz', centered_k7=lengths[7],
                        selected_k=ks[1], selected=new)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    upper = max(25, float(max(x.max() for x in lengths.values())))
    bins = np.arange(0,np.ceil(upper)+1,0.5)
    for k,x in lengths.items():
        axes[0].hist(x,bins=bins,histtype='step',linewidth=1.7,label=f'{k} nodes')
    axes[0].set(title='Presynaptic fragments: full distribution',yscale='log',ylabel='Unique fragments (log scale)')
    axes[0].legend(fontsize=8)
    axes[1].hist(new,bins=bins,color='#267b9f',edgecolor='white',linewidth=.3)
    axes[1].set(title=f'{ks[1]}-node fragments: central distribution',ylabel='Unique fragments',
                xlim=(0,max(30,float(np.percentile(new,99.5)))))
    if len(new):
        axes[1].axvline(new.mean(),color='#d56b2a',label=f'Mean {new.mean():.2f} µm')
        axes[1].legend()
    for ax in axes:
        ax.set_xlabel('Sum of fragment edge cable lengths (µm)')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'{len(paths):,} cells · scale 32 · full-resolution centers preserved')
    for ext in ('png','pdf'):
        fig.savefig(out/f'presynaptic_fragment_lengths.{ext}',dpi=180)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-id',type=int,default=0)
    p.add_argument('--num-tasks',type=int,default=1)
    p.add_argument('--plot',action='store_true')
    p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args()
    (a.out/'cells').mkdir(parents=True,exist_ok=True)
    if a.plot:
        plot(a.out)
    else:
        paths=sorted((NATIVE/'cells').glob('*.npz'))[a.task_id::a.num_tasks]
        for i,path in enumerate(paths,1):
            measure(path,a.out)
            print(f'{i}/{len(paths)} {path.stem}',flush=True)
