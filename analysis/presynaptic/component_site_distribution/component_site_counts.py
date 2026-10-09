"""Unfiltered connected-component site distributions for both TEASAR datasets."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from ..new_skeletons import skeleton_stats as shared

HERE=Path(__file__).resolve().parent
DATABASES={
    'Non-native': Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10'),
    'Native resampled': Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/native'),
}
COLORS={'Non-native':'#4477aa','Native resampled':'#ee7733'}


def cell_components(task):
    label,path=task
    with np.load(path,allow_pickle=False) as cell:
        n=len(cell['new_pos_nm'])
        edges=cell['new_edges']
        sites,_=shared.mapped_presynaptic_counts(cell,n)
        graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),
                         shape=(n,n)).tocsr()
        nc,component=connected_components(graph,directed=False)
        counts=np.bincount(component,weights=sites,minlength=nc).astype(np.int64)
        nodes=np.bincount(component,minlength=nc)
        edge_counts=np.bincount(component[edges[:,0]],minlength=nc)
        assert np.array_equal(component[edges[:,0]],component[edges[:,1]])
        if 'component_presynaptic_sites' in cell:
            np.testing.assert_array_equal(counts,cell['component_presynaptic_sites'])
    return pd.DataFrame(dict(dataset=label,root_id=path.stem,component_id=np.arange(nc),
                             n_nodes=nodes,n_edges=edge_counts,n_presynaptic_sites=counts))


def load_components(workers=None):
    tasks=[(label,path) for label,db in DATABASES.items()
           for path in shared.presynaptic_cell_paths(db)]
    for label in DATABASES:
        assert sum(name==label for name,_ in tasks)==2209
    signature={'version':1,'files':[(name,str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
                                   for name,p in tasks]}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]
    cache=HERE/key
    csv=cache/'components.csv'
    if (cache/'complete.json').exists():
        print('Loading cached connected-component site counts',flush=True)
        return pd.read_csv(csv,dtype={'root_id':str})
    workers=shared.default_branch_workers() if workers is None else workers
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,frame in enumerate(pool.map(cell_components,tasks,chunksize=1),1):
            rows.append(frame)
            if i==1 or i%250==0 or i==len(tasks):
                print(f'Connected components: {i}/{len(tasks)} cells',flush=True)
    result=pd.concat(rows,ignore_index=True)
    cache.mkdir(parents=True,exist_ok=True)
    tmp=csv.with_suffix('.tmp.csv')
    result.to_csv(tmp,index=False); os.replace(tmp,csv)
    (cache/'complete.json').write_text(json.dumps(signature))
    return result


def summary_table(frame):
    rows=[]
    for label,g in frame.groupby('dataset',sort=False):
        s=g.n_presynaptic_sites
        rows.append(dict(dataset=label,cells=g.root_id.nunique(),components=len(g),
            zero_site_components=int((s==0).sum()),one_to_five_components=int(s.between(1,5).sum()),
            components_gt5=int((s>5).sum()),median_sites=float(s.median()),
            p90_sites=float(s.quantile(.9)),p99_sites=float(s.quantile(.99)),max_sites=int(s.max())))
    return pd.DataFrame(rows).set_index('dataset')


def plot_distribution(frame, bin_width=100):
    if not isinstance(bin_width, int) or isinstance(bin_width, bool) or bin_width < 1:
        raise ValueError('bin_width must be a positive integer')
    groups=list(frame.groupby('dataset',sort=False))
    maximum=int(frame.n_presynaptic_sites.max())
    upper=((maximum+1+bin_width-1)//bin_width)*bin_width
    bins=np.arange(0,upper+1,bin_width,dtype=float)-.5
    fig,axes=plt.subplots(1,len(groups),figsize=(13,4.8),sharex=True,sharey=True,squeeze=False)
    for ax,(label,g) in zip(axes[0],groups):
        sites=g.n_presynaptic_sites.to_numpy()
        counts,_,_=ax.hist(sites,bins=bins,color=COLORS[label],alpha=.85,
                          edgecolor='white',linewidth=.3)
        assert int(counts.sum())==len(sites), 'Histogram must include every component'
        ax.axvline(5.5,color='black',linestyle='--',linewidth=1.2,
                   label='current filter: retain ≥6 sites')
        ax.set(title=f'{label}: full range, {bin_width}-site bins',
               xlabel='unique accepted presynaptic sites per connected component',
               ylabel='number of connected components',xlim=(-.5,upper-.5))
        ax.text(.98,.78,f'{len(sites):,} components (including zeros)' + chr(10) +
                f'Maximum: {sites.max():,} sites',transform=ax.transAxes,
                ha='right',va='top',fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=.2)
        ax.ticklabel_format(axis='both',style='plain',useOffset=False)
    fig.tight_layout()
    return fig


def cutoff_table(frame,cutoffs=(0,1,2,5,10,20,50,100,200,500,1000)):
    rows=[]
    for label,g in frame.groupby('dataset',sort=False):
        for cutoff in cutoffs:
            kept=g.n_presynaptic_sites>cutoff
            rows.append(dict(dataset=label,cutoff=int(cutoff),retained_components=int(kept.sum()),
                retained_component_fraction=float(kept.mean()),
                retained_edges=int(g.loc[kept,'n_edges'].sum()),
                retained_edge_fraction=float(g.loc[kept,'n_edges'].sum()/g.n_edges.sum())))
    return pd.DataFrame(rows)


def plot_cutoff_retention(frame):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for label,g in frame.groupby('dataset',sort=False):
        grouped=g.groupby('n_presynaptic_sites').agg(components=('component_id','size'),edges=('n_edges','sum'))
        grid=grouped.index.union(pd.Index([0,5])).sort_values()
        grouped=grouped.reindex(grid,fill_value=0)
        for ax,column in zip(axes,('components','edges')):
            fraction=1-grouped[column].cumsum()/grouped[column].sum()
            ax.step(grid,fraction,where='post',color=COLORS[label],label=label,linewidth=2)
    for ax,title in zip(axes,('Connected components retained','Edges retained')):
        ax.axvline(5,color='black',linestyle='--',label='current cutoff: >5')
        ax.set_xscale('symlog',linthresh=10)
        ax.set(xlabel='cutoff k (retain components with >k sites)',ylabel='retained fraction',
               title=title,ylim=(0,1.02))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout()
    return fig
