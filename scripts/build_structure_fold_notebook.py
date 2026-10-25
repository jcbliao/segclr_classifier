"""Five-fold companion to the cached structural intervention analysis."""
import json,shutil,sys
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_report_fold_sweep import BASE,read,save

def build():
    path=ROOT/'analysis/all_windows/structure_ablation_all_folds.ipynb'
    original=path.with_name('structure_ablation.ipynb')
    backup=BASE/'legacy_notebooks/analysis/all_windows/structure_ablation.ipynb'
    if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,backup)
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    notebook=nb.v4.new_notebook()
    notebook.cells=[md('''# Whole-cell structure analysis: all folds

The original GT, MPNN, and mean + ResNet checkpoints are evaluated on their corresponding held-out 10-node windows, without retraining the ablations. Primary metrics exclude cells without known nucleus coordinates; other cells use the original 5 µm Euclidean nucleus exclusion. These are the earlier eight-class models, with unsupported classes excluded from macro metrics by the original metric definition.

Every plot shows equal-weight fold means with pointwise 95% Student-t intervals across the five folds. For randomized interventions, average the three seeds **within each fold first**; the seeds are not additional folds. Fold training sets overlap, so the intervals describe fold variability. No five-fold figure is drawn until all five fold caches are present.

Q/K/V projections mix dimensions within nodes; attention mixes node value vectors. `uniform_attention` retains mixing with equal weights. `nodewise_mean_readout` disables attention updates to ordinary nodes while retaining uniform CLS pooling. `no_attention_update` also blocks CLS readout and is a degenerate diagnostic. Inference ablations measure reliance, not the performance of retrained ablated architectures. Attention/gamma and message-branch norms are descriptive, not fractions of causal prediction importance.

Activation statistics use class-stratified samples (up to 20,000 windows per supported class per fold). Their means describe those samples, not the distribution weighted by all test windows.'''),code('''%matplotlib inline
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts/train_gnn.py').exists())
sys.path.insert(0,str(root/'analysis/report_folds_20261012'))
from all_folds import mean_ci
base=root/'analysis/all_windows/structure_ablation'
folders=[base]+[base/f'fold{f}' for f in range(1,5)]
ready=all((p/'cache'/f'ablations_seed{s}.json').exists() for p in folders for s in range(3))
weights_ready=all((p/'cache/weight_analysis.json').exists() for p in folders)
display(pd.DataFrame([{'fold':f,'metrics_ready':all((p/'cache'/f'ablations_seed{s}.json').exists() for s in range(3)),
                      'weights_ready':(p/'cache/weight_analysis.json').exists()} for f,p in enumerate(folders)]))
rows=[]
if ready:
    for fold,folder in enumerate(folders):
        for seed in range(3):
            for row in json.loads((folder/'cache'/f'ablations_seed{seed}.json').read_text())['rows']:
                for unit in ['window','cell']:
                    rows.append({'fold':fold,'seed':seed,'model':row['model'],'condition':row['condition'],'unit':unit,
                                 **{k:v for k,v in row['strict_soma'][unit].items() if isinstance(v,(float,int))}})
    df=pd.DataFrame(rows).groupby(['fold','model','condition','unit'],as_index=False).mean(numeric_only=True)
if weights_ready:
    weights=[json.loads((p/'cache/weight_analysis.json').read_text()) for p in folders]
    stats=[w['activation_statistics'] for w in weights]
    def series(model,key):return np.array([s[model][key]['mean'] for s in stats])
'''),md('## Held-out inference interventions'),code('''if ready:
    fig,axes=plt.subplots(1,2,figsize=(17,8),layout='constrained')
    order=list(dict.fromkeys(df.condition))
    for ax,unit in zip(axes,['window','cell']):
        for model,offset in zip(['GT','MPNN','Mean control'],[-.2,0,.2]):
            points=df[(df.model==model)&(df.unit==unit)]
            for condition,values in points.groupby('condition'):
                mean,low,high=mean_ci(values.sort_values('fold').balanced_accuracy)
                ax.errorbar(mean,order.index(condition)+offset,xerr=[[mean-low],[high-mean]],fmt='o',capsize=2,label=model if condition=='identity' else None)
        ax.set(yticks=range(len(order)),yticklabels=order,xlabel='Balanced accuracy: mean and 95% fold CI',title=unit);ax.legend()
    plt.show()
    display(df.groupby(['model','condition','unit'])[['balanced_accuracy','macro_f1']].agg(['mean','std']))
else:print('Waiting for all five folds; no partial five-fold means displayed.')
'''),md('## Attention allocation across ordinary-node queries'),code('''if weights_ready:
    labels=[f'L{l+1}/H{h+1}' for l in range(4) for h in range(4)]
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for ax,term in zip(axes.flat,['self_mass','neighbor_mass','other_node_mass','cls_key_mass']):
        values=np.stack([series('GT',f'GT/layer{l}/head{h}/node/{term}') for l in range(4) for h in range(4)],axis=1)
        mean,low,high=mean_ci(values);x=np.arange(16)
        ax.plot(x,mean,'o-');ax.fill_between(x,low,high,alpha=.2)
        ax.set(xticks=x,xticklabels=labels,ylabel='Attention mass',title=term);ax.tick_params(axis='x',rotation=70)
    plt.show()
'''),md('## QK versus adjacency logit contributions'),code('''if weights_ready:
    fig,ax=plt.subplots(figsize=(12,5))
    for term in ['qk','adjacency']:
        values=np.stack([series('GT',f'GT/layer{l}/head{h}/node/{term}_centered_logit_rms') for l in range(4) for h in range(4)],axis=1)
        mean,low,high=mean_ci(values);line,=ax.plot(range(16),mean,'o-',label=term);ax.fill_between(range(16),low,high,color=line.get_color(),alpha=.2)
    ax.set(xticks=range(16),xticklabels=labels,ylabel='Centered logit RMS',title='After learned gamma scaling; mean and 95% fold CI');ax.tick_params(axis='x',rotation=70);ax.legend();plt.show()
'''),md('## CLS readout by radial rank'),code('''if weights_ready:
    fig,ax=plt.subplots(figsize=(10,5))
    for l in range(4):
        values=np.stack([np.mean([series('GT',f'GT/layer{l}/head{h}/cls/radial_rank{rank}_mass') for h in range(4)],axis=0) for rank in range(10)],axis=1)
        mean,low,high=mean_ci(values);x=np.arange(1,11);line,=ax.plot(x,mean,'o-',label=f'Layer {l+1}');ax.fill_between(x,low,high,color=line.get_color(),alpha=.2)
    ax.set(xlabel='Node rank by Euclidean distance to center',ylabel='CLS attention mass',title='Mean across heads, then folds; 95% fold CI');ax.legend();plt.show()
'''),md('## MPNN neighbor/self contribution distribution'),code('''if weights_ready:
    fig,ax=plt.subplots(figsize=(10,5))
    for l in range(2):
        hist=[s['MPNN'][f'MPNN/layer{l}/neighbor_norm_fraction']['histogram'] for s in stats]
        edges=np.array(hist[0]['edges']);width=np.diff(edges)
        values=np.array([np.array(h['counts'])/sum(h['counts'])/width for h in hist])
        mean,low,high=mean_ci(values);x=(edges[:-1]+edges[1:])/2
        line,=ax.plot(x,mean,label=f'Layer {l+1}');ax.fill_between(x,low,high,color=line.get_color(),alpha=.2)
    ax.set(xlabel='Neighbor norm / (neighbor norm + self norm)',ylabel='Density',title='Mean fold density and 95% fold CI');ax.legend();plt.show()
'''),md('## Parameter magnitudes\nDescriptive norms are not a ranking of aggregator versus decoder importance.'),code('''if weights_ready:
    rows=[]
    for model,parameters in weights[0]['parameter_statistics'].items():
        for name in parameters:
            values=[w['parameter_statistics'][model][name]['rms'] for w in weights]
            mean,low,high=mean_ci(values)
            rows.append({'model':model,'parameter':name,'RMS_mean':mean,'CI_low':low,'CI_high':high})
    display(pd.DataFrame(rows))
''')]
    notebook.metadata={'kernelspec':{'display_name':'segclr_db (.venv)','language':'python','name':'segclr_db'}}
    nb.write(notebook,path);nb.write(notebook,original)
    records=read(BASE/'notebooks.json');records=[r for r in records if r['all_folds']!=str(path.relative_to(ROOT))]
    records.append({'original':str(original.relative_to(ROOT)),'all_folds':str(path.relative_to(ROOT)),'category':'Whole-cell inference interventions','archived_original':str(backup.relative_to(ROOT))})
    save(BASE/'notebooks.json',records)

if __name__=='__main__':build()
