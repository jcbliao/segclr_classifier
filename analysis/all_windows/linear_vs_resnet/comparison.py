"""Cached reports and epoch metrics only; never loads a model or runs inference."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
METRICS=('balanced_accuracy','macro_precision','macro_f1')
LABELS=('Balanced accuracy','Macro precision','Macro F1')
COLORS={'Linear':'#4C78A8','ResNet':'#E45756'}

def refresh_cache():
    plan=json.loads((BASE/'plan.json').read_text());runs=[]
    for job in plan['jobs']:
        for head,key in [('Linear','linear_report'),('ResNet','source_report')]:
            p=ROOT/job[key];run=p.stem;csv=p.parent/run/'epoch_metrics.csv'
            entry=dict(architecture=job['architecture'],head=head,run=run,report=str(p.relative_to(ROOT)),complete=p.exists(),epochs=[])
            if csv.exists():entry['epochs']=pd.read_csv(csv).replace({np.nan:None}).to_dict('records')
            if p.exists():
                raw=p.read_bytes();d=json.loads(raw)
                if head=='ResNet':assert hashlib.sha256(raw).hexdigest()==job['source_sha256']
                assert d['classes']==job['classes']
                a=d['args'];assert a['cls_resnet']==(head=='ResNet') and not a['freeze_aggregator']
                assert a['num_embeddings']==10 and a['seed']==0 and a['epochs']==16 and a['class_balance']=='sample'
                source=json.loads((ROOT/job['source_report']).read_text())
                allowed={'cls_resnet','num_workers','resume','manifest','results_dir','dataset'}
                assert all(a[k]==v for k,v in source['args'].items() if k not in allowed)
                assert np.array_equal(np.asarray(d['window_test_metrics']['confusion_matrix']).sum(1),np.asarray(source['window_test_metrics']['confusion_matrix']).sum(1))
                entry.update(window=d['window_test_metrics'],cell=d['test_metrics'],classes=d['classes'],sha256=hashlib.sha256(raw).hexdigest())
            runs.append(entry)
    cache=dict(runs=runs,plan_sha256=hashlib.sha256((BASE/'plan.json').read_bytes()).hexdigest())
    (BASE/'report_cache.json').write_text(json.dumps(cache,indent=2)+'\n')
    return cache

def table(cache):
    rows=[]
    for r in cache['runs']:
        row={k:r[k] for k in ('architecture','head','complete')}
        row['epochs_available']=len(r['epochs'])
        for scope in ('window','cell'):
            for m in METRICS:row[f'{scope}_{m}']=r.get(scope,{}).get(m,np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def score_plot(cache,difference=False):
    df=table(cache);architectures=list(dict.fromkeys(df.architecture));fig,axs=plt.subplots(2,3,figsize=(17,8),sharey=not difference)
    for row,scope in enumerate(('window','cell')):
        for col,(metric,label) in enumerate(zip(METRICS,LABELS)):
            ax=axs[row,col];x=np.arange(len(architectures));key=f'{scope}_{metric}'
            if difference:
                wide=df.pivot(index='architecture',columns='head',values=key).reindex(architectures)
                ax.bar(x,wide['ResNet']-wide['Linear'],color='#8064A2');ax.axhline(0,color='black',lw=.8)
                if not wide[['ResNet','Linear']].notna().all(axis=1).any():ax.text(.5,.5,'Waiting for matched linear results',transform=ax.transAxes,ha='center')
            else:
                for j,(head,color) in enumerate(COLORS.items()):
                    vals=df[df['head']==head].set_index('architecture').reindex(architectures)[key]
                    ax.bar(x+(j-.5)*.36,vals,width=.36,label=head,color=color)
                ax.set_ylim(0,1)
            ax.set_xticks(x,architectures,rotation=22,ha='right',fontsize=9);ax.set_title(f'{scope.title()}: {label}');ax.grid(axis='y',alpha=.2)
    if not difference:axs[0,0].legend()
    fig.suptitle('ResNet − linear head (positive favors ResNet)' if difference else '10-node windows: linear versus ResNet head · one cell-held-out split')
    fig.tight_layout();return fig

def learning_curves(cache):
    arches=list(dict.fromkeys(r['architecture'] for r in cache['runs']));fig,axes=plt.subplots(1,4,figsize=(18,4),sharey=True)
    for ax,arch in zip(axes,arches):
        for r in cache['runs']:
            if r['architecture']!=arch or not r['epochs']:continue
            e=pd.DataFrame(r['epochs']);ax.plot(e.epoch+1,e.window_macro_f1,label=r['head'],color=COLORS[r['head']])
        ax.set_title(arch,fontsize=10);ax.set_xlabel('Epoch');ax.set_xlim(1,16);ax.grid(alpha=.2)
    axes[0].set_ylabel('Held-out window macro F1');axes[0].legend();fig.tight_layout();return fig

def confusions(cache,scope='window'):
    arches=list(dict.fromkeys(r['architecture'] for r in cache['runs']));fig,axes=plt.subplots(2,4,figsize=(20,10))
    for i,head in enumerate(COLORS):
        for j,arch in enumerate(arches):
            ax=axes[i,j];r=next(r for r in cache['runs'] if r['architecture']==arch and r['head']==head)
            ax.set_title(f'{arch}\n{head}',fontsize=10)
            if not r['complete']:ax.text(.5,.5,'Training pending / in progress',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
            cm=np.asarray(r[scope]['confusion_matrix'],float);cm=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm),where=cm.sum(1,keepdims=True)>0)
            im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
            for y in range(len(cm)):
                for x in range(len(cm)):ax.text(x,y,f'{cm[y,x]:.2f}',ha='center',va='center',fontsize=7,color='white' if cm[y,x]>.55 else 'black')
            names=[s.replace('putative_','') for s in r['classes']]
            ax.set_xticks(range(len(names)),names,rotation=65,ha='right',fontsize=8);ax.set_yticks(range(len(names)),names,fontsize=8)
            ax.set_xlabel('Predicted');ax.set_ylabel('True')
    fig.suptitle(f'{scope.title()} confusion matrices · rows normalized to one');fig.tight_layout();return fig
