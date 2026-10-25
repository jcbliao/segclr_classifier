"""Cached five-fold analysis, with folds as the uncertainty unit."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
METRICS=['accuracy','balanced_accuracy','macro_precision','macro_f1']

def mean_ci(values):
    """Pointwise 95% Student-t interval across five folds, not windows."""
    values=np.asarray(values,float)
    if len(values)!=5:raise ValueError('A five-fold mean requires all five folds')
    mean=np.mean(values,axis=0)
    half=t.ppf(.975,4)*np.std(values,axis=0,ddof=1)/np.sqrt(5)
    return mean,mean-half,mean+half

def describe(group):
    path=group['runs'][0]['report'];name=Path(path).name
    model=next((label for tag,label in [('pointwise_mlp','Pointwise MLP'),('mpnn','MPNN'),('gt','GT'),('mean','Mean')] if f'_{tag}_' in name),None)
    if model is None:model='Pointwise MLP'
    category=group['category'];family='';condition=''
    if category=='Whole-cell linear versus ResNet':condition='ResNet' if 'resnet' in name else 'Linear'
    elif category=='Morpho TEASAR density sweep':condition='bin'+re.search(r'/scale(\d+)/',path)[1]
    elif category=='Historical CAVE versus original mesh TEASAR':condition='CAVE' if '/cave_skeletons/' in path else 'Original mesh TEASAR'
    elif category=='V1DD addition':condition='With V1DD' if '/with_v1dd/' in path else 'MICrONS training only'
    elif category in ['TEASAR neighborhood pre/post','Single-embedding pre/post']:
        condition=next(c for c in ['pre_only','pre_mean_control','pre_post'] if f'/{c}/' in path)
    elif category=='CAVE six-class pre+post':condition='pre_post'
    elif category=='Morpho CAVE/TEASAR confidence sweep':
        family='CAVE' if '/morpho_confidence_cave/' in path else 'TEASAR'
        condition='conf'+re.search(r'/conf([^/]+)/',path)[1]
    elif category=='EM augmentation':condition=re.search(r'_embaug_(.*?)_sampled_',name)[1]
    elif category=='Paired axon versus axon+dendrite':condition=Path(path).parent.name
    return dict(group=group['id'],model=model,family=family,condition=condition,label=' / '.join(s for s in [family,model,condition] if s))

def load(category,family=None,model=None):
    plan=json.loads((BASE/'plan.json').read_text());records=[];status=[];curves=[]
    for group in plan['groups']:
        if group['category']!=category:continue
        meta=describe(group)
        if family and meta['family']!=family:continue
        if model and meta['model']!=model:continue
        available=[]
        for run in group['runs']:
            path=ROOT/run['report'];fold=run['fold']
            if path.exists():
                data=json.loads(path.read_text())
                if 'window_test_metrics' in data:
                    records.append({**meta,'fold':fold,'data':data});available.append(fold)
            folder=path.parent if path.name=='complete.json' else path.parent/path.stem
            cp=folder/'epoch_metrics.csv'
            if cp.exists():
                try:frame=pd.read_csv(cp)
                except pd.errors.EmptyDataError:continue
                for key,value in {**meta,'fold':fold}.items():frame[key]=value
                curves.append(frame)
            elif (folder/'epoch_metrics.json').exists():
                parsed=[]
                for epoch in json.loads((folder/'epoch_metrics.json').read_text()):
                    row={**meta,'fold':fold,'epoch':epoch['epoch'],'train_loss':epoch['train_loss']}
                    for scope,metrics in epoch['validation'].items():
                        row.update({f'{scope}_{k}':v for k,v in metrics.items() if k in METRICS})
                    parsed.append(row)
                curves.append(pd.DataFrame(parsed))
        status.append({**meta,'completed_folds':available,'complete':available==list(range(5))})
    result={'records':records,'status':pd.DataFrame(status),'curves':pd.concat(curves,ignore_index=True) if curves else pd.DataFrame()}
    return result

def complete_records(cache):
    if cache['status'].empty:return []
    groups=set(cache['status'].loc[cache['status'].complete,'group'])
    return [r for r in cache['records'] if r['group'] in groups]

def metric_rows(cache):
    rows=[]
    for r in complete_records(cache):
        data=r['data'];domains={'Combined':{'window':data['window_test_metrics'],'cell':data['test_metrics']}}
        domains.update(data.get('dataset_test_metrics',{}))
        for domain,scopes in domains.items():
            for scope in ['window','cell']:
                for metric in METRICS:
                    rows.append({k:r[k] for k in ['group','model','family','condition','label','fold']}|
                        dict(domain=domain,scope=scope,metric=metric,value=scopes[scope][metric]))
    return pd.DataFrame(rows)

def summary(cache):
    rows=metric_rows(cache)
    if rows.empty:return pd.DataFrame()
    output=[]
    for key,group in rows.groupby(['label','domain','scope','metric'],sort=False):
        mean,low,high=mean_ci(group.sort_values('fold').value)
        output.append(dict(zip(['label','domain','scope','metric'],key))|dict(mean=mean,ci_low=low,ci_high=high,folds=5))
    return pd.DataFrame(output)

def plot_scores(cache):
    rows=metric_rows(cache)
    if rows.empty:print('Waiting for complete five-fold groups.');return
    for domain in rows.domain.unique():
        fig,axes=plt.subplots(2,4,figsize=(18,9),layout='constrained')
        for ax,(scope,metric) in zip(axes.flat,[(s,m) for s in ['window','cell'] for m in METRICS]):
            values=rows[(rows.domain==domain)&(rows.scope==scope)&(rows.metric==metric)]
            labels=[]
            for i,(label,group) in enumerate(values.groupby('label',sort=False)):
                mean,low,high=mean_ci(group.sort_values('fold').value)
                ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=3)
                labels.append(label)
            ax.set(title=f'{scope}: {metric}',xlabel='Mean with 95% fold CI',yticks=range(len(labels)),yticklabels=labels);ax.set_xlim(0,1)
            ax.tick_params(axis='y',labelsize=7)
        fig.suptitle(domain);plt.show()

def plot_curves(cache):
    curves=cache['curves']
    if curves.empty:print('Waiting for epoch caches.');return
    for metric in ['train_loss','window_macro_f1','cell_macro_f1','selection_window_macro_f1']:
        if metric not in curves:continue
        fig,ax=plt.subplots(figsize=(11,5))
        drawn=False
        for label,group in curves.groupby('label',sort=False):
            pivot=group.pivot(index='epoch',columns='fold',values=metric).reindex(columns=range(5)).dropna()
            if pivot.empty:continue
            mean,low,high=mean_ci(pivot.to_numpy().T)
            line,=ax.plot(pivot.index,mean,label=label);ax.fill_between(pivot.index,low,high,color=line.get_color(),alpha=.18);drawn=True
        if drawn:
            ax.set(xlabel='Epoch',ylabel=metric,title='Five-fold mean and pointwise 95% fold CI');ax.legend(fontsize=8);plt.show()
        else:plt.close(fig)

def plot_trends(cache):
    rows=metric_rows(cache)
    if rows.empty:return
    prefix=next((p for p in ['conf','bin'] if all(str(v).startswith(p) for v in rows.condition)),None)
    if prefix is None:return
    for domain in rows.domain.unique():
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for ax,(scope,metric) in zip(axes.flat,[(s,m) for s in ['window','cell'] for m in ['macro_f1','balanced_accuracy','macro_precision']]):
            subset=rows[(rows.domain==domain)&(rows.scope==scope)&(rows.metric==metric)]
            for (family,model),group in subset.groupby(['family','model'],sort=False):
                label=' / '.join(v for v in [family,model] if v)
                values=[]
                for condition,points in group.groupby('condition'):
                    mean,low,high=mean_ci(points.sort_values('fold').value)
                    values.append((float(condition[len(prefix):]),mean,low,high))
                values=np.array(sorted(values));x,mean,low,high=values.T
                line,=ax.plot(x,mean,'o-',label=label);ax.fill_between(x,low,high,color=line.get_color(),alpha=.18)
            ax.set(xlabel='Morpho confidence cutoff' if prefix=='conf' else 'TEASAR bin',ylabel=metric,title=scope)
            if prefix=='bin':ax.set_xscale('log',base=2);ax.invert_xaxis()
            ax.legend()
        fig.suptitle(f'{domain}: five-fold mean and 95% fold CI');plt.show()

def plot_cohort_counts(cache):
    """Manifest cell counts; all-cohort counts are identical across folds."""
    records=complete_records(cache)
    if not records or not all(r['condition'].startswith('conf') for r in records):return
    import sys
    sys.path.insert(0,str(ROOT))
    from data.dataset_lcpn import load_hierarchy
    groups={}
    for r in records:groups.setdefault((r['family'],r['condition'],r['group']),[]).append(r)
    seen=set();values={};classes=records[0]['data']['classes']
    for (family,condition,group),runs in groups.items():
        if (family,condition) in seen:continue
        seen.add((family,condition));counts=[]
        for run in sorted(runs,key=lambda r:r['fold']):
            args=run['data']['args'];path=args.get('manifest') or str(Path(args['presynaptic_database'])/'manifest.json')
            manifest=json.loads(Path(path).read_text());hierarchy=load_hierarchy(manifest)
            cell_counts=np.zeros((3,len(classes)))
            for info in manifest['cells'].values():
                label=hierarchy.label_paths.get(info['cell_type'],[''])[-1]
                if label not in classes:continue
                j=classes.index(label);cell_counts[2,j]+=1
                if info['split'] in ['train','test']:cell_counts[['train','test'].index(info['split']),j]+=1
            counts.append(cell_counts)
        values[float(condition[4:])]=np.array(counts)
    x=np.array(sorted(values));fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for scope,ax in enumerate(axes):
        for j,label in enumerate(classes):
            arrays=np.stack([values[c][:,scope,j] for c in x],axis=1)
            mean,low,high=mean_ci(arrays);line,=ax.plot(x,mean,'o-',label=label);ax.fill_between(x,low,high,color=line.get_color(),alpha=.15)
        ax.set(xlabel='Morpho confidence cutoff',ylabel='Manifest cells',title=['Training','Held-out','Entire cohort (shared across folds)'][scope])
    axes[0].legend(fontsize=8);plt.show()

def plot_deltas(cache):
    rows=metric_rows(cache)
    if rows.empty:return
    pairs=[('Linear','ResNet'),('pre_only','pre_post'),('pre_mean_control','pre_post'),('MICrONS training only','With V1DD'),('clean','flip'),('clean','gray'),('clean','structured_low'),('axon_only','axon_dendrite'),('CAVE','Original mesh TEASAR')]
    for before,after in pairs:
        if not {before,after}<=set(rows.condition):continue
        for domain in rows.domain.unique():
            fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained');drawn=False
            for ax,(scope,metric) in zip(axes,[('window','macro_f1'),('window','balanced_accuracy'),('cell','macro_f1')]):
                subset=rows[(rows.domain==domain)&(rows.scope==scope)&(rows.metric==metric)]
                for i,((family,model),group) in enumerate(subset.groupby(['family','model'],sort=False)):
                    pivot=group.pivot(index='fold',columns='condition',values='value')
                    if not {before,after}<=set(pivot.columns):continue
                    values=pivot[after]-pivot[before]
                    if len(values.dropna())!=5:continue
                    mean,low,high=mean_ci(values);ax.errorbar(i,mean,yerr=[[mean-low],[high-mean]],fmt='o',capsize=4,label=model);drawn=True
                ax.axhline(0,color='grey',lw=1);ax.set(title=f'{scope}: {metric}',ylabel='Paired fold difference');ax.legend()
            if drawn:fig.suptitle(f'{domain}: {after} minus {before}');plt.show()
            else:plt.close(fig)

def _domains(data,scope):
    yield 'Combined',data['window_test_metrics' if scope=='window' else 'test_metrics']
    for domain,metrics in data.get('dataset_test_metrics',{}).items():yield domain,metrics[scope]

def plot_classes(cache):
    records=complete_records(cache)
    for scope in ['window','cell']:
        for kind in ['per_class_recall','per_class_precision','support']:
            grouped={}
            for r in records:
                for domain,metrics in _domains(r['data'],scope):
                    values=np.asarray(metrics['confusion_matrix']).sum(1) if kind=='support' else np.asarray([metrics[kind][c] for c in r['data']['classes']])
                    grouped.setdefault((domain,r['label']),[]).append((r['fold'],values,r['data']['classes']))
            for domain in dict.fromkeys(k[0] for k in grouped):
                fig,ax=plt.subplots(figsize=(12,5));labels=None
                for (d,label),values in grouped.items():
                    if d!=domain:continue
                    values.sort(key=lambda x:x[0]);assert len(values)==5
                    classes=values[0][2]
                    if labels is not None and classes!=labels:raise ValueError('Cannot average different class orders')
                    labels=classes;mean,low,high=mean_ci([v[1] for v in values]);x=np.arange(len(mean))
                    line,=ax.plot(x,mean,'o-',label=label);ax.fill_between(x,low,high,color=line.get_color(),alpha=.15)
                if labels is not None:
                    ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel=kind,title=f'{domain}: {scope} mean with 95% fold CI');ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=8);plt.show()
                else:plt.close(fig)

def plot_confusions(cache):
    records=complete_records(cache)
    for group in sorted(set(r['group'] for r in records)):
        runs=sorted([r for r in records if r['group']==group],key=lambda r:r['fold'])
        classes=runs[0]['data']['classes'];assert all(r['data']['classes']==classes for r in runs)
        for scope in ['window','cell']:
            for domain in dict(_domains(runs[0]['data'],scope)):
                matrices=[]
                for r in runs:
                    cm=np.asarray(dict(_domains(r['data'],scope))[domain]['confusion_matrix'],float)
                    support=cm.sum(1,keepdims=True)
                    if np.any(support==0):
                        cm=np.divide(cm,support,out=np.full_like(cm,np.nan),where=support>0)
                    else:cm=cm/support
                    matrices.append(cm)
                mean,low,high=mean_ci(matrices)
                fig,ax=plt.subplots(figsize=(8,7),layout='constrained');im=ax.imshow(mean,vmin=0,vmax=1,cmap='Blues')
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        text='unsupported' if not np.isfinite(mean[i,j]) else f'{100*mean[i,j]:.1f}%\n±{100*(high[i,j]-mean[i,j]):.1f}'
                        ax.text(j,i,text,ha='center',va='center',fontsize=7,color='white' if mean[i,j]>.5 else 'black')
                ax.set(xticks=range(len(classes)),yticks=range(len(classes)),xticklabels=classes,yticklabels=classes,
                    xlabel='Predicted',ylabel='True',title=f'{runs[0]["label"]}: {domain}, {scope}\nMean row-normalized confusion; ±95% fold CI')
                ax.tick_params(axis='x',rotation=45);fig.colorbar(im,ax=ax);plt.show()

def plot_augmentation_propagation():
    paths=[BASE/f'cache/augmentation_propagation_fold{fold}.json' for fold in range(5)]
    if not all(p.exists() for p in paths):
        print('Five-fold augmentation propagation inference pending.');return
    data=[json.loads(p.read_text()) for p in paths]
    stages=['node_cosine','mean_cosine','gt_cosine','trunk_cosine']
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for policy in ['gray','flip','structured_low']:
        values=[[d['policies'][policy][stage]['mean'] for stage in stages] for d in data]
        mean,low,high=mean_ci(values);x=np.arange(4)
        line,=axes[0].plot(x,mean,'o-',label=policy);axes[0].fill_between(x,low,high,color=line.get_color(),alpha=.18)
        values=[d['policies'][policy]['augmented_window_accuracy']-d['policies'][policy]['clean_window_accuracy'] for d in data]
        mean,low,high=mean_ci(values);i=['gray','flip','structured_low'].index(policy)
        axes[1].errorbar(i,mean,yerr=[[mean-low],[high-mean]],fmt='o',capsize=3)
    axes[0].set(xticks=range(4),xticklabels=['Node','Raw mean','GT output','ResNet trunk'],ylabel='Cosine similarity',title='Five-fold mean and 95% fold CI');axes[0].legend()
    axes[1].axhline(0,color='grey',lw=1);axes[1].set(xticks=range(3),xticklabels=['gray','flip','structured_low'],ylabel='Augmented − clean accuracy',title='Paired augmentation propagation sample')
    plt.show()
