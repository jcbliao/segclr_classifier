"""Compare paired fold 0, including separate MICrONS and V1DD metrics."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path('results/presynaptic/cave_v1dd_addition')
OUTPUT = Path('analysis/presynaptic/cave_v1dd_addition')
NAMES = {'mean':'mean','pointwise_mlp':'pointwise_mlp_L2','graph_transformer':'gt_L4_H4'}


def main():
    rows = []
    per_class = []
    for condition in ['no_v1dd','with_v1dd']:
        for architecture,tag in NAMES.items():
            for fold in (0,):
                name = f'gnn_lcpn_scratch_{tag}_resnet4x128_n10_mixed16_sampled_fold{fold}'
                data = json.loads((ROOT/condition/f'{name}.json').read_text())
                assert data['args']['architecture']==architecture
                domains = {'combined':{'cell':data['test_metrics'],'window':data['window_test_metrics']},**data['dataset_test_metrics']}
                assert 'microns' in domains and 'v1dd' in domains
                for domain,metrics in domains.items():
                    record = dict(condition=condition,architecture=architecture,fold=fold,dataset=domain,best_epoch=data['best_epoch'])
                    record['n_cells'] = sum(map(sum,metrics['cell']['confusion_matrix']))
                    record['n_windows'] = sum(map(sum,metrics['window']['confusion_matrix']))
                    for granularity in ['cell','window']:
                        m = metrics[granularity]
                        for metric in ['accuracy','balanced_accuracy','macro_precision','macro_f1']:
                            record[f'{granularity}_{metric}'] = m[metric]
                        for index,(label,value) in enumerate(m['per_class_recall'].items()):
                            per_class.append(dict(condition=condition,architecture=architecture,fold=fold,dataset=domain,
                                granularity=granularity,cell_type=label,recall=value,precision=m['per_class_precision'][label],
                                support=sum(m['confusion_matrix'][index])))
                    rows.append(record)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT/'summary_by_fold.csv',index=False)
    metric_cols = [c for c in df if c.startswith(('cell_','window_'))]
    summary = df.groupby(['condition','architecture','dataset'])[metric_cols].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUTPUT/'summary.csv',index=False)
    pairs = df[df.condition.eq('with_v1dd')].merge(df[df.condition.eq('no_v1dd')],on=['architecture','dataset','fold'],suffixes=('_with','_without'),validate='one_to_one')
    deltas = pairs[['architecture','dataset','fold']].copy()
    for col in metric_cols:
        deltas[col] = pairs[col+'_with']-pairs[col+'_without']
    deltas.to_csv(OUTPUT/'paired_deltas_by_fold.csv',index=False)
    agg = deltas.groupby(['architecture','dataset'])[metric_cols].agg(['mean','std'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg.reset_index().to_csv(OUTPUT/'paired_deltas.csv',index=False)
    pd.DataFrame(per_class).to_csv(OUTPUT/'per_class_metrics.csv',index=False)
    print(summary.to_string())


if __name__=='__main__':
    main()
