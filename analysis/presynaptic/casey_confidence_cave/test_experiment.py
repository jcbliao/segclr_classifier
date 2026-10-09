"""Exercise the real matched cohorts and CAVE windows through all three models."""
from pathlib import Path
import json
import sys
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.presynaptic.casey_confidence_cave import casey_confidence_comparison as comparison
from data.dataset_presynaptic import PresynapticWindowDataset
from gnn.model import ModelConfig, WindowClassifier


def test_matched_cohorts():
    for confidence in comparison.CONFIDENCES:
        for fold in range(5):
            directory = comparison.COHORT_ROOT / f'conf{confidence:g}/fold{fold}'
            manifest = json.loads((directory/'manifest.json').read_text())
            native = json.loads(Path(manifest['native_cohort_manifest']).read_text())
            assert manifest['cells'] == native['cells']
            assert manifest['hierarchy_tree'] == native['hierarchy_tree']
            assert (directory/'cells').resolve() == Path(manifest['source_database'])/'cells'
    counts = comparison.load_coarse_cohort_counts()
    assert not counts.attrs['planned']
    tc = counts[counts.coarse_type == 'thalamocortical']
    assert (tc.train_cells == 24).all() and (tc.test_cells == 6).all()


def test_models_forward_backward():
    torch.set_num_threads(2)
    manifest = json.loads((comparison.COHORT_ROOT/'conf0/fold0/manifest.json').read_text())
    graphs = []
    for split in ('train','test'):
        for label in ('BasketFam','thalamocortical','L2IT'):
            rid = next(r for r,info in manifest['cells'].items() if info['split']==split and info['cell_type']==label)
            probe = {**manifest,'cells':{rid:manifest['cells'][rid]}}
            ds = PresynapticWindowDataset(probe,split,'cave',database=comparison.COHORT_ROOT/'conf0/fold0',cell_cache_size=1)
            assert len(ds)>0
            graph = ds[0]
            assert graph.x.shape == (10,64) and torch.isfinite(graph.x).all()
            graphs.append(graph)
    batch = next(iter(DataLoader(graphs,batch_size=len(graphs))))
    for architecture in ('mean','pointwise_mlp','graph_transformer'):
        config = ModelConfig(architecture=architecture,cls_head_resnet=True,gt_depth=4,gt_heads=4)
        model = WindowClassifier(config,hierarchy=ds.hierarchy)
        embedding = model(batch.x,batch.edge_index,batch.batch,pos_enc=batch.pos_enc,rel_pos=batch.rel_pos,has_segclr=batch.has_segclr)
        loss = model.cls_head.compute_loss(embedding,batch.y_levels)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5).step()


def test_results_and_plots(tmp_path):
    """Exercise fold aggregation and best-checkpoint plots before training completes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    tags = ('mean','pointwise_mlp_L2','gt_L4_H4')
    metrics = {'accuracy':0.8,'balanced_accuracy':0.75,'macro_precision':0.7,
               'macro_f1':0.72,'confusion_matrix':[[8,2],[1,9]]}
    for confidence in comparison.CONFIDENCES:
        for fold in range(5):
            for tag in tags:
                run = tmp_path/f'conf{confidence:g}/fold{fold}'/f'gnn_lcpn_scratch_{tag}_resnet4x128_n10_mixed16_sampled_fold{fold}'
                run.mkdir(parents=True)
                (run/'best_metrics.json').write_text(json.dumps({'epoch':3,'classes':['a','b'],
                    'test_metrics':metrics,'window_test_metrics':metrics}))
                pd.DataFrame({'epoch':[1,2], 'train_loss':[1.,.5],
                    'window_macro_f1':[.5,.7],'cell_macro_f1':[.4,.6]}).to_csv(run/'epoch_metrics.csv',index=False)
    summary,payloads = comparison.load_summaries(tmp_path)
    curves = comparison.load_training_curves(tmp_path)
    assert len(summary)==75 and len(curves)==150
    assert (summary.best_epoch==3).all()
    figures = [comparison.plot_metric_comparison(summary),*comparison.plot_training_curves(curves),
               *comparison.plot_confusions(payloads),*comparison.plot_confusions(payloads,scope='cell')]
    assert len(figures)==16
    for fig in figures:
        fig.canvas.draw()
        plt.close(fig)


def test_submission_defaults_to_15_fold0_runs(tmp_path):
    """Exercise the submission script with a scheduler stub; submit no jobs."""
    import os
    import subprocess
    stub = tmp_path/'sbatch'
    log = tmp_path/'submitted.jsonl'
    stub.write_text('#!/usr/bin/python3\nimport json,os,sys\nwith open(os.environ["CASEY_SUBMISSION_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint("12345")\n')
    stub.chmod(0o755)
    script = tmp_path/'submit.sh'
    script.write_text((ROOT/'scripts/submit_casey_confidence_cave.sh').read_text().replace(
        'results/presynaptic/casey_confidence_cave/k10', str(tmp_path/'results')))
    environment = {**os.environ,'PATH':str(tmp_path)+':'+os.environ['PATH'],
                   'CASEY_SUBMISSION_LOG':str(log)}
    environment.pop('CASEY_FOLDS',None)
    subprocess.run(['bash',str(script)],env=environment,check=True,capture_output=True,text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls)==3
    arrays = []
    for call in calls[:2]:
        value = next(arg.split('=',1)[1] for arg in call if arg.startswith('--array='))
        arrays.append(set(map(int,value.split('%')[0].split(','))))
    assert len(arrays[0])==8 and len(arrays[1])==7
    assert not arrays[0]&arrays[1] and arrays[0]|arrays[1]==set(range(15))
    assert '--export=ALL,CASEY_FOLDS=0' in calls[2]
