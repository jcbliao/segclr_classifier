"""Map report analyses to frozen fold catalogs, and make all-fold siblings."""
import json,shutil,sys
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_report_fold_sweep import BASE,save

SPECS=[
 ('analysis/all_windows/linear_vs_resnet.ipynb','Whole-cell linear versus ResNet',None,None),
 ('analysis/all_windows/architecture_comparison.ipynb','Whole-cell linear versus ResNet',None,None),
 ('analysis/presynaptic/new_skeletons_native/skeleton_comparison_native.ipynb','Morpho TEASAR density sweep',None,None),
 ('analysis/presynaptic/skeleton_comparison/skeleton_comparison.ipynb','Historical CAVE versus original mesh TEASAR',None,None),
 ('analysis/presynaptic/cave_v1dd_addition/embedding_augmentation_comparison.ipynb','V1DD addition',None,None),
 ('analysis/presynaptic/native_skeletons_pre_post/pre_post_comparison.ipynb','TEASAR neighborhood pre/post',None,None),
 ('analysis/presynaptic/native_single_pre_post/pre_post_comparison.ipynb','Single-embedding pre/post',None,None),
 ('analysis/presynaptic/cave_skeletons_pre_post/six_class/pre_post_comparison.ipynb','CAVE six-class pre+post',None,None),
 ('analysis/presynaptic/cave_embedding_augmentation_conf0.7/embedding_augmentation_comparison.ipynb','EM augmentation',None,None),
 ('analysis/presynaptic/morpho_confidence_cave/morpho_confidence_comparison.ipynb','Morpho CAVE/TEASAR confidence sweep','CAVE',None),
 ('analysis/presynaptic/morpho_confidence_native/morpho_confidence_comparison.ipynb','Morpho CAVE/TEASAR confidence sweep','TEASAR',None),
 ('analysis/presynaptic/cave_teasar_confidence_5fold_20261010/comparison.ipynb','Morpho CAVE/TEASAR confidence sweep',None,None),
 ('analysis/presynaptic/axon_vs_whole_cell_morpho6/paired_comparison.ipynb','Paired axon versus axon+dendrite',None,None),
]

def build():
    specs=list(SPECS)
    # Preserve the existing model-specific six-class notebook entry points too.
    for folder,category in [('native_skeletons_pre_post','TEASAR neighborhood pre/post'),('native_single_pre_post','Single-embedding pre/post'),('cave_skeletons_pre_post/six_class','CAVE six-class pre+post')]:
        for p in (ROOT/'analysis/presynaptic'/folder).glob('gnn*_5folds.ipynb'):
            if 'three_class' in p.name:continue
            model='Pointwise MLP' if 'pointwise_mlp' in p.name else 'GT'
            specs.append((str(p.relative_to(ROOT)),category,None,model))
    # Mean and GT CAVE six-class pages have no historical page to overwrite.
    for model,tag in [('Mean','mean'),('GT','gt')]:
        specs.append((f'analysis/presynaptic/cave_skeletons_pre_post/six_class/{tag}_pre_post.ipynb','CAVE six-class pre+post',None,model))
    mapped=[]
    for path,category,family,model in specs:
        original=ROOT/path;all_path=original.with_name(original.stem+'_all_folds.ipynb')
        backup=BASE/'legacy_notebooks'/path
        existed=original.exists()
        if existed and not backup.exists():
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,backup)
        header=f'''# {category}: all folds

This analysis reads saved final reports and epoch caches through the frozen report catalog. It produces **equal-weight means across five folds**, with **pointwise 95% Student-t confidence intervals** (four degrees of freedom). Folds, not individual windows, are the uncertainty unit. The training sets overlap across folds, so these intervals describe fold variability; they are not independent-replicate or untouched-test confidence guarantees.

No five-fold summary is drawn for an incomplete group. Learning curves use only epochs present in all five folds. Paired differences are computed within each fold before averaging. Confusion matrices average **row-normalized matrices**, rather than pooling windows; each cell displays its mean and CI half-width. Unsupported class rows are marked rather than treated as zero accuracy.

Whole-cell analyses retain the historical eight-class hierarchy and 5 µm Euclidean nucleus exclusion. Density analyses use **six-class morpho labels at confidence ≥0.7**, with bins 32/16/8/4 and 9/17/33/65 nodes. Historical mesh-skeleton comparisons retain their five neuronal classes. V1DD models select epochs using **MICrONS validation-window macro F1 only**, while preserving separate MICrONS, V1DD, and combined final results. The paired axon/dendrite experiment retains its separate inner validation split; other historical experiments select on the reported held-out fold.

Source analysis: `{path}`. The original version, where present, is archived under `analysis/report_folds_20261012/legacy_notebooks/`. These pages cover the report's registered models; historical mixed-batch pilots are retained in that archive rather than mixed into the report comparison.
'''
        notebook=nb.v4.new_notebook()
        notebook.cells=[nb.v4.new_markdown_cell(header),nb.v4.new_code_cell(f'''%matplotlib inline
from pathlib import Path
import sys
from IPython.display import display
root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts/train_gnn.py').exists())
sys.path.insert(0,str(root/'analysis/report_folds_20261012'))
import all_folds as af
cache=af.load({category!r},family={family!r},model={model!r})
display(cache['status'][['label','completed_folds','complete']])
'''),nb.v4.new_markdown_cell('## Window and cell performance\nMean across folds with 95% fold confidence intervals.'),
            nb.v4.new_code_cell("display(af.summary(cache))\naf.plot_scores(cache)"),
            nb.v4.new_markdown_cell('## Cohort composition\nFor confidence sweeps: mean source-manifest cell counts and 95% fold CIs. Entire-cohort counts are identical across folds, so their intervals have zero width. These counts precede window eligibility filtering.'),nb.v4.new_code_cell('af.plot_cohort_counts(cache)'),
            nb.v4.new_markdown_cell('## Confidence / density trends\nFive-fold mean with shaded 95% fold CI, where applicable.'),nb.v4.new_code_cell('af.plot_trends(cache)'),
            nb.v4.new_markdown_cell('## Training progress\nMean curve and shaded 95% fold CI; no individual-fold overlays.'),nb.v4.new_code_cell('af.plot_curves(cache)'),
            nb.v4.new_markdown_cell('## Paired changes\nWithin-fold differences with 95% fold CI.'),nb.v4.new_code_cell('af.plot_deltas(cache)'),
            nb.v4.new_markdown_cell('## Class recall, precision, and support\nMeans and shaded 95% fold CIs.'),nb.v4.new_code_cell('af.plot_classes(cache)'),
            nb.v4.new_markdown_cell('## Confusion matrices\nEqual fold-weighted normalized confusion matrices, with 95% fold CI per cell.'),nb.v4.new_code_cell('af.plot_confusions(cache)')]
        notebook.metadata={'kernelspec':{'display_name':'segclr_db (.venv)','language':'python','name':'segclr_db'},
                           'report_fold_catalog':str((BASE/'plan.json').relative_to(ROOT)),'source_notebook':path}
        if category=='EM augmentation':
            notebook.cells += [nb.v4.new_markdown_cell('## How augmentation propagates through the clean-trained model\nClass-balanced samples of training windows, following the original diagnostic. Fold means and 95% fold CIs describe model/sample variation, not held-out accuracy.'),
                               nb.v4.new_code_cell('af.plot_augmentation_propagation()')]
        original.parent.mkdir(parents=True,exist_ok=True)
        nb.write(notebook,all_path)
        # The original entry point now uses the same registered reports and figures.
        nb.write(notebook,original)
        mapped.append({'original':path,'all_folds':str(all_path.relative_to(ROOT)),'category':category,'family':family,'model':model,'archived_original':str(backup.relative_to(ROOT)) if existed else None})
    if (BASE/'notebooks.json').exists():
        mapped += [r for r in json.loads((BASE/'notebooks.json').read_text()) if r['category']=='Whole-cell inference interventions']
    save(BASE/'notebooks.json',mapped)
    print(f'Mapped {len(mapped)} analyses and wrote {len(mapped)} *_all_folds.ipynb notebooks')

if __name__=='__main__':build()
