"""Run the existing augmentation propagation diagnostic for every clean fold."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import compare_augmentation_propagation as propagation

for fold in range(5):
    report=next((ROOT/'results/presynaptic/cave_embedding_augmentation_conf0.7').glob(f'gnn*_embaug_clean_sampled_fold{fold}.json'))
    args=json.loads(report.read_text())['args']
    propagation.MANIFEST=Path(args['manifest'])
    propagation.DATABASE=Path(args['presynaptic_database'])
    propagation.AUGMENTATIONS=Path(args['embedding_augmentation_database'])
    propagation.CHECKPOINT=report.parent/report.stem/'checkpoint_best.pt'
    propagation.OUTPUT=ROOT/f'analysis/report_folds_20261012/cache/augmentation_propagation_fold{fold}.json'
    if not propagation.OUTPUT.exists():propagation.main()
