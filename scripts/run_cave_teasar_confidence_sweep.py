"""Execute one frozen configuration from the audited paired sweep."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    job=json.loads((BASE/'plan.json').read_text())['jobs'][int(sys.argv[1])]
    cp=ROOT/job['config'];a=json.loads(cp.read_text())
    assert sha(cp)==job['config_sha256']
    assert sha(Path(a['manifest']))==job['manifest_sha256']
    assert sha(ROOT/job['template'])==job['template_sha256']
    assert a['class_balance']=='sample' and not a['freeze_aggregator'] and a['cls_resnet']
    assert a['num_workers']==15 and a['epochs']==30 and a['mixed_cell_batch_size']==16
    report=ROOT/job['report']
    if report.exists():
        d=json.loads(report.read_text());ignored={'num_workers','resume','manifest'}
        assert all(d['args'].get(k)==v for k,v in a.items() if k not in ignored)
        print('Matching final report already exists:',report,flush=True);return
    os.chdir(ROOT)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned GPU is unavailable; refusing CPU fallback')
    os.execv(sys.executable,[sys.executable,'-u','scripts/train_gnn.py','--config',str(cp)])
if __name__=='__main__':main()
