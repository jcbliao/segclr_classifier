"""Exercise real FP16 trainer, saved scaler state and checkpoint resume on CUDA."""
import json,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'analysis/presynaptic/cave_teasar_confidence_5fold_20261010/performance'
def main():
    assert torch.cuda.is_available()
    template=json.loads((BASE/'smoke_config.json').read_text())
    for arch in ['mean','pointwise_mlp','graph_transformer']:
        for epochs in ([1,2] if arch=='mean' else [1]):
            a=dict(template,architecture=arch,epochs=epochs,resume=epochs>1,amp=True,num_workers=2,results_dir=str(BASE/'fp16_smoke'/arch))
            config=BASE/f'fp16_smoke_{arch}.json';config.write_text(json.dumps(a))
            with (BASE/f'fp16_smoke_{arch}_{epochs}.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/train_gnn.py'),'--config',str(config)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            checkpoints=list(Path(a['results_dir']).glob('*/checkpoint_last.pt'));assert len(checkpoints)==1
            ckpt=torch.load(checkpoints[0],map_location='cpu',weights_only=False)
            assert ckpt['epoch']==epochs-1 and ckpt['amp_dtype']=='float16' and ckpt['grad_scaler_state']
            assert all(group['fused'] for group in ckpt['optimizer_state']['param_groups'])
            if arch=='mean' and epochs==1:
                # Simulate the pre-rollout Adam checkpoint: CPU step counters and no fused flag.
                for group in ckpt['optimizer_state']['param_groups']:group['fused']=None
                for state in ckpt['optimizer_state']['state'].values():state['step']=state['step'].cpu()
                torch.save(ckpt,checkpoints[0])
            for value in ckpt['model_state'].values():assert torch.isfinite(value).all()
            print(f'FP16 trainer + scaler checkpoint passed: {arch}, {epochs} epochs',flush=True)
    (BASE/'fp16_validation.json').write_text(json.dumps({'passed':True,'architectures':['mean','pointwise_mlp','graph_transformer'],'mean_resume':True},indent=2))
if __name__=='__main__':main()
