"""Generate the complete gray, deterministic-flip, and structured set per crop."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from segclr.inference import SegCLRInference
from scripts.generate_cave_embedding_augmentations import (
    CHECKPOINT, DATABASE, MANIFEST, OUTPUT, FastAugInference, _Skeleton,
    _atomic_savez, build_missing_record, target_nodes, validate_output,
)

VARIANTS = (
    *(("gray", sample) for sample in range(20)),
    *(("flip", sample) for sample in range(7)),
    *(("structured_low", sample) for sample in range(4)),
)

class MultiAugInference(FastAugInference):
    def __init__(self, *args, variants_per_forward=4, compile_model=False,
                 compare_compile=False, amp="none", compare_amp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.variants_per_forward = variants_per_forward
        self.eager_model = self.model
        self.compare_compile = compare_compile
        self._compared_compile = False
        self.amp_dtype = {"none": None, "bf16": torch.bfloat16,
                          "fp16": torch.float16}[amp]
        self.compare_amp = compare_amp
        self._compared_amp = False
        if compile_model:
            # A fixed effective batch gives Inductor one reusable graph. Partial
            # crop batches are padded per variant below and trimmed after inference.
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)

    def embed(self,root_id,coords,node_ids):
        self._generators={}
        for pi,(policy,sample) in enumerate(VARIANTS):
            key=self.seed+1_000_003*root_id+101*pi+sample
            self._generators[(policy,sample)]=torch.Generator(device='cuda').manual_seed(key%(2**63-1))
        return SegCLRInference._embed(self,root_id,_Skeleton(coords),node_ids.astype(np.int64))
    def _forward(self,slot,n_ok):
        # Transfer and normalize the crop once, then amortize each model call over
        # several independently augmented views.  Variant-major concatenation
        # makes it cheap to split the model output back into aligned node rows.
        em_buf, mask_buf = self._buffers[slot]
        em = em_buf[:n_ok].to(self.device, non_blocking=True).float().div_(255.0)
        base_mask = mask_buf[:n_ok].to(self.device, non_blocking=True)
        outputs = []
        for start in range(0, len(VARIANTS), self.variants_per_forward):
            batches = []
            for policy, sample in VARIANTS[start:start + self.variants_per_forward]:
                gen = self._generators[(policy, sample)]
                shape = (n_ok, 1, 1, 1, 1)
                mask = base_mask
                if policy == "gray":
                    alpha = .75 + .5 * torch.rand(shape, device=self.device, generator=gen)
                    shift = -.25 + .5 * torch.rand(shape, device=self.device, generator=gen)
                    gamma = 2.0 ** (-1. + 2. * torch.rand(shape, device=self.device, generator=gen))
                    batch = (em * alpha + shift).clamp(0, 1).pow(gamma)
                elif policy == "flip":
                    batch = em
                    # Samples 0..6 enumerate every nonidentity element of the
                    # three-axis flip group exactly once.  All nodes in a
                    # sample receive the same transform, so the seven views
                    # have equal, deterministic coverage rather than random
                    # duplication and omission.
                    bits = sample + 1
                    for bit, dim in enumerate((2, 3, 4)):
                        if bits & (1 << bit):
                            batch = batch.flip(dim)
                            mask = mask.flip(dim)
                else:
                    noise = torch.randn(em.shape, device=self.device, generator=gen)
                    for _ in range(2):
                        noise = (noise + noise.roll(1, 3) + noise.roll(-1, 3)
                                 + noise.roll(1, 4) + noise.roll(-1, 4)) / 5.
                    noise = noise - noise.mean(dim=(3, 4), keepdim=True)
                    noise = noise / noise.std(dim=(3, 4), keepdim=True).clamp_min(1e-6)
                    sigma = .002 + .008 * torch.rand(
                        (n_ok, 1, em.shape[2], 1, 1), device=self.device, generator=gen)
                    batch = (em + sigma * noise).clamp(0, 1)
                batch = torch.where(mask, batch, torch.zeros_like(batch))
                if n_ok < self.batch_size:
                    batch = torch.cat((batch, batch.new_zeros(
                        (self.batch_size - n_ok, *batch.shape[1:]))))
                batches.append(batch)
            model_batch = torch.cat(batches).to(memory_format=torch.channels_last_3d)
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                _, embeddings = self.model(model_batch, return_embeddings=True)
                if self.compare_compile and not self._compared_compile:
                    _, eager = self.eager_model(model_batch, return_embeddings=True)
                    cos = torch.nn.functional.cosine_similarity(embeddings.float(), eager.float())
                    print(json.dumps(dict(
                        paired_compile_cosine_mean=float(cos.mean()),
                        paired_compile_cosine_min=float(cos.min()),
                        paired_compile_max_abs=float((embeddings.float()-eager.float()).abs().max()),
                    )), flush=True)
                    self._compared_compile = True
            if self.compare_amp and not self._compared_amp:
                paired = {}
                with torch.inference_mode():
                    with torch.autocast("cuda", enabled=False):
                        _, ref = self.eager_model(model_batch, return_embeddings=True)
                    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
                        with torch.autocast("cuda", dtype=dtype):
                            _, test = self.eager_model(model_batch, return_embeddings=True)
                        cos = torch.nn.functional.cosine_similarity(ref.float(), test.float())
                        paired[name] = dict(cosine_mean=float(cos.mean()),
                            cosine_min=float(cos.min()),
                            max_abs=float((ref.float()-test.float()).abs().max()))
                print(json.dumps({"paired_amp": paired}), flush=True)
                self._compared_amp = True
            outputs.extend(x[:n_ok].float().cpu().numpy()
                           for x in embeddings.split(self.batch_size))
        return np.concatenate(outputs, axis=1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-id',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID',0)))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--seed',type=int,default=20260920)
    p.add_argument('--database',type=Path,default=DATABASE);p.add_argument('--manifest',type=Path,default=MANIFEST);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--variants-per-forward', type=int, default=4)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--compare-compile', action='store_true')
    p.add_argument('--amp', choices=('none','bf16','fp16'), default='none')
    p.add_argument('--compare-amp', action='store_true')
    a=p.parse_args(); plan=json.loads(a.plan.read_text()); roots=plan['shards'][a.task_id]
    memory=torch.cuda.get_device_properties(0).total_memory/2**30
    batch=a.batch_size or (64 if memory>=100 else 48 if memory>=70 else 32)
    inference=MultiAugInference('gray',0,a.seed,batch_size=batch,num_threads=a.workers,
        variants_per_forward=a.variants_per_forward,
        compile_model=a.compile or a.compare_compile,compare_compile=a.compare_compile,
        amp=a.amp,compare_amp=a.compare_amp)
    record_runner=None; written=0
    for ci,rid in enumerate(roots,1):
        rid=int(rid); source=a.database/'cells'/f'{rid}.npz'
        dests=[a.output/policy/f'sample{sample}'/'cells'/f'{rid}.npz' for policy,sample in VARIANTS]
        if not source.exists():
            if record_runner is None: record_runner=FastAugInference('clean',0,a.seed,batch,16)
            build_missing_record(rid,a.database,dests[0],record_runner,a.seed)
        node_ids,xyz=target_nodes(source)
        done=[validate_output(path,rid,node_ids,p,s) for path,(p,s) in zip(dests,VARIANTS,strict=True)]
        if all(done): print(f'[{ci}/{len(roots)}] {rid}: all cached',flush=True);continue
        got,all_embeddings=inference.embed(rid,xyz,np.arange(len(node_ids),dtype=np.int32))
        np.testing.assert_array_equal(got,np.arange(len(node_ids)))
        assert all_embeddings.shape==(len(node_ids),len(VARIANTS)*64)
        for i,((policy,sample),dest,is_done) in enumerate(zip(VARIANTS,dests,done,strict=True)):
            if is_done: continue
            _atomic_savez(dest,root_id=np.asarray(rid,np.uint64),node_ids=node_ids,
                embeddings=all_embeddings[:,i*64:(i+1)*64],augmentation_id=np.asarray(policy),
                augmentation_sample=np.asarray(sample,np.int32),seed=np.asarray(a.seed,np.int64),
                match_cutoff_nm=np.asarray(5000.,np.float32),checkpoint_path=np.asarray(str(CHECKPOINT)),
                inference_precision=np.asarray(a.amp))
        written+=1;print(f'[{ci}/{len(roots)}] {rid}: wrote missing variants for {len(node_ids)} nodes',flush=True)
    print(json.dumps(dict(task_id=a.task_id,cells=len(roots),written=written,batch_size=batch,
        variants_per_forward=a.variants_per_forward,compiled=a.compile,amp=a.amp)),flush=True)
if __name__=='__main__':main()
