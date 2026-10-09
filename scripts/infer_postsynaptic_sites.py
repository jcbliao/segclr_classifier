"""Resumable SegCLR inference at exact postsynaptic sites, mixing roots per batch."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path('/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1718')
CHECKPOINT = Path('/orcd/data/sdorkenw/001/collina/segclr_runs/resnet_860b_reshuffled_20260603_150412/checkpoints/checkpoint_e0_s95000.pt')


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def prepare(out, shard_size):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['shard_size'] != shard_size:
            raise ValueError('Existing dataset has a different shard size')
        return manifest
    sources = [REPO / 'data/synapse_cache/presynaptic_sites.parquet']
    extra = REPO / 'analysis/postsynaptic_targets_conf0.7/additional_synapses.parquet'
    if extra.exists():
        sources.append(extra)
    cols = ['synapse_id', 'cell_root_id', 'partner_root_id', 'partner_x_nm', 'partner_y_nm', 'partner_z_nm']
    frame = pd.concat([pd.read_parquet(p, columns=cols) for p in sources], ignore_index=True)
    # Overlapping source caches must agree on both identity and geometry.
    distinct = frame.drop_duplicates()
    if distinct.synapse_id.duplicated().any():
        raise ValueError('Conflicting paired sites for the same synapse ID')
    frame = distinct.rename(columns={'cell_root_id': 'presynaptic_root_id', 'partner_root_id': 'postsynaptic_root_id',
        'partner_x_nm': 'x_nm', 'partner_y_nm': 'y_nm', 'partner_z_nm': 'z_nm'})
    if not np.isfinite(frame[['x_nm', 'y_nm', 'z_nm']].to_numpy()).all():
        raise ValueError('Nonfinite postsynaptic coordinates')
    # Spatial order improves volume cache reuse; it does not group by cell.
    frame = frame.sort_values(['z_nm', 'y_nm', 'x_nm', 'synapse_id']).reset_index(drop=True)
    (out / 'sites').mkdir(exist_ok=True)
    (out / 'embeddings').mkdir(exist_ok=True)
    for part, start in enumerate(range(0, len(frame), shard_size)):
        frame.iloc[start:start + shard_size].to_parquet(out / 'sites' / f'part-{part:05d}.parquet', index=False)
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    manifest = dict(datastack='minnie65_public', materialization_version=1718, synapse_table='synapses_pni_2',
        sources=[str(p) for p in sources], n_sites=len(frame), n_unresolved=int((frame.postsynaptic_root_id == 0).sum()),
        n_shards=(len(frame) + shard_size - 1) // shard_size, shard_size=shard_size,
        experiment_id='resnet_860b_reshuffled', run_id='resnet_860b_reshuffled__20260603_150412',
        checkpoint_id='checkpoint_e0_s95000', checkpoint_path=str(CHECKPOINT), checkpoint_sha256=digest,
        embedding_dim=64, coordinate_units='nm', side='post', crop_size_voxels=129,
        masking='postsynaptic_root_id', normalization='EM / 255, zero outside mask', precision='fp16', compiled=True, compile_mode='reduce-overhead',
        batching='spatially ordered sites across roots; adaptive GPU batch size',
        statuses=['ok', 'unresolved_root', 'crop_failed', 'empty_mask'])
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def make_inference(batch_size, threads, dataset_manifest=None):
    import torch
    from segclr.inference import SegCLRInference
    from segclr.data.crops import CropLoader, mask_and_normalize
    from segclr.inference.inference import EM_PATH, EM_SCALE, SEGMENTATION, BOX_SIZE
    from segclr.utils.checkpointing import load_checkpoint_automatic

    class SiteInference(SegCLRInference):
        def __init__(self):
            self.batch_size = batch_size
            self.num_threads = threads
            self.precision = 'tf32'  # Enables TF32 outside autocast
            self.channels_last = True
            self.allow_partial = True
            self.device = torch.device('cuda')
            self.mat_version = dataset_manifest['materialization_version'] if dataset_manifest else 1718
            self._apply_precision()
            if dataset_manifest and dataset_manifest['datastack']=='v1dd_public':
                from v1dd_postsynaptic_crops import V1DDPostCrops
                self.crops = V1DDPostCrops(dataset_manifest)
            else:
                seg_path, seg_scale = SEGMENTATION[1718]
                self.crops = CropLoader(EM_PATH, seg_path, EM_SCALE, BOX_SIZE, oob='pad', seg_scale=seg_scale)
            self.model = load_checkpoint_automatic(CHECKPOINT, self.device).eval().to(memory_format=torch.channels_last_3d)
            assert self.model.bottleneck_dim == 64
            self.model = torch.compile(self.model, mode='reduce-overhead', dynamic=False)
            self.experiment = SimpleNamespace(embedding_dim=64)
            self._buffers = self._buffer_views = None
            self._n_crop_failures = 0
            self.gpu_batch_size = batch_size

        def _fill(self, pool, slot, root_id, centers, node_ids, start):
            em_view, mask_view = self._buffer_views[slot]
            counter, lock = [0], threading.Lock()
            def read(index):
                site = int(node_ids[index])
                try:
                    em, mask = self.crops.fetch_crop(centers[index], int(self.roots[site]))
                    if not mask.numpy().any():
                        self.errors[site] = ('empty_mask', 'No postsynaptic-root voxels in crop')
                        raise ValueError('Empty postsynaptic mask')
                except Exception as exc:
                    self.errors.setdefault(site, ('crop_failed', str(exc)[:1000]))
                    raise
                with lock:
                    row = counter[0]
                    counter[0] += 1
                em_view[row] = em.numpy()
                mask_view[row] = mask.numpy()
                return row
            return {pool.submit(read, i): int(node_ids[i]) for i in range(start, min(start + self.batch_size, len(node_ids)))}

        def _forward(self, slot, n_ok):
            result = []
            start = 0
            while start < n_ok:
                size = min(self.gpu_batch_size, n_ok - start)
                def forward():
                    em, mask = self._buffers[slot]
                    batch = mask_and_normalize(em[start:start+size].to(self.device, non_blocking=True),
                        mask[start:start+size].to(self.device, non_blocking=True)).to(memory_format=torch.channels_last_3d)
                    if size < self.gpu_batch_size:
                        batch = torch.cat((batch, batch.new_zeros((self.gpu_batch_size-size, *batch.shape[1:]))))
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        _, vector = self.model(batch, return_embeddings=True)
                    return vector[:size].float().cpu().numpy()
                try:
                    vectors = forward()
                except torch.cuda.OutOfMemoryError:
                    if dataset_manifest and dataset_manifest['datastack']=='v1dd_public':
                        raise  # Preserve the existing fixed-128 compilation cache.
                    if size == 1:
                        raise
                    self.gpu_batch_size = max(1, size // 2)
                    torch.cuda.empty_cache()
                    print(f'GPU batch reduced to {self.gpu_batch_size}', flush=True)
                    continue
                if vectors.shape != (size, 64) or not np.isfinite(vectors).all():
                    raise ValueError('Invalid model embeddings')
                result.append(vectors)
                start += size
            return np.concatenate(result)
    return SiteInference()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--finalize', action='store_true')
    ap.add_argument('--shard-size', type=int, default=8192)
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--world', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--limit-shards', type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.prepare:
        prepare(args.output, args.shard_size)
        return
    manifest = json.loads((args.output / 'manifest.json').read_text())
    if args.finalize:
        totals = {}
        for part in range(manifest['n_shards']):
            name = f'part-{part:05d}.parquet'
            sites = pd.read_parquet(args.output / 'sites' / name, columns=['synapse_id'])
            result = pd.read_parquet(args.output / 'embeddings' / name)
            np.testing.assert_array_equal(sites.synapse_id, result.synapse_id)
            for status, count in result.status.value_counts().items():
                totals[status] = totals.get(status, 0) + int(count)
            good = result[result.status == 'ok']
            if len(good):
                vectors = np.stack(good.embedding)
                assert vectors.shape == (len(good), 64) and np.isfinite(vectors).all()
        assert sum(totals.values()) == manifest['n_sites']
        atomic_json(args.output / 'summary.json', dict(n_sites=manifest['n_sites'], n_shards=manifest['n_shards'], statuses=totals))
        print(totals, flush=True)
        return
    assert 0 <= args.rank < args.world
    if manifest['datastack']=='v1dd_public':
        ap.error('V1DD defaults to shared CPU preparation; use prepare_v1dd_post_batch_queue.py and dispatch_v1dd_post_jobs.py')
    inference = None
    completed = 0
    for part in range(args.rank, manifest['n_shards'], args.world):
        source = args.output / 'sites' / f'part-{part:05d}.parquet'
        dest = args.output / 'embeddings' / source.name
        if dest.exists():
            continue
        frame = pd.read_parquet(source)
        if inference is None:
            inference = make_inference(args.batch_size, args.threads, manifest)
        inference.roots = frame.postsynaptic_root_id.to_numpy(np.int64)
        inference.errors = {}
        valid = np.flatnonzero(inference.roots != 0)
        xyz = frame[['x_nm', 'y_nm', 'z_nm']].to_numpy()
        # _embed needs a length-bearing coordinate container.
        class Points:
            coords = xyz
            def __len__(self): return len(xyz)
        got, vectors = inference._embed(0, Points(), valid)
        if len(valid) and not len(got):
            raise RuntimeError('Every crop failed; refusing to publish a failed shard')
        frame['status'] = np.where(inference.roots == 0, 'unresolved_root', 'crop_failed')
        frame['error'] = ''
        frame['embedding'] = pd.Series([None] * len(frame), dtype=object)
        for index, (status, error) in inference.errors.items():
            frame.at[index, 'status'] = status
            frame.at[index, 'error'] = error
        for index, vector in zip(got, vectors):
            frame.at[int(index), 'status'] = 'ok'
            frame.at[int(index), 'embedding'] = vector
        if manifest['datastack']=='v1dd_public' and frame.status.eq('crop_failed').any():
            raise RuntimeError('Transient V1DD crop failures must be retried before publishing this shard')
        tmp = dest.with_suffix('.tmp.parquet')
        frame.to_parquet(tmp, index=False)
        # Verify exact identity/order and vector roundtrip before publishing shard.
        check = pd.read_parquet(tmp)
        np.testing.assert_array_equal(check.synapse_id, frame.synapse_id)
        if len(got):
            np.testing.assert_array_equal(np.stack(check.iloc[got].embedding), vectors)
        os.replace(tmp, dest)
        print(f'part {part}: {frame.status.value_counts().to_dict()}, GPU batch={inference.gpu_batch_size}', flush=True)
        completed += 1
        if args.limit_shards and completed >= args.limit_shards:
            break

if __name__ == '__main__':
    main()
