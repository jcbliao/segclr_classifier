"""Generate stochastic SegCLR views for confidence-0.7 CAVE training nodes.

Only observed CAVE nodes used by a valid k=10 presynaptic window whose center
is within 5 um of a synapse are embedded. Test cells remain clean. One invocation
generates one complete ``(policy, sample)`` set across its assigned cells.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.build_presynaptic_axon_database import _atomic_savez as save_record, _sym_edges, _window, _window_arrays
from data.geodesic_window import build_csr_from_edges
from scripts.check_embedding_augmentation_stability import (
    CHECKPOINT, flips, gray, structured_noise_low,
)
from segclr.data.crops import CropLoader
from segclr.inference import SegCLRInference
from segclr.inference.inference import BOX_SIZE, EM_PATH, EM_SCALE, SEGMENTATION
from segclr.utils.checkpointing import load_checkpoint_automatic
from types import SimpleNamespace


DATABASE = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10")
MANIFEST = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0/manifest.json"
)
OUTPUT = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "cave_embedding_augmentations/conf0.7/fold0/cutoff5000"
)
POLICIES = {"gray": gray, "flip": flips, "structured_low": structured_noise_low}
MATCH_CUTOFF_NM = 5_000.0
CAVE_CACHE = Path("data/skeleton_cache")
TC_SOURCE = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_k17_full_tc_source/scale16/k17")


class _Skeleton:
    def __init__(self, coords): self.coords = coords
    def __len__(self): return len(self.coords)


class FastAugInference(SegCLRInference):
    """Production double-buffered crop pipeline with GPU-side augmentation."""
    def __init__(self, policy, sample, seed, batch_size=32, num_threads=8):
        self.batch_size=batch_size; self.num_threads=num_threads
        self.precision='tf32'; self.channels_last=True; self.allow_partial=False
        self.device=torch.device('cuda'); self.mat_version=1718
        self._apply_precision()
        seg_path,seg_scale=SEGMENTATION[1718]
        self.crops=CropLoader(em_path=EM_PATH,seg_path=seg_path,scale=EM_SCALE,
            seg_scale=seg_scale,box_size=BOX_SIZE,oob='pad')
        self.model=load_checkpoint_automatic(CHECKPOINT,self.device).eval()
        self.model=self.model.to(memory_format=torch.channels_last_3d)
        self.experiment=SimpleNamespace(embedding_dim=64)
        self._buffers=None; self._buffer_views=None; self._n_crop_failures=0
        self.policy=policy; self.sample=sample; self.seed=seed; self.generator=None

    def embed(self, root_id, coords, node_ids):
        policy_index=-1 if self.policy=='clean' else sorted(POLICIES).index(self.policy)
        key=self.seed+1_000_003*root_id+101*policy_index+self.sample
        self.generator=torch.Generator(device='cuda').manual_seed(key % (2**63-1))
        return self._embed(root_id,_Skeleton(coords),node_ids.astype(np.int64))

    def _forward(self, slot, n_ok):
        em_buf,mask_buf=self._buffers[slot]
        em=em_buf[:n_ok].to(self.device,non_blocking=True).float()/255.0
        mask=mask_buf[:n_ok].to(self.device,non_blocking=True)
        shape=(n_ok,1,1,1,1); gen=self.generator
        if self.policy=='clean':
            batch=em
        elif self.policy=='gray':
            alpha=.75+.5*torch.rand(shape,device=self.device,generator=gen)
            shift=-.25+.5*torch.rand(shape,device=self.device,generator=gen)
            gamma=2.0**(-1.+2.*torch.rand(shape,device=self.device,generator=gen))
            batch=(em*alpha+shift).clamp(0,1).pow(gamma)
        elif self.policy=='flip':
            batch=em
            # Per-crop flips retain all 2^3 combinations without serial crop work.
            for dim in (2,3,4):
                choose=torch.rand((n_ok,),device=self.device,generator=gen)<.5
                flipped=batch.flip(dim); flipped_mask=mask.flip(dim)
                view=choose.view(n_ok,1,1,1,1)
                batch=torch.where(view,flipped,batch); mask=torch.where(view,flipped_mask,mask)
        else:
            noise=torch.randn(em.shape,device=self.device,generator=gen)
            for _ in range(2):
                noise=(noise+noise.roll(1,3)+noise.roll(-1,3)+noise.roll(1,4)+noise.roll(-1,4))/5.
            noise=noise-noise.mean(dim=(3,4),keepdim=True)
            noise=noise/noise.std(dim=(3,4),keepdim=True).clamp_min(1e-6)
            sigma=.002+.008*torch.rand((n_ok,1,em.shape[2],1,1),device=self.device,generator=gen)
            batch=(em+sigma*noise).clamp(0,1)
        batch=torch.where(mask,batch,torch.zeros_like(batch)).to(memory_format=torch.channels_last_3d)
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            _,emb=self.model(batch,return_embeddings=True)
        return emb.float().cpu().numpy()


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def target_nodes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Observed nodes participating in the expanded 5 um window set."""
    with np.load(path) as z:
        pos = z["cave_pos_nm"].copy()
        edges = z["cave_edges"].copy()
        lengths = z["cave_edge_length_nm"].copy()
        observed = z["cave_observed"].copy()
        distances = z["cave_nearest_observed_distance_nm"].copy()
        centers = z["cave_nearest_observed_node"].copy()
        old_valid = z["cave_valid_k_window"].copy()
        old_offsets = z["cave_window_offsets"].copy()
        old_members = z["cave_window_members"].copy()

    used = np.zeros(len(pos), bool)
    for row in np.flatnonzero(old_valid):
        used[old_members[old_offsets[row]:old_offsets[row + 1]]] = True

    extra = np.flatnonzero((distances > 2_000.0) & (distances <= MATCH_CUTOFF_NM))
    if len(extra):
        offsets, neighbors, weights = build_csr_from_edges(
            _sym_edges(edges), np.concatenate([lengths, lengths])[:, None], len(pos)
        )
        cache: dict[int, np.ndarray | None] = {}
        for row in extra:
            center = int(centers[row])
            if center not in cache:
                found = _window(center, offsets, neighbors, weights, observed, 10)
                cache[center] = None if found is None else found[0]
            nodes = cache[center]
            if nodes is not None:
                used[nodes] = True

    node_ids = np.flatnonzero(used & observed).astype(np.int32)
    return node_ids, pos[node_ids].astype(np.float32)


def build_missing_record(root_id, database, aug_dest, inference, seed):
    """Build a soma-free CAVE record and gray/0 in the same crop worker."""
    with open(CAVE_CACHE / f"{root_id}.pkl", "rb") as f:
        cave = pickle.load(f)
    pos=np.asarray(cave.coords,np.float32); edges=np.asarray(cave.edges,np.int32)
    lengths=np.linalg.norm(pos[edges[:,0]].astype(np.float64)-pos[edges[:,1]].astype(np.float64),axis=1).astype(np.float32)
    ids=np.arange(len(pos),dtype=np.int32)
    inference.policy='clean'; got,clean_x=inference.embed(root_id,pos,ids); np.testing.assert_array_equal(got,ids)
    inference.policy='gray'; inference.sample=0; got,gray_x=inference.embed(root_id,pos,ids); np.testing.assert_array_equal(got,ids)
    sites=pd.read_parquet(TC_SOURCE/'sites'/f'{root_id}.parquet')
    observed=np.ones(len(pos),bool)
    windows=_window_arrays(sites,pos,edges,lengths,observed,10,match_cutoff_nm=MATCH_CUTOFF_NM)
    arrays=dict(root_id=np.asarray([root_id],np.uint64),root_cave_node_original=np.asarray([-1],np.int64),
        root_xyz_nm=np.full(3,np.nan,np.float32),soma_cut_applied=np.asarray(False),
        k_observed=np.asarray([10],np.int32),cave_pos_nm=pos,cave_edges=edges,
        cave_edge_length_nm=lengths,cave_observed_node_ids=ids,cave_observed_x=clean_x,cave_observed=observed)
    arrays.update({f'cave_{k}':v for k,v in windows.items()})
    save_record(database/'cells'/f'{root_id}.npz',**arrays)
    _atomic_savez(aug_dest,root_id=np.asarray(root_id,np.uint64),node_ids=ids,embeddings=gray_x,
        augmentation_id=np.asarray('gray'),augmentation_sample=np.asarray(0,np.int32),
        seed=np.asarray(seed),match_cutoff_nm=np.asarray(MATCH_CUTOFF_NM,np.float32),
        checkpoint_path=np.asarray(str(CHECKPOINT)))


def validate_output(path: Path, root_id: int, node_ids: np.ndarray,
                    policy: str, sample: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return (
                int(z["root_id"]) == root_id
                and np.array_equal(z["node_ids"], node_ids)
                and z["embeddings"].shape == (len(node_ids), 64)
                and str(z["augmentation_id"]) == policy
                and int(z["augmentation_sample"]) == sample
            )
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-tasks", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--policy", choices=sorted(POLICIES), required=True)
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--database", type=Path, default=DATABASE)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.sample < 0:
        raise ValueError("--sample must be non-negative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    manifest = json.loads(args.manifest.read_text())
    roots = sorted(
        int(root_id) for root_id, info in manifest["cells"].items()
        if info["split"] == "train"
    )
    # Multiplicative hashing distributes adjacent IDs and cell sizes better than
    # a contiguous or strided root-id split while remaining stable on resume.
    roots = [r for r in roots if ((r * 11400714819323198485) & ((1 << 64) - 1)) % args.num_tasks == args.task_id]
    if args.limit is not None:
        roots = roots[:args.limit]

    inference=FastAugInference(args.policy,args.sample,args.seed,args.batch_size,args.workers)

    completed = 0
    for cell_index, root_id in enumerate(roots, 1):
            source = args.database / "cells" / f"{root_id}.npz"
            dest = args.output / args.policy / f"sample{args.sample}" / "cells" / f"{root_id}.npz"
            if not source.exists():
                if (args.policy,args.sample) != ('gray',0):
                    raise RuntimeError(f"missing base record before {args.policy}/{args.sample}: {root_id}")
                build_missing_record(root_id,args.database,dest,inference,args.seed)
                print(f"[{cell_index}/{len(roots)}] {root_id}: built base record and gray/0",flush=True)
                completed+=1
                continue
            node_ids, xyz = target_nodes(source)
            if validate_output(dest, root_id, node_ids, args.policy, args.sample):
                print(f"[{cell_index}/{len(roots)}] {root_id}: cached {len(node_ids)}", flush=True)
                continue
            got,embeddings=inference.embed(root_id,xyz,np.arange(len(node_ids),dtype=np.int32))
            np.testing.assert_array_equal(got,np.arange(len(node_ids)))
            _atomic_savez(
                dest,
                root_id=np.asarray(root_id, np.uint64),
                node_ids=node_ids,
                embeddings=embeddings.astype(np.float32),
                augmentation_id=np.asarray(args.policy),
                augmentation_sample=np.asarray(args.sample, np.int32),
                seed=np.asarray(args.seed, np.int64),
                match_cutoff_nm=np.asarray(MATCH_CUTOFF_NM, np.float32),
                checkpoint_path=np.asarray(str(CHECKPOINT)),
            )
            completed += 1
            print(f"[{cell_index}/{len(roots)}] {root_id}: wrote {len(node_ids)} nodes", flush=True)
    print(json.dumps({"task_id": args.task_id, "policy": args.policy,
                      "sample": args.sample, "cells": len(roots),
                      "written": completed}), flush=True)


if __name__ == "__main__":
    main()
