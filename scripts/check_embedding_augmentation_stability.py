"""Measure SegCLR embedding stability under three image augmentations.

The diagnostic uses the exact resnet_860b_reshuffled checkpoint and crop loader
used for named-skeleton inference.  Every sampled crop is embedded clean and
under one isolated perturbation: the original grayscale transform, the original
axis flips, or mild per-section correlated noise.  Nothing is written to the
embedding database.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import torch

from segclr.data.crops import CropLoader, mask_and_normalize
from segclr.inference.inference import BOX_SIZE, EM_PATH, EM_SCALE, SEGMENTATION
from segclr.utils.checkpointing import load_checkpoint_automatic
from segclr.downstream.augmentations import (
    aug_curtaining,
    aug_curtaining_strong,
    aug_gaussian_noise_strong,
    aug_gaussian_noise_weak,
    aug_missing_sections,
    aug_per_slice_noise,
)


CHECKPOINT = Path(
    "/orcd/data/sdorkenw/001/collina/segclr_runs/"
    "resnet_860b_reshuffled_20260603_150412/checkpoints/checkpoint_e0_s95000.pt"
)
TOPOLOGY = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "resampled_teasar_2209/training_local_center/topology/full"
)
OUT = Path("analysis/presynaptic/embedding_augmentation_stability")


def gray(em: torch.Tensor, mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """The checkpoint's training-time brightness/contrast/gamma distribution."""
    alpha = 0.75 + 0.5 * torch.rand((), generator=gen)
    shift = -0.25 + 0.5 * torch.rand((), generator=gen)
    gamma = 2.0 ** (-1.0 + 2.0 * torch.rand((), generator=gen))
    aug = (em.float() / 255.0 * alpha + shift).clamp(0, 1).pow(gamma)
    return torch.where(mask, aug, torch.zeros_like(aug))


def flips(em: torch.Tensor, mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """The checkpoint's independent p=0.5 flips over Z, Y and X."""
    aug = em.float() / 255.0
    for dim in (1, 2, 3):
        if torch.rand((), generator=gen) < 0.5:
            aug = aug.flip(dim); mask = mask.flip(dim)
    return torch.where(mask, aug, torch.zeros_like(aug))


def structured_noise(em, mask, gen, sigma_lo=0.01, sigma_hi=0.05) -> torch.Tensor:
    """Independent per-section correlated noise at the requested RMS range.

    Each section gets low-pass-filtered random noise rather than voxel-white
    noise.  Two rounds of neighbor averaging give correlation without adding a
    scipy dependency to the GPU job; rescaling restores the requested RMS.
    """
    aug = em.float() / 255.0
    noise = torch.randn(aug.shape, generator=gen)
    for _ in range(2):
        noise = (
            noise + noise.roll(1, 2) + noise.roll(-1, 2)
            + noise.roll(1, 3) + noise.roll(-1, 3)
        ) / 5.0
    noise = noise - noise.mean(dim=(2, 3), keepdim=True)
    noise = noise / noise.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    sigma = sigma_lo + (sigma_hi - sigma_lo) * torch.rand(
        (1, aug.shape[1], 1, 1), generator=gen
    )
    aug = (aug + sigma * noise).clamp(0, 1)
    return torch.where(mask, aug, torch.zeros_like(aug))


def structured_noise_low(em, mask, gen):
    return structured_noise(em, mask, gen, 0.002, 0.01)


def structured_noise_very_low(em, mask, gen):
    return structured_noise(em, mask, gen, 0.001, 0.005)


def repo_augmentation(fn):
    """Adapt one repository NumPy perturbation to this masked-crop diagnostic."""
    def apply(em, mask, gen):
        rng = np.random.default_rng(gen.initial_seed())
        normalized = em[0].numpy().astype(np.float32) / 255.0
        augmented = torch.from_numpy(fn(normalized, rng=rng)).unsqueeze(0)
        return torch.where(mask, augmented, torch.zeros_like(augmented))
    return apply


CONDITIONS = (
    ("gray", gray),
    ("flip", flips),
    ("repo_gaussian_weak", repo_augmentation(aug_gaussian_noise_weak)),
    ("repo_gaussian_strong", repo_augmentation(aug_gaussian_noise_strong)),
    ("repo_slice_noise", repo_augmentation(aug_per_slice_noise)),
    ("repo_curtaining", repo_augmentation(aug_curtaining)),
    ("repo_curtaining_strong", repo_augmentation(aug_curtaining_strong)),
    ("repo_missing_sections", repo_augmentation(aug_missing_sections)),
    ("structured_noise_low", structured_noise_low),
    ("structured_noise_very_low", structured_noise_very_low),
)


def sample_locations(n_cells: int, nodes_per_cell: int, seed: int):
    rng = np.random.default_rng(seed)
    paths = sorted(TOPOLOGY.glob("*.npz"))
    chosen = rng.choice(len(paths), min(n_cells, len(paths)), replace=False)
    samples = []
    for pi in chosen:
        path = paths[int(pi)]
        with np.load(path) as z:
            pos = z["pos_nm"]
            root_id = int(z["root_id"])
        ids = rng.choice(len(pos), min(nodes_per_cell, len(pos)), replace=False)
        samples.extend((root_id, int(i), pos[int(i)]) for i in ids)
    return samples


def nearest_other(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Nearest clean reference other than the same crop index."""
    q = torch.nn.functional.normalize(query, dim=1)
    r = torch.nn.functional.normalize(reference, dim=1)
    sim = q @ r.T
    sim.fill_diagonal_(-torch.inf)
    return sim.argmax(dim=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-cells", type=int, default=16)
    p.add_argument("--nodes-per-cell", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    seg_path, seg_scale = SEGMENTATION[1718]
    crops = CropLoader(
        em_path=EM_PATH, seg_path=seg_path, scale=EM_SCALE, seg_scale=seg_scale,
        box_size=BOX_SIZE, oob="pad",
    )
    model = load_checkpoint_automatic(CHECKPOINT, device).eval()
    model = model.to(memory_format=torch.channels_last_3d)

    samples = sample_locations(args.n_cells, args.nodes_per_cell, args.seed)
    outputs = {name: [] for name in ("clean", *(name for name, _ in CONDITIONS))}
    kept = []

    def fetch(item):
        root_id, node_id, xyz = item
        center = crops.nm_to_voxel(xyz)
        em, mask = crops.fetch_crop(center, root_id)
        return root_id, node_id, em, mask

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(samples), args.batch_size):
            batch = list(pool.map(fetch, samples[start:start + args.batch_size]))
            variants = {name: [] for name in outputs}
            for root_id, node_id, em, mask in batch:
                key = args.seed + 1000003 * root_id + node_id
                variants["clean"].append(mask_and_normalize(em, mask))
                for offset, (name, fn) in enumerate(CONDITIONS, 1):
                    gen = torch.Generator().manual_seed((key + offset) % (2**63 - 1))
                    variants[name].append(fn(em, mask, gen))
                kept.append((root_id, node_id))
            for name, views in variants.items():
                x = torch.stack(views).to(device, non_blocking=True)
                x = x.to(memory_format=torch.channels_last_3d)
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    _, emb = model(x, return_embeddings=True)
                outputs[name].append(emb.float().cpu())
            print(f"embedded {min(start + len(batch), len(samples))}/{len(samples)}", flush=True)

    outputs = {k: torch.cat(v) for k, v in outputs.items()}
    clean = outputs["clean"]
    clean_nn = nearest_other(clean, clean)
    kept_roots = torch.tensor([x[0] for x in kept])
    rows = []
    for name, x in outputs.items():
        cos = torch.nn.functional.cosine_similarity(clean, x, dim=1)
        norm_ratio = x.norm(dim=1) / clean.norm(dim=1).clamp_min(1e-12)
        variant_nn = nearest_other(x, clean)
        row = {
            "condition": name,
            "n": len(x),
            "cosine_mean": cos.mean().item(),
            "cosine_p01": torch.quantile(cos, 0.01).item(),
            "cosine_p05": torch.quantile(cos, 0.05).item(),
            "cosine_min": cos.min().item(),
            "norm_ratio_mean": norm_ratio.mean().item(),
            "norm_ratio_p05": torch.quantile(norm_ratio, 0.05).item(),
            "norm_ratio_p95": torch.quantile(norm_ratio, 0.95).item(),
            "nearest_neighbor_retention": (variant_nn == clean_nn).float().mean().item(),
            "nearest_neighbor_same_cell": (
                kept_roots[variant_nn] == kept_roots
            ).float().mean().item(),
        }
        rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(CHECKPOINT),
        "seed": args.seed,
        "n_cells_requested": args.n_cells,
        "nodes_per_cell_requested": args.nodes_per_cell,
        "samples": len(kept),
        "conditions": rows,
    }
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    torch.save({"samples": kept, **outputs}, args.out / "embeddings.pt")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
