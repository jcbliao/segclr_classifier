"""Measure where embedding augmentation is attenuated by the trained GT."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from data.dataset_presynaptic import PresynapticWindowDataset
from gnn.model import WindowClassifier

MANIFEST = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_coarse_confidence_with_tc_folds/scale16/k17/conf0.7/fold0/manifest.json")
DATABASE = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10")
AUGMENTATIONS = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_embedding_training_choices/conf0.7/fold0/cutoff5000_fp16")
CHECKPOINT = Path("results/presynaptic/cave_embedding_augmentation_conf0.7/gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_embaug_clean_fold0/checkpoint_best.pt")
OUTPUT = Path("analysis/presynaptic/cave_embedding_augmentation_conf0.7/augmentation_propagation.json")
POLICIES = ("gray", "flip", "structured_low")


def cosine(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.cosine_similarity(a.float(), b.float()).cpu().numpy()


def summarize(values: list[np.ndarray]) -> dict:
    x = np.concatenate(values)
    return {"mean": float(x.mean()), "p01": float(np.quantile(x, .01)),
            "p05": float(np.quantile(x, .05)), "min": float(x.min())}


def mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    count = torch.bincount(batch, minlength=int(batch.max()) + 1).clamp_min(1)
    out = x.new_zeros((len(count), x.shape[1])); out.index_add_(0, batch, x)
    return out / count[:, None]


def main() -> None:
    torch.manual_seed(20260920)
    device = torch.device("cuda")
    manifest = json.loads(MANIFEST.read_text())
    clean = PresynapticWindowDataset(manifest, "train", "cave", database=DATABASE)
    rng = np.random.default_rng(20260920)
    indices = []
    for label in np.unique(clean.index_labels):
        candidates = np.flatnonzero(clean.index_labels == label)
        indices.extend(rng.choice(candidates, min(4000, len(candidates)), replace=False))
    indices = np.asarray(indices)
    # Preserve the balanced sample, but group it by cell so the two-entry
    # decompressed-cell cache is effective instead of reopening an NPZ for
    # nearly every randomly interleaved window.
    indices = indices[np.argsort(clean.index_root_ids[indices], kind="stable")]
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = WindowClassifier(checkpoint["config"], hierarchy=clean.hierarchy).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    clean_loader = DataLoader(Subset(clean, indices.tolist()), batch_size=512, shuffle=False)

    results = {"n_windows": int(len(indices)), "checkpoint_epoch": int(checkpoint["epoch"]),
               "policies": {}}
    for policy in POLICIES:
        augmented = PresynapticWindowDataset(
            manifest, "train", "cave", database=DATABASE,
            embedding_augmentation=policy, augmentation_database=AUGMENTATIONS,
        )
        aug_loader = DataLoader(Subset(augmented, indices.tolist()), batch_size=512, shuffle=False)
        metrics = {name: [] for name in ("node_cosine", "mean_cosine", "gt_cosine", "trunk_cosine")}
        n, agree, clean_correct, aug_correct, fixed, broken = 0, 0, 0, 0, 0, 0
        with torch.inference_mode():
            for clean_batch, aug_batch in zip(clean_loader, aug_loader, strict=True):
                clean_batch, aug_batch = clean_batch.to(device), aug_batch.to(device)
                if not torch.equal(clean_batch.root_id, aug_batch.root_id):
                    raise RuntimeError("paired loaders lost alignment")
                metrics["node_cosine"].append(cosine(clean_batch.x, aug_batch.x))
                metrics["mean_cosine"].append(cosine(
                    mean_pool(clean_batch.x, clean_batch.batch),
                    mean_pool(aug_batch.x, aug_batch.batch)))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    clean_g = model(clean_batch.x, clean_batch.edge_index, clean_batch.batch,
                        pos_enc=clean_batch.pos_enc, rel_pos=clean_batch.rel_pos,
                        has_segclr=clean_batch.has_segclr)
                    aug_g = model(aug_batch.x, aug_batch.edge_index, aug_batch.batch,
                        pos_enc=aug_batch.pos_enc, rel_pos=aug_batch.rel_pos,
                        has_segclr=aug_batch.has_segclr)
                    clean_h = model.cls_head.trunk(clean_g)
                    aug_h = model.cls_head.trunk(aug_g)
                    clean_pred = model.cls_head.predict_finest(clean_g)
                    aug_pred = model.cls_head.predict_finest(aug_g)
                metrics["gt_cosine"].append(cosine(clean_g, aug_g))
                metrics["trunk_cosine"].append(cosine(clean_h, aug_h))
                target = clean_batch.y_levels[:, -1]
                clean_ok, aug_ok = clean_pred == target, aug_pred == target
                n += len(target); agree += int((clean_pred == aug_pred).sum())
                clean_correct += int(clean_ok.sum()); aug_correct += int(aug_ok.sum())
                fixed += int((~clean_ok & aug_ok).sum()); broken += int((clean_ok & ~aug_ok).sum())
        results["policies"][policy] = {
            **{name: summarize(values) for name, values in metrics.items()},
            "prediction_agreement": agree / n,
            "clean_window_accuracy": clean_correct / n,
            "augmented_window_accuracy": aug_correct / n,
            "wrong_to_correct": fixed, "correct_to_wrong": broken,
        }
        print(policy, json.dumps(results["policies"][policy]), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
