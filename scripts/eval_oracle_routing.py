#!/usr/bin/env python3
"""Compare normal top-down LCPN inference with true-parent oracle routing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.dataset_lcpn import load_hierarchy, load_manifest
from data.dataset_windowed import WindowedGraphDatasetLCPN
from gnn.metrics import majority_vote_by_group, summarize
from gnn.model import WindowClassifier

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUN = "gnn_lcpn_scratch_mpnn_L2_position_lpe_resnet4x128_n40"


def cell_metrics(labels, predictions, roots, classes):
    cell_true, cell_pred = majority_vote_by_group(roots, labels, predictions)
    return summarize(cell_true, cell_pred, len(classes), classes)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", nargs="?", default=DEFAULT_RUN)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=15)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--limit-batches", type=int, help="smoke-test only")
    args = ap.parse_args()

    result_path = REPO / "results" / "all_windows" / f"{args.run}.json"
    checkpoint_path = REPO / "results" / "all_windows" / args.run / "checkpoint_best.pt"
    metadata = json.loads(result_path.read_text())
    train_args = metadata["args"]
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WindowClassifier(checkpoint["config"], hierarchy)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset = WindowedGraphDatasetLCPN(
        manifest, "test", pos_dim=int(train_args["gt_pos_dim"]),
        use_thickness=bool(train_args.get("gt_use_thickness", False)),
        num_embeddings=int(train_args["num_embeddings"]),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    labels, roots, top_down, oracle = [], [], [], []
    total = len(loader) if args.limit_batches is None else min(len(loader), args.limit_batches)
    for batch_index, batch in enumerate(tqdm(loader, total=total, desc="oracle routing")):
        if args.limit_batches is not None and batch_index >= args.limit_batches:
            break
        batch = batch.to(device)
        hidden = model(
            batch.x, batch.edge_index, batch.batch, batch.pos_enc, batch.rel_pos,
            getattr(batch, "thickness", None),
        )
        top_down.append(model.cls_head.predict_top_down(hidden)[:, -1].cpu().numpy())
        oracle.append(
            model.cls_head.predict_oracle_routed(hidden, batch.y_levels)[:, -1].cpu().numpy()
        )
        labels.append(batch.y_levels[:, -1].cpu().numpy())
        roots.append(batch.root_id.cpu().numpy().reshape(-1))

    labels_np, roots_np = np.concatenate(labels), np.concatenate(roots)
    predictions = {"top_down": np.concatenate(top_down), "oracle_routed": np.concatenate(oracle)}
    classes = list(dataset.classes)
    conditions = {}
    for name, pred in predictions.items():
        conditions[name] = {
            "window": summarize(labels_np, pred, len(classes), classes),
            "cell": cell_metrics(labels_np, pred, roots_np, classes),
        }
    payload = {
        "run": args.run, "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"), "split": "test/validation",
        "n_windows": int(len(labels_np)), "n_cells": int(len(np.unique(roots_np))),
        "oracle_definition": "true parent selects each local head; selected head predicts child",
        "conditions": conditions,
        "delta_oracle_minus_top_down": {
            granularity: {
                metric: conditions["oracle_routed"][granularity][metric]
                - conditions["top_down"][granularity][metric]
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            }
            for granularity in ("window", "cell")
        },
    }
    output = args.output or REPO / "results" / "all_windows" / args.run / "oracle_routing.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(output), "run": args.run,
        "top_down": {g: {m: conditions["top_down"][g][m]
                         for m in ("accuracy", "balanced_accuracy", "macro_f1")}
                     for g in ("window", "cell")},
        "oracle_routed": {g: {m: conditions["oracle_routed"][g][m]
                              for m in ("accuracy", "balanced_accuracy", "macro_f1")}
                          for g in ("window", "cell")},
        "delta": payload["delta_oracle_minus_top_down"],
    }, indent=2))


if __name__ == "__main__":
    main()
