"""Evaluate trained window classifiers under arrangement/topology shuffles.

This is an inference-only diagnostic: it loads existing ``checkpoint_best.pt``
files, applies deterministic interventions within each PyG window, and reports
window- and cell-level metrics through the same code used by training.

The important control is the mean architecture.  It reads neither node order
nor edges, so its predictions must be exactly unchanged by ``permute_x`` and
the edge interventions.  A failed invariance check means the harness is wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from data.dataset_windowed import WindowedGraphDatasetLCPN  # noqa: E402
from gnn.metrics import majority_vote_by_group, summarize  # noqa: E402
from gnn.model import WindowClassifier  # noqa: E402


DEFAULT_CHECKPOINTS = (
    "results/gnn_lcpn_scratch_meanpool/checkpoint_best.pt",
    "results/gnn_lcpn_scratch_meanpool_resnet4x128/checkpoint_best.pt",
    "results/gnn_lcpn_scratch_mpnn_L2_spatial/checkpoint_best.pt",
    "results/gnn_lcpn_scratch_mpnn_L2_spatial_resnet4x128/checkpoint_best.pt",
    "results/gnn_lcpn_scratch_mpnn_L2_spatial_frozenagg/checkpoint_best.pt",
)
CONDITIONS = (
    "none",
    "permute_x",
    "shuffle_edges",
    "drop_edges",
    "zero_pos_enc",
    "shuffle_edges_zero_pos_enc",
    "drop_edges_zero_pos_enc",
    "zero_rel_pos",
)


def within_window_permutation(ptr: torch.Tensor, seed: int, device: torch.device) -> torch.Tensor:
    """A global-node permutation whose mappings never cross graph boundaries."""
    generator = torch.Generator().manual_seed(seed)
    pieces = []
    # Generate on CPU so the same seed gives the same intervention on CPU/GPU.
    for start, stop in zip(ptr[:-1].cpu().tolist(), ptr[1:].cpu().tolist(), strict=True):
        pieces.append(torch.randperm(stop - start, generator=generator) + start)
    return torch.cat(pieces).to(device=device)


def intervened_inputs(data, condition: str, seed: int) -> dict[str, torch.Tensor | None]:
    """Return model inputs after one intervention; never mutate ``data``."""
    x = data.x
    edge_index = data.edge_index
    pos_enc = data.pos_enc
    rel_pos = data.rel_pos

    if condition == "permute_x":
        x = x[within_window_permutation(data.ptr, seed, x.device)]
    elif condition in ("shuffle_edges", "shuffle_edges_zero_pos_enc"):
        # Relabel both endpoints by the same within-window permutation.  This
        # preserves each window's abstract graph and degree multiset while
        # breaking which feature/position is attached to which graph vertex.
        mapping = within_window_permutation(data.ptr, seed, edge_index.device)
        edge_index = mapping[edge_index]
    elif condition in ("drop_edges", "drop_edges_zero_pos_enc"):
        edge_index = edge_index.new_empty((2, 0))

    if condition in (
        "zero_pos_enc", "shuffle_edges_zero_pos_enc", "drop_edges_zero_pos_enc"
    ):
        pos_enc = torch.zeros_like(pos_enc)
    if condition == "zero_rel_pos":
        rel_pos = torch.zeros_like(rel_pos)

    return {
        "x": x,
        "edge_index": edge_index,
        "batch_index": data.batch,
        "pos_enc": pos_enc,
        "rel_pos": rel_pos,
        "thickness": getattr(data, "thickness", None),
    }


def metrics(labels, preds, root_ids, classes):
    window = summarize(labels, preds, len(classes), classes)
    cell_true, cell_pred = majority_vote_by_group(root_ids, labels, preds)
    cell = summarize(cell_true, cell_pred, len(classes), classes)
    return window, cell


@torch.inference_mode()
def evaluate_checkpoint(path: Path, dataset, hierarchy, classes, args, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = WindowClassifier(config, hierarchy=hierarchy).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    predictions = defaultdict(list)
    labels, root_ids = [], []
    for batch_no, data in enumerate(tqdm(loader, desc=path.parent.name, unit="batch")):
        data = data.to(device)
        labels.append(data.y_levels[:, -1].cpu().numpy())
        root_ids.append(data.root_id.cpu().numpy().reshape(-1))
        for condition in args.conditions:
            inputs = intervened_inputs(data, condition, args.seed + batch_no)
            graph_embedding = model(**inputs)
            pred = model.cls_head.predict_top_down(graph_embedding)[:, -1]
            predictions[condition].append(pred.cpu().numpy())

    labels_array = np.concatenate(labels)
    roots_array = np.concatenate(root_ids)
    pred_arrays = {name: np.concatenate(parts) for name, parts in predictions.items()}

    # These interventions touch only information a mean model cannot read.
    # Exact prediction equality is stronger and less brittle than comparing
    # rounded metrics or floating-point pooled embeddings.
    invariant_conditions = {
        "permute_x", "shuffle_edges", "drop_edges",
        "shuffle_edges_zero_pos_enc", "drop_edges_zero_pos_enc",
    }
    if config.architecture == "mean" and "none" in pred_arrays:
        for condition in invariant_conditions.intersection(pred_arrays):
            if not np.array_equal(pred_arrays["none"], pred_arrays[condition]):
                changed = int(np.count_nonzero(pred_arrays["none"] != pred_arrays[condition]))
                raise AssertionError(
                    f"mean invariance check failed for {path.parent.name}/{condition}: "
                    f"{changed} predictions changed"
                )

    rows = []
    baseline = pred_arrays.get("none")
    for condition, pred in pred_arrays.items():
        window, cell = metrics(labels_array, pred, roots_array, classes)
        rows.append({
            "checkpoint": path.parent.name,
            "architecture": config.architecture,
            "condition": condition,
            "changed_window_predictions": (
                int(np.count_nonzero(pred != baseline)) if baseline is not None else None
            ),
            "window_accuracy": window["accuracy"],
            "window_balanced_accuracy": window["balanced_accuracy"],
            "window_macro_f1": window["macro_f1"],
            "cell_accuracy": cell["accuracy"],
            "cell_balanced_accuracy": cell["balanced_accuracy"],
            "cell_macro_f1": cell["macro_f1"],
            "window_per_class_recall": window["per_class_recall"],
            "cell_per_class_recall": cell["per_class_recall"],
        })
    return rows


def main(args) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    paths = [Path(p) if Path(p).is_absolute() else repo_root / p for p in args.checkpoints]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint(s):\n  " + "\n  ".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)
    # The default grid is intentionally one radius. Mixing radii would require
    # distinct datasets/membership caches, and checkpoint config does not store
    # window_nm because it is not a model-construction property.
    dataset = WindowedGraphDatasetLCPN(
        manifest, "test", pos_dim=args.pos_dim, window_nm=args.window_nm
    )
    classes = dataset.classes
    print(f"device={device} test={len(dataset)} windows conditions={args.conditions}")

    rows = []
    for path in paths:
        rows.extend(evaluate_checkpoint(path, dataset, hierarchy, classes, args, device))

    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    scalar_fields = [key for key, value in rows[0].items() if not isinstance(value, dict)]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in rows)

    for row in rows:
        print(
            f"{row['checkpoint']:50s} {row['condition']:29s} "
            f"window_f1={row['window_macro_f1']:.4f} cell_f1={row['cell_macro_f1']:.4f} "
            f"changed={row['changed_window_predictions']}"
        )
    print(f"wrote {output} and {csv_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--window-nm", type=float, default=10_000.0)
    parser.add_argument("--pos-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=31)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/shuffling_ablations.json")
    raise SystemExit(main(parser.parse_args()))
