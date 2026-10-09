"""Summarize completed Casey coarse-label confidence sweep results."""
from __future__ import annotations

import csv
import json
import os
from statistics import mean, stdev
from pathlib import Path

ROOT = Path("results/presynaptic/casey_confidence_cave/k10")
NAMES = {
    "mean": "gnn_lcpn_scratch_mean_resnet4x128_n10_mixed16_sampled_fold{fold}",
    "pointwise_mlp": "gnn_lcpn_scratch_pointwise_mlp_L2_resnet4x128_n10_mixed16_sampled_fold{fold}",
    "graph_transformer": "gnn_lcpn_scratch_gt_L4_H4_resnet4x128_n10_mixed16_sampled_fold{fold}",
}


def main() -> None:
    folds = tuple(int(value) for value in os.environ.get("CASEY_FOLDS", "0").split(","))
    if not folds or any(fold not in range(5) for fold in folds) or len(set(folds)) != len(folds):
        raise ValueError(f"Invalid CASEY_FOLDS: {folds}")
    rows = []
    missing = []
    for confidence in (0, 0.3, 0.5, 0.7, 0.9):
        for fold in folds:
            directory = ROOT / f"conf{confidence:g}" / f"fold{fold}"
            for architecture, template in NAMES.items():
                path = directory / f"{template.format(fold=fold)}.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                result = json.loads(path.read_text())
                cell = result["test_metrics"]
                window = result["window_test_metrics"]
                rows.append({
                    "confidence": confidence, "architecture": architecture,
                    "fold": fold,
                    "test_cell_accuracy": cell["accuracy"],
                    "test_cell_balanced_accuracy": cell["balanced_accuracy"],
                    "test_cell_macro_f1": cell["macro_f1"],
                    "test_window_accuracy": window["accuracy"],
                    "test_window_balanced_accuracy": window["balanced_accuracy"],
                    "test_window_macro_f1": window["macro_f1"],
                })
    if missing:
        raise SystemExit("Missing results:\n" + "\n".join(missing))
    destination = ROOT / "summary_by_fold.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = [key for key in rows[0] if key.startswith("test_")]
    aggregate = []
    for confidence in (0, 0.3, 0.5, 0.7, 0.9):
        for architecture in NAMES:
            group = [row for row in rows if row["confidence"] == confidence
                     and row["architecture"] == architecture]
            record = {"confidence": confidence, "architecture": architecture, "n_folds": len(group)}
            for metric in metrics:
                values = [row[metric] for row in group]
                record[f"{metric}_mean"] = mean(values)
                record[f"{metric}_sd"] = stdev(values) if len(values) > 1 else None
            aggregate.append(record)
    aggregate_path = ROOT / "summary.csv"
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(f"Wrote {len(rows)} fold rows to {destination} and {len(aggregate)} summaries to {aggregate_path}")


if __name__ == "__main__":
    main()
