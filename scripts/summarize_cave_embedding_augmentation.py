"""Summarize clean-test metrics for the four embedding augmentation models."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("results/presynaptic/cave_embedding_augmentation_conf0.7")
OUTPUT = Path("analysis/presynaptic/cave_embedding_augmentation_conf0.7/model_comparison.csv")
POLICIES = ("clean", "gray", "flip", "structured_low")


def main() -> None:
    rows = []
    for policy in POLICIES:
        matches = list(ROOT.glob(f"*_embaug_{policy}_sampled_fold0/best_metrics.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one completed {policy} result, found {matches}")
        data = json.loads(matches[0].read_text())
        cell = data["test_metrics"]
        window = data["window_test_metrics"]
        rows.append({
            "training_condition": policy,
            "run": data["run"],
            "best_epoch": data["epoch"],
            "cell_accuracy": cell["accuracy"],
            "cell_balanced_accuracy": cell["balanced_accuracy"],
            "cell_macro_f1": cell["macro_f1"],
            "window_accuracy": window["accuracy"],
            "window_balanced_accuracy": window["balanced_accuracy"],
            "window_macro_f1": window["macro_f1"],
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
