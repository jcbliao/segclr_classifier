"""Create five cell-level folds balanced by Casey fine type at every cutoff.

The confidence cohorts are nested. Assign roots once, processing high-confidence
bins first, so the same root has one held-out fold in every cohort containing it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "casey_coarse_confidence_with_tc/scale16/k17"
)
OUTPUT = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "casey_coarse_confidence_with_tc_folds/scale16/k17"
)
COMPARISON = Path("data/v1718_extended_axon_neurons_casey_comparison.csv")
N_FOLDS = 5
SEED = 0
CONFIDENCES = (0, 0.3, 0.5, 0.7, 0.9)


def fine_label(row: dict, casey: dict[str, dict]) -> str:
    if row["original_cell_type"] == "thalamocortical":
        return "thalamocortical"
    label = casey[row["root_id"]]["casey_fine"]
    if not label:
        raise ValueError(f"Missing Casey fine type for {row['root_id']}")
    return label


def confidence_bin(row: dict) -> int:
    if row["original_cell_type"] == "thalamocortical":
        return len(CONFIDENCES) - 1
    score = float(row["casey_coarse_confidence"])
    return max(i for i, threshold in enumerate(CONFIDENCES) if score >= threshold)


def make_assignments(rows: list[dict], casey: dict[str, dict]) -> dict[str, int]:
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in rows:
        strata[(fine_label(row, casey), confidence_bin(row))].append(row["root_id"])
    assignments: dict[str, int] = {}
    total_counts = [0] * N_FOLDS
    for label in sorted({label for label, _ in strata}):
        class_counts = [0] * N_FOLDS
        digest = hashlib.sha256(label.encode()).digest()
        rng = random.Random(SEED + int.from_bytes(digest[:8], "big"))
        tie_order = list(range(N_FOLDS))
        rng.shuffle(tie_order)
        rank = {fold: i for i, fold in enumerate(tie_order)}
        for bin_index in reversed(range(len(CONFIDENCES))):
            roots = sorted(strata.get((label, bin_index), ()))
            rng.shuffle(roots)
            for root_id in roots:
                fold = min(range(N_FOLDS), key=lambda i: (class_counts[i], total_counts[i], rank[i]))
                assignments[root_id] = fold
                class_counts[fold] += 1
                total_counts[fold] += 1
    if len(assignments) != len(rows):
        raise ValueError("Not every cell received one fold")
    return assignments


def main() -> None:
    casey_rows = list(csv.DictReader(COMPARISON.open(newline="")))
    casey = {row["root_id"]: row for row in casey_rows}
    base_rows = list(csv.DictReader((SOURCE / "conf0/cohort.csv").open(newline="")))
    assignments = make_assignments(base_rows, casey)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "fold_assignments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("root_id", "casey_fine_type", "confidence_bin", "held_out_fold"))
        writer.writeheader()
        writer.writerows({
            "root_id": row["root_id"], "casey_fine_type": fine_label(row, casey),
            "confidence_bin": confidence_bin(row), "held_out_fold": assignments[row["root_id"]],
        } for row in base_rows)

    for confidence in CONFIDENCES:
        source_dir = SOURCE / f"conf{confidence:g}"
        source_manifest = json.loads((source_dir / "manifest.json").read_text())
        source_rows = list(csv.DictReader((source_dir / "cohort.csv").open(newline="")))
        for fold in range(N_FOLDS):
            destination = OUTPUT / f"conf{confidence:g}" / f"fold{fold}"
            destination.mkdir(parents=True, exist_ok=True)
            cells = {
                root_id: {**info, "split": "test" if assignments[root_id] == fold else "train"}
                for root_id, info in source_manifest["cells"].items()
            }
            manifest = {
                **source_manifest, "cells": cells, "fold_index": fold,
                "n_folds": N_FOLDS, "fold_assignment_seed": SEED,
                "fold_stratification": "Casey fine type, confidence-bin-balanced; thalamocortical separate",
                "fold_assignments": str((OUTPUT / "fold_assignments.csv").resolve()),
            }
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (destination / "metadata.json").write_text((source_dir / "metadata.json").read_text())
            link = destination / "cells"
            target = (source_dir / "cells").resolve()
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
            elif link.resolve() != target:
                if not link.is_symlink():
                    raise ValueError(f"Wrong cells directory: {link}")
                link.unlink()
                link.symlink_to(target, target_is_directory=True)
            with (destination / "cohort.csv").open("w", newline="") as handle:
                fieldnames = list(source_rows[0]) + ["casey_fine_type", "held_out_fold"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in source_rows:
                    root_id = row["root_id"]
                    writer.writerow({
                        **row, "split": cells[root_id]["split"],
                        "casey_fine_type": fine_label(row, casey),
                        "held_out_fold": assignments[root_id],
                    })
        print(f"conf={confidence:g}: {len(source_rows)} cells, five folds", flush=True)

    # Every fine type must differ by at most one held-out cell between folds
    # within each nested threshold cohort.
    for confidence in CONFIDENCES:
        rows = list(csv.DictReader((OUTPUT / f"conf{confidence:g}/fold0/cohort.csv").open(newline="")))
        per_type: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            per_type[row["casey_fine_type"]][int(row["held_out_fold"])] += 1
        for label, counts in per_type.items():
            values = [counts[i] for i in range(N_FOLDS)]
            if max(values) - min(values) > 1:
                raise ValueError(f"Unbalanced {label} at confidence {confidence}: {values}")
    print("Verified per-type held-out counts differ by at most one in every cohort")


if __name__ == "__main__":
    main()
