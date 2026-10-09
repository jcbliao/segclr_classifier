"""Build k17 TEASAR manifests for Casey confidence cuts plus thalamocortical cells.

The comparison CSV already joins v1718 roots to Casey by root or stable nucleus
identity. Preserve the source database's cell splits and link its window files.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_DATABASE = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "casey_k17_full_tc_source/scale16/k17"
)
DEFAULT_COMPARISON = Path("data/v1718_extended_axon_neurons_casey_comparison.csv")
DEFAULT_OUTPUT = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "casey_coarse_confidence_with_tc/scale16/k17"
)
CONFIDENCES = (0, 0.3, 0.5, 0.7, 0.9)
FAMILIES = {"BasketFam", "MartFam", "BipFam", "NglFam"}
PYRAMIDAL_TYPES = {
    "L5ET", "L2IT", "L3IT", "L4IT", "L6IT", "L5IT", "L5NP", "L6CT"
}
HIERARCHY_TREE = {
    "neuron": {
        "excitatory": {
            "pyramidal": {
                "ET": ["L5ET"],
                "IT": ["L2IT", "L3IT", "L4IT", "L6IT", "L5IT"],
                "NP": ["L5NP"],
                "corticothalamic": ["L6CT"],
            },
            "thalamocortical": ["thalamocortical"],
        },
        "inhibitory": {family: None for family in sorted(FAMILIES)},
    },
    "non_neuron": {
        "non_neuron": {"glia": ["astrocyte", "oligo", "microglia", "OPC"]},
    },
}


def build(database: Path, comparison: Path, output: Path) -> None:
    metadata = json.loads((database / "metadata.json").read_text())
    if metadata.get("format") != "dense-presynaptic-v1" or metadata.get("k_observed") != 17:
        raise ValueError("Expected finalized dense TEASAR k17 database")
    if metadata.get("missing_root_ids"):
        raise ValueError("Source database is incomplete")
    source = json.loads((database / "manifest.json").read_text())
    rows = list(csv.DictReader(comparison.open(newline="")))
    by_root = {row["root_id"]: row for row in rows}
    if len(by_root) != len(rows):
        raise ValueError("Duplicate roots in Casey comparison")

    for confidence in CONFIDENCES:
        cells = {}
        audit = []
        excluded = Counter()
        for root_id, source_info in source["cells"].items():
            row = by_root.get(root_id)
            reason = None
            label = None
            # Casey has no coarse confidence for thalamocortical cells. Include
            # the source manifest's thalamocortical label at every threshold.
            thalamocortical = source_info["cell_type"] == "thalamocortical"
            if thalamocortical:
                label = "thalamocortical"
            elif row is None or not row["casey_coarse"] or not row["casey_cell_id"]:
                reason = "no_casey_identity_or_coarse_label"
            elif row["casey_coarse"] in {"ChC", "L6b"} or source_info["cell_type"] in {"ChC", "L6b"}:
                reason = "excluded_ChC_or_L6b"
            elif row["casey_class"] == "excitatory" and source_info["cell_type"] in PYRAMIDAL_TYPES:
                label = source_info["cell_type"]
            elif row["casey_class"] == "inhibitory" and row["casey_coarse"] in FAMILIES:
                label = row["casey_coarse"]
            else:
                reason = "outside_requested_cohort"
            if reason is None and not thalamocortical:
                try:
                    score = float(row["casey_coarse_confidence"])
                except (TypeError, ValueError):
                    reason = "missing_confidence"
                else:
                    if not 0 <= score <= 1:
                        raise ValueError(f"Invalid Casey confidence for {root_id}: {score}")
                    if score < confidence:
                        reason = "below_confidence"
            if reason:
                excluded[reason] += 1
                continue
            cells[root_id] = {**source_info, "cell_type": label}
            audit.append({
                "root_id": root_id, "split": source_info["split"],
                "original_cell_type": source_info["cell_type"],
                "training_label": label, "casey_class": row["casey_class"] if row else "",
                "casey_coarse": row["casey_coarse"] if row else "",
                "casey_coarse_confidence": row["casey_coarse_confidence"] if row else "",
                "casey_cell_id": row["casey_cell_id"] if row else "",
                "match_source": row["match_source"] if row else "",
                "label_source": "source_manifest" if thalamocortical else "Casey_coarse",
            })
        destination = output / f"conf{confidence:g}"
        destination.mkdir(parents=True, exist_ok=True)
        link = destination / "cells"
        if not link.exists():
            link.symlink_to((database / "cells").resolve(), target_is_directory=True)
        elif link.resolve() != (database / "cells").resolve():
            if not link.is_symlink():
                raise ValueError(f"Wrong cells directory: {link}")
            link.unlink()
            link.symlink_to((database / "cells").resolve(), target_is_directory=True)
        manifest = {
            **source, "cells": cells, "hierarchy_tree": HIERARCHY_TREE,
            "hierarchy_levels_dropped": 2,
            "drop_labels": ["astrocyte", "oligo", "microglia", "OPC"],
            "casey_comparison": str(comparison.resolve()),
            "casey_coarse_confidence_min": confidence,
            "casey_join": "root_id or stable nucleus identity (cell_id)",
            "thalamocortical_policy": "source-manifest label; included at every confidence threshold",
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (destination / "cohort.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
            writer.writeheader()
            writer.writerows(audit)
        summary = {
            "confidence": confidence, "n_cells": len(cells),
            "splits": dict(Counter(row["split"] for row in audit)),
            "labels": dict(Counter(row["training_label"] for row in audit)),
            "excluded": dict(excluded),
        }
        (destination / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"conf={confidence:g}: {len(cells)} cells, {summary['splits']}, excluded={dict(excluded)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.database, args.comparison, args.output)
