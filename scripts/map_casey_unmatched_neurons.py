"""Map v1718 extended-axon neuron roots absent from Casey via nucleus ID.

Reads the SegCLR store and Casey Parquet file, then queries v1718 CAVE
proofreading and ``nucleus_detection_v0`` tables. The latter's ``id`` is the stable nucleus
ID corresponding to Casey's ``cell_id``. Writes one CSV row per candidate
nucleus, plus a row for any root with no nucleus record. No store data changes.

Run with the project's segclr_db virtual environment::

    segclr_db/.venv/bin/python scripts/map_casey_unmatched_neurons.py

Authentication uses CAVE_TOKEN or the standard CloudVolume token file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import caveclient
import lance
import pyarrow.parquet as pq


DEFAULT_STORE = Path("/orcd/compute/sdorkenw/001/segclr-db/microns")
DEFAULT_CASEY = Path(
    "/home/jcbliao/rotation/segclr/misc/casey_class_hierarchy/"
    "microns_public_v1822_ct_csm_v2_sep15.parquet"
)
DEFAULT_OUTPUT = Path("data/v1718_extended_axon_neurons_casey_root_map.csv")
DEFAULT_UNRESOLVED_OUTPUT = Path("data/v1718_extended_axon_neurons_unresolved_casey.csv")
DEFAULT_COHORT_OUTPUT = Path("data/v1718_extended_axon_neurons_labels.csv")
DEFAULT_COMPARISON_OUTPUT = Path("data/v1718_extended_axon_neurons_casey_comparison.csv")
EXTENDED_AXON_STRATEGIES = {"axon_partially_extended", "axon_fully_extended"}
FIELDS = (
    "v1718_root_id", "strategy_axon", "nucleus_id", "casey_root_id", "casey_cell_id",
    "class", "ct_coarse", "ct_coarse_confidence", "ct_fine",
    "ct_fine_confidence", "status",
)


def token() -> str:
    value = os.environ.get("CAVE_TOKEN")
    if value:
        return value
    path = Path.home() / ".cloudvolume/secrets/global.daf-apis.com-cave-secret.json"
    if path.exists():
        return json.loads(path.read_text())["token"]
    raise RuntimeError("Set CAVE_TOKEN or install the standard CloudVolume CAVE token file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--casey", type=Path, default=DEFAULT_CASEY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unresolved-output", type=Path, default=DEFAULT_UNRESOLVED_OUTPUT)
    parser.add_argument("--cohort-output", type=Path, default=DEFAULT_COHORT_OUTPUT)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    metadata = json.loads((args.store / "meta.json").read_text())
    if metadata.get("mat_version") != 1718:
        raise ValueError(f"Expected a v1718 store, got {metadata.get('mat_version')}")

    cells = lance.dataset(str(args.store / "dims/cells.lance")).to_table(
        columns=["root_id", "is_neuron"]
    )
    neuron_roots = {
        int(root) for root, is_neuron in zip(
            cells["root_id"].to_pylist(), cells["is_neuron"].to_pylist(), strict=True
        ) if is_neuron is True
    }
    casey_rows = pq.read_table(args.casey).to_pylist()
    casey_roots = {int(row["root_id"]) for row in casey_rows}
    by_cell_id: dict[int, list[dict]] = defaultdict(list)
    for row in casey_rows:
        by_cell_id[int(row["cell_id"])].append(row)

    client = caveclient.CAVEclient(
        "minnie65_public", auth_token=token(), version=1718
    )
    strategies_by_root: dict[int, set[str]] = defaultdict(set)
    neuron_list = sorted(neuron_roots)
    for start in range(0, len(neuron_list), args.batch_size):
        batch = neuron_list[start:start + args.batch_size]
        frame = client.materialize.query_table(
            "proofreading_status_and_strategy", materialization_version=1718,
            filter_in_dict={"pt_root_id": batch},
        )
        for root_id, status, strategy in frame[
            ["pt_root_id", "status_axon", "strategy_axon"]
        ].itertuples(index=False, name=None):
            if bool(status) and strategy:
                strategies_by_root[int(root_id)].add(str(strategy))

    eligible = {}
    for root_id, strategies in strategies_by_root.items():
        selected = strategies & EXTENDED_AXON_STRATEGIES
        if root_id in neuron_roots and selected:
            if len(selected) != 1:
                raise ValueError(f"Conflicting extended-axon strategies for {root_id}: {selected}")
            eligible[root_id] = selected.pop()
    unmatched = sorted(set(eligible) - casey_roots)
    nuclei_by_root: dict[int, set[int]] = defaultdict(set)
    for start in range(0, len(unmatched), args.batch_size):
        batch = unmatched[start:start + args.batch_size]
        frame = client.materialize.query_table(
            "nucleus_detection_v0", materialization_version=1718,
            filter_in_dict={"pt_root_id": batch},
        )
        for root_id, nucleus_id in frame[["pt_root_id", "id"]].itertuples(index=False, name=None):
            root_id = int(root_id)
            if root_id not in batch:
                raise ValueError(f"CAVE returned unexpected root ID {root_id}")
            nuclei_by_root[root_id].add(int(nucleus_id))
        print(f"Queried {min(start + len(batch), len(unmatched))}/{len(unmatched)} roots", flush=True)

    output_rows = []
    for root_id in unmatched:
        nuclei = sorted(nuclei_by_root.get(root_id, ()))
        if not nuclei:
            output_rows.append({
                "v1718_root_id": root_id, "strategy_axon": eligible[root_id],
                "status": "no_v1718_nucleus",
            })
            continue
        for nucleus_id in nuclei:
            matches = by_cell_id.get(nucleus_id, ())
            if not matches:
                output_rows.append({
                    "v1718_root_id": root_id, "strategy_axon": eligible[root_id],
                    "nucleus_id": nucleus_id,
                    "status": "nucleus_absent_from_casey",
                })
            for match in matches:
                has_type = match["ct_coarse"] is not None and match["ct_fine"] is not None
                output_rows.append({
                    "v1718_root_id": root_id, "strategy_axon": eligible[root_id],
                    "nucleus_id": nucleus_id,
                    "casey_root_id": int(match["root_id"]),
                    "casey_cell_id": int(match["cell_id"]),
                    "class": match["class"],
                    "ct_coarse": match["ct_coarse"],
                    "ct_coarse_confidence": match["ct_coarse_confidence"],
                    "ct_fine": match["ct_fine"],
                    "ct_fine_confidence": match["ct_fine_confidence"],
                    "status": "matched_by_nucleus_id" if has_type else "casey_type_missing",
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    labels = lance.dataset(str(args.store / "dims/cell_labels.lance")).to_table(
        columns=["root_id", "label_set", "label"]
    ).to_pylist()
    classifications: dict[int, str] = {}
    for row in labels:
        if row["label_set"] != "cell_type":
            continue
        root_id = int(row["root_id"])
        label = str(row["label"])
        if root_id in classifications and classifications[root_id] != label:
            raise ValueError(f"Conflicting cell classifications for {root_id}")
        classifications[root_id] = label

    missing_labels = set(eligible) - set(classifications)
    if missing_labels:
        raise ValueError(f"Missing cell classifications for {len(missing_labels)} eligible roots")
    args.cohort_output.parent.mkdir(parents=True, exist_ok=True)
    with args.cohort_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "root_id", "axon_status", "cell_classification",
        ))
        writer.writeheader()
        writer.writerows({
            "root_id": root_id, "axon_status": eligible[root_id],
            "cell_classification": classifications[root_id],
        } for root_id in sorted(eligible))

    casey_by_root = {int(row["root_id"]): row for row in casey_rows}
    mapped_roots = {
        int(row["v1718_root_id"]): int(row["casey_root_id"])
        for row in output_rows if row.get("casey_root_id") is not None
    }
    comparison_fields = (
        "root_id", "axon_status", "cell_classification", "casey_root_id",
        "casey_cell_id", "casey_class", "casey_coarse", "casey_coarse_confidence",
        "casey_fine", "casey_fine_confidence", "casey_is_core", "match_source",
    )
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    with args.comparison_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        for root_id in sorted(eligible):
            casey_root_id = root_id if root_id in casey_by_root else mapped_roots.get(root_id)
            match = casey_by_root.get(casey_root_id) if casey_root_id is not None else None
            writer.writerow({
                "root_id": root_id,
                "axon_status": eligible[root_id],
                "cell_classification": classifications[root_id],
                "casey_root_id": casey_root_id if match else "",
                "casey_cell_id": match["cell_id"] if match else "",
                "casey_class": match["class"] if match else "",
                "casey_coarse": match["ct_coarse"] if match else "",
                "casey_coarse_confidence": match["ct_coarse_confidence"] if match else "",
                "casey_fine": match["ct_fine"] if match else "",
                "casey_fine_confidence": match["ct_fine_confidence"] if match else "",
                "casey_is_core": match["is_core"] if match else "",
                "match_source": ("root_id" if root_id in casey_by_root else
                                 "nucleus_id" if match else "unmatched"),
            })

    unresolved = []
    for row in output_rows:
        if row["status"] == "matched_by_nucleus_id":
            continue
        root_id = row["v1718_root_id"]
        if classifications.get(root_id) == "thalamocortical":
            continue
        unresolved.append({
            "root_id": root_id,
            "axon_status": row["strategy_axon"],
            "nucleus_id": row.get("nucleus_id", ""),
            "cell_classification": classifications.get(root_id, ""),
        })
    args.unresolved_output.parent.mkdir(parents=True, exist_ok=True)
    with args.unresolved_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "root_id", "axon_status", "nucleus_id", "cell_classification",
        ))
        writer.writeheader()
        writer.writerows(unresolved)

    mapped_roots = {row["v1718_root_id"] for row in output_rows
                    if row["status"] == "matched_by_nucleus_id"}
    print(f"v1718 neurons with partially or fully extended axons: {len(eligible)}")
    print(f"Already present by root ID: {len(set(eligible) & casey_roots)}")
    print(f"Missing by root ID: {len(unmatched)}")
    print(f"Recovered with Casey type through nucleus ID: {len(mapped_roots)}")
    print(f"Still without a Casey type: {len(unmatched) - len(mapped_roots)}")
    print(f"Wrote {len(output_rows)} rows to {args.output}")
    print(f"Wrote {len(unresolved)} unresolved rows to {args.unresolved_output}")
    print(f"Wrote {len(eligible)} cohort rows to {args.cohort_output}")
    print(f"Wrote {len(eligible)} joined rows to {args.comparison_output}")


if __name__ == "__main__":
    main()
