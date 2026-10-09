"""Pack clean plus all complete FP16 augmentation draws per cell/policy."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DEFAULT_DATABASE = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10")
DEFAULT_SOURCE = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "cave_embedding_augmentations/conf0.7/fold0/cutoff5000"
)
DEFAULT_OUTPUT = Path(
    "/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
    "cave_embedding_training_choices/conf0.7/fold0/cutoff5000_fp16"
)
POLICIES = ("gray", "flip", "structured_low")


def atomic_save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def pack_cell(root_id: int, database: Path, source: Path, output: Path, samples_by_policy: dict[str, list[int]]) -> tuple[int, int]:
    with np.load(database / "cells" / f"{root_id}.npz", allow_pickle=False) as z:
        clean_ids = z["cave_observed_node_ids"].copy()
        clean = z["cave_observed_x"].copy()
    written = 0
    for policy in POLICIES:
        paths = [source / policy / f"sample{sample}" / "cells" / f"{root_id}.npz"
                 for sample in samples_by_policy[policy]]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        augmented = []
        node_ids = None
        for sample, path in zip(samples_by_policy[policy], paths):
            with np.load(path, allow_pickle=False) as z:
                ids = z["node_ids"].copy()
                values = z["embeddings"].copy()
                if str(z["augmentation_id"]) != policy or int(z["augmentation_sample"]) != sample:
                    raise ValueError(f"wrong policy/sample metadata in {path}")
                if str(z.get("inference_precision", "")) != "fp16":
                    raise ValueError(f"non-FP16 augmentation file: {path}")
            if node_ids is None:
                node_ids = ids
            elif not np.array_equal(node_ids, ids):
                raise ValueError(f"node IDs differ across {policy} samples for {root_id}")
            if values.shape != (len(ids), 64) or not np.isfinite(values).all():
                raise ValueError(f"invalid embeddings in {path}")
            augmented.append(values)
        locations = np.searchsorted(clean_ids, node_ids)
        if (len(locations) and (locations.max() >= len(clean_ids) or
                               not np.array_equal(clean_ids[locations], node_ids))):
            raise ValueError(f"augmentation nodes are absent from clean embeddings for {root_id}")
        choices = np.stack([clean[locations], *augmented], axis=1).astype(np.float32, copy=False)
        destination = output / policy / "cells" / f"{root_id}.npz"
        atomic_save(destination, root_id=np.asarray(root_id, np.uint64), node_ids=node_ids,
                    embedding_choices=choices, augmentation_id=np.asarray(policy),
                    choice_labels=np.asarray(["clean", *[f"sample{i}" for i in samples_by_policy[policy]]]),
                    augmentation_precision=np.asarray("fp16"))
        written += 1
    return root_id, written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, nargs="+")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    roots = sorted({int(root) for path in args.manifest
                    for root, info in json.loads(path.read_text())["cells"].items()
                    if info["split"] == "train"})
    samples_by_policy = {}
    for policy in POLICIES:
        samples = sorted(int(p.name[6:]) for p in (args.source / policy).glob("sample*")
                         if p.is_dir() and all((p / "cells" / f"{root}.npz").is_file() for root in roots))
        if not samples:
            raise ValueError(f"No complete samples for {policy}")
        samples_by_policy[policy] = samples
    print(json.dumps(samples_by_policy), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(pack_cell, root, args.database, args.source, args.output, samples_by_policy)
                   for root in roots]
        for index, future in enumerate(futures, 1):
            root, written = future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] root={root} policies={written}", flush=True)
    metadata = {
        "format": "cave-embedding-choices-v1", "policies": list(POLICIES),
        "choices_per_node": {p: 1 + len(v) for p, v in samples_by_policy.items()},
        "choice_order": {p: ["clean", *v] for p, v in samples_by_policy.items()},
        "training_cells": len(roots), "augmentation_precision": "fp16",
        "source": str(args.source), "database": str(args.database),
        "manifests": [str(p) for p in args.manifest],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
