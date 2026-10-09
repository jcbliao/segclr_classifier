"""Plan balanced shards for confidence-0.7 cells absent from augmentation storage."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path

from scripts.generate_cave_embedding_augmentations import CAVE_CACHE, DATABASE, OUTPUT, target_nodes


def node_count(root_id: int) -> tuple[int, int]:
    source = DATABASE / "cells" / f"{root_id}.npz"
    if source.exists():
        return root_id, len(target_nodes(source)[0])
    with open(CAVE_CACHE / f"{root_id}.pkl", "rb") as stream:
        return root_id, len(pickle.load(stream).coords)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=18)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    roots: set[int] = set()
    for manifest in sorted(args.fold_root.glob("fold*/manifest.json")):
        payload = json.loads(manifest.read_text())
        roots.update(map(int, payload["cells"]))
    reference = OUTPUT / "gray" / "sample0" / "cells"
    missing = sorted(root_id for root_id in roots if not (reference / f"{root_id}.npz").exists())
    with mp.Pool(16) as pool:
        counts = dict(pool.map(node_count, missing))
    shards = [[] for _ in range(args.shards)]
    loads = [0] * args.shards
    for root_id in sorted(missing, key=counts.__getitem__, reverse=True):
        index = min(range(args.shards), key=loads.__getitem__)
        shards[index].append(root_id)
        loads[index] += counts[root_id]
    result = {"shards": shards, "node_loads": loads, "total_nodes": sum(loads),
              "total_cells": len(missing), "variants_per_cell": 31}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cells": len(missing), "nodes": sum(loads), "loads": loads}, indent=2))


if __name__ == "__main__":
    main()
