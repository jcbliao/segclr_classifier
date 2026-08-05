"""Generates an HDF5 matching segCLR_cell_classification's expected schema
(CellTypingDataset.__init__ in their src/data/datasets.py: seg_ids, cell_types,
coarse_cell_types, embeddings, optional nodes) from OUR real data
(data/manifest.json + data/graph_cache/*.pt + data/skeleton_cache/*.pkl), so
their actual training infrastructure (DeepResNet, their BaseTrainer, their
cell_level_accuracy majority-vote metric) can be used as the baseline
directly -- per explicit user direction, rather than reimplementing it
ourselves (see results/deprecated_own_baseline_reimplementation/README.md).

geodesic_mean at window_nm=25000 -- one row per covered skeleton node per
cell, NOT collapsed (reuses baseline.mean_pool_classifier.node_level_features
directly, since it already does exactly this). Their own CellTypingDataset
expects/handles per-point classification + majority vote internally, so no
extra aggregation step belongs here.

Also writes a non-standard extra dataset "splits" (a "train"/"val"/"test"
string per row) -- not part of their original schema, but harmless (their
loader only reads the keys it names), and it's what lets
scripts/lab_baseline_dataset_override.py replace their internal random
_split_by_cells with OUR exact split, which is required for the baseline and
the GNN to be evaluated on identical cells.

Run via sbatch (mit_normal, CPU only -- geodesic_mean is numba on CPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from baseline.mean_pool_classifier import WINDOW_NM, node_level_features  # noqa: E402
from data.dataset import load_manifest  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "lab_baseline_aggregated_25um.h5"


def main() -> int:
    manifest = load_manifest()
    cells = manifest["cells"]
    root_ids = sorted(int(r) for r in cells)
    print(f"{len(root_ids)} cells total")

    all_X, all_seg_ids, all_labels, all_nodes, all_splits = [], [], [], [], []
    n_failed = 0
    for root_id in tqdm(root_ids, desc="cells", unit="cell"):
        info = cells[str(root_id)]
        try:
            feats = node_level_features(root_id, WINDOW_NM)
        except Exception as e:  # noqa: BLE001 -- report and continue
            print(f"  {root_id} failed: {type(e).__name__}: {e}")
            n_failed += 1
            continue
        n = len(feats)
        all_X.append(feats.astype(np.float32))
        all_seg_ids.append(np.full(n, root_id, dtype=np.int64))
        all_labels.append(np.full(n, info["cell_type"], dtype=object))
        all_splits.append(np.full(n, info["split"], dtype=object))
        # orig_node_ids paired 1:1 with node_level_features' output rows (both
        # come from the same geodesic_mean call over data.orig_node_ids).
        import torch

        data = torch.load(
            Path(__file__).resolve().parent.parent / "data" / "graph_cache" / f"{root_id}.pt",
            weights_only=False,
        )
        all_nodes.append(data.orig_node_ids.numpy().astype(np.int64))

    print(f"built {len(root_ids) - n_failed} cells, {n_failed} failed")

    X = np.concatenate(all_X)
    seg_ids = np.concatenate(all_seg_ids)
    labels = np.concatenate(all_labels)
    nodes = np.concatenate(all_nodes)
    splits = np.concatenate(all_splits)
    print(f"total rows: {len(X)}  dim: {X.shape[1]}")

    # dtype=object (Python str elements) is what h5py's vlen-string special
    # dtype actually wants -- .astype(str) converts to a fixed-width numpy
    # unicode dtype ('<U9' etc.) instead, which h5py cannot write ("No
    # conversion path"). labels/splits are already object-dtype from
    # np.full(..., dtype=object) above; leave them as-is.
    str_dt = h5py.special_dtype(vlen=str)
    with h5py.File(OUT_PATH, "w") as f:
        f.create_dataset("embeddings", data=X, dtype=np.float32)
        f.create_dataset("seg_ids", data=seg_ids, dtype=np.int64)
        f.create_dataset("cell_types", data=labels, dtype=str_dt)
        # No real coarse taxonomy for these Allen-style labels (established
        # earlier: no dash-hierarchy to collapse) -- duplicate fine labels so
        # use_coarse=false is what actually selects granularity, and
        # use_coarse=true would just be a no-op rather than silently wrong.
        f.create_dataset("coarse_cell_types", data=labels, dtype=str_dt)
        f.create_dataset("nodes", data=nodes, dtype=np.int64)
        f.create_dataset("splits", data=splits, dtype=str_dt)
        f.attrs["window_nm"] = WINDOW_NM
        f.attrs["source_manifest"] = str(Path(__file__).resolve().parent.parent / "data" / "manifest.json")

    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
