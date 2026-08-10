"""Diagnostic: inspect the lab's own pre-built HDF5
(/orcd/compute/sdorkenw/001/collina/data/all_cells_aggregated_1718.h5) --
schema, label vocabulary, cell coverage vs. our existing 2193-cell manifest.

Run via sbatch only -- see scripts/sbatch/explore_lab_h5.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

H5_PATH = "/orcd/compute/sdorkenw/001/collina/data/all_cells_aggregated_1718.h5"

manifest_path = Path(__file__).resolve().parent.parent / "data" / "manifest.json"
manifest = json.loads(manifest_path.read_text())
our_root_ids = set(int(rid) for rid in manifest["cells"].keys())
our_labels_by_root = {int(rid): v["cell_type"] for rid, v in manifest["cells"].items()}
print(f"Our manifest: {len(our_root_ids)} cells", flush=True)

with h5py.File(H5_PATH, "r") as f:
    print("\n=== top-level keys ===", flush=True)
    for k in f.keys():
        ds = f[k]
        print(f"  {k}: shape={ds.shape} dtype={ds.dtype}", flush=True)

    print("\n=== attrs ===", flush=True)
    for k, v in f.attrs.items():
        print(f"  {k}: {v}", flush=True)

    seg_ids = f["seg_ids"][:]
    print(f"\ntotal rows: {len(seg_ids)}", flush=True)
    unique_seg_ids = np.unique(seg_ids)
    print(f"distinct cells (seg_ids): {len(unique_seg_ids)}", flush=True)

    label_key = "cell_types" if "cell_types" in f else None
    if label_key:
        raw = f[label_key][:]
        labels = np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in raw])
        vals, counts = np.unique(labels, return_counts=True)
        print(f"\n=== cell_types distribution (row-level, n={len(labels)}) ===", flush=True)
        for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
            print(f"  {v}: {c}", flush=True)

    if "coarse_cell_types" in f:
        raw_c = f["coarse_cell_types"][:]
        coarse = np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in raw_c])
        vals, counts = np.unique(coarse, return_counts=True)
        print(f"\n=== coarse_cell_types distribution ===", flush=True)
        for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
            print(f"  {v}: {c}", flush=True)

    # Cell-level (not row-level) label -- one label per unique seg_id
    if label_key:
        cell_label = {}
        for sid, lbl in zip(seg_ids, labels):
            cell_label.setdefault(int(sid), lbl)
        print(f"\n=== cell-level cell_types distribution (n={len(cell_label)} cells) ===", flush=True)
        vals, counts = np.unique(list(cell_label.values()), return_counts=True)
        for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
            print(f"  {v}: {c}", flush=True)

    h5_root_ids = set(int(s) for s in unique_seg_ids)
    overlap = our_root_ids & h5_root_ids
    only_ours = our_root_ids - h5_root_ids
    only_h5 = h5_root_ids - our_root_ids
    print(f"\n=== overlap with our manifest ===", flush=True)
    print(f"in both: {len(overlap)}", flush=True)
    print(f"only in our manifest (not in h5): {len(only_ours)}", flush=True)
    print(f"only in h5 (not in our manifest): {len(only_h5)}", flush=True)

    if overlap:
        n_match = 0
        n_diff = 0
        diffs = []
        for rid in overlap:
            ours = our_labels_by_root.get(rid)
            h5v = cell_label.get(rid) if label_key else None
            if ours == h5v:
                n_match += 1
            else:
                n_diff += 1
                if len(diffs) < 15:
                    diffs.append((rid, ours, h5v))
        print(f"\nlabel agreement on overlap: match={n_match} diff={n_diff}", flush=True)
        for rid, ours, h5v in diffs:
            print(f"  {rid}: ours={ours!r} h5={h5v!r}", flush=True)

    if "nodes" in f:
        nodes = f["nodes"][:5]
        print(f"\nsample 'nodes' values (first 5 rows): {nodes}", flush=True)

print("\ndone.", flush=True)
