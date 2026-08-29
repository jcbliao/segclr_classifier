"""Validate same-root coordinate alignment between graph embeddings and new skeletons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph_cache"
MANIFEST = ROOT / "data" / "manifest.json"
SKELETON = Path("/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "test", "all"])
    ap.add_argument("--output", default=str(ROOT / "analysis" / "new_skeleton_alignment.csv"))
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    ids = [int(r) for r, v in manifest["cells"].items()
           if args.split == "all" or v["split"] == args.split]
    rows, missing = [], []
    for root_id in tqdm(sorted(ids), desc="same-root alignment", unit="cell"):
        gp, sp = GRAPH / f"{root_id}.pt", SKELETON / f"{root_id}.npz"
        if not gp.exists() or not sp.exists():
            missing.append({"root_id": root_id, "graph": gp.exists(), "skeleton": sp.exists()})
            continue
        data = torch.load(gp, map_location="cpu", weights_only=False)
        embedding_xyz = data.pos.numpy().astype(np.float64)
        with np.load(sp) as z:
            stored_root = int(z["root_id"][0])
            skeleton_xyz = z["vertices"].astype(np.float64)
            radius = z["radius"]
        if stored_root != root_id:
            raise RuntimeError(f"{sp}: internal root_id={stored_root}, filename={root_id}")
        distance, nearest = cKDTree(skeleton_xyz).query(embedding_xyz, k=1)
        finite_radius = np.isfinite(radius[nearest])
        rows.append({
            "root_id": root_id, "cell_type": manifest["cells"][str(root_id)]["cell_type"],
            "n_embeddings": len(embedding_xyz), "n_skeleton_nodes": len(skeleton_xyz),
            "nearest_median_nm": float(np.median(distance)),
            "nearest_p95_nm": float(np.percentile(distance, 95)),
            "nearest_p99_nm": float(np.percentile(distance, 99)),
            "nearest_max_nm": float(distance.max()),
            "fraction_within_100nm": float(np.mean(distance <= 100)),
            "fraction_within_500nm": float(np.mean(distance <= 500)),
            "fraction_within_1000nm": float(np.mean(distance <= 1000)),
            "fraction_finite_radius": float(finite_radius.mean()),
            "embedding_center_x": float(np.median(embedding_xyz[:, 0])),
            "skeleton_center_x": float(np.median(skeleton_xyz[:, 0])),
            "embedding_center_y": float(np.median(embedding_xyz[:, 1])),
            "skeleton_center_y": float(np.median(skeleton_xyz[:, 1])),
            "embedding_center_z": float(np.median(embedding_xyz[:, 2])),
            "skeleton_center_z": float(np.median(skeleton_xyz[:, 2])),
        })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader(); writer.writerows(rows)
    summary = {
        "split": args.split, "manifest_cells": len(ids), "aligned_cells": len(rows),
        "missing": missing,
        "embedding_weighted_fraction_within_500nm": float(np.average(
            [r["fraction_within_500nm"] for r in rows], weights=[r["n_embeddings"] for r in rows]
        )) if rows else None,
        "cell_median_of_nearest_medians_nm": float(np.median(
            [r["nearest_median_nm"] for r in rows])) if rows else None,
        "cell_p95_of_nearest_p95_nm": float(np.percentile(
            [r["nearest_p95_nm"] for r in rows], 95)) if rows else None,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
