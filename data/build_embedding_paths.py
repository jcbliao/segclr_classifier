"""graph_cache -> perisomatic-restricted skeletons -> centred path databases.

One SLURM array task owns a strided slice of the cell list. Per cell: drop the
15 um perisomatic ball, then enumerate every centred path at each of the six
budgets. Reads no store -- nucleus positions come from the cache written by
scripts/dump_nucleus_positions.py -- so the array needs neither the v4 shim nor
a CAVE token.

    python -u data/build_embedding_paths.py --task-id 0 --num-tasks 64

Layout under --out:

    soma_restricted/<root_id>.npz          the cut skeleton and what it broke into
    paths/<config>/<root_id>.npz           one path database per cell per budget
    neighborhoods/<config>/<root_id>.npz   local subgraphs holding a fixed amount
                                             of skeleton (the cable-budget unit)

Resumable per (cell, config): an existing .npz is skipped, so a requeued task
loses at most the config it was mid-way through.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.cable_neighborhoods import neighborhoods  # noqa: E402
from data.embedding_paths import assert_forest, centered_paths, count_paths  # noqa: E402
from data.soma_restrict import DEFAULT_SOMA_RADIUS_NM, restrict  # noqa: E402

#: The seven path budgets. A "half" is per arm, so a k-node config yields paths
#: of at most 2k+1 nodes and an L-um config arms of at most L/2 um each -- the
#: config name is the *diameter*, matching how the window sizes are named.
#:
#: 80um exists so each node budget has a length budget of comparable extent to
#: compare against: measured node spacing is ~2 um, so 10/20/40 nodes span about
#: 20/40/80 um. Pairing 10node with 10um would compare objects that differ in
#: length by a factor of two before anything about the model is varied.
CONFIGS = {
    "10um":   {"half_nm": 5_000.0},
    "20um":   {"half_nm": 10_000.0},
    "40um":   {"half_nm": 20_000.0},
    "80um":   {"half_nm": 40_000.0},
    "10node": {"half_nodes": 5},
    "20node": {"half_nodes": 10},
    "40node": {"half_nodes": 20},
}

#: Neighbourhood budgets. A neighbourhood is a connected local subgraph grown
#: nearest-first until it holds a fixed amount of skeleton -- see
#: data/cable_neighborhoods.py. Names are deliberately NOT reused from CONFIGS:
#: under paths/ a "20um" is a 20 um *diameter* route (10 um per arm) and a
#: "20node" is 21 nodes (10 per arm), while here "cable20um" is 20 um of *total
#: cable* and "n20" is 20 nodes outright. Same directory tree, so the names have
#: to carry the difference.
NEIGHBORHOOD_CONFIGS = {
    "cable10um": {"cable_nm": 10_000.0},
    "cable20um": {"cable_nm": 20_000.0},
    "cable40um": {"cable_nm": 40_000.0},
    "cable80um": {"cable_nm": 80_000.0},
    "n10": {"n_nodes": 10},
    "n20": {"n_nodes": 20},
    "n40": {"n_nodes": 40},
}

PATHS_ROOT = Path("/orcd/scratch/orcd/013/jcbliao/embedding_paths")


def out_for(radius_nm: float) -> Path:
    """Output directory for a given perisomatic radius.

    The radius is in the path, not just inside the files, so two radii can never
    land in one directory -- an interrupted rebuild would otherwise leave a mix
    that only a per-file read could detect.
    """
    return PATHS_ROOT / f"r{radius_nm / 1000:g}um"


DEFAULT_OUT = out_for(DEFAULT_SOMA_RADIUS_NM)


def build_cell(rid, cache_dir, nucleus_xyz, out, radius_nm, configs, force=False):
    """Returns a per-config dict of stats, or None if the cell is unusable."""
    d = torch.load(cache_dir / f"{rid}.pt", weights_only=False)
    pos = d.pos.numpy()
    orig = d.orig_node_ids.numpy().astype(np.int64)

    r = restrict(pos, d.edge_index.numpy(), d.edge_attr.numpy().reshape(-1),
                 nucleus_xyz, radius_nm)
    keep = r["keep"]
    n_kept = r["n_nodes_after"]

    # Index of each surviving node in the graph_cache arrays, and its id in the
    # CAVE skeleton's vertex array. Both stored: the first is how you fetch the
    # embedding, the second is the real foreign key, and neither should ever be
    # re-derived by matching coordinates.
    cache_idx = np.flatnonzero(keep).astype(np.int32)
    orig_ids = orig[keep]

    sr = out / "soma_restricted" / f"{rid}.npz"
    sr.parent.mkdir(parents=True, exist_ok=True)
    if force or not sr.exists():
        _atomic_savez(
            sr, out,
            root_id=np.array([rid], np.uint64),
            keep=keep,
            dist_to_nucleus_nm=r["dist_to_nucleus_nm"],
            component=r["component"],
            cache_index=cache_idx,
            orig_node_ids=orig_ids,
            edge_index=r["edge_index"].astype(np.int32),
            edge_attr=r["edge_attr"],
            nucleus_xyz=(np.full(3, np.nan) if nucleus_xyz is None
                         else np.asarray(nucleus_xyz, np.float64)),
            cut_applied=np.array([r["cut_applied"]], bool),
            soma_radius_nm=np.array(
                [np.nan if nucleus_xyz is None else radius_nm], np.float64),
            n_nodes_before=np.array([r["n_nodes_before"]], np.int64),
            n_nodes_after=np.array([n_kept], np.int64),
            n_components=np.array([r["n_components"]], np.int64),
        )

    stats = {"n_nodes_before": r["n_nodes_before"], "n_nodes_after": n_kept,
             "n_components": r["n_components"], "cut_applied": r["cut_applied"],
             "configs": {}}
    if n_kept < 1:
        return stats

    offsets, neighbors, weights = r["csr"]
    assert_forest(offsets, neighbors)
    comp = r["component"]

    for name in [c for c in configs if c in NEIGHBORHOOD_CONFIGS]:
        dest = out / "neighborhoods" / name / f"{rid}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            stats["configs"][name] = "skipped"
            continue
        nb = neighborhoods(offsets, neighbors, weights, **NEIGHBORHOOD_CONFIGS[name])
        _atomic_savez(
            dest, out,
            root_id=np.array([rid], np.uint64),
            members=nb["members"],
            offsets=nb["offsets"],
            cable_nm=nb["cable_nm"],
            radius_nm=nb["radius_nm"],
            n_members=nb["n_members"],
            component=comp.astype(np.int32),
            cache_index=cache_idx,
            orig_node_ids=orig_ids,
        )
        stats["configs"][name] = {
            "n_paths": int(len(nb["cable_nm"])),
            "median_nodes": float(np.median(nb["n_members"])),
            "median_nm": float(np.median(nb["cable_nm"])),
        }

    for name in [c for c in configs if c in CONFIGS]:
        dest = out / "paths" / name / f"{rid}.npz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            stats["configs"][name] = "skipped"
            continue
        kw = CONFIGS[name]
        per_node, arms = count_paths(offsets, neighbors, weights, **kw)
        res = centered_paths(offsets, neighbors, weights, per_node=per_node,
                             arms=arms, **kw)
        center_node = np.repeat(np.arange(n_kept, dtype=np.int32), per_node)
        _atomic_savez(
            dest, out,
            root_id=np.array([rid], np.uint64),
            path_nodes=res["path_nodes"],
            path_offsets=res["path_offsets"],
            center_at=res["center_at"],
            center_node=center_node,
            geodesic_nm=res["geodesic_nm"],
            component=comp[center_node].astype(np.int32),
            cache_index=cache_idx,
            orig_node_ids=orig_ids,
            paths_per_node=per_node.astype(np.int32),
        )
        stats["configs"][name] = {
            "n_paths": int(len(res["center_at"])),
            "median_nodes": float(np.median(np.diff(res["path_offsets"]))),
            "median_nm": float(np.median(res["geodesic_nm"])),
        }
    return stats


def _atomic_savez(dest, out, **arrays):
    """Write via a partial beside the output, then rename.

    np.savez_compressed appends '.npz' to any path lacking it, so the partial
    already carries the suffix -- otherwise the rename chases a name numpy never
    wrote. Same filesystem, because os.replace is only atomic within one.
    """
    tmp_dir = out / "partial"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{os.getpid()}_{dest.parent.name}_{dest.name}"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="default: derived from --radius-nm")
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "graph_cache"))
    ap.add_argument("--radius-nm", type=float, default=DEFAULT_SOMA_RADIUS_NM)
    ap.add_argument("--configs", nargs="*",
                    default=list(CONFIGS) + list(NEIGHBORHOOD_CONFIGS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out) if a.out else out_for(a.radius_nm)
    nucleus = json.loads((ROOT / "data" / "nucleus_positions.json").read_text())
    positions = {int(k): tuple(v) for k, v in nucleus["positions"].items()}

    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    cells = manifest["cells"]
    # Every labelled cell is built. A cell the store has no nucleus position for
    # is built *uncut* rather than dropped -- see soma_restrict.restrict.
    ids = sorted((int(r) for r in cells),
                 key=lambda r: -cells[str(r)]["n_nodes_covered"])
    mine = ids[a.task_id::a.num_tasks]
    n_uncut = sum(1 for r in ids if r not in positions)

    print(f"task {a.task_id}/{a.num_tasks}: {len(mine)} cells of {len(ids)} "
          f"({n_uncut} have no nucleus position and are built uncut), "
          f"radius {a.radius_nm/1000:g} um, out={out}", flush=True)

    (out / "status").mkdir(parents=True, exist_ok=True)
    status = out / "status" / f"task_{a.task_id:04d}.tsv"
    done = failed = 0
    for i, rid in enumerate(mine):
        t = time.time()
        try:
            st = build_cell(rid, Path(a.cache_dir), positions.get(rid), out,
                            a.radius_nm, a.configs, force=a.force)
            done += 1
            per = "  ".join(
                f"{k}={v['n_paths']:,}" if isinstance(v, dict) else f"{k}=skip"
                for k, v in st["configs"].items())
            print(f"[{i+1}/{len(mine)}] {rid} {cells[str(rid)]['cell_type']}: "
                  f"{st['n_nodes_before']:,}->{st['n_nodes_after']:,} nodes, "
                  f"{st['n_components']} comp"
                  f"{'' if st['cut_applied'] else '  [UNCUT: no nucleus]'}  "
                  f"{per}  ({time.time()-t:.1f}s)", flush=True)
            with open(status, "a") as fh:
                fh.write("\t".join([str(rid), "ok", f"{time.time()-t:.1f}",
                                    str(st["n_nodes_before"]), str(st["n_nodes_after"]),
                                    str(st["n_components"])] +
                                   [str(st["configs"].get(c, {}).get("n_paths", ""))
                                    if isinstance(st["configs"].get(c), dict) else ""
                                    for c in a.configs]) + "\n")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{i+1}/{len(mine)}] {rid} FAILED {type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            with open(status, "a") as fh:
                fh.write(f"{rid}\terror\t{time.time()-t:.1f}\t\t\t\t{type(exc).__name__}: {exc}\n")

    print(f"\ntask {a.task_id}: {done} ok, {failed} failed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
