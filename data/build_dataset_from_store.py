"""Builds the GNN dataset from the real segclr-db store -- replaces the
deprecated nearest-neighbor pipeline (data/DEPRECATED.md). Read-only:
SegCLRDatabase + SkeletonCache(fetch_if_missing=False), never SegCLRWriter.

Data sources:
  - embeddings + skeletons: /orcd/compute/sdorkenw/001/segclr-db,
    dataset "microns", experiment "resnet_860b_reshuffled" (a model this lab
    trained/ran -- NOT Google's public SegCLR release; confirmed intentional,
    see data/DEPRECATED.md). node_id already indexes the same skeleton for
    both node_embeddings and skeleton_nodes/skeleton_edges by construction --
    no nearest-neighbor matching needed, unlike the deprecated pipeline.
  - labels: segclr_db's own registered `cell_labels` table
    (db.get_labels(label_set="cell_type")), not a direct CAVE query --
    switched 2026-08-06 per explicit user direction, once the store's
    permissions opened enough to read it (scripts/explore_db_cell_labels.py
    re-checked live and found it populated, not empty/blocked as earlier
    checks this session had found). Strictly more coverage than the CAVE
    `cortical_neurons`+status_axon=True query it replaces: 2410 labeled
    cells vs. that query's 2193 (100% label agreement on the overlap),
    including astrocyte/microglia/thalamocortical cells the old
    status_axon=True filter excluded outright -- the old "non_neuron branch
    never receives gradient" caveat no longer applies once this is rerun.
    Chandelier cells (ChC, n=1 -- too few to train or hold out on) are
    dropped here; see EXCLUDED_LABELS.

For each labeled cell (minus EXCLUDED_LABELS):
  1. fetch raw node_embeddings (root_id, node_id, embedding) -- one call,
     already exactly indexed to the skeleton
  2. fetch the skeleton (coords, edges) from the store (no CAVE call: already
     ingested)
  3. restrict to nodes with an embedding (should be ~all; some cells may be
     partially embedded), symmetrize edges, edge_attr = length (nm)
  4. save one torch_geometric Data per cell to data/graph_cache/
  5. also cache the Skeleton itself to data/skeleton_cache/*.pkl (shared
     format with the deprecated pipeline -- baseline/mean_pool_classifier.py (deleted 2026-08-06, deprecated cleanup)
     reads from there regardless of which pipeline produced it) so the
     geodesic-mean baseline can be recomputed without re-hitting the store

Also writes data/manifest.json: per-cell root_id, cell_type label (flat
Allen-style string, e.g. "L4IT" -- no dash-hierarchy like the deprecated
pipeline's labels, so there is no depth>flat grouping here yet), and a
deterministic stratified train/test split (reuses
data.build_dataset.stratified_split -- pure function, source-agnostic). No
separate val fraction: "val" is an alias for the test split at the consumer
level (scripts/train_gnn.py builds its "val" dataset from split=="test")
rather than a third partition computed here -- an 80/20 split, val=test.

Run via sbatch (mit_normal, >=32G -- reading tables this size needs real
memory, see scripts/explore_real_store.py's OOM lesson). Resumable: cells
with a cached .pt file are skipped.

    python data/build_dataset_from_store.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

from segclr_db import store as st  # noqa: E402
from segclr_db.database import SegCLRDatabase  # noqa: E402
from segclr_db.skeletons import SkeletonCache  # noqa: E402

from data import cave_skeletons as cs  # noqa: E402 -- shared pickle cache path/format only
from data.build_dataset import stratified_split  # noqa: E402 -- pure fn, source-agnostic

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
STORE_DATASET = "microns"
EXPERIMENT_ID = "resnet_860b_reshuffled"
LABEL_SET = "cell_type"  # the only label_set currently in segclr_db's cell_labels table
EXCLUDED_LABELS = {"ChC"}  # chandelier cells: n=1 in the store, too few to train or hold out on

GRAPH_CACHE_DIR = Path(__file__).resolve().parent / "graph_cache"
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
SPLIT_SEED = 0
SPLIT_FRACS = (0.8, 0.2)  # train, test -- val is an alias for test, see module docstring
DEFAULT_WORKERS = 8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("build_dataset_from_store")


def _cell_path(root_id: int) -> Path:
    return GRAPH_CACHE_DIR / f"{root_id}.pt"


def build_one_cell(root_id: int, skeleton, node_ids: np.ndarray, embeddings: np.ndarray):
    """Returns a torch_geometric.data.Data, or None if nothing usable.

    Unlike the deprecated pipeline's build_one_cell, there is no nearest-
    neighbor step: node_ids already indexes skeleton.coords/edges exactly.
    """
    import torch
    from torch_geometric.data import Data

    n_nodes = len(skeleton)
    if n_nodes == 0 or len(node_ids) == 0:
        return None
    if node_ids.min() < 0 or node_ids.max() >= n_nodes:
        raise ValueError(
            f"cell {root_id}: node_ids range {node_ids.min()}..{node_ids.max()} "
            f"out of bounds for skeleton with {n_nodes} nodes"
        )

    d = embeddings.shape[1]
    covered = np.zeros(n_nodes, dtype=bool)
    covered[node_ids] = True
    n_covered = int(covered.sum())
    if n_covered == 0:
        return None

    x_full = np.zeros((n_nodes, d), dtype=np.float32)
    x_full[node_ids] = embeddings.astype(np.float32)

    old_to_new = -np.ones(n_nodes, dtype=np.int64)
    old_to_new[covered] = np.arange(n_covered)

    edges = skeleton.edges
    if len(edges):
        keep = covered[edges[:, 0]] & covered[edges[:, 1]]
        e = edges[keep]
        src, dst = old_to_new[e[:, 0]], old_to_new[e[:, 1]]
        edge_index = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
        lengths = np.linalg.norm(
            skeleton.coords[e[:, 0]].astype(np.float64)
            - skeleton.coords[e[:, 1]].astype(np.float64),
            axis=1,
        ).astype(np.float32)
        edge_attr = np.concatenate([lengths, lengths])[:, None]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 1), dtype=np.float32)

    data = Data(
        x=torch.from_numpy(x_full[covered]),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr),
        pos=torch.from_numpy(skeleton.coords[covered].astype(np.float32)),
    )
    data.root_id = int(root_id)
    data.n_nodes_skeleton = n_nodes
    data.n_nodes_covered = n_covered
    data.orig_node_ids = torch.from_numpy(np.nonzero(covered)[0]).long()
    return data


def _fetch_and_build(root_id: int, db: SegCLRDatabase, skel_cache: SkeletonCache):
    """Worker body: one store read + one build_one_cell. db/skel_cache read
    from Lance files -- safe to share across threads for read-only access."""
    t0 = time.monotonic()
    try:
        skeleton = skel_cache.get_skeleton(root_id, fetch_if_missing=False)
        result = db.get_embeddings(EXPERIMENT_ID, root_ids=root_id)
        data = build_one_cell(root_id, skeleton, result.node_ids, result.embeddings)
        t1 = time.monotonic()
    except KeyError:
        return root_id, None, None, "no skeleton in store"
    except Exception as exc:  # noqa: BLE001 -- reported per-cell, loop continues
        return root_id, None, None, f"{type(exc).__name__}: {exc}"
    return root_id, data, skeleton, None if data is not None else "no covered nodes"


def fetch_labels(db: SegCLRDatabase) -> tuple[dict[int, str], list[int]]:
    """Raw granular cell_type label per cell, straight from segclr_db's own
    registered cell_labels table -- see module docstring for why this
    replaced the earlier direct CAVE query. Returns (labels, mat_versions)
    so main() can record the real mat_version(s) the data came from instead
    of a hardcoded constant that could silently drift out of sync with what
    the store actually holds.
    """
    df = db.get_labels(label_set=LABEL_SET)
    mat_versions = sorted(int(v) for v in df["mat_version"].dropna().unique())
    labels = {int(r.root_id): str(r.label) for r in df.itertuples()}

    n_before = len(labels)
    labels = {rid: lab for rid, lab in labels.items() if lab not in EXCLUDED_LABELS}
    n_excluded = n_before - len(labels)
    if n_excluded:
        logger.info("  dropped %d cells with excluded labels %s", n_excluded, sorted(EXCLUDED_LABELS))
    return labels, mat_versions


def main(args) -> int:
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    store = st.open_store(STORE_ROOT, STORE_DATASET)
    db = SegCLRDatabase(store=store)
    skel_cache = SkeletonCache(store)  # read-only: no cave_config

    logger.info("fetching labels from segclr_db cell_labels (label_set=%r) ...", LABEL_SET)
    labels, label_mat_versions = fetch_labels(db)
    logger.info("  %d labeled cells, %d distinct cell_type values", len(labels), len(set(labels.values())))

    root_ids = sorted(labels)
    todo = [r for r in root_ids if not _cell_path(r).exists()]
    logger.info(
        "%d total cells, %d already built, %d to do", len(root_ids), len(root_ids) - len(todo), len(todo)
    )

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    cells_manifest = manifest.get("cells", {})

    # Prune anything not in the CURRENT label set -- e.g. a ChC cell cached
    # under an older manifest.json from before EXCLUDED_LABELS existed, or
    # any other cell that dropped out on a label-source change -- and
    # refresh cell_type from that current source, so a stale on-disk
    # manifest.json never silently outlives a label-source switch (like
    # this one, CAVE query -> segclr_db cell_labels). Cells needing a fresh
    # fetch (not yet in cells_manifest at all) are added below.
    cells_manifest = {
        str(rid): {**cells_manifest[str(rid)], "cell_type": labels[rid]}
        for rid in labels
        if str(rid) in cells_manifest
    }

    if todo:
        logger.info("reading %d cells from the store with %d parallel workers ...", len(todo), args.workers)
        n_ok, n_skip, n_err = 0, 0, 0
        t0 = time.monotonic()
        import torch

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_fetch_and_build, r, db, skel_cache): r for r in todo}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="cells", unit="cell")
            for fut in pbar:
                root_id, data, skeleton, err = fut.result()
                if data is None:
                    if err and "no skeleton" in err:
                        n_err += 1
                        logger.warning("  cell %d: %s", root_id, err)
                    else:
                        n_skip += 1
                    pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)
                    continue

                torch.save(data, _cell_path(root_id))
                # Shared pickle cache format with the deprecated pipeline --
                # lets baseline/mean_pool_classifier.py (deleted 2026-08-06, deprecated cleanup) recompute geodesic_mean
                # for these cells too without touching the store again.
                cs.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                if not cs._cache_path(root_id).exists():
                    with open(cs._cache_path(root_id), "wb") as f:
                        pickle.dump(skeleton, f)

                cells_manifest[str(root_id)] = {
                    "cell_type": labels[root_id],
                    "n_nodes_skeleton": data.n_nodes_skeleton,
                    "n_nodes_covered": data.n_nodes_covered,
                }
                n_ok += 1
                pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)

        elapsed = time.monotonic() - t0
        logger.info(
            "built %d cells, skipped %d (no covered nodes), %d errors (no skeleton), in %.0fs (%.2f cells/s)",
            n_ok, n_skip, n_err, elapsed, n_ok / max(elapsed, 1e-9),
        )

    built_ids = {int(r) for r in cells_manifest}
    split_of = stratified_split({r: cells_manifest[str(r)]["cell_type"] for r in built_ids})
    for root_id, split in split_of.items():
        cells_manifest[str(root_id)]["split"] = split

    manifest["cells"] = cells_manifest
    manifest["experiment_id"] = EXPERIMENT_ID
    manifest["store_root"] = STORE_ROOT
    manifest["label_source"] = "segclr_db cell_labels"
    manifest["label_set"] = LABEL_SET
    manifest["label_mat_versions"] = label_mat_versions
    manifest["excluded_labels"] = sorted(EXCLUDED_LABELS)
    manifest["split_seed"] = SPLIT_SEED
    manifest["split_fracs"] = list(SPLIT_FRACS)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("wrote manifest for %d cells to %s", len(cells_manifest), MANIFEST_PATH)

    counts = {}
    for v in cells_manifest.values():
        counts[v["split"]] = counts.get(v["split"], 0) + 1
    logger.info("split sizes: %s", counts)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    raise SystemExit(main(p.parse_args()))
