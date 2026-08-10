"""Builds the local GNN dataset from the public MICrONS SegCLR release + real
ground-truth cell-type labels. No segclr_db store is used anywhere here --
see data/cave_skeletons.py for why CAVESkeletonSource is reused as a library
while the Store/Writer/Database registry is not.

For each of the 398 labeled cells in the official label table:
  1. fetch its CAVE skeleton (cached to data/skeleton_cache/*.pkl)
  2. fetch its raw per-node embeddings from the public GCS release, using the
     nm-coord / public-offset variant so xyz lines up with CAVE (validated in
     scripts/explore_cave_alignment.py: median nearest-neighbor residual
     ~740nm against a skeleton spanning >1mm)
  3. assign each embedding row to its nearest skeleton node (cKDTree), mean
     over collisions (multiple embedding rows commonly land on one skeleton
     node -- SegCLR's own sampling is denser than CAVE's skeleton nodes), and
     drop skeleton nodes with no assigned embedding at all
  4. symmetrize skeleton edges restricted to the covered node subset, with
     edge length (nm) as edge_attr
  5. save one torch_geometric Data object per cell under data/graph_cache/

Also writes data/manifest.json: per-cell root_id, cell_type (full string, so
downstream code can truncate to any hierarchy depth it wants), and a
deterministic stratified train/val/test split.

Parallelism: skeletons are fetched with ONE call to CAVESkeletonSource, which
already batches/paces internally against CAVE's own rate limits -- splitting
that across a SLURM array would not fetch any faster (the limit is on CAVE's
side, shared across however many callers hit it) and would just multiply job
startup overhead. The embedding download (independent, anonymous GCS zip
fetches from a public bucket, one per cell) is the part actually worth
parallelizing, and is I/O-bound -- each call spends almost all its time
waiting on the network, releasing the GIL, so a thread pool gets real
wall-clock speedup without the per-task interpreter/import startup cost a
SLURM array would pay ~40 times over (once per array task). This is also
exactly what the original access notebook does
(https://colab.research.google.com/gist/chinasaur/63f15b3f37b35b5bb27de31ba0a0087f,
`ThreadPoolExecutor(max_workers=10)`); --workers defaults a bit above that.

Run via sbatch (mit_normal, not mit_quicktest -- CAVE fetching + ~400 GCS zip
downloads will not reliably fit in 15 minutes even though most skeletons
already exist). Resumable: cells with a cached .pt file are skipped.

    python data/build_dataset.py --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import cave_skeletons as cs  # noqa: E402
from data import public_reader as pr  # noqa: E402

GRAPH_CACHE_DIR = Path(__file__).resolve().parent / "graph_cache_deprecated"
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest_deprecated.json"
DATA_KEY = "microns_nm_coord_public_offset_v343"
SPLIT_SEED = 0
# train, test -- there is no separate val fraction: "val" is an alias for
# the test split at the consumer level (scripts/train_gnn.py builds its
# "val" dataset from split=="test"), not a third partition computed here.
# See stratified_split's docstring.
SPLIT_FRACS = (0.8, 0.2)
DEFAULT_WORKERS = 12

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("build_dataset")


def _cell_path(root_id: int) -> Path:
    return GRAPH_CACHE_DIR / f"{root_id}.pt"


def build_one_cell(root_id: int, skeleton, cell: pr.RawCellEmbeddings):
    """Returns a torch_geometric.data.Data, or None if nothing usable."""
    import torch
    from torch_geometric.data import Data

    if cell.embeddings.shape[0] == 0 or len(skeleton) == 0:
        return None

    tree = cKDTree(skeleton.coords.astype(np.float64))
    dist, node_idx = tree.query(cell.xyz_nm.astype(np.float64))

    d = cell.embeddings.shape[1]
    n_nodes = len(skeleton)
    sums = np.zeros((n_nodes, d), dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(sums, node_idx, cell.embeddings.astype(np.float64))
    np.add.at(counts, node_idx, 1)

    covered = counts > 0
    n_covered = int(covered.sum())
    if n_covered == 0:
        return None

    x_full = np.zeros((n_nodes, d), dtype=np.float32)
    x_full[covered] = (sums[covered] / counts[covered, None]).astype(np.float32)

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
    data.match_dist_median_nm = float(np.median(dist))
    # FK back into the ORIGINAL (uncovered-included) skeleton node indexing --
    # segclr_db's aggregate.geodesic_mean needs the true skeleton topology
    # (including nodes this cell has no embedding for) to compute geodesic
    # windows correctly, so the baseline recomputes from skeleton_cache using
    # this rather than the covered-only, remapped edge_index above. Keeping
    # this FK explicit is the same "(root_id, node_id) is a real foreign key"
    # principle CLAUDE.md documents for segclr_db itself, just done locally.
    data.orig_node_ids = torch.from_numpy(np.nonzero(covered)[0]).long()
    return data


def _fetch_and_build(root_id: int, skeleton, fs) -> tuple[int, object, dict | None, str | None]:
    """Worker body: one GCS embedding fetch + one build_one_cell. Runs in a
    thread -- fs (gcsfs) and cKDTree/numpy calls are all fine to share a
    filesystem handle across threads for read-only access; each thread writes
    a distinct root_id's file, so there's no shared mutable state to race on.
    Returns (root_id, torch.save-ready Data or None, timing dict, error str).
    """
    t0 = time.monotonic()
    try:
        cell = pr.get_raw_cell_embeddings(root_id, fs, data_key=DATA_KEY)
        t_fetch = time.monotonic()
        data = build_one_cell(root_id, skeleton, cell)
        t_build = time.monotonic()
    except Exception as exc:  # noqa: BLE001 -- reported per-cell, loop continues
        return root_id, None, None, f"{type(exc).__name__}: {exc}"
    timing = {
        "n_embedding_rows": int(cell.embeddings.shape[0]),
        "fetch_s": t_fetch - t0,
        "build_s": t_build - t_fetch,
    }
    return root_id, data, timing, None


def stratified_split(labels: dict[int, str], seed: int = SPLIT_SEED, fracs=SPLIT_FRACS):
    """Deterministic per-class split, two-way (train/test only -- see
    SPLIT_FRACS's comment for why there's no separate val partition; callers
    that need a validation set during training use split=="test" for it,
    i.e. val and test are the SAME held-out cells by design, not two
    disjoint holdouts). Classes with < 2 examples can't be represented in
    both splits -- those go entirely to train, logged so it's visible
    rather than silently thinning the test split."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = {}
    for root_id, label in labels.items():
        by_class.setdefault(label, []).append(root_id)

    split_of: dict[int, str] = {}
    thin_classes = []
    for label, ids in by_class.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        if n < 2:
            thin_classes.append((label, n))
            for r in ids:
                split_of[r] = "train"
            continue
        n_train = max(1, round(fracs[0] * n))
        n_train = min(n_train, n - 1)  # leave >=1 for test
        for r in ids[:n_train]:
            split_of[r] = "train"
        for r in ids[n_train:]:
            split_of[r] = "test"

    if thin_classes:
        logger.warning(
            "%d classes with <2 examples went entirely to train: %s", len(thin_classes), thin_classes
        )
    return split_of


def main(args) -> int:
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("loading ground-truth labels ...")
    fs = pr.get_public_filesystem()
    labels_df = pr.get_celltype_labels(fs)
    labels = {int(row.seg_id): str(row.cell_type) for row in labels_df.itertuples()}
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

    if todo:
        import os

        token = os.environ.get("CAVE_TOKEN")
        if not token:
            logger.error("CAVE_TOKEN not set -- cannot fetch skeletons for the cells not yet built")
            return 2
        cave_config = cs.default_cave_config(token)

        logger.info("fetching %d skeletons from CAVE (one batched, rate-limited call) ...", len(todo))
        t0 = time.monotonic()
        skeletons = cs.fetch_skeletons(todo, cave_config, log=logger.info)
        logger.info("skeleton fetch done in %.0fs (%d/%d cells have a skeleton)", time.monotonic() - t0, len(skeletons), len(todo))

        has_skeleton = [r for r in todo if r in skeletons]
        n_no_skeleton = len(todo) - len(has_skeleton)
        if n_no_skeleton:
            logger.warning("%d cells have no skeleton (refused/not ready) -- will be skipped", n_no_skeleton)

        logger.info(
            "downloading %d embedding sets from public GCS with %d parallel workers ...",
            len(has_skeleton), args.workers,
        )
        n_ok, n_skip, n_err = 0, 0, 0
        fetch_times, node_counts = [], []
        t0 = time.monotonic()
        import torch

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_fetch_and_build, root_id, skeletons[root_id], fs): root_id
                for root_id in has_skeleton
            }
            pbar = tqdm(as_completed(futures), total=len(futures), desc="cells", unit="cell")
            for fut in pbar:
                root_id = futures[fut]
                root_id, data, timing, err = fut.result()
                if err is not None:
                    n_err += 1
                    logger.warning("  cell %d failed: %s", root_id, err)
                    pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)
                    continue
                if data is None:
                    n_skip += 1
                    pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)
                    continue

                torch.save(data, _cell_path(root_id))
                cells_manifest[str(root_id)] = {
                    "cell_type": labels[root_id],
                    "n_nodes_skeleton": data.n_nodes_skeleton,
                    "n_nodes_covered": data.n_nodes_covered,
                    "match_dist_median_nm": data.match_dist_median_nm,
                }
                n_ok += 1
                fetch_times.append(timing["fetch_s"])
                node_counts.append(timing["n_embedding_rows"])
                pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_err)

        elapsed = time.monotonic() - t0
        logger.info(
            "built %d cells, skipped %d (no covered nodes), %d errors, in %.0fs (%.2f cells/s)",
            n_ok, n_skip, n_err, elapsed, n_ok / max(elapsed, 1e-9),
        )
        if fetch_times:
            logger.info(
                "GCS fetch latency per cell: median=%.2fs p90=%.2fs max=%.2fs  |  embedding rows/cell: median=%d max=%d",
                float(np.median(fetch_times)), float(np.percentile(fetch_times, 90)), float(np.max(fetch_times)),
                int(np.median(node_counts)), int(np.max(node_counts)),
            )

    built_ids = {int(r) for r in cells_manifest}
    split_of = stratified_split({r: cells_manifest[str(r)]["cell_type"] for r in built_ids})
    for root_id, split in split_of.items():
        cells_manifest[str(root_id)]["split"] = split

    manifest["cells"] = cells_manifest
    manifest["data_key"] = DATA_KEY
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
    p.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="parallel threads for GCS embedding downloads (I/O-bound; skeleton fetch is not parallelized here -- see module docstring)",
    )
    raise SystemExit(main(p.parse_args()))
