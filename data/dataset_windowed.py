"""Per-window (per-point) GNN dataset -- the training/eval unit is a small
local-neighborhood subgraph around one skeleton node, scoped to roughly the
baseline's window_nm, not a whole cell. See CLAUDE.md's project-goal section
for the rationale: the baseline (and the SegCLR paper) classifies per point
from a small context window then majority-votes up to a cell-level answer;
the GNN's classifier does the same, just aggregating the window with a
GraphTransformer (or, for the baseline configuration in gnn/model.py, a plain
mean) instead of a geodesic mean baked into the cached data.

Reads data/graph_cache/*.pt (raw 64-dim segclr_db node embeddings + skeleton
edges, data/build_dataset_from_store.py) plus the per-cell window membership
data/build_window_membership.py precomputes, and extracts one window subgraph
per (cell, node) pair on the fly via data/geodesic_window.py::extract_window_subgraph.

The cached Data objects carry no y_levels -- computed here at load time from
the manifest's flat cell_type string and attached onto each whole-cell Data
object once, before any windows are cut from it.

`use_thickness=True` additionally joins in the spine-corrected dendrite shaft
radius (data/dendrite_thickness_cache/*.npz, see data/DENDRITE_THICKNESS.md)
as a per-node feature, normalized and NaN-masked once per cell here so the
per-window hot path only has to index it. It must be enabled in lockstep with
the model's own switch -- scripts/train_gnn.py drives both from one flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler

from data.dataset_lcpn import load_hierarchy, load_manifest, split_cells  # noqa: F401 -- re-exported
from data.geodesic_window import (
    DEFAULT_POS_DIM,
    DEFAULT_WINDOW_NM,
    THICKNESS_DIM,
    THICKNESS_SCALE_NM,
    extract_window_subgraph,
    membership_dir_name,
)

GRAPH_CACHE_DIR_NAME = "graph_cache"
THICKNESS_CACHE_DIR_NAME = "dendrite_thickness_cache"
DEFAULT_FIXED_NEIGHBORHOOD_ROOT = Path(
    "/orcd/scratch/orcd/013/jcbliao/embedding_paths/r5um"
)


def load_thickness_features(
    npz_path: Path, orig_node_ids: torch.Tensor, n_nodes: int
) -> tuple[torch.Tensor, bool]:
    """Per-node dendrite-thickness feature for one cell, (n_nodes, THICKNESS_DIM)
    float32, plus whether a cache file was actually found.

    The cache (data/build_dendrite_thickness.py) stores `radius_nm` indexed by
    SKELETON vertex, while the graph cache holds only the subset of vertices
    that had embeddings. `orig_node_ids` is the exact index of each graph node
    back into that skeleton vertex array (data/build_dataset_from_store.py),
    so this is a real id-based join, not a positional or coordinate one.

    Channel 0 is radius / THICKNESS_SCALE_NM with non-finite entries zeroed;
    channel 1 is 1.0 where the radius was actually measured, 0.0 otherwise.
    See THICKNESS_DIM's comment for why the second channel is not optional:
    NaN is the cache's normal, expected value for every axon node, every
    branch point, and every mesh-hole miss.

    A missing cache file is returned as an all-unmeasured block (both channels
    zero) rather than raised on, so one un-ingested cell can't take down a run
    -- the caller reports the count, and an all-zero mask channel is exactly
    the honest encoding of "nothing measured here."
    """
    if not npz_path.exists():
        return torch.zeros(n_nodes, THICKNESS_DIM, dtype=torch.float32), False

    radius_nm = np.load(npz_path)["radius_nm"][orig_node_ids.numpy()]
    measured = np.isfinite(radius_nm)
    feat = np.zeros((n_nodes, THICKNESS_DIM), dtype=np.float32)
    feat[measured, 0] = radius_nm[measured] / THICKNESS_SCALE_NM
    feat[:, 1] = measured
    return torch.from_numpy(feat), True


def inverse_sqrt_class_weights(class_indices: np.ndarray, num_classes: int) -> torch.Tensor:
    """Per-class sampling weight, ported from segCLR_cell_classification's
    `HierarchyCellTypingDataset._class_weights`: `1/sqrt(count)`, rescaled so
    the weights sum to `num_classes` (i.e. average 1.0).

    The square root is theirs and is the whole character of the correction --
    plain `1/count` equalizes the classes outright, while `1/sqrt(count)`
    leaves a residual tilt toward the populous ones. Over a ~3100x spread
    between pyramidal and OPC windows, that is the difference between
    resampling OPC's 2562 windows ~3100x per epoch and ~56x. Their own commit
    history shows `1/count` written first and then replaced, so this is a
    deliberate choice, not an approximation.

    Counts are clamped to 1 before the reciprocal. Theirs doesn't clamp, and
    on a zero-count class produces `inf`, which then makes `w.sum()` infinite
    and collapses EVERY weight to zero or NaN -- a silently degenerate
    sampler rather than a loud failure. Where every class has support the two
    agree exactly.
    """
    counts = torch.bincount(torch.as_tensor(class_indices), minlength=num_classes).float()
    weights = 1.0 / torch.sqrt(counts.clamp(min=1.0))
    return weights / weights.sum() * num_classes


def balanced_sampler(dataset: "WindowedGraphDatasetLCPN") -> WeightedRandomSampler:
    """Class-balanced resampling of `dataset`'s windows, the correction
    segCLR_cell_classification's own LCPN config uses
    (`weight_imbalanced_classes: sample`).

    One epoch still draws `len(dataset)` windows, but WITH REPLACEMENT and in
    proportion to each window's class weight -- so a rare class's windows
    recur many times per epoch and a populous class's are subsampled, and the
    loss itself stays unweighted. Weighting is by the finest level only, as
    theirs is; the coarser levels rebalance implicitly, since every window's
    whole path is determined by its finest-level class.
    """
    num_classes = len(dataset.hierarchy.level_classes[-1])
    class_weights = inverse_sqrt_class_weights(dataset.index_labels, num_classes)
    weights = class_weights[torch.from_numpy(dataset.index_labels)]
    # torch.multinomial, which WeightedRandomSampler calls, refuses more than
    # 2**24 categories. At one window per node this bites only at ~16.8M
    # windows; failing here names the reason rather than surfacing as an
    # opaque error from inside the sampler on the first epoch.
    if len(weights) > 2**24:
        raise ValueError(
            f"{len(weights)} windows exceeds torch.multinomial's 2**24 category limit, so "
            "WeightedRandomSampler cannot draw from them -- use --class-balance loss instead"
        )
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


class WindowedGraphDatasetLCPN(TorchDataset):
    """One split's worth of (cell, node) windows. __len__ is the total node
    count across the split's cells -- the same order of magnitude as the
    baseline's row count, not the cell count (see module docstring).

    Cell graphs + window membership are loaded once at construction and kept
    resident in memory -- re-reading a .pt/.npz file from disk on every
    __getitem__ across millions of window queries would dominate wall clock
    otherwise.
    """

    def __init__(
        self,
        manifest: dict,
        split: str,
        pos_dim: int = DEFAULT_POS_DIM,
        use_thickness: bool = False,
        window_nm: float = DEFAULT_WINDOW_NM,
        num_embeddings: int | None = 20,
        neighborhood_root: str | Path = DEFAULT_FIXED_NEIGHBORHOOD_ROOT,
    ):
        repo_root = Path(__file__).resolve().parent.parent
        graph_cache_dir = repo_root / "data" / GRAPH_CACHE_DIR_NAME
        if num_embeddings is not None and num_embeddings not in (10, 20, 40):
            raise ValueError("num_embeddings must be one of 10, 20, 40, or None")
        membership_dir = (
            Path(neighborhood_root) / "neighborhoods" / f"n{num_embeddings}"
            if num_embeddings is not None
            else repo_root / "data" / membership_dir_name(window_nm)
        )
        thickness_dir = repo_root / "data" / THICKNESS_CACHE_DIR_NAME
        self.window_nm = window_nm
        self.num_embeddings = num_embeddings
        if not membership_dir.is_dir():
            # Every radius needs its own precomputed cache. Failing here names
            # the exact command, rather than letting the per-cell "missing
            # membership" warning below fire for all of them and yield an
            # empty dataset.
            raise FileNotFoundError(
                f"no neighborhood membership cache at {membership_dir}"
            )
        self.pos_dim = pos_dim  # width of GraphTransformer's per-window Laplacian PE
        self.use_thickness = use_thickness

        self.hierarchy = load_hierarchy(manifest)
        self.classes = self.hierarchy.level_classes[-1]

        cells = split_cells(manifest, split, self.hierarchy)

        self.cell_data: dict[int, object] = {}
        self.cell_membership: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        index_root_ids: list[int] = []
        index_centers: list[int] = []
        index_membership_centers: list[int] = []
        index_labels: list[int] = []
        n_missing_membership = 0
        n_missing_thickness = 0
        for root_id, info in cells:
            membership_path = membership_dir / f"{root_id}.npz"
            if not membership_path.exists():
                n_missing_membership += 1
                continue
            data = torch.load(graph_cache_dir / f"{root_id}.pt", weights_only=False)
            data.y_levels = self._y_levels(info["cell_type"])
            if use_thickness:
                # Normalized + NaN-masked once here, per cell, rather than in
                # extract_window_subgraph -- that runs millions of times per
                # epoch and should only index an already-prepared tensor.
                if not hasattr(data, "orig_node_ids"):
                    raise AttributeError(
                        f"use_thickness=True but cell {root_id}'s cached Data has no "
                        "orig_node_ids, which is the id-based join back into the skeleton "
                        "vertex array the thickness cache is indexed by (see "
                        "data/DENDRITE_THICKNESS.md). Rerun data/build_dataset_from_store.py."
                    )
                data.thickness, found = load_thickness_features(
                    thickness_dir / f"{root_id}.npz", data.orig_node_ids, data.x.shape[0]
                )
                n_missing_thickness += not found
            npz = np.load(membership_path)
            if num_embeddings is not None:
                # Fixed-node neighborhoods are indexed in the soma-restricted
                # graph. cache_index is the lossless mapping back to graph_cache.
                cache_index = npz["cache_index"].astype(np.int64, copy=False)
                offsets = npz["offsets"]
                members = cache_index[npz["members"]]
                centers = cache_index
            else:
                offsets = npz["mem_offsets"]
                members = npz["members"]
                centers = np.arange(data.x.shape[0], dtype=np.int64)
            self.cell_data[root_id] = data
            self.cell_membership[root_id] = (offsets, members)
            n_nodes = len(centers)
            index_root_ids.extend([root_id] * n_nodes)
            index_centers.extend(centers.tolist())
            index_membership_centers.extend(range(n_nodes))
            # Every window cut from one cell inherits that cell's label, so the
            # finest-level class index is known here without touching a window.
            index_labels.extend([int(data.y_levels[0, -1])] * n_nodes)

        if n_missing_membership:
            import logging

            logging.getLogger(__name__).warning(
                "%d/%d cells in split %r have no precomputed window membership "
                "(run data/build_window_membership.py) -- excluded from this dataset",
                n_missing_membership, len(cells), split,
            )

        if n_missing_thickness:
            import logging

            # Warn rather than fail: these cells stay in the dataset with an
            # all-zero measured flag, which is a truthful "not measured," not a
            # fabricated radius. Loud because a large count means the ingest
            # (scripts/sbatch/build_dendrite_thickness.sh) is incomplete and the
            # feature is mostly absent rather than mostly present.
            logging.getLogger(__name__).warning(
                "%d/%d cells in split %r have no dendrite-thickness cache entry -- their "
                "thickness feature is all-unmeasured (both channels 0). Run "
                "scripts/sbatch/build_dendrite_thickness.sh to backfill.",
                n_missing_thickness, len(self.cell_data), split,
            )

        self.index_root_ids = np.array(index_root_ids, dtype=np.int64)
        self.index_centers = np.array(index_centers, dtype=np.int64)
        self.index_membership_centers = np.array(index_membership_centers, dtype=np.int64)
        #: Finest-level class index per window, parallel to the two arrays
        #: above. Feeds balanced_sampler() without a pass over the windows.
        self.index_labels = np.array(index_labels, dtype=np.int64)

    def _y_levels(self, cell_type: str) -> torch.Tensor:
        path = self.hierarchy.label_paths[cell_type]
        levels = [self.hierarchy.level_maps[lvl][path[lvl]] for lvl in range(self.hierarchy.depth)]
        return torch.tensor(levels, dtype=torch.long).unsqueeze(0)  # (1, depth)

    def __len__(self) -> int:
        return len(self.index_root_ids)

    def __getitem__(self, i: int):
        root_id = int(self.index_root_ids[i])
        center = int(self.index_centers[i])
        data = self.cell_data[root_id]
        mem_offsets, members = self.cell_membership[root_id]
        # Fixed-node offsets are indexed by the restricted centre ordinal,
        # whereas `center` is its graph_cache index. The dataset row ordinal
        # within this cell recovers the former without a coordinate join.
        if self.num_embeddings is not None:
            membership_center = int(self.index_membership_centers[i])
        else:
            membership_center = center
        window = extract_window_subgraph(
            data, membership_center, mem_offsets, members, pos_dim=self.pos_dim
        )
        window.y = window.y_levels[:, -1]  # finest-level alias
        return window
