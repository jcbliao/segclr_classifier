"""Per-node geodesic-window MEMBERSHIP -- which other skeleton nodes fall
within `window_nm` of each node, not just their mean embedding.

`segclr_db.aggregate.geodesic_mean` already does the bounded-Dijkstra-per-
source search this needs (numba-compiled, ~3.2us/node flat regardless of
cell size, per its own docstring), but it only returns the aggregated MEAN
per source -- the per-source node membership itself is computed internally
(the `touched` array) and thrown away once the mean is folded in. A GNN
needs the actual node SET (and the induced edges among them) to build one
small subgraph per point, not one number -- hence this sibling module rather
than editing segclr_db/src.

`window_membership()` below is structurally the same bounded-Dijkstra-per-
source kernel as aggregate.py's, with the accumulate-embedding step replaced
by "record every node touched" -- by construction, a node only ever enters
`touched` when it's been relaxed to a distance <= window (see the `nd <=
window` guard before every push), so `touched[:n_touched]` after a source's
search IS exactly that source's window membership, no extra filtering
needed. Two passes (count then fill) size the ragged CSR output, since numba
doesn't grow arrays -- doubles the compute but is still cheap at this
kernel's throughput, and this only ever runs once per cell at dataset-build
time (data/build_window_membership.py), not per training step.
"""

from __future__ import annotations

import numpy as np
import torch

# Default width of the per-window Laplacian positional encoding attached by
# extract_window_subgraph -- consumed by gnn/graph_transformer.py::GraphTransformer.
# Kept as a module constant (rather than only a function default) so
# data/dataset_windowed.py and scripts/train_gnn.py's --gt-pos-dim can
# share one source of truth for what "default" means.
DEFAULT_POS_DIM = 8

# Normalization scale (nm) for the relative-position feature below -- brings
# rel_pos into a roughly O(1) range instead of raw nm magnitudes (thousands),
# which would otherwise dominate the embedding features purely by scale, not
# by information content, once concatenated together. Matches WINDOW_NM
# (data/build_window_membership.py's 10um baseline window) conceptually, not
# imported from there to avoid a circular import (build_window_membership.py
# imports FROM this module) -- both are independent constants tied to the
# same "10um window" convention.
REL_POS_SCALE_NM = 10_000.0

# Normalization scale (nm) for the dendrite-thickness node feature, same role
# REL_POS_SCALE_NM plays for rel_pos: keep the feature near unit magnitude so
# it doesn't dominate the (roughly unit-scale) SegCLR embedding it gets
# concatenated with purely by numeric size. Spine-corrected shaft radii run in
# the high hundreds of nm (see data/DENDRITE_THICKNESS.md), so 1um puts a
# typical dendrite a bit under 1.0.
THICKNESS_SCALE_NM = 1_000.0

# Width of the per-node dendrite-thickness feature: [normalized radius,
# measured flag]. Two channels, not one, because the cache is NaN wherever a
# radius could not be measured -- non-dendrite compartment, branch point, or a
# mesh-hole miss -- and that is a large, systematic, label-correlated fraction
# of nodes, not noise. Feeding a NaN into a Linear poisons the whole batch, and
# silently substituting 0.0 would tell the model "this dendrite is
# infinitely thin" rather than "this was not measured." The flag lets it tell
# the two apart. See data/dataset_windowed.py::load_thickness_features.
THICKNESS_DIM = 2

_window_membership_kernel = None


def _kernel():
    global _window_membership_kernel
    if _window_membership_kernel is not None:
        return _window_membership_kernel

    import numba

    @numba.njit(cache=True)
    def kernel(offsets, neighbors, weights, window):
        n = len(offsets) - 1
        dist = np.full(n, np.inf)
        touched = np.empty(n, dtype=np.int64)

        capacity = len(neighbors) + 1
        heap_dist = np.empty(capacity, dtype=np.float64)
        heap_node = np.empty(capacity, dtype=np.int64)

        counts = np.zeros(n, dtype=np.int64)

        # Pass 1: count membership per source only.
        for source in range(n):
            dist[source] = 0.0
            touched[0] = source
            n_touched = 1
            heap_dist[0] = 0.0
            heap_node[0] = source
            heap_size = 1

            while heap_size > 0:
                d = heap_dist[0]
                node = heap_node[0]
                heap_size -= 1
                if heap_size > 0:
                    heap_dist[0] = heap_dist[heap_size]
                    heap_node[0] = heap_node[heap_size]
                    i = 0
                    while True:
                        left, right, smallest = 2 * i + 1, 2 * i + 2, i
                        if left < heap_size and heap_dist[left] < heap_dist[smallest]:
                            smallest = left
                        if right < heap_size and heap_dist[right] < heap_dist[smallest]:
                            smallest = right
                        if smallest == i:
                            break
                        heap_dist[i], heap_dist[smallest] = heap_dist[smallest], heap_dist[i]
                        heap_node[i], heap_node[smallest] = heap_node[smallest], heap_node[i]
                        i = smallest

                if d > dist[node]:
                    continue
                if d > window:
                    break

                for e in range(offsets[node], offsets[node + 1]):
                    nxt = neighbors[e]
                    nd = d + weights[e]
                    if nd <= window and nd < dist[nxt]:
                        if dist[nxt] == np.inf:
                            touched[n_touched] = nxt
                            n_touched += 1
                        dist[nxt] = nd
                        heap_dist[heap_size] = nd
                        heap_node[heap_size] = nxt
                        j = heap_size
                        heap_size += 1
                        while j > 0:
                            parent = (j - 1) // 2
                            if heap_dist[parent] <= heap_dist[j]:
                                break
                            heap_dist[j], heap_dist[parent] = heap_dist[parent], heap_dist[j]
                            heap_node[j], heap_node[parent] = heap_node[parent], heap_node[j]
                            j = parent

            counts[source] = n_touched
            for i in range(n_touched):
                dist[touched[i]] = np.inf

        mem_offsets = np.zeros(n + 1, dtype=np.int64)
        for source in range(n):
            mem_offsets[source + 1] = mem_offsets[source] + counts[source]
        members = np.empty(mem_offsets[n], dtype=np.int64)

        # Pass 2: same search, this time writing membership into place.
        for source in range(n):
            dist[source] = 0.0
            touched[0] = source
            n_touched = 1
            heap_dist[0] = 0.0
            heap_node[0] = source
            heap_size = 1

            while heap_size > 0:
                d = heap_dist[0]
                node = heap_node[0]
                heap_size -= 1
                if heap_size > 0:
                    heap_dist[0] = heap_dist[heap_size]
                    heap_node[0] = heap_node[heap_size]
                    i = 0
                    while True:
                        left, right, smallest = 2 * i + 1, 2 * i + 2, i
                        if left < heap_size and heap_dist[left] < heap_dist[smallest]:
                            smallest = left
                        if right < heap_size and heap_dist[right] < heap_dist[smallest]:
                            smallest = right
                        if smallest == i:
                            break
                        heap_dist[i], heap_dist[smallest] = heap_dist[smallest], heap_dist[i]
                        heap_node[i], heap_node[smallest] = heap_node[smallest], heap_node[i]
                        i = smallest

                if d > dist[node]:
                    continue
                if d > window:
                    break

                for e in range(offsets[node], offsets[node + 1]):
                    nxt = neighbors[e]
                    nd = d + weights[e]
                    if nd <= window and nd < dist[nxt]:
                        if dist[nxt] == np.inf:
                            touched[n_touched] = nxt
                            n_touched += 1
                        dist[nxt] = nd
                        heap_dist[heap_size] = nd
                        heap_node[heap_size] = nxt
                        j = heap_size
                        heap_size += 1
                        while j > 0:
                            parent = (j - 1) // 2
                            if heap_dist[parent] <= heap_dist[j]:
                                break
                            heap_dist[j], heap_dist[parent] = heap_dist[parent], heap_dist[j]
                            heap_node[j], heap_node[parent] = heap_node[parent], heap_node[j]
                            j = parent

            base = mem_offsets[source]
            for i in range(n_touched):
                members[base + i] = touched[i]
            for i in range(n_touched):
                dist[touched[i]] = np.inf

        return mem_offsets, members

    _window_membership_kernel = kernel
    return kernel


def build_csr_from_edges(
    edge_index: np.ndarray, edge_attr: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR adjacency from an ALREADY-symmetrized edge_index/edge_attr (i.e.
    the format data/build_dataset_from_lab_h5.py (deleted 2026-08-06, deprecated cleanup)'s Data objects already use
    -- both directions present, edge_attr in nm) -- deliberately not
    segclr_db.aggregate.build_csr, which symmetrizes raw CAVE skeleton_edges
    itself and would double-symmetrize (or need un-symmetrizing first) if
    handed a PyG Data's edge_index straight.
    """
    if edge_index.shape[1] == 0:
        return np.zeros(n_nodes + 1, np.int64), np.empty(0, np.int64), np.empty(0, np.float64)

    src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    weights = edge_attr.reshape(-1).astype(np.float64)

    order = np.argsort(src, kind="stable")
    neighbors = dst[order]
    sorted_weights = weights[order]

    offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(offsets, src + 1, 1)
    np.cumsum(offsets, out=offsets)
    return offsets, neighbors, sorted_weights


def window_membership(
    edge_index: np.ndarray, edge_attr: np.ndarray, n_nodes: int, window_nm: float
) -> tuple[np.ndarray, np.ndarray]:
    """(mem_offsets, members): members[mem_offsets[i]:mem_offsets[i+1]] are
    all node indices (including i itself) within window_nm geodesic distance
    of node i, over the SAME edge_index/edge_attr a cached whole-cell
    Data object already carries (see data/build_dataset_from_lab_h5.py (deleted 2026-08-06, deprecated cleanup))."""
    offsets, neighbors, weights = build_csr_from_edges(edge_index, edge_attr, n_nodes)
    kernel = _kernel()
    mem_offsets, members = kernel(offsets, neighbors, weights, float(window_nm))
    return mem_offsets, members


def _window_laplacian_pos_enc(local_edge_index: torch.Tensor, n: int, pos_dim: int) -> torch.Tensor:
    """Per-window Laplacian positional encoding for gnn/graph_transformer.py's
    GraphTransformer, computed on THIS window's own small real adjacency --
    not on a batch-padded (B, N, N) block, deliberately: see
    gnn/graph_transformer.py's module docstring for why batching many
    different-size real windows into one padded matrix before
    eigendecomposing would mix real eigenvectors with spurious ones from the
    zero-adjacency padding block. Runs once per window at extraction time
    instead, same convention this function already uses for local edges.

    Faithful port of ssl_neuron/utils.py::compute_eig_lapl_torch_batch's math
    (self-looped adjacency, descending eigenvalues via flip, indices
    [1:pos_dim+1], zero-pad the tail if the window has fewer than pos_dim
    nontrivial modes) -- just de-batched to one graph. float64 throughout the
    eigendecomposition for numerical stability on the tiny windows this runs
    on (avg 10.7 nodes), cast back to float32 at the end to match `x`.
    """
    adj = torch.zeros(n, n, dtype=torch.float64)
    if local_edge_index.numel():
        adj[local_edge_index[0], local_edge_index[1]] = 1.0
    adj.fill_diagonal_(1.0)  # self-loops, matching the reference's neighbors_to_adjacency_torch

    degree = adj.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = torch.diag(degree.pow(-0.5))
    lap = torch.eye(n, dtype=torch.float64) - d_inv_sqrt @ adj @ d_inv_sqrt

    _, eig_vec = torch.linalg.eigh(lap)
    eig_vec = torch.flip(eig_vec, dims=[1])
    pos_enc = eig_vec[:, 1 : pos_dim + 1]
    if pos_enc.shape[1] < pos_dim:
        pad = torch.zeros(n, pos_dim - pos_enc.shape[1], dtype=torch.float64)
        pos_enc = torch.cat([pos_enc, pad], dim=1)
    return pos_enc.float()


def extract_window_subgraph(
    data, center: int, mem_offsets: np.ndarray, members: np.ndarray, pos_dim: int = DEFAULT_POS_DIM
):
    """One local-neighborhood torch_geometric.data.Data for the window
    around `center` -- node features restricted to the window, edges
    induced among just those nodes (both endpoints in the window),
    remapped to local 0..len(window)-1 indices. `data` is a cached
    whole-cell Data (data/build_dataset_from_lab_h5.py (deleted 2026-08-06, deprecated cleanup)'s output); this
    is called on the fly at train time, not cached to disk per window --
    there'd be one such subgraph per node per cell, far too many .pt files.
    """
    from torch_geometric.data import Data

    lo, hi = mem_offsets[center], mem_offsets[center + 1]
    window_nodes = members[lo:hi]  # (W,) original node indices, W = window size

    n_full = data.x.shape[0]
    in_window = np.zeros(n_full, dtype=bool)
    in_window[window_nodes] = True

    old_to_new = -np.ones(n_full, dtype=np.int64)
    old_to_new[window_nodes] = np.arange(len(window_nodes))

    edge_index_np = data.edge_index.numpy()
    edge_attr_np = data.edge_attr.numpy()
    keep = in_window[edge_index_np[0]] & in_window[edge_index_np[1]]
    e = edge_index_np[:, keep]
    local_edge_index = old_to_new[e]
    local_edge_attr = edge_attr_np[keep]
    local_edge_index_t = torch.from_numpy(local_edge_index).long()

    if not hasattr(data, "pos") or data.pos is None:
        raise AttributeError(
            "extract_window_subgraph requires data.pos (whole-cell xyz coordinates, nm) for "
            "the relative-position node feature -- see data/build_dataset_from_store.py's "
            "Data(..., pos=...); got a cached whole-cell Data object with no pos attribute "
            "(likely built before pos= was added -- rerun data/build_dataset_from_store.py)."
        )
    # Relative position feature, per explicit user direction (2026-08-07): the
    # window's own center node (local index 0, see the center_local_idx
    # comment below) sits at (0,0,0); every other node's feature is its raw
    # xyz coordinate minus the center's -- so this is purely relative/
    # translation-invariant, never an absolute position in the cell. Scaled
    # by REL_POS_SCALE_NM so it sits in a comparable numeric range to the
    # (roughly unit-scale) SegCLR embedding features it gets concatenated
    # with in gnn/graph_transformer.py, rather than raw nm magnitudes
    # (thousands) dominating purely by scale.
    center_pos = data.pos[window_nodes[0]]
    rel_pos = (data.pos[window_nodes] - center_pos).float() / REL_POS_SCALE_NM

    # Dendrite thickness, (W, THICKNESS_DIM). Attached only when the dataset
    # was built with use_thickness=True (data/dataset_windowed.py), which
    # normalizes and NaN-masks it ONCE per cell at load time -- this hot path
    # runs millions of times per epoch and only indexes the prepared tensor.
    # Omitted entirely otherwise, rather than attached as zeros: a Data
    # carrying a silently-all-zero feature is far easier to train against by
    # accident than one where the attribute is simply absent.
    extra = {}
    thickness = getattr(data, "thickness", None)
    if thickness is not None:
        extra["thickness"] = thickness[window_nodes]

    return Data(
        **extra,
        x=data.x[window_nodes],
        edge_index=local_edge_index_t,
        edge_attr=torch.from_numpy(local_edge_attr),
        # Per-node Laplacian positional encoding for
        # gnn/graph_transformer.py::GraphTransformer -- unused by every other
        # architecture in gnn/model.py, but cheap enough (eigh on this
        # window's own tiny adjacency) to always attach rather than making it
        # conditional on which model will consume it later.
        pos_enc=_window_laplacian_pos_enc(local_edge_index_t, len(window_nodes), pos_dim),
        # Relative xyz position, see comment above -- also GraphTransformer-only.
        rel_pos=rel_pos,
        y_levels=data.y_levels,  # the cell's label -- every window from this cell shares it
        # Wrapped as a (1,)-shaped tensor, not a bare python int -- PyG's
        # Batch.from_data_list concatenates tensor graph-level attributes
        # reliably (same convention y_levels already uses); a bare int
        # attribute's batching behavior isn't something to rely on, and
        # root_id needs to survive batching intact for majority-voting
        # per-window predictions back up to a cell-level answer
        # (gnn/metrics.py::majority_vote_by_group in train_gnn.py).
        root_id=torch.tensor([data.root_id], dtype=torch.long),
        window_size=len(window_nodes),
        # window_membership's kernel always visits the source itself first
        # (touched[0] = source, see data/geodesic_window.py's kernel), so
        # members[mem_offsets[center]] == center by construction -- local
        # index 0 is always the window's own center node, no search needed.
        center_local_idx=0,
    )
