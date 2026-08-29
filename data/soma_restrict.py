"""Drop the perisomatic ball from a skeleton, and see what it breaks into.

Nodes within ``radius_nm`` of the cell's nucleus are removed. The point is not
that the soma is uninteresting but that it is *shared*: every neurite converges
there, so any path or window overlapping the soma mixes branches that are
otherwise far apart geodesically, and the resulting sequence says more about
where the soma is than about the process it came from.

Removing the ball **disconnects the skeleton**, and that is the intended
outcome, not a side effect to repair: the primary neurites are joined only
through the soma, so cutting it leaves one component per surviving primary
neurite (plus fragments wherever an arbor re-enters the ball). Every downstream
path is confined to a single component, so a path never jumps between two
neurites that only ever met at the soma.

Distance is **Euclidean to the nucleus centroid**, not geodesic along the
skeleton. The nucleus is a point in the volume, not a skeleton node -- there is
no geodesic distance to it -- and a ball in space is what "within 15 um of the
nucleus" means. A geodesic cut from the nearest node would instead follow cable,
which on a coiled proximal dendrite reaches much further out in space than the
radius suggests.

The nucleus position comes from the store's own ``cells`` dimension
(``soma_x_nm`` / ``soma_y_nm`` / ``soma_z_nm``, with ``nucleus_id``), keyed by
root_id -- so this needs no CAVE call and no token, and it is an id-based join
rather than a spatial guess.
"""

from __future__ import annotations

import numpy as np

#: The perisomatic radius this project cuts at. The output directory is derived
#: from it (data/build_embedding_paths.py::out_for), so changing this cannot
#: quietly mix two radii into one database.
DEFAULT_SOMA_RADIUS_NM = 5_000.0


def nucleus_positions(store_root, root_ids=None, dataset="microns"):
    """{root_id: (x, y, z) in nm} from the store's cells dimension.

    Cells whose soma position is null are omitted rather than defaulted -- a
    missing nucleus must shrink the dataset visibly, not silently place the
    ball at the origin and delete a different part of the cell.
    """
    from segclr_db.database import SegCLRDatabase

    db = SegCLRDatabase(root=str(store_root), dataset=dataset)
    cells = db.get_cells(root_ids=root_ids)
    cols = ("soma_x_nm", "soma_y_nm", "soma_z_nm")
    missing = [c for c in cols if c not in cells.columns]
    if missing:
        raise KeyError(f"cells dimension is missing {missing}; got {list(cells.columns)}")

    out = {}
    for rid, x, y, z in zip(cells["root_id"], cells[cols[0]], cells[cols[1]], cells[cols[2]]):
        if x is None or y is None or z is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        out[int(rid)] = (float(x), float(y), float(z))
    return out


def restrict(pos, edge_index, edge_attr, nucleus_xyz, radius_nm=DEFAULT_SOMA_RADIUS_NM):
    """Remove nodes within ``radius_nm`` of ``nucleus_xyz``.

    ``nucleus_xyz=None`` means **no cut**: every node is kept and the cell enters
    the database whole. That is the deliberate treatment for the cells the store
    has no nucleus position for -- guessing a centre, or dropping the cell, are
    both worse than leaving it uncut, and ``cut_applied`` records which happened
    so the two populations are never silently pooled.

    Returns a dict with the keep mask, the surviving graph reindexed to
    ``0..n_kept-1``, per-node component labels, and the counts worth recording.
    """
    pos = np.asarray(pos, np.float64)
    if nucleus_xyz is None:
        d = np.full(len(pos), np.nan)
        keep = np.ones(len(pos), bool)
        cut_applied = False
    else:
        d = np.linalg.norm(pos - np.asarray(nucleus_xyz, np.float64)[None, :], axis=1)
        keep = d > float(radius_nm)
        cut_applied = True

    n_old = len(pos)
    remap = np.full(n_old, -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)

    ei = np.asarray(edge_index, np.int64)
    ea = np.asarray(edge_attr, np.float64).reshape(-1)
    if ei.shape[1]:
        both = keep[ei[0]] & keep[ei[1]]
        ei = remap[ei[:, both]]
        ea = ea[both]
    else:
        ei = np.zeros((2, 0), np.int64)
        ea = np.zeros(0, np.float64)

    from data.embedding_paths import forest_order
    from data.geodesic_window import build_csr_from_edges

    n_kept = int(keep.sum())
    offsets, neighbors, weights = build_csr_from_edges(ei, ea, n_kept)
    _order, _parent, comp, n_comp = forest_order(offsets, neighbors)

    return {
        "keep": keep,
        "cut_applied": cut_applied,
        "dist_to_nucleus_nm": d.astype(np.float32),
        "edge_index": ei,
        "edge_attr": ea.astype(np.float32),
        "csr": (offsets, neighbors, weights),
        "component": comp,
        "n_components": int(n_comp),
        "n_nodes_before": n_old,
        "n_nodes_after": n_kept,
    }
