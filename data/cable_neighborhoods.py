"""Local neighbourhoods grown to a fixed amount of skeleton, per node.

The unit is a **connected local subgraph**, not a route. A path follows one way
through every fork and discards the other branches; a neighbourhood keeps them.
What is held constant across units is the **total cable length** the subgraph
contains -- the summed edge length of its induced subgraph -- so every unit holds
the same amount of skeleton and its spatial extent varies instead:

    unbranched neurite   40 um of cable reaches ~20 um from the centre
    branchy arbor        40 um of cable is spread over several short branches

That is the deliberate trade. The alternative -- a fixed geodesic *radius* -- is
the project's existing window (data/geodesic_window.py) and holds extent constant
while the amount of skeleton varies. Both are defensible; this module is the
"same amount of stuff" one, and it is what makes a cable budget directly
comparable to a node budget.

Growth is **nearest-first** (bounded Dijkstra from the centre), so a neighbourhood
is a geodesic ball grown until its cable runs out, not a cherry-picked set of
short edges. It stops at the **first node whose edge would overshoot** rather than
skipping that node and continuing -- skipping would keep hunting for short edges
far from the centre and stop being a ball at all. So total cable is <= the budget
and under-fills by at most one edge, the same convention the path arms use.

Skeletons are forests, so a connected set of n nodes induces exactly n-1 edges and
total cable is just the sum of the tree edges the search added. There is no
separate induced-edge pass and no chance of double counting.

Two budgets, one search:

    by cable    accumulate edge length until the next edge would exceed L
    by nodes    take the k geodesically nearest nodes, centre included
"""

from __future__ import annotations

import numpy as np

_kernels = None


def _get_kernels():
    global _kernels
    if _kernels is not None:
        return _kernels

    import numba

    @numba.njit(cache=True)
    def _grow(offsets, neighbors, weights, src, limit_nm, limit_nodes,
              dist, seen, touched, heap_d, heap_v, heap_w, out_nodes, w0):
        """Grow one neighbourhood. Returns (n_nodes, cable_nm, max_radius_nm, w).

        `dist`/`seen` are scratch shared across calls and reset via `touched`, so
        the per-node cost does not depend on the size of the whole cell.
        """
        n_touch = 0
        nh = 0
        # push the centre
        heap_d[0] = 0.0
        heap_v[0] = src
        heap_w[0] = 0.0
        nh = 1
        cable = 0.0
        count = 0
        maxr = 0.0
        w = w0
        while nh > 0:
            # pop min
            top_d = heap_d[0]
            top_v = heap_v[0]
            top_w = heap_w[0]
            nh -= 1
            if nh > 0:
                heap_d[0] = heap_d[nh]
                heap_v[0] = heap_v[nh]
                heap_w[0] = heap_w[nh]
                i = 0
                while True:
                    l = 2 * i + 1
                    r = l + 1
                    s = i
                    if l < nh and heap_d[l] < heap_d[s]:
                        s = l
                    if r < nh and heap_d[r] < heap_d[s]:
                        s = r
                    if s == i:
                        break
                    heap_d[i], heap_d[s] = heap_d[s], heap_d[i]
                    heap_v[i], heap_v[s] = heap_v[s], heap_v[i]
                    heap_w[i], heap_w[s] = heap_w[s], heap_w[i]
                    i = s
            if seen[top_v]:
                continue
            # budget checks happen at admission, so the centre is always in
            if count > 0:
                if limit_nm >= 0.0 and cable + top_w > limit_nm:
                    break
                if limit_nodes >= 0 and count >= limit_nodes:
                    break
            seen[top_v] = True
            touched[n_touch] = top_v
            n_touch += 1
            out_nodes[w] = top_v
            w += 1
            count += 1
            cable += top_w
            if top_d > maxr:
                maxr = top_d
            for e in range(offsets[top_v], offsets[top_v + 1]):
                v = neighbors[e]
                if seen[v]:
                    continue
                nd = top_d + weights[e]
                if nh >= len(heap_d):
                    continue
                heap_d[nh] = nd
                heap_v[nh] = v
                heap_w[nh] = weights[e]
                nh += 1
                j = nh - 1
                while j > 0:
                    par = (j - 1) // 2
                    if heap_d[par] <= heap_d[j]:
                        break
                    heap_d[par], heap_d[j] = heap_d[j], heap_d[par]
                    heap_v[par], heap_v[j] = heap_v[j], heap_v[par]
                    heap_w[par], heap_w[j] = heap_w[j], heap_w[par]
                    j = par
        for i in range(n_touch):
            seen[touched[i]] = False
        return count, cable, maxr, w

    @numba.njit(cache=True)
    def _all(offsets, neighbors, weights, limit_nm, limit_nodes, cap_nodes, heap_cap):
        n = len(offsets) - 1
        out_nodes = np.empty(n * cap_nodes, np.int32)
        out_off = np.zeros(n + 1, np.int64)
        cable = np.empty(n, np.float32)
        radius = np.empty(n, np.float32)
        dist = np.full(n, np.inf)
        seen = np.zeros(n, np.uint8)
        touched = np.empty(n, np.int64)
        heap_d = np.empty(heap_cap, np.float64)
        heap_v = np.empty(heap_cap, np.int64)
        heap_w = np.empty(heap_cap, np.float64)
        w = 0
        for i in range(n):
            if w + cap_nodes > len(out_nodes):
                return out_nodes[:0], out_off, cable, radius, -1
            c, cb, mr, w = _grow(offsets, neighbors, weights, i,
                                 limit_nm, limit_nodes, dist, seen, touched,
                                 heap_d, heap_v, heap_w, out_nodes, w)
            cable[i] = cb
            radius[i] = mr
            out_off[i + 1] = w
        return out_nodes[:w], out_off, cable, radius, 0

    _kernels = (_all,)
    return _kernels


def neighborhoods(offsets, neighbors, weights, *, cable_nm=None, n_nodes=None):
    """One neighbourhood per node.

    Exactly one of ``cable_nm`` / ``n_nodes`` sets the budget. Returns a dict with
    ``members`` / ``offsets`` (ragged, members[offsets[i]:offsets[i+1]] is node
    i's neighbourhood, centre first), ``cable_nm``, ``radius_nm`` (the furthest
    member's geodesic distance) and ``n_members``.
    """
    if (cable_nm is None) == (n_nodes is None):
        raise ValueError("pass exactly one of cable_nm / n_nodes")
    (_all,) = _get_kernels()
    n = len(offsets) - 1
    if n_nodes is not None:
        cap = int(n_nodes)
    else:
        w = weights[weights > 0]
        shortest = float(w.min()) if len(w) else 1.0
        cap = int(min(n, np.ceil(cable_nm / shortest) + 2))
    cap = max(cap, 1)
    heap_cap = max(1024, cap * 8)

    for _ in range(6):
        members, off, cable, radius, status = _all(
            offsets, neighbors, weights,
            -1.0 if cable_nm is None else float(cable_nm),
            -1 if n_nodes is None else int(n_nodes),
            cap, heap_cap)
        if status == 0:
            break
        cap *= 2
        heap_cap *= 2
    else:
        raise RuntimeError("neighbourhood growth kept overflowing its buffers")

    return {
        "members": members,
        "offsets": off,
        "cable_nm": cable,
        "radius_nm": radius,
        "n_members": np.diff(off).astype(np.int32),
    }
