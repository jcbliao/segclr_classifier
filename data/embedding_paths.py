"""Centered paths along a skeleton: every distinct route through each node.

The unit here is a **path**, not a window subgraph -- the same per-point regime
the rest of the project works in (see CLAUDE.md's project goal), but with the
local context laid out as a 1-D sequence of embeddings rather than a graph.

A path through a node is not unique once the node sits on or near a branch
point, and rather than pick one, **every route is enumerated**. For node i:

    arms(e)   every maximal route leaving i through out-edge e, taking each
              branch separately, until the budget is spent or a tip is reached
    paths(i)  reverse(a) + [i] + b  for every a in arms(e1), b in arms(e2),
              over every unordered pair of distinct out-edges e1 < e2

A node of degree 1 has only one arm set, so its paths are one-sided; a node of
degree 0 is its own path. Pairs are unordered so a route and its reverse are one
path, not two. The same physical route reappears centered on each of its nodes,
which is intended -- a row is "the context of node i", not "a route".

An arm is **maximal**: it stops only where no further step fits the budget. A
branch whose own edge would overshoot is simply not a route; it does not
truncate the arm beside it.

Two path families, both from the same enumeration:

    by length   each arm accumulates up to L/2 nm
    by nodes    each arm takes up to k//2 nodes

A node-count path has no length cap, deliberately: what geodesic extent k nodes
spans is a measured quantity, not an input, and it is what the notebooks report.

**Enumeration is combinatorial, so count before you materialize.** Paths per
node is the product of two arm counts, and arm count grows with the number of
branch points inside the budget -- benign on a dendrite, explosive on a dense
interneuron axon at 40 um. `count_paths` answers that in one cheap pass with no
allocation; `centered_paths` refuses to run without a per-node cap so a build
can never silently try to write more than the disk holds.
"""

from __future__ import annotations

import numpy as np

#: Returned by the counters when a node's arm or path count exceeds what is
#: worth counting exactly; the build caps long before this, so the exact value
#: past the ceiling is never load-bearing.
COUNT_CEILING = np.int64(1) << 40

_kernels = None


def _get_kernels():
    global _kernels
    if _kernels is not None:
        return _kernels

    import numba

    @numba.njit(cache=True)
    def _order_forest(offsets, neighbors):
        n = len(offsets) - 1
        order = np.empty(n, np.int64)
        parent = np.full(n, -1, np.int64)
        comp = np.full(n, -1, np.int32)
        head = 0
        n_comp = 0
        for root in range(n):
            if comp[root] != -1:
                continue
            comp[root] = n_comp
            order[head] = root
            head += 1
            read = head - 1
            while read < head:
                u = order[read]
                read += 1
                for e in range(offsets[u], offsets[u + 1]):
                    v = neighbors[e]
                    if comp[v] == -1:
                        comp[v] = n_comp
                        parent[v] = u
                        order[head] = v
                        head += 1
            n_comp += 1
        return order, parent, comp, n_comp

    @numba.njit(cache=True)
    def _count_arms(offsets, neighbors, weights, first, src, limit_nm, limit_nodes,
                    st_node, st_prev, st_k, st_acc):
        """Maximal routes leaving `src` through the edge that lands on `first`.

        Explicit-stack DFS; returns the route count, saturating at COUNT_CEILING
        so a pathological arbor cannot spin here forever.
        """
        ceiling = np.int64(1) << 40
        top = 0
        st_node[0] = first
        st_prev[0] = src
        st_k[0] = 1
        st_acc[0] = weights[0] * 0.0  # typed zero
        # recompute the real accumulated weight for the first hop
        for e in range(offsets[src], offsets[src + 1]):
            if neighbors[e] == first:
                st_acc[0] = weights[e]
                break
        top = 1
        count = np.int64(0)
        while top > 0:
            top -= 1
            cur = st_node[top]
            prev = st_prev[top]
            k = st_k[top]
            acc = st_acc[top]
            extended = False
            if (limit_nodes < 0 or k < limit_nodes):
                for e in range(offsets[cur], offsets[cur + 1]):
                    v = neighbors[e]
                    if v == prev:
                        continue
                    w = weights[e]
                    if limit_nm >= 0.0 and acc + w > limit_nm:
                        continue
                    if top >= len(st_node):
                        break
                    st_node[top] = v
                    st_prev[top] = cur
                    st_k[top] = k + 1
                    st_acc[top] = acc + w
                    top += 1
                    extended = True
            if not extended:
                count += 1
                if count >= ceiling:
                    return ceiling
        return count

    @numba.njit(cache=True)
    def _count_paths(offsets, neighbors, weights, limit_nm, limit_nodes, stack_cap):
        """(paths_per_node, arms_per_node) for the whole graph."""
        n = len(offsets) - 1
        ceiling = np.int64(1) << 40
        per_node = np.zeros(n, np.int64)
        arms_total = np.zeros(n, np.int64)
        st_node = np.empty(stack_cap, np.int64)
        st_prev = np.empty(stack_cap, np.int64)
        st_k = np.empty(stack_cap, np.int64)
        st_acc = np.empty(stack_cap, np.float64)
        arm_c = np.empty(64, np.int64)
        for i in range(n):
            deg = offsets[i + 1] - offsets[i]
            if deg == 0:
                per_node[i] = 1
                continue
            if deg > len(arm_c):
                arm_c = np.empty(deg, np.int64)
            na = 0
            for e in range(offsets[i], offsets[i + 1]):
                v = neighbors[e]
                w = weights[e]
                if limit_nm >= 0.0 and w > limit_nm:
                    arm_c[na] = 0
                elif limit_nodes >= 0 and limit_nodes < 1:
                    arm_c[na] = 0
                else:
                    arm_c[na] = _count_arms(offsets, neighbors, weights, v, i,
                                            limit_nm, limit_nodes,
                                            st_node, st_prev, st_k, st_acc)
                na += 1
            tot = np.int64(0)
            for a in range(na):
                arms_total[i] += arm_c[a]
            if deg == 1:
                tot = arm_c[0]
            else:
                for a in range(na):
                    for b in range(a + 1, na):
                        tot += arm_c[a] * arm_c[b]
                        if tot >= ceiling:
                            tot = ceiling
                            break
                    if tot >= ceiling:
                        break
            if tot == 0:
                # No two-sided route exists. Fall back to whatever one-sided
                # arms there are, and only then to the bare node -- otherwise a
                # node whose every arm leaves through one edge would be recorded
                # as contextless when it is not.
                tot = arms_total[i]
            if tot == 0:
                tot = 1
            per_node[i] = tot
        return per_node, arms_total

    @numba.njit(cache=True)
    def _emit_arms(offsets, neighbors, weights, first, src, limit_nm, limit_nodes,
                   cur, accw, it, pushed, arm_nodes, arm_off, arm_nm, base_arm, base_w):
        """Materialise every maximal route leaving `src` through `first`.

        Arms land in arm_nodes[arm_off[k]:arm_off[k+1]] for k in
        [base_arm, base_arm + n). Returns (n_arms, write_ptr), or (-1, -1) if a
        scratch buffer would overflow -- never a truncated answer.
        """
        w0 = 0.0
        for e in range(offsets[src], offsets[src + 1]):
            if neighbors[e] == first:
                w0 = weights[e]
                break
        if limit_nm >= 0.0 and w0 > limit_nm:
            return 0, base_w
        if limit_nodes >= 0 and limit_nodes < 1:
            return 0, base_w

        d = 0
        cur[0] = first
        accw[0] = w0
        it[0] = offsets[first]
        pushed[0] = False
        n = 0
        w = base_w
        while d >= 0:
            prev = src if d == 0 else cur[d - 1]
            advanced = False
            while it[d] < offsets[cur[d] + 1]:
                e = it[d]
                it[d] += 1
                v = neighbors[e]
                if v == prev:
                    continue
                ww = weights[e]
                if limit_nm >= 0.0 and accw[d] + ww > limit_nm:
                    continue
                if limit_nodes >= 0 and d + 2 > limit_nodes:
                    continue
                if d + 1 >= len(cur):
                    return -1, -1
                cur[d + 1] = v
                accw[d + 1] = accw[d] + ww
                it[d + 1] = offsets[v]
                pushed[d + 1] = False
                pushed[d] = True
                d += 1
                advanced = True
                break
            if advanced:
                continue
            if not pushed[d]:
                if w + d + 1 > len(arm_nodes) or base_arm + n + 1 >= len(arm_off):
                    return -1, -1
                for j in range(d + 1):
                    arm_nodes[w] = cur[j]
                    w += 1
                arm_nm[base_arm + n] = accw[d]
                n += 1
                arm_off[base_arm + n] = w
            d -= 1
        return n, w

    @numba.njit(cache=True)
    def _emit_paths(offsets, neighbors, weights, limit_nm, limit_nodes, out_off,
                    out_nodes, out_center, out_nm, depth_cap, arm_cap, arm_node_cap):
        """One row per centred path, for every node, in node order.

        out_off must already be the exclusive prefix sum of the counter's
        per-node path counts times nothing -- it is filled here, so it is passed
        as a writable array of size (total_paths + 1).
        """
        n = len(offsets) - 1
        cur = np.empty(depth_cap, np.int64)
        accw = np.empty(depth_cap, np.float64)
        it = np.empty(depth_cap, np.int64)
        arm_nodes = np.empty(arm_node_cap, np.int64)
        arm_off = np.empty(arm_cap + 1, np.int64)
        arm_nm = np.empty(arm_cap, np.float64)
        edge_lo = np.empty(64, np.int64)
        edge_hi = np.empty(64, np.int64)
        pushed_b = np.zeros(depth_cap, np.uint8)
        p = 0          # path index
        w = 0          # write pointer into out_nodes
        for i in range(n):
            deg = offsets[i + 1] - offsets[i]
            if deg > len(edge_lo):
                edge_lo = np.empty(deg, np.int64)
                edge_hi = np.empty(deg, np.int64)
            arm_off[0] = 0
            n_arms = 0
            aw = 0
            ok = True
            for a in range(deg):
                e = offsets[i] + a
                edge_lo[a] = n_arms
                got, aw2 = _emit_arms(offsets, neighbors, weights, neighbors[e], i,
                                      limit_nm, limit_nodes, cur, accw, it, pushed_b,
                                      arm_nodes, arm_off, arm_nm, n_arms, aw)
                if got < 0:
                    ok = False
                    break
                n_arms += got
                aw = aw2
                edge_hi[a] = n_arms
            if not ok:
                return -1, -1

            emitted = 0
            # two-sided: every arm of one edge against every arm of another
            for a in range(deg):
                for b in range(a + 1, deg):
                    for ka in range(edge_lo[a], edge_hi[a]):
                        la = arm_off[ka + 1] - arm_off[ka]
                        for kb in range(edge_lo[b], edge_hi[b]):
                            lb = arm_off[kb + 1] - arm_off[kb]
                            if w + la + 1 + lb > len(out_nodes):
                                return -1, -1
                            for j in range(la - 1, -1, -1):
                                out_nodes[w] = arm_nodes[arm_off[ka] + j]
                                w += 1
                            out_center[p] = la
                            out_nodes[w] = i
                            w += 1
                            for j in range(lb):
                                out_nodes[w] = arm_nodes[arm_off[kb] + j]
                                w += 1
                            out_nm[p] = arm_nm[ka] + arm_nm[kb]
                            p += 1
                            out_off[p] = w
                            emitted += 1
            if emitted == 0 and n_arms > 0:
                # one-sided fallback (degree 1, or every arm on a single edge)
                for k in range(n_arms):
                    lk = arm_off[k + 1] - arm_off[k]
                    if w + lk + 1 > len(out_nodes):
                        return -1, -1
                    out_center[p] = 0
                    out_nodes[w] = i
                    w += 1
                    for j in range(lk):
                        out_nodes[w] = arm_nodes[arm_off[k] + j]
                        w += 1
                    out_nm[p] = arm_nm[k]
                    p += 1
                    out_off[p] = w
                    emitted += 1
            if emitted == 0:
                if w + 1 > len(out_nodes):
                    return -1, -1
                out_center[p] = 0
                out_nodes[w] = i
                w += 1
                out_nm[p] = 0.0
                p += 1
                out_off[p] = w
        return p, w

    _kernels = (_order_forest, _count_paths, _emit_paths)
    return _kernels


def forest_order(offsets, neighbors):
    order_forest, _, _ = _get_kernels()
    order, parent, comp, n_comp = order_forest(offsets, neighbors)
    return order, parent, comp, int(n_comp)


def assert_forest(offsets, neighbors):
    """Refuse a graph carrying a cycle.

    Arm enumeration walks simple routes by never stepping back to the node it
    came from, which is only equivalent to "simple path" on a forest. On a cycle
    it would loop until the budget ran out and emit routes that revisit nodes --
    wrong, and wrong quietly.
    """
    n_nodes = len(offsets) - 1
    n_edges = len(neighbors) // 2
    _order, _parent, _comp, n_comp = forest_order(offsets, neighbors)
    if n_edges != n_nodes - n_comp:
        raise ValueError(
            f"not a forest: {n_nodes} nodes, {n_edges} undirected edges, "
            f"{n_comp} components (a forest would have {n_nodes - n_comp})"
        )
    return n_comp


def _stack_cap(offsets, weights, half_nm, half_nodes):
    n_nodes = len(offsets) - 1
    if half_nodes is not None:
        depth = int(half_nodes) + 2
    else:
        w = weights[weights > 0]
        shortest = float(w.min()) if len(w) else 1.0
        depth = int(min(n_nodes, np.ceil(half_nm / shortest))) + 2
    # DFS frontier is bounded by depth x max branching; 8 is generous for a
    # skeleton (degree is almost always <= 3) and the kernel bounds-checks anyway.
    return max(1024, depth * 8)


def count_paths(offsets, neighbors, weights, *, half_nm=None, half_nodes=None):
    """(paths_per_node, arms_per_node) without allocating any path.

    Run this before a build: the totals decide whether a config is storable at
    all, and which nodes need capping.
    """
    if (half_nm is None) == (half_nodes is None):
        raise ValueError("pass exactly one of half_nm / half_nodes")
    _, counter, _ = _get_kernels()
    return counter(
        offsets, neighbors, weights,
        -1.0 if half_nm is None else float(half_nm),
        -1 if half_nodes is None else int(half_nodes),
        _stack_cap(offsets, weights, half_nm, half_nodes),
    )


def centered_paths(offsets, neighbors, weights, *, half_nm=None, half_nodes=None,
                   per_node=None, arms=None):
    """Materialise every centred path. Returns a dict of flat arrays.

    ``path_nodes[path_offsets[k]:path_offsets[k+1]]`` is path k in walk order,
    ``center_at[k]`` is the centre's position within it, and ``center_node[k]``
    is the node it is centred on.
    """
    if (half_nm is None) == (half_nodes is None):
        raise ValueError("pass exactly one of half_nm / half_nodes")
    _, _, emit = _get_kernels()
    if per_node is None or arms is None:
        per_node, arms = count_paths(offsets, neighbors, weights,
                                     half_nm=half_nm, half_nodes=half_nodes)

    n_nodes = len(offsets) - 1
    total_paths = int(per_node.sum())
    if half_nodes is not None:
        depth = int(half_nodes) + 2
        max_len = 2 * int(half_nodes) + 1
    else:
        w = weights[weights > 0]
        shortest = float(w.min()) if len(w) else 1.0
        depth = int(min(n_nodes, np.ceil(half_nm / shortest))) + 2
        max_len = 2 * depth + 1
    # The node buffer cannot be sized exactly without walking twice, and sizing
    # it from max_len is unsafe: for a length budget, max_len derives from the
    # *shortest* edge, so one near-zero edge would demand a huge allocation. So
    # start from a typical path length and grow on overflow -- the kernel
    # reports overflow rather than truncating, which is what makes this safe.
    out_off = np.zeros(total_paths + 1, np.int64)
    out_center = np.empty(total_paths, np.int64)
    out_nm = np.empty(total_paths, np.float64)
    arm_cap = max(1024, int(arms.max()) * 2 + 8)

    typical = min(max_len, 64)
    total_nodes = max(1024, total_paths * typical)
    for _attempt in range(8):
        out_nodes = np.empty(total_nodes, np.int64)
        p, w = emit(offsets, neighbors, weights,
                    -1.0 if half_nm is None else float(half_nm),
                    -1 if half_nodes is None else int(half_nodes),
                    out_off, out_nodes, out_center, out_nm,
                    depth, arm_cap, arm_cap * (depth + 1))
        if p >= 0:
            break
        total_nodes *= 2
        arm_cap *= 2
    else:
        raise RuntimeError("path emission kept overflowing its scratch buffers")
    if p != total_paths:
        raise RuntimeError(f"emitted {p} paths, counter said {total_paths}")

    return {
        "path_nodes": out_nodes[:w].astype(np.int32),
        "path_offsets": out_off[:p + 1],
        "center_at": out_center[:p].astype(np.int32),
        "geodesic_nm": out_nm[:p].astype(np.float32),
    }
