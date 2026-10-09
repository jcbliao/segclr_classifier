"""Nested degree-2 contraction with stable original node IDs and cable lengths."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SCALE_WINDOWS = {1: 209, 2: 105, 4: 53, 8: 27, 16: 13, 32: 7}
# Registered native experiment: preserve (K-1)*factor = 256 across scales.
# Scale 32 uses 9 nodes; each finer scale doubles edges and adds one node.
# The full-resolution center is included in K; no cable-length cutoff.
NATIVE_SCALE_WINDOWS = {factor: 256 // factor + 1 for factor in SCALE_WINDOWS}


def contract_degree_two(pos, edges, lengths, original_ids, target_nodes):
    """Keep all degree != 2 nodes, then allocate the remaining node budget.

    Each maximal degree-2 chain receives a proportional share of its interior
    nodes. Samples are spread uniformly along its index sequence. Coordinates
    and original node IDs are never interpolated; contracted edge weights sum
    the original cable lengths. If anchors exceed the budget they all survive.
    Input must be a forest (isolated vertices are supported).
    """
    pos = np.asarray(pos)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    lengths = np.asarray(lengths, np.float64)
    original_ids = np.asarray(original_ids, np.int64)
    n = len(pos)
    if target_nodes < 0 or len(original_ids) != n or len(lengths) != len(edges):
        raise ValueError('invalid target or array lengths')
    if not np.isfinite(lengths).all() or np.any(lengths < 0):
        raise ValueError('invalid cable lengths')
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('edge endpoint outside vertex array')
    adjacency = [[] for _ in range(n)]
    for ei, (u, v) in enumerate(edges):
        adjacency[u].append((int(v), ei))
        adjacency[v].append((int(u), ei))
    degree = np.array([len(a) for a in adjacency])
    if n:
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n,n)).tocsr()
        components = connected_components(graph, directed=False, return_labels=False)
        if len(edges) != n - components:
            raise ValueError('branch-preserving subsampling requires a forest')
    anchors = np.flatnonzero(degree != 2)
    visited = np.zeros(len(edges), bool)
    chains = []
    for start in anchors:
        for neighbor, ei in adjacency[start]:
            if visited[ei]:
                continue
            nodes, arc = [int(start)], [0.0]
            current, edge = neighbor, ei
            while True:
                visited[edge] = True
                nodes.append(current)
                arc.append(arc[-1] + lengths[edge])
                if degree[current] != 2:
                    break
                options = adjacency[current]
                nxt, next_edge = options[0] if options[0][1] != edge else options[1]
                current, edge = nxt, next_edge
            chains.append((np.array(nodes), np.array(arc)))
    if not visited.all():
        raise ValueError('unvisited edges in skeleton')
    capacity = np.array([len(nodes)-2 for nodes, _ in chains], dtype=np.int64)
    budget = max(0, min(n, int(target_nodes)) - len(anchors))
    quotas = np.zeros(len(chains), np.int64)
    if capacity.sum():
        ideal = capacity * (budget / int(capacity.sum()))
        quotas = np.floor(ideal).astype(np.int64)
        remaining = budget - int(quotas.sum())
        order = np.argsort(-(ideal - quotas), kind='stable')
        quotas[order[:remaining]] += 1
    keep = np.zeros(n, bool)
    keep[anchors] = True
    new_edges, new_lengths = [], []
    for (nodes, arc), quota in zip(chains, quotas):
        # Spacing in the node sequence retains distinct nodes even for zero-
        # length edges; the dense input is already approximately uniform.
        selected = np.rint(np.linspace(0, len(nodes)-1, int(quota)+2)).astype(int)
        picked = nodes[selected]
        keep[picked] = True
        new_edges.extend(zip(picked[:-1], picked[1:]))
        new_lengths.extend(np.diff(arc[selected]))
    retained = np.flatnonzero(keep)
    remap = np.full(n, -1, np.int64)
    remap[retained] = np.arange(len(retained))
    reduced_edges = remap[np.asarray(new_edges, np.int64).reshape(-1,2)]
    return dict(pos_nm=pos[retained].astype(np.float32),
                edges=reduced_edges.astype(np.int32),
                edge_length_nm=np.asarray(new_lengths, np.float64),
                original_node_ids=original_ids[retained],
                requested_nodes=np.array(int(target_nodes)),
                mandatory_nodes=np.array(len(anchors)))


def nested_skeletons(pos, edges, lengths, original_ids):
    """Full resolution and nested 1/2 ... 1/32 of the post-cut node count."""
    n = len(pos)
    current = dict(pos_nm=np.asarray(pos,np.float32), edges=np.asarray(edges,np.int32).reshape(-1,2),
                   edge_length_nm=np.asarray(lengths,np.float64),
                   original_node_ids=np.asarray(original_ids,np.int64))
    result = {1: current}
    for factor in list(SCALE_WINDOWS)[1:]:
        current = contract_degree_two(current['pos_nm'],current['edges'],current['edge_length_nm'],
                                      current['original_node_ids'],(n+factor-1)//factor)
        result[factor] = current
    return result
