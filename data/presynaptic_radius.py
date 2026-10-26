"""Observed embedding bags in a bounded geodesic ball on the source skeleton."""
from __future__ import annotations

import heapq
import numpy as np
from scipy.sparse import csr_matrix

FORMAT = 'presynaptic-geodesic-radius-v1'


def radius_windows(pos, edges, observed_ids, centers, anchor_distances, radius_nm,
                   match_cutoff_nm=5000.0, anchor_eligible=None):
    """Snap using the existing presynaptic anchors; measure radius from that node.

    Traverse unobserved skeleton nodes too, but return only observed embeddings.
    A disconnected component contributes nothing, even if spatially nearby.
    The distance from the synapse to its anchor is a separate eligibility cutoff,
    not an extra skeleton edge. No minimum count or nearest-K truncation applies.
    """
    if not np.isfinite(radius_nm) or radius_nm <= 0:
        raise ValueError('radius_nm must be finite and positive')
    pos = np.asarray(pos, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    observed_ids = np.asarray(observed_ids, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.int64)
    distances = np.asarray(anchor_distances, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or not np.isfinite(pos).all():
        raise ValueError('Invalid skeleton coordinates')
    if distances.shape != centers.shape:
        raise ValueError('Anchor arrays differ in shape')
    if (np.any(edges < 0) or np.any(edges >= len(pos)) or
            np.any(observed_ids < 0) or np.any(observed_ids >= len(pos)) or
            len(np.unique(observed_ids)) != len(observed_ids)):
        raise ValueError('Invalid skeleton or observed node IDs')
    # Canonicalize duplicate/reversed edges before CSR construction (which sums
    # duplicate weights). Zero-length edges remain real edges.
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    graph = csr_matrix((np.r_[lengths, lengths],
                        (np.r_[edges[:, 0], edges[:, 1]],
                         np.r_[edges[:, 1], edges[:, 0]])), shape=(len(pos), len(pos)))
    observed = np.zeros(len(pos), bool)
    observed[observed_ids] = True
    valid = (centers >= 0) & (centers < len(pos)) & np.isfinite(distances)
    valid &= distances >= 0
    if anchor_eligible is None:
        valid &= distances <= match_cutoff_nm
    else:
        eligible = np.asarray(anchor_eligible, dtype=bool)
        if eligible.shape != centers.shape:
            raise ValueError('Anchor eligibility has the wrong shape')
        valid &= eligible
    valid[np.flatnonzero(valid)] &= observed[centers[valid]]
    cache = {}
    offsets = [0]
    bags = []
    for center, ok in zip(centers, valid, strict=True):
        if not ok:
            offsets.append(offsets[-1])
            continue
        if center not in cache:
            best = {int(center): 0.0}
            queue = [(0.0, int(center))]
            members = []
            while queue:
                distance, node = heapq.heappop(queue)
                if distance != best[node]:
                    continue
                if observed[node]:
                    members.append(node)
                for p in range(graph.indptr[node], graph.indptr[node + 1]):
                    nxt = int(graph.indices[p])
                    candidate = distance + float(graph.data[p])
                    if candidate <= radius_nm and candidate < best.get(nxt, np.inf):
                        best[nxt] = candidate
                        heapq.heappush(queue, (candidate, nxt))
            cache[center] = np.asarray(members, np.int32)
        bag = cache[center]
        bags.append(bag)
        offsets.append(offsets[-1] + len(bag))
    return (np.asarray(offsets, np.int64),
            np.concatenate(bags) if bags else np.empty(0, np.int32), valid)


def validate_radius_training(args, metadata):
    """Prevent silently loading K10 windows or geometry placeholders as radius data."""
    if (metadata.get('format') != FORMAT or not metadata.get('complete') or
            metadata.get('missing_root_ids') or args.window_nm != metadata.get('window_nm')):
        raise ValueError('Radius database is incomplete or has a different radius')
    if args.dataset != 'presynaptic_cave' or args.num_embeddings is not None:
        raise ValueError('Radius data require presynaptic_cave and --radius-dataset')
    if (args.architecture not in ('mean', 'pointwise_mlp') or args.spatial or
            args.position or args.lpe or args.single_presynaptic_embedding or
            args.presynaptic_memmap_root or args.embedding_augmentation not in (None, 'clean')):
        raise ValueError('Radius bags currently support clean, set-only mean/pointwise MLP models')
