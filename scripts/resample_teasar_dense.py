"""Densify TEASAR edges without deleting vertices or changing cable geometry.

Each edge is split into ceil(length / spacing) equal pieces. Thus spacing is
at most the target (apart from float32 rounding), not an exact uniform grid.
Original short edges remain; radii at inserted nodes are linearly interpolated.
"""
from pathlib import Path
import argparse
import json
import os

import numpy as np


def densify(vertices, edges, radius, spacing=111.0, subdivision='ceil'):
    v = np.asarray(vertices, dtype=np.float64)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    r = np.asarray(radius).reshape(-1)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be positive and finite')
    if v.ndim != 2 or v.shape[1] != 3 or len(r) != len(v):
        raise ValueError('invalid vertex/radius shapes')
    if not np.isfinite(v).all() or (e.size and (e.min() < 0 or e.max() >= len(v))):
        raise ValueError('invalid coordinates or edge endpoints')
    lengths = np.linalg.norm(v[e[:, 1]] - v[e[:, 0]], axis=1)
    if subdivision not in ('ceil', 'round'):
        raise ValueError('subdivision must be ceil or round')
    # np.rint matches Python round: nearest integer, ties to even.
    rounding = np.ceil if subdivision == 'ceil' else np.rint
    pieces = np.maximum(1, rounding(lengths / spacing).astype(np.int64))
    n_new = int((pieces - 1).sum())
    out_v = np.empty((len(v) + n_new, 3), np.float64)
    out_r = np.empty(len(out_v), np.float32)
    out_e = np.empty((int(pieces.sum()), 2), np.int32)
    out_v[:len(v)] = v
    out_r[:len(v)] = r
    node, edge = len(v), 0
    for (a, b), count in zip(e, pieces):
        count = int(count)
        ids = np.arange(node, node + count - 1)
        t = np.arange(1, count) / count
        out_v[ids] = v[a] + t[:, None] * (v[b] - v[a])
        out_r[ids] = r[a] + t * (r[b] - r[a])
        chain = np.concatenate(([a], ids, [b]))
        out_e[edge:edge + count] = np.column_stack((chain[:-1], chain[1:]))
        node += count - 1
        edge += count
    return out_v.astype(np.float32), out_e, out_r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--roots', type=Path, required=True)
    ap.add_argument('--source-dir', type=Path)
    ap.add_argument('--task-id', type=int, default=0)
    ap.add_argument('--num-tasks', type=int, default=1)
    args = ap.parse_args()
    dest_dir = args.out / 'resampled_111nm'
    dest_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for rid in args.roots.read_text().split()[args.task_id::args.num_tasks]:
        dest = dest_dir / f'{rid}.npz'
        if dest.exists():
            continue
        source = (args.source_dir or args.out / 'skeletons') / f'{rid}.npz'
        if not source.exists():
            failed.append(rid)
            continue
        with np.load(source) as raw:
            v, e, r = densify(raw['vertices'], raw['edges'], raw['radius'])
            lengths = np.linalg.norm(v[e[:, 0]].astype(float) - v[e[:, 1]], axis=1)
            # World coordinates are around 1e6 nm: float32 has sub-nm rounding.
            assert not len(lengths) or lengths.max() <= 111.5
            assert len(v) - len(e) == len(raw['vertices']) - len(raw['edges'])
            np.testing.assert_array_equal(v[:len(raw['vertices'])], raw['vertices'].astype(np.float32))
            report = dict(root_id=rid, original_nodes=len(raw['vertices']),
                          nodes=len(v), edges=len(e), spacing_nm=111,
                          edge_percentiles_nm=np.percentile(lengths, [0, 10, 50, 90, 100]).tolist() if len(lengths) else [],
                          radius_method='linear interpolation of original clearance radii')
            partial = dest.with_suffix('.partial.npz')
            np.savez_compressed(partial, root_id=np.array([int(rid)], np.uint64),
                                vertices=v, edges=e, radius=r, spacing_nm=111,
                                original_node_count=len(raw['vertices']))
            os.replace(partial, dest)
            dest.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
    if failed:
        raise RuntimeError(f'Missing raw skeletons: {failed}')


if __name__ == '__main__':
    main()
