"""Parallel, resumable CAVE/TEASAR cable comparison; all coordinates are nm.

Use prepare -> run (Slurm array or process pool) -> summarize. No network calls.
Distances are exact to target segments; integration along source cable uses
length-weighted midpoint quadrature, independent of original vertex density.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/orcd/scratch/orcd/013/jcbliao')
DEFAULT_OUT = BASE / 'skeleton_geometry_comparison'
VERSION = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stamp(path):
    path = Path(path)
    info = path.stat()
    return [str(path.resolve()), info.st_size, info.st_mtime_ns]


def geometry(vertices, edges):
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_edges = np.asarray(edges)
    if raw_edges.size and (not np.issubdtype(raw_edges.dtype, np.integer)):
        raise ValueError('Edge indices must be integers')
    edges = raw_edges.astype(np.int64).reshape(-1, 2)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Expected finite Nx3 coordinates in nm')
    if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
        raise ValueError('Edge index out of range')
    # Normalize undirected edges so duplicate storage does not double the cable.
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    return vertices, edges, lengths


def subdivide(vertices, edges, lengths, step, midpoint):
    """Equal subdivisions within each edge; return points and cable weights."""
    positive = lengths > 0
    edges, lengths = edges[positive], lengths[positive]
    count = np.maximum(1, np.ceil(lengths / step).astype(np.int64))
    edge_ids = np.repeat(np.arange(len(edges)), count)
    starts = np.repeat(np.cumsum(count) - count, count)
    fraction = (np.arange(len(edge_ids)) - starts + (0.5 if midpoint else 0)) / count[edge_ids]
    origin = vertices[edges[edge_ids, 0]]
    delta = vertices[edges[edge_ids, 1]] - origin
    points = origin + delta * fraction[:, None]
    weights = lengths[edge_ids] / count[edge_ids]
    if midpoint:
        return points, weights
    return points, points + delta / count[edge_ids, None]


class SegmentIndex:
    """Exact nearest-segment search with a KD tree of short-segment midpoints.

    Unvisited segment distances are bounded below by their midpoint distance
    minus the maximum half-length. Expand k until that bound exceeds the best
    projection distance. Limit query matrix size to bound temporary memory.
    """
    def __init__(self, vertices, edges, lengths, max_segment_nm=1000):
        if not np.any(lengths > 0):
            raise ValueError('Skeleton has no positive-length cable')
        self.start, end = subdivide(vertices, edges, lengths, max_segment_nm, False)
        self.delta = end - self.start
        self.length2 = np.einsum('ij,ij->i', self.delta, self.delta)
        self.half_max = np.sqrt(self.length2.max()) / 2
        self.tree = cKDTree((self.start + end) / 2)

    def distances(self, points):
        result = np.full(len(points), np.inf)
        remaining = np.arange(len(points))
        k = min(4, len(self.start))
        while len(remaining):
            pending = []
            batch = max(1, 200_000 // k)
            for offset in range(0, len(remaining), batch):
                ids = remaining[offset:offset + batch]
                midpoint_d, nearest = self.tree.query(points[ids], k=k, workers=1)
                midpoint_d = midpoint_d.reshape(len(ids), k)
                nearest = nearest.reshape(len(ids), k)
                relative = points[ids, None, :] - self.start[nearest]
                fraction = np.clip(np.einsum('ijk,ijk->ij', relative, self.delta[nearest]) /
                                   self.length2[nearest], 0, 1)
                residual = relative - fraction[..., None] * self.delta[nearest]
                best = np.sqrt(np.einsum('ijk,ijk->ij', residual, residual).min(axis=1))
                result[ids] = np.minimum(result[ids], best)
                if k < len(self.start):
                    unresolved = midpoint_d[:, -1] - self.half_max < result[ids]
                    pending.append(ids[unresolved])
            remaining = np.concatenate(pending) if pending else np.empty(0, dtype=int)
            k = min(2 * k, len(self.start))
        return result


def topology(vertices, edges, lengths):
    n = len(vertices)
    degree = np.bincount(edges.ravel(), minlength=n)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    components = connected_components(graph, directed=False, return_labels=False) if n else 0
    return dict(nodes=n, edges=len(edges), cable_um=float(lengths.sum() / 1000),
                components=int(components), isolated_nodes=int((degree == 0).sum()),
                endpoints=int((degree == 1).sum()), branch_nodes=int((degree >= 3).sum()),
                cycle_rank=int(len(edges) - n + components))


def directed(source, target, spacing, tolerances):
    points, weights = subdivide(*source, spacing, True)
    distances = SegmentIndex(*target).distances(points)
    if not len(points):
        raise ValueError('Skeleton has no positive-length cable')
    total = weights.sum()
    order = np.argsort(distances)
    cumulative = np.cumsum(weights[order]) / total
    p95 = distances[order[min(np.searchsorted(cumulative, .95), len(order) - 1)]]
    coverage = [float(weights[distances <= t].sum() / total) for t in tolerances]
    # Bounds at each tolerance follow from the 1-Lipschitz distance function:
    # every point in a source subdivision is within weight/2 of its midpoint.
    lower = [float(weights[distances + weights / 2 <= t].sum() / total) for t in tolerances]
    upper = [float(weights[distances - weights / 2 <= t].sum() / total) for t in tolerances]
    return dict(mean_um=float(np.dot(weights, distances) / total / 1000),
                p95_um=float(p95 / 1000), sampled_max_um=float(distances.max() / 1000),
                samples=len(points), coverage=coverage, coverage_lower=lower, coverage_upper=upper)


def compare(cave, teasar, spacing, tolerances):
    c = directed(cave, teasar, spacing, tolerances)
    t = directed(teasar, cave, spacing, tolerances)
    metrics = dict(symmetric_mean_um=(c['mean_um'] + t['mean_um']) / 2,
                   mean_quadrature_error_bound_um=spacing / 2000)
    for name, value in [('cave', c), ('teasar', t)]:
        metrics.update({f'{name}_{key}': v for key, v in value.items() if not key.startswith('coverage')})
    for i, tau in enumerate(tolerances):
        suffix = f'{tau / 1000:g}um'
        precision, recall = t['coverage'][i], c['coverage'][i]
        metrics[f'precision_{suffix}'] = precision
        metrics[f'recall_{suffix}'] = recall
        metrics[f'f1_{suffix}'] = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        for direction, value in [('precision', t), ('recall', c)]:
            for bound in ('lower', 'upper'):
                metrics[f'{direction}_{bound}_{suffix}'] = value[f'coverage_{bound}'][i]
    for name, skeleton in [('cave', cave), ('teasar', teasar)]:
        metrics.update({f'{name}_{k}': v for k, v in topology(*skeleton).items()})
    metrics['teasar_cave_cable_ratio'] = metrics['teasar_cable_um'] / metrics['cave_cable_um']
    return metrics


def prepare(args):
    """Freeze the joined cohort and export CAVE once, avoiding per-worker scans."""
    import lance
    rows = list(csv.DictReader(args.comparison.open(newline='')))
    if len({r['root_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate roots in comparison CSV')
    routes = {int(r['root_id']): r for r in json.loads(args.routes.read_text())}
    cells, excluded = [], []
    for row in rows:
        rid = int(row['root_id'])
        if not row['casey_cell_id'] or not row['casey_root_id']:
            excluded.append(dict(root_id=rid, reason='no_casey_match'))
            continue
        route = routes.get(rid)
        path = (Path(route['source']) if route and route.get('source') else
                Path(route['base']) / 'skeletons' / f'{rid}.npz' if route else None)
        if path is None or not path.is_file():
            excluded.append(dict(root_id=rid, reason='missing_teasar'))
            continue
        cells.append(dict(row, root_id=rid, teasar_path=str(path.resolve()),
                          cave_path=str((args.output / 'cave' / f'{rid}.npz').resolve())))
    args.output.mkdir(parents=True, exist_ok=True)
    nodes_ds = lance.dataset(str(args.store / 'skeletons/skeleton_nodes.lance'))
    edges_ds = lance.dataset(str(args.store / 'skeletons/skeleton_edges.lance'))
    retained = []
    for start in range(0, len(cells), 128):
        batch = cells[start:start + 128]
        clause = 'root_id IN (' + ','.join(str(r['root_id']) for r in batch) + ')'
        nodes = nodes_ds.to_table(columns=['root_id', 'node_id', 'x_nm', 'y_nm', 'z_nm'],
                                 filter=clause, use_scalar_index=False).to_pandas()
        edges = edges_ds.to_table(columns=['root_id', 'src', 'dst'], filter=clause,
                                 use_scalar_index=False).to_pandas()
        groups = {int(rid): group for rid, group in edges.groupby('root_id', sort=False)}
        ngroups = {int(rid): group for rid, group in nodes.groupby('root_id', sort=False)}
        for cell in batch:
            rid = cell['root_id']
            if rid not in ngroups or rid not in groups:
                excluded.append(dict(root_id=rid, reason='missing_cave_cable'))
                continue
            group = ngroups[rid].sort_values('node_id')
            node_ids = group.node_id.to_numpy()
            if len(np.unique(node_ids)) != len(node_ids):
                raise ValueError(f'Duplicate CAVE node IDs for {rid}')
            edge_ids = groups[rid][['src', 'dst']].to_numpy()
            remapped = np.searchsorted(node_ids, edge_ids)
            if (remapped >= len(node_ids)).any() or not np.array_equal(node_ids[remapped], edge_ids):
                raise ValueError(f'CAVE edges reference missing nodes for {rid}')
            dest = Path(cell['cave_path'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f'{dest.name}.tmp.{os.getpid()}')
            with tmp.open('wb') as handle:
                np.savez(handle, root_id=rid, vertices=group[['x_nm', 'y_nm', 'z_nm']].to_numpy(), edges=remapped)
            tmp.replace(dest)
            retained.append(cell)
        print(f'Prepared {min(start + 128, len(cells))}/{len(cells)} cells', flush=True)
    atomic_json(args.output / 'cohort.json', dict(version=VERSION, comparison=stamp(args.comparison),
                routes=stamp(args.routes), store=str(args.store.resolve()),
                input_cells=len(rows), cells=retained, excluded=excluded,
                scope='full skeletons; no soma cut, pruning, registration, or confidence threshold'))
    print(f'Cohort: {len(retained)} paired cells, {len(excluded)} excluded', flush=True)


def load(path, rid):
    with np.load(path, allow_pickle=False) as z:
        if 'root_id' in z and int(z['root_id'].item()) != rid:
            raise ValueError(f'Wrong internal root ID: {path}')
        return geometry(z['vertices'], z['edges'])


def run_cell(task):
    cell, output, spacing, tolerances, retry = task
    rid = cell['root_id']
    dest = Path(output) / 'cells' / f'{rid}.json'
    try:
        signature = dict(version=VERSION, cave=stamp(cell['cave_path']), teasar=stamp(cell['teasar_path']),
                         spacing_nm=spacing, tolerances_nm=tolerances)
        if dest.exists():
            saved = json.loads(dest.read_text())
            if saved.get('signature') == signature and (saved['status'] == 'ok' or not retry):
                return rid, saved['status'], True
        started = time.monotonic()
        metrics = compare(load(cell['cave_path'], rid), load(cell['teasar_path'], rid), spacing, tolerances)
        result = dict(cell, status='ok', signature=signature, seconds=time.monotonic() - started, **metrics)
    except Exception as exc:
        result = dict(cell, status='error', error=f'{type(exc).__name__}: {exc}',
                      signature=locals().get('signature'))
    atomic_json(dest, result)
    return rid, result['status'], False


def run(args):
    cohort = json.loads((args.output / 'cohort.json').read_text())
    cells = sorted(cohort['cells'], key=lambda c: c['root_id'])[args.task_id::args.num_tasks]
    if args.limit is not None:
        cells = cells[:args.limit]
    tasks = [(c, str(args.output), args.spacing_nm, args.tolerances_nm, args.retry_errors) for c in cells]
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_cell, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rid, status, cached = future.result()
            failures += status != 'ok'
            print(f'{i}/{len(tasks)} {rid} {status}' + (' cached' if cached else ''), flush=True)
    if failures:
        raise SystemExit(f'{failures} cells failed; inspect per-cell JSON and rerun with --retry-errors')


def summarize(args):
    cohort = json.loads((args.output / 'cohort.json').read_text())
    rows, missing, stale = [], [], []
    for cell in cohort['cells']:
        path = args.output / 'cells' / f"{cell['root_id']}.json"
        if not path.exists():
            missing.append(cell['root_id'])
            continue
        row = json.loads(path.read_text())
        expected = dict(version=VERSION, cave=stamp(cell['cave_path']), teasar=stamp(cell['teasar_path']),
                        spacing_nm=args.spacing_nm, tolerances_nm=args.tolerances_nm)
        if row.get('signature') != expected:
            stale.append(cell['root_id'])
            continue
        rows.append({k: v for k, v in row.items() if k != 'signature'})
    keys = sorted({k for r in rows for k in r})
    if rows:
        with (args.output / 'metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    good = [r for r in rows if r['status'] == 'ok']
    metrics = ['symmetric_mean_um', 'teasar_cave_cable_ratio'] + [f'f1_{t / 1000:g}um' for t in args.tolerances_nm]
    report = dict(cohort_cells=len(cohort['cells']), completed=len(good),
                  errors=[r['root_id'] for r in rows if r['status'] != 'ok'], missing=missing, stale=stale,
                  excluded=cohort['excluded'], spacing_nm=args.spacing_nm,
                  median_per_cell={m: float(np.median([r[m] for r in good])) for m in metrics} if good else {})
    atomic_json(args.output / 'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'excluded'}, indent=2), flush=True)
    if missing or stale or report['errors']:
        raise SystemExit('Incomplete comparison; see summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--comparison', type=Path, default=ROOT / 'data/v1718_extended_axon_neurons_casey_comparison.csv')
    parser.add_argument('--routes', type=Path, default=BASE / 'skeletons/segclr_registered_teasar_111nm_20260911/routes.json')
    parser.add_argument('--store', type=Path, default=Path('/orcd/compute/sdorkenw/001/segclr-db/microns'))
    parser.add_argument('--spacing-nm', type=float, default=250)
    parser.add_argument('--tolerances-nm', type=float, nargs='+', default=[250., 500., 1000., 2000.])
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--num-tasks', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    if (not np.isfinite(args.spacing_nm) or args.spacing_nm <= 0 or
        any(not np.isfinite(t) or t <= 0 for t in args.tolerances_nm) or
        args.workers < 1 or args.num_tasks < 1 or not 0 <= args.task_id < args.num_tasks or
        (args.limit is not None and args.limit < 1)):
        parser.error('Invalid spacing, tolerance, worker count, shard, or limit')
    dict(prepare=prepare, run=run, summarize=summarize)[args.mode](args)


if __name__ == '__main__':
    main()
