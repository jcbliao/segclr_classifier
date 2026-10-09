"""Full-resolution resampled TEASAR analysis for the 2,209-cell cohort.

Shared statistical definitions; native geometry and nearest-node site mappings.
"""
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from analysis.presynaptic.new_skeletons import skeleton_stats as shared

PRESYNAPTIC_DATABASE = Path('/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/resampled_teasar_2209/native')
NEW_SKELETONS = PRESYNAPTIC_DATABASE / 'originals'
MANIFEST = PRESYNAPTIC_DATABASE / 'manifest.json'
GMM_COMPONENTS = shared.GMM_COMPONENTS
PLOT_PERCENTILES = shared.PLOT_PERCENTILES
MIN_COMPONENT_SITES = 5
CACHE_ROOT = Path(__file__).resolve().parent / 'components_gt5_presynaptic/cache'
parameter_table = shared.parameter_table
plot_edge_length_by_upstream_count = shared.plot_edge_length_by_upstream_count


def _stamp(paths):
    return [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def _json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2)); os.replace(tmp, path)


def ensure_cache(database=PRESYNAPTIC_DATABASE):
    paths = sorted((Path(database) / 'cells').glob('*.npz'))
    if not paths:
        raise FileNotFoundError(database)
    signature = {'version': 1, 'min_component_sites': MIN_COMPONENT_SITES,
                 'files': _stamp(paths)}
    cache = CACHE_ROOT / _digest(signature)
    if (cache / 'complete.json').exists():
        return cache
    (cache / 'cells').mkdir(parents=True, exist_ok=True)
    total = kept = 0
    raw = cache / 'edge_lengths_nm.float32'
    tmp = raw.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        for i, path in enumerate(paths, 1):
            with np.load(path, allow_pickle=False) as cell:
                n = len(cell['new_pos_nm']); edges = cell['new_edges']
                sites, _ = shared.mapped_presynaptic_counts(cell, n)
                nodes = shared.component_node_mask(n, edges, sites, MIN_COMPONENT_SITES)
                lengths = cell['new_edge_length_nm']
                selected = np.asarray(lengths[nodes[edges].all(axis=1)], dtype=np.float32)
                selected.tofile(stream)
                np.savez_compressed(cache / 'cells' / path.name, new_edge_length_nm=selected)
                total += len(lengths); kept += len(selected)
            if i == 1 or i % 100 == 0 or i == len(paths):
                print(f'Component cache: {i:,}/{len(paths):,} cells', flush=True)
    if _stamp(paths) != signature['files']:
        raise RuntimeError('Source files changed while building component cache; rerun')
    os.replace(tmp, raw)
    _json(cache / 'complete.json', {'signature': signature, 'cells': len(paths),
                                  'source_edges': total, 'kept_edges': kept})
    return cache


def load_presynaptic_lengths(database=PRESYNAPTIC_DATABASE):
    cache = ensure_cache(database)
    info = json.loads((cache / 'complete.json').read_text())
    print(f">5-site components: {info['kept_edges']:,} / {info['source_edges']:,} edges; "
          f"{info['cells']:,} cells (persistent cache)", flush=True)
    return np.memmap(cache / 'edge_lengths_nm.float32', dtype=np.float32, mode='r')


def _value_cache(values):
    if isinstance(values, np.memmap) and isinstance(values.filename, (str, Path)):
        path = Path(values.filename)
        if (path.name == 'edge_lengths_nm.float32' and values.offset == 0
                and values.nbytes == path.stat().st_size
                and (path.parent / 'complete.json').exists()):
            return path.parent
    return None


def summarize(values):
    cache = _value_cache(values)
    path = cache / 'summary.csv' if cache else None
    if path and path.exists():
        return pd.read_csv(path, index_col=0)
    result = shared.summarize(values)
    if path:
        result.to_csv(path)
    return result


def streaming_quantiles(values, percentiles, resolution=shared.N_FINE_BINS, log_bins=True):
    cache = _value_cache(values)
    key = _digest([list(percentiles), resolution, log_bins])
    path = cache / f'quantiles-{key}.json' if cache else None
    if path and path.exists():
        return np.asarray(json.loads(path.read_text()))
    result = shared.streaming_quantiles(values, percentiles, resolution, log_bins)
    if path:
        _json(path, result.tolist())
    return result



def plot_distribution_and_gmm(values, bounds, log_space, n_components=GMM_COMPONENTS,
                              family="Gaussian", compare_gaussian=False):
    cache = _value_cache(values)
    key = _digest({"version": 1, "bounds": list(bounds), "log_space": log_space,
                   "n_components": n_components, "family": family,
                   "compare_gaussian": compare_gaussian,
                   "plot_bins": shared.N_PLOT_BINS, "fit_bins": shared.N_FINE_BINS,
                   "implementation": _stamp([Path(shared.__file__)])})
    path = cache / f'distribution-{key}.pkl' if cache else None
    if path and path.exists():
        print('Loading cached histogram and mixture fit', flush=True)
        with path.open('rb') as stream:
            prepared = pickle.load(stream)
    else:
        print('Building histogram and mixture fit (cached after this run)', flush=True)
        bins = (np.geomspace(*bounds, shared.N_PLOT_BINS) if log_space
                else np.linspace(*bounds, shared.N_PLOT_BINS))
        counts = shared.streaming_histogram(values, bins)
        model = shared.fit_histogram_gmm(values, bounds, log_space, n_components, family)
        gaussian = (shared.fit_histogram_gmm(values, bounds, log_space, n_components)
                    if compare_gaussian and family != "Gaussian" else None)
        prepared = (bins, counts, model, log_space, gaussian)
        if path:
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                pickle.dump(prepared, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
    return shared.plot_precomputed_distribution(*prepared)


def plot_distributions_by_cell_type(bounds, database=PRESYNAPTIC_DATABASE, manifest=None):
    return shared.plot_distributions_by_cell_type(bounds, ensure_cache(database), manifest or MANIFEST)


def collect_upstream_length_histograms(bounds, database=PRESYNAPTIC_DATABASE,
                                     originals=NEW_SKELETONS, manifest=None, workers=None):
    cache = ensure_cache(database)
    if manifest is None:
        manifest = MANIFEST
    originals = Path(originals)
    source_paths = sorted((Path(database) / 'cells').glob('*.npz'))
    key = _digest({'bounds': list(bounds), 'originals': _stamp([originals / p.name for p in source_paths]),
                   'metadata': _stamp([Path(database) / 'metadata.json', Path(manifest)]),
                   'implementation': _stamp([Path(shared.__file__)])})
    path = cache / f'upstream-{key}.pkl'
    if path.exists():
        print('Loading cached upstream histograms for >5-site components', flush=True)
        with path.open('rb') as f:
            return pickle.load(f)
    result = shared.collect_upstream_length_histograms(
        bounds, database=database, originals=originals, manifest=manifest, workers=workers,
        min_component_sites=MIN_COMPONENT_SITES)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return result
