"""Statistics and plots for presynaptic-derived TEASAR skeleton edge lengths."""

import json
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from time import monotonic
from pathlib import Path
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import cauchy, norm, t


PRESYNAPTIC_DATABASE = Path("/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10")
CHUNK_EDGES = 2_000_000
PLOT_PERCENTILES = (0.5, 99.5)
N_PLOT_BINS = 70
N_FINE_BINS = 20_000
GMM_COMPONENTS = 3
STUDENT_T_DF_BOUNDS = (0.05, 10_000.0)


def presynaptic_cell_paths(database):
    """Select the original database cohort, excluding explicitly discarded roots.

    Later CAVE-only additions can share the cells directory without carrying
    the TEASAR geometry used by these analyses.
    """
    database = Path(database)
    metadata_path = database / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    excluded = set(map(int, metadata.get("discarded_without_current_embeddings_root_ids", [])))
    return [p for p in sorted((database / "cells").glob("*.npz")) if int(p.stem) not in excluded]


def chunks(values, chunk_edges=CHUNK_EDGES):
    for start in range(0, len(values), chunk_edges):
        yield np.asarray(values[start : start + chunk_edges])


def load_presynaptic_lengths(database=PRESYNAPTIC_DATABASE):
    """Read each derived cell's edges once, after the database's 5 um soma cut.

    Only new_edge_length_nm contributes, not CAVE edges or repeated windows.
    Rebuild temporary disk backing on each load to avoid stale global caches.
    """
    cell_dir = Path(database) / "cells"
    paths = presynaptic_cell_paths(database)
    if not paths:
        raise FileNotFoundError(f"No presynaptic derived skeletons found: {cell_dir}")
    with TemporaryFile() as backing:
        for path in paths:
            with np.load(path, allow_pickle=False) as cell:
                lengths = np.asarray(cell["new_edge_length_nm"], dtype=np.float32)
            if lengths.ndim != 1:
                raise ValueError(f"Expected one-dimensional edge lengths: {path}")
            lengths.tofile(backing)
        if backing.tell() == 0:
            raise ValueError(f"No presynaptic derived skeleton edges found: {cell_dir}")
        backing.flush()
        # The mapping retains backing storage after the file handle closes.
        return np.memmap(backing, dtype=np.float32, mode="r")


def streaming_histogram(values, bins):
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for chunk in chunks(values):
        finite = chunk[np.isfinite(chunk) & (chunk > 0)]
        counts += np.histogram(finite, bins=bins)[0]
    return counts


def streaming_quantiles(values, percentiles, resolution=N_FINE_BINS, log_bins=True):
    lo = np.inf
    hi = -np.inf
    for chunk in chunks(values):
        x = chunk[np.isfinite(chunk) & (chunk > 0)]
        if len(x):
            lo = min(lo, float(x.min()))
            hi = max(hi, float(x.max()))
    if not np.isfinite(lo) or hi <= lo:
        raise ValueError("No positive finite edge lengths found")
    bins = np.geomspace(lo, hi, resolution + 1) if log_bins else np.linspace(lo, hi, resolution + 1)
    counts = streaming_histogram(values, bins)
    cumulative = counts.cumsum()
    targets = np.asarray(percentiles) / 100 * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return bins[np.minimum(indices, resolution - 1)]


def summarize(values):
    qs = [1, 5, 25, 50, 75, 95, 99]
    quantiles = streaming_quantiles(values, qs)
    n = 0
    total = total2 = 0.0
    for chunk in chunks(values):
        x = chunk[np.isfinite(chunk) & (chunk > 0)].astype(np.float64)
        n += len(x)
        total += x.sum()
        total2 += np.square(x).sum()
    mean = total / n
    row = {"edges": n, "mean_nm": mean, "std_nm": np.sqrt(max(0, total2 / n - mean**2))}
    row.update({f"p{q}_nm": value for q, value in zip(qs, quantiles)})
    return pd.DataFrame([row], index=["Presynaptic-derived TEASAR"])


def _weighted_gmm(x, sample_weight, n_components=GMM_COMPONENTS, max_iter=200, tol=1e-8):
    sample_weight = sample_weight.astype(np.float64)
    cumulative = np.cumsum(sample_weight) / sample_weight.sum()
    means = np.interp(np.linspace(0.2, 0.8, n_components), cumulative, x)
    span = x[-1] - x[0]
    sigmas = np.full(n_components, span / (2 * n_components))
    weights = np.full(n_components, 1 / n_components)
    previous = -np.inf
    for iteration in range(max_iter):
        joint = (np.log(weights)[None, :] - np.log(sigmas)[None, :]
                 - 0.5 * ((x[:, None] - means[None, :]) / sigmas[None, :]) ** 2
                 - 0.5 * np.log(2 * np.pi))
        normalizer = logsumexp(joint, axis=1)
        responsibility = np.exp(joint - normalizer[:, None]) * sample_weight[:, None]
        nk = responsibility.sum(axis=0)
        weights = nk / nk.sum()
        means = (responsibility * x[:, None]).sum(axis=0) / nk
        variances = (responsibility * (x[:, None] - means) ** 2).sum(axis=0) / nk
        sigmas = np.sqrt(np.maximum(variances, (span * 1e-6) ** 2))
        likelihood = float(sample_weight @ normalizer)
        if np.isfinite(previous) and abs(likelihood - previous) <= tol * (1 + abs(previous)):
            break
        previous = likelihood
    order = np.argsort(means)
    return {"weights": weights[order], "means": means[order], "sigmas": sigmas[order],
            "iterations": iteration + 1, "log_likelihood": likelihood}



def _weighted_cauchy_mixture(x, sample_weight, n_components=GMM_COMPONENTS):
    return _weighted_t_mixture(x, sample_weight, n_components, fit_df=False)


def _weighted_student_t_mixture(x, sample_weight, n_components=GMM_COMPONENTS):
    return _weighted_t_mixture(x, sample_weight, n_components, fit_df=True)


def _weighted_t_mixture(x, sample_weight, n_components, fit_df):

    """Fit a Student-t mixture with one shared df, or fixed df=1 for Cauchy.

    Locations and log scales are optimized on a unit interval for conditioning.
    A histogram-resolution scale floor prevents singular collapsed components.
    """
    x = np.asarray(x, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    if n_components < 1 or len(x) < n_components or x[-1] <= x[0]:
        raise ValueError("Need distinct populated bins for the requested components")
    origin, span = x[0], x[-1] - x[0]
    z = (x - origin) / span
    probability = sample_weight / sample_weight.sum()
    cumulative = probability.cumsum()
    scale_floor = max(float(np.min(np.diff(z))) / 2, 1e-8)

    def unpack(theta):
        logits = np.r_[theta[:n_components - 1], 0.0]
        weights = np.exp(logits - logsumexp(logits))
        locations = theta[n_components - 1:2 * n_components - 1]
        scales = np.exp(theta[2 * n_components - 1:3 * n_components - 1])
        df = float(np.exp(theta[-1])) if fit_df else 1.0
        return weights, locations, scales, df

    def objective(theta):
        weights, locations, scales, df = unpack(theta)
        joint = np.log(weights) + t.logpdf(
            z[:, None], df=df, loc=locations, scale=scales)
        return -float(probability @ logsumexp(joint, axis=1))

    candidates = []
    for lower, upper, initial_scale, initial_df in [(0.2, 0.8, 0.1, 1.0),
            (0.05, 0.95, 0.03, 4.0), (0.35, 0.65, 0.3, 15.0)]:
        quantiles = np.linspace(lower, upper, n_components) if n_components > 1 else [0.5]
        locations = np.interp(quantiles, cumulative, z)
        theta = np.r_[np.zeros(n_components - 1), locations,
                      np.full(n_components, np.log(max(initial_scale, scale_floor)))]
        if fit_df:
            theta = np.r_[theta, np.log(initial_df)]
        result = minimize(
            objective, theta, method="L-BFGS-B",
            bounds=([(-20, 20)] * (n_components - 1)
                    + [(0, 1)] * n_components
                    + [(np.log(scale_floor), np.log(10))] * n_components
                    + ([tuple(np.log(STUDENT_T_DF_BOUNDS))] if fit_df else [])),
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Student-t/Cauchy mixture optimization failed for all starts")
    best = min(candidates, key=lambda result: result.fun)
    weights, locations, scales, df = unpack(best.x)
    order = np.argsort(locations)
    return {"weights": weights[order], "means": origin + span * locations[order],
            "sigmas": span * scales[order], "family": "Student-t" if fit_df else "Cauchy",
            "df": df,
            "df_at_bound": bool(fit_df and any(np.isclose(df, b, rtol=1e-3)
                                              for b in STUDENT_T_DF_BOUNDS)),
            "iterations": best.nit,
            "log_likelihood": float((-best.fun - np.log(span)) * sample_weight.sum())}


def fit_histogram_gmm(values, bounds, log_space, n_components=GMM_COMPONENTS, family="Gaussian"):
    transformed_bounds = np.log10(bounds) if log_space else np.asarray(bounds, dtype=float)
    fit_bins = np.linspace(*transformed_bounds, N_FINE_BINS + 1)
    source_bins = 10 ** fit_bins if log_space else fit_bins
    counts = streaming_histogram(values, source_bins)
    centers = (fit_bins[:-1] + fit_bins[1:]) / 2
    keep = counts > 0
    if family not in {"Gaussian", "Cauchy", "Student-t"}:
        raise ValueError(f"Unknown mixture family: {family}")
    fitter = {"Gaussian": _weighted_gmm, "Cauchy": _weighted_cauchy_mixture,
              "Student-t": _weighted_student_t_mixture}[family]
    model = fitter(centers[keep], counts[keep], n_components=n_components)
    model["family"] = family
    return model


def plot_distribution_and_gmm(values, bounds, log_space, n_components=GMM_COMPONENTS,
                              family="Gaussian", compare_gaussian=False):
    bins = np.geomspace(*bounds, N_PLOT_BINS) if log_space else np.linspace(*bounds, N_PLOT_BINS)
    counts = streaming_histogram(values, bins)
    model = fit_histogram_gmm(values, bounds, log_space, n_components, family)
    gaussian = (fit_histogram_gmm(values, bounds, log_space, n_components)
                if compare_gaussian and family != "Gaussian" else None)
    return plot_precomputed_distribution(bins, counts, model, log_space, gaussian)


def plot_precomputed_distribution(bins, counts, model, log_space, gaussian=None):
    """Render a distribution from small histograms and fitted model parameters."""
    pmf = counts / counts.sum()
    family = model["family"]
    distribution = t if family == "Student-t" else (cauchy if family == "Cauchy" else norm)
    distribution_kwargs = {"df": model["df"]} if family == "Student-t" else {}
    model_bins = np.log10(bins) if log_space else bins
    centers = np.sqrt(bins[:-1] * bins[1:]) if log_space else (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.stairs(pmf, bins, color="#228833", linewidth=2.2, label="Presynaptic-derived TEASAR PMF")
    component_mass = []
    component_lines = []
    for i, (weight, mean, sigma) in enumerate(zip(model["weights"], model["means"], model["sigmas"]), 1):
        mass = weight * np.diff(distribution.cdf(model_bins, loc=mean, scale=sigma, **distribution_kwargs))
        component_mass.append(mass)
        center_nm = 10**mean if log_space else mean
        component_lines.append(ax.plot(centers, mass, linestyle=":", linewidth=1.8,
                                       label=f"component {i} (w={weight:.2f})")[0])
        ax.axvline(center_nm, linestyle="--", linewidth=1.3, alpha=0.8)
    total_mass = np.sum(component_mass, axis=0)
    # Condition model and observations on the same displayed percentile range.
    total_normalizer = total_mass.sum()
    for line in component_lines:
        line.set_ydata(line.get_ydata() / total_normalizer)
    ax.plot(centers, total_mass / total_normalizer, color="black", linewidth=2, label=(f"{family} mixture total (df={model['df']:.3g})"
                                    if family == "Student-t" else f"{family} mixture total"))
    if gaussian is not None:
        mass = sum(w * np.diff(norm.cdf(model_bins, loc=m, scale=s))
                   for w, m, s in zip(gaussian["weights"], gaussian["means"], gaussian["sigmas"]))
        ax.plot(centers, mass / mass.sum(), color="#4477aa", linestyle="--",
                linewidth=2, label="Gaussian mixture total")
    if log_space:
        ax.set_xscale("log")
    scale = "log10-space fit / log x-axis" if log_space else "linear-space fit / linear x-axis"
    ax.set(title=f"Presynaptic-derived TEASAR edge lengths ({scale})",
           xlabel="edge length (nm)", ylabel="probability mass per bin")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, model


def parameter_table(log_model, linear_model):
    rows = []
    for space, model in [("log10", log_model), ("linear", linear_model)]:
        for i, (weight, mean, sigma) in enumerate(zip(model["weights"], model["means"], model["sigmas"]), 1):
            rows.append({"fit_space": space, "component": i, "weight": weight,
                         "family": model.get("family", "Gaussian"),
                         "shared_df": model.get("df", np.nan),
                         "df_at_bound": model.get("df_at_bound", False),
                         "location_fit_units": mean, "scale_fit_units": sigma,
                         "component_center_nm": 10**mean if space == "log10" else mean,
                         "iterations": model["iterations"]})
    table = pd.DataFrame(rows)
    if all(model.get("family", "Gaussian") == "Gaussian"
           for model in (log_model, linear_model)):
        table = table.drop(columns=["shared_df", "df_at_bound"])
    return table


def plot_distributions_by_cell_type(bounds, database=PRESYNAPTIC_DATABASE, manifest=None):
    """Pool derived edges by manifest cell type; normalize each type separately."""
    if manifest is None:
        manifest = Path(__file__).resolve().parents[3] / "data" / "manifest.json"
    labels = json.loads(Path(manifest).read_text())["cells"]
    paths = presynaptic_cell_paths(database)
    if not paths:
        raise FileNotFoundError(f"No presynaptic derived skeletons found: {database}")
    bins = np.linspace(*bounds, N_PLOT_BINS)
    groups = {}
    for path in paths:
        cell_type = labels.get(path.stem, {}).get("cell_type") or "Unlabeled"
        if cell_type not in groups:
            groups[cell_type] = {"cells": 0, "positive_finite_edges": 0,
                                "counts": np.zeros(len(bins) - 1, dtype=np.int64)}
        group = groups[cell_type]
        group["cells"] += 1
        with np.load(path, allow_pickle=False) as cell:
            values = cell["new_edge_length_nm"]
        for chunk in chunks(values):
            valid = chunk[np.isfinite(chunk) & (chunk > 0)]
            group["positive_finite_edges"] += len(valid)
            group["counts"] += np.histogram(valid, bins=bins)[0]

    ncols = 5
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    rows = []
    ymax = 0.0
    for ax, (cell_type, group) in zip(axes.flat, sorted(groups.items())):
        shown = int(group["counts"].sum())
        rows.append({"cell_type": cell_type, "cells": group["cells"],
                     "positive_finite_edges": group["positive_finite_edges"],
                     "displayed_edges": shown})
        if shown:
            pmf = group["counts"] / shown
            ymax = max(ymax, float(pmf.max()))
            ax.stairs(pmf, bins, color="#228833", linewidth=1.6)
        else:
            ax.text(0.5, 0.5, "No edges in displayed range",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_title(f"{cell_type} ({group['cells']:,} cells)")
        ax.grid(alpha=0.2)
        ax.tick_params(labelbottom=True, labelleft=True)
    for ax in axes.flat:
        ax.set(xlim=bounds, ylim=(0, 1.05 * ymax if ymax else 1))
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.supxlabel("edge length (nm)")
    fig.supylabel("probability mass per bin (within cell type)")
    fig.suptitle("Presynaptic-derived TEASAR edge lengths by cell type")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.96))
    return fig, pd.DataFrame(rows)


NEW_SKELETONS = Path("/orcd/scratch/orcd/013/jcbliao/skeletons/segclr/skeletons")


def upstream_edge_counts(n_nodes, edges, boundary_roots, node_synapses):
    """Count branch vertices and synapses on each edge's root-to-parent path.

    The proximal endpoint is included; the distal endpoint is excluded.
    The cut root contributes zero branch points. Only trees with one cut-boundary root
    are eligible; excluded edges receive -1 in both returned arrays.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.sparse.linalg import spsolve_triangular

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    roots = np.unique(np.asarray(boundary_roots, dtype=np.int64))
    branches = np.full(len(edges), -1, dtype=np.int64)
    synapses = branches.copy()
    node_synapses = np.asarray(node_synapses, dtype=np.int64)
    if node_synapses.shape != (n_nodes,) or (node_synapses < 0).any():
        raise ValueError("node_synapses must contain one nonnegative count per node")
    graph = coo_matrix(
        (np.ones(2 * len(edges), dtype=np.float64),
         (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    nodes_per_component = np.bincount(labels, minlength=n_components)
    edge_components = labels[edges[:, 0]]
    edges_per_component = np.bincount(edge_components, minlength=n_components)
    roots_per_component = np.bincount(labels[roots], minlength=n_components)
    is_tree = edges_per_component == nodes_per_component - 1
    eligible = is_tree & (roots_per_component == 1)
    valid_nodes = eligible[labels]
    valid_edges = eligible[edge_components]
    stats = {"components": n_components,
             "rooted_tree_components": int(eligible.sum()),
             "no_boundary_root_components": int((roots_per_component == 0).sum()),
             "multiple_boundary_root_components": int((roots_per_component > 1).sum()),
             "non_tree_components": int((~is_tree).sum()),
             "excluded_edges": int((~valid_edges).sum()),
             "included_edges": int(valid_edges.sum()),
             "synapses_in_rooted_trees": int(node_synapses[valid_nodes].sum()),
             "synapses_in_excluded_components": int(node_synapses[~valid_nodes].sum())}
    valid_roots = roots[eligible[labels[roots]]]
    if not len(valid_roots):
        return branches, synapses, stats

    # Orient the eligible trees in compiled code. Sorting by hop distance puts
    # parents before children, giving a triangular path-sum system.
    distance, parent, _ = dijkstra(
        graph, directed=True, indices=valid_roots, min_only=True,
        return_predecessors=True, unweighted=True)
    nodes = np.flatnonzero(valid_nodes)
    nodes = nodes[np.argsort(distance[nodes], kind="stable")]
    remap = np.full(n_nodes, -1, dtype=np.int64)
    remap[nodes] = np.arange(len(nodes))
    children = nodes[parent[nodes] >= 0]
    child_ids = remap[children]
    parent_ids = remap[parent[children]]
    diagonal = np.arange(len(nodes))
    system = coo_matrix(
        (np.r_[np.ones(len(nodes)), -np.ones(len(children))],
         (np.r_[diagonal, child_ids], np.r_[diagonal, parent_ids])),
        shape=(len(nodes), len(nodes)),
    ).tocsr()
    is_branch = np.diff(graph.indptr) >= 3  # non-root: two or more children
    is_branch[valid_roots] = False
    values = np.column_stack((is_branch[nodes], node_synapses[nodes])).astype(np.float64)
    # path[node] - path[parent] = local_count[node].
    totals = spsolve_triangular(system, values, lower=True, unit_diagonal=True)
    a, b = edges[valid_edges].T
    upstream = np.where(parent[b] == a, a, b)
    edge_totals = np.rint(totals[remap[upstream]]).astype(np.int64)
    branches[valid_edges], synapses[valid_edges] = edge_totals.T
    return branches, synapses, stats


def mapped_presynaptic_counts(cell, n_nodes):
    """Count unique accepted presynaptic IDs at their stored mapped nodes."""
    ids = cell["new_synapse_id"]
    nodes = cell["new_nearest_observed_node"]
    cutoff_key = ("new_within_match_cutoff" if "new_within_match_cutoff" in cell
                  else "new_within_2um")
    accepted = cell[cutoff_key].astype(bool) & (nodes >= 0) & (nodes < n_nodes)
    accepted_ids, accepted_nodes = ids[accepted], nodes[accepted]
    unique_ids, first, inverse = np.unique(accepted_ids, return_index=True, return_inverse=True)
    if len(first) and not np.array_equal(accepted_nodes, accepted_nodes[first][inverse]):
        raise ValueError("A presynaptic ID maps to multiple skeleton nodes")
    counts = np.bincount(accepted_nodes[first], minlength=n_nodes)
    return counts, {"presynaptic_site_rows": len(ids),
                    "unmatched_presynaptic_site_rows": int((~accepted).sum()),
                    "duplicate_matched_site_rows": len(accepted_ids) - len(unique_ids),
                    "matched_unique_presynaptic_sites": len(unique_ids)}


def component_node_mask(n_nodes, edges, node_synapses, min_sites):
    """Keep entire connected components with strictly more than min_sites sites."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    graph = coo_matrix((np.ones(len(edges), dtype=np.uint8),
                       (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes)).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels, weights=node_synapses, minlength=n_components)
    return (counts > min_sites)[labels]


def derived_edge_upstream_counts(cell_path, original_path, soma_radius_nm, min_component_sites=None):
    """Recover exact cut-boundary roots and validate original/database alignment."""
    with np.load(cell_path, allow_pickle=False) as cell:
        positions = cell["new_pos_nm"]
        edges = cell["new_edges"]
        lengths = cell["new_edge_length_nm"]
        center = cell["root_xyz_nm"].astype(np.float64)
        node_synapses, synapse_stats = mapped_presynaptic_counts(cell, len(positions))
    with np.load(original_path, allow_pickle=False) as original:
        original_pos = original["vertices"]
        original_edges = original["edges"]
    keep = np.linalg.norm(original_pos.astype(np.float64) - center, axis=1) > soma_radius_nm
    remap = np.full(len(keep), -1, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    endpoints_kept = keep[original_edges]
    reconstructed_edges = remap[original_edges[endpoints_kept.all(axis=1)]]
    if (not np.array_equal(original_pos[keep].astype(np.float32), positions)
            or not np.array_equal(reconstructed_edges, edges)
            or len(lengths) != len(edges)):
        raise ValueError(f"Original skeleton does not reproduce the stored soma cut: {cell_path}")
    crossing_edges = original_edges[endpoints_kept[:, 0] != endpoints_kept[:, 1]]
    boundary_roots = remap[crossing_edges[keep[crossing_edges]]]
    if min_component_sites is not None:
        selected = component_node_mask(len(positions), edges, node_synapses, min_component_sites)
        selected_edges = selected[edges].all(axis=1)
        selected_remap = np.full(len(positions), -1, dtype=np.int64)
        selected_remap[selected] = np.arange(selected.sum())
        boundary_roots = selected_remap[boundary_roots[selected[boundary_roots]]]
        edges = selected_remap[edges[selected_edges]]
        lengths = lengths[selected_edges]
        positions = positions[selected]
        node_synapses = node_synapses[selected]
        synapse_stats = {f'source_{k}': v for k, v in synapse_stats.items()}
        synapse_stats['matched_unique_presynaptic_sites'] = int(node_synapses.sum())
    branch_counts, synapse_counts, stats = upstream_edge_counts(
        len(positions), edges, boundary_roots, node_synapses)
    stats.update(synapse_stats)
    return lengths, branch_counts, synapse_counts, stats


def _sparse_count_histogram(counts, lengths, length_bins):
    """Retain exact integer x counts without allocating a dense count grid."""
    from scipy.sparse import coo_matrix
    valid = (counts >= 0) & np.isfinite(lengths) & (lengths > 0)
    n_counts = int(counts[valid].max()) + 1 if valid.any() else 1
    visible = valid & (lengths >= length_bins[0]) & (lengths <= length_bins[-1])
    y = np.searchsorted(length_bins, lengths[visible], side="right") - 1
    y = np.minimum(y, len(length_bins) - 2)  # include final bin's right endpoint
    result = coo_matrix(
        (np.ones(int(visible.sum()), dtype=np.int64), (counts[visible], y)),
        shape=(n_counts, len(length_bins) - 1))
    result.sum_duplicates()
    return result


def _upstream_length_cell(task):
    """Compute both upstream metrics in one pass and return sparse histograms."""
    path, original_path, radius, cell_type, length_bins, min_sites = task
    lengths, branches, synapses, stats = derived_edge_upstream_counts(path, original_path, radius, min_sites)
    valid = (branches >= 0) & np.isfinite(lengths) & (lengths > 0)
    visible = valid & (lengths >= length_bins[0]) & (lengths <= length_bins[-1])
    stats.update(root_id=path.stem, cell_type=cell_type,
                 positive_finite_rooted_edges=int(valid.sum()),
                 displayed_edges=int(visible.sum()))
    histograms = {name: _sparse_count_histogram(counts, lengths, length_bins)
                  for name, counts in [("branches", branches), ("synapses", synapses)]}
    return cell_type, histograms, stats


def default_branch_workers():
    """Use at most four processes and respect CPU affinity/Slurm allocation."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated:
        available = min(available, int(allocated))
    return max(1, min(4, available))


def _upstream_cell_results(tasks, workers):
    if workers == 1:
        yield from map(_upstream_length_cell, tasks)
        return
    # Spawn avoids forking a live notebook kernel and its numerical-library
    # threads. Bound pending work so completed results cannot pile up in RAM.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        tasks = iter(tasks)
        pending = set()
        for _ in range(2 * workers):
            task = next(tasks, None)
            if task is None:
                break
            pending.add(pool.submit(_upstream_length_cell, task))
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(pool.submit(_upstream_length_cell, task))
        finally:
            for future in pending:
                future.cancel()


def collect_upstream_length_histograms(bounds, database=PRESYNAPTIC_DATABASE,
                                     originals=NEW_SKELETONS, manifest=None, workers=None,
                                     min_component_sites=None):
    """Process cells independently, retaining only compact histograms in the parent.

    workers=None uses up to four allocated CPUs; workers=1 runs serially.
    Each worker holds one cell's original and derived skeleton in memory.
    """
    database = Path(database)
    metadata = json.loads((database / "metadata.json").read_text())
    if manifest is None:
        manifest = Path(__file__).resolve().parents[3] / "data" / "manifest.json"
    labels = json.loads(Path(manifest).read_text())["cells"]
    paths = presynaptic_cell_paths(database)
    if not paths:
        raise FileNotFoundError(f"No presynaptic derived skeletons found: {database}")
    workers = default_branch_workers() if workers is None else workers
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer or None")
    workers = min(workers, len(paths))
    length_bins = np.linspace(*bounds, N_PLOT_BINS)
    tasks = [(path, Path(originals) / path.name, metadata["soma_radius_nm"],
              labels.get(path.stem, {}).get("cell_type") or "Unlabeled", length_bins,
              min_component_sites)
             for path in paths]
    histograms, diagnostics = {"branches": {}, "synapses": {}}, []
    started = last_report = monotonic()
    print(f"Upstream counts: {len(paths):,} cells, {workers} worker(s)", flush=True)
    for i, (cell_type, metrics, stats) in enumerate(_upstream_cell_results(tasks, workers), 1):
        diagnostics.append(stats)
        for metric, counts in metrics.items():
            groups = histograms[metric]
            if cell_type not in groups:
                groups[cell_type] = counts.tocsr()
            else:
                old = groups[cell_type]
                shape = (max(old.shape[0], counts.shape[0]), len(length_bins) - 1)
                old.resize(shape)
                counts.resize(shape)
                groups[cell_type] = old + counts.tocsr()
        now = monotonic()
        if i == 1 or i % 100 == 0 or now - last_report >= 10 or i == len(paths):
            elapsed = now - started
            print(f"Upstream counts: {i:,}/{len(paths):,} cells "
                  f"({elapsed:.1f}s, {i / max(elapsed, 1e-6):.1f} cells/s)", flush=True)
            last_report = now
    return {"direction": "upstream", "length_bins": length_bins,
            "metrics": {name: dict(sorted(groups.items())) for name, groups in histograms.items()},
            "diagnostics": pd.DataFrame(diagnostics).sort_values("root_id").reset_index(drop=True)}



def collect_branch_length_histograms(*args, **kwargs):
    """Reject old notebook cells after a helper reload."""
    raise RuntimeError(
        "Run the updated cell using collect_upstream_length_histograms(...), "
        "then plot with plot_edge_length_by_upstream_count(...)."
    )


def collect_downstream_length_histograms(*args, **kwargs):
    """Reject the previous downstream notebook cell after a helper reload."""
    raise RuntimeError(
        "The analysis now uses upstream counts. Run the updated cell using "
        "collect_upstream_length_histograms(...)."
    )


def plot_edge_length_by_upstream_count(data, metric="branches", by_cell_type=False,
                                        max_x_bins=200):
    """Plot either upstream path count using common linear axes and shared color bounds.

    Exact integer counts are retained in sparse histograms. If needed, only the
    rendering groups adjacent counts into up to max_x_bins equal-width bins.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator

    if data.get("direction") != "upstream":
        raise ValueError("Recollect with collect_upstream_length_histograms; these are not upstream results")
    if metric not in {"branches", "synapses"}:
        raise ValueError("metric must be 'branches' or 'synapses'")
    if max_x_bins < 1:
        raise ValueError("max_x_bins must be positive")
    histograms = data["metrics"][metric]
    if not histograms:
        raise ValueError("No cell-type histograms available")
    n_counts = max(counts.shape[0] for counts in histograms.values())
    width = max(1, int(np.ceil(n_counts / max_x_bins)))
    n_bins = (n_counts + width - 1) // width
    grouped = {}
    for name, counts in sorted(histograms.items()):
        counts = counts.tocoo()
        dense = np.zeros((n_bins, counts.shape[1]), dtype=np.int64)
        np.add.at(dense, (counts.row // width, counts.col), counts.data)
        grouped[name] = dense
    panels = grouped if by_cell_type else {"All cell types": sum(grouped.values())}
    ncols = 5 if by_cell_type else 1
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True,
                             figsize=(20 if by_cell_type else 9, 3.4 * nrows if by_cell_type else 5),
                             layout="constrained")
    maximum = max(int(counts.max()) for counts in panels.values())
    norm = LogNorm(vmin=1, vmax=max(2, maximum))
    count_bins = np.arange(n_bins + 1) * width - 0.5
    for ax, (name, counts) in zip(axes.flat, panels.items()):
        mesh = ax.pcolormesh(count_bins, data["length_bins"],
                             np.ma.masked_equal(counts.T, 0),
                             norm=norm, cmap="viridis", shading="flat", rasterized=True)
        ax.set(title=name, xlim=(count_bins[0], count_bins[-1]),
               ylim=(data["length_bins"][0], data["length_bins"][-1]))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelbottom=True, labelleft=True)
        if not counts.any():
            ax.text(0.5, 0.5, "No eligible edges in range",
                    ha="center", transform=ax.transAxes)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.colorbar(mesh, ax=list(axes.flat)[:len(panels)], label="edge count (log color scale)")
    label = "branch points" if metric == "branches" else "presynaptic sites"
    fig.supxlabel(f"number of upstream {label} ({width} count(s) per bin)")
    fig.supylabel("edge length (nm)")
    fig.suptitle(f"Edge length versus upstream {label}")
    return fig
