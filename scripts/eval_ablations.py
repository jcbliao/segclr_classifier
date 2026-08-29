"""Evaluate whether trained window classifiers use node arrangement or only a multiset.

The question is not answered by shuffling whole PyG batches. A collated batch is one
block-diagonal graph, so a batch-wide shuffle moves embeddings between windows belonging
to different cells and labels, while a batch-wide rewire invents cross-window edges. Both
would lower accuracy for reasons unrelated to spatial/topological use. Every intervention
here is therefore bounded by ``batch.ptr`` / ``batch.batch`` and applied to fresh tensors.

The headline ``permute_x`` condition preserves the graph, Laplacian positional encoding,
and center-relative geometry bit-for-bit while breaking only the correspondence between an
embedding and its location. ``rewire`` preserves the number of undirected edges per window
and recomputes the Laplacian PE from that rewired graph, making the graph and its structural
features self-consistent but wrong. ``rewire_lpe_stale`` instead retains the original PE; its
gap from ``rewire`` measures how much structural signal the model can recover from PE alone
rather than from message passing or attention over the new edges. ``rewire_lpe0`` is the
wrong-graph/no-PE corner. All three share the same sampled graph within a batch, so their gaps
do not contain a second, random topology change.

``recompute_lpe`` is what makes the rewire family readable. A Laplacian PE is not unique --
eigenvector signs are arbitrary, and on windows this small degenerate eigenvalues make whole
eigenspaces a free choice of the solver -- while ``identity`` carries the encoding extraction
computed on CPU and every rewired condition carries one recomputed here on GPU. Recomputing on
the unmodified graph isolates that arbitrary choice on its own, so it bounds how much of any
rewire delta is solver convention rather than topology. Read it before reading ``rewire``.

Window extraction includes induced-edge remapping and an O(W^3) Laplacian eigendecomposition,
so it is paid exactly once per batch and shared by every condition and checkpoint at one
radius. Re-evaluating one checkpoint at a time would repeat the dominant CPU work. Predictions
are retained as int8 because the resolved hierarchy has only a small number of classes; labels
and root ids are retained once because the unshuffled loader order is common to all runs.

MeanReadout over raw embeddings supplies a harness self-check. Its predictions must be exactly
unchanged by every condition. A moving meanpool result means the intervention escaped its
window or the wrong tensors reached a model, not that the baseline uses structure. The
permutation invariance would even hold for a hypothetical mean + spatial-feature model: the
mean of a concatenation is the concatenation of the block means, and permuting one block
independently cannot change its mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from data.dataset_windowed import WindowedGraphDatasetLCPN  # noqa: E402
from data.geodesic_window import DEFAULT_WINDOW_NM  # noqa: E402
from gnn.metrics import majority_vote_by_group, summarize  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402


CONDITIONS = (
    "identity", "recompute_lpe", "permute_x", "rewire", "rewire_lpe_stale",
    "rewire_lpe0", "drop_edges", "zero_pos_enc", "zero_rel_pos",
)
CSV_FIELDS = (
    "run", "window_nm", "architecture", "spatial", "cls_resnet", "frozen_agg",
    "condition", "n_windows", "n_cells", "window_accuracy",
    "window_balanced_accuracy", "window_macro_precision", "window_macro_f1",
    "cell_accuracy", "cell_balanced_accuracy", "cell_macro_precision", "cell_macro_f1",
    "d_window_macro_f1", "d_cell_macro_f1",
)
RADIUS_RE = re.compile(r"_w([0-9]+(?:\.[0-9]+)?)um(?:_|$)")


@dataclass
class LoadedRun:
    name: str
    window_nm: float
    model: WindowClassifier
    config: ModelConfig
    frozen_agg: bool


@dataclass
class Inputs:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    pos_enc: torch.Tensor
    rel_pos: torch.Tensor
    thickness: torch.Tensor | None


def _radius_from_name(run: str) -> float:
    match = RADIUS_RE.search(run)
    return float(match.group(1)) * 1000.0 if match else DEFAULT_WINDOW_NM


def _metadata_radius(results_dir: Path, run: str) -> float | None:
    path = results_dir / f"{run}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    try:
        return float(payload["args"]["window_nm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{path} exists but has no numeric args.window_nm") from exc


def discover_runs(results_dir: Path, requested: list[str] | None, window_nm: float) -> list[tuple[str, Path]]:
    checkpoints = {p.parent.name: p for p in results_dir.glob("*/checkpoint_best.pt")}
    if requested:
        missing = [run for run in requested if run not in checkpoints]
        if missing:
            raise SystemExit(f"--runs checkpoint(s) not found under {results_dir}: {', '.join(missing)}")
        names = requested
    else:
        names = sorted(checkpoints)

    selected = []
    for run in names:
        name_radius = _radius_from_name(run)
        json_radius = _metadata_radius(results_dir, run)
        if json_radius is not None and json_radius != name_radius:
            raise SystemExit(
                f"window radius disagreement for {run}: directory name implies "
                f"{name_radius:g} nm but {results_dir / (run + '.json')} says {json_radius:g} nm"
            )
        if name_radius == window_nm:
            selected.append((run, checkpoints[run]))
    if not selected:
        raise SystemExit(f"no checkpoint_best.pt runs resolve to --window-nm {window_nm:g}")
    return selected


def _load_runs(specs, hierarchy, device) -> list[LoadedRun]:
    loaded = []
    for name, path in specs:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt.get("config")
        if not isinstance(config, ModelConfig):
            got = "missing" if config is None else type(config).__name__
            raise SystemExit(
                f"cannot rebuild {name}: {path} config is {got}, expected gnn.model.ModelConfig"
            )
        model = WindowClassifier(config, hierarchy=hierarchy).to(device)
        # The checkpoint carries the ModelConfig but not the hierarchy, so the
        # LCPN heads are rebuilt from whatever data/dataset_lcpn.py::load_hierarchy
        # resolves to now. A run trained before a label was added to or removed
        # from DROP_LABELS therefore has per-node heads of a different width, and
        # surfaces here as an opaque size-mismatch traceback rather than as the
        # stale-checkpoint problem it actually is.
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as exc:
            raise SystemExit(
                f"{name}: {path} does not fit the current hierarchy "
                f"({len(hierarchy.level_classes[-1])} classes, dropping "
                f"{sorted(hierarchy.drop_labels)}). It was almost certainly trained against a "
                f"different DROP_LABELS / HIERARCHY_LEVELS_DROPPED -- retrain it or exclude it "
                f"with --runs.\n  {exc}"
            ) from exc
        model.eval()
        loaded.append(LoadedRun(
            name=name, window_nm=_radius_from_name(name), model=model, config=config,
            frozen_agg="_frozenagg" in name,
        ))
        tqdm.write(f"loaded {name} ({config.architecture}) onto {device}")
    return loaded


def _rewire(batch, edge_index: torch.Tensor) -> torch.Tensor:
    """Uniform endpoint sampling within each graph, preserving undirected edge counts."""
    num_graphs = int(batch.num_graphs)
    edge_graph = batch.batch[edge_index[0]]
    undirected_counts = torch.bincount(edge_graph, minlength=num_graphs) // 2
    graph_ids = torch.repeat_interleave(
        torch.arange(num_graphs, device=edge_index.device), undirected_counts
    )
    if graph_ids.numel() == 0:
        return edge_index.new_empty((2, 0))

    ptr = batch.ptr
    sizes = ptr[1:] - ptr[:-1]
    local_sizes = sizes[graph_ids]
    keep = local_sizes > 1
    graph_ids = graph_ids[keep]
    local_sizes = local_sizes[keep]
    if graph_ids.numel() == 0:
        return edge_index.new_empty((2, 0))

    # Multiplying a uniform [0,1) draw by each repeated local size is the
    # varying-upper-bound equivalent of randint; no per-graph Python loop is needed.
    u = (torch.rand(graph_ids.numel(), device=edge_index.device) * local_sizes).long()
    v = (torch.rand(graph_ids.numel(), device=edge_index.device) * local_sizes).long()
    # A one-step wrap chooses a different endpoint without rejection sampling.
    v = torch.where(v == u, (v + 1) % local_sizes, v)
    offsets = ptr[graph_ids]
    src, dst = offsets + u, offsets + v
    return torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)


def _recompute_pos_enc(
    edge_index: torch.Tensor, batch, pos_dim: int, device: torch.device
) -> torch.Tensor:
    """Recompute extraction-equivalent Laplacian PE on each rewired window."""
    counts = batch.ptr[1:] - batch.ptr[:-1]
    output = torch.zeros((batch.x.shape[0], pos_dim), dtype=torch.float32, device=device)
    edge_graph = batch.batch[edge_index[0]]
    edge_dst_graph = batch.batch[edge_index[1]]

    # Grouping by exact node count is what makes batching legitimate here, not
    # merely an optimization detail. Every (K, n, n) group contains no padding,
    # so its batched eigh is numerically the same operation as looping over K
    # windows. Eigendecomposing one padded mixed-size batch would instead mix
    # real eigenvectors with modes from the zero-adjacency padding block. There
    # are few distinct sizes (windows average ~10.7 nodes), so this costs only a
    # handful of batched GPU calls rather than thousands of tiny decompositions.
    for size_tensor in torch.unique(counts):
        n = int(size_tensor.item())
        graph_ids = torch.nonzero(counts == n, as_tuple=False).flatten()
        k = graph_ids.numel()
        adj = torch.zeros((k, n, n), dtype=torch.float64, device=device)

        if edge_index.numel():
            graph_to_group = torch.full(
                (batch.num_graphs,), -1, dtype=torch.long, device=device
            )
            graph_to_group[graph_ids] = torch.arange(k, device=device)
            group_rows = graph_to_group[edge_graph]
            keep = group_rows >= 0
            local_src = edge_index[0, keep] - batch.ptr[edge_graph[keep]]
            local_dst = edge_index[1, keep] - batch.ptr[edge_dst_graph[keep]]
            adj[group_rows[keep], local_src, local_dst] = 1.0
        adj.diagonal(dim1=-2, dim2=-1).fill_(1.0)

        degree = adj.sum(dim=2).clamp(min=1.0)
        d_inv_sqrt = torch.diag_embed(degree.pow(-0.5))
        eye = torch.eye(n, dtype=torch.float64, device=device).expand(k, -1, -1)
        lap = eye - d_inv_sqrt @ adj @ d_inv_sqrt
        _, eig_vec = torch.linalg.eigh(lap)
        eig_vec = torch.flip(eig_vec, dims=[2])
        group_pos_enc = eig_vec[:, :, 1 : pos_dim + 1]
        # For n==1 this slice has width zero and the same trailing-pad rule
        # used during extraction makes the lone node's entire PE exactly zero.
        if group_pos_enc.shape[2] < pos_dim:
            pad = torch.zeros(
                (k, n, pos_dim - group_pos_enc.shape[2]),
                dtype=torch.float64, device=device,
            )
            group_pos_enc = torch.cat([group_pos_enc, pad], dim=2)

        node_slots = batch.ptr[graph_ids, None] + torch.arange(n, device=device)[None, :]
        output[node_slots.reshape(-1)] = group_pos_enc.float().reshape(-1, pos_dim)
    return output


def intervene(
    batch,
    condition: str,
    pos_dim: int,
    rewired_edge_index: torch.Tensor | None = None,
    rewired_pos_enc: torch.Tensor | None = None,
    recomputed_pos_enc: torch.Tensor | None = None,
) -> Inputs:
    # Clone even identity: no condition may alias tensors owned by the loader batch.
    x = batch.x.clone()
    edge_index = batch.edge_index.clone()
    edge_attr = batch.edge_attr.clone()
    pos_enc = batch.pos_enc.clone()
    rel_pos = batch.rel_pos.clone()
    thickness = getattr(batch, "thickness", None)
    thickness = thickness.clone() if thickness is not None else None

    if condition == "permute_x":
        # rand is in [0,1), while adjacent graph offsets differ by 2. Sorting
        # therefore randomizes only inside each already-contiguous graph block.
        # float64 is load-bearing, not defensive: at batch_size 4096 the offset
        # reaches ~8190, where float32's 24-bit mantissa resolves steps of only
        # ~1e-3, quantizing the [0,1) draw to ~1024 levels. Windows average 10.7
        # nodes, so ties would be common and argsort would resolve them by node
        # index -- a permutation still, but biased toward the original order,
        # which is exactly the thing this condition is supposed to destroy.
        keys = batch.batch.to(torch.float64) * 2.0 + torch.rand(
            x.shape[0], device=x.device, dtype=torch.float64
        )
        x = x[torch.argsort(keys)]
    elif condition in ("rewire", "rewire_lpe_stale", "rewire_lpe0"):
        if rewired_edge_index is None:
            raise ValueError(f"{condition} requires one shared rewired graph for this batch")
        edge_index = rewired_edge_index.clone()
        # Nothing in gnn/ reads edge_attr, but mismatching its length with
        # edge_index would be a landmine for any later consumer of this batch.
        edge_attr = batch.edge_attr.new_zeros((edge_index.shape[1],) + batch.edge_attr.shape[1:])
        if condition == "rewire":
            if rewired_pos_enc is None or rewired_pos_enc.shape[1] != pos_dim:
                raise ValueError(
                    f"rewire requires recomputed PE with the trained width pos_dim={pos_dim}"
                )
            pos_enc = rewired_pos_enc.clone()
        elif condition == "rewire_lpe0":
            pos_enc.zero_()
    elif condition == "drop_edges":
        edge_index = edge_index.new_empty((2, 0))
        edge_attr = batch.edge_attr.new_zeros((0,) + batch.edge_attr.shape[1:])
        # With no edges, adding self-loops makes A=I and the normalized
        # Laplacian exactly zero. Its spectrum is completely degenerate, so
        # eigh may return an arbitrary numerical basis that describes LAPACK,
        # not the graph. Zero is the only meaningful encoding in this case.
        pos_enc.zero_()
    elif condition == "recompute_lpe":
        # The recomputation-fidelity control, and the row that decides whether
        # any `rewire` number can be read as a topology effect at all.
        #
        # A Laplacian PE is not unique. Eigenvector SIGNS are arbitrary, and
        # worse, the small windows here (paths and stars, ~10.7 nodes) routinely
        # have DEGENERATE eigenvalues, where the whole eigenspace -- not just a
        # sign -- is a free choice the solver makes. `identity` carries the PE
        # extraction computed on CPU via LAPACK; `rewire` carries one this
        # harness computes on GPU via cuSOLVER. So a `rewire` delta could in
        # principle be the two backends disagreeing about an arbitrary basis
        # rather than the model noticing the graph changed.
        #
        # This condition recomputes the PE on the UNMODIFIED graph, through the
        # exact code path `rewire` uses. Everything the model sees is
        # semantically identical to `identity`, differing only by that arbitrary
        # choice -- so its delta is a direct measurement of the confound, and
        # `rewire` is interpretable only to the extent this row reads ~0.
        if recomputed_pos_enc is None or recomputed_pos_enc.shape[1] != pos_dim:
            raise ValueError(
                f"recompute_lpe requires a recomputed PE of the trained width pos_dim={pos_dim}"
            )
        pos_enc = recomputed_pos_enc.clone()
    elif condition == "zero_pos_enc":
        pos_enc.zero_()
    elif condition == "zero_rel_pos":
        rel_pos.zero_()
    elif condition != "identity":
        raise ValueError(f"unknown condition {condition!r}")
    return Inputs(x, edge_index, edge_attr, pos_enc, rel_pos, thickness)


def _cell_metrics(labels, preds, root_ids, classes):
    cell_true, cell_pred = majority_vote_by_group(root_ids, labels, preds)
    return summarize(cell_true, cell_pred, len(classes), classes)


def _output_stem(limit_batches: int | None, window_nm: float) -> str:
    stem = "ablations"
    if limit_batches is not None:
        stem += f"_limit{limit_batches}batches"
    if window_nm != DEFAULT_WINDOW_NM:
        stem += f"_w{window_nm / 1000.0:g}um"
    return stem


@torch.no_grad()
def main(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    specs = discover_runs(results_dir, args.runs, args.window_nm)
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)
    models = _load_runs(specs, hierarchy, device)

    use_thickness = any(run.config.gt_use_thickness for run in models)
    pos_dims = {run.config.gt_pos_dim for run in models}
    if len(pos_dims) != 1:
        raise SystemExit(f"runs at one radius disagree on gt_pos_dim: {sorted(pos_dims)}")
    pos_dim = pos_dims.pop()
    dataset = WindowedGraphDatasetLCPN(
        manifest, "test", pos_dim=pos_dim, use_thickness=use_thickness,
        window_nm=args.window_nm,
    )
    classes = dataset.classes
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    conditions = list(dict.fromkeys(["identity", *args.conditions]))
    prediction_chunks = {(run.name, condition): [] for run in models for condition in conditions}
    label_chunks, root_id_chunks = [], []
    total = len(loader) if args.limit_batches is None else min(len(loader), args.limit_batches)
    tqdm.write(
        f"device={device}; radius={args.window_nm:g} nm; runs={len(models)}; "
        f"conditions={len(conditions)}; batches={total}"
    )

    bar = tqdm(loader, total=total, desc="eval ablations", unit="batch")
    for batch_idx, batch in enumerate(bar):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break
        batch = batch.to(device)
        label_chunks.append(batch.y_levels[:, -1].cpu().numpy())
        root_id_chunks.append(batch.root_id.cpu().numpy().reshape(-1))
        rewire_conditions = {"rewire", "rewire_lpe_stale", "rewire_lpe0"}
        rewired_edge_index = None
        rewired_pos_enc = None
        if rewire_conditions.intersection(conditions):
            rewired_edge_index = _rewire(batch, batch.edge_index)
        if "rewire" in conditions:
            rewired_pos_enc = _recompute_pos_enc(
                rewired_edge_index, batch, pos_dim, device
            )
        # Deliberately the same function on the ORIGINAL edges -- the control's
        # value depends on it sharing every step of the rewired path except the
        # graph it is handed.
        recomputed_pos_enc = None
        if "recompute_lpe" in conditions:
            recomputed_pos_enc = _recompute_pos_enc(
                batch.edge_index, batch, pos_dim, device
            )
        for condition in conditions:
            inputs = intervene(
                batch, condition, pos_dim, rewired_edge_index, rewired_pos_enc,
                recomputed_pos_enc,
            )
            for run in models:
                g = run.model(
                    inputs.x, inputs.edge_index, batch.batch, pos_enc=inputs.pos_enc,
                    rel_pos=inputs.rel_pos, thickness=inputs.thickness,
                )
                pred = run.model.cls_head.predict_top_down(g)[:, -1]
                prediction_chunks[(run.name, condition)].append(
                    pred.cpu().numpy().astype(np.int8, copy=False)
                )
        bar.set_postfix(windows=sum(len(x) for x in label_chunks), batch=batch_idx + 1)

    labels = np.concatenate(label_chunks)
    root_ids = np.concatenate(root_id_chunks)
    predictions = {key: np.concatenate(chunks) for key, chunks in prediction_chunks.items()}
    # A non-spatial mean run discards edge_index, pos_enc and rel_pos outright --
    # WindowClassifier._node_features returns x unmodified when
    # use_spatial_features is False, and MeanReadout is permutation-invariant --
    # so no condition here can reach it. Every one of them therefore re-runs an
    # identical computation, which makes this the strongest available test that
    # an intervention stayed inside its window and that the right tensors reached
    # the right model.
    #
    # The bar is a small tolerance rather than bitwise equality, and that is not
    # a weakened assertion. global_mean_pool reduces with CUDA atomics, whose
    # accumulation ORDER varies between otherwise identical calls, so two runs of
    # the same forward pass can differ in the last ulp. A deep classification
    # head amplifies that into a flipped argmax whenever two classes are near-tied
    # -- measured at 1 window in 2,423,696, and only for the resnet4x128 head,
    # never for the linear probe. What the tolerance still catches is the failure
    # it was written for: an intervention escaping its window would corrupt
    # features for a large fraction of windows, not four in ten million.
    max_flip_fraction = 1e-5
    for run in models:
        if run.config.architecture != "mean" or run.config.use_spatial_features:
            continue
        baseline = predictions[(run.name, "identity")]
        for condition in conditions:
            n_diff = int((baseline != predictions[(run.name, condition)]).sum())
            if n_diff == 0:
                continue
            fraction = n_diff / len(baseline)
            if fraction > max_flip_fraction:
                raise AssertionError(
                    f"meanpool harness self-check failed for run {run.name}, condition "
                    f"{condition}: {n_diff}/{len(baseline)} ({fraction:.2%}) window predictions "
                    "differ from identity, far beyond floating-point tie-breaking, but mean "
                    "pooling cannot see anything this condition changed. The intervention "
                    "escaped its window, or the wrong tensors reached the model."
                )
            tqdm.write(
                f"self-check: {run.name} / {condition} differs from identity on "
                f"{n_diff}/{len(baseline)} windows ({fraction:.2e}) -- within the "
                "nondeterministic-reduction tolerance, treated as float tie-breaking"
            )

    rows, json_rows = [], []
    for run in models:
        per_run = {}
        for condition in conditions:
            preds = predictions[(run.name, condition)]
            window_metrics = summarize(labels, preds, len(classes), classes)
            cell_metrics = _cell_metrics(labels, preds, root_ids, classes)
            per_run[condition] = (window_metrics, cell_metrics)
        identity_window, identity_cell = per_run["identity"]
        for condition in conditions:
            window_metrics, cell_metrics = per_run[condition]
            row = {
                "run": run.name, "window_nm": run.window_nm,
                "architecture": run.config.architecture,
                "spatial": run.config.use_spatial_features,
                "cls_resnet": run.config.cls_head_resnet, "frozen_agg": run.frozen_agg,
                "condition": condition, "n_windows": len(labels),
                "n_cells": len(np.unique(root_ids)),
                **{f"window_{key}": window_metrics[key] for key in
                   ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1")},
                **{f"cell_{key}": cell_metrics[key] for key in
                   ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1")},
                "d_window_macro_f1": window_metrics["macro_f1"] - identity_window["macro_f1"],
                "d_cell_macro_f1": cell_metrics["macro_f1"] - identity_cell["macro_f1"],
            }
            rows.append(row)
            json_rows.append({
                **row,
                "window_per_class_recall": window_metrics["per_class_recall"],
                "cell_per_class_recall": cell_metrics["per_class_recall"],
            })

    out_dir = results_dir / "eval_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(args.limit_batches, args.window_nm)
    csv_path, json_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({
        "classes": classes, "window_nm": args.window_nm,
        "limit_batches": args.limit_batches, "rows": json_rows,
    }, indent=2))
    tqdm.write(f"wrote {csv_path}")
    tqdm.write(f"wrote {json_path}")
    tqdm.write("=== ranked intervention deltas (window macro F1) ===")
    for row in sorted(rows, key=lambda item: item["d_window_macro_f1"]):
        tqdm.write(
            f"{row['d_window_macro_f1']:+.4f}  {row['run']}  {row['condition']} "
            f"(F1={row['window_macro_f1']:.4f})"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", help="explicit result-directory names to evaluate")
    parser.add_argument("--window-nm", type=float, default=DEFAULT_WINDOW_NM)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    cli_args = parser.parse_args()
    if cli_args.limit_batches is not None and cli_args.limit_batches < 1:
        parser.error("--limit-batches must be at least 1")
    raise SystemExit(main(cli_args))
