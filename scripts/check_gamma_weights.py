"""Measure the learned trade-off between global attention and the adjacency bias.

`gnn/graph_transformer.py::GraphAttention` builds its logits as
``gamma_0 * (QK^T / sqrt(d)) + gamma_1 * adj``, with ``gamma = exp(predict_gamma(x))``
predicted per query node per layer. So there is no single pair of numbers to read off a
checkpoint: gamma is a function of the node's hidden state, and a weight matrix alone says
nothing about the values it actually produces. This script runs each trained
GraphTransformer's best checkpoint over the real test windows and records the gammas the
model assigns, layer by layer.

Three things it separates, because collapsing any of them would misreport the mechanism:

- **CLS rows from node rows.** The CLS token's own adjacency row is a self-loop and nothing
  else (`adj_full[:, 0, 0] = 1`), so its ``gamma_1`` biases exactly one logit -- its own --
  while its ``gamma_0`` scales the attention that produces the returned window embedding.
  Averaging it into the node rows would mix a readout temperature with a locality knob.
- **Padding rows from real ones.** Padded positions still get a gamma predicted from an
  all-zero hidden state; they are dropped by the key-padding mask, so they are dropped here.
- **The raw gamma ratio from its effect.** ``gamma_1 / gamma_0`` is the bias measured in
  units of the attention logit -- but whether that shifts any probability depends on the
  spread of ``QK^T/sqrt(d)`` in that row and on how many of the row's keys are neighbors.
  So each row's attention is also computed twice, with the bias and with ``gamma_1`` forced
  to zero, and the neighbor probability mass of both is recorded. The gap is what the bias
  term actually buys, in probability, and ``uniform`` (the neighbor share of the row's keys)
  is what a structure-blind row would already put there.

Quantiles come from fixed log10-spaced histograms accumulated on device rather than from
retained samples: over ~25M node rows per layer, storing values to sort later would cost more
memory than the models, and 0.01-dex bins resolve a median to ~2%.

A `_frozenagg` run is a control, not a failure: its GraphTransformer never left random init,
where `predict_gamma.weight` is drawn from U(0, 0.01) with zero bias, so both gammas sit at
essentially exp(0) = 1. It is the "no training happened" reference every trained run's spread
should be read against.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.dataset_lcpn import load_hierarchy, load_manifest  # noqa: E402
from data.dataset_windowed import WindowedGraphDatasetLCPN  # noqa: E402
from eval_ablations import _metadata_radius, _radius_from_name  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402

# log10 histogram support for gamma_0, gamma_1 and their ratio. exp() of a
# 128-wide linear head cannot realistically leave this range, and out-of-range
# values are counted at the edges rather than dropped (see Histogram.update).
LOG_LO, LOG_HI, LOG_BINS = -4.0, 4.0, 800
# Linear histogram for probabilities (neighbor mass and its bias-induced delta).
PROB_BINS = 200
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

CSV_FIELDS = (
    "run", "window_nm", "best_epoch", "cls_resnet", "frozen_agg", "layer", "row_type",
    "n_rows", "gamma_0_mean", "gamma_1_mean", "ratio_mean",
    *(f"gamma_0_p{int(q * 100)}" for q in QUANTILES),
    *(f"gamma_1_p{int(q * 100)}" for q in QUANTILES),
    *(f"ratio_p{int(q * 100)}" for q in QUANTILES),
    "nbr_mass_biased_mean", "nbr_mass_unbiased_mean", "nbr_mass_delta_mean",
    "nbr_mass_delta_p50", "nbr_share_uniform_mean",
)


class Histogram:
    """Fixed-range histogram with edge clamping, accumulated on device."""

    def __init__(self, lo: float, hi: float, bins: int, device: torch.device, log10: bool):
        self.lo, self.hi, self.bins, self.log10 = lo, hi, bins, log10
        self.counts = torch.zeros(bins, dtype=torch.float64, device=device)

    def update(self, values: torch.Tensor) -> None:
        v = values.detach().flatten().float()
        if self.log10:
            # A gamma of exactly 0 is only reachable with use_exp off; log10 of it
            # would be -inf, which clamp still maps onto the first bin.
            v = torch.log10(v.clamp_min(torch.finfo(torch.float32).tiny))
        # Clamping rather than discarding keeps every row in the denominator, so a
        # quantile can never be computed over a silently filtered population.
        v = v.clamp(self.lo, self.hi - (self.hi - self.lo) / self.bins * 0.5)
        self.counts += torch.histc(v, bins=self.bins, min=self.lo, max=self.hi).double()

    def quantiles(self, qs=QUANTILES) -> list[float]:
        total = float(self.counts.sum().item())
        if total == 0:
            return [float("nan")] * len(qs)
        cdf = torch.cumsum(self.counts, dim=0) / total
        width = (self.hi - self.lo) / self.bins
        out = []
        for q in qs:
            idx = int(torch.searchsorted(cdf, torch.tensor(q, dtype=cdf.dtype, device=cdf.device)))
            idx = min(idx, self.bins - 1)
            centre = self.lo + (idx + 0.5) * width
            out.append(float(10.0**centre) if self.log10 else float(centre))
        return out


@dataclass
class RowStats:
    """One (layer, row-type) accumulator: exact means plus histograms for quantiles."""

    device: torch.device
    log_gamma: bool
    n: float = 0.0
    sums: dict = field(default_factory=dict)
    hists: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ("gamma_0", "gamma_1", "ratio", "nbr_biased", "nbr_unbiased",
                    "nbr_delta", "nbr_uniform"):
            self.sums[key] = 0.0
        for key in ("gamma_0", "gamma_1", "ratio"):
            self.hists[key] = Histogram(LOG_LO, LOG_HI, LOG_BINS, self.device, self.log_gamma)
        self.hists["nbr_delta"] = Histogram(0.0, 1.0, PROB_BINS, self.device, False)

    def update(self, values: dict[str, torch.Tensor]) -> None:
        count = values["gamma_0"].numel()
        if count == 0:
            return
        self.n += count
        for key, tensor in values.items():
            self.sums[key] += float(tensor.double().sum().item())
            if key in self.hists:
                self.hists[key].update(tensor)

    def report(self) -> dict:
        mean = lambda key: self.sums[key] / self.n if self.n else float("nan")  # noqa: E731
        row = {"n_rows": int(self.n)}
        for key in ("gamma_0", "gamma_1", "ratio"):
            row[f"{key}_mean"] = mean(key)
            for q, value in zip(QUANTILES, self.hists[key].quantiles()):
                row[f"{key}_p{int(q * 100)}"] = value
        row["nbr_mass_biased_mean"] = mean("nbr_biased")
        row["nbr_mass_unbiased_mean"] = mean("nbr_unbiased")
        row["nbr_mass_delta_mean"] = mean("nbr_delta")
        row["nbr_mass_delta_p50"] = self.hists["nbr_delta"].quantiles((0.5,))[0]
        row["nbr_share_uniform_mean"] = mean("nbr_uniform")
        return row


@dataclass
class LoadedRun:
    name: str
    window_nm: float
    best_epoch: int | None
    model: WindowClassifier
    config: ModelConfig
    stats: dict  # (layer, row_type) -> RowStats
    handles: list = field(default_factory=list)


def make_hook(run: LoadedRun, layer: int):
    """Recompute this layer's attention twice -- with the adjacency bias and with
    gamma_1 forced to zero -- alongside the gammas themselves.

    Everything here mirrors GraphAttention.forward exactly (same qkv projection, same
    head convention, same padding/attn masks); it is recomputed rather than returned by
    the module because the module returns only its projected output, and changing its
    signature to hand out internals would put a diagnostic in the training hot path.
    """

    @torch.no_grad()
    def hook(module, args, output):
        x, adj, key_padding_mask, attn_mask = args
        B, N, C = x.shape
        gamma = module.predict_gamma(x)
        if module.use_exp:
            gamma = torch.exp(gamma)
        g0, g1 = gamma[..., 0], gamma[..., 1]  # (B, N) each

        qkv = module.qkv_projection(x).view(B, N, 3, module.num_heads, module.dim)
        query, key, _ = qkv.permute(0, 3, 1, 2, 4).unbind(dim=3)
        logits = (query @ key.transpose(-2, -1)) * module.scale  # (B, H, N, N)
        unbiased = g0[:, None, :, None] * logits
        biased = unbiased + g1[:, None, :, None] * adj[:, None]

        pad = ~key_padding_mask[:, None, None, :]
        unbiased = unbiased.masked_fill(pad, float("-inf"))
        biased = biased.masked_fill(pad, float("-inf"))
        if attn_mask is not None:
            unbiased = unbiased.masked_fill(~attn_mask[:, None], float("-inf"))
            biased = biased.masked_fill(~attn_mask[:, None], float("-inf"))

        neighbor = (adj[:, None] > 0).to(logits.dtype)  # (B, 1, N, N), self-loops included
        mass_biased = (biased.softmax(dim=-1) * neighbor).sum(-1).mean(dim=1)  # (B, N)
        mass_unbiased = (unbiased.softmax(dim=-1) * neighbor).sum(-1).mean(dim=1)
        # What a row that ignored structure entirely would already place on neighbors:
        # the neighbor share of the keys it is allowed to attend to.
        allowed = key_padding_mask[:, None, :].to(logits.dtype)
        if attn_mask is not None:
            allowed = allowed * attn_mask.to(logits.dtype)
        uniform = (neighbor[:, 0] * allowed).sum(-1) / allowed.sum(-1).clamp_min(1.0)

        # CLS is row 0 of every window and is always real; its adjacency row is a
        # self-loop only, so its gamma_1 is a self-bias, not a locality knob. Every
        # other real row is a window node; padding rows are dropped outright.
        cls_rows = torch.zeros_like(key_padding_mask)
        cls_rows[:, 0] = True
        node_rows = key_padding_mask.clone()
        node_rows[:, 0] = False
        selections = {"cls": cls_rows, "node": node_rows}
        for row_type, mask in selections.items():
            run.stats[(layer, row_type)].update({
                "gamma_0": g0[mask], "gamma_1": g1[mask], "ratio": (g1 / g0)[mask],
                "nbr_biased": mass_biased[mask], "nbr_unbiased": mass_unbiased[mask],
                "nbr_delta": (mass_biased - mass_unbiased)[mask],
                "nbr_uniform": uniform[mask],
            })

    return hook


def discover_gt_runs(results_dir: Path, requested: list[str] | None) -> list[tuple[str, Path]]:
    checkpoints = {p.parent.name: p for p in results_dir.glob("*/checkpoint_best.pt")}
    if requested:
        missing = [run for run in requested if run not in checkpoints]
        if missing:
            raise SystemExit(f"checkpoint_best.pt not found for: {', '.join(missing)}")
        names = requested
    else:
        names = sorted(checkpoints)
    return [(name, checkpoints[name]) for name in names]


def load_run(name: str, path: Path, hierarchy, device: torch.device) -> LoadedRun | None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if not isinstance(config, ModelConfig):
        raise SystemExit(f"{path} carries no gnn.model.ModelConfig; cannot rebuild {name}")
    if config.architecture != "graph_transformer":
        return None
    if not config.gt_use_adj_bias:
        # predict_gamma was never built, so there is no gamma to report -- the whole
        # point of --gt-no-adj-bias is that those parameters do not exist.
        tqdm.write(f"skipping {name}: gt_use_adj_bias=False, the model has no gamma")
        return None

    name_radius = _radius_from_name(name)
    json_radius = _metadata_radius(path.parent.parent, name)
    if json_radius is not None and json_radius != name_radius:
        raise SystemExit(
            f"window radius disagreement for {name}: name implies {name_radius:g} nm, "
            f"{name}.json says {json_radius:g} nm"
        )

    model = WindowClassifier(config, hierarchy=hierarchy).to(device)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        raise SystemExit(
            f"{name}: {path} does not fit the current hierarchy "
            f"({len(hierarchy.level_classes[-1])} classes) -- it was trained against a "
            f"different DROP_LABELS. Retrain it or exclude it with --runs.\n  {exc}"
        ) from exc
    model.eval()

    run = LoadedRun(
        name=name, window_nm=name_radius, best_epoch=ckpt.get("epoch"), model=model,
        config=config,
        stats={
            (layer, row_type): RowStats(device=device, log_gamma=config.gt_use_exp)
            for layer in range(config.gt_depth) for row_type in ("cls", "node")
        },
    )
    for layer, block in enumerate(model.graph_transformer.blocks):
        run.handles.append(block.attn.register_forward_hook(make_hook(run, layer)))
    tqdm.write(
        f"loaded {name} (best epoch {run.best_epoch}, depth {config.gt_depth}, "
        f"heads {config.gt_heads}, scope {config.gt_attention_scope}) onto {device}"
    )
    return run


@torch.no_grad()
def probe_radius(runs: list[LoadedRun], manifest, hierarchy, window_nm: float, args) -> None:
    use_thickness = any(run.config.gt_use_thickness for run in runs)
    pos_dims = {run.config.gt_pos_dim for run in runs}
    if len(pos_dims) != 1:
        raise SystemExit(f"runs at {window_nm:g} nm disagree on gt_pos_dim: {sorted(pos_dims)}")
    dataset = WindowedGraphDatasetLCPN(
        manifest, "test", pos_dim=pos_dims.pop(), use_thickness=use_thickness,
        window_nm=window_nm,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    total = len(loader) if args.limit_batches is None else min(len(loader), args.limit_batches)
    bar = tqdm(loader, total=total, desc=f"gamma @ {window_nm / 1000:g}um", unit="batch")
    for batch_idx, batch in enumerate(bar):
        if args.limit_batches is not None and batch_idx >= args.limit_batches:
            break
        batch = batch.to(args.device)
        thickness = getattr(batch, "thickness", None)
        for run in runs:
            # The hooks do the recording; the returned embedding is not needed at all,
            # and no head is run -- gamma lives entirely inside the attention blocks.
            run.model(
                batch.x, batch.edge_index, batch.batch, pos_enc=batch.pos_enc,
                rel_pos=batch.rel_pos, thickness=thickness,
            )
        bar.set_postfix(batch=batch_idx + 1)
    bar.close()


def main(args) -> int:
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)

    runs = []
    for name, path in discover_gt_runs(results_dir, args.runs):
        run = load_run(name, path, hierarchy, args.device)
        if run is not None:
            runs.append(run)
    if not runs:
        raise SystemExit("no GraphTransformer checkpoint with an adjacency bias was found")

    by_radius: dict[float, list[LoadedRun]] = {}
    for run in runs:
        by_radius.setdefault(run.window_nm, []).append(run)
    tqdm.write(
        f"device={args.device}; runs={len(runs)}; radii="
        + ", ".join(f"{r / 1000:g}um x{len(v)}" for r, v in sorted(by_radius.items()))
    )
    for window_nm, group in sorted(by_radius.items()):
        probe_radius(group, manifest, hierarchy, window_nm, args)
    for run in runs:
        for handle in run.handles:
            handle.remove()

    rows = []
    for run in runs:
        for (layer, row_type), stats in sorted(run.stats.items()):
            rows.append({
                "run": run.name, "window_nm": run.window_nm, "best_epoch": run.best_epoch,
                "cls_resnet": run.config.cls_head_resnet,
                "frozen_agg": "_frozenagg" in run.name,
                "layer": layer, "row_type": row_type, **stats.report(),
            })

    out_dir = results_dir / "gamma_weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "gamma_weights" if args.limit_batches is None else f"gamma_weights_limit{args.limit_batches}batches"
    csv_path, json_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"limit_batches": args.limit_batches, "rows": rows}, indent=2))
    tqdm.write(f"wrote {csv_path}")
    tqdm.write(f"wrote {json_path}")

    tqdm.write("")
    tqdm.write("=== median gamma per layer (node rows; ratio = gamma_1 / gamma_0) ===")
    for run in runs:
        tqdm.write(f"{run.name}  (best epoch {run.best_epoch})")
        for row_type in ("node", "cls"):
            for row in rows:
                if row["run"] != run.name or row["row_type"] != row_type:
                    continue
                tqdm.write(
                    f"  L{row['layer']} {row_type:4s} "
                    f"g0={row['gamma_0_p50']:7.3f} g1={row['gamma_1_p50']:7.3f} "
                    f"ratio={row['ratio_p50']:8.3f}  "
                    f"nbr mass {row['nbr_mass_unbiased_mean']:.3f}->"
                    f"{row['nbr_mass_biased_mean']:.3f} "
                    f"(uniform {row['nbr_share_uniform_mean']:.3f})"
                )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", help="explicit result-directory names to probe")
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    cli_args = parser.parse_args()
    if cli_args.limit_batches is not None and cli_args.limit_batches < 1:
        parser.error("--limit-batches must be at least 1")
    raise SystemExit(main(cli_args))
