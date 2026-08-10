"""Supervised classification training for gnn/model.py::WindowClassifier.

Three aggregation methods, chosen with --architecture, and that choice is
the only thing that differs between a run of each:

  --architecture graph_transformer  (default) gnn/graph_transformer.py's
      AC-attention GraphTransformer. Four independent ablation switches:
      --gt-no-lpe, --gt-no-rel-pos, --gt-no-adj-bias, --gt-dist-bias, and
      --gt-attention-scope {global,neighborhood}, plus the off-by-default
      --gt-use-thickness node feature. Enabled switches are appended to the
      run name, so they never overwrite the full run.
  --architecture mpnn  gnn/encoder.py::MPNNEncoder -- plain GraphSAGE message
      passing, no attention, 2 layers by default -- followed by MeanReadout.
      One opt-in switch, --mpnn-lpe, concatenating the per-window Laplacian
      positional encoding onto the raw node features; it tags the run _lpe.
  --architecture mean  gnn/readout.py::MeanReadout straight over the raw
      per-node embeddings, no encoder -- the mean-pooling BASELINE.

All three run through this exact same pipeline: same windows, same LCPNHead,
same eval, so a comparison isolates the aggregation method and nothing else.
They form a ladder of how much learned mixing happens before the readout:
none, fixed local neighbor averaging over a few hops, or adjacency-biased
global attention.

Every run trains from scratch on the classification objective alone: there is
no pretraining stage and no checkpoint loading.

--cls-resnet swaps the classification head from a linear probe to the lab's
own shared ResNet backbone (gnn/resnet.py) feeding the per-node LCPN heads --
their `local_classifier_resnet_sngp`, minus SNGP. Orthogonal to
--architecture, so it composes with all three aggregation methods.

Trains and evaluates on per-window local subgraphs (data/dataset_windowed.py),
not whole cells -- see CLAUDE.md's project-goal section: the baseline
classifies per point from a small context window then majority-votes up to a
cell-level answer, and the GNN's classifier does the same here. Every batch
therefore produces one LCPN prediction per WINDOW; cell-level metrics come
from majority-voting those predictions by root_id
(gnn/metrics.py::majority_vote_by_group), exactly like the real baseline's
cell_level_accuracy. Window-level metrics are also reported, as a
diagnostic, not the headline number.

Run via sbatch (mit_normal_gpu):
    python scripts/train_gnn.py                          # GraphTransformer (default)
    python scripts/train_gnn.py --architecture mpnn      # 2-layer GraphSAGE + mean
    python scripts/train_gnn.py --architecture mpnn --mpnn-lpe  # -> ..._mpnn_L2_lpe
    python scripts/train_gnn.py --architecture mean      # mean-pool baseline
    python scripts/train_gnn.py --gt-no-lpe              # -> gnn_lcpn_scratch_gt_L4_H4_nolpe
    python scripts/train_gnn.py --gt-attention-scope neighborhood
    python scripts/train_gnn.py --gt-use-thickness       # -> ..._gt_L4_H4_thick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.dataset_lcpn import (  # noqa: E402
    load_hierarchy,
    load_manifest,
    train_window_counts_by_label,
)
from data.dataset_windowed import WindowedGraphDatasetLCPN  # noqa: E402
from gnn.lcpn import compute_node_class_weights  # noqa: E402
from gnn.metrics import majority_vote_by_group, summarize  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (finest_level_labels, finest_level_preds, root_ids), all at
    WINDOW granularity -- one entry per window subgraph, not per cell.
    LCPNHead's top-down cascade gives predictions at every level; the finest
    level is what's compared against the baseline's cell-level accuracy."""
    model.eval()
    preds, labels, root_ids = [], [], []
    for data in loader:
        data = data.to(device)
        g = model(
            data.x, data.edge_index, data.batch,
            pos_enc=data.pos_enc, rel_pos=data.rel_pos,
            thickness=getattr(data, "thickness", None), edge_attr=data.edge_attr,
        )
        level_preds = model.cls_head.predict_top_down(g)
        preds.append(level_preds[:, -1].cpu().numpy())
        labels.append(data.y_levels[:, -1].cpu().numpy())
        root_ids.append(data.root_id.cpu().numpy().reshape(-1))
    return np.concatenate(labels), np.concatenate(preds), np.concatenate(root_ids)


def cell_level_metrics(labels, preds, root_ids, num_classes, classes):
    """Window-level predictions -> majority vote per cell -> summarize().
    Same two-stage design the baseline uses."""
    cell_true, cell_pred = majority_vote_by_group(root_ids, labels, preds)
    return summarize(cell_true, cell_pred, num_classes, classes)


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)

    # There is no separate val fraction: "val" is an alias for the test split
    # (see data/build_dataset_from_store.py's module docstring), not a second
    # held-out partition. val_ds IS test_ds, not a second load of the same
    # cells -- avoids doubling the eager whole-split load
    # WindowedGraphDatasetLCPN does, and keeps it honest that checkpoint
    # selection and final test metrics are computed over identical cells.
    # One flag drives both the dataset and the model so they cannot drift:
    # the dataset attaches the feature iff the model is configured to read it.
    use_thickness = args.gt_use_thickness
    train_ds = WindowedGraphDatasetLCPN(
        manifest, "train", pos_dim=args.gt_pos_dim, use_thickness=use_thickness
    )
    test_ds = WindowedGraphDatasetLCPN(
        manifest, "test", pos_dim=args.gt_pos_dim, use_thickness=use_thickness
    )
    val_ds = test_ds
    classes = train_ds.classes  # finest-level names, for summarize()'s per_class_recall keys
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} windows, classes={classes}")

    # num_workers=0 (the DataLoader default) serializes all window extraction
    # on one core and leaves the GPU idle waiting -- measured as the actual
    # bottleneck, not batch size. See CLAUDE.md's DataLoader-throughput note.
    loader_kwargs = dict(num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    val_loader = test_loader  # val_ds is test_ds -- see the dataset construction comment above

    config = ModelConfig(
        in_dim=train_ds[0].x.shape[1],
        architecture=args.architecture,
        cls_head_hidden_dim=args.cls_hidden_dim,
        cls_head_resnet=args.cls_resnet,
        cls_resnet_hidden=args.cls_resnet_hidden,
        cls_resnet_layers=args.cls_resnet_layers,
        cls_resnet_dropout=args.cls_resnet_dropout,
        mpnn_hidden_dim=args.mpnn_hidden_dim,
        mpnn_out_dim=args.mpnn_hidden_dim,
        mpnn_layers=args.mpnn_layers,
        mpnn_dropout=args.mpnn_dropout,
        mpnn_use_lpe=args.mpnn_lpe,
        gt_dim=args.gt_dim,
        gt_depth=args.gt_depth,
        gt_heads=args.gt_heads,
        gt_mlp_ratio=args.gt_mlp_ratio,
        gt_pos_dim=args.gt_pos_dim,
        gt_use_exp=not args.gt_no_exp,
        gt_dropout=args.gt_dropout,
        gt_use_lpe=not args.gt_no_lpe,
        gt_use_rel_pos=not args.gt_no_rel_pos,
        gt_use_adj_bias=not args.gt_no_adj_bias,
        gt_attention_scope=args.gt_attention_scope,
        gt_use_dist_bias=args.gt_dist_bias,
        gt_use_thickness=use_thickness,
    )
    model = WindowClassifier(config, hierarchy=hierarchy).to(device)

    if not args.no_class_weights:
        # Per-node inverse-frequency weights, computed from TRAIN-split
        # window counts only (data/dataset_lcpn.py::train_window_counts_by_label)
        # -- see gnn/lcpn.py::compute_node_class_weights for why this exists:
        # unweighted, the LCPN loss just learns to predict populous classes
        # (severe imbalance -- L4IT ~2.45M windows vs. singleton classes --
        # gives no gradient pressure to learn the rare ones otherwise).
        node_weights = compute_node_class_weights(hierarchy, train_window_counts_by_label(manifest))
        model.cls_head.set_class_weights(node_weights)
        print("applied per-node class weights (train-split window counts)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # agg_tag identifies the aggregation method -- "meanpool" for the
    # mean-readout baseline, "mpnn_L{layers}" for the message-passing encoder,
    # "gt_L{depth}_H{heads}" for the AC-attention GraphTransformer -- so runs
    # that only differ by aggregation don't collide on the same checkpoint
    # dir / results file. The best-epoch
    # checkpoint is written to disk as soon as a new best is found, not just
    # kept in memory until the loop ends -- a killed/preempted job before the
    # last epoch used to lose the best state entirely.
    if args.architecture == "mean":
        agg_tag = "meanpool"
    elif args.architecture == "mpnn":
        # _lpe for the same reason the GraphTransformer's switches are tagged:
        # without it, --mpnn-lpe would land on the plain mpnn_L2 run's
        # directory and overwrite its epoch_metrics.csv.
        agg_tag = f"mpnn_L{args.mpnn_layers}" + ("_lpe" if args.mpnn_lpe else "")
    else:
        # Ablation switches go into the tag too -- without them, a full run
        # and any of its ablations would collide on one checkpoint dir and
        # silently overwrite each other's epoch_metrics.csv.
        agg_tag = f"gt_L{args.gt_depth}_H{args.gt_heads}"
        if args.gt_attention_scope == "neighborhood":
            agg_tag += "_nbhd"
        for flag, suffix in (
            (args.gt_no_lpe, "_nolpe"),
            (args.gt_no_rel_pos, "_norelpos"),
            (args.gt_no_adj_bias, "_noadjbias"),
            (args.gt_dist_bias, "_distbias"),
            (args.gt_use_thickness, "_thick"),
        ):
            if flag:
                agg_tag += suffix
    # Appended outside the architecture branch above: the head choice applies
    # to every architecture, so without it a --cls-resnet run would land on the
    # linear-probe run's directory and overwrite its epoch_metrics.csv.
    if args.cls_resnet:
        agg_tag += f"_resnet{args.cls_resnet_layers}x{args.cls_resnet_hidden}"
    run_name = f"gnn_lcpn_scratch_{agg_tag}"
    ckpt_dir = Path(__file__).resolve().parent.parent / "results" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- resume -----------------------------------------------------------
    # checkpoint_last.pt is written at the end of EVERY epoch, unlike
    # checkpoint_best.pt which is written only on improvement, and it carries
    # the optimizer and RNG state as well as the weights -- so a resumed run
    # continues the same trajectory instead of restarting Adam's moments from
    # zero. Preemption therefore costs at most one epoch rather than the whole
    # run, which is what makes long runs viable on mit_preemptable and makes a
    # 100-epoch run survivable inside mit_normal_gpu's 6h walltime cap across
    # successive submissions.
    last_path = ckpt_dir / "checkpoint_last.pt"
    start_epoch, best_val_bacc = 0, -1.0
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        # A mismatched config means this directory belongs to a different model
        # than the one just built -- resuming would silently load foreign
        # weights, or fail deep inside load_state_dict with an unhelpful shape
        # error. Refuse up front instead.
        if ckpt["config"] != config:
            raise SystemExit(
                f"--resume: {last_path} was written by a different ModelConfig.\n"
                f"  on disk: {ckpt['config']}\n"
                f"  now:     {config}\n"
                "Delete the directory to start fresh, or fix the flags."
            )
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_bacc = ckpt["best_val_bacc"]
        # .cpu() is load-bearing, not defensive: map_location=device above sends
        # EVERY tensor in the checkpoint to the GPU, RNG states included, and
        # both set_rng_state calls require a CPU ByteTensor -- a CUDA one raises
        # "RNG state must be a torch.ByteTensor". This only ever fires on a real
        # GPU resume, so it is invisible until the first preemption of a real
        # run (which is exactly when it costs the most).
        torch.set_rng_state(ckpt["cpu_rng_state"].cpu())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
        print(
            f"resumed from {last_path}: starting at epoch {start_epoch}, "
            f"best val_window_bacc so far {best_val_bacc:.4f}"
        )
        if start_epoch >= args.epochs:
            print(f"already at epoch {start_epoch} of {args.epochs} -- nothing left to train")
    elif args.resume:
        print(f"--resume given but {last_path} does not exist -- starting from epoch 0")

    # Per-epoch CSV -- everything needed to make figures later without
    # re-running anything: loss + window-level and cell-level accuracy/
    # balanced_accuracy/macro_precision/macro_f1, plus per-class recall AND
    # precision at both granularities. Per-class F1 is deliberately NOT
    # logged here -- it's cheaply derived in the analysis notebook from
    # recall+precision. Appended one row per epoch (opened fresh each time,
    # not held open for the whole run) so a killed/preempted job still leaves
    # every completed epoch's data on disk.
    csv_path = ckpt_dir / "epoch_metrics.csv"
    csv_fields = (
        ["epoch", "train_loss"]
        + [f"window_{k}" for k in ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1")]
        + [f"cell_{k}" for k in ("accuracy", "balanced_accuracy", "macro_precision", "macro_f1")]
        + [f"window_recall_{c}" for c in classes]
        + [f"cell_recall_{c}" for c in classes]
        + [f"window_precision_{c}" for c in classes]
        + [f"cell_precision_{c}" for c in classes]
    )
    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()
    else:
        # Resuming: keep only rows for epochs preceding where we restart, then
        # append from there. checkpoint_last.pt is saved AFTER the CSV row for
        # the same epoch, so the CSV is never behind the checkpoint -- but it
        # can be one row ahead (killed between the two writes), and its final
        # row can be torn (killed mid-append). Rewriting from the parsed rows
        # repairs both, and keeps epoch numbers unique so the analysis notebook
        # doesn't see duplicates.
        kept = []
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        if int(row["epoch"]) < start_epoch:
                            kept.append(row)
                    except (TypeError, ValueError):
                        continue  # torn final row from a mid-write kill
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(kept)
        print(f"resumed epoch_metrics.csv with {len(kept)} prior epochs retained")

    def _append_csv_row(epoch, train_loss, window_metrics, cell_metrics):
        row = {
            "epoch": epoch, "train_loss": train_loss,
            "window_accuracy": window_metrics["accuracy"],
            "window_balanced_accuracy": window_metrics["balanced_accuracy"],
            "window_macro_precision": window_metrics["macro_precision"],
            "window_macro_f1": window_metrics["macro_f1"],
            "cell_accuracy": cell_metrics["accuracy"],
            "cell_balanced_accuracy": cell_metrics["balanced_accuracy"],
            "cell_macro_precision": cell_metrics["macro_precision"],
            "cell_macro_f1": cell_metrics["macro_f1"],
        }
        for c in classes:
            row[f"window_recall_{c}"] = window_metrics["per_class_recall"][c]
            row[f"cell_recall_{c}"] = cell_metrics["per_class_recall"][c]
            row[f"window_precision_{c}"] = window_metrics["per_class_precision"][c]
            row[f"cell_precision_{c}"] = cell_metrics["per_class_precision"][c]
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

    # best_val_bacc comes from the resume block above (-1.0 on a fresh run).
    # The best weights live on disk in checkpoint_best.pt rather than in an
    # in-memory `best_state`: a resumed run may never beat a best set before
    # the interruption, so the final model has to be reloadable from disk.
    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        desc="train", unit="epoch", initial=start_epoch, total=args.epochs,
    )
    for epoch in epoch_bar:
        model.train()
        total_loss, n = 0.0, 0
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch}", unit="batch", leave=False)
        for data in batch_bar:
            data = data.to(device)
            opt.zero_grad()
            g = model(
                data.x, data.edge_index, data.batch,
                pos_enc=data.pos_enc, rel_pos=data.rel_pos,
                thickness=getattr(data, "thickness", None), edge_attr=data.edge_attr,
            )
            loss = model.cls_head.compute_loss(g, data.y_levels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.num_graphs
            n += data.num_graphs
            batch_bar.set_postfix(loss=f"{total_loss / max(1, n):.4f}")

        val_labels, val_preds, val_root_ids = evaluate(model, val_loader, device)
        # Window-level metrics via the SAME summarize() the cell-level ones
        # use -- lets us see whether imbalance bias shows up at the per-window
        # classification step itself, independent of what majority voting
        # does to it afterward.
        window_val_metrics = summarize(val_labels, val_preds, len(classes), classes)
        window_val_acc = window_val_metrics["accuracy"]
        val_metrics = cell_level_metrics(val_labels, val_preds, val_root_ids, len(classes), classes)
        # Checkpoint selection uses WINDOW balanced accuracy, not cell -- cell
        # metrics come from majority-voting a few hundred val cells, which is
        # small-sample and genuinely noisy epoch to epoch, whereas window
        # metrics average over ~1.8M val windows. The window metric is also
        # what the per-window training loss is directly shaped by, without the
        # nonlinear transform majority voting adds on top.
        #
        # NOTE: val_ds IS test_ds (see its construction above) -- checkpoint
        # selection is therefore not held out from the final reported test
        # metrics below. Accepted trade-off of the two-way split, in exchange
        # for both partitions getting the full 20% of held-out cells.
        if window_val_metrics["balanced_accuracy"] > best_val_bacc:
            best_val_bacc = window_val_metrics["balanced_accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(), "config": config,
                    "epoch": epoch, "val_window_balanced_accuracy": best_val_bacc,
                },
                ckpt_dir / "checkpoint_best.pt",
            )
            tqdm.write(f"  new best val_window_bacc={best_val_bacc:.3f} at epoch {epoch} -> checkpoint_best.pt")
        epoch_bar.set_postfix(
            train_loss=f"{total_loss / max(1, n):.4f}",
            val_cell_bacc=f"{val_metrics['balanced_accuracy']:.3f}",
            val_cell_acc=f"{val_metrics['accuracy']:.3f}",
        )
        # Every epoch, not throttled -- epochs are cheap (~1.5-3min with the
        # DataLoader worker settings), and per-epoch visibility into the val
        # curve makes convergence readable instead of guessed at.
        tqdm.write(
            f"epoch {epoch:4d}  train_loss={total_loss / max(1, n):.4f}  "
            f"val_window_acc={window_val_acc:.3f}  val_window_bacc={window_val_metrics['balanced_accuracy']:.3f}  "
            f"val_cell_bacc={val_metrics['balanced_accuracy']:.3f}  val_cell_acc={val_metrics['accuracy']:.3f}"
        )
        _append_csv_row(epoch, total_loss / max(1, n), window_val_metrics, val_metrics)

        # Written LAST, after the CSV row for this epoch, so the checkpoint can
        # never claim an epoch the CSV has no row for. Saved to a temp path and
        # renamed, since os.replace is atomic on POSIX -- being killed partway
        # through this write would otherwise leave a truncated file and cost
        # the whole run rather than one epoch.
        tmp_path = last_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_val_bacc": best_val_bacc,
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
            tmp_path,
        )
        os.replace(tmp_path, last_path)

    # Best weights from disk, not from an in-memory copy -- see the note above
    # the epoch loop. Absent only if the run trained zero epochs (already
    # complete on resume), in which case whatever is in memory is what we have.
    if (ckpt_dir / "checkpoint_best.pt").exists():
        best_ckpt = torch.load(ckpt_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])
        print(f"loaded best checkpoint from epoch {best_ckpt['epoch']} for final test evaluation")
    test_labels, test_preds, test_root_ids = evaluate(model, test_loader, device)
    window_test_metrics = summarize(test_labels, test_preds, len(classes), classes)
    test_metrics = cell_level_metrics(test_labels, test_preds, test_root_ids, len(classes), classes)
    print("=== test metrics (GNN) ===")
    print(f"window-level accuracy: {window_test_metrics['accuracy']:.4f}  balanced_accuracy: {window_test_metrics['balanced_accuracy']:.4f}")
    print(json.dumps(test_metrics, indent=2))

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{run_name}.json"
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "window_test_metrics": window_test_metrics,
                "test_metrics": test_metrics,
                "classes": classes,
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--architecture", default="graph_transformer",
        choices=["graph_transformer", "mpnn", "mean"],
        help="graph_transformer (default): gnn/graph_transformer.py's AC-attention "
             "GraphTransformer. mpnn: gnn/encoder.py::MPNNEncoder, plain GraphSAGE message "
             "passing (no attention) + MeanReadout. mean: MeanReadout over the raw node "
             "embeddings, no encoder -- the mean-pooling baseline.",
    )
    p.add_argument(
        "--mpnn-layers", type=int, default=2,
        help="message-passing hops for --architecture mpnn; 2 by default because windows "
             "average ~10.7 nodes and deeper stacks over-smooth a graph that small",
    )
    p.add_argument("--mpnn-hidden-dim", type=int, default=128)
    p.add_argument("--mpnn-dropout", type=float, default=0.1)
    p.add_argument(
        "--mpnn-lpe", action="store_true",
        help="concatenate the per-window Laplacian positional encoding onto the raw node "
             "features before the first SAGEConv (--architecture mpnn only). OFF by default, "
             "unlike the GraphTransformer's LPE, so the recorded mpnn_L{layers} runs stay "
             "reproducible; enabling it tags the run _lpe. Width comes from --gt-pos-dim, "
             "which is the one pos_dim the dataset itself was built with.",
    )
    p.add_argument(
        "--cls-hidden-dim", type=int, default=None,
        help="LCPN head hidden layer size; default None = plain Linear per node",
    )
    p.add_argument(
        "--cls-resnet", action="store_true",
        help="put a shared ResNet backbone (gnn/resnet.py::DeepResNetTrunk, ported from the "
             "lab's segCLR_cell_classification) between the readout and the per-node LCPN "
             "heads, instead of a linear probe. Reproduces their own "
             "local_classifier_resnet_sngp arrangement minus SNGP. Orthogonal to "
             "--architecture: works with mean, mpnn and graph_transformer alike.",
    )
    p.add_argument("--cls-resnet-hidden", type=int, default=128,
                   help="ResNet trunk width (their configs/local_classifier_sngp.yaml: 128)")
    p.add_argument("--cls-resnet-layers", type=int, default=4,
                   help="ResNet trunk residual blocks (theirs: 4)")
    p.add_argument("--cls-resnet-dropout", type=float, default=0.0)
    p.add_argument(
        "--no-class-weights", action="store_true",
        help="disable per-node inverse-frequency class weighting (on by default -- see "
             "gnn/lcpn.py::compute_node_class_weights)",
    )
    p.add_argument("--gt-dim", type=int, default=128, help="GraphTransformer hidden width")
    p.add_argument("--gt-depth", type=int, default=4, help="number of AC-attention blocks")
    p.add_argument("--gt-heads", type=int, default=4, help="attention heads per block")
    p.add_argument("--gt-mlp-ratio", type=int, default=4)
    p.add_argument(
        "--gt-pos-dim", type=int, default=8,
        help="width of the per-window Laplacian positional encoding "
             "(data/geodesic_window.py's DEFAULT_POS_DIM) -- must match what the dataset "
             "was constructed with; passed through to WindowedGraphDatasetLCPN here so "
             "they can't drift apart.",
    )
    p.add_argument(
        "--gt-no-exp", action="store_true",
        help="disable exp() on GraphAttention's predicted local/global trade-off (gamma) -- "
             "on by default, matching the ssl_neuron reference (keeps both weights positive)",
    )
    p.add_argument("--gt-dropout", type=float, default=0.0)
    # --- GraphTransformer ablation switches (all default to the full model) ---
    # Each is an OFF switch, so the default run is unchanged by their presence.
    # Any combination that is enabled gets appended to the run name (see
    # agg_tag above), so ablations never overwrite the full run's results.
    p.add_argument(
        "--gt-no-lpe", action="store_true",
        help="drop the per-window Laplacian positional encoding (the additive pos_enc term)",
    )
    p.add_argument(
        "--gt-no-rel-pos", action="store_true",
        help="drop the center-relative geometry concatenated onto the node features -- "
             "dx, dy, dz and their norm, all 4 channels together",
    )
    p.add_argument(
        "--gt-no-adj-bias", action="store_true",
        help="drop GraphDINO's additive gamma_1 * adj attention bias, leaving plain scaled "
             "dot-product attention (the learned per-node gamma_0 temperature goes with it)",
    )
    p.add_argument(
        "--gt-dist-bias", action="store_true",
        help="replace the binary adjacency in the attention bias term with a learned per-head "
             "scalar indexed by binned edge length (gnn/graph_transformer.py's "
             "DIST_BIAS_BOUNDARIES_NM). A strict generalization -- initialized to reproduce "
             "binary adjacency exactly -- so it starts as a no-op and learns away. Requires "
             "the adjacency bias, i.e. incompatible with --gt-no-adj-bias.",
    )
    p.add_argument(
        "--gt-use-thickness", action="store_true",
        help="concatenate the spine-corrected dendrite shaft radius (+ a measured flag) onto "
             "the node features. OFF by default, unlike the other switches, because it needs "
             "data/dendrite_thickness_cache/*.npz ingested "
             "(scripts/sbatch/build_dendrite_thickness.sh). This single flag also turns on the "
             "dataset side, so the two can't drift apart. Only --architecture "
             "graph_transformer consumes it.",
    )
    p.add_argument(
        "--gt-attention-scope", default="global", choices=["global", "neighborhood"],
        help="global (default): each node attends anywhere in the window, with adjacency "
             "entering only as a soft bias. neighborhood: hard -inf mask restricting "
             "attention to 1-hop graph neighbors (the CLS token stays fully connected, or "
             "the readout would see nothing). NOTE: under neighborhood scope the adjacency "
             "bias is nearly inert -- see gnn/graph_transformer.py's class docstring.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4096,
        help="number of WINDOW subgraphs per batch (not whole cells)",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--resume", action="store_true",
        help="continue from results/<run>/checkpoint_last.pt if it exists, restoring model, "
             "optimizer and RNG state and truncating epoch_metrics.csv to match. Starts from "
             "epoch 0 if no such file exists, so it is safe to leave on permanently -- which is "
             "what makes a preempted job recover on requeue instead of restarting. Refuses to "
             "resume across a changed ModelConfig.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument(
        "--num-workers", type=int, default=15,
        help="DataLoader worker processes for window extraction -- keep one below the job's "
             "--cpus-per-task, leaving a core for the main process (see CLAUDE.md)",
    )
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
