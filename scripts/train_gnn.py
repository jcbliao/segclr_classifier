"""Supervised classification training for gnn/model.py::WindowClassifier.

Five aggregation methods, chosen with --architecture, and that choice is
the only thing that differs between a run of each:

  --architecture graph_transformer  (default) gnn/graph_transformer.py's
      AC-attention GraphTransformer. Four independent ablation switches:
      --gt-no-lpe, --gt-no-rel-pos, --gt-no-adj-bias, and
      --gt-attention-scope {global,neighborhood}, plus the off-by-default
      --gt-use-thickness node feature. Enabled switches are appended to the
      run name, so they never overwrite the full run.
  --architecture mpnn  gnn/encoder.py::MPNNEncoder -- plain GraphSAGE message
      passing, no attention, 2 layers by default -- followed by MeanReadout.
  --architecture mpnn_complete  the same encoder over a per-window clique:
      message passing with the skeleton structure taken away.
  --architecture pointwise_mlp  gnn/pointwise_mlp.py::PointwiseMLPEncoder -- a per-node MLP
      (64 -> 128 -> 128), then the mean, then the ordinary head as rho. No
      attention, no positional features, no graph.
  --architecture linear  gnn/linear_encoder.py::LinearEncoder -- one per-node
      Linear (64 -> 128), no nonlinearity, then the mean. Embeddings only.
      The mean absorbs a node-wise linear map, so this is the control that
      isolates the pointwise MLP's nonlinearity rather than a rung above it.
  --architecture mean  gnn/readout.py::MeanReadout straight over the raw
      per-node embeddings, no encoder -- the mean-pooling BASELINE.

All six run through this exact same pipeline: same windows, same LCPNHead,
same eval, so a comparison isolates the aggregation method and nothing else.
They form a ladder of how much learned mixing happens before the readout:
none, a learned per-node transform, fixed local neighbor averaging over a few
hops, that same averaging over a clique, or adjacency-biased global
attention.

Every run trains from scratch on the classification objective alone: there is
no pretraining stage and no checkpoint loading.

--cls-resnet swaps the classification head from a linear probe to the lab's
own shared ResNet backbone (gnn/resnet.py) feeding the per-node LCPN heads --
their `local_classifier_resnet_sngp`, minus SNGP. Orthogonal to
--architecture, so it composes with every aggregation method.

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
    python scripts/train_gnn.py --architecture pointwise_mlp  # per-node MLP + mean
    python scripts/train_gnn.py --architecture linear    # one per-node Linear + mean
    python scripts/train_gnn.py --architecture mean      # mean-pool baseline
    python scripts/train_gnn.py --gt-no-lpe              # -> ..._gt_L4_H4_nolpe_..._fold0
    python scripts/train_gnn.py --gt-attention-scope neighborhood
    python scripts/train_gnn.py --gt-use-thickness       # -> ..._gt_L4_H4_thick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
)
from data.geodesic_window import DEFAULT_WINDOW_NM  # noqa: E402
from data.dataset_windowed import WindowedGraphDatasetLCPN, balanced_sampler  # noqa: E402
from data.dataset_presynaptic import (  # noqa: E402
    AttentionBudgetBatchSampler,
    PresynapticWindowDataset,
)
from data.mixed_cell_batch_sampler import MixedCellBatchSampler  # noqa: E402
from data.window_prediction_cache import save_prediction_cache  # noqa: E402
from gnn.metrics import majority_vote_by_group, summarize  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402


def _seed_loader_worker(worker_id: int) -> None:
    """Give Python/NumPy augmentation draws the worker's PyTorch seed."""
    del worker_id
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False):
    """Returns (finest_level_labels, finest_level_preds, root_ids), all at
    WINDOW granularity -- one entry per window subgraph, not per cell.
    LCPNHead's top-down cascade gives predictions at every level; the finest
    level is what's compared against the baseline's cell-level accuracy."""
    model.eval()
    preds, labels, root_ids = [], [], []
    for data in loader:
        data = data.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            g = model(
                data.x, data.edge_index, data.batch,
                pos_enc=data.pos_enc, rel_pos=data.rel_pos,
                thickness=getattr(data, "thickness", None),
                has_segclr=getattr(data, "has_segclr", None),
                postsynaptic_embedding=getattr(data, "postsynaptic_embedding", None),
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


def publish_test_prediction_cache(run_name, dataset, predictions, targets, num_embeddings):
    """Publish final held-out predictions without another inference pass."""
    if num_embeddings is None:
        print("radius-dataset run: shared fixed-window prediction cache not applicable")
        return
    if len(predictions) != len(dataset):
        raise RuntimeError(
            f"cannot cache {run_name}: {len(predictions)} predictions for {len(dataset)} windows"
        )
    xyz = np.empty((len(dataset), 3), np.float32)
    for root_id in np.unique(dataset.index_root_ids):
        rows = np.flatnonzero(dataset.index_root_ids == root_id)
        xyz[rows] = dataset.cell_data[int(root_id)].pos[
            dataset.index_centers[rows]
        ].numpy()
    path = save_prediction_cache(run_name, {
        "split": np.full(len(dataset), "test", dtype="U5"),
        "root_id": dataset.index_root_ids.astype(np.uint64),
        "center_index": dataset.index_centers.astype(np.int32),
        "center_xyz": xyz,
        "prediction": np.asarray(predictions, dtype=np.int16),
        "target": np.asarray(targets, dtype=np.int16),
        "num_embeddings": np.array([num_embeddings], np.int16),
    })
    print(f"published shared window predictions to {path}")


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
    elif args.dataset == 'presynaptic_dense':
        manifest = json.loads((Path(args.presynaptic_database) / 'manifest.json').read_text())
    else:
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
    if args.dataset == "all_windows":
        if args.num_embeddings not in (10, 20, 40):
            raise SystemExit("--dataset all_windows requires --num-embeddings 10, 20, or 40")
        train_ds = WindowedGraphDatasetLCPN(
            manifest, "train", pos_dim=args.gt_pos_dim, use_thickness=use_thickness,
            window_nm=args.window_nm, num_embeddings=args.num_embeddings,
            neighborhood_root=args.neighborhood_root,
        )
        test_ds = WindowedGraphDatasetLCPN(
            manifest, "test", pos_dim=args.gt_pos_dim, use_thickness=use_thickness,
            window_nm=args.window_nm, num_embeddings=args.num_embeddings,
            neighborhood_root=args.neighborhood_root,
        )
    else:
        allowed = {
            "presynaptic_cave": {"graph_transformer", "mean", "pointwise_mlp"},
            "presynaptic_new": {"graph_transformer_teasar"},
            "presynaptic_dense": {"graph_transformer", "graph_transformer_teasar", "mean", "linear", "pointwise_mlp"},
        }[args.dataset]
        if args.architecture not in allowed:
            raise SystemExit(
                f"--dataset {args.dataset} allows --architecture "
                f"{sorted(allowed)}, not {args.architecture}"
            )
        if args.dataset == 'presynaptic_dense':
            metadata = json.loads((Path(args.presynaptic_database) / 'metadata.json').read_text())
            if metadata['format'] != 'dense-presynaptic-v1' or metadata['missing_root_ids']:
                raise SystemExit('Dense presynaptic database is not finalized/complete')
            expected_k = 1 if args.single_presynaptic_embedding else metadata['k_observed']
            if args.num_embeddings != expected_k:
                raise SystemExit(f"Use --num-embeddings {expected_k} for this configuration")
        elif args.num_embeddings != 10:
            raise SystemExit("presynaptic databases were built with K=10; use --num-embeddings 10")
        if args.gt_no_lpe or args.gt_no_rel_pos or args.gt_no_adj_bias:
            raise SystemExit(
                "presynaptic training requires the full model: LPE, relative position, "
                "and adjacency bias must all remain enabled"
            )
        if use_thickness:
            raise SystemExit("the presynaptic database does not contain thickness features")
        variant = "cave" if args.dataset == "presynaptic_cave" else "new"
        train_ds = PresynapticWindowDataset(
            manifest, "train", variant, database=args.presynaptic_database,
            cell_cache_size=args.presynaptic_cell_cache,
            postsynaptic_cache=args.postsynaptic_cache,
            use_postsynaptic=args.use_postsynaptic,
            single_presynaptic_embedding=args.single_presynaptic_embedding,
            memmap_root=args.presynaptic_memmap_root,
            compartment_filter=args.presynaptic_compartment_filter,
            embedding_augmentation=(None if args.embedding_augmentation in (None, "clean")
                                    else args.embedding_augmentation),
            augmentation_database=args.embedding_augmentation_database,
        )
        test_ds = PresynapticWindowDataset(
            manifest, "test", variant, database=args.presynaptic_database,
            cell_cache_size=args.presynaptic_cell_cache,
            postsynaptic_cache=args.postsynaptic_cache,
            use_postsynaptic=args.use_postsynaptic,
            single_presynaptic_embedding=args.single_presynaptic_embedding,
            memmap_root=args.presynaptic_memmap_root,
            compartment_filter=args.presynaptic_compartment_filter,
        )
    val_ds = test_ds
    hierarchy = train_ds.hierarchy
    classes = train_ds.classes  # finest-level names, for summarize()'s per_class_recall keys
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} windows, classes={classes}")

    # num_workers=0 (the DataLoader default) serializes all window extraction
    # on one core and leaves the GPU idle waiting -- measured as the actual
    # bottleneck, not batch size. See CLAUDE.md's DataLoader-throughput note.
    loader_kwargs = dict(num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    train_generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs.update(worker_init_fn=_seed_loader_worker, generator=train_generator)
    # --class-balance sample resamples the training windows instead of shuffling
    # them (see data/dataset_windowed.py::balanced_sampler); a sampler and
    # shuffle=True are mutually exclusive in DataLoader.
    if args.dataset == "all_windows":
        train_sampler = balanced_sampler(train_ds, power=1.0 if args.class_balance == "equal" else 0.5) if args.class_balance in ("sample", "equal") else None
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=train_sampler, shuffle=train_sampler is None, **loader_kwargs,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    else:
        sampler_cls = MixedCellBatchSampler if args.mixed_cell_batch_size else AttentionBudgetBatchSampler
        sampler_kwargs = ({"cells_per_batch": args.mixed_cell_batch_size}
                          if args.mixed_cell_batch_size else {})
        train_batch_sampler = sampler_cls(
            train_ds, attention_budget=args.attention_budget,
            max_windows=args.batch_size, shuffle=True, seed=args.seed,
            balance_classes=args.class_balance in ("sample", "equal"),
            class_balance_power=1.0 if args.class_balance == "equal" else 0.5,
            **sampler_kwargs,
        )
        test_batch_sampler = AttentionBudgetBatchSampler(
            test_ds, attention_budget=args.attention_budget,
            max_windows=args.batch_size, shuffle=False, seed=args.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, **loader_kwargs)
        test_loader = DataLoader(test_ds, batch_sampler=test_batch_sampler, **loader_kwargs)
    val_loader = test_loader  # val_ds is test_ds -- see the dataset construction comment above

    config = ModelConfig(
        in_dim=train_ds[0].x.shape[1],
        postsynaptic_dim=64 if args.use_postsynaptic else 0,
        append_presynaptic_mean=args.append_presynaptic_mean,
        architecture=args.architecture,
        classifier=args.classifier,
        cls_head_hidden_dim=args.cls_hidden_dim,
        cls_head_resnet=args.cls_resnet,
        cls_resnet_hidden=args.cls_resnet_hidden,
        cls_resnet_layers=args.cls_resnet_layers,
        cls_resnet_dropout=args.cls_resnet_dropout,
        use_spatial_features=args.spatial,
        use_position=args.position,
        use_lpe=args.lpe,
        use_embeddings=not args.no_embeddings,
        mpnn_hidden_dim=args.mpnn_hidden_dim,
        mpnn_out_dim=args.mpnn_hidden_dim,
        mpnn_layers=args.mpnn_layers,
        mpnn_dropout=args.mpnn_dropout,
        pointwise_mlp_hidden_dim=args.pointwise_mlp_hidden_dim,
        pointwise_mlp_out_dim=args.pointwise_mlp_hidden_dim,
        pointwise_mlp_layers=args.pointwise_mlp_layers,
        linear_out_dim=args.linear_out_dim,
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
        gt_use_thickness=use_thickness,
    )
    model = WindowClassifier(config, hierarchy=hierarchy).to(device)

    if args.freeze_aggregator:
        n_frozen = model.freeze_aggregator()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"aggregator frozen at random init: {n_frozen:,} parameters held fixed, "
            f"{n_train:,} trainable (the classification head only)"
        )

    # Correct imbalance by changing which windows the model sees.  The loss
    # remains unweighted, matching segCLR_cell_classification's sampled local
    # classifier configuration.
    if args.class_balance == "equal":
        print("class balance: equal expected class shares, sampling with replacement (1/count)")
    elif args.class_balance == "sample":
        print("class balance: resampling train windows with replacement (1/sqrt(count), theirs)")
    else:
        print("class balance: none")

    # Frozen parameters are filtered out rather than left in with requires_grad
    # False: Adam would otherwise carry optimizer state for tensors it can
    # never update, and the count in --freeze-aggregator's log line would
    # disagree with what the optimizer actually holds.
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # agg_tag identifies the aggregation method -- "meanpool" for the
    # mean-readout baseline, "mpnn_L{layers}" for the message-passing encoder,
    # "gt_L{depth}_H{heads}" for the AC-attention GraphTransformer -- so runs
    # that only differ by aggregation don't collide on the same checkpoint
    # dir / results file. The best-epoch
    # checkpoint is written to disk as soon as a new best is found, not just
    # kept in memory until the loop ends -- a killed/preempted job before the
    # last epoch used to lose the best state entirely.
    if args.architecture == "pointwise_mlp":
        # No _position / _lpe variants to disambiguate: the model refuses those
        # switches for this architecture, so the depth is the only thing that
        # can vary between two pointwise_mlp runs.
        agg_tag = f"pointwise_mlp_L{args.pointwise_mlp_layers}"
    elif args.architecture == "linear":
        # No depth to record: it is one Linear by definition, and the same
        # refusals as pointwise_mlp mean there are no feature variants either.
        agg_tag = "linear"
    elif args.architecture in ("mean", "mpnn", "mpnn_complete"):
        if args.architecture == "mean":
            agg_tag = "mean"
        elif args.architecture == "mpnn_complete":
            agg_tag = f"mpnn_complete_L{args.mpnn_layers}"
        else:
            agg_tag = f"mpnn_L{args.mpnn_layers}"
        # Tagged for the same reason the GT ablations are: a --spatial run is a
        # different model on the same aggregator and must not land on the
        # raw-embedding run's directory.
        if args.position or args.spatial:
            agg_tag += "_position"
        if args.lpe or args.spatial:
            agg_tag += "_lpe"
    else:
        # Ablation switches go into the tag too -- without them, a full run
        # and any of its ablations would collide on one checkpoint dir and
        # silently overwrite each other's epoch_metrics.csv.
        gt_name = "gt_teasar" if args.architecture == "graph_transformer_teasar" else "gt"
        agg_tag = f"{gt_name}_L{args.gt_depth}_H{args.gt_heads}"
        if args.gt_attention_scope == "neighborhood":
            agg_tag += "_nbhd"
        for flag, suffix in (
            (args.gt_no_lpe, "_nolpe"),
            (args.gt_no_rel_pos, "_norelpos"),
            (args.gt_no_adj_bias, "_noadjbias"),
            (args.gt_use_thickness, "_thick"),
        ):
            if flag:
                agg_tag += suffix
    # Appended outside the architecture branch above: the head choice applies
    # to every architecture, so without it a --cls-resnet run would land on the
    # linear-probe run's directory and overwrite its epoch_metrics.csv.
    if args.cls_resnet:
        agg_tag += f"_resnet{args.cls_resnet_layers}x{args.cls_resnet_hidden}"
    # Window radius, likewise: the same aggregator over a 40um window is a
    # different model of the data than over 10um, and the two must not share a
    # results directory. The default radius stays untagged so the headline runs
    # keep the names the rest of the repo already refers to.
    if args.num_embeddings is not None:
        agg_tag += f"_n{args.num_embeddings}"
    elif args.window_nm != DEFAULT_WINDOW_NM:
        um = args.window_nm / 1000.0
        agg_tag += f"_w{um:g}um"
    # Applies to every architecture, like the head choice, so it is tagged
    # outside the aggregation branch -- a geometry-only run is a different
    # model on the same aggregator and must not land on its directory.
    if args.no_embeddings:
        agg_tag += "_noemb"
    # Same reasoning again: a frozen-aggregator run is the random-features
    # control for its architecture, not a rerun of it, and must not overwrite
    # the trained run's results.
    if args.freeze_aggregator:
        agg_tag += "_frozenagg"
    if args.mixed_cell_batch_size:
        agg_tag += f"_mixed{args.mixed_cell_batch_size}"
    if args.embedding_augmentation is not None:
        agg_tag += f"_embaug_{args.embedding_augmentation}"
    # Sampling changes the training distribution and therefore defines a
    # distinct run.  Presynaptic runs historically omitted this tag while
    # using loss weighting, so tag corrected runs to prevent --resume from
    # attaching them to those checkpoints.
    if args.dataset != "all_windows" and args.class_balance == "sample":
        agg_tag += "_sampled"
    elif args.class_balance == "equal":
        agg_tag += "_equal_sampled"
    elif args.class_balance == "none":
        agg_tag += "_unbalanced"
    # The fold goes LAST, after every other tag, and is never omitted -- not
    # even for fold 0. Everything above distinguishes two models of the same
    # data; this distinguishes the same model on two different splits, which
    # would otherwise land on one results directory and overwrite each other's
    # epoch_metrics.csv. Tagging it unconditionally is what makes that
    # impossible rather than merely unlikely: an untagged name would be
    # silently reused by whichever fold ran second.
    fold_index = manifest.get("fold_index", manifest.get("split_seed", 0))
    if args.postsynaptic_cache:
        agg_tag += ("_pre_mean_control" if args.append_presynaptic_mean else
                    "_pre_post" if args.use_postsynaptic else "_pre_only_matched")
    agg_tag += f"_fold{fold_index}"
    run_name = f"gnn_{args.classifier}_scratch_{agg_tag}"
    result_subdir = {
        "all_windows": Path("all_windows"),
        "presynaptic_cave": Path("presynaptic/cave_skeletons"),
        "presynaptic_new": Path("presynaptic/new_skeletons"),
        "presynaptic_dense": Path("presynaptic/dense_teasar") / Path(args.presynaptic_database).parent.name / Path(args.presynaptic_database).name,
    }[args.dataset]
    result_root = (Path(args.results_dir) if args.results_dir else
                   Path(__file__).resolve().parent.parent / "results" / result_subdir)
    if args.presynaptic_compartment_filter and not args.results_dir:
        result_root = result_root / 'compartment_filtered' / Path(args.presynaptic_compartment_filter).name
    ckpt_dir = result_root / run_name
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
    start_epoch, best_val_f1 = 0, -1.0
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
        # Not part of ModelConfig -- it is a property of the training
        # objective, not the model -- so it needs its own check. Resuming
        # across a change would silently train the second half of a run
        # against a different distribution than the first, and leave one
        # epoch_metrics.csv describing both. Checkpoints written before this
        # key existed carry the old default, "loss".
        if ckpt.get("class_balance", "loss") != args.class_balance:
            raise SystemExit(
                f"--resume: {last_path} was trained with --class-balance "
                f"{ckpt.get('class_balance', 'loss')}, not {args.class_balance}.\n"
                "Delete the directory to start fresh, or pass the original setting."
            )
        # Also not part of ModelConfig -- it changes which parameters get
        # gradient, not how the model is built, so two runs differing only in
        # it produce identical state_dicts and would resume into each other
        # without complaint.
        if ckpt.get("freeze_aggregator", False) != args.freeze_aggregator:
            raise SystemExit(
                f"--resume: {last_path} was trained with --freeze-aggregator="
                f"{ckpt.get('freeze_aggregator', False)}, not {args.freeze_aggregator}.\n"
                "Delete the directory to start fresh, or pass the original setting."
            )
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
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
            f"best window macro F1 so far {best_val_f1:.4f}"
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

    # best_val_f1 comes from the resume block above (-1.0 on a fresh run).
    # The best weights live on disk in checkpoint_best.pt rather than in an
    # in-memory `best_state`: a resumed run may never beat a best set before
    # the interruption, so the final model has to be reloadable from disk.
    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        desc="train", unit="epoch", initial=start_epoch, total=args.epochs,
    )
    for epoch in epoch_bar:
        if args.dataset != "all_windows":
            train_batch_sampler.set_epoch(epoch)
        model.train()
        total_loss, n = 0.0, 0
        batch_bar = tqdm(
            train_loader, desc=f"epoch {epoch}", unit="batch", leave=False,
        )
        for data in batch_bar:
            data = data.to(device)
            opt.zero_grad()
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=args.amp and device.type == "cuda",
            ):
                g = model(
                    data.x, data.edge_index, data.batch,
                    pos_enc=data.pos_enc, rel_pos=data.rel_pos,
                    thickness=getattr(data, "thickness", None),
                    has_segclr=getattr(data, "has_segclr", None),
                    postsynaptic_embedding=getattr(data, "postsynaptic_embedding", None),
                )
                loss = model.cls_head.compute_loss(g, data.y_levels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.num_graphs
            n += data.num_graphs
            batch_bar.set_postfix(loss=f"{total_loss / max(1, n):.4f}")

        val_labels, val_preds, val_root_ids = evaluate(model, val_loader, device, args.amp)
        # Window-level metrics via the SAME summarize() the cell-level ones
        # use -- lets us see whether imbalance bias shows up at the per-window
        # classification step itself, independent of what majority voting
        # does to it afterward.
        window_val_metrics = summarize(val_labels, val_preds, len(classes), classes)
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
        if window_val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = window_val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(), "config": config,
                    "epoch": epoch, "val_window_macro_f1": best_val_f1,
                },
                ckpt_dir / "checkpoint_best.pt",
            )
            # Small, login-node-safe companion to checkpoint_best.pt. Analysis
            # notebooks can show the best epoch's confusion matrix while this
            # run is still training, without loading a checkpoint or inferring.
            best_metrics_path = ckpt_dir / "best_metrics.json"
            best_metrics_tmp = ckpt_dir / "best_metrics.json.tmp"
            best_metrics_tmp.write_text(json.dumps({
                "run": run_name, "epoch": epoch, "classes": classes,
                "window_test_metrics": window_val_metrics,
                "test_metrics": val_metrics, "complete": False,
            }, indent=2))
            os.replace(best_metrics_tmp, best_metrics_path)
            tqdm.write(
                f"  new best selection metric (window macro F1)={best_val_f1:.3f} "
                f"at epoch {epoch} -> checkpoint_best.pt"
            )
        # Macro precision / recall / F1 at both granularities, rather than
        # accuracy and balanced accuracy. `balanced_accuracy` IS macro recall
        # (gnn/metrics.py::balanced_accuracy -- the mean of per-class recall),
        # so printing it as R alongside P and F1 is a renaming plus two extra
        # numbers, not three new metrics. The two extra numbers are the point:
        # under this class imbalance recall and precision move in OPPOSITE
        # directions as training proceeds -- an early, under-confident model
        # over-predicts rare classes, scoring high recall and low precision --
        # and a console showing only recall makes that look like the model
        # simply peaking at epoch 0. Raw accuracy stays in epoch_metrics.csv,
        # which logs all four at both granularities plus per-class recall and
        # precision; nothing is lost from the record by leaving it out here.
        def _macro(m: dict) -> str:
            return (
                f"P={m['macro_precision']:.3f} R={m['balanced_accuracy']:.3f} "
                f"F1={m['macro_f1']:.3f}"
            )

        epoch_bar.set_postfix(
            train_loss=f"{total_loss / max(1, n):.4f}",
            val_window_f1=f"{window_val_metrics['macro_f1']:.3f}",
            val_cell_f1=f"{val_metrics['macro_f1']:.3f}",
        )
        # Log every epoch so the validation curve remains visible in batch logs.
        tqdm.write(
            f"epoch {epoch:4d}  train_loss={total_loss / max(1, n):.4f}  "
            f"window[{_macro(window_val_metrics)}]  cell[{_macro(val_metrics)}]"
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
                "best_val_f1": best_val_f1,
                "class_balance": args.class_balance,
                "freeze_aggregator": args.freeze_aggregator,
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
    test_labels, test_preds, test_root_ids = evaluate(model, test_loader, device, args.amp)
    window_test_metrics = summarize(test_labels, test_preds, len(classes), classes)
    test_metrics = cell_level_metrics(test_labels, test_preds, test_root_ids, len(classes), classes)
    dataset_test_metrics = {}
    if any('source_dataset' in info for info in manifest['cells'].values()):
        domains = np.asarray([manifest['cells'][str(int(root))]['source_dataset']
                              for root in test_root_ids])
        for domain in sorted(set(domains)):
            selected = domains == domain
            dataset_test_metrics[domain] = {
                'n_cells': int(np.unique(test_root_ids[selected]).size),
                'n_windows': int(selected.sum()),
                'cell': cell_level_metrics(test_labels[selected], test_preds[selected],
                                          test_root_ids[selected], len(classes), classes),
                'window': summarize(test_labels[selected], test_preds[selected], len(classes), classes),
            }
    print("=== test metrics (GNN) ===")
    print(
        f"window-level  P={window_test_metrics['macro_precision']:.4f} "
        f"R={window_test_metrics['balanced_accuracy']:.4f} "
        f"F1={window_test_metrics['macro_f1']:.4f} "
        f"acc={window_test_metrics['accuracy']:.4f}"
    )
    print(json.dumps(test_metrics, indent=2))

    out_dir = result_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.json"
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "window_test_metrics": window_test_metrics,
                "test_metrics": test_metrics,
                "dataset_test_metrics": dataset_test_metrics,
                "classes": classes,
                "best_epoch": best_ckpt.get("epoch") if "best_ckpt" in locals() else None,
                # Which cell-held-out fold produced these numbers. A result
                # file is one fold; cross-fold summaries are written separately.
                "postsynaptic_cache": str(args.postsynaptic_cache) if args.postsynaptic_cache else None,
                "use_postsynaptic": args.use_postsynaptic,
                "append_presynaptic_mean": args.append_presynaptic_mean,
                "single_presynaptic_embedding": args.single_presynaptic_embedding,
                "dataset_counts": {"train_windows": len(train_ds), "test_windows": len(test_ds)},
                "fold": f"fold_{fold_index}",
                "fold_index": fold_index,
                "split_seed": manifest.get("split_seed"),
                "split_fracs": manifest.get("split_fracs"),
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    if args.dataset == "all_windows":
        publish_test_prediction_cache(
            run_name, test_ds, test_preds, test_labels, args.num_embeddings,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--postsynaptic-cache',
                   help='Prepared native-site cache; applies the same eligibility filter to both groups')
    p.add_argument('--append-presynaptic-mean', action='store_true',
                   help='Matched-width control: append the raw 64-dimensional presynaptic mean after pooling')
    p.add_argument('--single-presynaptic-embedding', action='store_true',
                   help='Use only the nearest observed presynaptic node; preserve original window eligibility')
    p.add_argument('--use-postsynaptic', action='store_true',
                   help='Concatenate 64 postsynaptic dimensions after pooling, before classification')
    p.add_argument(
        "--architecture", default="graph_transformer",
        choices=["graph_transformer", "graph_transformer_teasar", "mpnn_complete", "mpnn",
                 "pointwise_mlp", "linear", "mean"],
        help="graph_transformer (default): gnn/graph_transformer.py's AC-attention "
             "GraphTransformer. graph_transformer_teasar: mixed-node variant where "
             "has_segclr gates only the SegCLR modality and every TEASAR node remains in "
             "attention. mpnn: gnn/encoder.py::MPNNEncoder, plain GraphSAGE message "
             "passing (no attention) + MeanReadout. mpnn_complete: that same encoder over "
             "a per-window clique. pointwise_mlp: gnn/pointwise_mlp.py's "
             "PointwiseMLPEncoder, a per-node MLP + mean, with no graph and no spatial "
             "features. linear: that same shape with one Linear and no nonlinearity, which "
             "the mean absorbs. mean: MeanReadout "
             "over the raw node embeddings, no encoder -- the mean-pooling baseline.",
    )
    p.add_argument(
        "--dataset", default="all_windows",
        choices=["all_windows", "presynaptic_cave", "presynaptic_new", "presynaptic_dense"],
        help="dataset/result namespace; presynaptic_dense uses the finalized database's K",
    )
    p.add_argument(
        "--presynaptic-database",
        default="/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10",
    )
    p.add_argument("--manifest", help="explicit cohort/split manifest JSON")
    p.add_argument(
        "--embedding-augmentation", choices=("clean", "gray", "flip", "structured_low"),
        help="training-only node embedding choice set; augmented policies sample clean plus draws 0-3",
    )
    p.add_argument(
        "--embedding-augmentation-database",
        default="/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/"
                "cave_embedding_training_choices/conf0.7/fold0/cutoff5000_fp16",
        help="packed per-cell (clean + four draws) embedding choice database",
    )
    p.add_argument("--results-dir", help="Override the result directory containing model run folders")
    p.add_argument(
        "--attention-budget", type=int, default=1_806_336,
        help="maximum batch_windows * (largest_nodes + CLS)^2 for presynaptic dense attention",
    )
    p.add_argument(
        "--mixed-cell-batch-size", type=int, default=0,
        help="experimental: mix this many original single-cell batches into each batch "
             "while preserving the number of steps and using each window once per epoch",
    )
    p.add_argument(
        "--presynaptic-cell-cache", type=int, default=2,
        help="decompressed per-worker cell records retained by the presynaptic loader",
    )
    p.add_argument(
        "--presynaptic-memmap-root",
        help="directory of predecoded .npy cell arrays; preserves sampler order and window values",
    )
    p.add_argument('--presynaptic-compartment-filter',
                   help='completed majority-vote cache; exclude dendrite-majority windows before deduplication')
    p.add_argument(
        "--pointwise-mlp-layers", type=int, default=2,
        help="Linear+GELU layers in phi for --architecture pointwise_mlp; 2 with the default "
             "width is the 64 -> 128 -> 128 MLP it was specified as",
    )
    p.add_argument("--pointwise-mlp-hidden-dim", type=int, default=128,
                   help="phi's hidden AND output width for --architecture pointwise_mlp")
    p.add_argument("--linear-out-dim", type=int, default=128,
                   help="output width for --architecture linear; matches the pointwise MLP's "
                        "so the two hand the head the same number of channels")
    p.add_argument(
        "--mpnn-layers", type=int, default=2,
        help="message-passing hops for --architecture mpnn; 2 by default because windows "
             "average ~10.7 nodes and deeper stacks over-smooth a graph that small",
    )
    p.add_argument("--mpnn-hidden-dim", type=int, default=128)
    p.add_argument("--mpnn-dropout", type=float, default=0.1)
    p.add_argument(
        "--spatial", action="store_true",
        help="give --architecture mean/mpnn the same spatial node features the "
             "GraphTransformer builds for itself: the center-relative offset (dx, dy, dz and "
             "their norm) and the per-window Laplacian PE, concatenated onto the raw "
             "embeddings. Off by default, so those two see raw embeddings alone; enabling it "
             "tags the run _spatial. This is what makes a cross-architecture sweep compare "
             "aggregation rather than node features. Rejected with --architecture "
             "graph_transformer, which has --gt-no-lpe / --gt-no-rel-pos instead.",
    )
    p.add_argument("--position", action="store_true",
                   help="concatenate center-relative xyz and distance for mean/MPNN/MPNN complete")
    p.add_argument("--lpe", action="store_true",
                   help="concatenate the window Laplacian positional encoding for "
                        "mean/MPNN/MPNN complete")
    p.add_argument(
        "--cls-hidden-dim", type=int, default=None,
        help="LCPN head hidden layer size; default None = plain Linear per node",
    )
    p.add_argument(
        "--classifier", choices=["lcpn", "flat"], default="lcpn",
        help="lcpn (default) routes through local hierarchy heads; flat predicts the eight "
             "active finest classes with one softmax and never uses a coarse decision",
    )
    p.add_argument(
        "--cls-resnet", action=argparse.BooleanOptionalAction, default=True,
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
        "--class-balance", default="sample", choices=["sample", "equal", "none"],
        help="how to correct the class imbalance. sample (default): class-balanced "
             "resampling of the training windows with replacement, which is what "
             "segCLR_cell_classification's own LCPN config does "
             "(weight_imbalanced_classes: sample) -- their LCPN loss is never weighted. "
             "equal: use inverse-count weights for equal expected shares across active finest classes. "
             "none: disable resampling, which under this imbalance collapses balanced "
             "accuracy toward chance.",
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
        "--no-embeddings", action="store_true",
        help="drop the SegCLR embedding from the node input, leaving only morphology: the "
             "graph, the center-relative offset and the Laplacian PE. The geometry-only "
             "control for how much of a score comes from the embeddings vs. the shape they "
             "sit on. Requires --spatial for --architecture mean/mpnn, which otherwise would "
             "have no node input at all, and is rejected outright for --architecture "
             "pointwise_mlp / linear, whose only input is the embeddings. Tags the run "
             "_noemb.",
    )
    p.add_argument(
        "--freeze-aggregator", action="store_true",
        help="hold the aggregation stage (GraphTransformer, MPNNEncoder, or the pointwise "
             "MLP / linear phi) at its random initialization and train only the "
             "classification head -- the random-features "
             "control. Whatever such a run retains over mean pooling comes from the "
             "aggregator's STRUCTURE rather than from anything it learned, and it restores "
             "the mean-pool property that the head is the only thing training. Dropout in "
             "the frozen stage is disabled too. Rejected for --architecture mean, which has "
             "no aggregation parameters. Tags the run _frozenagg.",
    )
    p.add_argument(
        "--num-embeddings", type=int, default=20,
        help="fixed number of embeddings per neighborhood (default: 20); "
             "all_windows supports 10, 20, or 40; presynaptic_dense uses its database K",
    )
    p.add_argument(
        "--radius-dataset", dest="num_embeddings", action="store_const", const=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--neighborhood-root",
        default="/orcd/scratch/orcd/013/jcbliao/embedding_paths/r5um",
        help="root containing neighborhoods/n{10,20,40}; defaults to the new r5um dataset",
    )
    p.add_argument(
        "--window-nm", type=float, default=DEFAULT_WINDOW_NM,
        help="geodesic window RADIUS in nm. Default 10000 matches the baseline's 10um "
             "aggregation window. 0 gives single-node windows -- the unaggregated case, "
             "where 'mean' degenerates to a probe on one raw embedding and the two GNN "
             "architectures have no neighbors to pass messages from. Each radius needs its "
             "own membership cache (data/build_window_membership.py --window-nm ...), and "
             "any non-default radius is tagged into the run name.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4096,
        help="number of WINDOW subgraphs per batch (not whole cells)",
    )
    p.add_argument("--epochs", type=int, default=16,
                   help="total epochs; default 16 means epoch indices 0 through 15")
    p.add_argument(
        "--resume", action="store_true",
        help="continue from results/all_windows/<run>/checkpoint_last.pt if it exists, restoring model, "
             "optimizer and RNG state and truncating epoch_metrics.csv to match. Starts from "
             "epoch 0 if no such file exists, so it is safe to leave on permanently -- which is "
             "what makes a preempted job recover on requeue instead of restarting. Refuses to "
             "resume across a changed ModelConfig.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument(
        "--num-workers", type=int, default=15,
        help="DataLoader worker processes for window extraction -- keep one below the job's "
             "--cpus-per-task, leaving a core for the main process (see CLAUDE.md)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--amp", action="store_true",
        help="use BF16 autocast on CUDA (recommended for L40S presynaptic training)",
    )
    from scripts.training_config import parse_training_args

    main(parse_training_args(p))
