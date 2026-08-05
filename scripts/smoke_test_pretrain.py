"""Smoke test for the ACTUAL scripts/pretrain_gnn.py training loop -- not just
the model math (scripts/smoke_test_model.py already covers that on synthetic
tensors), but the real script's DataLoader/ReplacementPool/checkpoint-saving/
CLI-argument integration, which nothing has exercised yet.

Real data isn't ready (data/build_dataset.py may still be mid-run), so this
builds a tiny synthetic dataset under data/_smoke_dataset/ -- NEVER
data/graph_cache or data/manifest.json, which belong to the real pipeline and
may be actively written concurrently -- and monkeypatches data.dataset's
module-level MANIFEST_PATH/GRAPH_CACHE_DIR to point at it before calling the
real pretrain_gnn.main(), so this is the actual training code, just pointed
at fake data.

Run via sbatch (mit_quicktest -- tiny graphs, 2 epochs, CPU is plenty).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

SMOKE_DIR = Path(__file__).resolve().parent.parent / "data" / "_smoke_dataset"


def make_fake_cell(root_id: int, n_nodes: int, d: int, seed: int) -> Data:
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((n_nodes, d)), dtype=torch.float32)
    src, dst = np.arange(n_nodes - 1), np.arange(1, n_nodes)
    edge_index = torch.tensor(
        np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1), dtype=torch.long
    )
    edge_attr = torch.tensor(
        rng.uniform(50, 500, size=(edge_index.shape[1], 1)), dtype=torch.float32
    )
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.root_id = root_id
    data.orig_node_ids = torch.arange(n_nodes, dtype=torch.long)
    return data


def build_fake_dataset(n_cells: int = 12, d: int = 64):
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    graph_dir = SMOKE_DIR / "graph_cache"
    graph_dir.mkdir(parents=True)

    classes = ["C-N-I", "C-N-E", "C-G-OGC"]
    manifest = {"cells": {}}
    rng = np.random.default_rng(0)
    for i in range(n_cells):
        root_id = 1000 + i
        n_nodes = int(rng.integers(15, 40))
        data = make_fake_cell(root_id, n_nodes, d, seed=i)
        torch.save(data, graph_dir / f"{root_id}.pt")
        split = "train" if i < n_cells * 0.6 else ("val" if i < n_cells * 0.8 else "test")
        manifest["cells"][str(root_id)] = {"cell_type": classes[i % len(classes)], "split": split}

    manifest_path = SMOKE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path, graph_dir


def main() -> int:
    manifest_path, graph_dir = build_fake_dataset()
    print(f"fake dataset: {manifest_path}")
    print(f"fake graph cache: {graph_dir}")

    from data import dataset as ds_module

    ds_module.MANIFEST_PATH = manifest_path
    ds_module.GRAPH_CACHE_DIR = graph_dir

    from scripts import pretrain_gnn

    args = argparse.Namespace(
        depth=2,
        hidden_dim=16,
        num_layers=2,
        conv_type="sage",
        mask_prob=0.3,
        replace_strategy="random",
        use_smooth_l1=False,
        lambda_mag=1.0,
        epochs=2,
        lr=1e-3,
        weight_decay=1e-5,
        accum_steps=2,
        ckpt_every=1,
        seed=0,
    )
    pretrain_gnn.main(args)

    # Smoke-test checkpoints are reproducible junk, not real results -- clean
    # them up so they can't be mistaken for (or collide with) a real run's
    # results/pretrain_random/ output later.
    out_dir = Path(__file__).resolve().parent.parent / "results" / "pretrain_random"
    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"cleaned up smoke-test checkpoints at {out_dir}")

    shutil.rmtree(SMOKE_DIR)
    print(f"cleaned up fake dataset at {SMOKE_DIR}")

    print("\npretrain smoke test passed: DataLoader, ReplacementPool, masking, "
          "encoder/decoder, cosine loss, gradient accumulation, and checkpoint "
          "saving all ran end-to-end without error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
