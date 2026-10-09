"""Validate exact-unique presynaptic datasets and attention-budget batching."""

from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_lcpn import load_manifest  # noqa: E402
from data.dataset_presynaptic import (  # noqa: E402
    AttentionBudgetBatchSampler,
    PresynapticWindowDataset,
)

EXPECTED = {"cave": 1_664_524, "new": 1_743_421}


def connected(graph) -> bool:
    if graph.num_nodes == 0:
        return False
    reached = {0}
    changed = True
    edges = graph.edge_index.T.tolist()
    while changed:
        changed = False
        for u, v in edges:
            if u in reached and v not in reached:
                reached.add(v); changed = True
    return len(reached) == graph.num_nodes


def main() -> int:
    manifest = load_manifest()
    for variant in ("cave", "new"):
        datasets = [PresynapticWindowDataset(manifest, split, variant)
                    for split in ("train", "test")]
        total = sum(map(len, datasets))
        assert total == EXPECTED[variant], (variant, total, EXPECTED[variant])
        assert datasets[0].classes == [
            "putative_cge", "putative_parvalbumin", "putative_somatostatin",
            "pyramidal", "thalamocortical",
        ], datasets[0].classes
        sizes = np.concatenate([dataset.window_sizes for dataset in datasets])
        print(
            variant, "unique", f"{total:,}", "nodes p50/p95/p99/max",
            *np.percentile(sizes, [50, 95, 99]).astype(int), int(sizes.max()),
        )
        for dataset in datasets:
            picks = np.linspace(0, len(dataset) - 1, min(8, len(dataset)), dtype=int)
            for index in picks:
                graph = dataset[int(index)]
                assert int(graph.has_segclr.sum()) == 10
                assert graph.pos_enc.shape == (graph.num_nodes, 8)
                assert graph.rel_pos.shape == (graph.num_nodes, 3)
                assert graph.edge_index.numel() and connected(graph)
            sampler = AttentionBudgetBatchSampler(dataset, shuffle=False)
            for batch_indices in islice(iter(sampler), 100):
                largest = int(dataset.window_sizes[batch_indices].max()) + 1
                assert len(batch_indices) * largest**2 <= sampler.attention_budget
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
            batch = next(iter(loader))
            assert batch.has_segclr.dtype == torch.bool
            assert batch.num_graphs == len(
                next(iter(AttentionBudgetBatchSampler(dataset, shuffle=False)))
            )
    print("presynaptic dataset smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
