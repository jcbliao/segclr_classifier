from collections import OrderedDict

import numpy as np
import pytest
import torch

from data.packed_presynaptic import PackedPresynapticWindowDataset
from data.segmentation_augmentation import (
    TRAINING_POLICIES,
    variant_names,
    training_choice_indices,
)
from scripts.test_batched_presynaptic import fixture


def packed_fixture(tmp_path, policy="segmask_both3"):
    ds, h = fixture(precomputed=True, post=True)
    ds.hierarchy = h
    ds._cache = OrderedDict()
    ds.geometry_required = False
    ds.augmentation_database = tmp_path
    folder = tmp_path / "segmask" / "cells"
    folder.mkdir(parents=True, exist_ok=True)
    for root in (11, 22):
        cell = ds._load_cell(root)
        ids = cell["observed_ids"]
        values = np.empty((len(ids), 7, 64), np.float32)
        for row, node in enumerate(ids):
            for view in range(7):
                values[row, view] = root * 10000 + node * 100 + view
        np.savez(
            folder / f"{root}.npz",
            node_ids=ids,
            embedding_choices=values,
            augmentation_id=np.asarray("segmask"),
            choice_labels=np.asarray(variant_names(3)),
        )
    ds.embedding_augmentation = policy
    ds.__class__ = PackedPresynapticWindowDataset
    return ds


@pytest.mark.parametrize("policy", TRAINING_POLICIES)
def test_packed_views_keep_missing_nodes_zero_and_use_original_clean(tmp_path, policy):
    ds = packed_fixture(tmp_path, policy)
    assert ds.prepare_packed(workers=2)
    p = ds._packed
    indices = np.arange(p["ptr"][-1])
    clean = ds.sample_packed_x(indices, choices=np.zeros(len(indices), np.int64))
    torch.testing.assert_close(
        clean, torch.from_numpy(p["fields"]["x"]), rtol=0, atol=0
    )
    columns = training_choice_indices(policy, p["augmentation_labels"].tolist())
    for choice, column in enumerate(columns):
        got = ds.sample_packed_x(
            indices, choices=np.full(len(indices), choice, np.int64)
        )
        np.testing.assert_array_equal(
            got.numpy(), p["augmentation_bank"][p["augmentation_rows"], column]
        )
        assert not got[~torch.from_numpy(p["fields"]["has_segclr"])].any()


def test_packed_choices_redraw_and_sample_each_repeated_node_independently(tmp_path):
    ds = packed_fixture(tmp_path)
    ds.prepare_packed(workers=2)
    indices = np.repeat(0, 8192)
    torch.manual_seed(123)
    left = ds.sample_packed_x(indices)
    right = ds.sample_packed_x(indices)
    assert not torch.equal(left, right)
    assert len(torch.unique(left[:, 0])) == 7
    torch.manual_seed(123)
    torch.testing.assert_close(left, ds.sample_packed_x(indices))
    values, counts = torch.unique(left[:, 0], return_counts=True)
    assert (counts.float() / len(indices) - 1 / 7).abs().max() < 0.025


def test_persistent_bank_cache_is_shared_by_conditions(tmp_path, monkeypatch):
    from data.packing_cache import prepare_cached

    monkeypatch.setenv("SEGCLR_PACKING_CACHE", str(tmp_path / "cache"))
    first = packed_fixture(tmp_path / "source", "segmask_shrink1")
    assert prepare_cached(first, "shared", "train", workers=2, build=True)
    other = packed_fixture(tmp_path / "source", "segmask_expand3")
    monkeypatch.setattr(
        other, "prepare_packed", lambda **kwargs: pytest.fail("repacked shared bank")
    )
    assert prepare_cached(other, "shared", "train")
    assert isinstance(other._packed["augmentation_bank"], np.memmap)
    np.testing.assert_array_equal(
        first._packed["augmentation_bank"], other._packed["augmentation_bank"]
    )


def test_cache_identity_shares_pools_but_tracks_view_sources(tmp_path):
    from types import SimpleNamespace
    from data.packing_cache import identity

    packed_fixture(tmp_path)
    args = SimpleNamespace(
        dataset="presynaptic_cave",
        spatial=False,
        position=False,
        lpe=False,
        architecture="pointwise_mlp",
        presynaptic_database=str(tmp_path / "source"),
        postsynaptic_cache=None,
        embedding_augmentation="segmask_shrink1",
        embedding_augmentation_database=str(tmp_path),
        gt_use_thickness=False,
    )
    manifest = {"cells": {"11": {"split": "train"}, "22": {"split": "test"}}}
    key = identity(args, manifest)
    for policy in TRAINING_POLICIES:
        args.embedding_augmentation = policy
        assert identity(args, manifest) == key
    source = tmp_path / "segmask/cells/11.npz"
    with source.open("ab") as stream:
        stream.write(b"changed")
    assert identity(args, manifest) != key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_replays_redraw_views(tmp_path):
    ds = packed_fixture(tmp_path)
    ds.prepare_packed(workers=2)
    assert ds.prepare_device(torch.device("cuda"), reserve_bytes=0)
    indices = torch.zeros(2048, dtype=torch.long, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ds.sample_packed_x(indices, device=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = ds.sample_packed_x(indices, device=True)
    graph.replay()
    left = out.clone()
    graph.replay()
    right = out.clone()
    torch.cuda.synchronize()
    assert not torch.equal(left, right)
    assert len(torch.unique(left[:, 0])) == 7
