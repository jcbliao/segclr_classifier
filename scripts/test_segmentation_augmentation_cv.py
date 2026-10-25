from collections import OrderedDict
import numpy as np
import pytest
import torch
from data.segmentation_augmentation import (
    TRAINING_POLICIES,
    training_choice_indices,
    variant_names,
)


@pytest.mark.parametrize("policy", TRAINING_POLICIES)
def test_cumulative_pool_has_every_allowed_view_once(policy):
    labels = variant_names(3)
    indices = training_choice_indices(policy, labels)
    radius = int(policy[-1])
    assert indices.count(labels.index("clean")) == 1
    assert len(set(indices)) == len(indices)
    selected = [labels[i] for i in indices if i != labels.index("clean")]
    assert all(int(label[-1]) <= radius for label in selected)
    if "both" in policy:
        assert set(selected) == {
            f"{op}{r}" for op in ("erode", "dilate") for r in range(1, radius + 1)
        }
        assert len(indices) == 2 * radius + 1
    else:
        op = "erode" if "shrink" in policy else "dilate"
        assert selected == [f"{op}{r}" for r in range(1, radius + 1)]
        assert len(indices) == radius + 1


@pytest.mark.parametrize("policy", TRAINING_POLICIES)
def test_dataset_loads_shared_cache_and_original_clean_embeddings(
    tmp_path, monkeypatch, policy
):
    import data.dataset_presynaptic as module

    labels = variant_names(3)
    ids = np.array([2, 4], np.int32)
    original = np.full((2, 64), 9.0, np.float32)
    cell = dict(
        pos=np.zeros((5, 3), np.float32),
        edges=np.empty((0, 2), np.int32),
        observed_ids=ids,
        observed_x=original,
    )
    monkeypatch.setattr(
        module, "load_cell_arrays", lambda *args: {k: v.copy() for k, v in cell.items()}
    )
    directory = tmp_path / "segmask" / "cells"
    directory.mkdir(parents=True)
    choices = np.broadcast_to(
        np.arange(7, dtype=np.float32)[None, :, None] + 100, (2, 7, 64)
    ).copy()
    np.savez(
        directory / "1.npz",
        node_ids=ids,
        embedding_choices=choices,
        augmentation_id=np.asarray("segmask"),
        choice_labels=np.asarray(labels),
    )
    ds = module.PresynapticWindowDataset.__new__(module.PresynapticWindowDataset)
    ds._cache = OrderedDict()
    ds.memmap_root = None
    ds.cell_dir = tmp_path
    ds.variant = "cave"
    ds.embedding_augmentation = policy
    ds.augmentation_database = tmp_path
    ds.postsynaptic_cache = None
    ds.cell_cache_size = 1
    ds._targets = lambda root: torch.zeros((1, 1), dtype=torch.long)
    got = ds._load_cell(1)["augmentation_choices"]
    selected = training_choice_indices(policy, labels)
    for slot, index in enumerate(selected):
        np.testing.assert_array_equal(
            got[:, slot], original if index == 0 else choices[:, index]
        )


def test_task_mapping_covers_five_folds_and_nine_conditions():
    from scripts.segmentation_augmentation_cv import mapping, CONDITIONS

    assert {mapping(i) for i in range(45)} == {
        (f, c) for f in range(5) for c in CONDITIONS
    }
    with pytest.raises(ValueError):
        mapping(45)
