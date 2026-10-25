import numpy as np
import pytest
import torch
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from data.segmentation_augmentation import mask_variants, neighbor_step


def test_cohort_includes_held_out_only_when_requested():
    from scripts.generate_segmentation_augmentations import select_roots

    cohort = {
        "cells": {
            "1": {"split": "train", "cell_type": "A"},
            "2": {"split": "test", "cell_type": "A"},
        }
    }
    assert select_roots(cohort, 0) == [1]
    assert select_roots(cohort, 0, include_held_out=True) == [1, 2]


@pytest.mark.parametrize("neighbors", [6, 26])
@pytest.mark.parametrize("radius", [1, 2, 3])
def test_matches_scipy_with_halo(neighbors, radius):
    rng = np.random.default_rng(123)
    mask = rng.random((2, 1, 9 + 2 * radius, 9 + 2 * radius, 9 + 2 * radius)) > 0.3
    structure = generate_binary_structure(3, 1 if neighbors == 6 else 3)
    got = dict(mask_variants(torch.from_numpy(mask), radius, neighbors, 9))
    for layer in range(1, radius + 1):
        for prefix, op in [("erode", binary_erosion), ("dilate", binary_dilation)]:
            expected = np.stack(
                [
                    op(row[0], structure, iterations=layer)[
                        radius:-radius, radius:-radius, radius:-radius
                    ]
                    for row in mask
                ]
            )[:, None]
            np.testing.assert_array_equal(got[f"{prefix}{layer}"].numpy(), expected)


def test_halo_preserves_foreground_at_crop_boundary():
    mask = torch.ones((1, 1, 7, 7, 7), dtype=torch.bool)
    got = dict(mask_variants(mask, 2, 6, 3))
    assert got["erode2"].all()


def test_expansion_sees_foreground_outside_crop():
    mask = torch.zeros((1, 1, 5, 5, 5), dtype=torch.bool)
    mask[0, 0, 0, 2, 2] = True
    got = dict(mask_variants(mask, 1, 6, 3))
    assert not got["clean"].any()
    assert got["dilate1"].sum() == 1


def test_no_wrapping_and_simultaneous_update():
    mask = torch.zeros((1, 1, 7, 7, 7), dtype=torch.bool)
    mask[0, 0, 3, 3, 3] = True
    got = dict(mask_variants(mask, 1, 6, 5))
    assert got["dilate1"].sum() == 7
    assert not got["erode1"].any()


def test_validity_excludes_out_of_volume_expansion():
    mask = torch.ones((1, 1, 5, 5, 5), dtype=torch.bool)
    valid = mask.clone()
    valid[:, :, :2] = False
    got = dict(mask_variants(mask & valid, 1, 6, 3, valid))
    assert not got["dilate1"][:, :, 0].any()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        neighbor_step(torch.ones((1, 1, 5, 5, 5)), True)
    with pytest.raises(ValueError):
        list(mask_variants(torch.ones((1, 1, 5, 5, 5), dtype=torch.bool), 2, 6, 3))


def test_embedding_database_summary_and_training_export(tmp_path):
    import json
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.generate_segmentation_augmentations import summarize, pack, sha

    plan_path = tmp_path / "plan.json"
    output = tmp_path / "output"
    plan = dict(
        radius=1,
        neighbors=6,
        output=str(output),
        nodes_per_cell=2,
        shards=[[dict(root_id=1, n_nodes=2)]],
    )
    plan_path.write_text(json.dumps(plan))
    dest = output / "parts" / "1.parquet"
    dest.parent.mkdir(parents=True)
    vectors = np.tile(np.arange(1, 65, dtype=np.float32), (6, 1))
    table = pa.table(
        dict(
            root_id=[1] * 6,
            node_id=[3, 3, 3, 7, 7, 7],
            augmentation_id=[
                "segmask_n6_clean",
                "segmask_n6_erode1",
                "segmask_n6_dilate1",
            ]
            * 2,
            augmentation_sample=[0] * 6,
            embedding=pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1)), 64
            ),
            baseline_voxels=[10] * 3 + [20] * 3,
            mask_voxels=[10, 5, 15, 20, 0, 30],
            delta_voxels=[0, -5, 5, 0, -20, 10],
            halo_clipped=[False] * 6,
        )
    )
    table = table.replace_schema_metadata({b"plan_sha256": sha(plan_path).encode()})
    pq.write_table(table, dest)
    args = SimpleNamespace(plan=plan_path)
    summarize(args)
    import duckdb

    with duckdb.connect(str(output / "embeddings.duckdb"), read_only=True) as db:
        assert db.sql("SELECT count(*) FROM embeddings").fetchone()[0] == 6
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"]
    stats = {row["augmentation_id"]: row for row in summary["statistics"]}
    assert stats["segmask_n6_erode1"]["empty_masks"] == 1
    assert stats["segmask_n6_dilate1"]["mean_fractional_change"] == 0.5
    assert stats["segmask_n6_clean"]["mean_baseline_cosine"] == pytest.approx(1)
    pack(args)
    with np.load(output / "training_choices" / "segmask" / "cells" / "1.npz") as z:
        np.testing.assert_array_equal(z["node_ids"], [3, 7])
        assert z["embedding_choices"].shape == (2, 3, 64)
        np.testing.assert_array_equal(z["embedding_choices"].reshape(6, 64), vectors)


def test_grouped_forward_keeps_counts_and_vectors_aligned(monkeypatch):
    pytest.importorskip('segclr', reason='Grouped inference requires the optional SegCLR inference environment')
    from scripts.generate_segmentation_augmentations import inference_class
    from data.segmentation_augmentation import variant_names

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 0.0

    class Model:
        def __call__(self, batch, return_embeddings):
            return None, batch.mean(dim=(1, 2, 3, 4))[:, None].repeat(1, 64)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    cls = inference_class()
    runner = cls.__new__(cls)
    runner.radius = 1
    runner.neighbors = 6
    runner.names = variant_names(1)
    runner.step = neighbor_step
    runner.device = torch.device("cpu")
    runner.batch_size = 2
    runner.variants_per_forward = 2
    runner.amp_dtype = None
    runner.model = Model()
    runner.timings = []
    em = torch.full((2, 1, 129, 129, 129), 255, dtype=torch.uint8)
    masks = torch.zeros((2, 2, 131, 131, 131), dtype=torch.bool)
    masks[:, 1] = True
    masks[0, 0, 63:68, 63:68, 63:68] = True
    runner._buffers = [(em, masks)]
    result = runner._forward(0, 1)
    assert result.shape == (1, 3 * 64 + 3 + 1)
    counts = result[0, 3 * 64 : 3 * 64 + 3]
    assert counts[0] == 125 and counts[1] == 27 and counts[2] == 275
    for variant, count in enumerate(counts):
        np.testing.assert_allclose(
            result[0, variant * 64 : (variant + 1) * 64], count / (129**3), rtol=1e-6
        )
    assert result[0, -1] == 0
