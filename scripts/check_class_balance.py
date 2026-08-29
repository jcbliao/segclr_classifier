"""Verifies the class-imbalance correction ported from
segCLR_cell_classification (`weight_imbalanced_classes: sample`), and the
hierarchy `drop_labels` filtering that now runs alongside it.

Three things worth checking before a sweep is launched on top of them, all
silent if wrong:

  - the per-class sampling weights match their `_class_weights` formula
    (1/sqrt(count), rescaled to average 1.0) on the real train split;
  - the resulting EXPECTED per-class draw counts per epoch are what the
    weights claim -- the number that actually decides whether a rare class
    gets gradient, and not something the weights alone make legible;
  - drop_labels and unknown-label filtering remove exactly the cells they
    should, and no others.

Reports the effective imbalance before and after, since the point of the
sqrt is that it does NOT flatten the distribution -- a run that came out
perfectly flat would mean plain 1/count had been used by mistake.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.dataset_lcpn import (  # noqa: E402
    DROP_LABELS,
    load_hierarchy,
    load_manifest,
    split_cells,
    train_window_counts_by_label,
)
from gnn.hierarchy import with_dropped_labels  # noqa: E402
from data.dataset_windowed import (  # noqa: E402
    WindowedGraphDatasetLCPN,
    balanced_sampler,
    inverse_sqrt_class_weights,
)


def main(args) -> int:
    manifest = load_manifest()
    hierarchy = load_hierarchy(manifest)
    classes = hierarchy.level_classes[-1]
    print(f"hierarchy depth {hierarchy.depth}, {len(classes)} classes at the trained level")
    print(f"drop_labels: {sorted(hierarchy.drop_labels) or '(none)'}  [DROP_LABELS={sorted(DROP_LABELS) or '(none)'}]")

    # --- drop filtering ---------------------------------------------------
    for split in ("train", "test"):
        raw = [i for i in manifest["cells"].values() if i["split"] == split]
        kept = split_cells(manifest, split, hierarchy)
        dropped = {i["cell_type"] for i in raw} - {i["cell_type"] for _, i in kept}
        print(f"  split {split:>5}: {len(kept)}/{len(raw)} cells kept, "
              f"labels dropped: {sorted(dropped) or '(none)'}")

    # DROP_LABELS is empty by default, so the loop above proves only that
    # filtering is a no-op when nothing is dropped. Exercise the mechanism on
    # a label that really exists, without changing what any run trains on.
    victim = max(
        {i["cell_type"] for i in manifest["cells"].values()},
        key=lambda lab: sum(i["cell_type"] == lab for i in manifest["cells"].values()),
    )
    probe = with_dropped_labels(hierarchy, {victim})
    n_before = len(split_cells(manifest, "train", hierarchy))
    n_after = len(split_cells(manifest, "train", probe))
    n_victim = sum(
        i["cell_type"] == victim and i["split"] == "train" for i in manifest["cells"].values()
    )
    assert n_before - n_after == n_victim, (
        f"dropping {victim!r} removed {n_before - n_after} train cells, expected {n_victim}"
    )
    # An unknown label must be filtered too, not raise from label_paths.
    fake = dict(manifest, cells=dict(manifest["cells"]))
    fake_rid = next(iter(fake["cells"]))
    fake["cells"][fake_rid] = dict(fake["cells"][fake_rid], cell_type="__not_in_tree__")
    kept_ids = {rid for rid, _ in split_cells(fake, fake["cells"][fake_rid]["split"], hierarchy)}
    assert int(fake_rid) not in kept_ids, "a label absent from the hierarchy was not dropped"
    print(f"drop mechanism verified: dropping {victim!r} removes exactly its {n_victim} train "
          f"cells; an unknown label is filtered rather than raising")

    # --- sampling weights on the real split -------------------------------
    ds = WindowedGraphDatasetLCPN(manifest, "train", window_nm=args.window_nm)
    counts = np.bincount(ds.index_labels, minlength=len(classes))
    weights = inverse_sqrt_class_weights(ds.index_labels, len(classes))

    # Their formula, recomputed independently here rather than reusing ours,
    # so this is a check and not a tautology.
    ref = 1.0 / torch.sqrt(torch.as_tensor(counts, dtype=torch.float32).clamp(min=1.0))
    ref = ref / ref.sum() * len(classes)
    assert torch.allclose(weights, ref), "weights do not match 1/sqrt(count) normalised"
    print("\nper-class sampling weights match 1/sqrt(count) rescaled to mean 1.0")

    # Expected draws per epoch: len(ds) draws, each class drawn in proportion
    # to (its window count x its per-window weight).
    mass = counts * weights.numpy()
    expected = mass / mass.sum() * len(ds)

    print(f"\n{'class':>24}{'windows':>12}{'weight':>10}{'exp draws/ep':>15}{'x resampled':>13}")
    for i, c in enumerate(classes):
        print(f"{c:>24}{counts[i]:>12,}{weights[i]:>10.4f}{expected[i]:>15,.0f}"
              f"{expected[i] / max(counts[i], 1):>13.2f}")

    raw_ratio = counts.max() / max(counts.min(), 1)
    bal_ratio = expected.max() / max(expected.min(), 1e-9)
    print(f"\nimbalance (max/min windows):  before {raw_ratio:>10,.0f}x"
          f"   after resampling {bal_ratio:>8,.1f}x")
    print("The residual tilt is the point: 1/sqrt leaves it, 1/count would flatten it to 1.0x.")

    # --- sampler construction ---------------------------------------------
    sampler = balanced_sampler(ds)
    print(f"\nWeightedRandomSampler: {len(sampler)} draws/epoch over {len(ds)} windows "
          f"(replacement={sampler.replacement})")

    if args.draw:
        # One real draw, to confirm the empirical distribution matches the
        # expectation above rather than only the arithmetic doing so.
        idx = torch.tensor(list(iter(sampler)))
        drawn = np.bincount(ds.index_labels[idx.numpy()], minlength=len(classes))
        print(f"\n{'class':>24}{'expected':>14}{'drawn':>14}{'ratio':>9}")
        for i, c in enumerate(classes):
            print(f"{c:>24}{expected[i]:>14,.0f}{drawn[i]:>14,}{drawn[i] / max(expected[i], 1):>9.3f}")

    # --- the other lever still works --------------------------------------
    node_weights = train_window_counts_by_label(manifest, hierarchy)
    print(f"\n--class-balance loss path: window counts for {len(node_weights)} labels, "
          f"total {sum(node_weights.values()):,.0f}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--window-nm", type=float, default=10000.0)
    p.add_argument("--draw", action="store_true",
                   help="also draw a full epoch of indices and compare empirical to expected")
    raise SystemExit(main(p.parse_args()))
