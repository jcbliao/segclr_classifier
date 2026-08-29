"""What each classification level looks like, and how imbalanced it is.

For every truncation depth of LAB_HIERARCHY_TREE, reports the classes at that
level and the train-split WINDOW support behind each one. Window support, not
cell count, is what the class weights and the balanced-accuracy denominator are
actually built from (gnn/lcpn.py::compute_node_class_weights), so it is what
decides whether a class is learnable or noise.

Manifest-only: no .pt loading, no store access. Run via sbatch (mit_quicktest).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_lcpn import load_manifest, train_window_counts_by_label  # noqa: E402
from gnn.hierarchy import (  # noqa: E402
    LAB_HIERARCHY_TREE,
    get_local_classifier_nodes,
    parse_hierarchy,
    truncate_hierarchy,
)

MIN_USEFUL_WINDOWS = 10_000


def main() -> int:
    manifest = load_manifest()
    counts = train_window_counts_by_label(manifest)
    full = parse_hierarchy(LAB_HIERARCHY_TREE)

    n_cells_by_label: dict[str, int] = {}
    for info in manifest["cells"].values():
        if info["split"] == "train":
            n_cells_by_label[info["cell_type"]] = n_cells_by_label.get(info["cell_type"], 0) + 1

    print(f"manifest: {len(manifest['cells'])} cells, full hierarchy depth {full.depth}, "
          f"level sizes {[len(c) for c in full.level_classes]}")
    print(f"train windows total: {sum(counts.values()):,.0f}")

    for drop in range(0, full.depth - 1):
        h = full if drop == 0 else truncate_hierarchy(full, drop)
        classes = h.level_classes[-1]
        nodes = get_local_classifier_nodes(h)

        support: dict[str, float] = {c: 0.0 for c in classes}
        cells: dict[str, int] = {c: 0 for c in classes}
        for label, count in counts.items():
            leaf = h.label_paths[label][-1]
            support[leaf] += count
            cells[leaf] += n_cells_by_label.get(label, 0)

        present = {c: v for c, v in support.items() if v > 0}
        thin = [c for c, v in support.items() if 0 < v < MIN_USEFUL_WINDOWS]
        empty = [c for c, v in support.items() if v == 0]
        ratio = max(present.values()) / min(present.values()) if present else float("nan")

        print(f"\n{'=' * 78}")
        print(f"drop {drop} finest level(s)  ->  depth {h.depth}, "
              f"{len(classes)} classes, {len(nodes)} LCPN heads")
        print(f"  imbalance max/min = {ratio:,.0f}x   "
              f"classes with 0 train windows: {len(empty)}   "
              f"under {MIN_USEFUL_WINDOWS:,}: {len(thin)}")
        print(f"  {'class':<24} {'train windows':>15} {'share':>8} {'cells':>7}")
        total = sum(support.values())
        for c in sorted(classes, key=lambda c: -support[c]):
            share = 100 * support[c] / total if total else 0.0
            flag = "  <- empty" if support[c] == 0 else ("  <- thin" if support[c] < MIN_USEFUL_WINDOWS else "")
            print(f"  {c:<24} {support[c]:>15,.0f} {share:>7.2f}% {cells[c]:>7}{flag}")
        if empty:
            print(f"  never receives gradient: {sorted(empty)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
