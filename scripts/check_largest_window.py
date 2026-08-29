"""Finds which cells hold the largest geodesic windows at a given radius.

Reads only `mem_offsets` from each cached .npz -- window sizes are its
consecutive differences, so the far larger `members` array never has to be
decompressed. That keeps a full scan of every cell cheap.

Exists because the batch size a GraphTransformer run can use is set by the
single widest window in the split, not by the mean: attention pads each batch
to its largest member. A tail this long is worth being able to point at a
specific root_id for, rather than treating as an anonymous outlier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data.dataset_lcpn import load_manifest  # noqa: E402
from data.geodesic_window import membership_dir_name  # noqa: E402


def scan(cache_dir: Path, cells: dict) -> tuple[list, dict[str, list]]:
    """(per-cell rows, {cell_type: [window size arrays]}) for one radius."""
    rows = []
    by_class: dict[str, list] = {}
    for path in sorted(cache_dir.glob("*.npz")):
        offsets = np.load(path)["mem_offsets"]
        sizes = np.diff(offsets.astype(np.int64))
        if not len(sizes):
            continue
        center = int(np.argmax(sizes))
        rows.append((int(sizes.max()), path.stem, center, int(sizes.mean()), len(sizes)))
        label = cells.get(path.stem, {}).get("cell_type", "?")
        by_class.setdefault(label, []).append(sizes)
    return rows, by_class


def report_by_class(by_class: dict[str, list], cells: dict) -> None:
    n_cells: dict[str, int] = {}
    for info in cells.values():
        n_cells[info["cell_type"]] = n_cells.get(info["cell_type"], 0) + 1

    print(f"\n{'cell_type':<16}{'cells':>7}{'windows':>10}{'mean_W':>9}{'p50':>7}"
          f"{'p95':>7}{'p99':>7}{'max_W':>7}")
    # Densest first: this table exists to show which classes drive the padded
    # batch width, so ordering by the thing that drives it is the useful one.
    stats = []
    for label, arrays in by_class.items():
        sizes = np.concatenate(arrays)
        stats.append((sizes.mean(), label, sizes))
    for mean_w, label, sizes in sorted(stats, reverse=True):
        print(
            f"{label:<16}{n_cells.get(label, 0):>7}{len(sizes):>10}{mean_w:>9.1f}"
            f"{np.percentile(sizes, 50):>7.0f}{np.percentile(sizes, 95):>7.0f}"
            f"{np.percentile(sizes, 99):>7.0f}{sizes.max():>7}"
        )


def main(args) -> int:
    manifest = load_manifest()
    cells = manifest["cells"]
    cache_dir = Path(__file__).resolve().parent.parent / "data" / membership_dir_name(args.window_nm)
    rows, by_class = scan(cache_dir, cells)
    print(f"{len(rows)} cells in {cache_dir.name}")

    report_by_class(by_class, cells)

    rows.sort(reverse=True)
    print(f"\n{'max_W':>7}{'root_id':>21}{'center':>9}{'mean_W':>8}{'n_nodes':>9}  "
          f"{'cell_type':<12}{'split':<7}")
    for max_w, root_id, center, mean_w, n_nodes in rows[: args.top]:
        info = cells.get(root_id, {})
        print(f"{max_w:>7}{root_id:>21}{center:>9}{mean_w:>8}{n_nodes:>9}  "
              f"{info.get('cell_type', '?'):<12}{info.get('split', '?'):<7}")

    all_max = np.array([r[0] for r in rows])
    print(
        f"\nacross all cells: max {all_max.max()}, p99 {np.percentile(all_max, 99):.0f}, "
        f"median of per-cell maxima {np.median(all_max):.0f}"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--window-nm", type=float, default=40000.0)
    p.add_argument("--top", type=int, default=15)
    raise SystemExit(main(p.parse_args()))
