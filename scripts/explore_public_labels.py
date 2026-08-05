"""Smoke test: fetch the real ground-truth cell-type label table and one raw
embedding CSV shard from the public MICrONS SegCLR release, and report what's
actually in them. Run via sbatch (mit_quicktest) -- see
scripts/sbatch/explore_public_labels.sh. Not meant to write anything; this is
step 0 of ingestion, to ground the schema assumptions in
data/ingest_public_microns.py before it's written for real.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import public_reader as pr  # noqa: E402


def main() -> int:
    fs = pr.get_public_filesystem()

    print("=" * 70)
    print("1. ground-truth cell-type labels")
    print("=" * 70)
    df = pr.get_celltype_labels(fs)
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"dtypes:\n{df.dtypes}")
    print(f"\nhead:\n{df.head(10)}")
    if "cell_type" in df.columns:
        print(f"\ncell_type value_counts:\n{df['cell_type'].value_counts()}")
    if "seg_id" in df.columns:
        print(f"\n#unique seg_id: {df['seg_id'].nunique()}")
        example_root_id = int(df["seg_id"].iloc[0])
    else:
        example_root_id = None

    print()
    print("=" * 70)
    print("2. raw per-node embeddings for one example cell")
    print("=" * 70)
    test_ids = dict(
        microns=864691135293126156,  # from the access gist, known-good test id
    )
    if example_root_id is not None:
        test_ids["from_label_table"] = example_root_id

    for name, root_id in test_ids.items():
        for data_key in ("microns_v343", "microns_nm_coord_public_offset_v343"):
            try:
                cell = pr.get_raw_cell_embeddings(root_id, fs, data_key=data_key)
                print(
                    f"{name} ({root_id}) / {data_key}: "
                    f"{cell.embeddings.shape[0]} nodes, dim={cell.embeddings.shape[1]}"
                )
                if cell.embeddings.shape[0]:
                    print(f"  xyz[0]={cell.xyz_nm[0]}  embedding[0][:5]={cell.embeddings[0][:5]}")
                    print(
                        f"  xyz range: x=[{cell.xyz_nm[:, 0].min():.0f}, "
                        f"{cell.xyz_nm[:, 0].max():.0f}]  "
                        f"y=[{cell.xyz_nm[:, 1].min():.0f}, {cell.xyz_nm[:, 1].max():.0f}]  "
                        f"z=[{cell.xyz_nm[:, 2].min():.0f}, {cell.xyz_nm[:, 2].max():.0f}]"
                    )
                    import numpy as np

                    norms = np.linalg.norm(cell.embeddings, axis=1)
                    print(f"  embedding norms: mean={norms.mean():.3f} std={norms.std():.3f}")
            except Exception as e:  # noqa: BLE001 -- exploratory script, report and continue
                print(f"{name} ({root_id}) / {data_key}: FAILED -- {type(e).__name__}: {e}")

    print()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
