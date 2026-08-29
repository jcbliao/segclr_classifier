"""Validates the two synapse databases built by ``data/build_synapses.py``.

Every way these files can be wrong is silent, so each check below targets one
specific failure:

1. **Coverage** -- every manifest cell accounted for, and how many have no
   synapses on a side. A cell missing entirely means a lost shard, which the
   merge's own missing-part check should already have caught.
2. **Polarity, checked against CAVE itself** -- a sample of rows is re-queried
   by ``synapse_id`` and its ``pre_pt_root_id`` / ``post_pt_root_id`` compared
   with what the file says. This is what catches a swapped ``cell``/``partner``
   prefix, which no internal check can see because both sides are populated
   either way.
3. **Reciprocity** -- for a pair of our own cells, a synapse in A's
   presynaptic file must appear, with the same ``synapse_id``, in B's
   postsynaptic file, with the two ``cell``/``partner`` roles exactly swapped.
   An internal consistency check on polarity that costs no CAVE call.
4. **Units and geometry** -- coordinates inside the MICrONS volume, in
   nanometres. A frame that came back in (4, 4, 40) nm voxels would sit ~1/4
   and ~1/40 of the way into these ranges.
5. **Round trip against a fresh count** -- for a sample of cells, CAVE's own
   row count at this materialization vs. the file's, so a shard that was
   written short shows up as a mismatch rather than as a quiet undercount.

Run via sbatch (mit_normal -- reads two multi-million-row parquets and makes a
handful of CAVE calls).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.build_synapses import MERGED_NAME, OUTPUT_DIR  # noqa: E402
from data.synapses import MODES, build_client, count_synapses, positions_nm  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "data" / "manifest.json"

#: MICrONS minnie65 volume extent in nm, generous bounds -- this is a sanity
#: envelope, not a precise bounding box.
VOLUME_MAX_NM = (2_000_000.0, 2_000_000.0, 1_500_000.0)

N_POLARITY_SAMPLE = 200
N_COUNT_SAMPLE = 10
RNG_SEED = 0


def load(mode: str) -> pd.DataFrame:
    path = OUTPUT_DIR / MERGED_NAME[mode]
    print(f"reading {path.name} ...")
    return pd.read_parquet(path)


def check_coverage(frames: dict[str, pd.DataFrame], root_ids: set[int]) -> None:
    print("\n" + "=" * 70)
    print("1. coverage")
    print("=" * 70)
    for mode, frame in frames.items():
        cells = set(frame["cell_root_id"].unique().tolist())
        stray = cells - root_ids
        print(f"{mode:9s}: {len(frame):>10,} rows  {len(cells):>5}/{len(root_ids)} cells present")
        print(f"           {len(root_ids - cells):>5} cells with zero synapses on this side")
        print(f"           {len(stray):>5} cell_root_ids NOT in the manifest (must be 0)")
        per_cell = frame.groupby("cell_root_id").size()
        print(
            f"           per cell: median {per_cell.median():.0f}, "
            f"p90 {per_cell.quantile(0.9):.0f}, max {per_cell.max()}"
        )
        partners = frame["partner_root_id"]
        print(
            f"           {partners.nunique():,} distinct partner root_ids; "
            f"{(partners == 0).sum():,} rows with partner_root_id=0 (unresolved)"
        )
        print(f"           {(frame['cell_root_id'] == partners).sum():,} rows where partner == cell (autapse or merge)")


def check_units(frames: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 70)
    print("4. units and geometry")
    print("=" * 70)
    for mode, frame in frames.items():
        for prefix in ("cell", "partner", "ctr"):
            xyz = frame[[f"{prefix}_{a}_nm" for a in "xyz"]].to_numpy()
            lo, hi = xyz.min(axis=0), xyz.max(axis=0)
            ok = all(hi[i] <= VOLUME_MAX_NM[i] for i in range(3))
            print(
                f"{mode:9s} {prefix:8s} x[{lo[0]:>9.0f},{hi[0]:>9.0f}] "
                f"y[{lo[1]:>9.0f},{hi[1]:>9.0f}] z[{lo[2]:>9.0f},{hi[2]:>9.0f}]  "
                f"{'ok' if ok else 'OUT OF VOLUME'}"
            )
        # The cleft centroid should sit between the two partner points; far
        # apart would mean the three position columns came from different rows.
        d = np.linalg.norm(
            frame[["cell_x_nm", "cell_y_nm", "cell_z_nm"]].to_numpy()
            - frame[["ctr_x_nm", "ctr_y_nm", "ctr_z_nm"]].to_numpy(),
            axis=1,
        )
        print(f"{mode:9s} |cell_pt - ctr_pt|: median {np.median(d):.0f} nm, p99 {np.percentile(d, 99):.0f} nm")


def check_reciprocity(frames: dict[str, pd.DataFrame], root_ids: set[int]) -> None:
    print("\n" + "=" * 70)
    print("3. reciprocity between our own cells")
    print("=" * 70)
    out, inc = frames["outgoing"], frames["incoming"]
    internal = out[out["partner_root_id"].isin(root_ids)]
    print(f"{len(internal):,} presynaptic rows whose partner is also one of our cells")
    if internal.empty:
        return

    merged = internal.merge(
        inc[["synapse_id", "cell_root_id", "partner_root_id"]],
        on="synapse_id",
        how="left",
        suffixes=("_out", "_in"),
    )
    found = merged["cell_root_id_in"].notna()
    print(f"{found.sum():,}/{len(merged):,} of them appear in the postsynaptic file by synapse_id")
    matched = merged[found]
    roles_swapped = (
        (matched["cell_root_id_in"] == matched["partner_root_id_out"])
        & (matched["partner_root_id_in"] == matched["cell_root_id_out"])
    )
    print(f"{roles_swapped.sum():,}/{len(matched):,} have the two roles exactly swapped (must be all)")
    if len(matched) and not roles_swapped.all():
        print(matched[~roles_swapped].head())


def check_polarity_against_cave(frames: dict[str, pd.DataFrame], client, table: str) -> None:
    print("\n" + "=" * 70)
    print("2. polarity re-queried from CAVE")
    print("=" * 70)
    rng = np.random.default_rng(RNG_SEED)
    for mode, frame in frames.items():
        sample = frame.iloc[rng.choice(len(frame), size=min(N_POLARITY_SAMPLE, len(frame)), replace=False)]
        truth = client.materialize.query_table(
            table,
            filter_in_dict={"id": [int(s) for s in sample["synapse_id"]]},
            split_positions=True,
            desired_resolution=[1, 1, 1],
            metadata=False,
        )
        joined = sample.merge(truth, left_on="synapse_id", right_on="id", how="left")
        cell_col = f"{MODES[mode]['cell']}_root_id"
        partner_col = f"{MODES[mode]['partner']}_root_id"
        agree_cell = (joined["cell_root_id"] == joined[cell_col]).sum()
        agree_partner = (joined["partner_root_id"] == joined[partner_col]).sum()
        print(
            f"{mode:9s}: {len(joined)} sampled; cell side matches CAVE's {cell_col} "
            f"{agree_cell}/{len(joined)}; partner matches {partner_col} {agree_partner}/{len(joined)}"
        )
        pos = positions_nm(joined, MODES[mode]["cell"])
        ours = joined[["cell_x_nm", "cell_y_nm", "cell_z_nm"]].to_numpy(dtype="float64")
        offset = np.abs(pos - ours).max()
        print(f"           max |stored - CAVE| position error: {offset:.1f} nm (float32 rounding only)")


def check_counts_against_cave(frames: dict[str, pd.DataFrame], client, table: str, root_ids: list[int]) -> None:
    print("\n" + "=" * 70)
    print("5. per-cell row counts vs. a fresh CAVE count")
    print("=" * 70)
    rng = np.random.default_rng(RNG_SEED)
    sample = [int(r) for r in rng.choice(np.asarray(root_ids), size=N_COUNT_SAMPLE, replace=False)]
    for mode, frame in frames.items():
        counts = frame.groupby("cell_root_id").size()
        mismatched = 0
        for root_id in sample:
            ours = int(counts.get(root_id, 0))
            theirs = count_synapses(client, [root_id], mode, table)
            flag = "" if ours == theirs else "   MISMATCH"
            mismatched += ours != theirs
            print(f"{mode:9s} {root_id}: file {ours:>6}  CAVE {theirs:>6}{flag}")
        print(f"{mode:9s}: {mismatched}/{len(sample)} mismatched")


def main() -> int:
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)
    root_ids = sorted(int(r) for r in manifest["cells"])
    root_set = set(root_ids)

    with open(OUTPUT_DIR / "summary.json") as fh:
        summary = json.load(fh)
    print(json.dumps(summary, indent=2))

    frames = {mode: load(mode) for mode in ("outgoing", "incoming")}

    check_coverage(frames, root_set)
    check_reciprocity(frames, root_set)
    check_units(frames)

    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("\nCAVE_TOKEN not set -- skipping the two checks that need CAVE")
        return 1
    client = build_client(token, summary["datastack"], summary["mat_version"])
    check_polarity_against_cave(frames, client, summary["synapse_table"])
    check_counts_against_cave(frames, client, summary["synapse_table"], root_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
