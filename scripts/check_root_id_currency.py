"""Checks whether the labeled cells' root_ids (from the 2022 m343 label table)
are still LATEST roots in CAVE's live chunkedgraph, or have since been
superseded by further proofreading edits. A stale (non-latest) root_id may
still resolve for skeleton generation, or may hang/fail -- this is a direct
check of the hypothesis that the slow/stuck skeleton fetches are stale ids,
not a bug in which root_ids we're passing.

Run via sbatch (mit_quicktest).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import cave_skeletons as cs  # noqa: E402
from data import public_reader as pr  # noqa: E402


def main() -> int:
    import os

    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("CAVE_TOKEN not set")
        return 2

    fs = pr.get_public_filesystem()
    labels_df = pr.get_celltype_labels(fs)
    root_ids = sorted(int(x) for x in labels_df["seg_id"])
    print(f"{len(root_ids)} labeled root_ids")

    cave_config = cs.default_cave_config(token)
    client = cave_config.build_client()

    print("\nchecking is_latest_roots against the live chunkedgraph ...")
    is_latest = client.chunkedgraph.is_latest_roots(root_ids)
    is_latest = list(is_latest) if not isinstance(is_latest, dict) else [is_latest[r] for r in root_ids]
    n_latest = sum(bool(v) for v in is_latest)
    print(f"{n_latest}/{len(root_ids)} are latest roots; {len(root_ids) - n_latest} are stale")

    stale = [r for r, latest in zip(root_ids, is_latest, strict=True) if not latest]
    print(f"\nfirst 20 stale root_ids: {stale[:20]}")

    if stale:
        print("\nchecking skeletons_exist for a sample of stale root_ids (already generated, despite being stale?) ...")
        sample = stale[:20]
        exists = client.skeleton.skeletons_exist(
            root_ids=sample, datastack_name=cave_config.datastack, skeleton_version=cave_config.skeleton_version
        )
        print(exists)

        print("\nfor comparison, checking a sample of CURRENT (latest) root_ids ...")
        latest_sample = [r for r, latest in zip(root_ids, is_latest, strict=True) if latest][:20]
        exists_latest = client.skeleton.skeletons_exist(
            root_ids=latest_sample, datastack_name=cave_config.datastack, skeleton_version=cave_config.skeleton_version
        )
        print(exists_latest)

    out = {
        "n_total": len(root_ids),
        "n_latest": n_latest,
        "n_stale": len(root_ids) - n_latest,
        "stale_root_ids": stale,
    }
    out_path = Path(__file__).resolve().parent.parent / "data" / "root_id_currency.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
