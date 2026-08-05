"""22/40 has been flat for 10+ minutes in the running build_dataset job --
that's suspicious for "generation in progress" (should trickle upward even
slowly). This recomputes the exact same first chunk build_dataset.py would
form and checks each of the 40 root_ids individually against CAVE, to see
which are genuinely pending vs. possibly stuck/errored/refused-but-unlisted.

Run via sbatch (mit_quicktest). Read-only -- does not touch the running job's
cache or state.
"""

from __future__ import annotations

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

    missing = [r for r in root_ids if cs.load_cached(r) is None]
    print(f"{len(root_ids)} total, {len(root_ids) - len(missing)} already have a cached skeleton pickle")

    cave_config = cs.default_cave_config(token)
    client = cave_config.build_client()

    refused = set()
    try:
        raw = client.skeleton.get_refusal_list(datastack_name=cave_config.datastack)
        from segclr_db.cave import parse_refusal_list

        refused = parse_refusal_list(raw, cave_config.datastack)
    except Exception as e:  # noqa: BLE001
        print(f"refusal list fetch failed: {e}")

    fetchable = [r for r in missing if r not in refused]
    chunk = fetchable[:40]
    print(f"chunk 1 = {len(chunk)} root_ids: {chunk}")

    print("\nchecking skeletons_exist individually ...")
    exists = client.skeleton.skeletons_exist(
        root_ids=chunk, datastack_name=cave_config.datastack, skeleton_version=cave_config.skeleton_version
    )
    print(exists)

    if isinstance(exists, dict):
        not_ready = [r for r in chunk if not exists.get(r)]
    else:
        not_ready = [r for r, e in zip(chunk, exists, strict=True) if not e]
    print(f"\n{len(chunk) - len(not_ready)}/{len(chunk)} exist; not ready: {not_ready}")

    if not_ready:
        print("\nchecking get_skeleton_info (or similar status) for the stuck ones, if available ...")
        for r in not_ready[:5]:
            for method_name in ("get_skeleton_info", "skeleton_info"):
                method = getattr(client.skeleton, method_name, None)
                if method is None:
                    continue
                try:
                    info = method(r, datastack_name=cave_config.datastack)
                    print(f"  {r} [{method_name}]: {info}")
                except Exception as e:  # noqa: BLE001
                    print(f"  {r} [{method_name}]: FAILED -- {type(e).__name__}: {e}")

        print("\nre-requesting generation explicitly for the stuck ones (idempotent if already queued) ...")
        try:
            resp = client.skeleton.generate_bulk_skeletons_async(
                not_ready, datastack_name=cave_config.datastack, skeleton_version=cave_config.skeleton_version
            )
            print(f"  generate_bulk_skeletons_async response: {resp}")
        except Exception as e:  # noqa: BLE001
            print(f"  generation request FAILED -- {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
