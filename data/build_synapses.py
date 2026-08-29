"""manifest.json -> two synapse databases for the project's cells.

    data/synapse_cache/presynaptic_sites.parquet    our cell is PREsynaptic
    data/synapse_cache/postsynaptic_sites.parquet   our cell is POSTsynaptic

Both hold one row per synapse, with our cell's side under ``cell_*`` and the
other cell's under ``partner_*`` -- so ``partner_root_id`` is the postsynaptic
target in the presynaptic file and the presynaptic source in the postsynaptic
one. Identical columns in both (:data:`data.synapses.SCHEMA`), which is what
makes them concatenable. See :mod:`data.synapses` for where polarity comes from
and what is silent when it goes wrong.

Usage
-----
    # what would run, querying nothing
    python -u data/build_synapses.py --dry-run

    # one rank, a few cells, to measure throughput before committing an array
    python -u data/build_synapses.py --limit 20

    # the real run, as a SLURM array (rank/world come from the environment)
    NUM_TASKS=4 sbatch --array=0-3%4 scripts/sbatch/build_synapses.sh

    # once every part exists, fuse the shards into the two databases
    python -u data/build_synapses.py --merge

Re-running is how you resume: a chunk whose parquet part already exists is
skipped, so a preempted rank loses at most the chunk it was on. ``--redo``
re-queries instead. Parts are per (mode, chunk) files and each rank owns a
disjoint set of chunk indices, so ranks never write the same path.

Cells are chunked and queried in groups rather than one at a time because
``filter_in_dict`` takes a list: ~230 requests per direction instead of ~2,300,
against a service other labs share.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.synapses import (  # noqa: E402
    DEFAULT_DATASTACK,
    DEFAULT_MAT_VERSION,
    DEFAULT_SYNAPSE_TABLE,
    MODES,
    SCHEMA,
    build_client,
    fetch_synapses,
)

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "data" / "manifest.json"
OUTPUT_DIR = REPO / "data" / "synapse_cache"

#: The file each mode fuses into, named for our cell's role rather than for the
#: query direction -- "presynaptic sites" is what the rows are.
MERGED_NAME = {"outgoing": "presynaptic_sites.parquet", "incoming": "postsynaptic_sites.parquet"}

DEFAULT_CHUNK_SIZE = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--datastack", default=DEFAULT_DATASTACK)
    parser.add_argument(
        "--mat-version",
        type=int,
        default=DEFAULT_MAT_VERSION,
        help="materialization the root_ids belong to; must match the manifest's",
    )
    parser.add_argument("--synapse-table", default=DEFAULT_SYNAPSE_TABLE)
    parser.add_argument(
        "--modes",
        default="outgoing,incoming",
        help="which directions to build (comma-separated)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="pause between CAVE calls; this is a shared service",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap the cell list, for a pilot")
    parser.add_argument("--redo", action="store_true", help="re-query chunks that already have a part")
    parser.add_argument("--dry-run", action="store_true", help="report the plan, query nothing")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="fuse the parts into the two databases and write a summary; queries nothing",
    )
    return parser.parse_args()


def task_from_env() -> tuple[int, int]:
    """``(rank, world)`` from a job array or srun, defaulting to a solo run."""
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
        count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        return rank, int(count) if count else rank + 1
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"]), int(os.environ.get("SLURM_NTASKS", 1))
    return 0, 1


def load_root_ids(manifest_path: Path, limit: int | None) -> tuple[list[int], dict]:
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    root_ids = sorted(int(r) for r in manifest["cells"])
    if limit is not None:
        root_ids = root_ids[:limit]
    return root_ids, manifest


def plan_chunks(root_ids: list[int], chunk_size: int) -> list[list[int]]:
    """Fixed-size, order-stable groups of cells.

    Deterministic from the sorted cell list, so a chunk index means the same
    cells on every rank and on every resumed run -- which is what lets a part
    file on disk stand for "these cells are done". Contiguous rather than
    strided, so each merged database comes out sorted by ``cell_root_id`` and
    parquet row-group statistics can prune a per-cell read.
    """
    return [root_ids[i : i + chunk_size] for i in range(0, len(root_ids), chunk_size)]


def part_path(output: Path, mode: str, index: int) -> Path:
    return output / mode / f"part_{index:05d}.parquet"


def write_part(frame: pd.DataFrame, path: Path) -> None:
    """Write one chunk's rows, atomically.

    Via a temp file and a rename: a part killed mid-write would otherwise be a
    truncated parquet that the resume logic counts as done.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, schema=SCHEMA, preserve_index=False)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def build(args) -> int:
    rank, world = task_from_env()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in MODES:
            print(f"unknown mode {mode!r}; known: {', '.join(MODES)}")
            return 2

    root_ids, manifest = load_root_ids(args.manifest, args.limit)
    chunks = plan_chunks(root_ids, args.chunk_size)
    mine = [(i, c) for i, c in enumerate(chunks) if i % world == rank]

    manifest_mat = manifest.get("label_mat_version")
    print(f"rank {rank}/{world}: {len(root_ids)} cells -> {len(chunks)} chunks, {len(mine)} mine")
    print(f"datastack={args.datastack} mat_version={args.mat_version} table={args.synapse_table}")
    if manifest_mat is not None and int(manifest_mat) != int(args.mat_version):
        # Different materialization, different chunkedgraph state: the same
        # physical cell can carry a different root_id, and nothing downstream
        # would notice the mismatch.
        print(f"REFUSING: manifest is mat_version {manifest_mat}, query is {args.mat_version}")
        return 2

    todo = [
        (i, c, mode)
        for i, c in mine
        for mode in modes
        if args.redo or not part_path(args.output, mode, i).exists()
    ]
    print(f"{len(todo)} (chunk, mode) parts to fetch; {len(mine) * len(modes) - len(todo)} already on disk")

    if args.dry_run:
        for i, c, mode in todo[:5]:
            print(f"  would fetch {mode} chunk {i}: {len(c)} cells, first={c[0]}")
        return 0
    if not todo:
        return 0

    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("CAVE_TOKEN not set")
        return 2
    client = build_client(token, args.datastack, args.mat_version)

    started = time.time()
    n_rows = 0
    bar = tqdm(todo, desc=f"rank {rank}", unit="part")
    for index, cells, mode in bar:
        frame = fetch_synapses(
            client, cells, mode, synapse_table=args.synapse_table, sleep_s=args.sleep
        )
        write_part(frame, part_path(args.output, mode, index))
        n_rows += len(frame)
        bar.set_postfix(rows=n_rows, mode=mode)
    bar.close()

    elapsed = time.time() - started
    tqdm.write(
        f"rank {rank}: {len(todo)} parts, {n_rows} rows in {elapsed / 60:.1f} min "
        f"({elapsed / max(len(todo), 1):.1f} s/part)"
    )
    return 0


def merge(args) -> int:
    """Fuse each mode's parts into one database, checking coverage as it goes."""
    root_ids, manifest = load_root_ids(args.manifest, args.limit)
    chunks = plan_chunks(root_ids, args.chunk_size)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    summary = {
        "datastack": args.datastack,
        "mat_version": args.mat_version,
        "synapse_table": args.synapse_table,
        "n_cells": len(root_ids),
        "chunk_size": args.chunk_size,
        "built_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modes": {},
    }

    for mode in modes:
        missing = [i for i in range(len(chunks)) if not part_path(args.output, mode, i).exists()]
        if missing:
            print(f"{mode}: {len(missing)} parts missing (first: {missing[:5]}) -- not merging")
            return 1

        out_path = args.output / MERGED_NAME[mode]
        tmp = out_path.with_suffix(".parquet.tmp")
        writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
        # Streaming part by part rather than concatenating: the merged files run
        # to millions of rows, and there is no reason to hold them all at once.
        rows = 0
        seen: set[int] = set()
        partners: set[int] = set()
        unresolved = 0
        for index in tqdm(range(len(chunks)), desc=f"merge {mode}", unit="part"):
            table = pq.read_table(part_path(args.output, mode, index), schema=SCHEMA)
            writer.write_table(table)
            rows += table.num_rows
            cells = table.column("cell_root_id").to_pylist()
            seen.update(cells)
            partner_col = table.column("partner_root_id").to_pylist()
            partners.update(partner_col)
            unresolved += sum(1 for p in partner_col if p == 0)
        writer.close()
        tmp.replace(out_path)

        silent = sorted(set(root_ids) - seen)
        summary["modes"][mode] = {
            "file": MERGED_NAME[mode],
            "rows": rows,
            "cells_with_synapses": len(seen),
            "cells_with_none": len(silent),
            "distinct_partners": len(partners - {0}),
            "rows_with_unresolved_partner": unresolved,
        }
        print(
            f"{mode} -> {out_path.name}: {rows} rows, {len(seen)}/{len(root_ids)} cells, "
            f"{len(partners - {0})} distinct partners, {unresolved} rows with partner_root_id=0"
        )

    with open(args.output / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {args.output / 'summary.json'}")
    return 0


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return merge(args) if args.merge else build(args)


if __name__ == "__main__":
    raise SystemExit(main())
