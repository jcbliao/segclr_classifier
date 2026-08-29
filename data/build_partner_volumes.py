"""Fetch CAVE L2-chunk counts and segmented volumes for presynaptic partners.

For every distinct nonzero ``partner_root_id`` in postsynaptic_sites.parquet:

1. ask the chunkedgraph for the root's level-2 leaves;
2. ask the L2 cache for each leaf's ``size_nm3``;
3. store ``n_l2_chunks`` and the sum as ``volume_nm3``.

The root-leaf endpoint is one-root-at-a-time.  With millions of roots this is a
large shared-service job, so output is one small, atomic parquet part per input
chunk.  Re-running skips completed parts.  Use ``--limit`` for a throughput
pilot before choosing the SLURM array width.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.synapses import DEFAULT_DATASTACK, DEFAULT_MAT_VERSION, build_client

REPO = Path(__file__).resolve().parent.parent
POST = REPO / "data" / "synapse_cache" / "postsynaptic_sites.parquet"
OUTPUT = REPO / "data" / "partner_volume_cache"
DEFAULT_ROOTS_PER_PART = 100
DEFAULT_L2_BATCH_SIZE = 1_000

SCHEMA = pa.schema([
    pa.field("root_id", pa.uint64(), nullable=False),
    pa.field("n_l2_chunks", pa.uint64(), nullable=False),
    pa.field("n_l2_sizes_missing", pa.uint64(), nullable=False),
    pa.field("volume_nm3", pa.uint64(), nullable=False),
    pa.field("error", pa.string()),
])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--post", type=Path, default=POST)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--datastack", default=DEFAULT_DATASTACK)
    p.add_argument("--mat-version", type=int, default=DEFAULT_MAT_VERSION)
    p.add_argument("--roots-per-part", type=int, default=DEFAULT_ROOTS_PER_PART)
    p.add_argument("--l2-batch-size", type=int, default=DEFAULT_L2_BATCH_SIZE)
    p.add_argument("--request-workers", type=int, default=1,
                   help="concurrent root fetches within this process (I/O-bound)")
    p.add_argument("--root-retries", type=int, default=3,
                   help="attempts per root before recording an error")
    p.add_argument("--sleep", type=float, default=0.1, help="pause after each CAVE request")
    p.add_argument("--limit", type=int, help="pilot on only N distinct roots")
    p.add_argument("--sample", action="store_true",
                   help="with --limit, choose a deterministic hash sample instead of the first IDs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--merge", action="store_true")
    return p.parse_args()


def task_from_env() -> tuple[int, int]:
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
        world = os.environ.get("VOLUME_WORLD", os.environ.get("SLURM_ARRAY_TASK_COUNT", rank + 1))
        return rank, int(world)
    return 0, 1


def load_root_ids(path: Path, limit: int | None, sample: bool = False) -> list[int]:
    suffix = f" limit {int(limit)}" if limit is not None else ""
    ordering = "hash(partner_root_id), partner_root_id" if sample else "partner_root_id"
    rows = duckdb.sql(f"""
        select distinct partner_root_id
        from read_parquet('{path}')
        where partner_root_id != 0
        order by {ordering}
        {suffix}
    """).fetchall()
    return [int(row[0]) for row in rows]


def part_path(output: Path, index: int) -> Path:
    return output / "parts" / f"part_{index:06d}.parquet"


def l2_volume(client, root_id: int, batch_size: int, sleep_s: float) -> dict:
    l2_ids = np.unique(client.chunkedgraph.get_leaves(root_id, stop_layer=2))
    sizes: dict[str, dict] = {}
    for start in range(0, len(l2_ids), batch_size):
        batch = l2_ids[start : start + batch_size]
        sizes.update(client.l2cache.get_l2data(batch, attributes=["size_nm3"]))
        if sleep_s:
            time.sleep(sleep_s)

    values = [sizes.get(str(int(node)), {}).get("size_nm3") for node in l2_ids]
    present = [int(value) for value in values if value is not None]
    return {
        "root_id": root_id,
        "n_l2_chunks": len(l2_ids),
        "n_l2_sizes_missing": len(values) - len(present),
        # This is a lower bound if the cache omitted any L2 sizes; the missing
        # count makes that state explicit rather than silently claiming exactness.
        "volume_nm3": sum(present),
        "error": None,
    }


def write_part(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression="zstd")
    tmp.replace(path)


def build(args) -> int:
    if args.limit is not None and args.output.resolve() == OUTPUT.resolve():
        raise ValueError("--limit requires a separate --output pilot directory")
    roots = load_root_ids(args.post, args.limit, args.sample)
    chunks = [roots[i : i + args.roots_per_part] for i in range(0, len(roots), args.roots_per_part)]
    rank, world = task_from_env()
    mine = [(i, chunk) for i, chunk in enumerate(chunks) if i % world == rank]
    todo = [(i, chunk) for i, chunk in mine if not part_path(args.output, i).exists()]
    print(f"rank {rank}/{world}: {len(roots):,} roots, {len(chunks):,} parts, {len(todo):,} to fetch")
    if args.dry_run or not todo:
        return 0

    token = os.environ.get("CAVE_TOKEN")
    if not token:
        token_path = Path.home() / ".cloudvolume/secrets/global.daf-apis.com-cave-secret.json"
        if token_path.exists():
            token = json.loads(token_path.read_text())["token"]
    if not token:
        raise RuntimeError("CAVE_TOKEN is unset and the standard token file does not exist")
    local = threading.local()

    def fetch_one(root_id: int) -> dict:
        # requests.Session is not guaranteed thread-safe.  Give each request
        # stream its own CAVEclient/session, created lazily in that thread.
        if not hasattr(local, "client"):
            local.client = build_client(token, args.datastack, args.mat_version)
        for attempt in range(1, args.root_retries + 1):
            try:
                return l2_volume(local.client, root_id, args.l2_batch_size, args.sleep)
            except Exception as exc:
                if attempt == args.root_retries:
                    return {"root_id": root_id, "n_l2_chunks": 0,
                            "n_l2_sizes_missing": 0, "volume_nm3": 0,
                            "error": f"{type(exc).__name__}: {exc}"}
                time.sleep(min(2 ** (attempt - 1), 8))

    bar = tqdm(todo, desc=f"rank {rank}", unit="part")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.request_workers) as pool:
        for index, chunk in bar:
            rows = list(pool.map(fetch_one, chunk))
            write_part(rows, part_path(args.output, index))
    return 0


def merge(args) -> int:
    roots = load_root_ids(args.post, args.limit, args.sample)
    n_parts = (len(roots) + args.roots_per_part - 1) // args.roots_per_part
    missing = [i for i in range(n_parts) if not part_path(args.output, i).exists()]
    if missing:
        print(f"REFUSING: {len(missing):,} parts missing; first: {missing[:10]}")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "partner_volumes.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    duckdb.sql(f"""
        copy (select * from read_parquet('{args.output / 'parts' / '*.parquet'}') order by root_id)
        to '{tmp}' (format parquet, compression zstd)
    """)
    tmp.replace(out)
    print(f"wrote {out} ({len(roots):,} roots)")
    return 0


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(merge(parsed) if parsed.merge else build(parsed))
