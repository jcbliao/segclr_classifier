"""Submit missing postsynaptic targets to CAVE's asynchronous v4 skeleton queue.

One request contains up to 1,000 roots. Checkpoints record accepted submissions,
not completed skeletons. Re-running skips accepted batches.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from caveclient import CAVEclient
from segclr_db.cave import parse_refusal_list

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / 'analysis/postsynaptic_targets_conf0.7'
OUTPUT = INVENTORY / 'skeleton_generation_v4'
DATASTACK = 'minnie65_public'
VERSION = 4
BATCH_SIZE = 1000


def atomic_json(path, payload):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)


def retry(call):
    for attempt in range(4):
        try:
            return call()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(65)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / 'root_ids.json'
    if not manifest.exists():
        c = duckdb.connect()
        roots = [int(r[0]) for r in c.execute(
            'select root_id from read_parquet(?) where cave_skeleton_available=false order by root_id',
            [str(INVENTORY / 'inventory.parquet')],
        ).fetchall()]
        assert len(roots) == len(set(roots))
        atomic_json(manifest, roots)
    roots = json.loads(manifest.read_text())
    client = CAVEclient(DATASTACK, version=1718)
    raw = retry(lambda: client.skeleton.get_refusal_list(datastack_name=DATASTACK))
    refused = parse_refusal_list(raw, DATASTACK)
    # Keep the raw response for provenance; only explicit matching refusals apply.
    raw.to_csv(OUTPUT / 'service_refusal_list.csv', index=False)
    atomic_json(OUTPUT / 'refused_root_ids.json', sorted(set(roots) & refused))
    print(f'{len(roots):,} missing targets; {len(set(roots) & refused):,} refused', flush=True)
    parts = OUTPUT / 'batches'
    parts.mkdir(exist_ok=True)
    for start in range(0, len(roots), BATCH_SIZE):
        batch = roots[start:start + BATCH_SIZE]
        path = parts / f'batch_{start // BATCH_SIZE:06d}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved['root_ids'] == batch
            continue
        eligible = [r for r in batch if r not in refused]
        exists = retry(lambda: client.skeleton.skeletons_exist(
            root_ids=eligible, datastack_name=DATASTACK, skeleton_version=VERSION,
        )) if eligible else {}
        if isinstance(exists, bool):
            exists = {eligible[0]: exists}
        assert set(exists) == set(eligible), 'Incomplete availability response'
        needed = [r for r in eligible if not exists[r]]
        estimate = None
        if needed:
            estimate = retry(lambda: client.skeleton.generate_bulk_skeletons_async(
                root_ids=needed, datastack_name=DATASTACK, skeleton_version=VERSION,
            ))
            if not isinstance(estimate, (int, float)):
                raise RuntimeError(f'Unexpected generation acknowledgement: {estimate!r}')
        record = {
            'root_ids': batch, 'requested_root_ids': needed,
            'already_available_root_ids': [r for r in eligible if exists[r]],
            'refused_root_ids': [r for r in batch if r in refused],
            'acknowledged_at': datetime.now(timezone.utc).isoformat(),
            'server_estimate_seconds': estimate,
        }
        atomic_json(path, record)
        print(f'batch {start // BATCH_SIZE}: requested {len(needed)}, already available {len(eligible)-len(needed)}, refused {len(batch)-len(eligible)}', flush=True)
        records = [json.loads(p.read_text()) for p in sorted(parts.glob('batch_*.json'))]
        summary = {
            'datastack': DATASTACK, 'materialization': 1718, 'skeleton_version': VERSION,
            'candidate_roots': len(roots), 'batches_acknowledged': len(records),
            'requested': sum(len(r['requested_root_ids']) for r in records),
            'already_available': sum(len(r['already_available_root_ids']) for r in records),
            'refused': sum(len(r['refused_root_ids']) for r in records),
            'unprocessed': len(roots)-sum(len(r['root_ids']) for r in records),
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'note': 'Requested means acknowledged by the async service, not generation complete.',
        }
        atomic_json(OUTPUT / 'summary.json', summary)
        time.sleep(2)
    print('All generation batches processed.', flush=True)


if __name__ == '__main__':
    main()
