"""Cache the nucleus position per root_id, once, so builds need no store access.

Done as its own step for two reasons: the shared store is schema v4 while the
vendored clone reads v3 (see data/store_compat.py), so exactly one job should
carry that shim; and a build array of 100 tasks should not each re-query the
store for the same 2,335 rows.

    sbatch scripts/sbatch/dump_nucleus_positions.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.store_compat import use_v4  # noqa: E402

use_v4()

from data.soma_restrict import nucleus_positions  # noqa: E402

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
OUT = ROOT / "data" / "nucleus_positions.json"


def main() -> int:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    cells = manifest["cells"]
    ids = [int(r) for r in cells]

    nuc = nucleus_positions(STORE_ROOT, root_ids=ids)
    have = [r for r in ids if r in nuc]
    missing = [r for r in ids if r not in nuc]

    from collections import Counter
    by_type = Counter(cells[str(r)]["cell_type"] for r in missing)

    payload = {
        "store_root": STORE_ROOT,
        "schema_version": 4,
        "source": "cells dimension: soma_x_nm / soma_y_nm / soma_z_nm",
        "n_requested": len(ids),
        "n_present": len(have),
        "missing_root_ids": sorted(missing),
        "missing_by_cell_type": dict(by_type),
        "positions": {str(r): list(nuc[r]) for r in have},
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"{len(have)}/{len(ids)} cells have a nucleus position "
          f"({100 * len(have) / len(ids):.1f}%)", flush=True)
    print(f"missing by cell type: {dict(by_type)}", flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
