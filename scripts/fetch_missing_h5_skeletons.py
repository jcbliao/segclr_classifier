"""Fetch CAVE skeletons for the h5-only cells (mostly glia) that
scripts/check_h5_skeleton_coverage.py found missing from data/skeleton_cache/.
Resumable via data/cave_skeletons.py::fetch_skeletons (cache-hit skip).

Run via sbatch -- see scripts/sbatch/fetch_missing_h5_skeletons.sh.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.cave_skeletons import default_cave_config, fetch_skeletons  # noqa: E402
missing_path = REPO / "data" / "h5_missing_skeleton_root_ids.json"
root_ids = json.loads(missing_path.read_text())
print(f"{len(root_ids)} root_ids to fetch", flush=True)

token = subprocess.run(
    ["jq", "-r", ".token", str(Path.home() / ".cloudvolume/secrets/global.daf-apis.com-cave-secret.json")],
    capture_output=True, text=True, check=True,
).stdout.strip()

cave_config = default_cave_config(token)
result = fetch_skeletons(root_ids, cave_config)
print(f"done: {len(result)}/{len(root_ids)} skeletons now cached", flush=True)
