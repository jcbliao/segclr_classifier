"""List every label hierarchy registered in the shared v3 store.

Read-only: no writes, no CAVE calls, no token needed.
"""

import json

from segclr_db import SegCLRDatabase

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"

db = SegCLRDatabase(root=STORE_ROOT, dataset="microns")

hierarchies = db.list_hierarchies()
print("=== list_hierarchies() ===")
print(hierarchies.to_string())
print()

for hid in hierarchies["hierarchy_id"]:
    h = db.hierarchy(hid)
    print(f"=== hierarchy_id={hid!r} ===")
    print(f"levels: {len(h.level_classes)}")
    for i, level in enumerate(h.level_classes):
        print(f"  level {i} (n={len(level)}): {list(level)}")
    print(f"granular labels in label_paths: {len(h.label_paths)}")
    print("tree:")
    print(json.dumps(h.tree, indent=2, default=str))
    print()

print("=== experiments (which declare a hierarchy_id) ===")
exps = db.list_experiments()
cols = [c for c in ("experiment_id", "kind", "hierarchy_id", "notes") if c in exps.columns]
print(exps[cols].to_string())
print()

print("=== prediction_runs ===")
print(db.list_prediction_runs().to_string())
