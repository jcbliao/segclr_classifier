"""Assert gnn/hierarchy.py::HIERARCHY_V2_TREE parses to exactly the taxonomy
the store's `hierarchy_v2` row defines, then exercise the three sweep
configurations against it.

The tree is transcribed into this repo so training doesn't open the store, and
that transcription is the thing that can silently drift: a dropped label or a
mistyped group would not raise anywhere -- it would just train a different
taxonomy than the one the run claims. So the comparison here is against
db.hierarchy("hierarchy_v2"), which is the authority, and it is exact on
level_classes and on every granular label's path.

Also prints per-class cell and window support at the truncated level actually
trained on, and runs one forward/backward for each of meanpool / mpnn+spatial
/ gt, which is what catches a shape mismatch before a 100-epoch job does.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Imported BEFORE the repo root joins sys.path. The repo contains a clone
# directory literally named `segclr_db/`, and with the repo root on the path
# that bare directory resolves as a namespace package ("unknown location"),
# shadowing the editable install whose package root is actually
# segclr_db/src.
from segclr_db import SegCLRDatabase  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from data.dataset_lcpn import (  # noqa: E402
    HIERARCHY_LEVELS_DROPPED,
    load_hierarchy,
    load_manifest,
    train_window_counts_by_label,
)
from gnn.hierarchy import HIERARCHY_V2_TREE, parse_hierarchy  # noqa: E402
from gnn.lcpn import compute_node_class_weights  # noqa: E402
from gnn.model import ModelConfig, WindowClassifier  # noqa: E402

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
HIERARCHY_ID = "hierarchy_v2"

failures: list[str] = []

# --- 1. the transcription matches the store -------------------------------
store = SegCLRDatabase(root=STORE_ROOT, dataset="microns").hierarchy(HIERARCHY_ID)
parsed = parse_hierarchy(HIERARCHY_V2_TREE)

print(f"=== parse_hierarchy(HIERARCHY_V2_TREE) vs store {HIERARCHY_ID!r} ===")
print(f"depth: parsed={parsed.depth} store={len(store.level_classes)}")
for lvl in range(max(parsed.depth, len(store.level_classes))):
    mine = list(parsed.level_classes[lvl]) if lvl < parsed.depth else None
    theirs = list(store.level_classes[lvl]) if lvl < len(store.level_classes) else None
    ok = mine == theirs
    print(f"  level {lvl}: {'OK ' if ok else 'MISMATCH'} n={len(mine or [])} {mine}")
    if not ok:
        failures.append(f"level {lvl}: parsed {mine} != store {theirs}")
        print(f"      store: {theirs}")

# Paths, not just the class lists: two trees can agree on which classes exist
# at each level while disagreeing on which label routes where.
store_paths = {k: list(v) for k, v in store.label_paths.items()}
mine_paths = {k: list(v) for k, v in parsed.label_paths.items()}
if set(store_paths) != set(mine_paths):
    failures.append(
        f"granular label sets differ: only-parsed={sorted(set(mine_paths) - set(store_paths))} "
        f"only-store={sorted(set(store_paths) - set(mine_paths))}"
    )
for label in sorted(set(store_paths) & set(mine_paths)):
    if store_paths[label] != mine_paths[label]:
        failures.append(f"path for {label!r}: parsed {mine_paths[label]} != store {store_paths[label]}")
print(f"granular labels: parsed={len(mine_paths)} store={len(store_paths)}")

# --- 2. the level actually trained on -------------------------------------
manifest = load_manifest()
hierarchy = load_hierarchy(manifest)
classes = hierarchy.level_classes[-1]
print(f"\n=== trained level (depth {hierarchy.depth}, dropped {HIERARCHY_LEVELS_DROPPED}) ===")
print(f"{len(classes)} classes: {classes}")

cell_counts = {c: 0 for c in classes}
window_counts = {c: 0 for c in classes}
split_counts = {c: {"train": 0, "test": 0} for c in classes}
unmapped: set[str] = set()
for info in manifest["cells"].values():
    label = info["cell_type"]
    if label not in hierarchy.label_paths:
        unmapped.add(label)
        continue
    coarse = hierarchy.label_paths[label][-1]
    cell_counts[coarse] += 1
    window_counts[coarse] += info["n_nodes_covered"]
    split_counts[coarse][info["split"]] += 1

print(f"\n{'class':<24}{'cells':>8}{'train':>8}{'test':>8}{'windows':>12}")
for c in classes:
    print(
        f"{c:<24}{cell_counts[c]:>8}{split_counts[c]['train']:>8}"
        f"{split_counts[c]['test']:>8}{window_counts[c]:>12}"
    )
if unmapped:
    failures.append(f"manifest labels absent from the hierarchy: {sorted(unmapped)}")
for c in classes:
    if split_counts[c]["train"] == 0 or split_counts[c]["test"] == 0:
        print(f"  NOTE: {c} has an empty split (train={split_counts[c]['train']} test={split_counts[c]['test']})")

# --- 3. the three sweep configurations build and step ----------------------
weights = compute_node_class_weights(hierarchy, train_window_counts_by_label(manifest))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=== forward/backward on {device} ===")

# Two tiny windows, sized like the real ones (~10.7 nodes average).
n0, n1 = 11, 4
n = n0 + n1
x = torch.randn(n, 64, device=device)
edge_index = torch.tensor(
    [[i, i + 1] for i in range(n0 - 1)] + [[i + 1, i] for i in range(n0 - 1)]
    + [[n0 + i, n0 + i + 1] for i in range(n1 - 1)] + [[n0 + i + 1, n0 + i] for i in range(n1 - 1)],
    device=device,
).t().contiguous()
batch_index = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)]).to(device)
pos_enc = torch.randn(n, 8, device=device)
rel_pos = torch.randn(n, 3, device=device)
y_levels = torch.tensor(
    [[hierarchy.level_maps[lvl][hierarchy.label_paths["L4IT"][lvl]] for lvl in range(hierarchy.depth)],
     [hierarchy.level_maps[lvl][hierarchy.label_paths["astrocyte"][lvl]] for lvl in range(hierarchy.depth)]],
    device=device,
)

for name, cfg in (
    ("meanpool", ModelConfig(architecture="mean")),
    ("mpnn_L2_spatial", ModelConfig(architecture="mpnn", mpnn_layers=2, use_spatial_features=True)),
    ("gt_L4_H4", ModelConfig(architecture="graph_transformer", gt_depth=4, gt_heads=4)),
):
    try:
        model = WindowClassifier(cfg, hierarchy=hierarchy).to(device)
        model.cls_head.set_class_weights(weights)
        g = model(x, edge_index, batch_index, pos_enc=pos_enc, rel_pos=rel_pos)
        loss = model.cls_head.compute_loss(g, y_levels)
        loss.backward()
        preds = model.cls_head.predict_top_down(g)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"  {name:<18} g={tuple(g.shape)} loss={loss.item():.4f} "
            f"preds={preds[:, -1].tolist()} params={n_params:,}"
        )
    except Exception as exc:  # noqa: BLE001 -- report all three, don't stop at the first
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  {name:<18} FAILED: {type(exc).__name__}: {exc}")

print("\n=== result ===")
if failures:
    for f in failures:
        print(f"FAIL: {f}")
    raise SystemExit(1)
print("all checks passed")
