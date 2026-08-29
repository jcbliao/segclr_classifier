#!/home/jcbliao/rotation/segclr/gnn_classifier/segclr_db/.venv/bin/python
"""Print N random cells from the dataset manifest.

Pure stdlib -- no numpy, no torch, no store access -- so unlike everything else in
scripts/ this runs directly on the login node, no sbatch needed. Needs Python 3.7+
(the system `python3` is 3.6.8); the shebang points at the venv, so just run it:

    ./scripts/sample_cells.py 10

    python scripts/sample_cells.py 10                  # 10 random neurons
    python scripts/sample_cells.py 10 --seed 0         # reproducible
    python scripts/sample_cells.py 10 --tsv            # + type, split, node count
    python scripts/sample_cells.py 6 --stratify        # spread across cell types
    python scripts/sample_cells.py 5 --cell-type PV MC       # granular labels
    python scripts/sample_cells.py 5 --cell-type excitatory  # hierarchy node
    python scripts/sample_cells.py 5 --cell-type IT          # all IT descendants
    python scripts/sample_cells.py 20 --split train
    python scripts/sample_cells.py 10 --include-glia

Root ids go to stdout one per line, so it pipes:

    for r in $(python scripts/sample_cells.py 5); do ... done

"Neuron" is not a hardcoded list of glia to exclude -- it is the leaf set under
LAB_HIERARCHY_TREE's `neuron` branch, so it tracks the hierarchy if that changes.
The seed actually used is always reported on stderr, so an unseeded run that
turns up something interesting can still be reproduced.
"""

import sys

if sys.version_info < (3, 7):
    raise SystemExit(
        "needs Python 3.7+ (gnn/hierarchy.py uses `from __future__ import "
        "annotations`); the system python3 is 3.6.8.\n"
        "  run:  ./scripts/sample_cells.py ...\n"
        "  or:   segclr_db/.venv/bin/python scripts/sample_cells.py ...")

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gnn.hierarchy import LAB_HIERARCHY_TREE, parse_hierarchy  # noqa: E402

MANIFEST = ROOT / "data" / "manifest.json"


def _leaves(node):
    """Every granular label under a hierarchy subtree."""
    if isinstance(node, dict):
        out = set()
        for v in node.values():
            out |= _leaves(v)
        return out
    if isinstance(node, (list, tuple)):
        return {str(x) for x in node}
    return {str(node)}


NEURON_LABELS = _leaves(LAB_HIERARCHY_TREE["neuron"])


def _hierarchy_label_map(tree):
    """Map every hierarchy node name to its granular descendant labels.

    ``ParsedHierarchy.label_paths`` is the single source of truth for the
    classification semantics.  Taking the union over every occurrence also
    handles repeated/padded class names without depending on a fixed depth.
    """
    hierarchy = parse_hierarchy(tree)
    descendants = {}
    for granular, path in hierarchy.label_paths.items():
        descendants.setdefault(granular, set()).add(granular)
        for label in path:
            descendants.setdefault(label, set()).add(granular)
    return descendants


HIERARCHY_LABELS = _hierarchy_label_map(LAB_HIERARCHY_TREE)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int, help="how many cells to draw")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the draw; omitted means a random seed, reported on stderr")
    ap.add_argument("--cell-type", nargs="+", metavar="T",
                    help="restrict to these granular labels or hierarchy nodes; "
                         "a node includes all of its granular descendants")
    ap.add_argument("--split", choices=("train", "test"),
                    help="restrict to one split")
    ap.add_argument("--include-glia", action="store_true",
                    help="draw from all cells, not just the neuron branch")
    ap.add_argument("--stratify", action="store_true",
                    help="spread the draw evenly over cell types instead of "
                         "sampling uniformly, which otherwise returns mostly L4IT")
    ap.add_argument("--tsv", action="store_true",
                    help="also print cell_type, split and node count")
    ap.add_argument("--manifest", default=str(MANIFEST))
    a = ap.parse_args()

    cells = json.loads(Path(a.manifest).read_text())["cells"]
    pool = [(int(r), c) for r, c in cells.items()]
    if not a.include_glia:
        pool = [(r, c) for r, c in pool if c["cell_type"] in NEURON_LABELS]
    if a.cell_type:
        requested = set(a.cell_type)
        unknown = requested - set(HIERARCHY_LABELS)
        if unknown:
            ap.error(f"unknown hierarchy label(s) {sorted(unknown)}; available: "
                     f"{sorted(HIERARCHY_LABELS)}")
        want = set().union(*(HIERARCHY_LABELS[label] for label in requested))
        pool = [(r, c) for r, c in pool if c["cell_type"] in want]
        if not pool and not a.include_glia and want.isdisjoint(NEURON_LABELS):
            ap.error("the requested hierarchy label contains only non-neurons; "
                     "pass --include-glia")
    if a.split:
        pool = [(r, c) for r, c in pool if c["split"] == a.split]

    if not pool:
        ap.error("no cells match those filters")
    if a.n > len(pool):
        ap.error(f"asked for {a.n} but only {len(pool)} cells match")

    seed = a.seed if a.seed is not None else random.randrange(2**31)
    rng = random.Random(seed)

    if a.stratify:
        by_type = {}
        for r, c in pool:
            by_type.setdefault(c["cell_type"], []).append((r, c))
        for v in by_type.values():
            rng.shuffle(v)
        order = sorted(by_type)
        rng.shuffle(order)
        picked, i = [], 0
        # Round-robin over types, so a draw of 6 gives 6 different types where
        # possible rather than 6 L4IT (which is 23% of the neuron pool).
        while len(picked) < a.n:
            took = False
            for t in order:
                if i < len(by_type[t]):
                    picked.append(by_type[t][i])
                    took = True
                    if len(picked) == a.n:
                        break
            if not took:
                break
            i += 1
        chosen = picked
    else:
        chosen = rng.sample(pool, a.n)

    print(f"seed {seed}  ({len(pool)} cells matched"
          f"{'' if a.include_glia else ', neuron branch only'})", file=sys.stderr)
    if a.tsv:
        print("root_id\tcell_type\tsplit\tn_nodes")
    for r, c in chosen:
        if a.tsv:
            print(f"{r}\t{c['cell_type']}\t{c['split']}\t{c['n_nodes_covered']}")
        else:
            print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
