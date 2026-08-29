"""Generate the two analysis notebooks from the summary. Run, then execute them."""
from __future__ import annotations
import sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "analysis"
KERNEL = {"display_name": "segclr_db (.venv)", "language": "python", "name": "segclr_db"}

PREAMBLE = '''import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "analysis" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

# Root of the database, holding BOTH units: paths/ and neighborhoods/.
from data.build_embedding_paths import DEFAULT_OUT as DB_ROOT
from data.soma_restrict import DEFAULT_SOMA_RADIUS_NM
RADIUS_UM = DEFAULT_SOMA_RADIUS_NM / 1000
S = np.load(REPO_ROOT / "analysis" / "embedding_paths_summary.npz", allow_pickle=False)
CONFIGS = [str(c) for c in S["configs"]]
NCONFIGS = [str(c) for c in S["nconfigs"]]
CABLE = [c for c in NCONFIGS if c.startswith("cable")]
NNODE = [c for c in NCONFIGS if c.startswith("n")]
NPAIRS = [("n10", "cable20um"), ("n20", "cable40um"), ("n40", "cable80um")]
UM = [c for c in CONFIGS if c.endswith("um")]
NODE = [c for c in CONFIGS if c.endswith("node")]

# Matched pairs. Node spacing is ~2 um, so k nodes spans about 2k um -- pairing
# 10node with 10um would compare objects differing two-fold in extent before
# anything about the model is varied. These are the like-for-like comparisons.
PAIRS = [("10node", "20um"), ("20node", "40um"), ("40node", "80um")]

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})
COLOR = {"cable10um": "#86b6ef", "cable20um": "#2a78d6",
         "cable40um": "#1c5cab", "cable80um": "#0d366b",
         "n10": "#eb6834", "n20": "#e34948", "n40": "#4a3aa7",
         "10um": "#86b6ef", "20um": "#2a78d6", "40um": "#1c5cab", "80um": "#0d366b",
         "10node": "#eb6834", "20node": "#e34948", "40node": "#4a3aa7"}
print(f"{len(CONFIGS)} configs, summary loaded")'''


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


def geodesics_nb():
    c = [
        md("""# Geodesic length of centred embedding paths

What this measures, and why it is not obvious in advance.

The path database holds, for every skeleton node outside the 5 µm perisomatic
ball, **every distinct route through that node** at six budgets. Three budgets are
set in **geodesic length** (10 / 20 / 40 µm total, so L/2 per arm) and three in
**node count** (10 / 20 / 40 nodes, so k/2 per arm).

The two families are not interchangeable. A length budget is a physical distance;
a node budget is a count of embedding sites, and how much cable that spans depends
entirely on skeleton node spacing. This notebook measures the difference.

Every path is confined to one connected component of the cut skeleton, so no path
crosses between two neurites that only ever met at the soma."""),
        code(PREAMBLE),
        md("## 1. Headline table\n\nPercentiles are exact to the 500 nm histogram bin; node counts come from the path offsets."),
        code('''rows = []
for c in CONFIGS:
    q = S[f"{c}__pct_q"]; nm = S[f"{c}__pct_nm"]
    p = dict(zip(q.tolist(), nm.tolist()))
    nodes = S[f"{c}__sample_nodes"]
    rows.append({
        "config": c,
        "paths": int(S[f"{c}__n_paths"][0]),
        "nodes/path p50": float(np.median(nodes)),
        "nodes/path p90": float(np.percentile(nodes, 90)),
        "geodesic p50 (µm)": p[50] / 1000,
        "geodesic p90 (µm)": p[90] / 1000,
        "geodesic p99 (µm)": p[99] / 1000,
        "mean (µm)": float(S[f"{c}__mean_nm"][0]) / 1000,
    })
tbl = pd.DataFrame(rows).set_index("config")
tbl.style.format({"paths": "{:,.0f}", "nodes/path p50": "{:.0f}", "nodes/path p90": "{:.0f}",
                  "geodesic p50 (µm)": "{:.1f}", "geodesic p90 (µm)": "{:.1f}",
                  "geodesic p99 (µm)": "{:.1f}", "mean (µm)": "{:.1f}"})'''),
        md("""## 2. Matching a node budget to a length budget

Measured node spacing is ~2 µm, so **k nodes spans roughly 2k µm**. That is why the
length budgets run 10 / 20 / 40 / **80** µm: each node budget gets a length budget
of comparable extent to sit beside.

    10 nodes  <->  20 µm
    20 nodes  <->  40 µm
    40 nodes  <->  80 µm

Pairing `10node` with `10um` instead would compare two objects differing two-fold
in physical extent before anything about the model had been varied. Below: how
closely each matched pair actually lines up."""),
        code('''fig, ax = plt.subplots(figsize=(7.4, 3.6))
x = np.arange(len(PAIRS))
nd_med = [tbl.loc[n, "geodesic p50 (µm)"] for n, _ in PAIRS]
um_med = [tbl.loc[u, "geodesic p50 (µm)"] for _, u in PAIRS]
ax.bar(x - 0.19, nd_med, 0.36, label="node budget", color="#eb6834")
ax.bar(x + 0.19, um_med, 0.36, label="matched length budget", color="#2a78d6")
for xi, (n, u) in enumerate(zip(nd_med, um_med)):
    ax.text(xi - 0.19, n, f"{n:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(xi + 0.19, u, f"{u:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"{n} vs {u}" for n, u in PAIRS])
ax.set_ylabel("median geodesic length (µm)")
ax.set_title("Matched pairs: node budget vs the length budget of similar extent")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print("matched pairs, median geodesic length")
for (n, u), a, b in zip(PAIRS, nd_med, um_med):
    print(f"  {n:>7} {a:6.1f} µm   vs   {u:>5} {b:6.1f} µm    ratio {a/b:.2f}x")
print()
print("the naive (same-number) pairing, for contrast")
for n, u in zip(NODE, ["10um", "20um", "40um"]):
    a, b = tbl.loc[n, "geodesic p50 (µm)"], tbl.loc[u, "geodesic p50 (µm)"]
    print(f"  {n:>7} {a:6.1f} µm   vs   {u:>5} {b:6.1f} µm    ratio {a/b:.2f}x")'''),
        md("""## 3. Full distributions

**On reading the y-axis.** These are densities, so each curve integrates to 1 over
its own support — a distribution spread over a wider range is necessarily shorter.
That alone accounts for the difference in height between the two families, and it
is normalisation, not concentration:

| config | support | peak density |
|---|---|---|
| `10um` | 0–10.0 µm | 0.34 |
| `80um` | 0–80.0 µm | 0.18 |
| `10node` | 0–69.5 µm | 0.12 |
| `40node` | 0–215.6 µm | 0.031 |

The supports differ because **a length budget is hard-capped and a node budget is
not**. A `10um` arm physically cannot exceed 5 µm, so the config piles up against
its cap at exactly 10.0 µm. A `10node` path has no length cap at all — it takes 5
steps whatever they measure — so it reaches 69.5 µm where spacing is coarse. Bin
width is *not* the cause: recomputing on common 1 µm bins moves the peaks by under
8%.

So the panels below are drawn as **matched pairs on shared bins and shared axes**,
which is the only way the two families can be compared by eye."""),
        code('''fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for ax, (nd, um) in zip(axes, PAIRS):
    hi = max(S[f"{nd}__sample_nm"].max(), S[f"{um}__sample_nm"].max()) / 1000
    edges = np.linspace(0, hi, 121)          # one shared bin set per panel
    for c in (um, nd):
        ax.hist(S[f"{c}__sample_nm"] / 1000, bins=edges, histtype="step",
                lw=1.6, density=True, color=COLOR[c], label=c)
    ax.set_xlim(0, hi); ax.set_xlabel("geodesic length (µm)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{nd} vs {um}")
axes[0].set_ylabel("density")
fig.suptitle("Matched pairs, shared bins and axes", y=1.02, fontsize=10)
plt.tight_layout(); plt.show()'''),
        md("""The length budget stacks against its cap; the node budget leaks past
it on both sides. An ECDF makes that exact — it has no bin width and no density
normalisation, so all seven are directly comparable on one axis."""),
        code('''fig, ax = plt.subplots(figsize=(7.4, 4))
for c in CONFIGS:
    v = np.sort(S[f"{c}__sample_nm"] / 1000)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=1.6, color=COLOR[c],
            ls="--" if c.endswith("um") else "-", label=c)
ax.set_xlim(0, 160); ax.set_xlabel("geodesic length (µm)")
ax.set_ylabel("fraction of paths at most this long")
ax.set_title("ECDF: dashed = length budget, solid = node budget")
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()

print(f"{'config':>8} {'support':>16} {'p50':>7} {'p90':>7} {'p99':>7}  (µm)")
for c in CONFIGS:
    v = S[f"{c}__sample_nm"] / 1000
    print(f"{c:>8} {f'{v.min():.1f} - {v.max():.1f}':>16} "
          f"{np.median(v):>7.1f} {np.percentile(v, 90):>7.1f} "
          f"{np.percentile(v, 99):>7.1f}")'''),
        md("## 4. Implied node spacing\n\nIf a k-node path spans D µm over its k−1 steps, the implied spacing is D/(k−1). This is the quantity that makes the two families diverge."),
        code('''sp = []
for c, k in zip(NODE, (11, 21, 41)):
    v = S[f"{c}__sample_nm"] / 1000
    n = S[f"{c}__sample_nodes"].astype(float)
    ok = n > 1
    step = v[ok] / (n[ok] - 1)
    sp.append({"config": c, "nodes/path p50": float(np.median(n[ok])),
               "spacing p50 (µm)": float(np.median(step)),
               "spacing p90 (µm)": float(np.percentile(step, 90))})
sd = pd.DataFrame(sp).set_index("config")
display(sd.style.format("{:.2f}"))

fig, ax = plt.subplots(figsize=(6.4, 3.2))
for c in NODE:
    v = S[f"{c}__sample_nm"] / 1000
    n = S[f"{c}__sample_nodes"].astype(float)
    ok = n > 1
    ax.hist(v[ok] / (n[ok] - 1), bins=np.linspace(0, 8, 160), histtype="step",
            lw=1.5, density=True, color=COLOR[c], label=c)
ax.set_xlabel("implied inter-node spacing (µm)"); ax.set_ylabel("density")
ax.set_title("Skeleton node spacing, as seen by node-budgeted paths")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()'''),
        md("## 5. Per cell type\n\nMedian geodesic length per cell, grouped by type. Types whose skeletons are sampled more finely give shorter node-budgeted paths."),
        code('''ct = {int(r): str(t) for r, t in zip(S["restrict_root_id"], S["restrict_cell_type"])}
frames = {}
for c in CONFIGS:
    rid = S[f"{c}__cell_root_id"]
    frames[c] = pd.Series(S[f"{c}__cell_median_nm"] / 1000,
                          index=[ct.get(int(r), "?") for r in rid])
by_type = pd.DataFrame({c: frames[c].groupby(level=0).median() for c in CONFIGS})
by_type["n cells"] = frames[CONFIGS[0]].groupby(level=0).size()
by_type = by_type.sort_values("40node", ascending=False)
by_type.style.format({**{c: "{:.1f}" for c in CONFIGS}, "n cells": "{:.0f}"})'''),
        code('''fig, ax = plt.subplots(figsize=(9, 3.6))
sub = by_type[by_type["n cells"] >= 10]
xs = np.arange(len(sub))
for n, u in PAIRS:
    ax.plot(xs, sub[n], "o-", ms=4, lw=1.4, color=COLOR[n], label=n)
    ax.plot(xs, sub[u], "s--", ms=3, lw=1.1, color=COLOR[u], alpha=0.85, label=u)
ax.set_xticks(xs); ax.set_xticklabels(sub.index, rotation=45, ha="right")
ax.set_ylabel("median geodesic length (µm)")
ax.set_title("Cell types with >= 10 cells")
ax.legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout(); plt.show()'''),
        md("""## 6. How many paths does a node have?

One path per node would be the answer if skeletons never branched. They do, and
every route is enumerated, so a node near a branch point carries several. Neurons
are close to unbranched at these scales; glia are not."""),
        code('''rows = []
for c in CONFIGS:
    rid = S[f"{c}__cell_root_id"]
    types = np.array([ct.get(int(r), "?") for r in rid])
    rows.append({"config": c,
                 "mean paths/node": float(np.average(S[f"{c}__cell_mean_ppn"])),
                 "max paths/node": float(S[f"{c}__cell_max_ppn"].max()),
                 "mean, neurons": float(S[f"{c}__cell_mean_ppn"][
                     ~np.isin(types, ["astrocyte", "microglia", "oligo", "OPC"])].mean()),
                 "mean, glia": float(S[f"{c}__cell_mean_ppn"][
                     np.isin(types, ["astrocyte", "microglia", "oligo", "OPC"])].mean())})
pd.DataFrame(rows).set_index("config").style.format("{:.2f}")'''),
        md("""## 7. Reading one cell's paths, and joining embeddings

`path_nodes` indexes the **restricted** node array. `cache_index` maps that to the
row in `graph_cache/<root_id>.pt`, and `orig_node_ids` is the CAVE skeleton vertex
id — the real foreign key. Both are stored so neither has to be re-derived by
matching coordinates."""),
        code('''import torch

rid = int(S["10um__cell_root_id"][0])
P = np.load(DB_ROOT / "paths" / "10um" / f"{rid}.npz")
d = torch.load(REPO_ROOT / "data" / "graph_cache" / f"{rid}.pt", weights_only=False)

off, nodes = P["path_offsets"], P["path_nodes"]
k = int(np.argmax(np.diff(off)))                    # the longest path in this cell
p = nodes[off[k]:off[k + 1]]

emb = d.x.numpy()[P["cache_index"][p]]              # (len(path), 64) embeddings
print(f"cell {rid}: {len(off)-1:,} paths")
print(f"path {k}: {len(p)} nodes, {P['geodesic_nm'][k]:,.0f} nm, "
      f"centred on restricted node {P['center_node'][k]} at position {P['center_at'][k]}")
print(f"CAVE skeleton vertex ids: {P['orig_node_ids'][p][:6]} ...")
print(f"embedding sequence: {emb.shape}")'''),
        md("""## What to take from this

- **Compare a node budget to the length budget of matching extent**, not the one
  with the same number: 10node vs 20um, 20node vs 40um, 40node vs 80um. The
  same-number pairing differs about two-fold in physical extent, which would
  confound any model comparison built on it.
- Even matched, the pairs are not identical: a node budget has a **long right
  tail** wherever spacing is coarse, while a length budget is bounded by
  construction. Matching the medians does not match the distributions.
- **Length-budgeted paths under-fill their nominal diameter** by up to one edge per
  arm, because an arm stops before overshooting. That is a property of the
  definition, not a bug.
- **Paths per node is a branching measure**, and it separates glia from neurons far
  more sharply than it separates neuron types."""),
    ]
    nb = nbf.v4.new_notebook(cells=c)
    nb.metadata.kernelspec = KERNEL
    return nb


def restriction_nb():
    c = [
        md("""# The perisomatic cut

Before any unit is built — path or neighbourhood — every node within **5 µm
(Euclidean) of the nucleus** is removed. The radius is `data.soma_restrict.DEFAULT_SOMA_RADIUS_NM`, it is recorded
in every output file as `soma_radius_nm`, and the output directory is named after
it (`r5um/`) so two radii can never be mixed. The soma is shared by every neurite, so a path crossing it would join
branches that are otherwise far apart geodesically, and would describe where the
soma is rather than what the process looks like.

Removing that ball **disconnects the skeleton**, which is the intended outcome: the
primary neurites meet only at the soma, so cutting it leaves roughly one component
per surviving neurite, plus a fragment wherever an arbor re-enters the ball. Every
path is then confined to a single component.

Distance is Euclidean to the nucleus centroid, taken from the store's own `cells`
dimension (`soma_x_nm` / `soma_y_nm` / `soma_z_nm`), keyed by `root_id`. It is a
point in the volume, not a skeleton node, so there is no geodesic distance to it.

**Cells the store has no nucleus position for are built uncut** — every node kept —
rather than excluded. `cut_applied` marks them; section 4 covers them."""),
        code(PREAMBLE),
        code('''rid = S["restrict_root_id"]; before = S["restrict_before"]
after = S["restrict_after"]; comps = S["restrict_components"]
ctype = S["restrict_cell_type"]; cut = S["restrict_cut_applied"]
dropped = before - after
R = pd.DataFrame({"root_id": rid, "cell_type": ctype, "before": before,
                  "after": after, "dropped": dropped,
                  "dropped_frac": dropped / np.maximum(before, 1),
                  "components": comps, "cut": cut})
# Everything below describes the *cut* population. An uncut cell trivially keeps
# every node and stays connected, so pooling the two would dilute both numbers.
Rc = R[R["cut"]]
print(f"{len(R):,} cells ({len(Rc):,} cut, {(~R['cut']).sum()} uncut)")
print(f"cut cells: {Rc['before'].sum():,} -> {Rc['after'].sum():,} nodes "
      f"({100*Rc['after'].sum()/Rc['before'].sum():.1f}% kept, "
      f"{Rc['dropped'].sum():,} dropped)")
print(f"components (cut cells): median {Rc['components'].median():.0f}, "
      f"mean {Rc['components'].mean():.1f}, max {Rc['components'].max()}")
Rc.describe().T.style.format("{:,.2f}")'''),
        md("## 1. How much of each cell the cut removes\n\nThe fraction removed is small for most cells — the ball is 5 µm against arbors that run hundreds of µm — but it is not uniform across types."),
        code('''fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].hist(Rc["dropped_frac"] * 100, bins=60, color="#2a78d6")
axes[0].set_xlabel(f"% of nodes within {RADIUS_UM:g} µm of the nucleus"); axes[0].set_ylabel("cells")
axes[0].set_title("Fraction removed")
axes[1].hist(Rc["components"], bins=np.arange(0, Rc["components"].max() + 2) - 0.5,
             color="#eb6834")
axes[1].set_xlabel("connected components after the cut"); axes[1].set_ylabel("cells")
axes[1].set_yscale("log"); axes[1].set_title("Fragmentation")
plt.tight_layout(); plt.show()'''),
        md("## 2. Per cell type\n\nGlia fragment far more than neurons: their processes are short and radiate directly from the soma, so even a 5 µm ball severs several of them at once."),
        code('''g = Rc.groupby("cell_type").agg(
    cells=("root_id", "size"),
    median_before=("before", "median"),
    median_dropped_pct=("dropped_frac", lambda s: 100 * s.median()),
    median_components=("components", "median"),
    max_components=("components", "max"),
).sort_values("median_components", ascending=False)
g.style.format({"cells": "{:.0f}", "median_before": "{:,.0f}",
                "median_dropped_pct": "{:.1f}", "median_components": "{:.0f}",
                "max_components": "{:.0f}"})'''),
        code('''fig, ax = plt.subplots(figsize=(9, 3.6))
sub = g[g["cells"] >= 10]
xs = np.arange(len(sub))
ax.bar(xs, sub["median_components"], color="#1baf7a")
ax.set_xticks(xs); ax.set_xticklabels(sub.index, rotation=45, ha="right")
ax.set_ylabel("median components after the cut")
ax.set_title("Fragmentation by cell type (>= 10 cells)")
plt.tight_layout(); plt.show()'''),
        md("## 3. Does a bigger cell fragment more?\n\nComponent count reflects how many processes cross the 5 µm shell, which is a morphology fact rather than a size fact — so the relationship is weak within a type."),
        code('''fig, ax = plt.subplots(figsize=(5.6, 4))
glia = np.isin(Rc["cell_type"], ["astrocyte", "microglia", "oligo", "OPC"])
ax.scatter(Rc["before"][~glia], Rc["components"][~glia], s=5, alpha=0.35,
           color="#2a78d6", label="neuron")
ax.scatter(Rc["before"][glia], Rc["components"][glia], s=6, alpha=0.6,
           color="#eb6834", label="glia")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("nodes before the cut"); ax.set_ylabel("components after")
ax.legend(frameon=False, fontsize=8)
r = np.corrcoef(np.log10(Rc["before"]), np.log10(np.maximum(Rc["components"], 1)))[0, 1]
ax.set_title(f"log-log Pearson r = {r:.2f}")
plt.tight_layout(); plt.show()'''),
        md("""## 4. What the cut is actually doing

Two measurements together settle this, and neither was obvious in advance.

**Before the cut, every cell is one connected component.** The `graph_cache`
graphs are covered-node subgraphs of a CAVE skeleton, and subsetting to nodes with
embeddings does *not* fragment them — 400/400 cells spanning the whole size range
come back with exactly 1 component. So all the fragmentation below is caused by the
cut, none of it is pre-existing coverage gaps.

**At 5 µm the median cell loses one node and gains eight components.** That is only
possible if the removed node has degree >= 8: the CAVE skeleton represents the soma
as a single high-degree hub that every primary neurite attaches to. Deleting it
disconnects them all at once, which is precisely the intent — and it costs 0.024%
of the nodes to achieve.

A larger radius buys very little extra separation for a lot more data: at 15 µm the
cut removes 34x as many nodes (0.812%) to go from 8 components to 10."""),
        code('''# Degree of the nodes the cut removes, measured on one representative cell.
import torch
from data.geodesic_window import build_csr_from_edges

rid = int(R.sort_values("before", ascending=False).iloc[0]["root_id"])
g = torch.load(REPO_ROOT / "data" / "graph_cache" / f"{rid}.pt", weights_only=False)
with np.load(DB_ROOT / "soma_restricted" / f"{rid}.npz") as z:
    keep, dist = z["keep"], z["dist_to_nucleus_nm"]
n = g.pos.shape[0]
off, nbr, _w = build_csr_from_edges(g.edge_index.numpy(),
                                    g.edge_attr.numpy().reshape(-1), n)
degree = np.diff(off)
removed = np.flatnonzero(~keep)
print(f"cell {rid}: {n:,} nodes, {len(removed)} removed by the {RADIUS_UM:g} µm cut")
print(f"  degree of removed nodes: {sorted(degree[removed].tolist(), reverse=True)}")
print(f"  degree distribution of the whole cell: "
      f"max {degree.max()}, mean {degree.mean():.2f}")
print(f"  components after the cut: "
      f"{int(np.load(DB_ROOT / 'soma_restricted' / f'{rid}.npz')['n_components'][0])}")'''),
        md("""### Radius comparison

If the 15 µm build is still on disk, this compares the two directly. The output
directory is named after the radius (`r5um/`, `r15um/`), so both can coexist and
nothing can silently mix them."""),
        code('''rows = []
for tag in ("r5um", "r15um"):
    d = DB_ROOT.parent / tag / "soma_restricted"
    if not d.exists():
        continue
    b = a = 0; drops = []; comps = []
    for f in sorted(d.glob("*.npz")):
        with np.load(f) as z:
            if not bool(z["cut_applied"][0]):
                continue
            bb, aa = int(z["n_nodes_before"][0]), int(z["n_nodes_after"][0])
        b += bb; a += aa; drops.append(bb - aa)
        comps.append(int(np.load(f)["n_components"][0]))
    drops = np.array(drops); comps = np.array(comps)
    rows.append({"radius": tag, "cells": len(drops),
                 "% nodes dropped": 100 * (b - a) / b,
                 "dropped/cell (median)": float(np.median(drops)),
                 "components (median)": float(np.median(comps)),
                 "components (mean)": float(comps.mean())})
pd.DataFrame(rows).set_index("radius").style.format({
    "cells": "{:.0f}", "% nodes dropped": "{:.3f}",
    "dropped/cell (median)": "{:.0f}", "components (median)": "{:.0f}",
    "components (mean)": "{:.2f}"})'''),
        md("""## 5. Cells built uncut

19 of the 2,335 labelled cells have **no nucleus position** in the store's `cells`
dimension — exactly the 16 oligodendrocytes and 3 OPCs. Every neuron, astrocyte and
microglia has one.

There is no ball to cut for those, so they enter the database **whole**: no node is
removed. Guessing a centre would silently delete some other part of the cell, and
dropping them would shrink the dataset invisibly. `cut_applied` records which
treatment each cell got, so the two populations are never pooled by accident — and
their component counts are *not* comparable to the cut cells' below, since an uncut
skeleton is connected to begin with."""),
        code('''import json
nuc = json.loads((REPO_ROOT / "data" / "nucleus_positions.json").read_text())
cut = S["restrict_cut_applied"]
print(f"cells in database: {len(cut):,}   cut: {cut.sum():,}   uncut: {(~cut).sum():,}")
print("uncut by cell type:", nuc["missing_by_cell_type"])

U = R[~cut]
print()
print(f"uncut cells keep every node: dropped max = {U['dropped'].max()}")
display(U[["root_id", "cell_type", "before", "after", "components"]]
        .sort_values("before", ascending=False).head(10))'''),
    ]
    nb = nbf.v4.new_notebook(cells=c)
    nb.metadata.kernelspec = KERNEL
    return nb


def neighborhood_nb():
    c = [
        md("""# Total skeleton length in a local neighbourhood

The unit here is a **connected local subgraph**, not a route. A path follows one
way through every fork and throws the other branches away; a neighbourhood keeps
them, so it holds all the skeleton near a node rather than one line through it.

What is held constant is **total cable length** — the summed edge length of the
subgraph. Every `cable40um` unit contains 40 µm of skeleton, whatever shape it is
in: an unbranched neurite spends that reaching ~20 µm from the centre, a branchy
arbor spends it on several short branches and reaches much less far. Size constant,
extent variable.

That is the opposite trade from the project's existing geodesic window, which fixes
the **radius** and lets the amount of skeleton vary. Both are in the database; this
notebook measures the cable-budget one.

Growth is nearest-first (bounded Dijkstra from the centre), stopping at the first
node whose edge would overshoot — so cable is at or just under budget, and the unit
is a geodesic ball grown until its cable runs out, not a cherry-picked set of short
edges.

**Config names are not shared with `paths/`.** Under `neighborhoods/`, `cable20um`
is 20 µm of *total cable* and `n20` is 20 nodes outright; under `paths/`, `20um` is
a 20 µm *diameter* route and `20node` is 21 nodes. Different units, different
names, so a directory listing can never be misread."""),
        code(PREAMBLE),
        md("## 1. Headline table"),
        code('''rows = []
for c in NCONFIGS:
    q = S[f"nb_{c}__pct_q"]; cb = S[f"nb_{c}__pct_cable"]
    pc = dict(zip(q.tolist(), cb.tolist()))
    nodes = S[f"nb_{c}__sample_nodes"]; rad = S[f"nb_{c}__sample_radius"] / 1000
    rows.append({
        "config": c,
        "units": int(S[f"nb_{c}__n"][0]),
        "nodes p50": float(np.median(nodes)),
        "nodes p90": float(np.percentile(nodes, 90)),
        "cable p50 (µm)": pc[50] / 1000,
        "cable p90 (µm)": pc[90] / 1000,
        "radius p50 (µm)": float(np.median(rad)),
        "radius p90 (µm)": float(np.percentile(rad, 90)),
    })
tbl = pd.DataFrame(rows).set_index("config")
tbl.style.format({"units": "{:,.0f}", "nodes p50": "{:.0f}", "nodes p90": "{:.0f}",
                  "cable p50 (µm)": "{:.1f}", "cable p90 (µm)": "{:.1f}",
                  "radius p50 (µm)": "{:.1f}", "radius p90 (µm)": "{:.1f}"})'''),
        md("""## 2. Cable held constant, extent free

A cable budget fills almost exactly (the stop-before-overshoot rule leaves it one
edge short), while the **radius** it reaches varies over a wide range — that spread
is the branching structure, and it is what a fixed-radius window would hide."""),
        code('''fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
for c in CABLE:
    axes[0].hist(S[f"nb_{c}__sample_cable"] / 1000, bins=np.linspace(0, 90, 181),
                 histtype="step", lw=1.6, density=True, color=COLOR[c], label=c)
    axes[1].hist(S[f"nb_{c}__sample_radius"] / 1000, bins=np.linspace(0, 60, 181),
                 histtype="step", lw=1.6, density=True, color=COLOR[c], label=c)
axes[0].set_xlabel("total cable in the unit (µm)"); axes[0].set_title("what is held fixed")
axes[1].set_xlabel("geodesic radius reached (µm)"); axes[1].set_title("what varies")
axes[0].set_ylabel("density")
for ax in axes:
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print("cable is pinned; radius is not")
for c in CABLE:
    r = S[f"nb_{c}__sample_radius"] / 1000
    cb = S[f"nb_{c}__sample_cable"] / 1000
    print(f"  {c:>10}  cable {np.median(cb):6.1f} µm (p10-p90 "
          f"{np.percentile(cb,10):5.1f}-{np.percentile(cb,90):5.1f})   "
          f"radius {np.median(r):6.1f} µm (p10-p90 "
          f"{np.percentile(r,10):5.1f}-{np.percentile(r,90):5.1f})")'''),
        md("""## 3. Matching a node budget to a cable budget

A neighbourhood of k nodes is a tree with k-1 edges, so it holds about 2(k-1) µm of
cable at ~2 µm spacing. The matched pairs are therefore `n10` <-> `cable20um`,
`n20` <-> `cable40um`, `n40` <-> `cable80um`.

Pairing by the same number instead (`n10` with `cable10um`) compares units that
differ about two-fold in how much skeleton they hold, which would confound any
model comparison built on it. Both pairings are printed so the gap is visible."""),
        code('''fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
x = np.arange(len(NPAIRS))
for ax, (col, lab) in zip(axes, [("cable p50 (µm)", "total cable"),
                                 ("radius p50 (µm)", "geodesic radius")]):
    nd = [tbl.loc[n, col] for n, _ in NPAIRS]
    cb = [tbl.loc[u, col] for _, u in NPAIRS]
    ax.bar(x - 0.19, nd, 0.36, label="node budget", color="#eb6834")
    ax.bar(x + 0.19, cb, 0.36, label="matched cable budget", color="#2a78d6")
    for xi, (a, b) in enumerate(zip(nd, cb)):
        ax.text(xi - 0.19, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.19, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{n} vs {u}" for n, u in NPAIRS])
    ax.set_ylabel(f"median {lab} (µm)"); ax.set_title(lab)
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

for col, lab in [("cable p50 (µm)", "cable"), ("radius p50 (µm)", "radius")]:
    print(f"matched pairs, median {lab}")
    for n, u in NPAIRS:
        a, b = tbl.loc[n, col], tbl.loc[u, col]
        print(f"  {n:>4} {a:6.1f} µm  vs  {u:>10} {b:6.1f} µm   ratio {a/b:.2f}x")
    print(f"the naive same-number pairing, for contrast ({lab})")
    for n, u in zip(NNODE, CABLE[:3]):
        a, b = tbl.loc[n, col], tbl.loc[u, col]
        print(f"  {n:>4} {a:6.1f} µm  vs  {u:>10} {b:6.1f} µm   ratio {a/b:.2f}x")
    print()'''),
        md("""## 4. ECDFs

No bin width and no density normalisation, so every config is directly comparable
on one axis. Two panels because the unit fixes one quantity and frees the other:

- **cable** — the four `cable*` configs rise almost vertically at their budgets;
  the three `n*` configs cross them and keep going, because a node budget does not
  cap cable.
- **radius** — nothing is vertical. Even a pinned cable budget spreads over a wide
  range of radii, and that spread *is* the local branching."""),
        code('''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, key, lab, xmax in ((axes[0], "sample_cable", "total cable (µm)", 110),
                           (axes[1], "sample_radius", "geodesic radius (µm)", 80)):
    for c in NCONFIGS:
        v = np.sort(S[f"nb_{c}__{key}"] / 1000)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), lw=1.6, color=COLOR[c],
                ls="--" if c.startswith("cable") else "-", label=c)
    ax.set_xlim(0, xmax); ax.set_xlabel(lab); ax.set_ylabel("fraction of units at most this")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
axes[0].set_title("cable: what the budget pins")
axes[1].set_title("radius: what it leaves free")
fig.suptitle("dashed = cable budget, solid = node budget", y=1.02, fontsize=10)
plt.tight_layout(); plt.show()

print(f"{'config':>10} {'cable p50':>10} {'cable p90':>10} "
      f"{'radius p50':>11} {'radius p90':>11} {'radius p10-p90 spread':>22}  (µm)")
for c in NCONFIGS:
    cb = S[f"nb_{c}__sample_cable"] / 1000
    rd = S[f"nb_{c}__sample_radius"] / 1000
    lo, hi = np.percentile(rd, 10), np.percentile(rd, 90)
    print(f"{c:>10} {np.median(cb):>10.1f} {np.percentile(cb,90):>10.1f} "
          f"{np.median(rd):>11.1f} {np.percentile(rd,90):>11.1f} "
          f"{f'{lo:.1f}-{hi:.1f} ({hi/max(lo,1e-9):.1f}x)':>22}")'''),
        md("""## 5. How far does a fixed amount of cable actually reach?

The ratio radius / (cable/2) is 1.0 for a perfectly straight unbranched neurite
(cable spent equally in two directions) and falls toward 0 as the unit branches or
doubles back. It is a branching index that needs no extra data."""),
        code('''fig, ax = plt.subplots(figsize=(6.8, 3.6))
for c in CABLE:
    cb = S[f"nb_{c}__sample_cable"] / 1000
    rd = S[f"nb_{c}__sample_radius"] / 1000
    ok = cb > 0
    ax.hist(rd[ok] / (cb[ok] / 2), bins=np.linspace(0, 1.05, 160), histtype="step",
            lw=1.6, density=True, color=COLOR[c], label=c)
ax.set_xlabel("radius / (cable/2)   —   1.0 = straight, 0 = maximally branched")
ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=8)
ax.set_title("How much of the cable goes into reach rather than branching")
plt.tight_layout(); plt.show()

for c in CABLE:
    cb = S[f"nb_{c}__sample_cable"] / 1000
    rd = S[f"nb_{c}__sample_radius"] / 1000
    ok = cb > 0
    q = rd[ok] / (cb[ok] / 2)
    print(f"  {c:>10}  median {np.median(q):.3f}   "
          f"p10 {np.percentile(q,10):.3f}  p90 {np.percentile(q,90):.3f}")'''),
        md("## 6. Per cell type\n\nCable is fixed by construction, so what varies by type is the **radius** that cable reaches — small where the arbor branches densely."),
        code('''ct = {int(r): str(t) for r, t in zip(S["restrict_root_id"], S["restrict_cell_type"])}
rad = {}
for c in NCONFIGS:
    rid = S[f"nb_{c}__cell_root_id"]
    rad[c] = pd.Series(S[f"nb_{c}__cell_median_radius"] / 1000,
                       index=[ct.get(int(r), "?") for r in rid]).groupby(level=0).median()
by_type = pd.DataFrame(rad)
by_type["n cells"] = pd.Series(
    [ct.get(int(r), "?") for r in S["nb_cable40um__cell_root_id"]]).value_counts()
by_type = by_type.sort_values("cable80um")
display(by_type.style.format({**{c: "{:.1f}" for c in NCONFIGS}, "n cells": "{:.0f}"}))

fig, ax = plt.subplots(figsize=(9, 3.5))
sub = by_type[by_type["n cells"] >= 10]
xs = np.arange(len(sub))
for c in CABLE:
    ax.plot(xs, sub[c], "o-", ms=4, lw=1.3, color=COLOR[c], label=c)
ax.set_xticks(xs); ax.set_xticklabels(sub.index, rotation=45, ha="right")
ax.set_ylabel("median radius reached (µm)")
ax.set_title("How far a fixed amount of cable reaches, by cell type")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()'''),
        md("""## 7. Reading one neighbourhood, and joining embeddings

`members` indexes the restricted node array, centre first. `cache_index` maps to
the row in `graph_cache/<root_id>.pt`; `orig_node_ids` is the CAVE skeleton vertex
id."""),
        code('''import torch

rid = int(S["nb_cable40um__cell_root_id"][0])
N = np.load(DB_ROOT / "neighborhoods" / "cable40um" / f"{rid}.npz")
d = torch.load(REPO_ROOT / "data" / "graph_cache" / f"{rid}.pt", weights_only=False)

off, mem = N["offsets"], N["members"]
k = int(np.argmax(N["n_members"]))
m = mem[off[k]:off[k + 1]]
emb = d.x.numpy()[N["cache_index"][m]]
print(f"cell {rid}: {len(off)-1:,} neighbourhoods")
print(f"unit {k}: centre = restricted node {m[0]}, {len(m)} members")
print(f"  total cable {N['cable_nm'][k]:,.0f} nm, radius {N['radius_nm'][k]:,.0f} nm")
print(f"  CAVE vertex ids: {N['orig_node_ids'][m][:6]} ...")
print(f"  embedding set: {emb.shape}")'''),
        md("""## What to take from this

- **Cable is pinned, radius is not.** That is the whole point of the unit: every
  unit holds the same amount of skeleton, and how far it reaches is a measured
  property of the local branching rather than an input. The radius ECDF in section
  4 is the one to look at — it is the only quantity here the budget does not fix.
- **radius / (cable/2) is a free branching index.** 1.0 means the cable went
  entirely into reach (a straight unbranched neurite); lower means it went into
  branches. It needs no data beyond what is already stored.
- **A node budget is close to a cable budget** at these sizes, because spacing is
  near-uniform — `n10`/`n20`/`n40` land within a few percent of
  `cable20um`/`cable40um`/`cable80um`. Where they diverge is where spacing is
  irregular.
- **Do not read a `neighborhoods/` config name as a `paths/` one.** `cable20um` is
  20 µm of cable; `paths/20um` is a 20 µm diameter route."""),
    ]
    nb = nbf.v4.new_notebook(cells=c)
    nb.metadata.kernelspec = KERNEL
    return nb


def check(name, nb):
    """Parse every code cell before writing.

    Cell sources are Python inside Python here, so an escape can collapse one
    level too many and put a real newline inside an f-string. That surfaces as a
    SyntaxError halfway through a several-minute execution job; parsing first
    turns it into an immediate, located failure.
    """
    import ast
    bad = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        try:
            ast.parse(c.source)
        except SyntaxError as exc:
            bad.append(f"{name} cell {i}: {exc}")
    if bad:
        raise SystemExit("generated cells do not parse:\n  " + "\n  ".join(bad))


def main():
    AN.mkdir(exist_ok=True)
    for name, nb in (("neighborhood_cable", neighborhood_nb()),
                     ("embedding_path_geodesics", geodesics_nb()),
                     ("soma_restriction", restriction_nb())):
        check(name, nb)
        p = AN / f"{name}.ipynb"
        nbf.write(nb, p)
        print(f"wrote {p} ({len(nb.cells)} cells, code cells parse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
