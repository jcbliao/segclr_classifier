"""Generate analysis/synapse_inventory.ipynb. Run, then execute it in place.

The notebook does every aggregation in duckdb against the two parquet
databases, so nothing but per-cell summaries (2,335 rows) is ever materialised
in the kernel and there is no precomputed npz to keep in sync.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "analysis"
KERNEL = {"display_name": "segclr_db (.venv)", "language": "python", "name": "segclr_db"}


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


PREAMBLE = '''import json
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "analysis" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from gnn.hierarchy import LAB_HIERARCHY_TREE

CACHE = REPO_ROOT / "data" / "synapse_cache"
PRE = str(CACHE / "presynaptic_sites.parquet")    # our cell is PREsynaptic
POST = str(CACHE / "postsynaptic_sites.parquet")  # our cell is POSTsynaptic
PARTNER_VOLUMES = REPO_ROOT / "data" / "partner_volume_cache" / "partner_volumes.parquet"
SUMMARY = json.loads((CACHE / "summary.json").read_text())

manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text())
cells = pd.DataFrame(
    [{"root_id": int(r), "cell_type": v["cell_type"], "split": v["split"],
      "n_nodes": v["n_nodes_covered"]} for r, v in manifest["cells"].items()]
)


def _leaves(tree, trail=()):
    """Every granular label in LAB_HIERARCHY_TREE, in tree order, with its trail.

    Tree order rather than alphabetical or frequency order: it keeps the
    excitatory subtypes, the interneuron families and the glia in contiguous
    blocks, which is what makes a per-type figure readable at a glance.
    """
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from _leaves(value, trail + (key,))
        else:
            for leaf in value:
                yield leaf, trail + (key,)


TRAIL = dict(_leaves(LAB_HIERARCHY_TREE))
ORDER = [label for label in TRAIL if label in set(cells["cell_type"])]
GROUP = {label: ("glia" if trail[0] == "non_neuron" else trail[1]) for label, trail in TRAIL.items()}
cells["group"] = cells["cell_type"].map(GROUP)
GROUP_COLOR = {"excitatory": "#2a78d6", "inhibitory": "#e34948", "glia": "#7a7a7a"}
COLOR = {label: GROUP_COLOR[GROUP[label]] for label in TRAIL}

con = duckdb.connect()
con.register("cells", cells)

plt.rcParams.update({
    "figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})

print(f"{len(cells)} cells, {cells['cell_type'].nunique()} types")
print(f"synapses from {SUMMARY['synapse_table']} @ {SUMMARY['datastack']} "
      f"mat_version {SUMMARY['mat_version']}")
for mode, info in SUMMARY["modes"].items():
    print(f"  {mode:9s} {info['rows']:>10,} rows  {info['file']}")'''


PER_CELL = '''per_cell = con.sql(f"""
    with pre as (
        select cell_root_id                                           as root_id,
               count(*)                                               as n_boutons,
               count(distinct partner_root_id)
                   filter (where partner_root_id != 0)                as n_targets
        from read_parquet('{PRE}')
        group by 1
    ),
    post as (
        select p.cell_root_id                                         as root_id,
               count(*)                                               as n_postsyn_sites,
               count(*) filter (where p.partner_root_id != 0)         as n_in_resolved,
               count(*) filter (where c.root_id is not null)          as n_in_from_db,
               count(distinct p.partner_root_id)
                   filter (where p.partner_root_id != 0)              as n_input_cells,
               count(distinct c.root_id)                              as n_input_cells_in_db
        from read_parquet('{POST}') p
        left join cells c on c.root_id = p.partner_root_id
        group by 1
    )
    select c.root_id, c.cell_type, c.group, c.split, c.n_nodes,
           coalesce(pre.n_boutons, 0)          as n_boutons,
           coalesce(pre.n_targets, 0)          as n_targets,
           coalesce(post.n_postsyn_sites, 0)   as n_postsyn_sites,
           coalesce(post.n_in_resolved, 0)     as n_in_resolved,
           coalesce(post.n_in_from_db, 0)      as n_in_from_db,
           coalesce(post.n_input_cells, 0)     as n_input_cells,
           coalesce(post.n_input_cells_in_db, 0) as n_input_cells_in_db
    from cells c
    left join pre  on pre.root_id  = c.root_id
    left join post on post.root_id = c.root_id
""").df()

per_cell["n_in_unresolved"] = per_cell["n_postsyn_sites"] - per_cell["n_in_resolved"]
per_cell["frac_in_resolved"] = per_cell["n_in_resolved"] / per_cell["n_postsyn_sites"].clip(lower=1)
per_cell["frac_in_from_db"] = per_cell["n_in_from_db"] / per_cell["n_postsyn_sites"].clip(lower=1)
per_cell.to_csv(REPO_ROOT / "analysis" / "synapse_inventory_per_cell.csv", index=False)

print(f"{len(per_cell)} cells")
print(f"{(per_cell['n_boutons'] == 0).sum()} with no presynaptic sites, "
      f"{(per_cell['n_postsyn_sites'] == 0).sum()} with no postsynaptic sites")
per_cell.head(10)'''


BY_TYPE = '''def q(series, p):
    return series.quantile(p)

by_type = per_cell.groupby("cell_type").agg(
    n_cells=("root_id", "size"),
    boutons_total=("n_boutons", "sum"),
    boutons_med=("n_boutons", "median"),
    boutons_p10=("n_boutons", lambda s: q(s, 0.10)),
    boutons_p90=("n_boutons", lambda s: q(s, 0.90)),
    postsyn_total=("n_postsyn_sites", "sum"),
    postsyn_med=("n_postsyn_sites", "median"),
    postsyn_p10=("n_postsyn_sites", lambda s: q(s, 0.10)),
    postsyn_p90=("n_postsyn_sites", lambda s: q(s, 0.90)),
    nodes_med=("n_nodes", "median"),
).reindex(ORDER)

# Ratio of the medians, not the median of the ratios: cells with zero
# presynaptic sites (unproofread axon) would otherwise dominate a per-cell
# ratio with infinities.
by_type["post_per_bouton"] = by_type["postsyn_med"] / by_type["boutons_med"].replace(0, np.nan)
by_type.insert(0, "group", [GROUP[t] for t in by_type.index])

pd.set_option("display.width", 200, "display.max_columns", 40)
print(by_type.to_string(float_format=lambda v: f"{v:,.1f}"))'''


DIST_FIG = '''fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, column, title in [
    (axes[0], "n_boutons", "presynaptic sites per cell (boutons: this cell is presynaptic)"),
    (axes[1], "n_postsyn_sites", "postsynaptic sites per cell (spine + shaft pooled; see caveat)"),
]:
    data = [per_cell.loc[per_cell["cell_type"] == t, column].to_numpy() for t in ORDER]
    parts = ax.boxplot(data, positions=range(len(ORDER)), widths=0.62, showfliers=False,
                       patch_artist=True, medianprops={"color": "black"})
    for patch, t in zip(parts["boxes"], ORDER):
        patch.set_facecolor(COLOR[t])
        patch.set_alpha(0.55)
    for i, t in enumerate(ORDER):
        values = per_cell.loc[per_cell["cell_type"] == t, column].to_numpy()
        jitter = np.random.default_rng(0).normal(0, 0.07, size=len(values))
        ax.plot(i + jitter, np.clip(values, 0.7, None), ".", ms=2, color=COLOR[t], alpha=0.35)
    ax.set_yscale("log")
    ax.set_ylabel("synapses per cell")
    ax.set_title(title, loc="left", fontsize=10)

axes[1].set_xticks(range(len(ORDER)))
axes[1].set_xticklabels(ORDER, rotation=60, ha="right")
handles = [plt.Line2D([], [], color=c, lw=6, alpha=0.6, label=g) for g, c in GROUP_COLOR.items()]
axes[0].legend(handles=handles, frameon=False, ncol=3, loc="upper right")
fig.tight_layout()
plt.show()'''


SCATTER_FIG = '''fig, ax = plt.subplots(figsize=(6.2, 5.6))
for group, color in GROUP_COLOR.items():
    sub = per_cell[per_cell["group"] == group]
    ax.plot(sub["n_boutons"].clip(lower=0.7), sub["n_postsyn_sites"].clip(lower=0.7),
            ".", ms=4, alpha=0.5, color=color, label=f"{group} (n={len(sub)})")
lim = [0.7, max(per_cell["n_boutons"].max(), per_cell["n_postsyn_sites"].max()) * 1.4]
ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="equal")
ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
       xlabel="presynaptic sites (boutons)", ylabel="postsynaptic sites")
ax.set_title("output vs. input count per cell", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
plt.show()

zero_out = per_cell[per_cell["n_boutons"] == 0]
print(f"{len(zero_out)} cells have zero presynaptic sites; by type:")
print(zero_out["cell_type"].value_counts().to_string())'''


RESOLUTION = '''resolution = per_cell.groupby("cell_type").agg(
    n_cells=("root_id", "size"),
    postsyn_total=("n_postsyn_sites", "sum"),
    resolved=("n_in_resolved", "sum"),
    unresolved=("n_in_unresolved", "sum"),
    from_db=("n_in_from_db", "sum"),
).reindex(ORDER)

resolution["pct_resolved"] = 100 * resolution["resolved"] / resolution["postsyn_total"]
resolution["pct_from_db"] = 100 * resolution["from_db"] / resolution["postsyn_total"]
# Of the partners that resolved at all -- the denominator that separates "the
# segmentation could not name this partner" from "it named a cell we do not have".
resolution["pct_of_resolved_from_db"] = 100 * resolution["from_db"] / resolution["resolved"]

total = resolution[["postsyn_total", "resolved", "unresolved", "from_db"]].sum()
print(f"across all {len(per_cell)} cells, {total['postsyn_total']:,} postsynaptic sites")
print(f"  presynaptic partner resolved to a root_id : {total['resolved']:,} "
      f"({100 * total['resolved'] / total['postsyn_total']:.2f}%)")
print(f"  unresolved (partner_root_id = 0)          : {total['unresolved']:,} "
      f"({100 * total['unresolved'] / total['postsyn_total']:.2f}%)")
print(f"  partner is another cell in this database  : {total['from_db']:,} "
      f"({100 * total['from_db'] / total['postsyn_total']:.2f}% of all, "
      f"{100 * total['from_db'] / total['resolved']:.2f}% of resolved)\\n")
print(resolution.to_string(float_format=lambda v: f"{v:,.2f}"))'''


RESOLUTION_FIG = '''fig, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1])

x = np.arange(len(ORDER))
frac_db = 100 * resolution["from_db"] / resolution["postsyn_total"]
frac_other = 100 * (resolution["resolved"] - resolution["from_db"]) / resolution["postsyn_total"]
frac_none = 100 * resolution["unresolved"] / resolution["postsyn_total"]

axes[0].bar(x, frac_db, color="#2a78d6", label="partner is a cell in this database")
axes[0].bar(x, frac_other, bottom=frac_db, color="#b9cfe8",
            label="partner resolved, but outside this database")
axes[0].bar(x, frac_none, bottom=frac_db + frac_other, color="#d9534f",
            label="partner_root_id = 0 (unresolved)")
axes[0].set(ylabel="% of postsynaptic sites", ylim=(0, 100), xticks=x)
axes[0].set_xticklabels([])
axes[0].set_title("where each cell type's presynaptic partners come from", loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=8, loc="lower right")

# The in-database share on its own axis: it is a fraction of a percent, and
# stacked against the other two it would be an invisible sliver.
axes[1].bar(x, frac_db, color="#2a78d6")
axes[1].set(ylabel="% from this database", xticks=x)
axes[1].set_xticklabels(ORDER, rotation=60, ha="right")
fig.tight_layout()
plt.show()'''


IN_DB_MATRIX = '''pairs = con.sql(f"""
    select src.cell_type as pre_type, dst.cell_type as post_type, count(*) as n
    from read_parquet('{POST}') p
    join cells dst on dst.root_id = p.cell_root_id
    join cells src on src.root_id = p.partner_root_id
    group by 1, 2
""").df()

matrix = (pairs.pivot(index="pre_type", columns="post_type", values="n")
          .reindex(index=ORDER, columns=ORDER).fillna(0))

print(f"{int(matrix.values.sum()):,} synapses have both partners in this database "
      f"({matrix.astype(bool).values.sum()} of {len(ORDER)**2} type pairs occupied)")

fig, ax = plt.subplots(figsize=(9.5, 8))
shown = np.log10(matrix.to_numpy() + 1)
im = ax.imshow(shown, cmap="magma", aspect="equal")
ax.set(xticks=range(len(ORDER)), yticks=range(len(ORDER)),
       xlabel="postsynaptic cell type (our cell)", ylabel="presynaptic cell type (partner)")
ax.set_xticklabels(ORDER, rotation=60, ha="right", fontsize=7)
ax.set_yticklabels(ORDER, fontsize=7)
ax.grid(False)
ax.set_title("synapses with both partners in the database (log10 count + 1)", loc="left", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.8, label="log10(count + 1)")
fig.tight_layout()
plt.show()

matrix.astype(int).to_csv(REPO_ROOT / "analysis" / "synapse_in_db_type_matrix.csv")'''


OUTGOING_RESOLUTION = '''outgoing = con.sql(f"""
    select c.cell_type,
           count(*)                                        as presyn_total,
           count(*) filter (where p.partner_root_id != 0)  as resolved,
           count(*) filter (where d.root_id is not null)   as from_db
    from read_parquet('{PRE}') p
    join cells c on c.root_id = p.cell_root_id
    left join cells d on d.root_id = p.partner_root_id
    group by 1
""").df().set_index("cell_type").reindex(ORDER)

outgoing["pct_resolved"] = 100 * outgoing["resolved"] / outgoing["presyn_total"]
outgoing["pct_from_db"] = 100 * outgoing["from_db"] / outgoing["presyn_total"]
print(outgoing.to_string(float_format=lambda v: f"{v:,.2f}"))'''

PARTNER_ROOTS = '''# This cache is populated by data/cave_skeletons.py.  Filename
# matching is a read-only availability check; it does not submit generation
# requests to CAVE.
skeleton_cache = REPO_ROOT / "data" / "skeleton_cache"
cached_skeleton_ids = {
    int(path.stem) for path in skeleton_cache.glob("*.pkl") if path.stem.isdigit()
}
cached_skeletons = pd.DataFrame({"root_id": list(cached_skeleton_ids)}, dtype="uint64")
con.register("cached_skeletons", cached_skeletons)

# Keep the 3.5-million-row distinct set inside DuckDB: only this one-row
# summary enters the notebook kernel.
partner_summary = con.sql(f"""
    with partners as (
        select partner_root_id as root_id, count(*) as n_postsyn_sites
        from read_parquet('{POST}')
        where partner_root_id != 0
        group by 1
    )
    select count(*) as unique_root_ids,
           count(c.root_id) as cached_skeletons,
           avg(n_postsyn_sites) as mean_sites,
           median(n_postsyn_sites) as median_sites,
           quantile_cont(n_postsyn_sites, 0.90) as p90_sites,
           quantile_cont(n_postsyn_sites, 0.99) as p99_sites,
           max(n_postsyn_sites) as max_sites
    from partners p
    left join cached_skeletons c using (root_id)
""").df().iloc[0]

print(f"{int(partner_summary.unique_root_ids):,} unique resolved presynaptic partner root IDs")
print(f"{int(partner_summary.cached_skeletons):,} currently have a skeleton in "
      f"data/skeleton_cache ({100 * partner_summary.cached_skeletons / partner_summary.unique_root_ids:.3f}%)")
print("postsynaptic sites contributed per partner: "
      f"median {partner_summary.median_sites:,.0f}, mean {partner_summary.mean_sites:,.2f}, "
      f"p90 {partner_summary.p90_sites:,.0f}, p99 {partner_summary.p99_sites:,.0f}, "
      f"max {partner_summary.max_sites:,.0f}")'''

PARTNER_VOLUMES = '''if PARTNER_VOLUMES.exists():
    volume_summary = con.sql(f"""
        select count(*) as roots_fetched,
               count(*) filter (where error is null) as roots_ok,
               count(*) filter (where error is not null) as roots_failed,
               count(*) filter (where n_l2_sizes_missing > 0) as roots_incomplete,
               median(n_l2_chunks) filter (where error is null) as median_l2_chunks,
               quantile_cont(n_l2_chunks, 0.90) filter (where error is null) as p90_l2_chunks,
               median(volume_nm3 / 1e9) filter (where error is null) as median_volume_um3,
               quantile_cont(volume_nm3 / 1e9, 0.90) filter (where error is null) as p90_volume_um3
        from read_parquet('{PARTNER_VOLUMES}')
    """).df().iloc[0]
    print(f"volume metadata fetched for {int(volume_summary.roots_fetched):,} / "
          f"{int(partner_summary.unique_root_ids):,} partner roots")
    print(f"  successful: {int(volume_summary.roots_ok):,}; errors: "
          f"{int(volume_summary.roots_failed):,}; incomplete L2 sizes: "
          f"{int(volume_summary.roots_incomplete):,}")
    print(f"  L2 chunks: median {volume_summary.median_l2_chunks:,.0f}, "
          f"p90 {volume_summary.p90_l2_chunks:,.0f}")
    print(f"  segmented volume: median {volume_summary.median_volume_um3:,.2f} µm³, "
          f"p90 {volume_summary.p90_volume_um3:,.2f} µm³")
else:
    print("partner volume metadata has not been merged yet; run "
          "data/build_partner_volumes.py (see below)")'''


def build():
    cells = [
        md("""# Synaptic inventory of the labelled cells, by cell type

For every cell in `data/manifest.json`, from the two databases
`data/SYNAPSES.md` describes:

- **presynaptic sites** — where this cell synapses *onto* something else (its boutons)
- **postsynaptic sites** — where something synapses *onto* it
- for each postsynaptic site, whether the **presynaptic partner** resolved to a
  `root_id` at all, and whether that partner is **another cell in this database**

Polarity is read off CAVE's own `pre_pt_root_id` / `post_pt_root_id`; nothing here
infers direction from geometry.

### One thing this cannot tell you

**Spine vs. shaft is not in the synapse table.** `synapses_pni_2` records the two
partners, the cleft centroid and the cleft size — not the postsynaptic
compartment. So "postsynaptic sites" below pools spine and shaft synapses, and no
number in this notebook separates them. Splitting them would need a per-synapse
compartment call this project has not built: the nearest-skeleton-node join is
unreliable exactly where it would matter, since two branches of the same cell
frequently pass within a spine's length of each other.

### Two more things that bound how these numbers should be read

**Axon proofreading is uneven.** A cell whose axon was never proofread keeps only
the stub the automated segmentation found, so its presynaptic count reports
reconstruction effort as much as biology. Dendrites are far more completely
reconstructed, which is why the postsynaptic counts are both larger and more
uniform. Compare cell types with that asymmetry in mind, and read the
zero-presynaptic-site cells below as a reconstruction statistic.

**This database is a labelled subset, not a circuit.** 2,335 cells out of the
volume's tens of thousands, so the fraction of partners that are also in the
database is a property of *the subset's size and selection*, not a connection
probability."""),
        code(PREAMBLE),
        md("""## 1. Per-cell inventory

One row per cell. `n_boutons` and `n_postsyn_sites` are row counts in the two
databases; the `n_in_*` columns break the postsynaptic ones down by what happened
to the presynaptic partner. Written to
`analysis/synapse_inventory_per_cell.csv`."""),
        code(PER_CELL),
        md("""## 2. Per cell type

Medians and the 10–90% range rather than means: both counts are heavy-tailed, and
a single exceptionally well-reconstructed cell moves a mean by more than the type
does."""),
        code(BY_TYPE),
        md("""## 3. Distributions

Box plus every cell as a point, log scale. The spread *within* a type is the
thing to look at first — where it exceeds the spread between types, reconstruction
completeness is a bigger effect than cell class."""),
        code(DIST_FIG),
        md("""## 4. Output against input, per cell

The dashed line is parity. Distance below it is the input/output asymmetry that
follows from dendrites being reconstructed more completely than axons."""),
        code(SCATTER_FIG),
        md("""## 5. Where the presynaptic partners come from

Three exclusive buckets for every postsynaptic site: the partner supervoxel
resolved to a cell in **this database**, resolved to a cell **outside** it, or did
not resolve at all (`partner_root_id = 0`, the supervoxel naming no cell at this
materialization).

`pct_of_resolved_from_db` is the one to quote when asking how self-contained the
labelled set is — it drops the unresolved rows from the denominator rather than
mixing a segmentation failure in with a coverage fact."""),
        code(RESOLUTION),
        code(RESOLUTION_FIG),
        md("""## 6. Unique presynaptic partner root IDs

The 8,421,814 resolved postsynaptic-site rows collapse to **3,542,235 unique,
nonzero presynaptic partner root IDs**. (`0` is an unresolved sentinel, not a
cell, so it is excluded.) The code below recomputes the count directly from the
parquet database and also reports how many of those IDs already have a locally
cached CAVE skeleton.

Neither cell volume nor skeleton node count is stored in the synapse parquet.
Volume must be obtained separately from the segmentation/CAVE metadata, and
skeleton node count exists only after a skeleton has been downloaded or
generated. The cache check below is deliberately read-only: it identifies
existing skeletons without making millions of CAVE generation requests. Any
future bulk request should first deduplicate these IDs, subtract cached and CAVE
refusal-list IDs, and be chunked/resumable via `data/cave_skeletons.py`."""),
        code(PARTNER_ROOTS),
        md("""### CAVE volume and L2-chunk metadata

`data/build_partner_volumes.py` obtains the full set of L2 leaves for each root
at materialization 1718. `n_l2_chunks` is the number of those leaves;
`volume_nm3` is the sum of their CAVE L2-cache `size_nm3` values. Because CAVE
serves the leaf list one root at a time, this is a large resumable batch job,
not a notebook-side query. A nonzero `n_l2_sizes_missing` means the reported
volume is only a lower bound and must not be treated as exact.

```bash
# First measure throughput without contaminating the full-run cache.
python -u data/build_partner_volumes.py --limit 100 --sample \\
  --roots-per-part 100 --output data/partner_volume_pilot

# Full resumable run on the preemptable CPU pool.
sbatch --array=0-19%16 --export=ALL,VOLUME_WORLD=40 scripts/sbatch/build_partner_volumes.sh
sbatch --array=20-39%16 --export=ALL,VOLUME_WORLD=40 scripts/sbatch/build_partner_volumes.sh
MERGE=1 sbatch scripts/sbatch/build_partner_volumes.sh
```

A deterministic random 100-root pilot completed with 100/100 successful roots
and no missing L2 sizes. It found a median of 17.5 L2 chunks per root (mean
155.3, p90 420.5, maximum 3,146) and median segmented volume 1.35 µm³ (mean
16.32, p90 35.17, maximum 471.04 µm³). A four-stream pilot fetched the same
100 roots in 23 seconds versus 76 seconds sequentially, with no errors or
missing sizes. The full concurrent sweep is split between SLURM jobs 21473408
and 21473181, both on `mit_preemptable`: 40 resumable shards with at
most 32 SLURM tasks concurrently and an 8-hour walltime. Each
task pipelines four independent root requests, for up to 128 roots in flight;
each thread owns its own CAVE client/session and retries transient failures.

The cell below automatically summarizes the merged file when it exists."""),
        code(PARTNER_VOLUMES),
        md("""## 7. The part of the wiring diagram this database holds

Synapses where **both** partners are labelled cells here — the sub-network the
database closes on. The counts are small and heavily shaped by which types the
label set happens to hold many of, so read the matrix as coverage, not as
connectivity structure."""),
        code(IN_DB_MATRIX),
        md("""## 8. The same resolution question on the output side

For completeness: for each cell type's *presynaptic* sites, how often the
postsynaptic target resolved, and how often it is another cell in the database."""),
        code(OUTGOING_RESOLUTION),
        md("""## Regenerating

```bash
sbatch --array=0-3%4 scripts/sbatch/build_synapses.sh   # the two databases
MERGE=1 sbatch scripts/sbatch/build_synapses.sh
sbatch scripts/sbatch/check_synapses.sh                 # validate them
sbatch scripts/sbatch/make_synapse_notebook.sh          # rebuild + execute this notebook
```

Every aggregation runs in duckdb against the parquet files, so this notebook
reads the databases directly rather than a precomputed summary — nothing to keep
in sync, and only small summary/cache-index tables enter the kernel."""),
    ]

    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": KERNEL})
    out = AN / "synapse_inventory.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
