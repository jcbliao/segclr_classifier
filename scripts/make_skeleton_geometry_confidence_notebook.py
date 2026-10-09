"""Create and execute the geometry/confidence notebook on a Slurm worker."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'analysis/presynaptic/skeleton_comparison/geometry_vs_confidence.ipynb'


def markdown(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook(cells=[
    markdown('''# CAVE / TEASAR skeleton agreement versus Casey confidence

Each point represents one matched cell, comparing the two full skeletons under
the same v1718 root. Casey's identity and confidence come from the existing
root/nucleus join. Confidence describes **cell-type annotation confidence**;
the geometry metrics describe agreement between skeletonization methods.

The default uses Casey **coarse confidence** and **coarse cell type**. Change
the configuration to fine confidence/fine type or your original classification.
No confidence threshold or additional cell-type filter is applied.

Mean distance is lower for closer centerlines; overlap F1 is higher for greater
agreement. A cable-length ratio of 1 means equal total cable. Distances are exact
to target segments, with cable-weighted source integration at 250 nm spacing
(mean quadrature bound: 125 nm).'''),
    code('''%matplotlib inline
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "scripts/compare_skeleton_geometry.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.presynaptic.skeleton_comparison import geometry_confidence as gc

METRICS_CSV = gc.DEFAULT_CSV
CONFIDENCE_COLUMN = "casey_coarse_confidence"  # or "casey_fine_confidence"
CELL_TYPE_COLUMN = "casey_coarse"  # or "casey_fine", "cell_classification"
CONFIDENCE_LABEL = CONFIDENCE_COLUMN.replace("_", " ").capitalize()
METRICS = gc.DEFAULT_METRICS.copy()
# Optional additional panels, using any numeric column in metrics.csv:
# METRICS["precision_1um"] = "TEASAR cable coverage at 1 µm"
# METRICS["recall_1um"] = "CAVE cable coverage at 1 µm"
# METRICS["cave_p95_um"] = "CAVE → TEASAR 95th percentile distance (µm)"
BINS = 10
EXPORT = True
FIGURE_DIR = ROOT / "analysis/presynaptic/skeleton_comparison/geometry_confidence_figures" / f"{CONFIDENCE_COLUMN}__{CELL_TYPE_COLUMN}"
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

cells, audit = gc.load_metrics(METRICS_CSV, CONFIDENCE_COLUMN, CELL_TYPE_COLUMN)
display(audit.to_frame())
display(cells.groupby("cell_type").agg(
    cells=("root_id", "size"), median_confidence=("confidence", "median"),
    min_confidence=("confidence", "min"), max_confidence=("confidence", "max")))'''),
    markdown('''## Global plots

All cells have equal weight. Color indicates the selected cell type. Black
points/lines show medians in ten equal-width confidence bins (empty bins are
omitted), placed at the median confidence of cells in each bin. Confidence 1
belongs to the final bin. Confidence values are not jittered. Annotations show
sample size and Spearman rank correlation; correlation is undefined when a
variable is constant or there are fewer than three cells. Pooled correlations
can reflect differences in cell-type composition. Overlap and cable-ratio
axes span the observed values to show variation; their y axes need not start
at zero. All observations, including distance outliers, remain visible.'''),
    code('''fig = gc.plot_global(cells, METRICS, CONFIDENCE_LABEL, BINS)
if EXPORT:
    gc.save_figure(fig, FIGURE_DIR, "global")
plt.show()
plt.close(fig)'''),
    markdown('''## Plots by cell type

One figure per metric, with a panel for every observed cell type. Each figure
shares the same x/y scales across types to allow direct comparison. Black lines
and correlation annotations are computed separately within each type. Small
groups and types with little confidence variation remain visible.'''),
    code('''for metric, label in METRICS.items():
    fig = gc.plot_by_cell_type(cells, metric, label, CONFIDENCE_LABEL, BINS)
    if EXPORT:
        gc.save_figure(fig, FIGURE_DIR, "by_type_" + metric.replace(".", "p"))
    plt.show()
    plt.close(fig)'''),
    markdown('''## Numerical summaries and source cells

These are descriptive associations, not a causal test or a reconstruction
accuracy assessment. Both the global and within-type Spearman correlations are
included. Exported tables preserve root IDs as strings. The CSV includes
directional coverage bounds for checking uncertainty near each tolerance.'''),
    code('''correlations = gc.correlation_table(cells, METRICS)
display(correlations.pivot(index="cell_type", columns="metric", values="spearman_rho").round(3))
display(cells[["root_id", "cell_type", "confidence", *METRICS]].sort_values("confidence").head(20))
if EXPORT:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(FIGURE_DIR / "correlations.csv", index=False)
    cells.to_csv(FIGURE_DIR / "plotted_cells.csv", index=False)
    print(f"Exported PNG/PDF figures and tables to {FIGURE_DIR}")'''),
])
nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
nb.metadata['language_info'] = dict(name='python')
DEST.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, DEST)
NotebookClient(nb, timeout=600, kernel_name='python3',
               resources={'metadata': {'path': str(ROOT)}}).execute()
tmp = DEST.with_suffix('.tmp.ipynb')
nbf.write(nb, tmp)
tmp.replace(DEST)
print(f'Created and executed {DEST}', flush=True)
