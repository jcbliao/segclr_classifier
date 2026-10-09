"""Headless check that both analysis notebooks' figures and selectors build.

Neither notebook can be exercised on the login node -- `data.dataset_lcpn` pulls
in torch, and the feature summaries are millions of rows before they are
reduced -- so this is the sbatch-side stand-in: it walks the same calls the
notebooks make, asserts the derived per-class metrics agree with what the
trainer wrote, and renders every figure to a throwaway Agg canvas.

Run it after `--summarize-only` has rebuilt the feature summaries, since the
per-hierarchy-class bins it checks for do not exist in a summary written before
those groups were scored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.all_windows import architecture_comparison as ac  # noqa: E402
from analysis.all_windows import feature_prediction_correlation as fpc  # noqa: E402

RESULTS = ROOT / "results" / "all_windows"
failures: list[str] = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def render(name, figure):
    """A figure that draws is a figure that has real data in every artist.

    `plt.subplots` succeeds on empty data; the renderer is what actually walks
    the artists, so drawing the canvas is the only cheap check that catches a
    panel built from a column that came back empty.
    """
    if figure is None:
        check(name, False, "returned None")
        return
    figure.canvas.draw()
    plt.close(figure)
    check(name, True)


print("=== architecture_comparison ===")
payloads = ac.load_confusions(RESULTS)
check("payloads found", len(payloads) > 0, f"{len(payloads)} runs")
classes = next(iter(payloads.values()))["classes"]
groups = ac.hierarchy_class_groups(classes)
names = [g["name"] for g in groups]
print("  groups:", names)
check("hierarchy parsed, not the flat fallback", len(groups) > len(classes),
      f"{len(groups)} groups over {len(classes)} classes")
check("breadth-first from the root", names[:2] == ["neuron", "non_neuron"], str(names[:2]))
check("level 1 follows level 0", names[2:4] == ["excitatory", "inhibitory"], str(names[2:4]))
check("finest classes all present", set(classes) <= set(names))

# The derived leaf scores must reproduce what the trainer wrote, or every coarse
# score above them is being summed out of a matrix that means something else.
for run, payload in sorted(payloads.items())[:3]:
    metrics = payload["window_test_metrics"]
    scores = ac.group_scores(metrics["confusion_matrix"], payload["classes"], groups)
    # `per_class_recall` / `per_class_precision` are dicts keyed by class name,
    # so both sides are pulled by name rather than zipped by position.
    for stat in ("recall", "precision"):
        written = metrics[f"per_class_{stat}"]
        derived = np.array([scores[c][stat] for c in written])
        check(f"leaf {stat} matches per_class_{stat} ({run[:40]})",
              bool(np.allclose(derived, list(written.values()), atol=1e-9)))
    # A root-level node covers every window, so its recall is the raw accuracy
    # of the neuron/non_neuron question -- and the two roots' supports must add
    # back up to the whole confusion matrix.
    covered = sorted({leaf for g in groups if g["level"] == 0 for leaf in g["leaves"]})
    check(f"level-0 nodes partition the classes ({run[:40]})",
          covered == sorted(payload["classes"]))

frame = ac.load_best_epochs(RESULTS, payloads)
check("best-epoch frame built", len(frame) > 0, f"{len(frame)} runs")
for column in ("window_f1_neuron", "window_f1_excitatory", "window_f1_pyramidal",
               "cell_f1_neuron", "cell_recall_inhibitory", "window_precision_astrocyte"):
    check(f"column {column}", column in frame.columns)
check("payload_epoch recorded", frame.payload_epoch.notna().all(),
      f"{int(frame.payload_epoch.isna().sum())} missing")
known = frame[frame.payload_epoch.notna()]
check("payload epoch is the CSV's best epoch",
      bool((known.payload_epoch == known.best_epoch).all()),
      f"{int((known.payload_epoch != known.best_epoch).sum())} of {len(known)} disagree")

averaged = ac.mean_over_folds(frame)
check("fold averaging keeps per-class columns", "window_f1_neuron" in averaged.columns)
check("fold averaging emits a spread column", "window_f1_neuron_std" in averaged.columns)

render("plot_within_count", ac.plot_within_count(frame))
render("plot_across_counts (macro)", ac.plot_across_counts(frame))
for class_name in ("neuron", "excitatory", "pyramidal"):
    for stat, scope in (("f1", "window"), ("recall", "cell"), ("accuracy", "window")):
        column = ac.metric_column(stat, scope, class_name)
        render(f"plot_across_counts {column}", ac.plot_across_counts(frame, column))

model = ac.model_label(sorted(payloads)[0])
render(f"plot_confusion_grid ({model})", ac.plot_confusion_grid(payloads, model))
render(f"plot_confusion_grid raw counts ({model})",
       ac.plot_confusion_grid(payloads, model, scope="cell", normalize=False))
counts = ac.runs_by_count(payloads, model)
check("confusion grid keeps a slot per count", sorted(counts) == list(ac.COUNTS), str(counts))

try:
    import ipywidgets  # noqa: F401
    pairs = sorted({(ac.describe_run(r)[0], ac.model_label(r))
                    for r in payloads if ac.describe_run(r) is not None})
    architecture, ablation = ac.linked_model_pickers(pairs)
    first = architecture.value
    architecture.value = [a for a, _ in pairs if a != first][0]
    check("ablation dropdown follows the architecture",
          all(a == architecture.value for a, m in pairs if m == ablation.value),
          f"{architecture.value} -> {ablation.value}")
    # The dropdown shows the input switches alone; its value stays the model
    # label every table is keyed by.
    labels = [label for label, _ in ablation.options]
    print("  ablation labels:", labels)
    check("ablation labels drop the architecture",
          all(not label.startswith(architecture.value) for label in labels), str(labels))
    check("ablation labels are only position/LPE",
          all(label in {"no position, no LPE", "position", "LPE", "position + LPE"}
              for label in labels), str(labels))
    check("ablation values are full model labels",
          all(model in {m for _, m in pairs} for _, model in ablation.options))
except ImportError:
    check("ipywidgets available", False)

print("\n=== feature_prediction_correlation ===")
runs = fpc.usable_runs()
cached = [r for r in runs if fpc.has_summary(r)]
check("cached summaries found", len(cached) > 0, f"{len(cached)} of {len(runs)} runs")
# The notebook's own default selection, not the full cached list: an
# architecture whose first run by name has no cache must still reach the grid
# through a sibling that does, or it vanishes from every panel.
# What the notebook actually plots: every cached model, one panel each.
grid = fpc.cached_runs(runs)
check("cached_runs returns only cached runs", all(fpc.has_summary(r) for r in grid))
check("cached_runs covers every cached run", set(grid) == set(cached),
      f"{len(grid)} vs {len(cached)}")
grid_models = [fpc.model_of(r) for r in grid]
check("cached_runs keeps sibling ablations apart",
      len(set(grid_models)) == len({fpc.model_of(r) for r in cached}),
      f"{len(set(grid_models))} distinct models")
check("cached_runs groups an architecture's runs together",
      all(grid_models[i] == grid_models[i - 1] or grid_models[i] not in grid_models[:i - 1]
          for i in range(1, len(grid_models))), " | ".join(dict.fromkeys(grid_models)))

present = sorted({fpc.architecture_of(r) for r in cached}, key=ac.architecture_rank)
print("  architectures present:", present)
check("depth is part of the architecture",
      all(" L" in a or a == "Mean" for a in present), str(present))
default = fpc.select_runs(runs, architectures=present)
selected_architectures = {fpc.architecture_of(r) for r in default if fpc.has_summary(r)}
print("  default selection:", *default, sep="\n    ")
for architecture in present:
    check(f"default selection keeps {architecture}",
          architecture in selected_architectures)
frames = fpc.load_frames(cached)
bins, cells, overall, corr = (frames["bins"], frames["cells"],
                              frames["overall"], frames["correlations"])
check("bins loaded", len(bins) > 0, f"{len(bins)} rows")
check("class order handed back", frames["classes"] == list(classes))
options = fpc.class_options(bins, frames["classes"])
print("  class options:", options)
check("class options are breadth-first", options[:2] == ["neuron", "non_neuron"], str(options[:2]))
check("coarse classes are binned", "neuron" in set(bins.class_name.dropna()))
per_class_features = set(bins[bins.class_name == "neuron"].feature)
check("per-class bins cover every feature", per_class_features == set(fpc.FEATURES),
      f"missing {sorted(set(fpc.FEATURES) - per_class_features)}")

models = fpc.model_order(bins)
print("  models:", models)
check("models ordered by the architecture ladder", len(models) > 0)
for feature in ("path_distance_um", "node_density_per_um"):
    for class_name in (None, "neuron", "pyramidal"):
        render(f"plot_feature_by_model {feature} / {class_name}",
               fpc.plot_feature_by_model(bins, feature, class_name))
for family in fpc.FEATURE_FAMILIES:
    for class_name in (None, "inhibitory"):
        render(f"plot_family_by_count {family} / {class_name}",
               fpc.plot_family_by_count(bins, family, models[0], class_name))
check("per-cell F1 stored", "f1" in cells.columns)
check("per-cell F1 is 2r/(1+r)",
      bool(np.allclose(cells.f1, np.where(cells.recall > 0,
                                          2 * cells.recall / (1 + cells.recall), 0.0))))
# --- granular classes, quantile marks, and the per-class bar chart ---------
quantiles = frames["quantiles"]
check("quantiles table loaded", len(quantiles) > 0, f"{len(quantiles)} rows")
for key in ("p25", "p50", "p75", "p90"):
    check(f"quantile column {key}", key in quantiles.columns)
options = fpc.class_options(bins, frames["classes"])
granular = [row["name"] for row in fpc.scoring_groups(frames["classes"]) if row["granular"]]
print("  granular classes:", granular)
check("granular classes are offered", set(granular) & set(options) != set(),
      f"{len(set(granular) & set(options))} of {len(granular)} present")
check("granular classes follow the coarse ones",
      options.index(granular[0]) > options.index("oligo") if granular else False)
check("granular bins exist", "L4IT" in set(bins.class_name.dropna()))

for feature in ("node_density_per_um", "soma_path_um"):
    for class_name in (None, "L4IT"):
        render(f"plot_feature_by_model {feature} / {class_name} + quantiles",
               fpc.plot_feature_by_model(bins, feature, class_name, quantiles=quantiles))
for family in fpc.FEATURE_FAMILIES:
    check(f"family {family} is mean/median only" if "soma" not in family else f"family {family}",
          len(fpc.FEATURE_FAMILIES[family]) == 2, str(fpc.FEATURE_FAMILIES[family]))
    render(f"plot_family_by_count {family} + quantiles",
           fpc.plot_family_by_count(bins, family, models[0], quantiles=quantiles))

granular_overall = overall[overall.granular.fillna(False).astype(bool)]
check("granular overall rows", len(granular_overall) > 0, f"{len(granular_overall)} rows")
check("granular rows name their level-2 parent",
      granular_overall.coarse_class.notna().all())
order = fpc.granular_order(overall, frames["classes"])
print("  bar order:", order)
# Each level-2 parent must occupy one contiguous run of the x axis, or the
# dividers and headings the bar chart draws would land mid-group.
parent_of = granular_overall.drop_duplicates("class_name").set_index("class_name").coarse_class
sequence = [parent_of.get(name) for name in order]
runs_seen, previous = [], object()
for parent in sequence:
    if parent != previous:
        runs_seen.append(parent)
        previous = parent
check("bar order groups subtypes under their parent",
      len(runs_seen) == len(set(runs_seen)), " | ".join(map(str, runs_seen)))
render(f"plot_class_bars ({models[0]})",
       fpc.plot_class_bars(overall, models[0], frames["classes"]))

render(f"plot_score_cdf F1 ({models[0]})", fpc.plot_score_cdf(cells, models[0], overall))
render(f"plot_score_cdf recall ({models[0]})",
       fpc.plot_score_cdf(cells, models[0], overall, score="recall"))
check("correlations cover coarse classes",
      "neuron" in set(corr.class_name.dropna()))

print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
