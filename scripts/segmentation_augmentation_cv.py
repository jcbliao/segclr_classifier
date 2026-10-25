"""Prepare, validate, train and summarize nine mask augmentation conditions x five folds."""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis/presynaptic/segmentation_augmentation/cv"
RESULTS = ROOT / "results/presynaptic/cave_segmentation_augmentation_conf0.7/six_class"
BASELINES = ROOT / "results/presynaptic/cave_skeletons_pre_post/k10/conf0.7/six_class"
CONDITIONS = tuple(
    f"{direction}{radius}"
    for radius in (1, 2, 3)
    for direction in ("shrink", "expand", "both")
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapping(task):
    if not 0 <= task < 45:
        raise ValueError("task must be 0..44")
    return task // 9, CONDITIONS[task % 9]


def prepare():
    jobs = json.loads((ANALYSIS.parent / "full_jobs.json").read_text())
    augmentation = Path(jobs["output"]) / "training_choices"
    records = []
    folds = []
    cohort = None
    held_out = []
    for fold in range(5):
        baseline = next(
            (BASELINES / f"fold{fold}" / "pre_post").glob("*pointwise*.json")
        )
        saved = json.loads(baseline.read_text())
        args = saved["args"]
        manifest = Path(args["manifest"])
        data = json.loads(manifest.read_text())
        roots = set(data["cells"])
        if cohort is not None and roots != cohort:
            raise ValueError(
                "fold cohorts differ; the full embedding plan must cover their union"
            )
        cohort = roots
        held_out.append({r for r, v in data["cells"].items() if v["split"] == "test"})
        folds.append(
            dict(
                fold=fold,
                manifest=str(manifest),
                manifest_sha256=sha(manifest),
                baseline=str(baseline),
                baseline_sha256=sha(baseline),
            )
        )
        for condition in CONDITIONS:
            config = dict(
                args,
                embedding_augmentation="segmask_" + condition,
                embedding_augmentation_database=str(augmentation),
                results_dir=str(RESULTS / f"fold{fold}" / condition),
                resume=True,
            )
            path = ANALYSIS / "configs" / f"fold{fold}_{condition}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, indent=2) + "\n")
            records.append(
                dict(
                    task_id=len(records),
                    fold=fold,
                    condition=condition,
                    config=str(path),
                    config_sha256=sha(path),
                )
            )
    if set.union(*held_out) != cohort or sum(map(len, held_out)) != len(cohort):
        raise ValueError("held-out assignments are not a disjoint five-fold partition")
    spec = dict(
        conditions=CONDITIONS,
        folds=folds,
        tasks=records,
        cells=len(cohort),
        embedding_plan=jobs["plan"],
        augmentation_database=str(augmentation),
        results=str(RESULTS),
        embedding_finalization_job=jobs["finalization_job"],
        sampling="independent uniform choice per node, clean appears once",
        choice_counts={
            direction: {
                str(r): (2 * r + 1 if direction == "both" else r + 1) for r in (1, 2, 3)
            }
            for direction in ("shrink", "expand", "both")
        },
        strength_policy="cumulative layers 1 through maximum strength",
        evaluation="original clean embeddings",
        model="six-class pointwise MLP, pre/post, K10; saved baseline hyperparameters",
    )
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "experiment.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(
        json.dumps(
            dict(tasks=len(records), cells=len(cohort), conditions=CONDITIONS), indent=2
        )
    )


def preflight():
    import numpy as np
    from data.segmentation_augmentation import variant_names

    spec = json.loads((ANALYSIS / "experiment.json").read_text())
    plan_path = ROOT / spec["embedding_plan"]
    plan = json.loads(plan_path.read_text())
    expected = {i["root_id"]: i for shard in plan["shards"] for i in shard}
    if not plan.get("include_held_out") or plan["nodes_per_cell"]:
        raise ValueError(
            "CV requires full-node augmentation coverage, including fold0 held-out cells"
        )
    labels = variant_names(3)
    digest = sha(plan_path)
    summary = json.loads((Path(plan["output"]) / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("embedding generation is incomplete")
    for fold in spec["folds"]:
        if sha(fold["manifest"]) != fold["manifest_sha256"]:
            raise ValueError("fold manifest changed")
        if sha(fold["baseline"]) != fold["baseline_sha256"]:
            raise ValueError("baseline result changed")
        baseline = json.loads(Path(fold["baseline"]).read_text())
        database = Path(baseline["args"]["presynaptic_database"])
        manifest = json.loads(Path(fold["manifest"]).read_text())
        for root in manifest["cells"]:
            item = expected[int(root)]
            if (database / "cells" / f"{root}.npz").resolve() != Path(
                item["source"]
            ).resolve():
                raise ValueError(f"fold {fold['fold']} geometry differs for {root}")
    for root, item in expected.items():
        with np.load(
            Path(spec["augmentation_database"]) / "segmask" / "cells" / f"{root}.npz"
        ) as z:
            if str(z["plan_sha256"]) != digest or z["embedding_choices"].shape != (
                item["n_nodes"],
                7,
                64,
            ):
                raise ValueError(f"incomplete or mismatched augmentation cache: {root}")
            np.testing.assert_array_equal(z["choice_labels"], labels)
            if (
                len(np.unique(z["node_ids"])) != item["n_nodes"]
                or not np.isfinite(z["embedding_choices"]).all()
            ):
                raise ValueError(f"invalid augmentation cache: {root}")
    (ANALYSIS / "preflight.json").write_text(
        json.dumps(dict(passed=True, cells=len(expected), plan_sha256=digest), indent=2)
        + "\n"
    )
    print("CV preflight passed", flush=True)


def train(task):
    spec = json.loads((ANALYSIS / "experiment.json").read_text())
    fold, condition = mapping(task)
    record = spec["tasks"][task]
    if (
        record["fold"] != fold
        or record["condition"] != condition
        or sha(record["config"]) != record["config_sha256"]
    ):
        raise ValueError("task configuration changed")
    if not json.loads((ANALYSIS / "preflight.json").read_text())["passed"]:
        raise ValueError("preflight has not passed")
    import sys

    os.execv(
        sys.executable,
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts/train_gnn.py"),
            "--config",
            record["config"],
        ],
    )


def summarize():
    spec = json.loads((ANALYSIS / "experiment.json").read_text())
    rows = []

    def row(fold, condition, data):
        result = dict(fold=fold, condition=condition, best_epoch=data["best_epoch"])
        for key, prefix in [("window_test_metrics", "window")]:
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                result[f"{prefix}_{metric}"] = data[key][metric]
        return result

    for fold in spec["folds"]:
        rows.append(
            row(fold["fold"], "clean", json.loads(Path(fold["baseline"]).read_text()))
        )
        for condition in CONDITIONS:
            paths = list(
                (Path(spec["results"]) / f"fold{fold['fold']}" / condition).glob(
                    "*.json"
                )
            )
            completed = [
                json.loads(p.read_text())
                for p in paths
                if "test_metrics" in json.loads(p.read_text())
            ]
            if len(completed) != 1:
                raise ValueError(
                    f"expected one completed result: {fold['fold']}/{condition}"
                )
            rows.append(row(fold["fold"], condition, completed[0]))
    with (ANALYSIS / "fold_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for condition in ("clean", *CONDITIONS):
        own = [r for r in rows if r["condition"] == condition]
        result = dict(condition=condition, folds=len(own))
        for metric in [k for k in own[0] if k.startswith(("cell_", "window_"))]:
            values = [r[metric] for r in own]
            result[metric + "_mean"] = statistics.mean(values)
            result[metric + "_std"] = statistics.stdev(values)
            deltas = [
                r[metric]
                - next(
                    b[metric]
                    for b in rows
                    if b["condition"] == "clean" and b["fold"] == r["fold"]
                )
                for r in own
            ]
            result[metric + "_paired_delta_mean"] = statistics.mean(deltas)
        aggregates.append(result)
    with (ANALYSIS / "condition_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregates[0])
        writer.writeheader()
        writer.writerows(aggregates)
    print(ANALYSIS / "condition_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "train", "summarize"))
    parser.add_argument(
        "--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    args = parser.parse_args()
    if args.mode == "train":
        train(args.task_id)
    else:
        {"prepare": prepare, "preflight": preflight, "summarize": summarize}[
            args.mode
        ]()
