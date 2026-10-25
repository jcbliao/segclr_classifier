"""Build shared fast caches, benchmark full-size training, and release held CV tasks."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "analysis/presynaptic/segmentation_augmentation/cv"


def pack(fold):
    import torch
    from scripts.train_gnn import build_training_parser, build_training_datasets
    from scripts.training_config import parse_training_args
    from data.packing_cache import identity, prepare_cached

    torch.set_num_threads(1)
    args = parse_training_args(
        build_training_parser(),
        ["--config", str(CV / "configs" / f"fold{fold}_shrink1.json")],
    )
    train, test, manifest = build_training_datasets(args)
    key = identity(args, manifest)
    for split, ds in [("train", train), ("test", test)]:
        if not prepare_cached(ds, key, split, workers=8, build=True):
            raise RuntimeError("fast packing was bypassed")
    directory = CV / "fast_cache"
    directory.mkdir(exist_ok=True)
    (directory / f"fold{fold}.json").write_text(
        json.dumps(dict(fold=fold, key=key, complete=True), indent=2) + "\n"
    )
    print(f"Fold {fold}: fast CPU caches ready", flush=True)


def pilot():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "scripts/test_packed_segmentation_augmentation.py",
        ],
        check=True,
        cwd=ROOT,
    )
    original = CV / "configs/fold3_both2.json"
    config = json.loads(original.read_text())
    config.update(
        epochs=3,
        resume=False,
        optimized_loading=True,
        cuda_graphs=True,
        results_dir=str(CV / "fast_pilot_results"),
    )
    path = CV / "fast_pilot_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ, SEGCLR_PACKING_CACHE_ONLY="1")
    subprocess.run(
        [sys.executable, "-u", "scripts/train_gnn.py", "--config", str(path)],
        check=True,
        cwd=ROOT,
        env=env,
    )
    performances = list((CV / "fast_pilot_results").glob("*/performance.jsonl"))
    if len(performances) != 1:
        raise RuntimeError("expected one benchmark performance record")
    rows = [json.loads(line) for line in performances[0].read_text().splitlines()]
    if len(rows) != 3 or not all(
        r["resident_features"] and r["cuda_graphs"] for r in rows
    ):
        raise RuntimeError("default fast path was not active")
    slow = []
    results = (
        ROOT
        / "results/presynaptic/cave_segmentation_augmentation_conf0.7/six_class/fold3"
    )
    for file in results.glob("*/*/performance.jsonl"):
        slow.extend(
            json.loads(line)["training_seconds"]
            for line in file.read_text().splitlines()
            if json.loads(line)["epoch"] > 0
        )
    if not slow:
        raise RuntimeError("no previous fold3 timings available for comparison")
    fast = statistics.median(r["training_seconds"] for r in rows[1:])
    old = statistics.median(slow)
    report = dict(
        fast_training_seconds=fast,
        previous_training_seconds=old,
        speedup=old / fast,
        resident_features=True,
        cuda_graphs=True,
        passed=fast < old * 0.5,
    )
    (CV / "fast_training_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("less than 2x speedup; held jobs remain held")


def release():
    if not json.loads((CV / "fast_training_benchmark.json").read_text())["passed"]:
        raise RuntimeError("fast benchmark has not passed")
    released = []
    for task in range(32, 45):
        job = f"26130459_{task}"
        subprocess.run(["scontrol", "release", job], check=True)
        released.append(job)
    (CV / "fast_training_released.json").write_text(
        json.dumps(dict(jobs=released), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("pack", "pilot", "release"))
    p.add_argument(
        "--fold", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 3))
    )
    a = p.parse_args()
    if a.mode == "pack":
        pack(a.fold)
    elif a.mode == "pilot":
        pilot()
    else:
        release()
