"""One-off: append the fold tag to every existing run name and everything keyed to it.

`scripts/train_gnn.py` now ends every run name with `_fold{split_seed}`, so two
folds of the same model can no longer land on one results directory. Runs that
predate that tag are named without it, which leaves them unreadable by the
analysis parsers (which now require a fold) and, worse, still collidable: a
fold-1 rerun of an untagged run would have overwritten it.

This renames them in place -- results directories, their `<run>.json`
summaries, the feature-prediction caches, and the shared window-prediction
caches -- so nothing is left addressed by the old name. Every existing run in
this repo trained on split_seed 0, so they all become `_fold0`.

Two properties worth relying on:

- **Idempotent.** A name already carrying a fold tag is skipped, so a partial
  run (interrupted, or a cache directory that was offline) can simply be
  repeated.
- **Refuses rather than clobbers.** If a destination already exists the whole
  migration aborts before moving anything, since silently merging two runs'
  directories is exactly the collision the fold tag exists to prevent.

The run name is also embedded INSIDE two kinds of JSON -- `best_metrics.json`
and the feature-prediction `.summary.json` (in every one of its correlation,
bin and cell rows) -- so those are rewritten too. The `.npz` prediction caches
carry no internal run name (see data/window_prediction_cache.py's schema), so
renaming the file is the whole job there.

    python scripts/migrate_run_names_to_fold.py --dry-run
    python scripts/migrate_run_names_to_fold.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "all_windows"
FEATURE_CACHE = REPO / "analysis" / "all_windows" / "feature_prediction_cache"
MANIFEST = REPO / "data" / "manifest.json"
PREDICTION_CACHE = Path(
    os.environ.get(
        "SEGCLR_PREDICTION_CACHE",
        "/orcd/scratch/orcd/013/jcbliao/segclr/window_prediction_cache",
    )
)

# A fixed-node run name, with the embedding count and optionally the tags that
# may follow it, and NO fold tag yet. Anything else in results/ (the exported
# neuroglancer link .txt files, for instance) is left alone.
#
# The head prefix is a character class, not the literal "gnn_lcpn_scratch_":
# `gnn_flat_scratch_` runs exist too, and pinning one prefix would silently
# leave the other family untagged -- an orphan the analysis parsers would then
# skip without saying why.
UNTAGGED = re.compile(r"^gnn_[a-z]+_scratch_.*_n(?:10|20|40)(?:_noemb)?(?:_frozenagg)?$")
TAGGED = re.compile(r"_fold\d+$")


def running_jobs() -> list[str]:
    """This user's currently RUNNING SLURM jobs, as "id name" lines.

    A training job builds its checkpoint paths fresh every epoch, so renaming
    its results directory out from under it makes the next write land on a path
    that no longer exists. Checked rather than trusted to memory: this
    migration is one-off and easy to fire while a sweep is live.
    """
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "RUNNING", "-o", "%i %j"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def rewrite_run_field(value, old: str, new: str):
    """Replace the run name wherever it appears as a `run` value, at any depth."""
    if isinstance(value, dict):
        return {
            k: (new if k == "run" and v == old else rewrite_run_field(v, old, new))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [rewrite_run_field(v, old, new) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, move nothing")
    parser.add_argument(
        "--force", action="store_true",
        help="migrate even with jobs RUNNING (they may be writing into results/)",
    )
    args = parser.parse_args()

    busy = running_jobs()
    if busy and not (args.dry_run or args.force):
        print(
            f"REFUSING: {len(busy)} job(s) are RUNNING and may be writing into results/.\n"
            "Renaming a results directory under a live training job sends its next\n"
            "checkpoint write to a path that no longer exists. Let them drain, or pass\n"
            "--force if you know none of them touch results/:",
            file=sys.stderr,
        )
        for line in busy:
            print(f"  {line}", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST.read_text())
    split_seed = manifest.get("split_seed", 0)
    split_fracs = manifest.get("split_fracs")
    suffix = f"_fold{split_seed}"
    print(f"manifest split_seed={split_seed} -> appending {suffix!r}\n")

    runs = sorted(
        {
            p.name if p.is_dir() else p.stem
            for p in RESULTS.iterdir()
            if (p.is_dir() or p.suffix == ".json")
            and UNTAGGED.match(p.name if p.is_dir() else p.stem)
        }
    )
    already = sorted(p.name for p in RESULTS.iterdir() if p.is_dir() and TAGGED.search(p.name))
    if already:
        print(f"{len(already)} run(s) already tagged, skipping them")
    if not runs:
        print("nothing to migrate")
        return 0

    # (source, destination) for every path this migration touches.
    moves: list[tuple[Path, Path]] = []
    json_rewrites: list[tuple[Path, str, str]] = []
    for run in runs:
        new = run + suffix
        for src, dst in (
            (RESULTS / run, RESULTS / new),
            (RESULTS / f"{run}.json", RESULTS / f"{new}.json"),
            (FEATURE_CACHE / f"{run}.npz", FEATURE_CACHE / f"{new}.npz"),
            (FEATURE_CACHE / f"{run}.summary.json", FEATURE_CACHE / f"{new}.summary.json"),
            (PREDICTION_CACHE / f"{run}.npz", PREDICTION_CACHE / f"{new}.npz"),
        ):
            if src.exists():
                moves.append((src, dst))
        # Rewritten AFTER the move, so the paths named here are the new ones.
        for path in (
            RESULTS / new / "best_metrics.json",
            RESULTS / f"{new}.json",
            FEATURE_CACHE / f"{new}.summary.json",
        ):
            json_rewrites.append((path, run, new))

    collisions = [dst for _, dst in moves if dst.exists()]
    if collisions:
        print("REFUSING: destination(s) already exist -- resolve by hand:", file=sys.stderr)
        for dst in collisions:
            print(f"  {dst}", file=sys.stderr)
        return 1

    print(f"{len(runs)} run(s), {len(moves)} path(s) to move:")
    for src, dst in moves:
        print(f"  {src.relative_to(REPO) if REPO in src.parents else src}"
              f"\n    -> {dst.name}")
    if args.dry_run:
        print("\n--dry-run: nothing moved")
        return 0

    for src, dst in moves:
        os.replace(src, dst)
    print(f"\nmoved {len(moves)} path(s)")

    rewritten = 0
    for path, old, new in json_rewrites:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        updated = rewrite_run_field(payload, old, new)
        # The summary written before train_gnn.py recorded provenance has no
        # fold keys at all; fill them in so every artifact says which split it
        # came from, not just the ones written after the change.
        if path.parent == RESULTS:
            updated.setdefault("fold", f"fold_{split_seed}")
            updated.setdefault("split_seed", split_seed)
            updated.setdefault("split_fracs", split_fracs)
        if updated != payload:
            path.write_text(json.dumps(updated, indent=2))
            rewritten += 1
    print(f"rewrote the run name inside {rewritten} JSON file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
