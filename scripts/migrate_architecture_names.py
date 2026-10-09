"""One-off: carry the architecture renames onto everything already on disk.

Two aggregation methods were renamed for what they actually are:

    fully_connected -> mpnn_complete     (message passing over the complete graph)
    deepsets        -> pointwise_mlp     (a per-node MLP, then the mean)

`scripts/train_gnn.py` now tags their runs `mpnn_complete_L{layers}` and
`pointwise_mlp_L{layers}`, and `ModelConfig` carries the new names. Everything
written before that still carries the old spelling, in three places that all
matter:

- **The paths.** Results directories, `<run>.json` summaries, the
  feature-prediction caches and the shared window-prediction caches are keyed
  by run name, and `analysis/all_windows/architecture_comparison.py` now looks for the new
  tags. An un-renamed directory is silently skipped rather than plotted under
  the wrong label, which is quiet enough to miss.
- **The `ModelConfig` inside EVERY checkpoint.** `train_gnn.py --resume`
  refuses a `checkpoint_last.pt` whose stored config differs from the one just
  built, so a renamed field or architecture value strands a half-trained run:
  resume aborts, and a fresh start silently throws away the epochs already
  done. This is not confined to the renamed runs -- a `ModelConfig` records
  every architecture's fields whatever architecture it was built for, so a GT
  or mean checkpoint carries the `deepsets_*` fields too, and the dataclass's
  own `__eq__` raises `AttributeError` on the renamed one. Hence the checkpoint
  pass walks all of `results/`, not only the directories being moved.
- **The argparse destinations recorded in `<run>.json`.** Those summaries store
  the parsed CLI args verbatim, so a completed run's record still names
  `deepsets_layers`. Nothing reads them programmatically, but a record written
  in a vocabulary the code no longer uses is a trap for whoever reads it next.
- **The `state_dict` keys.** The submodule `WindowClassifier.deepsets` is now
  `WindowClassifier.pointwise_mlp`, and a state_dict is keyed by attribute
  path, so `load_state_dict` on an unmigrated checkpoint fails on every one of
  phi's weights.

Rewriting a checkpoint needs torch, so this goes through
`scripts/sbatch/migrate_architecture_names.sh` like any other execution here.

Two properties worth relying on:

- **Idempotent.** A run already carrying the new name is skipped, and a
  checkpoint already holding the new architecture and field names is left
  untouched, so a partial run (or a second rename landing later) can simply be
  repeated.
- **Refuses rather than clobbers.** If a destination path already exists the
  whole migration aborts before moving anything.

    python scripts/migrate_architecture_names.py --dry-run
    python scripts/migrate_architecture_names.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "all_windows"
FEATURE_CACHE = REPO / "analysis" / "all_windows" / "feature_prediction_cache"
PREDICTION_CACHE = Path(
    os.environ.get(
        "SEGCLR_PREDICTION_CACHE",
        "/orcd/scratch/orcd/013/jcbliao/segclr/window_prediction_cache",
    )
)

# The run-name tag, the `architecture` value, the `ModelConfig` field names and
# the state_dict prefix are four separate spellings of the same rename, so they
# are listed together rather than in four places that could drift apart.
RUN_TAGS = (("_fc_L", "_mpnn_complete_L"), ("_deepsets_L", "_pointwise_mlp_L"))
ARCHITECTURES = {"fully_connected": "mpnn_complete", "deepsets": "pointwise_mlp"}
CONFIG_FIELDS = {
    "deepsets_hidden_dim": "pointwise_mlp_hidden_dim",
    "deepsets_out_dim": "pointwise_mlp_out_dim",
    "deepsets_layers": "pointwise_mlp_layers",
}
STATE_PREFIXES = {"deepsets.": "pointwise_mlp."}


def all_checkpoints() -> list[Path]:
    """Every checkpoint under results/, whatever architecture it was built for.

    Not restricted to the runs being renamed: see the module docstring on why a
    GT or mean checkpoint carries the renamed fields too.
    """
    return sorted(RESULTS.glob("*/checkpoint_*.pt"))


def rename(name: str) -> str:
    for old, new in RUN_TAGS:
        name = name.replace(old, new)
    return name


def running_jobs() -> list[str]:
    """This user's currently RUNNING SLURM jobs, as "id name" lines.

    A training job builds its checkpoint paths fresh every epoch, so renaming
    its results directory out from under it makes the next write land on a path
    that no longer exists.
    """
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "RUNNING", "-o", "%i %j"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    # This migration itself runs under sbatch, so it is in its own listing.
    self_id = os.environ.get("SLURM_JOB_ID", "")
    return [
        line for line in out.stdout.splitlines()
        if line.strip() and line.split(maxsplit=1)[0] != self_id
    ]


def rewrite_json(value, run_old: str | None = None, run_new: str | None = None):
    """Rewrite the run name, the architecture value and the renamed arg keys.

    `run` is matched by key so an unrelated string that happens to equal the
    run name is left alone; `architecture` likewise, since these payloads
    record the parsed CLI args verbatim -- which is also why the argparse
    destinations (`deepsets_layers` and friends) are renamed as keys.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "run" and run_old is not None and item == run_old:
                out[key] = run_new
            elif key == "architecture" and item in ARCHITECTURES:
                out[key] = ARCHITECTURES[item]
            else:
                out[CONFIG_FIELDS.get(key, key)] = rewrite_json(item, run_old, run_new)
        return out
    if isinstance(value, list):
        return [rewrite_json(item, run_old, run_new) for item in value]
    return value


def rewrite_checkpoint(path: Path) -> bool:
    """Point a checkpoint's stored config and state_dict at the new names.

    Imported here rather than at module scope so `--dry-run` costs nothing and
    needs no torch import.
    """
    import torch

    from gnn.model import ModelConfig

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    changed = False

    config = ckpt.get("config")
    if config is not None:
        # Unpickled with the CURRENT dataclass, so the instance carries
        # whatever field names it was written with -- rebuilt from its own
        # __dict__ rather than mutated, since a stale field is a missing
        # attribute the dataclass's own __eq__ would trip over.
        fields = {CONFIG_FIELDS.get(k, k): v for k, v in vars(config).items()}
        architecture = fields.get("architecture")
        if architecture in ARCHITECTURES:
            fields["architecture"] = ARCHITECTURES[architecture]
        if fields != vars(config):
            ckpt["config"] = ModelConfig(**fields)
            changed = True

    for key in ("model_state",):
        state = ckpt.get(key)
        if not isinstance(state, dict):
            continue
        renamed = {}
        for name, tensor in state.items():
            for old, new in STATE_PREFIXES.items():
                if name.startswith(old):
                    name = new + name[len(old):]
                    changed = True
                    break
            renamed[name] = tensor
        ckpt[key] = renamed

    if not changed:
        return False
    # Written beside the original and moved into place, so an interrupted job
    # leaves the old checkpoint intact rather than a truncated one -- these are
    # the only copy of a half-trained run's weights.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return True


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

    runs = sorted(
        {
            name
            for name in (
                p.name if p.is_dir() else p.stem
                for p in RESULTS.iterdir()
                if p.is_dir() or p.suffix == ".json"
            )
            if rename(name) != name
        }
    )
    moves: list[tuple[Path, Path]] = []
    json_rewrites: list[tuple[Path, str, str]] = []
    for run in runs:
        new = rename(run)
        for src, dst in (
            (RESULTS / run, RESULTS / new),
            (RESULTS / f"{run}.json", RESULTS / f"{new}.json"),
            (FEATURE_CACHE / f"{run}.npz", FEATURE_CACHE / f"{new}.npz"),
            (FEATURE_CACHE / f"{run}.summary.json", FEATURE_CACHE / f"{new}.summary.json"),
            (PREDICTION_CACHE / f"{run}.npz", PREDICTION_CACHE / f"{new}.npz"),
        ):
            if src.exists():
                moves.append((src, dst))
        # Named at their post-move paths, since both passes run after the moves.
        for path in (
            RESULTS / new / "best_metrics.json",
            RESULTS / f"{new}.json",
            FEATURE_CACHE / f"{new}.summary.json",
        ):
            json_rewrites.append((path, run, new))
    # Every other results JSON gets the key/value renames with no run rename,
    # for the same reason the checkpoint pass is not confined to moved runs.
    named = {path for path, _, _ in json_rewrites}
    for path in sorted(RESULTS.glob("*.json")) + sorted(RESULTS.glob("*/best_metrics.json")):
        if path not in named:
            json_rewrites.append((path, None, None))

    collisions = [dst for _, dst in moves if dst.exists()]
    if collisions:
        print("REFUSING: destination(s) already exist -- resolve by hand:", file=sys.stderr)
        for dst in collisions:
            print(f"  {dst}", file=sys.stderr)
        return 1

    print(f"{len(runs)} run(s), {len(moves)} path(s) to move:")
    for src, dst in moves:
        print(f"  {src.relative_to(REPO) if REPO in src.parents else src}\n    -> {dst.name}")
    if args.dry_run:
        print(f"\nwould rewrite up to {len(all_checkpoints())} checkpoint(s)")
        print("--dry-run: nothing moved")
        return 0

    for src, dst in moves:
        os.replace(src, dst)
    print(f"\nmoved {len(moves)} path(s)")


    rewritten = 0
    for path, old, new in json_rewrites:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        updated = rewrite_json(payload, old, new)
        if updated != payload:
            path.write_text(json.dumps(updated, indent=2))
            rewritten += 1
    print(f"rewrote {rewritten} JSON file(s)")

    paths = all_checkpoints()
    touched = sum(rewrite_checkpoint(path) for path in paths)
    print(f"rewrote {touched} of {len(paths)} checkpoint(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
