# Experiment results

- `all_windows/`: fixed-node neighborhood experiments.
- `presynaptic/`: presynaptic experiments grouped by preparation, skeleton source, and input condition.
- `reproductions/`: suggested namespace for new reproductions launched with JSON configs.
- [`summary.csv`](summary.csv): generated inventory of final evaluation reports, with one row per experiment/run.

Run folders retain checkpoint and epoch-metric paths used by the trainer and analysis notebooks. Sibling `<run>.json` files are final evaluation reports. `best_metrics.json` inside a run directory can be an in-progress snapshot; these snapshots are excluded from the final-report catalog.

See the [model catalog](../models/README.md) for per-run cards and full metrics. The summary copies saved values without averaging folds or ranking different experiments. Its `report` and `report_sha256` columns identify the source. Checkpoints remain ignored by Git.

Regenerate the catalog through `sbatch scripts/sbatch/organize_repository.sh` from the repository root.
