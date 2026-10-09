# Reproduction and verification

## Environment

The lab environment is Python 3.11 in `segclr_db/.venv`, managed with `uv pip`. `segclr_db/` and `segCLR_cell_classification/` are separate dependency clones ignored by this repository. Their source locations and installation history are documented in [`CLAUDE.md`](../CLAUDE.md). Obtain those dependencies before recreating the environment; neither is bundled in a checkout of this project.

| Local checkout | Source remote |
|---|---|
| `segclr_db/` | [dorkenwald-lab/segclr_db](https://github.com/dorkenwald-lab/segclr_db) |
| `segCLR_cell_classification/` | [collina/segCLR_cell_classification](https://github.mit.edu/collina/segCLR_cell_classification) (MIT GitHub access) |

The existing [`setup_env.sh`](../scripts/sbatch/setup_env.sh) installs the GNN dependencies into an existing `segclr_db` environment. It uses lab-specific paths. Package installation and all project execution on the lab cluster must run through Slurm. The current environment is not a portable lockfile; record its exact dependency versions and dependency-clone commits when publishing a checkpoint.

## Data

Prepared datasets and caches are not distributed in Git. The trainer's defaults point to lab scratch storage. Supply paths appropriate to the experiment:

- Fixed-node neighborhoods: `--neighborhood-root` containing `neighborhoods/n10`, `n20`, or `n40`.
- Presynaptic experiments: `--dataset` plus the matching `--presynaptic-database` and, when needed, `--manifest`.
- Postsynaptic inputs: the experiment's prepared `--postsynaptic-cache`.

See [`data/`](../data/), [dense presynaptic preparation](../data/DENSE_PRESYNAPTIC.md), and [synapse data](../data/SYNAPSES.md). Preserve the original manifests, cell IDs, split seeds, label definitions, skeleton versions, and preprocessing settings with the checkpoint.

## Training

Submit from the repository root:

```bash
mkdir -p logs
sbatch scripts/sbatch/train_config.sh configs/baselines/mean.json \
  --results-dir results/reproductions/mean_n20
sbatch scripts/sbatch/train_config.sh configs/graph_transformer/fixed_node.json \
  --results-dir results/reproductions/gt_n20
```

The JSON supplies defaults and subsequent flags override them. The launcher requires an explicit output directory. Existing experiment launchers under `scripts/sbatch/` retain their paths and behavior.

For historical runs, start with the catalog entry's `config.json` and original evaluation report. These record the arguments available when the experiment ran, rather than an immutable environment or complete provenance. Fields absent from a historical report must be recovered from the original experiment metadata before claiming exact reproduction.

## Inference

Inference requires matching weights, model configuration, class hierarchy, and prepared input data. Existing entry points include [`infer_named_teasar.py`](../scripts/infer_named_teasar.py), [`export_neuroglancer_predictions.py`](../scripts/export_neuroglancer_predictions.py), and [`subcompartment_predict_all.py`](../scripts/subcompartment_predict_all.py). Use their matching Slurm launchers and dataset-specific options. A generic checkpoint is not substituted across these pipelines.

Each model card includes `artifact.json` with fields for the checkpoint download URL, SHA-256, and original training commit. Null fields mean the information has not been verified; the catalog does not publish or invent it.

## Notebooks

Use the [notebook index](../analysis/README.md). Experiment notebooks and their helper modules remain in their original directories. Open notebooks with this environment's Python kernel. Their working directory should be the notebook's directory; the execution verifier uses that convention too. External caches and services required by individual notebooks must be available.

The batch verifier executes copies in `logs/notebook_checks/`, preserving all original notebook sources, outputs, and prose. Execution reports distinguish failures and timeouts; syntax checking alone is not described as execution success. Full-data analyses use a 45-minute per-cell timeout by default; `--timeout` changes that allowance without changing notebook code or scientific parameters.

```bash
sbatch scripts/sbatch/verify_notebooks.sh --include-archive
```

## Catalog and structural checks

[Verification record](repository_verification.md) documents the repository checks and notebook execution fixes.

```bash
sbatch scripts/sbatch/organize_repository.sh
```

This regenerates the factual model catalog and `results/summary.csv`, checks Python and notebook syntax, verifies catalog hashes/configurations and local documentation links, and tests JSON config parsing. It does not recompute scientific metrics or select a preferred model. Saved final evaluations and ongoing best-epoch snapshots are kept separate.

The existing figure smoke test remains available as `SKIP_SUMMARIZE=1 sbatch scripts/sbatch/smoke_test_analysis_notebooks.sh`. Figure publication uses the existing [publishing workflow](interactive_figures.md); generating the catalog does not publish or push changes.
