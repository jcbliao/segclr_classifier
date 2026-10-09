# Repository verification

Verification uses the lab's `segclr_db/.venv` environment and prepared datasets. Notebook sources, saved notebook outputs, and authored prose are preserved; execution copies and JSON reports are stored under the ignored `logs/notebook_checks/` directory.

## Checks completed on 2026-10-08

- Python source and all 36 notebook files passed syntax checks.
- Generated catalog entries, source-report hashes, training arguments, and local documentation links passed consistency checks.
- Five focused tests passed for config parsing, CLI precedence, invalid settings, experiment namespaces, and preservation of publication metadata.
- The actual training CLI accepted the graph-transformer JSON recipe and rendered its help without starting training.
- The existing analysis figure smoke test completed with zero failures, including rendering and metric consistency checks.

## Notebook execution

All 36 notebooks completed full execution successfully, including both historical training-curve notebooks. [Per-notebook execution results](notebook_execution.json) record the successful runs and durations. Scientific parameters and dataset populations were not reduced for verification.

The execution reports are retained locally in `logs/notebook_checks/`. The full notebook index is in [`analysis/README.md`](../analysis/README.md).

One optional appendix in `native_single_pre_post/pre_post_comparison.ipynb` referenced an absent exclusion CSV. It now displays the missing path when that report is unavailable; its comparison cells and authored prose are unchanged. The notebook passed a subsequent complete execution. No exclusion counts were fabricated or substituted from another experiment.

The all-windows skeleton-statistics notebook requires a longer cell timeout for its full-population mixture fits. The verifier now defaults to 2700 seconds per cell, with scientific settings unchanged.

The verifier also inherits the existing native-notebook workaround of disabling NumPy transparent-huge-page allocation on the cluster. This affects allocation behavior, not scientific parameters. The successful full-population all-windows run took approximately 22 minutes before that launcher workaround was added.

Re-run the checks using the commands in the [reproduction guide](reproducibility.md).
