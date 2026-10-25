# Whole-cell structure analysis

Open `../structure_ablation.ipynb`. The notebook reads local JSON caches only; it never queries the database or runs inference.

- `plan.json`: frozen manifests and checkpoint SHA256 hashes, model and perturbation definitions.
- `cache/ablations_seed{0,1,2}.json`: window and cell metrics; seed zero includes deterministic interventions, all seeds include randomized interventions.
- `cache/predictions_seed{0,1,2}.npz`: one class index per original held-out window and condition (duplicate windows are preserved).
- `cache/weight_analysis.json`: class-stratified activation summaries, distribution histograms, and parameter norms.
- `jobs.json`: Slurm submission IDs.

Packed original windows live at `/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/whole_cell_structure_n10_20261011`. Root IDs, labels, pointers, geometry and embeddings are stored in NumPy arrays; `strict_soma_mask.npy` identifies the primary cohort. Packed data are validated against the original extractor before inference. Each GPU task memory-maps this cache and uploads fields when memory allows.

Run preparation with `scripts/analyze_whole_cell_structure.py prepare --workers 15`, GPU checkpoint validation with `validate`, full inference with `evaluate --seed N`, and activation analysis with `inspect`. CPU packing uses 16 allocated CPUs; GPU inference uses FP16 and batches all windows. No model training or database modification is performed.

See notebook explanations for the original eight-class taxonomy, missing-nucleus exclusion, checkpoint-selection limitation, and why disabling all CLS attention is degenerate. Uniform CLS readout retains collection of node information while disabling ordinary-node attention updates. This inference intervention is not a separately trained nodewise model.
