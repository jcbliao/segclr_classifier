# Linear versus ResNet heads: 10-node whole-cell windows

Open [the comparison notebook](../linear_vs_resnet.ipynb). It uses cached final reports and epoch CSV files; it never runs model inference. Rerunning the notebook refreshes its small `report_cache.json` snapshot.

Four new linear LCPN-head models pair with existing ResNet-head models: raw mean pooling, two-layer skeleton MPNN with LPE and relative-position inputs, full four-layer/four-head GT with LPE, relative position and adjacency bias, and two-layer pointwise MLP. Both learned aggregator and decoder train end to end. `freeze_aggregator=false` in every configuration. Mean pooling itself has no parameters. The `architecture=linear` node encoder is not part of this comparison.

The ResNet is the shared four-block, width-128 classification trunk, followed by the same local hierarchy classifiers. Disabling the trunk leaves one trainable linear layer per hierarchy branch (`cls_hidden_dim=null`), not one flat softmax. The nonlinear aggregators remain nonlinear with either decoder.

Matched settings: 10-node neighborhoods from the original CAVE skeleton/SegCLR caches, original eight-class taxonomy, fixed cell-held-out 80/20 split with seed 0, 16 epochs, batch size 4096, AdamW learning rate 1e-4 and weight decay 1e-5, class-balanced sampling, and no embedding augmentation or postsynaptic context. All new runs use 16 CPUs, 15 loader workers and a maximum of four concurrent L40S GPUs. Historical mean used 31 loader workers; this resource difference does not change the model or examples. Models train from scratch; `resume=true` permits recovery only within each new output directory.

The 5 µm nucleus-centered Euclidean cut removes nodes before neighborhood construction; the 16 roots without nucleus positions remain uncut as in the historical experiment. All 2,332 eligible roots have membership caches. Preflight checked 1,866 train cells / 9,702,190 windows and 466 held-out cells / 2,423,150 windows, with exact per-class held-out support matching all four ResNet reports. These are local windows sampled throughout a cell, rather than one whole-cell embedding.

The historical trainer selects checkpoints on the same held-out split used for reporting. Results therefore describe one model-selection split, not an untouched test set or five-fold cross-validation. Window counts do not provide independent cell-level replication. The notebook does not claim statistical significance or presuppose that the ResNet wins. Changing decoder capacity may change convergence, so it also displays all 16 epochs.

Files:

- `plan.json`: paired report paths, hashes and configuration paths.
- `manifest.json`: frozen labels, splits and explicit hierarchy.
- `configs/00.json`–`03.json`: complete replay configurations, changing the decoder and resource/output settings only.
- `preflight.py` / `preflight.json`: membership, taxonomy, class support and soma-exclusion audit.
- `smoke_check.py` / `smoke_check.json`: real cached-window backward passes verifying head and learned-encoder gradients.
- `comparison.py`: cached report/CSV loading, matched-setting checks and plotting.
- `report_cache.json`: regenerable metrics and learning-curve snapshot.
- `render.py`: execute the notebook without training or inference.
- `jobs.json`: submitted training and post-training notebook-render job IDs.

Original reports: `results/all_windows/*resnet4x128_n10_fold0.json`. New reports and checkpoints: `results/all_windows/linear_vs_resnet/`. Standard per-window predictions are published by the existing trainer to the configured `SEGCLR_PREDICTION_CACHE` (default `/orcd/scratch/orcd/013/jcbliao/segclr/window_prediction_cache`). The notebook itself needs only the reports and epoch CSVs.

Training launcher: `scripts/sbatch/train_whole_window_linear_heads.sh`. The dependent CPU render job refreshes notebook outputs after all four training jobs terminate, leaving unsuccessful pairs visibly incomplete if any job fails. Manual refresh: `segclr_db/.venv/bin/python analysis/all_windows/linear_vs_resnet/render.py` from the repository root.
