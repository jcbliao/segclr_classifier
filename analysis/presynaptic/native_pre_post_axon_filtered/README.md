# Native pre/post with dendrite-filtered windows

Same fold-0 bin16/k17/confidence >=0.7 cohort, three architectures (mean, pointwise MLP, GT), and three conditions (pre only, raw pre mean control, pre + post) as `../native_skeletons_pre_post/`. All nine runs use the saved geodesic 20 µm soma-exclusion dendrite-majority masks. This retains ties and other compartment labels; it is not a strict axon-only classifier selection.

Training settings match the original: 30 epochs, learning rate 1e-4, weight decay 1e-5, AMP, sampled class balance, mixed cell batches of 16, and the same prepared postsynaptic cache. Filtering is applied before shared postsynaptic eligibility and presynaptic node-set deduplication. Results and checkpoints are separate under `results/presynaptic/native_pre_post_axon_filtered/scale16/k17/conf0.7/fold0/`.

Open `pre_post_comparison.ipynb` for training curves, window/cell metrics, paired differences, per-class recall, confusion matrices, postsynaptic exclusions, and the filtering audit. No GPU is needed to read results.

Run loader checks with `segclr_db/.venv/bin/python analysis/presynaptic/native_pre_post_axon_filtered/check.py`.
Submit training and a dependent summary with `segclr_db/.venv/bin/python analysis/presynaptic/native_pre_post_axon_filtered/submit.py` (tasks 0–2 pre only, 3–5 pre + post, 6–8 raw pre mean control).
After completion run `segclr_db/.venv/bin/python analysis/presynaptic/native_pre_post_axon_filtered/summarize.py --include-controls`.
