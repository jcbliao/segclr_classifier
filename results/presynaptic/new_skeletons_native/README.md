# Priority native-skeleton classification

Submitted September 15, 2026 for the 2,209 priority proofread-axon cells.

- Source: audited `resampled_teasar_2209/native` geometry and presynaptic sites.
- Embeddings: per-cell registered inference routes; exact original-node ID lookup.
- Six scales: factors 32, 16, 8, 4, 2, and 1, with K=9, 17, 33, 65, 129, and 257 nodes including the center.
- Nested contraction preserves branch points, endpoints, and cable lengths.
- Same 5 µm soma cut, 2 µm site cutoff, exact window deduplication, and previous cell splits.
- Mean, ordinary graph transformer (depth 4 / heads 4, full geometry features), and pointwise MLP at each scale.
- 30 epochs, AMP, class-balanced loss, Adam learning rate 1e-4, weight decay 1e-5.
- Mean and pointwise MLP consume SegCLR embeddings only, as in existing baselines.
- Results and checkpoints: `local_center/scale<factor>/k<K>/<model-run-name>/`; final summaries alongside model folders.

The native skeleton windows contain only nodes with exact SegCLR embeddings; there are
no unembedded in-between routing nodes. Their graph transformer therefore uses
`graph_transformer`, as in the non-native CAVE skeleton experiment. The separate
`graph_transformer_teasar` model handles mixed embedded and unembedded nodes and
does not match this native input.

SLURM chain: priority verification `22600108` → build `22794358` (32 CPU tasks,
8 concurrent) → finalize `22794359` → training `22794360` (0=mean, 1=graph
transformer, 2=pointwise MLP; 2 concurrent L40S GPUs). Training has the existing
six-hour job limit and `--resume`; if a task reaches its time limit before 30
epochs, it needs a continuation from its saved checkpoint.

Non-priority inference arrays `22681428`, `22681429` and verification `22681434`
were canceled. Priority recovery array `22727496` initially failed tasks 0–3
because `balanced_normal` was missing from the inference argument choices;
that parser was corrected, those tasks requeued, and verification dependencies
refreshed.

Validation: seven dense-presynaptic regression tests passed. A real priority
cell produced 481 unique K=7 windows and passed forward evaluation through all
three model encoders. Shell syntax, Python compilation, and CLI parsing passed.

The original September 15 submission queued only the coarsest scale. The verified
September 16 relaunch queues all six scales and three architectures per scale.
Each scale now has its own build array and finalization job; its three training
tasks depend only on that scale's finalized dataset. See
`submission_20260916_0737.tsv` for current job IDs.
The five active build arrays were raised to task throttles of 3, 3, 3, 3,
and 4 for scales 16, 8, 4, 2, and 1, respectively: 16 concurrent tasks total.
Training arrays have no effective array throttle: both mean and pointwise MLP
tasks may run together, subject to the partition's resource limits.
