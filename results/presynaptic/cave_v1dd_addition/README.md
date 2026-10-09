# CAVE V1DD addition comparison

Six training runs compare mean pooling, pointwise MLP, and the full
GraphTransformer under two training conditions on paired fold 0.

- `no_v1dd`: existing Casey MICrONS training cells without a confidence cutoff, including
  the established thalamocortical cohort policy.
- `with_v1dd`: identical MICrONS training cells plus the fold's V1DD training cells.

Both conditions evaluate on exactly the same held-out MICrONS and V1DD cells.
MICrONS keeps its existing Casey fine-type stratified folds. V1DD is split by
cell into five folds, balanced by family and fine label with deterministic seed
0, and only fold 0 is trained/evaluated. The MICrONS source is the conf0 cohort
(2,029 cells before window eligibility); its exclusions contain no confidence
exclusions. V1DD test cells never enter either condition's training set. Separate metrics
are reported for MICrONS, V1DD, and the combined test set, at window and
majority-vote cell levels. This pipeline follows the existing trainer's use of
the held-out test fold for checkpoint selection; it does not add a separate
validation partition.

V1DD starts from 266 axon-status-positive cells: 71 ITC -> putative BipFam,
173 DTC -> putative MartFam, and 22 STC-Neurogliaform- -> NglFam. No dendrite
status filter or V1DD confidence threshold is applied. These ITC/DTC mappings
are candidate-family assignments rather than visually confirmed morphologies.

The presynaptic rules match the existing MICrONS CAVE K=10 cache:

1. Choose the skeleton node nearest the nucleus/soma position as soma center.
2. Remove nodes within 5 micrometers of that center and their incident edges.
3. Query all outgoing `synapses_v1dd` sites at materialization 1196, in nm,
   validating the returned row count against a separate count query.
4. Accept a site only within 2 micrometers of a surviving embedded node.
5. Collect the ten nearest nodes by cable distance, including the center;
   reject components too small for a full window. Use induced edges and the
   same window-local LPE.
6. Deduplicate by the complete sorted membership set within each root.

The 2-micrometer threshold is read from the existing k10 cache metadata;
the reusable builder's current 5-micrometer default is explicitly overridden.
MICrONS NPZ files are reused without rebuilding. V1DD embeddings are attached
by exact CAVE skeleton node ID. Inference covers every V1DD CAVE node; this
can differ from the coverage of the original MICrONS embedding release.
V1DD image voxels are 38.8 x 38.8 x 45 nm, whereas MICrONS training imagery
uses 32 x 32 x 40 nm.

Cells with no valid unique windows are excluded explicitly in
`window_inventory.csv`, identically for both comparison conditions.
Missing skeletons, embeddings, source caches, or failed queries fail preparation
instead of silently reducing the training cohort. Finalization verifies paired
split identities and exercises the real dataset loader on both domains.

Dataset and manifests:
`/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_v1dd_addition`.
Results are separated into `no_v1dd` and `with_v1dd` under this directory.
Analysis CSVs will be written to `analysis/presynaptic/cave_v1dd_addition`:
`summary_by_fold.csv`, `summary.csv`, `paired_deltas_by_fold.csv`,
`paired_deltas.csv`, and `per_class_metrics.csv` (with class support).

Training uses the existing ResNet classification head, 30 epochs, AMP,
class-balanced sampling, mixed-cell batches of 16, Adam learning rate 1e-4,
and weight decay 1e-5. Mean and pointwise MLP receive embeddings only; GT uses
relative positions, LPE, adjacency bias, and four layers/four heads.
All paired models use seed 0. Training and its existing checkpoint resume
behavior are unchanged apart from allowing pointwise MLP on CAVE windows and
adding dataset-separated final metrics.

Submitted September 30, 2026:

- topology/synapses: 24477886, starts before inference finishes;
- windows: 24477887, waits for topology/synapses and inference verification 24474619;
- dataset finalization: 24477888;
- mean/MLP: 24483983, four tasks on preemptable GPUs;
- GT: 24483984, two tasks on the advanced normal GPU account;
- summary: 24483985, waits for all six training tasks to succeed.

The earlier five-fold confidence-0.7 training arrays 24477889 and 24478025,
and summary 24478034, were cancelled before training started. Topology,
synapse preparation, windows, and finalization retain their existing jobs.

See `submission_20260930.tsv` for dependencies. Task index is
`6 * fold + 3 * condition + architecture`, where conditions are
`no_v1dd`, `with_v1dd` and architectures are mean, pointwise MLP, GT.
Training arrays each allow at most two simultaneous tasks.

Re-submit using `bash scripts/submit_cave_v1dd_addition.sh VERIFICATION_JOB_ID`.
Check existing jobs before re-submitting to avoid duplicate work.
