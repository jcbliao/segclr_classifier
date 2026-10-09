# Native skeleton pre/post comparison

Six configurations: mean, pointwise MLP, and GraphTransformer, each using
presynaptic features alone or presynaptic features plus a 64-dimensional
postsynaptic embedding concatenated after pooling, immediately before the
shared 4-layer/128-wide ResNet classification trunk and LCPN heads.

The existing native bin16, k17, Casey confidence >=0.7 fold-0 dataset, labels,
cell-held-out splits, and thalamocortical inclusion are reused. Both groups
exclude unresolved postsynaptic root IDs and empty postsynaptic masks before applying the same existing
window deduplication: uniqueness is the complete sorted presynaptic node set
within a root. Different postsynaptic points do not make duplicate fragments
unique. The first eligible valid site in original row order supplies the
postsynaptic vector for a retained fragment. No post vectors are averaged.

Both groups train anew with seed 0, 30 epochs, class-balanced sampling, BF16
training autocast, up to 4096 windows per batch, attention budget 14450688,
16 mixed cells per batch, existing memmaps, Adam lr=1e-4/weight_decay=1e-5,
and GT depth 4/heads 4. The extra 64 input dimensions increase the ResNet
input layer width. Presynaptic encoders keep the same input dimensions.

Runs live in scale16/k17/conf0.7/fold0/{pre_only,pre_post}. Result JSON files
record matched train/test window counts. comparison.csv and post_minus_pre.csv
will report window- and cell-level accuracy, balanced accuracy, precision, and
macro F1. The existing training script selects the best epoch on the held-out
fold, which is also used for final metrics; these are exploratory comparisons.

Dependencies and task mapping are recorded in submission.json. Preparation
runs after embedding verification; it validates exact synapse-ID alignment
and requires every retained cohort site to have a valid embedding. Empty masks
are excluded by user instruction; unexpected crop failures still stop preparation. Training
runs after preparation; comparison runs only after all six trainings succeed.
All six training configurations run on mit_normal_gpu with account and
QoS mit_amf_advanced_gpu. Failed or timed-out tasks can be resubmitted with
--resume through scripts/sbatch/train_native_pre_post.sh; incomplete training
does not automatically launch a replacement job.

Validation: scripts/check_native_pre_post.py checks the shuffled-ID embedding
join, unresolved exclusion, preservation of non-unique presynaptic fragments
with differing post points, identical pre-only/pre+post indices, PyG collation,
fusion dimensions, backward gradients, and evaluation for all six variants.

The 25 cohort sites with empty masks are exported in empty_mask_sites.csv.
Coordinates are the postsynaptic point in nanometres; both pre- and postsynaptic
root IDs are included. Root IDs must be read as integers or strings to avoid
spreadsheet floating-point rounding. Replacement jobs are in submission.json.

Three additional matched-width controls (tasks 6–8) append the 64-dimensional
mean of raw presynaptic node embeddings after pooling. They use identical data
and settings, with 128 input dimensions for mean and 192 for MLP/GT, matching
pre+post parameter counts. Window macro F1 is the primary comparison metric.

The three raw-presynaptic-mean controls were moved to mit_preemptable using
mit_general / normal at the user's request. Latest job IDs are in submission.json.
