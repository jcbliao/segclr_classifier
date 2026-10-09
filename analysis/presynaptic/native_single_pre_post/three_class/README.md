# Three-class single-embedding pre + post training

**Current submission:** preparation job 25175952 succeeded. Single-embedding
tasks 0, 3, 6, 9, and 12 in array 25175953 continue. The native k17 tasks in
that array were canceled and replaced by the CAVE k10 experiment in
`analysis/presynaptic/cave_skeletons_pre_post`.

Pointwise MLP with one raw presynaptic embedding and the original postsynaptic
features, trained on the original folds 0–4.

| Output class | Original classes |
| --- | --- |
| BasketFam | BasketFam |
| nonBasketInhibitory | BipFam, MartFam, NglFam |
| Excitatory | pyramidal, thalamocortical |

The manifests retain all original cells, labels, and split assignments. Only the
classification hierarchy changes: one root classifier predicts the three merged
classes directly. Sampling uses the same inverse-square-root frequency policy,
computed for the merged classes. Training remains 30 epochs with the same
optimizer, architecture, feature eligibility, and checkpoint selection settings.

Results go to
`results/presynaptic/native_single_pre_post/scale16/k1/conf0.7/three_class/foldN/pre_post`.
The `*_three_class_5folds.ipynb` notebook in the parent directory reads these
results; run it after trainings finish.

From the repository root, submit all three model configurations and five folds:

```bash
bash scripts/submit_native_three_class_folds.sh
```

The submission first prepares and validates manifests and exercises the
three-output classifier loss, backward pass, and predictions on a CPU job. The
15-task GPU array depends on successful preparation. Task `3 * fold` trains this
model; the following two tasks train the k17 MLP and GT. Successful submission
records job IDs in `submission.json` beside this file.

Local checks verified that all ten manifests preserve the original cells, splits,
and fold indices, and that every cell maps to one of the three classes. All 15
shell commands were resolved and checked without executing training.
Submission was attempted on 2026-10-07, but the scheduler could not be reached;
no job ID was received. The CPU model check and GPU training are still pending.
