# Three-class k17 pre + post training

**Superseded on 2026-10-07:** the native k17 MLP and GT tasks in array
25175953 were canceled at the user's request. Their replacements aggregate 10
CAVE embeddings and live in `analysis/presynaptic/cave_skeletons_pre_post`.
The existing single-embedding runs continue. Any k17 checkpoints here are
partial artifacts of the canceled runs.

Pointwise MLP and Graph Transformer with 17 raw presynaptic embeddings and the
original postsynaptic features, trained on the original folds 0–4.

| Output class | Original classes |
| --- | --- |
| BasketFam | BasketFam |
| nonBasketInhibitory | BipFam, MartFam, NglFam |
| Excitatory | pyramidal, thalamocortical |

The manifests retain all original cells, labels, and split assignments. Only the
classification hierarchy changes: one root classifier predicts the three merged
classes directly. Sampling uses the same inverse-square-root frequency policy,
computed for the merged classes. Training remains 30 epochs with the same
optimizer, architectures, feature eligibility, and checkpoint selection settings.

Results go to
`results/presynaptic/native_skeletons_pre_post/scale16/k17/conf0.7/three_class/foldN/pre_post`.
The two `*_three_class_5folds.ipynb` notebooks in the parent directory read these
results; run them after trainings finish.

From the repository root, submit all three model configurations and five folds:

```bash
bash scripts/submit_native_three_class_folds.sh
```

The submission first prepares and validates manifests and exercises the
three-output classifier loss, backward pass, and predictions on a CPU job. The
15-task GPU array depends on successful preparation. Task `3 * fold + 1` trains
the k17 MLP; `3 * fold + 2` trains the k17 GT. Task `3 * fold` trains the
single-embedding MLP. Successful submission records job IDs in `submission.json`
beside this file.

Local checks verified that all ten manifests preserve the original cells, splits,
and fold indices, and that every cell maps to one of the three classes. All 15
shell commands were resolved and checked without executing training.
Submission was attempted on 2026-10-07, but the scheduler could not be reached;
no job ID was received. The CPU model check and GPU training are still pending.
