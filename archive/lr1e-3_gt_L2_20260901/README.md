# GraphTransformer depth-2 sweep, discarded: wrong learning rate

Twelve runs (four ablations x 10/20/40 nodes) trained at **lr 1e-3** because
`scripts/sbatch/submit_embedding_sweep.sh` still defaulted to it, while every
other run in `results/` was trained at **1e-4**. A depth comparison against the
depth-4 sweep would have been confounded by the optimizer, not by depth.

Moved rather than deleted: nothing here is cited anywhere, but the epoch
metrics are a real record of what 1e-3 does to this model, and `--resume` finds
`checkpoint_last.pt` by run name -- leaving these in `results/` would have made
the re-run silently continue from 1e-3 weights instead of starting fresh.

The defaults in `scripts/sbatch/train_gnn.sh`,
`scripts/sbatch/submit_embedding_sweep.sh` and `scripts/train_gnn.py` are all
1e-4 now, so the three agree and no caller has to remember to pass it.
