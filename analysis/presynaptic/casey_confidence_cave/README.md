# Casey confidence sweep on CAVE skeletons

Matched to `../casey_confidence_native/`: confidence cutoffs 0, 0.3, 0.5,
0.7, and 0.9; Mean, Pointwise MLP, and full GT; fold 0 of the existing cell-held-out split
balanced by Casey fine type. Native labels, exclusions, thalamocortical
inclusion, and fold membership are copied exactly. CAVE uses K=10 observed
nodes rather than native K=17. Each confidence has 30 thalamocortical cells,
with 24 training and six held out in each fold.

The 15 training runs use 30 epochs, class-balanced window sampling,
mixed-cell batches of 16, AMP, learning rate 1e-4, weight decay 1e-5,
and GT depth/head count 4/4, matching the native training configuration.
Checkpoint selection and reported test metrics use the same held-out cells,
as in the native experiments.

Prepare cohorts and test actual CAVE windows and model optimization:

```bash
segclr_db/.venv/bin/python scripts/prepare_casey_confidence_cave.py
segclr_db/.venv/bin/python -m pytest -q analysis/presynaptic/casey_confidence_cave/test_experiment.py
```

Submit fold 0 (five confidence cutoffs × three models = 15 runs):

```bash
bash scripts/submit_casey_confidence_cave.sh
```

The submission script splits fold 0 into eight preemptable and seven normal-GPU
runs, with all three models in each array, followed by
result summarization. Results live in `results/presynaptic/casey_confidence_cave/k10`;
cohorts live in `/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/casey_confidence_cave/k10`.
The notebook preserves the native section layout, including cohort counts,
training progress, best-checkpoint metrics, and confusion matrices. With no
training results yet it displays cohort counts and explains that results are pending.
Use the **segclr_db (.venv)** kernel.
