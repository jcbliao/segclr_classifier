# CAVE V1DD addition analysis

Open `embedding_augmentation_comparison.ipynb`, adapted from the CAVE
confidence-0.7 augmentation notebook. It compares mean, pointwise MLP, and GT
with and without V1DD training cells on paired fold 0, without a confidence
cutoff. Figures and tables separate MICrONS, V1DD, and combined held-out data.

The notebook loads live epoch CSVs and completed result JSONs directly from
`results/presynaptic/cave_v1dd_addition`. It includes paired metric changes,
target-family recall and support, and window/cell confusion matrices. It can
be run before training starts and shows pending-result messages. Re-run cells
to refresh results; no inference or training is launched by this notebook.

Helpers: `analysis/presynaptic/cave_v1dd_addition/cave_v1dd_comparison.py`.
