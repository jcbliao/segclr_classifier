# CAVE embedding augmentation comparison (confidence 0.7)

Five folds use the maximum complete augmentation set available for all 1,773
cells. Evaluation always uses clean embeddings. Each training node uniformly
samples one available choice on each access.

| Training condition | Per-node choices |
|---|---|
| `clean` | clean only |
| `gray` | clean + gray samples 0–19 (21 choices) |
| `flip` | clean + flip samples 0–6 (8 choices) |
| `structured_low` | clean + structured-low samples 0–3 (5 choices) |

On 2026-09-24, submitted packing/validation job 23651120 and dependent training
arrays 23651172 (tasks 1–2,4–17) and 23651173 (tasks 18–19).
Task index = 4 * fold + condition index (clean, gray, flip, structured_low).
These launch all conditions for folds 1–4 and replace gray/flip fold 0.
The existing clean and structured-low fold 0 runs already use the full set.
Previous four-draw gray/flip fold 0 runs are preserved in
`archive/embaug_four_draws_20260924_102547`.

Packed data: `/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/cave_embedding_training_choices/conf0.7/all_folds/max_fp16`.
The packer checks precision, sample metadata, node alignment, embedding shapes,
and finite values before dependent training can start.
