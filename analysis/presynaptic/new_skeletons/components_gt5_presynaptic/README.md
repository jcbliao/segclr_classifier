# Components with more than five presynaptic sites

Dataset: `/orcd/scratch/orcd/013/jcbliao/presynaptic_axons/k10`.

Components are connected components of the full stored TEASAR skeleton after the existing 5 µm soma exclusion. Distinct presynaptic site IDs mapped within 2 µm are counted; duplicate rows and unmatched sites do not increase counts. All edges inside a qualifying component are retained. This is a component filter, not a per-node synapse filter or an axon classifier.

The original plotting helper is reused. Plot and fit bounds are recalculated from the filtered population (0.5th–99.5th percentiles).

Retained 136,241,017 of 149,925,558 edges in 5,439 components.

The plotting script accepts `--database`, `--out`, and `--title` arguments.
