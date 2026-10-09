# Native single-embedding pre/post analysis

Same fold-0 cohort and eligible, deduplicated k17 windows as native_skeletons_pre_post. Each presynaptic input is the raw 64D embedding at the nearest observed skeleton node to the synapse; no neighborhood averaging. The graph has one node, no edges, zero LPE, and zero relative position. Postsynaptic vectors and their exclusion policy are unchanged.

Run the notebook to compare mean, pointwise MLP, and graph transformer under pre_only, pre_post, and pre_mean_control (duplicates the single raw pre vector after pooling as the matched-width control). Training uses the original 30 epochs and hyperparameters. Results are in results/presynaptic/native_single_pre_post/scale16/k1/conf0.7/fold0.

Eligibility and deduplication remain based on the original k17 windows to keep the comparisons matched. Multiple retained windows may therefore select the same central embedding.
