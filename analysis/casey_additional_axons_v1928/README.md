# Casey candidates and extended axons at v1928

Queries `minnie65_phase3_v1` materializations 1718 and 1928. The public
datastack currently exposes 1926 rather than 1928. Casey labels come from
`microns_public_v1822_ct_csm_v2_sep15.parquet`; `cell_id` is matched to
`nucleus_detection_v0.id`, then the nucleus's v1928 `pt_root_id` is used to
join proofreading. No confidence threshold or dendrite-status filter is used.
Selection requires `strategy_axon` to be `axon_partially_extended` or
`axon_fully_extended`. All selected rows also have `status_axon=true`.

Existing identities are excluded using all 2,442 roots in the v1718 MICrONS
Lance store, not just a confidence-filtered training cohort. The v1718 nucleus
table resolves 2,412 of these roots; 30 lack nucleus records. Matching by
nucleus identity accounts for changes in segmentation root IDs.

`additional_casey_extended_axons.csv` and its Parquet counterpart contain the
one qualifying candidate: nucleus 305183, v1928 root 864691135611879943,
MartFam / MartFam-f, coarse confidence 0.95, fine confidence 0.0666667,
fully extended axon. This nucleus already had a fully extended axon at v1718
(root 864691135409856969); it is additional relative to the existing store,
not newly proofread between the two versions. There are no additional BipFam
or NglFam candidates satisfying these criteria.

The full inventories contain 2,155 extended roots at v1718 and 2,184 at v1928,
but each contains exactly the same 2,100 distinct nucleus IDs. Of those,
239 switch from partially to fully extended, 518 remain fully extended,
and 1,343 remain partially extended. There are no nucleus-identified additions
or losses. Roots without a nucleus record number 55 and 84 respectively;
these cannot be compared by nucleus identity or called additional cells.

`casey_candidates_audit.parquet` retains all 4,961 candidates after exclusion,
including ones that fail the extended-axon filter. Raw query snapshots,
extended inventories, exclusions, and roots without nucleus records are
preserved here. `summary.json` records metadata, timestamps and counts.

Reproduce with:

```bash
segclr_db/.venv/bin/python scripts/check_casey_additional_axons.py
```

The script reuses saved raw snapshots. Remove those snapshots only if a fresh
query is intended. These files are analysis exports; no training data is modified.
