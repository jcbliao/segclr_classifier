# teasar_target

Separate named skeletons in `/orcd/compute/sdorkenw/001/segclr-db`, dataset
`microns`, `skeleton_name="teasar_target"`. This uses the same 2,442 roots and
original TEASAR sources frozen in
`/orcd/scratch/orcd/013/jcbliao/skeletons/segclr_registered_teasar_111nm_20260911/routes.json`
(2,209 presynaptic-priority cells plus 233 others).

For each original edge of length L, use `n = max(1, round(L / 111 nm))`, then
split it into n equal pieces of length L/n. Rounding follows Python/NumPy
nearest-integer rounding, with exact half ties to even. Existing vertices,
short edges, topology, and cable paths are preserved. Inserted coordinates and
radii are linearly interpolated. Lengths may exceed 111 nm: this is a target,
not a maximum. Source geometry is never taken from already-densified skeletons.

Artifacts and immutable source signatures:
`/orcd/scratch/orcd/013/jcbliao/skeletons/teasar_target/`.
`plan.json` freezes inputs; `skeletons/` holds NPZ geometry; `status/` records
full database array readback after bounded, serialized commits; `completion.json`
records cohort verification and pooled edge-length statistics. Retrying identical
geometry is idempotent; conflicting existing named geometry is rejected.

Completed and verified: 2,442 named skeletons, 244,138,385 nodes,
244,135,943 edges. Mean edge length 110.9797 nm; standard deviation 15.9350 nm;
range 0.8432–166.5000 nm. Full stored geometry was compared with generated
arrays during registration; final root identities and node/edge counts match.
The pooled distribution is saved as `edge_length_histogram.png` and `.pdf`
beside `completion.json`.

CPU-only registration jobs:

- Tests: 22807151 (passed; temporary DB roundtrip, idempotence, conflict rejection).
- Preflight: 22807148 (all original sources exist; no preexisting teasar_target rows).
- Two-cell production pilot: 22807162 (full readback passed).
- Full preparation/registration array: 22807196 (completed).
- Final cohort verification: 22807201 (completed).

No inference jobs or embeddings are requested for `teasar_target`.
The existing native training fragment configuration is separately set to
9, 17, 33, 65, 129, 257 nodes at scales 32, 16, 8, 4, 2, 1, including the
true full-resolution center. Those fragment counts do not modify this geometry.

Implementation: `scripts/build_teasar_target.py`; batch wrapper:
`scripts/sbatch/build_teasar_target.sh` (`MODE=preflight`, `prepare`, or `verify`).
