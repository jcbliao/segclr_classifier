# Synapse labeler

Create a Spelunker link with the supplied cell in the normal `seg` layer and
two separate layers of partner L2 meshes:

- Orange: postsynaptic fragments at the cell's outgoing synapses.
- Cyan: presynaptic fragments at the cell's incoming synapses.

Run:

```bash
segclr_db/.venv/bin/python scripts/make_synapse_labeler.py ROOT_ID
```

Uses the existing CAVE credentials (or `CAVE_TOKEN`) and materialization 1718.
The root ID must belong to that snapshot; use `--mat-version` for another version.

For every outgoing/incoming synapse, the script resolves the opposite partner's supervoxel to
one L2 ID at the selected materialization timestamp. Only that L2 fragment is
shown; neighboring chunks are excluded. Repeated L2 IDs are deduplicated.
Unresolved partners (root 0) are reported and omitted from the mesh selection.

Each fragment layer uses the default segmentation source with only its mesh
subsource enabled. All selected L2 IDs within each layer have the same color; the layer's volume
and graph subsources are disabled to keep the selection local. The normal
segmentation layer retains the supplied cell root and its usual sources.

Optional flags:

- `--color '#00ccff'`: set the outgoing layer color (incoming stays cyan).
- `--output /path/link.txt`: choose the output location.
- `--mat-version 1718`: choose the synapse snapshot.
- `--synapse-table synapses_pni_2`: choose the synapse table.

The default output is `results/presynaptic/synapse_labeler_ROOT_ID.txt`.
Open the URL stored in that file in a browser with access to the default
segmentation. Adjacent `.state.json` and `.fragments.json` files contain the
viewer state and per-direction, per-synapse partner/L2 IDs, respectively. Empty neighborhoods
are reported; query failures stop execution instead of creating partial links.

Offline verification:

```bash
segclr_db/.venv/bin/python scripts/smoke_test_synapse_labeler.py
```
