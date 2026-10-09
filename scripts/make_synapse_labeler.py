#!/usr/bin/env python3
"""Create a Spelunker link showing a cell and its local postsynaptic L2 meshes."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synapses import (DEFAULT_DATASTACK, DEFAULT_MAT_VERSION,
                           DEFAULT_SYNAPSE_TABLE, fetch_synapses)
from make_neuroglancer_prediction_link import VIEWER, viewer_state


def collect_fragments(client, synapses, mode="outgoing"):
    if mode not in ("outgoing", "incoming"):
        raise ValueError(f"Unknown synapse mode: {mode}")
    partner_side = "postsynaptic" if mode == "outgoing" else "presynaptic"
    supervoxels = sorted({int(row.partner_supervoxel_id)
                          for row in synapses.itertuples(index=False)
                          if int(row.partner_root_id) != 0 and int(row.partner_supervoxel_id) != 0})
    lookup = {}
    if supervoxels:
        timestamp = client.materialize.get_timestamp()
        ids = client.chunkedgraph.get_roots(supervoxels, stop_layer=2, timestamp=timestamp)
        if len(ids) != len(supervoxels):
            raise RuntimeError("Incomplete postsynaptic supervoxel-to-L2 response")
        lookup = dict(zip(supervoxels, map(int, ids), strict=True))
    records, selected = [], set()
    for row in synapses.itertuples(index=False):
        partner = int(row.partner_root_id)
        supervoxel = int(row.partner_supervoxel_id)
        fragment = lookup.get(supervoxel, 0) if partner else 0
        if fragment and fragment >> 56 != 2:
            raise RuntimeError(f"Expected L2 mesh ID, received {fragment}")
        fragments = [fragment] if fragment else []
        selected.update(fragments)
        records.append({
            "synapse_id": str(row.synapse_id), f"{partner_side}_root_id": str(partner),
            f"{partner_side}_supervoxel_id": str(supervoxel),
            "point_nm": [float(getattr(row, f"ctr_{axis}_nm")) for axis in "xyz"],
            "l2_ids": list(map(str, fragments)),
        })
    return sorted(selected), records


def make_state(root_id, l2_ids, position_nm=None, color="#ff8800", points_nm=None, incoming_l2_ids=None):
    # Reuse the standard image, segmentation (including skeletons), and camera defaults.
    state = viewer_state("", {"run": "synapse_labeler"}, ["rainbow"],
                         "", [], {"rainbow": ""}, 0)
    state["layers"] = state["layers"][:2]
    state["layers"][1].update({
        "segments": [str(root_id)], "selectedAlpha": 0.5,
        "notSelectedAlpha": 0, "objectAlpha": 1,
    })
    source = copy.deepcopy(state["layers"][1]["source"][0])
    source["subsources"] = {"default": False, "graph": False, "bounds": False, "mesh": True}
    source["enableDefaultSubsources"] = False
    ids = [str(value) for value in l2_ids]
    name = "postsynaptic L2 fragments"
    state["layers"].append({
        "type": "segmentation", "name": name, "source": [source],
        "segments": ids, "segmentColors": {value: color for value in ids},
        "segmentDefaultColor": color, "selectedAlpha": 1, "notSelectedAlpha": 0,
        "objectAlpha": 1,
    })
    incoming = copy.deepcopy(state["layers"][2])
    incoming["name"] = "presynaptic L2 fragments (incoming)"
    incoming_ids = [str(value) for value in (incoming_l2_ids if incoming_l2_ids is not None else [])]
    incoming.update({"segments": incoming_ids,
                     "segmentColors": {value: "#00ccff" for value in incoming_ids},
                     "segmentDefaultColor": "#00ccff"})
    state["layers"][2]["name"] = "postsynaptic L2 fragments (outgoing)"
    name = state["layers"][2]["name"]
    state["layers"].append(incoming)
    for layer in state["layers"]:
        layer["visible"] = True
        layer["archived"] = False
    state["showSlices"] = True
    state["selectedLayer"] = {"layer": name, "visible": True}
    if position_nm is not None:
        state["position"] = (np.asarray(position_nm) / [8, 8, 40]).tolist()
    if points_nm is not None and len(points_nm):
        points = np.asarray(points_nm, dtype=float)
        lo, hi = points.min(axis=0), points.max(axis=0)
        state["position"] = ((lo + hi) / 2 / [8, 8, 40]).tolist()
        state["projectionScale"] = max(30000.0, float(np.linalg.norm(hi - lo)) * 1.25)
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root_id", type=int, help="presynaptic cell root ID at the materialization version")
    ap.add_argument("--mat-version", type=int, default=DEFAULT_MAT_VERSION)
    ap.add_argument("--synapse-table", default=DEFAULT_SYNAPSE_TABLE)
    ap.add_argument("--color", default="#ff8800", help="shared fragment color (#RRGGBB)")
    ap.add_argument("--output", type=Path, help="URL file; defaults to results/presynaptic/synapse_labeler_ROOT.txt")
    args = ap.parse_args()
    if not 0 < args.root_id < 2**64:
        ap.error("root_id must be a positive uint64")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", args.color):
        ap.error("color must be #RRGGBB")

    import caveclient

    client = caveclient.CAVEclient(DEFAULT_DATASTACK, version=args.mat_version,
                                   auth_token=os.environ.get("CAVE_TOKEN"))
    records_by_mode, ids_by_mode = {}, {}
    for mode in ("outgoing", "incoming"):
        print(f"Querying {mode} synapses for {args.root_id} (materialization {args.mat_version})...", flush=True)
        synapses = fetch_synapses(client, [args.root_id], mode, args.synapse_table)
        ids_by_mode[mode], records_by_mode[mode] = collect_fragments(client, synapses, mode)
    points = [record["point_nm"] for records in records_by_mode.values() for record in records]
    state = make_state(args.root_id, ids_by_mode["outgoing"], color=args.color,
                       points_nm=points, incoming_l2_ids=ids_by_mode["incoming"])
    output = args.output or ROOT / "results" / "presynaptic" / f"synapse_labeler_{args.root_id}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(VIEWER + quote(json.dumps(state, separators=(",", ":")), safe="") + "\n")
    output.with_suffix(".state.json").write_text(json.dumps(state, indent=2) + "\n")
    report = {"root_id": str(args.root_id), "mat_version": args.mat_version,
              "synapse_table": args.synapse_table,
              "directions": {mode: {"l2_ids": list(map(str, ids_by_mode[mode])),
                                    "synapses": records_by_mode[mode]}
                             for mode in ("outgoing", "incoming")}}
    output.with_suffix(".fragments.json").write_text(json.dumps(report, indent=2) + "\n")
    for mode, records in records_by_mode.items():
        empty = sum(not record["l2_ids"] for record in records)
        print(f"{mode}: {len(records)} synapses, {len(ids_by_mode[mode])} unique L2 meshes; "
              f"{empty} unresolved fragments.")
    print(output)



if __name__ == "__main__":
    main()
