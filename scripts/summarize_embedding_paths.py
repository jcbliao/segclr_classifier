"""Reduce the six path databases to one summary the notebooks can open instantly.

Reading 6 x 2,316 .npz files takes minutes; a notebook that does that on every
run does not get re-run. This does it once and stores exact aggregates plus a
random subsample of raw geodesic lengths, so the notebooks can still draw a real
distribution rather than only percentiles.

    sbatch scripts/sbatch/summarize_embedding_paths.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.build_embedding_paths import DEFAULT_OUT as PATHS  # noqa: E402

# Derived from the perisomatic radius, so the summary always describes the
# database the current settings build rather than whatever ran last.
OUT = ROOT / "analysis" / "embedding_paths_summary.npz"
from data.build_embedding_paths import CONFIGS as _CFG  # noqa: E402
from data.build_embedding_paths import NEIGHBORHOOD_CONFIGS as _NCFG  # noqa: E402

CONFIGS = list(_CFG)
NCONFIGS = list(_NCFG)

#: Raw values kept per config so the notebooks can draw real distributions.
SUBSAMPLE = 400_000
#: Geodesic histogram: 0 to 250 um in 0.5 um bins, plus an overflow bin.
NM_EDGES = np.arange(0, 250_001, 500, dtype=np.float64)
NODE_EDGES = np.arange(0, 202, dtype=np.float64)


def main() -> int:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    cells = manifest["cells"]
    rng = np.random.default_rng(0)

    out = {}
    per_cell = {c: [] for c in CONFIGS}

    # --- what the 15 um cut did, read once from the restricted skeletons ----
    restr = []
    for f in sorted((PATHS / "soma_restricted").glob("*.npz")):
        with np.load(f) as d:
            rid = int(d["root_id"][0])
            restr.append((rid, int(d["n_nodes_before"][0]), int(d["n_nodes_after"][0]),
                          int(d["n_components"][0]), int(bool(d["cut_applied"][0]))))
    restr = np.array(restr, dtype=np.int64)
    out["restrict_root_id"] = restr[:, 0]
    out["restrict_before"] = restr[:, 1]
    out["restrict_after"] = restr[:, 2]
    out["restrict_components"] = restr[:, 3]
    out["restrict_cut_applied"] = restr[:, 4].astype(bool)
    out["restrict_cell_type"] = np.array(
        [cells[str(r)]["cell_type"] for r in restr[:, 0]])
    print(f"soma_restricted: {len(restr):,} cells "
          f"({int((~out['restrict_cut_applied']).sum())} built uncut)", flush=True)

    for cfg in CONFIGS:
        files = sorted((PATHS / "paths" / cfg).glob("*.npz"))
        nm_hist = np.zeros(len(NM_EDGES) - 1, np.int64)
        node_hist = np.zeros(len(NODE_EDGES) - 1, np.int64)
        keep_nm, keep_nodes = [], []
        total = 0
        s_nm = 0.0
        rows = []
        # root_ids are kept in their own int64 list, never inside the float64
        # stats array: a CAVE root_id needs ~60 bits and float64 carries 53,
        # so round-tripping one through a float silently returns a different
        # -- and possibly real -- cell.
        row_ids = []
        for f in files:
            with np.load(f) as d:
                rid = int(d["root_id"][0])
                nm = d["geodesic_nm"].astype(np.float64)
                nodes = np.diff(d["path_offsets"]).astype(np.float64)
                ppn = d["paths_per_node"]
            if len(nm) == 0:
                continue
            nm_hist += np.histogram(nm, NM_EDGES)[0]
            node_hist += np.histogram(nodes, NODE_EDGES)[0]
            total += len(nm)
            s_nm += float(nm.sum())
            row_ids.append(rid)
            rows.append((len(nm), float(np.median(nm)), float(np.median(nodes)),
                         float(nm.max()), float(np.mean(ppn)), float(ppn.max())))
            # reservoir-ish: a fixed fraction per cell, so no cell dominates
            k = max(1, int(SUBSAMPLE * len(nm) / 60_000_000))
            if k < len(nm):
                idx = rng.choice(len(nm), size=k, replace=False)
            else:
                idx = np.arange(len(nm))
            keep_nm.append(nm[idx].astype(np.float32))
            keep_nodes.append(nodes[idx].astype(np.int16))

        sub_nm = np.concatenate(keep_nm) if keep_nm else np.zeros(0, np.float32)
        sub_nodes = np.concatenate(keep_nodes) if keep_nodes else np.zeros(0, np.int16)
        rows = np.array(rows, dtype=np.float64)

        out[f"{cfg}__nm_hist"] = nm_hist
        out[f"{cfg}__node_hist"] = node_hist
        out[f"{cfg}__sample_nm"] = sub_nm
        out[f"{cfg}__sample_nodes"] = sub_nodes
        out[f"{cfg}__n_paths"] = np.array([total], np.int64)
        out[f"{cfg}__mean_nm"] = np.array([s_nm / max(total, 1)])
        out[f"{cfg}__cell_root_id"] = np.array(row_ids, np.int64)
        out[f"{cfg}__cell_n_paths"] = rows[:, 0].astype(np.int64)
        out[f"{cfg}__cell_median_nm"] = rows[:, 1]
        out[f"{cfg}__cell_median_nodes"] = rows[:, 2]
        out[f"{cfg}__cell_max_nm"] = rows[:, 3]
        out[f"{cfg}__cell_mean_ppn"] = rows[:, 4]
        out[f"{cfg}__cell_max_ppn"] = rows[:, 5]

        # exact percentiles from the histogram (bin-resolution 500 nm)
        cum = np.cumsum(nm_hist)
        pct = {}
        for q in (1, 5, 25, 50, 75, 90, 95, 99):
            i = int(np.searchsorted(cum, cum[-1] * q / 100))
            pct[q] = float(NM_EDGES[min(i + 1, len(NM_EDGES) - 1)])
        out[f"{cfg}__pct_q"] = np.array(sorted(pct))
        out[f"{cfg}__pct_nm"] = np.array([pct[q] for q in sorted(pct)])
        print(f"{cfg:>7}: {total:>12,} paths  median {pct[50]:>9,.0f} nm  "
              f"p90 {pct[90]:>9,.0f}  mean {s_nm/max(total,1):>9,.0f}  "
              f"sample {len(sub_nm):,}", flush=True)

    # --- neighbourhoods: the cable-budget unit -----------------------------
    for cfg in NCONFIGS:
        files = sorted((PATHS / "neighborhoods" / cfg).glob("*.npz"))
        if not files:
            continue
        cab_hist = np.zeros(len(NM_EDGES) - 1, np.int64)
        node_hist = np.zeros(len(NODE_EDGES) - 1, np.int64)
        keep_c, keep_n, keep_r = [], [], []
        total = 0
        s_c = 0.0
        row_ids, rows = [], []
        for f in files:
            with np.load(f) as d:
                rid = int(d["root_id"][0])
                cab = d["cable_nm"].astype(np.float64)
                nm_ = d["n_members"].astype(np.float64)
                rad = d["radius_nm"].astype(np.float64)
            if len(cab) == 0:
                continue
            cab_hist += np.histogram(cab, NM_EDGES)[0]
            node_hist += np.histogram(nm_, NODE_EDGES)[0]
            total += len(cab)
            s_c += float(cab.sum())
            row_ids.append(rid)
            rows.append((len(cab), float(np.median(cab)), float(np.median(nm_)),
                         float(np.median(rad)), float(cab.max())))
            k = max(1, int(SUBSAMPLE * len(cab) / 60_000_000))
            idx = rng.choice(len(cab), size=k, replace=False) if k < len(cab) \
                else np.arange(len(cab))
            keep_c.append(cab[idx].astype(np.float32))
            keep_n.append(nm_[idx].astype(np.int16))
            keep_r.append(rad[idx].astype(np.float32))
        rows = np.array(rows, np.float64)
        out[f"nb_{cfg}__cable_hist"] = cab_hist
        out[f"nb_{cfg}__node_hist"] = node_hist
        out[f"nb_{cfg}__sample_cable"] = np.concatenate(keep_c)
        out[f"nb_{cfg}__sample_nodes"] = np.concatenate(keep_n)
        out[f"nb_{cfg}__sample_radius"] = np.concatenate(keep_r)
        out[f"nb_{cfg}__n"] = np.array([total], np.int64)
        out[f"nb_{cfg}__mean_cable"] = np.array([s_c / max(total, 1)])
        out[f"nb_{cfg}__cell_root_id"] = np.array(row_ids, np.int64)
        out[f"nb_{cfg}__cell_n"] = rows[:, 0].astype(np.int64)
        out[f"nb_{cfg}__cell_median_cable"] = rows[:, 1]
        out[f"nb_{cfg}__cell_median_nodes"] = rows[:, 2]
        out[f"nb_{cfg}__cell_median_radius"] = rows[:, 3]
        cum = np.cumsum(cab_hist)
        pct = {}
        for q in (1, 5, 25, 50, 75, 90, 95, 99):
            i = int(np.searchsorted(cum, cum[-1] * q / 100))
            pct[q] = float(NM_EDGES[min(i + 1, len(NM_EDGES) - 1)])
        out[f"nb_{cfg}__pct_q"] = np.array(sorted(pct))
        out[f"nb_{cfg}__pct_cable"] = np.array([pct[q] for q in sorted(pct)])
        print(f"nb {cfg:>10}: {total:>12,} units  cable median {pct[50]:>9,.0f} nm  "
              f"p90 {pct[90]:>9,.0f}  mean {s_c/max(total,1):>9,.0f}", flush=True)

    out["configs"] = np.array(CONFIGS)
    out["nconfigs"] = np.array(NCONFIGS)
    out["nm_edges"] = NM_EDGES
    out["node_edges"] = NODE_EDGES
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"\nwrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
