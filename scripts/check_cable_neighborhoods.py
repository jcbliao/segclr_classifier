"""Prove the cable-budget neighbourhood growth correct on real cells.

Checked against an independent brute-force Dijkstra sharing no code with the
kernel, plus the invariants that would otherwise fail silently: connectivity,
the centre being present, cable never exceeding the budget, and cable equalling
the induced subgraph's edge sum (which on a forest is n-1 edges exactly).

    sbatch scripts/sbatch/check_cable_neighborhoods.sh
"""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.cable_neighborhoods import neighborhoods  # noqa: E402
from data.soma_restrict import DEFAULT_SOMA_RADIUS_NM, restrict  # noqa: E402

CONFIGS = [("20um", {"cable_nm": 20_000.0}), ("80um", {"cable_nm": 80_000.0}),
           ("10node", {"n_nodes": 10}), ("40node", {"n_nodes": 40})]
N_CELLS = 6


def brute(adj, src, cable_nm=None, n_nodes=None):
    """Nearest-first growth, written independently of the kernel."""
    seen = set()
    got, cable = [], 0.0
    h = [(0.0, 0.0, src)]
    while h:
        d, w, v = heapq.heappop(h)
        if v in seen:
            continue
        if got:
            if cable_nm is not None and cable + w > cable_nm:
                break
            if n_nodes is not None and len(got) >= n_nodes:
                break
        seen.add(v)
        got.append(v)
        cable += w
        for u, ew in adj[v]:
            if u not in seen:
                heapq.heappush(h, (d + ew, ew, u))
    return set(got), cable


def main() -> int:
    man = json.loads((ROOT / "data" / "manifest.json").read_text())["cells"]
    nuc = json.loads((ROOT / "data" / "nucleus_positions.json").read_text())["positions"]
    order = sorted((r for r in man if r in nuc),
                   key=lambda r: man[r]["n_nodes_covered"])
    picks = [order[int(i * (len(order) - 1) / (N_CELLS - 1))] for i in range(N_CELLS)]

    bad = 0
    for rid in picks:
        d = torch.load(ROOT / "data" / "graph_cache" / f"{rid}.pt", weights_only=False)
        r = restrict(d.pos.numpy(), d.edge_index.numpy(),
                     d.edge_attr.numpy().reshape(-1), tuple(nuc[rid]),
                     DEFAULT_SOMA_RADIUS_NM)
        offsets, neigh, wts = r["csr"]
        n = r["n_nodes_after"]
        if n < 3:
            continue
        adj = [[] for _ in range(n)]
        wof = {}
        for a, b, w in zip(r["edge_index"][0], r["edge_index"][1], r["edge_attr"]):
            adj[int(a)].append((int(b), float(w)))
            wof[(int(a), int(b))] = float(w)

        print(f"\n=== {rid} {man[rid]['cell_type']}  {n:,} nodes, "
              f"{r['n_components']} comp ===", flush=True)
        for tag, kw in CONFIGS:
            out = neighborhoods(offsets, neigh, wts, **kw)
            off, mem = out["offsets"], out["members"]
            probs = []
            for i in range(n):
                s = mem[off[i]:off[i + 1]]
                if len(s) == 0 or int(s[0]) != i:
                    probs.append(f"node {i}: centre not first"); continue
                ss = set(int(v) for v in s)
                if len(ss) != len(s):
                    probs.append(f"node {i}: duplicate member")
                # connected, and cable == induced edge sum (forest: n-1 edges)
                edges = sum(1 for a in ss for b, _ in adj[a] if b in ss) // 2
                if edges != len(ss) - 1:
                    probs.append(f"node {i}: {len(ss)} nodes but {edges} induced edges")
                tot = sum(w for a in ss for b, w in adj[a] if b in ss) / 2
                if abs(tot - float(out["cable_nm"][i])) > 1e-3 * max(tot, 1.0):
                    probs.append(f"node {i}: cable {out['cable_nm'][i]:.1f} != {tot:.1f}")
                if "cable_nm" in kw and out["cable_nm"][i] > kw["cable_nm"] + 1e-6:
                    probs.append(f"node {i}: cable over budget")
                if "n_nodes" in kw and len(ss) > kw["n_nodes"]:
                    probs.append(f"node {i}: {len(ss)} members over cap")
                if len(probs) > 8:
                    break
            mism = 0
            for i in range(min(n, 60)):
                want, wc = brute(adj, i, **kw)
                got = set(int(v) for v in mem[off[i]:off[i + 1]])
                # Relative tolerance: cable_nm is stored float32, so at ~20,000 nm
                # its resolution is ~0.002 and an absolute 1e-6 can never pass.
                if want != got or abs(wc - float(out["cable_nm"][i])) > 1e-4 * max(wc, 1.0):
                    mism += 1
            nm = out["n_members"]
            flag = "OK" if not probs and not mism else f"{len(probs)} bad, {mism} mismatch"
            print(f"  {tag:>7}: members med {np.median(nm):>5.1f} max {nm.max():>4d}  "
                  f"cable med {np.median(out['cable_nm']):>9,.0f} nm  "
                  f"radius med {np.median(out['radius_nm']):>8,.0f} nm  [{flag}]", flush=True)
            for pmsg in probs[:4]:
                print(f"      ! {pmsg}", flush=True)
            bad += len(probs) + mism
    print(f"\nTOTAL PROBLEMS: {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
