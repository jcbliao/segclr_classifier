"""Prove the centred-path enumeration correct on real cells before building.

Every property here is one the pipeline could otherwise get wrong silently: a
path that skips an edge, a centre in the wrong slot, an arm over its budget, a
route that revisits a node, a path straddling two components that only ever met
at the soma, or an emitted set that disagrees with the counter that sized it.

It also checks the enumeration is *complete* on small graphs, against an
independent brute-force search that shares no code with the kernel.

    sbatch scripts/sbatch/check_embedding_paths.sh
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.store_compat import use_v4  # noqa: E402

use_v4()

from data.embedding_paths import assert_forest, centered_paths, count_paths  # noqa: E402
from data.soma_restrict import (  # noqa: E402
    DEFAULT_SOMA_RADIUS_NM, nucleus_positions, restrict,
)

STORE_ROOT = "/orcd/compute/sdorkenw/001/segclr-db"
N_CELLS = 8

CONFIGS = [("10um", {"half_nm": 5_000.0}), ("40um", {"half_nm": 20_000.0}),
           ("10node", {"half_nodes": 5}), ("40node", {"half_nodes": 20})]


def brute_force(adj, wt, i, half_nm=None, half_nodes=None):
    """Every centred path through i, found independently of the kernel."""
    def arms(first):
        out, stack = [], [([first], wt[(i, first)])]
        while stack:
            path, acc = stack.pop()
            ext = False
            for v in sorted(adj[path[-1]]):
                if len(path) > 1 and v == path[-2]:
                    continue
                if len(path) == 1 and v == i:
                    continue
                w = wt[(path[-1], v)]
                if half_nm is not None and acc + w > half_nm:
                    continue
                if half_nodes is not None and len(path) + 1 > half_nodes:
                    continue
                stack.append((path + [v], acc + w))
                ext = True
            if not ext:
                out.append((tuple(path), acc))
        return out

    nb = sorted(adj[i])
    per_edge = []
    for v in nb:
        w = wt[(i, v)]
        if half_nm is not None and w > half_nm:
            per_edge.append([])
        elif half_nodes is not None and half_nodes < 1:
            per_edge.append([])
        else:
            per_edge.append(arms(v))
    got = set()
    for a, b in combinations(range(len(nb)), 2):
        for pa, _ in per_edge[a]:
            for pb, _ in per_edge[b]:
                got.add(tuple(reversed(pa)) + (i,) + pb)
    if not got:
        one = {(i,) + p for lst in per_edge for p, _ in lst}
        got = one or {(i,)}
    return got


def main() -> int:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    cells = manifest["cells"]
    ids = [int(r) for r in cells]
    nuc = nucleus_positions(STORE_ROOT, root_ids=ids)
    have = [r for r in ids if r in nuc]
    order = sorted(have, key=lambda r: cells[str(r)]["n_nodes_covered"])
    picks = [order[int(i * (len(order) - 1) / (N_CELLS - 1))] for i in range(N_CELLS)]

    total_problems = 0
    for rid in picks:
        d = torch.load(ROOT / "data" / "graph_cache" / f"{rid}.pt", weights_only=False)
        r = restrict(d.pos.numpy(), d.edge_index.numpy(),
                     d.edge_attr.numpy().reshape(-1), nuc[rid], DEFAULT_SOMA_RADIUS_NM)
        offsets, neighbors, weights = r["csr"]
        n = r["n_nodes_after"]
        if n < 2:
            continue
        assert_forest(offsets, neighbors)

        adj = [set() for _ in range(n)]
        wt = {}
        for a, b, w in zip(r["edge_index"][0], r["edge_index"][1], r["edge_attr"]):
            adj[int(a)].add(int(b))
            wt[(int(a), int(b))] = float(w)

        print(f"\n=== {rid} {cells[str(rid)]['cell_type']}  {n:,} nodes, "
              f"{r['n_components']} components ===", flush=True)

        for tag, kw in CONFIGS:
            per_node, arms = count_paths(offsets, neighbors, weights, **kw)
            out = centered_paths(offsets, neighbors, weights, per_node=per_node,
                                 arms=arms, **kw)
            off, nodes = out["path_offsets"], out["path_nodes"]
            ca, nm = out["center_at"], out["geodesic_nm"]
            centre = np.repeat(np.arange(n), per_node)
            problems = []

            for k in range(len(off) - 1):
                p = nodes[off[k]:off[k + 1]]
                c = int(ca[k])
                if not (0 <= c < len(p)) or int(p[c]) != centre[k]:
                    problems.append(f"path {k}: centre slot wrong"); continue
                if len(set(p.tolist())) != len(p):
                    problems.append(f"path {k}: revisits a node")
                tot = 0.0
                bad = False
                for a, b in zip(p[:-1], p[1:]):
                    if int(b) not in adj[int(a)]:
                        problems.append(f"path {k}: {a}->{b} not an edge"); bad = True; break
                    tot += wt[(int(a), int(b))]
                if not bad and abs(tot - float(nm[k])) > 1e-3 * max(tot, 1.0):
                    problems.append(f"path {k}: nm {nm[k]:.1f} != measured {tot:.1f}")
                if len({int(r['component'][int(v)]) for v in p}) != 1:
                    problems.append(f"path {k}: spans >1 component")
                if "half_nodes" in kw:
                    if len(p) > 2 * kw["half_nodes"] + 1:
                        problems.append(f"path {k}: {len(p)} nodes over cap")
                else:
                    for arm in (p[:c + 1][::-1], p[c:]):
                        s = sum(wt[(int(a), int(b))] for a, b in zip(arm[:-1], arm[1:]))
                        if s > kw["half_nm"] + 1e-6:
                            problems.append(f"path {k}: arm {s:.0f} nm over budget"); break
                if len(problems) > 20:
                    break

            # completeness, on the first few nodes only (brute force is exponential)
            mism = 0
            for i in range(min(n, 40)):
                want = brute_force(adj, wt, i, **kw)
                lo = int(off[np.searchsorted(centre, i)])
                sel = np.flatnonzero(centre == i)
                got = {tuple(int(v) for v in nodes[off[k]:off[k + 1]]) for k in sel}
                got |= {tuple(reversed(t)) for t in got}
                if not want <= got:
                    mism += 1
            lens = np.diff(off)
            flag = "OK" if not problems and not mism else f"{len(problems)} bad, {mism} incomplete"
            print(f"  {tag:>7}: {len(off)-1:>9,} paths  nodes/path med "
                  f"{np.median(lens):>5.1f} max {lens.max():>4d}  geodesic med "
                  f"{np.median(nm):>9,.0f} nm max {nm.max():>9,.0f}  [{flag}]", flush=True)
            for pr in problems[:5]:
                print(f"      ! {pr}", flush=True)
            total_problems += len(problems) + mism

    print(f"\nTOTAL PROBLEMS: {total_problems}", flush=True)
    return 1 if total_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
