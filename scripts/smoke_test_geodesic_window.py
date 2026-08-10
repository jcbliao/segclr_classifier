"""Correctness smoke test for data/geodesic_window.py's numba kernel against
a hand-computable toy graph, before trusting it on the real 2442-cell
dataset. Run via sbatch (mit_quicktest -- tiny, CPU only, numba JIT compile
is the only real cost).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data.geodesic_window import window_membership  # noqa: E402


def chain_edges(n: int, spacing: float):
    """0 -- 1 -- 2 -- ... -- (n-1), each edge `spacing` nm, symmetrized."""
    src = np.arange(n - 1)
    dst = np.arange(1, n)
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_attr = np.full((2 * (n - 1), 1), spacing, dtype=np.float64)
    return edge_index, edge_attr


def main() -> int:
    # Chain of 7 nodes, 1000nm apart: 0-1-2-3-4-5-6. Window=2500nm from node 3
    # should reach nodes 1..5 (within 2000nm) but not 0 or 6 (3000nm away).
    n = 7
    edge_index, edge_attr = chain_edges(n, spacing=1000.0)
    mem_offsets, members = window_membership(edge_index, edge_attr, n, window_nm=2500.0)

    for center in range(n):
        lo, hi = mem_offsets[center], mem_offsets[center + 1]
        got = sorted(members[lo:hi].tolist())
        expected = sorted(i for i in range(n) if abs(i - center) * 1000.0 <= 2500.0)
        status = "OK" if got == expected else "MISMATCH"
        print(f"center={center}: got={got} expected={expected}  {status}", flush=True)
        assert got == expected, f"center {center}: got {got}, expected {expected}"
    print("chain test passed.", flush=True)

    # Star graph: center 0 connected to 1..5, each spoke a different length.
    # Window=50 from node 0 should include exactly the spokes <=50nm away.
    n = 6
    lengths = [10.0, 20.0, 30.0, 60.0, 100.0]
    src = np.array([0, 0, 0, 0, 0])
    dst = np.array([1, 2, 3, 4, 5])
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    edge_attr = np.array(lengths + lengths, dtype=np.float64).reshape(-1, 1)
    mem_offsets, members = window_membership(edge_index, edge_attr, n, window_nm=50.0)
    lo, hi = mem_offsets[0], mem_offsets[1]
    got = sorted(members[lo:hi].tolist())
    expected = [0, 1, 2, 3]  # 10, 20, 30 <= 50; 60, 100 > 50
    print(f"star center=0: got={got} expected={expected}", flush=True)
    assert got == expected
    print("star test passed.", flush=True)

    # Symmetry sanity check on the chain: window membership should be
    # mutual for a uniform-weight graph (if b is in a's window, a is in b's).
    edge_index, edge_attr = chain_edges(10, spacing=500.0)
    mem_offsets, members = window_membership(edge_index, edge_attr, 10, window_nm=1200.0)
    membership_sets = [
        set(members[mem_offsets[i]:mem_offsets[i + 1]].tolist()) for i in range(10)
    ]
    for a in range(10):
        for b in membership_sets[a]:
            assert a in membership_sets[b], f"asymmetric: {b} in window({a}) but {a} not in window({b})"
    print("symmetry test passed.", flush=True)

    print("\nall geodesic_window smoke tests passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
