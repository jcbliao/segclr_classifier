"""Local-mesh-patch fetching for dendrite thickness -- deliberately NEVER
downloads a whole neuron mesh (full meshes run into the millions of faces /
hundreds of MB-GB each; across 2192 cells that would be a large, unnecessary
hit on a shared CAVE service and on this project's own storage). Per explicit
user direction: "At most, each point likely only needs the 3x3x3 L2 cache
window from cave."

Validated live (scripts/check_local_mesh_patch.py, job 19884777) against a
real cell: `cv.mesh.get(root_id, bounding_box=bbox)` on CloudVolume's graphene
mesh source restricts the fetch server-side to just the L2 mesh fragments
intersecting `bbox` -- a real local patch (5734 vertices / 11312 faces for a
~12um-wide box), not the whole neuron. No separate chunkedgraph.get_leaves()
query needed; CloudVolume's own manifest endpoint already does the spatial
restriction.

Coordinate-unit gotcha caught by that same live test (a first attempt without
this conversion got a 500 from the manifest endpoint): `Bbox(..., unit="nm")`
is local bookkeeping ONLY -- cloudvolume's graphene mesh source serializes
raw coordinate values into the manifest request with no automatic unit
conversion, and the server expects VOXEL coordinates regardless of what
`unit=` claims. Always call `.convert_units("vx", resolution=cv.resolution)`
before passing a bbox to `mesh.get()`. `cv.resolution` for minnie65's
segmentation source measured as [8, 8, 40] nm/voxel -- NOT the (4,4,40)
convention some other CAVE services (e.g. do_merge) use, so always read it
from the live CloudVolume instance rather than hardcoding it.

Throughput fix (2026-08-07, after a first real bulk-ingestion attempt on 2
cells measured ~250s/cell and ~3.6k "connection pool is full" warnings per
cell): traced to `cloudfiles.CloudFiles.__init__`'s default `num_threads=20`
(cloudvolume's mesh-fragment fetch path uses this, independent of
`CloudVolume(..., parallel=1)` -- that setting only governs volumetric chunk
downloads, not mesh fragments) fetching fragments with MORE concurrency than
`requests`' default `pool_maxsize=10` -- every burst past 10 discards and
reopens a connection, adding a real fresh TCP+TLS handshake's worth of
latency each time, not just log noise. `_raise_connection_pool_size` below
patches `requests.adapters.DEFAULT_POOLSIZE` up to comfortably cover that
before any CloudVolume/GCS client gets constructed. Combined with a larger
DEFAULT_BUCKET_SIZE_NM in build_dendrite_thickness.py (fewer, larger patches
-> fewer total manifest/fragment fetches) rather than tuning pool size alone.
"""

from __future__ import annotations

import numpy as np

# Margin (nm) added around a batch of skeleton vertices' own bounding box
# before fetching its local mesh patch -- must be large enough that every
# ray (up to dendrite_thickness.DEFAULT_MAX_RADIUS_NM=2000nm from its
# origin) still lands on real mesh wall inside the patch, not empty space
# past its edge.
DEFAULT_MARGIN_NM = 2500.0

_pool_size_raised = False


def _raise_connection_pool_size(size: int = 32) -> None:
    """Match requests' HTTP connection pool size to cloudfiles' actual fetch
    concurrency (num_threads=20 by default) -- see module docstring. Must run
    BEFORE the first CloudVolume/google-cloud-storage client is constructed
    (HTTPAdapter reads this at its own construction time, not per-request),
    so local_cloudvolume() calls this first. Idempotent."""
    global _pool_size_raised
    if _pool_size_raised:
        return
    import requests.adapters

    requests.adapters.DEFAULT_POOLSIZE = max(size, requests.adapters.DEFAULT_POOLSIZE)
    _pool_size_raised = True


def local_cloudvolume(client):
    """CloudVolume instance for the client's own segmentation source (the
    same graphene/PCG source the skeleton and dataset build already key off
    of), read-only, HTTPS. One instance is reused across many mesh.get()
    calls by the caller -- constructing it touches client.info once, not per
    patch.

    cache=False (CloudVolume's default) deliberately, NOT a shared on-disk
    fragment cache -- tried that (a real, considered idea: adjacent buckets'
    local patches overlap in their DEFAULT_MARGIN_NM border region, so a
    disk cache would dedupe repeat fetches there) and it hung the first real
    production run (job 19888001) with zero progress for 8+ minutes across
    every shard: SLURM array tasks run on DIFFERENT compute nodes, all
    pointed at the same cache directory on a shared network filesystem --
    cloudfiles' cache locking over NFS is a classic hang/deadlock source,
    and every prior validated run (which never hit this because it never
    used cache=) completed within minutes. Reverted rather than debugged
    further -- not worth the risk for a secondary throughput optimization
    when the connection-pool-size + bucket-size fixes already validated
    real, sufficient improvement on their own.
    """
    _raise_connection_pool_size()
    from cloudvolume import CloudVolume

    cv_path = client.info.segmentation_source()
    return CloudVolume(cv_path, use_https=True, progress=False)


def fetch_local_mesh_patch(cv, root_id: int, points_nm: np.ndarray, margin_nm: float = DEFAULT_MARGIN_NM):
    """Local mesh patch covering the bounding box of `points_nm` (+ margin),
    NOT the whole neuron -- see module docstring.

    Parameters
    ----------
    cv : cloudvolume.CloudVolume
        From local_cloudvolume(client), reused across calls.
    root_id : int
    points_nm : (N, 3) array
        The skeleton vertices this patch needs to cover (e.g. one spatial
        bucket's worth -- see data/build_dendrite_thickness.py's batching).
    margin_nm : float

    Returns
    -------
    meshparty.trimesh_io.Mesh (or trimesh.Trimesh) covering just this
    local region, or None if the fetch returned nothing (e.g. a mesh hole /
    ungenerated chunk at this location).
    """
    from cloudvolume.lib import Bbox

    points_nm = np.asarray(points_nm, dtype=np.float64)
    lo = points_nm.min(axis=0) - margin_nm
    hi = points_nm.max(axis=0) + margin_nm
    resolution = np.asarray(cv.resolution, dtype=np.float64)
    bbox_vx = Bbox(lo, hi, unit="nm").convert_units("vx", resolution=resolution)

    meshes = cv.mesh.get(root_id, bounding_box=bbox_vx, remove_duplicate_vertices=True)
    cv_mesh = meshes[root_id] if isinstance(meshes, dict) else meshes
    if cv_mesh is None or len(np.asarray(cv_mesh.vertices)) == 0:
        return None

    # cv.mesh.get() returns cloudvolume's own lightweight Mesh (plain
    # vertices/faces/normals, no ray-casting) -- data/dendrite_thickness.py's
    # first_hit_distances needs a real trimesh.Trimesh for its .ray
    # accessor (embree-backed, via the embreex install). Caught by the
    # first real ingestion attempt: 'Mesh' object has no attribute 'ray'.
    import trimesh

    return trimesh.Trimesh(
        vertices=np.asarray(cv_mesh.vertices), faces=np.asarray(cv_mesh.faces), process=False
    )
