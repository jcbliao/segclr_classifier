"""One-off check: does CloudVolume's graphene mesh source's `bounding_box=`
argument actually restrict a mesh fetch to a small LOCAL patch (not the whole
neuron), per explicit user direction to never download a full neuron mesh --
"At most, each point likely only needs the 3x3x3 L2 cache window from cave."

Fetches a small nm-scale bounding box around ONE real skeleton vertex from an
already-cached cell and reports the returned mesh's vertex count and spatial
extent, so "is this actually local" is answered by real numbers rather than
inferred from reading cloudvolume's source.

Run via sbatch (mit_quicktest -- needs CAVE_TOKEN, no GPU):
    sbatch scripts/sbatch/check_local_mesh_patch.sh
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "segclr_db" / "src"))

import numpy as np  # noqa: E402

from data import cave_skeletons as cs  # noqa: E402

# Half-width of the local bbox around the query point, in nm -- generous
# margin over dendrite_thickness.py's DEFAULT_MAX_RADIUS_NM (2000nm) so rays
# have real mesh wall to hit in every direction.
HALF_WIDTH_NM = 6000.0


def main() -> int:
    import os

    from cloudvolume import CloudVolume
    from cloudvolume.lib import Bbox

    token = os.environ.get("CAVE_TOKEN")
    if not token:
        print("CAVE_TOKEN not set")
        return 2

    pkl_path = sorted(cs.CACHE_DIR.glob("*.pkl"))[0]
    with open(pkl_path, "rb") as f:
        skel = pickle.load(f)
    root_id = skel.root_id
    point_nm = np.asarray(skel.coords[len(skel.coords) // 2], dtype=np.float64)
    print(f"cell {root_id}: {len(skel.coords)} skeleton vertices, query point (nm)={point_nm.tolist()}")

    cave_config = cs.default_cave_config(token)
    client = cave_config.build_client()
    cv_path = client.info.segmentation_source()
    print(f"segmentation_source: {cv_path}")

    cv = CloudVolume(cv_path, use_https=True, progress=False)
    resolution = np.asarray(cv.resolution, dtype=np.float64)
    print(f"cv.mip resolution (nm/voxel): {resolution.tolist()}")

    # Bbox's unit= kwarg is local bookkeeping only -- fetch_manifest_remote
    # (cloudvolume/datasource/graphene/mesh/unsharded.py) serializes raw
    # coordinate VALUES into the request with no automatic unit conversion,
    # and the server expects voxel coordinates regardless of what unit=
    # claims. Convert explicitly rather than trusting Bbox to do it (this is
    # exactly what a first live test caught: nm values sent as if they were
    # voxels produced a 500 from the manifest endpoint).
    bbox_nm = Bbox(point_nm - HALF_WIDTH_NM, point_nm + HALF_WIDTH_NM, unit="nm")
    bbox_vx = bbox_nm.convert_units("vx", resolution=resolution)
    print(f"requesting bounding_box={bbox_vx} (unit={bbox_vx.unit}, converted from {bbox_nm})")

    meshes = cv.mesh.get(root_id, bounding_box=bbox_vx, remove_duplicate_vertices=True)
    mesh = meshes[root_id] if isinstance(meshes, dict) else meshes
    verts = np.asarray(mesh.vertices)
    extent = verts.max(axis=0) - verts.min(axis=0) if len(verts) else None
    print(
        f"local patch: {len(verts)} vertices, {len(mesh.faces)} faces, "
        f"extent (nm)={extent.tolist() if extent is not None else None}"
    )
    if extent is not None and np.any(extent > 10 * HALF_WIDTH_NM):
        print("WARNING: extent is much larger than the requested bbox -- bounding_box may not be restricting the fetch")
    else:
        print("OK: patch extent is consistent with a LOCAL fetch, not a whole-neuron download")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
