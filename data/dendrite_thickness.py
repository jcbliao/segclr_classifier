"""Spine-corrected dendrite shaft radius from a mesh plus skeleton, by ray
casting. Ported from isebenius/E-I's `src/dendrite_thickness.py` (private
repo, fetched via `gh api` per explicit user direction, 2026-08-07) -- the
core estimator functions below (`skeleton_neighbors` through
`estimate_dendrite_radius`, plus the synthetic-geometry validator at the
bottom) take a generic mesh + skeleton and return a per-vertex radius, with
no CAVE-specific code inside them; the one deliberate change from the
original is documented inline at `estimate_dendrite_radius`'s `degree > 2`
line (branch-point detection no longer needs a real meshparty Skeleton
object). What's new for this project is everything after
`estimate_dendrite_radius` -- bridging segclr_db's own `Skeleton` dataclass
(data/cave_skeletons.py, data/build_dataset_from_store.py's exported
data/skeleton_cache/*.pkl) into the minimal object this estimator actually
needs (`skeleton_for_ray_casting`). Mesh fetching lives in the sibling
data/neuron_mesh.py, via this project's own CAVE client convention
(data/cave_skeletons.py's default_cave_config), not here -- this module is
the ray-casting math only.

Biological intent (from the original): the `radius` CAVE skeletons already
carry (segclr_db.results.Skeleton.radii, confirmed already populated on every
cached skeleton this project has -- scripts/check_skeleton_compartment_radius.py)
is derived from the mesh WITH spines attached, so it reports "shaft plus spine
heads" rather than the shaft that actually conducts current -- inflating
cross-sectional area and flattening apparent taper. The correction casts rays
outward from a skeleton vertex within the plane perpendicular to the local
skeleton tangent: a ring in that plane samples a 1-D set of directions rather
than the 2-D sphere, and a spine is near point-like along its own axis, so a
ray only meets a spine whose mouth happens to cut that exact plane. Almost
all spines are therefore missed by construction, and the median first-hit
distance approaches the true shaft radius.

Units are nanometers throughout, matching this project's existing convention
(data.pos, edge_attr, REL_POS_SCALE_NM all in nm -- see data/geodesic_window.py).

Alignment note (per explicit user direction -- "using the root ID and
position, we should be able to find a direct CAVE skeleton node match"):
`data/build_dataset_from_store.py`'s cached whole-cell Data already carries
`orig_node_ids`, the exact index of each covered node back into its source
Skeleton's vertex array -- data.pos[i] == skeleton.coords[data.orig_node_ids[i]]
by construction, since both come from the SAME Skeleton object (no separate
position-based re-matching needed for skeleton-to-skeleton alignment; see
build_dendrite_thickness.py). The only alignment actually still open is
mesh-to-skeleton (does the freshly-downloaded mesh's coordinate frame agree
with the skeleton's), which build_dendrite_thickness.py checks empirically
(miss_fraction / radius sanity) rather than assumed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Skeleton vertex spacing in minnie65 CAVE skeletons is ~1.8 um, so a tangent
# window has to span several um to contain enough vertices to fit a direction.
DEFAULT_TANGENT_WINDOW_NM = 8000.0
DEFAULT_MIN_NEIGHBORS = 2
DEFAULT_N_RAYS = 64
# Measured (by the original author) on a v1718 basal arbor: the median radius
# is converged to ~0.4% by 32 rays, but recentring is still creeping at 3
# passes (304, 310, 314, 316, 317 nm for 1, 2, 3, 5, 8 passes). 5 lands within
# ~0.2% of the asymptote for ~0.1s.
DEFAULT_N_PASSES = 5
# Beyond this a "hit" is almost certainly a mesh hole letting the ray escape
# into a neighbouring process, not the local shaft wall.
DEFAULT_MAX_RADIUS_NM = 2000.0


def skeleton_neighbors(skeleton):
    """Undirected adjacency of a skeleton in flat CSR-like form.

    Returned as sorted neighbour lists so a vertex's neighbours are
    `neighbor[start[v] : start[v] + degree[v]]`. Built by sorting both edge
    orientations, which keeps it vectorized rather than a per-vertex Python loop.

    Returns
    -------
    neighbor : (2 * n_edges,) int array
    start : (n_vertices,) int array
    degree : (n_vertices,) int array
    """
    edges = np.asarray(skeleton.edges, dtype=np.int64)
    n = len(np.asarray(skeleton.vertices))

    source = np.concatenate([edges[:, 0], edges[:, 1]])
    target = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(source, kind="stable")

    degree = np.bincount(source, minlength=n)
    start = np.concatenate([[0], np.cumsum(degree)[:-1]])
    return target[order], start, degree


def local_tangents(skeleton):
    """Local cable direction at each vertex, from its immediate neighbours.

    The perpendicular plane is only as good as this tangent: a tilted plane
    lets rays run along the shaft and overshoot, so radius error from tilt is
    strictly one-directional (always inflating).

    For an interior vertex (degree 2) with neighbours A and B, the tangent is
    the unit angle bisector of the two incident edges:
    `normalize(B - M) - normalize(A - M)`. That equalizes the absolute angles
    to the two edges (equivalently: maximizes the worse of the two alignment
    cosines), so unequal edge lengths do not tilt the plane toward the longer
    side -- unlike the raw chord `B - A`, which is length-weighted. For a tip
    (degree 1) we fall back to the one-sided difference against its single
    neighbour. Branch points (degree > 2) get no tangent, since there is no
    single cable direction there.

    Returns
    -------
    tangents : (n_vertices, 3) np.ndarray
        Unit tangent per vertex; NaN at branch points and where an edge was
        degenerate. Sign is arbitrary, which is fine for a ring of directions.
    straightness : (n_vertices,) np.ndarray
        Chord length ||B - A|| over summed edge length for degree-2 vertices,
        so 1.0 is perfectly straight cable and lower values mean the
        neighbours fold back. NaN at tips and branch points.
    """
    verts = np.asarray(skeleton.vertices, dtype=np.float64)
    neighbor, start, degree = skeleton_neighbors(skeleton)

    tangents = np.full((len(verts), 3), np.nan)
    straightness = np.full(len(verts), np.nan)

    interior = np.where(degree == 2)[0]
    if len(interior):
        a = neighbor[start[interior]]
        b = neighbor[start[interior] + 1]
        edge_a = verts[a] - verts[interior]
        edge_b = verts[b] - verts[interior]
        len_a = np.linalg.norm(edge_a, axis=1)
        len_b = np.linalg.norm(edge_b, axis=1)
        chord_len = np.linalg.norm(verts[b] - verts[a], axis=1)
        arc_len = len_a + len_b
        with np.errstate(divide="ignore", invalid="ignore"):
            straightness[interior] = chord_len / arc_len
            unit_a = edge_a / len_a[:, None]
            unit_b = edge_b / len_b[:, None]
            bisector = unit_b - unit_a
            bisector_len = np.linalg.norm(bisector, axis=1)
            tangents[interior] = bisector / bisector_len[:, None]

    tips = np.where(degree == 1)[0]
    if len(tips):
        edge = verts[tips] - verts[neighbor[start[tips]]]
        edge_len = np.linalg.norm(edge, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tangents[tips] = edge / edge_len[:, None]

    tangents[~np.isfinite(tangents).all(axis=1)] = np.nan
    return tangents, straightness


def perpendicular_ray_directions(tangents, n_rays=DEFAULT_N_RAYS, seed=0):
    """Build a ring of unit directions in the plane perpendicular to each tangent.

    This is the step that rejects spines: confining directions to a plane
    means a spine is only sampled if its mouth intersects that plane.

    A random phase is added per vertex because the in-plane basis is a
    deterministic function of the tangent, and we do not want ray angles to
    be systematically aligned with local mesh structure.

    Returns
    -------
    (n_vertices, n_rays, 3) np.ndarray
        Unit directions, all orthogonal to the corresponding tangent.
    """
    tangents = np.asarray(tangents, dtype=np.float64)
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    helper = np.zeros_like(tangents)
    helper[np.arange(len(tangents)), np.argmin(np.abs(tangents), axis=1)] = 1.0
    e1 = np.cross(tangents, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(tangents, e1)  # already unit: tangent and e1 are orthonormal

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(len(tangents), 1))
    angle = phase + 2.0 * np.pi * np.arange(n_rays)[None, :] / n_rays

    return (
        np.cos(angle)[:, :, None] * e1[:, None, :]
        + np.sin(angle)[:, :, None] * e2[:, None, :]
    )


def isotropic_ray_directions(n_vertices, n_rays=DEFAULT_N_RAYS, seed=0):
    """Uniformly random directions over the full sphere, for the spine
    contrast test in validate_on_synthetic_cylinder -- sampling the sphere
    DOES hit spines, so comparing against perpendicular_ray_directions on the
    same vertices isolates how much spine surface the perpendicular ring
    excludes."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_vertices, n_rays, 3))
    return directions / np.linalg.norm(directions, axis=2, keepdims=True)


def first_hit_distances(mesh, origins, directions, max_rays_per_call=2_000_000):
    """Distance from each ray origin to the first mesh surface it meets.

    Uses `mesh.ray.intersects_first` (a single embree query returning
    triangle indices), then solves the ray-plane intersection ourselves
    rather than `intersects_location` -- that path materializes
    `mesh.triangles`/`mesh.face_normals` (hundreds of MB on a multi-million-
    face neuron mesh), does a Python-level plane solve, and accumulates hits
    in a Python list. Doing the arithmetic here touches only the triangles
    actually hit and stays in float64 while embree works internally in float32.

    Returns
    -------
    (n_rays,) np.ndarray
        Distance in nm to the first hit, NaN where the ray missed.
    """
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    directions = np.ascontiguousarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    out = np.full(len(origins), np.nan)

    for start in range(0, len(origins), max_rays_per_call):
        block = slice(start, start + max_rays_per_call)
        org, dirs = origins[block], directions[block]

        tri = mesh.ray.intersects_first(org, dirs)
        hit = np.asarray(tri) >= 0
        if not hit.any():
            continue

        corners = verts[faces[np.asarray(tri)[hit]]]
        v0 = corners[:, 0]
        normal = np.cross(corners[:, 1] - v0, corners[:, 2] - v0)

        denom = np.einsum("ij,ij->i", dirs[hit], normal)
        numer = np.einsum("ij,ij->i", v0 - org[hit], normal)
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = numer / denom
        dist[~np.isfinite(dist)] = np.nan

        block_out = np.full(len(org), np.nan)
        block_out[hit] = dist
        out[block] = block_out

    return out


def ray_radius(
    mesh,
    origins_nm,
    directions,
    n_passes=DEFAULT_N_PASSES,
    max_radius_nm=DEFAULT_MAX_RADIUS_NM,
    min_radius_nm=0.0,
    max_rays_per_call=2_000_000,
):
    """Median first-hit distance per vertex, with iterative recentring.

    A skeleton vertex does not sit exactly on the medial axis, and for an
    off-axis origin the ring of chord lengths is biased upward. Each pass
    therefore moves the origin to the centroid of its own hit ring and
    re-casts. Because every direction lies in the perpendicular plane, that
    displacement is in-plane by construction, so the origin cannot drift
    along the cable.

    The upper tail is deliberately not trimmed (a genuinely elliptical shaft
    should keep its long axis); only hits beyond `max_radius_nm` are dropped,
    on the grounds that they are mesh-hole escapes rather than the local wall.
    """
    origins = np.array(origins_nm, dtype=np.float64, copy=True)
    directions = np.asarray(directions, dtype=np.float64)
    directions = directions / np.linalg.norm(directions, axis=2, keepdims=True)
    n_vertices, n_rays = directions.shape[:2]

    dirs_flat = np.ascontiguousarray(directions.reshape(-1, 3))
    center_shift = np.zeros(n_vertices)
    per_ray = None

    for pass_index in range(max(int(n_passes), 1)):
        org_flat = np.repeat(origins, n_rays, axis=0)
        per_ray = first_hit_distances(
            mesh, org_flat, dirs_flat, max_rays_per_call=max_rays_per_call
        ).reshape(n_vertices, n_rays)
        per_ray[(per_ray <= min_radius_nm) | (per_ray > max_radius_nm)] = np.nan

        if pass_index == max(int(n_passes), 1) - 1:
            break

        valid = np.isfinite(per_ray)
        displacement = (np.where(valid, per_ray, 0.0)[:, :, None] * directions).sum(
            axis=1
        ) / np.maximum(valid.sum(axis=1), 1)[:, None]
        origins = origins + displacement
        center_shift += np.linalg.norm(displacement, axis=1)

    valid = np.isfinite(per_ray)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        radius = np.nanmedian(per_ray, axis=1)
        q75, q25 = (
            np.nanpercentile(per_ray, 75, axis=1),
            np.nanpercentile(per_ray, 25, axis=1),
        )

    return {
        "radius_nm": radius,
        "per_ray_nm": per_ray,
        "iqr_nm": q75 - q25,
        "n_hits": valid.sum(axis=1),
        "miss_fraction": 1.0 - valid.sum(axis=1) / n_rays,
        "center_shift_nm": center_shift,
        "origins_nm": origins,
    }


def estimate_dendrite_radius(
    mesh,
    skeleton,
    vertex_mask=None,
    skeleton_radius=None,
    compartment=None,
    tangents=None,
    exclude_branch_points=True,
    n_rays=DEFAULT_N_RAYS,
    n_passes=DEFAULT_N_PASSES,
    max_radius_nm=DEFAULT_MAX_RADIUS_NM,
    seed=0,
    return_per_ray=False,
):
    """Spine-corrected radius for every eligible skeleton vertex of one neuron.

    Vertices are skipped where the tangent could not be fit and, by default,
    at branch points, where there is no single cable direction and the
    perpendicular plane is meaningless.

    Returns
    -------
    pandas.DataFrame indexed by skeleton vertex index, with columns
        radius_nm, skeleton_radius_nm, radius_ratio, n_hits, miss_fraction,
        iqr_nm, center_shift_nm, straightness, degree, compartment.
    (df, per_ray_nm) if return_per_ray.
    """
    default_tangents, straightness = local_tangents(skeleton)
    if tangents is None:
        tangents = default_tangents
    else:
        tangents = np.asarray(tangents, dtype=np.float64)
    _, _, degree = skeleton_neighbors(skeleton)

    eligible = np.isfinite(tangents).all(axis=1)
    if vertex_mask is not None:
        eligible &= np.asarray(vertex_mask, dtype=bool)
    if exclude_branch_points:
        # A branch point IS exactly a vertex with more than 2 neighbours --
        # using this project's own already-computed `degree` (from
        # skeleton_neighbors above) instead of the original's
        # `skeleton.branch_points` (a meshparty.skeleton.Skeleton-specific
        # property) means this function only ever touches .vertices/.edges,
        # so `skeleton` can be any object with those two attributes, not
        # necessarily a real meshparty Skeleton -- see
        # skeleton_for_ray_casting below, which exploits exactly that.
        is_branch = degree > 2
        eligible &= ~is_branch

    idx = np.where(eligible)[0]
    if len(idx) == 0:
        raise ValueError("No eligible vertices; check vertex_mask.")

    directions = perpendicular_ray_directions(tangents[idx], n_rays=n_rays, seed=seed)
    result = ray_radius(
        mesh,
        np.asarray(skeleton.vertices, dtype=np.float64)[idx],
        directions,
        n_passes=n_passes,
        max_radius_nm=max_radius_nm,
    )

    df = pd.DataFrame(
        {
            "radius_nm": result["radius_nm"],
            "n_hits": result["n_hits"],
            "miss_fraction": result["miss_fraction"],
            "iqr_nm": result["iqr_nm"],
            "center_shift_nm": result["center_shift_nm"],
            "straightness": straightness[idx],
            "degree": degree[idx],
        },
        index=pd.Index(idx, name="skeleton_index"),
    )
    if skeleton_radius is not None:
        df["skeleton_radius_nm"] = np.asarray(skeleton_radius, dtype=np.float64)[idx]
        df["radius_ratio"] = df["radius_nm"] / df["skeleton_radius_nm"]
    if compartment is not None:
        df["compartment"] = np.asarray(compartment)[idx]

    if return_per_ray:
        return df, result["per_ray_nm"]
    return df


# --------------------------------------------------------------------------- #
# This project's bridge from segclr_db's Skeleton to a ray-castable skeleton.
# Not in the original -- the original bridges from CAVE's own wire-format
# sk_dict via skeleton_dict_to_meshparty; this project already HAS the
# skeleton (data/skeleton_cache/*.pkl, exported by
# data/build_dataset_from_store.py), with compartments/radii confirmed
# already populated (scripts/check_skeleton_compartment_radius.py), so no
# live CAVE skeleton fetch is needed at all -- only the mesh is new.
# --------------------------------------------------------------------------- #

# Compartment codes segclr_db.results.Skeleton uses (matches the SWC/CAVE
# convention isebenius/E-I's skeleton_utils.py documents: SOMA=1, AXON=2,
# DENDRITE_BASAL=3, DENDRITE_APICAL=4). Confirmed present (1, 2, 3; no 4
# sampled) on this project's own cached skeletons.
COMPARTMENT_SOMA = 1
COMPARTMENT_AXON = 2
COMPARTMENT_DENDRITE_BASAL = 3
COMPARTMENT_DENDRITE_APICAL = 4
DENDRITE_COMPARTMENTS = (COMPARTMENT_DENDRITE_BASAL, COMPARTMENT_DENDRITE_APICAL)


class _RayCastSkeleton:
    """Minimal duck-typed stand-in for meshparty.skeleton.Skeleton -- every
    function in this module only ever reads `.vertices`/`.edges` (see the
    comment on `degree > 2` above for why `.branch_points` specifically is
    no longer needed), so building a real meshparty Skeleton per cell (with
    its own KD-tree/distance-to-root/segment bookkeeping this module never
    uses) would be pure overhead across a 2192-cell bulk run for no benefit.
    """

    __slots__ = ("vertices", "edges")

    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges


def skeleton_for_ray_casting(skeleton):
    """segclr_db.results.Skeleton -> the minimal object estimate_dendrite_radius
    actually needs (see _RayCastSkeleton). Kept as a real function rather than
    just handing callers `skeleton.coords`/`skeleton.edges` directly so this
    stays the one place that would need updating if that ever changes."""
    coords = np.asarray(skeleton.coords, dtype=np.float64)
    edges = np.asarray(skeleton.edges, dtype=np.int64)
    return _RayCastSkeleton(vertices=coords, edges=edges)


# --------------------------------------------------------------------------- #
# Verification against known geometry -- unchanged from the original, no CAVE
# or mesh-download dependency, so this runs standalone as a correctness check
# (scripts/smoke_test_dendrite_thickness.py) before any real ingestion.
# --------------------------------------------------------------------------- #


def spiny_cylinder_mesh(
    shaft_radius_nm=300.0,
    length_nm=40000.0,
    n_spines=60,
    spine_head_radius_nm=250.0,
    overlap_fraction=0.6,
    sections=64,
    seed=0,
):
    """Mesh with a known shaft radius and spine-like protrusions, for validation.

    Spine heads are spheres placed so they overlap the shaft wall; the
    boolean union then removes the wall inside each mouth, exactly as a real
    spine neck opens into the shaft lumen.
    """
    import trimesh

    parts = [
        trimesh.creation.cylinder(radius=shaft_radius_nm, height=length_nm, sections=sections)
    ]

    rng = np.random.default_rng(seed)
    centre_offset = shaft_radius_nm + overlap_fraction * spine_head_radius_nm
    z = rng.uniform(-0.4 * length_nm, 0.4 * length_nm, size=n_spines)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_spines)
    for zi, ti in zip(z, theta):
        head = trimesh.creation.icosphere(subdivisions=2, radius=spine_head_radius_nm)
        head.apply_translation([centre_offset * np.cos(ti), centre_offset * np.sin(ti), zi])
        parts.append(head)

    mesh = trimesh.boolean.union(parts) if n_spines else parts[0]
    truth = {
        "shaft_radius_nm": shaft_radius_nm,
        "spine_tip_radius_nm": centre_offset + spine_head_radius_nm,
        "n_spines": int(n_spines),
    }
    return mesh, truth


def _axis_skeleton(length_nm, spacing_nm, offset_nm=0.0, rotation=None):
    """Straight skeleton along z, matching the sparse spacing of CAVE
    skeletons. `offset_nm` displaces the skeleton off the true axis, which is
    what makes the recentring step observable."""
    from meshparty import skeleton as mp_skeleton

    z = np.arange(-0.45 * length_nm, 0.45 * length_nm, spacing_nm)
    verts = np.column_stack([np.full(len(z), offset_nm), np.zeros(len(z)), z])
    if rotation is not None:
        verts = verts @ np.asarray(rotation, dtype=np.float64).T
    edges = np.column_stack([np.arange(len(z) - 1), np.arange(1, len(z))])
    return mp_skeleton.Skeleton(vertices=verts, edges=edges, root=0, voxel_scaling=None)


def validate_on_synthetic_cylinder(
    shaft_radius_nm=300.0,
    length_nm=40000.0,
    spacing_nm=1850.0,
    skeleton_offset_nm=60.0,
    n_spines=60,
    spine_head_radius_nm=250.0,
    n_rays=DEFAULT_N_RAYS,
    n_passes=DEFAULT_N_PASSES,
    tilt_deg=20.0,
    seed=0,
):
    """Check the estimator against geometry whose answer is known in advance.

    Three things are tested at once. On a bare cylinder the perpendicular
    ring should return `shaft_radius_nm` regardless of the skeleton sitting
    off-axis and the cable being tilted relative to the coordinate axes. On
    the spiny cylinder it should still return the shaft radius, because the
    ring misses spines. And an isotropic sampler on that same spiny mesh
    should come out clearly inflated, which is the effect being corrected for.
    """
    angle = np.deg2rad(tilt_deg)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )

    smooth, truth = spiny_cylinder_mesh(
        shaft_radius_nm=shaft_radius_nm, length_nm=length_nm, n_spines=0, seed=seed
    )
    spiny, _ = spiny_cylinder_mesh(
        shaft_radius_nm=shaft_radius_nm,
        length_nm=length_nm,
        n_spines=n_spines,
        spine_head_radius_nm=spine_head_radius_nm,
        seed=seed,
    )
    for mesh in (smooth, spiny):
        mesh.apply_transform(
            np.vstack([np.hstack([rotation, np.zeros((3, 1))]), [0, 0, 0, 1]])
        )

    sk = _axis_skeleton(length_nm, spacing_nm, offset_nm=skeleton_offset_nm, rotation=rotation)
    max_radius_nm = 4.0 * shaft_radius_nm

    out = {
        "truth_shaft_radius_nm": shaft_radius_nm,
        "truth_spine_tip_radius_nm": truth["spine_tip_radius_nm"] if n_spines else np.nan,
        "n_measured_vertices": 0,
        "tilt_deg": tilt_deg,
        "skeleton_offset_nm": skeleton_offset_nm,
    }

    for label, mesh in (("smooth", smooth), ("spiny", spiny)):
        df = estimate_dendrite_radius(
            mesh,
            sk,
            skeleton_radius=None,
            exclude_branch_points=False,
            n_rays=n_rays,
            n_passes=n_passes,
            max_radius_nm=max_radius_nm,
            seed=seed,
        )
        out[f"{label}_perpendicular_median_nm"] = float(df["radius_nm"].median())
        out[f"{label}_perpendicular_bias"] = float(df["radius_nm"].median()) / shaft_radius_nm - 1.0
        out[f"{label}_miss_fraction"] = float(df["miss_fraction"].mean())
        out["n_measured_vertices"] = int(len(df))

    tangents, _ = local_tangents(sk)
    keep = np.isfinite(tangents).all(axis=1)
    iso = ray_radius(
        spiny,
        np.asarray(sk.vertices)[keep],
        isotropic_ray_directions(int(keep.sum()), n_rays=n_rays, seed=seed),
        n_passes=1,
        max_radius_nm=max_radius_nm,
    )
    out["spiny_isotropic_median_nm"] = float(np.nanmedian(iso["radius_nm"]))
    out["spiny_isotropic_bias"] = float(np.nanmedian(iso["radius_nm"])) / shaft_radius_nm - 1.0
    return out
