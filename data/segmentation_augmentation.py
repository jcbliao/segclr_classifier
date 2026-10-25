"""Batched binary morphology using valid neighborhoods and explicit halos."""

from __future__ import annotations

import torch
import torch.nn.functional as F

TRAINING_POLICIES = tuple(
    f"segmask_{direction}{radius}"
    for radius in (1, 2, 3)
    for direction in ("shrink", "expand", "both")
)


def training_choice_indices(policy, labels):
    """Uniform per-node choice among clean and cumulative views up to radius."""
    if policy not in TRAINING_POLICIES:
        raise ValueError(f"unknown segmentation training policy: {policy}")
    radius = int(policy[-1])
    direction = policy[len("segmask_") : -1]
    labels = list(labels)
    names = ["clean"]
    for layer in range(1, radius + 1):
        if direction in ("shrink", "both"):
            names.append(f"erode{layer}")
        if direction in ("expand", "both"):
            names.append(f"dilate{layer}")
    return [labels.index(name) for name in names]


def neighbor_step(mask: torch.Tensor, expand: bool, neighbors: int = 6):
    """Simultaneous update; consume one halo voxel on each spatial face."""
    if mask.dtype != torch.bool or mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("mask must be bool [B,1,Z,Y,X]")
    if min(mask.shape[-3:]) < 3:
        raise ValueError("one step requires at least three voxels per spatial axis")
    if neighbors == 26:
        if expand:
            return F.max_pool3d(mask.float(), 3, stride=1).bool()
        return ~F.max_pool3d((~mask).float(), 3, stride=1).bool()
    if neighbors != 6:
        raise ValueError("neighbors must be 6 or 26")
    out = mask[:, :, 1:-1, 1:-1, 1:-1].clone()
    for axis in (2, 3, 4):
        for side in (slice(None, -2), slice(2, None)):
            selection = [
                slice(None),
                slice(None),
                slice(1, -1),
                slice(1, -1),
                slice(1, -1),
            ]
            selection[axis] = side
            if expand:
                out.logical_or_(mask[tuple(selection)])
            else:
                out.logical_and_(mask[tuple(selection)])
    return out


def center_crop(mask, size):
    starts = [(edge - size) // 2 for edge in mask.shape[-3:]]
    if size < 1 or any(start < 0 for start in starts):
        raise ValueError("invalid center crop")
    return mask[
        :,
        :,
        starts[0] : starts[0] + size,
        starts[1] : starts[1] + size,
        starts[2] : starts[2] + size,
    ]


def mask_variants(
    mask, radius=3, neighbors=6, crop_size=129, valid=None, step=neighbor_step
):
    """Yield baseline and +/-radius views, reusing each preceding layer.

    Input must include radius halo voxels. Out-of-volume voxels are zero
    background; ``valid`` excludes padded voxels from all final views.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if any(edge != crop_size + 2 * radius for edge in mask.shape[-3:]):
        raise ValueError("mask shape does not include the requested halo")
    validity = True if valid is None else center_crop(valid, crop_size)
    yield "clean", center_crop(mask, crop_size) & validity
    eroded = dilated = mask
    for layer in range(1, radius + 1):
        eroded = step(eroded, False, neighbors)
        dilated = step(dilated, True, neighbors)
        yield f"erode{layer}", center_crop(eroded, crop_size) & validity
        yield f"dilate{layer}", center_crop(dilated, crop_size) & validity


def variant_names(radius):
    return [
        "clean",
        *[name for r in range(1, radius + 1) for name in (f"erode{r}", f"dilate{r}")],
    ]
