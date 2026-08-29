"""Per-node segmentation mask volume: how many voxels of the SegCLR input were the cell.

For every node, SegCLR was fed one ``129^3`` crop centered on that node, with the EM
zeroed everywhere the segmentation was not this ``root_id``. This module recovers a
single scalar per node -- the number of voxels that survived that mask -- which is the
occupancy of the cell inside the exact box the embedding was computed from.

**Nothing is stored but the count.** A crop is read, reduced to an integer, and
discarded; the mask itself is never written to disk. Storing it would be 129^3 bytes
per node, and at 12.13M nodes that is ~25 TiB (~3 TiB bit-packed) against a 1 TB
scratch quota. The count is 4 bytes.

**Nothing is downloaded either.** The segmentation is already on ORCD disk as a sharded
precomputed volume and is read locally through TensorStore. There is no CAVE call and no
network I/O on this path.

Window geometry
---------------
Reproduced exactly from the inference run that produced the ``resnet_860b_reshuffled``
embeddings now in the store (``~/projects/segclr/src/inference/inference.py`` and
``src/data/crops.py``)::

    resolution = (32, 32, 40) nm/voxel     # the 1718 segmentation's only scale
    center_vox = trunc(coords_nm / resolution)
    start_vox  = center_vox - 129 // 2
    end_vox    = start_vox + 129           # half-open
    mask       = seg[start:end] == root_id

Three details are load-bearing and are why this module reuses ``CropLoader`` rather than
reimplementing the read:

1. **The center truncates, it does not round.** ``crops.py::nm_to_voxel`` documents that
   the model was trained on truncated centers, and that rounding instead shifts most
   crops by up to a voxel per axis -- measured at cos 0.99 mean / 0.82 min against
   embeddings computed the old way. A count taken from a rounded box is a count of a
   different box than the embedding saw.
2. **The materialization version selects the segmentation volume.** A ``root_id`` names a
   state of a cell, so a 1718 id masked against the 1300 segmentation finds nothing or
   finds a different cell, silently. 1718 is what this project's manifest records.
3. **Out-of-bounds reads zero-pad** (``oob="pad"``), matching inference. A window
   overhanging the dataset boundary really was fed to the model with zeros there, so
   padding is the faithful reproduction -- but the resulting count is not comparable to
   an interior node's, which is what :func:`clipped_flags` exists to mark.

Counts are exact voxel counts; multiply by :data:`VOXEL_VOLUME_NM3` for cubic nanometres.
"""

from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

#: The one crop-loading implementation, shared by that project's training and inference.
#: Its module docstring records a masking bug that produced garbage embeddings for months
#: while every pipeline involved kept running; re-deriving the read here would recreate
#: exactly the divergence that having a single implementation prevents.
#:
#: Loaded by file path rather than imported as ``segclr.data.crops``, because the segclr
#: package is installed only in ``~/.conda/envs/segclr`` and this project runs from its
#: own uv venv. ``crops.py`` imports nothing from its own package -- only json, os, numpy,
#: torch, and a lazy tensorstore -- so it loads standalone, and a path import keeps
#: ``sys.path`` clean and the failure legible if the file ever moves.
CROPS_PATH = Path("~/projects/segclr/src/data/crops.py").expanduser()


def _load_crops_module():
    if not CROPS_PATH.exists():
        raise ImportError(
            f"cannot find the segclr crop loader at {CROPS_PATH}. This module reuses it "
            "so the counted box is provably the embedded box; point CROPS_PATH at it "
            "rather than reimplementing the read."
        )
    spec = importlib.util.spec_from_file_location("segclr_crops", CROPS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_crops = _load_crops_module()
CropLoader = _crops.CropLoader


#: Model input edge in voxels. A property of the trained architecture, identical across
#: every checkpoint in this project.
BOX_SIZE = 129

#: nm per voxel of the 1718 segmentation's only scale, checked against its ``info``
#: rather than assumed. Asserted at runtime in :meth:`MaskVolumeCounter.resolution`.
EXPECTED_RESOLUTION_NM = (32.0, 32.0, 40.0)

#: One voxel in cubic nanometres: 32 * 32 * 40.
VOXEL_VOLUME_NM3 = 32 * 32 * 40

#: Which segmentation each materialization's root_ids must be masked against, and the
#: scale index that reaches 32x32x40 nm in that volume. The 1300 volume is a full pyramid
#: whose 32 nm level is index 2; the 1718 one stores that resolution alone at index 0.
SEGMENTATION: dict[int, tuple[str, int]] = {
    1300: ("precomputed://file:///orcd/compute/sdorkenw/001/collina/minnie_seg_fullvolume", 2),
    1718: ("precomputed://file:///orcd/compute/sdorkenw/001/collina/minnie_seg_1718_sharded", 0),
}

DEFAULT_MAT_VERSION = 1718

#: TensorStore chunk cache per process, matching the value ``crops.py`` was benchmarked
#: at. **Do not raise this on the theory that there is spare host memory.** Raising it to
#: 4 GiB drove resident memory to 32 GiB and killed 26 of 32 shards with OOM: the pool
#: limit bounds cached chunk data, not the chunks pinned by in-flight reads, and with 16
#: threads each touching ~20 chunks of a 129^3 window the two compound. The 500 MB default
#: is what the embedding pipeline runs this same read path with, for 11 hours at a time.
DEFAULT_CACHE_BYTES = 500 * 1024 * 1024  # 500 MB

#: Reads are I/O plus gzip decompression and release the GIL, so threads scale. The
#: inference pipeline measured throughput falling past 32; 16 was its optimum with EM
#: reads competing, and this path reads only the segmentation.
DEFAULT_NUM_THREADS = 16


def morton_order(centers: np.ndarray) -> np.ndarray:
    """Indices sorting nodes along a Z-order curve over 64-voxel cells.

    Consecutive skeleton nodes sit 1-2 um apart while a crop spans ~4 um, so neighbouring
    crops overlap heavily. Visiting them in spatial order lets TensorStore's chunk cache
    serve the overlap instead of re-reading it -- the inference pipeline measured this
    worth 23% over the skeleton's own node order and 70% over random.

    Ported from ``segclr.inference.morton_order`` so this module does not depend on the
    inference class, which would drag in torch model loading and a GPU device pick.
    """

    def spread(values: np.ndarray) -> np.ndarray:
        v = values.astype(np.int64) & 0x1FFFFF
        v = (v | (v << 32)) & 0x1F00000000FFFF
        v = (v | (v << 16)) & 0x1F0000FF0000FF
        v = (v | (v << 8)) & 0x100F00F00F00F00F
        v = (v | (v << 4)) & 0x10C30C30C30C30C3
        v = (v | (v << 2)) & 0x1249249249249249
        return v

    cells = ((centers - centers.min(axis=0)) // 64).astype(np.int64)
    codes = spread(cells[:, 0]) | (spread(cells[:, 1]) << 1) | (spread(cells[:, 2]) << 2)
    return np.argsort(codes, kind="stable")


def window_box_voxels(
    centers_vox: np.ndarray, box_size: int = BOX_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """``(start, end)`` voxel corners of each window, half-open, as ``(N, 3)`` int64.

    Pure geometry -- no volume handle, no read. This is the function to call to answer
    "which voxels went into the embedding at this node" without touching the data.
    """
    centers_vox = np.asarray(centers_vox, np.int64)
    start = centers_vox - box_size // 2
    return start, start + box_size


def clipped_flags(
    centers_vox: np.ndarray,
    vol_min: np.ndarray,
    vol_max: np.ndarray,
    box_size: int = BOX_SIZE,
) -> np.ndarray:
    """``(N,)`` bool: does this window overhang the segmentation's bounds?

    A clipped window was zero-padded at inference time, so its count is over fewer real
    voxels than an interior node's and the two are not directly comparable. Computed from
    geometry alone, so it costs nothing and is always available alongside the counts.
    """
    start, end = window_box_voxels(centers_vox, box_size)
    return np.any(start < vol_min, axis=1) | np.any(end > vol_max, axis=1)


class MaskVolumeCounter:
    """Counts ``seg == root_id`` voxels in each node's SegCLR window.

    Opens the segmentation lazily on first use, so the object is cheap to construct and
    safe to build before forking into SLURM ranks.

    Args:
        mat_version: which materialization the ``root_id``s belong to. Selects the
            segmentation volume; getting this wrong fails silently.
        box_size: model input edge in voxels. Leave at 129 unless reproducing a different
            architecture.
        num_threads: concurrent crop reads.
        cache_bytes: TensorStore chunk cache for this process.
    """

    def __init__(
        self,
        mat_version: int = DEFAULT_MAT_VERSION,
        box_size: int = BOX_SIZE,
        num_threads: int = DEFAULT_NUM_THREADS,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
    ):
        if mat_version not in SEGMENTATION:
            raise ValueError(
                f"no segmentation volume for materialization {mat_version}; "
                f"known: {sorted(SEGMENTATION)}"
            )
        seg_path, seg_scale = SEGMENTATION[mat_version]
        self.mat_version = int(mat_version)
        self.box_size = int(box_size)
        self.num_threads = int(num_threads)

        # CropLoader reads this module-level constant when it opens a volume, so it has
        # to be set before the first handle is created rather than passed per call.
        _crops.TS_CACHE_BYTES = int(cache_bytes)

        # em_path is required by the constructor but never touched: only `.seg` is used
        # below, and both handles open lazily, so the EM volume is never opened at all.
        self.crops = CropLoader(
            em_path="",
            seg_path=seg_path,
            scale=seg_scale,
            seg_scale=seg_scale,
            box_size=self.box_size,
            oob="pad",
        )

    @property
    def resolution(self) -> np.ndarray:
        """nm per voxel, ``(3,)`` float64, read from the volume and checked."""
        res = self.crops.resolution
        if not np.allclose(res, EXPECTED_RESOLUTION_NM):
            raise RuntimeError(
                f"segmentation resolution is {tuple(res)}, expected "
                f"{EXPECTED_RESOLUTION_NM}. The window geometry and VOXEL_VOLUME_NM3 "
                "both assume the latter; a different scale means a different box."
            )
        return res

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(vol_min, vol_max)`` voxel bounds of the segmentation, each ``(3,)`` int64."""
        domain = self.crops.seg.domain
        vol_min = np.array([domain[i].inclusive_min for i in range(3)], np.int64)
        vol_max = np.array([domain[i].exclusive_max for i in range(3)], np.int64)
        return vol_min, vol_max

    def centers_for(self, coords_nm: np.ndarray) -> np.ndarray:
        """``(N, 3)`` int64 voxel centers for nm coordinates. Truncates, as inference did."""
        self.resolution  # validate before any geometry is derived from it
        return self.crops.nm_to_voxel(coords_nm)

    def count_one(self, center_vox: np.ndarray, root_id: int) -> int:
        """Voxels equal to ``root_id`` in the window centered at ``center_vox``.

        The mask exists only inside this call. ``load_mask_block`` is the same public
        entry point inference used, so the block counted here is the block embedded.
        """
        mask = self.crops.load_mask_block(center_vox, int(root_id))
        return int(mask.sum())

    def count_many(self, centers_vox: np.ndarray, root_id: int) -> np.ndarray:
        """``(N,)`` int64 counts, one per center, **in the caller's input order**.

        Reads are issued Morton-ordered for cache locality and permuted back before
        returning, so ordering is purely an internal performance concern and a caller
        cannot misalign counts against nodes by forgetting to un-sort.
        """
        centers_vox = np.asarray(centers_vox, np.int64)
        if not len(centers_vox):
            return np.empty(0, np.int64)

        # Open the volume before the pool starts. The handle is a lazy property, so
        # letting N threads find it unset would have them all open it concurrently.
        self.crops.seg

        order = morton_order(centers_vox)
        counts = np.empty(len(centers_vox), np.int64)

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            results = pool.map(
                lambda index: (index, self.count_one(centers_vox[index], root_id)),
                order,
            )
            for index, count in results:
                counts[index] = count
        return counts
